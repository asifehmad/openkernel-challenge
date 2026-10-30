# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
"""Official check 9ea7b5b7: previous revision passed 9/9 at 0.772x.

This revision tests a baseline-equivalent router fallback for M=1 and M>8,
retaining the measured fast M=8 GEMV. Rerun official check to validate.
Only weights are prepared; every call uses fresh activations.
"""
from typing import Any, TypedDict

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # BF16 [2304,8192]
    o: torch.Tensor  # BF16 [8192,1024]
    pooler: torch.Tensor  # BF16 [4096,8192]
    unpooler: torch.Tensor  # BF16 [8192,4096]
    router: torch.Tensor  # FP32 [256,8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


if triton is not None:
    @triton.jit
    def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
              XM: tl.constexpr, XK: tl.constexpr,
              WN: tl.constexpr, WK: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr):
        # A CTA streams contiguous K elements from several weight rows. FP32
        # arithmetic also supports the exact FP32-router path; no TF32 here.
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        token = tl.program_id(1)
        kk = tl.arange(0, BK)
        acc = tl.full((BN, BK), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + kk
            x = tl.load(X + token * XM + k * XK, k < K, 0).to(tl.float32)
            w = tl.load(W + rows[:, None] * WN + k[None, :] * WK,
                        (rows[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
            acc = tl.fma(w, x[None, :], acc)
        out = tl.sum(acc, axis=1)
        tl.store(Y + token * N + rows, out, rows < N)

    @triton.jit
    def _split_gemm(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    XM: tl.constexpr, XK: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SPLIT: tl.constexpr):
        mi = tl.program_id(0) * BM + tl.arange(0, BM)
        ni = tl.program_id(1) * BN + tl.arange(0, BN)
        part = tl.program_id(2)
        kk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for chunk in range(tl.cdiv(K, BK * SPLIT)):
            k = (chunk * SPLIT + part) * BK + kk
            a = tl.load(X + mi[:, None] * XM + k[None, :] * XK,
                        (mi[:, None] < M) & (k[None, :] < K), 0)
            b = tl.load(W + ni[None, :] * WN + k[:, None] * WK,
                        (ni[None, :] < N) & (k[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
        tl.store(P + part * M * N + mi[:, None] * N + ni[None, :], acc,
                 (mi[:, None] < M) & (ni[None, :] < N))

    @triton.jit
    def _single_gemm(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     XM: tl.constexpr, XK: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # One CTA owns the entire K reduction for an output tile. This removes
        # split-K scratch/reduction traffic but can expose less parallelism.
        mi = tl.program_id(0) * BM + tl.arange(0, BM)
        ni = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for chunk in range(tl.cdiv(K, BK)):
            k = chunk * BK + kk
            a = tl.load(X + mi[:, None] * XM + k[None, :] * XK,
                        (mi[:, None] < M) & (k[None, :] < K), 0)
            b = tl.load(W + ni[None, :] * WN + k[:, None] * WK,
                        (ni[None, :] < N) & (k[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
        tl.store(Y + mi[:, None] * N + ni[None, :], acc,
                 (mi[:, None] < M) & (ni[None, :] < N))

    @triton.jit
    def _finish_split(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        acc = tl.full((B,), 0, tl.float32)
        for part in range(SPLIT):
            acc += tl.load(P + part * SIZE + i, i < SIZE, 0)
        tl.store(Y + i, acc, i < SIZE)

    @triton.jit
    def _router_pair(X, HI, LO, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     XM: tl.constexpr, XK: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        mi = tl.program_id(0) * BM + tl.arange(0, BM)
        ni = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        hi_acc = tl.full((BM, BN), 0, tl.float32)
        lo_acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + kk
            a = tl.load(X + mi[:, None] * XM + k[None, :] * XK,
                        (mi[:, None] < M) & (k[None, :] < K), 0)
            offsets = ni[None, :] * K + k[:, None]
            mask = (ni[None, :] < N) & (k[:, None] < K)
            hi = tl.load(HI + offsets, mask, 0)
            lo = tl.load(LO + offsets, mask, 0)
            hi_acc = tl.dot(a, hi, hi_acc)
            lo_acc = tl.dot(a, lo, lo_acc)
        tl.store(Y + mi[:, None] * N + ni[None, :], hi_acc + lo_acc,
                 (mi[:, None] < M) & (ni[None, :] < N))


def prepare(weights: Weights) -> dict[str, Any]:
    # Metadata checks only; no reads of tensor contents or host synchronization.
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        weight = weights[layer]
        dtype = torch.float32 if layer == "router" else torch.bfloat16
        if weight.ndim != 2 or weight.dtype != dtype:
            raise ValueError(f"{layer} weights must be a 2D {dtype} tensor")
    state: dict[str, Any] = {key: value.contiguous() for key, value in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (router - hi.float()).to(torch.bfloat16)
    # Also retain prepared transposed layouts for efficient tensor-core GEMM.
    # The original row-major hi/lo remain available to _router_pair.
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = state["router_lo"].t().contiguous()
    return state


def _checked_inputs(data: Inputs, state: dict[str, Any]):
    """Validate metadata before a raw-pointer launch; never inspect data values."""
    x, cfg = data["x"], data["config"]
    m, n, k, layer = cfg["m"], cfg["n"], cfg["k"], cfg["layer"]
    if layer not in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        raise ValueError("unknown projection layer")
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in (m, n, k)):
        raise ValueError("m, n, k must be positive integers")
    w = state[layer]
    dtype = torch.float32 if layer == "router" else torch.bfloat16
    if x.ndim != 2 or x.shape != (m, k) or x.dtype != torch.bfloat16:
        raise ValueError("activation shape/dtype does not match BF16 [m, k]")
    if w.ndim != 2 or w.shape[1] != k or n > w.shape[0] or w.dtype != dtype:
        raise ValueError("weight shape/dtype does not contain the requested [n, k] slice")
    if x.device != w.device:
        raise ValueError("activation and weights must share a device")
    return x, w, m, n, k, layer


def torch_kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    """Conservative development control; not the SGLang competition baseline.

    FP64 router fallback avoids depending on global TF32 settings. This route
    is for correctness/control, not a proposed fast GPU-router implementation.
    """
    x, w, _, n, _, layer = _checked_inputs(data, state)
    w = w[:n]
    if layer == "router":
        return (x.double() @ w.double().T).float()
    return x @ w.T


def _dispatch(data: Inputs, state: dict[str, Any], small_m_strategy: str) -> torch.Tensor:
    x, w, m, n, k, layer = _checked_inputs(data, state)
    if triton is None or not x.is_cuda:
        return torch_kernel_fn(data, state)
    # The first official check identified router M>8 as the largest regression.
    # Use the permitted BF16 hi+lo representation, as in SGLang, on those sizes.
    # Keep our faster M=8 FP32 GEMV; M=1 uses the native FP32 F.linear path.
    if layer == "router" and m == 1:
        return F.linear(x.float(), w[:n])
    if layer == "router" and m > 8:
        acc = torch.mm(x, state["router_hi_t"][:, :n], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"][:, :n], out_dtype=torch.float32)
        return acc
    # Dispatch depends only on public shapes and layer metadata.
    use_gemv = (layer == "router" and m <= 8) or (
        layer != "router" and ((small_m_strategy == "default" and m == 1) or
                                (small_m_strategy == "gemv" and m <= 32)))
    if use_gemv:
        out = torch.empty((m, n), device=x.device, dtype=w.dtype)
        _gemv[(triton.cdiv(n, 4), m)](
            x, w, out, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            BN=4, BK=512, num_warps=4)
        return out
    if layer == "router":
        out = torch.empty((m, n), device=x.device, dtype=torch.float32)
        _router_pair[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            x, state["router_hi"], state["router_lo"], out, m, n, k,
            x.stride(0), x.stride(1), BM=16, BN=32, BK=64, num_warps=4)
        return out
    if m <= 32:
        if small_m_strategy == "single_pass":
            out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
            _single_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
                x, w, out, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                BM=16, BN=64, BK=64, num_warps=4)
            return out
        split = 4 if k >= 4096 else 2
        partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _split_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 64), split)](
            x, w, partial, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            BM=16, BN=64, BK=64, SPLIT=split, num_warps=4)
        _finish_split[(triton.cdiv(m * n, 256),)](partial, out, m*n, split, B=256)
        return out
    return x @ w[:n].T


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    """First post-check router improvement candidate; unverified until rerun."""
    return _dispatch(data, state, "default")


def single_pass_kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    """Local experiment: one GEMM launch for BF16 M<=32, including M=1.

    Router behavior and large-M library fallback match kernel_fn. Selection is
    explicit in the local harness; the evaluator continues to call kernel_fn.
    """
    return _dispatch(data, state, "single_pass")


def gemv_kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    """Local experiment: per-token GEMV for BF16 M<=32. May reread weights M times.

    This is a measured-comparison candidate, not a claim that GEMV wins at M>1.
    Router behavior and large-M library fallback match kernel_fn.
    """
    return _dispatch(data, state, "gemv")
