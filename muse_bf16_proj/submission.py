# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
"""Independent Phase-A candidate. GPU correctness/performance not yet verified.

Only weights are prepared. Every invocation computes a fresh output from x.
Triton routes are experiments, not measured dispatch winners; see docs/experiments.md.
"""
from typing import Any, TypedDict

import torch

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
    state: dict[str, Any] = {key: value.contiguous() for key, value in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (router - hi.float()).to(torch.bfloat16)
    return state


def torch_kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    """Conservative development control; not the SGLang competition baseline.

    FP64 router fallback avoids depending on global TF32 settings. This route
    is for correctness/control, not a proposed fast GPU-router implementation.
    """
    x, cfg = data["x"], data["config"]
    w = state[cfg["layer"]][:cfg["n"]]
    if cfg["layer"] == "router":
        return (x.double() @ w.double().T).float()
    return x @ w.T


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x, cfg = data["x"], data["config"]
    m, n, k, layer = cfg["m"], cfg["n"], cfg["k"], cfg["layer"]
    if triton is None or not x.is_cuda:
        return torch_kernel_fn(data, state)
    w = state[layer]
    # Dispatch depends only on public shapes and layer metadata.
    if m == 1 or (layer == "router" and m <= 8):
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
        split = 4 if k >= 4096 else 2
        partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _split_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 64), split)](
            x, w, partial, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            BM=16, BN=64, BK=64, SPLIT=split, num_warps=4)
        _finish_split[(triton.cdiv(m * n, 256),)](partial, out, m*n, split, B=256)
        return out
    return x @ w[:n].T
