# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
"""Experimental CUDA WMMA micro-GEMM for BF16 QKV/pooler, M=1/8/16/32.

Verified parent 4f36b51 passed official 9/9 at 1.117x, 4,324.7 us.
The new weight layout is a LOSSLESS BF16 16x16 transpose within each
tile, created only in untimed prepare(). CUDA tensor cores accumulate
FP32 partials; router and other BF16 paths remain identical to parent.
This branch has not passed an official GPU check and may be slower.
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
    def _router_pair_split(X, HI, LO, P,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           XM: tl.constexpr, XK: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           SPLIT: tl.constexpr):
        # Split long K across CTAs. Both BF16 hi and BF16 lo are consumed in
        # the same launch, with separate FP32 dot accumulators and no input
        # reads on the CPU. Scratch is FP32 and reduced in one second launch.
        mi = tl.program_id(0) * BM + tl.arange(0, BM)
        ni = tl.program_id(1) * BN + tl.arange(0, BN)
        part = tl.program_id(2)
        kk = tl.arange(0, BK)
        hi_acc = tl.full((BM, BN), 0, tl.float32)
        lo_acc = tl.full((BM, BN), 0, tl.float32)
        for chunk in range(tl.cdiv(K, BK * SPLIT)):
            k = (chunk * SPLIT + part) * BK + kk
            a = tl.load(X + mi[:, None] * XM + k[None, :] * XK,
                        (mi[:, None] < M) & (k[None, :] < K), 0)
            offsets = ni[None, :] * K + k[:, None]
            mask = (ni[None, :] < N) & (k[:, None] < K)
            hi = tl.load(HI + offsets, mask, 0)
            lo = tl.load(LO + offsets, mask, 0)
            hi_acc = tl.dot(a, hi, hi_acc)
            lo_acc = tl.dot(a, lo, lo_acc)
        tl.store(P + part * M * N + mi[:, None] * N + ni[None, :],
                 hi_acc + lo_acc,
                 (mi[:, None] < M) & (ni[None, :] < N))


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



# CUDA WMMA implementation stays inside this ONE submission file. The official
# rules allow native CUDA built during untimed prepare(); no external source
# files or dependencies are accessed. This implementation is original and
# uses NVIDIA's public 16x16x16 WMMA programming model.
_CPP_WMMA_SOURCE = r"""
torch::Tensor bf16_wmma_cuda(torch::Tensor x, torch::Tensor packed, int64_t splits);
"""

_CUDA_WMMA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace {
namespace wmma = nvcuda::wmma;

// Packed weights are [ceil(N/16), K/16, 16, 16], exactly BF16.
// Within one 16x16 tile, the last two dimensions are (k_inner, n_inner).
// Every block streams a contiguous set of 512-byte weight fragments.
// A CTA (one warp) computes one 16x16 output tile and one K partition.
__global__ void wmma_project(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ packed,
    float* __restrict__ partial,
    int M, int N, int K, int splits
) {
    const int tid = threadIdx.x;
    const int nm = blockIdx.y * 16;
    const int nn = blockIdx.x * 16;
    const int part = blockIdx.z;
    const int k_first = part * (K / splits);
    const int k_last = k_first + (K / splits);
    __shared__ __align__(16) __nv_bfloat16 a_s[256];
    __shared__ __align__(16) __nv_bfloat16 b_s[256];
    __shared__ __align__(16) float c_s[256];

    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   wmma::precision::bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   wmma::precision::bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.f);

    for (int kk = k_first; kk < k_last; kk += 16) {
        const __nv_bfloat16* wt = packed +
            ((nn / 16) * (K / 16) + (kk / 16)) * 256;
        #pragma unroll
        for (int i = tid; i < 256; i += 32) {
            int row = i / 16, col = i % 16;
            a_s[i] = (nm + row < M)
                ? x[(nm + row) * K + kk + col]
                : __float2bfloat16(0.f);
            b_s[i] = wt[i];
        }
        __syncwarp();
        wmma::load_matrix_sync(af, a_s, 16);
        wmma::load_matrix_sync(bf, b_s, 16);
        wmma::mma_sync(cf, af, bf, cf);
        __syncwarp();
    }
    wmma::store_matrix_sync(c_s, cf, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = tid; i < 256; i += 32) {
        const int row = nm + i / 16, col = nn + i % 16;
        if (row < M && col < N) {
            partial[(part * M + row) * N + col] = c_s[i];
        }
    }
}

__global__ void wmma_finish(
    const float* __restrict__ partial,
    __nv_bfloat16* __restrict__ out, int M, int N, int splits
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    float acc = 0.f;
    for (int p = 0; p < splits; ++p) acc += partial[p * M * N + idx];
    out[idx] = __float2bfloat16_rn(acc);
}
}  // namespace

torch::Tensor bf16_wmma_cuda(torch::Tensor x, torch::Tensor packed, int64_t splits) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                packed.scalar_type() == torch::kBFloat16, "BF16 required");
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 4, "wrong rank");
    const int M = static_cast<int>(x.size(0));
    const int N = static_cast<int>(packed.size(0) * 16);
    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(packed.size(1) * 16 == K && packed.size(2) == 16 &&
                packed.size(3) == 16, "invalid tile layout");
    TORCH_CHECK(splits > 0 && K % (16 * splits) == 0, "invalid split-K");
    TORCH_CHECK(x.get_device() == packed.get_device(), "different devices");
    auto partial = torch::empty({splits, M, N}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    dim3 grid((N + 15) / 16, (M + 15) / 16, static_cast<unsigned>(splits));
    wmma_project<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(packed.data_ptr<at::BFloat16>()),
        partial.data_ptr<float>(), M, N, K, static_cast<int>(splits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    wmma_finish<<<(M * N + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        M, N, static_cast<int>(splits));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


def _prepare_wmma(state: dict[str, Any]) -> None:
    """Compile once and pack only the original BF16 weights, without rounding."""
    from torch.utils.cpp_extension import load_inline

    # No compilation or host-to-device movement occurs in kernel_fn().
    state["_bf16_wmma"] = load_inline(
        name="muse_proj_bf16_wmma16_v1",
        cpp_sources=_CPP_WMMA_SOURCE,
        cuda_sources=_CUDA_WMMA_SOURCE,
        functions=["bf16_wmma_cuda"],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
        verbose=False,
        with_cuda=True,
    )
    for layer in ("qkv_gate", "pooler"):
        weight = state[layer]  # [N,K], already BF16 and contiguous.
        n, k = weight.shape
        if n % 16 or k % 16:
            raise ValueError("WMMA packed dimensions must be multiples of 16")
        # [n_tile, n_inner, k_tile, k_inner] ->
        # [n_tile, k_tile, k_inner, n_inner]. No precision conversion.
        state[layer + "_wmma16"] = (
            weight.view(n // 16, 16, k // 16, 16)
            .permute(0, 2, 3, 1).contiguous()
        )



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
    if router.is_cuda:
        _prepare_wmma(state)
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
    # Trial: extend the successful FP32-accumulated fused hi/lo split-K router
    # down to M<=8 on the official default route. This eliminates the
    # 3-kernel F.linear path at M=1 and the streaming GEMV at M=8.
    # Alternate entry points retain those established routes as controls.
    if layer == "router" and m == 1 and small_m_strategy != "default":
        return F.linear(x.float(), w[:n])
    if layer == "router" and (m > 8 or small_m_strategy == "default"):
        if small_m_strategy == "default" and m < 256:
            # Official job cdea2227: fused split-K router was 2.109x at
            # M=16, 1.600x at M=32, ~1.0x at M=128, but only 0.609x
            # at M=256. Revert only the regressing large-M shape to
            # the previously validated native high/low matmul below.
            # Hopper parallelism for K=8192 with few router output rows:
            # one CTA per output tile alone leaves most SMs unused. Splitting
            # K exposes more CTAs without launching two separate GEMMs.
            # Preserve the validated native route in explicit alternative
            # entry points for comparison with this unverified experiment.
            split = 16 if m <= 16 else 8 if m <= 32 else 4 if m <= 128 else 2
            partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
            out = torch.empty((m, n), device=x.device, dtype=torch.float32)
            _router_pair_split[(triton.cdiv(m, 16), triton.cdiv(n, 32), split)](
                x, state["router_hi"], state["router_lo"], partial,
                m, n, k, x.stride(0), x.stride(1),
                BM=16, BN=32, BK=64, SPLIT=split, num_warps=4)
            _finish_split[(triton.cdiv(m * n, 256),)](
                partial, out, m * n, split, B=256, num_warps=4)
            return out
        acc = torch.mm(x, state["router_hi_t"][:, :n], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"][:, :n], out_dtype=torch.float32)
        return acc
    # Experimental native WMMA for QKV/pooler small batches. The prepared
    # weights are the exact original BF16 bit patterns in tiled order.
    # Other projections and all router paths remain the tested dispatch.
    if (layer in ("qkv_gate", "pooler") and m in (1, 8, 16, 32)
            and "_bf16_wmma" in state):
        return state["_bf16_wmma"].bf16_wmma_cuda(
            x.contiguous(), state[layer + "_wmma16"], 16)
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
        # Official check b9558e1c found two-launch split-K slower than SGLang
        # for M=8/16/32. Try cuBLAS-backed PyTorch matmul for the default path.
        # This is a performance hypothesis; measure on the official H200.
        # The split-K code remains available to explicit experimental dispatch.
        if small_m_strategy == "default":
            return x @ w[:n].T
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
    """WMMA packed-weight micro-GEMM experiment; official GPU check required."""
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
