# Profile-driven router split-K experiment

**Status: first fused router revision was verified at 1.053x, 9/9 correct; the new hybrid fallback remains untested.** This branch is based on the verified `feat/muse-bf16-proj` commit `ced1acb496f2ee25882e2254c5df2cbb1f32fac0`, which passed the official `okc check` 9/9 at **1.044x (4,625.7 µs)**, job `1e9be7af4434464b`.

## Evidence from official profile

The owner's uploaded text profile from that job reports:

- Router M=1: FP32 `F.linear` entails 3 GPU kernels: dot 4.0 µs, elementwise 2.1 µs, reduce 1.6 µs (7.7 µs profiled, 10.43 µs scored).
- Router M=8: custom `_gemv` GPU time 9.0 µs; scored 15.15 µs versus SGLang 25.79 µs (1.702x). **Do not alter** this winning route in the first experiment.
- Router M=16: two cuBLAS GEMMs 8.3 µs, two split-K reductions 3.6 µs, elementwise addition 0.9 µs (12.8 µs GPU, 14.71 µs scored).
- Router M=32: same 5-launch pattern, 12.9 µs profiled, 15.15 µs scored.
- Router M=128: same pattern, 13.8 µs profiled, 16.44 µs scored.
- Router M=256: same pattern, 14.9 µs profiled, 17.86 µs scored.

GPU-kernel profiler totals and scored cold-L2 CUDA-graph latencies are different measurements; do not subtract or conflate them.

**Important bound:** Replacing every router latency with its official theoretical floor while holding BF16 projections constant would give about **1.26x** aggregate, not 1.60x. Router optimization is necessary but insufficient: BF16 GEMMs must also improve.

## Hypothesis

The old custom `_router_pair` fused hi+lo arithmetic but under-parallelized the long K=8192 reduction, taking ~51 µs on previous official checks. The new `_router_pair_split` uses the same permitted BF16 hi/lo approximation with independent FP32 accumulators, but partitions K over multiple CTAs. The FP32 partial outputs are combined by the existing generic `_finish_split` reduction, giving two GPU launches rather than the baseline's five kernels.

| M | Split-K partitions | CTA grid (for N=256) |
|---|---:|---:|
| 9 or 16 | 16 | 128 |
| 32 | 8 | 128 |
| 128 | 4 | 256 |
| 256 | 2 (initial experiment; native fallback in hybrid) | 256 (initial experiment only) |

Tile configuration `BM=16, BN=32, BK=64`, four warps. This maintains the old unmodified M=1 FP32 `F.linear`, M=8 Triton GEMV, all BF16 projection routes and M>32 fallback. Only the default router path for M>8 is modified; local `single_pass_kernel_fn` and `gemv_kernel_fn` retain the proven native high/low router path as controls.

## Official fused-router check: job `cdea2227f2654a1f`

The owner ran `okc check` on the initial fused split-K router revision (commit `c2b0cf020c68b8e684fcd0256e13337c13d8c240`). All **9/9 correctness tests passed**. Overall speedup was **1.053x**, CI **1.037–1.069x**, time **4,587.3 µs**. The check was off-board, with 19/20 checks remaining in the owner's reported daily window.

| Router case | SGLang µs | Fused router µs | Ratio |
| --- | ---: | ---: | ---: |
| M=1 | 10.3 | 10.4 | 0.990x |
| M=8 | 26.3 | 16.4 | 1.602x |
| M=16 | 14.9 | 7.1 | **2.109x** |
| M=32 | 15.4 | 9.6 | **1.600x** |
| M=128 | 16.5 | 16.5 | 0.997x |
| M=256 | 17.5 | 28.7 | **0.609x** |

The whole-run improvement from 1.044x to 1.053x is smaller than the stated run-level confidence bounds; it is not evidence that the whole-run speedup is reliably higher. The M=16/32 changes, however, were large. For M=256, the added split-K work clearly lost to native matmul.

**Current hybrid experiment:** Retain fused router for M=9–255 but restore native high/low `torch.mm` for M>=256. Both math paths have passed official correctness tests separately, but their combination has **not** been officially checked. The prior best 1.044x feature branch and earlier fused-router commit remain available.

A replay using the displayed, rounded per-case timings predicts approximately **1.072x** total speedup with the M=256 router reduced to its previous ~17.5 µs; this is a mathematical counterfactual, **not a GPU measurement or guaranteed gain**. The true result may differ because every official run samples timing independently.

## Validation gates

1. On VPS, switch to `experiment/muse-router-fused-splitk`, compile Python and run `bash scripts/run_tests.sh`. CPU-only validation does **not** exercise this Triton kernel.
2. Before consuming an official check, optionally offline-compile the new kernel for representative M=9,16,32,128,256 if suitable Triton 3.6 environment is available.
3. With user's approval, use one off-board `okc check muse_bf16_proj/submission.py`; require 9/9 official correctness. Pay attention to M=9 outlier router and M=8 probe correctness.
4. Compare all 30 scored cases, especially M=16/32/128/256 router, with the saved 1.044x run. **Keep only if aggregate score improves beyond noise**, with no precision regression. Otherwise discard branch.
5. Never run `okc submit` automatically. No leaderboard entry is authorized.

This experiment uses a two-launch path and may still lose to cuBLAS. There is no claim of measured improvement, compliance certification, or Hopper occupancy.
