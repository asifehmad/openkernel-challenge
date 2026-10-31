# Profile-driven router split-K experiment

**Status: untested.** This branch is based on the verified `feat/muse-bf16-proj` commit `ced1acb496f2ee25882e2254c5df2cbb1f32fac0`, which passed the official `okc check` 9/9 at **1.044x (4,625.7 µs)**, job `1e9be7af4434464b`.

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
| 256 | 2 | 256 |

Tile configuration `BM=16, BN=32, BK=64`, four warps. This maintains the old unmodified M=1 FP32 `F.linear`, M=8 Triton GEMV, all BF16 projection routes and M>32 fallback. Only the default router path for M>8 is modified; local `single_pass_kernel_fn` and `gemv_kernel_fn` retain the proven native high/low router path as controls.

## Validation gates

1. On VPS, switch to `experiment/muse-router-fused-splitk`, compile Python and run `bash scripts/run_tests.sh`. CPU-only validation does **not** exercise this Triton kernel.
2. Before consuming an official check, optionally offline-compile the new kernel for representative M=9,16,32,128,256 if suitable Triton 3.6 environment is available.
3. With user's approval, use one off-board `okc check muse_bf16_proj/submission.py`; require 9/9 official correctness. Pay attention to M=9 outlier router and M=8 probe correctness.
4. Compare all 30 scored cases, especially M=16/32/128/256 router, with the saved 1.044x run. **Keep only if aggregate score improves beyond noise**, with no precision regression. Otherwise discard branch.
5. Never run `okc submit` automatically. No leaderboard entry is authorized.

This experiment uses a two-launch path and may still lose to cuBLAS. There is no claim of measured improvement, compliance certification, or Hopper occupancy.
