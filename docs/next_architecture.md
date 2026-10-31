# Strategic pivot: from tile tuning to H200 architecture

**Verified result:** the fused short-router revision (source commit `64e11666fd52e78e5859e14a9a74c2aadc333489`) passed official check `5fbcf345f0454920` **9/9**, **1.117x** at **4,324.7 µs** (95% CI 1.100–1.134). It has now been promoted unchanged. Previous feature-branch commit `f406563` scored 1.089x. User-reported first-place target approximately **1.6x** (roughly 3.0 ms). Never automatically use `okc submit`.

## Immediate controlled experiment: fuse the short router

The official stable profile records a router M=1 native FP32 `F.linear` path with three GPU kernels (7.6 µs cumulative GPU work, 10.12 µs scored) and an M=8 custom streaming FP32 GEMV (8.9 µs GPU, 15.32 µs scored). The already-tested M=16 fused `_router_pair_split` path takes 3.0 µs plus 1.2 µs finish (7.44 µs scored). The short-router experiment extends this same two-launch architecture to `1<=M<=8` in the default entry point using `BM=16, BN=32, BK=64, SPLIT=16`. The original F.linear and FP32 GEMV still exist as controls in alternative dispatch entry points.

The router's FP32 weights are represented as BF16 `hi` and `lo` with independent FP32 accumulation, as permitted. Official check `5fbcf345f0454920` passed 9/9, including M=8 router correctness; M=1 router scored 6.9 µs vs 10.5 µs SGLang, and M=8 scored 7.4 µs vs 26.4 µs. No input caching or value-dependent dispatch was introduced.

## Key result from latest check (high split-K, job `f6d322c21e96456f`)

Off-board `okc check` passed 9/9 but measured **1.081x (4,470.4 µs; 95% CI 1.064–1.097x)**, nominally below stable 1.089x. Pooler M=32 slowed to 24.9 µs (from about 21.9 µs stable); pooler M=16 improved to 20.4 µs vs about 21.3 µs stable. Do **not** promote the whole trial. Preserve the artifact in `experiment/muse-bf16-highsplit`.

## How to make materially larger gains

This trial is a fast, isolated proof point; **it is not enough for 1.6x**. Most BF16 GEMMs for M=8/16/32/128/256 currently execute near cuBLAS speed. Repeating Triton `BM/BN/BK/SPLIT` sweeps against remote checks alone has not yielded competitive speedups.

A public participant's performance write-up (https://thethings.ai/p/okc/muse-bf16-proj-report, retrieved 2026-10-31) reports independently measured **1.33x** with a custom CUDA `mma.sync m16n8k16` GEMV for M<=32, careful warp-level weight fragment organization, and a **lossless 12-bit BF16 weight representation** for some matrices. It also reports that a compressed WGMMA implementation for M=128/256 failed to beat cuBLAS in their tests. These are third-party claims, not our measurements, and do not constitute source to copy.

Proposed new architecture, ordered by impact and verification risk:

1. Implement a standalone, GPU-correct custom CUDA tensor-core GEMV/micro-GEMM for BF16 M<=32, **using weights at their original precision** and `prepare()` to create a more coalesced layout. Focus first on pooler/QKV and a benchmark path that provides register, occupancy and HBM-efficiency evidence. Benchmark several candidates on a rented H200 with CUDA graph replay, L2 eviction, random input replay, and paired cuBLAS comparisons **before** spending official checks.
2. Only after verifying the original-precision version, prototype **lossless** weight compression in `prepare()`, including full representable-range fallback and exact recovery for probe columns. Never mask weight errors as acceptable quantization; these are not permitted lossy weights. Test multiple seeds and outliers.
3. Explore Hopper `wgmma`/TMA for the large M=128/256 cases only if microbenchmarks show a realistic benefit. Avoid introducing costly pipelines merely because these instructions exist.
4. Do shape-by-shape promotion only for measured wins; preserve stable router and cuBLAS fallbacks. Demand 9/9 official tests, no speed regressions on nearby shapes, and net geometric-mean improvement beyond check noise.

## Quick validation for this short-router branch

```bash
cd /root/openkernel-challenge
git fetch origin
git switch -c experiment/muse-router-short-fused --track origin/experiment/muse-router-short-fused
python3 -m py_compile muse_bf16_proj/submission.py
bash scripts/run_tests.sh
python3 scripts/compile_kernels.py --json /tmp/short-router-sm90.json
# If all gates pass and the owner approves using an off-board check:
okc check muse_bf16_proj/submission.py
```

The remote CLI in job `f6d322c21e96456f` reported **15/20 checks remaining** in its current window and **10/10 actual submission attempts remaining**. Official counts may reset; inspect `okc problems`. Do not submit to the leaderboard until the user explicitly approves.
