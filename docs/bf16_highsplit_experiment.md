# High-parallelism split-K BF16 trial

This branch starts at the verified **1.089x** feature commit `f406563399c647c30fb6c62f2040c4bb71c1e21b`, official job `1b6a10d0c8f84b6b`, **9/9** passing, **4,437.5 µs**. **Experimental code is not GPU-tested.**

## Hypothesis

SGLang's native pooler and QKV kernels in the official GPU profile use cuBLASLt, with a main GEMM launch and a separate split-K reduction at M16/M32. For pooler M16/M32 the profile measured around 19–20 µs for main GEMM + ~1.7–1.9 µs for reduction. QKV around 8–10 µs + ~1.6–1.8 µs.

A Triton split-K design may expose more independent CTA work. The earlier split-K experiment with SPLIT=4 was inferior; this variant changes **SPLIT=16** and keeps `BM=16, BN=64, BK=64` with FP32 partials and a BF16 final store. Larger split-K can also increase memory traffic and launch overhead, so this is a speculative trial—not a performance claim. It does not use CUDA-specific TMA/WGMMA intrinsics.

Only the default BF16 **qkv_gate** and **pooler**, for **M=16 or M=32**, route through the new split-K configuration. All other BF16 cases, all router cases including the winning M16/M32 fused paths, and all M=1 custom GEMV remain unchanged. The code uses shape/layer metadata only and has no output caching or reference access.

## Validation

1. Switch to experiment branch `experiment/muse-bf16-highsplit`.
2. Run `python3 -m py_compile muse_bf16_proj/submission.py` and `bash scripts/run_tests.sh`.
3. In an environment supporting offline SM90 compilation, compile at least `_split_gemm` at SPLIT=16, M=16/32, N=2304/4096, K=8192 before consuming any official checks.
4. If those pass and user chooses to consume a check allowance, run `okc check muse_bf16_proj/submission.py`; require **9/9** and a total score better than stable 1.089x beyond noise before promoting.
5. Inspect four affected per-case timings; abandon promptly if the old ~19–21 µs pooler main kernel cannot be beaten. Preserve the source and profile for audit.

No leaderboard submission is authorized. The next realistic approach after this bounded test is local H200 GEMM microbenchmarking with more advanced Hopper-specific kernels, not blindly increasing splits.
