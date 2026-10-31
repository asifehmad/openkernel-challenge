# Experimental CUDA WMMA + lossless BF16 weight tiles

**Branch:** `experiment/muse-bf16-wmma-pack16` based on verified feature checkpoint `4f36b51ba415c3b76868d8ec38c9b6dc0b9c3eb4`.

**Last verified official result:** `5fbcf345f0454920`: **9/9**, **1.117x**, **4,324.7 µs**, 95% CI 1.100–1.134x. **This new WMMA source has not been compiled or benchmarked on an H200.**

## Purpose

Current BF16 QKV and pooler projections are essentially tied with cuBLAS at M=8/16/32. Earlier Triton single-pass and high-split variants regressed. A genuinely different micro-GEMM architecture is needed to target the user-reported 1.6x leaderboard leader.

This original CUDA experiment uses NVIDIA's documented `wmma::mma_sync` 16×16×16 BF16 tensor-core API. Within **untimed `prepare()`**, it calls PyTorch `torch.utils.cpp_extension.load_inline` to compile the source embedded in `submission.py` and creates 16×16 tiled, **bit-exact BF16** weights for `qkv_gate` and `pooler`. Every tile is stored as `(k_inner,n_inner)` for contiguous global streaming by one warp per CTA. This is a lossless **layout transpose**, not lower-precision quantization. Untimed preparation may take tens of seconds, but the official allowance is 300 s.

The timed native entry point executes two CUDA kernels: one WMMA kernel accumulating FP32 split-K partials, then one FP32 reduction with a BF16 output store. `SPLIT=16`. The first trial routes only the two BF16 layers at M=1/8/16/32; all router and other BF16 paths remain identical to the **1.117x** checkpoint. Unseen shapes use the stable fallback, not assumptions about the scorer. This approach may lose to cuBLAS even if correct.

## Validation

1. On the VPS switch to this branch, then `python3 -m py_compile muse_bf16_proj/submission.py` and `bash scripts/run_tests.sh`. CPU-only tests **do not** exercise the CUDA extension. Check `command -v nvcc`; the official image must provide nvcc.
2. If a CUDA H200 is available locally, compile/test a minimal CUDA smoke run of `prepare` and `kernel_fn` before an official check. Otherwise `okc check` will be the first full GPU compilation and correctness gate.
3. Run **one** off-board `okc check muse_bf16_proj/submission.py` only if the owner chooses to consume an allowance. Require 9/9, no compilation timeout (10 min total, prepare 300 s), and inspect all 30 scores.
4. If compilation fails, debug the native extension and retest. If some shapes regress, build shape-specific dispatch on a new experiment commit and retain only measured winners. If **no** new kernel beats native, revert to the protected feature branch.
5. No benchmark or >1.6x claim has been made; no leaderboard submission is authorized.

## Implementation checks

- [ ] Compile and load CUDA extension (not possible in the CPU-only assistant runtime).
- [ ] Confirm BF16 WMMA support and H200 `sm_90` compilation in official image.
- [ ] Verify exact 16×16 tile preparation and output correctness for M=1/8/16/32.
- [ ] Run tests with probe rows, fresh activations, outlier distributions, and nearby M.
- [ ] Check CUDA graphs and `prepare()` compilation/caching.
- [ ] Compare 30 scored shapes with 1.117x stable job.

**References:** Official rules https://openkernelchallenge.dev/rules; NVIDIA WMMA guide https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html; PyTorch load_inline https://docs.pytorch.org/docs/stable/cpp_extension.html.

Safety: keep `feat/muse-bf16-proj` unchanged; do not merge on a speculative speedup. This source contains no external files, hidden weights, network operations, or input-value dispatch.
