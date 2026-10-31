# Measured TinyGEMM winner, isolated to pooler M=8

## Verified checkpoint and source evidence

The protected feature branch `feat/muse-bf16-proj` commit `4f36b51ba415c3b76868d8ec38c9b6dc0b9c3eb4` passed owner-reported official check `5fbcf345f0454920`: **9/9**, **1.117x** (CI 1.100–1.134), **4,324.7 µs**.

The original FlashInfer trial on `experiment/muse-flashinfer-tinygemm` (`7565061`) passed official `okc check` job `96941f1938bc4ef4`: **9/9**, **1.112x** (CI 1.095–1.128), **4,345.2 µs**. The owner also supplied four GPU profiler lines naming `tinygemm_kernel`, proving the imported TinyGEMM ran on the evaluator. The owner's VPS virtual environment did not have FlashInfer installed, which does not reflect the official remote image.

| Scored shape | Feature branch ~µs | Full TinyGEMM trial µs | Decision |
|---|---:|---:|---|
| QKV M=1 | 14.0 | 14.4 | Revert to proven GEMV |
| QKV M=8 | 14.5 | 14.8 | Revert to cuBLAS |
| Pooler M=1 | 18.4 | 19.7 | Revert to proven GEMV |
| Pooler M=8 | 21.5 | **19.5** | Retain TinyGEMM experimentally |

**One-factor hybrid:** The default dispatch invokes `flashinfer.gemm.tinygemm_bf16(x, w, out)` only for `layer == "pooler" and m == 8`; every other shape follows the verified feature-branch code. Original BF16 weights and activations are used directly; no caching, lossy weight conversion or hidden state. If TinyGEMM is absent, the original verified path is used. All router optimizations are unchanged.

**Status:** New combination has **NOT** been checked. Expected aggregate improvement is small and might be within noise; do not claim a new best or spend an off-board check just to prove a sub-1% gain if a more meaningful GPU candidate is ready.

## Performance strategy toward reported ~1.6x leader

The remaining challenge is QKV/pooler/unpooler/O BF16, M=16/32/128/256, which mostly equal SGLang's cuBLASLt latency. The failed native WMMA (0.631x) and two Triton split-K versions demonstrate that changing tile sizes without cold-cache profiling is inefficient. The current 1.117x stable version must remain available.

FlashInfer's `mm_bf16` supports `cutlass`, `cudnn`, `cublaslt`, `cutile`, `tinygemm` and other backends depending on device; `tgv` is SM100-class in the current source, **not a Hopper/H200 option**. This library must be inspected inside the evaluator before betting check quota on any unsupported backend. `cutile` is nominally SM90+ but has separate runtime dependencies. Using the same cuBLAS backend will not beat it just by changing Python wrappers.

A more substantial strategy is **one-CTA, high-throughput BF16 GEMV with warp-level MMA and contiguous K streaming**, plus optional **lossless, exactly invertible BF16 bitpacking in untimed `prepare()`** to reduce HBM traffic, tested with arbitrary weight values and one-hot probe rows. Do not reuse previous inefficient shared-memory WMMA pipeline unchanged. Prefer a rentable local H200 for repeated CUDA-graph cold-cache A/B experiments rather than consuming a daily quota to compare each tile configuration.

## Local commands

```bash
git fetch origin
git switch -c experiment/muse-tinygemm-pooler-m8 --track origin/experiment/muse-tinygemm-pooler-m8
python3 -m py_compile muse_bf16_proj/submission.py
bash scripts/run_tests.sh
```

The source is self-contained except for FlashInfer, which is already known to exist in the remote evaluator. `okc check` is optional at user discretion: the reported 11/20 remaining in that day's window is not evidence of local GPU access. No leaderboard submission has been made.
