# Phase A validation record

Executed on 2026-10-30 on a CPU-only x86_64 cloud machine. Python 3.12.14,
PyTorch 2.10.0+cpu, Triton 3.6.0, NumPy 2.2.6, pytest 8.4.2, ruff 0.14.2.
No NVIDIA device, nvcc, Nsight Compute or SGLang was present.

| Check | Observed result |
| --- | --- |
| `VENV_PATH=/workspace/.venvs/openkernel bash scripts/setup.sh cpu` | Passed, including rerun with existing dependencies |
| `bash scripts/run_tests.sh` in the prepared venv | 16 unit/interpreter tests passed; 2 skipped; 100 reduced-size CPU correctness cases passed |
| `ruff check muse_bf16_proj/submission.py benchmarks scripts tests` | Passed |
| `python scripts/compile_kernels.py` with writable `TRITON_CACHE_DIR` | Five SM90 variants compiled offline; no CUDA execution |
| `python scripts/verify_environment.py` | CPU environment identified; CUDA unavailable |
| Full-shape H200 checks, CUDA graph replay, SGLang timings | Not run |
| Live official site/task/rules/leaderboard | Proxy HTTP 403; secondary task snapshots used |

The 16 tests comprise six contract tests, seven score/report tests and three
Triton interpreter tests. The 100 correctness cases use 32×64 weights, five
layers, four batch sizes and five input distributions. They exercise the CPU
fallback and output-retention checks, not the GPU dispatch.

Native BF16/FP32 GEMV runs in the CPU interpreter with strided tensors and tail
masks. Split-K/reduction and hi+lo-router interpreter tests widen BF16 values
losslessly into FP32 buffers to validate indexing, reductions and tails.
Triton 3.6.0's interpreter stores BF16 as uint16 but passes that storage directly
to NumPy in `create_dot`; native BF16-dot interpreter cases consequently produce
invalid arithmetic. Those **two tests are explicitly skipped for 3.6.0**, not
counted as passes. GPU BF16 tensor-core arithmetic is unverified. The GPU
checker retains its original tolerances and executes the native BF16 kernels.

Offline compiler metadata (four warps per CTA for these specializations):

| Variant | Shared memory, bytes | Execution |
| --- | ---: | --- |
| BF16 GEMV | 64 | Unrun |
| FP32 router GEMV | 64 | Unrun |
| Split-K GEMM | 10,240 | Unrun |
| Split-K reduction | 0 | Unrun |
| Hi+lo router | 8,192 | Unrun |

Register allocation, achieved occupancy, tensor-core utilization, preparation
GPU memory and time, all latency measurements, and the final score remain TBD.
Compiler metadata is not a measured hardware resource profile.
