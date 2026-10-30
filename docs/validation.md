# Phase A validation record

## Latest H200 preflight update (candidate unchanged from 41aeda0)

- **34 tests passed, 3 native BF16-dot interpreter tests skipped**, plus
  **200 reduced CPU cases passed** via `scripts/run_tests.sh`.
- Six representative kernels compiled with offline resource extraction;
  register/shared/stack/local allocations are documented in
  [the preflight report](h200_preflight.md). No GPU execution or occupancy claim.
- Docker registry manifest/config retrieved and config SHA256 verified. The
  image was not unpacked due to storage limits; installed runtime versions and
  actual SGLang imports remain unverified.
- GPU smoke correctly refuses CPU-only execution and records failure evidence.
  The real H200 smoke/first-session path remains unrun.
- Fresh task layout tested with explicitly synthetic CLI fixtures; official CLI
  installation, authentication and current-task reconciliation remain blocked.
  No identical request through the blocked proxy was repeated.
- Default submission.py is unchanged. New workflow preserves tested artifacts
  and rejects incomplete/mismatched evidence when comparing candidates.

## Previous follow-up audit validation (41aeda0)

On the same CPU environment, the follow-up audit executed:

- Full test suite: **28 passed, 3 skipped**. The skips are the native BF16
  single-pass GEMM, split-K GEMM and router-dot interpreter cases on Triton 3.6.0.
- **600 reduced-size CPU cases passed**: 200 each for `kernel_fn`,
  `single_pass_kernel_fn`, and `gemv_kernel_fn`, seed 94017. These use the CPU
  fallback; the alternatives' native GPU routes remain unrun.
- `python scripts/compile_kernels.py --all-shapes`: **56 distinct SM90
  specializations compiled offline**, including the scored/custom paths and
  published special-case custom paths. Compiler output saved locally under
  ignored `benchmarks/results/offline-compile-audit.json`.
- `ruff check` and Python compilation passed. Shell syntax checks passed.
- Single-file packaging tested by copying only submission.py into a temporary
  directory and importing/calling it in `python -I` with installed dependencies.
- All five preserved task/check/baseline snapshot hashes still match provenance.
- Synthetic A/A tests check bias/noise rejection, correction sensitivity and
  raw-vs-corrected accounting. **No A/A GPU calibration has run**.

The first H200 workflow, external CUDA event timing/capture, full-shape numerical
tests, SGLang, native BF16 tensor cores, and GPU performance all remain pending.
Direct official-source access still returns proxy HTTP 403; CLI installation,
authentication, official versions and fresh hashes remain blocked. No official
check quota or submission was used. See [audit](audit.md).

## Original Phase A checks (historical)

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
