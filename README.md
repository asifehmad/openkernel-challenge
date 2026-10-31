# Muse BF16 projection — H200 development

Independent kernel candidate for `muse_bf16_proj`, targeting the Muse Spark Open
Kernel Challenge. **Best verified official off-board result: 1.117x speedup,
4,324.7 µs, all nine correctness tests passed** (job `5fbcf345f0454920`).
The hybrid router keeps fused split-K at M=9–255 and uses native hi/lo matmul
at M>=256. No leaderboard submission has been made.

| Metric | Result |
| --- | --- |
| Official baseline | 4.8308 ms in a public October 24 API snapshot; live unverified |
| Current leaderboard target | 2.93 ms user-supplied target; live rank unverified |
| Best official-check latency | 4.3247 ms (job `5fbcf345f0454920`) |
| Best official-check speedup | 1.117x; 95% 1.100–1.134x |
| GPU correctness | Best current candidate: 9/9 passed in official check |
| Submission readiness | Not ready |

The immediate objective is <2.93 ms, with ≤2.5 ms as a stretch target. These
are targets, not forecasts. Optimize the geometric mean of **six weighted load
speedups**, not an arithmetic average of per-shape speedups.

## Files and provenance

- `muse_bf16_proj/submission.py`: single-file independent candidate. Weight-only
  preparation, fresh outputs, public shape dispatch, Triton GEMV/split-K GEMM,
  exact FP32 GEMV or permitted BF16 hi+lo router, and PyTorch GEMM for larger BF16
  shapes. The active paths have passed an official H200 check; individual optimization choices still need profiling.
- `muse_bf16_proj/{TASK.md,AGENTS.md,baseline.py,reference.py,checks.py}`:
  **byte-for-byte matched** against the owner's authenticated `okc 0.7.2 init`
  download on the VPS (all five task/agent/baseline/reference/check SHA256).
  See [contract](docs/contract.md) and [provenance](muse_bf16_proj/PROVENANCE.json).
  The original cloud agent could not access the official site, but the VPS
  comparison resolved the file-identity uncertainty. No competitor code was copied.
- `benchmarks/`: local graph timing, strict scoring and CSV/JSON/Markdown reports.
- `scripts/`: setup, environment checks, correctness and offline compilation.
- `docs/`: [contract](docs/contract.md), [research](docs/research.md),
  [optimization plan](docs/optimization_plan.md), [experiments](docs/experiments.md),
  [all 30 shapes](docs/performance.md), [H200 setup](docs/h200.md),
  [readiness checklist](docs/readiness.md).

## CPU development

```bash
bash scripts/setup.sh cpu
source .venv/bin/activate
bash scripts/run_tests.sh
TRITON_CACHE_DIR="$PWD/.cache/triton" python scripts/compile_kernels.py
python scripts/verify_environment.py
```

The CPU venv uses PyTorch 2.10.0+cpu and Triton 3.6.0 for offline SM90
compilation. These are development tools, **not verified evaluation-image
versions**. Do not replace the GPU image's torch/Triton packages with them.
CPU correctness tests use reduced dimensions and the PyTorch fallback. Offline
compilation does not execute kernels or prove correctness/occupancy.

## H200 workflow

Use [docs/h200.md](docs/h200.md) to start the official image on your rented H200.
Preserve its dependencies and record its immutable image digest and versions.
With the official CLI installed from the official site, initialize separately:

```bash
bash scripts/fetch_official.sh official
# Locate TASK.md in the resulting tree; use its directory as OFFICIAL_DIR.
export OFFICIAL_DIR="$PWD/official/muse_bf16_proj"
python scripts/check_correctness.py --device cuda --official-dir "$OFFICIAL_DIR" --seeds 94017 82173
python benchmarks/benchmark.py --official-dir "$OFFICIAL_DIR" --output benchmarks/results/h200-001
python benchmarks/compare.py benchmarks/results/h200-001/results.json --output benchmarks/results/h200-001/report.md
```

Use the actual directory emitted by `okc init`; do not assume the example path
exists. To diagnose against the included **historical snapshot**, pass
`--official-dir muse_bf16_proj`, and retain that provenance in reports.
`--only-layer` / `--only-m` are diagnostic runs with no aggregate score.

The benchmark requires SGLang and a CUDA GPU, checks the nine prerequisite cases
and every timed input, alternates baseline/candidate graph replays, and captures
external timing events and the payload in one graph to avoid host launch gaps.
It reads a buffer larger than L2 before the start event and requires SGLang A/A
calibration before and after each shape. Raw intervals are the default;
experimental empty-interval corrections and all raw samples remain available.
It computes weighted load ratios and is a **local approximation** of the judge,
not a substitute for official validation. Inspect raw timings and repeat full
runs; do not claim improvements smaller than noise.

`torch_kernel_fn` is an optional conservative development control, not the SGLang
baseline. The FP64 router control is deliberately conservative and slow.
Neither our runner nor setup scripts call `okc check`, `okc submit`, or accept
competition terms. A person must handle account/rules acceptance; official
checks and submissions each require explicit approval.

The follow-up [audit](docs/audit.md) adds metadata bounds checks and explicit
`single_pass_kernel_fn` / `gemv_kernel_fn` experiment entry points. Default
dispatch and large-M PyTorch behavior remain unchanged. Validation at `41aeda0`:
28 tests passed, 3 native BF16-dot interpreter tests skipped, 600 reduced CPU
cases passed across the three candidate entry points, and 56 SM90
specializations compiled offline. See [validation](docs/validation.md).

The next readiness update adds [Hostinger CLI instructions](docs/hostinger.md),
bundle-layout discovery, a one-configuration GPU smoke stage, and independent
candidate runs with retained source/task artifacts. GPU setup now installs
nothing into the evaluation image. Latest local checks: **34 passed, 3 skipped**
and **200 reduced CPU cases passed** in that earlier local development session.
Those historical CPU/compiler results are distinct from subsequent official GPU
checks. See [readiness update](docs/h200_preflight.md).

## Official check progress

| Job | Correctness | Speedup | Time |
| --- | --- | ---: | ---: |
| `9ea7b5b7bf0f43ec` | 9/9 | 0.772x | 6,253.9 µs |
| `b9558e1c1e85451c` | 9/9 | 0.979x | 4,934.9 µs |
| `1e9be7af4434464b` | 9/9 | 1.044x | 4,625.7 µs |
| `74d4ff146a4744f7` | 9/9 | 0.932x | 5,185.0 µs (rejected branch) |
| `cdea2227f2654a1f` | 9/9 | 1.053x | 4,587.3 µs |
| `1b6a10d0c8f84b6b` | 9/9 | 1.089x | 4,437.5 µs |
| `29a3b639821c40f8` | 9/9 | 1.083x | 4,461.9 µs (large-tile trial) |
| `3cecb11c38704af0` | 9/9 | 1.086x | 4,446.5 µs (M128 trial) |
| `f6d322c21e96456f` | 9/9 | 1.081x | 4,470.4 µs (BF16 highsplit trial) |
| **`5fbcf345f0454920`** | **9/9** | **1.117x** | **4,324.7 µs** |

The 1.117x candidate is the best verified. Its score CI is 1.100–1.134x.
The current target reported by the owner is a leaderboard leader at 1.60x.
See [latest official check](docs/official_check_10.md) and [experiment log](docs/experiments.md).
Do not present off-board checks as leaderboard submissions.

## Current limitations

The local development environment still lacks an H200 for controlled profiling,
but the official remote GPU checker has measured the candidate repeatedly.
The task files have been verified against the owner's authenticated official CLI.
Current leaderboard rank and detailed GPU occupancy are not independently known.
The project's best observed dispatch is selected using off-board checks. Raw results, compiled artifacts, credentials
and weights are ignored by Git. Existing cloud checkouts are isolated already;
use the checkout directly, with no worktree unless explicitly requested.
