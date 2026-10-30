# Muse BF16 projection — H200 development

Independent initial candidate for `muse_bf16_proj`, targeting the Muse Spark Open
Kernel Challenge. **Phase A only: no GPU correctness result or measured speedup.**
No official competition submission has been made.

| Metric | Result |
| --- | --- |
| Official baseline | 4.8308 ms in a public October 24 API snapshot; live unverified |
| Current leaderboard target | 2.93 ms user-supplied target; live rank unverified |
| Our measured latency | TBD |
| Our verified speedup | TBD |
| GPU correctness | Not tested |
| Submission readiness | Not ready |

The immediate objective is <2.93 ms, with ≤2.5 ms as a stretch target. These
are targets, not forecasts. Optimize the geometric mean of **six weighted load
speedups**, not an arithmetic average of per-shape speedups.

## Files and provenance

- `muse_bf16_proj/submission.py`: single-file independent candidate. Weight-only
  preparation, fresh outputs, public shape dispatch, Triton GEMV/split-K GEMM,
  exact FP32 GEMV or permitted BF16 hi+lo router, and PyTorch GEMM for larger BF16
  shapes. The routes are experimental until H200 measurement.
- `muse_bf16_proj/{TASK.md,AGENTS.md,baseline.py,reference.py,checks.py}`:
  **unchanged secondary-source task snapshot**, corroborated across two public
  repositories. See [provenance](muse_bf16_proj/PROVENANCE.json) and retained
  [license](third_party/okc_snapshot/LICENSE). The live official site was blocked
  by the cloud proxy. These files must be reconciled with the current official
  bundle before competition readiness. No competitor kernel code was copied.
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
and **200 reduced CPU cases passed**; the previous 56-specialization offline
result remains applicable because submission.py is unchanged. No GPU result
has been produced. See [readiness update](docs/h200_preflight.md).

## Current limitations

No H200, driver, CUDA runtime or SGLang execution is available here. No GPU
latencies, register counts, occupancy or tensor-core utilization are claimed.
The official site's indexed rules and problem description were independently reviewed on 2026-10-31; see [official website cross-check](docs/official_web_crosscheck.md). Direct live task/API and CLI access, authenticated task hashes, current leaderboard and GPU correctness remain unverified.
The project is prepared for experiments; the fastest valid dispatch is still
to be selected from measurements. Raw results, compiled artifacts, credentials
and weights are ignored by Git. Existing cloud checkouts are isolated already;
use the checkout directly, with no worktree unless explicitly requested.
