# H200 preflight update from 41aeda0

## Verified directly

- Existing branch/workspace was clean at 41aeda0. The candidate file and default
  dispatch remain byte-identical to that commit.
- Saved network settings still contain the official domain; no runtime access
  change or authorized VPS connection was available. Identical blocked official
  website requests were not repeated. CLI/version/authentication/current task
  and rules verification remain blocked. The new Hostinger guide is prepared
  for the owner's normal authorized VPS route; it has not been executed there.
- Docker registry access succeeded. The Linux/amd64 official-image manifest
  and its config were retrieved; the config digest was checked with SHA256.
  Metadata declares CUDA 13.0.3. These are registry facts, not installed runtime
  verification. The compressed layers total 23,866,593,205 bytes. With 29 GB free in
  this sandbox, the image was not unpacked or executed. Full runtime validation
  still requires an H200 host with sufficient storage.

## Corrected setup/workflow issues

GPU setup previously installed CPU development requirements, including a pinned
NumPy version. It now installs nothing and validates the existing image instead.
The environment report includes CUDA 12/13 cuBLAS package variants, NumPy,
compute capability 9.0 and the required external-event constructor argument.

The task resolver supports direct and nested bundles, requires all six files,
rejects ambiguity and escaping symlinks, and never treats discovery as proof of
official provenance. Fresh task differences stop the session for reconciliation.
The actual current CLI layout cannot be verified until the CLI is accessible.

The minimal GPU stage defaults to QKV, M16, using the actual supplied SGLang
baseline and one explicit candidate. It validates eager output, retention,
native launch, external-event capture and fresh normal/outlier/zero replay.
The CPU-only failure path correctly records that GPU execution did not occur.

The full session accepts `--entry-point` and `--smoke-only`, retains exact source
and task copies, and records a local selection only after full evidence checks.
Separate runs remain comparable only with matching hardware/software/task/timing
settings. A local best-observed entry point is not an official result and does
not change the evaluator's default. No official check or submission quota used.

## Kernel audit and compiler allocation evidence

No concrete launch/indexing/precision defect was found for the known contract.
`_single_gemm` masks all M/N/K tails, uses BF16 operands with FP32 accumulation,
then one BF16 store. Split-K partitions K disjointly and reduces FP32 partials.
Router and unpooler behavior remains as documented in the prior audit.

Triton 3.6 compilation uses the same four warps and three stages as runtime
defaults. It emits `.target sm_90a`, PTX 8.7 and 128-thread blocks. H200 supports
that architecture; actual driver/runtime launch is still untested. Optional
`scripts/compile_kernels.py --resources` uses the installed Triton-bundled CUDA 12.8
cuobjdump and reports compiler/cubin allocation without requiring a GPU:

| Representative kernel | Registers/thread | Dynamic shared launch bytes | Stack/local bytes |
| --- | ---: | ---: | ---: |
| BF16 GEMV |38|64|0/0|
| FP32 GEMV |32|64|0/0|
| Split-K GEMM |96|10,240|0/0|
| Single-pass GEMM |96|10,240|0/0|
| Split reduction |16|0|0/0|
| Router pair |128|8,192|0/0|

Cubin-reported shared allocation is retained separately in JSON, not added
blindly to the dynamic launch field. Zero local allocation is not a measured
spill-traffic result. No achieved occupancy, utilization or latency is claimed.

## Local validation and remaining blockers

`scripts/run_tests.sh`: 34 tests passed, 3 native BF16-dot interpreter tests skipped,
and 200 reduced CPU correctness cases passed. Bundle tests use explicit synthetic
flat/nested CLI fixtures, not organizer downloads. Candidate selection tests
reject mismatched/incomplete/CPU-only evidence and do not produce GPU results.
Six representative kernels were recompiled for resource extraction; the prior 56
offline specializations remain valid because candidate bytes did not change.
Ruff, Python compilation and shell syntax checks passed.

Pending: owner executes the Hostinger installer/help/authentication/fetch steps;
current rules and task hashes are reconciled; H200 image dependencies and driver
are executed; smoke, full native correctness, graph timing/A/A and performance
are measured. Until then, official compatibility and submission readiness are
not established.
