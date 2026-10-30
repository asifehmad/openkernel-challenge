# Follow-up audit

This audit continues `feat/muse-bf16-proj` from `23cfab6`. No previous candidate
was discarded. The existing default GPU dispatch remains in place until H200
measurements select a winner. All performance fields remain TBD.

## Primary-source access

Direct website, installer and API requests failed at the egress proxy CONNECT
step with HTTP403, before an origin response. Explicit network permission was
granted; an elevated installer request also failed. The saved environment draft
already includes `openkernelchallenge.dev`; saving a draft does not apply it to
the running proxy. Apply that domain in environment settings before retrying.
No proxy bypass, credential extraction or unofficial executable was used.

The official CLI could not be downloaded, inspected, installed or executed.
Its version and authentication are unverified. `OKC_TOKEN` is absent, but no
token is requested: connectivity/installation must work before authentication
can be diagnosed. No login, official check quota or submission was used.
No current rules, limits, leaderboard, or fresh official-file hashes are claimed.
See [machine-readable access evidence](official_access.json).

All five vendored starter files still match their recorded secondary-source
hashes exactly. This proves snapshot preservation, not equivalence with the live
task. The candidate is expected to differ from the official starter submission.
`scripts/compare_official.py` now hashes all six requested files and distinguishes
expected candidate changes from changes requiring contract review. It never
overwrites the candidate or snapshot and refuses to overwrite a prior report.

## Kernel findings and changes

| Path | Audit result | Change / remaining validation |
| --- | --- | --- |
| Input metadata | Previously inconsistent n/k/m could reach raw-pointer launches | Reject shape, dtype, device and bounds mismatches using metadata only |
| GEMV | Masked K/N, FP32 FMA/reduction, strided X/W reads; original FP32 router W | Interpreter checks include tails, probes, outliers, zeros and output guards; GPU pending |
| Split-K M≤32 | k=(chunk×S+part)×BK+lane covers each K once; partial writes disjoint; FP32 reduction then BF16 store | Preserved; full-shape GPU bounds/accuracy/graphs pending |
| Router M≤8 | Uses original FP32 W with BF16 activations converted to FP32 for arithmetic | Preserved, no TF32 |
| Router M>8 | hi=BF16(W), lo=BF16(W−FP32(hi)); separate FP32 accumulators added into FP32 output | Matches the permitted mirrored hi+lo approach; GPU tolerance still pending |
| Unpooler | Computes first cfg.n rows; n=1024 for M≤8, else 8192 in task inputs | Slice/metadata/retention tests expanded; no full-weight assumption |
| Large M BF16 | PyTorch GEMM | Preserved until another path wins measured comparisons |
| New single pass | One masked BF16-dot GEMM, one final BF16 store, no partial buffer | Explicit `single_pass_kernel_fn`; unmeasured |
| New small-M GEMV comparison | Per-token GEMV for all BF16 M≤32 | Explicit `gemv_kernel_fn`; may lose from rereading W M times |

State contains only weights and their permitted hi+lo preparation. Each call
allocates a fresh output. Dispatch and validation read shapes/dtypes/devices,
not activation contents; no host synchronization, output cache or external
resource access was introduced. Isolated `python -I` import/calls from a copied
single file pass on CPU. That is not proof of evaluation-container compatibility.

## Benchmark findings and changes

The former host-issued start-event → graph replay → end-event sequence could
include Python/launch gaps for microsecond kernels. `GraphTimer` now captures
cache eviction, two **external CUDA event nodes**, and the payload in one graph.
Only replay is host-issued; the GPU timestamps exclude eviction and host launch
gaps. The external-event graph path still needs actual GPU validation.

Raw in-graph intervals are the default. They already exclude graph-launch cost,
so blindly subtracting the judge's roughly 4 µs fixed cost would be incorrect.
The empty-event interval and both raw/corrected sample sets are retained.
Optional `--correction empty` is experimental, not judge-equivalent calibration.

Every shape now has SGLang-versus-SGLang A/A pairs before and after candidate
timing. Default acceptance bounds are ≤3% median speedup bias from 1 and ≤3%
relative median absolute deviation. Empty subtraction is rejected if its
interval exceeds 20% of raw baseline time or yields nonpositive times. A/A can
detect asymmetric bias/noise; it cannot establish absolute timing accuracy or
the private judge's overhead model. Failed gates preserve samples and prohibit
an aggregate score. `--aa-only` records a distinct calibration result type.

The official bundle's SGLang baseline is always the reference backend. The
optional `torch_kernel_fn` is explicitly a candidate-side control, never a
replacement for SGLang. Metadata separately records local result type, entry
point, hashes, and false official-check/submission flags. Reports rank tuning
priorities by candidate load share / 6, the local log-score sensitivity, rather
than ranking isolated microbenchmark speedups.

## Next decisions

Apply official-domain access, install/inspect the CLI, fetch a separate current
bundle and reconcile hashes/rules. Then use [the first H200 session](h200.md)
to test native kernels, fresh-input graph replay, unpooler/router edge cases,
A/A calibration and all 30 scored shapes. No route is promoted merely because
it compiled or passed reduced CPU tests. Official check quotas and competition
submissions each require explicit owner approval.
