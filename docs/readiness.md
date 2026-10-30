# Submission readiness

**Status: not ready for competition submission.** The implementation is a CPU-developed candidate. Compatible GPU execution, official validation, performance measurement and current-rule verification remain outstanding. No competition submission is authorized automatically.

| Metric | Result |
|---|---|
| Official baseline | 4.83 ms — user-supplied target; historical mirror records 4830.8 µs; not live-verified |
| Current leaderboard target | 2.93 ms — user-supplied target; current leaderboard unverified |
| Our measured latency | TBD |
| Our verified speedup | TBD |
| Correctness | GPU not tested |
| Submission readiness | Not ready |

## Before H200 measurements

Completed locally: isolated-file CPU packaging, metadata/bounds validation,
single-pass/GEMV comparison entry points, latest 34 passing tests (3 explicitly skipped),
latest 200 reduced CPU checks and the previous 56 offline SM90 compilations. Six
representative kernels now also have offline allocation diagnostics. These do not satisfy
any GPU gate below. See [audit](audit.md) and [validation](validation.md).

- [ ] Obtain access to an NVIDIA H200 SXM and confirm compute/runtime support.
- [ ] Record the immutable official image digest and all relevant tool versions.
- [ ] Fetch the current task through official `okc init`; confirm file hashes, full rules, memory/preparation caps, available dependencies and leaderboard.
- [ ] Confirm all required GPU imports and candidate JIT compilation succeed in that image.
- [ ] Pass the one-configuration smoke stage for each selected entry point before starting full tests.
- [ ] Run the local official reference/check routines on all nine published test cases and all 30 scored configurations.
- [ ] Test unseen weights/activation seeds, outliers, zero rows, signed power-of-two probes and nearby batch sizes.
- [ ] Test CUDA graph capture and replay with new activation contents and retained old outputs.
- [ ] Measure cold preparation time and extra GPU memory; keep room below current official limits.
- [ ] Check that the submission uses no external files, network, processes, reference imports, harness inspection, input-value dispatch, output caching or prohibited precision conversion.

## Before claiming a performance result

- [ ] Benchmark actual supplied SGLang baseline and candidate on the same GPU, with paired graph trials and a cold-cache protocol.
- [ ] Run `--aa-only` and pass per-shape SGLang A/A gates; validate external-event capture and correction sensitivity on the actual image.
- [ ] Retain raw samples, repeated-capture outcomes, A/A control, timing-overhead treatment and environment/source metadata.
- [ ] Review all 30 per-shape comparisons and six weighted load totals; use the geometric-mean load score.
- [ ] Clearly label local figures as provisional rather than official judge scores.
- [ ] Retain the tested candidate/task copies; compare only complete matching records using `summarize_candidates.py`.
- [ ] Verify any apparent gain exceeds observed noise; repeat promising changes without changing several variables at once.
- [ ] Verify GPU resources and bottlenecks with profiling where available; do not substitute profiler timings for graph latencies.

## Before a competition submission

- [ ] User has personally accepted current rules through the official login flow and supplied credentials securely if needed.
- [ ] Current permitted dependencies, file-size/memory/preparation/runtime limits and licenses are satisfied.
- [ ] Official checks passed on the exact final file; record job/result provenance separately from local runs.
- [ ] User explicitly approves any use of official check quota before it is consumed.
- [ ] Final file begins with `# okc: problem=muse_bf16_proj` and is self-contained.
- [ ] Exact final file SHA256 and tested Git commit are recorded; no later unvalidated edits.
- [ ] Repository contains no credentials, private keys, model weights, binaries or benchmark caches.
- [ ] Verified changes are committed and pushed to the requested feature branch.
- [ ] User explicitly approves the official competition submission.

Neither an available GPU nor a passing syntax check satisfies this checklist. Check marks should be added only after the named operation succeeds and its evidence is saved. A failed check should remain visible with a diagnosis rather than being bypassed by changing the official reference or tolerances.
