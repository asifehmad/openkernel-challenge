# Experiment log

Official off-board jobs `9ea7b5b7bf0f43ec`, `b9558e1c1e85451c` and `1e9be7af4434464b` passed all 9 tests, with scores 0.772x, 0.979x and **1.044x**, respectively. The **narrow-tile experiment** passed correctness but was rejected for performance (0.932x; job `74d4ff146a4744f7`). `TBD` means unmeasured.

| ID | Hypothesis / scope | Change | Correctness evidence | Previous latency | New latency | Aggregate effect | Decision |
|---|---|---|---|---|---|---|---|
| A000 | Establish the actual contract before tuning | Corroborate two public starter snapshots, inspect pinned SGLang and primary upstream kernel documentation | Source bytes match between mirrors; live organizer verification blocked | N/A | N/A | N/A | Retain provenance; refresh against official source before judge run |
| A001 | Different projection shapes need different execution paths | Initial single-file Triton/library dispatch: GEMV, small-M split-K GEMM, large-M library GEMM, exact FP32 and hi+lo router | All five variants compile offline for SM90; GPU numerical checks and capture/replay **not run** | TBD | TBD | TBD | Experimental candidate only; no fastest/best-valid claim |
| A002 | Catch arithmetic/indexing errors before GPU rental | CPU oracle, strict score tests, Triton interpreter and SM90 offline compiler | 16 unit/interpreter tests passed, 2 native BF16-dot interpreter tests skipped; 100 reduced-size CPU cases passed | N/A | N/A | N/A | Retain as Phase-A checks; GPU validation remains required |
| A003 | Compare single-pass GEMM / repeated-token GEMV against split-K | Add explicit alternative entry points; preserve default dispatch and large-M library path; metadata bounds checks | 56 offline specializations compile; 600 reduced CPU cases pass across three entry points; GPU unrun | TBD | TBD | TBD | Retain as experiments, not promoted winners |
| A004 | Remove host launch gaps and detect timing bias | In-graph external events, per-shape pre/post SGLang A/A, raw/corrected samples and score-sensitivity reports | Full CPU suite 28 passed/3 skipped; synthetic calibration cases pass; CUDA timing unrun | TBD | TBD | TBD | Retain harness change, require H200 A/A before interpreting performance |
| A005 | Reduce first GPU session setup risk without changing default kernels | Preserve image packages, discover bundle layout, add representative smoke and retained-candidate comparison | 34 tests pass/3 skip, 200 reduced CPU cases pass; registry manifest verified, runtime/CLI/GPU still blocked | TBD | TBD | TBD | Retain workflow; no kernel promotion |

| A006 | Official check of initial candidate | Original GEMV/split-K/custom-router dispatch | 9/9 official tests passed, job `9ea7b5b7bf0f43ec` | SGLang 4,830.8 µs historical baseline | 6,253.9 µs | 0.772x (95% 0.761–0.784x) | Identify router M>8 regression (~51 µs each); fix first |
| A007 | Replace regressed router paths while keeping M=8 win | FP32 `F.linear` for M=1; native BF16 high/low matmul for M>8; keep custom M=8 GEMV | 9/9 official tests passed, job `b9558e1c1e85451c` | Previous run 6,253.9 µs | 4,934.9 µs | 0.979x overall; 21.1% lower candidate latency | Retain router changes |
| A008 | Eliminate BF16 split-K overhead for M=8/16/32 | Default route uses native PyTorch BF16 matmul at 8<=M<=32; M=1 custom GEMV and router unchanged | Official 9/9 check job `1e9be7af4434464b` | 4,934.9 µs | 4,625.7 µs | 1.044x (95% CI 1.029–1.060x) | **Best verified**; preserve commit `ced1acb` |
| A009 | Narrow single-pass CTA to raise SM occupancy at M=16/32 | Reuse masked BF16 `_single_gemm`, set `BM=M, BN=16, BK=128`; keep M=1, M=8, router and M>32 unchanged | **9/9 passed**, job `74d4ff146a4744f7` | 4,625.7 µs | **5,185.0 µs** | **0.932x**, 95% 0.918–0.946x; +12.1% latency | **Reject**; retain verified `ced1acb` on feature branch; QKV M16/M32 ~2x slower |

Initial candidate milestone: `116d64f`. See [validation record](validation.md) for
commands, environment and the interpreter limitation. A static or CPU-only pass
must not change A001's GPU correctness status.

## Next experiment record

```text
ID / date:
Source commit and submission SHA256:
Official task/reference hashes:
GPU / driver / image digest / PyTorch / Triton / CUDA:
Hypothesis:
Affected (M, N, K, layer) configurations:
One major code change:
Correctness: cases executed, seeds, failures, skips, graph replay / retained outputs:
Preparation: cold seconds, incremental allocated/reserved memory:
Benchmark: warmup, repetitions, captures, cache eviction, overhead calibration:
Baseline and candidate raw JSON/CSV paths:
Per-shape before/after latency and uncertainty:
Six weighted load totals and aggregate speedup:
Noise/A-A control:
Profiler evidence (if used):
Decision: retained / reverted / inconclusive:
Reason and next hypothesis:
```

Keep raw local results and compiler caches outside version control by default. Publish only deliberately selected small result summaries with their source hashes, command, environment and validation provenance. Never label an estimated roofline, third-party report or CPU result as our measured GPU performance.
