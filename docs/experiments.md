# Experiment log

Six owner-reported off-board checks each passed **9/9**: 0.772x, 0.979x, 1.044x, rejected narrow-GEMM 0.932x, fused router 1.053x, and **hybrid-router 1.089x** (job `1b6a10d0c8f84b6b`). The subsequent large-router, M128-only, BF16 highsplit, and short-router attempts each passed 9/9; the short-router revision **1.117x** is best. All are official remote check results, not leaderboard submissions. `TBD` means unmeasured, not zero.

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
| A008 | Native BF16 matmul replaces custom split-K at M=8/16/32 | Default uses native PyTorch BF16 matmul, preserving M=1 GEMV | Official check `1e9be7af4434464b`: 9/9 | 4,934.9 µs | 4,625.7 µs | 1.044x | Preserve verified feature-branch checkpoint `ced1acb` |
| A009 | Narrow single-pass BF16 GEMM for M=16/32 | Reuse `_single_gemm` with `BM=M, BN=16, BK=128` | Official check `74d4ff146a4744f7`: 9/9 | 4,625.7 µs | 5,185.0 µs | 0.932x | Rejected; separate experiment branch |
| A010 | Fuse router high/low BF16 matmul with parallel split-K | `_router_pair_split`, split 16/8/4/2 by M, FP32 reduction | Official check `cdea2227f2654a1f`: 9/9 | 4,625.7 µs | 4,587.3 µs | 1.053x (95% CI 1.037–1.069x) | M16=2.109x and M32=1.600x; M256=0.609x regression |
| A011 | Hybrid router dispatch for M>=256 | Retain fused M16/M32 and restore native BF16 high/low router at M>=256 | **Official check `1b6a10d0c8f84b6b`: 9/9** | 4,587.3 µs | **4,437.5 µs** | **1.089x**, CI 1.072–1.105x | **Promote best verified unchanged**; next challenge is BF16 GEMM across M=8–256 |

| A012 | Larger router M128 and M256 tiles | BM32/SPLIT8 for M128/M256 | Official check `29a3b639821c40f8`: 9/9 | 4,437.5 µs | 4,461.9 µs | 1.083x; M128 12.8 µs but M256 regressed to 19.6 µs | Do not promote |
| A013 | Keep BM32 router only for M128 | restore native M256 | Official check `3cecb11c38704af0`: 9/9 | 4,437.5 µs | 4,446.5 µs | 1.086x, within run-to-run noise | Do not promote |
| A014 | Increase split-K for BF16 QKV/pooler M16/M32 | SPLIT16 | Official check `f6d322c21e96456f`: 9/9 | 4,437.5 µs | 4,470.4 µs | 1.081x; pooler M32 regressed | Reject |
| A015 | Extend fused hi/lo split-K router to M<=8 | Replaces F.linear M1 and FP32 GEMV M8, other paths unchanged | Official check `5fbcf345f0454920`: 9/9 | 4,437.5 µs | **4,324.7 µs** | **1.117x**, CI 1.100–1.134; M8 router 3.591x vs SGLang | **New best verified**; preserve source `64e11666` |

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
