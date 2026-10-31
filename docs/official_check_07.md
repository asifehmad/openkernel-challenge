# Official off-board check — hybrid router, job `1b6a10d0c8f84b6b`

**Evidence:** User-supplied authenticated VPS CLI output. The tested code was the hybrid router in commit `e8f697a6129aa1855c8e7b8b3a1acee331283614` (not independently associated with a source SHA256 from the job record). The exact `submission.py` file bytes are preserved by GitHub history and **not edited** during documentation promotion.

- Result: **9/9 official correctness tests passed**, off-board, no leaderboard entry.
- Speedup: **1.089x**, 95% interval **1.072–1.105x**.
- Candidate score time: **4,437.5 µs**, down from 4,587.3 µs at 1.053x and 4,625.7 µs at 1.044x.
- Job: `1b6a10d0c8f84b6b`, runtime 0:47, queue 0:11.
- CLI reported **18/20 checks** and **10/10 submissions** remaining in that day's window, as of this job.
- Raw profile and log remain on the VPS at `muse_bf16_proj/profiles/submission.1b6a10d0c8f84b6b.{txt,json,log}`.

## Official per-case times (owner-supplied, rounded µs)

| Load | M | QKV | O | Pooler | Unpooler | Router |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode 1 | 1 | 13.3 | 6.0 | 18.4 | 4.9 | 10.1 |
| Decode 16 | 16 | 13.9 | 6.6 | 21.3 | 18.5 | **7.4** |
| Decode 32 | 32 | 14.2 | 7.0 | 21.9 | 18.8 | **9.7** |
| Verify 1 | 8 | 13.8 | 6.5 | 21.2 | 7.0 | **15.3** |
| Verify 16 | 128 | 17.1 | 7.4 | 25.1 | 19.9 | 16.4 |
| Verify 32 | 256 | 19.5 | 9.3 | 31.0 | 24.3 | 17.6 |

Large improvements occur for router M16 (2.043x) and M32 (1.579x). The previous 28.7 µs M256 router regression was removed by reverting that public-shape dispatch to the native hi/lo matmul path. BF16 QKV, O, pooler, unpooler still approximately match SGLang at M8–256 and constrain progress toward the **owner-reported 1.60x leaderboard target**.

## Next recommended evidence

Capture the current profiler text and JSON on the VPS, especially router M128/256 and BF16 GEMM M16/M32/128/256. Compare actual GPU kernel counts and per-launch time. Keep a stable `1.089x` checkpoint while developing materially different BF16 GEMM approaches on a separate experimental branch; do not randomly vary tile sizes or consume checks for uncompiled code. Any new code requires its own official correctness and performance check before promotion. No official `okc submit` is authorized.
