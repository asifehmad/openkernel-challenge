# Router M=128-only: measured winner plus validated fallback

**Status: official 9/9 correctness passed; 1.086x nominally below stable 1.089x.** This experiment is based on the large-tile branch commit `1b54a13d1f4d640c080cbf6c1f2a36ec0b566c47` and preserves the latest verified stable candidate on `feat/muse-bf16-proj` (commit `f406563399c647c30fb6c62f2040c4bb71c1e21b`).

## What the official checks found

| Official job | Result | Candidate time | Correctness |
| --- | --- | --- | --- |
| `1b6a10d0c8f84b6b` (stable hybrid) | 1.089x (95% 1.072–1.105x) | 4,437.5 µs | 9/9 |
| `29a3b639821c40f8` (large M128/M256 tile) | 1.083x (95% 1.066–1.099x) | 4,461.9 µs | 9/9 |

The official large-tile check reported **M=128 router 12.8 µs** vs **16.45 µs** in the stable hybrid, but **M=256 router 19.6 µs** vs **17.58 µs** in stable hybrid. Although the overall two results overlap within their intervals, the M=128 improvement is meaningful enough to isolate. The M=256 slowdown is a regression worth reverting.

## Exact change

For router only:
- M=1: native FP32, unchanged.
- M=8: successful custom GEMV, unchanged.
- M=9–16: fused `BM=16/SPLIT=16`, unchanged.
- M=17–32: fused `BM=16/SPLIT=8`, unchanged.
- M=33–127: fused `BM=16/SPLIT=4`, unchanged.
- M=128: **fused `BM=32/SPLIT=8`**, retained from large-tile experiment.
- M>=256: **native high/low BF16 GEMM with FP32 accumulation**, restored to stable dispatch.

All BF16 projection routes are exactly as in the stable checkpoint. No input-dependent routing; no caching; every result uses current activations. M=128 and M>=256 have individually passed 9 official correctness tests in prior source revisions, but **this combination has not**.

## Official result: job `3cecb11c38704af0`

The owner ran an authenticated off-board `okc check` on this M128-only hybrid. **All nine correctness tests passed**, with **1.086x overall speedup** (95% interval **1.070–1.103x**) and **4,446.5 µs** estimated time. Job ran 0:48 and queued 0:09. The CLI reported **16 of 20** check runs left and **10 of 10** leaderboard submissions left in the day's window.

The targeted router M128 improved to **13.0 µs** (from 16.45 µs on the stable 1.089x check), and native M256 recovered to **17.3 µs** (vs 17.58 µs on the stable check). Despite that per-shape success, the whole-run score was **1.086x**, nominally below the still-best **1.089x** checkpoint; the intervals overlap, so this difference is inconclusive. **Do not promote on the aggregate result** without repeated evidence. Archive this as a successful M128 micro-optimization but an inconclusive overall optimization.

Profile: `muse_bf16_proj/profiles/submission.3cecb11c38704af0.txt` and corresponding `.json/.log` on VPS. Source branch commit `d7dc6aad651a15fe1f22922ef6554d9cb999c5ce`, exact file digest not independently tied to remote record.

## Next gate

On VPS, switch to this experimental branch and run `python3 -m py_compile muse_bf16_proj/submission.py` and `bash scripts/run_tests.sh`. If all pass, the user may choose to consume one `okc check` run off-board. Compare 9/9 correctness and aggregate speedup with stable **1.089x**, paying particular attention to M=128 router and restored M=256 router. If no convincing improvement, leave stable branch untouched.

The current leader target of **1.6x** requires substantially faster BF16 QKV/pooler/unpooler kernels in addition to router improvements. A marginal router dispatch improvement is not an end-state. No leaderboard submission is authorized.
