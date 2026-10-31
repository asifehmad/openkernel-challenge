# M=128/256 router: larger tile and split-K experiment

**Branch:** `experiment/muse-router-large-tile`, based on verified `feat/muse-bf16-proj` commit `f406563399c647c30fb6c62f2040c4bb71c1e21b`. This branch is **not GPU tested**; do not promote or submit before a full official check.

## Evidence

From the user's official profile for check `1b6a10d0c8f84b6b`:

- **9/9 correctness**, **1.08864x** (95% CI 1.07231–1.10497), scored latency **4,437.5 µs**.
- Router M=16: `_router_pair_split` 3.0 µs + `_finish_split` 1.2 µs = 4.2 µs GPU; scored **7.44 µs**.
- Router M=32: 4.3 + 1.0 = 5.3 µs GPU; scored **9.72 µs**.
- Router M=128: 10.2 + 1.1 = 11.2 µs GPU; scored **16.45 µs**, essentially native baseline 16.59 µs.
- Router M=256: native split hi/lo performs 2 cuBLAS GEMMs, 2 reductions and an elementwise add, **14.9 µs profiled**, **17.58 µs scored**.
- BF16 GEMMs retain original cuBLAS operations and are the bigger obstacle to the user-reported **1.6x leaderboard leader**.

**Hypothesis:** At large M, the fused router's `BM=16` repeats weight traffic over too many M tiles, and fewer split-K partitions leave long per-CTA reductions. Increasing `BM` from 16 to 32 and `SPLIT` to 8 preserves weight precision but may improve parallelism/reuse.

| Router M | Verified route | Experimental route |
| --- | --- | --- |
| 1 | Native FP32 | unchanged |
| 8 | Triton FP32 GEMV | unchanged |
| 9–16 | Fused `BM=16, SPLIT=16` | unchanged |
| 17–32 | Fused `BM=16, SPLIT=8` | unchanged |
| 33–127 | Fused `BM=16, SPLIT=4` | unchanged |
| 128 | Fused `BM=16, SPLIT=4` | **Fused `BM=32, SPLIT=8`** |
| 256 | Native BF16 hi+lo matmul | **Fused `BM=32, SPLIT=8`** |
| >256 | Native BF16 hi+lo matmul | unchanged |

Only the two scored shapes M=128 and M=256 receive different launch geometry. The split kernel still uses separate FP32 accumulators for both BF16 parts, followed by FP32 partial reduction. No input-dependent dispatch or caching.

## Validation

1. `git fetch origin && git switch -c experiment/muse-router-large-tile --track origin/experiment/muse-router-large-tile`
2. `python3 -m py_compile muse_bf16_proj/submission.py && bash scripts/run_tests.sh`
3. When authorized, run one off-board `okc check muse_bf16_proj/submission.py` and require **9/9**. The official check must also score all 30 cases; inspect M=128 and M=256 router, and the aggregate CI.
4. If only M=128 or only M=256 improves, hybridize that shape alone on a subsequent controlled commit. If neither improves, return to `feat/muse-bf16-proj` (`f406563`) without merging.
5. The official checker is the only authoritative performance result, and this experiment has **no measured result yet**. This branch does not authorize a leaderboard submission.

## Longer-term constraint

At fixed non-router timings, even the hypothetical case of pushing every router to the official theoretical floor yields only approximately **1.276x** aggregate. To challenge 1.60x, the BF16 projection paths also need large improvements. Custom Hopper WGMMA/TMA, pipeline scheduling and lossless weight reordering may be worth testing **on a local H200 first**, but none has been measured for this repository.
