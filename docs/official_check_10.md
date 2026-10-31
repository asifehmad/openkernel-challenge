# Official GPU check: fused short router (`5fbcf345f0454920`)

**Source:** owner-supplied `okc check muse_bf16_proj/submission.py` output on the remote evaluator. Source commit was `64e11666fd52e78e5859e14a9a74c2aadc333489`; file hash is not independently tied to the remote job log.

- **9/9 official tests passed.**
- **1.117x** speedup (95% CI **1.100–1.134x**), scored **4,324.7 µs**.
- Off-board check, **no leaderboard submission**.
- Remote run 0:51, queue 0:08; CLI showed 14/20 checks, 10/10 submissions remaining in that day’s quota.
- Profile available on owner VPS: `muse_bf16_proj/profiles/submission.5fbcf345f0454920.txt` and `.json`, `.log`.

Router before/after targeted configurations:

| Router M | Earlier stable `1b6a10d0` µs | Current µs | SGLang current µs | Current speedup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10.12 | **6.9** | 10.5 | **1.516x** |
| 8 | 15.32 | **7.4** | 26.4 | **3.591x** |
| 16 | 7.44 | 7.2 | 15.3 | 2.133x |
| 32 | 9.72 | 9.9 | 15.6 | 1.584x |
| 128 | 16.45 | 16.1 | 16.3 | 1.016x |
| 256 | 17.58 | 17.7 | 17.7 | 1.001x |

Non-router BF16 projections, especially pooler and QKV for M=8/16/32/128/256, remain near cuBLAS/SGLang times. The new score is a verified improvement; challenging the user-reported 1.6x leader requires memory-efficient tensor-core GEMV/micro-GEMM and potentially **lossless** weight compression. Any new source revision needs its own GPU check. Keep the verified source unchanged for rollback.
