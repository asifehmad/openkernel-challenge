# Official off-board GPU check 2 — `b9558e1c1e85451c`

Source: owner-supplied authenticated `okc check muse_bf16_proj/submission.py` output. The check tested the **router-fallback revision** (commit `5116bec93caac590e79b5413ef9578cd92d2d3d8`) and preceded the next BF16 small-M experiment. Its exact submitted source hash has not been independently read from the VPS job record.

- **9/9 official correctness tests passed**.
- **0.979x speedup** (95% CI: 0.964–0.994x), still slower than SGLang.
- Official estimated time: **4,934.9 µs**.
- Check ran 0:44 (queued 0:05). CLI reported **18/20 checks remaining**, **10/10 leaderboard submissions remaining** after this job.
- Off-board check only; no leaderboard submission.
- Raw profile and log remain on the owner's VPS under `muse_bf16_proj/profiles/submission.b9558e1c1e85451c.{txt,json,log}`.

| Load | Layer | Baseline µs | Candidate µs | Speedup |
|---|---|---:|---:|---:|
| decode 1 | qkv_gate | 14.4 | 13.7 | 1.051x |
| decode 1 | o | 6.7 | 5.6 | 1.194x |
| decode 1 | pooler | 21.5 | 18.1 | 1.192x |
| decode 1 | unpooler | 7.2 | 5.3 | 1.354x |
| decode 1 | router | 10.7 | 10.5 | 1.018x |
| decode 16 | qkv_gate | 14.0 | 18.8 | 0.744x |
| decode 16 | o | 6.6 | 8.1 | 0.813x |
| decode 16 | pooler | 21.6 | 22.5 | 0.958x |
| decode 16 | unpooler | 18.7 | 20.1 | 0.927x |
| decode 16 | router | 14.8 | 15.0 | 0.991x |
| decode 32 | qkv_gate | 14.3 | 19.5 | 0.732x |
| decode 32 | o | 6.9 | 9.7 | 0.717x |
| decode 32 | pooler | 22.1 | 24.2 | 0.913x |
| decode 32 | unpooler | 19.0 | 22.4 | 0.848x |
| decode 32 | router | 15.5 | 15.6 | 0.995x |
| verify 1 | qkv_gate | 14.4 | 19.1 | 0.751x |
| verify 1 | o | 6.7 | 8.2 | 0.822x |
| verify 1 | pooler | 21.6 | 22.5 | 0.960x |
| verify 1 | unpooler | 7.2 | 9.8 | 0.737x |
| verify 1 | router | 26.3 | 15.6 | 1.688x |
| verify 16 | qkv_gate | 17.2 | 17.0 | 1.009x |
| verify 16 | o | 7.6 | 7.5 | 1.012x |
| verify 16 | pooler | 25.1 | 25.1 | 1.002x |
| verify 16 | unpooler | 19.9 | 20.0 | 0.997x |
| verify 16 | router | 16.6 | 16.7 | 0.994x |
| verify 32 | qkv_gate | 20.1 | 20.1 | 1.000x |
| verify 32 | o | 9.6 | 9.6 | 0.997x |
| verify 32 | pooler | 30.6 | 30.5 | 1.001x |
| verify 32 | unpooler | 24.6 | 24.6 | 1.002x |
| verify 32 | router | 17.4 | 17.6 | 0.992x |

## Next hypothesis

The original two-launch BF16 split-K+finish path loses especially on M=8, M=16 and M=32. Try PyTorch BF16 matmul for these default routes, as already successfully used for BF16 M>32. Retain the winning M=1 BF16 GEMV and router routes. **The replacement has not been measured or correctness-tested on the remote GPU.** Recheck with the official CLI; do not submit to leaderboard yet.
