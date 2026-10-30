# Third official check — verified checkpoint `ced1acb`

**Source:** Owner-supplied CLI result of off-board `okc check muse_bf16_proj/submission.py` for job `1e9be7af4434464b`. The source was the preceding `feat/muse-bf16-proj` commit `ced1acb496f2ee25882e2254c5df2cbb1f32fac0`, before this experimental branch. The exact source SHA256 has not been tied to the remote job log.

- **9/9** correctness tests passed.
- **1.044x** faster than SGLang; 95% CI **1.029–1.060x**.
- Reported candidate time **4,625.7 µs**.
- Remote check ran 44 seconds, queued 14 seconds.
- Off-board only; CLI reported **17/20 checks** and **10/10 leaderboard submissions** left.
- Profile and raw log on the VPS under `muse_bf16_proj/profiles/submission.1e9be7af4434464b.{txt,json,log}`. Not fetched here.

| Load | M | SGLang sum µs | Candidate sum µs | Load ratio (approximate, rounded cases) |
|---|---:|---:|---:|---:|
| Decode 1 | 1 | 59.8 | 54.0 | 1.107x |
| Decode 16 | 16 | 75.6 | 75.6 | 1.000x |
| Decode 32 | 32 | 77.5 | 77.1 | 1.005x |
| Verify 1 | 8 | 75.0 | 64.0 | 1.172x |
| Verify 16 | 128 | 86.1 | 86.2 | 0.999x |
| Verify 32 | 256 | 102.6 | 102.6 | 1.000x |

The critical bottleneck is BF16 GEMM matching cuBLAS for M=16/32/128/256; improving only M=1 and the M=8 router cannot reach the user-reported 1.60x leader. The current experiment specifically tests one-launch Triton BF16 M=16/32 at narrow N tiles to exploit more SMs.

**Safety:** Leave the best-verified `feat/muse-bf16-proj` branch unchanged. Run a new official check on this branch only if local lint/tests succeed. Record new results and promote this change only if full correctness passes and total score improves beyond measurement noise. If this test loses, return to the feature branch. No automatic submission is authorized.
