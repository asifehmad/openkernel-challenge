# Official check evidence: job `9ea7b5b7bf0f43ec`

**Source:** Owner-supplied output of authenticated `okc check muse_bf16_proj/submission.py` on 2026-10-31. This check was **off-board**, on the pre-router-fallback source revision. It does not validate subsequent changes. The exact source SHA256 for this check has not yet been independently tied to the remote job; preserve the CLI profile/log for audit.

- 9/9 official tests passed.
- Reported score 0.772x (95% interval 0.761–0.784x); **slower** than SGLang.
- Candidate aggregate time 6,253.9 µs.
- Job 9ea7b5b7bf0f43ec; wall 0:48 and queue 0:10.
- CLI reported 19 of 20 check runs and 10 of 10 submissions remaining at that time.
- Official profiles on VPS: `muse_bf16_proj/profiles/submission.9ea7b5b7bf0f43ec.{txt,json,log}`. Not accessible through GitHub.

## Router timings identify the main regression

| Load | M | SGLang µs | Old candidate µs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| decode 1 | 1 | 10.5 | 16.7 | 0.630x |
| decode 16 | 16 | 14.9 | 50.9 | 0.292x |
| decode 32 | 32 | 15.4 | 50.9 | 0.302x |
| verify 1 | 8 | 26.0 | 15.2 | **1.713x** |
| verify 16 | 128 | 16.4 | 50.6 | 0.323x |
| verify 32 | 256 | 17.6 | 51.0 | 0.345x |

## Controlled follow-up

Keep M=8 router in the successful custom FP32 GEMV path. For M=1 use baseline-equivalent FP32 `F.linear`, and for M>8 use the baseline-equivalent two BF16 high/low `torch.mm` operations with FP32 accumulation. Both use actual weights and current activations. All other candidate routes remain unchanged.

**Status:** This is an experiment, not a measured performance improvement. Before another leaderboard submission, pull the latest commit on the VPS and run a new authorized `okc check`. Compare all 30 cases, especially BF16 small-M regressions, then tune further.
