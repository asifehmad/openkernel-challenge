# Official website cross-check — 2026-10-31

This record supplements, and **does not overwrite**, the agent's historical access diagnostics in `docs/official_access.json`. It is an independent review of the **published official website content as indexed by web search**. It is not an authenticated CLI download, a live API response, an H200 execution, or an official challenge check.

## Source access and evidence boundaries

- Official [rules](https://openkernelchallenge.dev/rules): the indexed text was retrievable on 2026-10-31. Opening the page directly returned a JavaScript-rendered HTML result with no readable lines in this environment.
- Official [problem description](https://openkernelchallenge.dev/problems/muse_bf16_proj): the indexed text was retrievable on 2026-10-31, confirming the projection contract and layer types.
- Official [home page](https://openkernelchallenge.dev/): indexed content confirms the competition's kernel/serving-stage distinction and CLI installer URL.
- Direct [task API](https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj) and the CLI installer bytes could **not** be retrieved here. A separate container test also failed DNS resolution for the challenge domain. Thus no official task-file SHA256, CLI version, authentication state, or daily account limits were obtained.
- The indexed website's leaderboard text appeared stale ("no scored runs"); it does **not** contradict the owner's separately observed 2.93 ms result. Do not use indexed leaderboard text as a current rank or score.

## Cross-check against the checked-in candidate

| Published rule or problem property | Repository source check | Status |
| --- | --- | --- |
| Kernel-stage hardware is one NVIDIA H200 SXM | H200 guide and `verify_environment.py` target this hardware | Aligned; hardware not run |
| Use the `lmsysorg/sglang:dev-muse-spark-open` image; no runtime network installs | H200 guide uses that image and preserves its packages | Aligned by setup design; container not run |
| Single Python file providing `kernel_fn` | `muse_bf16_proj/submission.py` is one file with the required function/header | Aligned by inspection |
| Optional weight-only `prepare()`, untimed up to 300 s | `prepare(weights)` makes contiguous weights and a router hi/lo pair | Aligned by inspection; timing/memory unmeasured |
| CUDA graph capture; no value-dependent branches/allocations or host synchronization | Dispatch uses configuration, shape/dtype/device metadata; kernels use fresh `x` | Aligned by inspection; CUDA replay untested |
| BF16 projections retain BF16 precision; router FP32 may use BF16 high/low with FP32 accumulation | BF16 paths use BF16 operands; router small-M reads FP32 weights and large-M uses hi/lo | Aligned by inspection; numerical checks untested on GPU |
| No output replay, benchmark tampering, external processes, harness access or network in submission | No such operations identified in reviewed `submission.py` | No issue found by source inspection; not official validation |
| Submission licensed under Apache License 2.0 | Repository has `LICENSE` (Apache-2.0) and submission SPDX header | Aligned |
| Kernel checks/submissions are quota-limited; organizer may change limits | Project does not invoke `okc check` or `okc submit` automatically | Aligned; current quota must be read with `okc problems` |
| Kernel leaderboard carries no cash prizes; prize track is serving | Project currently targets kernel phase as preparation | Informational |

The indexed problem specification still agrees with the checked-in secondary-source snapshot on the core matrix operation `X @ W.T`, BF16 activations, BF16 projection weights, FP32 router weights, and the unpooler output-width special case (first 1024 rows for up to eight tokens). The **complete live task bundle, nine checks, all 30 timing configurations, and scorer implementation were not independently re-downloaded**. Keep those items provisional.

## Required before H200 qualification

1. On the owner's authorized Hostinger VPS, use [the CLI handoff](hostinger.md) to inspect the official installer, install the CLI, complete interactive GitHub authorization if necessary, and fetch a fresh task bundle. Do not paste tokens into repository files or chat.
2. Run `scripts/compare_official.py`, inspect actual differences and reconcile them before measuring against the old snapshot. Do not silently overwrite the original files.
3. Use `scripts/first_h200_session.sh` only after official bundle reconciliation; run one-configuration smoke checks before full GPU correctness/benchmarking.
4. Only the owner can approve consumption of `okc check` quota or an `okc submit` run.

## Outcome

**No confirmed rules violation or source-code change requiring an immediate kernel rewrite was found in the reviewed commit `7fa658af6563dea7e6d418f90c11d2bb23ccac3f`.** Consequently the candidate kernel has been left unchanged. This record narrows the website/rules uncertainty, not the live CLI/API, numerical correctness, or performance uncertainty.
