# Provisional challenge contract

The repository preserves the mirrored starter `muse_bf16_proj/TASK.md`, `AGENTS.md`, `baseline.py`, `reference.py` and `checks.py` without changing their contents. They are usable for local development, but they are **secondary snapshots**, not directly authenticated current organizer downloads. See the accompanying provenance files and `third_party/okc_snapshot/LICENSE`. Official website access returned proxy HTTP 403; refresh with official `okc init muse_bf16_proj` and read the current full rules before declaring submission readiness.

## Source identity

The bytes match in these public repositories:

- `lilaclilac09/open-kernel-challenge`, commit `d9bdcc9f01dff1018f2f256a0a1a63911a16557e`, under `muse_bf16_proj/`.
- `Xuefeng-Zhu/open-kernel-challenge`, commit `5372a758c7d4c03431ed4887f75dfb7d5e53aa1b`, under `official/`. Its manifest says it downloaded `https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj` on 2026-10-24. The raw response includes the starter template, descriptions and scored configurations.

| File | SHA256 |
|---|---|
| `TASK.md` | `deb9118fa75cbc03c2ebd16b283e72a0314168d40299fa7501373eb1960a2bc4` |
| `AGENTS.md` | `d6f3fb0cedb7cfac9aaef929c5ec2999eba5e1103cc3fa240f49de4e732e21fc` |
| `baseline.py` | `39f52ea5b65505958aa9dee9c8129dd394cbba4e670046f862b5a88ce91a0574` |
| `reference.py` | `e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d` |
| `checks.py` | `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055` |

These are snapshot integrity checks, not proof that the live organizer still distributes the same files. The independent candidate does not import or depend on reference/check files during evaluation.

## API and computation

```python
state = prepare(weights)             # called once, untimed
out = kernel_fn(data, state)         # timed, CUDA-graph capturable
# data["x"]: BF16 [M,K]
# data["config"]: {"m": M, "n": N, "k": K, "layer": layer}
# out = data["x"] @ weights[layer][:N].T
```

| Layer | Full weight [rows,K] | Weight dtype | Output dtype | Output width N |
|---|---|---|---|---|
| qkv_gate | [2304,8192] | BF16 | BF16 | 2304 |
| o | [8192,1024] | BF16 | BF16 | 8192 |
| pooler | [4096,8192] | BF16 | BF16 | 4096 |
| unpooler | [8192,4096] | BF16 | BF16 | **1024 for M≤8**, otherwise 8192 |
| router | [256,8192] | FP32 | FP32 | 256 |

The sliced unpooler computes the **first 1024 rows** of W, not a smaller K and not a later slice. The task omits the full application's all-gather. Respect `config["n"]` and tensor strides as required by the supplied input; do not infer or cache output contents. Nearby M values also matter in correctness and final review.

`prepare` may keep weight-only reusable representations. BF16 weights must remain original precision or be compressed losslessly. The explicit router exception permits FP32 weights as BF16 hi+lo with FP32 accumulation at any M. BF16-only router weights do not meet the contract. BF16 activations remain BF16; native FP32 router arithmetic may promote those exact BF16 values.

Snapshot limits: preparation ≤300 seconds and whole judge run ≤10 minutes including compilation. The current memory cap and complete dependency/file-size restrictions have **not been authenticated**. Do not assume unlimited state or copy a competitor's reported cap into a compliance claim.

## All 30 scored configurations

A matrix shape below means `(M,N,K)` for `X[M,K] @ W[N,K].T`. Every configuration carries weight 62.

| Load | Layer | M | N | K | Weight | Our latency |
|---|---|---:|---:|---:|---:|---|
| decode 1 | qkv_gate | 1 | 2304 | 8192 | 62 | TBD |
| decode 1 | o | 1 | 8192 | 1024 | 62 | TBD |
| decode 1 | pooler | 1 | 4096 | 8192 | 62 | TBD |
| decode 1 | unpooler | 1 | 1024 | 4096 | 62 | TBD |
| decode 1 | router | 1 | 256 | 8192 | 62 | TBD |
| decode 16 | qkv_gate | 16 | 2304 | 8192 | 62 | TBD |
| decode 16 | o | 16 | 8192 | 1024 | 62 | TBD |
| decode 16 | pooler | 16 | 4096 | 8192 | 62 | TBD |
| decode 16 | unpooler | 16 | 8192 | 4096 | 62 | TBD |
| decode 16 | router | 16 | 256 | 8192 | 62 | TBD |
| decode 32 | qkv_gate | 32 | 2304 | 8192 | 62 | TBD |
| decode 32 | o | 32 | 8192 | 1024 | 62 | TBD |
| decode 32 | pooler | 32 | 4096 | 8192 | 62 | TBD |
| decode 32 | unpooler | 32 | 8192 | 4096 | 62 | TBD |
| decode 32 | router | 32 | 256 | 8192 | 62 | TBD |
| verify 1 | qkv_gate | 8 | 2304 | 8192 | 62 | TBD |
| verify 1 | o | 8 | 8192 | 1024 | 62 | TBD |
| verify 1 | pooler | 8 | 4096 | 8192 | 62 | TBD |
| verify 1 | unpooler | 8 | 1024 | 4096 | 62 | TBD |
| verify 1 | router | 8 | 256 | 8192 | 62 | TBD |
| verify 16 | qkv_gate | 128 | 2304 | 8192 | 62 | TBD |
| verify 16 | o | 128 | 8192 | 1024 | 62 | TBD |
| verify 16 | pooler | 128 | 4096 | 8192 | 62 | TBD |
| verify 16 | unpooler | 128 | 8192 | 4096 | 62 | TBD |
| verify 16 | router | 128 | 256 | 8192 | 62 | TBD |
| verify 32 | qkv_gate | 256 | 2304 | 8192 | 62 | TBD |
| verify 32 | o | 256 | 8192 | 1024 | 62 | TBD |
| verify 32 | pooler | 256 | 4096 | 8192 | 62 | TBD |
| verify 32 | unpooler | 256 | 8192 | 4096 | 62 | TBD |
| verify 32 | router | 256 | 256 | 8192 | 62 | TBD |

All latency cells intentionally remain unmeasured. Generated comparison reports should provide SGLang and candidate measurements, paired speedups and effective bandwidth for every row, retaining raw data and run metadata. Neither the task's theoretical floor nor a competitor's profile belongs in our measured-latency column.

## Correctness

The supplied reference computes in FP64 then casts to the required output dtype. Checks compare the candidate to that reference:

- Whole-output relative L2 ≤`1e-2` for BF16 projections and ≤`1e-4` for router.
- Any row ≤5 times its corresponding bound; near-zero rows use a denominator floor based on one tenth of the average reference row norm, as implemented in the preserved `checks.py`.
- Single-nonzero probe rows, whose powers-of-two activations expose original weight columns, must meet relative error ≤`1e-3`.
- Required shape/dtype/device must match and output values must be finite.

The generator uses BF16 normal activations, roughly one probe row in eight, at least one for M≥8, and an outlier distribution scaling about 1% of channels by 30. These descriptions inform coverage, never input-specific shortcuts or seed dispatch. Use unseen weight and activation seeds for local validation.

| Published required case | M | Layer | Distribution |
|---|---:|---|---|
| 1 | 1 | qkv_gate | normal |
| 2 | 7 | pooler | normal |
| 3 | 33 | o | normal |
| 4 | 130 | unpooler | normal |
| 5 | 5 | unpooler | normal |
| 6 | 8 | router | normal |
| 7 | 9 | router | outliers |
| 8 | 512 | qkv_gate | outliers |
| 9 | 1000 | unpooler | outliers |

Also validate all scored cases, graph replay with fresh activation data, separate weight preparations, output retention, zero rows, signed probes and nearby M. CPU checks of the same Python interface do not validate GPU kernels. Local checks using these mirrored routines are not official judge runs.

## Baseline and scoring

The preserved baseline names SGLang commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. BF16 projections call SGLang's `_bf16_gemm_dispatch_impl`; the router calls FP32 `F.linear` through M=8 and two BF16 hi/lo GEMMs with FP32 accumulation above M=8. Run this baseline in the official image; do not rename a substitute PyTorch baseline as SGLang.

For each load l, `B_l = Σ_i 62*b_i` and `C_l = Σ_i 62*c_i` for its five projections. The overall score is:

`S = exp((1/6) * Σ_l log(B_l/C_l))`.

The displayed time is the geometric-mean baseline load time divided by S. The historical mirrored API records 4830.8 µs; this is not a new measurement. The geometric mean of individual kernel ratios and the arithmetic average of load speedups are different metrics and must not be substituted.

The snapshot describes same-GPU alternating candidate/baseline CUDA-graph replays, fresh inputs, cold L2 before each timed replay, paired order-swapped repetitions, median ratios and subtraction of fixed graph replay cost (approximately 4 µs in the judge). Exact runner implementation and current calibration are unavailable. The local harness approximates that protocol and reports its own calibration; **local measurements are provisional and not judge-identical**.

## Compliance boundary

Branching on public shape/configuration metadata is allowed. Host synchronization, tensor-value printing, input-value-dependent allocation/dispatch, output caching, hidden-state tricks, reference imports, harness inspection/modification, external resources during evaluation and prohibited lower precision are not allowed. CUDA-graph replay must compute from current inputs each time. Reusable state may derive from preparation weights, not timed inputs.

The mirrored agent guide says a human must accept the rules through `okc login`. Never automate that acceptance or send credentials in chat. The user has expressly prohibited automatic competition submission; a ready artifact still needs explicit approval before `okc submit`.
