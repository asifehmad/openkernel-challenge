# Optimization plan

## Phase A: prepare for an evidence-driven H200 session

Keep a self-contained `muse_bf16_proj/submission.py` and separate research, testing and timing infrastructure. The candidate must consume the supplied activations and prepared weights on every call. Preserve the original reference/check sources and record provenance. CPU syntax, static analysis and score-math tests can validate infrastructure; they cannot establish GPU compilation, graph safety, numerical correctness or speed.

The initial candidate tests deliberately simple alternatives: streamed M=1 GEMV, small-M split-K tensor-core GEMM, a library path for large BF16 GEMM, exact small-M FP32 router reduction, and fused BF16 hi+lo router accumulation. No path is a measured improvement yet. Keep changes modular enough that the benchmark harness can compare versions without changing the submission interface.

## Phase B: establish trustworthy measurements first

1. Obtain H200 access through the user's chosen provider. Do not rent paid hardware or accept competition terms automatically.
2. Record GPU model, compute capability, SM count, memory, driver, clocks, power/thermal state and other GPU processes. Record the official image's immutable digest, PyTorch/CUDA/Triton/SGLang versions and compiler versions. The tag `lmsysorg/sglang:dev-muse-spark-open` is mutable.
3. Fetch fresh official task files with the official CLI and compare their hashes and contract against [contract.md](contract.md). Read current full rules, preparation limits, memory limits, available dependencies and live leaderboard. The memory cap is currently unknown.
4. Execute local reference checks for all nine published correctness cases, all 30 scored configurations, unseen seeds, zero inputs, signed one-hot/power-of-two probes and nearby batch counts. Validate dtype, shape, finiteness, aggregate/row/probe bounds and output retention.
5. Exercise capture/replay with changed activation contents between replays; validate the replayed result against the reference. A successful eager call alone is insufficient. Every call must initialize its own reduction scratch and compute fresh outputs.
6. Measure preparation wall time and incremental allocated/reserved GPU memory, accounting for retained original weights and all extra representations. Check a cold process too; cached JIT output can hide setup cost. Leave margin below the mirrored 300-second preparation and 10-minute whole-run caps.
7. Measure the supplied SGLang baseline and candidate on the same H200, input stream and process. Start with an A/A control. Use paired order-swapped CUDA-graph trials, cold L2 before each timed replay, sufficient warmup, repeated captures and recorded raw measurements. Local overhead subtraction/calibration must be explicit; do not assume a hardcoded 4 µs is the correct local correction.
8. Verify local score arithmetic independently: weighted sums within each load, then geometric mean of six load speedups. Label local scores as provisional; the harness is not proven judge-identical.

No `okc submit` is automatic. Any online check requires the user's existing authentication and human rule acceptance; follow the user's authorization and preserve quota. A local reference run is not an official judge pass.

## Prioritize weighted impact

For load l, `T_l = sum_i(weight_i * t_i)`; here every weight is 62. The overall speedup is `exp(mean_l(log(B_l / T_l)))`. A small change has approximate score sensitivity `-weight_i * delta_t_i / (6*T_l)`. Rank configurations by their contribution and attainable reduction, not by their standalone speedup. A large absolute saving on a small fraction of the score may matter less than a moderate gain across several loads.

The same five layers appear in each load, but the unpooler changes width between M=8 and M=16. Treat the sliced cases separately. Pooler/unpooler stream large weights and deserve early attention, but let measured load contributions decide the actual order. Router arithmetic is smaller yet its fixed latency may be material, especially in the smallest loads.

## Controlled experiments

| Priority | Hypothesis | Controlled variables | Acceptance criteria |
|---|---|---|---|
| 1 | Initial paths can satisfy the contract | Correctness first; small router native FP32; full-precision BF16 inputs/weights | All applicable checks, fresh-input replay and output retention pass |
| 2 | M1 GEMV improves streaming | Output tile width, K chunk, warps, vector width; one variable per comparison | Lower paired cold latency and aggregate score; no spills/regression on nearby M |
| 3 | Small-M tensor-core path can exploit reuse | M/N/K tiles, split factors 1/2/4/8, warps and stages | Account for both partial and reduction kernels; compare against cuBLAS and GEMV |
| 4 | Fused hi+lo reduces router launch cost | Exact FP32 vs two tensor-core contributions; tile/split geometry | Router 1e-4 aggregate and probe bounds across fresh/outlier inputs; end-to-end gain |
| 5 | Large-M library or custom tensor-core path wins by shape | cuBLAS baseline, conventional Triton, then persistent/TMA | Correctness plus improvements across captures/clock conditions, including total extra traffic |
| 6 | Preparation layout improves cold streaming | Lossless transpose/packing; retain exact dtype/bit pattern | Verify reversible representation and all value edge cases; include preparation memory/time |
| 7 | Lossless compression reduces enough bytes to pay for decoding | Codec, escape handling and metadata; small M first | Bit-exact BF16 reconstruction; arbitrary unseen weights; measured aggregate gain |
| 8 | Hopper-specific pipeline improves an identified bottleneck | TMA stages, producer/consumer allocation, WGMMA tile and persistent grid | Nsight evidence supports diagnosis; image compatibility and compilation budget remain valid |

Do not prioritize compression for large M simply because it helps GEMV. Do not use split-K by default when the output grid already fills the GPU: FP32 partial writes and a second reduction can dominate the saving. Likewise, a higher occupancy percentage does not prove a faster tensor-core kernel.

For Nsight Compute, collect register and shared-memory footprints, local-memory spills, occupancy limits, achieved occupancy, DRAM bytes/throughput, L2 traffic, tensor-pipe activity and stall reasons where the installed version permits. Query available metric names rather than assuming names from another release. Profile a selected shape after the correctness/timing pass; profiler timings are not interchangeable with benchmark timings.

## Retain or revert

For each experiment, record source SHA/hash, environment/image digest, affected configurations, hypothesis, exact major change, correctness outcomes, per-shape raw timing and distributions, preparation cost and score impact in [experiments.md](experiments.md). Compare in the same session and repeat promising wins. Changes within measurement noise are inconclusive. Keep the previous best validated artifact available and restore it when correctness fails or aggregate performance regresses.

Stop when the agreed rental budget is reached or repeated controlled changes fail to exceed the noise floor. The target is below 2.93 ms, with 2.5 ms as a stretch target; neither is a forecast. Only independently verified judge measurements can establish that the competition target has been beaten.
