# TinyGEMM experiment: BF16 QKV/pooler, batch 1 and 8

Branch `experiment/muse-flashinfer-tinygemm`, based on verified `4f36b51ba415c3b76868d8ec38c9b6dc0b9c3eb4` (job `5fbcf345f0454920`, **1.117x**, **9/9**, **4,324.7 µs**).

## Why this is different from the failed WMMA design

The previous first-party experimental CUDA WMMA kernel, job `9f4bf6209cf044a8`, passed **9/9** but regressed heavily to **0.631x (7,661.0 µs)**. Its warp/CTA scheduling, shared-memory loads and 16-way split-K overhead were inefficient. **Do not reuse or merge it.**

FlashInfer's documented `flashinfer.gemm.tinygemm_bf16(input, weight, out)` is a different, SM90+ optimized **warp-specialized** GEMM with TMA asynchronous bulk loads and BF16 tensor-core instructions with FP32 accumulation. The public API explicitly targets **M=1–8**, contiguous BF16 inputs and weights, K divisible by 64, N divisible by 16; all target cases meet these conditions. It directly computes the original `x @ W.T` without reduced-precision weights, caching outputs, or relying on input contents.

Only **QKV** and **pooler**, at **M=1 and M=8**, change. The original router (now strong at M=1,8,16) and all M>8 BF16 kernels are untouched. `prepare()` optionally resolves the installed `flashinfer.gemm.tinygemm_bf16` function. If unavailable it uses the unchanged tested fallback—thus a passing but unchanged check would not prove TinyGEMM actually ran. **Inspect the official profile for `tinygemm` kernel names, and verify import on VPS.**

**This source is not CUDA-tested.** We cannot claim the API exists in the actual evaluator image, works during CUDA graph capture, or beats the stable kernel. Current FlashInfer public docs: https://docs.flashinfer.ai/generated/flashinfer.gemm.tinygemm_bf16.html .

## Steps on VPS

```bash
cd /root/openkernel-challenge
git fetch origin
git switch -c experiment/muse-flashinfer-tinygemm --track origin/experiment/muse-flashinfer-tinygemm
python3 -m py_compile muse_bf16_proj/submission.py
bash scripts/run_tests.sh
python3 - <<'PY'
try:
    from flashinfer.gemm import tinygemm_bf16
    print("FlashInfer TinyGEMM import OK", tinygemm_bf16)
except Exception as exc:
    print("FlashInfer TinyGEMM unavailable locally:", type(exc).__name__, str(exc))
PY
```

If available in the evaluator image, use an off-board `okc check` at the owner's discretion; require **9/9** and a net score improvement beyond noise before promoting. Check all 30 shapes, especially QKV M1/M8 and pooler M1/M8. Verify the profiler actually invokes the optimized kernel. If missing, do not spend more official checks; first verify evaluator support. The stable feature branch remains at **1.117x**.

## Longer-term plan

TinyGEMM addresses only 4 of 30 scored cases. Reaching the reported **1.60x** leaderboard level requires larger GEMM speedups for M=16/32/128/256 and all BF16 layers; one package change is not sufficient. Later evaluate other legitimate installed backends such as FlashInfer's CUTLASS/cuBLASLt BF16 GEMM using isolated graph timing and original BF16 weights. Never import reference code, cache activations, or lower BF16 precision.

No `okc submit` performed.
