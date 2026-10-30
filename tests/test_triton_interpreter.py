"""CPU execution of Triton kernel algorithms, NOT CUDA compilation or GPU proof.

Each case launches a separate process with TRITON_INTERPRET=1 before importing
Triton. This checks arithmetic, strides and masked tails. It does not validate
Hopper code generation, scheduling, graphs, resource limits or performance.

Triton 3.6.0 stores BF16 as uint16 and create_dot does not decode it. Native
BF16 dot tests are explicitly skipped for that version. Widened tests preserve
BF16 values exactly in FP32 storage to exercise the original kernel algorithms,
without claiming BF16 tensor-core validation. GEMV uses explicit FP32 conversion
and can be interpreted with native BF16 input storage.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "triton", "numpy"))


def _worker(case):
    sys.path.insert(0, str(ROOT))
    import torch
    import triton
    from benchmarks.contract import check_implementation
    from muse_bf16_proj import submission as candidate

    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("worker must start with TRITON_INTERPRET=1")
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(723619)

    def inputs(m, n, k, dtype, strided=True):
        stride = 2 if strided else 1
        x = torch.randn(m, k * stride, generator=rng).bfloat16()[:, ::stride]
        w = (torch.randn(n, k * stride, generator=rng) / k ** .5).to(dtype)[:, ::stride]
        # A power-of-two probe should retain the exact BF16 weight values.
        x[0].zero_()
        x[0, k - 1] = 64
        return x, w

    def check(x, w, y, layer):
        m, k = x.shape
        n = w.shape[0]
        data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
        passed, message = check_implementation(data, y, {layer: w})
        if not passed:
            raise AssertionError(message)
        print(f"INTERPRETER ONLY: {case}, {layer}, ({m},{n},{k}): {message}")

    if case == "gemv":
        for dtype, layer in ((torch.bfloat16, "o"), (torch.float32, "router")):
            for distribution in ("normal", "outliers", "zero", "signed", "probes"):
                m, n, k = 9, 7, 73
                x, w = inputs(m, n + 6, k, dtype)
                if distribution == "outliers":
                    x[:, 3] *= 32
                elif distribution == "zero":
                    x.zero_()
                elif distribution == "signed":
                    x.sign_()
                elif distribution == "probes":
                    x.zero_()
                    for row in range(m):
                        x[row, (row * 7) % k] = 32 if row % 2 else -64
                guard = torch.full((m*n + 8,), -123, dtype=dtype)
                y = guard[4:-4].view(m, n)
                candidate._gemv[(triton.cdiv(n, 4), m)](
                    x, w, y, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                    BN=4, BK=64, num_warps=4)
                check(x, w[:n], y, layer)
                if not bool((guard[:4] == -123).all() & (guard[-4:] == -123).all()):
                    raise AssertionError("GEMV wrote outside the output slice")
    elif case in {"split_gemm", "split_gemm_bf16"}:
        m, n, k = 19, 37, 151
        x, w = inputs(m, n, k, torch.bfloat16)
        if case == "split_gemm":
            # Same exact BF16 values, wider storage: bypasses interpreter
            # BF16 dot limitation, NOT an emulation of tensor-core arithmetic.
            x, w = x.float(), w.float()
        for split in (1, 3, 4):
            partial = torch.full((split, m, n), float("nan"), dtype=torch.float32)
            y = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
            candidate._split_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32), split)](
                x, w, partial, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                BM=16, BN=32, BK=32, SPLIT=split, num_warps=4)
            if not bool(torch.isfinite(partial).all()):
                raise AssertionError("partial tensor has unwritten or nonfinite values")
            candidate._finish_split[(triton.cdiv(m*n, 128),)](
                partial, y, m*n, split, B=128)
            check(x, w, y, "o")
    elif case in {"single_gemm", "single_gemm_bf16"}:
        for m, n, k in ((1, 7, 73), (8, 13, 97), (19, 37, 151), (32, 13, 19)):
            x, w = inputs(m, n + 5, k, torch.bfloat16)
            if case == "single_gemm":
                x, w = x.float(), w.float()
            guard = torch.full((m*n + 8,), -123, dtype=torch.bfloat16)
            y = guard[4:-4].view(m, n)
            candidate._single_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
                x, w, y, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                BM=16, BN=32, BK=32, num_warps=4)
            check(x, w[:n], y, "unpooler")
            if not bool((guard[:4] == -123).all() & (guard[-4:] == -123).all()):
                raise AssertionError("single-pass GEMM wrote outside the output slice")
    elif case in {"router_pair", "router_pair_bf16"}:
        m, n, k = 19, 37, 151
        x, w = inputs(m, n, k, torch.float32)
        hi = w.bfloat16().contiguous()
        lo = (w - hi.float()).bfloat16().contiguous()
        if case == "router_pair":
            x, hi, lo = x.float(), hi.float(), lo.float()
        y = torch.full((m, n), float("nan"), dtype=torch.float32)
        candidate._router_pair[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            x, hi, lo, y, m, n, k, x.stride(0), x.stride(1),
            BM=16, BN=32, BK=32, num_warps=4)
        check(x, w, y, "router")
    else:
        raise ValueError(case)


@unittest.skipUnless(AVAILABLE, "optional CPU interpreter requires torch, Triton and NumPy")
class InterpreterTests(unittest.TestCase):
    def run_case(self, case):
        environment = os.environ.copy()
        environment["TRITON_INTERPRET"] = "1"
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", case],
                                env=environment, text=True, capture_output=True, timeout=90,
                                cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("INTERPRETER ONLY", result.stdout)

    def test_gemv_bf16_and_fp32_strides_and_tails(self):
        self.run_case("gemv")

    def test_split_gemm_indexing_with_widened_bf16(self):
        self.run_case("split_gemm")

    def test_router_pair_indexing_with_widened_bf16(self):
        self.run_case("router_pair")

    def test_single_gemm_indexing_slices_with_widened_bf16(self):
        self.run_case("single_gemm")

    def require_bf16_dot_support(self):
        import triton
        if triton.__version__ == "3.6.0":
            self.skipTest("Triton 3.6.0 interpreter create_dot multiplies BF16 uint16 storage; GPU validation remains required")

    def test_native_bf16_split_gemm_interpreter(self):
        self.require_bf16_dot_support()
        self.run_case("split_gemm_bf16")

    def test_native_bf16_router_pair_interpreter(self):
        self.require_bf16_dot_support()
        self.run_case("router_pair_bf16")

    def test_native_bf16_single_gemm_interpreter(self):
        self.require_bf16_dot_support()
        self.run_case("single_gemm_bf16")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
    else:
        unittest.main()
