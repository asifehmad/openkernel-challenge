"""Submission packaging/contract checks on CPU, not CUDA validation."""
import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "torch required for reduced CPU submission checks")
class SubmissionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from muse_bf16_proj import submission
        self.torch, self.submission = torch, submission
        torch.set_num_threads(2)
        rng = torch.Generator().manual_seed(823901)
        self.weights = {layer: torch.randn(37, 64, generator=rng).to(
            torch.float32 if layer == "router" else torch.bfloat16)
            for layer in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        self.state = submission.prepare(self.weights)
        self.entries = (submission.kernel_fn, submission.single_pass_kernel_fn,
                        submission.gemv_kernel_fn, submission.torch_kernel_fn)

    def data(self, m=8, n=13, layer="unpooler"):
        x = self.torch.randn(m, 64).bfloat16()
        return {"x": x, "config": dict(m=m, n=n, k=64, layer=layer)}

    def test_each_entry_obeys_output_slice_and_retains_outputs(self):
        from benchmarks.contract import check_implementation
        for entry in self.entries:
            for m in (1, 8, 9, 32, 33):
                data = self.data(m=m, n=13 if m <= 8 else 37)
                output = entry(data, self.state)
                ok, message = check_implementation(data, output, self.weights)
                self.assertTrue(ok, message)
                retained = output.clone()
                data["x"].zero_()
                fresh = entry(data, self.state)
                self.assertEqual(fresh.count_nonzero().item(), 0)
                self.assertTrue(self.torch.equal(output, retained))
                self.assertNotEqual(output.data_ptr(), fresh.data_ptr())

    def test_router_output_dtype_and_no_weight_state_mutation(self):
        from benchmarks.contract import check_implementation
        before = {name: tensor.clone() for name, tensor in self.state.items()}
        for entry in self.entries:
            data = self.data(m=9, layer="router")
            data["x"][:, 3] *= 32
            output = entry(data, self.state)
            self.assertEqual(output.dtype, self.torch.float32)
            ok, message = check_implementation(data, output, self.weights)
            self.assertTrue(ok, message)
        self.assertEqual(set(before), set(self.state))
        for name in before:
            self.assertTrue(self.torch.equal(before[name], self.state[name]), name)

    def test_invalid_metadata_rejected_before_raw_pointer_launch(self):
        original = self.data()
        changes = ({"m": 7}, {"k": 65}, {"n": 38}, {"n": 0},
                   {"m": -1}, {"m": True}, {"layer": "unknown"})
        for change in changes:
            data = copy.copy(original)
            data["config"] = {**original["config"], **change}
            for entry in self.entries:
                with self.subTest(change=change, entry=entry.__name__), self.assertRaises(ValueError):
                    entry(data, self.state)
        original["x"] = original["x"].float()
        with self.assertRaisesRegex(ValueError, "activation"):
            self.submission.kernel_fn(original, self.state)

    def test_prepare_rejects_wrong_weight_precision(self):
        weights = dict(self.weights)
        weights["router"] = weights["router"].bfloat16()
        with self.assertRaisesRegex(ValueError, "router"):
            self.submission.prepare(weights)
        weights = dict(self.weights)
        weights["unpooler"] = weights["unpooler"].float()
        with self.assertRaisesRegex(ValueError, "unpooler"):
            self.submission.prepare(weights)

    def test_submission_runs_as_an_isolated_single_file(self):
        code = '''
import importlib.util
import sys
import torch
spec = importlib.util.spec_from_file_location("standalone_submission", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
weights = {name: torch.ones((7, 64), dtype=torch.float32 if name == "router" else torch.bfloat16)
           for name in ("qkv_gate", "o", "pooler", "unpooler", "router")}
state = module.prepare(weights)
for name in weights:
    data = {"x": torch.ones((3, 64), dtype=torch.bfloat16),
            "config": dict(m=3, n=5, k=64, layer=name)}
    for entry in ("kernel_fn", "single_pass_kernel_fn", "gemv_kernel_fn", "torch_kernel_fn"):
        out = getattr(module, entry)(data, state)
        assert out.shape == (3, 5) and bool((out == 64).all())
print("standalone CPU import and calls passed; GPU unverified")
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            shutil.copyfile(ROOT / "muse_bf16_proj/submission.py", path)
            result = subprocess.run([sys.executable, "-I", "-c", code, str(path)],
                                    cwd=directory, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("standalone CPU", result.stdout)
