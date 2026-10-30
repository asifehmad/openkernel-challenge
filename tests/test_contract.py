"""CPU tests for manifest boundaries and adversarial numerical checks."""
import importlib.util
import unittest

from benchmarks import contract


class ManifestTests(unittest.TestCase):
    def test_unpooler_width_boundary(self):
        self.assertEqual(contract.out_features("unpooler", 8), 1024)
        self.assertEqual(contract.out_features("unpooler", 9), 8192)
        self.assertEqual(contract.out_features("pooler", 8), 4096)

    def test_scored_shapes_and_special_cases(self):
        rows = contract.configurations()
        self.assertEqual(len(rows), 30)
        self.assertEqual({r["m"] for r in rows}, {1, 8, 16, 32, 128, 256})
        self.assertEqual({r["weight"] for r in rows}, {62})
        self.assertEqual(len(contract.SPECIAL_CASES), 9)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU torch not installed")
class NumericalCheckTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.weights = {"router": torch.eye(16, dtype=torch.float32)}
        self.data = {"x": torch.ones(200, 16, dtype=torch.bfloat16),
                     "config": {"m": 200, "n": 16, "k": 16, "layer": "router"}}

    def test_reference_passes_and_shape_dtype_nonfinite_fail(self):
        t = self.torch
        correct = contract.ref_kernel(self.data, self.weights)
        self.assertTrue(contract.check_implementation(self.data, correct, self.weights)[0])
        self.assertFalse(contract.check_implementation(self.data, correct.bfloat16(), self.weights)[0])
        self.assertFalse(contract.check_implementation(self.data, correct[:1], self.weights)[0])
        correct[0, 0] = t.nan
        self.assertFalse(contract.check_implementation(self.data, correct, self.weights)[0])

    def test_row_error_cannot_hide_in_global_average(self):
        output = contract.ref_kernel(self.data, self.weights)
        output[0] *= 1.001
        passed, message = contract.check_implementation(self.data, output, self.weights)
        self.assertFalse(passed)
        self.assertIn("row relative", message)

    def test_probe_bound_stricter_than_bf16_global_bound(self):
        t = self.torch
        w = {"o": t.ones(16, 16, dtype=t.bfloat16)}
        data = {"x": t.zeros(16, 16, dtype=t.bfloat16),
                "config": {"m": 16, "n": 16, "k": 16, "layer": "o"}}
        data["x"][:, 0] = 1
        out = contract.ref_kernel(data, w)
        out[0, :] = 1.0078125
        passed, message = contract.check_implementation(data, out, w)
        self.assertFalse(passed)
        self.assertIn("probe relative", message)

    def test_zero_reference_rejects_nonzero(self):
        self.data["x"].zero_()
        out = contract.ref_kernel(self.data, self.weights)
        self.assertTrue(contract.check_implementation(self.data, out, self.weights)[0])
        out[0, 0] = 1e-6
        self.assertFalse(contract.check_implementation(self.data, out, self.weights)[0])


if __name__ == "__main__":
    unittest.main()
