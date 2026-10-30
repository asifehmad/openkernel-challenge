"""CPU tests of score aggregation and failure behavior, not GPU correctness."""
import copy
import math
import unittest

from benchmarks.benchmark import paired_summary
from benchmarks.common import aggregate, scored_shapes
from benchmarks.compare import render


def rows():
    return [dict(**s, baseline_us=10.0, candidate_us=5.0) for s in scored_shapes()]


class ScoringTests(unittest.TestCase):
    def test_thirty_shapes_and_unpooler_boundary(self):
        shapes = scored_shapes()
        self.assertEqual(len(shapes), 30)
        self.assertEqual(len({(s['m'], s['layer']) for s in shapes}), 30)
        for s in shapes:
            if s['layer'] == 'unpooler':
                self.assertEqual(s['n'], 1024 if s['m'] <= 8 else 8192)

    def test_uniform_speedup_and_units(self):
        score = aggregate(rows())
        self.assertAlmostEqual(score['speedup'], 2)
        self.assertAlmostEqual(score['baseline_ms'], 3.1)
        self.assertAlmostEqual(score['candidate_ms'], 1.55)
        self.assertEqual(len(score['loads']), 6)

    def test_load_geometric_mean_not_shape_average(self):
        data = rows()
        for r in data:
            r['candidate_us'] = 10.0
        # One kernel in one load is 5x slower; load total is 9/5 of baseline.
        data[0]['candidate_us'] = 50.0
        self.assertAlmostEqual(aggregate(data)['speedup'], (5 / 9) ** (1 / 6))

    def test_reject_bad_measurements_and_missing_shapes(self):
        for value in (0, -1, math.inf, math.nan, None, True, 'TBD'):
            data = rows()
            data[0]['candidate_us'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                aggregate(data)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            aggregate(rows()[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(rows() + [rows()[0]])

    def test_reject_wrong_dimensions_and_weight(self):
        for name in ('m', 'n', 'k', 'weight', 'load'):
            data = rows()
            data[0][name] = 'wrong'
            with self.subTest(name=name), self.assertRaises(ValueError):
                aggregate(data)

    def test_csv_string_numbers_and_report(self):
        data = [{k: str(v) for k, v in r.items()} for r in rows()]
        self.assertAlmostEqual(aggregate(data)['speedup'], 2)
        self.assertIn('not an official judge result', render(data))

    def test_swapped_pair_formula_and_nonpositive_rejection(self):
        samples = [dict(corrected_us=dict(baseline_ab=10, candidate_ab=4,
                                         candidate_ba=5, baseline_ba=8))]
        summary = paired_summary(samples)
        self.assertAlmostEqual(summary['speedup'], 2)
        self.assertAlmostEqual(summary['baseline_us'], 9)
        self.assertAlmostEqual(summary['candidate_us'], 4.5)
        # A single extreme pair must not dominate a median of several pairs.
        noisy = samples + samples + [dict(corrected_us=dict(
            baseline_ab=1000, candidate_ab=1, candidate_ba=1, baseline_ba=1000))]
        self.assertAlmostEqual(paired_summary(noisy)['speedup'], 2)
        # Reverse AB/BA order within a pair; its geometric pairing is invariant.
        reverse = [dict(corrected_us=dict(baseline_ab=8, candidate_ab=5,
                                         candidate_ba=4, baseline_ba=10))]
        self.assertEqual(paired_summary(reverse), summary)
        bad = copy.deepcopy(samples)
        for value in (0, -1, math.nan, math.inf):
            bad[0]['corrected_us']['baseline_ab'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                paired_summary(bad)
        with self.assertRaises(ValueError):
            paired_summary([])


if __name__ == '__main__':
    unittest.main()
