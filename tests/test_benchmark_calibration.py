"""Synthetic timing data only: these are not GPU measurements."""
import copy
import tempfile
import unittest
from pathlib import Path

from benchmarks.timing import aa_diagnostics, paired_summary
from benchmarks.compare import render
from benchmarks.common import scored_shapes
from scripts.compare_official import FILES, compare


def samples(a=10., b=10., overhead=1.):
    raw = dict(baseline_ab=a, candidate_ab=b, candidate_ba=b, baseline_ba=a)
    return [dict(raw_us=raw.copy(), corrected_us={k: v-overhead for k, v in raw.items()},
                 overhead_us=overhead) for _ in range(8)]


class CalibrationTests(unittest.TestCase):
    def test_aa_accepts_equal_payloads_but_not_as_proof_of_absolute_time(self):
        for correction in ('none', 'empty'):
            result = aa_diagnostics(samples(), correction)
            self.assertTrue(result['passed'])
            self.assertEqual(result['variants'][correction]['speedup'], 1.)

    def test_correction_rejected_when_overhead_dominates_or_is_nonpositive(self):
        for overhead in (3., 10., 11.):
            self.assertFalse(aa_diagnostics(samples(overhead=overhead), 'empty')['passed'])
            # Raw calibration can remain usable; do not silently subtract.
            self.assertTrue(aa_diagnostics(samples(overhead=overhead), 'none')['passed'])

    def test_biased_and_noisy_aa_rejected(self):
        self.assertFalse(aa_diagnostics(samples(a=11., b=10.))['passed'])
        noisy = samples()
        for i, sample in enumerate(noisy):
            for key in ('candidate_ab', 'candidate_ba'):
                sample['raw_us'][key] *= 1.3 if i % 2 else .8
        self.assertFalse(aa_diagnostics(noisy)['passed'])

    def test_explicit_raw_and_corrected_reports(self):
        data = samples(a=10., b=5., overhead=1.)
        self.assertEqual(paired_summary(data, 'raw_us')['speedup'], 2.)
        self.assertEqual(paired_summary(data, 'corrected_us')['speedup'], 2.25)
        broken = copy.deepcopy(data)
        broken[0]['corrected_us']['candidate_ab'] = -1
        with self.assertRaises(ValueError):
            paired_summary(broken)

    def test_aa_report_cannot_be_mistaken_for_candidate_improvement(self):
        rows = [dict(**s, baseline_us=10., candidate_us=10.) for s in scored_shapes()]
        report = render(rows, {'result_type': 'local_aa_calibration'})
        self.assertIn('not a candidate speedup', report)
        self.assertNotIn('Optimization priorities', report)
        self.assertIn('Score elasticity', render(rows))


class BundleComparisonTests(unittest.TestCase):
    def test_reports_changed_contract_and_expected_candidate_difference(self):
        with tempfile.TemporaryDirectory() as d:
            fresh, local = Path(d) / 'fresh', Path(d) / 'local'
            fresh.mkdir()
            local.mkdir()
            for name in FILES:
                (fresh / name).write_text('starter')
                (local / name).write_text('starter')
            (local / 'submission.py').write_text('candidate')
            (fresh / 'checks.py').write_text('new official checker')
            rows = {r['file']: r for r in compare(fresh, local)}
            self.assertTrue(rows['TASK.md']['identical'])
            self.assertFalse(rows['checks.py']['identical'])
            self.assertIn('review required', rows['checks.py']['interpretation'])
            self.assertIn('by design', rows['submission.py']['interpretation'])
            (fresh / 'TASK.md').unlink()
            with self.assertRaises(ValueError):
                compare(fresh, local)
