"""CPU fixture tests; no CLI service, GPU measurement or official validation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from benchmarks.bundle import TASK_FILES, require_matching_contract, resolve_bundle
from benchmarks.common import file_sha256, scored_shapes
from benchmarks.contract import validation_cases
from scripts.summarize_candidates import CONTEXT, collect


def bundle(path):
    path.mkdir(parents=True)
    for name in TASK_FILES:
        (path / name).write_text("synthetic fixture " + name)
    return path


class BundleTests(unittest.TestCase):
    def test_fetch_script_accepts_synthetic_cli_layout_and_refuses_overwrite(self):
        # Stub only the unavailable CLI, never contact a service or claim an
        # official download. Exercise the real shell/resolver integration.
        repository = Path(__file__).resolve().parents[1]
        for nesting in ('', 'muse_bf16_proj'):
            with self.subTest(nesting=nesting), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                binary = root / 'okc'
                binary.write_text('#!/bin/bash\nset -e\ntest "$1" = init\ntest "$2" = muse_bf16_proj\n'
                                  'mkdir -p "${FIXTURE_NESTING:-.}"\ncd "${FIXTURE_NESTING:-.}"\n'
                                  'for file in TASK.md AGENTS.md baseline.py reference.py checks.py submission.py; do\n'
                                  '  printf "synthetic fixture\\n" > "$file"\ndone\n')
                binary.chmod(0o755)
                environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], FIXTURE_NESTING=nesting)
                destination = root / 'download'
                command = ['bash', str(repository / 'scripts/fetch_official.sh'), str(destination)]
                result = subprocess.run(command, env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(str(resolve_bundle(destination)), result.stdout)
                repeat = subprocess.run(command, env=environment, capture_output=True, text=True)
                self.assertNotEqual(repeat.returncode, 0)
                self.assertIn('already exists', repeat.stderr)

    def test_direct_and_nested_layout_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = bundle(root / 'nested' / 'muse_bf16_proj')
            self.assertEqual(resolve_bundle(root), task)
            self.assertEqual(resolve_bundle(task), task)
            bundle(root / 'another')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                resolve_bundle(root)

    def test_missing_files_symlink_escape_and_contract_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, snapshot = bundle(root / 'task'), bundle(root / 'snapshot')
            require_matching_contract(task, snapshot)
            (task / 'submission.py').write_text('different starter is expected')
            require_matching_contract(task, snapshot)
            (task / 'checks.py').write_text('changed tolerance')
            with self.assertRaisesRegex(ValueError, 'checks.py'):
                require_matching_contract(task, snapshot)
            (task / 'checks.py').unlink()
            with self.assertRaisesRegex(ValueError, 'found 0'):
                resolve_bundle(task)
            (task / 'checks.py').symlink_to(snapshot / 'checks.py')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                resolve_bundle(task)


def measured_fixture(run, entry='kernel_fn', speedup=1.5):
    """Synthetic records, not results. All values are arbitrary test fixtures."""
    run.mkdir(exist_ok=True)
    (run / 'submission.py').write_text('# synthetic candidate, not executable\n')
    sha = file_sha256(run / 'submission.py')
    hashes = {'reference.py': 'fixture-reference', 'checks.py': 'fixture-checks'}
    meta = {key: 'synthetic' for key in CONTEXT}
    meta.update(result_type='local_candidate_benchmark', candidate_entry_point=entry,
                submission_sha256=sha, official_sha256=hashes,
                arguments={k: 'fixture' for k in ('seed','correction','eviction_mib','repeats','warmup','aa_repeats')})
    shapes = [dict(**s, baseline_us=10., candidate_us=10./speedup,
                   aa_before_diagnostics={'passed': True}, aa_after_diagnostics={'passed': True}) for s in scored_shapes()]
    (run / entry).mkdir()
    (run / entry / 'results.json').write_text(json.dumps(dict(status='complete', metadata=meta, shapes=shapes)))
    cases = [dict(seed=seed, m=m, layer=layer, distribution=dist, status='pass', graph='pass')
             for seed in (1,2) for m,layer,dist in validation_cases()]
    (run / f'correctness-{entry}.json').write_text(json.dumps(dict(
        submission_sha256=sha, device='cuda', quick=False, backend='submission',
        candidate_entry_point=entry, failed=0, passed=len(cases), cases=cases, official_file_hashes=hashes)))
    (run / f'smoke-{entry}.json').write_text(json.dumps(dict(submission_sha256=sha, status='pass',
        external_event_graph_executed=True, entry_point=entry, task_sha256=hashes)))


class SelectionTests(unittest.TestCase):
    def test_preserves_sources_and_ranks_only_comparable_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            measured_fixture(a, speedup=1.5)
            measured_fixture(b, 'single_pass_kernel_fn', 1.51)
            before = (a / 'submission.py').read_bytes()
            report = collect([a,b])
            self.assertEqual(report['best_observed']['entry_point'], 'single_pass_kernel_fn')
            self.assertTrue(report['comparison_inconclusive'])
            self.assertFalse(report['default_implementation_changed'])
            self.assertEqual((a / 'submission.py').read_bytes(), before)
            path = b / 'single_pass_kernel_fn' / 'results.json'
            doc = json.loads(path.read_text())
            doc['metadata']['torch'] = 'different-runtime'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'different hardware'):
                collect([a,b])

    def test_rejects_incomplete_validation_graph_failure_and_source_changes(self):
        for mutation in ('incomplete', 'graph-fail', 'wrong-task', 'source'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                measured_fixture(run)
                path = run / 'correctness-kernel_fn.json'
                doc = json.loads(path.read_text())
                if mutation == 'incomplete':
                    doc['cases'] = doc['cases'][:1]
                    doc['passed'] = 1
                elif mutation == 'graph-fail':
                    doc['cases'][0]['graph'] = 'unrun (CPU)'
                elif mutation == 'wrong-task':
                    doc['official_file_hashes']['checks.py'] = 'different'
                else:
                    (run / 'submission.py').write_text('changed after measurement')
                path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    collect([run])

    def test_rejects_failed_aa_and_partial_scoring(self):
        for mutation in ('aa', 'partial'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                measured_fixture(run)
                path = run / 'kernel_fn' / 'results.json'
                doc = json.loads(path.read_text())
                if mutation == 'aa':
                    doc['shapes'][0]['aa_before_diagnostics']['passed'] = False
                else:
                    doc['shapes'].pop()
                path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    collect([run])
