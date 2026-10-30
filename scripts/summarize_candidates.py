"""Retain the best observed local entry point; never edit evaluator dispatch.

Only full, matching GPU validation/benchmark records are eligible. This is not
an official result or a statistical guarantee that a close winner is faster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.common import aggregate, file_sha256  # noqa: E402
from benchmarks.contract import validation_cases  # noqa: E402

ENTRIES = ("kernel_fn", "single_pass_kernel_fn", "gemv_kernel_fn")
CONTEXT = ("gpu", "capability", "torch", "cuda", "triton", "sglang", "tf32_allowed",
           "bf16_reduced_precision_reduction", "official_sha256", "measurement")


def collect(run_roots):
    eligible = []
    context = None
    for run in map(Path, run_roots):
        source = run / "submission.py"
        if not source.is_file():
            raise ValueError(f"Missing retained submission: {source}")
        source_hash = file_sha256(source)
        found = False
        for entry in ENTRIES:
            result = run / entry / "results.json"
            if not result.is_file():
                continue
            found = True
            doc = json.loads(result.read_text())
            meta = doc.get("metadata", {})
            if doc.get("status") != "complete" or meta.get("result_type") != "local_candidate_benchmark":
                raise ValueError(f"Incomplete or non-candidate run: {result}")
            if meta.get("candidate_entry_point") != entry or meta.get("submission_sha256") != source_hash:
                raise ValueError(f"Entry point or retained source hash mismatch: {result}")
            for kind in ("smoke", "correctness"):
                report = run / f"{kind}-{entry}.json"
                if not report.is_file():
                    raise ValueError(f"Missing {kind} evidence: {report}")
                validation = json.loads(report.read_text())
                if validation.get("submission_sha256") != source_hash:
                    raise ValueError(f"Validation used a different source: {report}")
                if kind == "smoke":
                    valid = validation.get("status") == "pass" and validation.get("external_event_graph_executed") is True
                    valid = valid and validation.get("entry_point") == entry
                    hashes = validation.get("task_sha256", {})
                else:
                    valid = validation.get("device") == "cuda" and validation.get("quick") is False
                    cases = validation.get("cases", [])
                    valid = valid and validation.get("backend") == "submission" and validation.get("candidate_entry_point") == entry
                    valid = valid and validation.get("failed") == 0 and len(cases) > 0 and validation.get("passed") == len(cases)
                    valid = valid and all(c.get("status") == "pass" and c.get("graph") == "pass" for c in cases)
                    seeds = {c.get("seed") for c in cases}
                    valid = valid and len(seeds) >= 2 and None not in seeds
                    expected = set(validation_cases())
                    valid = valid and all(expected <= {(c.get("m"), c.get("layer"), c.get("distribution"))
                                                       for c in cases if c.get("seed") == seed} for seed in seeds)
                    hashes = validation.get("official_file_hashes", {})
                valid = valid and all(hashes.get(name) == meta.get("official_sha256", {}).get(name)
                                      and hashes.get(name) for name in ("reference.py", "checks.py"))
                if not valid:
                    raise ValueError(f"Invalid/full-GPU evidence missing or task hashes differ: {report}")
            for shape in doc.get("shapes", []):
                if any(shape.get(gate, {}).get("passed") is not True
                       for gate in ("aa_before_diagnostics", "aa_after_diagnostics")):
                    raise ValueError("A/A gate missing or failed")
            score = aggregate(doc.get("shapes", []))
            settings = {key: meta.get(key) for key in CONTEXT}
            # Run directory/entry point differ; measurement settings must not.
            settings["arguments"] = {k: meta.get("arguments", {}).get(k)
                                     for k in ("seed", "correction", "eviction_mib", "repeats", "warmup", "aa_repeats")}
            if any(settings[key] is None for key in CONTEXT):
                raise ValueError("Required hardware/measurement provenance missing")
            if context is not None and settings != context:
                raise ValueError("Runs have different hardware/dependencies/task/timing settings; compare separately")
            context = settings
            eligible.append(dict(entry_point=entry, submission=str(source.resolve()), submission_sha256=source_hash,
                                 result=str(result.resolve()), speedup=score["speedup"], candidate_ms=score["candidate_ms"]))
        if not found:
            raise ValueError(f"No complete candidate benchmark found in {run}")
    if not eligible:
        raise ValueError("No eligible candidates")
    eligible.sort(key=lambda item: item["speedup"], reverse=True)
    margin = eligible[0]["speedup"] / eligible[1]["speedup"] - 1 if len(eligible) > 1 else None
    return dict(kind="local_candidate_selection_not_official", candidates=eligible,
                best_observed=eligible[0], runner_up_margin=margin,
                comparison_inconclusive=margin is None or margin < .03,
                note="3% is a conservative heuristic, not a confidence interval. Repeat on fresh seeds. "
                     "Artifacts retain all entry points; no evaluator-default promotion was performed.",
                default_implementation_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        report = collect(args.runs)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
