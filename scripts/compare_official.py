"""Hash a separately fetched task bundle without overwriting any project files.

This tool records local bytes and caller-supplied provenance; it cannot certify
that a directory came from the organizer. Only a successful official CLI fetch
with its version recorded supplies that provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = ("TASK.md", "AGENTS.md", "baseline.py", "reference.py", "checks.py", "submission.py")


def compare(official, local):
    rows = []
    for name in FILES:
        fresh, previous = official / name, local / name
        if not fresh.is_file() or not previous.is_file():
            raise ValueError(f"Both directories must contain {name}")
        a = hashlib.sha256(fresh.read_bytes()).hexdigest()
        b = hashlib.sha256(previous.read_bytes()).hexdigest()
        rows.append(dict(file=name, fetched_sha256=a, repository_sha256=b, identical=a == b,
                         interpretation="candidate differs from starter by design" if name == "submission.py" else
                                        "match" if a == b else "review required before relying on contract"))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-dir", type=Path, required=True)
    p.add_argument("--local-dir", type=Path, default=Path(__file__).resolve().parents[1] / "muse_bf16_proj")
    p.add_argument("--cli-version", required=True, help="Actual output from installed okc version/help")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    try:
        files = compare(args.official_dir.resolve(), args.local_dir.resolve())
    except ValueError as exc:
        p.error(str(exc))
    report = dict(source_url="https://openkernelchallenge.dev/", provenance="caller-supplied CLI download directory",
                  cli_version=args.cli_version, files=files,
                  contract_matches=all(r['identical'] for r in files if r['file'] != 'submission.py'),
                  official_check_run=False, official_submission_run=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier provenance record.
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
