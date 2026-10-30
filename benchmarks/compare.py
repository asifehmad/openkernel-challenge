#!/usr/bin/env python3
"""Render a complete local benchmark JSON or CSV; never invent missing timings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import aggregate, scored_shapes


def render(rows: list[dict], metadata: dict | None = None) -> str:
    result = aggregate(rows)
    indexed = {(int(r["m"]), r["layer"]): r for r in rows}
    lines = ["# Local H200 comparison", "",
             "Local estimate only; this is not an official judge result.", "",
             "| Load | Layer | M | N | K | SGLang µs | Candidate µs | Speedup |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for shape in scored_shapes():
        r = indexed[(shape["m"], shape["layer"])]
        b, c = float(r["baseline_us"]), float(r["candidate_us"])
        lines.append(f"| {shape['load']} | {shape['layer']} | {shape['m']} | {shape['n']} | "
                     f"{shape['k']} | {b:.3f} | {c:.3f} | {b / c:.4f}× |")
    lines += ["", "| Load (weight 62 per layer) | SGLang ms | Candidate ms | Speedup |",
              "| --- | ---: | ---: | ---: |"]
    for r in result["loads"]:
        lines.append(f"| {r['load']} | {r['baseline_us'] / 1000:.4f} | "
                     f"{r['candidate_us'] / 1000:.4f} | {r['speedup']:.4f}× |")
    lines += ["", f"Geometric-mean load speedup: **{result['speedup']:.4f}×**.",
              f"Local displayed-latency equivalent: **{result['candidate_ms']:.4f} ms** "
              f"(local baseline {result['baseline_ms']:.4f} ms).", "",
              "Do not substitute the website's 4.83 ms baseline into these local measurements.",
              "A CSV lacks full provenance; retain the sibling JSON and raw samples."]
    if metadata:
        lines += ["", "Run provenance:", "", "```json", json.dumps(metadata, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.result.suffix == ".csv":
        with args.result.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        metadata = None
    else:
        document = json.loads(args.result.read_text())
        rows, metadata = document["shapes"], document.get("metadata")
        if document.get("status") != "complete":
            parser.error("Only complete benchmark runs can be compared")
    try:
        report = render(rows, metadata)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
