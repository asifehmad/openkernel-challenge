"""Scored workload and reporting, independent of PyTorch and GPU availability.

Provisional workload transcribed from the public task mirror documented in
 muse_bf16_proj/PROVENANCE.json. Reconcile it with the official task before judge use.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import statistics
import sys
from typing import Any

LAYERS = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
    "router": (256, 8192),
}
LOADS = (("decode 1", 1), ("decode 16", 16), ("decode 32", 32),
         ("verify 1", 8), ("verify 16", 128), ("verify 32", 256))


def scored_shapes() -> list[dict[str, Any]]:
    return [dict(load=load, m=m, n=1024 if layer == "unpooler" and m <= 8 else n,
                 k=k, layer=layer, weight=62)
            for load, m in LOADS for layer, (n, k) in LAYERS.items()]


def positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute load totals and geometric mean; reject partial or malformed runs."""
    expected = {(s["m"], s["layer"]): s for s in scored_shapes()}
    indexed = {}
    for row in rows:
        try:
            key = (int(row["m"]), row["layer"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each shape must identify m and layer") from exc
        if key not in expected:
            raise ValueError(f"Unknown scored shape: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate scored shape: {key}")
        shape = expected[key]
        for name in ("m", "n", "k", "weight"):
            try:
                equal = float(row[name]) == shape[name]
            except (KeyError, TypeError, ValueError):
                equal = False
            if not equal:
                raise ValueError(f"Invalid {name} for {key}; expected {shape[name]}")
        if row.get("load") != shape["load"]:
            raise ValueError(f"Invalid load for {key}; expected {shape['load']}")
        indexed[key] = {**shape,
                        "baseline_us": positive_number(row.get("baseline_us"), "baseline_us"),
                        "candidate_us": positive_number(row.get("candidate_us"), "candidate_us")}
    if set(indexed) != set(expected):
        raise ValueError(f"Incomplete run: missing {sorted(set(expected) - set(indexed))}")
    loads = []
    for load, m in LOADS:
        selected = [indexed[(m, layer)] for layer in LAYERS]
        baseline = sum(r["weight"] * r["baseline_us"] for r in selected)
        candidate = sum(r["weight"] * r["candidate_us"] for r in selected)
        loads.append(dict(load=load, m=m, baseline_us=baseline, candidate_us=candidate,
                          speedup=baseline / candidate))
    speedup = statistics.geometric_mean(r["speedup"] for r in loads)
    baseline_ms = statistics.geometric_mean(r["baseline_us"] for r in loads) / 1000
    return dict(loads=loads, speedup=speedup, baseline_ms=baseline_ms,
                candidate_ms=baseline_ms / speedup)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    """Load trusted local code. Official reference imports checks.py by name."""
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
