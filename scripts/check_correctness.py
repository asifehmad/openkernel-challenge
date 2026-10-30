#!/usr/bin/env python3
"""Local eager/graph correctness, never an official service or submission call."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import contract  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def small_weights(seed, device):
    import torch
    rng = torch.Generator(device=device).manual_seed(seed)
    return {layer: (torch.randn(32, 64, generator=rng, device=device) / 8).to(
        torch.float32 if layer == "router" else torch.bfloat16) for layer in contract.LAYERS}


def small_input(m, layer, seed, dist, device):
    import torch
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, 64, generator=rng, device=device)
    if dist == "zero":
        x.zero_()
    elif dist == "signed":
        x = x.sign()
    elif dist == "outliers":
        x[:, 3] *= 30
    if dist == "probes":
        x.zero_()
        for row in range(m):
            x[row, row % 64] = (-1 if row % 2 else 1) * (2 ** (3 + row % 5))
    elif dist != "zero" and m >= 8:
        x[0].zero_()
        x[0, 7] = 32
    n = 8 if layer == "unpooler" and m <= 8 else 32
    return {"x": x.bfloat16(), "config": {"m": m, "n": n, "k": 64, "layer": layer}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--backend", choices=["submission", "torch"], default="submission")
    p.add_argument("--submission", type=Path, default=ROOT / "muse_bf16_proj/submission.py")
    p.add_argument("--entry-point", choices=["kernel_fn", "torch_kernel_fn", "single_pass_kernel_fn", "gemv_kernel_fn"],
                   default="kernel_fn", help="Explicit candidate path; none is the official SGLang baseline")
    p.add_argument("--official-dir", type=Path, help="untouched current okc init task files")
    p.add_argument("--seeds", type=int, nargs="+", default=[94017, 173629, 812347])
    p.add_argument("--quick", action="store_true", help="reduced 32x64 weights, NEVER official validation")
    p.add_argument("--json", type=Path, help="write complete result record")
    args = p.parse_args()
    if args.device == "cpu" and not args.quick:
        p.error("CPU checks require --quick; full dimensions and graphs belong on H200")
    if args.quick and args.official_dir:
        p.error("--quick cannot use official shapes/checker")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; GPU correctness has not been run")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    oracle = contract
    official_hashes = {}
    if args.official_dir:
        directory = args.official_dir.resolve()
        for filename in ("reference.py", "checks.py"):
            file = directory / filename
            if not file.is_file():
                p.error(f"missing {file}")
            official_hashes[filename] = hashlib.sha256(file.read_bytes()).hexdigest()
        sys.path.insert(0, str(directory))
        load_module(directory / "checks.py", "checks")
        oracle = load_module(directory / "reference.py", "okc_local_reference")
    if args.backend == "submission":
        candidate = load_module(args.submission, "candidate_correctness")
        prepare, run = candidate.prepare, getattr(candidate, args.entry_point)
    else:
        # Independent FP32 matmul, distinct from FP64 reference.
        def prepare(weights):
            return weights
        def run(data, state):
            cfg = data["config"]
            out = data["x"].float() @ state[cfg["layer"]][:cfg["n"]].float().T
            return out.to(torch.float32 if cfg["layer"] == "router" else torch.bfloat16)
    if args.quick:
        cases = [(m, layer, dist) for layer in contract.LAYERS for m in (1, 7, 8, 9, 16, 31, 32, 33)
                 for dist in ("normal", "zero", "signed", "outliers", "probes")]
    else:
        cases = list(contract.SPECIAL_CASES)
        cases += [(s["m"], s["layer"], "normal") for s in contract.configurations()]
        cases += [(m, layer, "normal") for layer in contract.LAYERS for m in (2, 7, 9, 31, 33, 127, 129, 255, 257)]
        cases += [(m, layer, dist) for layer in contract.LAYERS
                  for m in (1, 7, 8, 9, 16, 31, 32, 33, 128, 256)
                  for dist in ("zero", "signed", "probes", "outliers")]
        cases = list(dict.fromkeys(cases))
    results = []
    qualification = ("REDUCED CPU/local checks" if args.quick else
                     "LOCAL execution using supplied task checker (verify provenance)" if args.official_dir else
                     "PROVISIONAL local contract, NOT official checker")
    print(qualification, flush=True)
    for seed in args.seeds:
        weights = small_weights(seed, args.device) if args.quick else oracle.generate_weights(seed, device=args.device)
        state = prepare(weights)
        for index, (m, layer, dist) in enumerate(cases):
            make_input = small_input if args.quick else oracle.generate_input
            # Official generate_input only supports normal/outliers. Local edge inputs
            # use the independent generator and are still checked by the official oracle.
            if not args.quick and dist in {"zero", "signed", "probes"}:
                make_input = contract.generate_input
            data = make_input(m, layer, seed + 10000 + index, dist=dist, device=args.device)
            fresh = make_input(m, layer, seed + 20000 + index, dist=dist, device=args.device)
            row = {"seed": seed, "m": m, "layer": layer, "distribution": dist}
            try:
                output = run(data, state)
                passed, details = oracle.check_implementation(data, output, weights)
                if not passed:
                    raise AssertionError(details)
                retained = output.clone()
                fresh_output = run(fresh, state)
                passed_fresh, fresh_details = oracle.check_implementation(fresh, fresh_output, weights)
                if not passed_fresh:
                    raise AssertionError("fresh eager input: " + fresh_details)
                if not torch.equal(output, retained):
                    raise AssertionError("a later call overwrote an earlier eager output")
                if args.device == "cuda":
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run(data, state)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = run(data, state)
                    graph.replay()
                    valid, msg = oracle.check_implementation(data, graph_output, weights)
                    if not valid:
                        raise AssertionError("initial graph replay: " + msg)
                    data["x"].copy_(fresh["x"])
                    graph.replay()
                    valid, msg = oracle.check_implementation(data, graph_output, weights)
                    if not valid:
                        raise AssertionError("fresh graph replay: " + msg)
                    # The captured buffer may be reused by its own replays, but
                    # an unrelated eager call must not overwrite that output.
                    retained_graph_output = graph_output.clone()
                    run(fresh, state)
                    if not torch.equal(graph_output, retained_graph_output):
                        raise AssertionError("eager call overwrote a captured graph output")
                    row["graph"] = "pass"
                else:
                    row["graph"] = "unrun (CPU)"
                row.update(status="pass", detail=details)
            except Exception as exc:
                row.update(status="fail", detail=f"{type(exc).__name__}: {exc}")
            results.append(row)
            print(f"{row['status'].upper()} seed={seed} {layer} m={m} {dist}: {row['detail']}", flush=True)
    failures = sum(row["status"] == "fail" for row in results)
    report = {"qualification": qualification, "official_service_run": False,
              "official_check_run": False, "official_submission_run": False,
              "candidate_entry_point": args.entry_point if args.backend == "submission" else "local_fp32_torch_control",
              "official_file_hashes": official_hashes, "quick": args.quick, "device": args.device,
              "backend": args.backend, "submission_sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest() if args.backend == "submission" else None,
              "passed": len(results) - failures, "failed": failures, "cases": results}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed: {len(results) - failures} passed; {failures} failed; GPU graph checks {'executed' if args.device == 'cuda' else 'UNRUN'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
