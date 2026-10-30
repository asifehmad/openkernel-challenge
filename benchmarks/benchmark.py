#!/usr/bin/env python3
"""Paired, cold-L2 CUDA graph benchmark against the downloaded SGLang baseline.

This implements a documented local approximation, not the private judge runner.
All input generation, cache eviction, reference checks and copies are untimed.
No remote checks or submissions are issued.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import aggregate, file_sha256, load_module, scored_shapes

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ((1, "qkv_gate", "normal"), (7, "pooler", "normal"),
             (33, "o", "normal"), (130, "unpooler", "normal"),
             (5, "unpooler", "normal"), (8, "router", "normal"),
             (9, "router", "outliers"), (512, "qkv_gate", "outliers"),
             (1000, "unpooler", "outliers"))


def git_output(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args], text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def paired_summary(samples: list[dict]) -> dict:
    """One sample is AB then BA; use geometric mean of the two ratios."""
    ratios, baselines = [], []
    for sample in samples:
        times = sample["corrected_us"]
        values = [times[key] for key in ("baseline_ab", "candidate_ab", "candidate_ba", "baseline_ba")]
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("Nonpositive overhead-corrected latency; timer resolution is insufficient")
        ratios.append(math.sqrt((times["baseline_ab"] / times["candidate_ab"]) *
                                (times["baseline_ba"] / times["candidate_ba"])))
        baselines.extend((times["baseline_ab"], times["baseline_ba"]))
    if not ratios:
        raise ValueError("No completed timing pairs")
    speedup = statistics.median(ratios)
    baseline = statistics.median(baselines)
    return dict(baseline_us=baseline, candidate_us=baseline / speedup,
                speedup=speedup, pair_count=len(ratios),
                paired_speedup_min=min(ratios), paired_speedup_max=max(ratios),
                paired_speedup_mad=statistics.median(abs(r - speedup) for r in ratios))


def require_correct(reference, data, output, weights, label: str) -> str:
    if output.device != data["x"].device:
        raise RuntimeError(f"{label}: output is on the wrong device")
    valid, message = reference.check_implementation(data, output, weights)
    if not valid:
        raise RuntimeError(f"{label}: {message}")
    return message


def capture(torch, function, data, state, warmup: int):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            function(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function(data, state)
    torch.cuda.synchronize()
    return graph, output


def prepare_module(torch, module, weights) -> tuple[object, dict]:
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    state = module.prepare(weights)
    torch.cuda.synchronize()
    duration = time.perf_counter() - begin
    metrics = dict(seconds=duration,
                   retained_extra_bytes=torch.cuda.memory_allocated() - allocated_before,
                   peak_extra_bytes=torch.cuda.max_memory_allocated() - allocated_before)
    if duration > 300:
        raise RuntimeError(f"prepare took {duration:.1f}s, exceeding the documented 300s limit")
    return state, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True,
                        help="Unmodified task bundle downloaded using the official CLI")
    parser.add_argument("--submission", type=Path, default=ROOT / "muse_bf16_proj/submission.py")
    parser.add_argument("--entry-point", choices=("kernel_fn", "torch_kernel_fn"), default="kernel_fn",
                        help="Candidate entry point; torch_kernel_fn is the local PyTorch control")
    parser.add_argument("--output", type=Path, help="New result directory (must not already exist)")
    parser.add_argument("--repeats", type=int, default=20, help="Swapped AB/BA pairs per shape (>=4)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261030, help="Change for unseen weights/activations")
    parser.add_argument("--eviction-mib", type=int, default=128, help="Read buffer size; must exceed L2")
    parser.add_argument("--only-m", type=int, choices=(1, 8, 16, 32, 128, 256))
    parser.add_argument("--only-layer", choices=("qkv_gate", "o", "pooler", "unpooler", "router"))
    parser.add_argument("--allow-other-gpu", action="store_true",
                        help="Permit diagnostic runs elsewhere; results remain hardware-specific")
    args = parser.parse_args()
    if not 4 <= args.repeats <= 10000 or not 1 <= args.warmup <= 1000:
        parser.error("Require 4..10000 repeat pairs and 1..1000 warmups")
    if not 64 <= args.eviction_mib <= 2048:
        parser.error("Require 64..2048 MiB eviction buffer")
    official = args.official_dir.resolve()
    for name in ("reference.py", "checks.py", "baseline.py", "TASK.md"):
        if not (official / name).is_file():
            parser.error(f"Missing official file: {official / name}; obtain the official task first")
    import torch
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and the official evaluation dependencies are required")
    properties = torch.cuda.get_device_properties(0)
    if "H200" not in properties.name and not args.allow_other_gpu:
        parser.error(f"Detected {properties.name}; use H200 or explicitly permit a diagnostic run")
    l2_bytes = getattr(properties, "L2_cache_size", None)
    eviction_bytes = args.eviction_mib * 1024 * 1024
    if l2_bytes and eviction_bytes <= l2_bytes:
        parser.error("Eviction buffer must be larger than the GPU's L2 cache")

    # Only harness code imports official modules. Submission stays self-contained.
    load_module(official / "checks.py", "checks")
    reference = load_module(official / "reference.py", "okc_local_reference")
    baseline = load_module(official / "baseline.py", "okc_local_baseline")
    candidate = load_module(args.submission, "okc_candidate")
    candidate_fn = getattr(candidate, args.entry_point, None)
    if not callable(candidate_fn):
        parser.error(f"Submission has no callable {args.entry_point}")
    output_dir = args.output or ROOT / "benchmarks/results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(kind="local_estimate_not_official_judge", utc=datetime.now(timezone.utc).isoformat(),
                    platform=platform.platform(), python=sys.version, torch=torch.__version__,
                    cuda=torch.version.cuda, triton=package_version("triton"),
                    sglang=package_version("sglang"), gpu=properties.name,
                    capability=list(torch.cuda.get_device_capability(0)),
                    total_memory_bytes=properties.total_memory,
                    sm_count=properties.multi_processor_count, l2_bytes=l2_bytes,
                    tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                    bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    git_commit=git_output("rev-parse", "HEAD"),
                    git_status=git_output("status", "--porcelain"),
                    submission_sha256=file_sha256(args.submission), candidate_entry_point=args.entry_point,
                    official_sha256={name: file_sha256(official / name)
                                     for name in ("reference.py", "checks.py", "baseline.py", "TASK.md")},
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    measurement="CUDA events around one graph replay; read-only L2 eviction outside timing; "
                                "median empty-graph overhead before/after each AB/BA pair is subtracted; "
                                "median geometric paired ratios; no claimed judge confidence interval")
    record = dict(schema_version=1, status="in_progress", metadata=metadata, shapes=[], preflight=[])
    result_path = output_dir / "results.json"

    def checkpoint():
        temp = output_dir / "results.json.tmp"
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temp.replace(result_path)

    checkpoint()
    try:
        with torch.inference_mode():
            weights = reference.generate_weights(args.seed, device="cuda")
            baseline_state, baseline_prepare = prepare_module(torch, baseline, weights)
            candidate_state, candidate_prepare = prepare_module(torch, candidate, weights)
            record["preparation"] = dict(baseline=baseline_prepare, candidate=candidate_prepare)
            for index, (m, layer, dist) in enumerate(PREFLIGHT):
                data = reference.generate_input(m=m, layer=layer, seed=args.seed + index + 1,
                                                dist=dist, device="cuda")
                for name, function, state in (("baseline", baseline.kernel_fn, baseline_state),
                                               ("candidate", candidate_fn, candidate_state)):
                    result = function(data, state)
                    message = require_correct(reference, data, result, weights, f"preflight {name} {m}/{layer}")
                    record["preflight"].append(dict(backend=name, m=m, layer=layer, dist=dist,
                                                    passed=True, detail=message))
                del result, data
            checkpoint()
            # torch.sum reads every element. Its result stays on GPU and is untimed.
            eviction = torch.ones(eviction_bytes // 4, device="cuda", dtype=torch.float32)
            eviction_sum = torch.empty((), device="cuda", dtype=torch.float32)
            torch.sum(eviction, dim=0, out=eviction_sum)
            flush_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(flush_graph):
                torch.sum(eviction, dim=0, out=eviction_sum)
            empty_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(empty_graph):
                pass
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            def time_graph(graph):
                flush_graph.replay()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1000

            for shape_index, shape in enumerate(scored_shapes()):
                if args.only_m is not None and shape["m"] != args.only_m:
                    continue
                if args.only_layer is not None and shape["layer"] != args.only_layer:
                    continue
                seed_base = args.seed + 10000 + shape_index * 100000
                data = reference.generate_input(m=shape["m"], layer=shape["layer"],
                                                seed=seed_base, device="cuda")
                bg, bout = capture(torch, baseline.kernel_fn, data, baseline_state, args.warmup)
                cg, cout = capture(torch, candidate_fn, data, candidate_state, args.warmup)
                # Capture outputs are checked after replay, not assumed valid from capture.
                bg.replay()
                cg.replay()
                require_correct(reference, data, bout, weights, "baseline initial graph replay")
                require_correct(reference, data, cout, weights, "candidate initial graph replay")
                samples = []
                record["active_shape"] = {**shape, "samples": samples}
                for pair in range(args.repeats):
                    fresh = reference.generate_input(m=shape["m"], layer=shape["layer"],
                                                     seed=seed_base + pair + 1, device="cuda")
                    data["x"].copy_(fresh["x"])
                    del fresh
                    overhead_before = [time_graph(empty_graph) for _ in range(3)]
                    raw = dict(baseline_ab=time_graph(bg), candidate_ab=time_graph(cg),
                               candidate_ba=time_graph(cg), baseline_ba=time_graph(bg))
                    overhead_after = [time_graph(empty_graph) for _ in range(3)]
                    overhead = statistics.median(overhead_before + overhead_after)
                    corrected = {name: value - overhead for name, value in raw.items()}
                    require_correct(reference, data, bout, weights, f"baseline fresh graph input {pair}")
                    require_correct(reference, data, cout, weights, f"candidate fresh graph input {pair}")
                    samples.append(dict(pair=pair, input_seed=seed_base + pair + 1,
                                        raw_us=raw, overhead_samples_us=overhead_before + overhead_after,
                                        overhead_us=overhead, corrected_us=corrected, correctness="pass"))
                    if (pair + 1) % 5 == 0:
                        checkpoint()
                summary = paired_summary(samples)
                width = 4 if shape["layer"] == "router" else 2
                nominal_bytes = shape["n"] * shape["k"] * width + shape["m"] * shape["k"] * 2 + shape["m"] * shape["n"] * width
                row = dict(**shape, **summary, nominal_bytes=nominal_bytes,
                           effective_bandwidth_gb_s=nominal_bytes / summary["candidate_us"] / 1000,
                           samples=samples)
                record["shapes"].append(row)
                record.pop("active_shape", None)
                checkpoint()
                print(f"{shape['load']:9s} {shape['layer']:9s}: baseline {summary['baseline_us']:.3f} µs, "
                      f"candidate {summary['candidate_us']:.3f} µs, {summary['speedup']:.3f}x", flush=True)
                del bg, cg, bout, cout, data
        complete = len(record["shapes"]) == len(scored_shapes())
        record["status"] = "complete" if complete else "partial"
        if complete:
            record["aggregate"] = aggregate(record["shapes"])
        fields = [key for key in record["shapes"][0] if key != "samples"]
        with (output_dir / "shapes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(record["shapes"])
        checkpoint()
        print(f"Local measurements written to {result_path}; status={record['status']}")
        if complete:
            print(json.dumps(record["aggregate"], indent=2))
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint()
        raise


if __name__ == "__main__":
    main()
