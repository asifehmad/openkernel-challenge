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
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import aggregate, file_sha256, load_module, scored_shapes
from benchmarks.timing import GraphTimer, aa_diagnostics, paired_summary

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


def require_correct(reference, data, output, weights, label: str) -> str:
    if output.device != data["x"].device:
        raise RuntimeError(f"{label}: output is on the wrong device")
    valid, message = reference.check_implementation(data, output, weights)
    if not valid:
        raise RuntimeError(f"{label}: {message}")
    return message


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
    parser.add_argument("--entry-point", choices=("kernel_fn", "torch_kernel_fn", "single_pass_kernel_fn", "gemv_kernel_fn"), default="kernel_fn",
                        help="Candidate entry point; torch_kernel_fn is the local PyTorch control")
    parser.add_argument("--aa-only", action="store_true", help="SGLang vs SGLang only; never a candidate result")
    parser.add_argument("--aa-repeats", type=int, default=8, help="A/A pairs before and after each shape")
    parser.add_argument("--correction", choices=("none", "empty"), default="none",
                        help="Default raw in-graph intervals; empty subtraction is experimental and A/A gated")
    parser.add_argument("--aa-max-bias", type=float, default=.03)
    parser.add_argument("--aa-max-relative-mad", type=float, default=.03)
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
    if not 4 <= args.aa_repeats <= 1000:
        parser.error("Require 4..1000 A/A pairs")
    if not 0 < args.aa_max_bias <= .1 or not 0 < args.aa_max_relative_mad <= .1:
        parser.error("A/A tolerance must be >0 and <=0.1")
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
    candidate = baseline if args.aa_only else load_module(args.submission, "okc_candidate")
    candidate_fn = baseline.kernel_fn if args.aa_only else getattr(candidate, args.entry_point, None)
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
                    submission_sha256=None if args.aa_only else file_sha256(args.submission),
                    candidate_entry_point="SGLang A/A control" if args.aa_only else args.entry_point,
                    result_type="local_aa_calibration" if args.aa_only else "local_candidate_benchmark",
                    official_check_run=False, official_submission_run=False,
                    official_sha256={name: file_sha256(official / name)
                                     for name in ("reference.py", "checks.py", "baseline.py", "TASK.md")},
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    measurement="External CUDA events captured with eviction and payload in one graph; "
                                "no host launch gap between event timestamps. Eviction excluded. "
                                "Raw timings and experimental empty-interval correction retained. "
                                "A/A bias/noise gates apply; no claimed judge confidence interval")
    record = dict(schema_version=2, status="in_progress", metadata=metadata, shapes=[], preflight=[])
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
            empty = GraphTimer(torch, lambda data, state: None, None, None,
                               eviction, eviction_sum, args.warmup)

            for shape_index, shape in enumerate(scored_shapes()):
                if args.only_m is not None and shape["m"] != args.only_m:
                    continue
                if args.only_layer is not None and shape["layer"] != args.only_layer:
                    continue
                seed_base = args.seed + 10000 + shape_index * 100000
                data = reference.generate_input(m=shape["m"], layer=shape["layer"],
                                                seed=seed_base, device="cuda")
                left = GraphTimer(torch, baseline.kernel_fn, data, baseline_state,
                                  eviction, eviction_sum, args.warmup)
                aa_right = GraphTimer(torch, baseline.kernel_fn, data, baseline_state,
                                      eviction, eviction_sum, args.warmup)
                right = aa_right if args.aa_only else GraphTimer(
                    torch, candidate_fn, data, candidate_state, eviction, eviction_sum, args.warmup)
                record["active_shape"] = dict(shape, aa_before=[], samples=[], aa_after=[])

                def sample_pairs(other, count, phase, offset, data=data, left=left):
                    samples = record["active_shape"][phase]
                    for pair in range(count):
                        fresh = reference.generate_input(m=shape["m"], layer=shape["layer"],
                                                         seed=seed_base + offset + pair, device="cuda")
                        data["x"].copy_(fresh["x"])
                        del fresh
                        overhead_before = [empty.elapsed_us() for _ in range(3)]
                        raw = dict(baseline_ab=left.elapsed_us(), candidate_ab=other.elapsed_us(),
                                   candidate_ba=other.elapsed_us(), baseline_ba=left.elapsed_us())
                        overhead_after = [empty.elapsed_us() for _ in range(3)]
                        overhead = statistics.median(overhead_before + overhead_after)
                        samples.append(dict(pair=pair, input_seed=seed_base + offset + pair,
                                            raw_us=raw, overhead_samples_us=overhead_before + overhead_after,
                                            overhead_us=overhead,
                                            corrected_us={name: value - overhead for name, value in raw.items()},
                                            correctness="pending"))
                        require_correct(reference, data, left.output, weights, f"{phase} left {pair}")
                        require_correct(reference, data, other.output, weights, f"{phase} right {pair}")
                        samples[-1]["correctness"] = "pass"
                        if (pair + 1) % 5 == 0:
                            checkpoint()
                    return samples

                before = sample_pairs(aa_right, args.aa_repeats, "aa_before", 1)
                before_gate = aa_diagnostics(before, args.correction, args.aa_max_bias, args.aa_max_relative_mad)
                record["active_shape"]["aa_before_diagnostics"] = before_gate
                if not before_gate["passed"]:
                    raise RuntimeError("Pre-timing SGLang A/A bias/noise/correction gate failed; inspect raw samples")
                samples = sample_pairs(right, args.repeats, "samples", 1001)
                after = sample_pairs(aa_right, args.aa_repeats, "aa_after", 20001)
                after_gate = aa_diagnostics(after, args.correction, args.aa_max_bias, args.aa_max_relative_mad)
                record["active_shape"]["aa_after_diagnostics"] = after_gate
                if not after_gate["passed"]:
                    raise RuntimeError("Post-timing SGLang A/A bias/noise/correction gate failed; inspect raw samples")
                field = "raw_us" if args.correction == "none" else "corrected_us"
                summary = paired_summary(samples, field)
                if args.aa_only:
                    calibration = aa_diagnostics(samples, args.correction, args.aa_max_bias, args.aa_max_relative_mad)
                    record["active_shape"]["main_aa_diagnostics"] = calibration
                    if not calibration["passed"]:
                        raise RuntimeError("Main A/A calibration failed")
                width = 4 if shape["layer"] == "router" else 2
                nominal_bytes = shape["n"] * shape["k"] * width + shape["m"] * shape["k"] * 2 + shape["m"] * shape["n"] * width
                row = dict(**shape, **summary, nominal_bytes=nominal_bytes,
                           effective_bandwidth_gb_s=nominal_bytes / summary["candidate_us"] / 1000,
                           samples=samples, aa_before=before, aa_after=after,
                           aa_before_diagnostics=before_gate, aa_after_diagnostics=after_gate,
                           correction=args.correction)
                record["shapes"].append(row)
                record.pop("active_shape", None)
                checkpoint()
                print(f"{shape['load']:9s} {shape['layer']:9s}: baseline {summary['baseline_us']:.3f} µs, "
                      f"candidate {summary['candidate_us']:.3f} µs, {summary['speedup']:.3f}x", flush=True)
                del sample_pairs, left, aa_right, right, data
        complete = len(record["shapes"]) == len(scored_shapes())
        record["status"] = "complete" if complete else "partial"
        if complete:
            record["aggregate"] = aggregate(record["shapes"])
        fields = [key for key, value in record["shapes"][0].items() if not isinstance(value, (list, dict))]
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
