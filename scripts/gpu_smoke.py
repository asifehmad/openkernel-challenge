"""One full-size projection on GPU, before the full local correctness suite.

Executes SGLang and one explicit candidate, including native kernel launch and
the actual external-event GraphTimer. This is local validation, not okc check.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.bundle import hashes, resolve_bundle  # noqa: E402
from benchmarks.common import file_sha256, load_module  # noqa: E402
from benchmarks.timing import GraphTimer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-dir", type=Path, required=True)
    p.add_argument("--submission", type=Path, default=ROOT / "muse_bf16_proj/submission.py")
    p.add_argument("--entry-point", choices=("kernel_fn", "single_pass_kernel_fn", "gemv_kernel_fn"), default="kernel_fn")
    p.add_argument("--layer", choices=("qkv_gate", "o", "pooler", "unpooler", "router"), default="qkv_gate")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--json", type=Path, required=True)
    args = p.parse_args()
    if args.m <= 0:
        p.error("m must be positive")
    if args.json.exists():
        p.error("Refusing to overwrite smoke results")
    directory = resolve_bundle(args.official_dir)
    record = dict(status="started", kind="local_gpu_smoke", official_check_run=False,
                  official_submission_run=False, entry_point=args.entry_point,
                  submission_sha256=file_sha256(args.submission), task_sha256=hashes(directory),
                  checks=[], shape={"m": args.m, "layer": args.layer})
    args.json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.json.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; GPU smoke was not executed")
        record.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
        load_module(directory / "checks.py", "checks")
        ref = load_module(directory / "reference.py", "smoke_reference")
        baseline = load_module(directory / "baseline.py", "smoke_sglang")
        candidate = load_module(args.submission, "smoke_candidate")
        fn = getattr(candidate, args.entry_point)
        with torch.inference_mode():
            weights = ref.generate_weights(94017, device="cuda")
            data = ref.generate_input(args.m, args.layer, 82173, device="cuda")
            record["shape"] = dict(data["config"])
            states = {"sglang": baseline.prepare(weights), "candidate": candidate.prepare(weights)}
            eviction = torch.ones(128 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
            eviction_sum = torch.empty((), dtype=torch.float32, device="cuda")

            def check(label, output):
                if not isinstance(output, torch.Tensor) or output.device != data["x"].device:
                    raise AssertionError(label + ": wrong output device/type")
                ok, detail = ref.check_implementation(data, output, weights)
                if not ok:
                    raise AssertionError(label + ": " + detail)
                record["checks"].append(dict(label=label, passed=True, detail=detail))

            # Initialize both libraries before capture; fresh eager outputs must survive calls.
            for name, function in (("sglang", baseline.kernel_fn), ("candidate", fn)):
                output = function(data, states[name])
                check(name + " eager", output)
                retained = output.clone()
                function(data, states[name])
                if not torch.equal(output, retained):
                    raise AssertionError(name + ": subsequent eager call overwrote output")
            timers = {
                "sglang": GraphTimer(torch, baseline.kernel_fn, data, states["sglang"], eviction, eviction_sum, 2),
                "candidate": GraphTimer(torch, fn, data, states["candidate"], eviction, eviction_sum, 2),
            }
            for distribution in ("normal", "outliers", "zero"):
                fresh = ref.generate_input(args.m, args.layer, 82174, dist="outliers" if distribution == "outliers" else "normal", device="cuda")
                if distribution == "zero":
                    fresh["x"].zero_()
                data["x"].copy_(fresh["x"])
                for name, timer in timers.items():
                    elapsed = timer.elapsed_us()
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError("External CUDA event timestamps are invalid")
                    check(name + " graph fresh " + distribution, timer.output)
            record.update(status="pass", external_event_graph_executed=True,
                          performance_claim=False)
        save()
        print(f"PASS local GPU smoke: {args.entry_point}, {args.layer} M={args.m}; no performance score")
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
