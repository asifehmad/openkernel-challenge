"""One-shape profiling target; never use profiler timings as benchmark results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.common import load_module  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-dir", type=Path, required=True)
    p.add_argument("--entry-point", choices=("kernel_fn", "single_pass_kernel_fn", "gemv_kernel_fn"), default="kernel_fn")
    p.add_argument("--layer", choices=("qkv_gate", "o", "pooler", "unpooler", "router"), required=True)
    p.add_argument("--m", type=int, required=True)
    args = p.parse_args()
    if args.m <= 0:
        p.error("m must be positive")
    import torch
    if not torch.cuda.is_available():
        p.error("CUDA GPU required; no profiling was performed")
    load_module(args.official_dir / "checks.py", "checks")
    ref = load_module(args.official_dir / "reference.py", "profile_reference")
    candidate = load_module(ROOT / "muse_bf16_proj/submission.py", "profile_candidate")
    fn = getattr(candidate, args.entry_point)
    weights = ref.generate_weights(94017, device="cuda")
    data = ref.generate_input(args.m, args.layer, 82173, device="cuda")
    state = candidate.prepare(weights)
    output = fn(data, state)
    ok, message = ref.check_implementation(data, output, weights)
    if not ok:
        raise RuntimeError(message)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn(data, state)
    graph.replay()
    torch.cuda.synchronize()
    # Nsight --profile-from-start off honors this range; references/JIT excluded.
    torch.cuda.profiler.start()
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    ok, message = ref.check_implementation(data, graph_output, weights)
    if not ok:
        raise RuntimeError(message)
    print("Local profiling range complete; no latency/score claim. " + message)


if __name__ == "__main__":
    main()
