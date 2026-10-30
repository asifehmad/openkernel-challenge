"""Local CUDA graph timing and CPU-testable calibration, not the judge runner."""
from __future__ import annotations

import math
import statistics

KEYS = ("baseline_ab", "candidate_ab", "candidate_ba", "baseline_ba")


def paired_summary(samples, field="corrected_us"):
    ratios, baselines = [], []
    for sample in samples:
        times = sample[field]
        if any(not math.isfinite(times[k]) or times[k] <= 0 for k in KEYS):
            raise ValueError("Nonpositive/nonfinite latency; reject this correction or timer")
        ratios.append(math.sqrt(times["baseline_ab"] / times["candidate_ab"] *
                                times["baseline_ba"] / times["candidate_ba"]))
        baselines.extend((times["baseline_ab"], times["baseline_ba"]))
    if not ratios:
        raise ValueError("No completed timing pairs")
    speedup = statistics.median(ratios)
    baseline = statistics.median(baselines)
    return dict(baseline_us=baseline, candidate_us=baseline / speedup,
                speedup=speedup, pair_count=len(ratios),
                paired_speedup_min=min(ratios), paired_speedup_max=max(ratios),
                paired_speedup_mad=statistics.median(abs(r - speedup) for r in ratios))


def aa_diagnostics(samples, correction="none", max_bias=.03, max_relative_mad=.03):
    """A/A can reject bias/noise, but cannot prove absolute timing accuracy."""
    report = {"correction": correction, "max_bias": max_bias,
              "max_relative_mad": max_relative_mad, "variants": {}}
    for name, field in (("none", "raw_us"), ("empty", "corrected_us")):
        try:
            summary = paired_summary(samples, field)
            relative_mad = summary["paired_speedup_mad"] / summary["speedup"]
            valid = abs(summary["speedup"] - 1) <= max_bias and relative_mad <= max_relative_mad
            report["variants"][name] = {**summary, "relative_mad": relative_mad, "passed": valid}
        except ValueError as exc:
            report["variants"][name] = {"passed": False, "error": str(exc)}
    selected = report["variants"][correction]
    report["passed"] = selected["passed"]
    overhead = statistics.median(s["overhead_us"] for s in samples)
    raw_base = report["variants"]["none"].get("baseline_us")
    report["median_empty_interval_us"] = overhead
    report["empty_fraction_of_raw_baseline"] = overhead / raw_base if raw_base else None
    # Empty-event subtraction can amplify noise while keeping A/A centered at 1.
    # Treat >20% overhead as unresolved for corrected results, not as calibration.
    if correction == "empty" and (raw_base is None or overhead / raw_base > .2):
        report["passed"] = False
        report["rejection"] = "Empty interval exceeds 20% of payload timing; correction is too sensitive"
    return report


class GraphTimer:
    """Capture eviction + external timing events + payload in ONE CUDA graph.

    No Python dispatch lies between event timestamps during replay. external=True
    retains real event-record nodes instead of capture-only dependencies. This
    requires a compatible PyTorch/CUDA version and must be tested on the H200.
    """
    def __init__(self, torch, function, data, state, eviction, eviction_sum, warmup):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                function(data, state)
            torch.sum(eviction, dim=0, out=eviction_sum)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            torch.sum(eviction, dim=0, out=eviction_sum)
            self.start.record()
            self.output = function(data, state)
            self.end.record()
        torch.cuda.synchronize()
        # Graph instantiation/first launch effects are not part of timed samples.
        for _ in range(warmup):
            self.elapsed_us()

    def elapsed_us(self):
        self.graph.replay()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000
