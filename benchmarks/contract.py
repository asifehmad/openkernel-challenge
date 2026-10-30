"""Independent local oracle based on a third-party mirror of the task contract.

This is NOT the official checker. Fetch the current task with ``okc init`` and
pass its directory to the correctness/benchmark scripts before GPU qualification.
The candidate never imports this module. See docs/research.md for provenance.
"""
from __future__ import annotations

import math

from benchmarks.common import LAYERS, scored_shapes

# Public special cases documented in the mirrored TASK.md, not scored cases.
SPECIAL_CASES = (
    (1, "qkv_gate", "normal"),
    (7, "pooler", "normal"),
    (33, "o", "normal"),
    (130, "unpooler", "normal"),
    (5, "unpooler", "normal"),
    (8, "router", "normal"),
    (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"),
    (1000, "unpooler", "outliers"),
)


def configurations():
    return scored_shapes()


def validation_cases(quick=False):
    if quick:
        return [(m, layer, dist) for layer in LAYERS for m in (1, 7, 8, 9, 16, 31, 32, 33)
                for dist in ("normal", "zero", "signed", "outliers", "probes")]
    cases = list(SPECIAL_CASES)
    cases += [(s["m"], s["layer"], "normal") for s in configurations()]
    cases += [(m, layer, "normal") for layer in LAYERS for m in (2, 7, 9, 31, 33, 127, 129, 255, 257)]
    cases += [(m, layer, dist) for layer in LAYERS for m in (1, 7, 8, 9, 16, 31, 32, 33, 128, 256)
              for dist in ("zero", "signed", "probes", "outliers")]
    return list(dict.fromkeys(cases))


def out_features(layer: str, m: int) -> int:
    return 1024 if layer == "unpooler" and m <= 8 else LAYERS[layer][0]


def generate_weights(seed: int, device: str = "cuda"):
    import torch

    rng = torch.Generator(device=device).manual_seed(seed)
    result = {}
    for layer, (n, k) in LAYERS.items():
        w = torch.randn(n, k, generator=rng, device=device) * (k ** -0.5)
        result[layer] = w.to(torch.float32 if layer == "router" else torch.bfloat16)
    return result


def generate_input(m: int, layer: str, seed: int, dist: str = "normal", device: str = "cuda", **_):
    """Independent samples; distribution is similar, not byte-identical to the judge."""
    import torch

    if dist not in {"normal", "outliers", "zero", "signed", "probes"}:
        raise ValueError(f"unsupported input distribution: {dist}")
    n, k = out_features(layer, m), LAYERS[layer][1]
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, generator=rng, device=device)
    if dist == "outliers":
        channels = torch.randperm(k, generator=rng, device=device)[:max(1, k // 100)]
        x[:, channels] *= 30
    elif dist == "zero":
        x.zero_()
    elif dist == "signed":
        x = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    if dist != "zero":
        mask = torch.rand(m, generator=rng, device=device) < 0.125
        if m >= 8:
            mask[torch.randint(m, (1,), generator=rng, device=device)] = True
        if dist == "probes":
            mask.fill_(True)
        columns = torch.randint(k, (m, 1), generator=rng, device=device)
        powers = torch.randint(3, 8, (m, 1), generator=rng, device=device)
        signs = 2 * torch.randint(2, (m, 1), generator=rng, device=device) - 1
        spikes = torch.zeros_like(x).scatter_(1, columns, signs * (2.0 ** powers))
        x = torch.where(mask[:, None], spikes, x)
    return {"x": x.to(torch.bfloat16), "config": {"m": m, "n": n, "k": k, "layer": layer}}


def ref_kernel(data, weights):
    import torch

    cfg = data["config"]
    target = torch.float32 if cfg["layer"] == "router" else torch.bfloat16
    result = data["x"].to(torch.float64) @ weights[cfg["layer"]][:cfg["n"]].to(torch.float64).T
    return result.to(target)


def check_implementation(data, output, weights):
    """Enforce output contract, global/row relative L2, and weight-column probes."""
    import torch

    cfg = data["config"]
    expected_dtype = torch.float32 if cfg["layer"] == "router" else torch.bfloat16
    if not isinstance(output, torch.Tensor):
        return False, "output is not a tensor"
    if tuple(output.shape) != (cfg["m"], cfg["n"]) or output.dtype != expected_dtype:
        return False, "incorrect output shape or dtype"
    if output.device != data["x"].device:
        return False, "incorrect output device"
    if not bool(torch.isfinite(output).all()):
        return False, "nonfinite output"
    reference = ref_kernel(data, weights)
    # Match the published checker's FP32 norm evaluation, with FP64 probe norms.
    reference32, actual32 = reference.float(), output.float()
    delta = actual32 - reference32
    tolerance = 1e-4 if cfg["layer"] == "router" else 1e-2
    global_error = float(delta.norm() / reference32.norm().clamp_min(1e-12))
    if not math.isfinite(global_error) or global_error > tolerance:
        return False, f"global relative L2 {global_error:.6g} exceeds {tolerance}"
    row_norm = reference32.norm(dim=-1)
    denominator = torch.maximum(row_norm, (0.1 * row_norm.mean()).clamp_min(1e-12))
    worst_row = float((delta.norm(dim=-1) / denominator).max())
    if not math.isfinite(worst_row) or worst_row > 5 * tolerance:
        return False, f"row relative L2 {worst_row:.6g} exceeds {5 * tolerance}"
    probe = (data["x"].count_nonzero(dim=-1) == 1)
    worst_probe = 0.0
    if bool(probe.any()):
        r = reference[probe].double()
        error = (output[probe].double() - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
        worst_probe = float(error.max())
        if not math.isfinite(worst_probe) or worst_probe > 1e-3:
            return False, f"probe relative L2 {worst_probe:.6g} exceeds 0.001"
    return True, f"global={global_error:.3g}, row={worst_row:.3g}, probe={worst_probe:.3g}"
