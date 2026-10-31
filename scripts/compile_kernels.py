"""Offline SM90 compile: validates lowering, NOT GPU execution or correctness."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import triton

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile


def find_cuobjdump():
    bundled = Path(triton.__file__).resolve().parent / "backends/nvidia/bin/cuobjdump"
    if bundled.is_file():
        return str(bundled)
    return shutil.which("cuobjdump")


def parse_resource_usage(output, name):
    """Read the named kernel only, not unrelated functions in the cubin."""
    match = re.search(r"Function\s+" + re.escape(name) + r":\s*\n([^\n]+)", output)
    if not match:
        raise ValueError(f"cuobjdump did not report the kernel {name}")
    values = dict(re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)\b", match.group(1)))
    if set(values) != {"REG", "STACK", "SHARED", "LOCAL"}:
        raise ValueError("cuobjdump resource output lacks expected allocation fields")
    return dict(registers_per_thread=int(values["REG"]), stack_bytes=int(values["STACK"]),
                cubin_reported_shared_bytes=int(values["SHARED"]), local_bytes=int(values["LOCAL"]))


def offline_resources(kernel, tool):
    result = dict(kind="offline_compiler_allocation_not_runtime_occupancy",
                  dynamic_shared_launch_bytes=kernel.metadata.shared,
                  ptx_directives=[line.strip() for line in kernel.asm["ptx"].splitlines()
                                  if line.startswith((".version", ".target", ".address_size", ".reqntid"))])
    if tool is None:
        return dict(result, status="unavailable", reason="cuobjdump not found; no tool was installed")
    try:
        with tempfile.TemporaryDirectory(prefix="okc-cubin-") as directory:
            path = Path(directory) / "kernel.cubin"
            path.write_bytes(kernel.asm["cubin"])
            proc = subprocess.run([tool, "--dump-resource-usage", str(path)], text=True,
                                  capture_output=True, timeout=30, check=False)
        if proc.returncode:
            return dict(result, status="unavailable", reason=f"cuobjdump exit {proc.returncode}")
        return dict(result, status="reported", **parse_resource_usage(proc.stdout, kernel.metadata.name))
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return dict(result, status="unavailable", reason=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--all-shapes", action="store_true",
                        help="Compile all custom-kernel specializations used by scored and nine test shapes")
    parser.add_argument("--resources", action="store_true",
                        help="Optional offline cubin register/stack/local allocations; not GPU occupancy")
    args = parser.parse_args()
    resource_tool = find_cuobjdump() if args.resources else None
    resource_tool_version = None
    if resource_tool:
        try:
            proc = subprocess.run([resource_tool, "--version"], text=True, capture_output=True, timeout=30)
            if proc.returncode == 0:
                resource_tool_version = proc.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    path = Path(__file__).resolve().parents[1] / "muse_bf16_proj/submission.py"
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = [
        ("gemv_bf16", module._gemv, {"X": "*bf16", "W": "*bf16", "Y": "*bf16"},
         dict(N=2304, K=8192, XM=8192, XK=1, WN=8192, WK=1, BN=4, BK=512)),
        ("gemv_router", module._gemv, {"X": "*bf16", "W": "*fp32", "Y": "*fp32"},
         dict(N=256, K=8192, XM=8192, XK=1, WN=8192, WK=1, BN=4, BK=512)),
        ("split_gemm", module._split_gemm, {"X": "*bf16", "W": "*bf16", "P": "*fp32"},
         dict(M=16, N=2304, K=8192, XM=8192, XK=1, WN=8192, WK=1, BM=16, BN=64, BK=64, SPLIT=4)),
        ("single_gemm", module._single_gemm, {"X": "*bf16", "W": "*bf16", "Y": "*bf16"},
         dict(M=16, N=2304, K=8192, XM=8192, XK=1, WN=8192, WK=1, BM=16, BN=64, BK=64)),
        ("finish_split", module._finish_split, {"P": "*fp32", "Y": "*bf16"},
         dict(SIZE=16*2304, SPLIT=4, B=256)),
        ("router_pair", module._router_pair, {"X": "*bf16", "HI": "*bf16", "LO": "*bf16", "Y": "*fp32"},
         dict(M=128, N=256, K=8192, XM=8192, XK=1, BM=16, BN=32, BK=64)),
        ("short_router_split_m1", module._router_pair_split,
         {"X": "*bf16", "HI": "*bf16", "LO": "*bf16", "P": "*fp32"},
         dict(M=1, N=256, K=8192, XM=8192, XK=1, BM=16, BN=32, BK=64, SPLIT=16)),
        ("short_router_split_m8", module._router_pair_split,
         {"X": "*bf16", "HI": "*bf16", "LO": "*bf16", "P": "*fp32"},
         dict(M=8, N=256, K=8192, XM=8192, XK=1, BM=16, BN=32, BK=64, SPLIT=16)),
        ("short_router_finish_m1", module._finish_split,
         {"P": "*fp32", "Y": "*fp32"}, dict(SIZE=256, SPLIT=16, B=256)),
        ("short_router_finish_m8", module._finish_split,
         {"P": "*fp32", "Y": "*fp32"}, dict(SIZE=2048, SPLIT=16, B=256)),
    ]
    if args.all_shapes:
        sys.path.insert(0, str(path.parents[1]))
        from benchmarks.common import LAYERS, scored_shapes
        from benchmarks.contract import SPECIAL_CASES, out_features

        cases = scored_shapes() + [dict(m=m, layer=layer, n=out_features(layer, m), k=LAYERS[layer][1])
                                   for m, layer, _ in SPECIAL_CASES]
        for shape in cases:
            m, n, k, layer = (shape[key] for key in ("m", "n", "k", "layer"))
            label = f"{layer}_m{m}"
            if m <= 32 and layer != "router" or layer == "router" and m <= 8:
                wtype = "*fp32" if layer == "router" else "*bf16"
                jobs.append((f"gemv_{label}", module._gemv, {"X": "*bf16", "W": wtype, "Y": wtype},
                             dict(N=n, K=k, XM=k, XK=1, WN=k, WK=1, BN=4, BK=512)))
            if layer == "router" and m > 8:
                jobs.append((f"router_pair_{label}", module._router_pair,
                             {"X": "*bf16", "HI": "*bf16", "LO": "*bf16", "Y": "*fp32"},
                             dict(M=m, N=n, K=k, XM=k, XK=1, BM=16, BN=32, BK=64)))
            if layer != "router" and m <= 32:
                constants = dict(M=m, N=n, K=k, XM=k, XK=1, WN=k, WK=1, BM=16, BN=64, BK=64)
                jobs.append((f"single_gemm_{label}", module._single_gemm,
                             {"X": "*bf16", "W": "*bf16", "Y": "*bf16"}, constants))
                if m > 1:
                    split = 4 if k >= 4096 else 2
                    jobs.append((f"split_gemm_{label}", module._split_gemm,
                                 {"X": "*bf16", "W": "*bf16", "P": "*fp32"},
                                 dict(**constants, SPLIT=split)))
                    jobs.append((f"finish_split_{label}", module._finish_split,
                                 {"P": "*fp32", "Y": "*bf16"}, dict(SIZE=m*n, SPLIT=split, B=256)))
    results = []
    seen = set()
    for name, fn, signature, constants in jobs:
        key = (fn.__name__, tuple(sorted(signature.items())), tuple(sorted(constants.items())))
        if key in seen:
            continue
        seen.add(key)
        signature.update({key: "constexpr" for key in constants})
        kernel = compile(ASTSource(fn, signature, constants), target=GPUTarget("cuda", 90, 32),
                         options={"num_warps": 4, "num_stages": 3})
        # Loading CUDA handles requires a device; report compile metadata only.
        result = {"kernel": name, "status": "compiled-not-executed",
                  "constants": constants, "shared_bytes": kernel.metadata.shared,
                  "num_warps": kernel.metadata.num_warps}
        if args.resources:
            result["resources"] = offline_resources(kernel, resource_tool)
        results.append(result)
    report = {"target": "sm90", "gpu_validation": False,
              "all_shapes": args.all_shapes, "kernels": results}
    if args.resources:
        report["resource_tool"] = dict(path=resource_tool, version=resource_tool_version,
                                       triton_version=triton.__version__)
        report["resource_notes"] = (
            "Offline compiler/cubin allocations only. Dynamic shared launch bytes and cuobjdump's "
            "reported shared field are recorded separately, not treated as achieved usage or occupancy. "
            "No GPU execution, register-spill traffic, occupancy or performance was measured.")
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
