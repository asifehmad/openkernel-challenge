"""Offline SM90 compile: validates lowering, NOT GPU execution or correctness."""
import argparse
import importlib.util
import json
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
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
        ("finish_split", module._finish_split, {"P": "*fp32", "Y": "*bf16"},
         dict(SIZE=16*2304, SPLIT=4, B=256)),
        ("router_pair", module._router_pair, {"X": "*bf16", "HI": "*bf16", "LO": "*bf16", "Y": "*fp32"},
         dict(M=128, N=256, K=8192, XM=8192, XK=1, BM=16, BN=32, BK=64)),
    ]
    results = []
    for name, fn, signature, constants in jobs:
        signature.update({key: "constexpr" for key in constants})
        kernel = compile(ASTSource(fn, signature, constants), target=GPUTarget("cuda", 90, 32),
                         options={"num_warps": 4, "num_stages": 3})
        # Loading CUDA handles requires a device; report compile metadata only.
        results.append({"kernel": name, "status": "compiled-not-executed",
                        "shared_bytes": kernel.metadata.shared,
                        "num_warps": kernel.metadata.num_warps})
    text = json.dumps({"target": "sm90", "gpu_validation": False, "kernels": results}, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
