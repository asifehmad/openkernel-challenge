"""Collect versions/device facts only; never dump environment variables."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-h200", action="store_true")
    args = parser.parse_args()
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for package in ("torch", "triton", "sglang", "nvidia-cublas-cu12"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    for binary, flag in (("nvcc", "--version"), ("nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id"), ("ncu", "--version")):
        path = shutil.which(binary)
        if path:
            command = [path, flag] + (["--format=csv,noheader"] if binary == "nvidia-smi" else [])
            proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
            result[binary] = {"returncode": proc.returncode, "output": proc.stdout.strip()}
        else:
            result[binary] = None
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        result["devices"] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append({"name": props.name, "capability": [props.major, props.minor],
                                      "memory_bytes": props.total_memory, "sms": props.multi_processor_count})
        if torch.cuda.is_available():
            x = torch.ones((16, 16), dtype=torch.bfloat16, device="cuda")
            result["bf16_matmul_smoke"] = bool(torch.all((x @ x) == 16).item())
    except ImportError:
        result["cuda_available"] = False
    print(json.dumps(result, indent=2))
    if args.require_h200:
        valid = result["cuda_available"] and result.get("bf16_matmul_smoke")
        valid = valid and bool(result.get("devices")) and "H200" in result["devices"][0]["name"]
        valid = valid and all(result["packages"][p] for p in ("torch", "triton", "sglang"))
        if not valid:
            print("H200 + CUDA torch/Triton/SGLang required; environment not ready for GPU scoring.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
