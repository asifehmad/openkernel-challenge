#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-cpu}
case "$mode" in
  cpu)
    venv=${VENV_PATH:-$PWD/.venv}
    python3 -m venv "$venv"
    "$venv/bin/python" -m pip install -r requirements-dev.txt
    "$venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
    "$venv/bin/python" -m pip install triton==3.6.0
    echo "CPU tools installed. Activate: source $venv/bin/activate"
    ;;
  gpu)
    # Run INSIDE the official image. Preserve its coupled torch/CUDA/sglang.
    # GPU runners use only the image packages and Python's standard library.
    # Do not install CPU dev pins (especially NumPy) over evaluation packages.
    python3 scripts/verify_environment.py --require-h200
    ;;
  *) echo 'usage: scripts/setup.sh cpu|gpu' >&2; exit 2 ;;
esac
