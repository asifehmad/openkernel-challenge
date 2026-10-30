#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q muse_bf16_proj benchmarks scripts tests
python3 -m pytest -q
python3 scripts/check_correctness.py --device cpu --quick --seeds 94017
