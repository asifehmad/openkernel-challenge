#!/usr/bin/env bash
# Local tests/measurements only. No login, rental, okc check or okc submit.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OFFICIAL_DIR:?Set OFFICIAL_DIR to the freshly downloaded task directory}"
run_root=${1:?Usage: OFFICIAL_DIR=/absolute/task/dir scripts/first_h200_session.sh benchmarks/results/new-run}
if [ -e "$run_root" ]; then
  echo "Refusing to overwrite existing result directory: $run_root" >&2
  exit 1
fi
mkdir -p "$run_root"
python3 scripts/verify_environment.py --require-h200 > "$run_root/environment.json"
for entry in kernel_fn single_pass_kernel_fn gemv_kernel_fn; do
  python3 scripts/check_correctness.py --device cuda --official-dir "$OFFICIAL_DIR" \
    --entry-point "$entry" --seeds 94017 82173 --json "$run_root/correctness-$entry.json"
done
# Establish timer calibration with actual SGLang, never the PyTorch control.
python3 benchmarks/benchmark.py --official-dir "$OFFICIAL_DIR" --aa-only \
  --warmup 20 --repeats 30 --aa-repeats 12 --seed 94017 --output "$run_root/aa"
python3 benchmarks/compare.py "$run_root/aa/results.json" --output "$run_root/aa/report.md"
for entry in kernel_fn single_pass_kernel_fn gemv_kernel_fn; do
  python3 benchmarks/benchmark.py --official-dir "$OFFICIAL_DIR" --entry-point "$entry" \
    --warmup 20 --repeats 30 --aa-repeats 12 --seed 94017 --output "$run_root/$entry"
  python3 benchmarks/compare.py "$run_root/$entry/results.json" --output "$run_root/$entry/report.md"
done
