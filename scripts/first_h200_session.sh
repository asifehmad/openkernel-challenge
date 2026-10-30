#!/usr/bin/env bash
# Local tests/measurements only. No login, rental, okc check or okc submit.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OFFICIAL_DIR:?Set OFFICIAL_DIR to the freshly downloaded task directory}"
run_root=${1:?Usage: scripts/first_h200_session.sh NEW_RUN [--smoke-only] [--entry-point NAME]}
shift
smoke_only=0
entry_choice=all
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke-only) smoke_only=1; shift ;;
    --entry-point) entry_choice=${2:?Missing entry point}; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$entry_choice" in
  all) entries=(kernel_fn single_pass_kernel_fn gemv_kernel_fn) ;;
  kernel_fn|single_pass_kernel_fn|gemv_kernel_fn) entries=("$entry_choice") ;;
  *) echo "Invalid entry point: $entry_choice" >&2; exit 2 ;;
esac
OFFICIAL_DIR=$(python3 scripts/resolve_task.py "$OFFICIAL_DIR" --check-contract)
if [ -e "$run_root" ]; then
  echo "Refusing to overwrite existing result directory: $run_root" >&2
  exit 1
fi
mkdir -p "$run_root"
python3 scripts/verify_environment.py --require-h200 > "$run_root/environment.json"
# Retain exact tested bytes even if the checkout or downloaded task later changes.
cp muse_bf16_proj/submission.py "$run_root/submission.py"
chmod a-w "$run_root/submission.py"
mkdir "$run_root/task"
for name in TASK.md AGENTS.md baseline.py reference.py checks.py submission.py; do
  cp "$OFFICIAL_DIR/$name" "$run_root/task/$name"
done
OFFICIAL_DIR=$(python3 scripts/resolve_task.py "$run_root/task")
for entry in "${entries[@]}"; do
  python3 scripts/gpu_smoke.py --official-dir "$OFFICIAL_DIR" \
    --submission "$run_root/submission.py" --entry-point "$entry" --json "$run_root/smoke-$entry.json"
done
if [ "$smoke_only" -eq 1 ]; then
  echo 'Representative local smoke completed. Full correctness and performance remain unrun.'
  exit 0
fi
for entry in "${entries[@]}"; do
  python3 scripts/check_correctness.py --device cuda --official-dir "$OFFICIAL_DIR" \
    --submission "$run_root/submission.py" --entry-point "$entry" \
    --seeds 94017 82173 --json "$run_root/correctness-$entry.json"
done
# Establish timer calibration with actual SGLang, never the PyTorch control.
python3 benchmarks/benchmark.py --official-dir "$OFFICIAL_DIR" --aa-only \
  --warmup 20 --repeats 30 --aa-repeats 12 --seed 94017 --output "$run_root/aa"
python3 benchmarks/compare.py "$run_root/aa/results.json" --output "$run_root/aa/report.md"
for entry in "${entries[@]}"; do
  python3 benchmarks/benchmark.py --official-dir "$OFFICIAL_DIR" --entry-point "$entry" \
    --submission "$run_root/submission.py" \
    --warmup 20 --repeats 30 --aa-repeats 12 --seed 94017 --output "$run_root/$entry"
  python3 benchmarks/compare.py "$run_root/$entry/results.json" --output "$run_root/$entry/report.md"
done
python3 scripts/summarize_candidates.py "$run_root" --output "$run_root/selection.json"
