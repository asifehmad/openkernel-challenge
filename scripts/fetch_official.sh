#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repository_root=$PWD
command -v okc >/dev/null || { echo 'Install CLI from https://openkernelchallenge.dev/install.sh (not PyPI okc).' >&2; exit 1; }
# A separate directory prevents overwriting our candidate or snapshot files.
destination=${1:-official}
if [ -e "$destination" ]; then
  echo "Destination already exists: $destination. Choose a new empty destination." >&2
  exit 1
fi
mkdir -p "$destination"
cd "$destination"
okc init muse_bf16_proj
python3 "$repository_root/scripts/resolve_task.py" "$PWD"
echo 'The preceding absolute path is the task directory. Inspect TASK.md/AGENTS.md and compare hashes before GPU testing.'
