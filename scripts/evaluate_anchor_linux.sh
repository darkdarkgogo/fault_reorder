#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="${1:-runs/anchor_train_1024}"
manifest="${2:-configs/anchor_validation_6.json}"
manifest_name="$(basename -- "$manifest" .json)"
output="${3:-$run_dir/evaluation-$manifest_name}"
checkpoint="$run_dir/best.pt"

if [[ ! -f "$checkpoint" ]]; then
  echo "error: best checkpoint does not exist: $checkpoint" >&2
  exit 1
fi
if [[ ! -f "$manifest" ]]; then
  echo "error: evaluation manifest does not exist: $manifest" >&2
  exit 1
fi

exec python3 -m fault_order_rl evaluate \
  --checkpoint "$checkpoint" \
  --manifest "$manifest" \
  --output "$output"
