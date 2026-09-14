#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="${1:-runs/anchor_train_1024}"
manifest="${2:-configs/anchor_validation_6.json}"
manifest_name="$(basename -- "$manifest" .json)"
output_base="${3:-$run_dir/evaluation-$manifest_name}"
best_checkpoint="$run_dir/best.pt"
latest_checkpoint="$run_dir/latest.pt"

if [[ ! -f "$best_checkpoint" ]]; then
  echo "error: best checkpoint does not exist: $best_checkpoint" >&2
  exit 1
fi
if [[ ! -f "$latest_checkpoint" ]]; then
  echo "error: latest checkpoint does not exist: $latest_checkpoint" >&2
  exit 1
fi
if [[ ! -f "$manifest" ]]; then
  echo "error: evaluation manifest does not exist: $manifest" >&2
  exit 1
fi

python3 -m fault_order_rl evaluate \
  --checkpoint "$best_checkpoint" \
  --manifest "$manifest" \
  --output "${output_base}-best"

python3 -m fault_order_rl evaluate \
  --checkpoint "$latest_checkpoint" \
  --manifest "$manifest" \
  --output "${output_base}-latest"
