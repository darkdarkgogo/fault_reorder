#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rounds="${1:-100}"
output="${2:-runs/anchor_train_1024}"
threads="${3:-1}"
batch_size="${4:-16}"

if [[ ! "$rounds" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: rounds must be a positive integer" >&2
  exit 2
fi
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: threads must be a positive integer" >&2
  exit 2
fi
if [[ ! "$batch_size" =~ ^[0-9]+$ ]]; then
  echo "error: batch_size must be a non-negative integer" >&2
  exit 2
fi

if [[ -f "$output/latest.pt" ]]; then
  echo "Resuming $output/latest.pt to round $rounds; saved thread and batch settings are preserved."
  exec python3 -m fault_order_rl train \
    --resume "$output/latest.pt" \
    --rounds "$rounds"
fi

exec python3 -m fault_order_rl train \
  --manifest configs/anchor_train_1024.json \
  --output "$output" \
  --rounds "$rounds" \
  --threads "$threads" \
  --batch-size "$batch_size"
