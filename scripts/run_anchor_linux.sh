#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output="${1:-runs/anchor_train_1024}"

if [[ -f "$output/latest.pt" ]]; then
  echo "Resuming the fixed five-round run from $output/latest.pt."
  exec python3 -m fault_order_rl train \
    --resume "$output/latest.pt"
fi

exec python3 -m fault_order_rl train \
  --manifest configs/anchor_train_1024.json \
  --validation-manifest configs/anchor_validation_6.json \
  --output "$output"
