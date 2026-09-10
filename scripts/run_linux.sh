#!/usr/bin/env bash
set -e

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rounds="${1:-100}"
output="${2:-runs/shared_scorer}"

exec python3 -m fault_order_rl train \
  --manifest configs/all_benchmarks.json \
  --rounds "$rounds" \
  --output "$output"
