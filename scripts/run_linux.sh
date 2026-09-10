#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_linux.sh validate [manifest]
  ./scripts/run_linux.sh smoke [rounds] [output]
  ./scripts/run_linux.sh train [rounds] [output]
  ./scripts/run_linux.sh resume <latest.pt> [total-rounds]
  ./scripts/run_linux.sh evaluate [best.pt] [output]

Defaults:
  validate manifest  configs/all_benchmarks.json
  smoke              1 round, runs/linux-smoke
  train              100 rounds, runs/shared_scorer
  evaluate           runs/shared_scorer/best.pt

Environment:
  PYTHON_BIN          Python command or path to use (default: python)
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
command_name="${1:-help}"
if (( $# > 0 )); then
  shift
fi

case "$command_name" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ $python_bin == */* ]]; then
  if [[ ! -x $python_bin ]]; then
    echo "error: Python interpreter is not executable: $python_bin" >&2
    exit 1
  fi
  python_dir="$(cd -- "$(dirname -- "$python_bin")" && pwd)"
  python_bin="$python_dir/$(basename -- "$python_bin")"
elif ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: Python interpreter not found: $python_bin" >&2
  exit 1
fi
cd "$repo_root"

case "$command_name" in
  validate)
    if (( $# > 1 )); then usage >&2; exit 2; fi
    manifest="${1:-configs/all_benchmarks.json}"
    exec "$python_bin" -m fault_order_rl validate --manifest "$manifest"
    ;;
  smoke)
    if (( $# > 2 )); then usage >&2; exit 2; fi
    rounds="${1:-1}"
    output="${2:-runs/linux-smoke}"
    exec "$python_bin" -m fault_order_rl train \
      --manifest configs/smoke_benchmarks.json \
      --rounds "$rounds" --evaluate-every 1 --output "$output"
    ;;
  train)
    if (( $# > 2 )); then usage >&2; exit 2; fi
    rounds="${1:-100}"
    output="${2:-runs/shared_scorer}"
    exec "$python_bin" -m fault_order_rl train \
      --manifest configs/all_benchmarks.json \
      --rounds "$rounds" --output "$output"
    ;;
  resume)
    if (( $# < 1 || $# > 2 )); then usage >&2; exit 2; fi
    checkpoint="$1"
    if (( $# == 2 )); then
      exec "$python_bin" -m fault_order_rl train --resume "$checkpoint" --rounds "$2"
    fi
    exec "$python_bin" -m fault_order_rl train --resume "$checkpoint"
    ;;
  evaluate)
    if (( $# > 2 )); then usage >&2; exit 2; fi
    checkpoint="${1:-runs/shared_scorer/best.pt}"
    if (( $# == 2 )); then
      exec "$python_bin" -m fault_order_rl evaluate --checkpoint "$checkpoint" --output "$2"
    fi
    exec "$python_bin" -m fault_order_rl evaluate --checkpoint "$checkpoint"
    ;;
  *)
    echo "error: unknown command: $command_name" >&2
    usage >&2
    exit 2
    ;;
esac
