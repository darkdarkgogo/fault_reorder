#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_linux.sh [manifest]

Check the currently active offline Python environment, build the native
cpp_podem extension with the bundled pybind11 headers, and validate the bundled
training data. The manifest defaults to configs/all_benchmarks.json.

Environment:
  PYTHON_BIN   Python command or path to use (default: python)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
manifest="${1:-configs/all_benchmarks.json}"

if [[ $(uname -s) != "Linux" ]]; then
  echo "error: setup_linux.sh must be run on Linux" >&2
  exit 1
fi
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
if ! command -v c++ >/dev/null 2>&1; then
  echo "error: C++ compiler not found; install your distribution's build tools" >&2
  exit 1
fi

"$python_bin" - <<'PY'
import sys
from pathlib import Path
import sysconfig
if sys.version_info < (3, 9):
    raise SystemExit("error: fault_order_rl requires Python 3.9 or newer")
header = Path(sysconfig.get_path("include")) / "Python.h"
if not header.is_file():
    raise SystemExit("error: Python development headers not found: " + str(header))
try:
    import numpy
    import setuptools
    import torch
except ImportError as exc:
    raise SystemExit(
        "error: the active offline environment is missing " + str(exc.name)
        + "; activate the prepared d2l environment"
    )
print("Using Python", sys.version.split()[0], "from", sys.executable)
print("Using NumPy", numpy.__version__, "and PyTorch", torch.__version__)
PY

cd "$repo_root"

pushd PODEM >/dev/null
"$python_bin" setup.py build_ext --inplace
"$python_bin" - <<'PY'
import cpp_podem
for name in ("catalog_stuck_at", "run_stuck_at_ordered"):
    if not hasattr(cpp_podem, name):
        raise SystemExit("error: cpp_podem is missing " + name)
print("Built cpp_podem:", cpp_podem.__file__)
PY
popd >/dev/null

"$python_bin" -m fault_order_rl validate --manifest "$manifest"
echo "Linux setup and data validation completed successfully."
