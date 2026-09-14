#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is not available in PATH" >&2
  exit 1
fi
if ! command -v c++ >/dev/null 2>&1; then
  echo "error: a C++ compiler named c++ is required to build cpp_podem" >&2
  exit 1
fi

python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ is required"'
python3 -c 'import numpy, setuptools, torch'
python3 -c 'import os, sysconfig; p=os.path.join(sysconfig.get_path("include"), "Python.h"); assert os.path.isfile(p), "Python development header is missing: " + p'

python3 PODEM/setup.py build_ext --inplace
python3 -c 'import sys; sys.path.insert(0, "PODEM"); import cpp_podem; assert hasattr(cpp_podem, "catalog_stuck_at") and hasattr(cpp_podem, "run_stuck_at_ordered")'

echo "cpp_podem is ready for $(python3 -c 'import sys; print(sys.executable)')"
