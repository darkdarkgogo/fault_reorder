#!/usr/bin/env bash
set -e

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec python3 PODEM/setup.py build_ext --inplace
