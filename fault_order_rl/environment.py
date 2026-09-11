"""Validated Python wrapper around the ordered PODEM binding."""

import importlib
from pathlib import Path
import sys


RESULT_FIELDS = (
    "pattern_count",
    "detected_collapsed_faults",
    "detected_equivalent_faults",
    "uncollapsed_faults",
    "aborted_faults",
    "redundant_faults",
    "redundant_equivalent_faults",
    "podem_calls",
    "total_backtracks",
)


def load_cpp_podem(module_dir=None):
    if module_dir is not None:
        resolved = str(Path(module_dir).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        module = importlib.import_module("cpp_podem")
    except ImportError as exc:
        raise RuntimeError(
            "cpp_podem is unavailable; build PODEM/setup.py for this Python"
        ) from exc
    if not hasattr(module, "catalog_stuck_at") or not hasattr(
        module, "run_stuck_at_ordered"
    ):
        raise RuntimeError("cpp_podem does not provide the ordered ATPG API")
    return module


class PodemEnvironment:
    def __init__(self, module_dir=None, backtrack_limit=5000, seed=14):
        if backtrack_limit <= 0:
            raise ValueError("backtrack_limit must be positive")
        self.module = load_cpp_podem(module_dir)
        self.backtrack_limit = int(backtrack_limit)
        self.seed = int(seed)

    def catalog(self, bench_path, faultmap_path):
        self._check_files(bench_path, faultmap_path)
        return dict(
            self.module.catalog_stuck_at(str(bench_path), str(faultmap_path))
        )

    def run(self, bench_path, faultmap_path, ordered_fault_ids):
        self._check_files(bench_path, faultmap_path)
        raw = dict(
            self.module.run_stuck_at_ordered(
                str(bench_path),
                str(faultmap_path),
                list(ordered_fault_ids),
                self.backtrack_limit,
                self.seed,
            )
        )
        missing = [field for field in RESULT_FIELDS if field not in raw]
        if missing:
            raise RuntimeError("PODEM result is missing fields: {}".format(missing))
        result = {field: int(raw[field]) for field in RESULT_FIELDS}
        if any(value < 0 for value in result.values()):
            raise RuntimeError("PODEM returned a negative metric")
        if result["detected_equivalent_faults"] > result["uncollapsed_faults"]:
            raise RuntimeError("PODEM detected count exceeds the fault total")
        if result["detected_collapsed_faults"] > len(ordered_fault_ids):
            raise RuntimeError("PODEM collapsed detection count exceeds the catalog")
        if result["redundant_faults"] > len(ordered_fault_ids):
            raise RuntimeError("PODEM collapsed redundant count exceeds the catalog")
        covered = (result["detected_equivalent_faults"]
                   + result["redundant_equivalent_faults"])
        if covered > result["uncollapsed_faults"]:
            raise RuntimeError("PODEM covered count exceeds the fault total")
        result["covered_equivalent_faults"] = covered
        result["fault_coverage"] = covered / result["uncollapsed_faults"]
        return result

    @staticmethod
    def _check_files(bench_path, faultmap_path):
        for path in (bench_path, faultmap_path):
            if not Path(path).is_file():
                raise ValueError("missing PODEM input: {}".format(path))
