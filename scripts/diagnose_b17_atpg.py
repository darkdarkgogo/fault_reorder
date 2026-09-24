"""Run the native b17_C baseline with opt-in DTC stall diagnostics."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PODEM_DIR = ROOT / "PODEM"
BENCH_PATH = ROOT / "datasets" / "validation" / "b17_C.bench"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("--max-steps must be a positive integer")
    return number


def _load_cpp_podem():
    if str(PODEM_DIR) not in sys.path:
        sys.path.insert(0, str(PODEM_DIR))
    try:
        import cpp_podem
    except ImportError as exc:
        raise RuntimeError(
            "cpp_podem is unavailable; build it with "
            "'python PODEM/setup.py build_ext --inplace'"
        ) from exc
    return cpp_podem


def create_session():
    if not BENCH_PATH.is_file():
        raise FileNotFoundError("missing b17_C BENCH input: {}".format(BENCH_PATH))
    cpp_podem = _load_cpp_podem()
    return cpp_podem.StuckAtSession(
        str(BENCH_PATH),
        "",
        backtrack_limit=100,
        seed=14,
        dtc_enabled=True,
        stc_enabled=False,
    )


def run(max_steps: Optional[int] = None) -> int:
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")

    os.environ["PODEM_DTC_DIAGNOSTICS"] = "1"
    session = create_session()
    completed = 0
    previous_dtc_calls = 0
    previous_dtc_backtracks = 0

    while max_steps is None or completed < max_steps:
        remaining = session.remaining_fault_ids()
        if not remaining:
            break
        primary = remaining[0]
        step_number = completed + 1
        print(
            "[B17-DEBUG] STEP START step={} selectable={} primary={}".format(
                step_number, len(remaining), primary
            ),
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        result = session.step(primary)
        current_dtc_calls = int(result["current_dtc_secondary_calls"])
        current_dtc_backtracks = int(result["current_dtc_backtracks"])
        print(
            "[B17-DEBUG] STEP DONE step={} remaining={} dtc_attempted={} "
            "dtc_embedded={} step_dtc_calls={} step_dtc_backtracks={} "
            "elapsed_s={:.3f}".format(
                step_number,
                len(result["remaining_fault_ids"]),
                len(result["dtc_attempted_fault_ids"]),
                len(result["dtc_embedded_fault_ids"]),
                current_dtc_calls - previous_dtc_calls,
                current_dtc_backtracks - previous_dtc_backtracks,
                time.perf_counter() - started,
            ),
            file=sys.stderr,
            flush=True,
        )
        previous_dtc_calls = current_dtc_calls
        previous_dtc_backtracks = current_dtc_backtracks
        completed += 1

    return completed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose native lazy-DTC stalls on b17_C without running STC."
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        help="stop after N Primary faults; omit to run every remaining Primary",
    )
    args = parser.parse_args(argv)
    try:
        run(args.max_steps)
    except (FileNotFoundError, RuntimeError) as exc:
        print("[B17-DEBUG] ERROR {}".format(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
