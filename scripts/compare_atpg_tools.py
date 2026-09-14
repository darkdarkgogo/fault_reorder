"""Compare Atalanta and the native ordered PODEM on one manifest."""

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

from fault_order_rl.data import load_manifest
from fault_order_rl.environment import PodemEnvironment


PATTERN_LINE = re.compile(r"^\s*\d+:\s")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _count_patterns(path):
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if PATTERN_LINE.match(line))


def _run_atalanta(exe, circuit, output_dir, seed):
    circuit_dir = output_dir / "atalanta" / circuit.name
    circuit_dir.mkdir(parents=True, exist_ok=True)
    pattern_path = circuit_dir / (circuit.name + ".test")
    log_path = circuit_dir / "atalanta.log"
    if pattern_path.is_file():
        return json.loads((circuit_dir / "result.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            [str(exe), "-s", str(seed), "-t", str(pattern_path), str(circuit.bench_path)],
            cwd=str(circuit_dir), stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        "tool": "atalanta",
        "circuit": circuit.name,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "pattern_count": _count_patterns(pattern_path) if pattern_path.is_file() else None,
        "pattern_file": str(pattern_path),
        "log_file": str(log_path),
    }
    _write_json(circuit_dir / "result.json", result)
    return result


def _run_podem(environment, circuit, output_dir, backtrack_limit):
    circuit_dir = output_dir / "reorder_podem" / circuit.name
    circuit_dir.mkdir(parents=True, exist_ok=True)
    result_path = circuit_dir / "result.json"
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("backtrack_limit") == backtrack_limit:
            return saved
    catalog = environment.catalog(circuit.bench_path, circuit.faultmap_path)
    started = time.perf_counter()
    ordered_fault_ids = [item["fault_id"] for item in catalog["faults"]]
    metrics = environment.run(circuit.bench_path, circuit.faultmap_path, ordered_fault_ids)
    result = {
        "tool": "reorder_podem",
        "circuit": circuit.name,
        "backtrack_limit": backtrack_limit,
        "seconds": time.perf_counter() - started,
        **metrics,
    }
    _write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atalanta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--backtrack-limit", type=int, default=200)
    parser.add_argument("--skip-atalanta", action="store_true")
    parser.add_argument("--skip-podem", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    # This comparison does not use the learned model; only BENCH and faultmap
    # are needed for the two ATPG engines.
    circuits = manifest.circuits
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = PodemEnvironment(
        manifest.module_dir,
        backtrack_limit=args.backtrack_limit,
        seed=args.seed,
    )
    rows = {}
    for circuit in circuits:
        row = {"circuit": circuit.name}
        if not args.skip_atalanta:
            row["atalanta"] = _run_atalanta(args.atalanta.resolve(), circuit, args.output, args.seed)
        if not args.skip_podem:
            row["reorder_podem"] = _run_podem(
                environment, circuit, args.output, args.backtrack_limit
            )
        rows[circuit.name] = row
        _write_json(args.output / "circuits" / (circuit.name + ".json"), row)

    fields = [
        "circuit", "atalanta_seconds", "atalanta_pattern_count", "atalanta_returncode",
        "reorder_podem_seconds", "reorder_podem_pattern_count", "reorder_podem_coverage",
        "reorder_podem_podem_calls", "reorder_podem_total_backtracks",
    ]
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in (c.name for c in circuits):
            row = rows[name]
            atalanta = row.get("atalanta", {})
            podem = row.get("reorder_podem", {})
            writer.writerow({
                "circuit": name,
                "atalanta_seconds": atalanta.get("seconds"),
                "atalanta_pattern_count": atalanta.get("pattern_count"),
                "atalanta_returncode": atalanta.get("returncode"),
                "reorder_podem_seconds": podem.get("seconds"),
                "reorder_podem_pattern_count": podem.get("pattern_count"),
                "reorder_podem_coverage": podem.get("fault_coverage"),
                "reorder_podem_podem_calls": podem.get("podem_calls"),
                "reorder_podem_total_backtracks": podem.get("total_backtracks"),
            })


if __name__ == "__main__":
    main()
