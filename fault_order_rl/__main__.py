"""Command line entry point: validate, train, evaluate."""

import argparse
import json
from pathlib import Path
import sys

from .data import load_all_circuits, load_manifest
from .environment import PodemEnvironment
from .trainer import TrainConfig, Trainer, evaluate_checkpoint


def _print_evaluation_by_circuit(report):
    columns = (
        ("circuit", "circuit"),
        ("checkpoint", "checkpoint_kind"),
        ("round", "round"),
        ("native_cov", "native_fault_coverage"),
        ("model_cov", "model_fault_coverage"),
        ("cov_delta_pp", "fault_coverage_increase_percentage_points"),
        ("native_patterns", "native_pattern_count"),
        ("model_patterns", "model_pattern_count"),
        ("pattern_reduction", "pattern_reduction"),
        ("reduction_pct", "pattern_reduction_percent"),
    )
    print("\t".join(label for label, _ in columns))
    for comparison in report["comparison_by_circuit"].values():
        values = []
        for _, key in columns:
            value = comparison[key]
            if value is None:
                values.append("N/A")
            elif key in ("native_fault_coverage", "model_fault_coverage"):
                values.append("{:.6%}".format(value))
            elif key in ("fault_coverage_increase_percentage_points",
                         "pattern_reduction_percent"):
                values.append("{:+.6f}".format(value))
            else:
                values.append(str(value))
        print("\t".join(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared fault ordering with listwise REINFORCE")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate all artifacts without ATPG episodes")
    validate.add_argument("--manifest", type=Path, required=True)
    train = commands.add_parser("train")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--resume", type=Path)
    train.add_argument("--output", type=Path)
    defaults = TrainConfig()
    for name, value in vars(defaults).items():
        if name == "backtrack_limit":
            continue
        train.add_argument("--" + name.replace("_", "-"), type=type(value), default=None)
    evaluate = commands.add_parser(
        "evaluate", help="Fresh deterministic best/latest evaluation and rank export")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path,
                          help="Evaluate the checkpoint on a separate manifest")
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            circuits = load_all_circuits(manifest, PodemEnvironment(manifest.module_dir))
            print(json.dumps({"validated": len(circuits), "faults": {c.name: c.fault_count for c in circuits}}, indent=2))
        elif args.command == "train":
            argument_values = vars(args)
            supplied = {
                name: argument_values[name]
                for name in vars(defaults)
                if name in argument_values and argument_values[name] is not None
            }
            if args.resume:
                if args.output is not None or set(supplied) - {"rounds"}:
                    raise ValueError("resume preserves saved configuration/output; only --rounds may extend the target")
                trainer = Trainer.resume(args.resume, args.rounds)
            else:
                if args.output is None:
                    raise ValueError("new training requires --output")
                trainer = Trainer.create(args.manifest, TrainConfig(**supplied), args.output)
            report = trainer.train()
            print(json.dumps({"checkpoint_kind": report.get("checkpoint_kind", "best"),
                              "round": report["round"],
                              "coverage_eligible": report.get("coverage_eligible", report.get("eligible")),
                              "coverage_shortfall": report.get("coverage_shortfall", 0),
                              "patterns": report["totals"]["pattern_count"],
                              "pattern_reduction": report["pattern_reduction"]}, indent=2))
        else:
            if args.output is not None:
                output = args.output
            elif args.manifest is not None:
                output = (args.checkpoint.resolve().parent
                          / ("evaluation-" + args.manifest.stem))
            else:
                output = args.checkpoint.resolve().parent / "evaluation"
            report = evaluate_checkpoint(args.checkpoint, output, manifest=args.manifest)
            _print_evaluation_by_circuit(report)
    except KeyboardInterrupt:
        print("Interrupted; resume from the last committed latest.pt.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
