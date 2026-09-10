"""Command line entry point: validate, train, evaluate."""

import argparse
import json
from pathlib import Path
import sys

from .data import load_all_circuits, load_manifest
from .environment import PodemEnvironment
from .trainer import TrainConfig, Trainer, evaluate_checkpoint


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
    evaluate = commands.add_parser("evaluate", help="Fresh deterministic best evaluation and rank export")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            circuits = load_all_circuits(manifest, PodemEnvironment(manifest.module_dir))
            print(json.dumps({"validated": len(circuits), "faults": {c.name: c.fault_count for c in circuits}}, indent=2))
        elif args.command == "train":
            supplied = {name: getattr(args, name) for name in vars(defaults) if getattr(args, name) is not None}
            if args.resume:
                if args.output is not None or set(supplied) - {"rounds"}:
                    raise ValueError("resume preserves saved configuration/output; only --rounds may extend the target")
                trainer = Trainer.resume(args.resume, args.rounds)
            else:
                if args.output is None:
                    raise ValueError("new training requires --output")
                trainer = Trainer.create(args.manifest, TrainConfig(**supplied), args.output)
            report = trainer.train()
            print(json.dumps({"best_round": report["round"], "patterns": report["totals"]["pattern_count"],
                              "pattern_reduction": report["pattern_reduction"]}, indent=2))
        else:
            output = args.output or args.checkpoint.resolve().parent / "evaluation"
            report = evaluate_checkpoint(args.checkpoint, output)
            print(json.dumps(report["totals"], indent=2))
    except KeyboardInterrupt:
        print("Interrupted; resume from the last committed latest.pt.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
