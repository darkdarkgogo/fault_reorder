"""Materialize DeepTPI ITC22 graphs as reorderATPG training artifacts."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PODEM"))
sys.path.insert(0, str(ROOT / "PODEM" / "scripts"))

from convert_binary_bench import convert_binary_bench
from fault_embedding.__main__ import run as export_fault_embeddings
from fault_embedding.inference import DEFAULT_CHECKPOINT, DEFAULT_SOURCE as DEFAULT_DEEPGATE_SOURCE
from fault_order_rl.checkpoint import write_json
from fault_order_rl.data import _sha256
from fault_order_rl.deeptpi_import import graph_to_bench, load_graphs, sha256


DEFAULT_SOURCE = ROOT.parent / "DeepTPI-main" / "DeepTPI-main" / "data" / "ITC22_dataset"
DEFAULT_OUTPUT = ROOT / "datasets" / "deeptpi_itc22"


def copy_verified(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise ValueError("missing DeepTPI source NPZ: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256(source) != sha256(destination):
            raise ValueError("existing raw NPZ differs from source: {}".format(destination))
        return
    temporary = destination.with_name("." + destination.name + ".copying")
    shutil.copy2(str(source), str(temporary))
    os.replace(str(temporary), str(destination))


def write_binding(faultmap, binary):
    lines = [line.split() for line in faultmap.read_text(encoding="utf-8").splitlines()]
    headers = {parts[0]: parts[1] for parts in lines if len(parts) == 2}
    required = {"source_hash", "circuit_hash"}
    if not required.issubset(headers):
        raise ValueError("generated fault map is missing hash headers")
    write_json(Path(str(faultmap) + ".binding.json"), {
        "bench_sha256": _sha256(binary),
        "faultmap_sha256": _sha256(faultmap),
        "source_hash": headers["source_hash"],
        "circuit_hash": headers["circuit_hash"],
    })


def expected_files(directory, name):
    binary_stem = name + "_binary"
    return [
        directory / (name + ".bench"),
        directory / (binary_stem + ".bench"),
        directory / (binary_stem + ".faultmap"),
        directory / (binary_stem + ".faultmap.binding.json"),
        directory / "embeddings" / (binary_stem + ".faults.json"),
        directory / "embeddings" / (binary_stem + ".gates.npz"),
        directory / "embeddings" / (binary_stem + ".fault_embeddings.npz"),
        directory / "import.json",
    ]


def prepare_circuit(dataset_root, split, name, graph, source_npz_sha256):
    split_dir = dataset_root / split
    final = split_dir / name
    if final.is_dir():
        missing = [path for path in expected_files(final, name) if not path.is_file()]
        if missing:
            raise ValueError("incomplete existing circuit directory: {}".format(final))
        metadata = json.loads((final / "import.json").read_text(encoding="utf-8"))
        if metadata.get("source_npz_sha256") != source_npz_sha256:
            raise ValueError("existing circuit came from a different NPZ: {}".format(name))
        print("Existing circuit: {} / {}".format(split, name), flush=True)
        return
    split_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + name + ".", dir=str(split_dir)))
    try:
        source = temporary / (name + ".bench")
        binary = temporary / (name + "_binary.bench")
        faultmap = temporary / (name + "_binary.faultmap")
        source.write_text(graph_to_bench(name, graph), encoding="utf-8")
        stats = convert_binary_bench(source, binary, faultmap)
        write_binding(faultmap, binary)
        embeddings = temporary / "embeddings"
        export_fault_embeddings(SimpleNamespace(
            command="export",
            bench=binary,
            faultmap=faultmap,
            out_dir=embeddings,
            checkpoint=DEFAULT_CHECKPOINT,
            deepgate_source=DEFAULT_DEEPGATE_SOURCE,
            seed=0,
        ))
        write_json(temporary / "import.json", {
            "schema": "deeptpi_itc22_import_v1",
            "split": split,
            "circuit": name,
            "source_npz_sha256": source_npz_sha256,
            "source_bench_sha256": _sha256(source),
            "binary_bench_sha256": _sha256(binary),
            "faultmap_sha256": _sha256(faultmap),
            "conversion": stats,
        })
        os.replace(str(temporary), str(final))
        print("Prepared circuit: {} / {}".format(split, name), flush=True)
    except BaseException:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def manifest_entry(split, name, dataset_root, config_dir):
    base = Path(os.path.relpath(dataset_root / split / name, config_dir))
    binary_stem = name + "_binary"
    return {
        "name": name,
        "bench": (base / (binary_stem + ".bench")).as_posix(),
        "faultmap": (base / (binary_stem + ".faultmap")).as_posix(),
        "embeddings": (base / "embeddings" / (binary_stem + ".fault_embeddings.npz")).as_posix(),
        "metadata": (base / "embeddings" / (binary_stem + ".faults.json")).as_posix(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=9)
    args = parser.parse_args(argv)
    try:
        output = args.output_root.resolve()
        sources = {
            split: args.source_root.resolve() / split / "benchmarks_circuits_graphs.npz"
            for split in ("train", "test")
        }
        raw = {
            split: output / "raw" / split / "benchmarks_circuits_graphs.npz"
            for split in ("train", "test")
        }
        for split in ("train", "test"):
            copy_verified(sources[split], raw[split])
        graphs = {
            "train": load_graphs(raw["train"], args.train_count),
            "test": load_graphs(raw["test"], args.test_count),
        }
        digests = {split: sha256(raw[split]) for split in raw}
        train_names = {name for name, _ in graphs["train"]}
        test_names = {name for name, _ in graphs["test"]}
        if train_names & test_names:
            raise ValueError("training and test circuit names overlap")
        for split in ("train", "test"):
            for name, graph in graphs[split]:
                prepare_circuit(output, split, name, graph, digests[split])

        args.config_dir.mkdir(parents=True, exist_ok=True)
        manifest_paths = {}
        podem_dir = Path(os.path.relpath(ROOT / "PODEM", args.config_dir.resolve())).as_posix()
        for split in ("train", "test"):
            path = args.config_dir / "deeptpi_itc22_{}_{}.json".format(
                split, len(graphs[split]))
            write_json(path, {
                "version": 1,
                "cpp_podem_dir": podem_dir,
                "circuits": [
                    manifest_entry(split, name, output, args.config_dir.resolve())
                    for name, _ in graphs[split]
                ],
            })
            manifest_paths[split] = str(path.resolve())
        write_json(output / "dataset.json", {
            "schema": "deeptpi_itc22_dataset_v1",
            "status": "complete",
            "source_npz_sha256": digests,
            "selection": {"train": "first {}".format(args.train_count),
                          "test": "first {} (complete source set)".format(args.test_count)},
            "circuits": {split: [name for name, _ in graphs[split]] for split in graphs},
            "manifests": manifest_paths,
        })
        print(json.dumps({"prepared": {split: len(values) for split, values in graphs.items()},
                          "manifests": manifest_paths}, indent=2), flush=True)
    except (ValueError, RuntimeError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
