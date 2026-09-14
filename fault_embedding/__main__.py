"""Run with python -m fault_embedding prepare/export --help."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from .circuit import build_graph, read_bench
from .faults import read_faultmap
from .features import compose_fault_embeddings
from .inference import DEFAULT_CHECKPOINT, DEFAULT_SOURCE, infer_gate_embeddings


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args):
    circuit = read_bench(args.bench)
    graph = build_graph(circuit)
    faultmap = args.faultmap or args.bench.with_suffix(".faultmap")
    metadata = read_faultmap(faultmap, circuit, graph)
    metadata["bench_path"] = str(args.bench.resolve())
    metadata["faultmap_path"] = str(faultmap.resolve())
    prefix = args.bench.stem
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise ValueError("Output directory already exists; choose a new --out-dir")
    final_paths = [out_dir / f"{prefix}.{suffix}" for suffix in ("graph.json", "faults.json")]
    if args.command == "export":
        final_paths.extend(out_dir / f"{prefix}.{suffix}" for suffix in ("gates.npz", "fault_embeddings.npz"))
    if args.command == "export":
        gate_arrays, provenance = infer_gate_embeddings(graph, args.checkpoint, args.deepgate_source, args.seed)
        functional_emb = gate_arrays["hf"]
        features = compose_fault_embeddings(functional_emb, metadata["faults"])
        metadata["feature_layout"] = {"gate_feature": "hf",
                                     "gate_dimension": functional_emb.shape[1],
                                     "fault_dimension": features.shape[1],
                                     "columns": ["gate_hf", "connected_hf", "sa_value"],
                                     "GO_rule": "connected_hf equals gate_hf",
                                     "GI_rule": "connected_hf comes from the input source signal"}
        metadata["provenance"] = provenance
        metadata["status"] = "embeddings_exported"
    else:
        metadata["status"] = "positions_prepared_no_embeddings"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    paths = [temp_dir / path.name for path in final_paths]
    try:
        write_json(paths[0], graph)
        write_json(paths[1], metadata)
        if args.command == "export":
            common = {"graph_sha256": np.asarray(graph["graph_sha256"]),
                      "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True))}
            np.savez_compressed(paths[2], hf=gate_arrays["hf"], **common,
                                node_names=np.asarray([node["name"] for node in graph["nodes"]]))
            records = metadata["faults"]
            np.savez_compressed(paths[3], embeddings=features, **common,
                                fault_ids=np.asarray([f["fault_id"] for f in records], dtype=str),
                                gate_function_anchor_nodes=np.asarray(
                                    [f["gate_function_anchor_node"] for f in records], dtype=np.int64),
                                connected_function_anchor_nodes=np.asarray(
                                    [f["connected_function_anchor_node"] for f in records], dtype=np.int64),
                                sa_values=features[:, -1].astype(np.int8),
                                eqv_fault_nums=np.asarray([f["eqv_fault_num"] for f in records], dtype=np.int64),
                                feature_layout_json=np.asarray(json.dumps(metadata["feature_layout"])))
        os.replace(temp_dir, out_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    if args.command == "export":
        print(f"Exported {features.shape[0]} collapsed fault embeddings, dimension {features.shape[1]}")
    else:
        print(f"Prepared {metadata['fault_count']} collapsed faults; no numeric embeddings generated")
    for path in final_paths:
        print(path.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collapsed fault features: [gate_hf, connected_hf, SA]")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("--bench", type=Path, required=True, help="Binary BENCH (unary NOT/BUFF accepted)")
        sub.add_argument("--faultmap", type=Path, help="Companion V2/V3 map; defaults to BENCH with .faultmap suffix")
        sub.add_argument("--out-dir", type=Path, required=True, help="Directory with no conflicting output files")
        if command == "export":
            sub.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
            sub.add_argument("--deepgate-source", type=Path, default=DEFAULT_SOURCE)
            sub.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
