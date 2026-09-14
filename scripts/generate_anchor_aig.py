"""Generate anchor-preserving AIGs and DeepGate2 embeddings for original faults."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PODEM"))

from fault_embedding.anchor_aig import (
    build_aigmap,
    compose_original_fault_embeddings,
    lower_to_anchor_aig,
    resolve_original_faults,
    sha256,
    validate_aigmap,
    verify_anchor_equivalence,
    write_json,
)
from fault_embedding.circuit import build_graph, read_bench
from fault_embedding.inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_SOURCE_SHA256,
    infer_gate_embeddings,
)


OUTPUT_SUFFIXES = (
    ".bench",
    ".aigmap.json",
    ".faults.json",
    ".gates.npz",
    ".fault_embeddings.npz",
)
ALLOWED_DATASET_TARGETS = frozenset({"train_AIG", "validation_AIG"})
EXPECTED_SPLIT_COUNTS = {"train": 1024, "validation": 6}
METADATA_AUDIT_PATH_FIELDS = ("atpg_bench", "aig_bench", "aigmap")


def native_path(path):
    resolved = Path(path).resolve()
    if os.name != "nt":
        return str(resolved)
    try:
        relative = os.path.relpath(str(resolved), str(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)
    return relative if relative.isascii() else str(resolved)


def load_cpp_podem():
    try:
        import cpp_podem
    except ImportError as exc:
        raise RuntimeError(
            "cpp_podem is unavailable for this Python; build PODEM/setup.py"
        ) from exc
    if not hasattr(cpp_podem, "catalog_stuck_at"):
        raise RuntimeError("cpp_podem does not provide catalog_stuck_at")
    return cpp_podem


def original_catalog(source_path):
    module = load_cpp_podem()
    return dict(module.catalog_stuck_at(native_path(source_path), ""))


def artifact_paths(directory, stem):
    directory = Path(directory)
    return {
        "aig": directory / (stem + ".bench"),
        "aigmap": directory / (stem + ".aigmap.json"),
        "metadata": directory / (stem + ".faults.json"),
        "gates": directory / (stem + ".gates.npz"),
        "embeddings": directory / (stem + ".fault_embeddings.npz"),
    }


def _generate_circuit_files(source_path, output_dir, recorded_output_dir=None):
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    recorded_output_dir = Path(recorded_output_dir or output_dir).resolve()
    paths = artifact_paths(output_dir, source_path.stem)
    recorded = artifact_paths(recorded_output_dir, source_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [path for path in paths.values() if path.exists()]
    if conflicts:
        raise ValueError("Refusing to overwrite generated artifact: {}".format(conflicts[0]))

    lowered = lower_to_anchor_aig(source_path)
    with paths["aig"].open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(lowered.text)
    aig_circuit = read_bench(paths["aig"])
    verify_anchor_equivalence(lowered.source, aig_circuit)
    graph = build_graph(aig_circuit)

    mapping = build_aigmap(
        source_path, paths["aig"], lowered, graph,
        recorded_aig_path=recorded["aig"],
    )
    write_json(paths["aigmap"], mapping)
    validate_aigmap(
        mapping, source_path, paths["aig"], graph,
        expected_aig_path=recorded["aig"],
    )

    catalog = original_catalog(source_path)
    faults = resolve_original_faults(catalog, lowered.source, graph)
    gate_arrays, provenance = infer_gate_embeddings(
        graph, checkpoint=DEFAULT_CHECKPOINT, source=DEFAULT_SOURCE, seed=0
    )
    features = compose_original_fault_embeddings(gate_arrays["hf"], faults)

    feature_layout = {
        "gate_feature": "hf",
        "gate_dimension": 128,
        "fault_dimension": 257,
        "columns": ["receiver_or_driver_hf", "connected_signal_hf", "sa_value"],
        "GO_rule": "both hf vectors use the original driver anchor",
        "GI_rule": "first hf uses receiver anchor; second uses original input signal anchor",
    }
    metadata = {
        "schema": "original_fault_anchor_embeddings_v1",
        "status": "embeddings_exported",
        "atpg_bench": str(source_path),
        "atpg_bench_sha256": sha256(source_path),
        "aig_bench": str(recorded["aig"]),
        "aig_bench_sha256": sha256(paths["aig"]),
        "aigmap": str(recorded["aigmap"]),
        "aigmap_sha256": sha256(paths["aigmap"]),
        "graph_sha256": graph["graph_sha256"],
        "fault_count": len(faults),
        "uncollapsed_total": int(catalog["uncollapsed_total"]),
        "feature_layout": feature_layout,
        "provenance": provenance,
        "faults": faults,
    }
    write_json(paths["metadata"], metadata)

    common = {
        "graph_sha256": np.asarray(graph["graph_sha256"]),
        "aig_bench_sha256": np.asarray(sha256(paths["aig"])),
        "atpg_bench_sha256": np.asarray(sha256(source_path)),
        "aigmap_sha256": np.asarray(sha256(paths["aigmap"])),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    np.savez_compressed(
        str(paths["gates"]),
        hs=gate_arrays["hs"],
        hf=gate_arrays["hf"],
        node_names=np.asarray([node["name"] for node in graph["nodes"]], dtype=str),
        node_origins=np.asarray(
            [node.get("origin") or "" for node in graph["nodes"]], dtype=str
        ),
        **common
    )
    np.savez_compressed(
        str(paths["embeddings"]),
        embeddings=features,
        fault_ids=np.asarray([fault["fault_id"] for fault in faults], dtype=str),
        gate_function_anchor_nodes=np.asarray(
            [fault["gate_function_anchor_node"] for fault in faults], dtype=np.int64
        ),
        connected_function_anchor_nodes=np.asarray(
            [fault["connected_function_anchor_node"] for fault in faults], dtype=np.int64
        ),
        sa_values=np.asarray([fault["sa_value"] for fault in faults], dtype=np.int8),
        eqv_fault_nums=np.asarray(
            [fault["eqv_fault_num"] for fault in faults], dtype=np.int64
        ),
        feature_layout_json=np.asarray(json.dumps(feature_layout, sort_keys=True)),
        **common
    )
    return {
        "name": source_path.stem,
        "source_gates": len(lowered.source.gates),
        "aig_gates": len(aig_circuit.gates),
        "feature_nodes": len(graph["nodes"]),
        "helpers": len(lowered.helper_origins),
        "faults": len(faults),
        "uncollapsed_faults": int(catalog["uncollapsed_total"]),
    }


def generate_circuit(source_path, output_dir):
    """Generate a single circuit transactionally into a new directory."""
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Single-circuit output directory already exists: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Path.mkdir inherits the dataset ACL on Windows. tempfile.mkdtemp uses a
    # private directory ACL, which would make an atomically installed result
    # unreadable to other local training processes.
    temporary = output_dir.parent / ".{}-anchor-aig-{}".format(
        source_path.stem, uuid.uuid4().hex
    )
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        result = _generate_circuit_files(source_path, temporary, output_dir)
        validate_generated_circuit(
            source_path, temporary, recorded_output_dir=output_dir
        )
        os.replace(str(temporary), str(output_dir))
        return result
    finally:
        if temporary.is_dir():
            shutil.rmtree(str(temporary))


def validate_generated_circuit(source_path, output_dir, recorded_output_dir=None):
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    recorded_output_dir = Path(recorded_output_dir or output_dir).resolve()
    paths = artifact_paths(output_dir, source_path.stem)
    recorded = artifact_paths(recorded_output_dir, source_path.stem)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Missing generated artifact: {}".format(missing[0]))
    source = read_bench(source_path)
    aig = read_bench(paths["aig"])
    verify_anchor_equivalence(source, aig)
    graph = build_graph(aig)
    try:
        mapping = json.loads(paths["aigmap"].read_text(encoding="utf-8"))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read generated metadata: {}".format(exc))
    validate_aigmap(
        mapping, source_path, paths["aig"], graph,
        expected_aig_path=recorded["aig"],
    )
    catalog = original_catalog(source_path)
    expected_faults = resolve_original_faults(catalog, source, graph)
    if metadata.get("schema") != "original_fault_anchor_embeddings_v1":
        raise ValueError("Unsupported original fault embedding metadata")
    if metadata.get("atpg_bench_sha256") != sha256(source_path):
        raise ValueError("ATPG BENCH hash mismatch in metadata")
    if metadata.get("aig_bench_sha256") != sha256(paths["aig"]):
        raise ValueError("AIG BENCH hash mismatch in metadata")
    if metadata.get("aigmap_sha256") != sha256(paths["aigmap"]):
        raise ValueError("AIG mapping hash mismatch in metadata")
    if metadata.get("graph_sha256") != graph["graph_sha256"]:
        raise ValueError("AIG graph hash mismatch in metadata")
    for key in METADATA_AUDIT_PATH_FIELDS:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError("Invalid audit path in metadata: {}".format(key))
    if metadata.get("faults") != expected_faults:
        raise ValueError("Original fault metadata is not aligned with current catalog")
    if metadata.get("fault_count") != len(expected_faults):
        raise ValueError("Original fault count mismatch")
    if metadata.get("uncollapsed_total") != int(catalog["uncollapsed_total"]):
        raise ValueError("Original uncollapsed fault total mismatch")

    ids = tuple(fault["fault_id"] for fault in expected_faults)
    eqv = np.asarray([fault["eqv_fault_num"] for fault in expected_faults], dtype=np.int64)
    with np.load(str(paths["gates"]), allow_pickle=False) as arrays:
        hf = np.asarray(arrays["hf"], dtype=np.float32)
        hs = np.asarray(arrays["hs"], dtype=np.float32)
        if hf.shape != (len(graph["nodes"]), 128) or hs.shape != hf.shape:
            raise ValueError("DeepGate2 gate embedding shape mismatch")
        if not np.isfinite(hf).all() or not np.isfinite(hs).all():
            raise ValueError("DeepGate2 gate embeddings contain non-finite values")
        if str(arrays["graph_sha256"].item()) != graph["graph_sha256"]:
            raise ValueError("Gate embedding graph hash mismatch")
        expected_names = np.asarray([node["name"] for node in graph["nodes"]], dtype=str)
        expected_origins = np.asarray(
            [node.get("origin") or "" for node in graph["nodes"]], dtype=str
        )
        if not np.array_equal(np.asarray(arrays["node_names"], dtype=str), expected_names):
            raise ValueError("Gate embedding node names do not match graph")
        if not np.array_equal(np.asarray(arrays["node_origins"], dtype=str), expected_origins):
            raise ValueError("Gate embedding node origins do not match graph")
        for key, value in (
            ("aig_bench_sha256", sha256(paths["aig"])),
            ("atpg_bench_sha256", sha256(source_path)),
            ("aigmap_sha256", sha256(paths["aigmap"])),
        ):
            if str(arrays[key].item()) != value:
                raise ValueError("Gate embedding {} mismatch".format(key))
    with np.load(str(paths["embeddings"]), allow_pickle=False) as arrays:
        features = np.asarray(arrays["embeddings"], dtype=np.float32)
        embedded_ids = tuple(str(value) for value in arrays["fault_ids"].tolist())
        embedded_eqv = np.asarray(arrays["eqv_fault_nums"], dtype=np.int64)
        provenance = json.loads(str(arrays["provenance_json"].item()))
        if features.shape != (len(ids), 257) or not np.isfinite(features).all():
            raise ValueError("Original fault embedding matrix is invalid")
        if embedded_ids != ids or not np.array_equal(embedded_eqv, eqv):
            raise ValueError("Original fault embedding rows do not match catalog")
        expected_gate_anchors = np.asarray(
            [fault["gate_function_anchor_node"] for fault in expected_faults], dtype=np.int64
        )
        expected_connected_anchors = np.asarray(
            [fault["connected_function_anchor_node"] for fault in expected_faults], dtype=np.int64
        )
        expected_sa = np.asarray(
            [fault["sa_value"] for fault in expected_faults], dtype=np.int8
        )
        if not np.array_equal(arrays["gate_function_anchor_nodes"], expected_gate_anchors):
            raise ValueError("Fault embedding gate anchors do not match catalog")
        if not np.array_equal(arrays["connected_function_anchor_nodes"], expected_connected_anchors):
            raise ValueError("Fault embedding connected anchors do not match catalog")
        if not np.array_equal(arrays["sa_values"], expected_sa):
            raise ValueError("Fault embedding SA values do not match catalog")
        recomposed = compose_original_fault_embeddings(hf, expected_faults)
        if not np.array_equal(features, recomposed):
            raise ValueError("Fault embeddings do not equal their referenced gate embeddings")
        if str(arrays["graph_sha256"].item()) != graph["graph_sha256"]:
            raise ValueError("Fault embedding graph hash mismatch")
        for key, value in (
            ("aig_bench_sha256", sha256(paths["aig"])),
            ("atpg_bench_sha256", sha256(source_path)),
            ("aigmap_sha256", sha256(paths["aigmap"])),
        ):
            if str(arrays[key].item()) != value:
                raise ValueError("Fault embedding {} mismatch".format(key))
        expected_layout = json.dumps(metadata["feature_layout"], sort_keys=True)
        if str(arrays["feature_layout_json"].item()) != expected_layout:
            raise ValueError("Fault embedding feature layout mismatch")
        if provenance != metadata.get("provenance"):
            raise ValueError("Fault embedding provenance does not match metadata")
        if provenance.get("checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("Fault embeddings do not use the official checkpoint")
        if provenance.get("source_sha256") != OFFICIAL_SOURCE_SHA256:
            raise ValueError("Fault embeddings do not use the pinned DeepGate2 source")
    return len(ids)


def ensure_safe_dataset_target(dataset_root, target, expected_name):
    dataset_root, target = Path(dataset_root).resolve(), Path(target).resolve()
    canonical_root = (ROOT / "datasets").resolve()
    if (dataset_root != canonical_root
            or expected_name not in ALLOWED_DATASET_TARGETS
            or target.parent != dataset_root or target.name != expected_name):
        raise ValueError("Unsafe dataset replacement target: {}".format(target))
    return target


def stage_directory_replacement(dataset_root, replacements):
    dataset_root = Path(dataset_root).resolve()
    if set(replacements) - ALLOWED_DATASET_TARGETS:
        raise ValueError("Only train_AIG and validation_AIG may be replaced")
    transaction = uuid.uuid4().hex
    operations = []
    for name, staged in replacements.items():
        final = ensure_safe_dataset_target(dataset_root, dataset_root / name, name)
        staged = Path(staged).resolve()
        if staged.parent.parent != dataset_root or staged.name != name or not staged.is_dir():
            raise ValueError("Unsafe staged dataset directory: {}".format(staged))
        if not final.is_dir():
            raise ValueError("Existing AIG directory is missing: {}".format(final))
        backup = dataset_root / (".{}-old-{}".format(name, transaction))
        if backup.exists():
            raise ValueError("Replacement backup already exists: {}".format(backup))
        operations.append({"name": name, "staged": staged, "final": final,
                           "backup": backup})
    try:
        for operation in operations:
            os.replace(str(operation["final"]), str(operation["backup"]))
            os.replace(str(operation["staged"]), str(operation["final"]))
    except BaseException:
        rollback_directory_replacement(operations)
        raise
    return operations


def rollback_directory_replacement(operations):
    for operation in reversed(operations):
        final, backup = operation["final"], operation["backup"]
        if backup.is_dir():
            if final.is_dir():
                shutil.rmtree(str(final))
            os.replace(str(backup), str(final))


def finalize_directory_replacement(operations):
    for operation in operations:
        backup = operation["backup"]
        if backup.is_dir():
            shutil.rmtree(str(backup))


def replace_directories(dataset_root, replacements):
    operations = stage_directory_replacement(dataset_root, replacements)
    finalize_directory_replacement(operations)


def source_files(directory, expected_count=None):
    files = sorted(Path(directory).resolve().glob("*.bench"), key=lambda path: path.name)
    if not files:
        raise ValueError("No source BENCH files in {}".format(directory))
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(
            "Expected {} source BENCH files in {}, found {}".format(
                expected_count, directory, len(files)
            )
        )
    return files


def check_split(source_dir, output_dir, recorded_output_dir=None, expected_count=None):
    sources = source_files(source_dir, expected_count=expected_count)
    output_dir = Path(output_dir).resolve()
    recorded_output_dir = Path(recorded_output_dir or output_dir).resolve()
    expected = set()
    total_faults = 0
    for source in sources:
        validate_generated_circuit(
            source, output_dir, recorded_output_dir=recorded_output_dir
        )
        total_faults += json.loads(
            (Path(output_dir) / (source.stem + ".faults.json")).read_text(encoding="utf-8")
        )["fault_count"]
        expected.update(path.name for path in artifact_paths(output_dir, source.stem).values())
    entries = list(output_dir.iterdir())
    actual = {path.name for path in entries if path.is_file()}
    unexpected_directories = sorted(path.name for path in entries if path.is_dir())
    if unexpected_directories:
        raise ValueError(
            "Unexpected directories in generated split: {}".format(unexpected_directories)
        )
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError("Generated file set mismatch; extra={}, missing={}".format(extra, missing))
    return {"circuits": len(sources), "faults": total_faults}


def directory_file_hashes(directory):
    directory = Path(directory).resolve()
    entries = list(directory.iterdir())
    directories = sorted(path.name for path in entries if path.is_dir())
    if directories:
        raise ValueError("Unexpected directories in artifact split: {}".format(directories))
    return {
        path.name: sha256(path)
        for path in sorted(entries, key=lambda item: item.name)
        if path.is_file()
    }


def validate_dataset_summary(dataset_root, checked_splits):
    """Validate the committed inventory without trusting generation-machine paths."""
    dataset_root = Path(dataset_root).resolve()
    summary_path = dataset_root / "anchor_aig_dataset.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read anchor AIG dataset summary: {}".format(exc))
    if summary.get("schema") != "anchor_aig_dataset_v1" or summary.get("status") != "complete":
        raise ValueError("Unsupported or incomplete anchor AIG dataset summary")
    if summary.get("deepgate_checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("Anchor AIG dataset checkpoint provenance changed")
    if summary.get("deepgate_source_sha256") != OFFICIAL_SOURCE_SHA256:
        raise ValueError("Anchor AIG dataset source provenance changed")

    expected_hashes = {
        name: directory_file_hashes(dataset_root / name)
        for name in checked_splits
    }
    if summary.get("file_sha256") != expected_hashes:
        raise ValueError("Anchor AIG dataset file inventory or hashes changed")
    recorded_splits = summary.get("splits", {})
    if set(recorded_splits) != set(checked_splits):
        raise ValueError("Anchor AIG dataset split inventory changed")
    for name, checked in checked_splits.items():
        recorded = recorded_splits.get(name, {})
        for key in ("circuits", "faults"):
            if recorded.get(key) != checked[key]:
                raise ValueError(
                    "Anchor AIG dataset {} {} summary changed".format(name, key)
                )
    return {
        "schema": summary["schema"],
        "status": summary["status"],
        "files": sum(len(files) for files in expected_hashes.values()),
    }


def validate_existing_inventory(source_dir, aig_dir, expected_count):
    sources = source_files(source_dir, expected_count=expected_count)
    existing = source_files(aig_dir, expected_count=expected_count)
    source_names = {path.name for path in sources}
    existing_names = {path.name for path in existing}
    if source_names != existing_names:
        raise ValueError(
            "Existing AIG inventory does not match source split; extra={}, missing={}".format(
                sorted(existing_names - source_names), sorted(source_names - existing_names)
            )
        )
    return sources


def run_b12_original_order_smoke(dataset_root):
    source = Path(dataset_root).resolve() / "validation" / "b12_C.bench"
    catalog = original_catalog(source)
    ids = [str(item["fault_id"]) for item in catalog["faults"]]
    module = load_cpp_podem()
    raw = dict(module.run_stuck_at_ordered(native_path(source), "", ids, 5000, 14))
    required = (
        "pattern_count", "detected_collapsed_faults", "detected_equivalent_faults",
        "uncollapsed_faults", "aborted_faults", "redundant_faults",
        "redundant_equivalent_faults", "podem_calls", "total_backtracks",
    )
    result = {key: int(raw[key]) for key in required}
    result["collapsed_faults"] = len(ids)
    result["covered_equivalent_faults"] = (
        result["detected_equivalent_faults"]
        + result["redundant_equivalent_faults"]
    )
    if (len(ids) != 3444 or int(catalog["uncollapsed_total"]) != 5906
            or result["uncollapsed_faults"] != 5906
            or result["covered_equivalent_faults"] != 5906
            or result["pattern_count"] != 218):
        raise ValueError("b12_C original-order ATPG baseline changed: {}".format(result))
    return result


def batch_generate(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    if dataset_root != (ROOT / "datasets").resolve():
        raise ValueError("Batch replacement is restricted to the repository datasets directory")
    splits = {"train": "train_AIG", "validation": "validation_AIG"}
    sources_by_split = {}
    for source_name, output_name in splits.items():
        sources_by_split[source_name] = validate_existing_inventory(
            dataset_root / source_name,
            ensure_safe_dataset_target(dataset_root, dataset_root / output_name, output_name),
            EXPECTED_SPLIT_COUNTS[source_name],
        )
    baseline = run_b12_original_order_smoke(dataset_root)
    stage_root = dataset_root / ".anchor-aig-stage-{}".format(uuid.uuid4().hex)
    stage_root.mkdir(parents=False, exist_ok=False)
    summaries = {}
    staged_hashes = {}
    started = time.perf_counter()
    try:
        for source_name, output_name in splits.items():
            source_dir = dataset_root / source_name
            stage_dir = stage_root / output_name
            final_dir = dataset_root / output_name
            sources = sources_by_split[source_name]
            stage_dir.mkdir(parents=True, exist_ok=False)
            aggregate = {
                "circuits": 0, "faults": 0, "uncollapsed_faults": 0,
                "feature_nodes": 0, "helpers": 0,
            }
            for index, source in enumerate(sources, 1):
                result = _generate_circuit_files(source, stage_dir, final_dir)
                aggregate["circuits"] += 1
                aggregate["faults"] += result["faults"]
                aggregate["uncollapsed_faults"] += result["uncollapsed_faults"]
                aggregate["feature_nodes"] += result["feature_nodes"]
                aggregate["helpers"] += result["helpers"]
                print(
                    "[{}/{}] {}: nodes={} faults={}".format(
                        index, len(sources), source.name,
                        result["feature_nodes"], result["faults"],
                    ),
                    flush=True,
                )
            checked = check_split(
                source_dir, stage_dir, recorded_output_dir=final_dir,
                expected_count=EXPECTED_SPLIT_COUNTS[source_name],
            )
            if checked["circuits"] != aggregate["circuits"] or checked["faults"] != aggregate["faults"]:
                raise ValueError("Staging summary changed during validation")
            summaries[output_name] = aggregate
            staged_hashes[output_name] = directory_file_hashes(stage_dir)

        operations = stage_directory_replacement(
            dataset_root, {name: stage_root / name for name in splits.values()}
        )
        try:
            for output_name in splits.values():
                installed = directory_file_hashes(dataset_root / output_name)
                if installed != staged_hashes[output_name]:
                    raise ValueError("Installed artifact hashes changed for {}".format(output_name))
            after = run_b12_original_order_smoke(dataset_root)
            if after != baseline:
                raise ValueError("b12_C ATPG result changed after AIG replacement")
        except BaseException:
            rollback_directory_replacement(operations)
            raise
        finalize_directory_replacement(operations)
        elapsed_seconds = time.perf_counter() - started
        write_json(dataset_root / "anchor_aig_dataset.json", {
            "schema": "anchor_aig_dataset_v1",
            "status": "complete",
            "deepgate_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
            "deepgate_source_sha256": OFFICIAL_SOURCE_SHA256,
            "b12_original_order_atpg": baseline,
            "file_sha256": staged_hashes,
            "elapsed_seconds": elapsed_seconds,
            "splits": summaries,
        })
        result = dict(summaries)
        result["seconds"] = elapsed_seconds
        return result
    finally:
        if stage_root.is_dir():
            shutil.rmtree(str(stage_root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (args.source is None) != (args.output_dir is None):
            raise ValueError("--source and --output-dir must be supplied together")
        if args.source is not None:
            if args.check_only:
                count = validate_generated_circuit(args.source, args.output_dir)
                result = {"validated": str(args.source.resolve()), "faults": count}
            else:
                result = generate_circuit(args.source, args.output_dir)
        elif args.check_only:
            checked = {
                "train_AIG": check_split(
                    args.dataset_root / "train", args.dataset_root / "train_AIG",
                    expected_count=EXPECTED_SPLIT_COUNTS["train"],
                ),
                "validation_AIG": check_split(
                    args.dataset_root / "validation", args.dataset_root / "validation_AIG",
                    expected_count=EXPECTED_SPLIT_COUNTS["validation"],
                ),
            }
            result = dict(checked)
            result["dataset_summary"] = validate_dataset_summary(
                args.dataset_root, checked
            )
        else:
            result = batch_generate(args.dataset_root)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except KeyboardInterrupt:
        print("Interrupted before a complete dataset replacement.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
