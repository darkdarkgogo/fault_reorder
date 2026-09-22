"""Manifest loading and strict embedding/catalog validation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from fault_embedding.circuit import build_graph, read_bench
from fault_embedding.inference import OFFICIAL_CHECKPOINT_SHA256, OFFICIAL_SOURCE_SHA256

from .progress import progress


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    bench_path: Path
    faultmap_path: Path
    embeddings_path: Path
    metadata_path: Path
    aig_bench_path: Path = None
    aigmap_path: Path = None


@dataclass
class CircuitData:
    spec: CircuitSpec
    embeddings: torch.Tensor
    fault_ids: tuple
    eqv_fault_nums: np.ndarray
    artifact_digest: str

    @property
    def name(self):
        return self.spec.name

    @property
    def fault_count(self):
        return len(self.fault_ids)


@dataclass
class TrainingManifest:
    path: Path
    module_dir: Path
    circuits: list
    digest: str


def load_manifest(path):
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read training manifest {}: {}".format(path, exc))
    if raw.get("version") != 1:
        raise ValueError("training manifest version must be 1")
    base = path.parent
    module_dir = (base / raw.get("cpp_podem_dir", "../PODEM")).resolve()
    entries = raw.get("circuits")
    if not isinstance(entries, list) or not entries:
        raise ValueError("training manifest must contain circuits")
    specs = []
    names = set()
    for entry in entries:
        name = str(entry.get("name", ""))
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("circuit names must be unique safe file names")
        names.add(name)
        faultmap = entry.get("faultmap")
        aig_bench = entry.get("aig_bench")
        aigmap = entry.get("aigmap")
        specs.append(
            CircuitSpec(
                name=name,
                bench_path=(base / entry["bench"]).resolve(),
                faultmap_path=(base / faultmap).resolve() if faultmap else None,
                embeddings_path=(base / entry["embeddings"]).resolve(),
                metadata_path=(base / entry["metadata"]).resolve(),
                aig_bench_path=(base / aig_bench).resolve() if aig_bench else None,
                aigmap_path=(base / aigmap).resolve() if aigmap else None,
            )
        )
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return TrainingManifest(path, module_dir, specs, hashlib.sha256(canonical).hexdigest())


def load_circuit_data(spec, catalog):
    paths = [spec.bench_path, spec.embeddings_path, spec.metadata_path]
    paths.extend(path for path in (
        spec.faultmap_path, spec.aig_bench_path, spec.aigmap_path
    ) if path is not None)
    for path in paths:
        if not path.is_file():
            raise ValueError("missing circuit artifact: {}".format(path))

    try:
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        with np.load(str(spec.embeddings_path), allow_pickle=False) as arrays:
            embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
            fault_ids = tuple(str(value) for value in arrays["fault_ids"].tolist())
            eqv_fault_nums = np.asarray(arrays["eqv_fault_nums"], dtype=np.int64)
            graph_sha256 = str(arrays["graph_sha256"].item())
            provenance = json.loads(str(arrays["provenance_json"].item()))
            embedding_hashes = {
                key: str(arrays[key].item())
                for key in ("atpg_bench_sha256", "aig_bench_sha256", "aigmap_sha256")
                if key in arrays
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load embeddings for {}: {}".format(spec.name, exc))

    if embeddings.ndim != 2 or embeddings.shape[1] != 257:
        raise ValueError("{} embeddings must have shape [N, 257]".format(spec.name))
    if embeddings.shape[0] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("{} embeddings are empty or non-finite".format(spec.name))
    if len(fault_ids) != embeddings.shape[0] or len(set(fault_ids)) != len(fault_ids):
        raise ValueError("{} embedding fault IDs are missing or duplicated".format(spec.name))
    if eqv_fault_nums.shape != (embeddings.shape[0],) or np.any(eqv_fault_nums <= 0):
        raise ValueError("{} equivalence counts are invalid".format(spec.name))
    if any(identifier.startswith("__smartatpg_bin_") for identifier in fault_ids):
        raise ValueError("{} contains helper-derived fault IDs".format(spec.name))

    metadata_ids = tuple(str(item["fault_id"]) for item in metadata.get("faults", []))
    metadata_eqv = np.asarray(
        [int(item["eqv_fault_num"]) for item in metadata.get("faults", [])],
        dtype=np.int64,
    )
    catalog_faults = list(catalog.get("faults", []))
    catalog_ids = tuple(str(item["fault_id"]) for item in catalog_faults)
    if fault_ids != metadata_ids or fault_ids != catalog_ids:
        raise ValueError("{} fault IDs do not match metadata and PODEM catalog".format(spec.name))
    catalog_eqv = np.asarray(
        [int(item["eqv_fault_num"]) for item in catalog_faults], dtype=np.int64
    )
    if not np.array_equal(eqv_fault_nums, catalog_eqv):
        raise ValueError("{} equivalence counts do not match PODEM".format(spec.name))
    if not np.array_equal(metadata_eqv, catalog_eqv):
        raise ValueError("{} metadata equivalence counts do not match PODEM".format(spec.name))
    if int(catalog.get("uncollapsed_total", -1)) != int(eqv_fault_nums.sum()):
        raise ValueError("{} uncollapsed fault total does not match".format(spec.name))
    if metadata.get("status") != "embeddings_exported":
        raise ValueError("{} metadata is not an embedding export".format(spec.name))
    if metadata.get("graph_sha256") != graph_sha256:
        raise ValueError("{} graph provenance does not match".format(spec.name))

    if metadata.get("schema") == "original_fault_anchor_embeddings_v1":
        if spec.faultmap_path is not None:
            raise ValueError("{} original-BENCH data must not use a fault map".format(spec.name))
        if spec.aig_bench_path is None or spec.aigmap_path is None:
            raise ValueError("{} anchor data requires aig_bench and aigmap".format(spec.name))
        expected_hashes = {
            "atpg_bench_sha256": _sha256(spec.bench_path),
            "aig_bench_sha256": _sha256(spec.aig_bench_path),
            "aigmap_sha256": _sha256(spec.aigmap_path),
        }
        for key, value in expected_hashes.items():
            if metadata.get(key) != value or embedding_hashes.get(key) != value:
                raise ValueError("{} {} provenance does not match".format(spec.name, key))
        if metadata.get("fault_count") != len(fault_ids):
            raise ValueError("{} metadata fault count does not match".format(spec.name))
        if metadata.get("uncollapsed_total") != int(eqv_fault_nums.sum()):
            raise ValueError("{} metadata uncollapsed total does not match".format(spec.name))
        if build_graph(read_bench(spec.aig_bench_path))["graph_sha256"] != graph_sha256:
            raise ValueError("{} graph does not match current AIG BENCH".format(spec.name))
    else:
        if spec.faultmap_path is None:
            raise ValueError("{} legacy data requires a fault map".format(spec.name))
        if metadata.get("bench_sha256") != _sha256(spec.bench_path):
            raise ValueError("{} BENCH provenance does not match".format(spec.name))
        if metadata.get("faultmap_sha256") != _sha256(spec.faultmap_path):
            raise ValueError("{} fault-map provenance does not match".format(spec.name))
        if build_graph(read_bench(spec.bench_path))["graph_sha256"] != graph_sha256:
            raise ValueError("{} graph does not match current BENCH".format(spec.name))
    if provenance != metadata.get("provenance") or (
        provenance.get("checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256
        or provenance.get("source_sha256") != OFFICIAL_SOURCE_SHA256
        or provenance.get("backend") != "official-python-deepgate"
        or provenance.get("is_known_official_checkpoint") is not True
    ):
        raise ValueError("{} pretrained embedding provenance does not match".format(spec.name))

    artifact_paths = [spec.bench_path, spec.embeddings_path, spec.metadata_path]
    artifact_paths.extend(path for path in (
        spec.faultmap_path, spec.aig_bench_path, spec.aigmap_path
    ) if path is not None)
    artifact_digest = hashlib.sha256(
        (spec.name + "".join(_sha256(path) for path in artifact_paths)).encode("utf-8")
    ).hexdigest()
    tensor = torch.from_numpy(np.array(embeddings, copy=True))
    return CircuitData(spec, tensor, fault_ids, eqv_fault_nums, artifact_digest)


def load_all_circuits(manifest, environment):
    # C++ input historically terminates on missing files. Preflight the entire
    # dataset before entering the binding, even for catalog-only validation.
    for spec in manifest.circuits:
        for path in (spec.bench_path, spec.embeddings_path, spec.metadata_path,
                     spec.faultmap_path, spec.aig_bench_path, spec.aigmap_path):
            if path is None:
                continue
            if not path.is_file():
                raise ValueError("missing circuit artifact: {}".format(path))
    circuits = []
    for spec in manifest.circuits:
        started = time.perf_counter()
        progress("LOAD", "catalog start", circuit=spec.name)
        try:
            catalog = environment.catalog(spec.bench_path, spec.faultmap_path)
        except Exception as exc:
            progress(
                "LOAD", "catalog error", circuit=spec.name,
                error_type=type(exc).__name__, error=str(exc),
                elapsed_s="{:.3f}".format(time.perf_counter() - started),
            )
            raise
        progress(
            "LOAD", "catalog done", circuit=spec.name,
            elapsed_s="{:.3f}".format(time.perf_counter() - started),
        )

        started = time.perf_counter()
        progress("LOAD", "validation start", circuit=spec.name)
        try:
            circuit = load_circuit_data(spec, catalog)
        except Exception as exc:
            progress(
                "LOAD", "validation error", circuit=spec.name,
                error_type=type(exc).__name__, error=str(exc),
                elapsed_s="{:.3f}".format(time.perf_counter() - started),
            )
            raise
        progress(
            "LOAD", "validation done", circuit=spec.name,
            elapsed_s="{:.3f}".format(time.perf_counter() - started),
        )
        circuits.append(circuit)
    return circuits
