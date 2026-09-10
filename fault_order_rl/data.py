"""Manifest loading and strict embedding/catalog validation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from fault_embedding.circuit import build_graph, read_bench
from fault_embedding.inference import OFFICIAL_CHECKPOINT_SHA256, OFFICIAL_SOURCE_SHA256


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
        specs.append(
            CircuitSpec(
                name=name,
                bench_path=(base / entry["bench"]).resolve(),
                faultmap_path=(base / entry["faultmap"]).resolve(),
                embeddings_path=(base / entry["embeddings"]).resolve(),
                metadata_path=(base / entry["metadata"]).resolve(),
            )
        )
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return TrainingManifest(path, module_dir, specs, hashlib.sha256(canonical).hexdigest())


def load_circuit_data(spec, catalog):
    for path in (
        spec.bench_path,
        spec.faultmap_path,
        spec.embeddings_path,
        spec.metadata_path,
    ):
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
    catalog_faults = list(catalog.get("faults", []))
    catalog_ids = tuple(str(item["fault_id"]) for item in catalog_faults)
    if fault_ids != metadata_ids or fault_ids != catalog_ids:
        raise ValueError("{} fault IDs do not match metadata and PODEM catalog".format(spec.name))
    catalog_eqv = np.asarray(
        [int(item["eqv_fault_num"]) for item in catalog_faults], dtype=np.int64
    )
    if not np.array_equal(eqv_fault_nums, catalog_eqv):
        raise ValueError("{} equivalence counts do not match PODEM".format(spec.name))
    if int(catalog.get("uncollapsed_total", -1)) != int(eqv_fault_nums.sum()):
        raise ValueError("{} uncollapsed fault total does not match".format(spec.name))
    if metadata.get("status") != "embeddings_exported":
        raise ValueError("{} metadata is not an embedding export".format(spec.name))
    if metadata.get("graph_sha256") != graph_sha256:
        raise ValueError("{} graph provenance does not match".format(spec.name))

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

    artifact_digest = hashlib.sha256(
        (spec.name + _sha256(spec.bench_path) + _sha256(spec.faultmap_path)
         + _sha256(spec.embeddings_path) + _sha256(spec.metadata_path)).encode(
            "utf-8"
        )
    ).hexdigest()
    tensor = torch.from_numpy(np.array(embeddings, copy=True))
    return CircuitData(spec, tensor, fault_ids, eqv_fault_nums, artifact_digest)


def load_all_circuits(manifest, environment):
    # C++ input historically terminates on missing files. Preflight the entire
    # dataset before entering the binding, even for catalog-only validation.
    for spec in manifest.circuits:
        for path in (spec.bench_path, spec.faultmap_path, spec.embeddings_path, spec.metadata_path):
            if not path.is_file():
                raise ValueError("missing circuit artifact: {}".format(path))
    circuits = []
    for spec in manifest.circuits:
        catalog = environment.catalog(spec.bench_path, spec.faultmap_path)
        circuits.append(load_circuit_data(spec, catalog))
    return circuits
