"""Compact generated fault metadata to the fields required for offline training."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fault_order_rl.data import load_manifest


REQUIRED_FIELDS = (
    "schema",
    "bench_sha256",
    "graph_sha256",
    "faultmap_sha256",
    "fault_count",
    "uncollapsed_total",
    "provenance",
    "status",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    for spec in manifest.circuits:
        raw = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        compact = {name: raw[name] for name in REQUIRED_FIELDS}
        with np.load(spec.embeddings_path, allow_pickle=False) as arrays:
            packed = {name: np.array(arrays[name], copy=True) for name in arrays.files}
        provenance = json.loads(str(packed["provenance_json"].item()))
        # Machine-local source paths are not evidence and are not portable.
        # Keep the immutable source/checkpoint hashes while using ASCII labels
        # that compare identically on Windows and Linux.
        provenance["checkpoint"] = "official-python-deepgate/pretrained/model.pth"
        provenance["source_path"] = "official-python-deepgate"
        packed["provenance_json"] = np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        )
        with tempfile.NamedTemporaryFile(
            "wb", dir=spec.embeddings_path.parent,
            prefix="." + spec.embeddings_path.name + ".", delete=False
        ) as stream:
            np.savez_compressed(stream, **packed)
            temporary_npz = Path(stream.name)
        os.replace(temporary_npz, spec.embeddings_path)
        compact["provenance"] = provenance
        compact["faults"] = [
            {"fault_id": str(item["fault_id"])} for item in raw["faults"]
        ]
        content = json.dumps(
            compact, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=spec.metadata_path.parent,
            prefix="." + spec.metadata_path.name + ".", delete=False
        ) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        os.replace(temporary, spec.metadata_path)
        print("Compacted", spec.metadata_path)


if __name__ == "__main__":
    main()
