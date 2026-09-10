"""Export pinned pretrained embeddings for a training manifest, without training."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fault_order_rl.data import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "all_benchmarks.json")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    for spec in manifest.circuits:
        if spec.embeddings_path.is_file() and spec.metadata_path.is_file():
            print("Existing export: " + spec.name, flush=True)
            continue
        expected = spec.embeddings_path.parent / (spec.bench_path.stem + ".fault_embeddings.npz")
        metadata = spec.embeddings_path.parent / (spec.bench_path.stem + ".faults.json")
        if spec.embeddings_path != expected or spec.metadata_path != metadata:
            raise ValueError("manifest export names must match the BENCH stem")
        print("Exporting pretrained embeddings: " + spec.name, flush=True)
        subprocess.run([sys.executable, "-m", "fault_embedding", "export", "--bench", str(spec.bench_path),
                        "--faultmap", str(spec.faultmap_path), "--out-dir", str(spec.embeddings_path.parent)],
                       cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, "-m", "fault_order_rl", "validate", "--manifest", str(manifest.path)],
                   cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
