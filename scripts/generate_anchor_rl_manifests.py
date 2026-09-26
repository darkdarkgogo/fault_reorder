"""Generate full RL manifests for the anchor-preserving dataset."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_EXCLUDED_STEMS = frozenset({"b17_C"})
SPLITS = {
    "train": (
        1024,
        ROOT / "configs" / "anchor_train_1024.json",
        frozenset(),
    ),
    "validation": (
        5,
        ROOT / "configs" / "anchor_validation_5.json",
        VALIDATION_EXCLUDED_STEMS,
    ),
}


def circuit_entry(split, source):
    stem = source.stem
    prefix = "../datasets/{}/{}".format(split, stem)
    aig_prefix = "../datasets/{}_AIG/{}".format(split, stem)
    return {
        "name": stem,
        "bench": prefix + ".bench",
        "aig_bench": aig_prefix + ".bench",
        "aigmap": aig_prefix + ".aigmap.json",
        "embeddings": aig_prefix + ".fault_embeddings.npz",
        "metadata": aig_prefix + ".faults.json",
    }


def generate_manifest(split, expected_count, output, excluded_stems=()):
    source_dir = ROOT / "datasets" / split
    sources = sorted(
        (
            source
            for source in source_dir.glob("*.bench")
            if source.stem not in excluded_stems
        ),
        key=lambda path: path.name,
    )
    if len(sources) != expected_count:
        raise ValueError(
            "Expected {} {} circuits, found {}".format(
                expected_count, split, len(sources)
            )
        )
    entries = [circuit_entry(split, source) for source in sources]
    for entry in entries:
        for key in ("bench", "aig_bench", "aigmap", "embeddings", "metadata"):
            if not (output.parent / entry[key]).resolve().is_file():
                raise ValueError("Missing {} artifact: {}".format(key, entry[key]))
    content = {
        "version": 1,
        "purpose": "anchor-preserving {} fault-order RL".format(split),
        "cpp_podem_dir": "../PODEM",
        "circuits": entries,
    }
    output.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"split": split, "circuits": len(entries), "output": str(output)}


def write_single_manifests(validation, single_dir):
    single_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        entry["name"] + ".json" for entry in validation["circuits"]
    }
    for existing in single_dir.glob("*.json"):
        if existing.name not in expected_names:
            existing.unlink()
    for entry in validation["circuits"]:
        output = single_dir / (entry["name"] + ".json")
        # Paths in the combined manifest are relative to configs/. Single
        # manifests live one level deeper.
        adjusted = dict(entry)
        for key in ("bench", "aig_bench", "aigmap", "embeddings", "metadata"):
            adjusted[key] = "../" + adjusted[key]
        output.write_text(
            json.dumps({
                "version": 1,
                "purpose": "single-circuit anchor validation",
                "cpp_podem_dir": "../../PODEM",
                "circuits": [adjusted],
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main():
    results = [
        generate_manifest(split, count, output, excluded_stems)
        for split, (count, output, excluded_stems) in SPLITS.items()
    ]
    validation_manifest = SPLITS["validation"][1]
    validation = json.loads(
        validation_manifest.read_text(encoding="utf-8")
    )
    single_dir = ROOT / "configs" / "anchor_validation_single"
    write_single_manifests(validation, single_dir)
    results.append({
        "split": "validation-single",
        "circuits": len(validation["circuits"]),
        "output": str(single_dir),
    })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
