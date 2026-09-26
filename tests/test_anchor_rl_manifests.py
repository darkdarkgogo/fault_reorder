import json
from pathlib import Path

from fault_order_rl.trainer import DEFAULT_VALIDATION_MANIFEST
from scripts import generate_anchor_rl_manifests as manifests


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["b12_C", "b15_C", "b20_C", "b21_C", "b22_C"]


def test_default_validation_manifest_has_five_circuits_without_b17():
    assert DEFAULT_VALIDATION_MANIFEST.name == "anchor_validation_5.json"
    payload = json.loads(
        DEFAULT_VALIDATION_MANIFEST.read_text(encoding="utf-8")
    )
    assert [entry["name"] for entry in payload["circuits"]] == EXPECTED
    assert sorted(
        path.stem
        for path in (ROOT / "configs" / "anchor_validation_single").glob("*.json")
    ) == EXPECTED


def test_validation_generator_excludes_b17_and_removes_stale_single(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(manifests, "ROOT", tmp_path)
    source_dir = tmp_path / "datasets" / "validation"
    aig_dir = tmp_path / "datasets" / "validation_AIG"
    config_dir = tmp_path / "configs"
    source_dir.mkdir(parents=True)
    aig_dir.mkdir(parents=True)
    config_dir.mkdir()
    for name in ["b12_C", "b15_C", "b17_C", "b20_C", "b21_C", "b22_C"]:
        (source_dir / f"{name}.bench").write_text("", encoding="utf-8")
        for suffix in (
            ".bench",
            ".aigmap.json",
            ".fault_embeddings.npz",
            ".faults.json",
        ):
            (aig_dir / f"{name}{suffix}").write_text("", encoding="utf-8")
    output = config_dir / "anchor_validation_5.json"
    manifests.generate_manifest(
        "validation", 5, output, excluded_stems={"b17_C"}
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    single_dir = config_dir / "anchor_validation_single"
    single_dir.mkdir()
    (single_dir / "b17_C.json").write_text("stale", encoding="utf-8")
    manifests.write_single_manifests(payload, single_dir)
    assert [entry["name"] for entry in payload["circuits"]] == EXPECTED
    assert sorted(path.stem for path in single_dir.glob("*.json")) == EXPECTED
