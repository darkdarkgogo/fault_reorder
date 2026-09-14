import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from fault_embedding.anchor_aig import (
    build_aigmap,
    compose_original_fault_embeddings,
    lower_to_anchor_aig,
    resolve_original_faults,
    validate_aigmap,
    verify_anchor_equivalence,
    write_json,
)
from fault_embedding.circuit import build_graph, read_bench
from scripts.generate_anchor_aig import (
    ensure_safe_dataset_target,
    original_catalog,
    replace_directories,
)
import scripts.generate_anchor_aig as generator


def write_source(tmp_path, body, name="tiny.bench"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def lower_file(tmp_path, body):
    source = write_source(tmp_path, body)
    lowered = lower_to_anchor_aig(source)
    output = tmp_path / "lowered.bench"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(lowered.text)
    aig = read_bench(output)
    verify_anchor_equivalence(lowered.source, aig)
    return source, output, lowered, aig, build_graph(aig)


@pytest.mark.parametrize("kind", ["AND", "NAND", "OR", "NOR", "NOT", "BUF"])
def test_lowering_preserves_every_original_signal(kind, tmp_path):
    pins = "a" if kind in ("NOT", "BUF") else "a,b"
    source, output, lowered, aig, graph = lower_file(
        tmp_path,
        "INPUT(a)\nINPUT(b)\nOUTPUT(y)\ny={}({})\n".format(kind, pins),
    )
    assert set(gate.kind for gate in aig.gates.values()) <= {"AND", "NOT"}
    assert graph["anchors"]["a"] != graph["anchors"]["y"]
    assert output.read_text(encoding="utf-8") == lower_to_anchor_aig(source).text
    mapping = build_aigmap(source, output, lowered, graph)
    write_json(tmp_path / "tiny.aigmap.json", mapping)
    assert validate_aigmap(mapping, source, output, graph).outputs == ["y"]


def test_lowering_preserves_intermediate_original_anchor(tmp_path):
    source, _, lowered, _, graph = lower_file(
        tmp_path,
        "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(z)\n"
        "n1=NAND(a,b)\nz=OR(n1,c)\n",
    )
    assert set(graph["anchors"]) >= {"a", "b", "c", "n1", "z"}
    assert len({graph["anchors"][name] for name in ("a", "b", "c", "n1", "z")}) == 5
    assert lowered.helper_origins
    assert not (set(lowered.helper_origins) & {"a", "b", "c", "n1", "z"})
    assert source.is_file()


def test_original_faults_map_to_anchor_rows_in_catalog_order(tmp_path):
    _, _, lowered, _, graph = lower_file(
        tmp_path,
        "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(x)\nOUTPUT(y)\n"
        "n=AND(a,b)\nx=AND(n,c)\ny=OR(n,c)\n",
    )
    catalog = {
        "uncollapsed_total": 7,
        "faults": [
            {"fault_id": "n:GO:sa0", "node_name": "n", "input_wire_name": "-",
             "io": 1, "input_index": -1, "input_occurrence": -1,
             "fault_type": 0, "eqv_fault_num": 3},
            {"fault_id": "x:GI1:sa1", "node_name": "x", "input_wire_name": "n",
             "io": 0, "input_index": 1, "input_occurrence": 0,
             "fault_type": 1, "eqv_fault_num": 1},
            {"fault_id": "dummy_gate1:GO:sa1", "node_name": "dummy_gate1",
             "input_wire_name": "-", "io": 1, "input_index": -1,
             "input_occurrence": -1, "fault_type": 1, "eqv_fault_num": 1},
            {"fault_id": "dummy_gate4:GI0:sa0", "node_name": "dummy_gate4",
             "input_wire_name": "x", "io": 0, "input_index": 0,
             "input_occurrence": 0, "fault_type": 0, "eqv_fault_num": 2},
        ],
    }
    faults = resolve_original_faults(catalog, lowered.source, graph)
    assert [fault["catalog_row"] for fault in faults] == list(range(4))
    assert faults[0]["gate_function_anchor_node"] == graph["anchors"]["n"]
    assert faults[1]["gate_function_anchor_node"] == graph["anchors"]["x"]
    assert faults[1]["connected_function_anchor_node"] == graph["anchors"]["n"]
    assert faults[2]["gate_function_anchor_node"] == graph["anchors"]["a"]
    assert faults[3]["gate_function_anchor_node"] != graph["anchors"]["x"]
    assert faults[3]["connected_function_anchor_node"] == graph["anchors"]["x"]
    functions = np.arange(len(graph["nodes"]) * 128, dtype=np.float32).reshape(-1, 128)
    features = compose_original_fault_embeddings(functions, faults)
    assert features.shape == (4, 257)
    assert features[0, -1] == 0 and features[1, -1] == 1


def test_mapping_rejects_any_missing_original_fault_anchor(tmp_path):
    _, _, lowered, _, graph = lower_file(
        tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n"
    )
    catalog = {
        "uncollapsed_total": 1,
        "faults": [{"fault_id": "missing:GO:sa0", "node_name": "missing",
                    "input_wire_name": "-", "io": 1, "input_index": -1,
                    "input_occurrence": -1, "fault_type": 0, "eqv_fault_num": 1}],
    }
    with pytest.raises(ValueError, match="has no AIG anchor"):
        resolve_original_faults(catalog, lowered.source, graph)


def test_mapping_rejects_fault_id_or_receiver_pin_disagreement(tmp_path):
    _, _, lowered, _, graph = lower_file(
        tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n"
    )
    base = {"node_name": "y", "input_wire_name": "a", "io": 0,
            "input_index": 0, "input_occurrence": 0, "fault_type": 1,
            "eqv_fault_num": 1}
    bad_id = dict(base, fault_id="y:GI1:sa1")
    with pytest.raises(ValueError, match="identity fields disagree"):
        resolve_original_faults({"uncollapsed_total": 1, "faults": [bad_id]},
                                lowered.source, graph)
    bad_pin = dict(base, fault_id="y:GI0:sa1", input_wire_name="b")
    with pytest.raises(ValueError, match="actual receiver pin"):
        resolve_original_faults({"uncollapsed_total": 1, "faults": [bad_pin]},
                                lowered.source, graph)


def test_reserved_helper_prefix_is_rejected(tmp_path):
    source = write_source(
        tmp_path,
        "INPUT(a)\nINPUT(b)\n__anchor_aig_bad=AND(a,b)\nOUTPUT(__anchor_aig_bad)\n",
    )
    with pytest.raises(ValueError, match="reserved"):
        lower_to_anchor_aig(source)


def test_mapping_hash_tampering_is_rejected(tmp_path):
    source, output, lowered, _, graph = lower_file(
        tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n"
    )
    mapping = build_aigmap(source, output, lowered, graph)
    mapping["aig_bench_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aig_bench_sha256"):
        validate_aigmap(mapping, source, output, graph)


def test_mapping_audit_paths_do_not_bind_artifacts_to_generation_machine(tmp_path):
    source, output, lowered, _, graph = lower_file(
        tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n"
    )
    mapping = build_aigmap(source, output, lowered, graph)
    mapping["source_bench"] = r"E:\\old-machine\\dataset\\tiny.bench"
    mapping["aig_bench"] = r"E:\\old-machine\\dataset_AIG\\tiny.bench"
    assert validate_aigmap(mapping, source, output, graph).outputs == ["y"]


@pytest.mark.parametrize("field", ["source_bench", "aig_bench"])
def test_mapping_requires_nonempty_audit_paths(field, tmp_path):
    source, output, lowered, _, graph = lower_file(
        tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n"
    )
    mapping = build_aigmap(source, output, lowered, graph)
    mapping[field] = ""
    with pytest.raises(ValueError, match="invalid audit path"):
        validate_aigmap(mapping, source, output, graph)


def test_all_boolean_assignments_survive_multigate_lowering(tmp_path):
    source, output, lowered, aig, _ = lower_file(
        tmp_path,
        "INPUT(a)\nINPUT(b)\nINPUT(c)\n"
        "n1=NAND(a,b)\nn2=NOR(b,c)\ny=OR(n1,n2)\nOUTPUT(y)\n",
    )
    # verify_anchor_equivalence already checks every original anchor with packed
    # random vectors. Re-run one vector for every assignment to make this test
    # exhaustive without depending on the random sample.
    for values in itertools.product((0, 1), repeat=3):
        bits = dict(zip(("a", "b", "c"), values))
        from fault_embedding.anchor_aig import _evaluate
        original_values = _evaluate(lowered.source, bits, 1)
        aig_values = _evaluate(aig, bits, 1)
        assert all(original_values[name] == aig_values[name]
                   for name in lowered.source.inputs + lowered.source.order)
    assert source.is_file() and output.is_file()


def test_real_b12_original_catalog_maps_every_fault():
    root = Path(__file__).resolve().parents[1]
    source = root / "datasets" / "validation" / "b12_C.bench"
    if not source.is_file():
        pytest.skip("new validation dataset is not available")
    lowered = lower_to_anchor_aig(source)
    temporary = root / ".pytest_bytecode" / "b12-anchor-map-test.bench"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(lowered.text)
        graph = build_graph(read_bench(temporary))
        faults = resolve_original_faults(original_catalog(source), lowered.source, graph)
        assert len(faults) == 3444
        assert sum(fault["eqv_fault_num"] for fault in faults) == 5906
        assert all(fault["gate_function_anchor_node"] >= 0 for fault in faults)
        assert all(fault["connected_function_anchor_node"] >= 0 for fault in faults)
    finally:
        temporary.unlink(missing_ok=True)


def test_existing_anchor_artifacts_validate_after_directory_relocation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "datasets" / "train" / "train_0001_iscas89_s13207_g7920.bench"
    artifacts = root / "datasets" / "train_AIG"
    if not source.is_file() or not artifacts.is_dir():
        pytest.skip("anchor training dataset is not available")
    relocated_source_dir = tmp_path / "linux-clone" / "datasets" / "train"
    relocated_aig_dir = tmp_path / "linux-clone" / "datasets" / "train_AIG"
    relocated_source_dir.mkdir(parents=True)
    relocated_aig_dir.mkdir(parents=True)
    relocated_source = relocated_source_dir / source.name
    shutil.copy2(source, relocated_source)
    for path in generator.artifact_paths(artifacts, source.stem).values():
        shutil.copy2(path, relocated_aig_dir / path.name)
    assert generator.validate_generated_circuit(
        relocated_source, relocated_aig_dir
    ) == 96


def test_directory_replacement_removes_old_after_staged_install(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    dataset = tmp_path / "datasets"
    old = dataset / "train_AIG"
    staged = dataset / ".stage" / "train_AIG"
    old.mkdir(parents=True)
    staged.mkdir(parents=True)
    (old / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    replace_directories(dataset, {"train_AIG": staged})
    assert not (dataset / "train_AIG" / "old.txt").exists()
    assert (dataset / "train_AIG" / "new.txt").read_text(encoding="utf-8") == "new"
    assert not list(dataset.glob(".train_AIG-old-*"))


@pytest.mark.parametrize("fail_on", [2, 4])
def test_directory_replacement_rolls_back_if_interrupted(tmp_path, monkeypatch, fail_on):
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    dataset = tmp_path / "datasets"
    for name in ("train_AIG", "validation_AIG"):
        directory = dataset / name
        directory.mkdir(parents=True)
        (directory / "old.txt").write_text(name, encoding="utf-8")
    train_staged = dataset / ".stage" / "train_AIG"
    train_staged.mkdir(parents=True)
    (train_staged / "new.txt").write_text("new", encoding="utf-8")
    validation_staged = dataset / ".stage" / "validation_AIG"
    validation_staged.mkdir(parents=True)
    (validation_staged / "new.txt").write_text("new", encoding="utf-8")
    real_replace = generator.os.replace
    calls = {"count": 0}

    def interrupt_once(source, target):
        calls["count"] += 1
        if calls["count"] == fail_on:
            raise KeyboardInterrupt()
        return real_replace(source, target)

    monkeypatch.setattr(generator.os, "replace", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        replace_directories(dataset, {
            "train_AIG": train_staged,
            "validation_AIG": validation_staged,
        })
    assert (dataset / "train_AIG" / "old.txt").read_text(encoding="utf-8") == "train_AIG"
    assert (dataset / "validation_AIG" / "old.txt").read_text(encoding="utf-8") == "validation_AIG"
    assert not list(dataset.glob(".*_AIG-old-*"))


def test_dataset_target_guard_rejects_broad_or_nested_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        ensure_safe_dataset_target(dataset, dataset, "datasets")
    with pytest.raises(ValueError, match="Unsafe"):
        ensure_safe_dataset_target(dataset, dataset / "nested" / "train_AIG", "train_AIG")
