import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fault_embedding.circuit import build_graph, read_bench
from fault_embedding.faults import read_faultmap
from fault_embedding.features import compose_fault_embeddings


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "PODEM" / "sample_circuits"


def circuit_from_text(tmp_path, text):
    path = tmp_path / "test.bench"
    path.write_text(text, encoding="utf-8")
    circuit = read_bench(path)
    return circuit, build_graph(circuit)


def mapped_faults(tmp_path, circuit, graph, records):
    path = tmp_path / "test.faultmap"
    total = sum(int(line.split()[-1]) for line in records)
    path.write_text("SMARTATPG_FAULT_MAP_V2\nsource_hash 0000000000000000\n"
                    "circuit_hash 0000000000000000\n"
                    f"count {len(records)}\nuncollapsed_total {total}\n"
                    + "\n".join(records) + "\nend\n", encoding="utf-8")
    sidecar = Path(str(path) + ".binding.json")
    import hashlib
    sidecar.write_text(json.dumps({"bench_sha256": circuit.sha256,
                                   "faultmap_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "source_hash": "0000000000000000",
                                   "circuit_hash": "0000000000000000"}), encoding="utf-8")
    return read_faultmap(path, circuit, graph)


def evaluate_graph(graph, inputs):
    values = []
    for node in graph["nodes"]:
        if node["kind"] == "INPUT":
            result = inputs[node["origin"]]
        elif node["kind"] == "NOT":
            result = 1 - values[node["inputs"][0]]
        else:
            result = values[node["inputs"][0]] & values[node["inputs"][1]]
        values.append(result)
    return values


@pytest.mark.parametrize("kind", ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "NOT", "BUF"])
def test_all_gate_truth_tables_and_unique_anchors(tmp_path, kind):
    pins = "a" if kind in ("NOT", "BUF") else "a,b"
    circuit, graph = circuit_from_text(tmp_path, f"INPUT(a)\nINPUT(b)\nOUTPUT(y)\ny = {kind}({pins})\n")
    assert len(set(graph["anchors"].values())) == len(graph["anchors"])
    for a, b in itertools.product((0, 1), repeat=2):
        expected = {"AND": a & b, "NAND": 1 - (a & b), "OR": a | b,
                    "NOR": 1 - (a | b), "XOR": a ^ b, "XNOR": 1 - (a ^ b),
                    "NOT": 1 - a, "BUF": a}[kind]
        values = evaluate_graph(graph, {"a": a, "b": b})
        assert values[graph["anchors"]["y"]] == expected
        po = graph["boundary_aliases"]["dummy_gate3"]
        assert values[po["anchor"]] == expected
    assert graph == build_graph(circuit)


def test_fanout_uses_receiver_and_polarity_not_source(tmp_path):
    circuit, graph = circuit_from_text(tmp_path, "INPUT(a)\nINPUT(b)\nINPUT(c)\nINPUT(d)\n"
                                     "n=AND(a,b)\nx=AND(n,c)\ny=AND(d,n)\nOUTPUT(x)\nOUTPUT(y)\n")
    records = ["fault n:GO:sa0 n - 1 -1 0 1", "fault n:GO:sa1 n - 1 -1 1 1",
               "fault x:GI0:sa1 x n 0 0 1 1", "fault y:GI1:sa1 y n 0 0 1 1"]
    metadata = mapped_faults(tmp_path, circuit, graph, records)
    faults = metadata["faults"]
    gates = np.arange(len(graph["nodes"]) * 4, dtype=np.float32).reshape(-1, 4)
    features = compose_fault_embeddings(gates, faults)
    assert features.shape == (4, 9)
    np.testing.assert_array_equal(features[0, :4], gates[graph["anchors"]["n"]])
    np.testing.assert_array_equal(features[0, 4:8], gates[graph["anchors"]["n"]])
    np.testing.assert_array_equal(features[0, :-1], features[1, :-1])
    assert features[0, -1] == 0 and features[1, -1] == 1
    np.testing.assert_array_equal(features[2, :4], gates[graph["anchors"]["x"]])
    np.testing.assert_array_equal(features[3, :4], gates[graph["anchors"]["y"]])
    np.testing.assert_array_equal(features[2, 4:8], gates[graph["anchors"]["n"]])
    np.testing.assert_array_equal(features[3, 4:8], gates[graph["anchors"]["n"]])
    # x's n input becomes pin 1 after PODEM sorts c(level 0), n(level 1).
    assert faults[2]["bench_pin_index"] == 0
    assert faults[2]["pin_index"] == 1
    assert features[2, -1] == 1


def test_same_receiver_uses_faulted_input_function_instead_of_pin_number(tmp_path):
    circuit, graph = circuit_from_text(tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n")
    records = ["fault y:GO:sa1 y - 1 -1 1 1", "fault y:GI0:sa1 y a 0 0 1 1",
               "fault y:GI1:sa1 y b 0 0 1 1"]
    faults = mapped_faults(tmp_path, circuit, graph, records)["faults"]
    functions = np.arange(len(graph["nodes"]) * 3, dtype=np.float32).reshape(-1, 3)
    features = compose_fault_embeddings(functions, faults)
    y, a, b = (graph["anchors"][name] for name in ("y", "a", "b"))
    np.testing.assert_array_equal(features[0, :6], np.concatenate([functions[y], functions[y]]))
    np.testing.assert_array_equal(features[1, :6], np.concatenate([functions[y], functions[a]]))
    np.testing.assert_array_equal(features[2, :6], np.concatenate([functions[y], functions[b]]))
    assert len(set(f["anchor_node"] for f in faults)) == 1


def test_input_swap_does_not_change_fault_embedding(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_circuit, first_graph = circuit_from_text(
        first_dir, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n")
    second_circuit, second_graph = circuit_from_text(
        second_dir, "INPUT(a)\nINPUT(b)\ny=AND(b,a)\nOUTPUT(y)\n")
    first_fault = mapped_faults(
        first_dir, first_circuit, first_graph, ["fault y:GI0:sa1 y a 0 0 1 1"])["faults"]
    second_fault = mapped_faults(
        second_dir, second_circuit, second_graph, ["fault y:GI1:sa1 y a 0 1 1 1"])["faults"]
    # Assign equal functional vectors by signal name in both graph layouts.
    values = {"a": [1, 2], "b": [3, 4], "y": [5, 6]}
    def named_functions(graph):
        result = np.zeros((len(graph["nodes"]), 2), dtype=np.float32)
        for name, vector in values.items():
            result[graph["anchors"][name]] = vector
        return result
    first_feature = compose_fault_embeddings(named_functions(first_graph), first_fault)
    second_feature = compose_fault_embeddings(named_functions(second_graph), second_fault)
    np.testing.assert_array_equal(first_feature, second_feature)
    assert first_fault[0]["pin_index"] == 0 and second_fault[0]["pin_index"] == 1


def test_boundary_faults_have_explicit_anchors(tmp_path):
    circuit, graph = circuit_from_text(tmp_path, "INPUT(a)\nOUTPUT(a)\n")
    records = ["fault dummy_gate1:GO:sa0 dummy_gate1 - 1 -1 0 1",
               "fault dummy_gate2:GI0:sa1 dummy_gate2 a 0 0 1 1"]
    faults = mapped_faults(tmp_path, circuit, graph, records)["faults"]
    assert faults[0]["anchor_node"] == graph["anchors"]["a"]
    assert faults[1]["anchor_node"] != graph["anchors"]["a"]
    assert faults[1]["connected_function_anchor_node"] == graph["anchors"]["a"]


@pytest.mark.parametrize("text, reason", [
    ("INPUT(a)\ny=AND(a,a,a)\nOUTPUT(y)", "requires 2 inputs"),
    ("INPUT(a)\ny=AND(a,b)\nOUTPUT(y)", "Undefined"),
    ("INPUT(a)\ny=NOT(z)\nz=NOT(y)\nOUTPUT(y)", "cycle"),
    ("INPUT(a)\nINPUT(a)\nOUTPUT(a)", "duplicate"),
    ("INPUT(a)\na=NOT(a)\nOUTPUT(a)", "both"),
])
def test_bad_bench_fails(tmp_path, text, reason):
    with pytest.raises(ValueError, match=reason):
        circuit_from_text(tmp_path, text)


def test_repeated_source_preserves_duplicate_rows(tmp_path):
    circuit, graph = circuit_from_text(tmp_path, "INPUT(a)\ny=AND(a,a)\nOUTPUT(y)\n")
    faults = mapped_faults(tmp_path, circuit, graph,
                          ["fault y:GI1:sa1 y a 0 1 1 1",
                           "fault y:GI0:sa1@dup1 y a 0 1 1 1"])["faults"]
    assert [fault["pin_index"] for fault in faults] == [1, 1]
    functions = np.arange(len(graph["nodes"]) * 2, dtype=np.float32).reshape(-1, 2)
    features = compose_fault_embeddings(functions, faults)
    np.testing.assert_array_equal(features[0], features[1])
    assert [fault["mapped_location_duplicate_index"] for fault in faults] == [0, 1]


@pytest.mark.parametrize("path", sorted(SAMPLES.glob("*_binary.faultmap")), ids=lambda p: p.stem)
def test_all_bundled_fault_maps_resolve(path):
    circuit = read_bench(path.with_suffix(".bench"))
    graph = build_graph(circuit)
    metadata = read_faultmap(path, circuit, graph)
    assert all(f["position_index"] in (0, 1, 2) for f in metadata["faults"])


def test_faultmap_rejects_wrong_bench_pairing(tmp_path):
    circuit, graph = circuit_from_text(tmp_path, "INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n")
    path = tmp_path / "unknown.faultmap"
    path.write_text("SMARTATPG_FAULT_MAP_V2\nsource_hash 0000000000000000\n"
                    "circuit_hash 0000000000000000\ncount 1\nuncollapsed_total 1\n"
                    "fault y:GO:sa0 y - 1 -1 0 1\nend\n", encoding="utf-8")
    wrong_sha = "0" * 64
    import hashlib
    Path(str(path) + ".binding.json").write_text(
        json.dumps({"bench_sha256": wrong_sha,
                    "faultmap_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_hash": "0000000000000000",
                    "circuit_hash": "0000000000000000"}), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch for bench_sha256"):
        read_faultmap(path, circuit, graph)


def test_invalid_features_fail():
    with pytest.raises(ValueError, match="non-finite"):
        compose_fault_embeddings(np.array([[np.nan]], np.float32), [])
    assert compose_fault_embeddings(np.ones((2, 4), np.float32), []).shape == (0, 9)
