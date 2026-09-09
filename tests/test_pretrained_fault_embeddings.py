"""Optional real-model integration checks; never substitute random weights."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest

from fault_embedding.circuit import build_graph, read_bench
from fault_embedding.faults import read_faultmap
from fault_embedding.features import compose_fault_embeddings
from fault_embedding.inference import DEFAULT_CHECKPOINT, infer_gate_embeddings


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Official checkpoint has not been downloaded")
def test_inference_rejects_preloaded_module_without_source(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    fake = types.ModuleType("deepgate.model")
    monkeypatch.setitem(sys.modules, "deepgate.model", fake)
    samples = Path(__file__).resolve().parents[1] / "PODEM" / "sample_circuits"
    graph = build_graph(read_bench(samples / "c432_binary.bench"))
    with pytest.raises(ValueError, match="no verifiable source file"):
        infer_gate_embeddings(graph)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Official checkpoint has not been downloaded")
def test_official_c432_embeddings_are_finite_repeatable_and_aligned():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    samples = Path(__file__).resolve().parents[1] / "PODEM" / "sample_circuits"
    circuit = read_bench(samples / "c432_binary.bench")
    graph = build_graph(circuit)
    faults = read_faultmap(samples / "c432_binary.faultmap", circuit, graph)["faults"]
    first, provenance = infer_gate_embeddings(graph, seed=0)
    second, _ = infer_gate_embeddings(graph, seed=0)
    for key in ("hs", "hf"):
        assert first[key].shape == (len(graph["nodes"]), 128)
        np.testing.assert_array_equal(first[key], second[key])
    features = compose_fault_embeddings(first["hf"], faults)
    assert features.shape == (533, 257)
    assert np.isfinite(features).all()
    assert provenance["is_known_official_checkpoint"]


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Official checkpoint has not been downloaded")
def test_official_hf_fault_feature_is_invariant_to_commutative_input_swap(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    paths = [tmp_path / "ab.bench", tmp_path / "ba.bench"]
    paths[0].write_text("INPUT(a)\nINPUT(b)\ny=AND(a,b)\nOUTPUT(y)\n", encoding="utf-8")
    paths[1].write_text("INPUT(a)\nINPUT(b)\ny=AND(b,a)\nOUTPUT(y)\n", encoding="utf-8")
    features = []
    for path in paths:
        graph = build_graph(read_bench(path))
        arrays, _ = infer_gate_embeddings(graph, seed=0)
        fault = {"fault_id": "y:a:sa1", "io": "GI", "sa_value": 1,
                 "gate_function_anchor_node": graph["anchors"]["y"],
                 "connected_function_anchor_node": graph["anchors"]["a"]}
        features.append(compose_fault_embeddings(arrays["hf"], [fault]))
    np.testing.assert_array_equal(features[0], features[1])
