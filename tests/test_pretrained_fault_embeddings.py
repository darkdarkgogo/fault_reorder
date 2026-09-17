"""Optional real-model integration checks; never substitute random weights."""

import numpy as np
import pytest

from fault_embedding.circuit import build_graph, read_bench
from fault_embedding.features import compose_fault_embeddings
from fault_embedding.inference import DEFAULT_CHECKPOINT, infer_gate_embeddings


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
