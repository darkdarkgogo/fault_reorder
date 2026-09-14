"""DeepTPI graph selection and BENCH conversion tests."""

from pathlib import Path

import numpy as np
import pytest

from fault_order_rl.deeptpi_import import graph_to_bench, load_graphs, validate_graph


def graph_data():
    x = np.zeros((5, 10), dtype=np.float32)
    x[:, 0] = np.arange(5)
    x[:, 1] = [0, 0, 1, 2, 3]
    edges = np.asarray([[0, 2], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    return {"x": x, "edge_index": edges}


def test_graph_to_bench_maps_deeptpi_gate_types_and_ports():
    bench = graph_to_bench("tiny", graph_data())
    assert "INPUT(N0)" in bench
    assert "INPUT(N1)" in bench
    assert "OUTPUT(N4)" in bench
    assert "N2 = AND(N0, N1)" in bench
    assert "N3 = NOT(N2)" in bench
    assert "N4 = BUFF(N3)" in bench


def test_graph_validation_rejects_wrong_arity_and_cycle():
    wrong_arity = graph_data()
    wrong_arity["edge_index"] = wrong_arity["edge_index"][:-1]
    with pytest.raises(ValueError, match="fanins"):
        validate_graph("wrong", wrong_arity)

    cycle = graph_data()
    cycle["x"][0, 1] = 3
    cycle["edge_index"] = np.concatenate(
        [cycle["edge_index"], np.asarray([[4, 0]], dtype=np.int64)])
    with pytest.raises(ValueError, match="cycle"):
        validate_graph("cycle", cycle)


def test_npz_selection_preserves_dictionary_order(tmp_path):
    circuits = {
        "first": graph_data(),
        "second": graph_data(),
        "third": graph_data(),
    }
    path = tmp_path / "benchmarks_circuits_graphs.npz"
    np.savez_compressed(path, circuits=circuits)
    selected = load_graphs(path, 2)
    assert [name for name, _ in selected] == ["first", "second"]
    with pytest.raises(ValueError, match="fewer than 4"):
        load_graphs(path, 4)


def test_checked_in_deeptpi_split_has_expected_boundaries():
    root = Path(__file__).resolve().parents[2]
    source = root / "DeepTPI-main" / "DeepTPI-main" / "data" / "ITC22_dataset"
    if not source.is_dir():
        pytest.skip("DeepTPI ITC22 data is not available")
    train = load_graphs(source / "train" / "benchmarks_circuits_graphs.npz", 512)
    test = load_graphs(source / "test" / "benchmarks_circuits_graphs.npz", 9)
    assert train[0][0] == "DMA_aig_000"
    assert train[-1][0] == "s35932_aig_121"
    assert [name for name, _ in test] == [
        "b12_C", "b15_C", "b17_C", "b20_C", "b21_C", "b22_C",
        "i2c_aig", "mem_ctrl_aig", "max_aig",
    ]
    assert not ({name for name, _ in train} & {name for name, _ in test})
