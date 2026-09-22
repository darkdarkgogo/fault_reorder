from pathlib import Path
from types import SimpleNamespace

import pytest

from fault_order_rl import data as data_module
from fault_order_rl.data import CircuitSpec
from fault_order_rl.trainer import Trainer


class _RecordingStream:
    def __init__(self):
        self.text = ""
        self.flushes = 0

    def write(self, value):
        self.text += value
        return len(value)

    def flush(self):
        self.flushes += 1


def _spec(tmp_path, name="tiny"):
    bench = tmp_path / (name + ".bench")
    embeddings = tmp_path / (name + ".npz")
    metadata = tmp_path / (name + ".json")
    for path in (bench, embeddings, metadata):
        path.write_text("x", encoding="utf-8")
    return CircuitSpec(name, bench, None, embeddings, metadata)


def test_progress_is_flushed_and_sampling_is_bounded(monkeypatch):
    from fault_order_rl import progress as progress_module

    stream = _RecordingStream()
    monkeypatch.setattr(progress_module.sys, "stderr", stream)
    progress_module.progress("LOAD", "catalog start", circuit="tiny")

    assert stream.text == "[LOAD] catalog start circuit=tiny\n"
    assert stream.flushes == 1
    assert [index for index in range(1, 202)
            if progress_module.sample_progress(index, 5, 100)] == [1, 2, 3, 4, 5, 100, 200]


def test_load_all_circuits_logs_catalog_then_validation(tmp_path, monkeypatch, capsys):
    spec = _spec(tmp_path)
    manifest = SimpleNamespace(circuits=[spec])
    catalog = {"faults": [], "uncollapsed_total": 0}

    class Environment:
        def catalog(self, bench_path, faultmap_path):
            assert bench_path == spec.bench_path
            return catalog

    monkeypatch.setattr(
        data_module, "load_circuit_data",
        lambda actual_spec, actual_catalog: (actual_spec.name, actual_catalog),
    )

    assert data_module.load_all_circuits(manifest, Environment()) == [("tiny", catalog)]
    lines = capsys.readouterr().err.splitlines()
    assert [line.split(" elapsed_s=")[0] for line in lines] == [
        "[LOAD] catalog start circuit=tiny",
        "[LOAD] catalog done circuit=tiny",
        "[LOAD] validation start circuit=tiny",
        "[LOAD] validation done circuit=tiny",
    ]


@pytest.mark.parametrize("failure", ["catalog", "validation"])
def test_load_all_circuits_logs_errors(tmp_path, monkeypatch, capsys, failure):
    spec = _spec(tmp_path)
    manifest = SimpleNamespace(circuits=[spec])

    class Environment:
        def catalog(self, bench_path, faultmap_path):
            if failure == "catalog":
                raise RuntimeError("catalog exploded")
            return {"faults": [], "uncollapsed_total": 0}

    if failure == "validation":
        def fail_validation(spec, catalog):
            raise ValueError("validation exploded")

        monkeypatch.setattr(data_module, "load_circuit_data", fail_validation)

    with pytest.raises((RuntimeError, ValueError), match="exploded"):
        data_module.load_all_circuits(manifest, Environment())

    error = capsys.readouterr().err
    assert "[LOAD] {} error circuit=tiny".format(failure) in error
    assert "error_type={}".format(
        "RuntimeError" if failure == "catalog" else "ValueError") in error


def test_native_session_logs_sampled_steps_and_finalize(capsys):
    class Session:
        def __init__(self):
            self.remaining_fault_ids = tuple("f{}".format(i) for i in range(101))

        def step(self, primary, secondaries):
            assert primary == self.remaining_fault_ids[0]
            self.remaining_fault_ids = self.remaining_fault_ids[1:]
            return {}

        def finish(self):
            return {"patterns_after_stc": 7}

    session = Session()

    class Environment:
        def start_session(self, bench_path, faultmap_path):
            return session

    trainer = Trainer.__new__(Trainer)
    trainer._check_result = lambda circuit, result: None
    circuit = SimpleNamespace(
        name="tiny",
        spec=SimpleNamespace(bench_path=Path("tiny.bench"), faultmap_path=None),
    )

    result, _ = trainer._run_native(circuit, Environment())

    assert result["patterns_after_stc"] == 7
    lines = capsys.readouterr().err.splitlines()
    assert any("[NATIVE] session start circuit=tiny mode=baseline" in line for line in lines)
    assert any("[NATIVE] session ready circuit=tiny mode=baseline selectable=101" in line for line in lines)
    starts = [line for line in lines if "[ATPG] step start" in line]
    dones = [line for line in lines if "[ATPG] step done" in line]
    assert len(starts) == 6
    assert len(dones) == 6
    assert any("step=100" in line for line in starts)
    assert any("[STC] finalize start circuit=tiny mode=baseline" in line for line in lines)
    assert any("[STC] finalize done circuit=tiny mode=baseline patterns=7" in line for line in lines)
