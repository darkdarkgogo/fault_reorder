import os

import pytest


class _FakeSession:
    def __init__(self, remaining, increments=None):
        self._remaining = list(remaining)
        self._increments = iter(increments or [(2, 3)] * len(remaining))
        self.selected = []
        self.dtc_calls = 0
        self.dtc_backtracks = 0

    def remaining_fault_ids(self):
        return list(self._remaining)

    def step(self, primary):
        assert primary == self._remaining[0]
        self.selected.append(primary)
        self._remaining.pop(0)
        calls, backtracks = next(self._increments)
        self.dtc_calls += calls
        self.dtc_backtracks += backtracks
        return {
            "remaining_fault_ids": list(self._remaining),
            "dtc_attempted_fault_ids": tuple(
                "secondary-{}".format(index) for index in range(calls)
            ),
            "dtc_embedded_fault_ids": ("secondary-a",),
            "current_dtc_secondary_calls": self.dtc_calls,
            "current_dtc_backtracks": self.dtc_backtracks,
        }

    def result(self):
        raise AssertionError("diagnostic driver must not finalize the session")


def test_run_exhausts_native_primary_order_without_finalizing(monkeypatch, capsys):
    from scripts import diagnose_b17_atpg

    session = _FakeSession(
        ["primary-1", "primary-2", "primary-3"],
        increments=[(2, 3), (5, 7), (1, 4)],
    )

    def create_session():
        assert os.environ["PODEM_DTC_DIAGNOSTICS"] == "1"
        return session

    monkeypatch.delenv("PODEM_DTC_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(diagnose_b17_atpg, "create_session", create_session)

    assert diagnose_b17_atpg.run() == 3
    assert session.selected == ["primary-1", "primary-2", "primary-3"]
    stderr = capsys.readouterr().err
    assert "[B17-DEBUG] STEP START step=1 selectable=3 primary=primary-1" in stderr
    done_lines = [line for line in stderr.splitlines() if "STEP DONE" in line]
    assert "step=1 remaining=2 dtc_attempted=2" in done_lines[0]
    assert "step_dtc_calls=2 step_dtc_backtracks=3" in done_lines[0]
    assert "step=2 remaining=1 dtc_attempted=5" in done_lines[1]
    assert "step_dtc_calls=5 step_dtc_backtracks=7" in done_lines[1]
    assert "step=3 remaining=0 dtc_attempted=1" in done_lines[2]
    assert "step_dtc_calls=1 step_dtc_backtracks=4" in done_lines[2]


def test_run_max_steps_truncates_without_reordering(monkeypatch):
    from scripts import diagnose_b17_atpg

    session = _FakeSession(["primary-1", "primary-2", "primary-3"])
    monkeypatch.setattr(diagnose_b17_atpg, "create_session", lambda: session)

    assert diagnose_b17_atpg.run(max_steps=2) == 2
    assert session.selected == ["primary-1", "primary-2"]
    assert session.remaining_fault_ids() == ["primary-3"]


@pytest.mark.parametrize("value", [0, -1])
def test_run_rejects_nonpositive_max_steps(value):
    from scripts import diagnose_b17_atpg

    with pytest.raises(ValueError, match="positive"):
        diagnose_b17_atpg.run(max_steps=value)


def test_create_session_uses_fixed_native_baseline_config(tmp_path, monkeypatch):
    from scripts import diagnose_b17_atpg

    bench = tmp_path / "b17_C.bench"
    bench.write_text("INPUT(a)\nOUTPUT(a)\n", encoding="ascii")
    captured = {}
    sentinel = object()

    class FakeModule:
        @staticmethod
        def StuckAtSession(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(diagnose_b17_atpg, "BENCH_PATH", bench)
    monkeypatch.setattr(diagnose_b17_atpg, "_load_cpp_podem", lambda: FakeModule)

    assert diagnose_b17_atpg.create_session() is sentinel
    assert captured["args"] == (str(bench), "")
    assert captured["kwargs"] == {
        "backtrack_limit": 100,
        "seed": 14,
        "dtc_enabled": True,
        "stc_enabled": False,
    }


def test_main_reports_native_setup_errors(monkeypatch, capsys):
    from scripts import diagnose_b17_atpg

    def fail_session():
        raise RuntimeError("native extension missing")

    monkeypatch.setattr(diagnose_b17_atpg, "create_session", fail_session)

    assert diagnose_b17_atpg.main(["--max-steps", "1"]) == 1
    assert "[B17-DEBUG] ERROR native extension missing" in capsys.readouterr().err
