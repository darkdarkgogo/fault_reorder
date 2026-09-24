import os

import pytest


class _FakeSession:
    def __init__(self, remaining):
        self._remaining = list(remaining)
        self.selected = []
        self.dtc_calls = 0
        self.dtc_backtracks = 0

    def remaining_fault_ids(self):
        return list(self._remaining)

    def step(self, primary):
        assert primary == self._remaining[0]
        self.selected.append(primary)
        self._remaining.pop(0)
        self.dtc_calls += 2
        self.dtc_backtracks += 3
        return {
            "remaining_fault_ids": list(self._remaining),
            "dtc_attempted_fault_ids": ("secondary-a", "secondary-b"),
            "dtc_embedded_fault_ids": ("secondary-a",),
            "current_dtc_secondary_calls": self.dtc_calls,
            "current_dtc_backtracks": self.dtc_backtracks,
        }

    def result(self):
        raise AssertionError("diagnostic driver must not finalize the session")


def test_run_exhausts_native_primary_order_without_finalizing(monkeypatch, capsys):
    from scripts import diagnose_b17_atpg

    session = _FakeSession(["primary-1", "primary-2", "primary-3"])

    def create_session():
        assert os.environ["PODEM_DTC_DIAGNOSTICS"] == "1"
        return session

    monkeypatch.delenv("PODEM_DTC_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(diagnose_b17_atpg, "create_session", create_session)

    assert diagnose_b17_atpg.run() == 3
    assert session.selected == ["primary-1", "primary-2", "primary-3"]
    stderr = capsys.readouterr().err
    assert "[B17-DEBUG] STEP START step=1 selectable=3 primary=primary-1" in stderr
    assert "[B17-DEBUG] STEP DONE step=3 remaining=0" in stderr
    assert "step_dtc_calls=2 step_dtc_backtracks=3" in stderr


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
