from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_stuck_at_dtc_uses_monotonic_cube_validation_without_fault_replay():
    source = (ROOT / "PODEM" / "src" / "saf_compaction.cpp").read_text(
        encoding="utf-8"
    )
    header = (ROOT / "PODEM" / "src" / "atpg.h").read_text(encoding="utf-8")
    atpg = (ROOT / "PODEM" / "src" / "atpg.cpp").read_text(encoding="utf-8")

    assert "validate_stuck_at_monotonic_cube" in source
    assert "validate_stuck_at_monotonic_cube" in header
    assert "stuck_at_preserved_faults" not in source + header + atpg
    assert "for (fptr preserved" not in source


def test_podemx_diagnostics_keep_required_fields_and_counter_boundaries():
    source = (ROOT / "PODEM" / "src" / "saf_compaction.cpp").read_text(
        encoding="utf-8"
    )

    assert "constexpr double kDtcProgressIntervalSeconds = 10.0;" in source
    assert "constexpr double kPodemxSlowSeconds = 2.0;" in source
    assert (
        "[ATPG][PODEMX] progress primary=%s secondary=%s "
        "iterations=%zu decisions=%zu depth=%zu backtracks=%d elapsed_s=%.3f"
    ) in source
    assert (
        "[ATPG][PODEMX] slow-done primary=%s secondary=%s status=%s "
        "iterations=%zu decisions=%zu backtracks=%d elapsed_s=%.3f"
    ) in source
    assert 'status == TRUE ? "detected"' in source
    assert 'status == MAYBE ? "aborted" : "failed"' in source
    assert source.index("++iterations;") < source.index(
        "if (stuck_at_cube_detects(fault))"
    )
    assert source.index("decisions.push_back") < source.index(
        "++forward_decisions;"
    )
