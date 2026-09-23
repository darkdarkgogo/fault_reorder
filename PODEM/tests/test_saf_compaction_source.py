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
