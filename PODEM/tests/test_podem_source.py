import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_trace_unknown_path_deduplicates_each_top_level_query():
    source = (ROOT / "PODEM" / "src" / "podem.cpp").read_text(
        encoding="utf-8"
    )
    header = (ROOT / "PODEM" / "src" / "atpg.h").read_text(
        encoding="utf-8"
    )

    assert "bool trace_unknown_path(wptr, unordered_set<wptr> &);" in header
    assert "unordered_set<wptr> visited;" in source
    assert "return trace_unknown_path(w, visited);" in source
    assert "if (!visited.insert(w).second)" in source
    assert "trace_unknown_path(output, visited)" in source


def test_stuck_at_detection_reporting_remains_after_n_detect_thresholds():
    source = (ROOT / "PODEM" / "src" / "faultsim.cpp").read_text(
        encoding="utf-8"
    )

    report = source.index("newly_detected_faults->push_back(detected_fault)")
    threshold_sites = [
        match.start()
        for match in re.finditer("detected_time == detected_num", source)
    ]
    assert len(threshold_sites) == 3
    assert max(threshold_sites) < report
    assert source.index("/* Drop detected faults in list order") < report
