from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_random_order_compaction_rechecks_shifted_vector_after_erase():
    source = (ROOT / "PODEM" / "src" / "tdfsim.cpp").read_text(encoding="utf-8")
    start = source.index("void ATPG::random_order_fault_sim()")
    body = source[start:]

    assert "for (size_t i = 0; i < vectors.size();)" in body
    assert "else\n\t\t{\n\t\t\t++i;\n\t\t}" in body
