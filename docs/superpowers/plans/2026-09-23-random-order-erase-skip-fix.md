# Random-order Erase-skip Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `random_order_fault_sim()` evaluate the vector shifted into the current position after erasing a redundant vector.

**Architecture:** Keep the existing shuffled-vector and fault-simulation flow. Change only the loop's index advancement: erase keeps the current index, while retention advances it.

**Tech Stack:** C++14, Python `unittest`/pytest, setuptools/pybind11

**Spec:** `docs/superpowers/specs/2026-09-23-random-order-erase-skip-design.md`

## Global Constraints

- Change only the traversal in `ATPG::random_order_fault_sim()`.
- Preserve shuffling, seed updates, fault-list resets, reverse-order compression, TDF behavior, and RL behavior.
- Use `size_t` so the loop index matches `vectors.size()`.

---

### Task 1: Prevent erase from skipping the shifted vector

**Files:**
- Create: `PODEM/tests/test_tdfsim_source.py`
- Modify: `PODEM/src/tdfsim.cpp:737-743`

**Interfaces:**
- Consumes: `ATPG::tdfault_sim_a_vector(const string &, int &) -> bool`
- Produces: unchanged `ATPG::random_order_fault_sim() -> void`

- [ ] **Step 1: Write the failing source regression test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_random_order_compaction_rechecks_shifted_vector_after_erase():
    source = (ROOT / "PODEM" / "src" / "tdfsim.cpp").read_text(encoding="utf-8")
    start = source.index("void ATPG::random_order_fault_sim()")
    body = source[start:]

    assert "for (size_t i = 0; i < vectors.size();)" in body
    assert "else\n\t\t{\n\t\t\t++i;\n\t\t}" in body
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM\tests\test_tdfsim_source.py -q -p no:cacheprovider
```

Expected: FAIL because the current loop uses an unconditional `i++`.

- [ ] **Step 3: Implement manual index advancement**

```cpp
for (size_t i = 0; i < vectors.size();)
{
    bool redundant = tdfault_sim_a_vector(vectors[i], current_detect_num);
    if (redundant)
    {
        vectors.erase(vectors.begin() + i);
    }
    else
    {
        ++i;
    }
}
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Rebuild the native extension**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
```

Expected: exit code 0 and the in-place `cpp_podem` module is updated.

- [ ] **Step 6: Run the regression suite**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM\tests\test_tdfsim_source.py PODEM\tests\test_fault_mapping.py tests\test_fault_order_rl.py tests\test_runtime_progress.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add PODEM/src/tdfsim.cpp PODEM/tests/test_tdfsim_source.py docs/superpowers/plans/2026-09-23-random-order-erase-skip-fix.md
git commit -m "fix: recheck shifted vector during TDF compaction"
```
