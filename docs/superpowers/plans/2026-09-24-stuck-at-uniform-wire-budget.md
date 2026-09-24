# Stuck-at Uniform Wire Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every stuck-at DTC unknown-output reverse-BFS use a 15-wire expansion budget, regardless of circuit input count.

**Architecture:** Preserve the existing small/default protocol fields and budget-selection branches, but set both default values to 15 in the C++ protocol and Python protocol identity. Update the existing large-input regression and user-facing protocol documentation; leave all TDF code untouched.

**Tech Stack:** C++14, pybind11, Python 3.9, unittest/pytest, Markdown

**Spec:** `docs/superpowers/specs/2026-09-24-stuck-at-uniform-wire-budget-design.md`

## Global Constraints

- All stuck-at DTC unknown outputs use a wire expansion budget of exactly 15.
- Lazy baseline and ranked/RL stuck-at DTC use the same protocol configuration.
- Keep `dtc_bfs_small_input_threshold`, `dtc_bfs_small_select_fault_try`, and `dtc_bfs_default_select_fault_try` in the public protocol schema.
- Do not modify `PODEM/src/tdfatpg.cpp` or TDF behavior.
- Preserve the existing definition of `visited_wire_count` and reset the budget for each unknown PO.

---

### Task 1: Unify the stuck-at wire budget and document it

**Files:**
- Modify: `PODEM/tests/test_fault_mapping.py:205-220,804-824`
- Modify: `PODEM/src/atpg.h:75-77`
- Modify: `fault_order_rl/environment.py:8-18`
- Modify: `docs/fault-reorder-algorithm.md:35-42`
- Modify: `docs/fault-order-rl.md:10-17`

**Interfaces:**
- Consumes: `StuckAtProtocolConfig::dtc_bfs_default_select_fault_try` and Python `PROTOCOL_CONFIG` validation.
- Produces: unchanged config/schema keys whose small and default wire-budget values are both 15.

- [ ] **Step 1: Change the large-input and protocol-default expectations**

In `PODEM/tests/test_fault_mapping.py`, rename the large-input test and change its assertions:

```python
def test_large_input_dtc_uses_15_wire_budget(self):
    # Existing 33-input fixture remains unchanged.
    self.assertEqual(state["select_fault_try"], 15)
    self.assertLessEqual(state["visited_wire_count"], 15)
```

In `test_stuck_at_session_reports_fixed_protocol`, change:

```python
"dtc_bfs_default_select_fault_try": 15,
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run with a new repository-local `--basetemp` directory:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  PODEM/tests/test_fault_mapping.py `
  -k "large_input_dtc or reports_fixed_protocol" -q -p no:cacheprovider `
  --basetemp tmp/pytest-uniform-wire-budget-red
```

Expected: both selected tests fail because native/Python defaults still report 100 for large circuits.

- [ ] **Step 3: Change the C++ and Python protocol defaults**

In `PODEM/src/atpg.h`, set:

```cpp
int dtc_bfs_default_select_fault_try{15};
```

In `fault_order_rl/environment.py`, set:

```python
"dtc_bfs_default_select_fault_try": 15,
```

Do not change `PODEM/src/tdfatpg.cpp`.

- [ ] **Step 4: Rebuild and rerun the focused tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM/setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  PODEM/tests/test_fault_mapping.py `
  -k "large_input_dtc or reports_fixed_protocol" -q -p no:cacheprovider `
  --basetemp tmp/pytest-uniform-wire-budget-green
```

Expected: both selected tests pass.

- [ ] **Step 5: Update current user-facing documentation**

Replace the 15/100 split in `docs/fault-reorder-algorithm.md` with:

```text
all ncktin : select_fault_try = 15
```

Replace the corresponding fixed-protocol bullet in `docs/fault-order-rl.md` with:

```text
- 每个 unknown PO 的 `select_fault_try` 统一为 `15`。
```

Do not rewrite historical files under `docs/superpowers/specs` or `docs/superpowers/plans`.

- [ ] **Step 6: Run full regression and consistency checks**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests -q `
  -p no:cacheprovider --basetemp tmp/pytest-uniform-wire-budget-full
rg -n "dtc_bfs_default_select_fault_try.*100|select_fault_try.*100" `
  PODEM fault_order_rl tests docs/fault-reorder-algorithm.md docs/fault-order-rl.md
git diff --check
```

Expected: all PODEM tests pass; the search finds no current stuck-at default/budget of 100; `git diff --check` is clean. Any TDF `select_fault_try` logic remains unchanged because TDF is outside the searched replacement scope and outside the diff.

- [ ] **Step 7: Commit the implementation**

```powershell
git add -- PODEM/tests/test_fault_mapping.py PODEM/src/atpg.h `
  fault_order_rl/environment.py docs/fault-reorder-algorithm.md `
  docs/fault-order-rl.md
git commit -m "perf: cap stuck-at DTC wire budget at 15"
```

