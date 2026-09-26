# Stuck-at DTC Wire Budget One Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set both fixed stuck-at DTC reverse-BFS wire budgets to 1 while preserving the existing protocol schema and TDF behavior.

**Architecture:** Keep the existing small/default configuration fields and selection branch. Change their C++ and Python fixed values together, then update exact protocol, batch-budget, and documentation expectations.

**Tech Stack:** C++14, Python 3, pybind11, pytest, Markdown

**Spec:** `docs/superpowers/specs/2026-09-26-stuck-at-dtc-wire-budget-one-design.md`

## Global Constraints

- Both stuck-at DTC wire-budget fields equal 1.
- Preserve their existing names and public schema.
- Do not modify `PODEM/src/tdfatpg.cpp`; its TDF budget remains 15.
- Candidate sets, patterns, coverage, and runtime may differ from budget 15.
- Do not rewrite historical specs or plans.

---

### Task 1: Change and verify the fixed stuck-at budgets

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `fault_order_rl/environment.py`
- Modify: `PODEM/tests/test_fault_mapping.py`
- Modify: `tests/test_fault_order_rl.py`
- Modify: `docs/fault-reorder-algorithm.md`
- Modify: `docs/fault-order-rl.md`

**Interfaces:**
- Consumes: `StuckAtProtocolConfig::dtc_bfs_small_select_fault_try`, `StuckAtProtocolConfig::dtc_bfs_default_select_fault_try`, and Python `PROTOCOL_CONFIG`.
- Produces: the unchanged protocol fields with fixed value 1.

- [ ] **Step 1: Change the exact test expectations to 1**

Update small/large DTC batch assertions and both protocol dictionaries:

```python
self.assertEqual(state["select_fault_try"], 1)
self.assertLessEqual(state["visited_wire_count"], 1)
```

```python
"dtc_bfs_small_select_fault_try": 1,
"dtc_bfs_default_select_fault_try": 1,
```

- [ ] **Step 2: Run focused tests and verify the old implementation fails**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  PODEM/tests/test_fault_mapping.py tests/test_fault_order_rl.py `
  -k "wire_budget or fixed_protocol" -q -p no:cacheprovider `
  --basetemp .pytest-wire-budget-one-red
```

Expected: assertions report the existing value 15 instead of 1.

- [ ] **Step 3: Change the C++ and Python fixed protocol values**

Set:

```cpp
int dtc_bfs_small_select_fault_try{1};
int dtc_bfs_default_select_fault_try{1};
```

and:

```python
"dtc_bfs_small_select_fault_try": 1,
"dtc_bfs_default_select_fault_try": 1,
```

- [ ] **Step 4: Update current documentation**

Document `select_fault_try = 1` for all stuck-at circuits in
`docs/fault-reorder-algorithm.md` and `docs/fault-order-rl.md`. Leave historical
files under `docs/superpowers` unchanged.

- [ ] **Step 5: Rebuild and run focused tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM/setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  PODEM/tests/test_fault_mapping.py tests/test_fault_order_rl.py `
  -k "wire_budget or fixed_protocol" -q -p no:cacheprovider `
  --basetemp .pytest-wire-budget-one-green
```

Expected: all selected tests pass.

- [ ] **Step 6: Run the PODEM suite and static consistency checks**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests -q `
  -p no:cacheprovider --basetemp .pytest-wire-budget-one-full
rg -n "dtc_bfs_(small|default)_select_fault_try.*15|select_fault_try.*15" `
  PODEM/src/atpg.h fault_order_rl/environment.py `
  docs/fault-reorder-algorithm.md docs/fault-order-rl.md
rg -n "select_fault_try = 15" PODEM/src/tdfatpg.cpp
git diff --check
```

Expected: no current stuck-at value remains 15; TDF still reports 15; no
whitespace errors; all PODEM tests pass.

- [ ] **Step 7: Commit the implementation**

```powershell
git add PODEM/src/atpg.h fault_order_rl/environment.py `
  PODEM/tests/test_fault_mapping.py tests/test_fault_order_rl.py `
  docs/fault-reorder-algorithm.md docs/fault-order-rl.md
git commit -m "perf: reduce stuck-at DTC wire budget to one"
```
