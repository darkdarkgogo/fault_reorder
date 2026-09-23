# SCOAP Fanout Temp Warning Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the undefined read of `temp` when a SCOAP fanout branch connects directly to an output.

**Architecture:** Keep the existing fanout minimum-reduction intact. Initialize each branch-local candidate to the current `co[i]` as a neutral value, then set it to zero for `OUTPUT`, so the common `if (temp < co[i])` reduction never reads stale or uninitialized state.

**Tech Stack:** C++11/14, MinGW warning diagnostics, setuptools/pybind11, pytest.

**Spec:** `C:\Users\acer\Downloads\fault_reorder_warning_fix_design.pdf`

## Global Constraints

- Modify only branch-local `temp` initialization and the `OUTPUT` case in the fanout-branch loop of `ATPG::calculate_scoap()`.
- Do not change signed/unsigned loop types in this task.
- Do not change PODEM, DTC/STC, RL policy, or protocol configuration.
- Preserve `CO = 0` for a branch connected directly to a primary output.

---

### Task 1: Fix and verify the SCOAP fanout branch value

**Files:**
- Modify: `PODEM/src/tdfsim.cpp:651`

**Interfaces:**
- Consumes: the existing per-branch `temp` minimum-reduction in `ATPG::calculate_scoap()`.
- Produces: a defined `temp == 0` value for an `OUTPUT` fanout branch.

- [ ] **Step 1: Reproduce the compiler diagnostic**

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -Wmaybe-uninitialized `
  -I PODEM\src -c PODEM\src\tdfsim.cpp -o tdfsim-before.o
```

Expected: `tdfsim.cpp:680: warning: 'temp' may be used uninitialized`.

- [ ] **Step 2: Apply the minimal correction**

```cpp
n = w->onode[j];
temp = co[i];
switch (n->type)
{
case OUTPUT:
    temp = 0;
    break;
}
```

- [ ] **Step 3: Re-run the focused compiler diagnostic**

Run the Step 1 command with output `tdfsim-after.o`.

Expected: no `temp may be used uninitialized` warning; existing unrelated warnings may remain.

- [ ] **Step 4: Rebuild the Python extension**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
```

Expected: C++ compilation succeeds. If the in-place DLL is locked by an existing process, use the successfully produced `PODEM/build/lib.win-amd64-cpython-39` extension for tests and report the lock.

- [ ] **Step 5: Run the relevant regression suites**

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest `
  PODEM\tests\test_fault_mapping.py `
  tests\test_fault_order_rl.py `
  tests\test_runtime_progress.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the fix**

```powershell
git add PODEM/src/tdfsim.cpp docs/superpowers/plans/2026-09-23-scoap-temp-warning-fix.md
git commit -m "fix: initialize SCOAP output branch cost"
```
