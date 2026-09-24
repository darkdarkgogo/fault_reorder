# b17_C ATPG Stall Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in native lazy-DTC/PODEMX progress logs and a b17_C-only driver that runs every Primary by default without entering STC.

**Architecture:** `saf_compaction.cpp` captures the environment-gated diagnostics flag once per lazy-DTC/secondary call and emits flushed, single-line `stderr` records from existing control-flow checkpoints. A standalone Python driver owns the b17_C native session and iterates the native remaining-fault order; focused tests cover log gating/result invariance and driver sequencing.

**Tech Stack:** C++14, pybind11, Python 3.9+, argparse, pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-09-24-b17-atpg-stall-diagnostics-design.md`

## Global Constraints

- Diagnostics are enabled only when `PODEM_DTC_DIAGNOSTICS` is exactly `1`.
- Do not change Primary/secondary ordering, DTC eligibility, wire budgets, or backtrack limits.
- Do not add a search stop condition.
- The b17_C driver defaults to all Primary faults and never finalizes/STC-compacts the session.
- New diagnostics write to `stderr` and flush each record.

---

### Task 1: Native lazy-DTC and PODEMX diagnostics

**Files:**
- Modify: `PODEM/src/saf_compaction.cpp`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: `stuck_at_active_step.selected_fault_id`, existing lazy-DTC counters, `fault_identifier(fptr)`, and `podemx_backtrack_limit`.
- Produces: opt-in `[ATPG][DTC-LAZY] start|progress|done` and `[ATPG][PODEMX] progress|slow-done` records; no public API changes.

- [ ] **Step 1: Add a failing native diagnostics regression**

Add a pytest function using `capfd` and `monkeypatch`. Run the existing lazy-DTC fixture once with the environment variable absent and once with it set to `1`, capture file-descriptor-level `stderr`, and assert:

```python
assert "[ATPG][DTC-LAZY]" not in quiet_stderr
assert "[ATPG][DTC-LAZY] start primary=x:GO:sa0" in diagnostic_stderr
assert "[ATPG][DTC-LAZY] done primary=x:GO:sa0" in diagnostic_stderr
assert quiet_result == diagnostic_result
```

- [ ] **Step 2: Run the focused test and verify the missing logs fail it**

Run: `python -m pytest PODEM/tests/test_fault_mapping.py -k dtc_diagnostics -v`

Expected: FAIL because the diagnostic records do not exist.

- [ ] **Step 3: Implement environment gating and PODEMX counters**

In the file-local namespace, add constants for 10-second progress and 2-second slow completion, plus:

```cpp
bool dtc_diagnostics_enabled()
{
    const char *value = std::getenv("PODEM_DTC_DIAGNOSTICS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}
```

Add `<cstdlib>` and `<cstring>`. In `stuck_at_podemx_secondary()`, capture the flag/start time, increment `iterations` at the top of the loop, increment `forward_decisions` only when a valid decision is pushed, emit progress at 10-second intervals, and emit `slow-done` before returning when elapsed time is at least 2 seconds. Map `TRUE/FALSE/MAYBE` to `detected/failed/aborted`.

- [ ] **Step 4: Implement lazy-DTC aggregate logs**

Capture start time and existing attempted/embedded sizes at function entry. Track processed unknown POs and cumulative expanded wires. Use one local emission helper/lambda to print the exact fields from the spec. Check the 100-PO, 500-attempt, and 10-second thresholds only at existing loop checkpoints. Print `start` at entry and `done` at normal exit.

- [ ] **Step 5: Rebuild and run the focused diagnostics test**

Run:

```powershell
python PODEM/setup.py build_ext --inplace
python -m pytest PODEM/tests/test_fault_mapping.py -k dtc_diagnostics -v
```

Expected: PASS.

- [ ] **Step 6: Commit the native diagnostics**

```powershell
git add PODEM/src/saf_compaction.cpp PODEM/tests/test_fault_mapping.py
git commit -m "feat: add opt-in lazy DTC diagnostics"
```

---

### Task 2: b17_C native diagnostic driver

**Files:**
- Create: `scripts/diagnose_b17_atpg.py`
- Create: `tests/test_diagnose_b17_atpg.py`

**Interfaces:**
- Consumes: `cpp_podem.StuckAtSession`, `remaining_fault_ids()`, and `step(primary)`.
- Produces: `run(max_steps: Optional[int] = None) -> int` and CLI `--max-steps N`; returns the number of completed Primary steps.

- [ ] **Step 1: Write failing driver tests with a fake native session**

Cover default exhaustion and explicit truncation. The fake session records selected IDs, returns cumulative DTC call/backtrack counts, and raises if `result()` is called. Assert default selection follows the first remaining ID and `PODEM_DTC_DIAGNOSTICS` is set before session construction.

- [ ] **Step 2: Run the focused driver tests and verify import failure**

Run: `python -m pytest tests/test_diagnose_b17_atpg.py -v`

Expected: FAIL because `scripts.diagnose_b17_atpg` does not exist.

- [ ] **Step 3: Implement the driver**

Resolve repository paths from `Path(__file__)`, insert `PODEM/` into `sys.path`, validate the bench and native extension import, and construct:

```python
cpp_podem.StuckAtSession(
    str(ROOT / "datasets" / "validation" / "b17_C.bench"),
    "",
    backtrack_limit=100,
    seed=14,
    dtc_enabled=True,
    stc_enabled=False,
)
```

The loop prints/flushed `[B17-DEBUG] STEP START` before `step()` and `[B17-DEBUG] STEP DONE` after it. It stops only when remaining faults are empty or `max_steps` has completed, and never calls `result()`.

- [ ] **Step 4: Run the focused driver tests**

Run: `python -m pytest tests/test_diagnose_b17_atpg.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the driver**

```powershell
git add scripts/diagnose_b17_atpg.py tests/test_diagnose_b17_atpg.py
git commit -m "feat: add b17 native ATPG diagnostic driver"
```

---

### Task 3: Targeted regression and b17 smoke diagnosis

**Files:**
- Modify only if a directly related regression fails.

**Interfaces:**
- Consumes: rebuilt in-place `cpp_podem` extension and b17_C validation bench.
- Produces: evidence that normal results remain unchanged and b17 step 1 emits both driver/native records.

- [ ] **Step 1: Run directly affected native tests**

Run:

```powershell
python -m pytest PODEM/tests/test_fault_mapping.py -k "dtc or lazy or limited" -v
python -m pytest PODEM/tests/test_saf_compaction_source.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the driver tests**

Run: `python -m pytest tests/test_diagnose_b17_atpg.py -v`

Expected: PASS.

- [ ] **Step 3: Run one b17_C Primary as a bounded smoke diagnosis**

Run: `python scripts/diagnose_b17_atpg.py --max-steps 1`

Expected: `[B17-DEBUG] STEP START`, `[ATPG][PRIMARY]`, `[ATPG][DTC-LAZY] start`, `[ATPG][DTC-LAZY] done`, and `[B17-DEBUG] STEP DONE`, with no STC finalization.

- [ ] **Step 4: Verify the worktree contains only intended changes**

Run: `git status --short` and `git diff --check`.

Expected: no unrelated changes and no whitespace errors.
