# Stuck-at Step Fault Scan Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant per-step fault-ID scans and string-set differencing while preserving exact ordered ATPG results.

**Architecture:** Report newly detected faults from the existing ordered fault-dropping traversal through an optional output parameter, and construct Python-visible strings only at the step boundary. Add a stable ID lookup and incremental collapsed-detection count, then remove native-loop scans that duplicate step results. Measure FSIM, bookkeeping, and total step time without changing ATPG decisions.

**Tech Stack:** C++14, `std::forward_list`, pybind11, Python 3, pytest

**Spec:** `docs/superpowers/specs/2026-09-26-stuck-at-step-fault-scan-optimization-design.md`

## Global Constraints

- Keep both stuck-at DTC wire budgets at 15.
- Preserve exact ordered `newly_detected_fault_ids` and `remaining_fault_ids`.
- Preserve PODEM, DTC, fault-simulation, N-detect, coverage, pattern, and STC behavior.
- Do not collect step-local detection pointers during ordinary fault simulation or STC replay.
- Run focused tests first and only then the relevant full PODEM suite.

---

### Task 1: Report dropped faults directly from fault simulation

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/faultsim.cpp`
- Modify: `PODEM/src/atpg.cpp`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: `flist_undetect`, `FAULT::detect`, and the existing ordered fault-dropping traversal.
- Produces: `void fault_sim_a_vector(const string &, int &, vector<fptr> *newly_detected_faults = nullptr)`.

- [ ] **Step 1: Add an exact-order regression test**

Add a fixture whose generated vector detects both a direct-output fault and a packet-simulated fault, and assert the complete ordered list:

```python
self.assertEqual(step["newly_detected_fault_ids"], expected_catalog_order)
```

Also retain the existing redundant, aborted, later-detected, and DTC-equivalence assertions.

- [ ] **Step 2: Run the focused mapping tests before the change**

Run:

```powershell
python -m pytest PODEM/tests/test_fault_mapping.py -q -p no:cacheprovider `
  --basetemp tmp/pytest-fault-scan-baseline
```

Expected: existing behavior passes and records the baseline order.

- [ ] **Step 3: Extend the simulator interface and replace `remove_if`**

Use an optional output pointer and an explicit ordered erase loop:

```cpp
void ATPG::fault_sim_a_vector(
    const string &vec,
    int &num_of_current_detect,
    vector<fptr> *newly_detected_faults)
{
    auto before = flist_undetect.before_begin();
    auto current = flist_undetect.begin();
    while (current != flist_undetect.end())
    {
        fptr fault = *current;
        if (fault->detect == TRUE)
        {
            if (newly_detected_faults != nullptr)
                newly_detected_faults->push_back(fault);
            num_of_current_detect += fault->eqv_fault_num;
            current = flist_undetect.erase_after(before);
        }
        else
        {
            before = current;
            ++current;
        }
    }
}
```

- [ ] **Step 4: Remove before/after differencing from the step**

Delete `stuck_at_step_before_ids`. In `complete_stuck_at_step()`, pass a local
`vector<fptr>` to fault simulation, materialize its IDs once, and retain the
single `get_selectable_fault_ids()` call for `remaining_fault_ids`.

- [ ] **Step 5: Rebuild and run focused tests**

Run:

```powershell
python PODEM/setup.py build_ext --inplace
python -m pytest PODEM/tests/test_fault_mapping.py -q -p no:cacheprovider `
  --basetemp tmp/pytest-fault-scan-task1
```

Expected: all mapping/session tests pass with exact list order unchanged.

### Task 2: Remove selected-ID and result-count rescans

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`

**Interfaces:**
- Consumes: canonical IDs after `generate_fault_list()` and the task-1 detection vector.
- Produces: session-local `unordered_map<string, fptr> stuck_at_faults_by_id` and `int stuck_at_detected_collapsed_faults`.

- [ ] **Step 1: Build the stable lookup during session preparation**

Populate the map once from owning `flist` after fault IDs are final. Reject
duplicates defensively. Resolve `fault_id` through the map and validate with:

```cpp
fault->detect == FALSE && !fault->test_tried
```

Count selectable entries for diagnostics without constructing strings.

- [ ] **Step 2: Make collapsed detection count incremental**

Reset `stuck_at_detected_collapsed_faults` with other session counters,
increment it by `newly_detected_faults.size()`, and return it directly from
`get_stuck_at_result()`.

- [ ] **Step 3: Reuse step remaining results in the native loop**

Use `AtpgStepResult::remaining_fault_ids` to choose the next primary instead
of calling `get_selectable_fault_ids()` after every step:

```cpp
vector<string> selectable = get_selectable_fault_ids();
while (!selectable.empty())
    selectable = step_stuck_at(selectable.front()).remaining_fault_ids;
```

- [ ] **Step 4: Run focused error and equivalence tests**

Run:

```powershell
python -m pytest PODEM/tests/test_fault_mapping.py `
  -k "invalid or unknown or equivalence or coverage or stc" -q `
  -p no:cacheprovider --basetemp tmp/pytest-fault-scan-task2
```

Expected: unknown and non-selectable errors, coverage, DTC equivalence, and STC remain unchanged.

### Task 3: Add segmented timing and complete regression

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: the existing `steady_clock` primary timing and step lifecycle.
- Produces: `[ATPG][FSIM]`, `[ATPG][BOOK]`, and `[ATPG][STEP]` elapsed-time diagnostics.

- [ ] **Step 1: Measure only the intended regions**

Store the step start time when a valid step begins. Time only the
`fault_sim_a_vector()` call as FSIM, and time ID materialization plus remaining
list/result assembly as BOOK. Emit nonnegative elapsed seconds and existing
fault counts; do not change result data.

- [ ] **Step 2: Add a diagnostic smoke assertion**

Use the existing subprocess/session fixture to assert that one successful step
emits all three timing tags and numeric `elapsed_s` values without asserting
wall-clock magnitudes.

- [ ] **Step 3: Run the relevant full suite**

Run:

```powershell
python -m pytest PODEM/tests -q -p no:cacheprovider `
  --basetemp tmp/pytest-fault-scan-full
```

Expected: all PODEM tests pass.

- [ ] **Step 4: Perform final static and diff checks**

Run:

```powershell
git diff --check
rg -n "stuck_at_step_before_ids|after_set|undetected_ids" PODEM/src
rg -n "dtc_bfs_(small|default)_select_fault_try" PODEM/src/atpg.h fault_order_rl/environment.py
```

Expected: no whitespace errors; deleted bookkeeping identifiers are absent;
both wire-budget defaults remain 15.

- [ ] **Step 5: Review the final diff against the spec**

Verify exact output ordering, optional simulator output isolation, counter
resets, error-message preservation, and absence of unrelated refactoring.
