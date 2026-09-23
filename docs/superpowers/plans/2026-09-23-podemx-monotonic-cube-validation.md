# PODEM-X Monotonic PI Cube Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove production preserved-fault replay and accept successful secondary cubes only after an O(number of PIs) monotonic-refinement check.

**Architecture:** Keep `stuck_at_accepted_pi_cube` as the canonical state shared by native lazy and ranked DTC. A private ATPG validator checks cube size, value domain, and fixed-PI preservation immediately before a proposed cube is committed. The obsolete preserved-fault vector and every replay mutation are deleted.

**Tech Stack:** C++14 ATPG core, pybind11 extension, Python pytest/unittest regression tests.

**Spec:** `docs/superpowers/specs/2026-09-23-podemx-monotonic-cube-validation-design.md`

## Global Constraints

- Production code must contain no preserved-fault replay state, loop, feature flag, sampled fallback, or debug fallback.
- Native TDF-lazy and ranked-RL paths must continue to use `attempt_stuck_at_dtc_secondary()`.
- `dtc_rollback_algorithm` remains `accepted_pi_cube_resim_v1`.
- `compression_algorithm_version` becomes `stuck_at_podemx_bfs_ranked_dtc_monotonic_v4`.
- Primary PODEM, candidate discovery, ranking, random fill, final fault simulation, and STC behavior do not change.

---

### Task 1: Lock the production structure with failing tests

**Files:**
- Create: `PODEM/tests/test_saf_compaction_source.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: ATPG source files and `fault_order_rl.trainer.SOLVER_PROTOCOL`.
- Produces: regression requirements for the monotonic validator, removed replay state, and v4 protocol identity.

- [ ] **Step 1: Write the source regression test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stuck_at_dtc_uses_monotonic_cube_validation_without_fault_replay():
    source = (ROOT / "PODEM" / "src" / "saf_compaction.cpp").read_text(encoding="utf-8")
    header = (ROOT / "PODEM" / "src" / "atpg.h").read_text(encoding="utf-8")
    atpg = (ROOT / "PODEM" / "src" / "atpg.cpp").read_text(encoding="utf-8")

    assert "validate_stuck_at_monotonic_cube" in source
    assert "validate_stuck_at_monotonic_cube" in header
    assert "stuck_at_preserved_faults" not in source + header + atpg
    assert "for (fptr preserved" not in source
```

- [ ] **Step 2: Require the v4 protocol identity**

Add to `test_fixed_protocol_and_training_defaults()`:

```python
assert SOLVER_PROTOCOL["compression_algorithm_version"] == (
    "stuck_at_podemx_bfs_ranked_dtc_monotonic_v4"
)
```

- [ ] **Step 3: Run tests and verify the intended failures**

Run:

```powershell
python -m pytest PODEM/tests/test_saf_compaction_source.py -q -p no:cacheprovider
```

Expected: FAIL because the validator is absent and replay state still exists.

The protocol test may require the repository's torch-enabled environment; if it cannot collect, record the missing dependency and verify it after implementation where available.

### Task 2: Replace preserved replay with the monotonic postcondition

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/src/saf_compaction.cpp`

**Interfaces:**
- Consumes: `stuck_at_accepted_pi_cube`, `cktin`, and the proposed normalized PI cube.
- Produces: `void ATPG::validate_stuck_at_monotonic_cube(const vector<int>&, const vector<int>&) const`.

- [ ] **Step 1: Declare the validator and remove replay state**

Declare the private validator beside the other stuck-at compaction helpers and delete `vector<fptr> stuck_at_preserved_faults`.

- [ ] **Step 2: Implement the complete postcondition**

Implement a validator that:

```cpp
if (accepted_cube.size() != cktin.size() ||
    proposed_cube.size() != cktin.size())
    throw runtime_error("Stuck-at DTC PI cube size mismatch");
```

For every index, normalize through `good_value()`, reject values outside
`{0, 1, U}`, and reject any `old_value != U && new_value != old_value`. Include
the PI index and values in the diagnostic.

- [ ] **Step 3: Commit successful secondary cubes directly**

Replace the historical replay block in `attempt_stuck_at_dtc_secondary()` with:

```cpp
if (embedded)
{
    validate_stuck_at_monotonic_cube(
        stuck_at_accepted_pi_cube, proposed_cube);
    stuck_at_accepted_pi_cube = proposed_cube;
    stuck_at_active_step.dtc_embedded_fault_ids.push_back(identifier);
}
```

- [ ] **Step 4: Delete initialization and Primary insertion sites**

Remove the preserved-vector `clear()` from `reset_stuck_at_active_step()` and the Primary `push_back()` from `begin_stuck_at_step_impl()`.

- [ ] **Step 5: Rebuild the C++ extension**

Run:

```powershell
python PODEM/setup.py build_ext --inplace
```

Expected: successful compilation and an updated `cpp_podem` extension.

- [ ] **Step 6: Run the source and available DTC regression tests**

Run:

```powershell
python -m pytest PODEM/tests/test_saf_compaction_source.py -q -p no:cacheprovider
```

Expected: PASS.

Run the DTC integration selection in an environment providing the project dependencies:

```powershell
python -m pytest PODEM/tests/test_fault_mapping.py -k dtc -q -p no:cacheprovider
```

Expected: PASS.

### Task 3: Version and document the new solver protocol

**Files:**
- Modify: `fault_order_rl/trainer.py`
- Modify: `tests/test_fault_order_rl.py`
- Modify: `docs/fault-reorder-algorithm.md`

**Interfaces:**
- Consumes: the unchanged native `PROTOCOL_CONFIG` and new C++ acceptance behavior.
- Produces: v4 checkpoint/experiment identity and current algorithm documentation.

- [ ] **Step 1: Change the solver protocol version**

Set:

```python
"compression_algorithm_version": "stuck_at_podemx_bfs_ranked_dtc_monotonic_v4"
```

- [ ] **Step 2: Update the algorithm description**

Replace the preserved-fault acceptance text with the invariant: successful
secondary cubes are committed only after fixed PIs are proven unchanged; failed
or limited searches restore the previous accepted cube; no historical faults
are replayed in production.

- [ ] **Step 3: Run protocol and static tests**

Run:

```powershell
python -m pytest PODEM/tests/test_saf_compaction_source.py tests/test_fault_order_rl.py::test_fixed_protocol_and_training_defaults -q -p no:cacheprovider
```

Expected: PASS when torch is installed; otherwise the source test must pass and the dependency failure must be reported.

- [ ] **Step 4: Review the complete diff**

Run `git diff --check`, confirm `rg -n "stuck_at_preserved_faults|for \(fptr preserved" PODEM/src` returns no matches, and inspect the diff for unrelated changes.
