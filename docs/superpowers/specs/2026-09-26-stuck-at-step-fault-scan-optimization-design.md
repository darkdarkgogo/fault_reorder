# Stuck-at Step Fault Scan Optimization Design

## Purpose

Reduce bookkeeping work in each ordered stuck-at ATPG step without changing
PODEM, DTC, fault-simulation, coverage, pattern, STC, or Python/RL-visible
semantics. Keep both DTC wire budgets at 15; changing them is a separate
algorithm experiment.

The baseline is commit `5c8673e342eecc6474e8fecd640edee890016096`.

## Current Problem

Each step performs work beyond the required PODEM, DTC, and fault simulation:

- it scans the owning fault catalog to resolve the selected string ID;
- it materializes all selectable IDs and hashes them only to validate one ID;
- it scans `flist_undetect` again to build a before snapshot;
- after simulation, it materializes all surviving IDs, hashes them, and diffs
  them against the before snapshot to infer newly detected faults;
- it scans `flist_undetect` again to construct `remaining_fault_ids`;
- it scans the owning catalog in `get_stuck_at_result()` to recompute the
  detected collapsed-fault count;
- the native `run_stuck_at()` loop performs an additional selectable-ID scan
  after discarding the same information returned by the step.

The repeated string construction and hashing are avoidable. At least one
post-step traversal remains necessary while the Python/RL API requires the
complete ordered `remaining_fault_ids` list.

## Considered Approaches

### A. Merge only the before-step scans

Generate selectable and before IDs in one traversal. This is the smallest
change but retains the expensive after-set construction and string diff. It
also becomes dead-end work once fault simulation reports detections directly.

### B. Record detections at every `detect = TRUE` assignment

This removes the string diff, but detection occurs through several paths:
direct primary-output faults, propagated gate-input faults, and packet
simulation. Instrumenting every transition is easy to make incomplete and can
change the externally visible ordering of `newly_detected_fault_ids`.

### C. Collect newly detected faults during fault dropping (selected)

Extend `fault_sim_a_vector()` with an optional output for newly detected fault
pointers. During its existing ordered fault-dropping traversal, append each
fault that is about to be removed. The traversal already defines the exact set
used to update `num_of_current_detect`, and raw pointers remain valid because
the owning `flist` retains the `unique_ptr<FAULT>` objects.

This approach centralizes detection reporting, preserves list order, and does
not affect fault-simulation callers that do not request the optional output.

## Design

### Fault-simulation result reporting

Change the private fault-simulation interface to accept an optional
`vector<fptr> *newly_detected_faults`. The pointer defaults to `nullptr` so
ordinary simulation and STC replay remain unchanged.

The fault-dropping code will use an explicit ordered `forward_list`
erase-after loop. For every entry whose `detect == TRUE`, it will:

1. append the `fptr` when an output vector was supplied;
2. add `eqv_fault_num` to `num_of_current_detect`;
3. erase the entry from `flist_undetect`.

Collection happens exactly once per dropped collapsed fault and follows the
pre-drop `flist_undetect` order.

### Step result construction

`complete_stuck_at_step()` will pass a local `vector<fptr>` to fault
simulation only when a pattern was generated. It will materialize fault IDs
from that vector after simulation and assign them to
`newly_detected_fault_ids`.

The before snapshot, post-simulation survivor snapshot, hash set, and string
diff will be removed. The existing single construction of ordered
`remaining_fault_ids` remains.

If no pattern was generated, fault simulation is not called and
`newly_detected_fault_ids` remains empty.

### Selected-fault lookup and validation

Build a stable `fault_id -> fptr` lookup when the fault catalog is prepared.
`begin_stuck_at_step_impl()` will resolve the requested ID through this lookup
and validate selectability directly with membership/state checks instead of
materializing and hashing every selectable string.

The distinction between an unknown fault ID and a known but non-selectable
fault ID must remain unchanged. The primary-start diagnostic may obtain the
selectable count without materializing identifiers.

### Incremental detected-collapsed count

Maintain a stuck-at session counter for detected collapsed faults and increment
it from the newly detected pointer vector. `get_stuck_at_result()` will read
the counter instead of rescanning the owning catalog on every step. Session
initialization/reset paths must initialize the counter consistently.

### Native loop

The native `run_stuck_at()` path will avoid rebuilding and discarding full
selectable-ID vectors around each step. It may select the next eligible `fptr`
directly while preserving current fault order. The Python/RL step API will
continue returning the full ordered remaining-ID list.

### Timing

Use `std::chrono::steady_clock` to measure FSIM, bookkeeping, and total step
time. Timing output must be behind the existing diagnostics/progress behavior
or aggregated so that per-step I/O does not dominate the measured work.
Existing PRIMARY and DTC timing remains intact.

## Correctness Constraints

- `newly_detected_fault_ids` must contain exactly the collapsed faults dropped
  by this step's generated vector, in the same order as the baseline before/
  after diff.
- `detected_equivalent_faults` must still increase by the sum of the same
  `eqv_fault_num` values.
- N-detect behavior must report a fault only when `detected_time` reaches
  `detected_num` and the fault is actually dropped.
- Redundant and aborted primaries must never be reported as newly detected
  unless a later generated vector genuinely detects and drops them.
- Fault-dropping time and ordering must not otherwise change.
- `remaining_fault_ids` must retain the exact current filter and order:
  `!test_tried && detect != REDUNDANT` over `flist_undetect`.
- Baseline and ranked-DTC paths must expose identical step-result semantics.
- STC replay and other fault-simulation callers must not accumulate step-local
  detection pointers.
- The wire-budget constants remain 15.

## Verification

Add focused tests for:

- exact ordered `newly_detected_fault_ids` equality against expected catalog
  order when direct and packet-detected faults occur in one vector;
- no newly detected IDs for redundant and aborted steps without a pattern;
- an earlier aborted fault detected by a later vector;
- N-detect values greater than one;
- a redundant tail entry and packet flush behavior;
- baseline and ranked-DTC equivalence;
- STC preserving fault IDs, equivalent coverage, and pattern results;
- unknown versus known-nonselectable error behavior;
- native and Python/RL completion producing the same final metrics.

Run the existing PODEM mapping/session tests first, then the complete repository
test suite. Finally compare b12_C, b15_C, and b20_C at the same seed and fixed
wire budget for fault catalog, ordered DTC attempts/embeds, generated vectors,
coverage, and STC output. Performance acceptance requires lower bookkeeping
time and no statistically meaningful total-runtime regression; it does not
assume a fixed speedup before profiling.

## Out of Scope

- Changing either DTC wire budget from 15 to 3.
- Changing DTC candidate selection, ranking, or PODEMX order.
- Replacing the Python/RL full remaining-ID list with a lazy or compact API.
- Parallelizing fault simulation.
