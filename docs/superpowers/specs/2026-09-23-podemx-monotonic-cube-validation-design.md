# PODEM-X Monotonic PI Cube Validation Design

## Goal

Remove the quadratic preserved-fault replay performed after every successful
stuck-at DTC secondary. Replace it with an O(number of PIs) postcondition that
requires every accepted PI cube to be a monotonic refinement of the previous
accepted cube.

## Correctness invariant

For every PI position, a previously fixed `0` or `1` must keep the same value.
An old `U` may remain `U` or become `0` or `1`. Both cubes must have exactly
`cktin.size()` entries and every normalized entry must be in `{0, 1, U}`.

This is sufficient for the repository's five-valued combinational simulation:
`check_test()` succeeds only for a definite `D` or `D_bar` at a primary output,
and refining remaining `U` inputs cannot change a definite simulated value.

## Production behavior

- `stuck_at_podemx_secondary()` remains responsible for finding a detecting
  refinement without changing PIs that were fixed at entry.
- A successful secondary produces a proposed good-circuit PI cube.
- Before acceptance, `validate_stuck_at_monotonic_cube()` checks the complete
  old/new cube postcondition and throws a diagnostic `runtime_error` on an
  invariant violation.
- A valid proposed cube is accepted immediately. Production code has no
  preserved-fault vector, replay loop, debug fallback, or sampled replay.
- FALSE and MAYBE results continue to restore the previous accepted PI cube.
- Native TDF-lazy and ranked-RL paths continue to share the same secondary
  atomic operation.

## State and protocol changes

- Delete `stuck_at_preserved_faults` and all initialization and update sites.
- Keep `stuck_at_accepted_pi_cube` as the sole accepted-cube state.
- Change `compression_algorithm_version` to
  `stuck_at_podemx_bfs_ranked_dtc_monotonic_v4` so checkpoints and experiment
  metadata cannot silently mix the old replay protocol with the new invariant.
- Keep `dtc_rollback_algorithm=accepted_pi_cube_resim_v1`; rollback mechanics
  do not change.

## Tests and acceptance

- Source-level regression checks require the monotonic validator and reject
  any return of the preserved-fault state or replay loop.
- Existing DTC integration tests continue to cover successful embedding,
  failed/MAYBE restoration, native lazy execution, ranked execution, XOR fault
  mapping, final coverage, and STC preservation.
- Protocol tests require the v4 identifier.
- The C++ extension must rebuild cleanly and the available targeted regression
  suites must pass.

## Non-goals

This change does not alter Primary PODEM, secondary search order, BFS budgets,
RL ranking, random fill, final fault simulation, or STC.
