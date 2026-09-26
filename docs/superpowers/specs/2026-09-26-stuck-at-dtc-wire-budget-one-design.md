# Stuck-at DTC Wire Budget One Design

## Goal

Reduce the stuck-at DTC reverse-BFS wire expansion budget from 15 to 1 for
every circuit size so profiling can evaluate the resulting DTC speedup and
ATPG quality trade-off.

## Design

Keep the existing small/default protocol fields and circuit-size branch, but
set both fixed values to 1:

- `dtc_bfs_small_select_fault_try = 1`
- `dtc_bfs_default_select_fault_try = 1`

Update the matching Python protocol identity, protocol tests, DTC batch tests,
and current user-facing algorithm documentation. Do not add a new CLI option
or environment variable.

The DTC search, candidate filtering, ranking, PODEMX order, rollback, and
backtrack limits remain unchanged. This is intentionally an algorithm-parameter
change: candidate sets, embedded secondary faults, pattern count, coverage, and
runtime are allowed to differ from the prior budget-15 baseline.

## Scope

- Change only stuck-at DTC configuration and expectations.
- Keep `PODEM/src/tdfatpg.cpp` and its TDF `select_fault_try = 15` unchanged.
- Do not rewrite historical specs or plans that describe earlier experiments.
- Preserve the existing public protocol field names and schema.

## Verification

- Assert both C++/Python fixed-protocol values are 1.
- Assert small- and large-input DTC batches report `select_fault_try == 1` and
  visit at most one wire.
- Run the focused protocol/DTC tests, followed by the PODEM test suite.
- Confirm `tdfatpg.cpp` still contains `select_fault_try = 15`.
