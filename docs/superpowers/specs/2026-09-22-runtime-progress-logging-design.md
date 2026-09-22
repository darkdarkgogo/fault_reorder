# Fault Reorder Runtime Progress Logging Design

## Goal

Add diagnostic progress logs that distinguish circuit catalog construction, Python artifact validation, native session initialization, Primary PODEM, DTC, and STC/finalization without changing ATPG or PPO behavior.

## Logging contract

- Diagnostic messages go to stderr and are flushed immediately.
- Every message names its phase and circuit where the caller has that context.
- Long loops are rate limited: Python ATPG steps log the first five and every 100th step; DTC candidates log the first five and every 1000th candidate.
- Start/done pairs include elapsed time on completion. Python catalog and validation failures emit an error event and then re-raise the original exception.

## Boundaries

- `fault_order_rl/data.py` logs catalog and validation start/done/error.
- `fault_order_rl/trainer.py` logs baseline ownership, native session readiness, sampled ATPG steps, and STC/finalize start/done.
- `PODEM/src/python_bindings.cpp` splits the post-`generate_fault_list` path into fault-list, catalog-copy, and Python-dict conversion checkpoints.
- `PODEM/src/atpg.cpp` surrounds each Primary PODEM call.
- `PODEM/src/saf_compaction.cpp` reports DTC progress at the configured sampling cadence.

## Non-goals

Do not change backtrack limits, fault ordering, fault simulation, DTC/STC behavior, PPO/GAE, reward calculation, seeds, checkpoints, or returned data structures.

## Verification

- Unit tests assert catalog/validation ordering, error logging, sampling cadence, and finalize markers.
- Rebuild `cpp_podem` and run the focused Python/C++ test suites.
- Existing result-validation tests remain authoritative for algorithm behavior.
