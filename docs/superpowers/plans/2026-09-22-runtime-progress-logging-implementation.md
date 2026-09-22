# Runtime Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add flushed, rate-limited progress logs that identify whether a run is in catalog construction, validation, session setup, Primary PODEM, DTC, or STC.

**Architecture:** Python owns circuit-level and session-level markers; the C++ binding owns catalog conversion markers; ATPG core code owns Primary and DTC markers. All diagnostics use stderr and preserve existing return values and solver state.

**Tech Stack:** Python 3, pytest, C++14/pybind11, setuptools.

**Spec:** `docs/superpowers/specs/2026-09-22-runtime-progress-logging-design.md`

## Global Constraints

- Do not change Primary backtrack limit 100 or DTC secondary backtrack limit 50.
- Do not change fault ordering, simulation, DTC/STC behavior, PPO/GAE, reward, seed, checkpoint data, or API return structures.
- Flush every diagnostic marker immediately and rate-limit long-loop progress.

---

### Task 1: Python LOAD and session progress

**Files:**
- Create: `fault_order_rl/progress.py`
- Modify: `fault_order_rl/data.py`
- Modify: `fault_order_rl/trainer.py`
- Test: `tests/test_runtime_progress.py`

**Interfaces:**
- Produces: `progress(scope: str, event: str, **fields) -> None` and `sample_progress(index: int, first: int, every: int) -> bool`.
- Consumes: Existing manifest, environment, session, and trainer interfaces unchanged.

- [ ] Write focused tests for flushed stderr output, catalog/validation ordering and errors, ATPG sampling, and finalize markers.
- [ ] Run `pytest tests/test_runtime_progress.py -q` and confirm the new tests fail.
- [ ] Implement the progress helper and Python call-site markers.
- [ ] Run `pytest tests/test_runtime_progress.py -q` and confirm it passes.

### Task 2: Native catalog, Primary, and DTC progress

**Files:**
- Modify: `PODEM/src/python_bindings.cpp`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/src/saf_compaction.cpp`

**Interfaces:**
- Consumes: Existing `catalog_stuck_at`, `step_stuck_at`, and `run_stuck_at_dtc` signatures.
- Produces: stderr-only progress; native APIs and result structures remain unchanged.

- [ ] Add post-fault-list, catalog-copy, and dict-conversion markers to the binding.
- [ ] Add Primary start/done timing around `podem`.
- [ ] Add DTC start, sampled candidate, and done timing.
- [ ] Rebuild with `python PODEM/setup.py build_ext --inplace`.
- [ ] Run focused PODEM and fault-order tests.

### Task 3: Regression verification

**Files:**
- Modify only if a targeted test exposes a logging defect.

**Interfaces:**
- Consumes: The unchanged solver and trainer APIs.
- Produces: Evidence that the extension is rebuilt and relevant tests pass.

- [ ] Run the runtime-progress tests and existing fault-order test module.
- [ ] Run `git diff --check` and inspect only the files changed by this feature.
- [ ] Confirm `git status --short` contains no generated build artifacts.
