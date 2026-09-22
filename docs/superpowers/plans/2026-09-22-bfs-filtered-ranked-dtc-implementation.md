# BFS-Filtered Ranked-DTC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make stuck-at DTC discover candidates with the original unknown-PO reverse BFS, apply the TDF `select_fault_try` budget, let RL reorder only each returned candidate batch using the Primary forward pass scores, check the target PO after every attempted secondary fault, and replace per-candidate full-wire snapshots with accepted-PI-cube restoration.

**Architecture:** C++ owns Primary PODEM, canonical good-circuit state, BFS eligibility, candidate execution, PO termination, and final fault simulation through a phase-based session. Python owns protocol validation and supplies either identity BFS order or an RL ordering callback. The actor-critic runs once per Primary step; PPO stores and replays the Primary choice plus the executed prefix of each BFS batch from that one score tensor.

**Tech Stack:** C++14 ATPG core, pybind11, Python 3, PyTorch, NumPy, pytest, setuptools.

**Spec:** `docs/superpowers/specs/2026-09-22-bfs-filtered-ranked-dtc-design.md`

## Global Constraints

- Keep Primary backtrack limit `100`, DTC secondary backtrack limit `50`, Primary seed `14`, and STC seed `7` unchanged.
- Candidate eligibility is exclusively the C++ unknown-PO reverse BFS. Python and RL may reorder a returned batch but may not add, remove, or reuse a candidate.
- Use `select_fault_try = 15` when `ncktin <= 32`, otherwise `100`; count wires actually dequeued and expanded, and reset the budget for each unknown PO.
- Run the actor-critic exactly once for each Primary step during rollout and exactly once for each transition during PPO replay.
- After every secondary attempt, restore the accepted fault-free PI cube, re-establish canonical good-circuit implication, and then test whether the current target PO is still `U`.
- Remove the per-candidate `vector<Snapshot>` full-wire save/restore path. Persistent rollback state must be O(number of PIs); candidate-local logs may scale with actual search changes.
- Do not change reward, GAE, PPO hyperparameters, fault-simulation drop semantics, or STC behavior.
- Bump checkpoint and solver protocol identities; do not silently load schema-4/full-remaining-ranking checkpoints.
- Keep review scoped to the changed interfaces, action probability, candidate eligibility, and rollback invariants. Do not perform unrelated refactors or run the complete validation benchmark suite.

---

### Task 1: Introduce the native phase protocol and fixed solver identity

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/src/python_bindings.cpp`
- Modify: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Replace the public two-argument full-ranking entry point with:

```cpp
StuckAtPhaseResult begin_stuck_at_step(const string &fault_id);
StuckAtPhaseResult rank_stuck_at_dtc_candidates(
    const vector<string> &ranked_candidate_fault_ids);
AtpgStepResult step_stuck_at(const string &fault_id);
```

- Add a `StuckAtPhaseResult` envelope containing `phase`, the completed `AtpgStepResult` when `phase == "complete"`, and the active batch fields when `phase == "dtc"`: `selected_fault_id`, `unknown_po_id`, `dtc_candidate_fault_ids`, `dtc_batch_index`, `select_fault_try`, and `visited_wire_count`.
- Add explicit native phase state (`idle` or `awaiting_dtc_order`) and reject begin/finalize/continuation calls made in the wrong phase before mutating solver state.
- Extend `StuckAtProtocolConfig` and `protocol_to_dict()` with exact typed fields:

```text
dtc_bfs_small_input_threshold = 32
dtc_bfs_small_select_fault_try = 15
dtc_bfs_default_select_fault_try = 100
dtc_rollback_algorithm = "accepted_pi_cube_resim_v1"
```

**Steps:**

- [ ] In `PODEM/tests/test_fault_mapping.py`, add a small session test that calls `begin_step(primary)`, asserts that the result phase is either `dtc` or `complete`, and proves that a second `begin_step`, premature `result()`, or continuation in the wrong phase raises without changing `remaining_fault_ids()` or cumulative counters.
- [ ] Add a protocol-config assertion for the four BFS/rollback fields and their exact Python types.
- [ ] Run `python PODEM/setup.py build_ext --inplace` and `python -m pytest PODEM/tests/test_fault_mapping.py -q`; confirm the new phase/config tests fail because the APIs do not exist.
- [ ] Define the phase/result structs and session-owned state in `atpg.h`. Keep incomplete-step data private; do not expose raw pointers through pybind.
- [ ] Split `ATPG::step_stuck_at` in `atpg.cpp` so `begin_stuck_at_step` executes and accounts for exactly one Primary. FALSE/MAYBE and DTC-disabled cases finish immediately; a successful DTC-enabled Primary delegates candidate discovery to the next tasks.
- [ ] Make single-argument `step_stuck_at` the heuristic convenience loop: call `begin_stuck_at_step`, and while the phase is `dtc`, submit the returned BFS list unchanged to `rank_stuck_at_dtc_candidates`.
- [ ] In `python_bindings.cpp`, expose `StuckAtSession.begin_step(id)`, `rank_dtc_candidates(ids)`, and one-argument `step(id)`. Delete the production binding that accepts all remaining secondary IDs.
- [ ] Serialize phase envelopes with one helper so `complete` returns the existing step fields and `dtc` returns only immutable batch metadata plus cumulative progress fields needed for logging.
- [ ] Rebuild and rerun the focused native test module; confirm phase/config tests pass before continuing.
- [ ] Commit: `git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py && git commit -m "refactor: add phased stuck-at DTC session"`

---

### Task 2: Implement unknown-PO reverse BFS and `select_fault_try`

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/saf_compaction.cpp`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Add a private batch finder equivalent to:

```cpp
bool find_next_stuck_at_dtc_batch(DtcBatchState &batch);
```

- `DtcBatchState` retains the target PO pointer internally and publishes stable IDs and counters only.
- Candidate order must be `cktout` order, FIFO reverse BFS, gate fan-in order, then wire `udflist` order, with first occurrence winning deduplication.

**Steps:**

- [ ] Add focused netlist fixtures in `PODEM/tests/test_fault_mapping.py` that contain: two unknown POs; a reconvergent all-`U` cone; a known-valued fan-in branch; a fault outside the cone; and enough buffer wires to cross the 15-wire budget.
- [ ] Add assertions that only faults reached through `U` fan-ins appear, the outside/known-branch faults never appear, reconvergence produces no duplicate IDs, and repeated runs return the same BFS order.
- [ ] Add a small-input fixture proving `visited_wire_count <= 15`, candidates below the cutoff are absent, and the next PO gets a fresh budget. Add a generated 33-input fixture proving the reported budget is `100` without asserting wall time.
- [ ] Run the new tests and confirm they fail against the phase scaffold.
- [ ] Implement FIFO traversal with `std::queue<wptr>`, per-PO visited wires, per-batch candidate-ID deduplication, and per-Primary attempted-ID exclusion.
- [ ] At discovery time intersect `udflist` with the current selectable fault catalog, excluding Primary, `test_tried`, `REDUNDANT`, and any fault already attempted in this Primary DTC.
- [ ] Count only wires popped and expanded. Stop deeper expansion when the configured budget is reached, but retain candidates discovered while processing the last permitted wire.
- [ ] If a PO has no candidates, continue to the next stable `cktout` entry. If no PO yields candidates, complete vector fill and fault simulation exactly once.
- [ ] Ensure `run_stuck_at_ordered()` and the one-argument heuristic session use the batch list unchanged; ordered Primary input must never be interpreted as secondary order.
- [ ] Rebuild and run `python -m pytest PODEM/tests/test_fault_mapping.py -q`; confirm the eligibility, ordering, deduplication, and 15/100 budget tests pass.
- [ ] Commit: `git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/saf_compaction.cpp PODEM/tests/test_fault_mapping.py && git commit -m "feat: filter stuck-at DTC with bounded BFS"`

---

### Task 3: Execute ranked batches with per-fault PO checks and optimized rollback

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/saf_compaction.cpp`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Add private helpers with one responsibility each:

```cpp
void restore_stuck_at_good_cube(const vector<int> &accepted_pi_cube);
bool stuck_at_cube_detects(fptr fault);
StuckAtPhaseResult try_stuck_at_dtc_batch(
    const vector<string> &ranked_candidate_fault_ids);
```

- The active step stores `accepted_pi_cube`, ordered `preserved_faults`, per-Primary `attempted_fault_ids`, `embedded_fault_ids`, accumulated DTC counters, current target PO, and current candidate IDs.

**Steps:**

- [ ] Replace the old “ranked DTC follows supplied prefix” fixture with a batch-local test: obtain the returned candidates, submit their reverse order, and assert attempted IDs are a contiguous prefix of that reversed batch and never contain a non-candidate.
- [ ] Add a fixture where the first successful secondary makes the target PO known. Assert only one secondary is attempted even though the submitted permutation has a tail, and assert the next returned batch—if any—comes from a fresh BFS on the modified cube.
- [ ] Add a failed/MAYBE candidate fixture that records the observable canonical state before the attempt and verifies afterward: accepted PI values match, internal good-circuit values match, the target PO check sees no residual `D`/`D_bar`, the fault remains selectable, and cumulative changes are limited to DTC calls/backtracks.
- [ ] Add a preserved-fault fixture proving temporary injection checks return to the accepted good-circuit cube before the PO check and the next BFS.
- [ ] Add invalid permutation cases (missing, duplicate, extra, unknown, stale) and assert they fail before attempted IDs, counters, phase, or cube change.
- [ ] Run the focused native test module and confirm the new tests fail.
- [ ] Implement `restore_stuck_at_good_cube`: clear transient scheduled/changed/assignment state, restore good-value PIs from `accepted_pi_cube`, set internal wires to `U`, and run normal implication to a deterministic fault-free state.
- [ ] Refactor `stuck_at_podemx_secondary` so its decision stack includes only PIs that were `U` in the accepted cube. Clear its decision/backtrack state and propagate-tree marks on every exit path.
- [ ] Delete `struct Snapshot` and `vector<Snapshot>` from `run_stuck_at_dtc`; do not replace them with another all-wire saved copy.
- [ ] Before any attempt, validate that the submitted IDs are exactly one permutation of the exposed batch. Then execute in that order, appending each ID before search and updating calls/backtracks once.
- [ ] On failure/MAYBE, restore the accepted PI cube and re-implicate. On success, validate Primary and every previously embedded fault; after every validation injection, restore the accepted good cube. Commit a new accepted PI cube only after all preserved checks succeed.
- [ ] After every attempted candidate—TRUE, FALSE, or MAYBE—restore canonical good-circuit state, read the current target PO, and stop the submitted ranking immediately when it is no longer `U`.
- [ ] Mark attempted faults only in the current Primary-step set. Do not set global `test_tried`, `REDUNDANT`, or `MAYBE` for a secondary attempt.
- [ ] Rebuild and run `python -m pytest PODEM/tests/test_fault_mapping.py -q`; confirm rollback, preserved-fault, PO-stop, prefix, and invalid-order tests pass.
- [ ] Run `rg -n "Snapshot|vector<Snapshot>" PODEM/src/saf_compaction.cpp` and require no match.
- [ ] Commit: `git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/saf_compaction.cpp PODEM/tests/test_fault_mapping.py && git commit -m "feat: stop ranked DTC by PO and restore PI cubes"`

---

### Task 4: Validate and orchestrate batch phases in Python

**Files:**
- Modify: `fault_order_rl/environment.py`
- Modify: `tests/test_fault_order_rl.py`
- Modify: `tests/test_runtime_progress.py`

**Interfaces:**
- Change the Python wrapper to:

```python
PodemSession.step(fault_id, rank_dtc_candidates=None) -> dict
```

- `rank_dtc_candidates` is called as `ranker(candidate_fault_ids, batch_metadata)` and returns a complete permutation of that batch. `None` means identity/BFS order.
- Final result adds `dtc_batches`, each containing `unknown_po_id`, `bfs_candidate_fault_ids`, `requested_fault_ids`, `executed_prefix_fault_ids`, `embedded_fault_ids`, `select_fault_try`, and `visited_wire_count`.

**Steps:**

- [ ] Replace the fake native session in `tests/test_fault_order_rl.py` with a deterministic two-batch phase machine. Test the default identity ranker and a reverse ranker without involving the model.
- [ ] Test Python-side rejection of a ranker returning missing, duplicate, extra, unknown, or cross-batch IDs. Verify the wrapper does not call native continuation after validation fails.
- [ ] Test that native `dtc_attempted_fault_ids` can only grow by each batch's requested contiguous prefix, embedded IDs are an ordered subsequence, no secondary repeats across batches, and all attempted IDs belonged to their logged BFS candidate set.
- [ ] Update `tests/test_runtime_progress.py` fake sessions to the new one-argument/default-ranker interface while preserving existing sampled progress assertions.
- [ ] Run `python -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`; confirm the phase-oriented tests fail first.
- [ ] Add `PHASE_FIELDS` and batch validators to `environment.py`. Validate string IDs, exact permutations, phase transitions, stable Primary ID, monotonically accumulated native counters, and strict remaining-set reduction only after completion.
- [ ] Implement `PodemSession.step` as the begin/continue loop. Keep session accounting uncommitted until `phase == "complete"`; if a ranker raises, leave the native session awaiting the same batch so the caller can retry or terminate explicitly.
- [ ] Return immutable tuples for IDs and batch sequences so trainer traces cannot mutate protocol evidence after the step.
- [ ] Update `PROTOCOL_CONFIG` with the four new native fields and exact values/types from Task 1.
- [ ] Run both focused Python modules and confirm they pass.
- [ ] Commit: `git add fault_order_rl/environment.py tests/test_fault_order_rl.py tests/test_runtime_progress.py && git commit -m "refactor: orchestrate BFS DTC batches in Python"`

---

### Task 5: Split Primary selection from cached-score candidate ranking

**Files:**
- Modify: `fault_order_rl/policy.py`
- Modify: `fault_order_rl/__init__.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Replace full-remaining ranking helpers with:

```python
sample_primary(scores, remaining_rows, temperature, stochastic, generator=None)
sample_candidate_ranking(scores, remaining_rows, candidate_rows,
                         temperature, stochastic, generator=None)
joint_action_stats(model, embeddings, remaining_rows, primary_row,
                   dtc_batches, temperature)
```

- `scores[i]` corresponds to `remaining_rows[i]`. Candidate ranking must index this tensor and must not call the model.
- Each replay batch supplies `bfs_candidate_rows` and `executed_prefix_rows`; requested but unexecuted suffixes do not contribute to log probability or entropy.

**Steps:**

- [ ] Replace the existing full-ranking policy test with a hand-computed three-fault Primary categorical probability plus two DTC executed-prefix Plackett–Luce terms. Assert `joint_action_stats` matches the sum, returns one critic value, and averages entropy over all executed conditional choices.
- [ ] Add a counting actor-critic test proving `joint_action_stats` calls `forward` exactly once even with multiple DTC batches.
- [ ] Add candidate-order tests proving non-candidate high scores are ignored, stochastic output is exactly a candidate permutation, and deterministic equal-score ties use ascending catalog row.
- [ ] Add malformed replay tests for duplicate/missing candidate rows, executed rows outside a batch, non-prefix execution evidence, repeated secondary rows across batches, a Primary outside remaining rows, and non-finite scores/value/statistics.
- [ ] Run `python -m pytest tests/test_fault_order_rl.py -q`; confirm the new policy tests fail.
- [ ] Implement a shared conditional prefix-stat helper over explicit local indices. Use it for the Primary categorical term and each candidate-batch prefix without rebuilding features.
- [ ] Implement `sample_primary` as one categorical sample/argmax. Implement `sample_candidate_ranking` by mapping candidate catalog rows to their unique locations in `remaining_rows`, then applying Plackett–Luce sampling or stable score sort only within that subset.
- [ ] Implement `joint_action_stats` with one `build_dynamic_features` call and one model forward. Sum Primary and batch log probabilities; average entropy over the Primary choice and each actually executed secondary choice.
- [ ] Remove `sample_ranking` and `executed_prefix_stats` from production exports after all call sites are migrated; do not keep a compatibility path that can rank all remaining secondaries.
- [ ] Run the focused policy tests and confirm gradients remain finite for actor and critic parameters.
- [ ] Commit: `git add fault_order_rl/policy.py fault_order_rl/__init__.py tests/test_fault_order_rl.py && git commit -m "feat: score Primary and BFS DTC actions jointly"`

---

### Task 6: Use one cached forward in rollout and PPO replay

**Files:**
- Modify: `fault_order_rl/trainer.py`
- Modify: `tests/test_fault_order_rl.py`
- Modify: `tests/test_runtime_progress.py`

**Interfaces:**
- Each rollout decision stores:

```text
remaining_rows
primary_row
old_joint_log_probability
old_value
mean_entropy
dtc_batches[]:
  unknown_po_id
  bfs_candidate_rows
  requested_rows
  executed_prefix_rows
  embedded_rows
  select_fault_try
  visited_wire_count
```

- Native baseline uses `session.step(primary_id)` and never constructs model scores.
- RL rollout computes `scores, value` once, samples/selects the Primary, and closes over those scores in the batch ranker passed to `session.step`.

**Steps:**

- [ ] Add a counting-model rollout test with two native DTC batches and assert exactly one forward call for the whole Primary step. Assert both requested batch orders equal orders derived from the same captured score tensor.
- [ ] Add a PPO replay test asserting exactly one forward per stored transition, not one per DTC batch, and equality between rollout and replay joint log probability before an optimizer update.
- [ ] Update trace assertions to prove every requested row exactly covers its BFS candidate rows, every executed list is a requested prefix, embedded rows are an ordered executed subsequence, and no secondary row repeats within the Primary step.
- [ ] Add a deterministic evaluation test for equal Primary/candidate scores and catalog-row tie-breaking.
- [ ] Run `python -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`; confirm the trainer tests fail.
- [ ] In `_run_policy`, build dynamic features and call the model once. Select only `primary_row`; create an ID-to-row map and a ranker closure that calls `sample_candidate_ranking` on each native batch using the cached `scores`.
- [ ] Convert returned batch IDs back to rows, verify they all belong to the step-start `remaining_rows - {primary_row}`, and compute rollout statistics from the already available `scores`/`value` without another model forward.
- [ ] Replace decision fields `requested_rows`/`executed_rows` at step level with `primary_row` and nested `dtc_batches`. Update `_evaluation_export`, logging, and best-result metadata to read the new structure.
- [ ] In `_ppo_update`, call `joint_action_stats` once per transition and preserve the existing clipping, value loss, entropy coefficient, gradient clipping, GAE, and minibatch behavior.
- [ ] In `_run_native`, call the one-argument session path so heuristic DTC submits each BFS batch unchanged and does not instantiate/call the model.
- [ ] Run both focused test modules and confirm all rollout, replay, deterministic evaluation, and progress tests pass.
- [ ] Commit: `git add fault_order_rl/trainer.py tests/test_fault_order_rl.py tests/test_runtime_progress.py && git commit -m "feat: reuse Primary scores for ranked DTC"`

---

### Task 7: Version checkpoints and serialize nested DTC evidence

**Files:**
- Modify: `fault_order_rl/checkpoint.py`
- Modify: `fault_order_rl/trainer.py`
- Modify: `fault_order_rl/__main__.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Set checkpoint schema to `5`.
- Set `POLICY_IDENTITY = "dynamic_bfs_ranked_dtc_actor_critic_ppo_v2"`.
- Set `SOLVER_PROTOCOL["compression_algorithm_version"] = "stuck_at_podemx_bfs_ranked_dtc_v3"` and include all fixed BFS/rollback config fields.
- Flatten nested trajectory data into non-object NumPy arrays using two levels of offsets:

```text
primary_rows
dtc_batch_step_offsets
dtc_candidate_rows / dtc_candidate_offsets
dtc_requested_rows / dtc_requested_offsets
dtc_executed_rows / dtc_executed_offsets
dtc_embedded_rows / dtc_embedded_offsets
dtc_select_fault_try
dtc_visited_wire_count
```

**Steps:**

- [ ] Update checkpoint tests so schemas 1–4 each produce an explicit retraining error; schema 4 must mention the obsolete full-remaining Ranked-DTC action space.
- [ ] Add an NPZ round-trip test with zero, one, and multiple DTC batches across successive Primary steps. Load with `allow_pickle=False` and reconstruct the exact nested rows from step and batch offsets.
- [ ] Add resume validation tests asserting schema 5, policy identity, solver version, BFS thresholds/budgets, and rollback identity must all match before optimizer/RNG state is restored.
- [ ] Run `python -m pytest tests/test_fault_order_rl.py -q`; confirm new schema/serialization tests fail.
- [ ] Bump checkpoint creation and loading to schema 5, add the explicit schema-4 rejection, and update CLI/help text that names the checkpoint schema.
- [ ] Update `_write_circuit` to emit typed `int64` flat arrays and monotonic offsets. Do not use object arrays, pickled nested data, or omit empty batches.
- [ ] Update policy/solver identities and ensure saved `protocol` is compared as an exact mapping during resume and final-model loading.
- [ ] Run the focused test module and inspect one generated NPZ with `np.load(path, allow_pickle=False)`.
- [ ] Commit: `git add fault_order_rl/checkpoint.py fault_order_rl/trainer.py fault_order_rl/__main__.py tests/test_fault_order_rl.py && git commit -m "feat: version BFS-ranked DTC trajectories"`

---

### Task 8: Update documentation and perform only required integration checks

**Files:**
- Modify: `docs/fault-order-rl.md`
- Modify: `docs/fault-reorder-algorithm.md`
- Modify only if a focused defect is found: files changed in Tasks 1–7

**Interfaces:**
- Documentation must distinguish Primary selection over all selectable faults from secondary ordering over each BFS-filtered batch.
- Documentation must state that each secondary attempt is followed by canonical good-circuit restoration and a target-PO `U` check.

**Steps:**

- [ ] Update both docs with the phased API, 15/100 `select_fault_try` rule, per-candidate PO stop rule, cached-score rule, joint probability, schema 5, and accepted-PI-cube rollback. Remove claims that DTC receives all remaining faults.
- [ ] Rebuild the extension: `python PODEM/setup.py build_ext --inplace`.
- [ ] Run native/binding regression: `python -m pytest PODEM/tests/test_fault_mapping.py -q`.
- [ ] Run policy/trainer/progress regression: `python -m pytest tests/test_fault_order_rl.py tests/test_runtime_progress.py -q`.
- [ ] Run one small real integration using `validation/bench/b12.ckt` with DTC enabled and STC disabled. Record candidate count, attempted prefix count, DTC calls, patterns, coverage, and backtracks; verify every attempted ID belongs to its logged BFS batch and no batch contains a cone-external fault. Do not assert wall-clock duration and do not run the full validation suite.
- [ ] Compare that one circuit against DTC-disabled execution only for accounting sanity: both sessions terminate, counters are internally consistent, and DTC does not fabricate detected IDs. Do not require identical pattern count or coverage trajectory.
- [ ] Run `git diff --check` and `git status --short`. Inspect only files changed by this feature and remove generated extension/build artifacts from the commit if they are untracked.
- [ ] Confirm no obsolete production call remains: `rg -n "sample_ranking|executed_prefix_stats|ranked_secondary_fault_ids|vector<Snapshot>" PODEM/src fault_order_rl tests` may match only deliberate migration-error tests or historical docs outside the changed runtime paths.
- [ ] Commit: `git add docs/fault-order-rl.md docs/fault-reorder-algorithm.md && git commit -m "docs: describe BFS-filtered ranked DTC"`

## Completion Criteria

- Unknown-PO reverse BFS is the only source of secondary candidates, honors stable ordering and the 15/100 per-PO wire budget, and excludes all faults outside the current all-`U` cone.
- Heuristic execution preserves BFS order; RL changes only the order inside each returned batch using scores cached from the single Primary forward pass.
- After each attempted secondary, canonical good-circuit state is restored and the target PO is checked; a known PO truncates the requested ranking to the recorded executed prefix.
- Failed and rejected candidates leave no solver state behind except documented call/backtrack metrics, and no per-candidate full-wire snapshot remains.
- PPO rollout/replay probabilities cover the Primary plus only actual DTC executed prefixes and use one actor-critic forward per transition.
- Schema-4 checkpoints are rejected, schema-5 nested trajectory evidence round-trips without pickle, and solver identity fixes the BFS and rollback protocol.
- The three focused test modules and the one b12 DTC-on/STC-off integration pass; no unrelated full-suite or broad review is required.
