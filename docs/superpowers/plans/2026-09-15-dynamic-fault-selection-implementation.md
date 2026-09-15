# Dynamic Fault Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed full-permutation fault policy with a stateful policy that recomputes scores from each fault embedding, the current remaining-fault mean, and the remaining ratio after every PODEM/fault-simulation step.

**Architecture:** A stateful C++ PODEM session exposes one-primary-fault steps and the current selectable fault IDs. Python builds 515-dimensional dynamic features and samples one masked categorical action per state; after the solver episode, it replays recorded masks under autograd and applies the existing coverage-gated REINFORCE reward, normalized by the circuit's initial fault count.

**Tech Stack:** C++11/14, pybind11, Python 3.9+, PyTorch, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-dynamic-fault-selection-design.md`

## Global Constraints

- Keep every DeepGate2 fault embedding frozen at exactly 257 dimensions.
- Scorer input is exactly `[fault embedding (257), remaining mean (257), remaining ratio (1)]`, or 515 dimensions.
- Recompute context and scores after every TRUE, FALSE, or MAYBE primary-fault attempt.
- A selectable fault is undetected, not redundant, and not previously attempted in the current one-attempt episode.
- Keep PODEM seed 14, backtrack limit 5000, one attempt, and STC/DTC/SCOAP/TDF disabled.
- Keep the existing pattern-count reward, native resolved-coverage guard, EMA advantage, Adam optimizer, temperature schedule, minibatch transaction, and best-selection key.
- Normalize trajectory log probability by the initial catalog fault count, never by the realized decision count.
- Do not load schema-2 static checkpoints into the dynamic model; require a fresh schema-3 run.
- Preserve existing user changes in `README.md` and `docs/fault-reorder-algorithm.md` unless the documentation task explicitly updates overlapping text.

---

### Task 1: Incremental PODEM core and pybind session

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/atpg.cpp`
- Modify: `PODEM/src/python_bindings.cpp`
- Test: `PODEM/tests/test_fault_mapping.py`

**Interfaces:**
- Consumes: the existing `ATPG::podem`, `ATPG::fault_sim_a_vector`, fault catalog, and ordered stuck-at configuration.
- Produces: `ATPG::get_selectable_fault_ids() const -> vector<string>`, `ATPG::step_stuck_at(const string&) -> AtpgStepResult`, `ATPG::get_stuck_at_result() const -> AtpgRunResult`, and Python `cpp_podem.StuckAtSession` with `catalog()`, `remaining_fault_ids()`, `step(fault_id)`, and `result()`.

- [ ] **Step 1: Write failing binding tests for an incremental session**

Add this test beside the existing ordered-ATPG tests:

```python
def test_incremental_session_matches_native_complete_run(self):
    _, binary, fault_map, _, catalog = self.convert(
        "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\ny = AND(a,b,c)\n"
    )
    ids = [str(item["fault_id"]) for item in catalog["faults"]]
    expected = cpp_podem.run_stuck_at_ordered(
        str(binary), str(fault_map), ids, 5000, 14
    )
    session = cpp_podem.StuckAtSession(
        str(binary), str(fault_map), 5000, 14
    )
    seen = []
    while session.remaining_fault_ids():
        selected = session.remaining_fault_ids()[0]
        before = set(session.remaining_fault_ids())
        step = session.step(selected)
        seen.append(selected)
        self.assertEqual(step["selected_fault_id"], selected)
        self.assertLess(set(step["remaining_fault_ids"]), before)
    self.assertTrue(seen)
    self.assertEqual(session.result(), expected)
```

Add rejection coverage:

```python
def test_incremental_session_rejects_non_selectable_fault(self):
    _, binary, fault_map, _, catalog = self.convert(
        "INPUT(a)\nOUTPUT(y)\ny = BUF(a)\n"
    )
    fault_id = str(catalog["faults"][0]["fault_id"])
    session = cpp_podem.StuckAtSession(str(binary), str(fault_map), 5000, 14)
    session.step(fault_id)
    with self.assertRaisesRegex(RuntimeError, "not selectable"):
        session.step(fault_id)
    with self.assertRaisesRegex(RuntimeError, "Unknown fault ID"):
        session.step("missing:GO:sa0")
```

- [ ] **Step 2: Run the new tests and verify the class is missing**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -k incremental -v
```

Expected: FAIL because `cpp_podem.StuckAtSession` does not exist.

- [ ] **Step 3: Add cumulative step state and refactor the complete run**

In `atpg.h`, add an `AtpgStepResult` containing the selected ID, `target_status`, `generated_pattern`, `newly_detected_fault_ids`, `remaining_fault_ids`, and cumulative `AtpgRunResult`. Add private cumulative counters initialized by `configure_ordered_stuck_at`.

Implement the following behavior in `atpg.cpp`:

```cpp
vector<string> ATPG::get_selectable_fault_ids() const {
    vector<string> result;
    for (fptr fault : flist_undetect) {
        if (!fault->test_tried && fault->detect != REDUNDANT)
            result.push_back(fault_identifier(fault));
    }
    return result;
}
```

`step_stuck_at` must locate an exact selectable ID, call `podem` once, run `fault_sim_a_vector` only on TRUE, mark the target tried, update cumulative counters, determine dropped IDs by comparing `flist_undetect` before and after fault simulation, and return the new selectable list. Seed `rand()` and validate unsupported physical XOR/EQV once at the first step. Refactor `run_stuck_at` to repeatedly call `step_stuck_at(get_selectable_fault_ids().front())`; this makes the old full-run API and the new session share one solver path.

- [ ] **Step 4: Bind the stateful session**

Add a small owning wrapper in `python_bindings.cpp`:

```cpp
class StuckAtSession {
public:
    StuckAtSession(const string &circuit, const string &faultmap,
                   int backtrack_limit, int seed);
    py::dict catalog() const;
    vector<string> remaining_fault_ids() const;
    py::dict step(const string &fault_id);
    py::dict result() const;
private:
    ATPG atpg_;
    vector<ATPG::FaultCatalogEntry> catalog_;
};
```

Release the GIL during circuit initialization and each `step`, but construct Python dictionaries only while holding the GIL. Reuse one `summary_to_dict` helper for legacy and session results.

- [ ] **Step 5: Build and run C++/binding tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe PODEM\setup.py build_ext --inplace
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest PODEM/tests/test_fault_mapping.py -v
```

Expected: all fault-mapping, full ordered-run, and incremental-session tests PASS; the session native-first result exactly equals the legacy native permutation result.

- [ ] **Step 6: Commit the incremental solver boundary**

```powershell
git add PODEM/src/atpg.h PODEM/src/atpg.cpp PODEM/src/python_bindings.cpp PODEM/tests/test_fault_mapping.py
git commit -m "feat: expose incremental PODEM fault sessions"
```

---

### Task 2: Dynamic feature construction and categorical policy math

**Files:**
- Modify: `fault_order_rl/model.py`
- Modify: `fault_order_rl/policy.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: validated `[N,257]` frozen embeddings and catalog-row indices.
- Produces: `build_dynamic_features(embeddings, remaining_rows) -> Tensor[K,515]`, `sample_action(scores, temperature, generator=None) -> (local_index, logits)`, `deterministic_action(scores) -> local_index`, and `trajectory_log_prob(model, embeddings, decisions, temperature) -> scalar Tensor`.

- [ ] **Step 1: Write failing tests for the 515-dimensional input**

Add tests that use hand-checkable tensors:

```python
def test_dynamic_features_append_remaining_mean_and_ratio():
    embeddings = torch.arange(4 * 257, dtype=torch.float32).reshape(4, 257)
    rows = torch.tensor([1, 3])
    features = build_dynamic_features(embeddings, rows)
    assert features.shape == (2, 515)
    assert torch.equal(features[:, :257], embeddings[rows])
    expected_mean = embeddings[rows].mean(dim=0)
    assert torch.equal(features[:, 257:514], expected_mean.expand(2, -1))
    assert torch.equal(features[:, 514], torch.full((2,), 0.5))
```

Also assert that empty, duplicate, out-of-range, and non-one-dimensional row tensors raise `ValueError`, and that `FaultScorer()` rejects `[N,257]` while accepting `[N,515]`.

- [ ] **Step 2: Run the feature tests and verify failure**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_features or scorer" -v
```

Expected: FAIL because the feature builder is absent and the scorer still expects 257 dimensions.

- [ ] **Step 3: Implement the dynamic scorer input**

Change `FaultScorer`'s default input dimension to 515 while retaining the current nonlinear topology:

```text
LayerNorm(515) -> Linear(515,256) -> ReLU
-> Linear(256,128) -> ReLU -> Linear(128,1)
```

Implement `build_dynamic_features` without modifying or cloning the source embedding matrix. Compute ratio as `K / N` and expand the shared mean and scalar across the `K` candidate rows.

- [ ] **Step 4: Write failing masked categorical and replay tests**

Add a fixed-seed sampling test and a direct-formula replay test:

```python
def test_dynamic_trajectory_log_prob_matches_direct_steps():
    embeddings = torch.randn(4, 257, generator=torch.Generator().manual_seed(9))
    model = FaultScorer()
    decisions = [
        {"remaining_rows": (0, 1, 2, 3), "selected_row": 2},
        {"remaining_rows": (1, 3), "selected_row": 3},
    ]
    actual = trajectory_log_prob(model, embeddings, decisions, 0.7)
    direct = []
    for state in decisions:
        rows = torch.tensor(state["remaining_rows"])
        logits = centered_logits(model(build_dynamic_features(embeddings, rows)), 0.7)
        local = state["remaining_rows"].index(state["selected_row"])
        direct.append(torch.log_softmax(logits, 0)[local])
    assert torch.allclose(actual, torch.stack(direct).sum())
```

- [ ] **Step 5: Implement categorical action helpers and trajectory replay**

Use `torch.multinomial(torch.softmax(logits, 0), 1, generator=generator)` for training selection. Use stable first-maximum selection for evaluation. In `trajectory_log_prob`, verify that every selected row occurs exactly once in its saved remaining rows, rebuild dynamic features for each state, and sum the selected `log_softmax` terms.

- [ ] **Step 6: Run policy tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_features or dynamic_trajectory or categorical or scorer" -v
```

Expected: all selected tests PASS with finite gradients through the scorer.

- [ ] **Step 7: Commit model and policy math**

```powershell
git add fault_order_rl/model.py fault_order_rl/policy.py tests/test_fault_order_rl.py
git commit -m "feat: add dynamic fault context policy"
```

---

### Task 3: Validated Python session environment

**Files:**
- Modify: `fault_order_rl/environment.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: `cpp_podem.StuckAtSession` from Task 1.
- Produces: `PodemEnvironment.start_session(bench_path, faultmap_path) -> PodemSession`; `PodemSession.remaining_fault_ids() -> tuple[str,...]`, `step(fault_id) -> dict`, and `result() -> dict`.

- [ ] **Step 1: Write failing wrapper validation tests**

Create a fake binding session whose first step changes `("f0","f1","f2")` to `("f2",)` and reports `newly_detected_fault_ids=("f0","f1")`. Assert that `PodemEnvironment.start_session` forwards paths, limit 5000, and seed 14; assert that the wrapper rejects duplicate/unknown remaining IDs, a remaining set that grows, a TRUE step without a pattern increment, and a final result missing any `RESULT_FIELDS` key.

- [ ] **Step 2: Run wrapper tests and verify failure**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "session_wrapper" -v
```

Expected: FAIL because `start_session` and `PodemSession` do not exist.

- [ ] **Step 3: Implement the wrapper and share result validation**

Extract the current final-result checks into `_validate_result(raw, catalog_size)`. `PodemSession` stores the immutable initial IDs and previous remaining tuple. On every step it verifies:

```text
selected ID was in the previous remaining set
new remaining IDs are unique, known, and a strict subset
newly detected IDs are known and absent from the new remaining set
generated_pattern changes cumulative pattern_count by exactly one
non-pattern steps do not change cumulative pattern_count
calls increase by one and backtracks never decrease
```

Keep `PodemEnvironment.run` supported and route its aggregate dictionary through the same `_validate_result` helper.

- [ ] **Step 4: Run environment tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "environment or session_wrapper" -v
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the Python environment boundary**

```powershell
git add fault_order_rl/environment.py tests/test_fault_order_rl.py
git commit -m "feat: validate incremental PODEM sessions"
```

---

### Task 4: Dynamic trainer episodes and fixed initial-count normalization

**Files:**
- Modify: `fault_order_rl/trainer.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: Tasks 2-3 policy helpers and `PodemEnvironment.start_session`.
- Produces: `Trainer._run_native(circuit)`, `Trainer._run_policy(circuit, model, temperature, stochastic)`, saved per-step decision states, and dynamic REINFORCE updates.

- [ ] **Step 1: Replace the fake complete-run environment with a fake session**

In the trainer fixture, make `FakeEnvironment.start_session` return a deterministic session over the test circuit's IDs. Its first selected even-numbered fault must also drop the next selectable fault, so a test can prove that the second decision's remaining set differs by more than the selected target alone. Keep complete aggregate metrics deterministic from the selected-row trajectory.

- [ ] **Step 2: Write failing dynamic-episode tests**

Assert that:

```python
metrics, elapsed, decisions, trace = trainer._run_policy(
    circuit, trainer.model, temperature=1.0, stochastic=False
)
assert decisions[0]["remaining_rows"] == tuple(range(circuit.fault_count))
assert len(decisions[1]["remaining_rows"]) < circuit.fault_count - 1
assert trace[1]["remaining_ratio"] == (
    len(decisions[1]["remaining_rows"]) / circuit.fault_count
)
```

Add a loss test that constructs three decisions for a five-fault circuit and checks the exact denominator is `5`, not `3`.

- [ ] **Step 3: Run trainer tests and verify failure**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "dynamic_episode or initial_fault_normalization" -v
```

Expected: FAIL because the trainer still requests one complete permutation.

- [ ] **Step 4: Implement native and policy session runners**

`_run_native` repeatedly selects the first ID returned by the session. `_run_policy` maps session IDs back to immutable catalog rows, rebuilds `[K,515]` features on every iteration, samples or chooses one local candidate, performs exactly one session step, and records:

```python
{"remaining_rows": tuple(rows), "selected_row": selected_row}
```

The human trace separately records selected ID/score, remaining count/ratio, target status, generated-pattern flag, newly detected IDs, and cumulative metrics. Reject unknown, duplicate, reordered-to-unknown, or nonshrinking session states.

- [ ] **Step 5: Replace static round collection and gradient calculation**

Within each batch, collect all circuit episodes using one unchanged candidate-model snapshot under `torch.no_grad()`. Then call `trajectory_log_prob` for each saved decision list and compute exactly:

```python
loss = (
    -record["advantage"]
    * log_prob
    / circuit.fault_count
    / len(batch)
)
```

Do not update model parameters between collection and replay for any episode in the batch. Keep transactional exception handling and RNG restoration unchanged.

- [ ] **Step 6: Encode compact round trajectories**

For each circuit save three numeric arrays in the round NPZ: `<name>__selected_rows`, `<name>__remaining_offsets`, and `<name>__remaining_rows`. Offsets start at zero and terminate at the flattened remaining-row array length, allowing every decision state to be reconstructed without pickle.

- [ ] **Step 7: Run trainer recovery and minibatch tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "shared_update or minibatch or failed_round or dynamic_episode or initial_fault_normalization" -v
```

Expected: continuous and resumed runs produce byte-identical selected-row and remaining-row arrays, equal model parameters, and equal baselines.

- [ ] **Step 8: Commit dynamic trainer behavior**

```powershell
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: train on dynamic fault trajectories"
```

---

### Task 5: Dynamic deterministic evaluation and schema-3 checkpoints

**Files:**
- Modify: `fault_order_rl/checkpoint.py`
- Modify: `fault_order_rl/trainer.py`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: deterministic `_run_policy(..., stochastic=False)` and dynamic traces from Task 4.
- Produces: schema-3 checkpoints, initial-ranking diagnostic NPZ files, and `<circuit>.trajectory.jsonl` execution traces.

- [ ] **Step 1: Write failing checkpoint migration tests**

Save minimal version-1 and version-2 dictionaries and assert distinct errors:

```python
with pytest.raises(ValueError, match="detected-only coverage"):
    load_checkpoint(schema1)
with pytest.raises(ValueError, match="static 257-dimensional policy"):
    load_checkpoint(schema2)
```

Assert a schema-3 payload contains `policy_version == "dynamic_remaining_mean_v1"` and `input_dimension == 515`.

- [ ] **Step 2: Upgrade checkpoint validation and payloads**

Make `load_checkpoint` accept only version 3. Preserve the schema-1 message and add an explicit schema-2 restart message. Add the two dynamic-policy identity fields to latest and derived-best payloads, and check them during resume/evaluation compatibility.

- [ ] **Step 3: Write failing dynamic evaluation artifact tests**

For every evaluated circuit, assert the NPZ contains:

```text
fault_ids, scores, ranks, permutation,
selected_rows, selected_scores, selected_steps,
exit_steps, exit_reasons
```

Treat `scores/ranks/permutation` as initial-state diagnostics. Assert `selected_steps == -1` for a fault dropped by another selected target and assert the JSONL trace identifies the selecting target and the dropped ID at the same step.

- [ ] **Step 4: Implement dynamic evaluation and artifacts**

Replace all static `model(circuit.embeddings)` plus complete-order runs with deterministic dynamic sessions. Build the initial diagnostic ranking from the first dynamic state only. Derive selected and exit arrays from the step trace, using Unicode strings for exit reasons so `np.load(..., allow_pickle=False)` remains valid. Write JSONL atomically next to each ranking NPZ.

Update external-manifest resumable evaluation so a circuit is complete only when metrics JSON, ranking NPZ, and trajectory JSONL all exist with the expected checkpoint/evaluation identity.

- [ ] **Step 5: Run checkpoint and evaluation tests**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "checkpoint or evaluation or latest or best" -v
```

Expected: all selected tests PASS; fresh deterministic re-evaluation matches the saved best metrics exactly.

- [ ] **Step 6: Commit evaluation and checkpoint migration**

```powershell
git add fault_order_rl/checkpoint.py fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: evaluate dynamic fault trajectories"
```

---

### Task 6: Real-solver integration, documentation, and full verification

**Files:**
- Modify: `tests/test_fault_order_rl.py`
- Modify: `docs/fault-order-rl.md`

**Interfaces:**
- Consumes: completed schema-3 dynamic policy and rebuilt `cpp_podem` extension.
- Produces: verified real-circuit dynamic training/evaluation behavior and current user documentation.

- [ ] **Step 1: Add a real dynamic smoke assertion**

Extend the existing real PODEM smoke test to run a one-round dynamic trainer on its tiny converted circuit. Assert that the round NPZ contains selected rows and remaining offsets, that at least one successive remaining set shrinks, and that deterministic evaluation writes both ranking and trajectory artifacts while preserving native resolved coverage.

- [ ] **Step 2: Run the focused real-solver smoke test**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k real_podem -v
```

Expected: PASS with a real incremental C++ session and a finite optimizer update.

- [ ] **Step 3: Update algorithm and operating documentation**

Document the 515-dimensional feature layout, eligible remaining-mask semantics, per-step deterministic and sampled selection, trajectory log probability, initial-fault-count normalization, stateful PODEM boundary, schema-3 restart requirement, and the distinction between initial ranking and actual selected trajectory. Remove statements in this tracked operating guide that say the model generates one complete permutation per episode. Leave the user's existing `README.md` edit and untracked `docs/fault-reorder-algorithm.md` unchanged.

- [ ] **Step 4: Run formatting and placeholder checks**

Run:

```powershell
git diff --check
rg -n "complete permutation" docs/fault-order-rl.md fault_order_rl PODEM/src
```

Expected: `git diff --check` is clean; any remaining `complete permutation` occurrence refers only to legacy behavior or migration context.

- [ ] **Step 5: Run the complete relevant test suite**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py PODEM/tests/test_fault_mapping.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Validate one production manifest without training**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -m fault_order_rl validate --manifest configs/anchor_smoke_train.json
```

Expected: manifest, catalog, embedding dimensions, IDs, equivalent counts, and provenance validate successfully.

- [ ] **Step 7: Review the complete change**

Invoke the `requesting-code-review` skill. Resolve every correctness finding, rerun the smallest affected test followed by the complete relevant suite, and verify `git status --short` contains no build products or unrelated staged changes.

- [ ] **Step 8: Commit verified documentation and integration tests**

```powershell
git add tests/test_fault_order_rl.py docs/fault-order-rl.md
git commit -m "docs: describe dynamic fault selection"
```

- [ ] **Step 9: Report verification evidence**

Report the exact passing test commands, dynamic smoke circuit, checkpoint incompatibility, remaining known limitations, and the paths of the design and implementation plan. Do not claim pattern-count improvement from a smoke run; improvement requires a real training experiment.
