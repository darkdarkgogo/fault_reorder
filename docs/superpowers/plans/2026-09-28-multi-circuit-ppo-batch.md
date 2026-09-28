# Multi-Circuit PPO Batch Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train PPO from rollout batches containing up to four circuits, with transition-equal minibatches of at most 128 samples and a fresh shuffle in every PPO epoch.

**Architecture:** Preserve per-circuit environment execution and trajectory finalization, but make a rollout batch the transaction boundary. Flatten finalized transitions into `(circuit, decision)` samples only inside PPO replay, normalize their advantages once across the rollout batch, and use a fresh `torch.randperm` per epoch before stepping the optimizer on consecutive minibatches.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, the existing atomic checkpoint helpers.

**Spec:** `docs/superpowers/specs/2026-09-28-multi-circuit-ppo-batch-design.md`

## Global Constraints

- `circuit_batch_size = 4` and `ppo_minibatch_size = 128` by default.
- Every transition has equal weight; do not add per-circuit weighting.
- Keep the final smaller transition minibatch without repartitioning it.
- Do not add KL early stopping or other update guards.
- Keep reward, terminal correction, GAE, model, solver, validation, DTC/STC, and wire-budget behavior unchanged.
- Collect every circuit in a rollout batch before the first optimizer step.
- Re-shuffle all transitions independently in every PPO epoch.
- Old checkpoints remain evaluable but cannot resume the new training protocol.

---

### Task 1: Batch configuration and training-protocol compatibility

**Files:**
- Modify: `fault_order_rl/trainer.py:29-75`
- Modify: `tests/test_fault_order_rl.py:85-105`

**Interfaces:**
- Produces: `TRAINING_PROTOCOL = "multi_circuit_ppo_batch_v1"`.
- Produces: `TrainConfig.circuit_batch_size: int = 4`.
- Produces: `TrainConfig.ppo_minibatch_size: int = 128`.
- Consumes: existing `Trainer._payload()` and `Trainer._check_compatibility()` checkpoint paths.

- [ ] **Step 1: Write failing configuration and compatibility tests**

Add assertions that defaults are 4 and 128, that zero/bool values are rejected,
that new payloads contain `training_protocol`, and that a saved latest checkpoint
without the matching identity raises `checkpoint training protocol changed` on
resume while remaining loadable for evaluation.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_fault_order_rl.py -k "fixed_protocol or training_protocol" -v`

Expected: FAIL because the new fields and identity do not exist.

- [ ] **Step 3: Implement the configuration and identity**

Add the two integer fields, validate them with the existing positive-integer
rules, add the protocol identity to checkpoint payloads, and check it only from
`Trainer._check_compatibility()` so standalone evaluation remains compatible
with old policy checkpoints.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_fault_order_rl.py -k "fixed_protocol or training_protocol" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: configure multi-circuit PPO batches"
```

---

### Task 2: Transition minibatch PPO replay

**Files:**
- Modify: `fault_order_rl/trainer.py:552-598`
- Modify: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: `rollouts`, a sequence of dictionaries containing `circuit` and finalized `decisions`.
- Produces: `Trainer._ppo_update(model, optimizer, rollouts) -> dict`.
- Produces update summary keys: `circuits`, `circuit_count`, `transition_count`, `minibatch_count`, `optimizer_steps`, and `epochs`.
- Each `epochs` entry contains `epoch` and `minibatches`; every minibatch record contains `minibatch`, `sample_count`, `loss`, `gradient_norm`, `actor_loss`, `critic_loss`, `entropy`, `approx_kl`, and `clip_fraction`.

- [ ] **Step 1: Write failing replay tests**

Create small synthetic circuits and decisions, monkeypatch
`joint_action_stats`/`ppo_objective` where necessary, and assert:

```python
summary = trainer._ppo_update(model, optimizer, rollouts)
assert summary["transition_count"] == 350
assert [item["sample_count"] for item in summary["epochs"][0]["minibatches"]] == [128, 128, 94]
assert summary["minibatch_count"] == 3
assert summary["optimizer_steps"] == 12
```

Also capture normalized advantages to prove normalization occurs over the
combined transitions, record circuit identities during replay to prove each
sample uses its owning embeddings, and replace `torch.randperm` with a recorder
to prove it is called once per epoch.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_fault_order_rl.py -k "rollout_batch_ppo or transition_minibatch" -v`

Expected: FAIL because `_ppo_update` still accepts one circuit and performs one
full-trajectory optimizer step per epoch.

- [ ] **Step 3: Implement flattened transition replay**

Build an immutable flattened list of `(circuit, decision)` pairs. Create tensors
for old log probabilities, returns, and one globally normalized advantage
vector. In every epoch, call `torch.randperm(transition_count)`, slice the result
in `ppo_minibatch_size` chunks, replay only those indices with the matching
circuit embeddings, and perform the existing finite checks, gradient clipping,
and optimizer step for each chunk.

- [ ] **Step 4: Return complete observational metrics**

Record every minibatch's sample count and existing PPO metrics. Derive
`minibatch_count = ceil(transition_count / ppo_minibatch_size)` and
`optimizer_steps = ppo_epochs * minibatch_count`; do not use either value to
change training.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_fault_order_rl.py -k "rollout_batch_ppo or transition_minibatch or reward_terminal" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: optimize PPO in transition minibatches"
```

---

### Task 3: Rollout-batch transaction, records, and resume

**Files:**
- Modify: `fault_order_rl/trainer.py:638-708`
- Modify: `tests/test_fault_order_rl.py:460-520`

**Interfaces:**
- Consumes: `TrainConfig.circuit_batch_size` and the Task 2 `_ppo_update()` summary.
- Produces: rollout-batch-aligned `Trainer.step()` behavior.
- Preserves: episode records returned from `step()` and per-circuit JSON/NPZ artifacts.

- [ ] **Step 1: Replace the per-circuit transaction test with failing batch tests**

Use at least five fake training circuits. Assert that failure while collecting
the second circuit leaves `next_circuit_index == 0`, model/optimizer/checkpoint
unchanged, and no committed episode artifact for the batch. After retry, assert
that the first checkpoint boundary is index 4 and the final partial batch moves
the index through circuit 5 before validation completes the round.

- [ ] **Step 2: Add a no-update-during-collection test**

Monkeypatch `_run_policy` to snapshot candidate model parameters on all four
calls and monkeypatch `_ppo_update` to count entry. Assert all four snapshots
are identical and `_ppo_update` is entered only after the fourth rollout.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `python -m pytest tests/test_fault_order_rl.py -k "batch_checkpoint or batch_collection or final_partial" -v`

Expected: FAIL because `step()` currently updates and checkpoints after each
circuit.

- [ ] **Step 4: Implement the rollout-batch transaction**

Iterate from `next_circuit_index` in slices of at most
`circuit_batch_size`. Capture RNG once before each slice, create one candidate
model and optimizer, collect and finalize all circuits against that unchanged
candidate, call `_ppo_update` once, then create episode records, write all
artifacts, save one latest checkpoint with the slice end index, and finally
publish candidate state to the trainer. On any exception, restore the
pre-batch RNG and leave trainer state/index unchanged.

- [ ] **Step 5: Attach shared batch diagnostics to episode records**

Extend `_record` with rollout-batch index, circuit list, circuit count,
transition count, minibatch count, optimizer steps, and the returned epoch
details. Emit one `PPO` progress line per successful rollout batch with these
counts.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_fault_order_rl.py -k "batch_checkpoint or batch_collection or final_partial or nested_dtc" -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: make rollout batches transactional"
```

---

### Task 4: User documentation and full verification

**Files:**
- Modify: `docs/fault-order-rl.md`
- Modify: `docs/fault-reorder-algorithm.md`
- Modify: `README.md` if its training summary states per-circuit updates.

**Interfaces:**
- Documents the exact defaults and transaction/replay behavior implemented by Tasks 1-3.

- [ ] **Step 1: Update training documentation**

Replace statements that PPO updates immediately after each circuit with the
four-circuit rollout-batch flow. Document per-circuit GAE, combined advantage
normalization, transition-equal weighting, per-epoch shuffle, 128-transition
minibatches, retained tail minibatches, and final partial circuit batches.

- [ ] **Step 2: Run all Python tests**

Run: `python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 3: Run repository checks**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 4: Commit**

```bash
git add docs/fault-order-rl.md docs/fault-reorder-algorithm.md README.md
git commit -m "docs: explain multi-circuit PPO training"
```

- [ ] **Step 5: Request code review**

Review all commits from the design baseline through `HEAD` against the spec,
fix every Critical or Important finding, rerun focused tests for changed code,
then rerun `python -m pytest tests -v` and `git diff --check`.
