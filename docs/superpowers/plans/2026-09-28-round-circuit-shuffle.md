# Per-Round Circuit Shuffle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shuffle all training circuits independently at the start of every round while preserving deterministic checkpoint resume and stable artifact identities.

**Architecture:** Add a pure helper that derives a permutation of manifest indices from the fixed training seed and one-based round number using an isolated `random.Random`. Treat `next_circuit_index` as an offset in that permutation, while keeping episode `circuit_index` and artifact filenames tied to the original manifest index.

**Tech Stack:** Python 3 standard library `random`, PyTorch trainer, pytest, existing checkpoint helpers.

**Spec:** `docs/superpowers/specs/2026-09-28-multi-circuit-ppo-batch-design.md`

## Global Constraints

- Shuffle the complete training circuit set once per round before forming 4-circuit rollout batches.
- Derive the order only from `TrainConfig.seed` and the one-based round number.
- Do not consume or mutate Python's global RNG when deriving the order.
- Resume must reconstruct the same active-round order from checkpoint config and round.
- `next_circuit_index` is the completed-position offset in the shuffled round order.
- Episode records and artifact filenames keep the original manifest index.
- Preserve PPO, GAE, transition weighting, minibatching, validation, and solver behavior.
- Reject resume from the earlier unshuffled multi-circuit training protocol while keeping those checkpoints evaluable.

---

### Task 1: Deterministic round-order helper and protocol identity

**Files:**
- Modify: `fault_order_rl/trainer.py:30,313-345,720-815`
- Modify: `tests/test_fault_order_rl.py:90-115,520-700`

**Interfaces:**
- Produces: `TRAINING_PROTOCOL = "shuffled_multi_circuit_ppo_batch_v1"`.
- Produces: `Trainer._round_circuit_indices(round_number: int) -> tuple[int, ...]`.
- Consumes: `self.config.seed` and `len(self.circuits)` only.

- [ ] **Step 1: Write failing deterministic-order tests**

Add a test that creates a lightweight trainer with seven circuits and verifies:

```python
first = trainer._round_circuit_indices(1)
assert sorted(first) == list(range(7))
assert first == trainer._round_circuit_indices(1)
assert first != trainer._round_circuit_indices(2)
```

Capture `random.getstate()` before and after the calls and assert equality to
prove the helper uses an isolated generator. Assert the new training protocol
identity and that the previous `multi_circuit_ppo_batch_v1` checkpoint is
rejected for resume.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "round_circuit_order or training_defaults" -q -p no:cacheprovider --basetemp=E:\桌面\fault-reorder\tmp\pytest-shuffle-red`

Expected: FAIL because `_round_circuit_indices` does not exist and the protocol
identity still names the unshuffled implementation.

- [ ] **Step 3: Implement the pure order helper**

Use an integer seed derived without hashing:

```python
round_seed = (self.config.seed << 32) | round_number
indices = list(range(len(self.circuits)))
random.Random(round_seed).shuffle(indices)
return tuple(indices)
```

Reject non-positive or non-integer round numbers. Update the training protocol
identity so old unshuffled checkpoints cannot resume.

- [ ] **Step 4: Run focused tests**

Run the Step 2 command again.

Expected: PASS.

---

### Task 2: Execute and resume from shuffled round positions

**Files:**
- Modify: `fault_order_rl/trainer.py:720-805`
- Modify: `tests/test_fault_order_rl.py:520-700`

**Interfaces:**
- Consumes: `Trainer._round_circuit_indices(number)`.
- Preserves: checkpoint field `next_circuit_index`, reinterpreted as a shuffled-order offset.
- Preserves: `_write_circuit(number, manifest_index, ...)` artifact naming.

- [ ] **Step 1: Update failing rollout-order and resume tests**

For the four-circuit collection test, calculate the expected round-one order
from `_round_circuit_indices(1)` and assert `_ppo_update` receives circuit names
in that order. For the five-circuit resume test, inject failure into the second
shuffled circuit, verify offset 0 remains committed, then inject failure into
the fifth shuffled circuit, verify offset 4 is committed, and assert artifacts
exist for the manifest indices in the first shuffled batch.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -k "collects_four or rollout_batch_checkpoint" -q -p no:cacheprovider --basetemp=E:\桌面\fault-reorder\tmp\pytest-shuffle-resume-red`

Expected: FAIL because `step()` still indexes `self.circuits` directly by the
completed-position offset.

- [ ] **Step 3: Apply the round permutation in `step()`**

At the start of `step()`, derive `circuit_indices` for `number`. Slice batch
positions using `next_circuit_index`, map each position through
`circuit_indices[position]`, and use that manifest index for the circuit,
episode record, and artifact filename. Continue checkpointing `batch_end` as
the next shuffled-order position.

- [ ] **Step 4: Run focused and full PPO tests**

Run: `C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests/test_fault_order_rl.py -q -p no:cacheprovider --basetemp=E:\桌面\fault-reorder\tmp\pytest-shuffle-ppo`

Expected: all tests in the file PASS.

- [ ] **Step 5: Commit Tasks 1-2**

```bash
git add fault_order_rl/trainer.py tests/test_fault_order_rl.py
git commit -m "feat: shuffle circuits independently each round"
```

---

### Task 3: Documentation and complete verification

**Files:**
- Modify: `docs/fault-order-rl.md`
- Modify: `docs/fault-reorder-algorithm.md`

**Interfaces:**
- Documents the deterministic per-round shuffle and shuffled-offset resume semantics.

- [ ] **Step 1: Update user documentation**

State that every round derives a fresh circuit permutation from the fixed seed
and round number, rollout batches follow that order, resume reconstructs it,
and artifact indices remain manifest-stable. Replace the old training-protocol
identity in checkpoint compatibility text.

- [ ] **Step 2: Run the complete test suite**

Run: `C:\Users\acer\.conda\envs\d2l\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=E:\桌面\fault-reorder\tmp\pytest-shuffle-full`

Expected: `75 passed, 1 skipped` or more passing tests if the new tests increase
the total; no failures.

- [ ] **Step 3: Run repository checks**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/fault-order-rl.md docs/fault-reorder-algorithm.md
git commit -m "docs: explain per-round circuit shuffling"
```

- [ ] **Step 5: Review the complete change**

Review the diff from `3b66c84` through `HEAD` against the spec. Verify round
order determinism, global RNG isolation, manifest-stable artifacts, resume from
offset 4, old-protocol resume rejection, and unchanged PPO/GAE behavior. Fix
all Critical or Important findings, then rerun Tasks 2 Step 4 and 3 Steps 2-3.
