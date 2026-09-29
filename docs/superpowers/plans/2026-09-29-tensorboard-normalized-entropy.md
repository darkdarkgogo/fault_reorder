# TensorBoard and Normalized Entropy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add normalized-entropy PPO regularization and transaction-safe TensorBoard monitoring with resumable global steps.

**Architecture:** Policy replay computes raw and per-conditional normalized entropy together. Trainer updates remain transactional: checkpoint schema 6 persists candidate state and proposed counters before events are emitted, while schema 5 remains evaluation-only. TensorBoard metrics use three explicit clocks for optimizer updates, rollout batches, and rounds.

**Tech Stack:** Python, PyTorch, TensorBoard, pytest

**Spec:** `docs/superpowers/specs/2026-09-29-tensorboard-normalized-entropy-design.md`

## Global Constraints

- Keep `POLICY_IDENTITY` unchanged.
- Set `entropy_coef_normalized` to `0.05` by default.
- Schema 5 remains evaluable but cannot resume training.
- Write TensorBoard events only after their checkpoint transaction commits.
- Do not add multi-seed experiments or extra optimizer update guards.

---

### Task 1: Conditional normalized entropy

**Files:**
- Modify: `fault_order_rl/policy.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Produces: `joint_action_stats*() -> (log_prob, entropy_raw, entropy_normalized, value)`

- [ ] Add failing tests for uniform, concentrated, singleton, and mixed Primary/DTC choices.
- [ ] Run the focused policy tests and verify the old three-value interface fails.
- [ ] Return raw and normalized entropy from `_conditional_prefix_stats()` and aggregate them at transition level.
- [ ] Run the focused policy tests and verify finite gradients and theoretical bounds.
- [ ] Commit the policy and test changes.

### Task 2: Normalized-entropy PPO objective

**Files:**
- Modify: `fault_order_rl/reward.py`
- Modify: `fault_order_rl/trainer.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Consumes: separate raw and normalized transition entropy tensors.
- Produces: objective details keys `entropy_raw` and `entropy_normalized`.

- [ ] Add a failing test proving raw entropy cannot affect total loss while normalized entropy can.
- [ ] Run the focused objective test and verify failure.
- [ ] Update `ppo_objective()` and trainer call sites; rename the config coefficient.
- [ ] Update weighted PPO summaries and rename `gradient_norm_pre_clip`.
- [ ] Run the focused PPO tests and commit.

### Task 3: Checkpoint counters and compatibility

**Files:**
- Modify: `fault_order_rl/checkpoint.py`
- Modify: `fault_order_rl/trainer.py`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Produces: schema 6 fields `global_optimizer_step` and `global_rollout_batch_step`.
- Preserves: schema 5 loading for evaluation only.

- [ ] Add failing tests for schema acceptance, resume rejection, counter persistence, and resume continuity.
- [ ] Run the focused checkpoint tests and verify failure.
- [ ] Upgrade new payloads to schema 6 and restore counters on resume.
- [ ] Upgrade `TRAINING_PROTOCOL` without changing `POLICY_IDENTITY`.
- [ ] Run the focused checkpoint tests and commit.

### Task 4: Post-commit TensorBoard logging

**Files:**
- Modify: `fault_order_rl/trainer.py`
- Modify: `requirements-fault-order.txt`
- Test: `tests/test_fault_order_rl.py`

**Interfaces:**
- Produces: TensorBoard events in `<output>/tensorboard` using optimizer, rollout, and round clocks.

- [ ] Add a recording-writer test for tag names, exact one-based steps, and checkpoint-before-event ordering.
- [ ] Run the focused monitoring test and verify failure.
- [ ] Add a writer factory, post-commit PPO/Rollout writes, round-level Validation writes, flushing, and lifecycle closure.
- [ ] Add the explicit `tensorboard` dependency.
- [ ] Run the focused monitoring tests and commit.

### Task 5: Documentation and full verification

**Files:**
- Modify: `docs/fault-reorder-algorithm.md`
- Modify: `docs/fault-order-rl.md`
- Modify: `fault_order_rl/__main__.py`

**Interfaces:**
- Documents: schema 6, normalized entropy coefficient, TensorBoard directory, metrics, and resume behavior.

- [ ] Update user-facing protocol, CLI description, checkpoint schema, and monitoring documentation.
- [ ] Run the complete pytest suite.
- [ ] Inspect the final diff and repository status for unrelated changes.
- [ ] Commit the documentation and any final test adjustments.

