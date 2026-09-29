# TensorBoard and Normalized Entropy Design

## Goal

Add transaction-safe TensorBoard monitoring to multi-circuit PPO training and
replace the raw-entropy training bonus with entropy normalized independently for
each conditional action space.

## Entropy semantics

`_conditional_prefix_stats()` records the raw entropy of every executed
conditional choice. For a choice with `N > 1`, it also records
`entropy / log(N)`. Choices with `N == 1` remain part of raw-entropy diagnostics
but are excluded from the normalized mean. A transition with no nontrivial
choice has normalized entropy zero and contributes zero entropy gradient.

`joint_action_stats*()` returns log probability, raw transition entropy,
normalized transition entropy, and value. `ppo_objective()` accepts both
entropy vectors, reports both means, and uses only normalized entropy in the
loss. The new configuration field is `entropy_coef_normalized=0.05`; the old
`entropy_coef=0.01` training objective is retired.

To disambiguate forced transitions, PPO summaries and TensorBoard also record
`choice_transition_fraction`, the fraction of transitions containing at least
one conditional action space with more than one candidate.

## Monitoring clocks

- `PPO/*` uses one-based `global_optimizer_step`, once per minibatch update.
- `Rollout/*` uses one-based `global_rollout_batch_step`, once per committed
  rollout batch.
- `Validation/*` uses the completed round number.

PPO tags are total loss, actor loss, critic loss, raw entropy, normalized
entropy, choice-transition fraction, approximate KL, clip fraction, and
pre-clipping gradient norm. Rollout tags are transition count, mean episode
return, and mean episode steps. Validation tags are total pattern-reduction
percentage, fault coverage, covered equivalent faults, coverage shortfall, and
coverage eligibility.

PPO aggregate metrics are weighted by minibatch sample count. Validation
pattern reduction is computed from the ratio of total pattern sums, never the
mean of per-circuit percentages.

## Transaction and resume semantics

`_ppo_update()` does not write TensorBoard. A rollout batch computes candidate
model state, per-minibatch metrics, and proposed counters. Circuit artifacts and
`latest.pt` are committed first. Only then are candidate state and counters
installed and TensorBoard events emitted and flushed. A checkpoint failure
therefore produces no events. A crash after checkpoint commit may leave an
acceptable logging gap but cannot create an event for an uncommitted update.

Both global counters are stored in checkpoint schema 6 and restored directly;
event files are never used to reconstruct training state. The writer logs under
`<output>/tensorboard` and is closed by `train()` in a `finally` block.

## Compatibility

The policy architecture and action semantics are unchanged, so
`POLICY_IDENTITY` remains unchanged. `TRAINING_PROTOCOL` is upgraded for the
new normalized-entropy objective. `load_checkpoint()` accepts schema 5 and 6 so
existing checkpoints remain evaluable. `Trainer.resume()` requires schema 6,
the current training protocol, and both persisted counters, so schema 5 cannot
resume under the new objective.

## Tests

Tests cover uniform and concentrated normalized entropy, singleton choices,
independent Primary/DTC normalization, objective entropy selection, weighted
aggregation, 12 consecutive optimizer steps for the 350-transition case,
counter persistence and resume, checkpoint-before-event ordering, validation
aggregation, old-checkpoint evaluation, and old-checkpoint resume rejection.

