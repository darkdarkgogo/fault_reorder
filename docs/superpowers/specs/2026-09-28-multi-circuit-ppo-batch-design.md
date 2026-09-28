# Multi-Circuit PPO Batch Training Design

## Goal

Reduce validation oscillation, circuit-order bias, and forgetting caused by
updating the PPO policy immediately after every circuit. Preserve the existing
environment, reward, terminal correction, actor-critic network, and on-policy
PPO action semantics.

## Chosen approach

Collect up to four complete circuit episodes with one unchanged policy, then
perform one PPO update phase over their combined transitions. This replaces the
current one-circuit rollout batch without changing what constitutes an action
or how an individual episode is executed.

Two alternatives are intentionally not used:

- Per-circuit loss weighting is not used. Every transition has equal weight, so
  longer circuits contribute more transitions and therefore more gradient.
- Equal-sized minibatch repartitioning is not used. A final minibatch smaller
  than 128 transitions is accepted as its own optimizer step.

## Configuration

The fixed first-version settings are:

| Setting | Value |
| --- | ---: |
| `circuit_batch_size` | 4 circuits |
| `ppo_minibatch_size` | 128 transitions |
| `ppo_epochs` | 4 |
| `learning_rate` | `1e-4` |
| `ppo_clip` | `0.2` |
| `gamma` | `1.0` |
| `gae_lambda` | `0.95` |

The existing reward, terminal correction, value coefficient, entropy
coefficient, gradient clipping, Actor-Critic architecture, `LayerNorm(515)`,
DTC/STC behavior, and wire budget remain unchanged. No target-KL early stop or
additional update guard is introduced.

## Rollout data flow

1. At the start of every round, derive a new deterministic random permutation
   of all training circuit indices from the fixed training seed and the
   one-based round number. Use an isolated Python `random.Random` instance so
   constructing the circuit order does not consume the global rollout RNG.
2. At a circuit-batch boundary, copy the current model and optimizer into a
   transactional candidate, as the current trainer does for one circuit.
3. Put the candidate model in evaluation mode and do not call
   `optimizer.step()` while collecting the batch.
4. Run the next four circuits in the round permutation. If fewer than four
   circuits remain at the end of a round, collect and train on that smaller
   final batch.
5. For each circuit, independently compute step rewards, add terminal
   correction to its final transition, and compute GAE/returns in the original
   temporal order. GAE must never cross a circuit boundary.
6. Concatenate all raw advantages from the collected circuits and normalize
   once with population standard deviation (`unbiased=False`). Do not normalize
   again per circuit or per minibatch.
7. Treat every transition as one equally weighted PPO sample. Each sample keeps
   its owning circuit so replay uses that circuit's embeddings, pre-step
   remaining rows, Primary action, and executed DTC prefixes.
8. For each PPO epoch, generate a fresh random permutation of all transition
   indices. Split it into consecutive minibatches of at most 128 transitions.
9. Replay the selected transitions, compute the existing clipped PPO objective,
   and perform one optimizer step per minibatch. Keep the final smaller
   minibatch and give its mean loss the same optimizer-step treatment as every
   other minibatch.
10. After four PPO epochs, discard the rollout data and commit the candidate
   model, optimizer, checkpoint, and per-circuit artifacts.

For 350 transitions, each epoch produces minibatches of 128, 128, and 94, for
12 optimizer steps across four epochs.

## Transaction and resume semantics

The atomic training transaction moves from one circuit to one rollout batch.
If any rollout, trajectory finalization, PPO minibatch, artifact write, or
checkpoint write fails, restore the RNG state captured before the batch and do
not expose a partially updated model as the current trainer state.

`next_circuit_index` represents the number of completed positions in the
current round permutation, not a manifest index. It advances only after the
complete rollout batch succeeds. On resume, the seed and active round recreate
the identical permutation before continuing at that position. Circuit records
and artifact filenames continue to use the original manifest index so every
circuit retains a stable identity. The latest checkpoint is written only at a
batch boundary. A final partial batch advances the position by its actual
number of circuits.

New checkpoints carry a training-protocol identity for shuffled multi-circuit
batching. Old checkpoints remain evaluable because the policy architecture and
action semantics are unchanged, but they cannot be resumed into the new
training protocol.

## Records and diagnostics

Each episode keeps its existing trajectory and reward fields. Its record also
identifies the rollout batch and includes the shared PPO update summary. The
summary contains:

- circuit names and circuit count;
- transition count;
- minibatch count per epoch;
- total optimizer-step count;
- per-minibatch loss, actor loss, critic loss, entropy, approximate KL, clip
  fraction, gradient norm, sample count, and epoch number.

The diagnostics are observational only. They do not trigger early stopping or
otherwise change updates.

## Tests

Automated tests must verify:

- default and invalid batch-size configuration;
- four circuits are collected before the first optimizer step;
- every round uses the deterministic seed/round permutation and consecutive
  rounds derive their orders independently;
- interrupted and resumed training reconstructs the same active-round order;
- artifact indices remain tied to manifest positions after shuffling;
- a final partial circuit batch is retained;
- terminal correction and GAE remain isolated by circuit;
- advantages are normalized once across the combined transitions;
- each epoch uses a fresh transition permutation;
- a 350-transition batch produces `128/128/94` minibatches and 12 optimizer
  steps;
- replay uses the correct circuit embeddings for every transition;
- a failed second circuit rolls back the entire batch and leaves
  `next_circuit_index` at the batch start;
- successful checkpoint/resume occurs at rollout-batch boundaries;
- old training checkpoints are rejected for resume while remaining available
  to standalone evaluation.

## Out of scope

- Per-circuit weighting or balanced circuit sampling.
- Special handling for a smaller final PPO minibatch.
- KL-based early stopping or extra PPO update protection.
- Multi-seed benchmarking as an implementation acceptance requirement.
- Reward, GAE hyperparameter, model architecture, solver, or validation-policy
  changes.
