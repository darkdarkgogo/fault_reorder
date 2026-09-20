# Dynamic Fault Ranking Actor-Critic PPO Strict Design

## Status and authority

This specification records the design approved on 2026-09-20. It implements the
user-provided `Fault_Reorder_Dynamic_PPO_Training_Design_v2.3.pdf` together with
`Fault_Reorder_PPO_v2.3_Strict_Revision_Implementation_Guide.pdf`. Where those
documents disagree, the strict revision wins. This specification adds two final
repository-specific constraints agreed during review:

- DTC attempts must be a contiguous prefix of the supplied secondary ranking.
- `latest.pt` must commit progress after every completed training circuit, not
  only after a full round.

The existing fixed solver protocol remains unchanged except that the DTC
secondary order is supplied by the policy instead of being restored to catalog
order internally.

## Goal

Replace the current dynamic-primary REINFORCE learner with an actor-critic PPO
learner that controls both the primary fault and the actual DTC secondary
priority. Optimize final `patterns_after_stc` subject to every circuit meeting
its heuristic covered-equivalent-fault baseline. Use an independent validation
manifest for checkpoint selection and preserve exact interruption/resume
behavior at circuit boundaries.

## Chosen approach

The implementation will make the full-stack change rather than approximate DTC
ordering in Python or stage a primary-only PPO migration.

Alternatives rejected during design review were:

1. Keep catalog-ordered DTC and apply PPO only to primary selection. This is a
   smaller change but makes the claimed ranking action differ from the action
   executed by the solver.
2. Preserve the v2.3 terminal formula without correction. Dense detection
   shaping could then compensate for a coverage violation or change the final
   optimization target when the detected/redundant split changes.
3. Save only at round boundaries. A crash could discard as many as 1024 costly
   circuit rollouts and their PPO updates.

## Fixed experiment configuration

The first implementation uses these defaults:

```text
training rounds                 = 5
PPO epochs per circuit rollout  = 4
gamma                           = 1.0
GAE lambda                      = 0.95
PPO clip epsilon                = 0.2
value loss coefficient          = 0.5
entropy coefficient             = 0.01
actor learning rate             = 1e-4
critic learning rate            = 1e-4
gradient norm clip              = 1.0
step shaping alpha              = 0.1
coverage penalty beta           = 10.0
training manifest               = caller supplied
validation manifest             = configs/anchor_validation_6.json by default
```

The existing PODEM protocol remains:

```text
primary backtrack limit         = 200
primary seed                    = 14
attempts per primary fault      = 1
stuck-at DTC                    = enabled
DTC secondary backtrack limit   = 50
stuck-at STC                    = enabled
STC reverse-order compaction    = enabled
STC shuffle seed                = 7
STC no-improvement limit        = 5
SCOAP ordering                  = disabled
transition-delay mode           = disabled
```

All numeric configuration values must be finite and must satisfy their natural
range constraints. The formal training workflow requires exactly 5 rounds;
resume may only complete the remainder of those same 5 rounds. The checkpoint
stores the exact configuration and rejects incompatible resume.

## State and actor-critic model

For initial fault embeddings `e_i` of dimension 257 and the selectable set
`F_t`, candidate features remain:

```text
x_i,t = [e_i, MeanPool({e_j | j in F_t}), |F_t| / |F_0|]
```

The feature dimension remains 515. Empty sets, duplicate rows, out-of-range
rows, non-finite embeddings, and non-finite model outputs are errors.

The existing scorer becomes an actor-critic network:

```text
candidate feature [K, 515]
  -> LayerNorm(515)
  -> Linear(515, 256) + ReLU
  -> Linear(256, 128) + ReLU
  -> candidate encoding [K, 128]

actor head:
  Linear(128, 1) -> score per candidate

critic head:
  mean pool candidate encodings
  -> Linear(128, 64) + ReLU
  -> Linear(64, 1) -> V(s_t)
```

There is no dropout. Replaying a stored state under unchanged parameters must
reproduce its scores and value exactly. Actor and critic may share the encoder,
but the optimizer must include all encoder and head parameters in one
transactional update.

## Ranking action and probability

At every environment step, the actor scores every currently selectable fault
and samples one complete Plackett-Luce ranking without replacement. A Gumbel
top-k implementation is acceptable if its distribution is exactly equivalent
and its RNG state is checkpointed.

The first ranked fault is the primary. All later faults are passed to C++ as the
ordered DTC candidate list. Deterministic validation orders by descending score
and breaks exact ties by the original catalog row.

The environment may execute only:

```text
executed_sequence = [primary] + dtc_attempted_fault_ids
```

For an executed sequence of length `K_t`, its log probability is the sum of
conditional masked-categorical log probabilities for those `K_t` selections.
Unvisited ranking tail faults do not enter the log probability. Attempted but
failed secondary faults do enter it.

Each conditional selection also has a categorical entropy. The entropy used by
the PPO loss for one ATPG step is the mean across the executed decisions:

```text
H_t = mean(H_t,1, ..., H_t,K_t)
```

This prevents longer DTC prefixes from receiving a larger entropy bonus solely
because of length.

Rollout storage for each ATPG step contains at least:

```text
remaining catalog rows before the step
requested ranking rows
executed sequence rows
old executed-sequence log probability
old value estimate
step shaping reward
done flag
pattern increment
newly detected equivalent-fault increment
DTC attempted and embedded IDs
solver trace and cumulative metrics
```

The requested full ranking is retained for audit and prefix validation even
though the PPO ratio uses only the executed sequence.

## Native ranked-DTC contract

The native and Python-facing session API changes logically from:

```python
step(primary_fault_id)
```

to:

```python
step(primary_fault_id, ranked_secondary_fault_ids)
```

At call time:

- `primary_fault_id` must be selectable.
- `ranked_secondary_fault_ids` must contain every other selectable fault exactly
  once, with no duplicates, unknown IDs, primary ID, omitted candidates, or
  extra candidates.
- The combined primary and secondary list is therefore a permutation of the
  pre-step selectable set.

C++ passes the ordered vector unchanged from the binding to the ATPG step and
then to stuck-at DTC. `saf_compaction.cpp` must not sort by `fault_no` or silently
fall back to catalog order.

The solver must return DTC attempted IDs in their exact execution order.
Attempted IDs must equal a contiguous prefix of
`ranked_secondary_fault_ids`. If DTC reaches a stopping condition, it stops; it
must not skip an intermediate candidate and continue with a later candidate.
Embedded IDs must be an order-preserving subset of attempted IDs. Python treats
any reordering, gap, duplicate, unknown ID, or non-prefix result as a protocol
error before using the trajectory for training.

The existing result fields remain, including exact attempted/embedded IDs,
newly detected IDs, current pattern count, split backtracks, final STC counts,
and covered equivalent-fault metrics. Fault simulation, not DTC embedding,
continues to determine which faults were newly detected and dropped.

The legacy ordered-run interface may remain for regression compatibility, but
training, validation, and dynamic evaluation use the ranked session interface.

## Reward and terminal correction

Let `InitialEqv` be the validated sum of equivalent-fault multiplicities for the
circuit. After each solver step:

```text
I_pattern,t = current_pattern_count_t - current_pattern_count_t-1
DeltaDetectedEqv_t = sum(eqv multiplicity of newly detected fault IDs)
r_t = (-I_pattern,t + 0.1 * DeltaDetectedEqv_t) / InitialEqv
```

`I_pattern,t` is read from the actual cumulative pattern-count difference; it
is not inferred from target status. It must be 0 or 1 under the one-primary-call
protocol. Redundant and aborted primary steps with no detected faults therefore
receive zero shaping reward.

After all primary attempts and STC, define:

```text
shortfall = max(heuristic_covered_eqv - rl_covered_eqv, 0)

G_target = 1 - patterns_after_stc / InitialEqv       if shortfall == 0
G_target = -10 * shortfall / InitialEqv              otherwise

G_step = sum(r_t)
R_terminal = G_target - G_step
```

`R_terminal` is added to the final transition before returns and GAE are
computed. With gamma 1.0, the undiscounted episode return is exactly
`G_target`; dense shaping cannot compensate for a coverage failure or change
the ordering among coverage-valid trajectories.

The environment/trainer asserts:

```text
0 <= patterns_after_stc <= patterns_before_stc <= InitialEqv
```

The last inequality follows from at most one generated pattern per selected
collapsed fault and `collapsed_fault_count <= InitialEqv`. It is checked rather
than assumed. Consequently, valid targets are non-negative and invalid targets
are negative, giving strict coverage-first separation.

## GAE and PPO update

The terminal state's value is zero. After adding terminal correction to the
last transition, compute returns and GAE with gamma 1.0 and lambda 0.95.

Advantage normalization is performed over the valid transitions in the current
circuit trajectory only when there are at least two advantages and population
standard deviation (`unbiased=False`) exceeds epsilon. Otherwise raw advantages
are preserved.

For each transition:

```text
ratio = exp(new_executed_log_prob - old_executed_log_prob)
actor loss = -mean(min(ratio * A, clip(ratio, 0.8, 1.2) * A))
critic loss = mean((V(s_t) - return_t)^2)
entropy bonus = mean(H_t)
total loss = actor loss + 0.5 * critic loss - 0.01 * entropy bonus
```

One circuit is rolled out completely with fixed parameters. The same stored
trajectory is then optimized four times without re-running ATPG. New log
probabilities, values, and entropies are recomputed from stored pre-step states
and executed sequences on every epoch. Parameters never change during the
rollout itself.

Before committing an update, advantages, log ratios, ratios, each loss,
gradient norm, and every parameter must be finite. The gradient norm is clipped
to 1.0. Approximate KL and clip fraction are logged. A failure aborts the
current uncommitted circuit and leaves the last checkpoint and its RNG state
unchanged.

## Training and validation separation

The trainer owns two disjoint collections:

- `train_circuits` from the caller-supplied training manifest.
- `validation_circuits` from an explicit validation manifest, defaulting to
  `configs/anchor_validation_6.json`.

The manifests must resolve to different files. Circuit names and artifact
identities must be disjoint; overlap is a configuration error. Each split runs
and caches its own catalog-first heuristic/native baseline with the same fixed
PODEM, DTC, and STC protocol.

Training circuits are used only for stochastic rollout, GAE/PPO updates, and
training diagnostics. Validation circuits are used only for deterministic
inference after each completed round. Validation performs no backward pass and
no optimizer step.

The best checkpoint key is computed only from validation results:

```text
(
  sum(max(validation_heuristic_covered_eqv - validation_rl_covered_eqv, 0)),
  sum(validation_patterns_after_stc),
)
```

Lower is better. Thus the system retains the least-shortfall checkpoint even
before any fully coverage-valid checkpoint exists. Once a zero-shortfall model
exists, no positive-shortfall model can replace it.

No test dataset is created by this change. The evaluation command continues to
accept an explicitly supplied external manifest and must support `best.pt` and
`final.pt` without updating parameters.

## Five-round execution flow

Training uses a stable circuit order stored in the checkpoint. For every round
from 1 through 5:

1. Starting at the checkpointed next circuit index, collect one complete
   stochastic trajectory for the training circuit.
2. Apply four PPO epochs to that trajectory.
3. Atomically append/replace the circuit's training artifacts and atomically
   publish `latest.pt` with the next circuit index and post-update RNG state.
4. Continue until every training circuit has been processed.
5. Run deterministic validation on the independent validation set.
6. Update the in-memory best record by the validation key, atomically commit the
   completed-round `latest.pt`, then derive/publish `best.pt`.
7. Advance to the first circuit of the next round.

After round 5, atomically save `final.pt` unconditionally from the current model
and retain `best.pt`. `best.pt` and `final.pt` may identify different rounds.

The default behavior is one trajectory followed immediately by its four PPO
epochs. The old multi-circuit REINFORCE minibatch option is removed rather than
given ambiguous PPO semantics.

## Checkpoint schema and exact resume

The checkpoint schema advances from version 3 to version 4 because the model,
optimizer, policy probability, reward state, validation state, and progress
semantics are incompatible. Loading versions 1-3 for resume or evaluation must
produce a clear retraining-required error.

`latest.pt` contains at least:

```text
actor-critic state
optimizer state
completed round and active round
next training circuit index
stable training circuit traversal order
Python, NumPy, and Torch RNG states
training and validation manifest paths/digests
training and validation artifact provenance
native baselines for both splits
configuration and fixed solver protocol identity
solver binary digest and Torch version
current best model/report/key
committed per-round/per-circuit training progress
```

After a circuit checkpoint is committed, a crash during the next circuit
restores the exact model, optimizer, progress, and RNG state from before that
next rollout. Re-running from that point must reproduce requested rankings,
executed sequences, model parameters, optimizer state, logs, and checkpoint
selection.

`best.pt` and `final.pt` omit optimizer and RNG state, retain model/config/
provenance, record their checkpoint kind, and are not resumable. `best.pt`
records the validation key, report, and originating round. `final.pt` records
round 5 and its validation-independent final-policy identity.

## Logging and artifacts

Training records include:

- circuit, round, circuit index, episode steps, and `InitialEqv`;
- patterns before/after STC and pattern increments;
- training heuristic coverage, RL coverage, shortfall, and validity;
- sum step reward, terminal correction, and exact target/episode return;
- advantage mean/std, actor loss, critic loss, entropy, approximate KL, clip
  fraction, and gradient norm for each PPO epoch;
- requested ranking audit data, executed sequence, DTC attempted/embedded
  counts, backtracks, and runtime.

Compact NPZ trajectory artifacts retain remaining/requested/executed row arrays
with offsets. Human-readable deterministic validation/evaluation traces remain
JSONL. Validation reports and comparison CSVs must identify the validation
manifest and must not contain aggregate training metrics.

## Failure handling

The current circuit update is transactional. The following conditions abort it
without publishing model, optimizer, progress, logs as committed, or RNG state:

- invalid features, values, rewards, advantages, ratios, losses, gradients, or
  parameters;
- native ranking input that is not a permutation of the selectable set;
- attempted DTC IDs that are not a contiguous prefix of the requested order;
- embedded IDs that are not an order-preserving subset of attempted IDs;
- unknown, duplicate, reordered, or silently skipped native trace IDs;
- pattern or coverage counters that regress or violate final STC invariants;
- final episode return differing from `G_target` beyond floating tolerance;
- training/validation manifest overlap or provenance mismatch;
- checkpoint configuration, solver digest, or runtime incompatibility.

Coverage shortfall is a valid training outcome and receives the strict negative
target; it is not a program error.

## Required tests

### Model and policy

- Actor scores and critic value have correct shapes and finite gradients.
- The same state replays identical scores and value before an update.
- Stochastic ranking is a valid permutation and deterministic ranking has stable
  catalog-row tie breaking.
- Executed-prefix log probability matches direct sequential categorical
  calculation.
- Unvisited tail rows do not change executed log probability or entropy.
- Attempted-but-failed rows remain in both quantities.
- Entropy is finite for `K_t=1` and is the mean, not sum, for longer prefixes.

### Reward, GAE, and PPO

- Redundant and aborted zero-increment steps receive zero shaping reward.
- Equivalent-fault multiplicities, not collapsed-fault counts, determine the
  detection increment.
- Valid coverage produces exact return `1 - P_after/InitialEqv`.
- Invalid coverage produces exact return `-10 * shortfall/InitialEqv` regardless
  of accumulated shaping.
- Valid targets are non-negative and invalid targets are negative.
- Terminal correction is attached to the last transition before GAE.
- Single/zero-variance advantages are not normalized; normal batches are.
- Four PPO epochs reuse one rollout, yield finite metrics, and never call the
  environment again.
- PPO clipping, critic loss, entropy coefficient, and gradient clipping match
  the configured values.

### Native ranked DTC

- Binding and C++ accept primary plus a complete ordered secondary permutation.
- Requested order is preserved through ATPG and compaction.
- Attempted results are a contiguous prefix, including failed attempts.
- Reordered, skipped-middle, duplicate, unknown, omitted, and extra candidates
  are rejected.
- Embedded results are an order-preserving subset of attempted results.
- Existing cube rollback, primary preservation, fault simulation, backtrack,
  STC, concurrency, and deterministic-session regressions continue to pass.

### Training, validation, and checkpoints

- Training and validation manifests/baselines are separate and overlap is
  rejected.
- Training circuits never enter validation or best selection.
- Validation performs no parameter or optimizer update.
- Best selection uses `(shortfall, patterns_after_stc)` and can select a
  positive-shortfall model until a zero-shortfall model exists.
- `latest.pt` advances after each completed circuit and resumes at the exact next
  circuit.
- Failure during rollout or any PPO epoch leaves the preceding circuit
  checkpoint and RNG state unchanged.
- Continuous and interrupted/resumed runs produce identical rankings, traces,
  models, optimizer states, best selection, and final checkpoint.
- After five rounds, `latest.pt`, `best.pt`, and `final.pt` all exist and
  best/final may differ.
- Schema versions 1-3 fail with a clear retraining-required message.
- External deterministic evaluation accepts both best and final checkpoints.

## Documentation and compatibility

Update CLI help, README, and `docs/fault-order-rl.md` to describe the validation
manifest, fixed five-round PPO defaults, ranked DTC semantics, return correction,
checkpoint kinds, and restart requirement. Existing static/listwise helper
functions may remain only when still used by compatibility tests; dead
REINFORCE/EMA training paths and misleading documentation are removed.

The change does not alter frozen fault embeddings, fault-map semantics, fault
collapsing, BENCH/AIG anchoring, or the fixed PODEM/STC algorithms beyond the
explicit external DTC priority order.
