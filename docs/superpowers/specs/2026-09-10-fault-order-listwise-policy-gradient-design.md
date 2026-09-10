# Shared fault ordering with listwise policy gradient

## Goal

Train one shared neural fault scorer on all 16 bundled binary benchmark
circuits. For each circuit, score every existing collapsed logical fault, sort
the complete fault list, run one stuck-at PODEM ATPG job, and use the generated
pattern count as reinforcement-learning feedback. Repeat this process in rounds
and retain the best deterministic scorer checkpoint.

The optimization target is fewer ATPG patterns without reducing detected fault
count. The ATPG configuration is fixed at one attempt per fault, backtrack limit
3000, seed 14, and no static test compression, dynamic test compression, SCOAP
ordering, transition-delay mode, or other compression pass.

## Fixed fault and embedding semantics

This feature consumes the logical collapsed fault catalog and embeddings that
already exist. It does not regenerate faults or change fault collapsing.

- XOR and multi-input gate decomposition helpers remain physical execution and
  embedding anchors only. A helper name must never become a `fault_id`.
- Every action is one complete `fault_id` from the companion fault map.
- Every catalog fault must occur exactly once in each submitted permutation.
- GI and GO mapping, logical XOR input handling, duplicate `@dup` records, and
  equivalence counts retain their current behavior.
- Each fault has one 257-dimensional embedding produced by the existing
  pretrained DeepGate2 path. The scorer must not silently synthesize missing
  embeddings or fall back to random features.
- The ordered fault IDs in an embedding artifact must exactly match the PODEM
  catalog IDs for the same binary BENCH and fault map before training starts.

The training set is all 16 bundled binary circuits:

```text
c1355, c1908, c2670, c3540, c432, c499, c5315, c6288, c7552,
s13207_scan, s15850_scan, s35932_scan, s38417_scan, s38584_scan,
s5378_scan, s9234_scan
```

## Architecture

The system has four isolated layers:

1. **PODEM ordered-run API** validates a complete logical fault permutation,
   applies it to `flist_undetect`, runs stuck-at ATPG, and returns structured
   metrics.
2. **Dataset layer** loads immutable embedding artifacts and validates their
   provenance and fault IDs against the PODEM catalog.
3. **Listwise policy** maps every 257-dimensional embedding to one scalar,
   samples a complete permutation during training, and creates a deterministic
   descending-score permutation during evaluation.
4. **Trainer** owns per-circuit pattern baselines, reward calculation, shared
   optimizer updates, checkpoints, evaluation, and logs.

The PODEM solver remains the environment. Gradients never pass through PODEM;
the sampled ranking receives one terminal reward after the complete ATPG run.

## PODEM integration

### Structured run result

Refactor the stuck-at portion of `ATPG::test()` so the underlying operation can
return a result structure while the existing executable can continue printing
its current report. The Python path must return at least:

```text
pattern_count
detected_collapsed_faults
detected_equivalent_faults
uncollapsed_faults
aborted_faults
redundant_faults
podem_calls
total_backtracks
```

Each Python call constructs a fresh `ATPG` instance so no vectors, detection
state, or list mutations leak between episodes.

### Ordered Python API

Extend `cpp_podem` with a callable equivalent to:

```python
run_stuck_at_ordered(
    circuit_path: str,
    fault_map_path: str,
    ordered_fault_ids: list[str],
    backtrack_limit: int = 3000,
    seed: int = 14,
) -> dict
```

The function initializes the mapped logical catalog, validates that
`ordered_fault_ids` is an exact permutation of catalog IDs, reorders
`flist_undetect`, and runs one-attempt stuck-at ATPG. Unknown IDs, duplicate
IDs, missing IDs, helper-derived IDs, malformed maps, and empty incomplete
orders are hard errors raised before ATPG starts.

No `-fault-order <file>` executable option is introduced. The reinforcement
learning path uses the in-memory Python binding. Existing executable behavior
and native ordering remain available for baseline comparisons.

## Shared scorer

Use one scorer for every circuit:

```text
LayerNorm(257)
Linear(257, 256) + ReLU
Linear(256, 128) + ReLU
Linear(128, 1)
```

The scorer has no dropout so deterministic evaluation is stable. It processes
faults in chunks when necessary, but produces one scalar tensor in catalog
order. Inference sorts by descending score and breaks exact ties by original
catalog row, preserving deterministic behavior.

All circuits update the same model parameters. There is no circuit-specific
output head and no circuit identity feature in the initial implementation.

## Listwise policy

For fault scores `s`, form centered per-circuit logits and divide by temperature
`tau`. During training, sample independent Gumbel noise and rank:

```text
permutation = argsort(logits + gumbel_noise, descending=True)
```

This samples a Plackett-Luce permutation. For a sampled permutation `p`, compute
its log probability from the unperturbed logits:

```text
log P(p) = sum_t(logit[p[t]] - logsumexp(logit[p[t:]]))
```

Use reverse `logcumsumexp` so this calculation is linear after sorting rather
than quadratic in fault count. Divide each permutation log probability by its
fault count before combining circuits; otherwise large circuits would dominate
the optimizer merely because their lists contain more terms.

Training uses Gumbel exploration. Evaluation and exported rankings use raw
scorer values with no noise. Temperature is configurable and checkpointed;
the default starts at 1.0 and decays exponentially to a floor of 0.1 over the
configured training rounds.

## Baselines and reward

Two values with different purposes are maintained per circuit:

- `previous_pattern_count` is the environment baseline used to calculate the
  requested ATPG reward. Its initial value comes from native catalog order.
- `reward_ema` is a control variate used only to reduce policy-gradient
  variance. It does not change the reported reward.

For a coverage-valid episode:

```text
raw_reward = previous_pattern_count - current_pattern_count
previous_pattern_count = current_pattern_count
```

A smaller pattern count is positive reward; a larger pattern count is negative
reward. The policy advantage is:

```text
advantage = (raw_reward - reward_ema) / max(native_pattern_count, 1)
```

The native pattern count denominator gives the 16 circuits comparable influence
without changing the raw pattern-count reward recorded in logs.

`reward_ema` starts at zero. The current stored EMA is used to compute the
advantage, then after the optimizer step it is updated with decay 0.9 from the
current raw reward, including a coverage penalty when present. Advantages are
detached constants; gradients flow only through permutation log probability.

Each circuit also stores `required_detected_equivalent_faults`, initialized from
its native-order run and increased if a later run detects more faults. If an
episode detects fewer than this requirement, it is invalid and receives:

```text
raw_reward = -max(native_pattern_count,
                  previous_pattern_count,
                  current_pattern_count,
                  1)
```

An invalid episode does not update `previous_pattern_count` or the required
detection count. This prevents the learner from reducing pattern count by
sacrificing coverage.

The REINFORCE objective for one round is:

```text
loss = -mean_circuit(advantage[c] * log_probability[c] / fault_count[c])
```

Gradients are clipped to a configurable norm, default 1.0. Adam is used with a
default learning rate of `1e-4`. Optimizer defaults remain configuration values,
not hidden constants in the training loop.

## Training round

One round means all 16 circuits participate once:

1. Freeze one shared scorer snapshot for data collection.
2. Score each circuit and sample one complete fault permutation.
3. Run one ordered stuck-at ATPG episode per circuit with that permutation.
4. Record solver metrics and compute each circuit's reward.
5. Recompute policy log probabilities from the frozen snapshot, sampled
   permutations, and cached embeddings. This avoids retaining autograd graphs
   while long ATPG jobs run.
6. Aggregate all 16 policy losses and perform one shared optimizer step.
7. Update valid environment baselines, reward EMAs, RNG state, and round logs.
8. Save a resumable latest checkpoint atomically.

The first setup phase runs native order once for every circuit to establish
pattern and detection baselines. These 16 baseline runs are not policy updates.

The initial implementation executes ATPG episodes sequentially. Parallel solver
workers are excluded until deterministic equivalence and process isolation are
verified, because ATPG runtime rather than scorer inference is expected to be
the dominant cost.

## Deterministic evaluation and model selection

At a configurable interval, default every 10 training rounds, score all faults
without Gumbel noise, sort descending, and run all 16 circuits once. A candidate
checkpoint is eligible only when every circuit meets its current required
detected-equivalent-fault count.

Eligible checkpoints are compared by the sum of deterministic pattern counts
over all 16 circuits. Lower is better. Ties are resolved by, in order:

1. More detected equivalent faults.
2. Fewer total PODEM calls.
3. Fewer total backtracks.
4. Earlier training round.

Both `latest` and `best` checkpoints are retained. Final reported results must
come from a fresh deterministic evaluation of `best`, not from a noisy training
episode.

## Artifacts, configuration, and resume

Add a training manifest that lists all 16 circuits and, for each one, its binary
BENCH, fault map, embedding NPZ, and embedding metadata JSON. Paths resolve
relative to the manifest. Startup validates that all files exist and that
catalog IDs, row counts, feature dimensions, circuit digests, fault-map digests,
and checkpoint provenance agree.

The Python package is organized by responsibility:

```text
fault_order_rl/
  data.py          manifest and embedding validation
  model.py         shared fault scorer
  policy.py        Gumbel/Plackett-Luce ranking and log probability
  environment.py   cpp_podem ordered-run wrapper and result validation
  trainer.py       baseline collection, rounds, rewards, updates, evaluation
  checkpoint.py    atomic save/load and RNG restoration
  __main__.py      train and evaluate commands
```

The public commands are:

```powershell
python -m fault_order_rl validate --manifest configs/all_benchmarks.json
python -m fault_order_rl train --manifest configs/all_benchmarks.json --rounds 100 --output runs/shared_scorer
python -m fault_order_rl train --resume runs/shared_scorer/latest.pt
python -m fault_order_rl evaluate --checkpoint runs/shared_scorer/best.pt
```

One hundred rounds is the default initial experiment, not an acceptance
threshold. `validate` performs all catalog, feature, and provenance checks
without running ATPG. `train` collects native baselines when starting a new
run; `--resume` restores the manifest and configuration from its checkpoint.

A checkpoint contains model and optimizer states, completed round, temperature
state, Python/NumPy/PyTorch RNG states, per-circuit pattern baselines, required
detection counts, reward EMA values, manifest digest, embedding provenance, and
best evaluation summary. Resume rejects incompatible manifests or embeddings.

Write append-only JSONL episode and evaluation logs plus a compact JSON summary.
Every episode row includes circuit, round, seed, temperature, pattern count,
detection metrics, raw reward, advantage, solver effort, elapsed time, and
checkpoint identity. Fault permutations are stored as catalog-row integer
arrays in compressed artifacts rather than repeated strings in JSONL.

## Failure handling

- Missing or incompatible embeddings fail before baseline ATPG runs.
- A bad permutation fails before solver execution and never becomes a training
  sample.
- Solver exceptions, non-finite scores or losses, and malformed result metrics
  abort the current round without changing model parameters or baselines.
- The latest checkpoint is written only after all 16 episodes and the optimizer
  step complete successfully.
- Interrupted training resumes from the last complete round with restored RNG
  state, producing the same next sampled permutations.
- A coverage-invalid episode is a valid negative training sample, not a crash.

## Testing

### C++ and binding tests

- Native catalog order through the new API matches existing stuck-at metrics.
- Reversed and selected known permutations are applied exactly.
- Missing, duplicate, unknown, helper-derived, and cross-circuit fault IDs are
  rejected before ATPG.
- Result counters match the existing executable on small mapped circuits.
- Repeated calls with fresh instances and seed 14 are deterministic.
- STC, DTC, SCOAP, and transition-delay paths remain disabled in ordered runs.

### Python unit tests

- The scorer emits exactly one finite scalar per 257-dimensional fault row.
- Deterministic sorting uses catalog-row tie-breaking.
- Gumbel sampling returns an exact permutation under fixed RNG seeds.
- Linear-time log-probability matches a small direct Plackett-Luce calculation.
- Reward, baseline updates, coverage penalty, EMA advantage, and equal circuit
  weighting follow the formulas in this document.
- Checkpoint resume restores model, optimizer, baselines, and sampled order.
- Manifest validation catches reordered IDs and provenance mismatches.

### End-to-end tests

- A tiny mapped multi-input/XOR circuit completes baseline, one training round,
  checkpoint, resume, and deterministic evaluation without introducing helper
  fault IDs.
- A short smoke run includes more than one real benchmark and demonstrates that
  one optimizer update receives losses from both circuits.
- Before a full experiment, all 16 artifacts pass validation and all 16 native
  baselines are recorded under the fixed ATPG settings.

## Acceptance criteria

The implementation is ready for full training when:

1. Every one of the 16 circuits has an exact catalog-to-embedding match.
2. Ordered ATPG accepts only complete logical fault permutations and returns
   structured deterministic metrics.
3. One round runs all 16 circuits, computes the specified reward, and performs
   one shared scorer update.
4. No training or evaluation run enables STC, DTC, SCOAP, compression, or TDF.
5. Coverage-regressing rankings receive the specified penalty and cannot become
   the best checkpoint.
6. Interrupted training resumes at a round boundary with reproducible sampling.
7. The best checkpoint exports a deterministic score and rank for every fault
   in every circuit and reports per-circuit and aggregate ATPG results.

## Out of scope

- Changing fault creation, collapsing, helper mapping, or embedding semantics.
- Adding a text-file `-fault-order` CLI option.
- Reproducing DeepTPI's sequential DQN action selection literally.
- Static or dynamic test compression and transition-delay ATPG.
- Jointly fine-tuning the pretrained DeepGate2 encoder.
- Parallel/distributed ATPG execution in the first implementation.
