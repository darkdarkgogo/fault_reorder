"""Necessary tests for dynamic ranked-DTC actor-critic PPO."""

import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from fault_order_rl.checkpoint import load_checkpoint, save_checkpoint
from fault_order_rl.data import CircuitData, CircuitSpec
from fault_order_rl.environment import PROTOCOL_CONFIG, PodemEnvironment, PodemSession
from fault_order_rl.model import FaultActorCritic, build_dynamic_features
from fault_order_rl.policy import (centered_logits, joint_action_stats,
                                   joint_action_stats_from_scores,
                                   sample_candidate_ranking, sample_primary)
from fault_order_rl.reward import (compute_gae, normalize_advantages,
                                   ppo_objective, step_reward,
                                   target_return, terminal_correction)
from fault_order_rl.trainer import (POLICY_IDENTITY, SOLVER_PROTOCOL,
                                    TRAINING_PROTOCOL, TrainConfig, Trainer,
                                    evaluate_checkpoint, evaluation_key,
                                    state_dict_equal)


class _RecordingWriter:
    def __init__(self, operations=None):
        self.calls = []
        self.operations = operations
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, float(value), int(step)))
        if self.operations is not None:
            self.operations.append("event:" + tag)

    def flush(self):
        if self.operations is not None:
            self.operations.append("flush")

    def close(self):
        self.closed = True


def test_actor_critic_shapes_and_dynamic_features():
    embeddings = torch.arange(4 * 257, dtype=torch.float32).reshape(4, 257)
    features = build_dynamic_features(embeddings, (1, 3))
    assert features.shape == (2, 515)
    assert torch.equal(features[:, :257], embeddings[[1, 3]])
    assert torch.equal(features[:, 257:514], embeddings[[1, 3]].mean(0).expand(2, -1))
    assert torch.equal(features[:, 514], torch.full((2,), 0.5))
    scores, value = FaultActorCritic()(features)
    assert scores.shape == (2,)
    assert value.shape == ()


def test_primary_and_bfs_batch_joint_probability_have_gradients():
    model = FaultActorCritic()
    embeddings = torch.randn(4, 257)
    rows = (3, 0, 2)
    scores, _ = model(build_dynamic_features(embeddings, rows))
    equal_scores = torch.zeros_like(scores)
    assert int(sample_primary(equal_scores, rows, 1.0, False)) == 0
    assert sample_candidate_ranking(
        equal_scores, rows, (3, 2), 1.0, False).tolist() == [2, 3]
    sampled = sample_candidate_ranking(
        equal_scores, rows, (3, 2), 1.0, True,
        generator=torch.Generator().manual_seed(7))
    assert sorted(sampled.tolist()) == [2, 3]
    batches = ({
        "bfs_candidate_rows": (3, 2),
        "requested_rows": (2, 3),
        "executed_prefix_rows": (2,),
    },)
    log_prob, entropy_raw, entropy_normalized, value = joint_action_stats(
        model, embeddings, rows, 0, batches, 1.0)
    replay_scores, _ = model(build_dynamic_features(embeddings, rows))
    logits = centered_logits(replay_scores, 1.0)
    manual = (torch.log_softmax(logits, 0)[1]
              + torch.log_softmax(logits[[0, 2]], 0)[1])
    assert torch.allclose(log_prob, manual)
    assert 0.0 <= float(entropy_normalized) <= 1.0
    loss = -(log_prob + 0.05 * entropy_normalized) + value.square()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_entropy_is_normalized_per_conditional_action_space():
    scores = torch.zeros(4, requires_grad=True)
    value = scores.sum() * 0.0
    batches = ({
        "bfs_candidate_rows": (1, 2, 3),
        "requested_rows": (1, 2, 3),
        "executed_prefix_rows": (1, 2, 3),
    },)

    _, entropy_raw, entropy_normalized, _ = joint_action_stats_from_scores(
        scores, value, (0, 1, 2, 3), 0, batches, 1.0)

    expected_raw = (
        np.log(4.0) + np.log(3.0) + np.log(2.0)
    ) / 4.0
    assert float(entropy_raw) == pytest.approx(expected_raw)
    assert float(entropy_normalized) == pytest.approx(1.0)


def test_normalized_entropy_handles_singleton_and_concentrated_choices():
    singleton = torch.tensor([0.0], requires_grad=True)
    _, singleton_raw, singleton_normalized, _ = joint_action_stats_from_scores(
        singleton, singleton.sum() * 0.0, (0,), 0, (), 1.0)
    assert float(singleton_raw) == 0.0
    assert float(singleton_normalized) == 0.0
    assert torch.isfinite(singleton_normalized)

    concentrated = torch.tensor([20.0, 0.0, -20.0], requires_grad=True)
    _, _, concentrated_normalized, _ = joint_action_stats_from_scores(
        concentrated, concentrated.sum() * 0.0, (0, 1, 2), 0, (), 1.0)
    assert 0.0 <= float(concentrated_normalized) < 1e-6


def test_reward_terminal_target_gae_and_ppo_are_finite():
    rewards = [step_reward(1, 2, 10), step_reward(0, 1, 10)]
    valid_target = target_return(3, 10, 0)
    rewards[-1] += terminal_correction(rewards, valid_target)
    assert sum(rewards) == pytest.approx(0.7)
    assert target_return(3, 10, 2) == pytest.approx(-2.0)
    returns, advantages = compute_gae(rewards, [0.2, 0.1], 1.0, 0.95)
    normalized = normalize_advantages(advantages)
    new_log_probs = torch.tensor([-0.3, -0.4], requires_grad=True)
    values = torch.tensor([0.1, 0.2], requires_grad=True)
    loss, details = ppo_objective(
        new_log_probs, torch.tensor([-0.35, -0.45]), normalized,
        values, returns, torch.tensor([0.5, 0.4]),
        torch.tensor([0.8, 0.6]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in details.values())


def test_ppo_objective_uses_only_normalized_entropy_for_bonus():
    common = (
        torch.zeros(2), torch.zeros(2), torch.zeros(2),
        torch.zeros(2), torch.zeros(2),
    )
    first, first_details = ppo_objective(
        *common, torch.tensor([0.0, 10.0]), torch.tensor([0.25, 0.75]),
        entropy_coef_normalized=0.05)
    second, second_details = ppo_objective(
        *common, torch.tensor([100.0, 200.0]), torch.tensor([0.25, 0.75]),
        entropy_coef_normalized=0.05)
    third, _ = ppo_objective(
        *common, torch.tensor([0.0, 10.0]), torch.tensor([0.0, 0.0]),
        entropy_coef_normalized=0.05)

    assert float(first) == pytest.approx(-0.025)
    assert torch.equal(first, second)
    assert float(third) == 0.0
    assert float(first_details["entropy_raw"]) == pytest.approx(5.0)
    assert float(second_details["entropy_raw"]) == pytest.approx(150.0)
    assert float(first_details["entropy_normalized"]) == pytest.approx(0.5)


def test_fixed_protocol_and_training_defaults():
    assert PROTOCOL_CONFIG["primary_backtrack_limit"] == 100
    assert PROTOCOL_CONFIG["dtc_secondary_backtrack_limit"] == 50
    assert PROTOCOL_CONFIG["dtc_bfs_small_select_fault_try"] == 1
    assert PROTOCOL_CONFIG["dtc_bfs_default_select_fault_try"] == 1
    assert SOLVER_PROTOCOL["primary_backtrack_limit"] == 100
    assert SOLVER_PROTOCOL["compression_algorithm_version"] == (
        "stuck_at_podemx_bfs_ranked_dtc_monotonic_v4"
    )
    config = TrainConfig()
    config.validate()
    assert config.rounds == 5
    assert config.circuit_batch_size == 4
    assert config.ppo_minibatch_size == 128
    assert config.ppo_epochs == 4
    assert config.entropy_coef_normalized == pytest.approx(0.05)
    assert config.backtrack_limit == 100
    with pytest.raises(ValueError, match="100"):
        TrainConfig(backtrack_limit=200).validate()
    for changes in (
            {"circuit_batch_size": 0}, {"circuit_batch_size": True},
            {"ppo_minibatch_size": 0}, {"ppo_minibatch_size": False}):
        with pytest.raises(ValueError, match="positive integer"):
            TrainConfig(**changes).validate()
    assert TRAINING_PROTOCOL == (
        "normalized_entropy_shuffled_multi_circuit_ppo_batch_v2")


def test_round_circuit_order_is_deterministic_distinct_and_rng_isolated():
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainConfig(seed=14)
    trainer.circuits = [object() for _ in range(7)]
    rng_before = random.getstate()

    first = trainer._round_circuit_indices(1)
    second = trainer._round_circuit_indices(2)

    assert sorted(first) == list(range(7))
    assert sorted(second) == list(range(7))
    assert first == trainer._round_circuit_indices(1)
    assert first != second
    assert random.getstate() == rng_before
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            trainer._round_circuit_indices(invalid)


def _step_summary(**changes):
    result = {
        "pattern_count": 1,
        "current_pattern_count": 1,
        "finalized": False,
        "patterns_before_stc": 1,
        "patterns_after_stc": 1,
        "stc_removed_patterns": 0,
        "stc_shuffle_attempts": 0,
        "stc_coverage_preserved": False,
        "detected_collapsed_faults": 1,
        "detected_equivalent_faults": 1,
        "covered_equivalent_faults": 1,
        "uncollapsed_faults": 3,
        "aborted_faults": 0,
        "redundant_faults": 0,
        "redundant_equivalent_faults": 0,
        "podem_calls": 1,
        "primary_podem_calls": 1,
        "dtc_secondary_calls": 1,
        "primary_backtracks": 0,
        "dtc_backtracks": 0,
        "total_backtracks": 0,
        "selected_fault_id": "f0",
        "target_status": "detected",
        "generated_pattern": True,
        "generated_test_vector": "01",
        "dtc_attempted_fault_ids": ("f2",),
        "dtc_embedded_fault_ids": ("f2",),
        "newly_detected_fault_ids": ("f0",),
        "remaining_fault_ids": ("f1", "f2"),
        "current_podem_calls": 1,
        "current_dtc_secondary_calls": 1,
        "current_primary_backtracks": 0,
        "current_dtc_backtracks": 0,
        "current_total_backtracks": 0,
    }
    result.update(changes)
    return result


class _NativePrefixSession:
    def __init__(self, attempted=("f2",)):
        self.attempted = attempted
        self.seen = None
        self.begin_calls = 0

    def config(self):
        return dict(PROTOCOL_CONFIG)

    def catalog(self):
        return {
            "faults": [{"fault_id": identifier}
                       for identifier in ("f0", "f1", "f2")],
            "uncollapsed_total": 3,
        }

    def remaining_fault_ids(self):
        return ("f0", "f1", "f2")

    def begin_step(self, primary):
        self.begin_calls += 1
        self.primary = primary
        return {
            "phase": "dtc", "selected_fault_id": primary,
            "unknown_po_id": "po0", "dtc_candidate_fault_ids": ("f2", "f1"),
            "dtc_batch_index": 0, "select_fault_try": 1,
            "visited_wire_count": 1,
            "last_dtc_attempted_fault_ids": (),
            "last_dtc_embedded_fault_ids": (),
        }

    def rank_dtc_candidates(self, candidates):
        self.seen = (self.primary, tuple(candidates))
        return {**_step_summary(
            dtc_attempted_fault_ids=self.attempted,
            dtc_embedded_fault_ids=self.attempted,
            dtc_secondary_calls=len(self.attempted),
            current_dtc_secondary_calls=len(self.attempted)),
            "phase": "complete",
            "last_dtc_attempted_fault_ids": self.attempted,
            "last_dtc_embedded_fault_ids": self.attempted,
        }


class _NativeLazySession(_NativePrefixSession):
    def __init__(self):
        super().__init__()
        self.step_calls = []

    def begin_step(self, primary):
        raise AssertionError("baseline must not enter ranked DTC phases")

    def step(self, primary):
        self.step_calls.append(primary)
        return _step_summary()


def test_python_session_without_ranker_calls_native_lazy_step():
    native = _NativeLazySession()
    session = PodemSession(native)
    result = session.step("f0")
    assert native.step_calls == ["f0"]
    assert native.begin_calls == 0
    assert result["dtc_attempted_fault_ids"] == ("f2",)
    assert result["dtc_batches"] == ()


def test_python_session_ranks_only_bfs_batch_and_accepts_only_prefix():
    native = _NativePrefixSession()
    session = PodemSession(native)
    result = session.step("f0", lambda candidates, _: tuple(candidates))
    assert native.seen == ("f0", ("f2", "f1"))
    assert result["dtc_attempted_fault_ids"] == ("f2",)
    assert result["dtc_batches"][0]["bfs_candidate_fault_ids"] == ("f2", "f1")
    bad = PodemSession(_NativePrefixSession(("f1",)))
    with pytest.raises(RuntimeError, match="requested batch prefix"):
        bad.step("f0", lambda candidates, _: candidates)
    with pytest.raises(ValueError, match="exactly the current BFS"):
        PodemSession(_NativePrefixSession()).step(
            "f0", lambda candidates, _: candidates[:-1])


def test_python_session_retries_ranker_exception_without_repeating_primary():
    native = _NativePrefixSession()
    session = PodemSession(native)

    def fail_once(_candidates, _metadata):
        raise RuntimeError("ranker failed")

    with pytest.raises(RuntimeError, match="ranker failed"):
        session.step("f0", fail_once)
    result = session.step("f0", lambda candidates, _: candidates)
    assert native.begin_calls == 1
    assert result["dtc_attempted_fault_ids"] == ("f2",)


def _metrics(fault_count, raw_patterns, final_patterns=1):
    return {
        "pattern_count": final_patterns,
        "current_pattern_count": raw_patterns,
        "finalized": True,
        "patterns_before_stc": raw_patterns,
        "patterns_after_stc": final_patterns,
        "stc_removed_patterns": raw_patterns - final_patterns,
        "stc_shuffle_attempts": 0,
        "stc_coverage_preserved": True,
        "detected_collapsed_faults": fault_count,
        "detected_equivalent_faults": fault_count,
        "covered_equivalent_faults": fault_count,
        "uncollapsed_faults": fault_count,
        "fault_coverage": 1.0,
        "aborted_faults": 0,
        "redundant_faults": 0,
        "redundant_equivalent_faults": 0,
        "podem_calls": fault_count,
        "primary_podem_calls": fault_count,
        "dtc_secondary_calls": fault_count * (fault_count - 1) // 2,
        "primary_backtracks": 0,
        "dtc_backtracks": 0,
        "total_backtracks": 0,
    }


class _FakeSession:
    def __init__(self, identifiers, ranking_log):
        self.remaining_fault_ids = tuple(identifiers)
        self.initial = tuple(identifiers)
        self.ranking_log = ranking_log
        self.patterns = 0
        self.calls = 0
        self.dtc_calls = 0

    def step(self, primary, rank_dtc_candidates=None):
        assert primary in self.remaining_fault_ids
        candidates = tuple(identifier for identifier in self.remaining_fault_ids
                           if identifier != primary)
        metadata = {"unknown_po_id": "po0", "dtc_batch_index": 0,
                    "select_fault_try": 1, "visited_wire_count": 1}
        requested = (candidates if rank_dtc_candidates is None or not candidates
                     else tuple(rank_dtc_candidates(candidates, metadata)))
        assert set(requested) == set(candidates)
        self.ranking_log.append((primary, requested))
        attempted = requested
        self.patterns += 1
        self.calls += 1
        self.dtc_calls += len(attempted)
        self.remaining_fault_ids = tuple(
            identifier for identifier in self.remaining_fault_ids
            if identifier != primary)
        return {
            **_metrics(len(self.initial), self.patterns, self.patterns),
            "finalized": False,
            "stc_coverage_preserved": False,
            "selected_fault_id": primary,
            "target_status": "detected",
            "generated_pattern": True,
            "generated_test_vector": "0" * len(self.initial),
            "dtc_attempted_fault_ids": attempted,
            "dtc_embedded_fault_ids": (),
            "dtc_batches": ({
                **metadata,
                "bfs_candidate_fault_ids": candidates,
                "requested_fault_ids": requested,
                "executed_prefix_fault_ids": attempted,
                "embedded_fault_ids": (),
            },) if candidates else (),
            "newly_detected_fault_ids": (primary,),
            "remaining_fault_ids": self.remaining_fault_ids,
            "current_podem_calls": self.calls,
            "current_dtc_secondary_calls": self.dtc_calls,
            "current_primary_backtracks": 0,
            "current_dtc_backtracks": 0,
            "current_total_backtracks": 0,
            "primary_podem_calls": self.calls,
            "podem_calls": self.calls,
            "dtc_secondary_calls": self.dtc_calls,
        }

    def finish(self):
        return _metrics(len(self.initial), self.patterns)


class _FakeEnvironment:
    module = None

    def __init__(self, circuits):
        self.circuits = circuits
        self.ranking_log = []
        self.starts = []
        self.fail_once = set()

    def catalog(self, bench_path, faultmap_path):
        circuit = next(c for c in self.circuits
                       if c.spec.bench_path == bench_path)
        return {
            "faults": [
                {"fault_id": identifier, "eqv_fault_num": 1}
                for identifier in circuit.fault_ids],
            "uncollapsed_total": circuit.fault_count,
        }

    def start_session(self, bench_path, faultmap_path):
        name = Path(bench_path).stem
        self.starts.append(name)
        if name in self.fail_once:
            self.fail_once.remove(name)
            raise RuntimeError("injected circuit failure")
        circuit = next(c for c in self.circuits
                       if c.spec.bench_path == bench_path)
        return _FakeSession(circuit.fault_ids, self.ranking_log)


def _circuit(root, name, digest, count=3):
    spec = CircuitSpec(
        name=name,
        bench_path=root / (name + ".bench"),
        faultmap_path=None,
        embeddings_path=root / (name + ".npz"),
        metadata_path=root / (name + ".json"),
    )
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    return CircuitData(
        spec=spec,
        embeddings=torch.randn(count, 257, generator=generator),
        fault_ids=tuple(name + "_f" + str(index) for index in range(count)),
        eqv_fault_nums=np.ones(count, dtype=np.int64),
        artifact_digest=digest,
    )


def _manifest(path, names):
    path.write_text(json.dumps({
        "version": 1,
        "cpp_podem_dir": ".",
        "circuits": [{
            "name": name,
            "bench": name + ".bench",
            "embeddings": name + ".npz",
            "metadata": name + ".json",
        } for name in names],
    }), encoding="utf-8")
    return path


def test_best_key_is_coverage_first():
    first = {"coverage_shortfall": 1, "totals": {"pattern_count": 1}}
    second = {"coverage_shortfall": 0, "totals": {"pattern_count": 99}}
    assert evaluation_key(second) < evaluation_key(first)


def test_policy_rollout_uses_one_forward_per_primary(tmp_path):
    circuit = _circuit(tmp_path, "counted", "counted-digest", count=3)
    environment = _FakeEnvironment([circuit])

    class CountingModel(FaultActorCritic):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            return super().forward(features)

    model = CountingModel()
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainConfig()
    trainer.environment = environment
    trainer._check_result = lambda actual_circuit, result: None
    _, _, decisions, _ = trainer._run_policy(
        circuit, model, 1.0, False, environment)
    assert model.calls == len(decisions) == 3
    assert all("primary_row" in decision and "dtc_batches" in decision
               for decision in decisions)


def test_rollout_batch_ppo_uses_transition_minibatches_and_circuit_data(
        tmp_path, monkeypatch):
    from fault_order_rl import trainer as trainer_module

    counts = (90, 80, 100, 80)
    circuits = [
        _circuit(tmp_path, "ppo_" + str(index), "digest-" + str(index), count=1)
        for index in range(len(counts))
    ]
    for index, circuit in enumerate(circuits):
        circuit.embeddings[:, 0] = float(index + 1)
    rollouts = []
    raw_advantages = []
    owners = []
    for circuit, count in zip(circuits, counts):
        decisions = []
        for sample in range(count):
            advantage = float(len(raw_advantages) - 175)
            decisions.append({
                "remaining_rows": (0,),
                "primary_row": 0,
                "dtc_batches": (),
                "old_log_prob": 0.0,
                "return": 1.0,
                "advantage": advantage,
            })
            raw_advantages.append(advantage)
            owners.append(float(len(rollouts) + 1))
        rollouts.append({"circuit": circuit, "decisions": decisions})

    class TinyActorCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.seen = []

        def forward(self, features):
            self.seen.append(float(features[0, 0]))
            scores = features[:, 0] * self.weight
            value = features[:, 0].mean() * self.weight
            return scores, value

    permutations = [
        torch.roll(torch.arange(sum(counts)), shifts=epoch)
        for epoch in range(4)
    ]
    permutation_calls = []

    def fake_randperm(size):
        assert size == sum(counts)
        result = permutations[len(permutation_calls)]
        permutation_calls.append(result.clone())
        return result

    captured_advantages = []
    real_objective = ppo_objective

    def capture_objective(new_log_probs, old_log_probs, advantages, values,
                          returns, raw_entropies, normalized_entropies,
                          *args, **kwargs):
        captured_advantages.append(advantages.detach().clone())
        return real_objective(
            new_log_probs, old_log_probs, advantages, values, returns,
            raw_entropies, normalized_entropies, *args, **kwargs)

    monkeypatch.setattr(torch, "randperm", fake_randperm)
    monkeypatch.setattr(trainer_module, "ppo_objective", capture_objective)
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainConfig()
    model = TinyActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    summary = trainer._ppo_update(model, optimizer, rollouts)

    assert summary["circuits"] == [circuit.name for circuit in circuits]
    assert summary["circuit_count"] == 4
    assert summary["transition_count"] == 350
    assert summary["minibatch_count"] == 3
    assert summary["optimizer_steps"] == 12
    assert all(np.isfinite(summary[key]) for key in (
        "approx_kl", "clip_fraction", "gradient_norm_pre_clip",
        "critic_loss", "entropy_raw", "entropy_normalized",
        "choice_transition_fraction"))
    assert [minibatch["optimizer_offset"]
            for minibatch in summary["flat_minibatches"]] == list(range(1, 13))
    assert len(permutation_calls) == 4
    assert [
        minibatch["sample_count"]
        for minibatch in summary["epochs"][0]["minibatches"]
    ] == [128, 128, 94]
    expected_advantages = normalize_advantages(torch.tensor(raw_advantages))
    assert torch.allclose(
        torch.cat(captured_advantages[:3]), expected_advantages[permutations[0]])
    assert model.seen[:350] == [owners[index] for index in permutations[0]]


def test_rollout_batch_collects_four_circuits_before_optimizer_update(
        tmp_path, monkeypatch):
    train_circuits = [
        _circuit(tmp_path, "batch_" + str(index), "batch-" + str(index))
        for index in range(4)
    ]
    validation_circuits = [
        _circuit(tmp_path, "validation", "validation-digest")]
    train_env = _FakeEnvironment(train_circuits)
    validation_env = _FakeEnvironment(validation_circuits)
    train_manifest = _manifest(
        tmp_path / "train.json", [circuit.name for circuit in train_circuits])
    validation_manifest = _manifest(
        tmp_path / "validation.json", ["validation"])
    from fault_order_rl import trainer as trainer_module
    operations = []
    writer = _RecordingWriter(operations)
    real_save_checkpoint = trainer_module.save_checkpoint

    def recording_save_checkpoint(path, state):
        operations.append("checkpoint:" + Path(path).name)
        return real_save_checkpoint(path, state)

    monkeypatch.setattr(
        trainer_module, "load_all_circuits",
        lambda manifest, environment: environment.circuits)
    monkeypatch.setattr(
        trainer_module, "_create_summary_writer", lambda path: writer)
    monkeypatch.setattr(
        trainer_module, "save_checkpoint", recording_save_checkpoint)
    trainer = Trainer.create(
        train_manifest, TrainConfig(), tmp_path / "run", train_env,
        validation_manifest, validation_env)
    operations.clear()
    shuffled_names = [
        train_circuits[index].name
        for index in trainer._round_circuit_indices(1)
    ]
    snapshots = []
    update_entries = []
    real_run_policy = trainer._run_policy
    real_ppo_update = trainer._ppo_update

    def capture_rollout(circuit, model, *args, **kwargs):
        if circuit in train_circuits:
            snapshots.append(copy.deepcopy(model.state_dict()))
        return real_run_policy(circuit, model, *args, **kwargs)

    def capture_update(model, optimizer, rollouts):
        update_entries.append((len(snapshots), [
            rollout["circuit"].name for rollout in rollouts]))
        return real_ppo_update(model, optimizer, rollouts)

    monkeypatch.setattr(trainer, "_run_policy", capture_rollout)
    monkeypatch.setattr(trainer, "_ppo_update", capture_update)
    trainer.step()

    assert update_entries == [(4, shuffled_names)]
    assert len(snapshots) == 4
    assert all(state_dict_equal(snapshots[0], snapshot)
               for snapshot in snapshots[1:])
    ppo_calls = [call for call in writer.calls if call[0].startswith("PPO/")]
    assert sorted({step for _, _, step in ppo_calls}) == [1, 2, 3, 4]
    assert ("Rollout/transition_count", 12.0, 1) in writer.calls
    assert all(step == 1 for tag, _, step in writer.calls
               if tag.startswith("Validation/"))
    first_ppo_event = next(
        index for index, item in enumerate(operations)
        if item.startswith("event:PPO/"))
    first_validation_event = next(
        index for index, item in enumerate(operations)
        if item.startswith("event:Validation/"))
    latest_checkpoints = [
        index for index, item in enumerate(operations)
        if item == "checkpoint:latest.pt"]
    assert first_ppo_event > latest_checkpoints[0]
    assert first_validation_event > latest_checkpoints[1]


def test_checkpoint_failure_writes_no_tensorboard_events(tmp_path, monkeypatch):
    train_circuit = _circuit(tmp_path, "transaction_train", "train-digest")
    validation_circuit = _circuit(
        tmp_path, "transaction_validation", "validation-digest")
    train_env = _FakeEnvironment([train_circuit])
    validation_env = _FakeEnvironment([validation_circuit])
    train_manifest = _manifest(tmp_path / "train.json", [train_circuit.name])
    validation_manifest = _manifest(
        tmp_path / "validation.json", [validation_circuit.name])
    from fault_order_rl import trainer as trainer_module

    writer = _RecordingWriter()
    monkeypatch.setattr(
        trainer_module, "load_all_circuits",
        lambda manifest, environment: environment.circuits)
    monkeypatch.setattr(
        trainer_module, "_create_summary_writer", lambda path: writer)
    output = tmp_path / "run"
    trainer = Trainer.create(
        train_manifest, TrainConfig(), output, train_env,
        validation_manifest, validation_env)

    def fail_checkpoint(path, state):
        raise OSError("injected checkpoint failure")

    monkeypatch.setattr(trainer_module, "save_checkpoint", fail_checkpoint)
    with pytest.raises(OSError, match="checkpoint failure"):
        trainer.step()

    assert writer.calls == []
    assert trainer.global_optimizer_step == 0
    assert trainer.global_rollout_batch_step == 0
    committed = load_checkpoint(output / "latest.pt")
    assert committed["global_optimizer_step"] == 0
    assert committed["global_rollout_batch_step"] == 0


def test_validation_tensorboard_reduction_uses_ratio_of_totals():
    trainer = Trainer.__new__(Trainer)
    trainer.writer = _RecordingWriter()
    trainer.validation_native_metrics = {
        "small": {"pattern_count": 10},
        "large": {"pattern_count": 30},
    }
    report = {
        "totals": {
            "pattern_count": 20,
            "fault_coverage": 0.975,
            "covered_equivalent_faults": 390,
        },
        "coverage_shortfall": 2,
        "eligible": False,
    }

    trainer._write_validation_events(report, 3)

    assert ("Validation/pattern_reduction_pct_total", 50.0, 3) in (
        trainer.writer.calls)
    assert ("Validation/coverage_shortfall", 2.0, 3) in trainer.writer.calls
    assert ("Validation/coverage_eligible", 0.0, 3) in trainer.writer.calls


def test_nested_dtc_trajectory_npz_uses_offsets_without_pickle(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.output = tmp_path
    decisions = [{
        "remaining_rows": (0, 1, 2), "primary_row": 0,
        "dtc_batches": ({
            "bfs_candidate_rows": (2, 1), "requested_rows": (1, 2),
            "executed_prefix_rows": (1,), "embedded_rows": (),
            "select_fault_try": 1, "visited_wire_count": 1,
        },),
        "old_log_prob": -1.0, "old_value": 0.0, "reward": 0.1,
        "return": 0.2, "advantage": 0.3,
    }, {
        "remaining_rows": (1, 2), "primary_row": 2, "dtc_batches": (),
        "old_log_prob": -0.5, "old_value": 0.1, "reward": 0.0,
        "return": 0.0, "advantage": -0.1,
    }]
    trainer._write_circuit(1, 0, {"kind": "episode"}, decisions)
    path = tmp_path / "rounds" / "round-000001" / "circuit-000000.npz"
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["primary_rows"].tolist() == [0, 2]
        assert arrays["dtc_batch_step_offsets"].tolist() == [0, 1, 1]
        assert arrays["dtc_candidate_rows"].tolist() == [2, 1]
        assert arrays["dtc_candidate_rows_offsets"].tolist() == [0, 2]
        assert arrays["dtc_executed_rows"].tolist() == [1]
        assert arrays["dtc_executed_rows_offsets"].tolist() == [0, 1]


def test_schema_1_to_4_require_retraining(tmp_path):
    for version in (1, 2, 3, 4):
        path = tmp_path / (str(version) + ".pt")
        save_checkpoint(path, {"version": version})
        with pytest.raises(ValueError, match="retrain|retraining"):
            load_checkpoint(path)


def test_rollout_batch_checkpoint_resume_validation_best_and_final(
        tmp_path, monkeypatch):
    train_circuits = [
        _circuit(tmp_path, "train_a", "train-a"),
        _circuit(tmp_path, "train_b", "train-b"),
        _circuit(tmp_path, "train_c", "train-c"),
        _circuit(tmp_path, "train_d", "train-d"),
        _circuit(tmp_path, "train_e", "train-e"),
    ]
    validation_circuits = [
        _circuit(tmp_path, "validation_a", "validation-a")]
    train_env = _FakeEnvironment(train_circuits)
    validation_env = _FakeEnvironment(validation_circuits)
    train_manifest = _manifest(
        tmp_path / "train.json",
        ["train_a", "train_b", "train_c", "train_d", "train_e"])
    validation_manifest = _manifest(
        tmp_path / "validation.json", ["validation_a"])

    from fault_order_rl import trainer as trainer_module

    def fake_load_all(manifest, environment):
        return environment.circuits

    monkeypatch.setattr(trainer_module, "load_all_circuits", fake_load_all)
    monkeypatch.setattr(
        trainer_module, "_create_summary_writer",
        lambda path: _RecordingWriter())
    output = tmp_path / "run"
    trainer = Trainer.create(
        train_manifest, TrainConfig(), output, train_env,
        validation_manifest, validation_env)
    initial = load_checkpoint(output / "latest.pt")
    assert initial["version"] == 6
    assert initial["global_optimizer_step"] == 0
    assert initial["global_rollout_batch_step"] == 0
    assert initial["training_protocol"] == TRAINING_PROTOCOL
    legacy = copy.deepcopy(initial)
    legacy["version"] = 5
    legacy["config"]["entropy_coef"] = 0.01
    legacy["config"].pop("entropy_coef_normalized")
    legacy["training_protocol"] = "multi_circuit_ppo_batch_v1"
    save_checkpoint(output / "legacy-latest.pt", legacy)
    with pytest.raises(ValueError, match="schema 6"):
        Trainer.resume(
            output / "legacy-latest.pt", environment=train_env,
            validation_environment=validation_env)
    legacy_report = evaluate_checkpoint(
        output / "legacy-latest.pt", tmp_path / "legacy-evaluation",
        environment=validation_env)
    assert legacy_report["checkpoint_kind"] == "latest"
    round_order = trainer._round_circuit_indices(1)
    ordered_names = [train_circuits[index].name for index in round_order]
    train_env.fail_once.add(ordered_names[1])
    with pytest.raises(RuntimeError, match="injected"):
        trainer.step()
    committed = load_checkpoint(output / "latest.pt")
    assert committed["round"] == 0
    assert committed["next_circuit_index"] == 0
    assert not (output / "rounds" / "round-000001").exists()
    initial_parameters = copy.deepcopy(committed["model"])

    resumed = Trainer.resume(
        output / "latest.pt", environment=train_env,
        validation_environment=validation_env)
    assert resumed.next_circuit_index == 0
    assert resumed._round_circuit_indices(1) == round_order
    assert all(torch.equal(initial_parameters[key], resumed.model.state_dict()[key])
               for key in initial_parameters)
    train_env.fail_once.add(ordered_names[4])
    with pytest.raises(RuntimeError, match="injected"):
        resumed.step()
    first_batch = load_checkpoint(output / "latest.pt")
    assert first_batch["round"] == 0
    assert first_batch["next_circuit_index"] == 4
    assert first_batch["global_optimizer_step"] == 4
    assert first_batch["global_rollout_batch_step"] == 1
    for index in round_order[:4]:
        assert (output / "rounds" / "round-000001"
                / "circuit-{:06d}.json".format(index)).is_file()
    assert not (
        output / "rounds" / "round-000001"
        / "circuit-{:06d}.json".format(round_order[4])
    ).exists()

    resumed = Trainer.resume(
        output / "latest.pt", environment=train_env,
        validation_environment=validation_env)
    assert resumed.next_circuit_index == 4
    assert resumed.global_optimizer_step == 4
    assert resumed.global_rollout_batch_step == 1
    records = resumed.step()
    assert resumed.round == 1
    assert [record["circuit"] for record in records
            if record["kind"] == "episode"] == [ordered_names[4]]
    episode = next(record for record in records if record["kind"] == "episode")
    assert episode["rollout_batch_circuits"] == [ordered_names[4]]
    assert episode["rollout_batch_circuit_count"] == 1
    assert episode["rollout_batch_transition_count"] == 3
    assert episode["ppo_minibatch_count"] == 1
    assert episode["optimizer_steps"] == 4
    committed_round = load_checkpoint(output / "latest.pt")
    assert committed_round["global_optimizer_step"] == 8
    assert committed_round["global_rollout_batch_step"] == 2
    assert (output / "best.pt").is_file()
    best = load_checkpoint(output / "best.pt")
    assert best["kind"] == "best"
    assert best["policy"] == POLICY_IDENTITY
    assert "optimizer" not in best and "rng" not in best
    assert all(name.startswith("validation")
               for name in best["evaluation"]["circuits"])

    for _ in range(4):
        resumed.step()
    report = resumed.train()
    final = load_checkpoint(output / "final.pt")
    assert resumed.round == 5
    assert final["kind"] == "final"
    assert final["round"] == 5
    assert "optimizer" not in final and "rng" not in final
    assert report["checkpoint_kind"] == "best"
    best_report = evaluate_checkpoint(
        output / "best.pt", tmp_path / "evaluate-best",
        environment=validation_env)
    final_report = evaluate_checkpoint(
        output / "final.pt", tmp_path / "evaluate-final",
        environment=validation_env)
    assert best_report["checkpoint_kind"] == "best"
    assert final_report["checkpoint_kind"] == "final"


def test_training_validation_overlap_is_rejected(tmp_path, monkeypatch):
    circuit = _circuit(tmp_path, "same", "same-digest")
    train_env = _FakeEnvironment([circuit])
    validation_env = _FakeEnvironment([circuit])
    train_manifest = _manifest(tmp_path / "train.json", ["same"])
    validation_manifest = _manifest(tmp_path / "validation.json", ["same"])
    from fault_order_rl import trainer as trainer_module
    monkeypatch.setattr(
        trainer_module, "load_all_circuits",
        lambda manifest, environment: environment.circuits)
    with pytest.raises(ValueError, match="independent"):
        Trainer(
            train_manifest, TrainConfig(), tmp_path / "run", train_env,
            validation_manifest, validation_env)
