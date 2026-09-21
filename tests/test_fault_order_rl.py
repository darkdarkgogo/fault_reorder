"""Necessary tests for dynamic ranked-DTC actor-critic PPO."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fault_order_rl.checkpoint import load_checkpoint, save_checkpoint
from fault_order_rl.data import CircuitData, CircuitSpec
from fault_order_rl.environment import PROTOCOL_CONFIG, PodemEnvironment, PodemSession
from fault_order_rl.model import FaultActorCritic, build_dynamic_features
from fault_order_rl.policy import executed_prefix_stats, sample_ranking
from fault_order_rl.reward import (compute_gae, normalize_advantages,
                                   ppo_objective, step_reward,
                                   target_return, terminal_correction)
from fault_order_rl.trainer import (POLICY_IDENTITY, SOLVER_PROTOCOL,
                                    TrainConfig, Trainer, evaluate_checkpoint,
                                    evaluation_key)


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


def test_full_ranking_and_executed_prefix_have_gradients():
    model = FaultActorCritic()
    embeddings = torch.randn(4, 257)
    rows = (3, 0, 2)
    scores, _ = model(build_dynamic_features(embeddings, rows))
    scores = torch.zeros_like(scores)
    assert sample_ranking(scores, rows, 1.0, False).tolist() == [0, 2, 3]
    sampled = sample_ranking(
        scores, rows, 1.0, True,
        generator=torch.Generator().manual_seed(7))
    assert sorted(sampled.tolist()) == sorted(rows)
    log_prob, entropy, value = executed_prefix_stats(
        model, embeddings, rows, (0, 3), 1.0)
    loss = -(log_prob + 0.01 * entropy) + value.square()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


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
        values, returns, torch.tensor([0.5, 0.4]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in details.values())


def test_fixed_protocol_and_training_defaults():
    assert PROTOCOL_CONFIG["primary_backtrack_limit"] == 100
    assert PROTOCOL_CONFIG["dtc_secondary_backtrack_limit"] == 50
    assert SOLVER_PROTOCOL["primary_backtrack_limit"] == 100
    config = TrainConfig()
    config.validate()
    assert config.rounds == 5
    assert config.ppo_epochs == 4
    assert config.backtrack_limit == 100
    with pytest.raises(ValueError, match="100"):
        TrainConfig(backtrack_limit=200).validate()


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

    def step(self, primary, secondaries):
        self.seen = (primary, tuple(secondaries))
        return _step_summary(
            dtc_attempted_fault_ids=self.attempted,
            dtc_embedded_fault_ids=self.attempted,
            dtc_secondary_calls=len(self.attempted),
            current_dtc_secondary_calls=len(self.attempted))


def test_python_session_passes_full_ranking_and_accepts_only_prefix():
    native = _NativePrefixSession()
    session = PodemSession(native)
    result = session.step("f0", ("f2", "f1"))
    assert native.seen == ("f0", ("f2", "f1"))
    assert result["dtc_attempted_fault_ids"] == ("f2",)
    bad = PodemSession(_NativePrefixSession(("f1",)))
    with pytest.raises(RuntimeError, match="ranked prefix"):
        bad.step("f0", ("f2", "f1"))
    with pytest.raises(ValueError, match="every non-primary"):
        PodemSession(_NativePrefixSession()).step("f0", ("f1",))


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

    def step(self, primary, secondaries):
        assert primary in self.remaining_fault_ids
        assert set(secondaries) == set(self.remaining_fault_ids) - {primary}
        assert len(secondaries) == len(self.remaining_fault_ids) - 1
        self.ranking_log.append((primary, tuple(secondaries)))
        attempted = tuple(secondaries)
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


def test_schema_1_to_3_require_retraining(tmp_path):
    for version in (1, 2, 3):
        path = tmp_path / (str(version) + ".pt")
        save_checkpoint(path, {"version": version})
        with pytest.raises(ValueError, match="retrain|retraining"):
            load_checkpoint(path)


def test_per_circuit_checkpoint_resume_validation_best_and_final(
        tmp_path, monkeypatch):
    train_circuits = [
        _circuit(tmp_path, "train_a", "train-a"),
        _circuit(tmp_path, "train_b", "train-b"),
    ]
    validation_circuits = [
        _circuit(tmp_path, "validation_a", "validation-a")]
    train_env = _FakeEnvironment(train_circuits)
    validation_env = _FakeEnvironment(validation_circuits)
    train_manifest = _manifest(tmp_path / "train.json", ["train_a", "train_b"])
    validation_manifest = _manifest(
        tmp_path / "validation.json", ["validation_a"])

    from fault_order_rl import trainer as trainer_module

    def fake_load_all(manifest, environment):
        return environment.circuits

    monkeypatch.setattr(trainer_module, "load_all_circuits", fake_load_all)
    output = tmp_path / "run"
    trainer = Trainer.create(
        train_manifest, TrainConfig(), output, train_env,
        validation_manifest, validation_env)
    assert load_checkpoint(output / "latest.pt")["version"] == 4
    train_env.fail_once.add("train_b")
    with pytest.raises(RuntimeError, match="injected"):
        trainer.step()
    committed = load_checkpoint(output / "latest.pt")
    assert committed["round"] == 0
    assert committed["next_circuit_index"] == 1
    first_parameters = copy.deepcopy(committed["model"])

    resumed = Trainer.resume(
        output / "latest.pt", environment=train_env,
        validation_environment=validation_env)
    assert resumed.next_circuit_index == 1
    assert all(torch.equal(first_parameters[key], resumed.model.state_dict()[key])
               for key in first_parameters)
    records = resumed.step()
    assert resumed.round == 1
    assert [record["circuit"] for record in records
            if record["kind"] == "episode"] == ["train_b"]
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
