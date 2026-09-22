"""Dynamic BFS-filtered ranked-DTC actor-critic PPO training."""

import copy
import csv
from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from .checkpoint import (atomic_write, capture_rng, load_checkpoint, restore_rng,
                         save_checkpoint, write_json, write_npz)
from .data import _sha256, load_all_circuits, load_manifest
from .environment import PROTOCOL_CONFIG, PodemEnvironment
from .model import FaultActorCritic, build_dynamic_features
from .policy import (joint_action_stats, joint_action_stats_from_scores,
                     sample_candidate_ranking, sample_primary)
from .progress import progress, sample_progress
from .reward import (compute_gae, normalize_advantages, ppo_objective,
                     step_reward, target_return, terminal_correction)


POLICY_IDENTITY = "dynamic_bfs_ranked_dtc_actor_critic_ppo_v2"
SOLVER_PROTOCOL = {
    **PROTOCOL_CONFIG,
    "compression_algorithm_version": "stuck_at_podemx_bfs_ranked_dtc_v3",
}
DEFAULT_VALIDATION_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "anchor_validation_6.json"
)


@dataclass
class TrainConfig:
    rounds: int = 5
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    alpha: float = 0.1
    beta: float = 10.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    temperature: float = 1.0
    seed: int = 14
    backtrack_limit: int = 100
    threads: int = 1

    def validate(self):
        for key in ("rounds", "ppo_epochs", "backtrack_limit", "threads"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(key + " must be a positive integer")
        for key in ("learning_rate", "gradient_clip", "beta", "temperature"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + " must be finite and positive")
        for key in ("alpha", "value_coef", "entropy_coef"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(key + " must be finite and non-negative")
        for key in ("gamma", "gae_lambda", "ppo_clip"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(key + " must be in [0, 1]")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        if self.backtrack_limit != 100:
            raise ValueError("the production protocol requires backtrack_limit=100")
        if self.rounds != 5:
            raise ValueError("the strict training protocol requires exactly 5 rounds")


def eligible(report, states):
    return report is not None and all(
        report["circuits"][name]["covered_equivalent_faults"]
        >= state["native_covered_equivalent_faults"]
        for name, state in states.items()
    )


def evaluation_key(report):
    """Coverage-first best-model key required by the strict design."""
    return (report["coverage_shortfall"], report["totals"]["pattern_count"])


def state_dict_equal(first, second):
    return first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first)


def _evaluation_totals(results):
    keys = (
        "pattern_count", "detected_equivalent_faults", "detected_collapsed_faults",
        "redundant_equivalent_faults", "covered_equivalent_faults",
        "uncollapsed_faults", "podem_calls", "total_backtracks",
        "aborted_faults", "redundant_faults",
    )
    totals = {key: sum(result[key] for result in results.values()) for key in keys}
    totals["fault_coverage"] = (
        totals["covered_equivalent_faults"] / totals["uncollapsed_faults"]
        if totals["uncollapsed_faults"] else 0.0
    )
    return totals


def _coverage_summary(report, states):
    shortfalls = {
        name: max(0, state["native_covered_equivalent_faults"]
                  - report["circuits"][name]["covered_equivalent_faults"])
        for name, state in states.items()
    }
    report["coverage_shortfall_by_circuit"] = shortfalls
    report["coverage_shortfall"] = sum(shortfalls.values())
    report["eligible"] = report["coverage_shortfall"] == 0
    return report


def _evaluation_export(circuit, scores, order, decisions, trace, metrics):
    count = circuit.fault_count
    ranks = np.empty(count, dtype=np.int64)
    ranks[order] = np.arange(1, count + 1)
    selected_steps = np.full(count, -1, dtype=np.int64)
    selected_scores = np.full(count, np.nan, dtype=np.float32)
    exit_steps = np.full(count, -1, dtype=np.int64)
    exit_reasons = np.full(count, "", dtype="<U32")
    for step_index, (decision, event) in enumerate(zip(decisions, trace)):
        selected = decision["primary_row"]
        selected_steps[selected] = step_index
        selected_scores[selected] = event["selected_score"]
        before = set(decision["remaining_rows"])
        after = (set(decisions[step_index + 1]["remaining_rows"])
                 if step_index + 1 < len(decisions) else set())
        for row in before - after:
            exit_steps[row] = step_index
            exit_reasons[row] = (event["target_status"]
                                 if row == selected else "fault_sim_drop")
    return {
        "fault_ids": np.asarray(circuit.fault_ids),
        "scores": scores.detach().cpu().numpy(),
        "ranks": ranks,
        "permutation": order,
        "selected_rows": np.asarray(
            [decision["primary_row"] for decision in decisions],
            dtype=np.int64),
        "selected_scores": selected_scores,
        "selected_steps": selected_steps,
        "exit_steps": exit_steps,
        "exit_reasons": exit_reasons,
        "patterns_before_stc": np.asarray(metrics["patterns_before_stc"], dtype=np.int64),
        "patterns_after_stc": np.asarray(metrics["patterns_after_stc"], dtype=np.int64),
        "primary_podem_calls": np.asarray(metrics["primary_podem_calls"], dtype=np.int64),
        "dtc_secondary_calls": np.asarray(metrics["dtc_secondary_calls"], dtype=np.int64),
        "primary_backtracks": np.asarray(metrics["primary_backtracks"], dtype=np.int64),
        "dtc_backtracks": np.asarray(metrics["dtc_backtracks"], dtype=np.int64),
        "total_backtracks": np.asarray(metrics["total_backtracks"], dtype=np.int64),
        "_trajectory": trace,
    }


class Trainer:
    def __init__(self, manifest, config, output, environment=None,
                 validation_manifest=None, validation_environment=None):
        config.validate()
        self.config = config
        self.manifest = load_manifest(manifest)
        self.environment = environment or PodemEnvironment(
            self.manifest.module_dir, config.backtrack_limit, 14)
        self.circuits = load_all_circuits(self.manifest, self.environment)
        self.output = Path(output).resolve()
        self.provenance = {c.name: c.artifact_digest for c in self.circuits}

        self.validation_manifest = None
        self.validation_environment = None
        self.validation_circuits = []
        self.validation_provenance = {}
        if validation_manifest is not None:
            self.validation_manifest = load_manifest(validation_manifest)
            if self.validation_manifest.path == self.manifest.path:
                raise ValueError("training and validation manifests must be different")
            self.validation_environment = validation_environment or environment or PodemEnvironment(
                self.validation_manifest.module_dir, config.backtrack_limit, 14)
            self.validation_circuits = load_all_circuits(
                self.validation_manifest, self.validation_environment)
            self.validation_provenance = {
                c.name: c.artifact_digest for c in self.validation_circuits
            }
            overlap = set(self.provenance.values()) & set(
                self.validation_provenance.values())
            training_paths = {
                path.resolve()
                for circuit in self.circuits
                for path in (
                    circuit.spec.bench_path, circuit.spec.faultmap_path,
                    circuit.spec.embeddings_path, circuit.spec.metadata_path,
                    circuit.spec.aig_bench_path, circuit.spec.aigmap_path)
                if path is not None
            }
            validation_paths = {
                path.resolve()
                for circuit in self.validation_circuits
                for path in (
                    circuit.spec.bench_path, circuit.spec.faultmap_path,
                    circuit.spec.embeddings_path, circuit.spec.metadata_path,
                    circuit.spec.aig_bench_path, circuit.spec.aigmap_path)
                if path is not None
            }
            if overlap or training_paths & validation_paths:
                raise ValueError("training and validation artifacts must be independent")

        module_path = getattr(getattr(self.environment, "module", None), "__file__", None)
        self.solver_digest = _sha256(module_path) if module_path else None
        validation_module_path = getattr(
            getattr(self.validation_environment, "module", None), "__file__", None)
        self.validation_solver_digest = (
            _sha256(validation_module_path) if validation_module_path else None)
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        self.model = FaultActorCritic()
        self.optimizer = self._optimizer(self.model)
        self.round = 0
        self.next_circuit_index = 0
        self.states = {}
        self.validation_states = {}
        self.native_metrics = {}
        self.validation_native_metrics = {}
        self.best = None
        self.run_id = uuid.uuid4().hex

    def _optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    @classmethod
    def create(cls, manifest, config, output, environment=None,
               validation_manifest=None, validation_environment=None):
        output = Path(output).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory is not empty; use --resume or a new directory")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        validation_manifest = (
            DEFAULT_VALIDATION_MANIFEST if validation_manifest is None
            else validation_manifest)
        self = cls(manifest, config, output, environment, validation_manifest,
                   validation_environment)
        for circuit in self.circuits:
            progress("BASELINE", "start", split="train", circuit=circuit.name)
            metrics, elapsed = self._run_native(
                circuit, self.environment, "baseline-train")
            progress(
                "BASELINE", "done", split="train", circuit=circuit.name,
                elapsed_s="{:.3f}".format(elapsed),
            )
            self.native_metrics[circuit.name] = metrics
            self.states[circuit.name] = {
                "native_covered_equivalent_faults": metrics[
                    "covered_equivalent_faults"]
            }
        for circuit in self.validation_circuits:
            progress("BASELINE", "start", split="validation", circuit=circuit.name)
            metrics, elapsed = self._run_native(
                circuit, self.validation_environment, "baseline-validation")
            progress(
                "BASELINE", "done", split="validation", circuit=circuit.name,
                elapsed_s="{:.3f}".format(elapsed),
            )
            self.validation_native_metrics[circuit.name] = metrics
            self.validation_states[circuit.name] = {
                "native_covered_equivalent_faults": metrics[
                    "covered_equivalent_faults"]
            }
        save_checkpoint(self.output / "latest.pt", self._payload())
        return self

    @classmethod
    def resume(cls, checkpoint, rounds=None, environment=None,
               validation_environment=None):
        checkpoint = Path(checkpoint).resolve()
        saved = load_checkpoint(checkpoint)
        if saved.get("kind") != "latest":
            raise ValueError("resume requires a latest training checkpoint")
        config = TrainConfig(**saved["config"])
        if rounds is not None:
            if rounds != config.rounds:
                raise ValueError("resume cannot change the fixed five-round target")
        self = cls(
            saved["manifest_path"], config, checkpoint.parent, environment,
            saved["validation_manifest_path"], validation_environment)
        self._check_compatibility(saved)
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.round = saved["round"]
        self.next_circuit_index = saved["next_circuit_index"]
        self.states = saved["states"]
        self.validation_states = saved["validation_states"]
        self.native_metrics = saved["native_metrics"]
        self.validation_native_metrics = saved["validation_native_metrics"]
        self.best = saved["best"]
        self.run_id = saved["run_id"]
        restore_rng(saved["rng"])
        if self.best is not None:
            self._publish_best()
        if self.round == self.config.rounds and self.next_circuit_index == 0:
            self._publish_final()
        return self

    def _check_compatibility(self, saved):
        checks = (
            (saved["manifest_digest"], self.manifest.digest,
             "checkpoint training manifest changed"),
            (saved["artifacts"], self.provenance,
             "checkpoint training artifacts changed"),
            (saved["training_circuit_order"],
             [circuit.name for circuit in self.circuits],
             "checkpoint training circuit order changed"),
            (saved["validation_manifest_digest"], self.validation_manifest.digest,
             "checkpoint validation manifest changed"),
            (saved["validation_artifacts"], self.validation_provenance,
             "checkpoint validation artifacts changed"),
            (saved["solver_digest"], self.solver_digest,
             "checkpoint PODEM binary changed"),
            (saved["validation_solver_digest"], self.validation_solver_digest,
             "checkpoint validation PODEM binary changed"),
            (saved["torch_version"], str(torch.__version__),
             "checkpoint PyTorch version changed"),
            (saved.get("policy"), POLICY_IDENTITY,
             "checkpoint policy identity changed"),
            (saved.get("solver_protocol"), SOLVER_PROTOCOL,
             "checkpoint solver protocol changed"),
        )
        for actual, expected, message in checks:
            if actual != expected:
                raise ValueError(message)

    def _check_result(self, circuit, result):
        if result["uncollapsed_faults"] != int(circuit.eqv_fault_nums.sum()):
            raise RuntimeError("PODEM fault total differs from validated embeddings")
        if result.get("stc_coverage_preserved") is not True:
            raise RuntimeError("PODEM STC coverage was not preserved")
        if result["patterns_after_stc"] > result["uncollapsed_faults"]:
            raise RuntimeError("PODEM pattern count exceeds initial equivalent faults")

    def _run_native(self, circuit, environment, mode="baseline"):
        start = time.perf_counter()
        session_started = time.perf_counter()
        progress("NATIVE", "session start", circuit=circuit.name, mode=mode)
        session = environment.start_session(
            circuit.spec.bench_path, circuit.spec.faultmap_path)
        progress(
            "NATIVE", "session ready", circuit=circuit.name, mode=mode,
            selectable=len(session.remaining_fault_ids),
            elapsed_s="{:.3f}".format(time.perf_counter() - session_started),
        )
        step_number = 0
        while session.remaining_fault_ids:
            step_number += 1
            ranking = tuple(session.remaining_fault_ids)
            sampled = sample_progress(step_number, 5, 100)
            if sampled:
                progress(
                    "ATPG", "step start", circuit=circuit.name, mode=mode,
                    step=step_number, selectable=len(ranking), primary=ranking[0],
                )
            step_started = time.perf_counter()
            session.step(ranking[0])
            if sampled:
                progress(
                    "ATPG", "step done", circuit=circuit.name, mode=mode,
                    step=step_number, remaining=len(session.remaining_fault_ids),
                    elapsed_s="{:.3f}".format(time.perf_counter() - step_started),
                )
        finalize_started = time.perf_counter()
        progress("STC", "finalize start", circuit=circuit.name, mode=mode)
        result = session.finish()
        progress(
            "STC", "finalize done", circuit=circuit.name, mode=mode,
            patterns=result.get("patterns_after_stc", result.get("pattern_count", "unknown")),
            elapsed_s="{:.3f}".format(time.perf_counter() - finalize_started),
        )
        self._check_result(circuit, result)
        return result, time.perf_counter() - start

    def _run_policy(self, circuit, model, temperature, stochastic,
                    environment=None):
        environment = environment or self.environment
        start = time.perf_counter()
        session_started = time.perf_counter()
        progress("NATIVE", "session start", circuit=circuit.name, mode="policy")
        session = environment.start_session(
            circuit.spec.bench_path, circuit.spec.faultmap_path)
        progress(
            "NATIVE", "session ready", circuit=circuit.name, mode="policy",
            selectable=len(session.remaining_fault_ids),
            elapsed_s="{:.3f}".format(time.perf_counter() - session_started),
        )
        row_by_id = {
            identifier: row for row, identifier in enumerate(circuit.fault_ids)}
        eqv_by_id = {
            identifier: int(circuit.eqv_fault_nums[row])
            for row, identifier in enumerate(circuit.fault_ids)}
        initial_eqv = int(circuit.eqv_fault_nums.sum())
        previous_patterns = 0
        decisions, trace = [], []
        step_number = 0
        while session.remaining_fault_ids:
            step_number += 1
            try:
                rows = tuple(row_by_id[identifier]
                             for identifier in session.remaining_fault_ids)
            except KeyError as exc:
                raise RuntimeError(
                    "PODEM remaining fault is absent from embeddings") from exc
            features = build_dynamic_features(circuit.embeddings, rows)
            scores, value = model(features)
            primary_row = int(sample_primary(
                scores, rows, temperature, stochastic))
            primary_id = circuit.fault_ids[primary_row]
            def rank_batch(candidate_ids, _metadata):
                try:
                    candidate_rows = tuple(row_by_id[identifier]
                                           for identifier in candidate_ids)
                except KeyError as exc:
                    raise RuntimeError(
                        "PODEM DTC candidate is absent from embeddings") from exc
                if (primary_row in candidate_rows
                        or not set(candidate_rows) <= set(rows) - {primary_row}):
                    raise RuntimeError(
                        "PODEM DTC candidate is outside the step-start remaining set")
                ranking = sample_candidate_ranking(
                    scores, rows, candidate_rows, temperature, stochastic)
                return tuple(circuit.fault_ids[int(row)]
                             for row in ranking.tolist())
            sampled = sample_progress(step_number, 5, 100)
            if sampled:
                progress(
                    "ATPG", "step start", circuit=circuit.name, mode="policy",
                    step=step_number, selectable=len(rows),
                    primary=primary_id,
                )
            step_started = time.perf_counter()
            step = session.step(primary_id, rank_batch)
            if sampled:
                progress(
                    "ATPG", "step done", circuit=circuit.name, mode="policy",
                    step=step_number, remaining=len(session.remaining_fault_ids),
                    elapsed_s="{:.3f}".format(time.perf_counter() - step_started),
                )
            dtc_batches = tuple({
                "unknown_po_id": batch["unknown_po_id"],
                "bfs_candidate_rows": tuple(
                    row_by_id[identifier]
                    for identifier in batch["bfs_candidate_fault_ids"]),
                "requested_rows": tuple(
                    row_by_id[identifier]
                    for identifier in batch["requested_fault_ids"]),
                "executed_prefix_rows": tuple(
                    row_by_id[identifier]
                    for identifier in batch["executed_prefix_fault_ids"]),
                "embedded_rows": tuple(
                    row_by_id[identifier]
                    for identifier in batch["embedded_fault_ids"]),
                "select_fault_try": batch["select_fault_try"],
                "visited_wire_count": batch["visited_wire_count"],
            } for batch in step["dtc_batches"])
            with torch.no_grad():
                old_log_prob, entropy, cached_value = joint_action_stats_from_scores(
                    scores, value, rows, primary_row, dtc_batches, temperature)
            pattern_increment = step["current_pattern_count"] - previous_patterns
            newly_detected_eqv = sum(
                eqv_by_id[identifier]
                for identifier in step["newly_detected_fault_ids"])
            reward = step_reward(
                pattern_increment, newly_detected_eqv, initial_eqv,
                self.config.alpha)
            selected_local = rows.index(primary_row)
            decisions.append({
                "remaining_rows": rows,
                "primary_row": primary_row,
                "dtc_batches": dtc_batches,
                "old_log_prob": float(old_log_prob),
                "old_value": float(cached_value),
                "entropy": float(entropy),
                "reward": reward,
                "pattern_increment": pattern_increment,
                "newly_detected_eqv": newly_detected_eqv,
            })
            trace.append({
                "selected_fault_id": primary_id,
                "selected_row": primary_row,
                "selected_score": float(scores[selected_local].detach()),
                "dtc_batches": step["dtc_batches"],
                "remaining_count": len(rows),
                "remaining_ratio": len(rows) / circuit.fault_count,
                "target_status": step["target_status"],
                "generated_test_vector": step["generated_test_vector"],
                "dtc_attempted_fault_ids": step["dtc_attempted_fault_ids"],
                "dtc_embedded_fault_ids": step["dtc_embedded_fault_ids"],
                "newly_detected_fault_ids": step["newly_detected_fault_ids"],
                "pattern_increment": pattern_increment,
                "newly_detected_eqv": newly_detected_eqv,
                "step_reward": reward,
                "primary_podem_calls": step["primary_podem_calls"],
                "dtc_secondary_calls": step["dtc_secondary_calls"],
                "primary_backtracks": step["primary_backtracks"],
                "dtc_backtracks": step["dtc_backtracks"],
                "total_backtracks": step["total_backtracks"],
                "current_pattern_count": step["current_pattern_count"],
            })
            previous_patterns = step["current_pattern_count"]
        finalize_started = time.perf_counter()
        progress("STC", "finalize start", circuit=circuit.name, mode="policy")
        result = session.finish()
        progress(
            "STC", "finalize done", circuit=circuit.name, mode="policy",
            patterns=result.get("patterns_after_stc", result.get("pattern_count", "unknown")),
            elapsed_s="{:.3f}".format(time.perf_counter() - finalize_started),
        )
        self._check_result(circuit, result)
        return result, time.perf_counter() - start, decisions, trace

    def _finish_trajectory(self, circuit, metrics, decisions, baseline):
        shortfall = max(
            0, baseline["native_covered_equivalent_faults"]
            - metrics["covered_equivalent_faults"])
        target = target_return(
            metrics["patterns_after_stc"], int(circuit.eqv_fault_nums.sum()),
            shortfall, self.config.beta)
        correction = terminal_correction(
            [decision["reward"] for decision in decisions], target)
        decisions[-1]["reward"] += correction
        values = [decision["old_value"] for decision in decisions]
        returns, advantages = compute_gae(
            [decision["reward"] for decision in decisions], values,
            self.config.gamma, self.config.gae_lambda)
        for decision, return_value, advantage in zip(
                decisions, returns.tolist(), advantages.tolist()):
            decision["return"] = return_value
            decision["advantage"] = advantage
        return {
            "coverage_shortfall": shortfall,
            "target_return": target,
            "terminal_correction": correction,
            "reward_sum": sum(decision["reward"] for decision in decisions),
        }

    def _ppo_update(self, model, optimizer, circuit, decisions):
        old_log_probs = torch.tensor(
            [decision["old_log_prob"] for decision in decisions],
            dtype=torch.float32)
        returns = torch.tensor(
            [decision["return"] for decision in decisions], dtype=torch.float32)
        advantages = normalize_advantages(torch.tensor(
            [decision["advantage"] for decision in decisions],
            dtype=torch.float32))
        epochs = []
        model.train()
        for epoch in range(self.config.ppo_epochs):
            new_log_probs, entropies, values = [], [], []
            for decision in decisions:
                log_prob, entropy, value = joint_action_stats(
                    model, circuit.embeddings, decision["remaining_rows"],
                    decision["primary_row"], decision["dtc_batches"],
                    self.config.temperature)
                new_log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
            loss, details = ppo_objective(
                torch.stack(new_log_probs), old_log_probs, advantages,
                torch.stack(values), returns, torch.stack(entropies),
                self.config.ppo_clip, self.config.value_coef,
                self.config.entropy_coef)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.gradient_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite PPO gradient")
            optimizer.step()
            if not all(torch.isfinite(parameter).all()
                       for parameter in model.parameters()):
                raise RuntimeError("non-finite model after PPO update")
            epochs.append({
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(grad_norm),
                **{key: float(value.detach())
                   for key, value in details.items()},
            })
        return epochs

    def evaluate_model(self, model, round_number, circuits=None,
                       environment=None, states=None):
        circuits = self.validation_circuits if circuits is None else circuits
        environment = self.validation_environment if environment is None else environment
        states = self.validation_states if states is None else states
        results, exports = {}, {}
        model.eval()
        with torch.no_grad():
            for circuit in circuits:
                metrics, elapsed, decisions, trace = self._run_policy(
                    circuit, model, self.config.temperature, False, environment)
                metrics = dict(metrics, seconds=elapsed)
                results[circuit.name] = metrics
                first_rows = tuple(range(circuit.fault_count))
                scores, _ = model(build_dynamic_features(
                    circuit.embeddings, first_rows))
                order = sample_candidate_ranking(
                    scores, first_rows, first_rows, self.config.temperature,
                    False).detach().cpu().numpy()
                exports[circuit.name] = _evaluation_export(
                    circuit, scores, order, decisions, trace, metrics)
        report = {
            "round": round_number,
            "circuits": results,
            "totals": _evaluation_totals(results),
        }
        return _coverage_summary(report, states), exports

    @staticmethod
    def _choose_best(best, model, report):
        if best is None or evaluation_key(report) < evaluation_key(best["report"]):
            return {
                "model": copy.deepcopy(model.state_dict()),
                "report": copy.deepcopy(report),
                "key": evaluation_key(report),
            }
        return best

    def _record(self, circuit, circuit_index, round_number, metrics, elapsed,
                trajectory, decisions, reward_summary, ppo_epochs):
        advantages = np.asarray(
            [decision["advantage"] for decision in decisions],
            dtype=np.float64)
        return {
            "kind": "episode",
            "circuit": circuit.name,
            "circuit_index": circuit_index,
            "round": round_number,
            "episode_steps": len(decisions),
            "initial_equivalent_faults": int(circuit.eqv_fault_nums.sum()),
            "seed": 14,
            "training_seed": self.config.seed,
            "checkpoint_identity": self.run_id + ":" + str(round_number),
            "seconds": elapsed,
            "trajectory": trajectory,
            "ppo_epochs": ppo_epochs,
            "coverage_valid": reward_summary["coverage_shortfall"] == 0,
            "episode_return": reward_summary["target_return"],
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            **reward_summary,
            **metrics,
        }

    def step(self):
        if self.round >= self.config.rounds:
            raise ValueError("training already reached the configured round target")
        number = self.round + 1
        records = []
        for index in range(self.next_circuit_index, len(self.circuits)):
            rng_before = capture_rng()
            circuit = self.circuits[index]
            try:
                candidate = copy.deepcopy(self.model)
                optimizer = self._optimizer(candidate)
                optimizer.load_state_dict(copy.deepcopy(
                    self.optimizer.state_dict()))
                candidate.eval()
                with torch.no_grad():
                    metrics, elapsed, decisions, trace = self._run_policy(
                        circuit, candidate, self.config.temperature, True,
                        self.environment)
                reward_summary = self._finish_trajectory(
                    circuit, metrics, decisions, self.states[circuit.name])
                ppo_epochs = self._ppo_update(
                    candidate, optimizer, circuit, decisions)
                record = self._record(
                    circuit, index, number, metrics, elapsed, trace,
                    decisions, reward_summary, ppo_epochs)
                self._write_circuit(number, index, record, decisions)
                payload = self._payload(
                    model=candidate, optimizer=optimizer,
                    next_circuit_index=index + 1)
                save_checkpoint(self.output / "latest.pt", payload)
            except BaseException:
                restore_rng(rng_before)
                raise
            self.model = candidate
            self.optimizer = optimizer
            self.next_circuit_index = index + 1
            records.append(record)

        report, exports = self.evaluate_model(self.model, number)
        best = self._choose_best(self.best, self.model, report)
        payload = self._payload(
            round_number=number, next_circuit_index=0, best=best)
        self._write_validation(number, report, exports)
        save_checkpoint(self.output / "latest.pt", payload)
        self.round = number
        self.next_circuit_index = 0
        self.best = best
        self._publish_best()
        return records + [{"kind": "validation", "report": report}]

    def _payload(self, model=None, optimizer=None, round_number=None,
                 next_circuit_index=None, best=None):
        model = self.model if model is None else model
        optimizer = self.optimizer if optimizer is None else optimizer
        return {
            "version": 5,
            "kind": "latest",
            "run_id": self.run_id,
            "policy": POLICY_IDENTITY,
            "input_dimension": 515,
            "solver_protocol": copy.deepcopy(SOLVER_PROTOCOL),
            "manifest_path": str(self.manifest.path),
            "manifest_digest": self.manifest.digest,
            "artifacts": self.provenance,
            "validation_manifest_path": str(self.validation_manifest.path),
            "validation_manifest_digest": self.validation_manifest.digest,
            "validation_artifacts": self.validation_provenance,
            "solver_digest": self.solver_digest,
            "validation_solver_digest": self.validation_solver_digest,
            "torch_version": str(torch.__version__),
            "config": asdict(self.config),
            "round": self.round if round_number is None else round_number,
            "active_round": (
                (self.round if round_number is None else round_number) + 1
                if (self.round if round_number is None else round_number)
                < self.config.rounds else None),
            "next_circuit_index": (
                self.next_circuit_index if next_circuit_index is None
                else next_circuit_index),
            "training_circuit_order": [
                circuit.name for circuit in self.circuits],
            "validation_circuit_order": [
                circuit.name for circuit in self.validation_circuits],
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "states": copy.deepcopy(self.states),
            "validation_states": copy.deepcopy(self.validation_states),
            "native_metrics": copy.deepcopy(self.native_metrics),
            "validation_native_metrics": copy.deepcopy(
                self.validation_native_metrics),
            "rng": capture_rng(),
            "best": copy.deepcopy(self.best if best is None else best),
        }

    def _write_circuit(self, number, index, record, decisions):
        directory = self.output / "rounds" / "round-{:06d}".format(number)
        stem = directory / "circuit-{:06d}".format(index)
        arrays = {}
        remaining_values, remaining_offsets = [], [0]
        for decision in decisions:
            remaining_values.extend(decision["remaining_rows"])
            remaining_offsets.append(len(remaining_values))
        arrays["remaining_rows"] = np.asarray(remaining_values, dtype=np.int64)
        arrays["remaining_rows_offsets"] = np.asarray(
            remaining_offsets, dtype=np.int64)
        arrays["primary_rows"] = np.asarray(
            [decision["primary_row"] for decision in decisions], dtype=np.int64)

        batches = []
        batch_step_offsets = [0]
        for decision in decisions:
            batches.extend(decision["dtc_batches"])
            batch_step_offsets.append(len(batches))
        arrays["dtc_batch_step_offsets"] = np.asarray(
            batch_step_offsets, dtype=np.int64)
        for output_name, batch_key in (
                ("dtc_candidate_rows", "bfs_candidate_rows"),
                ("dtc_requested_rows", "requested_rows"),
                ("dtc_executed_rows", "executed_prefix_rows"),
                ("dtc_embedded_rows", "embedded_rows")):
            values, offsets = [], [0]
            for batch in batches:
                values.extend(batch[batch_key])
                offsets.append(len(values))
            arrays[output_name] = np.asarray(values, dtype=np.int64)
            arrays[output_name + "_offsets"] = np.asarray(
                offsets, dtype=np.int64)
        arrays["dtc_select_fault_try"] = np.asarray(
            [batch["select_fault_try"] for batch in batches], dtype=np.int64)
        arrays["dtc_visited_wire_count"] = np.asarray(
            [batch["visited_wire_count"] for batch in batches], dtype=np.int64)
        arrays["old_log_probs"] = np.asarray(
            [decision["old_log_prob"] for decision in decisions], dtype=np.float32)
        arrays["old_values"] = np.asarray(
            [decision["old_value"] for decision in decisions], dtype=np.float32)
        arrays["rewards"] = np.asarray(
            [decision["reward"] for decision in decisions], dtype=np.float32)
        arrays["returns"] = np.asarray(
            [decision["return"] for decision in decisions], dtype=np.float32)
        arrays["advantages"] = np.asarray(
            [decision["advantage"] for decision in decisions], dtype=np.float32)
        write_npz(stem.with_suffix(".npz"), **arrays)
        write_json(stem.with_suffix(".json"), record)

    def _write_validation(self, number, report, exports):
        output = self.output / "validation" / "round-{:06d}".format(number)
        validation_report = _add_native_comparison(
            report, self.validation_native_metrics,
            self.validation_states, "validation")
        validation_report["validation_manifest"] = str(
            self.validation_manifest.path)
        validation_report["validation_manifest_digest"] = (
            self.validation_manifest.digest)
        _write_evaluation(output, validation_report, exports)

    def _derived_payload(self, kind, model_state, round_number, evaluation):
        payload = self._payload()
        payload.pop("optimizer")
        payload.pop("rng")
        payload.pop("best")
        payload["kind"] = kind
        payload["model"] = copy.deepcopy(model_state)
        payload["round"] = round_number
        payload["next_circuit_index"] = 0
        payload["evaluation"] = copy.deepcopy(evaluation)
        payload["validation_key"] = evaluation_key(evaluation)
        return payload

    def _publish_best(self):
        if self.best is None:
            return
        payload = self._derived_payload(
            "best", self.best["model"], self.best["report"]["round"],
            self.best["report"])
        save_checkpoint(self.output / "best.pt", payload)

    def _publish_final(self, report=None):
        if report is None:
            report, _ = self.evaluate_model(self.model, self.round)
        payload = self._derived_payload(
            "final", self.model.state_dict(), self.round, report)
        save_checkpoint(self.output / "final.pt", payload)

    def train(self):
        while self.round < self.config.rounds:
            records = self.step()
            episodes = [record for record in records
                        if record["kind"] == "episode"]
            validation = records[-1]["report"]
            print(
                "round {}/{}: train_patterns={}, validation_shortfall={}, "
                "validation_patterns={}".format(
                    self.round, self.config.rounds,
                    sum(record["pattern_count"] for record in episodes),
                    validation["coverage_shortfall"],
                    validation["totals"]["pattern_count"]),
                flush=True)
        final_report, _ = self.evaluate_model(self.model, self.round)
        self._publish_final(final_report)
        best_report = copy.deepcopy(self.best["report"])
        return _complete_report(
            best_report, self.output / "best.pt",
            self.validation_native_metrics, self.validation_states, "best")


def _add_native_comparison(report, native_metrics, states, checkpoint_kind):
    report = copy.deepcopy(report)
    report["checkpoint_kind"] = checkpoint_kind
    report["native_metrics"] = native_metrics
    report["native_pattern_total"] = sum(
        metrics["pattern_count"] for metrics in native_metrics.values())
    report["pattern_reduction"] = (
        report["native_pattern_total"] - report["totals"]["pattern_count"])
    comparisons = {}
    for name, metrics in report["circuits"].items():
        native = native_metrics[name]
        reduction = native["pattern_count"] - metrics["pattern_count"]
        reduction_percent = (
            100.0 * reduction / native["pattern_count"]
            if native["pattern_count"] else 0.0)
        coverage_increase = metrics["fault_coverage"] - native["fault_coverage"]
        comparisons[name] = {
            "circuit": name,
            "checkpoint_kind": checkpoint_kind,
            "round": report["round"],
            "native_fault_coverage": native["fault_coverage"],
            "model_fault_coverage": metrics["fault_coverage"],
            "fault_coverage_increase": coverage_increase,
            "fault_coverage_increase_percentage_points": 100.0 * coverage_increase,
            "native_covered_equivalent_faults": native[
                "covered_equivalent_faults"],
            "model_covered_equivalent_faults": metrics[
                "covered_equivalent_faults"],
            "covered_fault_increase": (
                metrics["covered_equivalent_faults"]
                - native["covered_equivalent_faults"]),
            "native_pattern_count": native["pattern_count"],
            "model_pattern_count": metrics["pattern_count"],
            "pattern_reduction": reduction,
            "pattern_reduction_percent": reduction_percent,
            "coverage_eligible": (
                metrics["covered_equivalent_faults"]
                >= native["covered_equivalent_faults"]),
        }
    report["comparison_by_circuit"] = comparisons
    _coverage_summary(report, states)
    report["coverage_eligible"] = report["eligible"]
    return report


def _complete_report(report, checkpoint, native_metrics, states, checkpoint_kind):
    report = _add_native_comparison(
        report, native_metrics, states, checkpoint_kind)
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = _sha256(checkpoint)
    return report


def _write_evaluation(output, report, exports):
    output = Path(output)
    for name, arrays in exports.items():
        arrays = dict(arrays)
        trajectory = arrays.pop("_trajectory", [])
        write_npz(output / (name + ".ranking.npz"), **arrays)
        metrics = report["circuits"][name]
        events = [dict(event, kind="decision") for event in trajectory]
        events.append({
            "kind": "stc_summary",
            "patterns_before_stc": metrics["patterns_before_stc"],
            "patterns_after_stc": metrics["patterns_after_stc"],
            "stc_removed_patterns": metrics["stc_removed_patterns"],
            "stc_shuffle_attempts": metrics["stc_shuffle_attempts"],
            "stc_coverage_preserved": metrics["stc_coverage_preserved"],
        })
        encoded = "".join(
            json.dumps(event, allow_nan=False) + "\n" for event in events
        ).encode("utf-8")
        atomic_write(output / (name + ".trajectory.jsonl"),
                     lambda stream, data=encoded: stream.write(data))
    rows = list(report.get("comparison_by_circuit", {}).values())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]) if rows else ["circuit"],
        lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    encoded = stream.getvalue().encode("utf-8")
    atomic_write(output / "comparison_by_circuit.csv",
                 lambda destination: destination.write(encoded))
    write_json(output / "summary.json", report)


def _standalone_evaluator(saved, output, manifest, environment):
    trainer = Trainer(
        manifest, TrainConfig(**saved["config"]), output, environment,
        validation_manifest=None)
    if saved["solver_digest"] != trainer.solver_digest:
        raise ValueError("checkpoint PODEM binary changed")
    if saved["torch_version"] != str(torch.__version__):
        raise ValueError("checkpoint PyTorch version changed")
    if saved.get("policy") != POLICY_IDENTITY:
        raise ValueError("checkpoint policy identity changed")
    if saved.get("solver_protocol") != SOLVER_PROTOCOL:
        raise ValueError("checkpoint solver protocol changed")
    trainer.model.load_state_dict(saved["model"])
    return trainer


def evaluate_checkpoint(checkpoint, output, environment=None, manifest=None):
    checkpoint = Path(checkpoint).resolve()
    saved = load_checkpoint(checkpoint)
    checkpoint_kind = saved.get("kind")
    if checkpoint_kind not in ("best", "latest", "final"):
        raise ValueError("evaluation requires best.pt, latest.pt, or final.pt")
    latest_path = checkpoint.parent / "latest.pt"
    if checkpoint_kind == "best" and latest_path.is_file():
        latest = load_checkpoint(latest_path)
        if latest.get("run_id") == saved.get("run_id"):
            expected = latest.get("best")
            current = (
                expected is not None
                and saved.get("round") == expected["report"]["round"]
                and saved.get("evaluation") == expected["report"]
                and saved.get("validation_key") == expected["key"]
                and state_dict_equal(saved["model"], expected["model"])
            )
            if not current:
                raise ValueError(
                    "best.pt is stale relative to latest.pt; resume latest.pt to repair it")
    saved_validation = manifest is None
    evaluation_manifest = (
        Path(manifest).resolve() if manifest is not None
        else Path(saved["validation_manifest_path"]).resolve())
    rng = capture_rng()
    try:
        trainer = _standalone_evaluator(
            saved, output, evaluation_manifest, environment)
        if saved_validation and (
                trainer.manifest.digest != saved["validation_manifest_digest"]
                or trainer.provenance != saved["validation_artifacts"]
                or trainer.solver_digest != saved["validation_solver_digest"]):
            raise ValueError(
                "checkpoint validation manifest, artifacts, or solver changed")
        native_metrics = {}
        states = {}
        for circuit in trainer.circuits:
            metrics, _ = trainer._run_native(circuit, trainer.environment)
            native_metrics[circuit.name] = metrics
            states[circuit.name] = {
                "native_covered_equivalent_faults": metrics[
                    "covered_equivalent_faults"]
            }
        report, exports = trainer.evaluate_model(
            trainer.model, saved["round"], trainer.circuits,
            trainer.environment, states)
        expected_report = saved.get("evaluation")
        if saved_validation and checkpoint_kind in ("best", "final"):
            if expected_report is None:
                raise ValueError("checkpoint is missing its validation report")
            for name, metrics in report["circuits"].items():
                expected_metrics = expected_report["circuits"].get(name)
                if expected_metrics is None or any(
                        metrics[key] != value
                        for key, value in expected_metrics.items()
                        if key != "seconds"):
                    raise RuntimeError(
                        "fresh validation differs from the saved checkpoint report")
        report["training_manifest"] = saved["manifest_path"]
        report["training_manifest_digest"] = saved["manifest_digest"]
        report["evaluation_manifest"] = str(trainer.manifest.path)
        report["evaluation_manifest_digest"] = trainer.manifest.digest
        report = _complete_report(
            report, checkpoint, native_metrics, states, checkpoint_kind)
        _write_evaluation(Path(output).resolve(), report, exports)
        return report
    finally:
        restore_rng(rng)
