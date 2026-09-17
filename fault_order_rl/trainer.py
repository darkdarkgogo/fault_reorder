"""CPU listwise REINFORCE with transactional rounds and coverage guards."""

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
from .environment import PodemEnvironment
from .model import FaultScorer, build_dynamic_features
from .policy import (deterministic_permutation, select_categorical_action,
                     trajectory_log_prob)


@dataclass
class TrainConfig:
    rounds: int = 100
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.9
    temperature_start: float = 1.0
    temperature_min: float = 0.1
    temperature_rounds: int = 100
    evaluate_every: int = 10
    seed: int = 14
    backtrack_limit: int = 200
    threads: int = 1
    # Zero preserves the historical full-manifest update behavior.
    batch_size: int = 0

    def validate(self):
        for key in ("rounds", "temperature_rounds", "evaluate_every", "backtrack_limit", "threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(key + " must be a positive integer")
        for key in ("learning_rate", "gradient_clip", "temperature_start", "temperature_min"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + " must be finite and positive")
        if not 0 <= self.ema_decay < 1 or not math.isfinite(self.ema_decay):
            raise ValueError("ema_decay must be in [0, 1)")
        if self.temperature_min > self.temperature_start:
            raise ValueError("temperature_min exceeds temperature_start")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        if self.backtrack_limit != 200:
            raise ValueError("the compressed fault-order experiment requires backtrack_limit=200")

    def temperature(self, round_number):
        fraction = min(max(round_number - 1, 0) / max(self.temperature_rounds - 1, 1), 1)
        return self.temperature_start * (self.temperature_min / self.temperature_start) ** fraction


def reward_transition(state, metrics, ema_decay):
    """Return new state; never mutate a baseline during episode collection."""
    new = dict(state)
    patterns = metrics["patterns_after_stc"]
    covered = metrics["covered_equivalent_faults"]
    valid = covered >= state["native_covered_equivalent_faults"]
    if valid:
        reward = state["previous_pattern_count"] - patterns
        new["previous_pattern_count"] = patterns
    else:
        reward = -max(state["native_pattern_count"], state["previous_pattern_count"], patterns, 1)
    advantage = (reward - state["reward_ema"]) / max(state["native_pattern_count"], 1)
    new["reward_ema"] = ema_decay * state["reward_ema"] + (1 - ema_decay) * reward
    return new, {"raw_reward": reward, "advantage": advantage, "coverage_valid": valid}


def eligible(report, states):
    return report is not None and all(
        report["circuits"][name]["covered_equivalent_faults"]
        >= state["native_covered_equivalent_faults"]
        for name, state in states.items()
    )


def evaluation_key(report):
    totals = report["totals"]
    return (totals["pattern_count"], -totals["covered_equivalent_faults"],
            totals["podem_calls"], totals["total_backtracks"], report["round"])


def state_dict_equal(first, second):
    return first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first)


def _normalized_policy_loss(advantage, log_prob, initial_fault_count, batch_size):
    if initial_fault_count <= 0:
        raise ValueError("initial_fault_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return -advantage * log_prob / initial_fault_count / batch_size


class Trainer:
    def __init__(self, manifest, config, output, environment=None):
        config.validate()
        self.config = config
        self.manifest = load_manifest(manifest)
        self.environment = environment or PodemEnvironment(
            self.manifest.module_dir, config.backtrack_limit, 14)
        self.circuits = load_all_circuits(self.manifest, self.environment)
        self.output = Path(output).resolve()
        self.provenance = {c.name: c.artifact_digest for c in self.circuits}
        module_path = getattr(getattr(self.environment, "module", None), "__file__", None)
        self.solver_digest = _sha256(module_path) if module_path else None
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        self.model = FaultScorer()
        self.optimizer = self._optimizer(self.model)
        self.round = 0
        self.states = {}
        self.native_metrics = {}
        self.best = None
        self.run_id = uuid.uuid4().hex

    def _optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    @classmethod
    def create(cls, manifest, config, output, environment=None):
        output = Path(output).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory is not empty; use --resume or a new directory")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self = cls(manifest, config, output, environment)
        # All embeddings have already passed validation before any ATPG run.
        records = []
        for circuit in self.circuits:
            metrics, elapsed = self._run_native(circuit)
            self.native_metrics[circuit.name] = metrics
            self.states[circuit.name] = {
                "native_pattern_count": metrics["pattern_count"],
                "previous_pattern_count": metrics["pattern_count"],
                "native_covered_equivalent_faults": metrics["covered_equivalent_faults"],
                "reward_ema": 0.0,
            }
            records.append(self._record("baseline", circuit.name, 0, metrics, elapsed))
        report, _ = self.evaluate_model(self.model, 0, self.states)
        self.best = self._choose_best(None, self.model, report, self.states)
        records.append({"kind": "evaluation", "report": report})
        self._write_round(0, records, {})
        save_checkpoint(self.output / "latest.pt", self._payload())
        self._publish_best()
        return self

    @classmethod
    def resume(cls, checkpoint, rounds=None, environment=None):
        checkpoint = Path(checkpoint).resolve()
        saved = load_checkpoint(checkpoint)
        if saved.get("kind") != "latest":
            raise ValueError("resume requires a latest training checkpoint")
        config = TrainConfig(**saved["config"])
        if rounds is not None:
            if rounds < saved["round"]:
                raise ValueError("--rounds is a total target, below the completed round")
            config.rounds = rounds
        self = cls(saved["manifest_path"], config, checkpoint.parent, environment)
        self._check_compatibility(saved)
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.round = saved["round"]
        self.states = saved["states"]
        self.native_metrics = saved["native_metrics"]
        self.best = saved["best"]
        self.run_id = saved["run_id"]
        restore_rng(saved["rng"])
        self._publish_best()  # repair an interrupted derived-artifact write
        return self

    def _check_compatibility(self, saved):
        if saved["manifest_digest"] != self.manifest.digest or saved["artifacts"] != self.provenance:
            raise ValueError("checkpoint manifest or embedding artifacts changed")
        if saved["solver_digest"] != self.solver_digest:
            raise ValueError("checkpoint PODEM binary changed")
        if saved["torch_version"] != str(torch.__version__):
            raise ValueError("checkpoint PyTorch version changed; exact resume cannot be guaranteed")

    def _check_result(self, circuit, result):
        if result["uncollapsed_faults"] != int(circuit.eqv_fault_nums.sum()):
            raise RuntimeError("PODEM fault total differs from validated embeddings")
        if result.get("stc_coverage_preserved") is not True:
            raise RuntimeError("PODEM STC coverage was not preserved")

    def _run_native(self, circuit):
        start = time.perf_counter()
        session = self.environment.start_session(
            circuit.spec.bench_path, circuit.spec.faultmap_path)
        while session.remaining_fault_ids:
            session.step(session.remaining_fault_ids[0])
        result = session.finish()
        self._check_result(circuit, result)
        return result, time.perf_counter() - start

    def _run_policy(self, circuit, model, temperature, stochastic):
        start = time.perf_counter()
        session = self.environment.start_session(
            circuit.spec.bench_path, circuit.spec.faultmap_path)
        row_by_id = {identifier: row for row, identifier in enumerate(circuit.fault_ids)}
        decisions, trace = [], []
        while session.remaining_fault_ids:
            try:
                rows = tuple(row_by_id[identifier]
                             for identifier in session.remaining_fault_ids)
            except KeyError as exc:
                raise RuntimeError("PODEM remaining fault is absent from embeddings") from exc
            features = build_dynamic_features(circuit.embeddings, rows)
            scores = model(features)
            selected_row = select_categorical_action(
                scores, rows, temperature, stochastic)
            selected_id = circuit.fault_ids[selected_row]
            local = rows.index(selected_row)
            step = session.step(selected_id)
            decisions.append({"remaining_rows": rows, "selected_row": selected_row})
            trace.append({
                "selected_fault_id": selected_id,
                "selected_row": selected_row,
                "selected_score": float(scores[local].detach()),
                "remaining_count": len(rows),
                "remaining_ratio": len(rows) / circuit.fault_count,
                "target_status": step["target_status"],
                "generated_test_vector": step["generated_test_vector"],
                "dtc_attempted_fault_ids": step["dtc_attempted_fault_ids"],
                "dtc_embedded_fault_ids": step["dtc_embedded_fault_ids"],
                "newly_detected_fault_ids": step["newly_detected_fault_ids"],
                "primary_podem_calls": step["primary_podem_calls"],
                "dtc_secondary_calls": step["dtc_secondary_calls"],
                "primary_backtracks": step["primary_backtracks"],
                "dtc_backtracks": step["dtc_backtracks"],
                "total_backtracks": step["total_backtracks"],
                "current_pattern_count": step["current_pattern_count"],
            })
        result = session.finish()
        self._check_result(circuit, result)
        return result, time.perf_counter() - start, decisions, trace

    def _record(self, kind, name, round_number, metrics, elapsed):
        return {"kind": kind, "circuit": name, "round": round_number,
                "seed": 14, "training_seed": self.config.seed,
                "checkpoint_identity": self.run_id + ":" + str(round_number),
                "seconds": elapsed, **metrics}

    def evaluate_model(self, model, round_number, states):
        results, exports = {}, {}
        model.eval()
        with torch.no_grad():
            for circuit in self.circuits:
                metrics, elapsed, decisions, trace = self._run_policy(
                    circuit, model, temperature=1.0, stochastic=False)
                metrics = dict(metrics, seconds=elapsed)
                results[circuit.name] = metrics
                first_rows = tuple(range(circuit.fault_count))
                scores = model(build_dynamic_features(circuit.embeddings, first_rows))
                order = deterministic_permutation(scores).numpy()
                ranks = np.empty(circuit.fault_count, dtype=np.int64)
                ranks[order] = np.arange(1, circuit.fault_count + 1)
                exports[circuit.name] = {"fault_ids": np.asarray(circuit.fault_ids),
                                         "scores": scores.numpy(), "ranks": ranks,
                                         "permutation": order,
                                         "selected_rows": np.asarray(
                                             [d["selected_row"] for d in decisions],
                                             dtype=np.int64)}
        totals = {key: sum(result[key] for result in results.values()) for key in
                  ("pattern_count", "detected_equivalent_faults", "detected_collapsed_faults",
                   "redundant_equivalent_faults", "covered_equivalent_faults",
                   "uncollapsed_faults", "podem_calls", "total_backtracks",
                   "aborted_faults", "redundant_faults")}
        totals["fault_coverage"] = (
            totals["covered_equivalent_faults"] / totals["uncollapsed_faults"])
        report = {"round": round_number, "circuits": results, "totals": totals}
        report["eligible"] = eligible(report, states)
        return report, exports

    def _round_batches(self):
        indices = list(range(len(self.circuits)))
        if self.config.batch_size:
            random.shuffle(indices)
        size = self.config.batch_size or len(indices)
        return [indices[start:start + size] for start in range(0, len(indices), size)]

    @staticmethod
    def _choose_best(best, model, report, states):
        if best is not None and not eligible(best["report"], states):
            best = None
        if eligible(report, states) and (best is None or evaluation_key(report) < evaluation_key(best["report"])):
            best = {"model": copy.deepcopy(model.state_dict()), "report": copy.deepcopy(report)}
        return best

    def step(self):
        if self.round >= self.config.rounds:
            raise ValueError("training already reached the configured round target")
        rng_before = capture_rng()
        try:
            number = self.round + 1
            temperature = self.config.temperature(number)
            states = copy.deepcopy(self.states)
            records, trajectories = [], {}
            candidate = copy.deepcopy(self.model)
            optimizer = self._optimizer(candidate)
            optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
            candidate.train()
            for batch_index, circuit_indices in enumerate(self._round_batches()):
                batch = [self.circuits[index] for index in circuit_indices]
                batch_records = []
                # Every episode in this batch sees the same parameter snapshot.
                with torch.no_grad():
                    for circuit in batch:
                        metrics, elapsed, decisions, trace = self._run_policy(
                            circuit, candidate, temperature, stochastic=True)
                        trajectories[circuit.name] = decisions
                        states[circuit.name], reward = reward_transition(
                            states[circuit.name], metrics, self.config.ema_decay)
                        batch_records.append(dict(
                            self._record("episode", circuit.name, number, metrics, elapsed),
                            temperature=temperature, batch_index=batch_index,
                            trace=trace, **reward))
                optimizer.zero_grad()
                for circuit, record in zip(batch, batch_records):
                    log_prob = trajectory_log_prob(
                        candidate, circuit.embeddings,
                        trajectories[circuit.name], temperature)
                    loss = _normalized_policy_loss(
                        record["advantage"], log_prob,
                        circuit.fault_count, len(batch))
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite policy loss")
                    record["loss_contribution"] = float(loss.detach())
                    record["log_probability"] = float(log_prob.detach())
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    candidate.parameters(), self.config.gradient_clip)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("non-finite policy gradient")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in candidate.parameters()):
                    raise RuntimeError("non-finite model after optimizer update")
                records.extend(batch_records)
                records.append({
                    "kind": "update",
                    "round": number,
                    "batch_index": batch_index,
                    "gradient_norm": float(grad_norm),
                    "circuits": len(batch),
                    "circuit_names": [circuit.name for circuit in batch],
                })
            best = self.best
            if best is not None and not eligible(best["report"], states):
                best = None
            if number % self.config.evaluate_every == 0 or number == self.config.rounds:
                report, _ = self.evaluate_model(candidate, number, states)
                best = self._choose_best(best, candidate, report, states)
                records.append({"kind": "evaluation", "report": report})
            payload = self._payload(model=candidate, optimizer=optimizer, round_number=number,
                                    states=states, best=best)
            self._write_round(number, records, trajectories)
            # This atomic replacement is the commit point. Failed collection,
            # evaluation or optimizer steps cannot alter latest or live state.
            save_checkpoint(self.output / "latest.pt", payload)
        except BaseException:
            restore_rng(rng_before)
            raise
        self.model, self.optimizer, self.states, self.best, self.round = candidate, optimizer, states, best, number
        self._publish_best()
        return records

    def _payload(self, model=None, optimizer=None, round_number=None, states=None, **overrides):
        return {"version": 2, "kind": "latest", "run_id": self.run_id,
                "manifest_path": str(self.manifest.path), "manifest_digest": self.manifest.digest,
                "artifacts": self.provenance, "solver_digest": self.solver_digest,
                "torch_version": str(torch.__version__), "config": asdict(self.config),
                "round": self.round if round_number is None else round_number,
                "model": (self.model if model is None else model).state_dict(),
                "optimizer": (self.optimizer if optimizer is None else optimizer).state_dict(),
                "states": self.states if states is None else states,
                "native_metrics": self.native_metrics, "rng": capture_rng(),
                "best": overrides.get("best", self.best)}

    def _write_round(self, number, records, trajectories):
        # One immutable JSONL file per committed round. Files beyond latest's
        # round are incomplete attempts and may be replaced when resuming.
        prefix = self.output / "rounds" / ("round-{:06d}".format(number))
        if trajectories:
            arrays = {}
            for name, decisions in trajectories.items():
                selected = np.asarray(
                    [decision["selected_row"] for decision in decisions],
                    dtype=np.int64)
                offsets = [0]
                remaining = []
                for decision in decisions:
                    remaining.extend(decision["remaining_rows"])
                    offsets.append(len(remaining))
                arrays[name + "__selected_rows"] = selected
                arrays[name + "__remaining_offsets"] = np.asarray(offsets, dtype=np.int64)
                arrays[name + "__remaining_rows"] = np.asarray(remaining, dtype=np.int64)
            write_npz(prefix.with_suffix(".npz"), **arrays)
        encoded = "".join(json.dumps(r, allow_nan=False) + "\n" for r in records).encode("utf-8")
        atomic_write(prefix.with_suffix(".jsonl"), lambda stream: stream.write(encoded))

    def _publish_best(self):
        payload = self._payload()
        payload.pop("optimizer")
        payload.pop("rng")
        payload.pop("best")
        payload["kind"] = "best"
        payload["available"] = self.best is not None
        if self.best is not None:
            payload["model"] = self.best["model"]
            payload["round"] = self.best["report"]["round"]
            payload["evaluation"] = self.best["report"]
        else:
            payload.pop("model")
        save_checkpoint(self.output / "best.pt", payload)

    def train(self):
        while self.round < self.config.rounds:
            records = self.step()
            episodes = [r for r in records if r["kind"] == "episode"]
            print("round {}/{}: patterns={}, reward={}".format(
                self.round, self.config.rounds, sum(r["pattern_count"] for r in episodes),
                sum(r["raw_reward"] for r in episodes)), flush=True)
        if self.best is not None:
            return evaluate_checkpoint(self.output / "best.pt", self.output / "evaluation",
                                       environment=self.environment)
        report, exports = self.evaluate_model(self.model, self.round, self.states)
        report = _complete_report(report, self.output / "latest.pt", self.native_metrics,
                                  self.states, "latest")
        _write_evaluation(self.output / "evaluation", report, exports)
        return report


def _complete_report(report, checkpoint, native_metrics, states, checkpoint_kind):
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = _sha256(checkpoint)
    report["checkpoint_kind"] = checkpoint_kind
    report["native_metrics"] = native_metrics
    report["native_pattern_total"] = sum(m["pattern_count"] for m in native_metrics.values())
    report["pattern_reduction"] = report["native_pattern_total"] - report["totals"]["pattern_count"]
    comparisons = {}
    for name, metrics in report["circuits"].items():
        native = native_metrics[name]
        reduction = native["pattern_count"] - metrics["pattern_count"]
        if native["pattern_count"]:
            reduction_percent = 100.0 * reduction / native["pattern_count"]
        elif metrics["pattern_count"] == 0:
            reduction_percent = 0.0
        else:
            reduction_percent = None
        native_coverage = native["fault_coverage"]
        model_coverage = metrics["fault_coverage"]
        coverage_increase = model_coverage - native_coverage
        comparisons[name] = {
            "circuit": name,
            "checkpoint_kind": checkpoint_kind,
            "round": report["round"],
            "native_fault_coverage": native_coverage,
            "model_fault_coverage": model_coverage,
            "fault_coverage_increase": coverage_increase,
            "fault_coverage_increase_percentage_points": 100.0 * coverage_increase,
            "native_covered_equivalent_faults": native["covered_equivalent_faults"],
            "model_covered_equivalent_faults": metrics["covered_equivalent_faults"],
            "covered_fault_increase": (
                metrics["covered_equivalent_faults"]
                - native["covered_equivalent_faults"]
            ),
            "native_pattern_count": native["pattern_count"],
            "model_pattern_count": metrics["pattern_count"],
            "pattern_reduction": reduction,
            "pattern_reduction_percent": reduction_percent,
            "coverage_eligible": (
                metrics["covered_equivalent_faults"]
                >= native["covered_equivalent_faults"]
            ),
        }
    report["comparison_by_circuit"] = comparisons
    shortfalls = {
        name: max(0, state["native_covered_equivalent_faults"]
                  - report["circuits"][name]["covered_equivalent_faults"])
        for name, state in states.items()
    }
    report["coverage_eligible"] = report["eligible"]
    report["coverage_shortfall"] = sum(shortfalls.values())
    report["coverage_shortfall_by_circuit"] = shortfalls
    return report


def _write_evaluation(output, report, exports, include_aggregate=True):
    output = Path(output)
    for name, arrays in exports.items():
        write_npz(output / (name + ".ranking.npz"), **arrays)
    rows = list(report["comparison_by_circuit"].values())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["circuit"],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    encoded = stream.getvalue().encode("utf-8")
    atomic_write(output / "comparison_by_circuit.csv",
                 lambda destination: destination.write(encoded))
    summary = copy.deepcopy(report)
    if not include_aggregate:
        for key in ("totals", "native_pattern_total", "pattern_reduction",
                    "coverage_shortfall"):
            summary.pop(key, None)
    write_json(output / "summary.json", summary)


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


def _evaluate_external_manifest(saved, checkpoint, output, manifest, environment):
    trainer = Trainer(manifest, TrainConfig(**saved["config"]), output, environment)
    if saved["solver_digest"] != trainer.solver_digest:
        raise ValueError("checkpoint PODEM binary changed")
    if saved["torch_version"] != str(torch.__version__):
        raise ValueError("checkpoint PyTorch version changed; evaluation is not reproducible")
    trainer.model.load_state_dict(saved["model"])
    output = Path(output).resolve()
    checkpoint_digest = _sha256(checkpoint)
    identity = {
        "checkpoint_sha256": checkpoint_digest,
        "evaluation_manifest_digest": trainer.manifest.digest,
        "artifacts": trainer.provenance,
        "solver_digest": trainer.solver_digest,
    }
    status_path = output / "status.json"
    completed = []
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("identity") != identity:
            raise ValueError("evaluation output belongs to a different checkpoint or manifest")
        completed = list(status.get("completed", []))
    elif output.exists() and any(output.iterdir()):
        raise ValueError("evaluation output is non-empty and has no resumable status")

    results, native_metrics = {}, {}
    circuit_names = [circuit.name for circuit in trainer.circuits]
    completed_set = set(completed)
    unknown = completed_set - set(circuit_names)
    if unknown:
        raise ValueError("evaluation status contains unknown circuits")
    for circuit in trainer.circuits:
        metrics_path = output / "circuits" / (circuit.name + ".metrics.json")
        ranking_path = output / (circuit.name + ".ranking.npz")
        if circuit.name in completed_set:
            if not metrics_path.is_file() or not ranking_path.is_file():
                raise ValueError("completed evaluation artifact is missing: " + circuit.name)
            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if saved_metrics.get("identity") != identity:
                raise ValueError("circuit metrics identity changed: " + circuit.name)
            native_metrics[circuit.name] = saved_metrics["native"]
            results[circuit.name] = saved_metrics["model"]
            continue

        native, native_elapsed = trainer._run_native(circuit)
        native = dict(native, seconds=native_elapsed)
        trainer.model.eval()
        with torch.no_grad():
            first_rows = tuple(range(circuit.fault_count))
            scores = trainer.model(build_dynamic_features(
                circuit.embeddings, first_rows))
            order = deterministic_permutation(scores).numpy()
            model_metrics, model_elapsed, decisions, _ = trainer._run_policy(
                circuit, trainer.model, temperature=1.0, stochastic=False)
        model_metrics = dict(model_metrics, seconds=model_elapsed)
        ranks = np.empty(circuit.fault_count, dtype=np.int64)
        ranks[order] = np.arange(1, circuit.fault_count + 1)
        write_npz(ranking_path, fault_ids=np.asarray(circuit.fault_ids),
                  scores=scores.numpy(), ranks=ranks, permutation=order,
                  selected_rows=np.asarray(
                      [decision["selected_row"] for decision in decisions],
                      dtype=np.int64))
        write_json(metrics_path, {"identity": identity, "native": native,
                                  "model": model_metrics})
        native_metrics[circuit.name] = native
        results[circuit.name] = model_metrics
        completed.append(circuit.name)
        completed_set.add(circuit.name)
        write_json(status_path, {
            "identity": identity,
            "complete": False,
            "completed": completed,
            "pending": [name for name in circuit_names if name not in completed_set],
        })

    states = {
        name: {"native_covered_equivalent_faults": metrics["covered_equivalent_faults"]}
        for name, metrics in native_metrics.items()
    }
    report = {"round": saved["round"], "circuits": results,
              "totals": _evaluation_totals(results)}
    report["eligible"] = eligible(report, states)
    report["training_manifest"] = saved["manifest_path"]
    report["training_manifest_digest"] = saved["manifest_digest"]
    report["evaluation_manifest"] = str(trainer.manifest.path)
    report["evaluation_manifest_digest"] = trainer.manifest.digest
    report = _complete_report(report, checkpoint, native_metrics, states, saved["kind"])
    _write_evaluation(output, report, {}, include_aggregate=False)
    write_json(status_path, {"identity": identity, "complete": True,
                             "completed": circuit_names, "pending": []})
    return report


def evaluate_checkpoint(checkpoint, output, environment=None, manifest=None):
    checkpoint = Path(checkpoint).resolve()
    saved = load_checkpoint(checkpoint)
    checkpoint_kind = saved.get("kind")
    if checkpoint_kind not in ("best", "latest"):
        raise ValueError("evaluation requires a best.pt or latest.pt checkpoint")
    if checkpoint_kind == "best" and not saved.get("available"):
        raise ValueError("no coverage-eligible best checkpoint; continue training from latest.pt")
    # latest.pt is the commit point. A crash between committing latest and
    # publishing its derived best file must never permit a stale standalone
    # evaluation with obsolete coverage requirements.
    latest_path = checkpoint.parent / "latest.pt"
    if checkpoint_kind == "best" and latest_path.is_file():
        latest = load_checkpoint(latest_path)
        if latest.get("run_id") == saved.get("run_id"):
            expected = latest.get("best")
            current = (expected is not None
                       and saved.get("round") == expected["report"]["round"]
                       and saved.get("evaluation") == expected["report"]
                       and saved.get("states") == latest.get("states")
                       and state_dict_equal(saved["model"], expected["model"]))
            if not current:
                raise ValueError("best.pt is stale relative to latest.pt; resume latest.pt once to repair it")
    rng = capture_rng()
    try:
        if manifest is not None:
            return _evaluate_external_manifest(
                saved, checkpoint, output, manifest, environment)
        trainer = Trainer(saved["manifest_path"], TrainConfig(**saved["config"]), output, environment)
        trainer._check_compatibility(saved)
        trainer.model.load_state_dict(saved["model"])
        report, exports = trainer.evaluate_model(trainer.model, saved["round"], saved["states"])
        if checkpoint_kind == "best" and not report["eligible"]:
            raise RuntimeError("fresh best evaluation failed coverage requirements")
        if checkpoint_kind == "best":
            for name, metrics in report["circuits"].items():
                expected = saved["evaluation"]["circuits"][name]
                if any(metrics[key] != expected[key] for key in expected if key != "seconds"):
                    raise RuntimeError("fresh best evaluation differs from saved deterministic metrics")
        report = _complete_report(
            report, checkpoint, saved["native_metrics"], saved["states"], checkpoint_kind)
        _write_evaluation(output, report, exports)
        return report
    finally:
        restore_rng(rng)
