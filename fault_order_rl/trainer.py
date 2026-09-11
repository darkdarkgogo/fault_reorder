"""CPU listwise REINFORCE with transactional rounds and coverage guards."""

import copy
from dataclasses import asdict, dataclass
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
from .model import FaultScorer
from .policy import (centered_logits, deterministic_permutation,
                     plackett_luce_log_prob, sample_permutation)


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
    backtrack_limit: int = 5000
    threads: int = 1

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
        if self.backtrack_limit != 5000:
            raise ValueError("the fault-order experiment requires backtrack_limit=5000")

    def temperature(self, round_number):
        fraction = min(max(round_number - 1, 0) / max(self.temperature_rounds - 1, 1), 1)
        return self.temperature_start * (self.temperature_min / self.temperature_start) ** fraction


def reward_transition(state, metrics, ema_decay):
    """Return new state; never mutate a baseline during episode collection."""
    new = dict(state)
    patterns = metrics["pattern_count"]
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
            metrics, elapsed = self._run(circuit, np.arange(circuit.fault_count))
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

    def _run(self, circuit, permutation):
        start = time.perf_counter()
        result = self.environment.run(circuit.spec.bench_path, circuit.spec.faultmap_path,
                                      [circuit.fault_ids[int(i)] for i in permutation])
        if result["uncollapsed_faults"] != int(circuit.eqv_fault_nums.sum()):
            raise RuntimeError("PODEM fault total differs from validated embeddings")
        return result, time.perf_counter() - start

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
                scores = model(circuit.embeddings)
                order = deterministic_permutation(scores).numpy()
                metrics, elapsed = self._run(circuit, order)
                metrics = dict(metrics, seconds=elapsed)
                results[circuit.name] = metrics
                ranks = np.empty(circuit.fault_count, dtype=np.int64)
                ranks[order] = np.arange(1, circuit.fault_count + 1)
                exports[circuit.name] = {"fault_ids": np.asarray(circuit.fault_ids),
                                         "scores": scores.numpy(), "ranks": ranks, "permutation": order}
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
            records, orders = [], {}
            # No retained autograd graph during solver calls; this model remains
            # unchanged until every circuit has contributed to a single update.
            with torch.no_grad():
                for circuit in self.circuits:
                    order, _ = sample_permutation(self.model(circuit.embeddings), temperature)
                    orders[circuit.name] = order.numpy()
                    metrics, elapsed = self._run(circuit, orders[circuit.name])
                    states[circuit.name], reward = reward_transition(
                        states[circuit.name], metrics, self.config.ema_decay)
                    records.append(dict(self._record("episode", circuit.name, number, metrics, elapsed),
                                        temperature=temperature, **reward))
            candidate = copy.deepcopy(self.model)
            optimizer = self._optimizer(candidate)
            optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
            optimizer.zero_grad()
            for circuit, record in zip(self.circuits, records):
                logits = centered_logits(candidate(circuit.embeddings), temperature)
                log_prob = plackett_luce_log_prob(logits, torch.from_numpy(orders[circuit.name]))
                loss = -record["advantage"] * log_prob / circuit.fault_count / len(self.circuits)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite policy loss")
                record["loss_contribution"] = float(loss.detach())
                record["log_probability"] = float(log_prob.detach())
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), self.config.gradient_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite policy gradient")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in candidate.parameters()):
                raise RuntimeError("non-finite model after optimizer update")
            best = self.best
            if best is not None and not eligible(best["report"], states):
                best = None
            if number % self.config.evaluate_every == 0 or number == self.config.rounds:
                report, _ = self.evaluate_model(candidate, number, states)
                best = self._choose_best(best, candidate, report, states)
                records.append({"kind": "evaluation", "report": report})
            records.append({"kind": "update", "round": number, "gradient_norm": float(grad_norm),
                            "circuits": len(self.circuits)})
            payload = self._payload(model=candidate, optimizer=optimizer, round_number=number,
                                    states=states, best=best)
            self._write_round(number, records, orders)
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

    def _write_round(self, number, records, orders):
        # One immutable JSONL file per committed round. Files beyond latest's
        # round are incomplete attempts and may be replaced when resuming.
        prefix = self.output / "rounds" / ("round-{:06d}".format(number))
        if orders:
            write_npz(prefix.with_suffix(".npz"), **orders)
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
    shortfalls = {
        name: max(0, state["native_covered_equivalent_faults"]
                  - report["circuits"][name]["covered_equivalent_faults"])
        for name, state in states.items()
    }
    report["coverage_eligible"] = report["eligible"]
    report["coverage_shortfall"] = sum(shortfalls.values())
    report["coverage_shortfall_by_circuit"] = shortfalls
    return report


def _write_evaluation(output, report, exports):
    output = Path(output)
    for name, arrays in exports.items():
        write_npz(output / (name + ".ranking.npz"), **arrays)
    write_json(output / "summary.json", report)


def evaluate_checkpoint(checkpoint, output, environment=None):
    checkpoint = Path(checkpoint).resolve()
    saved = load_checkpoint(checkpoint)
    if saved.get("kind") != "best" or not saved.get("available"):
        raise ValueError("no coverage-eligible best checkpoint; continue training from latest.pt")
    # latest.pt is the commit point. A crash between committing latest and
    # publishing its derived best file must never permit a stale standalone
    # evaluation with obsolete coverage requirements.
    latest_path = checkpoint.parent / "latest.pt"
    if latest_path.is_file():
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
        trainer = Trainer(saved["manifest_path"], TrainConfig(**saved["config"]), output, environment)
        trainer._check_compatibility(saved)
        trainer.model.load_state_dict(saved["model"])
        report, exports = trainer.evaluate_model(trainer.model, saved["round"], saved["states"])
        if not report["eligible"]:
            raise RuntimeError("fresh best evaluation failed coverage requirements")
        for name, metrics in report["circuits"].items():
            expected = saved["evaluation"]["circuits"][name]
            if any(metrics[key] != expected[key] for key in expected if key != "seconds"):
                raise RuntimeError("fresh best evaluation differs from saved deterministic metrics")
        report = _complete_report(report, checkpoint, saved["native_metrics"], saved["states"], "best")
        _write_evaluation(output, report, exports)
        return report
    finally:
        restore_rng(rng)
