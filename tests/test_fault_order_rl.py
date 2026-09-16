"""Policy math, shared updates, coverage gating and exact round recovery."""

import copy
import csv
from dataclasses import replace
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fault_order_rl.checkpoint import capture_rng, load_checkpoint, save_checkpoint
from fault_order_rl.data import (
    CircuitData, CircuitSpec, _sha256, load_circuit_data, load_manifest,
)
from fault_order_rl.environment import PodemEnvironment
from fault_order_rl.model import FaultScorer, build_dynamic_features
from fault_order_rl.policy import (
    centered_logits, deterministic_permutation, plackett_luce_log_prob,
    sample_permutation, select_categorical_action, trajectory_log_prob,
)
from fault_order_rl.trainer import (
    TrainConfig,
    Trainer,
    _complete_report,
    _write_evaluation,
    eligible,
    evaluate_checkpoint,
    reward_transition,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_features_append_remaining_mean_and_ratio():
    embeddings = torch.arange(4 * 257, dtype=torch.float32).reshape(4, 257)
    unchanged = embeddings.clone()
    rows = torch.tensor([1, 3])
    features = build_dynamic_features(embeddings, rows)
    assert features.shape == (2, 515)
    assert torch.equal(features[:, :257], embeddings[rows])
    mean = embeddings[rows].mean(dim=0)
    assert torch.equal(features[:, 257:514], mean.expand(2, -1))
    assert torch.equal(features[:, 514], torch.full((2,), 0.5))
    assert torch.equal(embeddings, unchanged)


@pytest.mark.parametrize('rows', [[], [1, 1], [-1], [4], [[1, 2]]])
def test_dynamic_features_reject_invalid_rows(rows):
    with pytest.raises(ValueError):
        build_dynamic_features(torch.zeros(4, 257), rows)


def test_dynamic_scorer_requires_515_features():
    model = FaultScorer()
    assert model(torch.zeros(2, 515)).shape == (2,)
    with pytest.raises(ValueError, match='515'):
        model(torch.zeros(2, 257))


def test_categorical_action_samples_only_remaining_and_reproduces_seed():
    scores = torch.tensor([0.2, 1.0, -0.5])
    rows = (5, 2, 8)
    first = select_categorical_action(scores, rows, 0.7, stochastic=True,
                                      generator=torch.Generator().manual_seed(12))
    second = select_categorical_action(scores, rows, 0.7, stochastic=True,
                                       generator=torch.Generator().manual_seed(12))
    assert first == second
    assert first in rows
    assert select_categorical_action(scores, rows, 0.7, stochastic=False) == 2
    assert select_categorical_action(torch.tensor([1., 1., 0.]), rows, 0.7,
                                     stochastic=False) == 2


@pytest.mark.parametrize('scores,rows,temperature', [
    (torch.tensor([0., float('nan')]), (0, 1), 1.),
    (torch.tensor([0., 1.]), (0, 1), 0.),
    (torch.tensor([0., 1.]), (0, 0), 1.),
    (torch.tensor([0., 1.]), (0,), 1.),
])
def test_categorical_action_rejects_invalid_input(scores, rows, temperature):
    with pytest.raises(ValueError):
        select_categorical_action(scores, rows, temperature, stochastic=False)


def test_dynamic_trajectory_log_prob_matches_direct_steps():
    embeddings = torch.randn(4, 257, generator=torch.Generator().manual_seed(9))
    model = FaultScorer()
    decisions = [
        {"remaining_rows": (0, 1, 2, 3), "selected_row": 2},
        {"remaining_rows": (1, 3), "selected_row": 3},
    ]
    actual = trajectory_log_prob(model, embeddings, decisions, 0.7)
    direct = []
    for state in decisions:
        rows = torch.tensor(state["remaining_rows"])
        logits = centered_logits(model(build_dynamic_features(embeddings, rows)), 0.7)
        local = state["remaining_rows"].index(state["selected_row"])
        direct.append(torch.log_softmax(logits, 0)[local])
    assert torch.allclose(actual, torch.stack(direct).sum())
    actual.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_dynamic_trajectory_rejects_selected_row_outside_remaining():
    with pytest.raises(ValueError, match='selected row'):
        trajectory_log_prob(FaultScorer(), torch.zeros(3, 257),
                            [{"remaining_rows": (0, 2), "selected_row": 1}], 1.)


def test_policy_distribution_and_gradient_match_direct_formula():
    logits = torch.tensor([0.3, -0.6, 1.2], dtype=torch.float64, requires_grad=True)
    probabilities = []
    for values in itertools.permutations(range(3)):
        order = torch.tensor(values)
        actual = plackett_luce_log_prob(logits, order)
        direct = sum(logits[order[i]] - torch.logsumexp(logits[order[i:]], 0) for i in range(3))
        assert torch.allclose(actual, direct)
        assert torch.allclose(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                              torch.autograd.grad(direct, logits, retain_graph=True)[0])
        probabilities.append(actual.exp())
    assert torch.allclose(sum(probabilities), torch.tensor(1., dtype=torch.float64))


def test_sampling_and_deterministic_ties():
    scores = torch.tensor([1., 1., 0., 1.])
    assert deterministic_permutation(scores).tolist() == [0, 1, 3, 2]
    first, _ = sample_permutation(scores, 0.7, torch.Generator().manual_seed(12))
    second, _ = sample_permutation(scores, 0.7, torch.Generator().manual_seed(12))
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(4))
    with pytest.raises(ValueError):
        plackett_luce_log_prob(scores, torch.tensor([0, 0, 2, 3]))


def test_reward_and_coverage_penalty_do_not_mutate_previous_state():
    state = dict(native_pattern_count=20, previous_pattern_count=18,
                 native_covered_equivalent_faults=100, reward_ema=2.)
    new, reward = reward_transition(
        state, dict(pattern_count=15, detected_equivalent_faults=95,
                    redundant_equivalent_faults=5, covered_equivalent_faults=100), .9)
    assert reward == dict(raw_reward=3, advantage=.05, coverage_valid=True)
    assert new["previous_pattern_count"] == 15
    assert new["native_covered_equivalent_faults"] == 100
    assert new["reward_ema"] == pytest.approx(2.1)
    invalid, penalty = reward_transition(
        state, dict(pattern_count=2, detected_equivalent_faults=98,
                    redundant_equivalent_faults=1, covered_equivalent_faults=99), .9)
    assert penalty == dict(raw_reward=-20, advantage=-1.1, coverage_valid=False)
    assert invalid["previous_pattern_count"] == 18
    assert invalid["native_covered_equivalent_faults"] == 100
    assert state["reward_ema"] == 2.


class FakeEnvironment:
    """Deterministic order-dependent solver used only by trainer unit tests."""
    fail_name = None

    def run(self, bench, faultmap, ids):
        if Path(bench).name == self.fail_name:
            raise RuntimeError("injected solver failure")
        score = sum((i + 1) * int(identifier[1:]) for i, identifier in enumerate(ids))
        return dict(pattern_count=3 + score % 7, detected_equivalent_faults=len(ids),
                    detected_collapsed_faults=len(ids), uncollapsed_faults=len(ids),
                    aborted_faults=0, redundant_faults=0,
                    redundant_equivalent_faults=0, covered_equivalent_faults=len(ids),
                    fault_coverage=1.0, podem_calls=len(ids), total_backtracks=score)


@pytest.fixture
def mock_training(tmp_path, monkeypatch):
    from fault_order_rl import trainer as module
    circuits = []
    entries = []
    for name, count in (("small", 5), ("larger", 8)):
        spec = CircuitSpec(name, Path(name), Path(name + '.map'), Path(name + '.npz'), Path(name + '.json'))
        features = torch.randn(count, 257, generator=torch.Generator().manual_seed(count))
        circuits.append(CircuitData(spec, features, tuple('f' + str(i) for i in range(count)),
                                    np.ones(count, dtype=np.int64), name + '-digest'))
        entries.append(dict(name=name, bench=name, faultmap=name+'.map', embeddings=name+'.npz', metadata=name+'.json'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(version=1, circuits=entries)), encoding='utf-8')
    monkeypatch.setattr(module, 'load_all_circuits', lambda manifest, environment: circuits)
    return manifest, FakeEnvironment()


def assert_model_equal(a, b):
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_checkpoint_schema_one_requires_restart(tmp_path):
    checkpoint = tmp_path/'old.pt'
    save_checkpoint(checkpoint, {'version': 1})
    with pytest.raises(ValueError, match='detected-only coverage; restart training'):
        load_checkpoint(checkpoint)


def test_shared_update_and_exact_resume(mock_training, tmp_path):
    manifest, env = mock_training
    config = TrainConfig(rounds=2, evaluate_every=1)
    continuous = Trainer.create(manifest, config, tmp_path/'continuous', env)
    initial = copy.deepcopy(continuous.model.state_dict())
    records = continuous.step()
    assert len([r for r in records if r['kind'] == 'episode']) == 2
    assert all('loss_contribution' in r for r in records if r['kind'] == 'episode')
    continuous.step()
    assert any(not torch.equal(initial[k], continuous.model.state_dict()[k]) for k in initial)
    interrupted = Trainer.create(manifest, config, tmp_path/'interrupted', env)
    interrupted.step()
    resumed = Trainer.resume(tmp_path/'interrupted/latest.pt', environment=env)
    resumed.step()
    assert_model_equal(continuous.model.state_dict(), resumed.model.state_dict())
    assert continuous.states == resumed.states
    for name in ('small', 'larger'):
        with np.load(tmp_path/'continuous/rounds/round-000002.npz') as a, np.load(tmp_path/'interrupted/rounds/round-000002.npz') as b:
            np.testing.assert_array_equal(a[name], b[name])
    evaluated = evaluate_checkpoint(tmp_path/'interrupted/best.pt', tmp_path/'exports', env)
    assert evaluated['eligible']
    for name in ('small', 'larger'):
        with np.load(tmp_path/'exports'/ (name+'.ranking.npz')) as arrays:
            assert sorted(arrays['ranks']) == list(range(1, len(arrays['fault_ids'])+1))


def test_minibatch_updates_and_exact_resume(mock_training, tmp_path):
    manifest, env = mock_training
    config = TrainConfig(rounds=2, evaluate_every=1, batch_size=1)
    continuous = Trainer.create(manifest, config, tmp_path/'continuous-batches', env)
    first_records = continuous.step()
    updates = [record for record in first_records if record['kind'] == 'update']
    assert len(updates) == 2
    assert {tuple(record['circuit_names']) for record in updates} == {('small',), ('larger',)}
    assert {record['batch_index'] for record in updates} == {0, 1}
    continuous.step()

    interrupted = Trainer.create(manifest, config, tmp_path/'interrupted-batches', env)
    interrupted.step()
    resumed = Trainer.resume(
        tmp_path/'interrupted-batches/latest.pt', environment=env)
    resumed.step()
    assert_model_equal(continuous.model.state_dict(), resumed.model.state_dict())
    assert continuous.states == resumed.states


def test_512_circuits_form_32_batches_of_16():
    trainer = object.__new__(Trainer)
    trainer.circuits = list(range(512))
    trainer.config = SimpleNamespace(batch_size=16)
    state = capture_rng()
    try:
        torch.manual_seed(14)
        np.random.seed(14)
        import random
        random.seed(14)
        first = trainer._round_batches()
        random.seed(14)
        second = trainer._round_batches()
    finally:
        from fault_order_rl.checkpoint import restore_rng
        restore_rng(state)
    assert len(first) == 32
    assert all(len(batch) == 16 for batch in first)
    assert sorted(index for batch in first for index in batch) == list(range(512))
    assert first == second


def test_failed_round_preserves_checkpoint_model_baselines_and_rng(mock_training, tmp_path):
    manifest, env = mock_training
    trainer = Trainer.create(manifest, TrainConfig(rounds=2), tmp_path/'run', env)
    before = _sha256(tmp_path/'run/latest.pt')
    states = copy.deepcopy(trainer.states)
    model = copy.deepcopy(trainer.model.state_dict())
    rng = capture_rng()
    env.fail_name = 'larger'
    with pytest.raises(RuntimeError, match='injected'):
        trainer.step()
    assert trainer.round == 0
    assert trainer.states == states
    assert_model_equal(model, trainer.model.state_dict())
    assert _sha256(tmp_path/'run/latest.pt') == before
    assert torch.equal(capture_rng()['torch'], rng['torch'])
    env.fail_name = None
    trainer.step()
    assert trainer.round == 1


def test_evaluation_rejects_best_from_before_latest_commit(mock_training, tmp_path):
    manifest, env = mock_training
    trainer = Trainer.create(manifest, TrainConfig(rounds=1, evaluate_every=1), tmp_path/'run', env)
    stale = (tmp_path/'run/best.pt').read_bytes()
    trainer.step()
    (tmp_path/'run/best.pt').write_bytes(stale)
    with pytest.raises(ValueError, match='stale'):
        evaluate_checkpoint(tmp_path/'run/best.pt', tmp_path/'exports', env)
    Trainer.resume(tmp_path/'run/latest.pt', environment=env)
    assert evaluate_checkpoint(tmp_path/'run/best.pt', tmp_path/'exports', env)['eligible']


def test_resume_rejects_changed_manifest(mock_training, tmp_path):
    manifest, env = mock_training
    Trainer.create(manifest, TrainConfig(rounds=1), tmp_path/'run', env)
    raw = json.loads(manifest.read_text())
    raw['circuits'][0]['bench'] = 'changed'
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='manifest'):
        Trainer.resume(tmp_path/'run/latest.pt', environment=env)


def test_best_requires_native_coverage_for_each_circuit():
    report = dict(round=0, circuits={'c': dict(covered_equivalent_faults=9)},
                  totals=dict(pattern_count=2, covered_equivalent_faults=9,
                              podem_calls=3, total_backtracks=1))
    states = {'c': dict(native_covered_equivalent_faults=10)}
    assert not eligible(report, states)
    assert Trainer._choose_best({'report': report}, FaultScorer(), report, states) is None


def test_best_prioritizes_pattern_count_after_native_coverage():
    states = {'c': dict(native_covered_equivalent_faults=10)}
    model = FaultScorer()
    fewer_patterns = dict(round=1, circuits={'c': dict(covered_equivalent_faults=10)},
                          totals=dict(pattern_count=4, covered_equivalent_faults=10,
                                      podem_calls=5, total_backtracks=7))
    more_coverage = dict(round=2, circuits={'c': dict(covered_equivalent_faults=11)},
                         totals=dict(pattern_count=5, covered_equivalent_faults=11,
                                     podem_calls=4, total_backtracks=6))
    best = Trainer._choose_best(None, model, fewer_patterns, states)
    assert Trainer._choose_best(best, model, more_coverage, states) is best


def test_completed_training_falls_back_to_latest_when_no_best(mock_training, tmp_path):
    manifest, env = mock_training
    trainer = Trainer.create(manifest, TrainConfig(rounds=1), tmp_path/'run', env)
    for state in trainer.states.values():
        state['native_covered_equivalent_faults'] += 1
    trainer.best = None
    trainer.round = trainer.config.rounds
    save_checkpoint(tmp_path/'run/latest.pt', trainer._payload())
    trainer._publish_best()
    resumed = Trainer.resume(tmp_path/'run/latest.pt', environment=env)
    report = resumed.train()
    assert report['checkpoint_kind'] == 'latest'
    assert not report['coverage_eligible']
    assert report['coverage_shortfall'] == len(trainer.states)
    assert report['pattern_reduction'] == (
        report['native_pattern_total'] - report['totals']['pattern_count'])
    assert (tmp_path/'run/evaluation/summary.json').is_file()
    assert not load_checkpoint(tmp_path/'run/best.pt')['available']
    for name in trainer.states:
        assert (tmp_path/'run/evaluation'/(name+'.ranking.npz')).is_file()


def test_evaluation_writes_per_circuit_coverage_and_pattern_comparison(tmp_path):
    native = {
        'better': dict(pattern_count=10, covered_equivalent_faults=5,
                       fault_coverage=0.5),
        'worse': dict(pattern_count=8, covered_equivalent_faults=5,
                      fault_coverage=0.5),
        'zero': dict(pattern_count=0, covered_equivalent_faults=5,
                     fault_coverage=0.5),
    }
    states = {name: dict(native_covered_equivalent_faults=5) for name in native}
    report = dict(
        round=1,
        eligible=True,
        circuits={
            'better': dict(pattern_count=7, covered_equivalent_faults=6,
                           fault_coverage=0.6),
            'worse': dict(pattern_count=9, covered_equivalent_faults=4,
                          fault_coverage=0.4),
            'zero': dict(pattern_count=2, covered_equivalent_faults=5,
                         fault_coverage=0.5),
        },
        totals=dict(pattern_count=18),
    )
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    report = _complete_report(report, checkpoint, native, states, 'best')
    _write_evaluation(tmp_path/'evaluation', report, {})
    with (tmp_path/'evaluation/comparison_by_circuit.csv').open(newline='') as stream:
        rows = {row['circuit']: row for row in csv.DictReader(stream)}
    assert float(rows['better']['fault_coverage_increase_percentage_points']) == pytest.approx(10)
    assert int(rows['better']['covered_fault_increase']) == 1
    assert int(rows['better']['pattern_reduction']) == 3
    assert float(rows['better']['pattern_reduction_percent']) == pytest.approx(30)
    assert int(rows['worse']['covered_fault_increase']) == -1
    assert int(rows['worse']['pattern_reduction']) == -1
    assert rows['worse']['coverage_eligible'] == 'False'
    assert int(rows['zero']['pattern_reduction']) == -2
    assert rows['zero']['pattern_reduction_percent'] == ''
    summary = json.loads((tmp_path/'evaluation/summary.json').read_text())
    comparison = summary['comparison_by_circuit']
    assert comparison['better']['covered_fault_increase'] == 1
    assert comparison['better']['fault_coverage_increase_percentage_points'] == pytest.approx(10)
    assert comparison['worse']['pattern_reduction'] == -1
    assert comparison['zero']['pattern_reduction_percent'] is None


def test_best_can_be_evaluated_on_separate_resumable_manifest(mock_training, tmp_path):
    manifest, env = mock_training
    trainer = Trainer.create(
        manifest, TrainConfig(rounds=1, evaluate_every=1), tmp_path/'run-external', env)
    trainer.step()
    test_manifest = tmp_path/'test-manifest.json'
    raw = json.loads(manifest.read_text())
    raw['purpose'] = 'external-test'
    test_manifest.write_text(json.dumps(raw))
    output = tmp_path/'external-evaluation'
    report = evaluate_checkpoint(
        tmp_path/'run-external/best.pt', output, env, manifest=test_manifest)
    assert report['training_manifest_digest'] != report['evaluation_manifest_digest']
    assert report['evaluation_manifest'] == str(test_manifest.resolve())
    assert report['checkpoint_kind'] == 'best'
    assert json.loads((output/'status.json').read_text())['complete'] is True
    assert (output/'comparison_by_circuit.csv').is_file()
    summary = json.loads((output/'summary.json').read_text())
    assert 'totals' not in summary
    assert 'pattern_reduction' not in summary
    for name in ('small', 'larger'):
        assert (output/'circuits'/(name+'.metrics.json')).is_file()
        assert (output/(name+'.ranking.npz')).is_file()
    resumed = evaluate_checkpoint(
        tmp_path/'run-external/best.pt', output, env, manifest=test_manifest)
    assert resumed['totals']['pattern_count'] == report['totals']['pattern_count']


def test_latest_can_be_evaluated_on_separate_manifest(mock_training, tmp_path):
    manifest, env = mock_training
    trainer = Trainer.create(
        manifest, TrainConfig(rounds=1, evaluate_every=1), tmp_path/'run-latest', env)
    trainer.step()
    latest_path = tmp_path/'run-latest/latest.pt'
    latest = load_checkpoint(latest_path)
    latest['model'] = {
        key: torch.zeros_like(value) for key, value in latest['model'].items()
    }
    save_checkpoint(latest_path, latest)
    test_manifest = tmp_path/'latest-validation.json'
    raw = json.loads(manifest.read_text())
    raw['purpose'] = 'latest-external-test'
    test_manifest.write_text(json.dumps(raw))

    report = evaluate_checkpoint(
        tmp_path/'run-latest/latest.pt', tmp_path/'latest-evaluation', env,
        manifest=test_manifest)

    assert report['checkpoint_kind'] == 'latest'
    assert report['round'] == 1
    assert all(row['checkpoint_kind'] == 'latest'
               for row in report['comparison_by_circuit'].values())
    for circuit in trainer.circuits:
        with np.load(tmp_path/'latest-evaluation'/(circuit.name+'.ranking.npz')) as arrays:
            np.testing.assert_array_equal(
                arrays['permutation'], np.arange(circuit.fault_count))
        model_metrics = report['circuits'][circuit.name]
        native_metrics = report['native_metrics'][circuit.name]
        assert {key: value for key, value in model_metrics.items() if key != 'seconds'} == {
            key: value for key, value in native_metrics.items() if key != 'seconds'
        }


def test_linux_run_script_auto_resumes_existing_output():
    script = (ROOT/'scripts/run_linux.sh').read_text(encoding='utf-8')
    assert '[[ -f "$output/latest.pt" ]]' in script
    assert '--resume "$output/latest.pt"' in script


def test_anchor_linux_scripts_are_offline_rootless_and_use_anchor_manifests():
    setup = (ROOT/'scripts/setup_anchor_linux.sh').read_text(encoding='utf-8')
    train = (ROOT/'scripts/run_anchor_linux.sh').read_text(encoding='utf-8')
    evaluate = (ROOT/'scripts/evaluate_anchor_linux.sh').read_text(encoding='utf-8')
    forbidden = ('sudo ', 'apt ', 'apt-get ', 'pip install', 'conda ')
    assert not any(token in setup for token in forbidden)
    assert 'PODEM/setup.py build_ext --inplace' in setup
    assert 'configs/anchor_train_1024.json' in train
    assert '[[ -f "$output/latest.pt" ]]' in train
    assert '--resume "$output/latest.pt"' in train
    assert 'configs/anchor_validation_6.json' in evaluate
    assert '--checkpoint "$best_checkpoint"' in evaluate
    assert '--checkpoint "$latest_checkpoint"' in evaluate
    assert '"${output_base}-best"' in evaluate
    assert '"${output_base}-latest"' in evaluate


def test_evaluate_cli_prints_only_per_circuit_changes(tmp_path, monkeypatch, capsys):
    from fault_order_rl import __main__ as cli
    comparison = {
        'tiny': {
            'circuit': 'tiny', 'checkpoint_kind': 'latest', 'round': 7,
            'native_fault_coverage': 0.75, 'model_fault_coverage': 0.875,
            'fault_coverage_increase_percentage_points': 12.5,
            'native_pattern_count': 10, 'model_pattern_count': 8,
            'pattern_reduction': 2, 'pattern_reduction_percent': 20.0,
        }
    }
    monkeypatch.setattr(cli, 'evaluate_checkpoint',
                        lambda *args, **kwargs: {'comparison_by_circuit': comparison})
    result = cli.main([
        'evaluate', '--checkpoint', str(tmp_path/'latest.pt'),
        '--manifest', str(tmp_path/'validation.json'),
    ])
    output = capsys.readouterr().out
    assert result == 0
    assert 'circuit\tcheckpoint\tround\tnative_cov\tmodel_cov\tcov_delta_pp' in output
    assert 'tiny\tlatest\t7\t75.000000%\t87.500000%\t+12.500000' in output
    assert 'totals' not in output


def test_existing_pretrained_artifact_and_bench_provenance(tmp_path):
    directory = ROOT.parent/'artifacts/c432-v3-embeddings'
    if not directory.exists():
        pytest.skip('local reference artifact unavailable')
    bench = ROOT/'PODEM/sample_circuits/c432_binary.bench'
    spec = CircuitSpec('c432', bench, bench.with_suffix('.faultmap'),
                       directory/'c432_binary.fault_embeddings.npz', directory/'c432_binary.faults.json')
    env = PodemEnvironment(ROOT/'PODEM')
    catalog = env.catalog(spec.bench_path, spec.faultmap_path)
    assert load_circuit_data(spec, catalog).fault_count == 533
    changed = tmp_path/'changed.bench'
    changed.write_bytes(bench.read_bytes() + b'\n# changed\n')
    with pytest.raises(ValueError, match='BENCH provenance'):
        load_circuit_data(replace(spec, bench_path=changed), catalog)


@pytest.mark.parametrize('field,value', [('learning_rate', float('nan')), ('temperature_min', 2.), ('rounds', 0), ('backtrack_limit', 3000), ('batch_size', -1)])
def test_bad_training_config(field, value):
    with pytest.raises(ValueError):
        replace(TrainConfig(), **{field: value}).validate()


def test_missing_solver_file_is_python_error(tmp_path):
    env = PodemEnvironment(ROOT/'PODEM')
    with pytest.raises(ValueError, match='missing'):
        env.catalog(tmp_path/'absent.bench', tmp_path/'absent.map')


def test_environment_derives_uncollapsed_resolved_coverage(tmp_path, monkeypatch):
    from fault_order_rl import environment as module
    bench = tmp_path/'tiny.bench'
    faultmap = tmp_path/'tiny.faultmap'
    bench.write_text('INPUT(a)\nOUTPUT(a)\n')
    faultmap.write_text('map\n')
    calls = []

    class Binding:
        @staticmethod
        def run_stuck_at_ordered(bench_path, faultmap_path, ids, limit, seed):
            calls.append((bench_path, faultmap_path, ids, limit, seed))
            return dict(pattern_count=2, detected_collapsed_faults=2,
                        detected_equivalent_faults=4, uncollapsed_faults=7,
                        aborted_faults=0, redundant_faults=1,
                        redundant_equivalent_faults=3, podem_calls=3,
                        total_backtracks=8)

    monkeypatch.setattr(module, 'load_cpp_podem', lambda module_dir: Binding())
    result = PodemEnvironment(tmp_path).run(bench, faultmap, ['a', 'b', 'c'])
    assert result['covered_equivalent_faults'] == 7
    assert result['fault_coverage'] == 1.0
    assert calls[0][3:] == (5000, 14)


def test_environment_passes_empty_faultmap_for_original_bench(tmp_path, monkeypatch):
    from fault_order_rl import environment as module
    bench = tmp_path/'tiny.bench'
    bench.write_text('INPUT(a)\nOUTPUT(a)\n')
    calls = []

    class Binding:
        @staticmethod
        def catalog_stuck_at(bench_path, faultmap_path):
            calls.append((bench_path, faultmap_path))
            return {'faults': [], 'uncollapsed_total': 0}

        @staticmethod
        def run_stuck_at_ordered(*args):
            raise AssertionError('not used')

    monkeypatch.setattr(module, 'load_cpp_podem', lambda module_dir: Binding())
    env = PodemEnvironment(tmp_path)
    env.catalog(bench, None)
    assert calls == [(str(bench), '')]


def test_anchor_preserving_manifest_loads_original_faults_without_faultmap():
    manifest_path = ROOT/'configs/anchor_smoke_train.json'
    if not manifest_path.is_file():
        pytest.skip('anchor smoke manifest is unavailable')
    manifest = load_manifest(manifest_path)
    spec = manifest.circuits[0]
    assert spec.faultmap_path is None
    assert spec.bench_path.parent.name == 'train'
    assert spec.aig_bench_path.parent.name == 'train_AIG'
    env = PodemEnvironment(manifest.module_dir)
    circuit = load_circuit_data(spec, env.catalog(spec.bench_path, None))
    assert circuit.fault_count == 96
    assert int(circuit.eqv_fault_nums.sum()) == 248


def test_environment_rejects_binding_without_equivalent_redundant_count(tmp_path, monkeypatch):
    from fault_order_rl import environment as module
    bench = tmp_path/'tiny.bench'
    faultmap = tmp_path/'tiny.faultmap'
    bench.write_text('INPUT(a)\nOUTPUT(a)\n')
    faultmap.write_text('map\n')

    class OldBinding:
        @staticmethod
        def run_stuck_at_ordered(*args):
            return dict(pattern_count=1, detected_collapsed_faults=1,
                        detected_equivalent_faults=1, uncollapsed_faults=1,
                        aborted_faults=0, redundant_faults=0, podem_calls=1,
                        total_backtracks=0)

    monkeypatch.setattr(module, 'load_cpp_podem', lambda module_dir: OldBinding())
    with pytest.raises(RuntimeError, match='redundant_equivalent_faults'):
        PodemEnvironment(tmp_path).run(bench, faultmap, ['a'])


def test_real_mapped_circuit_training_resume_and_export(tmp_path, monkeypatch):
    from fault_embedding.__main__ import main as export_main
    monkeypatch.syspath_prepend(str(ROOT/'PODEM/scripts'))
    from convert_binary_bench import convert_binary_bench
    # Non-ASCII filenames reproduce the Windows UTF-8 bridge regression.
    source = tmp_path/'原始.bench'
    binary = tmp_path/'二值.bench'
    faultmap = binary.with_suffix('.faultmap')
    source.write_text('INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\nOUTPUT(G1)\n'
                      'y = AND(a,b,c)\n# G1 = XOR(a,b)\nW1 = NOT(a)\nZ1 = NOT(b)\n'
                      'X1 = NAND(a,Z1)\nY1 = NAND(b,W1)\nG1 = NAND(X1,Y1)\n', encoding='utf-8')
    convert_binary_bench(source, binary, faultmap)
    headers = dict(line.split() for line in faultmap.read_text().splitlines()[1:5])
    Path(str(faultmap) + '.binding.json').write_text(json.dumps(dict(
        bench_sha256=_sha256(binary), faultmap_sha256=_sha256(faultmap),
        source_hash=headers['source_hash'], circuit_hash=headers['circuit_hash'])), encoding='utf-8')
    export = tmp_path/'embeddings'
    assert export_main(['export', '--bench', str(binary), '--faultmap', str(faultmap),
                        '--out-dir', str(export)]) == 0
    manifest = tmp_path/'real.json'
    manifest.write_text(json.dumps(dict(version=1, cpp_podem_dir=str(ROOT/'PODEM'), circuits=[
        dict(name='tiny', bench=str(binary), faultmap=str(faultmap),
             embeddings=str(export/(binary.stem+'.fault_embeddings.npz')),
             metadata=str(export/(binary.stem+'.faults.json')))])), encoding='utf-8')
    trainer = Trainer.create(manifest, TrainConfig(rounds=2, evaluate_every=1), tmp_path/'real-run')
    assert all(not f.startswith('__smartatpg_bin_') for f in trainer.circuits[0].fault_ids)
    native = trainer.native_metrics['tiny']
    circuit = trainer.circuits[0]
    assert trainer.environment.run(binary, faultmap, circuit.fault_ids) == native
    reversed_metrics = trainer.environment.run(binary, faultmap, tuple(reversed(circuit.fault_ids)))
    assert trainer.environment.run(binary, faultmap, tuple(reversed(circuit.fault_ids))) == reversed_metrics
    trainer.step()
    resumed = Trainer.resume(tmp_path/'real-run/latest.pt')
    resumed.step()
    report = evaluate_checkpoint(tmp_path/'real-run/best.pt', tmp_path/'real-eval')
    assert report['eligible']
    assert report['totals']['covered_equivalent_faults'] >= native['covered_equivalent_faults']


def test_training_seed_does_not_change_fixed_solver_protocol(mock_training, tmp_path, monkeypatch):
    from fault_order_rl import trainer as module
    manifest, env = mock_training
    calls = []
    def factory(module_dir, limit, seed):
        calls.append((limit, seed))
        return env
    monkeypatch.setattr(module, 'PodemEnvironment', factory)
    Trainer(manifest, TrainConfig(seed=93), tmp_path/'seed-test')
    assert calls == [(5000, 14)]


def test_train_cli_skips_internal_config_without_argument(tmp_path, monkeypatch, capsys):
    from fault_order_rl import __main__ as cli
    captured = {}
    fake = SimpleNamespace(train=lambda: {
        'round': 0, 'totals': {'pattern_count': 1}, 'pattern_reduction': 0,
    })

    def create(manifest, config, output):
        captured['config'] = config
        return fake

    monkeypatch.setattr(cli.Trainer, 'create', create)
    result = cli.main([
        'train', '--manifest', str(tmp_path/'manifest.json'),
        '--rounds', '1', '--output', str(tmp_path/'run'),
    ])
    assert result == 0
    assert captured['config'].rounds == 1
    assert captured['config'].backtrack_limit == 5000
    assert json.loads(capsys.readouterr().out)['checkpoint_kind'] == 'best'


def test_physical_xor_is_rejected_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'PODEM/scripts'))
    from convert_binary_bench import convert_binary_bench
    source = tmp_path/'raw.bench'
    binary = tmp_path/'binary.bench'
    source.write_text('INPUT(a)\nINPUT(b)\nOUTPUT(q)\nq = XOR(a,b)\n')
    faultmap = binary.with_suffix('.faultmap')
    convert_binary_bench(source, binary, faultmap)
    env = PodemEnvironment(ROOT/'PODEM')
    ids = [f['fault_id'] for f in env.catalog(binary, faultmap)['faults']]
    with pytest.raises(RuntimeError, match='expanded'):
        env.run(binary, faultmap, ids)
