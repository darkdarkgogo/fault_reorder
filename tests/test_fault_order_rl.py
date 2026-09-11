"""Policy math, shared updates, coverage gating and exact round recovery."""

import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fault_order_rl.checkpoint import capture_rng, load_checkpoint, save_checkpoint
from fault_order_rl.data import CircuitData, CircuitSpec, _sha256, load_circuit_data
from fault_order_rl.environment import PodemEnvironment
from fault_order_rl.model import FaultScorer
from fault_order_rl.policy import deterministic_permutation, plackett_luce_log_prob, sample_permutation
from fault_order_rl.trainer import TrainConfig, Trainer, eligible, evaluate_checkpoint, reward_transition


ROOT = Path(__file__).resolve().parents[1]


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
                 required_detected_equivalent_faults=100, reward_ema=2.)
    new, reward = reward_transition(state, dict(pattern_count=15, detected_equivalent_faults=101), .9)
    assert reward == dict(raw_reward=3, advantage=.05, coverage_valid=True)
    assert new["previous_pattern_count"] == 15
    assert new["required_detected_equivalent_faults"] == 100
    assert new["reward_ema"] == pytest.approx(2.1)
    invalid, penalty = reward_transition(state, dict(pattern_count=2, detected_equivalent_faults=99), .9)
    assert penalty == dict(raw_reward=-20, advantage=-1.1, coverage_valid=False)
    assert invalid["previous_pattern_count"] == 18
    assert invalid["required_detected_equivalent_faults"] == 100
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
                    aborted_faults=0, redundant_faults=0, podem_calls=len(ids), total_backtracks=score)


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


def test_old_best_is_invalidated_when_detection_requirement_rises():
    report = dict(round=0, circuits={'c': dict(detected_equivalent_faults=9)},
                  totals=dict(pattern_count=2, detected_equivalent_faults=9, podem_calls=3, total_backtracks=1))
    states = {'c': dict(required_detected_equivalent_faults=10)}
    assert not eligible(report, states)
    assert Trainer._choose_best({'report': report}, FaultScorer(), report, states) is None


def test_completed_training_falls_back_to_latest_when_no_best(mock_training, tmp_path):
    manifest, env = mock_training
    trainer = Trainer.create(manifest, TrainConfig(rounds=1), tmp_path/'run', env)
    for state in trainer.states.values():
        state['required_detected_equivalent_faults'] += 1
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


def test_linux_run_script_auto_resumes_existing_output():
    script = (ROOT/'scripts/run_linux.sh').read_text(encoding='utf-8')
    assert '[[ -f "$output/latest.pt" ]]' in script
    assert '--resume "$output/latest.pt"' in script


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


@pytest.mark.parametrize('field,value', [('learning_rate', float('nan')), ('temperature_min', 2.), ('rounds', 0), ('backtrack_limit', 30)])
def test_bad_training_config(field, value):
    with pytest.raises(ValueError):
        replace(TrainConfig(), **{field: value}).validate()


def test_missing_solver_file_is_python_error(tmp_path):
    env = PodemEnvironment(ROOT/'PODEM')
    with pytest.raises(ValueError, match='missing'):
        env.catalog(tmp_path/'absent.bench', tmp_path/'absent.map')


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
    assert report['totals']['detected_equivalent_faults'] >= native['detected_equivalent_faults']


def test_training_seed_does_not_change_fixed_solver_protocol(mock_training, tmp_path, monkeypatch):
    from fault_order_rl import trainer as module
    manifest, env = mock_training
    calls = []
    def factory(module_dir, limit, seed):
        calls.append((limit, seed))
        return env
    monkeypatch.setattr(module, 'PodemEnvironment', factory)
    Trainer(manifest, TrainConfig(seed=93), tmp_path/'seed-test')
    assert calls == [(3000, 14)]


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
    assert captured['config'].backtrack_limit == 3000
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
