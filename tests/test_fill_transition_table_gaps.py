import argparse
import json

import numpy as np
import pytest

from scripts.fill_transition_table_gaps import apply_witness, ingest_results, load_piece, prepare_cases
from scripts.build_unrestricted_transition_chart import aggregate, allowed_pairs
from scripts.probe_transition_qp import save_npz
import torch
import scripts.fill_transition_table_gaps as updater
from isaac_rl.trajectory_qp import QPConfig


NAMES = ['snare', 'hihat', 'tom1', 'tom2', 'cymbal_left', 'cymbal_right', 'floor_tom']


def fixture_piece():
    piece = dict(hand=np.asarray(0), source=np.asarray(2),
        destination=np.array([4, 4, 4, 1]), sample=np.array([0, 2, 3, 0]),
        q=np.arange(12, dtype=np.float32).reshape(4, 3), v=np.zeros((4, 3), np.float32),
        paths=np.zeros((4, 100, 3), np.float32), minimum_s=np.array([np.inf, .3, np.inf, .2]),
        decoder_minimum_s=np.full(4, np.inf))
    for key in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err'):
        piece[key] = np.zeros(4)
    return piece


def fixture_values(success=True):
    values = {key: np.array([0.]) for key in ('torque_excess_max', 'velocity_excess_max',
        'joint_box_excess_max', 'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err')}
    values['success'] = np.array([success])
    return values


def test_commit_preserves_departure_and_updates_only_matching_row_and_statistics():
    piece = fixture_piece()
    q, v = piece['q'].copy(), piece['v'].copy()
    case = dict(hand=0, source=2, destination=4, sample=3)
    path = np.ones((1, 10, 3), np.float32)
    assert apply_witness(piece, case, q[2:3], v[2:3], path, fixture_values())
    np.testing.assert_array_equal(piece['q'], q)
    np.testing.assert_array_equal(piece['v'], v)
    np.testing.assert_array_equal(piece['paths'][2, :10], path[0])
    assert not piece['paths'][2, 10:].any()
    assert np.isinf(piece['minimum_s'][0])
    result = aggregate(NAMES, [piece])
    assert result['support'][0, 2, 4] == 2
    assert result['trials'][0, 2, 4] == 3
    assert result['support_fraction'][0, 2, 4] == 2/3
    assert result['mean_minimum_s'][0, 2, 4] == .25
    assert result['median_minimum_s'][0, 2, 4] == .25
    assert result['p90_minimum_s'][0, 2, 4] == pytest.approx(.29)
    assert result['std_minimum_s'][0, 2, 4] == pytest.approx(.1/np.sqrt(2))
    assert np.isinf(result['all_start_p90_s'][0, 2, 4])


@pytest.mark.parametrize('failure', ['q', 'v', 'sample', 'destination', 'hand', 'source', 'success', 'path'])
def test_rejects_bad_witness_without_mutating_piece(failure):
    piece = fixture_piece()
    before = {k: v.copy() for k, v in piece.items()}
    case = dict(hand=0, source=2, destination=4, sample=0)
    q, v = piece['q'][:1].copy(), piece['v'][:1].copy()
    path, values = np.ones((1, 10, 3)), fixture_values()
    if failure == 'q': q += .01
    elif failure == 'v': v += .01
    elif failure == 'success': values['success'][0] = False
    elif failure == 'path': path[0, 0, 0] = np.nan
    else: case[failure] = 9
    with pytest.raises((ValueError, AssertionError)):
        apply_witness(piece, case, q, v, path, values)
    for key in before:
        np.testing.assert_array_equal(piece[key], before[key])


def test_slower_witness_does_not_replace_shorter_path():
    piece = fixture_piece()
    assert not apply_witness(piece, dict(hand=0, source=2, destination=4, sample=2),
        piece['q'][1:2], piece['v'][1:2], np.ones((1, 20, 3)), fixture_values())
    assert piece['minimum_s'][1] == .3
    assert not piece['paths'].any()


def test_decoder_support_is_retained_when_qp_witness_is_already_shorter():
    piece = fixture_piece()
    assert apply_witness(piece, dict(hand=0, source=2, destination=4, sample=2),
        piece['q'][1:2], piece['v'][1:2], np.ones((1, 20, 3)), fixture_values(), raw_minimum_s=.4)
    assert piece['minimum_s'][1] == .3
    assert piece['decoder_minimum_s'][1] == .4
    assert not piece['paths'].any()


def test_final_worker_result_is_ingested_after_previous_scan(tmp_path):
    table = tmp_path/'table'; out = tmp_path/'out'
    (table/'w0').mkdir(parents=True); (out/'w0').mkdir(parents=True)
    args = argparse.Namespace(table_run=table, out=out)
    piece = fixture_piece(); source = table/'w0/h0_source_2.npz'
    save_npz(source, **piece)
    applied = set()
    assert not ingest_results(args, applied)

    artifact = out/'w0/case_0.npz'
    save_npz(artifact, q=piece['q'][:1], v=piece['v'][:1], trajectory=np.ones((1, 6, 3)))
    result = dict(case=dict(hand=0, source=2, destination=4, sample=0), status='complete',
        artifact=str(artifact), final_metrics={k: float(v[0]) for k, v in fixture_values().items()},
        raw_minimum_s=None)
    (out/'w0/case_0.json').write_text(json.dumps(result))
    assert ingest_results(args, applied)
    assert load_piece(source)['minimum_s'][0] == .12
    assert not ingest_results(args, applied)


@pytest.mark.parametrize('passing_horizon', [75, None])
@pytest.mark.parametrize('per_start', [False, True])
def test_recovery_search_keeps_departures_and_does_not_prune_after_failed_long_deadline(
        tmp_path, monkeypatch, passing_horizon, per_start):
    source = tmp_path/'table/w0'; source.mkdir(parents=True)
    folder = tmp_path/'out/w0'; folder.mkdir(parents=True)
    piece = fixture_piece()
    piece['q'] = np.zeros((4, 17), np.float32)
    piece['q'][:, 0] = np.arange(4)
    piece['v'] = np.full((4, 17), .123, np.float32)
    piece['paths'] = np.zeros((4, 100, 17), np.float32)
    save_npz(source/'h0_source_2.npz', **piece)
    args = argparse.Namespace(table_run=source.parent, device='cpu', recovery_starts=3)
    case = dict(case=0, hand=0, source=2, destination=4, hand_name='left',
                source_name='tom1', destination_name='cymbal_left', recovery=True)
    if per_start: case.update(per_start=True, sample=3)
    horizons = []
    def fake_context(kit, h, sources, destinations, N, device):
        return np.full((len(sources), 2), -1), N, N
    def fake_metrics(cost, q, v, trajectory, ctx, lengths, *a, **kw):
        assert torch.all(v == .123)
        result = {k: np.zeros(len(q)) for k in fixture_values()}
        result['merit'] = q[:, 0].numpy().copy()
        result['success'] = np.full(len(q), bool(trajectory[0, 0, 1] == 99 and ctx == passing_horizon))
        return result
    def fake_refine(cost, q, v, trajectory, ctx, lengths, h, *a, **kw):
        horizons.append(ctx)
        path = trajectory.clone(); path[:, :, 1] = 99
        return path, None, fake_metrics(cost, q, v, path, ctx, lengths), None
    class Generator:
        def predict(self, q, v, *a, **kw):
            return dict(traj=q[:, None].expand(-1, 100, -1).clone())
    import amortized_opt.train_cost as train_cost
    monkeypatch.setattr(train_cost, 'dls_terminal_teacher', lambda traj, *a, **kw: (traj, None, None))
    monkeypatch.setattr(updater, 'context', fake_context)
    monkeypatch.setattr(updater, 'metrics', fake_metrics)
    monkeypatch.setattr(updater, 'refine', fake_refine)
    result = updater.recover_witness(args, case, None, None, Generator(),
        lambda N, dense=False: N, QPConfig(), folder)
    assert 100 in horizons
    record = json.loads((folder/'recovery_0.json').read_text())
    assert record['sample_ids'] == ([3] if per_start else [0, 2, 3])
    if passing_horizon is None:
        assert result is None
        assert set(horizons) == {10, 20, 30, 40, 50, 75, 100}
        assert record['status'] == 'unresolved_budget_exhausted'
    else:
        assert result is not None
        current, q, v, path = result
        assert current['initial_s'] == 1.5 and current['sample'] == (3 if per_start else 0)
        row = 2 if per_start else 0
        torch.testing.assert_close(q, torch.tensor(piece['q'][row:row+1]))
        torch.testing.assert_close(v, torch.tensor(piece['v'][row:row+1]))
        assert record['status'] == 'witness_found'


def test_parallel_recovery_forwards_results_without_mutating_source_and_final_count(tmp_path):
    from scripts.recover_transition_table_gaps import forward_results
    out, importer = tmp_path/'recovery', tmp_path/'importer'
    (out/'w0').mkdir(parents=True); importer.mkdir()
    result = dict(status='complete', case=dict(hand=1, source=3, destination=6, sample=1),
                  minimum_s=.8, artifact='/saved/path.npz')
    (out/'w0/case_0.json').write_text(json.dumps(result))
    forwarded = set()
    forward_results(out, importer, forwarded)
    paths = list(importer.glob('w*/witness_*.json'))
    assert len(paths) == 1
    assert not list(importer.glob('w*/case_*.json'))
    actual = json.loads(paths[0].read_text())
    assert actual['status'] == 'verified_witness'
    assert actual['case'] == result['case']
    forward_results(out, importer, forwarded)
    assert len(list(importer.glob('w*/witness_*.json'))) == 1


def test_per_start_queue_includes_failed_samples_in_supported_pairs_and_excludes_floor_left(tmp_path):
    table, out = tmp_path/'table', tmp_path/'out'
    (table/'w0').mkdir(parents=True); out.mkdir()
    piece = fixture_piece()
    save_npz(table/'w0/h0_source_2.npz', **piece)
    excluded = fixture_piece(); excluded['source'] = np.asarray(6)
    excluded['minimum_s'][:] = np.inf
    save_npz(table/'w0/h0_source_6.npz', **excluded)
    aggregated = aggregate(NAMES, [piece])
    assert np.isfinite(aggregated['minimum_s'][0, 2, 4])
    save_npz(table/'transition_table.npz', drum_names=np.asarray(NAMES), **aggregated)
    cases = prepare_cases(argparse.Namespace(table_run=table, out=out, per_start=True, stride=3))
    assert [(c['source'], c['destination'], c['sample']) for c in cases] == [(2, 4, 0), (2, 4, 3)]
    assert all(c['per_start'] and c['recovery'] for c in cases)
    np.testing.assert_array_equal(load_piece(table/'w0/h0_source_2.npz')['q'], piece['q'])


def test_per_start_queue_waits_for_all_writers_and_reports_failed_prerequisite(tmp_path):
    from scripts.queue_transition_start_recovery import prerequisites_complete
    a, b = tmp_path/'a', tmp_path/'b'
    a.mkdir(); b.mkdir()
    (a/'run.json').write_text(json.dumps(dict(status='complete')))
    (b/'run.json').write_text(json.dumps(dict(status='running', launcher_pid=123)))
    assert not prerequisites_complete([a, b])
    (b/'run.json').write_text(json.dumps(dict(status='failed', error='test failure')))
    with pytest.raises(RuntimeError, match='Prerequisite failed'):
        prerequisites_complete([a, b])
    (b/'run.json').write_text(json.dumps(dict(status='complete')))
    assert prerequisites_complete([a, b])
