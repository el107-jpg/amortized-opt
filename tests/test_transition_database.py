import numpy as np
import pytest
from isaac_rl.transition_assignment import TransitionTable
from scripts.rebuild_transition_database import accepted_gates, combine_minima


def inputs():
    shape = (2, 2, 2, 1)
    table = TransitionTable(('a', 'b'), np.array([0.]), np.full(shape, np.inf),
        np.zeros(shape, int), np.full(shape, 3), np.zeros((3, 3)),
        np.ones((3, 3), bool), {'samples': 3})
    keys = np.array([[0, 0, 1, 0, s] for s in range(3)]+[[1, 1, 0, 0, 0]])
    eligible = np.array([True, True, False, True])
    deadlines = np.array([1., .1, .2])
    success = np.array([[False, False, True], [False, False, False],
                        [True, True, True], [False, False, False]])
    return table, keys, eligible, deadlines, success


def test_rebuild_preserves_shorter_raw_witnesses_and_censored_admissible_starts():
    table, keys, eligible, times, success = inputs()
    rows = dict(keys=keys[:3], before_success=success[:3, 0], after_success=np.array([True]*3))
    result = combine_minima(table, keys, eligible, times, success, [(1., rows)])
    k = (0, 0, 1, 0)
    np.testing.assert_array_equal(result['raw_minima'][k], [.2, np.inf, np.inf])
    np.testing.assert_array_equal(result['repaired_minima'][k], [.2, 1., np.inf])
    assert result['trials'][k] == 2  # invalid departure excluded from both tables
    assert result['repaired_stats']['count'][k] == 2
    assert result['repaired_stats']['mean'][k] == pytest.approx(.6)
    assert result['repaired_stats']['std'][k] == pytest.approx(.4)
    other = (1, 1, 0, 0)
    assert result['trials'][other] == 1  # unrepairable valid start stays in denominator
    assert result['repaired_stats']['count'][other] == 0
    assert np.isinf(result['repaired_stats']['mean'][other])
    assert result['coverage'].sum() == 3
    assert result['baseline_acceptance_mismatches'] == 0


def test_rebuild_finds_minimum_across_nonmonotonic_deadlines_and_rejects_duplicate_shards():
    table, keys, eligible, times, success = inputs()
    rows = dict(keys=keys[:1], before_success=np.array([False]), after_success=np.array([True]))
    result = combine_minima(table, keys, eligible, times, success, [(1., rows), (.1, rows)])
    assert result['repaired_minima'][tuple(keys[0])] == pytest.approx(.1)
    with pytest.raises(ValueError, match='Overlapping'):
        combine_minima(table, keys, eligible, times, success, [(.1, rows), (.1, rows)])


def test_database_checks_nonlinear_gates_for_every_claimed_qp_witness():
    rows = {f'after_{k}': np.zeros(2) for k in ('terminal_region_dist', 'collision_min_sdf',
        'torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max')}
    rows.update(after_success=np.ones(2, bool), after_contact_dir_cos=np.ones(2),
                after_stick_elev_deg=np.full(2, 20.))
    accepted_gates(rows)
    rows['after_torque_excess_max'][1] = .01
    with pytest.raises(ValueError, match='physical diagnostics'): accepted_gates(rows)


def test_hand_policy_removes_sources_and_destinations_from_minima_and_support():
    table, keys, eligible, times, success = inputs()
    table.metadata['forbidden_drums_by_hand'] = {'left': ['b']}
    rows = dict(keys=keys, before_success=success[:, 0], after_success=np.ones(4, bool))
    result = combine_minima(table, keys, eligible, times, success, [(1., rows)])
    assert np.isinf(result['repaired_minima'][0]).all()
    assert result['trials'][0].sum() == 0
    assert result['repaired_stats']['count'][0].sum() == 0
    assert result['repaired_minima'][1, 1, 0, 0, 0] == 1.
    assert not result['allowed_groups'][0, 1].any()
    assert not result['allowed_groups'][0, :, 1].any()
    assert result['coverage'].sum() == 4  # retain the original diagnostic sweep


def test_floor_tom_policy_excludes_52_groups_without_double_counting():
    shape = (2, 7, 7, 4)
    table = TransitionTable(tuple('abcdef')+('floor_tom',), np.arange(4.),
        np.full(shape, np.inf), np.zeros(shape, int), np.zeros(shape, int),
        np.zeros((8, 8)), np.ones((8, 8), bool),
        {'forbidden_drums_by_hand': {'left': ['floor_tom']}})
    assert (~table.allowed_groups()).sum() == 52
    assert table.allowed_groups().sum() == 340


def test_saved_starts_are_rechecked_when_idle_arm_is_ignored(tmp_path):
    import torch
    from types import SimpleNamespace
    from scripts.audit_transition_deadlines import records
    shape = (2, 2, 2, 1)
    table = TransitionTable(('snare', 'floor_tom'), np.array([.3]), np.full(shape, np.inf),
        np.zeros(shape, int), np.ones(shape, int), np.zeros((3, 3)),
        np.ones((3, 3), bool), {'samples': 1, 'forbidden_drums_by_hand': {'left': ['floor_tom']}})
    (tmp_path/'samples').mkdir()
    for hand in (0, 1):
        for source in (0, 1):
            np.savez(tmp_path/'samples'/f'h{hand}_d{source}_y0.npz',
                q=np.zeros((1, 17), np.float32), v=np.zeros((1, 17), np.float32),
                start_valid=[False], ik_error_m=[0.], outward_speed_m_s=[.5], durations=[[1.]],
                **{f'{d}_{name}': np.array([[False]]) for d in (0, 1) for name in ('success', 'geometric')})
    seen = []
    def collision(frames, B, S, ctx, masks, raw, viol, live, diag, counts, sp, **options):
        seen.append(masks)
        assert options['active_hand_constraints_only']
        diag['collision_min_sdf'] = torch.ones(B)*.01
        diag['collision_stick_min_sdf'] = torch.ones(B)*1000
    cost = SimpleNamespace(fk=SimpleNamespace(forward=lambda q: {}), _collision=collision)
    from amortized_opt.rhythm import DrumKit
    original = records(tmp_path, table, 'cpu')
    assert not original['valid'].any()
    updated = records(tmp_path, table, 'cpu', cost=cost, kit=DrumKit.from_layout(), ignore_idle_arm=True)
    assert updated['valid'].sum() == 5
    valid_keys = updated['keys'][updated['valid']]
    assert not ((valid_keys[:, 0] == 0) & ((valid_keys[:, 1] == 1) | (valid_keys[:, 2] == 1))).any()
    assert len(seen) == 4
