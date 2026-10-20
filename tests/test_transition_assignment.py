import numpy as np
import pytest

from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl.assignment_search import hit_groups
from isaac_rl.transition_assignment import (
    AssignmentScoreConfig, TransitionTable, minimum_statistics, slack_cost,
    score_assignment, score_assignments, search_assignments,
)


def table():
    shape = (2, 2, 2, 3)
    return TransitionTable(('a', 'b'), np.array([-.5, 0., .5]), np.full(shape, .2),
        np.full(shape, 8), np.full(shape, 8), np.zeros((3, 3)), np.ones((3, 3), bool), {})


def clip(rows):
    hits = [Hit(t, t, d, ('a', 'b')[d], 'left', .5) for t, d in rows]
    return RhythmClip('test', hits, hits[-1].time)


def test_average_minimum_per_start_preserves_censored_failures():
    duration = np.array([[[.1, .4, .8], [.1, .3, .9], [.2, .4, .8]]])
    ok = np.array([[[True, True, False], [False, True, True], [False, False, False]]])
    result = minimum_statistics(duration, ok)
    np.testing.assert_allclose(result['mean'], [.2])
    np.testing.assert_allclose(result['std'], [.1])
    assert result['count'][0] == 2 and result['trials'][0] == 3
    assert np.isinf(result['minima'][0, 2])
    with np.errstate(invalid='raise'):
        unsupported = minimum_statistics(duration, np.zeros_like(ok))
    assert np.isinf(unsupported['mean']).all()
    assert (unsupported['count'] == 0).all()


def test_descending_search_starts_at_one_second_and_keeps_shorter_successes():
    from isaac_rl.transition_calibration import descending_deadlines
    deadlines = descending_deadlines()
    assert deadlines[0] == 1.
    np.testing.assert_allclose(deadlines[:47], np.arange(50, 3, -1)*.02)
    assert len(deadlines) == 51
    # A failed long reference must not censor a valid shorter neural output.
    success = np.zeros((1, 1, len(deadlines)), bool)
    success[0, 0, np.flatnonzero(np.isclose(deadlines, .4))[0]] = True
    stats = minimum_statistics(deadlines[None, None], success)
    assert stats['mean'][0] == pytest.approx(.4)


def test_full_range_search_includes_failures_of_reference_and_all_horizon_ticks():
    from isaac_rl.transition_calibration import full_range_deadlines
    deadlines = full_range_deadlines()
    assert deadlines[0] == 1.
    assert len(deadlines) == 100
    np.testing.assert_allclose(np.sort(deadlines), np.arange(1, 101)*.02)
    success = np.zeros((1, 2, len(deadlines)), bool)
    success[0, 0, np.flatnonzero(np.isclose(deadlines, .02))[0]] = True
    success[0, 1, np.flatnonzero(np.isclose(deadlines, 1.82))[0]] = True
    stats = minimum_statistics(np.broadcast_to(deadlines, success.shape), success)
    np.testing.assert_allclose(stats['minima'], [[.02, 1.82]])
    assert stats['count'][0] == 2


def test_parallel_audit_merges_disjoint_states_and_rejects_overlap(tmp_path):
    from scripts.audit_transition_deadlines import combine_worker_audits
    keys = np.asarray([[0, 0, 1, 0, 0], [0, 0, 1, 0, 1]])
    deadlines = np.array([1., .98])
    files = []
    for rank in range(2):
        live = np.arange(2) == rank
        success = np.zeros((2, 2), bool); success[rank, rank] = True
        gap = np.full((2, 2), np.nan); gap[rank] = [.01*rank, .02*rank]
        path = tmp_path/f'w{rank}.npz'
        np.savez(path, keys=keys, valid_start=live, deadlines_s=deadlines,
                 success=success, terminal_region_dist=gap)
        files.append(path)
    merged_keys, valid, grid, result = combine_worker_audits(files)
    np.testing.assert_array_equal(merged_keys, keys)
    assert valid.all()
    np.testing.assert_array_equal(result['success'], np.eye(2, dtype=bool))
    assert np.isfinite(result['terminal_region_dist']).all()
    np.testing.assert_array_equal(grid, deadlines)
    with pytest.raises(ValueError, match='overlap'):
        combine_worker_audits([files[0], files[0]])


def test_slack_cost_steep_near_minimum_and_flat_after_100ms():
    cost = slack_cost(np.array([-.05, 0., .02, .1, .2]))
    assert np.all(np.diff(cost) < 0)
    assert cost[0] > 100*cost[1]
    assert cost[3] < .001
    step = .0001
    near_slope = (slack_cost(0.)-slack_cost(step))/step
    far_slope = (slack_cost(.1)-slack_cost(.1+step))/step
    assert near_slope > 1000*far_slope


def test_search_uses_full_hand_intervals_and_preserves_simultaneous_hits():
    t = table()
    rhythm = clip([(.3, 0), (.34, 1), (.7, 0), (.7, 1)])
    initial = np.full((2, 2, 3), .1)
    best, rows = search_assignments(rhythm, t, yaw_now=0, initial_minimum_s=initial)
    assert len(rows) == 8
    # Avoid the 40 ms same-hand transition by allocating the first two hits differently.
    assert best['actions'][0] != best['actions'][1]
    assert set(best['assigned_drums'][-1]) == {0, 1}
    assert len(best['edges']) == 4


def test_probe_recognizes_unassignable_pairs_and_preserves_allowed_pairs():
    from scripts.probe_transition_assignment import hand_choices
    names=('hihat','cymbal_left','floor_tom')
    shape=(2,3,3,1)
    t=TransitionTable(names,np.array([0.]),np.full(shape,.2),np.ones(shape,int),
        np.ones(shape,int),np.zeros((4,4)),np.ones((4,4),bool),
        {'forbidden_drums_by_hand':{'left':['floor_tom'],'right':['hihat','cymbal_left']}})
    def rhythm(drums):
        hits=[Hit(.3,.3,d,names[d],'left',.5) for d in drums]
        return RhythmClip('pair',hits,.3)
    assert hand_choices(rhythm([0,1]),t)==[[]]
    assert hand_choices(rhythm([0,2]),t)==[[0]]
    assert hand_choices(rhythm([2]),t)==[[1]]


def test_shared_next_goal_timeline_and_time_weighted_yaw():
    t = table()
    t.pair_yaw[0, 0] = .5
    t.pair_yaw[0, 1] = -.5
    rhythm = clip([(.3, 0), (.4, 0), (.6, 1)])
    row = score_assignment(hit_groups(rhythm), [0, 1, 1], t, yaw_now=0,
                           initial_minimum_s=np.full((2, 2, 3), .1))
    assert row['interval_pair_goals'] == [[0, 0], [2, 0], [2, 1]]
    assert row['interval_yaw_rad'] == [.5, 0., 0.]
    # Right's first transition integrates .5 for .3 s and zero for .1 s.
    assert row['edges'][1]['average_yaw_rad'] == pytest.approx(.375)


def test_yaw_cost_counts_initial_movement_once_and_penalizes_short_deadlines():
    t = table(); t.pair_yaw[0, 2] = .5
    cfg = AssignmentScoreConfig(slack_weight=0, deficit_weight=0)
    initial = np.full((2, 2, 3), .01)
    slow = score_assignment(hit_groups(clip([(.5, 0)])), [0], t, yaw_now=0,
                            initial_minimum_s=initial, cfg=cfg)
    fast = score_assignment(hit_groups(clip([(.1, 0)])), [0], t, yaw_now=0,
                            initial_minimum_s=initial, cfg=cfg)
    already_turned = score_assignment(hit_groups(clip([(.1, 0)])), [0], t, yaw_now=.5,
                            initial_minimum_s=initial, cfg=cfg)
    assert fast['yaw_cost'] > slow['yaw_cost'] > 0
    assert already_turned['yaw_cost'] == 0


def test_unknown_initial_state_and_missing_evidence_are_explicit(tmp_path):
    t = table()
    with pytest.raises(ValueError, match='measured-state'):
        search_assignments(clip([(.3, 0)]), t, yaw_now=0)
    initial = np.full((2, 2, 3), np.inf)
    best, _ = search_assignments(clip([(.3, 0)]), t, yaw_now=0, initial_minimum_s=initial)
    assert best['missing_edges'] == 1
    assert best['total'] >= 1000 and np.isfinite(best['total'])
    t.minimum_s[:] = np.inf; t.support[:] = 0
    path = tmp_path/'table.npz'; t.save(path)
    loaded = TransitionTable.load(path)
    assert loaded.drum_names == t.drum_names and np.isinf(loaded.minimum_s).all()


def test_vectorized_bank_matches_scalar_with_coupled_yaw_and_missing_evidence():
    from itertools import product
    rng = np.random.default_rng(13)
    t = table()
    t.minimum_s[:] = rng.uniform(.08, .4, t.minimum_s.shape)
    t.minimum_s[0, 0, 1, 1] = np.inf
    t.support[:] = rng.integers(0, 9, t.support.shape)
    t.pair_yaw[:] = rng.uniform(-.5, .5, t.pair_yaw.shape)
    t.pair_supported[0, 1] = False
    groups = hit_groups(clip([(.14, 0), (.3, 1), (.3, 0), (.44, 1), (.8, 0)]))
    actions = np.asarray(list(product((0, 1), repeat=len(groups))))
    initial = rng.uniform(.1, .5, (2, 2, 3)); initial[0, 1, 1] = np.inf
    kwargs = dict(yaw_now=.12, initial_drums=(-1, 1), initial_minimum_s=initial)
    batch = score_assignments(groups, actions, t, **kwargs)
    for i, action in enumerate(actions):
        scalar = score_assignment(groups, action, t, **kwargs)
        for key, value in batch.items():
            assert value[i] == pytest.approx(scalar[key])


def test_hand_eligibility_prunes_search_and_cannot_be_bypassed_by_live_initial_estimates(tmp_path):
    t = table()
    t.metadata['forbidden_drums_by_hand'] = {'left': ['b']}
    path = tmp_path/'restricted.npz'; t.save(path)
    t = TransitionTable.load(path)
    initial = np.full((2, 2, 3), .01)
    rhythm = clip([(.3, 1), (.6, 0), (.6, 1)])
    groups = hit_groups(rhythm)
    kwargs = dict(yaw_now=0., initial_minimum_s=initial)
    best, candidates = search_assignments(rhythm, t, **kwargs)
    assert len(candidates) == 1
    assert best['assigned_drums'] == [[-1, 1], [0, 1]]
    actions = np.array([[0, 0], [1, 0]])
    batch = score_assignments(groups, actions, t, **kwargs)
    assert np.isinf(batch['total'][0]) and np.isfinite(batch['total'][1])
    for i, action in enumerate(actions):
        scalar = score_assignment(groups, action, t, **kwargs)
        for key, values in batch.items():
            assert values[i] == pytest.approx(scalar[key])
    with pytest.raises(ValueError, match='no assignment'):
        search_assignments(clip([(.3, 1), (.3, 1)]), t, **kwargs)
    with pytest.raises(ValueError, match='initial drum violates'):
        search_assignments(rhythm, t, initial_drums=(1, 0), **kwargs)


@pytest.mark.parametrize('yaw', [-.3, np.array([-.35, -.30, -.25, -.20])])
def test_near_head_starts_use_environment_frame_and_preserve_outward_velocity(yaw):
    import torch
    from amortized_opt.cost import CostConfig, FusedCost
    from amortized_opt.rhythm import DrumKit
    from amortized_opt.ik import tip_position
    from amortized_opt.strike import default_contact_sigma
    from isaac_rl.rnea_cache import build_or_load
    from isaac_rl.transition_calibration import near_head_starts
    cost = FusedCost(CostConfig(), max_horizon=10, device='cpu', rnea=build_or_load(device='cpu'))
    kit = DrumKit.from_layout()
    source = kit.names.index('snare')
    q, v, valid, diag = near_head_starts(cost, kit, 'left', source, yaw, 4, np.random.default_rng(7))
    assert np.min(diag['ik_error_m']) < .003
    assert np.max(diag['ik_error_m']) < .01
    assert valid.any()
    assert np.min(diag['outward_speed_m_s']) > .05
    sigma = default_contact_sigma('snare')
    point = tip_position(cost.fk, q, 'left', sigma=sigma, frame='env')
    height = (point-point.new_tensor(kit.positions[source]))@point.new_tensor(kit.axes[source])
    assert torch.max(torch.abs(height)) < .04
    torch.testing.assert_close(q[:, 0], torch.as_tensor(np.broadcast_to(yaw, (4,)).copy(), dtype=q.dtype))
    np.testing.assert_array_equal(diag['sampled_yaw_rad'], q[:, 0].numpy())
    assert torch.all(v[:, :3] == 0)


def test_calibration_checks_the_held_arm_for_full_transition_horizon(monkeypatch):
    from types import SimpleNamespace
    import torch
    from amortized_opt.constants import ready_joint_pos
    from amortized_opt.ik import arm_joint_indices
    from amortized_opt.rhythm import DrumKit
    from isaac_rl.transition_calibration import evaluate_trials
    monkeypatch.setattr('isaac_rl.transition_calibration.exact_contact_region',
                        lambda cost, traj, ctx, **kwargs: traj.new_zeros(len(traj)))
    kit = DrumKit.from_layout()
    q = torch.as_tensor(ready_joint_pos()[None], dtype=torch.float32)
    captures = []
    class Cost:
        clearance = .01
        def forward(self, Q, traj, V, ctx, *, lengths):
            captures.append((traj.clone(), lengths.clone()))
            zero = Q.new_zeros(len(Q))
            diag = {k: zero for k in ('terminal_region_dist',
                'joint_box_excess_max', 'torque_excess_max', 'velocity_excess_max')}
            diag['contact_speed_err'] = zero+2.
            diag.update(contact_dir_cos=zero+1, stick_elev_deg=zero+20,
                        collision_min_sdf=zero+self.clearance, collision_stick_min_sdf=zero+.01)
            return SimpleNamespace(total=zero, diagnostics=diag)
    class Generator:
        cost = Cost()
        def predict(self, Q, V, *args, **kwargs):
            assert kwargs['return_plan']
            return dict(traj=Q.new_zeros(len(Q), 10, 17),
                        lengths=Q.new_tensor([[10, 0, 10]]).expand(len(Q), -1).clone())
    generator = Generator(); generator.kit = kit
    result = evaluate_trials(generator, q, torch.zeros_like(q), 0, [0], [1], [0.],
                             np.array([[.2]]))
    traj, lengths = captures[0]
    idle = arm_joint_indices('right')
    torch.testing.assert_close(traj[:, :, idle], q[:, None, idle].expand(-1, 10, -1))
    assert lengths.tolist() == [[10, 10, 10]]
    assert not result['success'].any()
    relaxed = evaluate_trials(generator, q, torch.zeros_like(q), 0, [0], [1], [0.],
                              np.array([[.2]]), require_hit_speed=False)
    assert relaxed['success'].all()
    np.testing.assert_array_equal(relaxed['contact_speed_err'], result['contact_speed_err'])
    generator.cost.clearance = -.01
    colliding = evaluate_trials(generator, q, torch.zeros_like(q), 0, [0], [1], [0.],
                                np.array([[.2]]), require_hit_speed=False)
    assert not colliding['success'].any()


def test_exact_shaft_contact_is_not_rejected_by_training_softmin_average():
    from types import SimpleNamespace
    import torch
    from isaac_rl.transition_calibration import exact_contact_region
    from amortized_opt.sdf import segment_hit_region
    class FK:
        base_z = -.1
        def frame(self, q, name):
            return torch.eye(3)[None].expand(len(q), -1, -1), q.new_tensor([0., 0., self.base_z])[None].expand(len(q), -1)
    cost = SimpleNamespace(fk=FK(), cfg=SimpleNamespace(contact_mode='softmin',
        terminal_axial_lo=0., terminal_axial_hi=.01),
        c_mount_p={'left': torch.zeros(3), 'right': torch.zeros(3)},
        c_mount_axis={'left': torch.tensor([0., 0., 1.]), 'right': torch.tensor([0., 0., 1.])})
    z = torch.zeros(1, 2, 3)
    ctx = SimpleNamespace(active=torch.tensor([[1., 0.]]), hit_index=torch.ones(1, 2, dtype=torch.long),
        stick_axis_local=None, stick_tip_sigma=None, contact_sigma=torch.full((1, 2), .2),
        shaft_allowed=torch.tensor([[1., 0.]]), p_star=z, n_out=torch.tensor([[[0., 0., 1.]]]).expand(1, 2, 3),
        hit_radius=torch.full((1, 2), .1), disk_offset=z, half_plane_dir=z,
        has_half_plane=torch.zeros(1, 2))
    traj = torch.zeros(1, 1, 17)
    assert exact_contact_region(cost, traj, ctx).item() < 1e-6
    result = segment_hit_region(torch.tensor([[0., 0., -.05]]), torch.tensor([[0., 0., .1]]),
        z[:, 0], ctx.n_out[:, 0], ctx.hit_radius[:, 0], z[:, 0], z[:, 0],
        ctx.has_half_plane[:, 0], 0., .01, return_candidates=True)
    d, valid = result[4], result[5]
    weights = torch.softmax(-(d+1000*(1-valid))/.01, -1)
    assert (weights*d).sum().item() > 1e-6
    ctx.shaft_allowed[:] = 0.
    assert exact_contact_region(cost, traj, ctx).item() > .08
    # The aim is 10 mm inward from the physical head: the 70 mm band
    # extends to aim-relative -60 mm, preserving the +10 mm outer face.
    cost.fk.base_z = -.24  # tip at aim-relative -40 mm
    assert exact_contact_region(cost, traj, ctx).item() > .039
    assert exact_contact_region(cost, traj, ctx, depth_band_m=.07).item() < 1e-6
    cost.fk.base_z = -.261  # tip at -61 mm, outside the widened band
    assert exact_contact_region(cost, traj, ctx, depth_band_m=.07).item() > .0009
    cost.fk.base_z = -.189  # tip at +11 mm, outside the unchanged outer face
    assert exact_contact_region(cost, traj, ctx, depth_band_m=.07).item() > .0009
    assert (cost.cfg.terminal_axial_lo, cost.cfg.terminal_axial_hi) == (0., .01)
    with pytest.raises(ValueError, match='finite and positive'):
        exact_contact_region(cost, traj, ctx, depth_band_m=-.07)
