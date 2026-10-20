import numpy as np
import torch

from amortized_opt.constants import ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.rhythm import DrumKit
from isaac_rl.rnea_cache import build_or_load
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import QPConfig, captured_constraints, cost_for_horizon, solve_step


def test_transition_evaluation_keeps_both_physical_arms_live_through_deadline(monkeypatch):
    from types import SimpleNamespace
    import isaac_rl.transition_calibration as calibration
    from amortized_opt.ik import arm_joint_indices
    kit = DrumKit.from_layout()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float32).expand(2, -1).clone()
    v = torch.zeros_like(q)
    for hand in (0, 1):
        recorded = []
        def forward(Q, trajectory, V, ctx, *, lengths):
            recorded.append((trajectory.clone(), lengths.clone()))
            z = Q.new_zeros(len(Q))
            diag = {name: z for name in ('collision_min_sdf', 'collision_stick_min_sdf',
                'joint_box_excess_max', 'torque_excess_max', 'velocity_excess_max',
                'contact_speed_err', 'terminal_region_dist')}
            diag.update(contact_dir_cos=z+1, stick_elev_deg=z+15)
            return SimpleNamespace(total=z, diagnostics=diag)
        def predict(Q, V, drums, times, speeds, valid, group, **kw):
            N = (times[:, 0]/.02).round()
            lengths = torch.stack((N, N if hand == 0 else N*0-1,
                                   N if hand == 1 else N*0-1), -1)
            return dict(traj=Q.new_zeros(len(Q), 100, 17), lengths=lengths)
        generator = SimpleNamespace(kit=kit, cost=SimpleNamespace(forward=forward), predict=predict)
        monkeypatch.setattr(calibration, 'exact_contact_region',
                            lambda cost, traj, ctx, **kw: traj.new_zeros(len(traj)))
        calibration.evaluate_trials(generator, q, v, hand, [0, 0], [1, 1], [0., 0.],
                                    np.array([[.02], [1.]]), chunk=2)
        traj, lengths = recorded[0]
        torch.testing.assert_close(lengths, torch.tensor([[1., 1., 1.], [50., 50., 50.]]))
        idle = arm_joint_indices(('left', 'right')[1-hand])
        torch.testing.assert_close(traj[:, :, idle], q[:, None, idle].expand(-1, 100, -1))


def test_elastic_qp_repairs_linear_bound_and_handles_short_horizons():
    for horizon in (1, 2, 5):
        step, info = solve_step(np.ones((1, horizon)), np.array([.05]),
            horizon, 1, QPConfig(), region_rows=np.array([False]))
        assert info['usable']
        assert np.max(np.abs(step)) <= .12001
        assert .05+step.sum() <= 0
    # The trust region prevents a single large correction; the nonnegative
    # elastic slack reports the uncorrectable remainder in this local QP.
    step, info = solve_step(np.ones((1, 1)), np.array([2.]), 1, 1, QPConfig())
    assert info['usable'] and info['max_slack'] > 1.8
    assert abs(step[0, 0]) <= .12001


def test_independent_stalled_plan_does_not_block_repair_or_repeat_identical_qps(monkeypatch):
    import isaac_rl.trajectory_qp as qp
    # One constraint is uncorrectable; another needs several trust-region
    # steps. Their independent batch must behave like separate optimizations.
    def bounds(q, path):
        value = path[:, :, 0].mean(-1)-.25
        return torch.where(q[:, 0] < .5, value*0+1., value)[:, None]
    def captured(cost, q, v, path, ctx, lengths, depth, **kw):
        return bounds(q, path), None, ['bound_0']
    def measured(cost, q, v, path, ctx, lengths, depth, **kw):
        g = bounds(q, path).detach().numpy()
        return dict(success=(g <= 1e-6).all(-1), merit=(np.maximum(g, 0)**2).sum(-1))
    calls = []
    original = qp.solve_step
    def solved(jac, g, *args, **kw):
        calls.append(float(g[0]))
        return original(jac, g, *args, **kw)
    monkeypatch.setattr(qp, 'captured_constraints', captured)
    monkeypatch.setattr(qp, 'metrics', measured)
    monkeypatch.setattr(qp, 'solve_step', solved)
    q = torch.zeros(2, 17); q[1, 0] = 1.
    path = torch.zeros(2, 1, 17); path[:, :, 0] = .6
    repaired, _, after, _ = qp.refine(None, q, q*0, path, None,
        torch.ones(2, 3, dtype=torch.long), 0, cfg=QPConfig(iterations=8))
    assert not after['success'][0] and after['success'][1]
    torch.testing.assert_close(repaired[0], path[0])
    assert sum(value == 1. for value in calls) == 1
    assert len(calls) == 1+after['accepted_steps'][1]
    assert after['accepted_steps'][1] > 1
    alone, _, single_after, _ = qp.refine(None, q[1:], q[1:]*0, path[1:], None,
        torch.ones(1, 3, dtype=torch.long), 0, cfg=QPConfig(iterations=8))
    assert single_after['success'][0]
    torch.testing.assert_close(repaired[1:], alone)


def test_captured_constraints_match_real_gates_and_keep_spline_boundaries():
    torch.set_num_threads(1)
    cost = FusedCost(CostConfig(w_velocity=1., w_joint_box=1.), max_horizon=8,
        device='cpu', dtype=torch.float64, rnea=build_or_load(device='cpu', dtype=torch.float64))
    kit = DrumKit.from_layout()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float64)
    v = torch.zeros_like(q); v[:, 3] = .2
    traj = q[:, None].expand(-1, 8, -1).clone()
    traj[:, :, 3] += torch.linspace(.005, .02, 8)
    lengths = torch.full((1, 3), 5, dtype=torch.long)
    ctx = strike_context(kit, np.array([[0, -1]]), np.array([[0, -1]]), [5], 'cpu').to('cpu', torch.float64)
    ctx.target_top_entry_depth_m = .07
    capture = {}
    original = cost(q, traj, v, ctx, lengths=lengths)
    captured = cost(q, traj, v, ctx, lengths=lengths, constraint_capture=capture)
    torch.testing.assert_close(original.total, captured.total, rtol=0, atol=0)
    d = original.diagnostics
    torch.testing.assert_close(capture['torque'].clamp_min(0).amax(-1), d['torque_excess_max'])
    torch.testing.assert_close(capture['velocity'].clamp_min(0).amax(-1), d['velocity_excess_max'])
    box = torch.cat([capture['joint_lo'], capture['joint_hi']], -1).clamp_min(0).amax(-1)
    torch.testing.assert_close(box, d['joint_box_excess_max'])
    collision = torch.cat([val for name, val in capture.items() if name.startswith('collision')], -1).amax(-1)
    torch.testing.assert_close(collision, -torch.minimum(d['collision_min_sdf'], d['collision_stick_min_sdf']))
    short = cost_for_horizon(cost, 5)
    trimmed = traj[:, :5].clone().requires_grad_(True)
    g, terms, _ = captured_constraints(short, q, v, trimmed, ctx, lengths, .07)
    for name in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                 'contact_dir_cos', 'stick_elev_deg', 'collision_min_sdf'):
        torch.testing.assert_close(terms.diagnostics[name], d[name], atol=1e-8, rtol=1e-8)
    gradient = torch.autograd.grad(g.sum(), trimmed)[0]
    assert torch.isfinite(gradient).all()
    binned, binned_terms, names = captured_constraints(short, q, v,
        trimmed, ctx, lengths, .07, torque_time_bins=4)
    torque = binned[:, [i for i, name in enumerate(names) if name.startswith('torque_')]]
    assert torque.shape[1] == 4*q.shape[1]
    torch.testing.assert_close(torque.clamp_min(0).amax(-1)*10,
                               d['torque_excess_max'], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(binned_terms.total, terms.total, rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(binned.sum(), trimmed)[0]).all()
    sp = short.spline.build(q, trimmed, v, lengths)
    torch.testing.assert_close(sp.x[:, 0], q)
    torch.testing.assert_close(sp.m[:, 0], v)


def test_calibration_scope_ignores_idle_arm_constraints_but_preserves_active_dynamics(monkeypatch):
    from amortized_opt.ik import arm_joint_indices
    from isaac_rl.transition_calibration import static_clearance
    torch.set_num_threads(1)
    cost = FusedCost(CostConfig(w_velocity=1., w_joint_box=1., physics_mode='knot_fd'),
        max_horizon=4, device='cpu', dtype=torch.float64,
        rnea=build_or_load(device='cpu', dtype=torch.float64))
    kit = DrumKit.from_layout()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float64); q[:, 0] = .3
    v = torch.zeros_like(q)
    ctx = strike_context(kit, [[-1, 2]], [[-1, 2]], [4], 'cpu').to('cpu', torch.float64)
    traj = q[:, None].expand(-1, 4, -1).clone()
    lengths = torch.full((1, 3), 4, dtype=torch.long)
    base_capture = {}; base = cost(q, traj, v, ctx, lengths=lengths, constraint_capture=base_capture)
    assert base.diagnostics['collision_min_sdf_left'].item() < -.004
    cost.calibration_active_hand_only = True
    scoped_capture = {}; scoped = cost(q, traj, v, ctx, lengths=lengths, constraint_capture=scoped_capture)
    assert scoped.diagnostics['collision_min_sdf_left'].item() > 0
    idle = arm_joint_indices('left'); active = [0, 1, 2]+list(arm_joint_indices('right'))
    for name in ('torque', 'velocity', 'joint_lo', 'joint_hi'):
        assert (scoped_capture[name][:, idle] < 0).all()
        torch.testing.assert_close(scoped_capture[name][:, active], base_capture[name][:, active])
    # Full physical dynamics and waist constraints remain unchanged; only
    # which actuator violations participate in acceptance is different.
    np.testing.assert_allclose(
        static_clearance(cost, kit, q, np.array([[-1, 2]])).cpu(), [-.00498754], atol=1e-6)
    assert static_clearance(cost, kit, q, np.array([[-1, 2]]), ignore_idle_arm=True).item() > -.004
    monkeypatch.setattr('amortized_opt.cost.cross_hand_sdf',
        lambda a, *args, **kwargs: a.new_full(a.shape[:-1], -.5))
    scoped = cost(q, traj, v, ctx, lengths=lengths)
    assert scoped.diagnostics['collision_stick_min_sdf'].item() > 0
    cost.calibration_active_hand_only = False
    both = cost(q, traj, v, ctx, lengths=lengths)
    assert both.diagnostics['collision_stick_min_sdf'].item() == -.5
