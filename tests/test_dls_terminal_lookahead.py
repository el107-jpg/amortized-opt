"""Training-time terminal DLS lookahead for the phase-2 actor."""

from __future__ import annotations

from dataclasses import replace

import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.train_cost import (
    Phase2Config,
    _dls_contact_local,
    _dls_contact_position,
    _spline_smoothness_hessians,
    curvature_direct_vjp_correction,
    curvature_precondition_vjps,
    curvature_terminal_teacher,
    dls_terminal_lookahead_loss,
    dls_terminal_teacher,
    group_lengths,
)


H = 12


def _fixture():
    cost = FusedCost(
        CostConfig(n_sub=1), max_horizon=H, device="cpu", dtype=torch.float32,
    )
    B = 2
    q = torch.as_tensor(ready_joint_pos(), dtype=torch.float32)
    traj = q.view(1, 1, -1).expand(B, H, -1).contiguous()
    ones = torch.ones(B, 2)
    zeros = torch.zeros(B, 2)
    strike = StrikeContext(
        hit_index=torch.tensor([[H, H], [H - 2, H]], dtype=torch.long),
        active=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.zeros(B, 2, 3),
        v_star=torch.zeros(B, 2, 3),
        n_out=torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).expand(B, 2, 3),
        hit_radius=0.08 * ones,
        disk_offset=torch.zeros(B, 2, 3),
        half_plane_dir=torch.zeros(B, 2, 3),
        has_half_plane=zeros.clone(),
        contact_sigma=0.20 * ones,
        shaft_allowed=zeros.clone(),
        start_shaft_allowed=zeros.clone(),
        elev_required=zeros.clone(),
    )
    local = _dls_contact_local(cost, strike, 0)
    current = _dls_contact_position(cost.fk, traj[:, -1], "left", local)
    p_star = strike.p_star.clone()
    p_star[:, 0] = current + torch.tensor([0.04, -0.02, 0.03])
    strike = replace(strike, p_star=p_star)
    return cost, traj, strike


def test_dls_teacher_reduces_contact_error_and_only_moves_approach_arm():
    cost, traj, strike = _fixture()
    teacher, mask, stats = dls_terminal_teacher(
        traj, strike, cost, steps=2, window=5, damping=0.05,
        max_step_rad=0.35,
    )
    assert float(stats["dls_terminal_err_after_m"]) < float(
        stats["dls_terminal_err_before_m"]
    )
    left = [
        i for i, name in enumerate(G1_UPPER_BODY_JOINTS)
        if name.startswith("left_")
    ]
    untouched = [i for i in range(N_DOF) if i not in left]
    assert torch.equal(teacher[:, :, untouched], traj[:, :, untouched])
    assert torch.equal(mask[:, :, untouched], torch.zeros_like(mask[:, :, untouched]))
    # Row 0 hits at H and has a five-knot approach: earlier knots stay exact.
    assert torch.equal(teacher[0, : H - 5], traj[0, : H - 5])
    assert not torch.equal(teacher[0, -1, left], traj[0, -1, left])


def test_dls_lookahead_has_a_finite_straight_through_actor_gradient():
    cost, traj0, strike = _fixture()
    traj = traj0.clone().requires_grad_(True)
    q0 = traj0[:, 0].clone()
    v0 = torch.zeros_like(q0)
    lengths = group_lengths(strike.hit_index)
    cfg = Phase2Config(
        compile_model=False, compile_cost=False,
        dls_terminal_frac=1.0, w_dls_terminal_cost=0.25,
        w_dls_terminal_tie=10.0, dls_terminal_steps=1,
        dls_terminal_window=5,
    )
    look_cost, tie, torque, _ = dls_terminal_lookahead_loss(
        traj, q0, v0, strike, lengths, {}, cost, cfg,
    )
    total = cfg.w_dls_terminal_cost * look_cost + cfg.w_dls_terminal_tie * tie
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(torque)
    assert traj.grad is not None and torch.isfinite(traj.grad).all()
    assert float(traj.grad.abs().sum()) > 0.0


def test_analytic_spline_smoothness_hessian_matches_autograd():
    cost, traj, strike = _fixture()
    q0 = traj[:, 0]
    v0 = torch.zeros_like(q0)
    lengths = group_lengths(strike.hit_index)
    x = (traj + 0.01 * torch.randn_like(traj)).requires_grad_(True)
    direction = torch.zeros_like(x)
    direction[:, :, 3] = torch.linspace(-0.01, 0.02, H)

    def gradient(waypoints):
        sp = cost.spline.build(q0, waypoints, v0, lengths)
        ints = sp.segment_integrals()
        n = sp.seg_mask.sum(dim=(1, 2)).clamp_min(1.0)
        val = (
            cost.cfg.w_smooth_acc * ints["acc2"].sum(dim=(1, 2))
            + cost.cfg.w_smooth_jerk * ints["jerk2"].sum(dim=(1, 2))
        ).div(n).sum()
        return torch.autograd.grad(val, waypoints)[0]

    g0 = gradient(x)
    g1 = gradient((x.detach() + direction).requires_grad_(True))
    h = _spline_smoothness_hessians(cost)
    counts = cost.spline.build(q0, x.detach(), v0, lengths).seg_mask.sum(dim=(1, 2))
    # Joint 3 is on the left arm, including a shorter second-row horizon.
    for row in range(traj.shape[0]):
        horizon = int(lengths[row, 1])
        expected = h[horizon] @ direction[row, :, 3] / counts[row]
        torch.testing.assert_close(g1[row, :, 3] - g0[row, :, 3], expected,
                                   rtol=3e-3, atol=2e-3)


def test_curvature_teacher_never_admits_a_higher_full_cost_and_backprops():
    cost, traj0, strike = _fixture()
    traj = traj0.clone().requires_grad_(True)
    q0 = traj0[:, 0].clone()
    v0 = torch.zeros_like(q0)
    lengths = group_lengths(strike.hit_index)
    cfg = Phase2Config(
        compile_model=False, compile_cost=False,
        dls_terminal_frac=1.0, w_dls_terminal_cost=0.25,
        w_dls_terminal_tie=10.0, terminal_lookahead_mode="curvature",
    )
    teacher, mask, stats = curvature_terminal_teacher(
        traj, q0, v0, strike, lengths, {}, cost, cfg,
    )
    before = cost.forward(q0, traj.detach(), v0, strike, lengths=lengths).total
    after = cost.forward(q0, teacher, v0, strike, lengths=lengths).total
    assert torch.isfinite(teacher).all()
    assert torch.isfinite(mask).all()
    assert torch.all(after <= before + 1e-4)
    assert 0.0 <= float(stats["curvature_teacher_accept_frac"]) <= 1.0

    look_cost, tie, torque, _ = dls_terminal_lookahead_loss(
        traj, q0, v0, strike, lengths, {}, cost, cfg,
    )
    (cfg.w_dls_terminal_cost * look_cost + cfg.w_dls_terminal_tie * tie).backward()
    assert torch.isfinite(torque)
    assert traj.grad is not None and torch.isfinite(traj.grad).all()


def test_direct_curvature_replaces_trajectory_vjp_without_changing_forward_loss():
    cost, traj0, strike = _fixture()
    traj = (traj0 + 0.01).clone().requires_grad_(True)
    q0 = traj0[:, 0].clone()
    lengths = group_lengths(strike.hit_index)
    cfg = Phase2Config(
        compile_model=False, compile_cost=False,
        dls_terminal_frac=0.0, curvature_direct_frac=1.0,
    )
    total = traj.square().sum()
    torque = traj[..., 3].square().mean()
    raw = torch.autograd.grad(total, traj, retain_graph=True)[0]
    raw_torque = torch.autograd.grad(torque, traj, retain_graph=True)[0]
    expected, stats = curvature_precondition_vjps(
        torch.stack([raw.detach(), raw_torque.detach()], dim=-1),
        traj.detach(), q0, strike,
        lengths, {}, cost, cfg,
    )
    delta, torque_proxy, correction_stats = curvature_direct_vjp_correction(
        total, traj, q0, strike, lengths, {}, cost, cfg,
        torque_term_mean=torque,
    )
    torch.testing.assert_close(torque_proxy.detach(), torque.detach())
    total.backward(retain_graph=True)
    torch.autograd.backward(traj, grad_tensors=delta, retain_graph=True)
    torch.testing.assert_close(traj.grad, expected[..., 0], rtol=1e-4, atol=1e-4)
    assert torch.isfinite(traj.grad).all()
    assert not torch.allclose(traj.grad, raw)
    assert float(stats["curvature_direct_cosine"]) > 0.0
    assert float(correction_stats["curvature_direct_fallback_frac"]) == 0.0
    torque_proxy_grad = torch.autograd.grad(torque_proxy, traj)[0]
    torch.testing.assert_close(
        torque_proxy_grad, expected[..., 1], rtol=1e-4, atol=1e-4,
    )
