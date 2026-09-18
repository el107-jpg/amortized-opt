"""Numerical derivative tests for RNEA and its torque-limit reductions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from amortized_opt.constants import CONTROL_DT, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext, smooth_max
from amortized_opt.limits import available_torque_torch, isaac_armature
from amortized_opt.rnea import BatchedRNEA
from amortized_opt.tspline import TrajectorySplineBuilder


def _directional_table(fn, inputs, directions, steps):
    values = [x.detach().clone().requires_grad_(True) for x in inputs]
    f0 = fn(*values)
    grads = torch.autograd.grad(f0, values)
    autograd = sum((g * d).sum() for g, d in zip(grads, directions)).detach()
    rows = []
    with torch.no_grad():
        for h in steps:
            plus = fn(*(x + h * d for x, d in zip(values, directions)))
            minus = fn(*(x - h * d for x, d in zip(values, directions)))
            central = (plus - minus) / (2.0 * h)
            rows.append((h, central, (central - autograd).abs() / autograd.abs()))
    return autograd, rows


def test_rnea_directional_derivative_matches_central_difference():
    torch.manual_seed(20)
    dtype = torch.float64
    rnea = BatchedRNEA.from_urdf(
        with_armature=True, dtype=dtype, device="cpu"
    )
    q = torch.randn(5, N_DOF, dtype=dtype) * 0.25
    qd = torch.randn(5, N_DOF, dtype=dtype) * 2.0
    qdd = torch.randn(5, N_DOF, dtype=dtype) * 100.0
    directions = tuple(
        torch.randn_like(x) / torch.randn_like(x).numel() ** 0.5
        for x in (q, qd, qdd)
    )
    probe = torch.randn_like(q)

    def fn(a, b, c):
        return (rnea.rnea(a, b, c) * probe).mean()

    autograd, rows = _directional_table(
        fn, (q, qd, qdd), directions, (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
    )
    best = min(float(error) for _h, _fd, error in rows)
    assert torch.isfinite(autograd)
    assert best < 2e-7


def test_rnea_float32_tracks_float64_at_strike_scale():
    torch.manual_seed(21)
    r64 = BatchedRNEA.from_urdf(
        with_armature=True, dtype=torch.float64, device="cpu"
    )
    r32 = r64.to("cpu", torch.float32)
    q64 = torch.randn(64, N_DOF, dtype=torch.float64) * 0.35
    qd64 = torch.randn(64, N_DOF, dtype=torch.float64) * 8.0
    qdd64 = torch.randn(64, N_DOF, dtype=torch.float64) * 300.0
    ref = r64.rnea(q64, qd64, qdd64)
    got = r32.rnea(q64.float(), qd64.float(), qdd64.float()).double()
    relative_rms = (got - ref).square().mean().sqrt() / ref.square().mean().sqrt()
    assert torch.isfinite(got).all()
    assert float(relative_rms) < 2e-5


def test_spline_torque_warning_and_pnorm_derivative_is_correct_away_from_kinks():
    torch.manual_seed(22)
    dtype = torch.float64
    B, H = 2, 8
    builder = TrajectorySplineBuilder(
        max_segments=H, dt=CONTROL_DT, dtype=dtype
    )
    rnea = BatchedRNEA.from_urdf(
        with_armature=True, dtype=dtype, device="cpu"
    )
    q0 = torch.randn(B, N_DOF, dtype=dtype) * 0.1
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=dtype) * 0.015, dim=1
    )
    v0 = torch.randn(B, N_DOF, dtype=dtype) * 0.2
    lengths = torch.tensor([[H, H - 1, H], [H - 2, H, H - 1]])
    # A deliberately low wall keeps every material element away from the ReLU
    # threshold, so this checks derivatives rather than active-set convention.
    limit = torch.full((1, 1, N_DOF), 0.1, dtype=dtype)

    def fn(waypoints):
        spline = builder.build(q0, waypoints, v0, lengths)
        q, qd, qdd = spline.knot_finite_differences(v0)
        tau = rnea.rnea(
            q.reshape(-1, N_DOF),
            qd.reshape(-1, N_DOF),
            qdd.reshape(-1, N_DOF),
        ).reshape_as(q)
        index = torch.arange(H + 1).view(1, -1, 1)
        mask = (index <= spline.lengths.unsqueeze(1)).to(dtype)
        warning = (
            F.softplus(tau.abs() - limit) * mask
        ).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
        over = (tau.abs() - limit).clamp_min(0.0) * mask
        peak = smooth_max(over, mask, "pnorm", 12.0, 0.05)
        return (warning + 2.0 * peak).mean()

    direction = torch.randn_like(wp)
    direction /= direction.abs().max()
    autograd, rows = _directional_table(
        fn, (wp,), (direction,), (1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
    )
    best = min(float(error) for _h, _fd, error in rows)
    assert torch.isfinite(autograd)
    assert best < 2e-5


def test_hard_max_crossing_has_valid_nonsmooth_not_incorrect_derivative():
    # At a tie, amax's autograd is the midpoint subgradient.  Central FD agrees,
    # while the two one-sided directional derivatives differ.  A derivative
    # audit must classify this as a max crossing, not an autograd defect.
    x = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([1.0, -1.0], dtype=torch.float64)
    y = x.amax()
    (grad,) = torch.autograd.grad(y, x)
    h = 1e-5
    plus = (x.detach() + h * direction).amax()
    minus = (x.detach() - h * direction).amax()
    central = (plus - minus) / (2.0 * h)
    forward = (plus - y.detach()) / h
    backward = (y.detach() - minus) / h
    assert float((grad * direction).sum()) == pytest.approx(0.0, abs=1e-15)
    assert float(central) == pytest.approx(0.0, abs=1e-10)
    assert float(forward) == pytest.approx(1.0)
    assert float(backward) == pytest.approx(-1.0)


def test_linear_dynamic_torque_limit_gradcheck_and_zero_speed_kink():
    dtype = torch.float64
    effort = torch.tensor([[[25.0, 5.0]]], dtype=dtype)
    velocity = torch.tensor([[[37.0, 22.0]]], dtype=dtype)
    qd = torch.tensor([[[3.0, -2.0]]], dtype=dtype, requires_grad=True)

    def fn(x):
        return available_torque_torch(
            x,
            effort,
            velocity,
            eta=0.8,
            saturation_scale=1.0,
            derate="linear",
        )

    assert torch.autograd.gradcheck(fn, (qd,), eps=1e-6, atol=1e-8, rtol=1e-5)

    zero = torch.zeros(1, 1, 2, dtype=dtype, requires_grad=True)
    value = fn(zero).sum()
    (grad,) = torch.autograd.grad(value, zero)
    h = 1e-6
    forward = (fn(zero.detach() + h).sum() - value.detach()) / h
    backward = (value.detach() - fn(zero.detach() - h).sum()) / h
    assert torch.equal(grad, torch.zeros_like(grad))
    assert float(forward) < 0.0 < float(backward)


def test_knot_fd_mask_count_and_terminal_acceleration_are_consistent():
    dtype = torch.float64
    B, H = 1, 10
    builder = TrajectorySplineBuilder(max_segments=H, dtype=dtype)
    q0 = torch.zeros(B, N_DOF, dtype=dtype)
    wp = torch.randn(B, H, N_DOF, dtype=dtype) * 0.02
    v0 = torch.zeros_like(q0)
    lengths = torch.tensor([[4, 7, 9]])
    spline = builder.build(q0, wp, v0, lengths)
    q, qd, qdd = spline.knot_finite_differences(v0)
    index = torch.arange(H + 1).view(1, -1, 1)
    mask = (index <= spline.lengths.unsqueeze(1)).to(dtype)
    expected = float((spline.lengths + 1).sum())
    assert float(mask.sum()) == expected
    for joint, terminal in enumerate(spline.lengths[0].tolist()):
        if terminal > 0:
            want = (qd[0, terminal, joint] - qd[0, terminal - 1, joint]) / CONTROL_DT
            assert qdd[0, terminal, joint] == pytest.approx(float(want))
        assert float(qd[0, terminal + 1 :, joint].abs().max()) == 0.0
        assert float(qdd[0, terminal + 1 :, joint].abs().max()) == 0.0


def test_armature_gradient_is_exact_diagonal():
    dtype = torch.float64
    rnea = BatchedRNEA.from_urdf(
        with_armature=True, dtype=dtype, device="cpu"
    )
    no_armature = rnea.without_armature()
    q = torch.randn(3, N_DOF, dtype=dtype)
    qd = torch.randn_like(q)
    qdd = torch.randn_like(q, requires_grad=True)
    extra = rnea.rnea(q, qd, qdd) - no_armature.rnea(q, qd, qdd)
    probe = torch.randn_like(extra)
    (grad,) = torch.autograd.grad((extra * probe).sum(), qdd)
    want = (
        torch.as_tensor(isaac_armature(), dtype=dtype).view(1, -1) * probe
    )
    assert torch.allclose(grad, want, atol=1e-12, rtol=1e-12)


def _minimal_strike_context(batch: int, horizon: int, dtype):
    active = torch.ones(batch, 2, dtype=dtype)
    return StrikeContext(
        hit_index=torch.full((batch, 2), horizon, dtype=torch.long),
        active=active,
        target_drum=torch.zeros(batch, 2, dtype=torch.long),
        start_drum=torch.full((batch, 2), -1, dtype=torch.long),
        p_star=torch.full((batch, 2, 3), 0.3, dtype=dtype),
        v_star=torch.full((batch, 2, 3), -1.0, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
        .view(1, 1, 3)
        .expand(batch, 2, 3),
        hit_radius=torch.full((batch, 2), 0.08, dtype=dtype),
        disk_offset=torch.zeros(batch, 2, 3, dtype=dtype),
        half_plane_dir=torch.zeros(batch, 2, 3, dtype=dtype),
        has_half_plane=torch.zeros(batch, 2, dtype=dtype),
        contact_sigma=torch.full((batch, 2), 0.2, dtype=dtype),
        shaft_allowed=torch.zeros(batch, 2, dtype=dtype),
        start_shaft_allowed=torch.zeros(batch, 2, dtype=dtype),
        elev_required=active,
    )


def test_knot_fd_torque_is_exactly_independent_of_spline_subsampling():
    """``n_sub`` feeds geometry, not torque, under the live ``knot_fd`` mode."""
    torch.manual_seed(23)
    dtype = torch.float64
    B, H = 2, 7
    shared = FusedCost(
        CostConfig(n_sub=1, physics_mode="knot_fd", w_torque_peak=2.0),
        max_horizon=H,
        dtype=dtype,
    )
    dense = FusedCost(
        CostConfig(n_sub=4, physics_mode="knot_fd", w_torque_peak=2.0),
        max_horizon=H,
        dtype=dtype,
        fk=shared.fk,
        rnea=shared.rnea,
        scene=shared.scene,
    )
    q0 = torch.randn(B, N_DOF, dtype=dtype) * 0.1
    waypoints = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=dtype) * 0.02, dim=1
    )
    v0 = torch.randn_like(q0) * 0.2
    lengths = torch.tensor([[H, H - 2, H], [H - 1, H, H - 3]])
    context = _minimal_strike_context(B, H, dtype)
    one = shared.forward(q0, waypoints, v0, context, lengths=lengths)
    four = dense.forward(q0, waypoints, v0, context, lengths=lengths)
    for key in ("torque", "torque_peak"):
        assert torch.equal(one.raw[key], four.raw[key])
    assert torch.equal(one.terms["torque"], four.terms["torque"])
    assert torch.equal(
        one.diagnostics["torque_excess_max"],
        four.diagnostics["torque_excess_max"],
    )
