"""Boundary conditions and acceleration demand of the analytic strike seed."""

import numpy as np
import pytest
import torch

from amortized_opt.strike_seed import (
    ACCEL_RAMP_FRACTION,
    accelerating_path_numpy,
    accelerating_path_torch,
    accelerating_apex_path_numpy,
    accelerating_terminal_velocity_numpy,
    quintic_seed,
    strike_seed,
)


def test_two_segment_apex_path_hits_both_knots_and_matches_apex_velocity():
    q0 = np.array([0.2, -0.1])
    v0 = np.array([0.3, -0.4])
    apex = np.array([0.5, 0.2])
    terminal = np.array([0.8, -0.3])
    ka, kh, H, dt = 20, 60, 65, 0.02
    q = accelerating_apex_path_numpy(q0, v0, apex, terminal, ka, kh, H, dt)
    np.testing.assert_allclose(q[0], q0)
    np.testing.assert_allclose(q[ka], apex, atol=1e-13)
    np.testing.assert_allclose(q[kh], terminal, atol=1e-13)
    va = accelerating_terminal_velocity_numpy(q0, v0, apex, ka, dt)
    # Exact one-sided stencils for the quadratic plateau on the left and the
    # cubic zero-acceleration ramp on the right.
    left = (3*q[ka] - 4*q[ka - 1] + q[ka - 2]) / (2*dt)
    right = (-11*q[ka] + 18*q[ka + 1] - 9*q[ka + 2] + 2*q[ka + 3]) / (6*dt)
    np.testing.assert_allclose(left, va, atol=1e-11)
    np.testing.assert_allclose(right, va, atol=1e-11)


@pytest.mark.parametrize("v0", [[0., 0., 0.], [0.4, -0.8, 1.5], [-0.3, 2., -1.]])
@pytest.mark.parametrize("dt", [0.001, 0.002])
def test_actual_start_velocity_zero_start_acceleration_and_exact_hit(v0, dt):
    q0 = np.array([0.2, -0.1, 0.5])
    terminal = np.array([0.7, -0.4, 0.5])  # includes zero displacement with motion
    k = 1000
    q = accelerating_path_numpy(q0, v0, terminal, k, 1200, dt)
    np.testing.assert_allclose(q[0], q0)
    np.testing.assert_allclose(q[k], terminal, atol=1e-14)
    # Exact one-sided derivative stencils for the cubic acceleration ramp.
    velocity = (-11*q[0] + 18*q[1] - 9*q[2] + 2*q[3]) / (6*dt)
    acceleration = (2*q[0] - 5*q[1] + 4*q[2] - q[3]) / dt**2
    np.testing.assert_allclose(velocity, v0, atol=1e-10)
    np.testing.assert_allclose(acceleration, 0, atol=1e-8)
    r = ACCEL_RAMP_FRACTION
    T = k * dt
    plateau = (terminal - q0 - np.asarray(v0)*T) / (T*T*(0.5 - r/2 + r*r/6))
    acc = np.diff(q, n=2, axis=0) / dt**2
    np.testing.assert_allclose(acc[150:990], np.broadcast_to(plateau, (840, 3)), atol=1e-8)
    # Post-hit continuation tapers acceleration, then coasts.
    np.testing.assert_allclose(acc[1101:], 0, atol=1e-8)
    assert np.max(np.abs(q[k+1] - q[k])) > 1e-5
    # No acceleration jump at either end of the plateau (finite grid error).
    assert np.max(np.abs(np.diff(acc, axis=0))) < 0.011 * np.max(np.abs(plateau))


def test_rest_start_preserves_line_and_reduces_peak_acceleration():
    dt, k = 0.001, 1000
    q = accelerating_path_numpy([0., 0.], [0., 0.], [1., -2.], k, k, dt)
    np.testing.assert_allclose(q[:, 1], -2*q[:, 0])
    assert (np.diff(q[:, 0]) >= 0).all()
    peak = np.abs(np.diff(q[:, 0], n=2) / dt**2).max()
    assert peak < 0.39 * (10 / np.sqrt(3))  # >61% below the quintic peak
    assert (q[-1, 0] - q[-2, 0]) / dt > 2.0  # arrive moving, not at rest


def test_coasting_velocity_needs_no_acceleration():
    q0, v0 = np.array([0.2, -0.5]), np.array([0.7, -0.3])
    dt, k, H = 0.02, 25, 40
    q = accelerating_path_numpy(q0, v0, q0 + v0*k*dt, k, H, dt)
    np.testing.assert_allclose(q, q0 + np.arange(H + 1)[:, None]*dt*v0, atol=1e-14)


def test_numpy_torch_agree_for_unequal_windows_and_nonzero_velocities():
    rng = np.random.default_rng(5)
    q0, v0, terminal = (rng.normal(size=(3, 7)) for _ in range(3))
    lengths, H, dt = np.array([1, 10, 40]), 50, 0.02
    expected = np.stack([
        accelerating_path_numpy(q0[b], v0[b], terminal[b], int(lengths[b]), H, dt)[1:]
        for b in range(3)
    ])
    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (q0, v0, terminal)]
    actual = accelerating_path_torch(*tensors, torch.tensor(lengths), H, dt)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    actual.square().mean().backward()
    for t in tensors:
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_mixed_seed_changes_only_active_arms_and_respects_each_hit():
    names = ("waist_yaw_joint", "left_elbow_joint", "right_elbow_joint")
    q0 = torch.zeros(2, 3, dtype=torch.float64)
    v0 = torch.tensor([[0.1, -0.2, 0.4], [-0.2, 0.3, -0.1]], dtype=q0.dtype)
    term = torch.ones_like(q0) * 0.2
    for t in (q0, v0, term):
        t.requires_grad_()
    lengths = torch.tensor([[20, 10, 20], [15, 15, 8]])
    active = torch.tensor([[True, False], [True, True]])
    out = strike_seed(q0, v0, term, lengths, 25, active=active, joint_names=names)
    legacy = quintic_seed(q0, v0, term, lengths, 25)
    torch.testing.assert_close(out[:, :, 0], legacy[:, :, 0])
    torch.testing.assert_close(out[0, :, 2], legacy[0, :, 2])
    torch.testing.assert_close(out.gather(1, (lengths - 1)[:, None, :]).squeeze(1), term)
    assert not torch.allclose(out[:, :, 1], legacy[:, :, 1])
    out.square().sum().backward()
    for t in (q0, v0, term):
        assert torch.isfinite(t.grad).all()


def test_warmstart_and_gpu_arm_seed_agree_with_measured_velocity():
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
    from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, warmstart_trajectory

    q0 = ready_joint_pos()
    v0 = np.linspace(-0.4, 0.5, len(G1_UPPER_BODY_JOINTS))
    ws = warmstart_trajectory(
        q0, v0, horizon=40, goals={"left": ("snare", 0.4), "right": None},
        library=GoalPoseLibrary(), clamp=False,
    )
    cols, k = arm_slice("left"), ws.hit_index["left"]
    actual = accelerating_path_torch(
        torch.tensor(q0[cols])[None], torch.tensor(v0[cols])[None],
        torch.tensor(ws.q[k, cols])[None], torch.tensor([k]), 40,
    )
    np.testing.assert_allclose(actual[0].numpy(), ws.q[1:, cols], atol=1e-12)
