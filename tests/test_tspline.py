"""Conformance / autograd tests for the differentiable spline layer.

The reference is ``scipy.interpolate.make_interp_spline`` -- the very function
``amortized_opt.spline.joint_path_spline`` calls -- with a matching ``bc_type``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.interpolate import make_interp_spline

from amortized_opt.constants import CONTROL_DT, N_DOF
from amortized_opt.spline import joint_path_spline
from amortized_opt.tspline import TrajectorySplineBuilder, spline_gain_matrices

TOL = 1e-6


def _scipy_ref(Q: np.ndarray, dt: float, bc_start: str, bc_end: str, v0: np.ndarray | None):
    """scipy C2 cubic on the uniform time grid with the requested end rules."""
    n = Q.shape[1]
    t = np.arange(Q.shape[0], dtype=float) * dt
    if bc_start == "not-a-knot" and bc_end == "not-a-knot":
        bc = None
    else:
        lo = (
            None
            if bc_start == "not-a-knot"
            else [(1, np.asarray(v0, float))] if bc_start == "clamped"
            else [(2, np.zeros(n))]
        )
        hi = None if bc_end == "not-a-knot" else [(2, np.zeros(n))]
        if lo is None or hi is None:
            pytest.skip("scipy cannot mix not-a-knot with a derivative bc on one end")
        bc = (lo, hi)
    return make_interp_spline(t, Q, k=3, bc_type=bc)


@pytest.mark.parametrize("bc", [("clamped", "natural"), ("natural", "natural"),
                                ("not-a-knot", "not-a-knot")])
@pytest.mark.parametrize("K", [4, 9, 24, 40])
def test_conformance_to_scipy(bc, K):
    bc_start, bc_end = bc
    rng = np.random.default_rng(7 * K + len(bc_start))
    Q = rng.normal(size=(K + 1, N_DOF)) * 0.4
    v0 = rng.normal(size=N_DOF) * 0.8
    dt = CONTROL_DT

    ref = _scipy_ref(Q, dt, bc_start, bc_end, v0)
    ts = np.linspace(0.0, K * dt, 7 * K + 1)
    q_ref = ref(ts)
    v_ref = ref.derivative(1)(ts)
    a_ref = ref.derivative(2)(ts)

    b = TrajectorySplineBuilder(
        max_segments=K, dt=dt, bc_start=bc_start, bc_end=bc_end, dtype=torch.float64
    )
    q0 = torch.tensor(Q[0])[None]
    wp = torch.tensor(Q[1:])[None]
    vv = torch.tensor(v0)[None]
    lengths = torch.full((1, b.n_groups), K, dtype=torch.long)
    sp = b.build(q0, wp, vv, lengths)

    # evaluate on the same dense grid, segment by segment
    seg = np.clip((ts / dt).astype(int), 0, K - 1)
    loc = ts - seg * dt
    a = sp.coeffs[0].detach().numpy()  # (K, n, 4)
    q = (a[seg, :, 0] + a[seg, :, 1] * loc[:, None] + a[seg, :, 2] * loc[:, None] ** 2
         + a[seg, :, 3] * loc[:, None] ** 3)
    v = (a[seg, :, 1] + 2 * a[seg, :, 2] * loc[:, None] + 3 * a[seg, :, 3] * loc[:, None] ** 2)
    acc = 2 * a[seg, :, 2] + 6 * a[seg, :, 3] * loc[:, None]

    assert np.abs(q - q_ref).max() < TOL, np.abs(q - q_ref).max()
    assert np.abs(v - v_ref).max() < TOL, np.abs(v - v_ref).max()
    assert np.abs(acc - a_ref).max() < TOL, np.abs(acc - a_ref).max()


def test_not_a_knot_is_the_joint_path_spline_rule():
    """``joint_path_spline`` is scipy's default (not-a-knot) C2 cubic.

    Same interpolation rule, different parameter (arc length vs time): feeding
    the *arc-length* knots as if they were uniform reproduces the same
    coefficients up to the parameter scaling, so we assert the rule directly --
    tspline with not-a-knot equals ``make_interp_spline(u, Q, k=3)`` when the
    knots happen to be uniformly spaced in arc length.
    """
    rng = np.random.default_rng(0)
    # Build a path whose consecutive chord lengths are equal, so arc-length
    # parameterization IS a uniform grid and the two are directly comparable.
    K = 12
    dirs = rng.normal(size=(K, N_DOF))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Q = np.concatenate([np.zeros((1, N_DOF)), np.cumsum(dirs * 0.3, axis=0)], axis=0)

    path, length = joint_path_spline(Q)
    assert abs(length - 0.3 * K) < 1e-9
    b = TrajectorySplineBuilder(
        max_segments=K, dt=0.3, bc_start="not-a-knot", bc_end="not-a-knot",
        dtype=torch.float64,
    )
    sp = b.build(
        torch.tensor(Q[0])[None], torch.tensor(Q[1:])[None],
        torch.zeros(1, N_DOF, dtype=torch.float64),
        torch.full((1, b.n_groups), K, dtype=torch.long),
    )
    us = np.linspace(0.0, length, 200)
    seg = np.clip((us / 0.3).astype(int), 0, K - 1)
    loc = us - seg * 0.3
    a = sp.coeffs[0].numpy()
    q = (a[seg, :, 0] + a[seg, :, 1] * loc[:, None] + a[seg, :, 2] * loc[:, None] ** 2
         + a[seg, :, 3] * loc[:, None] ** 3)
    assert np.abs(q - path(us)).max() < TOL


def test_short_horizons_fall_back_gracefully():
    for K in (1, 2, 3):
        gx, gv = spline_gain_matrices(K, 8, CONTROL_DT, "clamped", "natural")
        assert np.all(gx[K + 1 :] == 0.0) and np.all(gx[:, K + 1 :] == 0.0)
        gx2, _ = spline_gain_matrices(K, 8, CONTROL_DT, "not-a-knot", "not-a-knot")
        assert np.isfinite(gx2).all()


def test_start_velocity_is_honoured_exactly():
    K, dt = 10, CONTROL_DT
    rng = np.random.default_rng(3)
    Q = torch.tensor(rng.normal(size=(1, K + 1, N_DOF)) * 0.3)
    v0 = torch.tensor(rng.normal(size=(1, N_DOF)))
    b = TrajectorySplineBuilder(max_segments=K, dt=dt, dtype=torch.float64)
    sp = b.build(Q[:, 0], Q[:, 1:], v0, torch.full((1, b.n_groups), K, dtype=torch.long))
    assert torch.allclose(sp.m[:, 0], v0, atol=1e-12)
    # natural end: acceleration at the terminal knot is zero
    _q, _v, acc, _j = sp.eval_offsets(torch.tensor([dt], dtype=torch.float64))
    assert acc[0, K - 1, 0].abs().max() < 1e-10


def test_masking_is_exact():
    """Values in masked elements cannot change anything the mask keeps."""
    H, dt = 30, CONTROL_DT
    torch.manual_seed(0)
    b = TrajectorySplineBuilder(max_segments=H, dt=dt, dtype=torch.float64)
    q0 = torch.randn(4, N_DOF, dtype=torch.float64)
    wp = torch.randn(4, H, N_DOF, dtype=torch.float64)
    v0 = torch.randn(4, N_DOF, dtype=torch.float64)
    lengths = torch.tensor([[20, 12, 20], [30, 30, 17], [8, 8, 5], [25, 25, 25]])
    sp = b.build(q0, wp, v0, lengths)

    wp2 = wp.clone()
    per = lengths[:, b._group_np]
    idx = torch.arange(H).view(1, -1, 1)
    beyond = idx >= per.unsqueeze(1)  # waypoint j (knot j+1) beyond the horizon
    wp2[beyond] = 1e3 * torch.randn(int(beyond.sum()), dtype=torch.float64)
    sp2 = b.build(q0, wp2, v0, lengths)

    for name in ("coeffs", "m"):
        a, c = getattr(sp, name), getattr(sp2, name)
        assert torch.equal(a, c), name
    ii = sp.segment_integrals()
    jj = sp2.segment_integrals()
    for k in ii:
        assert torch.equal(ii[k], jj[k])


def test_analytic_vs_subsampled_velocity_max():
    H, dt = 50, CONTROL_DT
    torch.manual_seed(1)
    b = TrajectorySplineBuilder(max_segments=H, dt=dt, dtype=torch.float64)
    q0 = torch.randn(16, N_DOF, dtype=torch.float64) * 0.3
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(16, H, N_DOF, dtype=torch.float64) * 0.05, dim=1
    )
    v0 = torch.randn(16, N_DOF, dtype=torch.float64)
    sp = b.build(q0, wp, v0, torch.full((16, b.n_groups), H, dtype=torch.long))
    exact = sp.segment_velocity_max_analytic()
    for n in (3, 5, 9):
        approx = sp.segment_velocity_max(n)
        rel = ((exact - approx) / exact.clamp_min(1e-6)).abs().max().item()
        print(f"n_probe={n}: max relative under-estimate of |qd|max = {rel:.3e}")
        assert approx.max() <= exact.max() + 1e-9
    # Measured: 5 probes under-estimate the true per-segment |qd|max by up to
    # 7.4% on a joint-space random walk.  That is why the cost defaults to the
    # ANALYTIC max (velocity is quadratic per segment, so the extremum is one
    # clamped division -- unlike the torque locator the GPU review shelved).
    assert ((exact - sp.segment_velocity_max(5)) / exact.clamp_min(1e-6)).abs().max() < 0.10
    assert ((exact - sp.segment_velocity_max(9)) / exact.clamp_min(1e-6)).abs().max() < 0.05


def test_integrals_match_dense_quadrature():
    H, dt = 12, CONTROL_DT
    torch.manual_seed(2)
    b = TrajectorySplineBuilder(max_segments=H, dt=dt, dtype=torch.float64)
    q0 = torch.randn(3, N_DOF, dtype=torch.float64)
    wp = torch.randn(3, H, N_DOF, dtype=torch.float64)
    v0 = torch.randn(3, N_DOF, dtype=torch.float64)
    sp = b.build(q0, wp, v0, torch.full((3, b.n_groups), H, dtype=torch.long))
    closed = sp.segment_integrals()
    n = 4096
    s = (torch.arange(n, dtype=torch.float64) + 0.5) * (dt / n)
    _q, qd, qdd, qddd = sp.eval_offsets(s)
    num = {
        "vel2": (qd**2).sum(2) * (dt / n),
        "acc2": (qdd**2).sum(2) * (dt / n),
        "jerk2": (qddd**2).sum(2) * (dt / n),
    }
    for k, v in num.items():
        rel = ((closed[k] - v).abs() / v.abs().clamp_min(1e-9)).max().item()
        assert rel < 1e-6, (k, rel)


def test_gradcheck():
    torch.manual_seed(4)
    H = 6
    b = TrajectorySplineBuilder(max_segments=H, dt=CONTROL_DT, dtype=torch.float64)
    q0 = torch.randn(2, N_DOF, dtype=torch.float64, requires_grad=True)
    wp = torch.randn(2, H, N_DOF, dtype=torch.float64, requires_grad=True)
    v0 = torch.randn(2, N_DOF, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([[H, H - 2, H], [H - 1, H - 1, H - 3]])

    def f(q0_, wp_, v0_):
        sp = b.build(q0_, wp_, v0_, lengths)
        ss = sp.subsample(2)
        return torch.stack(
            [
                (ss.q * ss.mask).sum(),
                (ss.qd * ss.mask).sum(),
                (ss.qdd * ss.mask).sum(),
                sp.segment_integrals()["acc2"].sum(),
            ]
        )

    assert torch.autograd.gradcheck(f, (q0, wp, v0), eps=1e-6, atol=1e-7)


def test_subsample_knot_alignment():
    H, n = 9, 3
    b = TrajectorySplineBuilder(max_segments=H, dt=CONTROL_DT, dtype=torch.float64)
    q0 = torch.randn(2, N_DOF, dtype=torch.float64)
    wp = torch.randn(2, H, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(2, N_DOF, dtype=torch.float64)
    sp = b.build(q0, wp, v0, torch.full((2, b.n_groups), H, dtype=torch.long))
    ss = sp.subsample(n)
    assert ss.n_samples == H * n + 1
    for i in range(H + 1):
        assert torch.allclose(ss.q[:, i * n], sp.x[:, i], atol=1e-12)
