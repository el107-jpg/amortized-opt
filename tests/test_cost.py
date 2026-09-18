"""Unit tests for the fused cost: conformance, masking, gradients, syncs.

The heavier baseline validation (hinge separation, noise monotonicity, descent)
lives in ``test_cost_validation.py``; the full report is
``scripts/validate_cost.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import N_DOF, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.fk import stick_axis_local

_HERE = Path(__file__).resolve().parent
FIXTURE = _HERE / "data" / "trajopt_hpipm_fixture.npz"
FIXTURE_CASES = _HERE / "data" / "trajopt_hpipm_fixture_cases.json"


# ------------------------------------------------------------------ setup --
@pytest.fixture(scope="module")
def fixture_run():
    if not FIXTURE.exists():
        pytest.skip("trajopt fixture missing")
    blob = json.loads(FIXTURE_CASES.read_text())
    cases = [A.EvalCase.from_dict(d) for d in blob["cases"]]
    z = np.load(FIXTURE)
    trajs = {int(k.split("_")[1]): np.asarray(z[k], float) for k in z.files
             if k.startswith("case_")}
    cases = [c for c in cases if c.case_id in trajs]
    return cases, trajs


@pytest.fixture(scope="module")
def cost_cpu():
    return FusedCost(CostConfig(), max_horizon=40, device="cpu", dtype=torch.float64)


def _pack(cases, trajs, H, dtype=torch.float64):
    B = len(cases)
    q0 = torch.zeros(B, N_DOF, dtype=dtype)
    v0 = torch.zeros(B, N_DOF, dtype=dtype)
    wp = torch.zeros(B, H, N_DOF, dtype=dtype)
    for i, c in enumerate(cases):
        Q = trajs[c.case_id]
        q0[i] = torch.as_tensor(Q[0], dtype=dtype)
        v0[i] = torch.as_tensor(c.qdot_start, dtype=dtype)
        wp[i, : c.N] = torch.as_tensor(Q[1 : c.N + 1], dtype=dtype)
        wp[i, c.N :] = torch.as_tensor(Q[c.N], dtype=dtype)
    return q0, wp, v0


def _synthetic_ctx(B, D, device, dtype, *, kl, kr, active=(1.0, 1.0)):
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=torch.tensor(active, device=device, dtype=dtype).expand(B, 2).contiguous(),
        target_drum=torch.zeros(B, 2, device=device, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, device=device, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, device=device, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, device=device, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, device=device, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, device=device, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, device=device, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        elev_required=ones,
    )


# ------------------------------------------------------- harness agreement --
def test_torque_matches_the_scorer_exactly(fixture_run, cost_cpu):
    """``physics_mode="knot_fd"`` reproduces ``TrajectoryScorer``'s torque."""
    cases, trajs = fixture_run
    H = cost_cpu.max_horizon
    q0, wp, v0 = _pack(cases, trajs, H)
    ctx = A.strike_context_from_cases(cases, drum_index=cost_cpu.drum_index,
                                      dtype=torch.float64)
    out = cost_cpu.forward(q0, wp, v0, ctx)
    scorer = A.TrajectoryScorer(device="cpu")
    ref_max = np.array([scorer.score(c, trajs[c.case_id]).tau_max_nm for c in cases])
    ref_ex = np.array(
        [scorer.score(c, trajs[c.case_id]).tau_excess_max_nm for c in cases]
    )
    got_max = out.diagnostics["torque_max"].numpy()
    got_ex = out.diagnostics["torque_excess_max"].numpy()
    assert np.abs(got_max - ref_max).max() < 1e-9, np.abs(got_max - ref_max).max()
    assert np.abs(got_ex - ref_ex).max() < 1e-9


def test_contact_geometry_matches_the_scorer(fixture_run, cost_cpu):
    cases, trajs = fixture_run
    q0, wp, v0 = _pack(cases, trajs, cost_cpu.max_horizon)
    ctx = A.strike_context_from_cases(cases, drum_index=cost_cpu.drum_index,
                                      dtype=torch.float64)
    out = cost_cpu.forward(q0, wp, v0, ctx)
    scorer = A.TrajectoryScorer(device="cpu")
    rows = [scorer.score(c, trajs[c.case_id]) for c in cases]
    ref_pos = np.array([r.contact_pos_err_mm for r in rows]) / 1000.0
    ref_elev = np.abs(np.array([r.stick_elev_deg for r in rows]))
    assert np.abs(out.diagnostics["contact_pos_err"].numpy() - ref_pos).max() < 1e-9
    assert np.abs(out.diagnostics["stick_elev_deg"].numpy() - ref_elev).max() < 1e-9


def test_collision_agrees_with_the_scorer_when_substepping_matches(fixture_run):
    """At ``n_sub = 4`` the cost sees the same interpolated path the scorer does.

    The residual gap is the interpolant: the scorer walks a LINEAR interpolation
    between waypoints, the cost walks the C2 spline.
    """
    cases, trajs = fixture_run
    cost = FusedCost(CostConfig(n_sub=4), max_horizon=40, device="cpu",
                     dtype=torch.float64)
    q0, wp, v0 = _pack(cases, trajs, cost.max_horizon)
    ctx = A.strike_context_from_cases(cases, drum_index=cost.drum_index,
                                      dtype=torch.float64)
    out = cost.forward(q0, wp, v0, ctx)
    scorer = A.TrajectoryScorer(device="cpu")
    ref = np.array(
        [scorer.score(c, trajs[c.case_id]).coll_min_sdf_all_m for c in cases]
    )
    got = out.diagnostics["collision_min_sdf"].numpy()
    assert np.abs(got - ref).max() < 0.02, np.abs(got - ref).max()


# ----------------------------------------------------------------- masking --
def test_masked_elements_cannot_change_the_cost():
    torch.manual_seed(0)
    H, B = 24, 6
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.03, dim=1
    )
    v0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.3
    kl = torch.tensor([10, 24, 7, 18, 12, 24])
    kr = torch.tensor([24, 11, 20, 6, 24, 9])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    base = cost.forward(q0, wp, v0, ctx)

    lengths = cost._lengths_from_context(ctx)
    per = lengths[:, cost.spline._group_np]  # (B, n)
    idx = torch.arange(H).view(1, -1, 1)
    beyond = idx >= per.unsqueeze(1)
    wp2 = wp.clone()
    wp2[beyond] = 17.0 * torch.randn(int(beyond.sum()), dtype=torch.float64)
    other = cost.forward(q0, wp2, v0, ctx)

    assert torch.equal(base.total, other.total)
    for k in base.raw:
        assert torch.equal(base.raw[k], other.raw[k]), k


def test_each_hand_lands_on_its_own_hit_index():
    """Asymmetric horizons: the terminal term reads each hand's own knot."""
    torch.manual_seed(1)
    H, B = 30, 4
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.04, dim=1
    )
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    kl = torch.tensor([11, 30, 17, 8])
    kr = torch.tensor([30, 14, 9, 22])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    out = cost.forward(q0, wp, v0, ctx)

    # Recompute the diagnostic by hand, FK at each hand's own knot.
    x = torch.cat([q0.unsqueeze(1), wp], dim=1)
    manual = torch.zeros(B, dtype=torch.float64)
    for hi, (hand, k) in enumerate((("left", kl), ("right", kr))):
        qk = x[torch.arange(B), k]
        R, p = cost.fk.frame(qk, f"{hand}_wrist")
        loc = torch.as_tensor(
            A.stick_contact_offset(hand, 0.20), dtype=torch.float64
        ).view(1, 3, 1)
        pc = p + (R @ loc).squeeze(-1)
        manual = manual + (pc - ctx.p_star[:, hi]).norm(dim=-1)
    assert torch.allclose(out.diagnostics["contact_pos_err"], manual / 2.0, atol=1e-10)

    # And the terminal term is sensitive to each hand's own knot, not the other's.
    for hi, k in enumerate((kl, kr)):
        wp2 = wp.clone()
        rows = torch.arange(B)
        wp2[rows, (k - 1).clamp_min(0)] += 0.05
        d = cost.forward(q0, wp2, v0, ctx).raw["terminal_region"]
        assert not torch.allclose(d, out.raw["terminal_region"])


def test_inactive_hand_contributes_no_terminal_cost():
    torch.manual_seed(2)
    H, B = 20, 3
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.03, dim=1
    )
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    both = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    left_only = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k,
                               active=(1.0, 0.0))
    a = cost.forward(q0, wp, v0, both).raw["terminal_region"]
    b = cost.forward(q0, wp, v0, left_only).raw["terminal_region"]
    assert not torch.allclose(a, b)
    # ... and moving the RIGHT arm cannot change a left-only terminal cost
    wp2 = wp.clone()
    right = cost.spline._group_cols[2]
    wp2[:, :, right] += 0.3
    c = cost.forward(q0, wp2, v0, left_only).raw["terminal_region"]
    assert torch.allclose(b, c, atol=1e-12)


# --------------------------------------------------------------- gradients --
def test_gradcheck_small():
    torch.manual_seed(3)
    H, B = 6, 2
    cfg = CostConfig(
        w_terminal_aim=1.0, w_smooth_vel=1e-3, w_axial_twist=1e-2,
        contact_mode="softmin",
    )
    cost = FusedCost(cfg, max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = (torch.randn(B, N_DOF, dtype=torch.float64) * 0.2).requires_grad_(True)
    wp = (torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.2).requires_grad_(True)
    v0 = (torch.randn(B, N_DOF, dtype=torch.float64) * 0.2).requires_grad_(True)
    kl = torch.tensor([6, 4])
    kr = torch.tensor([5, 6])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    ctx.shaft_allowed = torch.ones(B, 2, dtype=torch.float64)

    def f(a, b, c):
        return cost.forward(a, b, c, ctx).total

    assert torch.autograd.gradcheck(f, (q0, wp, v0), eps=1e-6, atol=2e-6, rtol=1e-3)


def test_gradients_are_finite_at_training_scale():
    torch.manual_seed(4)
    H, B = 50, 64
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float32)
    q0 = torch.randn(B, N_DOF) * 0.2
    wp = (q0.unsqueeze(1) + torch.cumsum(torch.randn(B, H, N_DOF) * 0.03, 1)
          ).requires_grad_(True)
    v0 = torch.randn(B, N_DOF) * 0.5
    kl = torch.randint(20, H + 1, (B,))
    kr = torch.randint(20, H + 1, (B,))
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float32, kl=kl, kr=kr)
    out = cost.forward(q0, wp, v0, ctx)
    out.total.sum().backward()
    assert torch.isfinite(out.total).all()
    assert torch.isfinite(wp.grad).all()
    assert float(wp.grad.abs().max()) > 0.0


def test_elevation_near_pole_gradients_are_finite():
    """Stick along the head normal used to NaN ``asin'(±1)`` in ``terminal_elev``."""
    H, B = 8, 2
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float32)
    q0 = torch.as_tensor(ready_joint_pos(), dtype=torch.float32).expand(B, -1).contiguous()
    v0 = torch.zeros(B, N_DOF)
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    R, _ = cost.fk.frame(q0, "left_wrist")
    u = (
        R @ torch.as_tensor(stick_axis_local("left"), dtype=torch.float32).view(1, 3, 1)
    ).squeeze(-1)
    for tilt in (0.0, 1e-3):
        wp = q0.unsqueeze(1).expand(B, H, N_DOF).contiguous().clone().requires_grad_(True)
        n = -u + tilt * torch.tensor([1.0, 0.0, 0.0])
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float32, kl=kl, kr=kr)
        ctx.n_out[:, 0] = n
        ctx.n_out[:, 1] = n
        out = cost.forward(q0, wp, v0, ctx)
        out.raw["terminal_elev"].sum().backward()
        assert torch.isfinite(out.raw["terminal_elev"]).all(), tilt
        assert torch.isfinite(wp.grad).all(), tilt
        wp2 = q0.unsqueeze(1).expand(B, H, N_DOF).contiguous().clone().requires_grad_(True)
        out2 = cost.forward(q0, wp2, v0, ctx)
        out2.total.sum().backward()
        assert torch.isfinite(out2.total).all(), tilt
        assert torch.isfinite(wp2.grad).all(), tilt


# ------------------------------------------------------------------ device --
def test_no_device_host_sync(cuda_device):
    H, B = 50, 32
    cost = FusedCost(CostConfig(), max_horizon=H, device=cuda_device,
                     dtype=torch.float32)
    torch.manual_seed(5)
    q0 = (torch.randn(B, N_DOF) * 0.2).to(cuda_device)
    wp = (q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, device=cuda_device) * 0.03, 1)).requires_grad_(True)
    v0 = torch.randn(B, N_DOF, device=cuda_device) * 0.5
    kl = torch.randint(20, H + 1, (B,), device=cuda_device)
    kr = torch.randint(20, H + 1, (B,), device=cuda_device)
    ctx = _synthetic_ctx(B, cost.n_drums, cuda_device, torch.float32, kl=kl, kr=kr)
    cost.forward(q0, wp, v0, ctx).total.sum().backward()  # warm the caches
    wp.grad = None
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = cost.forward(q0, wp, v0, ctx)
        out.total.sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(out.total).all()


def test_cuda_matches_cpu(cuda_device):
    torch.manual_seed(6)
    H, B = 20, 8
    cpu = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float32)
    gpu = FusedCost(CostConfig(), max_horizon=H, device=cuda_device,
                    dtype=torch.float32)
    q0 = torch.randn(B, N_DOF) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(torch.randn(B, H, N_DOF) * 0.03, 1)
    v0 = torch.randn(B, N_DOF) * 0.3
    kl = torch.randint(8, H + 1, (B,))
    kr = torch.randint(8, H + 1, (B,))
    a = cpu.forward(q0, wp, v0,
                    _synthetic_ctx(B, cpu.n_drums, "cpu", torch.float32, kl=kl, kr=kr))
    b = gpu.forward(q0.to(cuda_device), wp.to(cuda_device), v0.to(cuda_device),
                    _synthetic_ctx(B, gpu.n_drums, cuda_device, torch.float32,
                                   kl=kl.to(cuda_device), kr=kr.to(cuda_device)))
    rel = ((a.total - b.total.cpu()).abs() / a.total.abs().clamp_min(1e-6)).max()
    assert float(rel) < 2e-3, float(rel)


# -------------------------------------------------------------- lagrangian --
def test_augmented_lagrangian_grows_then_relaxes(tmp_path):
    al = A.AugmentedLagrangian()
    v = {g: torch.tensor([0.05, 0.05]) for g in A.CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(v)
    grown = dict(al.lam)
    assert all(grown[g] > 1.0 for g in A.CONSTRAINT_GROUPS), grown
    zero = {g: torch.zeros(2) for g in A.CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(zero)
    assert all(al.lam[g] == grown[g] for g in A.CONSTRAINT_GROUPS)  # no drift at 0

    al2 = A.AugmentedLagrangian(A.ALConfig(decay=0.9))
    for _ in range(5):
        al2.update(v)
    hi = dict(al2.lam)
    for _ in range(20):
        al2.update(zero)
    assert all(al2.lam[g] < hi[g] for g in A.CONSTRAINT_GROUPS)

    p = al.save(tmp_path / "al.json")
    back = A.AugmentedLagrangian.load(p)
    assert back.lam == al.lam and back.steps == al.steps
    assert al.lam["collision"] <= al.cfg.lam_max


def test_lagrangian_weights_reach_the_cost():
    torch.manual_seed(7)
    H, B = 12, 3
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.2
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    base = cost.forward(q0, wp, v0, ctx)
    doubled = cost.forward(q0, wp, v0, ctx, weights={"torque": 2.0})
    warning = cost.cfg.w_torque * base.raw["torque"]
    # warning (cfg.w * softplus mean) is invariant to λ
    assert torch.allclose(cost.cfg.w_torque * doubled.raw["torque"], warning, atol=1e-12)
    # hard half (terms - warning) scales with λ: family key broadcasts onto
    # every torque_* live residual.  Peak is off (w_torque_peak=0).
    from amortized_opt.lagrangian import ACTUATOR_SEGMENT_NAMES, limit_group_name
    hard = sum(
        base.violations[limit_group_name("torque", s)]
        for s in ACTUATOR_SEGMENT_NAMES
    )
    assert torch.allclose(base.terms["torque"] - warning, hard, atol=1e-12)
    assert torch.allclose(doubled.terms["torque"] - warning, 2.0 * hard, atol=1e-12)
    assert torch.allclose(doubled.terms["collision"], base.terms["collision"])


def test_violations_are_zero_when_feasible():
    """The AL is driven by the RAW excess, which vanishes on a feasible path --
    unlike the softplus penalty, which never does."""
    H, B = 10, 2
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = torch.zeros(B, H, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    out = cost.forward(q0, wp, v0, ctx)
    for g in ("torque", "velocity", "joint_box"):
        assert float(out.violations[g].max()) == 0.0, g
        assert float(out.raw[g].min()) > 0.0, g  # softplus floor


def test_terminal_aim_is_not_al_scaled():
    """``lam_terminal`` must not multiply the in-plane aim pull."""
    torch.manual_seed(5)
    H, B = 8, 2
    cfg = CostConfig(
        w_terminal_aim=10.0, w_terminal_region=0.0,
        w_terminal_velocity=0.0, w_terminal_elev=0.0,
    )
    cost = FusedCost(cfg, max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.03, dim=1
    )
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    a = cost.forward(q0, wp, v0, ctx, weights={"terminal": 1.0})
    b = cost.forward(q0, wp, v0, ctx, weights={"terminal": 4.0})
    assert torch.allclose(a.terms["terminal_aim"], b.terms["terminal_aim"])
    assert float(a.raw["terminal_aim"].mean()) >= 0.0
    assert "terminal_aim" in a.terms
