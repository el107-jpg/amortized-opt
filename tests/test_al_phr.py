"""PHR quadratic, Bertsekas ``pen``, PID used-λ, nonzero targets.

Default :class:`ALConfig` stays the historical I-only object.  Phase-2 turns
the rest on via :func:`~amortized_opt.train_cost.phase2_al_config`.
"""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import ALConfig, AugmentedLagrangian, CONSTRAINT_GROUPS
from amortized_opt.train_cost import clamp_resumed_al_penalty, phase2_al_config

DT = 0.02
WRIST_COLS = (8, 9, 15, 16)
ONE_ELEMENT_AMP = 0.10


def _ctx(B, device="cpu", dtype=torch.float64, *, kl, kr):
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=ones.clone(),
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
        elev_required=ones.clone(),
    )


def _batch(*, amp: float, H: int = 12, B: int = 4):
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    if amp:
        for b in range(B):
            j = WRIST_COLS[b % len(WRIST_COLS)]
            k = 3 + b
            wp[b, k, j] = amp * (-1.0) ** b
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    return dict(q0=q0, wp=wp, v0=v0, kl=kl, kr=kr, H=H, B=B)


def _cost(**cfg_kw):
    return FusedCost(CostConfig(**cfg_kw), max_horizon=12, device="cpu",
                     dtype=torch.float64, dt=DT)


def _fwd(b, cost, **kw):
    return cost.forward(b["q0"], b["wp"], b["v0"],
                        _ctx(b["B"], kl=b["kl"], kr=b["kr"]), **kw)


def _i_only(**kw):
    d = dict(rho=1.0, decay=1.0, ema=0.0, lam_min=0.0, lam_max=1.0e6,
             kp=0.0, kd=0.0, pen=0.0, pen_grow=1.0, pen_shrink=1.0, target=0.0)
    d.update(kw)
    return ALConfig(**d)


# -------------------------------------------------------- default is I-only --
def test_default_config_grows_then_freezes_at_zero_violation():
    """Bit-identical to the pre-PHR object: V=0 and decay=1 leaves λ frozen."""
    al = AugmentedLagrangian()
    v = {g: torch.tensor([0.05, 0.05]) for g in CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(v)
    grown = dict(al.lam)
    assert all(grown[g] > 1.0 for g in CONSTRAINT_GROUPS)
    assert all(al.pen[g] == 0.0 for g in CONSTRAINT_GROUPS)
    zero = {g: torch.zeros(2) for g in CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(zero)
    assert all(al.lam[g] == grown[g] for g in CONSTRAINT_GROUPS)


def test_default_config_decay_relaxes_when_feasible():
    al = AugmentedLagrangian(ALConfig(decay=0.9))
    v = {g: 0.05 for g in CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(v)
    hi = dict(al.lam)
    zero = {g: 0.0 for g in CONSTRAINT_GROUPS}
    for _ in range(20):
        al.update(zero)
    assert all(al.lam[g] < hi[g] for g in CONSTRAINT_GROUPS)


# --------------------------------------------------------------------- PID --
def test_pid_p_term_raises_used_lambda_above_integral():
    al = AugmentedLagrangian(_i_only(rho=2.0, kp=2.0, kd=0.0))
    assert al.weights()["torque_waist"] == pytest.approx(al.lam["torque_waist"])
    used = al.update({"torque_waist": 4.0})
    assert al.lam["torque_waist"] == pytest.approx(1.0 + 2.0 * 4.0)
    assert used["torque_waist"] == pytest.approx(al.lam["torque_waist"] + 2.0 * 4.0)
    assert al.err_prev["torque_waist"] == pytest.approx(4.0)
    assert al.weights()["torque_waist"] == pytest.approx(used["torque_waist"])


def test_pid_d_term_is_zero_on_first_observation_then_damps_a_drop():
    al = AugmentedLagrangian(_i_only(rho=1.0, kp=0.0, kd=0.5))
    first = al.update({"torque_waist": 5.0})
    assert first["torque_waist"] == pytest.approx(al.lam["torque_waist"])
    assert al._d["torque_waist"] == pytest.approx(0.0)
    i_before = al.lam["torque_waist"]
    second = al.update({"torque_waist": 1.0})
    assert al.lam["torque_waist"] == pytest.approx(i_before + 1.0)
    assert al._d["torque_waist"] == pytest.approx(0.5 * (1.0 - 5.0))
    assert second["torque_waist"] == pytest.approx(al.lam["torque_waist"] + al._d["torque_waist"])
    assert second["torque_waist"] < al.lam["torque_waist"]
    assert al.weights()["torque_waist"] == pytest.approx(second["torque_waist"])


# ---------------------------------------------------------------- Bertsekas --
def test_bertsekas_grows_pen_on_stale_violation_and_still_updates_lambda():
    al = AugmentedLagrangian(_i_only(rho=1.0, pen=2.0, pen_grow=2.0, pen_shrink=2.0,
                                     pen_max=1.0e6, improve_tol=0.9))
    v = {g: 1.0 for g in CONSTRAINT_GROUPS}
    al.update(v)                       # first look: do not grow
    assert al.pen["torque_waist"] == pytest.approx(2.0)
    lam0 = al.lam["torque_waist"]
    al.update(v)
    assert al.pen["torque_waist"] == pytest.approx(4.0)
    assert al.lam["torque_waist"] > lam0


def test_bertsekas_shrinks_pen_once_inside_target():
    al = AugmentedLagrangian(_i_only(
        rho=1.0, target=0.5, pen=8.0, pen_grow=2.0, pen_shrink=2.0,
        pen_min=1.0e-6, pen_max=1.0e6,
    ))
    al.update({"torque_waist": 1.0})
    al.update({"torque_waist": 0.0})
    assert al.pen["torque_waist"] == pytest.approx(4.0)


def test_pen_shrink_preserves_exactly_zero_disabled_penalty():
    al = AugmentedLagrangian(_i_only(
        target=0.5, pen=0.0, pen_min=1.0e-6,
        pen_grow=1.0, pen_shrink=2.0,
    ))
    for _ in range(5):
        al.update({"torque_waist": 0.0})
    assert al.pen["torque_waist"] == 0.0
    assert "torque_waist" not in al.pen_weights()


def test_resume_clamp_preserves_disabled_zero_penalty():
    assert clamp_resumed_al_penalty(0.0, 1.0e-6, 200.0) == 0.0


def test_resume_clamp_applies_bounds_to_enabled_penalty():
    assert clamp_resumed_al_penalty(1.0e-9, 1.0e-6, 200.0) == 1.0e-6
    assert clamp_resumed_al_penalty(300.0, 1.0e-6, 200.0) == 200.0


def test_positive_penalty_still_shrinks_to_configured_floor():
    al = AugmentedLagrangian(_i_only(
        target=0.5, pen=3.0e-6, pen_min=1.0e-6,
        pen_grow=1.0, pen_shrink=2.0,
    ))
    al.update({"torque_waist": 0.0})
    assert al.pen["torque_waist"] == pytest.approx(1.5e-6)
    al.update({"torque_waist": 0.0})
    assert al.pen["torque_waist"] == pytest.approx(1.0e-6)


def test_pen_grow_one_with_shrink_above_one_is_decay_only():
    al = AugmentedLagrangian(_i_only(
        target=0.5, pen=8.0, pen_min=1.0e-6,
        pen_grow=1.0, pen_shrink=2.0, improve_tol=0.9,
    ))
    al.update({"torque_waist": 1.0})
    al.update({"torque_waist": 1.0})
    assert al.pen["torque_waist"] == pytest.approx(8.0)
    al.update({"torque_waist": 0.0})
    assert al.pen["torque_waist"] == pytest.approx(4.0)


def test_pen_does_not_grow_while_lambda_is_at_the_ceiling():
    """The 32k continuation: λ pinned, V hovering, 1.5^20 in 2k steps."""
    al = AugmentedLagrangian(_i_only(
        rho=1.0, pen=2.0, pen_grow=2.0, pen_max=1.0e6, lam_max=5.0, lam_init=5.0,
    ))
    v = {g: 1.0 for g in CONSTRAINT_GROUPS}
    for _ in range(20):
        al.update(v)
    assert al.lam["torque_waist"] == pytest.approx(5.0)
    assert al.pen["torque_waist"] == pytest.approx(2.0)


def test_pen_patience_requires_consecutive_stale_updates():
    al = AugmentedLagrangian(_i_only(
        rho=1.0, pen=2.0, pen_grow=2.0, pen_max=1.0e6, pen_patience=3,
    ))
    v = {g: 1.0 for g in CONSTRAINT_GROUPS}
    al.update(v)                       # first look
    al.update(v)                       # stale 1
    al.update(v)                       # stale 2
    assert al.pen["torque_waist"] == pytest.approx(2.0)
    al.update(v)                       # stale 3 → grow, counter resets
    assert al.pen["torque_waist"] == pytest.approx(4.0)
    al.update(v)                       # stale 1 again
    assert al.pen["torque_waist"] == pytest.approx(4.0)


def test_phase2_fork_at_lam_max_cannot_compound_pen():
    """Replay of the failed resume: torque at 100, residual above target."""
    al = AugmentedLagrangian(phase2_al_config())
    g = "torque_waist"
    al.lam[g] = al.lam_max[g]
    pen0 = al.pen[g]
    for _ in range(20):
        al.update({g: 1.0})
    assert al.lam[g] == pytest.approx(al.lam_max[g])
    assert al.pen[g] == pytest.approx(pen0)
    # grow is frozen on the live recipe, so collision cannot compound either
    # (that path is what took the first segment resume 1e4 → 6e6).
    c = "collision"
    pen_c = al.pen[c]
    for _ in range(22):
        al.update({c: 0.02})
    assert al.pen[c] == pytest.approx(pen_c)


def test_phase2_config_turns_phr_and_pid_on():
    cfg = phase2_al_config()
    assert cfg.target["torque_peak"] == 0.0
    assert cfg.target["joint_box_waist"] > 0.0
    assert cfg.kp["torque_waist"] == pytest.approx(cfg.rho["torque_waist"])
    assert cfg.kd["torque_waist"] == pytest.approx(0.25 * cfg.rho["torque_waist"])
    assert cfg.pen["torque_waist"] == 0.0
    assert cfg.pen["torque_peak"] == 0.0
    assert cfg.pen_grow == 1.0
    assert cfg.pen["joint_box_waist"] == pytest.approx(800.0)
    assert cfg.pen_patience == 10
    assert cfg.feasible_decay < cfg.decay
    al = AugmentedLagrangian(cfg)
    assert al.pen_weights()
    assert "pen_torque_waist" not in al.cost_weights()
    assert "pen_joint_box_waist" in al.cost_weights()
    assert "pen_torque_waist" not in al.weights()
    al.update({"torque_waist": 0.0})
    assert al.pen["torque_waist"] == 0.0
    assert "pen_torque_waist" not in al.cost_weights()


# ---------------------------------------------------------- cost quadratic --
def test_absent_pen_key_is_bit_identical_to_the_pre_quadratic_term():
    b = _batch(amp=ONE_ELEMENT_AMP)
    cost = _cost(w_torque_peak=1.0)
    ref = _fwd(b, cost)
    same = _fwd(b, cost, weights={"torque": 1.0, "torque_peak": 1.0, "smooth": 1.0})
    assert torch.equal(ref.terms["torque"], same.terms["torque"])
    assert torch.equal(ref.total, same.total)
    assert not ref.violations["torque"].requires_grad
    assert not ref.violations["torque_peak"].requires_grad


def test_pen_torque_adds_half_pen_v_squared_and_has_grad_wrt_traj():
    b = _batch(amp=ONE_ELEMENT_AMP, B=2)
    cost = _cost(w_torque_peak=1.0)
    ref = _fwd(b, cost)
    pen = 1.0e4
    hot = _fwd(b, cost, weights={
        "pen_torque_left_wrist": torch.tensor(pen, dtype=torch.float64),
    })
    extra = 0.5 * pen * ref.violations["torque_left_wrist"].clamp_min(0.0).square()
    assert torch.allclose(hot.terms["torque"], ref.terms["torque"] + extra, atol=1e-12)
    assert float((hot.terms["torque"] - ref.terms["torque"]).min()) > 0.0

    wp = b["wp"].clone().requires_grad_(True)
    ctx = _ctx(b["B"], kl=b["kl"], kr=b["kr"])
    out = cost.forward(b["q0"], wp, b["v0"], ctx,
                       weights={"pen_torque_left_wrist": torch.tensor(pen, dtype=torch.float64)})
    out.terms["torque"].sum().backward()
    assert wp.grad is not None and torch.isfinite(wp.grad).all()
    assert float(wp.grad.abs().max()) > 0.0


def test_pen_key_does_not_leak_into_the_other_half_when_only_peak_is_priced():
    """``pen_torque_peak`` lands on ``terms['torque']`` (same term key) but
    squares the smooth-max residual, not the detached hard max."""
    b = _batch(amp=0.0, B=2)
    # Two wrist knots, so the p-norm sits strictly above the hard max.
    b["wp"][:, 4, 8] = 0.25
    b["wp"][:, 7, 9] = -0.25
    cost = _cost(w_torque_peak=1.0, torque_peak_p=2.0)
    ref = _fwd(b, cost)
    pen = 0.5
    hot = _fwd(b, cost, weights={
        "pen_torque_peak": torch.tensor(pen, dtype=torch.float64),
    })
    extra = 0.5 * pen * ref.raw["torque_peak"].clamp_min(0.0).square()
    hard = 0.5 * pen * ref.violations["torque_peak"].clamp_min(0.0).square()
    assert torch.allclose(hot.terms["torque"], ref.terms["torque"] + extra, atol=1e-12)
    # Two violated cells make the p-norm strictly larger than ``amax``, so
    # this locks the penalty onto the smooth residual rather than the hard max.
    assert float((extra - hard).abs().max()) > 1e-8
    assert torch.equal(hot.violations["torque"], ref.violations["torque"])
    assert torch.equal(hot.violations["torque_peak"], ref.violations["torque_peak"])
    assert not hot.violations["torque_peak"].requires_grad


def test_wtensors_emits_nonzero_pen_keys_and_omits_zeros():
    from amortized_opt.train_cost import WeightSchedule, _wtensors

    lam = {g: 1.0 for g in CONSTRAINT_GROUPS}
    mult = WeightSchedule().multipliers(0)
    none = _wtensors(mult, lam, "cpu", torch.float32)
    assert all(not k.startswith("pen_") for k in none)
    out = _wtensors(mult, lam, "cpu", torch.float32,
                    pen={"torque_waist": 10.0, "velocity_waist": 0.0})
    assert float(out["pen_torque_waist"]) == 10.0
    assert "pen_velocity_waist" not in out


def test_feasible_decay_unsticks_a_dual_at_the_ceiling():
    """With target slack, a residual under target plus feasible_decay actually
    brings λ down; the historical decay=0.999 from lam_max=100 cannot."""
    al = AugmentedLagrangian(phase2_al_config())
    al.lam["joint_box_waist"] = al.lam_max["joint_box_waist"]
    al.update({"joint_box_waist": 0.012})
    assert al.avg_violation["joint_box_waist"] < al.target["joint_box_waist"]
    assert al.lam["joint_box_waist"] < al.lam_max["joint_box_waist"]
