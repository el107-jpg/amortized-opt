"""``waist_pr_band`` as its own augmented-Lagrangian group.

Decoder waist pitch/roll must stay inside a deadband of the encoder
set-point.  The residual is a peak ReLU excess,

    relu(|q_pr - stopgrad(sp)| - band)

so a 2 rad dump cannot hide in a mean over the sub-sample × 2 lean rows,
and ``stopgrad(sp)`` keeps this group from moving the encoder (two-way
gradient is how yaw SP co-drifted).  This is not ``w_waist_pr_sp_tether``
(encoder residual to upright).
"""
from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import CONSTRAINT_GROUPS, ALConfig, AugmentedLagrangian
from amortized_opt.train_cost import migrate_al_state, phase2_al_config, phase2_cost_config

DT = 0.02
H = 10
B = 4


def _strike(B: int, H: int, pr_ref: torch.Tensor | None) -> StrikeContext:
    kl = torch.full((B,), H - 1, dtype=torch.long)
    return StrikeContext(
        hit_index=torch.stack([kl, kl], dim=1),
        active=torch.ones(B, 2, dtype=torch.float64),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=torch.float64),
        v_star=torch.full((B, 2, 3), -1.0, dtype=torch.float64),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=torch.float64),
        disk_offset=torch.zeros(B, 2, 3, dtype=torch.float64),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=torch.float64),
        has_half_plane=torch.zeros(B, 2, dtype=torch.float64),
        contact_sigma=torch.full((B, 2), 0.20, dtype=torch.float64),
        shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        start_shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        elev_required=torch.ones(B, 2, dtype=torch.float64),
        waist_pr_ref=pr_ref,
    )


def _cost(**cfg_kw):
    return FusedCost(CostConfig(n_sub=1, **cfg_kw), max_horizon=H,
                     device="cpu", dtype=torch.float64, dt=DT)


def _hold(cost, *, pitch, roll, dump_pitch=None, dump_knot=None):
    """Waypoints holding ``(pitch, roll)``; optional one-knot pitch dump."""
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    pitch_i = cost.joint_names.index("waist_pitch_joint")
    roll_i = cost.joint_names.index("waist_roll_joint")
    q0[:, pitch_i] = float(pitch)
    q0[:, roll_i] = float(roll)
    wp[:, :, pitch_i] = float(pitch)
    wp[:, :, roll_i] = float(roll)
    if dump_pitch is not None:
        k = 3 if dump_knot is None else int(dump_knot)
        wp[:, k, pitch_i] = float(dump_pitch)
    return q0, wp, v0, pitch_i, roll_i


def _sp(value_p=0.1, value_r=0.0, *, requires_grad=False):
    sp = torch.full((B, H, 2), 0.0, dtype=torch.float64)
    sp[..., 0] = float(value_p)
    sp[..., 1] = float(value_r)
    if requires_grad:
        sp = sp.clone().requires_grad_(True)
    return sp


# ============================================================== 1. off ======
def test_zero_weight_is_a_noop_on_the_total_but_still_reports_peak_viol():
    cost = _cost(w_waist_pr_band=0.0, waist_pr_band_rad=0.15)
    assert cost.cfg.w_waist_pr_band == 0.0
    sp = _sp(0.1)
    q0, wp, v0, _, _ = _hold(cost, pitch=0.1, roll=0.0, dump_pitch=2.0)
    out = cost.forward(q0, wp, v0, _strike(B, H, sp))
    assert "waist_pr_band" in out.violations
    assert "waist_pr_band" in out.terms
    # 2.0 - 0.1 - 0.15 = 1.75 on the dumped knot; sample 0 is masked.
    assert torch.allclose(out.violations["waist_pr_band"],
                          torch.full((B,), 1.75, dtype=torch.float64))
    # warning is product-zero; hard wall is λ=1 × live
    assert torch.allclose(out.terms["waist_pr_band"],
                          out.violations["waist_pr_band"], atol=1e-12)


def test_defaults_validate_and_band_zero_is_legal():
    cfg = CostConfig().validate()
    assert cfg.w_waist_pr_band == 0.0
    assert cfg.waist_pr_band_rad == 0.15
    CostConfig(w_waist_pr_band=0.0, waist_pr_band_rad=0.0).validate()
    with pytest.raises(ValueError, match="w_waist_pr_band"):
        CostConfig(w_waist_pr_band=-1.0).validate()
    with pytest.raises(ValueError, match="waist_pr_band_rad"):
        CostConfig(waist_pr_band_rad=-0.01).validate()


# =========================================================== 2. in band =====
def test_inside_the_band_pays_zero_relu_viol():
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    sp = _sp(0.1)
    q0, wp, v0, _, _ = _hold(cost, pitch=0.1, roll=0.0)
    out = cost.forward(q0, wp, v0, _strike(B, H, sp))
    assert torch.allclose(out.violations["waist_pr_band"],
                          torch.zeros(B, dtype=torch.float64), atol=1e-12)

    q0b, wpb, v0b, _, _ = _hold(cost, pitch=0.20, roll=0.0)  # |0.20-0.10|=0.10 < 0.15
    outb = cost.forward(q0b, wpb, v0b, _strike(B, H, sp))
    assert torch.allclose(outb.violations["waist_pr_band"],
                          torch.zeros(B, dtype=torch.float64), atol=1e-12)


# ========================================================== 3. outside ======
def test_outside_viol_equals_peak_excess():
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    sp = _sp(0.1)
    q0, wp, v0, _, _ = _hold(cost, pitch=0.1, roll=0.0, dump_pitch=2.0)
    out = cost.forward(q0, wp, v0, _strike(B, H, sp))
    expect = 2.0 - 0.1 - 0.15  # 1.75
    assert torch.allclose(out.violations["waist_pr_band"],
                          torch.full((B,), expect, dtype=torch.float64))
    assert not out.violations["waist_pr_band"].requires_grad
    assert out.raw["waist_pr_band"].shape == (B,)


# ======================================================== 4. stopgrad(sp) ===
def test_stopgrad_sp_zero_grad_on_ref_nonzero_on_dumped_q():
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    sp = _sp(0.1, requires_grad=True)
    q0, wp, v0, pitch_i, _ = _hold(cost, pitch=0.1, roll=0.0, dump_pitch=2.0)
    wp = wp.clone().requires_grad_(True)
    out = cost.forward(q0, wp, v0, _strike(B, H, sp))
    out.total.sum().backward()
    assert sp.grad is None or float(sp.grad.abs().max()) == 0.0
    assert wp.grad is not None
    # dumped knot (index 3) must feel the hinge; a knot still on SP should not
    assert float(wp.grad[:, 3, pitch_i].abs().max()) > 0.0


# ====================================================== 5. peak not mean ====
def test_raw_is_sum_of_waypoint_relu_viol_is_max():
    """One dumped knot: raw == viol == excess; two dumps: raw = 2×viol."""
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    sp = _sp(0.1)
    q0, wp, v0, pitch_i, _ = _hold(cost, pitch=0.1, roll=0.0, dump_pitch=2.0, dump_knot=3)
    out = cost.forward(q0, wp, v0, _strike(B, H, sp))
    excess = 2.0 - 0.1 - 0.15
    assert torch.allclose(out.raw["waist_pr_band"],
                          torch.full((B,), excess, dtype=torch.float64))
    assert torch.allclose(out.violations["waist_pr_band"],
                          torch.full((B,), excess, dtype=torch.float64))
    wp2 = wp.clone()
    wp2[:, 7, pitch_i] = 2.0
    out2 = cost.forward(q0, wp2, v0, _strike(B, H, sp))
    assert torch.allclose(out2.raw["waist_pr_band"],
                          torch.full((B,), 2.0 * excess, dtype=torch.float64))
    assert torch.allclose(out2.violations["waist_pr_band"],
                          torch.full((B,), excess, dtype=torch.float64))


# =========================================================== 6. no ref ======
def test_no_ref_zero_weight_reports_exact_zeros():
    """Weight=0 keeps the term exactly zero regardless of pose."""
    cost = _cost(w_waist_pr_band=0.0, waist_pr_band_rad=0.15)
    q0, wp, v0, _, _ = _hold(cost, pitch=0.0, roll=0.0, dump_pitch=2.0)
    out = cost.forward(q0, wp, v0, _strike(B, H, None))
    z = torch.zeros(B, dtype=torch.float64)
    assert torch.equal(out.violations["waist_pr_band"], z)
    assert torch.equal(out.raw["waist_pr_band"], z)
    assert torch.equal(out.terms["waist_pr_band"], z)


def test_no_ref_nonzero_weight_uses_upright_as_reference():
    """With no SP head, band should price lean beyond band_rad of zero (upright)."""
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    # dump_pitch=2.0 puts the trajectory at 2.0 rad pitch, well past 0.15
    q0, wp, v0, _, _ = _hold(cost, pitch=0.0, roll=0.0, dump_pitch=2.0)
    out = cost.forward(q0, wp, v0, _strike(B, H, None))
    # excess = 2.0 - 0.15 = 1.85 per waypoint, so violations > 0
    assert out.violations["waist_pr_band"].min().item() > 0.0
    assert out.raw["waist_pr_band"].min().item() > 0.0
    assert out.terms["waist_pr_band"].min().item() > 0.0


def test_no_ref_within_band_reports_zero_violation():
    """Lean within band_rad of upright produces no violation."""
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    q0, wp, v0, _, _ = _hold(cost, pitch=0.0, roll=0.0, dump_pitch=0.05)
    out = cost.forward(q0, wp, v0, _strike(B, H, None))
    z = torch.zeros(B, dtype=torch.float64)
    assert torch.equal(out.violations["waist_pr_band"], z)


# ============================================== 7. other waist terms ========
def test_does_not_mix_into_posture_and_upright_stopgrads_sp():
    sp = _sp(0.1, requires_grad=True)
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    # dump so both the band and the upright well have something to price
    cost_on = _cost(w_waist_pr_band=1.0, w_waist_upright=1.0, waist_pr_band_rad=0.15)
    pitch_i = cost_on.joint_names.index("waist_pitch_joint")
    wp[:, :, pitch_i] = 2.0
    wp = wp.clone().requires_grad_(True)
    off = _cost(w_waist_pr_band=0.0, w_waist_upright=1.0, waist_pr_band_rad=0.15)
    ctx = _strike(B, H, sp)
    out_on = cost_on.forward(q0, wp, v0, ctx)
    out_off = off.forward(q0, wp.detach(), v0, ctx)
    assert "waist_pr_band" in out_on.terms
    assert torch.equal(out_on.terms["posture"], out_off.terms["posture"])
    assert float(out_on.terms["waist_pr_band"].abs().max()) > 0.0
    # upright is a decoder tax: stopgrad(sp), nonzero grad on dumped q
    out_on.terms["waist_upright"].sum().backward()
    assert sp.grad is None or float(sp.grad.abs().max()) == 0.0
    assert float(wp.grad[:, :, pitch_i].abs().max()) > 0.0


# ======================================================== 8. AL wiring ======
def test_group_is_last_and_phase2_al_has_every_dict():
    assert CONSTRAINT_GROUPS[-1] == "velocity_peak"
    assert CONSTRAINT_GROUPS[-5:-1] == (
        "torque_peak", "collision_peak", "joint_box_peak", "waist_pr_band",
    )
    cfg = phase2_al_config()
    for name in ("rho", "target", "pen", "lam_max"):
        d = getattr(cfg, name)
        assert "waist_pr_band" in d, name
        assert "velocity_peak" in d, name
    assert cfg.rho["waist_pr_band"] == pytest.approx(1.0)
    assert cfg.target["waist_pr_band"] == 0.0
    assert cfg.pen["waist_pr_band"] == pytest.approx(2.0)
    assert cfg.lam_max["waist_pr_band"] == pytest.approx(100.0)
    assert phase2_cost_config().w_waist_pr_band == 1.0
    # constructing the object raises if any per-group dict is missing a key
    AugmentedLagrangian(cfg)


def test_al_update_grows_only_the_hot_group():
    al = AugmentedLagrangian(phase2_al_config(ema=0.0, decay=1.0))
    before = dict(al.lam)
    hot = {g: 0.0 for g in CONSTRAINT_GROUPS}
    hot["waist_pr_band"] = 1.5
    al.update(hot)
    assert al.lam["waist_pr_band"] > before["waist_pr_band"]
    for g in CONSTRAINT_GROUPS:
        if g == "waist_pr_band":
            continue
        # V=0: may leak via feasible_decay / target slack, but must not grow.
        assert al.lam[g] <= before[g], (g, before[g], al.lam[g])


# ========================================================== 9. resume =======
def test_resume_without_the_key_stays_at_lam_init_and_is_not_seeded():
    old = AugmentedLagrangian(ALConfig(
        groups=("torque", "velocity", "joint_box", "collision", "terminal"),
        rho=1.0,
    ))
    old.update({g: 0.4 for g in old.groups})
    state = old.state_dict()
    assert "waist_pr_band" not in state["lam"]
    assert state["lam"]["joint_box"] != 1.0

    fresh = AugmentedLagrangian(phase2_al_config())
    fresh.load_state_dict(state)
    assert fresh.lam["waist_pr_band"] == 1.0  # lam_init -- not silently dropped

    seeded = AugmentedLagrangian(phase2_al_config())
    migrated = migrate_al_state(state)
    seeded.load_state_dict(migrated)
    # must NOT copy joint_box λ onto the new price
    assert seeded.lam["waist_pr_band"] == 1.0
    assert seeded.lam["waist_pr_band"] != state["lam"]["joint_box"]
    assert "waist_pr_band" not in state["lam"]  # caller dict not mutated


# =========================================== 10. FusedCost w dict applies λ ==
def test_fusedcost_w_dict_applies_the_group_lambda():
    cost = _cost(w_waist_pr_band=1.0, waist_pr_band_rad=0.15)
    sp = _sp(0.1)
    q0, wp, v0, _, _ = _hold(cost, pitch=0.1, roll=0.0, dump_pitch=2.0)
    ctx = _strike(B, H, sp)
    one = cost.forward(q0, wp, v0, ctx, weights={"waist_pr_band": 1.0})
    seven = cost.forward(q0, wp, v0, ctx, weights={"waist_pr_band": 7.0})
    warning = cost.cfg.w_waist_pr_band * one.raw["waist_pr_band"]
    live = one.violations["waist_pr_band"]
    assert float(one.terms["waist_pr_band"].abs().max()) > 0.0
    assert torch.allclose(one.raw["waist_pr_band"], seven.raw["waist_pr_band"])
    assert torch.allclose(one.terms["waist_pr_band"], warning + 1.0 * live, atol=1e-12)
    assert torch.allclose(seven.terms["waist_pr_band"], warning + 7.0 * live, atol=1e-12)
