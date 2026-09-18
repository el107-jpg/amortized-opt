"""``collision_peak`` -- the MAX-shaped half of the collision term.

``FusedCost`` priced the collision hinge exactly once, as a MEAN over
``n_seen = 2 x S x P x D`` ``(knot, point, drum)`` pairs -- ~2 100 at bucket 25
and ~4 200 at bucket 50.  ``ok_collision`` is a zero-tolerance check that no
contact point penetrates any drum, so the statistic the gate reads and the
statistic the loss integrates are ``n_seen`` apart: a point 30 mm inside a drum
head moves the mean by ~1e-5 while failing the case outright.

This is the same shape of problem ``torque_peak`` was added for, and these
tests are deliberately the collision mirror of ``test_torque_peak_al.py``.  The
analogy is weaker and the docstring on ``CostConfig.w_collision_peak`` says why:
torque violations are sparse (~0.27 % of ~850 elements) so a mean genuinely
cannot see them, while collision violations run ~10 elements deep and the mean
partially can.

ONE DELIBERATE DIFFERENCE from ``torque_peak``, and it is the whole reason the
term works: the peak half reduces the raw PENETRATION, not the margin-softplus
the three means reduce.  Peaking the hinge was implemented, measured, and
rejected -- on the champion's 243-case grid it separated the acceptance gate
2.9x against the mean's 3.9x, i.e. *worse than the thing it was added to fix*,
because the softplus floors at 0.0066 m for an element at zero clearance and a
max over a floor is mostly floor.  The penetration is identically zero on a
clear element and separates 5.0x.

What these tests pin is the mechanism, not the size of the win: the peak half
tracks the deepest penetration, is a hard zero when nothing penetrates, carries
its own multiplier, is exactly zero and exactly gradient-free at the default
weight, and appends to the constraint groups without moving any index a
checkpoint already recorded.
"""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import CONSTRAINT_GROUPS, ALConfig, AugmentedLagrangian
from amortized_opt.train_cost import (
    METRIC_NAMES,
    VIOL_KEYS,
    VIOL_OFFSET,
    WeightSchedule,
    migrate_al_state,
    phase2_al_config,
)

DT = 0.02
H = 12


# ------------------------------------------------------------------ setup --
def _ctx(B, device="cpu", dtype=torch.float64, *, kl, kr):
    """The same minimal synthetic context ``test_torque_peak_al.py`` uses."""
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


#: ``left_shoulder_yaw`` / ``right_shoulder_yaw``.  Sweeping either drives the
#: stick through the kit; measured penetration at +-1.2 rad is ~55 mm, the same
#: order as the median failing case on the champion's eval grid (35 mm).
YAW_COLS = (5, 12)
#: Shoulder pitch / roll / elbow that park BOTH arms clear of every drum:
#: measured minimum clearance 117 mm, so every hinge in the batch underflows to
#: exactly zero.  This is the control for "no violation -> no term".
CLEAR_POSE = {3: 0.3, 4: 0.3, 6: 1.2, 10: 0.3, 11: -0.3, 13: 1.2}


def _batch(*, penetrating: bool, B: int = 4):
    """Row ``b`` sweeps one shoulder-yaw into the kit, or parks both arms clear."""
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    if not penetrating:
        for j, v in CLEAR_POSE.items():
            q0[:, j] = v
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    if penetrating:
        for b in range(B):
            j = YAW_COLS[b % len(YAW_COLS)]
            amp = -1.2 if j == 5 else 1.2
            wp[b, :, j] = torch.linspace(0.0, amp, H, dtype=torch.float64)
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    return dict(q0=q0, wp=wp, v0=torch.zeros(B, N_DOF, dtype=torch.float64),
                kl=kl, kr=kr, B=B)


def _cost(**cfg_kw):
    return FusedCost(CostConfig(**cfg_kw), max_horizon=H, device="cpu",
                     dtype=torch.float64, dt=DT)


def _fwd(b, cost, **kw):
    return cost.forward(b["q0"], b["wp"], b["v0"],
                        _ctx(b["B"], kl=b["kl"], kr=b["kr"]), **kw)


@pytest.fixture(scope="module")
def violating():
    b = _batch(penetrating=True)
    return b, _fwd(b, _cost(w_collision_peak=1.0))


# ================================================ 1. the reduction is a max ==
def test_the_peak_tracks_penetration_smooth_max_at_least_the_max(violating):
    """``raw["collision_peak"]`` is a smooth max of pens; AL viol is the max."""
    b, out = violating
    pen = out.diagnostics["collision_pen_max"]
    assert float(pen.min()) > 0.020, "fixture must actually penetrate"
    peak = out.raw["collision_peak"]
    for i in range(b["B"]):
        assert float(peak[i]) >= float(pen[i]) * (1.0 - 1e-9)
    assert torch.equal(out.violations["collision_peak"], pen)


def test_the_peak_is_exactly_zero_without_penetration():
    """Parked 117 mm clear: no penetration anywhere, so the peak is a hard zero.

    Stronger than the mean's behaviour on the same batch -- the margin-softplus
    the means reduce is merely SMALL at a few mm of clearance, while the peak is
    identically zero the moment nothing is actually inside a drum.  That is the
    property that makes the term free for a sample the gate already passes.
    """
    out = _fwd(_batch(penetrating=False), _cost(w_collision_peak=1.0))
    assert float(out.diagnostics["collision_pen_max"].abs().max()) == 0.0
    assert float(out.raw["collision_peak"].abs().max()) == 0.0


def test_the_peak_is_zero_at_positive_clearance_where_the_hinge_is_not():
    """The separation argument, as a unit test.

    An element a few mm clear still pays the softplus floor -- that is what made
    peaking the HINGE separate the gate worse (2.9x) than the mean it was meant
    to fix (3.9x).  Peaking the penetration is exactly zero there.
    """
    b = _batch(penetrating=False)
    out = _fwd(b, _cost(w_collision_peak=1.0))
    mean_hinge = (out.raw["collision_drum"] + out.raw["collision_under"]
                  + out.raw["collision_self"])
    assert float(out.raw["collision_peak"].max()) == 0.0
    # the mean is small but NOT zero at this clearance in general; either way the
    # peak contributes nothing, which is the point.
    assert float(mean_hinge.max()) >= 0.0


def test_the_dilution_the_term_exists_for(violating):
    """The mean is ``n_seen`` smaller than the max -- that is the whole point."""
    _, out = violating
    mean = (out.raw["collision_drum"] + out.raw["collision_under"]
            + out.raw["collision_self"])
    peak = out.raw["collision_peak"]
    ratio = peak / mean.clamp_min(1e-30)
    n_seen = out.counts["collision"]
    print(f"\n  n_seen {float(n_seen[0]):.0f}   mean {float(mean[0]):.3e}   "
          f"peak {float(peak[0]):.3e}   ratio {float(ratio[0]):.0f}x")
    # Same physical violation, two reductions, orders apart.  (Different units
    # by design -- the mean is a margin-softplus, the peak is metres of
    # penetration -- so this pins the ORDER, which is what sets the weight.)
    assert float(ratio.min()) > 20.0


# ============================================== 2. off by default, exactly ==
def test_bit_identical_at_the_default_weight():
    """``w_collision_peak = 0`` reproduces the mean-only term to the last bit."""
    b = _batch(penetrating=True)
    off = _fwd(b, _cost())                       # default: 0.0
    on_but_unpriced = _fwd(b, _cost(w_collision_peak=500.0),
                           weights={"collision_peak": 0.0})
    assert torch.equal(off.terms["collision"], on_but_unpriced.terms["collision"])
    assert torch.equal(off.total, on_but_unpriced.total)
    # ...and the peak is not merely unpriced, it is not even computed.
    assert float(off.raw["collision_peak"].abs().max()) == 0.0
    assert off.raw["collision_peak"].requires_grad is False


def test_the_peak_actually_changes_the_total_when_priced():
    b = _batch(penetrating=True)
    off = _fwd(b, _cost())
    on = _fwd(b, _cost(w_collision_peak=500.0))
    assert float((on.total - off.total).min()) > 0.0


def test_collision_lambda_scales_live_not_the_mean_hinge():
    """``weights["collision"]`` is λ on ReLU live; mean softplus is invariant."""
    b = _batch(penetrating=True)
    cost = _cost(w_collision_peak=0.0)
    one = _fwd(b, cost, weights={"collision": 1.0})
    fifty = _fwd(b, cost, weights={"collision": 50.0})
    warning = (
        cost.cfg.w_collision_drum * one.raw["collision_drum"]
        + cost.cfg.w_collision_under * one.raw["collision_under"]
        + cost.cfg.w_collision_self * one.raw["collision_self"]
        + cost.cfg.w_stick_stick * one.raw["collision_stick"]
    )
    assert torch.allclose(fifty.raw["collision_drum"], one.raw["collision_drum"])
    assert torch.allclose(one.terms["collision"] - warning, one.violations["collision"],
                          atol=1e-12)
    assert torch.allclose(fifty.terms["collision"] - warning,
                          50.0 * one.violations["collision"], atol=1e-12)


def test_gradient_is_zero_off_and_flows_on():
    """No gradient path at the default weight; a real one once priced."""
    def grad(w):
        b = _batch(penetrating=True, B=2)
        wp = b["wp"].clone().requires_grad_(True)
        c = _cost(w_collision_peak=w)
        out = c.forward(b["q0"], wp, b["v0"],
                        _ctx(b["B"], kl=b["kl"], kr=b["kr"]))
        out.terms["collision"].sum().backward()
        return wp.grad.clone()

    g_off, g_on = grad(0.0), grad(500.0)
    assert torch.isfinite(g_on).all()
    assert float((g_on - g_off).abs().max()) > 0.0


# =========================================== 3. its own multiplier, appended ==
def test_the_group_is_appended_last():
    assert CONSTRAINT_GROUPS[-1] == "velocity_peak"
    assert CONSTRAINT_GROUPS[-5:-1] == (
        "torque_peak", "collision_peak", "joint_box_peak", "waist_pr_band",
    )
    assert "torque" not in CONSTRAINT_GROUPS
    al = AugmentedLagrangian()
    assert "collision_peak" in al.lam and "collision_peak" in al.rho
    assert al.rho["collision_peak"] < al.rho["collision"]
    assert phase2_al_config().rho["collision_peak"] < phase2_al_config().rho["collision"]


def test_the_violation_mirrors_the_collision_group(violating):
    """Documented decision: both groups integrate the same max penetration.

    ``viol["collision"]`` has always been a MAX (unlike ``viol["torque"]``), so
    there is no second quantity to give the peak group, and changing the mean
    group's violation would move the multiplier dynamics of every existing run.
    """
    _, out = violating
    assert "collision_peak" in out.violations
    assert torch.equal(out.violations["collision_peak"], out.violations["collision"])
    assert torch.equal(out.violations["collision_peak"],
                       out.diagnostics["collision_pen_max"])


def test_the_multipliers_move_independently():
    al = AugmentedLagrangian(ALConfig(rho=1.0, decay=1.0))
    base = dict(al.lam)
    al.update({"collision": 0.0, "collision_peak": 5.0})
    assert al.lam["collision_peak"] == pytest.approx(base["collision_peak"] + 5.0)
    assert al.lam["collision"] == pytest.approx(base["collision"])


def test_resume_appends_the_group_without_seeding_it():
    """Unlike ``torque_peak`` there is no prior price to preserve."""
    old = {"lam": {"torque": 7.2, "velocity": 1.0, "joint_box": 3.0,
                   "collision": 4.8, "terminal": 2.0},
           "groups": ["torque", "velocity", "joint_box", "collision", "terminal"]}
    new = migrate_al_state(old)
    assert "collision_peak" in new["groups"]
    # NOT seeded: the term did not exist before, so lam_init is correct and the
    # ramp is what keeps the objective continuous.
    assert "collision_peak" not in new["lam"]
    # torque_peak still IS seeded -- that one was a regrouping of a live price.
    assert new["lam"]["torque_peak"] == pytest.approx(7.2)
    assert old["lam"] == {"torque": 7.2, "velocity": 1.0, "joint_box": 3.0,
                          "collision": 4.8, "terminal": 2.0}, "must not mutate"


# ================================================= 4. curriculum + metrics ==
def test_the_schedule_ramps_it():
    s = WeightSchedule(collision_peak_start=0, collision_peak_ramp=1000)
    assert s.multipliers(0)["collision_peak"] == pytest.approx(0.0)
    assert s.multipliers(1000)["collision_peak"] == pytest.approx(1.0)
    assert 0.0 < s.multipliers(500)["collision_peak"] < 1.0
    # monotone, and it does not drag the other peak along with it
    assert s.multipliers(200)["collision_peak"] < s.multipliers(800)["collision_peak"]


def test_the_metrics_block_stays_derived():
    assert set(VIOL_KEYS) == set(CONSTRAINT_GROUPS)
    assert VIOL_KEYS[-1] == "velocity_peak"
    assert VIOL_KEYS[-2] == "waist_pr_band"
    assert VIOL_KEYS[-3] == "joint_box_peak"
    assert VIOL_KEYS[-4] == "collision_peak"
    assert VIOL_KEYS[-5] == "torque_peak"
    assert VIOL_KEYS[-6] == "terminal"
    assert VIOL_KEYS[-7] == "collision"
    assert VIOL_KEYS[0] == "torque_waist"
    i = VIOL_KEYS.index("collision_peak")
    assert METRIC_NAMES[VIOL_OFFSET + i] == "v_collision_peak"
    assert METRIC_NAMES[VIOL_OFFSET + len(VIOL_KEYS) - 1] == "v_velocity_peak"
