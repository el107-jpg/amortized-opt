"""Section 7d leg 2: the strike-masked ``tau^2`` effort term.

``CostConfig.w_effort`` prices the mean squared RNEA torque over the trajectory,
with the weight faded to zero over the last ``effort_fade_wp`` waypoints before
each hand's own hit.  The fade is the whole point: a strike IS kinetic energy,
so an unmasked energy penalty is in direct opposition to
``w_terminal_velocity``.

What is pinned here: the fade window (value AND zero gradient inside it), the
per-hand / waist column routing, the element mask, agreement with a hand-rolled
``tau^2`` mean on a fixed batch, and the default-off guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext

WAIST = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("waist_")]
LEFT = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("left_")]
RIGHT = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("right_")]

H = 16
FADE = 5


@pytest.fixture(scope="module")
def cost():
    return FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)


def _ctx(kl, kr, active=(1.0, 1.0), B=1, dtype=torch.float64):
    dev = "cpu"
    ones = torch.ones(B, 2, device=dev, dtype=dtype)
    return StrikeContext(
        hit_index=torch.tensor([[kl, kr]] * B, device=dev, dtype=torch.long),
        active=torch.tensor([list(active)] * B, device=dev, dtype=dtype),
        target_drum=torch.zeros(B, 2, device=dev, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, device=dev, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, device=dev, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, device=dev, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, device=dev, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, device=dev, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, device=dev, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, device=dev, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, device=dev, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, device=dev, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, device=dev, dtype=dtype),
        elev_required=ones,
    )


def _wiggly(B=1, seed=0, dtype=torch.float64):
    """A trajectory with real accelerations, so the torques are not gravity-only."""
    g = torch.Generator().manual_seed(seed)
    q0 = 0.1 * torch.randn(B, N_DOF, generator=g, dtype=dtype)
    v0 = 0.2 * torch.randn(B, N_DOF, generator=g, dtype=dtype)
    wp = q0.unsqueeze(1) + 0.15 * torch.randn(B, H, N_DOF, generator=g, dtype=dtype)
    return q0, wp, v0


# ============================================================= default off ===
def test_effort_defaults_to_off_and_is_free():
    c = CostConfig()
    assert c.w_effort == 0.0
    assert c.effort_fade_wp == 5


def test_effort_contributes_exactly_zero_by_default(cost):
    q0, wp, v0 = _wiggly()
    out = cost.forward(q0, wp, v0, _ctx(H, H))
    assert out.raw["effort"].item() > 0.0            # measured...
    assert out.terms["effort"].item() == 0.0         # ...but not priced
    on = cost.with_config(w_effort=1e-3).forward(q0, wp, v0, _ctx(H, H))
    assert on.terms["effort"].item() == pytest.approx(1e-3 * on.raw["effort"].item())
    assert on.total.item() > out.total.item()


def test_effort_validates():
    with pytest.raises(ValueError):
        CostConfig(w_effort=-1.0).validate()
    with pytest.raises(ValueError):
        CostConfig(effort_fade_wp=-1).validate()


# ============================================================== fade shape ===
def _fade(cost, ctx, n=None, dtype=torch.float64):
    n = (cost.max_horizon + 1) if n is None else n
    return cost._effort_fade(ctx, n, dtype)


def test_fade_is_hard_zero_over_the_window_then_ramps_back(cost):
    """d = knots to go.  0 for d in [0, W], (d-W)/W for d in [W, 2W], then 1."""
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    f = _fade(c, _ctx(12, 12))[0]                    # (n_knots, 17)
    for d in range(0, FADE + 1):                     # the hard-zero core
        for col in (LEFT[0], RIGHT[0], WAIST[0]):
            assert f[12 - d, col].item() == 0.0, (d, col)
    for d in range(FADE, 2 * FADE + 1):              # the ramp back
        expect = (d - FADE) / FADE
        assert f[12 - d, LEFT[0]].item() == pytest.approx(expect), d
    assert f[12 - 2 * FADE - 1, LEFT[0]].item() == pytest.approx(1.0)


def test_fade_window_length_is_the_config_field(cost):
    for W in (2, 5, 7):
        c = cost.with_config(w_effort=1.0, effort_fade_wp=W)
        f = _fade(c, _ctx(15, 15))[0]
        col = LEFT[0]
        zero = [i for i in range(16) if f[i, col].item() == 0.0]
        assert zero == list(range(15 - W, 16)), (W, zero)
        below = [i for i in range(16) if f[i, col].item() < 1.0 - 1e-12]
        assert below == list(range(max(15 - 2 * W + 1, 0), 16)), (W, below)


def test_fade_is_released_once_the_hit_is_past(cost):
    """A hand that has already struck must stop protecting the WAIST.

    Without the ``d < 0`` branch a left-hand hit at knot 4 would hold the waist
    factor at 0 for the entire remaining horizon, which silently turns the term
    off on every two-hit context.
    """
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    f = _fade(c, _ctx(4, 15))[0]
    assert f[4, WAIST[0]].item() == 0.0              # left's own strike
    assert f[5, WAIST[0]].item() == pytest.approx(1.0)   # released immediately
    assert f[15, WAIST[0]].item() == 0.0             # right's strike now bites


def test_fade_zero_disables_the_mask(cost):
    c = cost.with_config(w_effort=1.0, effort_fade_wp=0)
    f = _fade(c, _ctx(10, 10))
    assert float(f.min()) == 1.0 and float(f.max()) == 1.0


def test_fade_routes_arms_to_their_own_hand_and_waist_to_the_nearer_hit(cost):
    """Left hits at 6, right at 14: the left arm is protected around 6 only, the
    right arm around 14 only, and the waist around BOTH."""
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    f = _fade(c, _ctx(6, 16))[0]
    assert f[6, LEFT[0]].item() == 0.0
    assert f[6, RIGHT[0]].item() == pytest.approx(1.0)     # 10 knots from its hit
    assert f[16, RIGHT[0]].item() == 0.0
    assert f[16, LEFT[0]].item() == pytest.approx(1.0)     # its hit is long past
    for col in WAIST:
        assert f[6, col].item() == 0.0                     # left's strike
        assert f[16, col].item() == 0.0                    # right's strike
        # between the two: left is released, right is on its ramp ((7-5)/5)
        assert f[9, col].item() == pytest.approx(0.4)
        assert f[13, col].item() == pytest.approx(0.0)     # inside right's core


def test_an_inactive_hand_gets_no_fade(cost):
    """No strike to protect -> the arm is priced everywhere, waist follows the
    only active hand."""
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    f = _fade(c, _ctx(6, 16, active=(0.0, 1.0)))[0]
    assert f[:, LEFT[0]].min().item() == pytest.approx(1.0)
    assert f[16, RIGHT[0]].item() == 0.0
    assert f[16, WAIST[0]].item() == 0.0                   # right's hit still bites
    assert f[6, WAIST[0]].item() == pytest.approx(1.0)     # left's does not


# ============================================================ correctness ====
def test_effort_matches_a_hand_rolled_tau_squared_mean(cost):
    """The reference is computed from the SCORER-convention torques directly."""
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    q0, wp, v0 = _wiggly(B=4, seed=3)
    ctx = _ctx(11, 9, B=4)
    out = c.forward(q0, wp, v0, ctx)

    # rebuild tau exactly as _torque does, then the weighted mean by hand
    sp = c.spline.build(q0, wp, v0, c._lengths_from_context(ctx))
    q, qd, qdd = sp.knot_finite_differences(v0)
    B, n, D = q.shape
    tau = c.rnea.rnea(q.reshape(B * n, D), qd.reshape(B * n, D),
                      qdd.reshape(B * n, D)).reshape(B, n, D)
    idx = torch.arange(n).view(1, -1, 1)
    mask = (idx <= sp.lengths.unsqueeze(1)).to(q.dtype)

    def _f1(k, i):
        d = k - i
        return 1.0 if d < 0 else float(np.clip((d - FADE) / FADE, 0.0, 1.0))

    fade = np.ones((B, n, D))
    kk = [11, 9]
    for b in range(B):
        for j in range(D):
            name = G1_UPPER_BODY_JOINTS[j]
            if name.startswith("left_"):
                ks = [kk[0]]
            elif name.startswith("right_"):
                ks = [kk[1]]
            else:
                ks = kk
            for i in range(n):
                fade[b, i, j] = min(_f1(k, i) for k in ks)
    ft = torch.as_tensor(fade, dtype=q.dtype)
    m = mask * ft
    ref = ((tau * tau * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0))
    assert torch.allclose(out.raw["effort"], ref, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out.counts["effort"], m.sum(dim=(1, 2)), atol=1e-12)


def test_effort_respects_the_element_mask(cost):
    """Garbage past the horizon must not reach the term."""
    c = cost.with_config(w_effort=1.0, effort_fade_wp=0)
    q0, wp, v0 = _wiggly(seed=5)
    ctx = _ctx(6, 6)
    ref = c.forward(q0, wp, v0, ctx).raw["effort"].item()
    wp2 = wp.clone()
    wp2[:, 7:] += 3.0                                  # far outside the horizon
    got = c.forward(q0, wp2, v0, ctx).raw["effort"].item()
    assert got == pytest.approx(ref, rel=1e-12)


# =============================================================== gradient ====
def test_no_gradient_inside_the_fade_window(cost):
    """The waypoints the strike is made of must be untouched by the term.

    Waypoint ``j`` of the plan is knot ``j + 1``.  With a hit at knot ``k`` and
    a window ``W`` the knots ``k-W .. k`` carry a hard-zero weight, so the last
    ``W`` waypoints (indices ``k-W .. k-1``) are the STRIKE and must receive no
    gradient.  ``tau`` at a knot reads its neighbours through the finite
    differences, so a knot one OUTSIDE the window still couples inward; what is
    asserted here is the strictly interior part, which is what carries the
    approach.
    """
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    q0, wp, v0 = _wiggly(seed=7)
    wp = wp.clone().requires_grad_(True)
    k = 12
    c.forward(q0, wp, v0, _ctx(k, k)).raw["effort"].sum().backward()
    g = wp.grad[0]                                     # (H, 17)
    for j in range(k - FADE + 1, k):                   # knots k-W+2 .. k
        assert g[j, LEFT].abs().max().item() == 0.0, j
        assert g[j, RIGHT].abs().max().item() == 0.0, j
        assert g[j, WAIST].abs().max().item() == 0.0, j
    # ...and well before the window the term is live
    assert g[: k - 2 * FADE, :].abs().max().item() > 0.0


def test_gradient_routes_per_hand(cost):
    """With the right hand INACTIVE its arm keeps full price inside the window
    that silences the left arm -- the routing, seen through autograd."""
    c = cost.with_config(w_effort=1.0, effort_fade_wp=FADE)
    q0, wp, v0 = _wiggly(seed=11)
    wp = wp.clone().requires_grad_(True)
    k = 14
    c.forward(q0, wp, v0, _ctx(k, k, active=(1.0, 0.0))).raw["effort"].sum().backward()
    g = wp.grad[0]
    j = k - 2                                          # deep inside left's window
    assert g[j, LEFT].abs().max().item() == 0.0
    assert g[j, RIGHT].abs().max().item() > 0.0
    # NOTE the waist waypoints are NOT gradient-free here, and correctly so: the
    # fade zeroes the waist's own torque COLUMN, but the waist joint is upstream
    # of both arms in the RNEA chain, so it still receives gradient through the
    # right arm's (unfaded) torques.  The column-level routing is pinned in
    # ``test_fade_routes_arms_to_their_own_hand_and_waist_to_the_nearer_hit``.


def test_effort_does_not_disturb_the_other_terms(cost):
    """It is additive: every other term is bit-identical with it on."""
    q0, wp, v0 = _wiggly(seed=13)
    ctx = _ctx(10, 10)
    a = cost.forward(q0, wp, v0, ctx)
    b = cost.with_config(w_effort=5e-3).forward(q0, wp, v0, ctx)
    for k in a.terms:
        if k == "effort":
            continue
        assert torch.equal(a.terms[k], b.terms[k]), k
    assert b.total.item() == pytest.approx(a.total.item() + b.terms["effort"].item())


def test_effort_carries_no_al_multiplier(cost):
    """``weights["torque"]`` must not scale it; ``weights["effort"]`` must."""
    q0, wp, v0 = _wiggly(seed=17)
    ctx = _ctx(10, 10)
    c = cost.with_config(w_effort=1e-3)
    base = c.forward(q0, wp, v0, ctx)
    hot_tau = c.forward(q0, wp, v0, ctx, weights={"torque": 9.0})
    assert torch.equal(hot_tau.terms["effort"], base.terms["effort"])
    ramp = c.forward(q0, wp, v0, ctx, weights={"effort": 0.25})
    assert torch.allclose(ramp.terms["effort"], 0.25 * base.terms["effort"],
                          atol=1e-15)


def test_effort_diagnostics_are_unfaded_means(cost):
    """``torque_sq_mean`` / ``torque_abs_mean`` cover the WHOLE horizon, so the
    report can say what happened to the energy including the strike."""
    q0, wp, v0 = _wiggly(seed=19)
    out = cost.forward(q0, wp, v0, _ctx(10, 10))
    no_fade = cost.with_config(effort_fade_wp=0).forward(q0, wp, v0, _ctx(10, 10))
    assert out.diagnostics["torque_sq_mean"].item() == pytest.approx(
        no_fade.raw["effort"].item(), rel=1e-12
    )
    assert out.diagnostics["torque_abs_mean"].item() > 0.0
    # the faded mean drops the strike, where the torques are largest
    assert out.raw["effort"].item() < out.diagnostics["torque_sq_mean"].item()
