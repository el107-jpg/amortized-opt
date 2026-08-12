"""``torque_peak`` as its own augmented-Lagrangian group.

``FusedCost`` prices the torque hinge twice -- a MEAN over ~850 ``(knot,
joint)`` elements and a smooth MAX over the same elements -- and until now both
halves were multiplied by ``lam_torque``, a multiplier integrated from the MEAN
raw violation.  Raising the peak price therefore also amplified a multiplier
that has no idea the peak exists, which is what destabilized the two
continuations recorded in the README ("What is left of the torque gap").

These tests pin the split: the peak half carries ``lam_torque_peak``, driven by
the per-sample MAX excess over the HARD effort limits; the two multipliers move
independently; the term is still bit-identical at ``w_torque_peak = 0``; and a
resume from a checkpoint written before the split keeps the objective
continuous by seeding the new multiplier from the old one.
"""

from __future__ import annotations

import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import CONSTRAINT_GROUPS, ALConfig, AugmentedLagrangian
from amortized_opt.train_cost import (
    METRIC_NAMES,
    VIOL_KEYS,
    VIOL_OFFSET,
    migrate_al_state,
    phase2_al_config,
)

DT = 0.02


# ------------------------------------------------------------------ setup --
def _ctx(B, device="cpu", dtype=torch.float64, *, kl, kr):
    """The same minimal synthetic context ``tests/test_torque_peak.py`` uses."""
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


#: The 5 N*m wrist columns (``wrist_pitch`` / ``wrist_yaw``, left then right).
WRIST_COLS = (8, 9, 15, 16)
#: Displacement of ONE knot that puts exactly ONE (knot, joint) element over the
#: hard effort limit -- the shape 0.27 % of the recorded phase-2 failures have.
#: A single displaced knot makes ``qdd`` at that knot twice its neighbours'
#: (``(q_{k+1} - 2 q_k + q_{k-1}) / dt^2``), so there is a window in which only
#: the centre element trips; 0.1 rad at a 20 ms tick is inside it.
ONE_ELEMENT_AMP = 0.10


def _batch(*, amp: float, H: int = 12, B: int = 4):
    """A hand-made batch: row ``b`` displaces ONE knot of ONE wrist joint.

    ``amp = 0`` is a motionless hold (gravity only, comfortably feasible).  At
    ``ONE_ELEMENT_AMP`` every row is over the hard limit on exactly one of its
    221 valid ``(knot, joint)`` elements, which makes the mean raw violation
    exactly ``max / n_valid`` -- the 1/221 dilution the peak group exists for.
    """
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    if amp:
        for b in range(B):
            j = WRIST_COLS[b % len(WRIST_COLS)]
            k = 3 + b               # ... at a different knot per row
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


@pytest.fixture(scope="module")
def violating():
    b = _batch(amp=ONE_ELEMENT_AMP)
    cost = _cost(w_torque_peak=1.0)
    return b, cost, _fwd(b, cost)


# ============================== 1. the violation the new group is driven by ==
def test_violation_is_exactly_the_max_excess_diagnostic(violating):
    b, cost, out = violating
    assert "torque_peak" in out.violations
    v = out.violations["torque_peak"]
    assert v.shape == (b["B"],) and v.dtype == out.violations["torque"].dtype
    assert not v.requires_grad                      # detached, like every other
    # the contract: it IS ``diag["torque_excess_max"]``
    assert torch.equal(v, out.diagnostics["torque_excess_max"])
    assert bool((v > 0).all()), "the fixture batch must actually violate"


def test_violation_matches_an_independent_recomputation(violating):
    """Recompute ``max(|tau| - hard_limit, 0)`` from the spline, outside the cost."""
    b, cost, out = violating
    sp = cost.spline.build(b["q0"], b["wp"], b["v0"],
                           cost._lengths_from_context(_ctx(b["B"], kl=b["kl"], kr=b["kr"])))
    q, qd, qdd = sp.knot_finite_differences(b["v0"])
    n = q.shape[1]
    idx = torch.arange(sp.max_segments + 1, dtype=torch.long).view(1, -1, 1)
    mask = (idx <= sp.lengths.unsqueeze(1)).to(q.dtype)
    tau = cost.rnea.rnea(q.reshape(-1, N_DOF), qd.reshape(-1, N_DOF),
                         qdd.reshape(-1, N_DOF)).reshape(b["B"], n, N_DOF)
    over = (tau.abs() - cost.tau_hard.view(1, 1, -1)).clamp_min(0.0) * mask
    expect = over.amax(dim=(1, 2))
    got = out.violations["torque_peak"]
    print("analytic max excess vs violations['torque_peak']: "
          f"{[round(float(x), 6) for x in expect]} vs "
          f"{[round(float(x), 6) for x in got]}")
    assert torch.allclose(got, expect, atol=0.0, rtol=0.0)


def test_violation_is_zero_on_a_feasible_batch():
    b = _batch(amp=0.0)                # a motionless hold: gravity only
    out = _fwd(b, _cost(w_torque_peak=1.0))
    assert float(out.diagnostics["torque_max"].max()) > 0.0     # RNEA did run
    assert float(out.violations["torque_peak"].abs().max()) == 0.0
    assert float(out.violations["torque"].abs().max()) == 0.0
    # ... and the hinge, which is a softplus, is NOT zero -- that is the whole
    # reason the multipliers are driven by the violation and not by the penalty.
    assert float(out.raw["torque_peak"].min()) > 0.0


def test_the_two_violations_are_orders_of_magnitude_apart(violating):
    """Why the peak cannot share a multiplier calibrated on the mean."""
    _b, _cost_, out = violating
    mean = out.violations["torque"]
    peak = out.violations["torque_peak"]
    ratio = (peak / mean.clamp_min(1e-12))
    print("per-sample  mean excess / max excess / ratio:")
    for i in range(mean.numel()):
        print(f"  {float(mean[i]):12.6f}  {float(peak[i]):10.4f}  "
              f"{float(ratio[i]):9.1f}x")
    assert float(ratio.min()) > 50.0


# =========================================== 2. the multipliers are separate ==
def test_the_group_exists_and_is_last():
    assert CONSTRAINT_GROUPS[-1] == "torque_peak"
    assert CONSTRAINT_GROUPS[:5] == ("torque", "velocity", "joint_box",
                                     "collision", "terminal")
    al = AugmentedLagrangian()
    assert "torque_peak" in al.lam and "torque_peak" in al.rho
    assert al.rho["torque_peak"] < al.rho["torque"]


def test_pure_al_groups_are_independent():
    al = AugmentedLagrangian(ALConfig(rho=1.0, decay=1.0))
    base = dict(al.lam)
    al.update({"torque": 0.0, "torque_peak": 5.0})
    assert al.lam["torque_peak"] == pytest.approx(base["torque_peak"] + 5.0)
    assert al.lam["torque"] == pytest.approx(base["torque"])
    for g in ("velocity", "joint_box", "collision", "terminal"):
        assert al.lam[g] == base[g]                 # untouched groups: absent

    al2 = AugmentedLagrangian(ALConfig(rho=1.0, decay=1.0))
    al2.update({"torque": 5.0, "torque_peak": 0.0})
    assert al2.lam["torque"] == pytest.approx(base["torque"] + 5.0)
    assert al2.lam["torque_peak"] == pytest.approx(base["torque_peak"])


def test_peak_lambda_rises_while_the_mean_lambda_does_not(violating):
    """THE point of the change, on a real ``FusedCost`` batch.

    A single over-limit element makes BOTH violations strictly positive, so the
    two are separated the way a converging run separates them: the mean is
    inside its tolerance (``target`` above the batch's actual mean violation --
    0.27 % of the elements over is what an accepted-but-marginal case looks
    like) while the max is not, ``target = 0`` being the acceptance gate itself.
    """
    _b, _cost_, out = violating
    viol = {k: out.violations[k] for k in out.violations}
    mean_v = float(viol["torque"].mean())
    peak_v = float(viol["torque_peak"].mean())
    # ``decay`` and ``ema`` are separate relaxation machinery; switch them off
    # so the assertion is about the SIGN of the violation-driven term alone.
    al = AugmentedLagrangian(phase2_al_config(
        target={g: (2.0 * mean_v if g == "torque" else 0.0)
                for g in CONSTRAINT_GROUPS},
        ema=0.0, decay=1.0,
    ))
    lam_t, lam_p = [al.lam["torque"]], [al.lam["torque_peak"]]
    for _ in range(20):
        al.update(viol)
        lam_t.append(al.lam["torque"])
        lam_p.append(al.lam["torque_peak"])
    print(f"mean violation {mean_v:.3e} (target {2 * mean_v:.3e}), "
          f"peak violation {peak_v:.4f} (target 0)")
    print(f"  lam_torque      {lam_t[0]:.4f} -> {lam_t[-1]:.4f}")
    print(f"  lam_torque_peak {lam_p[0]:.4f} -> {lam_p[-1]:.4f}")
    assert all(b > a for a, b in zip(lam_p, lam_p[1:])), lam_p
    assert all(b <= a for a, b in zip(lam_t, lam_t[1:])), lam_t
    assert lam_p[-1] > lam_p[0] and lam_t[-1] < lam_t[0]


def test_al_weights_reach_the_two_halves_separately(violating):
    b, cost, _out = violating
    al = AugmentedLagrangian(ALConfig(lam_init={"torque": 2.0, "torque_peak": 7.0,
                                                "velocity": 1.0, "joint_box": 1.0,
                                                "collision": 1.0, "terminal": 1.0}))
    out = _fwd(b, cost, weights=al.weights())
    expect = (2.0 * cost.cfg.w_torque * out.raw["torque"]
              + 7.0 * cost.cfg.w_torque_peak * out.raw["torque_peak"])
    assert torch.equal(out.terms["torque"], expect)


# ================================================== 3. the term, term by term ==
def test_the_two_halves_are_priced_independently(violating):
    b, cost, _o = violating
    ref = _fwd(b, cost, weights={"torque": 1.0, "torque_peak": 1.0})
    mean_half = cost.cfg.w_torque * ref.raw["torque"]
    peak_half = cost.cfg.w_torque_peak * ref.raw["torque_peak"]
    assert torch.equal(ref.terms["torque"], mean_half + peak_half)

    # moving one multiplier leaves the OTHER half bit-identical
    for wt, wp_ in ((5.0, 1.0), (1.0, 9.0), (0.0, 4.0), (3.0, 0.0)):
        out = _fwd(b, cost, weights={"torque": wt, "torque_peak": wp_})
        assert torch.equal(out.terms["torque"], wt * mean_half + wp_ * peak_half)
    # ... and the peak knob alone cannot move the mean half: at w_torque = 0 the
    # term is a pure multiple of the peak.
    only_peak = _fwd(b, cost, weights={"torque": 0.0, "torque_peak": 2.0})
    assert torch.equal(only_peak.terms["torque"], 2.0 * peak_half)


def test_zero_peak_weight_is_bit_identical_to_the_mean_only_term():
    b = _batch(amp=ONE_ELEMENT_AMP)
    cost = _cost()                      # w_torque_peak = 0.0, the pinned default
    assert cost.cfg.w_torque_peak == 0.0
    out = _fwd(b, cost)
    assert torch.equal(out.terms["torque"], cost.cfg.w_torque * out.raw["torque"])
    # ... and stays so however hard the peak multiplier is turned up
    for wp_ in (0.0, 1.0, 1e3):
        hot = _fwd(b, cost, weights={"torque_peak": wp_})
        assert torch.equal(hot.terms["torque"], out.terms["torque"])
    # the group multiplier still scales the whole (mean-only) term
    doubled = _fwd(b, cost, weights={"torque": 2.0})
    assert torch.equal(doubled.terms["torque"], 2.0 * out.terms["torque"])


# ============================================================== 4. plumbing ==
def test_metric_names_line_up_with_the_key_tuples():
    assert METRIC_NAMES[VIOL_OFFSET:VIOL_OFFSET + len(VIOL_KEYS)] == tuple(
        f"v_{k}" for k in VIOL_KEYS
    )
    assert METRIC_NAMES[VIOL_OFFSET] == "v_terminal"
    assert METRIC_NAMES[VIOL_OFFSET + len(VIOL_KEYS) - 1] == "v_torque_peak"
    # 5 losses + len(TERM_KEYS).  The literal is here to catch an ACCIDENTAL
    # shift; a deliberate append to TERM_KEYS updates it (7 terms -> 12 before
    # section 7d added ``effort``, 8 -> 13 after).
    from amortized_opt.train_cost import LOSS_KEYS, TERM_KEYS

    assert VIOL_OFFSET == len(LOSS_KEYS) + len(TERM_KEYS) == 14
    assert set(VIOL_KEYS) == set(CONSTRAINT_GROUPS)
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))


def test_cost_tuple_reports_the_new_violation(violating):
    from amortized_opt.train_cost import cost_tuple

    b, cost, out = violating
    ctx = _ctx(b["B"], kl=b["kl"], kr=b["kr"])
    _tot, _terms, viol, _diag = cost_tuple(
        cost, b["q0"], b["wp"], b["v0"], ctx, None, None
    )
    assert viol.numel() == len(VIOL_KEYS)
    # ``cost_tuple`` casts to float32 for the metrics tensor, hence rel=1e-6.
    i = VIOL_KEYS.index("torque_peak")
    assert float(viol[i]) == pytest.approx(
        float(out.violations["torque_peak"].mean()), rel=1e-6
    )


def test_resume_without_the_group_seeds_it_from_lam_torque():
    """An old checkpoint's AL state must not silently drop the peak price."""
    old = AugmentedLagrangian(ALConfig(groups=CONSTRAINT_GROUPS[:5]))
    old.update({g: 0.4 for g in old.groups})
    state = old.state_dict()
    assert "torque_peak" not in state["lam"]

    fresh = AugmentedLagrangian(phase2_al_config())
    fresh.load_state_dict(state)
    assert fresh.lam["torque_peak"] == 1.0        # lam_init -- the silent drop

    seeded = AugmentedLagrangian(phase2_al_config())
    seeded.load_state_dict(migrate_al_state(state))
    assert seeded.lam["torque_peak"] == seeded.lam["torque"] == state["lam"]["torque"]
    assert seeded.lam["torque"] != 1.0            # the migration is doing work
    # every other group is carried over untouched
    for g in CONSTRAINT_GROUPS[:5]:
        assert seeded.lam[g] == state["lam"][g]
    # ... and the migrated state does not mutate the caller's dict
    assert "torque_peak" not in state["lam"]


def test_resume_with_the_group_is_left_alone():
    al = AugmentedLagrangian(phase2_al_config())
    al.update({g: 0.2 for g in al.groups})
    state = al.state_dict()
    assert migrate_al_state(state) is state
    assert migrate_al_state(None) is None
    assert migrate_al_state({}) == {}


def test_train_phase2_resume_path_is_wired_to_the_migration():
    """The seeding is on the real resume path, not only in the helper."""
    import pathlib

    import amortized_opt.train_cost as T

    # the file, not ``inspect.getsource``: line numbers drift, this does not
    src = pathlib.Path(T.__file__).read_text()
    assert "al.load_state_dict(migrate_al_state(al_state))" in src


# =========================================== 5. the demonstration, printed ===
def test_demonstration_one_over_limit_element():
    """One over-limit element: the two violations, their ratio, both lambdas."""
    b = _batch(amp=ONE_ELEMENT_AMP, B=1)
    cost = _cost(w_torque_peak=1.0)
    out = _fwd(b, cost)
    mean_v = float(out.violations["torque"][0])
    peak_v = float(out.violations["torque_peak"][0])
    n_valid = float(out.counts["torque"][0])
    frac = float(out.diagnostics["torque_frac_over"][0])
    print(f"\nn_valid {n_valid:.0f}, elements over the hard limit "
          f"{frac * n_valid:.0f} ({100 * frac:.2f} %)")
    print(f"violations['torque']      (mean excess) = {mean_v:.6e} N*m")
    print(f"violations['torque_peak'] (max  excess) = {peak_v:.6f} N*m")
    print(f"ratio peak/mean = {peak_v / mean_v:.1f}x  (n_valid = {n_valid:.0f})")

    viol = {k: out.violations[k] for k in out.violations}
    al = AugmentedLagrangian(phase2_al_config(ema=0.0, decay=1.0))
    print("  20 AL updates, pure integral (ema 0, decay 1):")
    print("  step   lam_torque   lam_torque_peak")
    for s in range(21):
        if s % 5 == 0:
            print(f"  {s:4d}   {al.lam['torque']:10.5f}   "
                  f"{al.lam['torque_peak']:14.5f}")
        al.update(viol)

    # ... and where the DEFAULT dynamics (decay 0.999) put each multiplier:
    # lam* = rho * v / (1 - decay).
    cfg = phase2_al_config()
    for g, v in (("torque", mean_v), ("torque_peak", peak_v)):
        eq = cfg.rho[g] * v / (1.0 - cfg.decay)
        print(f"  default-config equilibrium lam_{g:11s} = {eq:8.4f} "
              f"(rho {cfg.rho[g]:.1e} x violation {v:.4f})")
    assert peak_v > mean_v


def test_no_group_can_be_dropped_from_the_al_view(violating):
    """Every constraint group the cost reports is a group the AL integrates."""
    _b, _c, out = violating
    reported = {k for k in out.violations}
    assert reported == set(CONSTRAINT_GROUPS), reported ^ set(CONSTRAINT_GROUPS)
    al = AugmentedLagrangian(phase2_al_config(decay=1.0))
    before = dict(al.lam)
    al.update(out.violations)
    assert all(al.lam[g] >= before[g] for g in CONSTRAINT_GROUPS)


def test_the_full_batch_shape_of_the_new_group(violating):
    _b, _c, out = violating
    assert out.violations["torque_peak"].ndim == 1
    assert torch.isfinite(out.violations["torque_peak"]).all()
    assert bool((out.violations["torque_peak"] >= 0).all())


def test_al_history_records_the_new_group():
    al = AugmentedLagrangian(phase2_al_config())
    al.update({g: 0.1 for g in CONSTRAINT_GROUPS})
    assert "torque_peak" in al.history[-1]
    assert "torque_peak" in al.weights()
    assert "torque_peak" in repr(al)


def test_public_export_surface_unchanged():
    assert A.CONSTRAINT_GROUPS is CONSTRAINT_GROUPS
    assert "torque_peak" in A.CONSTRAINT_GROUPS
