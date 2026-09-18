"""The unified tracking-torque limit system.

The cost prices the RNEA of the PLAN; the actuators have to produce whatever the
PD controller DEMANDS while tracking it, and in this plant those are different
quantities by an order of magnitude on the wrists -- the servo's ``kd * qd_cmd``
feed-forward asks for ~0.08 N*m per rad/s^2 of commanded acceleration against a
physical wrist inertia of 0.012, and none of that appears in
``rnea(q, qd, qdd)``.  This file pins the four pieces that close the gap:

1. :func:`~amortized_opt.limits.available_torque_np` /
   ``_torch`` -- the ONE definition of the limit, including its derate to zero
   at the speed rating;
2. the FIR estimator: kernels identified against the plant reproduce the plant,
   and the DC removal makes a constant command cost exactly nothing;
3. **everything is off by default, bit for bit** -- ``FusedCost`` with both
   tracking weights at zero, and ``simulate_pd`` with ``saturate_effort=False``,
   proved by SABOTAGE (the estimator is monkeypatched to return a garbage value
   that would be impossible to miss if any of it were wired in);
4. the pieces do what they claim when turned ON: the clamp clamps, the term
   prices a wrist snap and its gradient relieves it, and the eval gate moves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import CONTROL_DT, N_DOF, PD_SUBSTEPS, G1_UPPER_BODY_JOINTS
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.limits import (
    ETA_TORQUE,
    available_torque,
    available_torque_np,
    available_torque_torch,
    isaac_effort_limits,
    isaac_runtime_damping,
    isaac_runtime_stiffness,
    isaac_velocity_limits,
)
from amortized_opt.pd_sim import (
    BatchedPDSim,
    PDSimConfig,
    extract_torque_kernels,
    feedback_torque,
    simulate_pd,
)

DT = 0.02
H = 12
F64 = torch.float64
JOINT = {n: i for i, n in enumerate(G1_UPPER_BODY_JOINTS)}
WRIST = JOINT["right_wrist_pitch_joint"]


# ============================================== 1. the single limit function ==
def test_available_torque_no_derate_is_the_scaled_rating():
    tau = isaac_effort_limits()
    got = available_torque_np(np.zeros(N_DOF), eta=1.0, derate="none")
    assert np.allclose(got, tau)
    got = available_torque_np(np.full(N_DOF, 1e3), eta=0.8, derate="none")
    assert np.allclose(got, 0.8 * tau), "derate='none' must ignore qd entirely"


def test_available_torque_derates_linearly_to_zero_at_the_speed_rating():
    tau, vel = isaac_effort_limits(), isaac_velocity_limits()
    assert np.allclose(available_torque_np(np.zeros(N_DOF), eta=1.0), tau)
    assert np.allclose(available_torque_np(0.5 * vel, eta=1.0), 0.5 * tau)
    # AT the rating, and beyond it, the joint has nothing left.
    assert np.allclose(available_torque_np(vel, eta=1.0), 0.0)
    assert np.allclose(available_torque_np(2.0 * vel, eta=1.0), 0.0)
    assert np.all(available_torque_np(-0.25 * vel, eta=1.0) == 0.75 * tau), "|qd|"


def test_available_torque_eta_is_a_plain_factor():
    vel = isaac_velocity_limits()
    a = available_torque_np(0.3 * vel, eta=1.0)
    b = available_torque_np(0.3 * vel, eta=ETA_TORQUE)
    assert np.allclose(b, ETA_TORQUE * a)


def test_available_torque_saturation_scale_caps_at_the_rating():
    """``sat_scale > 1`` may not hand a joint more than its rating at rest."""
    tau, vel = isaac_effort_limits(), isaac_velocity_limits()
    at_rest = available_torque_np(np.zeros(N_DOF), eta=1.0, saturation_scale=2.0)
    assert np.allclose(at_rest, tau)
    # ...but it does widen the band over which the derate bites.
    half = available_torque_np(0.5 * vel, eta=1.0, saturation_scale=2.0)
    assert np.allclose(half, tau)


def test_available_torque_torch_matches_numpy_and_dispatches():
    vel = isaac_velocity_limits()
    qd = np.linspace(-1.0, 1.0, N_DOF) * vel
    ref = available_torque_np(qd, eta=0.8)
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=F64)
    got = available_torque_torch(t(qd), t(isaac_effort_limits()), t(vel), eta=0.8)
    assert torch.allclose(got, t(ref))
    disp = available_torque(t(qd), effort_limit=t(isaac_effort_limits()),
                            velocity_limit=t(vel), eta=0.8)
    assert torch.allclose(disp, got)
    assert isinstance(available_torque(qd, eta=0.8), np.ndarray)


def test_available_torque_rejects_an_unknown_derate():
    with pytest.raises(ValueError, match="derate"):
        available_torque_np(np.zeros(N_DOF), derate="quadratic")


# =========================================== 2. the kernel and its DC removal ==
def _nominal(dtype=F64):
    kp = torch.as_tensor(isaac_runtime_stiffness(), dtype=dtype)
    kd = torch.as_tensor(isaac_runtime_damping(), dtype=dtype)
    return kp, kd


#: A frozen diagonal mobility: crude, but the kernel and the reference rollout
#: below use the SAME one, which is what makes the comparison exact-in-principle.
MINV_DIAG = torch.tensor(
    [1.7, 0.9, 1.1, 5.0, 5.8, 42.0, 22.0, 96.0, 68.0, 84.0,
     5.0, 5.8, 42.0, 22.0, 96.0, 68.0, 84.0], dtype=F64
)


def _kernel(**kw):
    kp, kd = _nominal()
    return extract_torque_kernels(kp, kd, MINV_DIAG, dt=DT, substeps=PD_SUBSTEPS, **kw)


def _sim_demand(x_cmd, *, reduce="max"):
    """Ground truth: the plant's own demanded-torque series for ``x_cmd``.

    Started SETTLED AT ``x_cmd[0]`` (``qa0 = x[0]``, ``va0 = 0``), because that
    is the initial condition the estimator assumes: it convolves the command's
    deviation from itself with the pre-horizon command held at ``x[0]``, which
    is the same statement.  A plan's row 0 IS the pinned start pose, so this is
    also the real one.
    """
    kp, kd = _nominal()
    B, _, n = x_cmd.shape
    _qa, _va, _ig, tau = simulate_pd(
        x_cmd, torch.diag_embed(MINV_DIAG), kp.expand(B, n), kd.expand(B, n), DT,
        x_cmd[:, 0].clone(), torch.zeros(B, n, dtype=F64),
        substeps=PD_SUBSTEPS, return_torque=True, torque_reduce=reduce,
    )
    return tau


def test_kernel_is_short_and_decays():
    k = _kernel()
    assert k.shape[0] == N_DOF
    assert 3 <= k.shape[1] <= 24, k.shape
    # the tail really is negligible where it was cut
    assert float(k[:, -1].abs().max()) < 1e-2 * float(k.abs().max())


def _synthetic_commands(B=4, K=40, seed=0):
    torch.manual_seed(seed)
    t = torch.linspace(0.0, 1.0, K + 1, dtype=F64).view(1, -1, 1)
    return 0.3 * torch.sin(2 * np.pi * torch.rand(B, 1, N_DOF, dtype=F64) * 4.0 * t
                           + torch.rand(B, 1, N_DOF, dtype=F64) * 6.0)


def _rel_error(x, *, reduce, taps, tol):
    ref = _sim_demand(x, reduce=reduce)                        # (B, K, n)
    k = _kernel(taps=taps, tol=tol, torque_reduce=reduce)
    est = feedback_torque(x, k)[:, 1:]                         # knot i <-> tick i
    err = (est - ref).abs().amax(dim=(0, 1))
    return (err / ref.abs().amax(dim=(0, 1)).clamp_min(1e-9)).numpy()


def test_kernel_reproduces_the_plant_on_synthetic_commands():
    """The FIR is the plant, on inputs it was not identified on.

    Random smooth commands, same diagonal mobility and same initial condition
    the kernel assumes.  At the SHIPPED settings the residual is a few percent,
    and it comes from two named places, both bounded below:
    """
    rel = _rel_error(_synthetic_commands(), reduce="max", taps=24, tol=1e-3)
    assert rel.max() < 0.06, dict(zip(G1_UPPER_BODY_JOINTS, rel.round(4)))


def test_the_lti_path_is_exact_up_to_truncation():
    """Source 1 of 2: the ``"max"`` reduction. Remove it and the FIR is the plant.

    With ``torque_reduce="last"`` the command-to-demand map is exactly linear,
    so a long enough kernel reproduces the rollout to ~1e-5 -- what is left is
    the DC-removal convention (subtracting ``sum(k) * x[t]`` rather than the
    linear-algebra-exact ``sum(k) * x[0]``, which differ only by the truncated
    kernel's residual DC gain), not the estimator's structure.
    """
    rel = _rel_error(_synthetic_commands(), reduce="last", taps=60, tol=0.0)
    fast = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if not n.startswith("waist_")]
    assert rel[fast].max() < 1e-4, rel[fast]
    assert rel.max() < 1e-3, rel


def test_truncation_is_the_other_error_source_and_it_shrinks_with_taps():
    """Source 2 of 2: cutting the tail.  More taps, less error, monotonically."""
    x = _synthetic_commands()
    short = _rel_error(x, reduce="last", taps=6, tol=0.0).max()
    long_ = _rel_error(x, reduce="last", taps=60, tol=0.0).max()
    assert long_ < short / 10.0, (short, long_)


def test_dc_removal_kills_a_constant_offset():
    """A command held anywhere costs the servo nothing, at any joint angle.

    This is the double-counting guard: the quasi-static torque is the RNEA
    term's job.  Note the raw convolution does NOT have this property -- the
    truncated kernel's DC gain is small but not zero -- so the assertion below
    is a statement about ``feedback_torque``, not about the kernel.
    """
    k = _kernel()
    assert float(k.sum(dim=1).abs().max()) > 1e-3, "kernel DC really is nonzero"
    for c in (0.0, 0.5, -2.0):
        x = torch.full((3, 30, N_DOF), c, dtype=F64)
        assert float(feedback_torque(x, k).abs().max()) < 1e-12


def test_dc_removal_is_exactly_the_documented_algebra():
    """``tau_fb = (k * x) - sum(k) * x``, boundary included."""
    torch.manual_seed(1)
    k = _kernel()
    L = k.shape[1]
    x = torch.randn(2, 25, N_DOF, dtype=F64)
    got = feedback_torque(x, k)
    want = torch.zeros_like(x)
    for i in range(x.shape[1]):
        for l in range(L):
            want[:, i] += k[:, l] * (x[:, max(i - l, 0)] - x[:, i])
    assert torch.allclose(got, want, atol=1e-10)


def test_feedback_torque_is_differentiable_in_the_command():
    x = torch.zeros(1, 20, N_DOF, dtype=F64, requires_grad=True)
    feedback_torque(x, _kernel()).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()


# ===================================== 3. the defaults are OFF, bit for bit ===
def _ctx(B, dtype=F64):
    ones = torch.ones(B, 2, dtype=dtype)
    return StrikeContext(
        hit_index=torch.full((B, 2), H, dtype=torch.long),
        active=ones.clone(),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, dtype=dtype),
        elev_required=ones.clone(),
    )


def _cost(**kw):
    return FusedCost(CostConfig(**kw), max_horizon=H, device="cpu", dtype=F64, dt=DT)


def _fwd(cost, q0, wp, **kw):
    return cost.forward(q0, wp, torch.zeros_like(q0), _ctx(q0.shape[0]), **kw)


def _random_batch(B=6, seed=17):
    g = torch.Generator().manual_seed(seed)
    q0 = torch.randn(B, N_DOF, generator=g, dtype=F64) * 0.4
    wp = q0.unsqueeze(1) + torch.randn(B, H, N_DOF, generator=g, dtype=F64) * 0.15
    return q0, wp


def _flat(out):
    items = [("total", out.total)]
    for name, d in (("term", out.terms), ("raw", out.raw), ("viol", out.violations),
                    ("diag", out.diagnostics), ("count", out.counts)):
        items += [(f"{name}:{k}", v) for k, v in d.items()]
    return dict(items)


@pytest.mark.parametrize("peak_w", [0.0, 500.0])
def test_cost_default_off_is_bit_for_bit(monkeypatch, peak_w):
    """At both tracking weights 0, no tracking number can reach anything shared.

    Proved by sabotage: ``feedback_torque`` is replaced by one that returns a
    kilonewton-metre on every element, which would blow up the torque term, the
    ``torque`` violation the augmented Lagrangian integrates, and every torque
    diagnostic if any of them were wired to it.  Everything must come back bit
    for bit, INCLUDING the three tracking diagnostics, which are exact zeros.
    """
    import amortized_opt.pd_sim as P

    q0, wp = _random_batch()
    cost = _cost(w_torque_peak=peak_w)
    clean = _flat(_fwd(cost, q0, wp))
    monkeypatch.setattr(P, "feedback_torque", lambda x, k: torch.full_like(x, 1e3))
    dirty = _flat(_fwd(cost, q0, wp))

    assert set(clean) == set(dirty)
    for k in clean:
        assert torch.equal(clean[k], dirty[k]), f"{k} moved with both weights at 0"
    for k in ("track_torque_excess_mean", "track_torque_excess_max", "track_torque_max"):
        assert float(clean[f"diag:{k}"].abs().max()) == 0.0
    # ...and the sabotage is real: it lands the moment the term is priced.
    on = _flat(_fwd(_cost(w_track_torque=1.0), q0, wp))
    assert float(on["diag:track_torque_max"].min()) > 1e2


def test_cost_default_weights_are_zero_and_metrics_are_exported():
    from amortized_opt.train_cost import DIAG_KEYS, DIAG_NAMES, METRIC_NAMES

    assert CostConfig().w_track_torque == 0.0
    assert CostConfig().w_track_torque_peak == 0.0
    assert DIAG_NAMES[DIAG_KEYS.index("track_torque_excess_mean")] == "d_tau_track"
    assert DIAG_NAMES[DIAG_KEYS.index("track_torque_excess_max")] == "d_tau_track_max"
    assert "d_tau_track" in METRIC_NAMES and "d_tau_track_max" in METRIC_NAMES
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))
    assert len(DIAG_NAMES) == len(DIAG_KEYS)


def test_negative_tracking_weights_rejected():
    with pytest.raises(ValueError, match="w_track_torque"):
        CostConfig(w_track_torque=-1.0).validate()
    with pytest.raises(ValueError, match="w_track_torque_peak"):
        CostConfig(w_track_torque_peak=-1.0).validate()


def test_no_kernel_is_built_unless_the_term_is_priced():
    assert _cost().c_tau_kernel is None
    assert _cost(w_track_torque=1.0).c_tau_kernel is not None
    assert _cost(w_track_torque_peak=1.0).c_tau_kernel is not None


def test_a_with_config_clone_that_turns_it_on_still_works():
    on = _cost().with_config(w_track_torque=1.0)
    q0, wp = _random_batch(B=2)
    out = _fwd(on, q0, wp)
    assert torch.isfinite(out.total).all()
    assert on.c_tau_kernel is not None


def test_simulate_pd_saturation_default_off_is_bit_for_bit():
    torch.manual_seed(3)
    B, K, n = 3, 12, N_DOF
    x = torch.randn(B, K + 1, n, dtype=F64) * 0.2
    kp, kd = _nominal()
    args = (torch.diag_embed(MINV_DIAG), kp.expand(B, n), kd.expand(B, n), DT,
            torch.zeros(B, n, dtype=F64), torch.zeros(B, n, dtype=F64))
    a = simulate_pd(x, *args, substeps=PD_SUBSTEPS)
    b = simulate_pd(x, *args, substeps=PD_SUBSTEPS, saturate_effort=False)
    c = simulate_pd(x, *args, substeps=PD_SUBSTEPS, return_torque=True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # asking for the torque series must not perturb the states either
    assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])
    assert len(a) == 3 and len(c) == 4
    assert c[3].shape == (B, K, n)


# ================================================== 4. the pieces, turned on ==
def test_the_clamp_actually_clamps():
    """A command no actuator could follow: the applied force stays inside the band.

    Rigged high-stiffness, big-step case -- the unclamped plant demands
    thousands of N*m -- checked against ``available_torque(va)`` evaluated at
    the very velocities the clamped rollout reaches.
    """
    B, K, n = 2, 8, N_DOF
    x = torch.zeros(B, K + 1, n, dtype=F64)
    x[:, 1:] = 1.5                                  # a 1.5 rad step, held
    kp = torch.full((B, n), 400.0, dtype=F64)
    kd = torch.full((B, n), 40.0, dtype=F64)
    tau_lim = torch.as_tensor(isaac_effort_limits(), dtype=F64)
    vel_lim = torch.as_tensor(isaac_velocity_limits(), dtype=F64)
    args = (torch.diag_embed(MINV_DIAG), kp, kd, DT,
            torch.zeros(B, n, dtype=F64), torch.zeros(B, n, dtype=F64))

    _q, _v, _i, loose = simulate_pd(x, *args, substeps=PD_SUBSTEPS, return_torque=True)
    assert float(loose.abs().max()) > 1e3, "the rig must actually be over the limit"

    qa, va, _i, tight = simulate_pd(
        x, *args, substeps=PD_SUBSTEPS, return_torque=True,
        saturate_effort=True, tau_limit=tau_lim, vel_limit=vel_lim,
    )
    # The reported demand is the CLAMPED one, and the clamp is the velocity
    # derate at the state each tick reaches, so bound it by the widest band any
    # substep of that tick could have offered (the rating at zero speed).
    assert float(tight.abs().max()) <= float(tau_lim.max()) + 1e-9
    assert torch.all(tight.abs() <= tau_lim.view(1, 1, n) + 1e-9)
    # ...and the clamp really did change the trajectory
    assert not torch.allclose(qa, _q)
    assert torch.isfinite(qa).all() and torch.isfinite(va).all()


def test_the_clamp_respects_the_velocity_derate():
    """At the speed rating a joint has zero torque, so the plant stops driving."""
    n = N_DOF
    vel = torch.as_tensor(isaac_velocity_limits(), dtype=F64)
    x = torch.zeros(1, 3, n, dtype=F64)
    x[:, 1:] = 5.0
    kp, kd = _nominal()
    _q, _v, _i, tau = simulate_pd(
        x, torch.diag_embed(MINV_DIAG), kp.view(1, n), kd.view(1, n), DT,
        torch.zeros(1, n, dtype=F64), vel.view(1, n),   # already AT the rating
        substeps=PD_SUBSTEPS, return_torque=True,
        saturate_effort=True, saturate_derate="linear",
    )
    assert float(tau[:, 0].abs().max()) == 0.0


def _snap_batch(B=2, amp=0.45):
    """A plan whose right wrist snaps: RNEA-clean, PD-demand-hostile."""
    q0 = torch.zeros(B, N_DOF, dtype=F64)
    wp = q0.unsqueeze(1).repeat(1, H, 1).clone()
    wp[:, 4:, WRIST] = amp          # one-tick step, then held
    return q0, wp


def test_a_wrist_snap_prices_and_the_gradient_relieves_it():
    cost = _cost(w_track_torque=1.0, w_track_torque_peak=1.0)
    q0, wp = _snap_batch()
    out = _fwd(cost, q0, wp)
    assert float(out.diagnostics["track_torque_excess_max"].min()) > 0.0
    assert float(out.raw["track_torque"].min()) > 0.0
    assert float(out.raw["track_torque_peak"].min()) > 0.0
    # the RNEA hinge cannot see it: a step, then held, is quasi-static
    assert float(out.diagnostics["torque_excess_max"].max()) < float(
        out.diagnostics["track_torque_excess_max"].min()
    )

    wp_g = wp.clone().requires_grad_(True)
    loss = _fwd(cost, q0, wp_g).raw["track_torque"].sum()
    loss.backward()
    g = wp_g.grad
    assert float(g.abs().max()) > 0.0
    assert float(g[:, :, WRIST].abs().max()) > 0.0, "the snapping joint gets gradient"
    step = 1e-3 / float(g.abs().max())
    after = _fwd(cost, q0, (wp - step * g)).raw["track_torque"].sum()
    assert float(after) < float(loss.detach())


def test_the_priced_term_takes_over_the_torque_violation():
    """One multiplier, one constraint: the AL group switches quantity."""
    q0, wp = _snap_batch()
    off = _fwd(_cost(), q0, wp)
    on = _fwd(_cost(w_track_torque=1.0), q0, wp)
    # OFF: the group integrates the plan's own RNEA excess.  ON: it integrates
    # the excess of the PD DEMAND, which on a wrist snap is the larger number by
    # an order of magnitude -- the whole reason for the switch.
    assert torch.equal(on.violations["torque"], on.diagnostics["track_torque_excess_mean"])
    assert not torch.equal(on.violations["torque"], off.violations["torque"])
    assert float(on.violations["torque"].min()) > 5.0 * float(
        off.violations["torque"].max()
    )
    assert float(on.terms["torque"].min()) > float(off.terms["torque"].max())


def test_the_peak_half_rides_the_limit_alpha_ramp():
    """``w_track_torque_peak`` has no AL group, so the ramp is its only gate."""
    q0, wp = _snap_batch()
    cost = _cost(w_track_torque_peak=1.0)
    full = _fwd(cost, q0, wp)
    half = _fwd(cost, q0, wp, weights={"limit_alpha": torch.full((N_DOF,), 0.5,
                                                                dtype=F64)})
    assert float(half.raw["track_torque_peak"].max()) < float(
        full.raw["track_torque_peak"].max()
    )
    # the ramp must NOT reach the raw violation the multiplier integrates
    assert torch.allclose(half.violations["torque"], full.violations["torque"])


# ================================================= 5. the eval gate switches ==
@pytest.fixture(scope="module")
def grid_case():
    import amortized_opt as A

    sc = A.TrajectoryScorer(device="cpu")
    return A.generate_grid(seed=0, grid="standard", scorer=sc)[0]


def test_eval_switch_changes_ok_torque(grid_case):
    """The same trajectory: RNEA-clean, demand-hostile.

    The plan holds the start pose and dithers one wrist by 0.03 rad on alternate
    ticks.  That is a commanded 150 rad/s^2, which against a 0.015 kg m^2 wrist
    is ~2 N*m of inverse dynamics -- comfortably inside the 5 N*m rating -- but
    the servo's velocity feed-forward asks for ~6x that.  The legacy gate passes
    it; the tracking gate must not.
    """
    import amortized_opt as A

    case = grid_case
    Q = np.repeat(np.asarray(case.q_start, float)[None, :], case.n_waypoints, axis=0)
    Q[::2, WRIST] += 0.03

    off = A.TrajectoryScorer(device="cpu", track_torque=False).score(case, Q)
    on = A.TrajectoryScorer(device="cpu", track_torque=True).score(case, Q)
    assert off.ok_torque and not on.ok_torque
    assert on.tau_excess_max_nm > off.tau_excess_max_nm
    assert on.tau_max_nm > off.tau_max_nm
    assert not on.accepted


def test_eval_switch_is_a_no_op_on_the_torque_columns_when_off(grid_case):
    """A scorer built with the switch off is bit-for-bit the historical one."""
    import amortized_opt as A

    case = grid_case
    Q = np.repeat(np.asarray(case.q_start, float)[None, :], case.n_waypoints, axis=0)
    a = A.TrajectoryScorer(device="cpu").score(case, Q)
    b = A.TrajectoryScorer(device="cpu", track_torque=False).score(case, Q)
    same = [f.name for f in __import__("dataclasses").fields(a)
            if f.name not in ("solve_s", "travel_waste")]   # both NaN by design
    for name in same:
        assert getattr(a, name) == getattr(b, name), name


def test_the_scorer_and_the_cost_share_one_kernel():
    """A gate identified from a different filter than the loss is not a gate."""
    import amortized_opt as A

    sc = A.TrajectoryScorer(device="cpu", track_torque=True)
    cost = _cost(w_track_torque=1.0)
    assert sc._tau_kernel.shape == cost.c_tau_kernel.shape
    assert torch.allclose(sc._tau_kernel, cost.c_tau_kernel.to(F64), atol=1e-12)


# ============================================== 6. the plant flag is plumbed ==
def test_pd_sim_config_carries_the_flag_to_the_rollout():
    sim = BatchedPDSim(PDSimConfig(dt=DT, saturate_effort=True), dtype=F64)
    assert sim.cfg.saturate_effort
    x = torch.zeros(1, 4, N_DOF, dtype=F64)
    x[:, 1:] = 2.0
    kp = torch.full((1, N_DOF), 400.0, dtype=F64)
    kd = torch.full((1, N_DOF), 40.0, dtype=F64)
    Minv = torch.diag_embed(MINV_DIAG).unsqueeze(0)
    _q, _v, _i, tau = sim.rollout(
        x, qa0=torch.zeros(1, N_DOF, dtype=F64), va0=torch.zeros(1, N_DOF, dtype=F64),
        kp=kp, kd=kd, Minv=Minv, return_torque=True,
    )
    assert torch.all(tau.abs() <= sim.tau_limit.view(1, 1, -1) + 1e-9)


def test_phase2_config_and_mpc_gen_default_to_off():
    from amortized_opt.mpc_gen import MPCGenConfig
    from amortized_opt.train_cost import Phase2Config

    assert Phase2Config().plant_saturate_effort is False
    assert MPCGenConfig().saturate_effort is False
    assert PDSimConfig().saturate_effort is False
