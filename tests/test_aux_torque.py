"""The auxiliary TORQUE label: which physical quantity ``head_torque`` learns.

``aux_targets`` grew a second definition on 2026-08-16.  The head used to be
regressed onto the plan's own OPEN-LOOP inverse dynamics -- a per-segment
``max(|tau_rnea|)`` -- while the cost had (since 2026-08-15) started pricing the
torque the PD servo actually DEMANDS, ``|tau_rnea + tau_fb|``.  ``torque_mode``
lets a run ask for the second, and this file pins the three things that matter:

1. **the default is untouched, bit for bit.**  Two phase-2 runs were mid-flight
   when this landed and their ``_aux`` step calls ``aux_targets`` every step; a
   default that moved would have silently redefined their objective.  The guard
   below re-derives the historical formula from scratch and demands exact
   equality -- not ``allclose``.
2. ``"tracking"`` is THE SAME ARRAY the cost prices, asserted against
   :meth:`~amortized_opt.cost.FusedCost._track_torque`'s own output rather than
   against a reimplementation of it.
3. shapes, masking and the new phase-1 metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import N_DOF, G1_UPPER_BODY_JOINTS
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK
from amortized_opt.limits import isaac_effort_limits
from amortized_opt.pd_sim import feedback_torque, nominal_torque_kernels
from amortized_opt.rnea import BatchedRNEA
from amortized_opt.train_warmstart import METRIC_NAMES, LossScales, forward_loss
from amortized_opt.tspline import TrajectorySplineBuilder
from amortized_opt.warmstart_data import aux_targets

DT = 0.02
H = 12
F64 = torch.float64


# ================================================================ fixtures ===
@pytest.fixture(scope="module")
def parts():
    fk = ChainFK(device="cpu", dtype=F64)
    rnea = BatchedRNEA.from_urdf(with_armature=True, device="cpu", dtype=F64)
    builder = TrajectorySplineBuilder(max_segments=H, dt=DT, device="cpu", dtype=F64)
    return builder, fk, rnea


def _batch(B=5, seed=17, full_mask=True):
    g = torch.Generator().manual_seed(seed)
    q0 = torch.randn(B, N_DOF, generator=g, dtype=F64) * 0.4
    v0 = torch.randn(B, N_DOF, generator=g, dtype=F64) * 0.2
    target = q0.unsqueeze(1) + torch.randn(B, H, N_DOF, generator=g, dtype=F64) * 0.15
    em = torch.ones(B, H, N_DOF, dtype=F64)
    if not full_mask:
        # Ragged per-group horizons, the shape a real bucket has: the right
        # arm's columns stop early on half the batch.
        right = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("right")]
        em[: B // 2, H // 2 :, right] = 0.0
    return q0, target, v0, em


def _aux(parts, q0, target, v0, em, **kw):
    builder, fk, rnea = parts
    return aux_targets(q0, target, v0, em, builder=builder, fk=fk, rnea=rnea,
                       dt=DT, **kw)


# ============================== 1. the default label has not moved an ulp ====
def test_default_mode_is_the_historical_rnea_segment_max_bit_for_bit(parts):
    """The guard for the two runs that were training when this landed."""
    builder, _fk, rnea = parts
    q0, target, v0, em = _batch()
    got = _aux(parts, q0, target, v0, em)["torque"]

    # The formula as it stood before 2026-08-16, written out here on purpose so
    # this test fails if warmstart_data's default is ever quietly re-plumbed.
    sp = builder.from_masks(q0, target, v0, em)
    xk, qd, qdd = sp.knot_finite_differences(v0)
    B, n, D = xk.shape
    tau = rnea.rnea(
        xk.reshape(B * n, D), qd.reshape(B * n, D), qdd.reshape(B * n, D)
    ).reshape(B, n, D).abs()
    want = torch.maximum(tau[:, :-1], tau[:, 1:])

    assert torch.equal(got, want), "the DEFAULT aux torque target changed"


def test_default_mode_is_the_default(parts):
    q0, target, v0, em = _batch()
    a = _aux(parts, q0, target, v0, em)["torque"]
    b = _aux(parts, q0, target, v0, em, torque_mode="rnea_segmax")["torque"]
    assert torch.equal(a, b)


def test_tau_kernel_is_ignored_under_the_default(parts):
    """Passing a bank must not be able to perturb the historical label."""
    q0, target, v0, em = _batch()
    junk = torch.full((N_DOF, 4), 1e3, dtype=F64)
    a = _aux(parts, q0, target, v0, em)["torque"]
    b = _aux(parts, q0, target, v0, em, tau_kernel=junk)["torque"]
    assert torch.equal(a, b)


def test_unknown_mode_raises(parts):
    q0, target, v0, em = _batch()
    with pytest.raises(ValueError, match="torque_mode"):
        _aux(parts, q0, target, v0, em, torque_mode="pd")


def test_only_the_torque_label_depends_on_the_mode(parts):
    q0, target, v0, em = _batch()
    a = _aux(parts, q0, target, v0, em)
    b = _aux(parts, q0, target, v0, em, torque_mode="tracking")
    for k in ("vel", "acc", "fk_pos", "fk_vel"):
        assert torch.equal(a[k], b[k]), k
    assert not torch.allclose(a["torque"], b["torque"])


def test_cost_forward_aux_labels_match_aux_targets_bit_for_bit(parts):
    """Train step reuses the cost plant; labels must not drift from aux_targets."""
    q0, target, v0, em = _batch()
    want = _aux(parts, q0, target, v0, em, torque_mode="rnea_segmax")
    cost = _cost()
    ct = cost.forward(
        q0, target, v0, _ctx(q0.shape[0]),
        elem_mask=em, aux_mode="rnea_segmax",
    )
    assert ct.aux_labels is not None
    for k in ("vel", "acc", "torque", "fk_pos", "fk_vel"):
        assert torch.equal(ct.aux_labels[k], want[k]), k


def test_cost_forward_tracking_aux_labels_match_aux_targets(parts):
    q0, target, v0, em = _batch()
    cost = _cost(w_track_torque=1.0)
    want = _aux(parts, q0, target, v0, em, torque_mode="tracking",
                tau_kernel=cost.c_tau_kernel)
    ct = cost.forward(
        q0, target, v0, _ctx(q0.shape[0]),
        elem_mask=em, aux_mode="tracking", aux_tau_kernel=cost.c_tau_kernel,
    )
    assert ct.aux_labels is not None
    for k in ("vel", "acc", "torque", "fk_pos", "fk_vel"):
        assert torch.equal(ct.aux_labels[k], want[k]), k


# ================= 2. "tracking" IS the array the cost prices ================
def _cost(**kw):
    return FusedCost(CostConfig(**kw), max_horizon=H, device="cpu", dtype=F64, dt=DT)


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


def test_tracking_label_equals_the_costs_own_tau_hat(parts):
    """The whole point: label and price are ONE estimator.

    Asserted against the cost object's own kernel and RNEA, at the same
    nominal gains, through the same :func:`feedback_torque` -- and then
    cross-checked against a real ``FusedCost.forward`` diagnostic below.
    """
    builder, _fk, _rnea = parts
    q0, target, v0, em = _batch()
    cost = _cost(w_track_torque=1.0)
    assert cost.c_tau_kernel is not None, "the bank should be built at construction"

    got = _aux(parts, q0, target, v0, em, torque_mode="tracking",
               tau_kernel=cost.c_tau_kernel)["torque"]

    sp = builder.from_masks(q0, target, v0, em)
    xk, qd, qdd = sp.knot_finite_differences(v0)
    B, n, D = xk.shape
    tau_r = cost.rnea.rnea(
        xk.reshape(B * n, D), qd.reshape(B * n, D), qdd.reshape(B * n, D)
    ).reshape(B, n, D)
    tau_hat = tau_r + feedback_torque(xk, cost.c_tau_kernel)
    assert torch.equal(got, tau_hat[:, 1:].abs())


def test_tracking_label_reproduces_the_costs_track_torque_diagnostic(parts):
    """End to end through ``FusedCost.forward``, not through any helper.

    ``diag["track_torque_max"]`` is ``max |tau_hat|`` over knots ``0..H``.  The
    label covers knots ``1..H``; knot 0 carries no feedback demand by
    construction (the FIR is DC-removed and the pre-horizon command is held at
    ``x[0]``), so the missing element is exactly ``|rnea(q0, v0, qdd_fd[0])|``.
    """
    builder, _fk, _rnea = parts
    q0, target, v0, em = _batch()
    cost = _cost(w_track_torque=1.0)
    out = cost.forward(q0, target, v0, _ctx(q0.shape[0]), elem_mask=em)
    got = out.diagnostics["track_torque_max"]

    lab = _aux(parts, q0, target, v0, em, torque_mode="tracking",
               tau_kernel=cost.c_tau_kernel)["torque"]
    sp = builder.from_masks(q0, target, v0, em)
    xk, qd, qdd = sp.knot_finite_differences(v0)
    tau0 = cost.rnea.rnea(xk[:, 0], qd[:, 0], qdd[:, 0]).abs()   # knot 0, fb == 0
    want = torch.maximum(lab.amax(dim=(1, 2)), tau0.amax(dim=1))

    assert torch.allclose(got, want, rtol=0, atol=1e-9)


def test_knot_zero_carries_no_feedback_demand(parts):
    """The premise the test above leans on, pinned separately."""
    kernel = nominal_torque_kernels(parts[2], dt=DT, dtype=F64)
    x = torch.randn(3, H + 1, N_DOF, dtype=F64)
    fb = feedback_torque(x, kernel)
    assert torch.allclose(fb[:, 0], torch.zeros_like(fb[:, 0]), atol=1e-12)


def test_tracking_is_materially_bigger_than_rnea_on_the_wrists(parts):
    """Not a tolerance -- the reason the change exists."""
    q0, target, v0, em = _batch(B=16, seed=3)
    a = _aux(parts, q0, target, v0, em)["torque"]
    b = _aux(parts, q0, target, v0, em, torque_mode="tracking")["torque"]
    wr = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS)
          if "wrist_pitch" in n or "wrist_yaw" in n]
    assert len(wr) == 4
    rms = lambda t: float(t[..., wr].pow(2).mean().sqrt())  # noqa: E731
    assert rms(b) > 2.0 * rms(a), (rms(a), rms(b))


# ============================ 3. shapes, masking, and the new metrics ========
@pytest.mark.parametrize("mode", ["rnea_segmax", "tracking"])
@pytest.mark.parametrize("full", [True, False])
def test_shapes_and_masking_are_mode_independent(parts, mode, full):
    q0, target, v0, em = _batch(full_mask=full)
    out = _aux(parts, q0, target, v0, em, torque_mode=mode)
    B = q0.shape[0]
    assert out["torque"].shape == (B, H, N_DOF)
    assert out["vel"].shape == out["acc"].shape == (B, H, N_DOF)
    assert out["fk_pos"].shape[:2] == (B, H) and out["fk_pos"].shape[-1] == 3
    assert torch.isfinite(out["torque"]).all()
    assert (out["torque"] >= 0).all(), "the label is an absolute torque"


def test_ragged_mask_does_not_leak_across_the_batch(parts):
    """A row's label may not depend on any other row's horizon."""
    q0, target, v0, em = _batch(full_mask=False)
    both = _aux(parts, q0, target, v0, em, torque_mode="tracking")["torque"]
    for i in (0, len(q0) - 1):
        one = _aux(parts, q0[i : i + 1], target[i : i + 1], v0[i : i + 1],
                   em[i : i + 1], torque_mode="tracking")["torque"]
        assert torch.allclose(both[i : i + 1], one, atol=1e-12)


def test_new_torque_metrics_are_present_and_finite():
    names = list(METRIC_NAMES)
    assert "rmse_torque_frac" in names and "rmse_torque_wrist_nm" in names

    torch.manual_seed(0)
    scales = LossScales(torch.device("cpu"))
    rating = torch.as_tensor(np.asarray(isaac_effort_limits(), float), dtype=torch.float32)
    assert float(scales.wrist_mask.sum()) == 4.0
    assert torch.allclose(rating * scales.wrist_mask.view(-1),
                          5.0 * scales.wrist_mask.view(-1))

    B, D = 4, N_DOF
    aux = {k: torch.randn(B, H, D) for k in ("vel", "acc", "torque")}
    fkp = torch.randn(B, H, 6, 3)
    em = torch.ones(B, H, D)
    pm = torch.ones(B, H, 6)

    class _Out:
        pass

    class _Module(torch.nn.Module):
        def encode_context(self, context):
            return context

        def forward(self, init, state, enc, masks):
            o = _Out()
            o.traj = init
            o.vel, o.acc = torch.randn(B, H, D), torch.randn(B, H, D)
            o.torque = torch.randn(B, H, D) * 5.0
            o.fk_pos, o.fk_vel = torch.randn(B, H, 6, 3), torch.randn(B, H, 6, 3)
            return o

    class _Masks:
        elem_mask = em
        point_mask = pm

    from amortized_opt.train_warmstart import LossWeights

    loss, metrics = forward_loss(
        _Module(), torch.randn(B, H, D), None, None, _Masks(), torch.randn(B, H, D),
        aux["vel"], aux["acc"], aux["torque"], fkp, fkp.clone(),
        LossScales(torch.device("cpu")), LossWeights(),
    )
    assert metrics.numel() == len(METRIC_NAMES)
    rec = dict(zip(METRIC_NAMES, metrics.tolist()))
    assert np.isfinite(rec["rmse_torque_frac"]) and rec["rmse_torque_frac"] > 0
    assert np.isfinite(rec["rmse_torque_wrist_nm"]) and rec["rmse_torque_wrist_nm"] > 0
    # The wrist figure is in N*m against a 5 N*m rating, the averaged one is
    # dominated by the 88 N*m waist -- they must not be the same number.
    assert rec["rmse_torque_wrist_nm"] != pytest.approx(rec["rmse_torque_nm"])


# ================================================= 4. the opt-in plumbing ====
def test_phase2_config_defaults_off():
    from amortized_opt.train_cost import Phase2Config

    assert Phase2Config().aux_torque_tracking is False


def test_phase1_train_config_defaults_off_and_pool_agrees():
    from amortized_opt.train_warmstart import TrainConfig
    from amortized_opt.warmstart_data import WarmstartPool

    assert TrainConfig().aux_torque_tracking is False
    assert "aux_torque_tracking" in WarmstartPool.__init__.__code__.co_varnames
