"""Section 9b: the three waist-ablation mechanisms.

1. ``CostConfig.w_waist_yaw_vel`` -- a quadratic well on the waist-yaw RATE.
   The exact complement of ``w_waist_upright`` (section 7d), which prices
   pitch/roll POSITION: that one says where the torso may lean, this one says
   how fast it may turn.  It exists because the terminal term prices tip speed
   at contact and the torso is the cheapest place to buy it.

2. ``ModelConfig.frozen_joints`` -- columns the decoder does not control at all.
   Not a price, a structural removal: there is nothing to trade against.

3. The ``waist_yaw_ref`` WIRING.  ``CostConfig.w_waist_aim`` was reachable from
   the CLI, documented in section 7d, and identically zero on every phase-2
   batch, because the only place that ever built the reference
   (``MPCRoller.strike_context``) is not on the training path.  The tests at the
   bottom are what stop that from silently coming back.
"""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt import build_model
from amortized_opt.model import ModelConfig
from amortized_opt.train_cost import TERM_KEYS, phase2_cost_config

# ``pytest.ini`` puts ``tests`` on the path, so this is a top-level import.
from test_waist_upright import ARM_COLS, DT, H, I_PITCH, I_ROLL, I_YAW, _ctx

PR = [I_PITCH, I_ROLL]


@pytest.fixture(scope="module")
def cost_yv():
    """The rate well at unit weight, everything else at the shipped defaults."""
    return FusedCost(CostConfig(w_waist_yaw_vel=1.0), max_horizon=H,
                     device="cpu", dtype=torch.float64)


@pytest.fixture(scope="module")
def cost_off():
    return FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)


def _ramp(rate_yaw: float, dtype=torch.float64):
    """A trajectory whose waist yaw travels at a CONSTANT ``rate_yaw`` rad/s.

    A straight line is exactly representable by the cubic spline given the
    matching clamped start velocity, so the sampled ``qd`` is the rate itself
    and the term's value is a number we can predict in closed form rather than
    merely compare against itself.
    """
    q0 = torch.zeros(1, N_DOF, dtype=dtype)
    v0 = torch.zeros(1, N_DOF, dtype=dtype)
    v0[:, I_YAW] = rate_yaw
    k = torch.arange(1, H + 1, dtype=dtype) * DT
    wp = q0.unsqueeze(1).repeat(1, H, 1).clone()
    wp[0, :, I_YAW] = rate_yaw * k
    return q0, wp, v0


# ------------------------------------------------------- the rate well ------
def test_off_by_default_is_exactly_zero(cost_off):
    """``w_waist_yaw_vel = 0`` must leave the objective bit-identical."""
    assert CostConfig().w_waist_yaw_vel == 0.0
    q0, wp, v0 = _ramp(2.0)
    ct = cost_off.forward(q0, wp, v0, _ctx(1), lengths=None, weights=None)
    assert float(ct.terms["waist_yaw_vel"]) == 0.0
    # ...and the raw statistic is still reported, so a baseline run has a
    # number to compare the ablation against.
    assert float(ct.raw["waist_yaw_vel"]) > 0.0


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0])
def test_a_steady_turn_pays_the_weight_times_the_rate_squared(cost_yv, rate):
    """The scale claim in the config docstring, as an equality.

    The reduction is over the yaw row ALONE, which is what makes the weight
    readable as a target rate in (rad/s)^2 instead of a number whose meaning
    depends on how many joints happen to be in the mean.
    """
    q0, wp, v0 = _ramp(rate)
    ct = cost_yv.forward(q0, wp, v0, _ctx(1), lengths=None, weights=None)
    assert float(ct.raw["waist_yaw_vel"]) == pytest.approx(rate ** 2, rel=1e-6)
    assert float(ct.terms["waist_yaw_vel"]) == pytest.approx(rate ** 2, rel=1e-6)
    assert float(ct.diagnostics["waist_yawd_max"]) == pytest.approx(abs(rate), rel=1e-6)


def test_the_rate_well_is_yaw_only(cost_yv):
    """Pitch/roll motion must not move it -- they have their own term."""
    q0, wp, v0 = _ramp(1.0)
    ct_a = cost_yv.forward(q0, wp, v0, _ctx(1), lengths=None, weights=None)

    wp2 = wp.clone()
    v02 = v0.clone()
    v02[:, PR] = 1.5
    k = torch.arange(1, H + 1, dtype=wp.dtype) * DT
    for c in PR:
        wp2[0, :, c] = 1.5 * k
    ct_b = cost_yv.forward(q0, wp2, v02, _ctx(1), lengths=None, weights=None)

    assert float(ct_b.raw["waist_yaw_vel"]) == pytest.approx(
        float(ct_a.raw["waist_yaw_vel"]), rel=1e-9
    )
    # the pitch/roll RATE diagnostic, by contrast, must see it
    assert float(ct_b.diagnostics["waist_prd_max"]) == pytest.approx(1.5, rel=1e-3)
    assert float(ct_a.diagnostics["waist_prd_max"]) == pytest.approx(0.0, abs=1e-9)


def test_the_rate_well_gradient_reaches_yaw_and_nothing_else(cost_yv):
    q0, wp, v0 = _ramp(1.0)
    wp = wp.clone().requires_grad_(True)
    ct = cost_yv.forward(q0, wp, v0, _ctx(1), lengths=None, weights=None)
    ct.terms["waist_yaw_vel"].sum().backward()
    g = wp.grad[0]
    assert g[:, I_YAW].abs().max() > 0.0
    assert g[:, PR].abs().max() == 0.0
    assert g[:, ARM_COLS].abs().max() == 0.0


def test_the_rate_well_is_registered_for_logging():
    """A term that is not in ``TERM_KEYS`` never reaches ``metrics.jsonl``."""
    assert "waist_yaw_vel" in TERM_KEYS


# ---------------------------------------------------- the frozen columns ----
def _model_cfg(**kw):
    return ModelConfig(max_horizon=8, dec_layers=1, d_model=64, d_ctx=64,
                       ctx_layers=1, **kw)


def test_frozen_joints_defaults_to_nothing_frozen():
    m = build_model(_model_cfg(), "cpu")
    assert ModelConfig().frozen_joints == ()
    assert m.decoder.n_frozen == 0


def test_a_bare_string_is_not_read_character_by_character():
    """``frozen_joints="waist_pitch_joint"`` must freeze ONE joint, not 17."""
    cfg = _model_cfg(frozen_joints="waist_pitch_joint")
    assert cfg.frozen_joints == ("waist_pitch_joint",)


def test_an_unknown_joint_name_is_rejected():
    with pytest.raises(ValueError, match="frozen_joints"):
        _model_cfg(frozen_joints=("waist_twist_joint",))


def test_a_list_round_trips_as_a_tuple():
    """A checkpoint's ``model_cfg`` comes back through JSON as a list."""
    cfg = _model_cfg(frozen_joints=["waist_pitch_joint", "waist_roll_joint"])
    assert cfg.frozen_joints == ("waist_pitch_joint", "waist_roll_joint")


POOL_H = 25


@pytest.fixture(scope="module")
def small_pool():
    """A real scenario pool, generated IN PROCESS.

    ``generate_cost_pool``'s worker path forks, which pytest's autograd state
    refuses; ``generate_cost_scenarios`` is the same generator without the pool.
    """
    import amortized_opt as A
    from amortized_opt.phase2_data import (
        Phase2Pool,
        Phase2ScenarioConfig,
        generate_cost_scenarios,
    )

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = Phase2ScenarioConfig(horizon=POOL_H, buckets=(POOL_H,))
    raw = generate_cost_scenarios(120, clips, seed=3, cfg=cfg)
    return Phase2Pool(raw["buckets"], cfg=cfg, device="cpu")


def _model_cfg_for(pool, **kw):
    return ModelConfig(max_horizon=POOL_H, dec_layers=1, d_model=64, d_ctx=64,
                       ctx_layers=1, num_drums=pool.cfg.num_drums,
                       num_future=pool.cfg.num_future, **kw)


@pytest.fixture(scope="module")
def frozen_pair(small_pool):
    """(unfrozen, frozen) models sharing one set of weights, plus a batch."""
    torch.manual_seed(0)
    free = build_model(_model_cfg_for(small_pool), "cpu")
    torch.manual_seed(0)
    pinned = build_model(
        _model_cfg_for(small_pool,
                       frozen_joints=("waist_pitch_joint", "waist_roll_joint")),
        "cpu",
    )
    # the freeze must not be an architecture change
    pinned.load_state_dict(free.state_dict(), strict=True)
    batch = small_pool.batch(POOL_H, torch.arange(4), noise=False)
    return free, pinned, batch


def test_frozen_columns_settle_at_the_ready_pose(frozen_pair):
    """After the ramp the pinned columns hold the ready pose exactly."""
    from amortized_opt.constants import G1_READY_JOINT_OFFSETS

    free, pinned, b = frozen_pair
    init, state, ctx, masks = b["init"], b["state"], b["context"], b["masks"]
    with torch.no_grad():
        out = pinned(init, state, pinned.encode_context(ctx), masks)
        ref = free(init, state, free.encode_context(ctx), masks)

    ramp = pinned.decoder.cfg.frozen_ramp_wp
    for c in PR:
        want = G1_READY_JOINT_OFFSETS.get(G1_UPPER_BODY_JOINTS[c], 0.0)
        tail = out.traj[:, ramp:, c]
        assert tail.sub(want).abs().max() < 1e-6
        # ...and the unfrozen twin, on the same weights, does NOT sit there
        assert ref.traj[:, ramp:, c].sub(want).abs().max() > 1e-6

    # every other column is untouched, bit for bit
    keep = [i for i in range(N_DOF) if i not in PR]
    assert torch.equal(out.traj[:, :, keep], ref.traj[:, :, keep])
    assert torch.equal(out.vel[:, :, keep], ref.vel[:, :, keep])


def test_the_frozen_ramp_starts_at_the_measured_pose_not_at_ready(frozen_pair):
    """THE POINT of the ramp: a start 0.5 rad off zero must not be commanded to
    zero on waypoint 1.  The pool samples waist pitch/roll at 0.09 rad rms with
    tails at the full URDF +-0.52, and a step that size inside one 20 ms period
    puts the acceleration straight onto ``torque_peak`` -- the statistic the
    acceptance gate thresholds on."""
    _free, pinned, b = frozen_pair
    state = b["state"].clone()
    state[:, PR] = 0.5
    with torch.no_grad():
        out = pinned(b["init"], state, pinned.encode_context(b["context"]), b["masks"])

    ramp = pinned.decoder.cfg.frozen_ramp_wp
    for c in PR:
        col = out.traj[0, :, c]
        # waypoint 1 has barely moved off the measured pose (quintic smoothstep
        # leaves at zero slope), and the whole ramp is monotone down to zero
        step_1 = float((0.5 - col[0]).abs())
        assert step_1 < 0.5 / ramp, f"waypoint 1 jumped {step_1:.3f} rad"
        assert float(col[ramp - 1].abs()) < 1e-6
        assert torch.all(col[:ramp].diff() <= 1e-9)
        # the per-waypoint travel never exceeds what a step would have demanded
        assert float(col.diff().abs().max()) < 0.5


def test_no_gradient_reaches_the_head_through_a_frozen_column(frozen_pair):
    _free, pinned, b = frozen_pair
    init, state, ctx, masks = b["init"], b["state"], b["context"], b["masks"]
    pinned.zero_grad(set_to_none=True)
    out = pinned(init, state, pinned.encode_context(ctx), masks)
    out.traj[:, :, PR].sum().backward()
    g = pinned.decoder.head_traj.weight.grad
    assert g is None or float(g.abs().max()) == 0.0


def test_freezing_nothing_is_bit_identical(frozen_pair, small_pool):
    """The default path must not pay for, or be perturbed by, the feature."""
    free, _pinned, b = frozen_pair
    torch.manual_seed(0)
    also_free = build_model(_model_cfg_for(small_pool, frozen_joints=()), "cpu")
    also_free.load_state_dict(free.state_dict(), strict=True)
    init, state, ctx, masks = b["init"], b["state"], b["context"], b["masks"]
    with torch.no_grad():
        a = free(init, state, free.encode_context(ctx), masks)
        b = also_free(init, state, also_free.encode_context(ctx), masks)
    assert torch.equal(a.traj, b.traj)


# ------------------------------------------------- the waist_aim wiring -----
def test_a_fresh_pool_batch_carries_the_waist_reference(small_pool):
    """THE REGRESSION.  ``w_waist_aim`` priced nothing before 2026-08-12."""
    batch = small_pool.batch(POOL_H, torch.arange(8), noise=False)
    ref = batch["strike"].waist_yaw_ref
    assert ref is not None, "the waist-aim IL term has nothing to price"
    assert ref.shape == (8, POOL_H)
    # it is the AIM, not a constant: the reference moves within a sample and
    # differs between samples
    assert ref.std(dim=1).max() > 1e-3


def test_the_reference_is_the_seed_s_own_aim_ramp(small_pool):
    batch = small_pool.batch(POOL_H, torch.arange(8), noise=False)
    assert torch.equal(batch["strike"].waist_yaw_ref, batch["seed"][:, :, I_YAW])


def test_the_waist_aim_term_is_now_non_zero(small_pool):
    batch = small_pool.batch(POOL_H, torch.arange(8), noise=False)
    cost = FusedCost(phase2_cost_config(w_waist_aim=500.0), max_horizon=POOL_H,
                     device="cpu")
    ct = cost.forward(batch["q0"], batch["seed"], batch["v0"], batch["strike"],
                      lengths=None, weights=None)
    assert float(ct.terms["waist_aim"].max()) > 0.0


def test_the_replay_pool_never_derives_the_reference_from_its_own_init():
    """Its ``init`` is the model's shifted previous plan; using it would price
    the model against itself and read zero whatever the torso did."""
    from amortized_opt.train_cost import ReplayPool

    assert ReplayPool.waist_ref_from_seed is False


def test_an_unharvested_replay_pool_returns_no_reference(small_pool):
    """``waist_ref_from_seed=False`` + no harvest must mean NO anchor at all.

    Before 2026-08-13 the replay pool always allocated a ``waist_ref`` column
    of zeros and the read path checked ``"waist_ref" in d`` before the class
    attribute -- so with harvesting off, every replay row would have been
    priced toward yaw = 0 the moment anything stopped masking the term.
    """
    from amortized_opt.train_cost import ReplayPool

    replay = ReplayPool(small_pool, 64, buckets=[POOL_H])   # waist_ref=False
    assert "waist_ref" not in replay.data[POOL_H]
    assert replay.waist_ref(POOL_H) is None
    # ...and the harvesting configuration does carry (and require) the column
    harvesting = ReplayPool(small_pool, 64, buckets=[POOL_H], waist_ref=True)
    assert "waist_ref" in harvesting.data[POOL_H]
    row = {k: v[:2] for k, v in small_pool.data[POOL_H].items()}
    with pytest.raises(ValueError, match="waist_ref"):
        harvesting.add(POOL_H, row)   # a wave WITHOUT the reference is a bug


def test_the_seed_itself_pays_zero_waist_aim(small_pool):
    """Producer/consumer knot alignment (the off-by-one regression).

    The reference rows are knots 1..H (``StrikeContext.waist_yaw_ref``), and so
    are the trajectory's waypoint rows -- so a trajectory EXACTLY equal to the
    seed must read ``raw["waist_aim"] == 0``.  Before 2026-08-13 the cost
    indexed the reference as knots 0..H-1, reading the fresh pool's ramp one
    knot ahead of where the seed actually is.
    """
    batch = small_pool.batch(POOL_H, torch.arange(8), noise=False)
    cost = FusedCost(
        phase2_cost_config(w_waist_aim=500.0, waist_aim_yaw_only=True),
        max_horizon=POOL_H, device="cpu",
    )
    ct = cost.forward(batch["q0"], batch["seed"], batch["v0"], batch["strike"],
                      lengths=None, weights=None)
    assert float(ct.raw["waist_aim"].abs().max()) < 1e-9

    # sensitivity: the OLD convention (rows = knots 0..H-1, i.e. the same ramp
    # shifted one knot late) must NOT read zero, or this test pins nothing
    from dataclasses import replace as dc_replace

    ref = batch["strike"].waist_yaw_ref
    q0_yaw = batch["q0"][:, I_YAW].unsqueeze(1)
    shifted = dc_replace(batch["strike"],
                         waist_yaw_ref=torch.cat([q0_yaw, ref[:, :-1]], dim=1))
    ct2 = cost.forward(batch["q0"], batch["seed"], batch["v0"], shifted,
                       lengths=None, weights=None)
    assert float(ct2.raw["waist_aim"].abs().max()) > 1e-7


def test_bptt_second_solve_keeps_waist_yaw_ref_none(small_pool):
    """``Phase2Config.waist_sp_from_head``'s docstring says BPTT keeps
    ``waist_yaw_ref=None``; before 2026-08-13 the fresh ``pool.batch`` strike
    carried the seed's analytic ramp into the chain and fought the learned
    anchor with ``w_bptt * w_waist_aim`` of gradient."""
    from types import SimpleNamespace

    from amortized_opt.train_cost import Phase2Config, _bptt_term

    model = build_model(_model_cfg_for(small_pool), "cpu")
    cfg = Phase2Config(compile_mode=None, device="cpu", bptt_batch=4)
    captured = {}

    def bptt_fn(init, state, q0, v0, enc1, masks1, s1, gl1, k_first,
                term2, enc2, masks2, s2, gl2, elen2, w, kp, kd,
                hand, normal, e_rest, qn, qdn, tvn):
        captured["s1"], captured["s2"] = s1, s2
        z = torch.zeros(())
        return z, z

    n_arm = small_pool.data[POOL_H]["q0"].shape[-1]
    sim = SimpleNamespace(sample_gains=lambda n, generator=None: (
        torch.ones(n, n_arm), torch.ones(n, n_arm), None))
    gen = torch.Generator()
    _bptt_term(small_pool, SimpleNamespace(), bptt_fn, sim, cfg, POOL_H,
               torch.device("cpu"), torch.float32, gen, None, {}, model=model)
    assert captured["s1"].waist_yaw_ref is None
    assert captured["s2"].waist_yaw_ref is None
    # ...deliberately: the pool batch it was built from DOES carry a reference
    b = small_pool.batch(POOL_H, torch.arange(2), noise=False)
    assert b["strike"].waist_yaw_ref is not None
