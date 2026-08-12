"""The idle arm: the park posture, the cost terms, the seed, and the roller.

Section "the idle arm" of ``amortized/README.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS as JOINTS
from amortized_opt.constants import ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK
from amortized_opt.idle_home import (
    _FRAMES,
    IdleHomeConfig,
    IdleHomeTable,
    build_idle_home_table,
    default_idle_home_path,
    load_idle_home_table,
    resolve_idle_home,
)
from amortized_opt.limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from amortized_opt.sdf import DrumSceneSpec, SdfPenalty
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, warmstart_trajectory

_N_ARM = 7
_HANDS = ("left", "right")


@pytest.fixture(scope="module")
def table() -> IdleHomeTable:
    if not default_idle_home_path().exists():
        pytest.skip("idle-home asset not built (scripts/build_idle_home.py)")
    return load_idle_home_table()


@pytest.fixture(scope="module")
def fk_sdf():
    fk = ChainFK(None, joint_names=JOINTS, frames=dict(_FRAMES), dtype=torch.float64)
    scene = DrumSceneSpec.from_layout(None, frame="pelvis")
    return fk, SdfPenalty(scene, fk=fk, margin=0.06, power=2, dtype=torch.float64)


def _sweep(arm_q: np.ndarray, yaws: np.ndarray) -> np.ndarray:
    """(n_yaw, n_dof) poses: ``arm_q`` held while the waist yaws through ``yaws``."""
    q = np.repeat(ready_joint_pos(JOINTS)[None, :], len(yaws), axis=0)
    q[:, JOINTS.index("waist_yaw_joint")] = yaws
    for nm in ("waist_pitch_joint", "waist_roll_joint"):
        q[:, JOINTS.index(nm)] = 0.0
    for hi, hand in enumerate(_HANDS):
        q[:, arm_slice(hand, JOINTS)] = arm_q[hi]
    return q


def _clearance(fk, sdf, q: np.ndarray) -> dict[str, np.ndarray]:
    with torch.no_grad():
        _p, det = sdf.forward(
            frames=fk.forward(torch.as_tensor(q)), return_details=True
        )
    return {h: det.extras[f"min_sdf_{h}"].numpy() for h in _HANDS}


# ====================================================== the park posture ====
def test_pose_shape(table):
    """ONE posture per hand -- torso-relative, not a function of waist yaw."""
    assert table.q_arm.shape == (2, _N_ARM)
    assert table.tip_torso.shape == (2, 3)
    # the API takes no yaw, by construction
    assert np.array_equal(table.arm_pose("left"), table.q_arm[0])


def test_pose_is_inside_the_posture_box(table):
    lo, hi = urdf_position_limits(JOINTS, posture_limits=True)
    for hi_, hand in enumerate(_HANDS):
        idx = arm_slice(hand, JOINTS)
        assert (table.q_arm[hi_] >= lo[idx] - 1e-9).all()
        assert (table.q_arm[hi_] <= hi[idx] + 1e-9).all()


def test_the_SAME_pose_clears_the_kit_at_every_waist_yaw(table, fk_sdf):
    """THE point: the arm is mounted on the torso, so one posture rides the
    waist -- but the KIT is world-fixed, so that posture sweeps a circle which
    has to miss every drum."""
    fk, sdf = fk_sdf
    cl = _clearance(fk, sdf, _sweep(table.q_arm, table.yaw_grid))
    for hand in _HANDS:
        assert cl[hand].min() > 0.0, f"{hand} park posture COLLIDES at some yaw"
        assert cl[hand].min() > table.margin - 5e-3


def test_park_beats_the_ready_pose_it_replaces(table, fk_sdf):
    """An idle arm used to hold URDF-neutral, which collides at many yaws."""
    fk, sdf = fk_sdf
    ready = np.stack([ready_joint_pos(JOINTS)[arm_slice(h, JOINTS)] for h in _HANDS])
    bad_ready = _clearance(fk, sdf, _sweep(ready, table.yaw_grid))
    bad_home = _clearance(fk, sdf, _sweep(table.q_arm, table.yaw_grid))
    for hand in _HANDS:
        assert int((bad_ready[hand] < 0).sum()) > 0, "the baseline should be broken"
        assert int((bad_home[hand] < 0).sum()) == 0
        assert bad_home[hand].min() > bad_ready[hand].min()


def test_park_reaches_every_drum_within_one_second(table):
    """The acceptance bar: one second is long enough to get from home to a drum.

    Quintic ease peaks at 1.875x its mean rate, so the shortest feasible move is
    ``1.875 * |dq| / (eta * rating)``.  Measured worst case is ~0.28 s, i.e. a
    3.6x margin -- which is also why the readiness anchor is not a big lever:
    the speed ratings make almost any in-box park pose reachable in time.
    """
    lib = GoalPoseLibrary(joint_names=JOINTS)
    v_lim = isaac_velocity_limits(JOINTS)
    worst = 0.0
    for hand in _HANDS:
        idx = arm_slice(hand, JOINTS)
        cap = ETA_VELOCITY * v_lim[idx]
        park = table.arm_pose(hand)
        for drum in lib.drums(hand):
            for y in table.yaw_grid[::8]:
                dq = np.abs(lib.select(hand, drum, float(y)).arm_q - park)
                worst = max(worst, float((1.875 * dq / cap).max()))
    assert worst < 1.0, f"park -> drum takes {worst:.3f} s, over the 1 s budget"


def test_worst_clearance_is_reported(table):
    assert table.worst_clearance == pytest.approx(float(np.min(table.min_sdf)))
    assert table.worst_clearance > 0.0


def test_roundtrip(tmp_path, table):
    p = table.save(tmp_path / "t.json")
    back = load_idle_home_table(p, joint_names=JOINTS)
    assert np.allclose(back.q_arm, table.q_arm)
    assert np.allclose(back.tip_torso, table.tip_torso)
    assert resolve_idle_home(None) is None
    assert resolve_idle_home(str(p)) is not None


def test_old_yaw_indexed_file_is_rejected(tmp_path):
    """The format changed shape; a stale file must fail loudly, not silently."""
    import json

    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "joint_names": list(JOINTS),
        "q_arm": np.zeros((2, 5, _N_ARM)).tolist(),   # the yaw-INDEXED shape
    }))
    with pytest.raises(ValueError, match="yaw-INDEXED"):
        load_idle_home_table(p, joint_names=JOINTS)


def test_build_is_reproducible_and_clears():
    """A tiny build, so the solver itself is covered rather than just the asset."""
    cfg = IdleHomeConfig(n_yaw_checks=9, iters=120, restarts=2, seed=7)
    a = build_idle_home_table(cfg)
    b = build_idle_home_table(cfg)
    assert np.allclose(a.q_arm, b.q_arm)
    assert a.q_arm.shape == (2, _N_ARM)
    assert a.min_sdf.shape == (2, 9)
    assert a.worst_clearance > 0.0


# ============================================================ the cost ======
def _ctx(B: int, H: int, active: list[list[float]], dtype=torch.float64) -> StrikeContext:
    z = lambda *s: torch.zeros(*s, dtype=dtype)  # noqa: E731
    return StrikeContext(
        hit_index=torch.full((B, 2), H, dtype=torch.long),
        active=torch.tensor(active, dtype=dtype),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=z(B, 2, 3), v_star=z(B, 2, 3), n_out=z(B, 2, 3),
        hit_radius=torch.full((B, 2), 0.08, dtype=dtype),
        disk_offset=z(B, 2, 3), half_plane_dir=z(B, 2, 3), has_half_plane=z(B, 2),
        contact_sigma=torch.full((B, 2), 0.2, dtype=dtype),
        shaft_allowed=z(B, 2), start_shaft_allowed=z(B, 2), elev_required=z(B, 2),
    )


def _cost(table=None, **kw) -> FusedCost:
    c = FusedCost(CostConfig(**kw), max_horizon=12, dtype=torch.float64)
    if table is not None:
        c.set_idle_home(table)
    return c


def _flat(B: int, H: int, q0: np.ndarray):
    q = torch.as_tensor(np.repeat(np.asarray(q0, float)[None, :], B, axis=0))
    return q, q.unsqueeze(1).repeat(1, H, 1).clone(), torch.zeros(B, len(JOINTS), dtype=torch.float64)


def test_term_is_zero_without_a_table():
    q0, wp, v0 = _flat(2, 12, ready_joint_pos(JOINTS))
    out = _cost(w_idle_home=1.0, w_idle_vel=1.0).forward(q0, wp, v0, _ctx(2, 12, [[0., 0.], [1., 0.]]))
    assert out.raw["posture_idle_home"].abs().max().item() == 0.0
    assert out.raw["posture_idle_vel"].abs().max().item() == 0.0


def test_term_is_zero_at_zero_weight(table):
    """Bit-identical to the pre-idle cost, which is what protects checkpoints."""
    q0, wp, v0 = _flat(2, 12, ready_joint_pos(JOINTS))
    ctx = _ctx(2, 12, [[0.0, 0.0], [1.0, 0.0]])
    a = _cost(table).forward(q0, wp, v0, ctx)
    b = _cost().forward(q0, wp, v0, ctx)
    assert a.terms["posture"].allclose(b.terms["posture"])
    assert a.total.allclose(b.total)


def test_only_idle_hands_are_priced(table):
    q0, wp, v0 = _flat(3, 12, ready_joint_pos(JOINTS))
    ctx = _ctx(3, 12, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = _cost(table, w_idle_home=1.0).forward(q0, wp, v0, ctx)
    raw = out.raw["posture_idle_home"]
    assert raw[2].item() == 0.0, "both hands striking -> nothing idle to price"
    assert raw[0].item() > 0.0 and raw[1].item() > 0.0


def test_value_is_the_terminal_distance_to_home(table):
    """``|q_arm(K) - home|^2`` at the TERMINAL knot, like a strike's terminal.

    Not an average over the horizon: mid-horizon the arm is free, and only where
    it ends up is priced -- exactly how a striking hand is scored.
    """
    q0, wp, v0 = _flat(1, 12, ready_joint_pos(JOINTS))
    out = _cost(table, w_idle_home=1.0).forward(q0, wp, v0, _ctx(1, 12, [[1.0, 0.0]]))
    home = table.arm_pose("right")
    q_arm = ready_joint_pos(JOINTS)[arm_slice("right", JOINTS)]
    assert out.raw["posture_idle_home"].item() == pytest.approx(
        float(((q_arm - home) ** 2).sum()), rel=1e-9
    )


def test_only_the_terminal_knot_is_priced(table):
    """Moving a MID-horizon knot changes nothing; moving the last one does."""
    H = 12
    idx = arm_slice("right", JOINTS)
    base = ready_joint_pos(JOINTS)
    cost = _cost(table, w_idle_home=1.0)

    def val(shift):
        q0, wp, v0 = _flat(1, H, base)
        if shift is not None:
            wp[0, shift, idx] += 0.3
        out = cost.forward(q0, wp, v0, _ctx(1, H, [[1.0, 0.0]]))
        return out.raw["posture_idle_home"].item()

    ref = val(None)
    assert val(H // 2) == pytest.approx(ref, rel=1e-9), "mid-horizon must be free"
    assert abs(val(H - 1) - ref) > 1e-6, "the terminal knot must be priced"


def test_zero_at_home(table):
    q0 = ready_joint_pos(JOINTS)
    q0[arm_slice("left", JOINTS)] = table.arm_pose("left")
    q0_t, wp, v0 = _flat(1, 12, q0)
    out = _cost(table, w_idle_home=1.0).forward(q0_t, wp, v0, _ctx(1, 12, [[0.0, 1.0]]))
    assert out.raw["posture_idle_home"].item() == pytest.approx(0.0, abs=1e-18)


def test_gradient_reaches_only_the_idle_arm(table):
    """No yaw gradient: the bin index is a rounded division, deliberately."""
    q0, wp, v0 = _flat(1, 12, ready_joint_pos(JOINTS))
    wp.requires_grad_(True)
    out = _cost(table, w_idle_home=1.0).forward(q0, wp, v0, _ctx(1, 12, [[1.0, 0.0]]))
    g = torch.autograd.grad(out.raw["posture_idle_home"].sum(), wp)[0][0]
    assert g[:, :3].abs().max().item() == 0.0, "the waist must carry no idle gradient"
    assert g[:, arm_slice("left", JOINTS)].abs().max().item() == 0.0
    assert g[:, arm_slice("right", JOINTS)].abs().max().item() > 0.0


def test_gradient_points_at_home(table):
    """One step downhill moves the idle arm toward home."""
    q0, wp, v0 = _flat(1, 12, ready_joint_pos(JOINTS))
    wp.requires_grad_(True)
    cost = _cost(table, w_idle_home=1.0)
    out = cost.forward(q0, wp, v0, _ctx(1, 12, [[1.0, 0.0]]))
    g = torch.autograd.grad(out.raw["posture_idle_home"].sum(), wp)[0]
    stepped = (wp - 0.05 * g).detach()
    after = cost.forward(q0, stepped, v0, _ctx(1, 12, [[1.0, 0.0]]))
    assert after.raw["posture_idle_home"].item() < out.raw["posture_idle_home"].item()


def test_velocity_term_prices_the_END_of_the_horizon(table):
    """"final velocity of 0": the same |qd| costs far more at the terminal knot."""
    H = 12
    q0 = ready_joint_pos(JOINTS)
    idx = arm_slice("right", JOINTS)
    early, late = [], []
    for shift in (1, H - 1):
        _q, wp, v0 = _flat(1, H, q0)
        wp[0, shift, idx] += 0.2       # a one-knot bump = local velocity
        cost = _cost(table, w_idle_vel=1.0)
        q0_t = torch.as_tensor(q0[None, :])
        out = cost.forward(q0_t, wp, v0, _ctx(1, H, [[1.0, 0.0]]))
        (early if shift == 1 else late).append(out.raw["posture_idle_vel"].item())
    assert late[0] > 5.0 * early[0]


def test_velocity_term_is_zero_when_parked(table):
    q0 = ready_joint_pos(JOINTS)
    q0[arm_slice("right", JOINTS)] = table.arm_pose("right")
    q0_t, wp, v0 = _flat(1, 12, q0)
    out = _cost(table, w_idle_vel=1.0).forward(q0_t, wp, v0, _ctx(1, 12, [[1.0, 0.0]]))
    assert out.raw["posture_idle_vel"].item() == pytest.approx(0.0, abs=1e-20)


def test_the_term_is_exactly_yaw_independent(table):
    """The SAME arm pose costs the SAME at every waist yaw.

    Arm joints are torso-relative, so the park target does not move when the
    torso turns -- an idle arm genuinely sits still instead of tracking.  The
    earlier yaw-INDEXED design failed this on purpose; this one holds it by
    construction, because no waist column enters the term at all.
    """
    vals = []
    for yaw in (-1.2, 0.0, 1.2):
        q0 = ready_joint_pos(JOINTS)
        q0[JOINTS.index("waist_yaw_joint")] = yaw
        q0_t, wp, v0 = _flat(1, 12, q0)
        out = _cost(table, w_idle_home=1.0).forward(q0_t, wp, v0, _ctx(1, 12, [[1.0, 0.0]]))
        vals.append(out.raw["posture_idle_home"].item())
    assert max(vals) - min(vals) == pytest.approx(0.0, abs=1e-15)


def test_set_idle_home_validates_the_shape(table):
    with pytest.raises(ValueError, match=r"\(2, 7\)"):
        _cost().set_idle_home(table.q_arm[:, :3])
    assert _cost().set_idle_home(None).c_idle_q is None
    assert _cost().set_idle_home(table.q_arm).c_idle_q is not None


# ============================================================ the seed ======
def test_seed_parks_the_idle_arm_and_stops(table):
    lib = GoalPoseLibrary(joint_names=JOINTS)
    q0 = ready_joint_pos(JOINTS)
    q0[JOINTS.index("left_shoulder_pitch_joint")] = 0.6
    ws = warmstart_trajectory(
        q0, np.zeros(len(JOINTS)), horizon=25,
        goals={"left": None, "right": ("snare", 0.4)}, library=lib,
        idle_home=table, idle_settle_wp=10,
    )
    idx = arm_slice("left", JOINTS)
    home = table.arm_pose("left")
    assert np.abs(ws.q[-1, idx] - home).max() < 1e-9
    v = np.diff(ws.q, axis=0) / CONTROL_DT
    assert np.abs(v[-1, idx]).max() == pytest.approx(0.0, abs=1e-9), "arrives at rest"
    assert np.abs(v[-5:, idx]).max() == pytest.approx(0.0, abs=1e-9), "and sits there"
    assert ws.hit_index["left"] == -1


def test_seed_without_a_table_is_unchanged():
    lib = GoalPoseLibrary(joint_names=JOINTS)
    q0 = ready_joint_pos(JOINTS)
    q0[JOINTS.index("left_shoulder_pitch_joint")] = 0.6
    kw = dict(horizon=25, goals={"left": None, "right": ("snare", 0.4)}, library=lib)
    ws = warmstart_trajectory(q0, np.zeros(len(JOINTS)), **kw)
    idx = arm_slice("left", JOINTS)
    assert np.abs(ws.q[:, idx] - q0[idx]).max() == 0.0


def test_seed_respects_the_speed_rating(table):
    """A long retraction stretches the ramp instead of commanding a jump."""
    from amortized_opt.limits import (
        ETA_VELOCITY,
        isaac_velocity_limits,
        urdf_position_limits,
    )

    lib = GoalPoseLibrary(joint_names=JOINTS)
    v_lim = isaac_velocity_limits(JOINTS)
    lo, hi = urdf_position_limits(JOINTS)
    q0 = ready_joint_pos(JOINTS)
    idx = arm_slice("left", JOINTS)
    # inside the box: row 0 of the seed is pinned to ``q0`` unclamped by design,
    # so an out-of-box start would show up as a jump that is not the ramp's.
    q0[idx] = np.clip(table.arm_pose("left") + 1.2, lo[idx], hi[idx])
    ws = warmstart_trajectory(
        q0, np.zeros(len(JOINTS)), horizon=60,
        goals={"left": None, "right": ("snare", 0.5)}, library=lib,
        idle_home=table, idle_settle_wp=2,           # ask for an impossible ramp
    )
    peak = np.abs(np.diff(ws.q[:, idx], axis=0) / CONTROL_DT).max(axis=0)
    assert (peak <= ETA_VELOCITY * v_lim[idx] + 1e-6).all()


# ========================================================== the roller ======
def _roller_bits():
    """A small roller over a real scenario pool, on CPU."""
    import amortized_opt as A
    from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig
    from amortized_opt.phase2_data import (
        Phase2Pool,
        Phase2ScenarioConfig,
        generate_cost_scenarios,
    )
    from amortized_opt.train_cost import MPCRoller, Phase2Config, build_costs

    H = 25
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = Phase2ScenarioConfig(horizon=H, buckets=(H,), apex_init_frac=0.0)
    raw = generate_cost_scenarios(200, clips, seed=3, cfg=cfg)
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device="cpu")
    torch.manual_seed(0)
    model = A.build_model(A.ModelConfig(max_horizon=H), "cpu")
    p2 = Phase2Config(compile_mode=None, device="cpu")
    costs = build_costs(pool.buckets, p2.cost, device="cpu")
    sim = BatchedPDSim(PDSimConfig(dt=cfg.dt), device="cpu", dtype=torch.float32,
                       rnea=costs[H].rnea)
    return MPCRoller, pool, model, sim, costs[H], H


def test_roller_seeds_idle_arms_at_home(table):
    """Every idle hand's arm columns end at home, and hold there."""
    MPCRoller, pool, model, sim, cost, H = _roller_bits()
    r = MPCRoller(
        pool, model=model, sim=sim, rnea=cost.rnea, fk=cost.fk, horizon=H,
        slots=32, chain_frac=0.0, device="cpu", dtype=torch.float32, seed=0,
        idle_home=table, idle_settle_wp=8, rest_ticks=25,
    )
    r.tick()
    st = r.state
    idle = ~st["active"]
    assert bool(idle.any()), "the pool should contain single-hand-idle scenarios"
    for h in (0, 1):
        rows = idle[:, h].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        cols = r.arm_cols[h]
        want = r.idle_q[h].view(1, -1).expand(rows.numel(), -1)
        got = st["init"].index_select(0, rows)[:, -1, cols]     # last waypoint
        assert torch.allclose(got, want, atol=1e-5)


def test_roller_without_a_table_is_unchanged():
    MPCRoller, pool, model, sim, cost, H = _roller_bits()
    kw = dict(horizon=H, slots=16, chain_frac=0.0, device="cpu",
              dtype=torch.float32, seed=0)
    a = MPCRoller(pool, model=model, sim=sim, rnea=cost.rnea, fk=cost.fk, **kw)
    b = MPCRoller(pool, model=model, sim=sim, rnea=cost.rnea, fk=cost.fk,
                  idle_home=None, rest_ticks=0, **kw)
    for _ in range(2):
        sa, sb = a.tick(), b.tick()
    assert sa["idle_seed"] == 0.0
    assert torch.equal(a.state["init"], b.state["init"])


def test_rest_ticks_keep_a_both_idle_slot_alive(table):
    """The roller used to DELETE rest stretches; now it plays them out."""
    MPCRoller, pool, model, sim, cost, H = _roller_bits()
    kw = dict(pool=pool, model=model, sim=sim, rnea=cost.rnea, fk=cost.fk,
              horizon=H, slots=48, chain_frac=0.0, device="cpu",
              dtype=torch.float32, seed=1, idle_home=table)
    deleted = MPCRoller(**kw, rest_ticks=0)
    kept = MPCRoller(**kw, rest_ticks=25)
    # drive both into a rest and count how often a slot survives with no goal
    n_del = n_keep = 0
    for _ in range(12):
        deleted.tick()
        kept.tick()
        n_del += int((~deleted.state["active"].any(dim=1)).sum())
        n_keep += int((~kept.state["active"].any(dim=1)).sum())
    assert n_keep >= n_del


# =========================================================== phase 1 ========
# The warm start has to agree with phase 2 about what an idle arm does.  Before
# 2026-08-12 it did not: ``generate_scenarios`` called ``warmstart_trajectory``
# and ``build_context_inputs`` with no table, so the target held the idle arm at
# its start pose while phase 2 priced a retraction to home, and the encoder's
# IDLE token was masked out of every batch -- ``idle_proj`` and the 4th
# ``type_emb`` row reached phase 2 at exactly their random init.
def _phase1_pool(idle_home_path, *, n=900, seed=0, rest_frac=0.25, horizon=25,
                 buckets=None):
    import amortized_opt as A
    from amortized_opt.warmstart_data import NoiseConfig, ScenarioConfig, generate_pool

    cfg = ScenarioConfig(
        horizon=horizon, buckets=tuple(buckets or (horizon,)),
        idle_home_path=idle_home_path, rest_frac=rest_frac,
    )
    fk = ChainFK(device="cpu", dtype=torch.float32)
    rnea = A.BatchedRNEA.from_urdf(with_armature=True, device="cpu", dtype=torch.float32)
    # ``workers=1`` takes ``generate_pool_parallel``'s in-process branch.  The
    # pool workers fork, and the cost tests above this section have already run
    # autograd in this interpreter, which torch refuses to fork after.
    pool = generate_pool(n, seed=seed, cfg=cfg, noise=NoiseConfig(), split="train",
                         workers=1, device="cpu", fk=fk, rnea=rnea)
    return cfg, pool


@pytest.fixture(scope="module")
def p1_home(table):
    return _phase1_pool("default")


@pytest.fixture(scope="module")
def p1_none():
    return _phase1_pool(None)


def test_phase1_fills_the_idle_context_token(p1_home):
    """``idle_valid`` is the complement of ``active``, with a real payload."""
    _cfg, pool = p1_home
    for d in pool.data.values():
        assert torch.equal(d["idle_valid"], ~d["active"])
        idle = d["idle_valid"]
        assert bool(idle.any()), "pool should contain idle hands"
        # payload present exactly where valid, and exactly zero where not
        assert float(d["idle_home_q"][idle].abs().sum()) > 0.0
        assert float(d["idle_home_q"][~idle].abs().sum()) == 0.0
        assert float(d["idle_home_tip"][~idle].abs().sum()) == 0.0


def test_phase1_target_parks_the_idle_arm(p1_home, table):
    """The IMITATION TARGET ends at home -- not at the start pose."""
    _cfg, pool = p1_home
    seen = 0
    for d in pool.data.values():
        for hi, hand in enumerate(_HANDS):
            rows = (~d["active"][:, hi]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            seen += int(rows.numel())
            cols = list(arm_slice(hand, JOINTS))
            want = torch.as_tensor(table.arm_pose(hand), dtype=d["target"].dtype)
            got = d["target"][rows][:, -1, cols]
            assert torch.allclose(got, want.expand_as(got), atol=1e-5)
            # ...and it is a real journey, not a pose that was already there
            start = d["q0"][rows][:, cols]
            assert float((start - want).abs().max()) > 0.5
    assert seen > 0


def test_phase1_idle_retraction_respects_the_speed_rating(p1_home):
    """The settle window stretches, so a far park never outruns the joint."""
    _cfg, pool = p1_home
    v_lim = torch.as_tensor(isaac_velocity_limits(JOINTS), dtype=torch.float32)
    for d in pool.data.values():
        rate = torch.cat([d["q0"][:, None, :], d["target"]], 1).diff(dim=1).abs() / CONTROL_DT
        for hi, hand in enumerate(_HANDS):
            rows = (~d["active"][:, hi]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = list(arm_slice(hand, JOINTS))
            worst = float((rate[rows][:, :, cols] / v_lim[cols]).max())
            assert worst <= 1.0, f"{hand} idle retraction peaks at {worst:.2f}x its rating"


def test_phase1_without_a_table_holds_the_start_pose(p1_none):
    """``idle_home_path=None`` reproduces the pre-2026-08-12 pool exactly."""
    _cfg, pool = p1_none
    assert pool.stats["rest"] == 0, "rests need a table to park at"
    for d in pool.data.values():
        assert not bool(d["idle_valid"].any())
        assert float(d["idle_home_q"].abs().sum()) == 0.0
        for hi, hand in enumerate(_HANDS):
            rows = (~d["active"][:, hi]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = list(arm_slice(hand, JOINTS))
            held = d["target"][rows][:, :, cols]
            start = d["q0"][rows][:, None, cols]
            assert torch.allclose(held, start.expand_as(held), atol=1e-6)


def test_phase1_rests_are_kept_and_spread_across_buckets(table):
    """A both-idle draw is a rest, and it must not all land in one bucket."""
    _cfg, pool = _phase1_pool("default", n=1600, seed=5, rest_frac=1.0,
                              horizon=50, buckets=(25, 50))
    assert pool.stats["rest"] > 0
    assert pool.stats["reject_idle"] == 0, "rest_frac=1 keeps every rest"
    per_bucket = {
        b: int((~d["active"]).all(dim=1).sum()) for b, d in pool.data.items()
    }
    assert len(per_bucket) > 1
    assert all(v > 0 for v in per_bucket.values()), per_bucket
    # a rest has no hit, so it is not run-up-to-a-strike data
    assert pool.stats["short_horizon"] <= pool.stats["accepted"] - pool.stats["rest"]


def test_phase1_rest_frac_zero_rejects_every_rest(table):
    _cfg, pool = _phase1_pool("default", n=600, seed=7, rest_frac=0.0)
    assert pool.stats["rest"] == 0
    for d in pool.data.values():
        assert not bool((~d["active"]).all(dim=1).any())


def test_phase1_trains_the_idle_token(p1_home, p1_none):
    """THE regression this section exists for: a masked-out token gets no
    gradient, so without a payload ``idle_proj`` reaches phase 2 random."""
    import amortized_opt as A
    from amortized_opt import train_warmstart as TW

    def idle_grads(pool, cfg):
        torch.manual_seed(0)
        model = A.build_model(
            A.ModelConfig(max_horizon=cfg.horizon, num_drums=cfg.num_drums,
                          num_future=cfg.num_future),
            "cpu",
        )
        step = TW.make_step(model, TW.LossScales(device="cpu"), TW.LossWeights(),
                            compile_mode=None)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        for b, idx in TW.batch_sampler(pool, 128, rng=np.random.default_rng(0), steps=1):
            batch = pool.batch(b, idx, generator=gen)
        aux = batch["aux"]
        out = step(batch["init"], batch["state"], batch["context"], batch["masks"],
                   batch["target"],
                   *[aux[k] for k in ("vel", "acc", "torque", "fk_pos", "fk_vel")])
        (out[0] if isinstance(out, (tuple, list)) else out).backward()
        enc = model.encoder
        return (float(enc.idle_proj.weight.grad.abs().sum()),
                float(enc.type_emb.weight.grad[3].abs().sum()))

    proj_on, type_on = idle_grads(*reversed(p1_home))
    proj_off, type_off = idle_grads(*reversed(p1_none))
    assert proj_off == 0.0 and type_off == 0.0, "no payload must mean no gradient"
    assert proj_on > 0.0, "the idle projection must train when the pool feeds it"
    assert type_on > 0.0, "the IDLE type embedding must train too"
