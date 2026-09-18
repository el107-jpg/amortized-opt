"""Cross-hand (stick vs stick) collision: the primitive, the cost term, the guard.

Self-collision in this stack has always been *one hand* against the TORSO
capsule and the HEAD sphere -- in ``FusedCost._collision``, in
``SdfPenalty.forward`` and in ``TrajectoryScorer._self_sdf``.  The two hands'
geometry was never compared to each other, so nothing in the cost, the screen or
the eval harness ever objected to the two sticks passing through one another.

The term is a CAPSULE pair -- the whole 0.40 m stick as one segment -- built on
:func:`~amortized_opt.sdf.segment_segment_sdf`.  Per-hand drum and self
collision use the same capsule (see ``FusedCost._collision``); this file pins
the cross-hand pair.

What is pinned here:

1. the primitive against brute-force sampling, including the two degenerate
   families it has to survive (parallel and zero-length) and NaN-free gradients;
2. the cost term: a crossing pair prices, its gradient pushes the sticks apart,
   a separated pair prices ~0;
3. **the default is OFF, bit for bit** -- at ``w_stick_stick = 0`` no
   cross-hand quantity can reach the total, the augmented-Lagrangian violation,
   the peak group or ``collision_min_sdf``.  This is the regression guard for
   runs that are already in flight;
4. the roller's re-goal separation guard: a goal promoted onto the drum the
   other hand is playing has its contact point slid along the head instead of
   landing on top of the other bead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import (
    N_DOF,
    STICK_MIN_SEP,
    G1_UPPER_BODY_JOINTS,
)
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.sdf import segment_segment_sdf

DT = 0.02
H = 12
JOINT = {n: i for i, n in enumerate(G1_UPPER_BODY_JOINTS)}


# ======================================================== 1. the primitive ===
def _brute(a0, a1, b0, b1, n=513):
    """Min distance between two segments by dense sampling of both."""
    u = torch.linspace(0.0, 1.0, n, dtype=a0.dtype).unsqueeze(-1)
    A = a0 + u * (a1 - a0)
    B = b0 + u * (b1 - b0)
    return (A[:, None, :] - B[None, :, :]).norm(dim=-1).min()


@pytest.mark.parametrize(
    "name, a0, a1, b0, b1, want",
    [
        # parallel, overlapping projections: the perpendicular offset
        ("parallel", (0, 0, 0), (1, 0, 0), (0, 0.3, 0), (1, 0.3, 0), 0.3),
        # parallel, disjoint projections: end-to-end, through the offset
        ("parallel_offset", (0, 0, 0), (1, 0, 0), (2, 0.3, 0), (3, 0.3, 0),
         float(np.hypot(1.0, 0.3))),
        # collinear, disjoint: a pure gap
        ("collinear", (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), 1.0),
        # crossing in projection, separated in z: the z offset
        ("crossing", (0, 0, 0), (1, 0, 0), (0.5, -1, 0.05), (0.5, 1, 0.05), 0.05),
        # genuinely intersecting axes
        ("intersecting", (0, 0, 0), (1, 0, 0), (0.5, -1, 0), (0.5, 1, 0), 0.0),
        # zero-length vs segment == point-to-segment
        ("point_segment", (0.5, 0.4, 0), (0.5, 0.4, 0), (0, 0, 0), (1, 0, 0), 0.4),
        # zero-length vs zero-length == point-to-point
        ("point_point", (0, 0, 0), (0, 0, 0), (0, 0, 3), (0, 0, 3), 3.0),
        # both degenerate AND coincident
        ("coincident", (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), 0.0),
    ],
)
def test_primitive_analytic_cases(name, a0, a1, b0, b1, want):
    t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    got = segment_segment_sdf(t(a0), t(a1), t(b0), t(b1))
    assert float(got) == pytest.approx(want, abs=1e-8), name


def test_primitive_subtracts_both_radii():
    t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    d = segment_segment_sdf(
        t((0, 0, 0)), t((1, 0, 0)), t((0, 0.3, 0)), t((1, 0.3, 0)), 0.01, 0.02
    )
    assert float(d) == pytest.approx(0.3 - 0.03, abs=1e-9)


def test_primitive_matches_brute_force_random():
    """200 random pairs, batched, against dense sampling of both segments."""
    g = torch.Generator().manual_seed(11)
    n = 200
    a0 = torch.randn(n, 3, generator=g, dtype=torch.float64)
    a1 = a0 + torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.5
    b0 = torch.randn(n, 3, generator=g, dtype=torch.float64)
    b1 = b0 + torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.5
    got = segment_segment_sdf(a0, a1, b0, b1)
    ref = torch.stack([_brute(a0[i], a1[i], b0[i], b1[i], 2049) for i in range(n)])
    # The reference is a sampled minimum, so it can only OVERestimate.
    assert torch.all(got <= ref + 1e-12)
    assert float((ref - got).abs().max()) < 1e-5
    # ... and the primitive is symmetric in its two segments.
    assert torch.allclose(segment_segment_sdf(b0, b1, a0, a1), got, atol=1e-12)


def test_primitive_matches_brute_force_degenerate_families():
    """Nearly-parallel and nearly-zero-length pairs: the clamped paths."""
    g = torch.Generator().manual_seed(5)
    d1 = torch.randn(64, 3, generator=g, dtype=torch.float64)
    a0 = torch.randn(64, 3, generator=g, dtype=torch.float64)
    a1 = a0 + d1
    # b is a's direction plus a vanishing perturbation: den -> 0
    for eps in (0.0, 1e-12, 1e-8, 1e-4):
        pert = torch.randn(64, 3, generator=g, dtype=torch.float64) * eps
        b0 = a0 + torch.randn(64, 3, generator=g, dtype=torch.float64) * 0.3
        b1 = b0 + d1 + pert
        got = segment_segment_sdf(a0, a1, b0, b1)
        ref = torch.stack([_brute(a0[i], a1[i], b0[i], b1[i], 2049) for i in range(64)])
        assert torch.isfinite(got).all()
        # 1e-4 rather than 1e-5: ``den = A*E - B*B`` cancels catastrophically as
        # the pair goes parallel, which costs ~10 um of accuracy in the reported
        # distance -- three orders below the 10 mm bead this term protects.
        assert float((ref - got).abs().max()) < 1e-4, f"parallel eps={eps}"
        assert torch.allclose(
            segment_segment_sdf(b0, b1, a0, a1), got, atol=1e-12
        ), f"asymmetric at parallel eps={eps}"
    # b shrinks to a point: A or E -> 0
    for L in (0.0, 1e-14, 1e-6):
        b0 = torch.randn(64, 3, generator=g, dtype=torch.float64)
        b1 = b0 + torch.randn(64, 3, generator=g, dtype=torch.float64) * L
        got = segment_segment_sdf(a0, a1, b0, b1)
        ref = torch.stack([_brute(a0[i], a1[i], b0[i], b1[i], 2049) for i in range(64)])
        assert torch.isfinite(got).all()
        assert float((ref - got).abs().max()) < 1e-5, f"degenerate L={L}"
        # The same pair with the DEGENERATE segment FIRST.  This is the
        # asymmetry the branchy transcription of Ericson had: with ``b``
        # collapsed to a point it returned ``|a0 - b0|``, never the foot of the
        # perpendicular, and the two argument orders disagreed by metres.
        assert torch.allclose(
            segment_segment_sdf(b0, b1, a0, a1), got, atol=1e-9
        ), f"asymmetric at L={L}"


def test_primitive_gradients_are_finite_everywhere():
    """No NaN gradient from the clamped/degenerate paths, coincidence included."""
    cases = [
        ((0, 0, 0), (1, 0, 0), (0, 0.3, 0), (1, 0.3, 0)),   # parallel
        ((0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0)),       # identical
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),       # all degenerate
        ((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)),       # two points
        ((0, 0, 0), (1, 0, 0), (0.5, -1, 0), (0.5, 1, 0)),  # intersecting axes
    ]
    for c in cases:
        xs = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in c]
        segment_segment_sdf(*xs).backward()
        for x in xs:
            assert torch.isfinite(x.grad).all(), c


def test_primitive_broadcasts():
    a0 = torch.zeros(4, 7, 3, dtype=torch.float64)
    a1 = a0 + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    b0 = a0 + torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
    b1 = b0 + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = segment_segment_sdf(a0, a1, b0, b1, 0.01, 0.01)
    assert out.shape == (4, 7)
    assert torch.allclose(out, torch.full_like(out, 0.48))


# ========================================================= 2. the cost term ===
def _ctx(B, *, dtype=torch.float64):
    """The minimal synthetic context ``test_collision_peak.py`` uses."""
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


#: Both arms out front, elbows bent: the two sticks meet at the midline.  The
#: extra left shoulder YAW takes the pair off exact axis-intersection, so the
#: configuration is a real crossing rather than a measure-zero degeneracy --
#: measured cross-hand SDF ~ -14 mm (the sticks are 8 mm shafts with 10 mm
#: beads, so this is deep interpenetration).
CROSSING = {
    JOINT["left_shoulder_pitch_joint"]: -1.2,
    JOINT["right_shoulder_pitch_joint"]: -1.2,
    JOINT["left_elbow_joint"]: 0.97,
    JOINT["right_elbow_joint"]: 0.97,
    JOINT["left_shoulder_yaw_joint"]: 0.15,
}
#: The same pose with both shoulders rolled outward: ~150 mm of clearance.
SEPARATED = dict(CROSSING)
SEPARATED[JOINT["left_shoulder_yaw_joint"]] = 0.0
SEPARATED[JOINT["left_shoulder_roll_joint"]] = -0.4
SEPARATED[JOINT["right_shoulder_roll_joint"]] = 0.4


def _pose(spec: dict, B: int = 2):
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    for j, v in spec.items():
        q0[:, j] = v
    return q0, q0.unsqueeze(1).repeat(1, H, 1)


def _cost(**kw):
    return FusedCost(CostConfig(**kw), max_horizon=H, device="cpu",
                     dtype=torch.float64, dt=DT)


def _fwd(cost, q0, wp, **kw):
    return cost.forward(q0, wp, torch.zeros_like(q0), _ctx(q0.shape[0]), **kw)


def test_crossing_sticks_are_detected():
    q0, wp = _pose(CROSSING)
    out = _fwd(_cost(w_stick_stick=100.0), q0, wp)
    assert float(out.diagnostics["collision_stick_min_sdf"].min()) < 0.0
    assert float(out.raw["collision_stick"].min()) > 0.0
    assert float(out.diagnostics["collision_stick"].min()) > 0.0


def test_separated_sticks_price_nothing():
    q0, wp = _pose(SEPARATED)
    out = _fwd(_cost(w_stick_stick=100.0), q0, wp)
    assert float(out.diagnostics["collision_stick_min_sdf"].min()) > 0.05
    assert float(out.raw["collision_stick"].max()) < 1e-8


def _gap(cost: FusedCost, q: torch.Tensor) -> torch.Tensor:
    """(B,) cross-hand clearance for a batch of POSES, re-derived from FK.

    Independent of ``_collision``'s bookkeeping on purpose: this is the
    ground truth the term is asserted to move.
    """
    frames = cost.fk.forward(q)
    P = cost.n_points
    pl = cost._points_world(frames, "left").reshape(-1, P, 3)
    pr = cost._points_world(frames, "right").reshape(-1, P, 3)
    rs, rw = cost.c_stick_r, cost.c_wrist_r
    return torch.minimum(
        torch.minimum(
            segment_segment_sdf(pl[:, 1], pl[:, 0], pr[:, 1], pr[:, 0], rs, rs),
            segment_segment_sdf(pl[:, -1], pl[:, -1], pr[:, 1], pr[:, 0], rw, rs),
        ),
        torch.minimum(
            segment_segment_sdf(pr[:, -1], pr[:, -1], pl[:, 1], pl[:, 0], rw, rs),
            segment_segment_sdf(pl[:, -1], pl[:, -1], pr[:, -1], pr[:, -1], rw, rw),
        ),
    )


def test_gradient_pushes_the_sticks_apart():
    """One descent step on the cross-hand term alone must open the gap.

    Measured on the WAYPOINTS, not on ``collision_stick_min_sdf``: the
    trajectory's first sub-sample is ``q0``, which no waypoint gradient can
    move, so the reported minimum is pinned by the boundary condition.
    """
    cost = _cost(w_stick_stick=100.0)
    q0, wp = _pose(CROSSING)
    wp = wp.clone().requires_grad_(True)
    out = _fwd(cost, q0, wp)
    pen_before = float(out.raw["collision_stick"].sum())
    gap_before = _gap(cost, wp.detach().reshape(-1, N_DOF))
    assert float(gap_before.max()) < 0.0        # the sticks really do cross

    out.raw["collision_stick"].sum().backward()
    g = wp.grad
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 1e-4
    with torch.no_grad():
        wp2 = wp - (0.02 / float(g.abs().max())) * g
    after = _fwd(cost, q0, wp2)
    assert float(after.raw["collision_stick"].sum()) < 0.5 * pen_before
    gap_after = _gap(cost, wp2.reshape(-1, N_DOF))
    assert torch.all(gap_after > gap_before)


def test_gradient_is_negligible_when_the_sticks_are_clear():
    """The softplus tail, not a hard zero -- 8 decades below the crossing case."""
    cost = _cost(w_stick_stick=100.0)
    q0, wp = _pose(SEPARATED)
    wp = wp.clone().requires_grad_(True)
    out = _fwd(cost, q0, wp)
    out.raw["collision_stick"].sum().backward()
    assert float(wp.grad.abs().max()) < 1e-7


def test_term_reaches_the_total_only_when_priced():
    q0, wp = _pose(CROSSING)
    # λ_collision=0 so the delta is only the softplus mean (pricing stick also
    # folds its max ReLU into collision live, which default λ=1 would add).
    off = _fwd(_cost(), q0, wp, weights={"collision": 0.0})
    on = _fwd(_cost(w_stick_stick=100.0), q0, wp, weights={"collision": 0.0})
    assert float(on.total.sum()) > float(off.total.sum())
    delta = (on.terms["collision"] - off.terms["collision"])
    assert torch.allclose(delta, 100.0 * off.raw["collision_stick"], atol=1e-12)


# ============================================ 3. the default is OFF, exactly ===
def _random_batch(B=6, seed=17):
    g = torch.Generator().manual_seed(seed)
    q0 = torch.randn(B, N_DOF, generator=g, dtype=torch.float64) * 0.4
    wp = q0.unsqueeze(1) + torch.randn(
        B, H, N_DOF, generator=g, dtype=torch.float64
    ) * 0.15
    return q0, wp


def _flat(out):
    """Every number a training step reads, as one sorted list of (key, tensor)."""
    items = [("total", out.total)]
    for name, d in (("term", out.terms), ("raw", out.raw), ("viol", out.violations),
                    ("diag", out.diagnostics), ("count", out.counts)):
        items += [(f"{name}:{k}", v) for k, v in d.items()]
    return dict(items)


@pytest.mark.parametrize("peak_w", [0.0, 500.0])
def test_default_off_is_bit_for_bit(monkeypatch, peak_w):
    """At ``w_stick_stick = 0`` no cross-hand number can reach anything shared.

    Proved by SABOTAGE rather than by comparison with an old build: the
    cross-hand primitive is monkeypatched to report a metre of interpenetration
    on every pair, which would swamp the collision mean, the peak smooth-max,
    the augmented-Lagrangian violation and ``collision_min_sdf`` if any of them
    were wired to it.  Stick-vs-torso uses the same geometry function under a
    different name, so the sabotage does not also move ``collision_self``.
    Everything except the two cross-hand diagnostics must come back bit for
    bit.  This is the guard for runs already in flight.
    """
    import amortized_opt.cost as C

    q0, wp = _random_batch()
    cost = _cost(w_collision_peak=peak_w)
    clean = _flat(_fwd(cost, q0, wp))

    real = C.cross_hand_sdf
    monkeypatch.setattr(
        C, "cross_hand_sdf",
        lambda a0, a1, b0, b1, ra=0.0, rb=0.0: real(a0, a1, b0, b1, ra, rb) - 1.0,
    )
    dirty = _flat(_fwd(cost, q0, wp))

    stick_only = {"raw:collision_stick", "diag:collision_stick",
                  "diag:collision_stick_min_sdf", "count:collision_stick"}
    assert set(clean) == set(dirty)
    for k in clean:
        if k in stick_only:
            continue
        assert torch.equal(clean[k], dirty[k]), f"{k} moved at w_stick_stick=0"
    # the sabotage really did land somewhere
    assert not torch.equal(clean["raw:collision_stick"], dirty["raw:collision_stick"])


def test_priced_term_joins_the_shared_reductions():
    """The mirror image: at a positive weight the peak / violation DO see it."""
    q0, wp = _pose(CROSSING)
    off = _fwd(_cost(w_collision_peak=500.0), q0, wp)
    on = _fwd(_cost(w_collision_peak=500.0, w_stick_stick=100.0), q0, wp)
    assert float(on.raw["collision_peak"].max()) > float(off.raw["collision_peak"].max())
    assert float(on.violations["collision"].max()) >= float(
        off.violations["collision"].max()
    )
    assert float(on.diagnostics["collision_min_sdf"].min()) <= float(
        off.diagnostics["collision_min_sdf"].min()
    )


def test_default_weight_is_zero_and_metrics_are_exported():
    from amortized_opt.train_cost import DIAG_KEYS, DIAG_NAMES, METRIC_NAMES

    assert CostConfig().w_stick_stick == 0.0
    assert DIAG_NAMES[DIAG_KEYS.index("collision_stick")] == "d_c_stick"
    assert DIAG_NAMES[DIAG_KEYS.index("collision_stick_min_sdf")] == "d_stick_sdf"
    assert "d_c_stick" in METRIC_NAMES and "d_stick_sdf" in METRIC_NAMES
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))


def test_negative_weight_rejected():
    with pytest.raises(ValueError, match="w_stick_stick"):
        CostConfig(w_stick_stick=-1.0).validate()


# ================================================ 4. the re-goal separation ===
def test_shared_min_separation_constant():
    from amortized_opt.phase2_data import Phase2ScenarioConfig

    assert Phase2ScenarioConfig().pair_same_min_sep == STICK_MIN_SEP


HORIZON = 25


@pytest.fixture(scope="module")
def roller():
    """One :class:`MPCRoller` on the CPU, two slots, re-goal on."""
    import amortized_opt as A
    from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig
    from amortized_opt.phase2_data import (
        Phase2Pool,
        Phase2ScenarioConfig,
        generate_cost_scenarios,
    )
    from amortized_opt.rhythm import DrumKit
    from amortized_opt.train_cost import MPCRoller, Phase2Config, build_costs

    kit = DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = Phase2ScenarioConfig(horizon=HORIZON, buckets=(HORIZON,))
    raw = generate_cost_scenarios(120, clips, seed=3, cfg=cfg)
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device="cpu")
    torch.manual_seed(0)
    model = A.build_model(A.ModelConfig(max_horizon=HORIZON), "cpu")
    costs = build_costs(pool.buckets, Phase2Config(compile_mode=None,
                                                   device="cpu").cost, device="cpu")
    sim = BatchedPDSim(PDSimConfig(dt=cfg.dt), device="cpu", dtype=torch.float32,
                       rnea=costs[HORIZON].rnea)
    return MPCRoller(
        pool, model=model, sim=sim, rnea=costs[HORIZON].rnea, fk=costs[HORIZON].fk,
        horizon=HORIZON, slots=2, chain_frac=0.0, device="cpu",
        dtype=torch.float32, seed=0, regoal=True,
    )


def _stage_same_drum(r, offset_m: float):
    """LEFT playing ``snare``; RIGHT fires now, its next future on the same head.

    Returns ``(drum, left point, right point, axis, disk centre, hit radius)``,
    all env-frame points and pelvis-frame geometry, as the roller stores them.
    """
    st = r.state
    d = r.kit_names.index("snare")
    pelvis = r.pool._pelvis
    axis = r.geom["axis"][:, d]                      # (R, 3) unit head normal
    c = r.geom["disk_center"][:, d]                  # (R, 3) PELVIS frame
    hit_r = r.geom["hit_radius"][:, d]               # (R,)
    # any in-plane unit direction on the head
    seed = torch.zeros_like(axis)
    seed[:, 0] = 1.0
    seed = torch.where(axis[:, 0:1].abs() > 0.9, torch.roll(seed, 1, dims=1), seed)
    u = seed - (seed * axis).sum(-1, keepdim=True) * axis
    u = u / u.norm(dim=-1, keepdim=True)
    p_left = c + pelvis                              # env frame, disk centre
    p_right = c + offset_m * u + pelvis

    st["active"][:, 0] = True
    st["hit"][:, 0] = 10
    st["drum"][:, 0] = d
    st["goal_drum"][:, 0] = d
    st["goal_valid"][:, 0] = True
    st["goal_pos"][:, 0] = p_left
    st["goal_vel"][:, 0] = -2.0 * axis
    st["goal_tth"][:, 0] = 10 * r.dt
    st["future_valid"][:, 0] = False

    st["active"][:, 1] = True
    st["hit"][:, 1] = 0                              # fires this tick
    st["drum"][:, 1] = d
    st["goal_valid"][:, 1] = True
    st["goal_tth"][:, 1] = 0.0
    st["future_valid"][:, 1] = False
    st["future_valid"][:, 1, 0] = True
    st["future_drum"][:, 1, 0] = d
    st["future_pos"][:, 1, 0] = p_right
    st["future_vel"][:, 1, 0] = -2.0 * axis
    st["future_gap"][:, 1, 0] = 0.2                  # 10 waypoints from now
    return d, p_left, p_right, axis, c, hit_r


def test_regoal_pushes_a_colliding_promotion_apart(roller):
    """A promotion 15 mm from the other bead comes out >= STICK_MIN_SEP away."""
    r = roller
    d, p_left, p_right, axis, c, hit_r = _stage_same_drum(r, 0.015)
    assert float((p_right - p_left).norm(dim=-1).max()) < STICK_MIN_SEP  # the bug case

    _fired, stats = r._regoal()
    st = r.state

    assert stats["sep_push"] == float(st["hit"].shape[0])
    assert bool(st["goal_valid"][:, 1].all())        # the goal was KEPT
    assert int(st["drum"][0, 1]) == d
    # the left hand was not touched
    assert torch.allclose(st["goal_pos"][:, 0], p_left, atol=0.0)
    # separated, on the head, inside the legal hit disk
    got = st["goal_pos"][:, 1]
    assert float((got - p_left).norm(dim=-1).min()) >= STICK_MIN_SEP - 1e-5
    rel = got - r.pool._pelvis - c
    axial = (rel * axis).sum(-1)
    inplane = (rel - axial.unsqueeze(-1) * axis).norm(dim=-1)
    assert float(inplane.max()) <= float(hit_r.max()) + 1e-6
    # the SLIDE keeps the promoted point's own axial offset (it stays on the
    # head, it is not lifted off it)
    axial0 = ((p_right - r.pool._pelvis - c) * axis).sum(-1)
    assert torch.allclose(axial, axial0, atol=1e-6)
    # the re-priced strike row followed the moved point
    assert torch.allclose(st["p_star"][:, 1], got - r.pool._pelvis, atol=1e-6)


def test_regoal_leaves_a_well_separated_promotion_bit_for_bit(roller):
    """Same drum, 120 mm apart: nothing moves, and nothing is even rewritten."""
    r = roller
    _d, p_left, p_right, _axis, _c, _hit_r = _stage_same_drum(r, 0.12)
    _fired, stats = r._regoal()
    assert stats["sep_push"] == 0.0
    assert torch.equal(r.state["goal_pos"][:, 1], p_right)
    assert torch.equal(r.state["goal_pos"][:, 0], p_left)


def test_regoal_ignores_the_other_hand_on_a_different_drum(roller):
    """Close points on DIFFERENT drums are not this guard's business."""
    r = roller
    _d, p_left, p_right, _axis, _c, _hit_r = _stage_same_drum(r, 0.015)
    r.state["drum"][:, 0] = r.kit_names.index("tom1")
    r.state["goal_drum"][:, 0] = r.kit_names.index("tom1")
    _fired, stats = r._regoal()
    assert stats["sep_push"] == 0.0
    assert torch.equal(r.state["goal_pos"][:, 1], p_right)
