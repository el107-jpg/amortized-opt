"""Per-drum RADIUS domain randomization, end to end (2026-08-14).

Radius was the last drum feature that never varied: ``drum_features`` carries it
(1 of ``DRUM_FEAT_DIM = 9``), but every training sample gave a drum id its
nominal radius, so radius was perfectly confounded with drum identity and the
weights reading that token were UNIDENTIFIED -- the ``yaw_now`` failure class,
where an input that never varies independently is read in an arbitrary,
untrained way the moment deployment presents a novel value.

Randomizing it is only half the fix.  The other half is FIX B (see
``StrikeContext.drum_head_centers``): everything that PRICES the radius has to
read the randomized value, or the objective and the scene disagree.  Three
consumers, one number:

* the drum TOKEN            -- ``ContextInputs.drum_radius``
* the STRIKE row            -- ``hit_radius`` (``strike_aim_on_head``)
* the COLLISION cylinder    -- ``StrikeContext.drum_radii``

The end-to-end test below is the one that matters: it asserts all three come
from the same draw, per sample, on a real phase-2 pool.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import TIP_RADIUS, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK
from amortized_opt.phase2_data import (
    Phase2ScenarioConfig,
    generate_cost_scenarios,
    Phase2Pool,
)
from amortized_opt.rhythm import DrumKit, KitRandomization
from amortized_opt.strike import strike_aim_on_head

HANDS = ("left", "right")
FRAC = 0.15


# ------------------------------------------------------------ the draw ------
def test_radii_are_per_drum_and_inside_the_fraction():
    kit = DrumKit.from_layout()
    rng = np.random.default_rng(0)
    cfg = KitRandomization(per_drum_radius=FRAC)

    seen = []
    for _ in range(64):
        k = kit.randomized(rng, cfg)
        ratio = np.asarray(k.radii, float) / np.asarray(kit.radii, float)
        assert ratio.min() >= 1.0 - FRAC - 1e-12
        assert ratio.max() <= 1.0 + FRAC + 1e-12
        # INDEPENDENT per drum -- a single global scale would be a second
        # "one number for the whole kit" confound, not a fix for the first.
        assert ratio.std() > 1e-3, ratio
        seen.append(ratio)

    seen = np.stack(seen)
    # ...and independent across samples, per drum.
    assert seen.std(axis=0).min() > 1e-2
    # The heights, which are NOT randomized, are untouched.
    assert np.array_equal(kit.randomized(rng, cfg).heights, kit.heights)


def test_zero_fraction_leaves_the_rng_stream_untouched():
    """``per_drum_radius = 0`` is the pre-2026-08-14 kit, bit for bit.

    The draw happens only when the knob is on (the cold-start rule), so a pool
    generated before radius randomization existed reproduces exactly.
    """
    kit = DrumKit.from_layout()
    old = KitRandomization()                       # what every run used to pass
    new = KitRandomization(per_drum_radius=0.0)
    for seed in (0, 1, 2):
        a = kit.randomized(np.random.default_rng(seed), old)
        b = kit.randomized(np.random.default_rng(seed), new)
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.axes, b.axes)
        assert np.array_equal(a.radii, b.radii)
        assert np.array_equal(a.radii, kit.radii)


def test_a_fraction_of_one_or_more_is_rejected():
    with pytest.raises(ValueError):
        KitRandomization(per_drum_radius=1.0)
    with pytest.raises(ValueError):
        KitRandomization(per_drum_radius=-0.1)


# --------------------------------------------------- end-to-end agreement ---
@pytest.fixture(scope="module")
def randomized_pool():
    """A small phase-2 pool with radius randomization ON."""
    kit = DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = Phase2ScenarioConfig(
        horizon=25, buckets=(25,),
        kit_randomization=KitRandomization(per_drum_radius=FRAC),
        cold_start_frac=0.0,
    )
    raw = generate_cost_scenarios(24, clips, seed=3, cfg=cfg, kit=kit)
    if not raw["buckets"]:
        pytest.skip("no scenario survived the acceptance filters")
    pool = Phase2Pool(raw["buckets"], cfg=cfg, dtype=torch.float64)
    return kit, cfg, pool


def test_token_strike_and_collision_scene_share_one_radius(randomized_pool):
    """THE fix-B invariant, per sample: one draw, three consumers."""
    kit, cfg, pool = randomized_pool
    b = pool.buckets[0]
    n = pool.size(b)
    idx = torch.arange(n, device=pool.device)
    batch = pool.batch(b, idx, noise=False)
    ctx, strike = batch["context"], batch["strike"]

    # 1. the COLLISION scene is the drum token, not a parallel copy
    assert strike.drum_radii is not None, "the pool must carry per-sample radii"
    assert torch.equal(strike.drum_radii, ctx.drum_radius)

    # 2. the radii really were randomized (else this test proves nothing)
    D = kit.num_drums
    nominal = torch.as_tensor(kit.radii, dtype=ctx.drum_radius.dtype)
    r = ctx.drum_radius[:, :D]
    assert float((r - nominal).abs().max()) > 1e-3
    assert float((r / nominal - 1.0).abs().max()) <= FRAC + 1e-9

    # 3. the STRIKE row's hit radius comes from the SAME number.  The map from
    #    head radius to legal disk is ``strike_aim_on_head`` (full radius on a
    #    cymbal, rim-buffered elsewhere), so it is re-derived here from the
    #    token rather than assumed proportional.
    hit = strike.hit_radius
    active = strike.active > 0.5
    checked = 0
    for i in range(n):
        for hi in range(2):
            if not bool(active[i, hi]):
                continue
            d = int(strike.target_drum[i, hi])
            assert d >= 0
            name = kit.names[d]
            _aim, want = strike_aim_on_head(
                np.zeros(3), np.array([0.0, 0.0, 1.0]),
                float(ctx.drum_radius[i, d]), name=name,
            )
            assert float(hit[i, hi]) == pytest.approx(float(want), abs=1e-6), (
                f"sample {i} hand {hi} drum {name}: strike row priced "
                f"{float(hit[i, hi]):.6f} against token radius "
                f"{float(ctx.drum_radius[i, d]):.6f}"
            )
            checked += 1
    assert checked > 0, "no active strike in the pool -- nothing was checked"


def test_the_goal_point_stays_on_the_randomized_head(randomized_pool):
    """A shrunken drum's aim point is inside the SHRUNKEN legal disk.

    ``sample_hit_point`` jitters about the aim and CLAMPS into the legal disk of
    the kit it was handed, so a goal built against the nominal radius on a drum
    that shrank would sit outside the disk the terminal term then prices --
    the same disagreement as the collision one, on the aim side.
    """
    _kit, _cfg, pool = randomized_pool
    b = pool.buckets[0]
    idx = torch.arange(pool.size(b), device=pool.device)
    batch = pool.batch(b, idx, noise=False)
    strike = batch["strike"]
    active = strike.active > 0.5
    # ``disk_offset`` is p_star -> disk centre, in the head plane; its length is
    # the aim's in-plane distance from the centre, which is what
    # ``FusedCost._terminal`` measures against ``hit_radius``.
    off = strike.disk_offset.norm(dim=-1)
    for i in range(idx.numel()):
        for hi in range(2):
            if not bool(active[i, hi]):
                continue
            assert float(off[i, hi]) <= float(strike.hit_radius[i, hi]) + 1e-6


# ------------------------------------------------ the cost actually moves ---
@pytest.fixture(scope="module")
def cost64():
    return FusedCost(CostConfig(), max_horizon=8, device="cpu", dtype=torch.float64)


def _ctx_at(cost, heads, radii, dtype):
    """A one-row context with NOTHING exempt, so every drum is priced."""
    B, D = 1, heads.shape[1]
    z = lambda *s: torch.zeros(*s, device=cost.device, dtype=dtype)  # noqa: E731
    return StrikeContext(
        hit_index=torch.full((B, 2), cost.max_horizon, device=cost.device,
                             dtype=torch.long),
        active=z(B, 2),
        target_drum=torch.full((B, 2), -1, device=cost.device, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, device=cost.device, dtype=torch.long),
        p_star=z(B, 2, 3), v_star=z(B, 2, 3),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=cost.device, dtype=dtype)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, device=cost.device, dtype=dtype),
        disk_offset=z(B, 2, 3), half_plane_dir=z(B, 2, 3), has_half_plane=z(B, 2),
        contact_sigma=torch.full((B, 2), 0.20, device=cost.device, dtype=dtype),
        shaft_allowed=z(B, 2), start_shaft_allowed=z(B, 2), elev_required=z(B, 2),
        drum_head_centers=heads, drum_radii=radii,
    )


def _tip_and_scene(cost):
    """Ready pose, plus a scene with ONE drum wrapped around the left tip.

    The drum is placed so the tip sits at the cylinder's axial midpoint at a
    chosen in-plane distance ``rho`` from the axis; every other drum is pushed
    10 m away so the reported penetration is unambiguously this one's.  Placing
    the DRUM rather than the point is what makes this a test of the collision
    pathway instead of a test of ``point_cylinder_sdf``.
    """
    dtype = cost.dtype
    q0 = torch.as_tensor(ready_joint_pos(cost.joint_names), device=cost.device,
                         dtype=dtype).unsqueeze(0)
    frames = cost.fk.forward(q0)
    R, p = frames["left_wrist"]
    tip = ChainFK.points_world(R, p, cost.c_points["left"][:1])[0, 0]   # pelvis frame
    return q0, tip


def _pen_max(cost, q0, heads, radii):
    H = cost.max_horizon
    wp = q0.unsqueeze(1).expand(-1, H, -1).contiguous()
    v0 = torch.zeros_like(q0)
    out = cost.forward(q0, wp, v0, _ctx_at(cost, heads, radii, cost.dtype))
    return float(out.diagnostics["collision_pen_max_left"][0])


@pytest.mark.parametrize(
    "rho_frac, scale, want_pen",
    [
        # just INSIDE the nominal rim, on a drum that SHRANK -> now clear
        (0.90, 0.6, False),
        (0.90, 1.0, True),
        # just OUTSIDE the nominal rim, on a drum that GREW -> now penetrating
        (1.10, 1.4, True),
        (1.10, 1.0, False),
    ],
)
def test_collision_prices_the_randomized_radius(cost64, rho_frac, scale, want_pen):
    cost = cost64
    q0, tip = _tip_and_scene(cost)
    D = cost.n_drums
    dtype = cost.dtype

    d = 0                                    # the drum we wrap around the tip
    r_nom = float(cost.scene.radii[d])
    half_h = float(cost.c_half_h[d])
    axis = cost.c_axes[d]
    # An in-plane unit vector, so the tip's radial coordinate is exactly rho.
    e = torch.tensor([1.0, 0.0, 0.0], device=cost.device, dtype=dtype)
    e = e - (e @ axis) * axis
    e = e / e.norm()
    rho = rho_frac * r_nom
    heads = torch.full((1, D, 3), 10.0, device=cost.device, dtype=dtype)
    heads[0, d] = tip + half_h * axis + rho * e

    radii = torch.as_tensor(cost.scene.radii, device=cost.device,
                            dtype=dtype).unsqueeze(0).clone()
    radii[0, d] *= scale
    pen = _pen_max(cost, q0, heads, radii)

    # The tip is a SPHERE of ``TIP_RADIUS``, so "clear" means clear of the
    # cylinder wall by more than the bead; the parametrization above leaves
    # that margin on both sides.
    margin = abs(rho - scale * r_nom) - TIP_RADIUS
    assert margin > 1e-3, "the case is too tight to read -- fix the parameters"
    if want_pen:
        assert pen > 1e-3, f"expected penetration, got {pen:.6f}"
    else:
        assert pen == 0.0, f"expected clearance, got penetration {pen:.6f}"


def test_passing_the_nominal_radii_reproduces_the_baked_in_scene(cost64):
    """The per-sample path is a no-op on nominal radii (bit-identical).

    The pool ALWAYS supplies ``drum_radii`` (as it always supplies
    ``drum_head_centers``), so ``per_drum_radius = 0`` must leave the cost
    unchanged -- otherwise the knob's default is not free.
    """
    cost = cost64
    q0, tip = _tip_and_scene(cost)
    D = cost.n_drums
    heads = torch.as_tensor(cost.scene.head_centers, device=cost.device,
                            dtype=cost.dtype).unsqueeze(0)
    radii = torch.as_tensor(cost.scene.radii, device=cost.device,
                            dtype=cost.dtype).unsqueeze(0)
    H = cost.max_horizon
    wp = q0.unsqueeze(1).expand(-1, H, -1).contiguous()
    v0 = torch.zeros_like(q0)
    a = cost.forward(q0, wp, v0, _ctx_at(cost, heads, None, cost.dtype))
    b = cost.forward(q0, wp, v0, _ctx_at(cost, heads, radii, cost.dtype))
    assert torch.equal(a.raw["collision_drum"], b.raw["collision_drum"])
    assert torch.equal(a.raw["collision_under"], b.raw["collision_under"])
    assert torch.equal(a.total, b.total)
