"""Idle safety cylinder: library, tip hinge cost, and warmstart park.

Replaces the joint-home coverage in ``test_idle_home.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS as JOINTS
from amortized_opt.constants import ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK
from amortized_opt.idle_cylinder import (
    default_idle_cylinder_path,
    derive_cylinder,
    load_idle_cylinder_library,
    nearest_arm_pose,
    tip_in_torso,
    tip_inside,
)
from amortized_opt.limits import urdf_position_limits
from amortized_opt.rhythm import DrumKit
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, warmstart_trajectory

_N_ARM = 7
_HANDS = ("left", "right")


@pytest.fixture(scope="module")
def library():
    if not default_idle_cylinder_path().exists():
        pytest.skip("idle-cylinder asset not built (scripts/build_idle_cylinder_library.py)")
    return load_idle_cylinder_library()


@pytest.fixture(scope="module")
def fk():
    return ChainFK(None, joint_names=JOINTS, dtype=torch.float64)


@pytest.fixture(scope="module")
def cylinder(fk):
    return derive_cylinder(DrumKit.from_layout(), fk)


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


def _cost(cyl=None, **kw) -> FusedCost:
    c = FusedCost(CostConfig(**kw), max_horizon=12, dtype=torch.float64)
    if cyl is not None:
        c.set_idle_cylinder(cyl)
    return c


def _flat(B: int, H: int, q0: np.ndarray):
    q = torch.as_tensor(np.repeat(np.asarray(q0, float)[None, :], B, axis=0))
    return q, q.unsqueeze(1).repeat(1, H, 1).clone(), torch.zeros(B, len(JOINTS), dtype=torch.float64)


# ============================================================== library ====
def test_library_loads_default_shape(library):
    assert library.q_arm.shape[0] == 2
    assert library.q_arm.shape[2] == _N_ARM
    assert 15 <= library.K <= 30
    assert library.q_arm.shape == (2, library.K, _N_ARM)


def test_derive_cylinder_and_tip_inside(library, fk, cylinder):
    for hi, hand in enumerate(_HANDS):
        q = library.full_pose(hand, 0)
        tip = tip_in_torso(fk, torch.as_tensor(q[None]), hand).squeeze(0).numpy()
        assert tip_inside(cylinder, tip)
        assert tip_inside(library.cylinder, tip)


def test_cylinder_on_yaw_axis_misses_drums(fk, cylinder):
    """XY at the torso origin: yaw cannot sweep the volume into a reachable drum."""
    assert abs(float(cylinder.center[0])) < 1e-9
    assert abs(float(cylinder.center[1])) < 1e-9
    kit = DrumKit.from_layout()
    lo, hi = urdf_position_limits(JOINTS, posture_limits=True)
    yaw_i = JOINTS.index("waist_yaw_joint")
    yaws = np.linspace(float(lo[yaw_i]), float(hi[yaw_i]), 129)
    q0 = ready_joint_pos(JOINTS)
    q = np.repeat(q0[None], len(yaws), 0)
    q[:, yaw_i] = yaws
    q[:, JOINTS.index("waist_pitch_joint")] = 0.0
    q[:, JOINTS.index("waist_roll_joint")] = 0.0
    qt = torch.as_tensor(q, dtype=torch.float64)
    Rt, pt = fk.frame(qt, "torso")
    heads = torch.as_tensor(kit.positions_in("pelvis"), dtype=torch.float64)
    rel = heads.unsqueeze(0) - pt.unsqueeze(1)
    heads_t = torch.einsum("nij,ndj->ndi", Rt.transpose(1, 2), rel).numpy()
    z_lo = float(cylinder.center[2] - cylinder.half_height)
    z_hi = heads_t[..., 2].max(axis=0)
    keep = (z_lo - z_hi) <= 0.30
    assert keep.any()
    rho = np.linalg.norm(heads_t[..., :2], axis=-1)
    rim = rho - kit.radii[None, :]
    min_rim = float(rim[:, keep].min())
    assert min_rim > float(cylinder.radius)
    assert float(cylinder.radius) == pytest.approx(min_rim - 1e-3, abs=2e-3)
    # Snare / floor tom sit well below the lower face and must not pinch radius.
    names = list(kit.names)
    for far in ("snare", "floor_tom"):
        if far in names:
            assert not bool(keep[names.index(far)])


def test_nearest_prefers_shoulder_near_ready(library):
    """Smoke: shoulder-weighted nearest stays near ready's proximal joints."""
    ready = ready_joint_pos(JOINTS)
    for hand in _HANDS:
        q_arm = ready[arm_slice(hand, JOINTS)]
        _i, near = library.nearest_arm_pose(hand, q_arm, shoulder_weight=10.0)
        # Same entry as the free-function API.
        j, near2 = nearest_arm_pose(library.arm_poses(hand), q_arm, shoulder_weight=10.0)
        assert _i == j
        assert np.allclose(near, near2)
        # Shoulder block should be closer than a uniform-weight pick would guarantee
        # only as a soft check: distance on first 3 DoFs is finite and small-ish.
        d_sh = float(np.linalg.norm(near[:3] - q_arm[:3]))
        assert d_sh < 1.5


# ================================================================ cost =====
def test_no_cylinder_installed_is_zero():
    q0, wp, v0 = _flat(2, 12, ready_joint_pos(JOINTS))
    out = _cost(w_idle_cylinder=1.0).forward(
        q0, wp, v0, _ctx(2, 12, [[0.0, 0.0], [1.0, 0.0]])
    )
    assert out.raw["posture_idle_cylinder"].abs().max().item() == 0.0


def test_outside_positive_inside_near_zero(library, fk, cylinder):
    # Ready tip is radially outside the waist-centered cylinder; library pose 0 sits inside.
    q_out = ready_joint_pos(JOINTS)
    q_in = q_out.copy()
    q_in[arm_slice("right", JOINTS)] = library.arm_poses("right")[0]

    cost = _cost(cylinder, w_idle_cylinder=1.0)
    ctx = _ctx(1, 12, [[1.0, 0.0]])  # right idle

    q0, wp, v0 = _flat(1, 12, q_out)
    out_out = cost.forward(q0, wp, v0, ctx)
    outside_cost = out_out.raw["posture_idle_cylinder"].item()
    assert outside_cost > 1e-4

    q0, wp, v0 = _flat(1, 12, q_in)
    tip = tip_in_torso(fk, q0, "right").squeeze(0).numpy()
    assert tip_inside(cylinder, tip)
    out_in = cost.forward(q0, wp, v0, ctx)
    inside_cost = out_in.raw["posture_idle_cylinder"].item()
    # softplus hinge is not exact-zero on the boundary; just near-zero vs outside
    assert inside_cost < 1e-3
    assert inside_cost < 0.01 * outside_cost


def test_mid_horizon_outside_affects_cost(library, cylinder):
    """Cylinder hinge is all-knot: an early outside tip raises the term."""
    H = 12
    idx = arm_slice("right", JOINTS)
    q_in = ready_joint_pos(JOINTS)
    q_in[idx] = library.arm_poses("right")[0]
    q_out = ready_joint_pos(JOINTS)

    cost = _cost(cylinder, w_idle_cylinder=1.0)
    ctx = _ctx(1, H, [[1.0, 0.0]])

    q0, wp_hold, v0 = _flat(1, H, q_in)
    hold = cost.forward(q0, wp_hold, v0, ctx).raw["posture_idle_cylinder"].item()

    q0, wp_bump, v0 = _flat(1, H, q_in)
    wp_bump[0, H // 2, idx] = torch.as_tensor(q_out[idx])
    bumped = cost.forward(q0, wp_bump, v0, ctx).raw["posture_idle_cylinder"].item()
    assert bumped > hold + 1e-5


def test_set_idle_cylinder_none_uninstalls(cylinder):
    c = _cost(cylinder, w_idle_cylinder=1.0)
    assert c.c_idle_cyl is not None
    c.set_idle_cylinder(None)
    assert c.c_idle_cyl is None
    q0, wp, v0 = _flat(1, 12, ready_joint_pos(JOINTS))
    out = c.forward(q0, wp, v0, _ctx(1, 12, [[1.0, 0.0]]))
    assert out.raw["posture_idle_cylinder"].item() == 0.0


# ============================================================ warmstart ====
def test_warmstart_inside_holds_outside_moves(library, fk, cylinder):
    lib_goals = GoalPoseLibrary(joint_names=JOINTS)
    idx = arm_slice("left", JOINTS)

    q_in = ready_joint_pos(JOINTS)
    q_in[idx] = library.arm_poses("left")[0]
    tip = tip_in_torso(fk, torch.as_tensor(q_in[None]), "left").squeeze(0).numpy()
    assert tip_inside(cylinder, tip)

    ws_hold = warmstart_trajectory(
        q_in, np.zeros(len(JOINTS)), horizon=25,
        goals={"left": None, "right": ("snare", 0.4)}, library=lib_goals,
        idle_library=library, idle_cylinder=cylinder, fk=fk,
    )
    assert np.abs(ws_hold.q[:, idx] - q_in[idx]).max() < 1e-9

    q_out = ready_joint_pos(JOINTS)
    tip_out = tip_in_torso(fk, torch.as_tensor(q_out[None]), "left").squeeze(0).numpy()
    assert not tip_inside(cylinder, tip_out)

    ws_move = warmstart_trajectory(
        q_out, np.zeros(len(JOINTS)), horizon=25,
        goals={"left": None, "right": ("snare", 0.4)}, library=lib_goals,
        idle_library=library, idle_cylinder=cylinder, fk=fk, idle_settle_wp=10,
    )
    assert np.abs(ws_move.q[-1, idx] - q_out[idx]).max() > 1e-3
    tip_end = tip_in_torso(
        fk, torch.as_tensor(ws_move.q[-1][None]), "left"
    ).squeeze(0).numpy()
    assert tip_inside(cylinder, tip_end) or float(
        np.linalg.norm(tip_end - tip_out)
    ) > 1e-2
