"""Unitree G1 asset helpers for the drummer environment.

In addition to locating the upstream Unitree G1 USD, this module *composes* a
derived USD that rigidly attaches a drumstick to each wrist. The stick is a
capsule shaft + tip sphere added as collision geometry on the wrist-yaw link,
mounted at an angle so wrist roll/pitch/yaw all move the tip. Hits are detected
geometrically from the tracked tip position/velocity.

Standalone port of the main repo's ``g1_drummer.assets.g1``, with three
deliberate changes:

1. Every stick and body-collider dimension is now *imported* from
   :mod:`amortized_opt.constants` rather than restated here. In the repo the two
   copies were kept in sync by hand, and they are the same numbers the analytic
   FK/SDF stack scores against -- a drift between them means PhysX and the SDF
   disagree about where the stick tip is. Importing makes that drift impossible.
2. The composed-USD cache lives under ``amortized/assets/robots/_generated/``
   (override with ``$AMORTIZED_USD_CACHE``) instead of the repo root.
3. The upper-body joint list comes from
   :data:`amortized_opt.constants.G1_UPPER_BODY_JOINTS`.

External inputs that are NOT vendored, because they are third-party rather than
repo-internal: ``isaaclab``, ``unitree_rl_lab`` (for the stock G1 articulation
cfg) and the upstream Unitree G1 USD located via ``$UNITREE_MODEL_DIR``.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

from amortized_opt.constants import (
    G1_UPPER_BODY_JOINTS,
    HEAD_SPHERE,
    SHAFT_HIT_SPAN,
    STICK_LENGTH,
    STICK_MOUNT_POS_LEFT,
    STICK_MOUNT_POS_RIGHT,
    STICK_MOUNT_ROT_LEFT_WXYZ,
    STICK_MOUNT_ROT_RIGHT_WXYZ,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
)

_G1_USD_REL_PATH = Path("G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd")


def _stock_g1_cfg():
    """The upstream Unitree G1 articulation cfg.

    Deferred, like the ``pxr`` imports below and like
    ``isaac_rl_ppo.env._isaac_imports``: ``unitree_rl_lab`` pulls in Isaac Lab,
    which needs a launched Kit app (``omni.client``). Keeping it out of module
    scope lets the stick/collider geometry above -- the half that has to agree
    with :mod:`amortized_opt.fk` -- be imported and tested headless.
    """
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

    return UNITREE_G1_29DOF_CFG

# --- drumstick geometry / mounting -----------------------------------------
# The stick is attached as COLLISION GEOMETRY on each wrist-yaw link (not a
# separate rigid body): a child collider becomes part of the wrist's existing
# rigid body, so it is rigidly held, every wrist DOF moves it, and it produces
# real contacts. Hits are sensed/tracked on the wrist link itself.
LEFT_WRIST_LINK = "left_wrist_yaw_link"
RIGHT_WRIST_LINK = "right_wrist_yaw_link"
# Visual/collision child prim name under each wrist link.
_STICK_CHILD = {LEFT_WRIST_LINK: "left_stick", RIGHT_WRIST_LINK: "right_stick"}

# STICK_LENGTH / STICK_RADIUS / TIP_RADIUS are imported above (0.40 / 0.008 /
# 0.014 m) -- the tip radius is deliberately fatter than the shaft so the
# striking end contacts reliably.

# Per-collider restitution (bound in USD on tip/shaft only). Scene default and drum
# heads stay at 0; with restitution_combine_mode="max" only tip–drum contacts bounce.
STICK_TIP_RESTITUTION = 0.88
STICK_SHAFT_RESTITUTION = 0.0

# --- self-collision body geometry -------------------------------------------
# The upstream Unitree G1 USD ships with NO collision geometry at all: every
# link carries an empty `collisions` scope whose internal reference targets
# /colliders/<link>, a prim that does not exist in any shipped layer. The whole
# stage has zero prims with UsdPhysics.CollisionAPI. The articulation ALREADY
# has enabled_self_collisions=True (UnitreeUsdFileCfg), so self-collision has
# always been a no-op purely for want of geometry — the sticks swept straight
# through the chest and head, and the only thing objecting was the analytic SDF
# reward term.
#
# We author the torso capsule + head sphere that the analytic safety stack
# already models (control.safety.torch_fk.TORSO_CAPSULE / HEAD_SPHERE, itself
# mirroring trajopt.robot._TORSO_CAPSULE / _HEAD_SPHERE) so PhysX and the SDF
# agree on the body by construction. head_link / logo_link are fixed-joint
# children folded into the torso_link rigid body, so both primitives ride
# torso_link's frame — which is exactly the waist_pitch joint frame the
# constants are expressed in.
#
# Because these are the ONLY body colliders on the robot, turning self-collision
# on cannot regress the leg/hip/shoulder chains: there is no other geometry for
# PhysX to depenetrate at spawn. The reachable self pairs are stick↔torso,
# stick↔head and stick↔stick (shaft/tip within one wrist share a rigid body and
# are excluded automatically).
#
# This geometry is now the SOLE enforcement of self-collision: the analytic self
# term is off by default (G1DrummerEnvCfg.safety_sdf_include_self), so the policy
# is not charged for approaching its own body, it is simply prevented from
# passing through it. Verified by scripts/test_self_collision.py.
BODY_COLLIDER_LINK = "torso_link"
# Zero restitution on the body. NOTE: the scene uses restitution_combine_mode=
# "max" (G1DrummerEnvCfg.physics_material), so a tip — restitution 0.88 so that
# tip–drum contacts bounce — striking the torso still rebounds at 0.88 whatever
# material the body carries; "max" ignores this value on that pair. If a policy
# is ever seen pushing off its own chest, the fix is to move the high restitution
# from the stick tip onto the drum heads (max(0, 0.88) still bounces on drums,
# max(0, 0) does not on the body) rather than to re-tune this constant.
BODY_COLLIDER_RESTITUTION = 0.0
BODY_COLLIDER_COLOR = (0.30, 0.34, 0.42)
# Render the body colliders? Off by default. These primitives are coarse stand-ins
# (a Ø0.24 m capsule and a Ø0.22 m sphere) that fully envelop the real chest and
# head meshes, so leaving them visible replaces the G1's actual appearance with a
# grey capsule-and-sphere snowman in the viewport. USD `visibility` is a
# render-only attribute — PhysX parses CollisionAPI regardless — so hiding them
# costs nothing physically. Flip to True to eyeball the collision volumes.
BODY_COLLIDER_VISIBLE = False

# --- frozen legs -------------------------------------------------------------
# The drummer sits on a fixed base: the pelvis is anchored to the world (see the
# fix_root_link comment in get_g1_cfg, backed by G1DrummerEnvCfg.pin_pelvis), and
# the legs carry NO collision geometry (see BODY_COLLIDER_LINK -- torso capsule +
# head sphere are the robot's only body colliders). So the 12 leg DoF are pure
# simulation overhead:
#
#   * the policy never commands them -- the action space is the 17 upper-body
#     joints (G1DrummerEnvCfg.upper_body_joint_names), and everything else is left
#     to hold its default pose on the stock PD drives;
#   * they cannot influence the arms -- an anchored root absorbs every reaction
#     force and torque, so leg inertia never reaches the shoulders;
#   * they cannot touch anything -- no colliders, hence no ground contact, no
#     drum contact, no self-collision.
#
# They were nevertheless 12 of the articulation's 29 DoF in every PhysX solve, and
# they drift/jitter slightly under the drives. ``freeze_legs`` welds each leg
# revolute joint into a FIXED joint in the composed USD, so PhysX drops them from
# the reduced-coordinate articulation entirely (29 -> 17 DoF) instead of solving
# and servoing DoF that exist only to look like legs.
#
# The weld angle for each joint is exactly the angle the upstream default pose
# puts it at (see _leg_weld_angles), baked into the joint frame, so the robot's
# stance is visually unchanged -- it just stops being articulated.
LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Mount transform of the stick within the wrist-yaw link's local frame.
#   pos: where the stick's origin sits relative to the wrist link.
#   rot: quaternion (w, x, y, z). Local +Z (tip) maps through this to the
#        wrist frame. Base +90° Y (along forearm) then ±28° Z (inward flare).
# The tip sits at local +Z = STICK_LENGTH/2 within the stick's own frame.
# Wrist frame: +X along forearm (distal). Palm offset is along ±Y and is
# sagittal-mirrored (left −Y into palm; right +Y into palm — same local −Y
# would sit above the right hand).
#
# Both pos and rot are imported from amortized_opt.constants, so the collider
# authored below and the analytic tip in ``amortized_opt.fk`` are built from one
# set of numbers and cannot drift apart.
STICK_MOUNT_POS = STICK_MOUNT_POS_LEFT  # back-compat alias
_STICK_MOUNT_POS = {LEFT_WRIST_LINK: STICK_MOUNT_POS_LEFT, RIGHT_WRIST_LINK: STICK_MOUNT_POS_RIGHT}
# Base: +90° about Y — shaft local +Z → wrist +X (along forearm).
# Then ±28° about Z — tips angled inward (toward midline), L/R sagittal mirrors.


def _unit_quat(q):
    """Normalize a (w, x, y, z) quaternion.

    ``constants.STICK_MOUNT_ROT_*_WXYZ`` is stored rounded to 6 dp, which leaves
    it a hair off the unit sphere (|q| = 0.9999995). That matters twice over:

    * a non-unit quaternion is not a rotation -- the matrix USD builds from it is
      a skew (R^T R != I) that the renderer and PhysX resolve differently;
    * ``_quat_rotate`` below applies the raw Hamilton formula, which scales the
      rotated vector by |q|^2, while ``amortized_opt.fk.stick_mount_transform``
      re-normalizes before building its matrix. Left unnormalized the two place
      the stick tip 2.1e-7 m apart -- physically nil against a 14 mm tip, but it
      makes the collider and the cost function disagree by construction, which
      is precisely what importing the shared constants was meant to prevent.

    Normalizing here, once, before the value reaches BOTH ``_author_stick`` (the
    collider) and ``_tip_offset_in_wrist`` (the tracked tip), takes the gap to
    3e-18. ``tests/test_g1_scene.py`` pins it.
    """
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


STICK_MOUNT_ROT_LEFT = _unit_quat(STICK_MOUNT_ROT_LEFT_WXYZ)
STICK_MOUNT_ROT_RIGHT = _unit_quat(STICK_MOUNT_ROT_RIGHT_WXYZ)
_STICK_MOUNT_ROT = {LEFT_WRIST_LINK: STICK_MOUNT_ROT_LEFT, RIGHT_WRIST_LINK: STICK_MOUNT_ROT_RIGHT}


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q=(w,x,y,z). Returns a tuple."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_xyz, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


# Striking-tip position expressed in the WRIST link frame: the mount offset plus
# the tip's position within the (angled) stick. The env uses this to track the
# tip pose/velocity from the wrist link pose. Per-hand (mirrored mount + rot).
def _tip_offset_in_wrist(mount_pos, mount_rot):
    rotated = _quat_rotate(mount_rot, (0.0, 0.0, STICK_LENGTH / 2.0))
    return (
        mount_pos[0] + rotated[0],
        mount_pos[1] + rotated[1],
        mount_pos[2] + rotated[2],
    )


TIP_OFFSET_IN_WRIST_LEFT = _tip_offset_in_wrist(STICK_MOUNT_POS_LEFT, STICK_MOUNT_ROT_LEFT)
TIP_OFFSET_IN_WRIST_RIGHT = _tip_offset_in_wrist(STICK_MOUNT_POS_RIGHT, STICK_MOUNT_ROT_RIGHT)
# Backwards-compatible alias (== left) for older scripts/imports.
TIP_OFFSET_IN_WRIST = TIP_OFFSET_IN_WRIST_LEFT


# Points on the stick SHAFT (behind the tip), expressed in the WRIST link frame.
# A real cymbal sounds when ANY part of the stick reaches it -- a crash is
# normally played with the shoulder of the stick, not the bead -- so the hit
# detector tracks these extra probe points alongside the tip and lets them fire
# strikes on the drums flagged `shaft_hit` in the layout YAML (the cymbals).
# Membrane drums stay tip-only: laying a shaft across a snare head is not a hit.
#
# Samples run from the tip BACKWARD along the stick's own +Z axis, covering the
# ``span_m`` of shaft immediately behind the tip. The default span is half the
# stick, i.e. the whole length between the grip (the mount sits at the stick's
# midpoint) and the tip; sampling further back would reach the butt end sticking
# out behind the hand.
def _shaft_offsets_in_wrist(mount_pos, mount_rot, span_m: float, n_samples: int):
    if n_samples <= 0 or span_m <= 0.0:
        return []
    step = float(span_m) / float(n_samples)
    offsets = []
    for k in range(1, int(n_samples) + 1):
        z = STICK_LENGTH / 2.0 - k * step
        rotated = _quat_rotate(mount_rot, (0.0, 0.0, z))
        offsets.append(
            (
                mount_pos[0] + rotated[0],
                mount_pos[1] + rotated[1],
                mount_pos[2] + rotated[2],
            )
        )
    return offsets


#: Default shaft coverage: the 0.20 m of stick between the grip and the tip.
#: Same value the analytic shaft probes use (``constants.SHAFT_HIT_SPAN``).
SHAFT_HIT_SPAN_M = SHAFT_HIT_SPAN


def stick_shaft_offsets_in_wrist(hand: str, span_m: float = SHAFT_HIT_SPAN_M, n_samples: int = 4):
    """Shaft probe offsets for one hand, in that wrist-yaw link's frame.

    Ordered tip-ward first (sample 1 is ``span_m / n_samples`` behind the tip).
    ``hand`` is ``"left"`` or ``"right"``. Returns a list of (x, y, z) tuples,
    empty when shaft hits are disabled (``n_samples <= 0``).
    """
    if hand == "left":
        mount_pos, mount_rot = STICK_MOUNT_POS_LEFT, STICK_MOUNT_ROT_LEFT
    elif hand == "right":
        mount_pos, mount_rot = STICK_MOUNT_POS_RIGHT, STICK_MOUNT_ROT_RIGHT
    else:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    return _shaft_offsets_in_wrist(mount_pos, mount_rot, span_m, n_samples)


def _find_unitree_model_dir() -> Path:
    candidates = []

    if env_path := os.environ.get("UNITREE_MODEL_DIR"):
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            Path.home() / "workspace" / "unitree_model",
            Path.home() / "workspace" / "unitree_usd",
        ]
    )

    for candidate in candidates:
        if (candidate / _G1_USD_REL_PATH).is_file():
            return candidate

    searched = ", ".join(str(path / _G1_USD_REL_PATH) for path in candidates)
    raise FileNotFoundError(
        "Unitree G1 USD was not found. Set UNITREE_MODEL_DIR or clone "
        f"unitree_model into ~/workspace/unitree_model. Searched: {searched}"
    )


def _composed_usd_path(freeze_legs: bool = False) -> Path:
    """Cache path for the generated G1-with-sticks USD.

    Frozen-leg and articulated-leg compositions are different files: the DoF count
    they produce differs (17 vs 29), so they must not share a staleness-checked
    cache entry that flips underneath a running experiment.

    Defaults to ``amortized/assets/robots/_generated/`` (gitignored -- it is a
    derived artifact of the upstream USD, not a vendored asset). Set
    ``$AMORTIZED_USD_CACHE`` to redirect it, e.g. onto scratch when the checkout
    is read-only.
    """
    if env_dir := os.environ.get("AMORTIZED_USD_CACHE"):
        cache_dir = Path(env_dir).expanduser()
    else:
        cache_dir = Path(__file__).resolve().parents[1] / "assets" / "robots" / "_generated"
    name = "g1_with_sticks_frozen_legs.usd" if freeze_legs else "g1_with_sticks.usd"
    return cache_dir / name


def _find_link_paths(src_usd: Path) -> dict:
    """Map each link/joint we edit to its prim path in the source USD."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(src_usd))
    wanted = {LEFT_WRIST_LINK, RIGHT_WRIST_LINK, BODY_COLLIDER_LINK}
    wanted_joints = set(LEG_JOINT_NAMES)
    found: dict[str, object] = {}
    joints: dict[str, object] = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in wanted and name not in found:
            found[name] = prim.GetPath()
        if name in wanted_joints and name not in joints:
            joints[name] = prim.GetPath()
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"Could not find link(s) {sorted(missing)} in {src_usd}.")
    missing_joints = wanted_joints - set(joints)
    if missing_joints:
        raise RuntimeError(f"Could not find leg joint(s) {sorted(missing_joints)} in {src_usd}.")
    default_prim_path = stage.GetDefaultPrim().GetPath()
    return {"paths": found, "joint_paths": joints, "default_prim": default_prim_path}


def _bind_physics_material(stage, collision_prim, mat_name: str, *, restitution: float) -> None:
    """Bind a physics-purpose material with the given restitution to a collider."""
    from pxr import UsdPhysics, UsdShade

    mat_path = collision_prim.GetPath().GetParentPath().AppendChild(mat_name)
    material = UsdShade.Material.Define(stage, mat_path)
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateRestitutionAttr().Set(float(restitution))
    mat_api.CreateStaticFrictionAttr().Set(0.6)
    mat_api.CreateDynamicFrictionAttr().Set(0.5)

    binding = UsdShade.MaterialBindingAPI.Apply(collision_prim)
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _author_stick(stage, wrist_path, wrist_name, color):
    """Attach stick collision geometry to a wrist link (no separate body).

    The shaft + tip colliders are children of the wrist link, so PhysX folds
    them into the wrist's existing rigid body: rigidly held and moved by every
    wrist DOF. Hits are detected geometrically from the tracked tip position; the
    colliders just give the stick a physical presence (and visible geometry).

    Only the tip collider carries high restitution; the shaft is explicitly
    dead so incidental shaft–drum grazes do not rebound.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    stick_name = _STICK_CHILD[wrist_name]
    stick_path = wrist_path.AppendChild(stick_name)
    stick = UsdGeom.Xform.Define(stage, stick_path)
    # Place the stick in the wrist's local frame (drummer-grip angle).
    stick.AddTranslateOp().Set(Gf.Vec3d(*_STICK_MOUNT_POS[wrist_name]))
    w, x, y, z = _STICK_MOUNT_ROT[wrist_name]
    stick.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    # Shaft: capsule along local Z. Capsule length = height + 2*radius.
    shaft = UsdGeom.Capsule.Define(stage, stick_path.AppendChild("shaft"))
    shaft.CreateAxisAttr("Z")
    shaft.CreateRadiusAttr(STICK_RADIUS)
    shaft.CreateHeightAttr(max(STICK_LENGTH - 2.0 * STICK_RADIUS, 1e-3))
    shaft.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())
    _bind_physics_material(
        stage, shaft.GetPrim(), "shaft_physics_mat", restitution=STICK_SHAFT_RESTITUTION
    )

    # Tip: sphere at the striking end (+Z), the reliable contact point.
    tip = UsdGeom.Sphere.Define(stage, stick_path.AppendChild("tip"))
    tip.CreateRadiusAttr(TIP_RADIUS)
    tip.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, STICK_LENGTH / 2.0))
    tip.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.95, 0.95)])
    UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
    _bind_physics_material(
        stage, tip.GetPrim(), "tip_physics_mat", restitution=STICK_TIP_RESTITUTION
    )


def _author_body_colliders(stage, torso_path):
    """Attach the torso capsule + head sphere colliders to ``torso_link``.

    These are the robot's ONLY body colliders (see BODY_COLLIDER_LINK), and they
    reuse the exact primitives the analytic safety stack models, so a contact
    PhysX reports is a contact the SDF penalty would also have seen.

    Unlike the sticks (which are the only representation the drumsticks have, so
    they must render) these duplicate geometry the G1 already ships meshes for,
    and hide by default — see BODY_COLLIDER_VISIBLE.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    # TORSO_CAPSULE / HEAD_SPHERE come from amortized_opt.constants (imported at
    # module level), which is also what amortized_opt.sdf scores against. Both
    # are expressed in the waist_pitch (== torso_link) frame, so a contact PhysX
    # reports here is a contact the SDF penalty would also have seen.
    (a, b, torso_r) = TORSO_CAPSULE
    if abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9:
        raise RuntimeError(
            "TORSO_CAPSULE is no longer Z-aligned; _author_body_colliders would "
            f"place it wrong (a={a}, b={b}). Author an orient op here."
        )

    root_path = torso_path.AppendChild("self_collision")
    root = UsdGeom.Xform.Define(stage, root_path)
    if not BODY_COLLIDER_VISIBLE:
        # Invisibility is inherited down the subtree in USD, so one call on the
        # scope hides both primitives while their CollisionAPI stays active.
        root.MakeInvisible()

    # Torso: capsule along local Z. USD `height` is the segment between the two
    # hemispherical caps, i.e. exactly |b - a| (matches coal.Capsule(r, length)).
    segment = abs(b[2] - a[2])
    torso = UsdGeom.Capsule.Define(stage, root_path.AppendChild("torso"))
    torso.CreateAxisAttr("Z")
    torso.CreateRadiusAttr(float(torso_r))
    torso.CreateHeightAttr(float(segment))
    torso.AddTranslateOp().Set(
        Gf.Vec3d((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)
    )
    torso.GetDisplayColorAttr().Set([Gf.Vec3f(*BODY_COLLIDER_COLOR)])
    UsdPhysics.CollisionAPI.Apply(torso.GetPrim())
    _bind_physics_material(
        stage, torso.GetPrim(), "torso_physics_mat", restitution=BODY_COLLIDER_RESTITUTION
    )

    # Head: sphere riding the same (fixed-jointed) torso rigid body.
    head_c, head_r = HEAD_SPHERE
    head = UsdGeom.Sphere.Define(stage, root_path.AppendChild("head"))
    head.CreateRadiusAttr(float(head_r))
    head.AddTranslateOp().Set(Gf.Vec3d(*head_c))
    head.GetDisplayColorAttr().Set([Gf.Vec3f(*BODY_COLLIDER_COLOR)])
    UsdPhysics.CollisionAPI.Apply(head.GetPrim())
    _bind_physics_material(
        stage, head.GetPrim(), "head_physics_mat", restitution=BODY_COLLIDER_RESTITUTION
    )


def _leg_weld_angles() -> dict:
    """Angle (rad) each leg joint is welded at: its upstream default-pose angle.

    Read out of ``UNITREE_G1_29DOF_CFG.init_state.joint_pos`` (whose keys are
    regexes) rather than hard-coded, so the welded stance tracks the upstream
    default pose instead of silently diverging from it. Joints the default pose
    does not mention sit at 0.
    """
    patterns = dict(_stock_g1_cfg().init_state.joint_pos or {})
    angles = {}
    for joint_name in LEG_JOINT_NAMES:
        value = 0.0
        for pattern, pattern_value in patterns.items():
            if re.fullmatch(pattern, joint_name):
                value = float(pattern_value)
        angles[joint_name] = value
    return angles


def _freeze_leg_joints(stage, joint_paths: dict) -> None:
    """Weld the 12 leg revolute joints into fixed joints at their default angle.

    A ``PhysicsRevoluteJoint`` constrains its two frames to agree up to a rotation
    ``theta`` about ``physics:axis``; a ``PhysicsFixedJoint`` constrains them to
    agree exactly. Retyping one into the other therefore has to bake the current
    ``theta`` into a frame, or the leg snaps to the joint's zero angle. We fold it
    into body0's frame: ``localRot0' = localRot0 * quat(axis, theta)``.

    The angular drive is removed as well -- a fixed joint has no DoF for it to act
    on, so leaving it applied would be a dangling opinion for the USD physics
    parser to interpret.
    """
    from pxr import Gf, UsdPhysics

    axis_vectors = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
    weld_angles = _leg_weld_angles()

    for joint_name, joint_path in joint_paths.items():
        theta = weld_angles[joint_name]
        source = stage.GetPrimAtPath(joint_path)
        if not source.IsValid():
            raise RuntimeError(f"Leg joint {joint_path} vanished from the composed stage.")
        if source.GetTypeName() != "PhysicsRevoluteJoint":
            raise RuntimeError(
                f"Expected {joint_path} to be a PhysicsRevoluteJoint, got "
                f"{source.GetTypeName()}; the weld math below assumes revolute."
            )
        axis_token = source.GetAttribute("physics:axis").Get()
        if axis_token not in axis_vectors:
            raise RuntimeError(f"Leg joint {joint_path} has unsupported axis {axis_token!r}.")
        rot0 = source.GetAttribute("physics:localRot0").Get()

        half = 0.5 * theta
        ax, ay, az = axis_vectors[axis_token]
        sin_half = math.sin(half)
        spin = Gf.Quatf(
            math.cos(half), Gf.Vec3f(ax * sin_half, ay * sin_half, az * sin_half)
        )

        joint = stage.OverridePrim(joint_path)
        joint.SetTypeName("PhysicsFixedJoint")
        joint.GetAttribute("physics:localRot0").Set(rot0 * spin)
        joint.RemoveAPI(UsdPhysics.DriveAPI, "angular")


def _compose_g1_with_sticks(src_usd: Path, out_usd: Path, *, freeze_legs: bool = False) -> None:
    """Create ``out_usd`` referencing ``src_usd`` with a stick on each wrist and
    torso/head self-collision geometry on the torso link."""
    from pxr import Usd, UsdGeom

    info = _find_link_paths(src_usd)
    wrist_paths = info["paths"]
    default_prim_path = info["default_prim"]

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Reference the whole source robot onto the same prim path so the internal
    # hierarchy (and thus the wrist link paths) is preserved.
    robot = stage.OverridePrim(default_prim_path)
    robot.GetReferences().AddReference(assetPath=str(src_usd))
    stage.SetDefaultPrim(stage.GetPrimAtPath(default_prim_path))

    _author_stick(stage, wrist_paths[LEFT_WRIST_LINK], LEFT_WRIST_LINK, (0.55, 0.35, 0.18))
    _author_stick(stage, wrist_paths[RIGHT_WRIST_LINK], RIGHT_WRIST_LINK, (0.55, 0.35, 0.18))
    _author_body_colliders(stage, wrist_paths[BODY_COLLIDER_LINK])
    if freeze_legs:
        _freeze_leg_joints(stage, info["joint_paths"])

    stage.GetRootLayer().Save()


def _ensure_composed_usd(src_usd: Path, *, freeze_legs: bool = False) -> Path:
    """Regenerate the composed USD if missing or stale.

    Stale = older than the source G1 USD *or* older than this module (so editing
    the stick geometry/composition here forces a rebuild without manual cleanup).
    """
    import fcntl
    from amortized_opt import constants
    out_usd = _composed_usd_path(freeze_legs)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    # Geometry is imported from constants. Its edits must invalidate the USD,
    # and concurrent rollout workers must not compose into the same file.
    with out_usd.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        newest_input = max(src_usd.stat().st_mtime, Path(__file__).stat().st_mtime,
                           Path(constants.__file__).stat().st_mtime)
        if not out_usd.is_file() or out_usd.stat().st_mtime < newest_input:
            _compose_g1_with_sticks(src_usd, out_usd, freeze_legs=freeze_legs)
    return out_usd


def get_g1_cfg(
    *,
    use_ideal_pd: bool = False,
    pd_stiffness_scale: float = 1.0,
    pd_damping_scale: float | None = None,
    freeze_legs: bool = True,
):
    """Return a copy of the Unitree G1 articulation config (with sticks).

    ``use_ideal_pd``: convert ImplicitActuator groups to IdealPDActuator so the
    env can inject gravity (or other) feedforward via
    ``set_joint_effort_target`` (τ = Kp e + Kd ė + τ_ff). Required for true
    gravity FF; leave False for vanilla PhysX joint drives.

    ``pd_stiffness_scale`` / ``pd_damping_scale``: multiply every actuator group's
    configured Kp / Kd (Unitree defaults are ~40 / 1 on the arms). Pass the same
    scale for both to approximately preserve damping ratio when stiffening.

    ``freeze_legs`` (default on): weld the 12 leg joints into fixed joints, leaving
    a 17-DoF articulation of exactly the joints the policy drives. See
    :data:`LEG_JOINT_NAMES` for why this is free here. Pass False to get the
    original 29-DoF robot back (e.g. if the pelvis is ever unpinned).
    """
    import isaaclab.sim as sim_utils

    cfg = _stock_g1_cfg().copy()
    src_usd = _find_unitree_model_dir() / _G1_USD_REL_PATH
    cfg.spawn.usd_path = str(_ensure_composed_usd(src_usd, freeze_legs=freeze_legs))
    if freeze_legs:
        _drop_leg_joint_entries(cfg)

    # Anchor the pelvis to the world so the robot cannot fall over.
    # Adds a fixed joint between the root link and the world frame, turning the
    # floating-base G1 into a fixed-base articulation. The legs/waist are then
    # just held at their default pose by the existing PD drives.
    if cfg.spawn.articulation_props is None:
        cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()
    cfg.spawn.articulation_props.fix_root_link = True
    # Stick↔torso/head self-collision. UnitreeUsdFileCfg already defaults this on,
    # but it is what makes the body colliders above load-bearing (see
    # BODY_COLLIDER_LINK), so pin it here rather than inherit it silently.
    cfg.spawn.articulation_props.enabled_self_collisions = True

    if use_ideal_pd:
        cfg.actuators = _actuators_as_ideal_pd(cfg.actuators)
    if pd_damping_scale is None:
        pd_damping_scale = pd_stiffness_scale
    if float(pd_stiffness_scale) != 1.0 or float(pd_damping_scale) != 1.0:
        cfg.actuators = _scale_actuator_pd(
            cfg.actuators,
            stiffness_scale=float(pd_stiffness_scale),
            damping_scale=float(pd_damping_scale),
        )
    return cfg


def _matches_any(pattern: str, names) -> bool:
    """True if a joint-name regex matches at least one of ``names``."""
    return any(re.fullmatch(pattern, name) for name in names)


# Per-joint fields on an actuator cfg that may be a {joint-regex: value} dict and
# therefore need the same leg-pattern pruning as joint_names_expr.
_PER_JOINT_ACTUATOR_FIELDS = (
    "effort_limit",
    "effort_limit_sim",
    "velocity_limit",
    "velocity_limit_sim",
    "stiffness",
    "damping",
    "armature",
    "friction",
    "dynamic_friction",
    "viscous_friction",
)


def _drop_leg_joint_entries(cfg) -> None:
    """Strip (in place) config entries that only ever matched the welded leg joints.

    Isaac Lab raises when an actuator group -- or a per-joint gain dict, or an
    initial-state entry -- resolves to zero joints, so every upstream pattern that
    existed solely for the legs has to go once ``freeze_legs`` removes those DoF.
    ``cfg`` is the caller's private ``.copy()``, so mutating it is safe.
    """
    surviving = _UPPER_BODY_JOINT_NAMES

    for field in ("joint_pos", "joint_vel"):
        entries = getattr(cfg.init_state, field, None)
        if entries:
            setattr(
                cfg.init_state,
                field,
                {p: v for p, v in entries.items() if _matches_any(p, surviving)},
            )

    kept_actuators = {}
    for name, act in cfg.actuators.items():
        exprs = act.joint_names_expr
        exprs = [exprs] if isinstance(exprs, str) else list(exprs)
        kept_exprs = [expr for expr in exprs if _matches_any(expr, surviving)]
        if not kept_exprs:
            # Leg-only group (e.g. the hip-roll/knee actuator): nothing left to drive.
            continue
        act.joint_names_expr = kept_exprs
        for field in _PER_JOINT_ACTUATOR_FIELDS:
            value = getattr(act, field, None)
            if not isinstance(value, dict):
                continue
            kept = {p: v for p, v in value.items() if _matches_any(p, surviving)}
            if not kept:
                raise RuntimeError(
                    f"Actuator group {name!r} still drives {kept_exprs} but its "
                    f"{field} dict only had leg patterns ({sorted(value)}); Isaac "
                    "Lab would leave those joints unconfigured."
                )
            setattr(act, field, kept)
        kept_actuators[name] = act
    cfg.actuators = kept_actuators


def _scale_actuator_pd(actuators: dict, *, stiffness_scale: float, damping_scale: float) -> dict:
    """Return actuator cfgs with stiffness/damping multiplied."""
    out = {}
    for name, act in actuators.items():
        kwargs = {}
        stiff = getattr(act, "stiffness", None)
        damp = getattr(act, "damping", None)
        if stiff is not None:
            kwargs["stiffness"] = _scale_gain(stiff, stiffness_scale)
        if damp is not None:
            kwargs["damping"] = _scale_gain(damp, damping_scale)
        if not kwargs:
            out[name] = act
            continue
        if hasattr(act, "replace"):
            out[name] = act.replace(**kwargs)
        else:
            from dataclasses import replace

            out[name] = replace(act, **kwargs)
    return out


def _scale_gain(gain, scale: float):
    if isinstance(gain, dict):
        return {k: float(v) * scale for k, v in gain.items()}
    return float(gain) * scale


# The 17 joints the drummer policy actually drives (waist + both 7-DoF arms).
# The repo hand-duplicated this tuple here "to stay laptop-runnable"; the
# standalone port imports it, since amortized_opt.constants is pure-python and
# pulls in nothing heavier than numpy.
_UPPER_BODY_JOINT_NAMES = G1_UPPER_BODY_JOINTS


def _drives_upper_body(joint_names_expr) -> bool:
    """True if an actuator group's joint regexes match any upper-body joint."""
    exprs = [joint_names_expr] if isinstance(joint_names_expr, str) else list(joint_names_expr)
    return any(
        re.fullmatch(pattern, joint_name)
        for pattern in exprs
        for joint_name in _UPPER_BODY_JOINT_NAMES
    )


def _actuators_as_ideal_pd(actuators: dict) -> dict:
    """Rebuild upper-body actuator groups as IdealPDActuatorCfg, preserving gains/limits.

    Only groups that drive a joint in :data:`_UPPER_BODY_JOINT_NAMES` are converted.
    The env's RNEA/gravity feedforward and velocity-target writes
    (``g1_drummer_env.py``, ``set_joint_velocity_target`` / ``set_joint_effort_target``)
    are scoped to ``upper_body_joint_ids`` only, so converting leg actuators here
    too would leave them running as an unfed IdealPD -- Kd braking toward qd=0 with
    no gravity compensation (the exact bug fixed for the arms in the 2026-07-16
    reference-filter rework, reappearing on legs). Legs stay on the vanilla
    ImplicitActuator PD drive, matching the original "held at default pose by the
    existing PD drives" design (see the ``fix_root_link`` comment in ``get_g1_cfg``).
    """
    from isaaclab.actuators import IdealPDActuatorCfg

    out = {}
    for name, act in actuators.items():
        if not _drives_upper_body(act.joint_names_expr):
            out[name] = act
            continue
        # Collect kwargs IdealPD shares with Implicit; ignore unknown attrs.
        kwargs = {"joint_names_expr": act.joint_names_expr}
        for key in (
            "effort_limit",
            "effort_limit_sim",
            "velocity_limit",
            "velocity_limit_sim",
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
            "viscous_friction",
        ):
            if hasattr(act, key):
                val = getattr(act, key)
                if val is not None:
                    kwargs[key] = val
        # Pin the explicit clip to the sim limit. An EXPLICIT actuator with
        # effort_limit=None resolves it from the USD joint prim, which can
        # disagree with ``effort_limit_sim``; the Python-side clip must match
        # what PhysX enforces so applied_torque and the amortized cost tables
        # (``limits.isaac_effort_limits``) stay aligned with the actuator.
        if kwargs.get("effort_limit") is None and kwargs.get("effort_limit_sim") is not None:
            kwargs["effort_limit"] = kwargs["effort_limit_sim"]
        out[name] = IdealPDActuatorCfg(**kwargs)
    return out
