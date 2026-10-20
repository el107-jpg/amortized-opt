"""Geometry / control-grid constants for the amortized optimizer (standalone).

Vendored from ``source/g1_drummer/g1_drummer/control/trajopt/problem.py``
(2026-08-10). Nothing here imports the repo package; the asset paths point at
``amortized/assets/`` so the folder runs with ``UNITREE_MODEL_DIR`` unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ control --
# Resolved decision 1 (docs/amortized_optimizer_plan.md): the amortized stack
# runs a true 50 Hz waypoint grid with the PD / physics loop at 250 Hz, i.e.
# decimation 5. This applies to the amortized folder ONLY; the main repo keeps
# SIM_DT = 1/120, DECIMATION = 2.
CONTROL_HZ: float = 50.0
CONTROL_DT: float = 0.02
PD_HZ: float = 250.0
PD_DT: float = 1.0 / 250.0
# 2026-08-12: 5 -> 10.  This is NOT a claim that Isaac steps physics at 500 Hz --
# it still steps at PD_HZ.  It is an INTEGRATION-ACCURACY knob, and it had to move
# when the runtime damping went to Kd 4.0 (section 9).
#
# Isaac's arm actuators are ``ImplicitActuatorCfg``: the PD law is solved INSIDE
# the PhysX solver, implicitly, and is therefore unconditionally stable at any
# gain.  ``simulate_pd`` approximates that with an EXPLICIT substep recursion,
# which is only conditionally stable -- roughly ``h * Kd / M_eff < 2``.  The
# wrists are armature-dominated (M_eff ~ 0.01), so at Kd 4.0 and h = 1/250 that
# ratio is ~1.6 at the nominal gain and crosses 2 on the upper tail of the
# log-normal gain randomization.  Measured over 512 randomized rollouts of a held
# 50 mrad step, 40 knots:
#
#     Kd 2.0 (old), S=5   ->  max|q| 9.0e-2      correct
#     Kd 4.0,       S=5   ->  max|q| 3.3e+23     silently wrong, and FINITE
#     Kd 4.0,       S=10  ->  max|q| 6.4e-2      0.9 % from the S=40 answer
#     Kd 4.0,       S=40  ->  max|q| 6.3e-2      converged
#
# The 3.3e23 is the dangerous case: it is finite, so no divergence guard fires,
# and it would poison the chained-BPTT term (which rolls open loop over the whole
# horizon) rather than crash it.  S=10 also matches the convergence table already
# in the README: terminal-velocity error 8.7e-1 / 3.9e-1 / 1.4e-1 / 6.9e-2 /
# 2.5e-2 rad/s for S = 1 / 2 / 5 / 10 / 25.
# Cost: 2x the work in every ``simulate_pd`` call.
PD_SUBSTEPS: int = 10
#: Hit times are quantized onto the control grid (20 ms).
HIT_QUANTIZATION_S: float = CONTROL_DT

# ------------------------------------------------------------------- joints --
#: The 17 actuated joints, in the canonical optimizer ordering.
G1_UPPER_BODY_JOINTS: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
N_DOF: int = len(G1_UPPER_BODY_JOINTS)

#: Per-joint ``|RNEA(q,qd,qdd)| / |RNEA(q,0,0)|`` RMS ratios on decoder
#: plans (bucket 50, phase-1 init, training pool).  Gravity-only static torque
#: tightens by dividing the effort limit by this factor:
#: ``lim_static_j = eta_torque * hard_j / ratio_j``.  Coupled dynamic RNEA keeps
#: waist at the full hard rating (:func:`torque_limit_eta_scale`).  Re-measure
#: with ``scripts/compare_static_vs_dynamic_torque.py``.
STATIC_TORQUE_DYN_STATIC_RATIO: dict[str, float] = {
    # Roll/yaw τ_g≈0 so dyn/static ratios were unreliable; cap at ~20 % of hard
    # (η/ratio = 0.20 → ratio 4.0 at η=0.8).
    "waist_yaw_joint": 4.0,
    "waist_pitch_joint": 1.89,
    "waist_roll_joint": 4.0,
    "left_shoulder_pitch_joint": 1.57,
    "left_shoulder_roll_joint": 4.62,
    "left_shoulder_yaw_joint": 4.94,
    "left_elbow_joint": 1.65,
    "left_wrist_roll_joint": 3.99,
    "left_wrist_pitch_joint": 2.59,
    "left_wrist_yaw_joint": 8.36,
    "right_shoulder_pitch_joint": 1.57,
    "right_shoulder_roll_joint": 4.62,
    "right_shoulder_yaw_joint": 4.94,
    "right_elbow_joint": 1.65,
    "right_wrist_roll_joint": 3.99,
    "right_wrist_pitch_joint": 2.59,
    "right_wrist_yaw_joint": 8.36,
}


def static_torque_tighten_divisors(
    joint_names: tuple[str, ...] | list[str] = G1_UPPER_BODY_JOINTS,
    *,
    mode: str = "decoder_measured",
    cap: float = 10.0,
    floor: float = 1.0,
) -> np.ndarray:
    """Per-joint limit divisors for gravity-only static torque tightening.

    ``mode="decoder_measured"`` uses :data:`STATIC_TORQUE_DYN_STATIC_RATIO`.
    Unknown joints default to ``1.0`` (no extra tightening).
    """
    if mode in ("off", "none", ""):
        return np.ones(len(joint_names), dtype=float)
    if mode != "decoder_measured":
        raise ValueError(
            f"static_torque_tighten mode must be 'decoder_measured' or off, "
            f"got {mode!r}"
        )
    out = np.ones(len(joint_names), dtype=float)
    for i, nm in enumerate(joint_names):
        r = float(STATIC_TORQUE_DYN_STATIC_RATIO.get(nm, 1.0))
        out[i] = float(np.clip(r, floor, cap))
    return out


def torque_limit_eta_scale(
    joint_names: tuple[str, ...] | list[str] = G1_UPPER_BODY_JOINTS,
    *,
    eta_torque: float = 0.8,
) -> np.ndarray:
    """Per-joint η for coupled RNEA torque hinges.

    Waist joints use ``1.0`` (full hard rating).  Arms use ``eta_torque``.
    """
    out = np.full(len(joint_names), float(eta_torque), dtype=float)
    for i, nm in enumerate(joint_names):
        if nm.startswith("waist_"):
            out[i] = 1.0
    return out


#: Absolute joint angles of the drummer ready pose (problem.py::ready_joint_pos).
G1_READY_JOINT_OFFSETS: dict[str, float] = {
    "left_shoulder_pitch_joint": -1.20,
    "right_shoulder_pitch_joint": -1.20,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.25,
    "right_shoulder_yaw_joint": -0.25,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}

#: Elbow-down / palm-out posture preference (problem.py::G1_DRUM_POSTURE_LIMITS_RAD).
#:
#: These are a PREFERENCE encoded as a hard box, and they bind hard: shoulder
#: yaw keeps 1.7 rad of the URDF's 5.24, wrist roll 1.2 of 3.94 -- ~30 % each.
#: That matters for TORQUE (2026-08-16): wrist roll is an ``N5020-16`` rated
#: 25 N*m while wrist pitch/yaw are ``W4010-25`` at 5 N*m, and the 5 N*m pair
#: is what saturates during a strike (2.4x rating at p90).  The joint with
#: five times the headroom is the one tied down.
#:
#: ``AMORTIZED_POSTURE_LIMITS=wide`` empties this box so the URDF (and Isaac,
#: which never had the preference) is the only bound.  Env-driven rather than
#: threaded through call sites because ``urdf_position_limits()`` is called
#: with defaults from ~everywhere -- the cost's joint box, the start sampler,
#: the seed clip and the goal-pose filter must all agree or the pool carries a
#: box the loss does not price.
#:
#: HOMOTOPY NOTE (the obvious worry): widening wrist roll does NOT introduce a
#: 180-deg flip ambiguity.  The stick mount sits 0.2 m off the roll axis, so
#: the tip sweeps 0.33 m monotonically across the full range and only rolls
#: within +-0.066 rad reproduce a given tip position -- there is no second
#: solution to jump to.  Seeds also still come from ``drum_goal_poses.json``,
#: which is NOT re-solved here, so the seed keeps selecting the branch and the
#: wider box only lets the trajectory breathe around it.
_POSTURE_LIMITS_FULL: dict[str, tuple[float, float]] = {
    "left_shoulder_yaw_joint": (-0.9, 0.8),
    "right_shoulder_yaw_joint": (-0.8, 0.9),
    "left_wrist_roll_joint": (-0.2, 1.0),
    "right_wrist_roll_joint": (-1.0, 0.2),
}
G1_DRUM_POSTURE_LIMITS_RAD: dict[str, tuple[float, float]] = (
    {} if os.environ.get("AMORTIZED_POSTURE_LIMITS", "").lower() in ("wide", "off", "0")
    else dict(_POSTURE_LIMITS_FULL)
)


def ready_joint_pos(joint_names: tuple[str, ...] | list[str] = G1_UPPER_BODY_JOINTS) -> np.ndarray:
    return np.array([G1_READY_JOINT_OFFSETS.get(n, 0.0) for n in joint_names], dtype=float)


# ------------------------------------------------------------------- sticks --
STICK_LENGTH = 0.40  # m
STICK_RADIUS = 0.008  # m
TIP_RADIUS = 0.010  # m
STICK_MOUNT_POS_LEFT = (0.2, -0.03, 0.0)
STICK_MOUNT_POS_RIGHT = (0.2, 0.03, 0.0)
STICK_MOUNT_ROT_LEFT_WXYZ = (0.656636, 0, 0.717789, 0)
STICK_MOUNT_ROT_RIGHT_WXYZ = (0.656636, 0, 0.717789, 0)
#: Arc coordinate of the striking tip along the stick axis.
STICK_TIP_SIGMA = STICK_LENGTH / 2.0
#: Contact span back from the tip that counts as a shaft hit (cymbals/hihat).
SHAFT_HIT_SPAN = STICK_LENGTH / 2.0
SHAFT_HIT_SAMPLES = 4
#: Distal stick length NOT treated as colliding shaft (torch_fk.SHAFT_TIP_CLEAR).
SHAFT_TIP_CLEAR: float = 0.18
#: Minimum distance between the two hands' CONTACT POINTS when both sticks are
#: asked to strike the same drum head.  One number, two consumers -- the pool's
#: same-drum sampler (``Phase2ScenarioConfig.pair_same_min_sep``) and the MPC
#: roller's re-goal separation guard (``MPCRoller._regoal``) -- so a scenario
#: drawn fresh and a goal promoted mid-episode cannot disagree about how close
#: the beads may get.  Two 8 mm shafts with 10 mm beads need well under this;
#: the margin is for the WRISTS and the forearms behind them, which is what
#: actually collides when both hands crowd one 0.14 m head.
STICK_MIN_SEP: float = 0.06

# --------------------------------------------------------- body / self-coll --
#: Torso capsule + head sphere in the waist_pitch (torso) joint frame.
TORSO_CAPSULE = ((0.0, 0.0, 0.05), (0.0, 0.0, 0.32), 0.12)
HEAD_SPHERE = ((0.02, 0.0, 0.52), 0.11)
WRIST_RADIUS = 0.0225

# Shared conservative arm collision proxies. Shoulder links have 3 cm URDF
# cylinders; distal links use meshes. These capsules are geometric surrogates,
# not replacement Isaac colliders, and are shared by pose and trajectory costs.
UPPER_ARM_RADIUS = 0.040
FOREARM_RADIUS = 0.035

#: Fixed lower-body collision capsules in the pelvis frame.  The optimizer has
#: no leg DoF: the Isaac drummer configuration welds the twelve leg joints at
#: the upstream ``UNITREE_G1_29DOF_CFG`` standing pose (hip pitch ``-0.10``,
#: knee ``+0.30``, ankle pitch ``-0.20`` radians; all other leg joints zero).
#: These centre lines are the resulting URDF joint-frame positions, with a
#: conservative envelope for the pelvis/hip, thigh, shin and foot meshes.
#:
#: They intentionally remain PELVIS-fixed -- unlike :data:`TORSO_CAPSULE`, they
#: must not lean or yaw with the commanded waist.  This gives the analytic cost
#: and start/eval screens the same fixed-base lower-body reference.  They are
#: collision *proxies*, not new lower-body dynamics or a replacement for a full
#: mesh SDF.
#:
#: ``(name, endpoint_a, endpoint_b, radius_m)``.  The mirrored leg values are
#: generated from ``assets/g1_29dof_rev_1_0.urdf`` at that exact frozen pose.
FIXED_LOWER_BODY_CAPSULES = (
    ("pelvis", (0.0, 0.0, 0.055), (0.0, 0.0, -0.145), 0.140),
    ("left_thigh", (0.000000, 0.064452, -0.102700),
     (0.033601, 0.118601, -0.437614), 0.082),
    ("left_shin", (0.033601, 0.118601, -0.437614),
     (-0.026002, 0.118506, -0.731644), 0.072),
    ("left_foot", (-0.096002, 0.118506, -0.749202),
     (0.124000, 0.118506, -0.749202), 0.055),
    ("right_thigh", (0.000000, -0.064452, -0.102700),
     (0.033601, -0.118601, -0.437614), 0.082),
    ("right_shin", (0.033601, -0.118601, -0.437614),
     (-0.026002, -0.118506, -0.731644), 0.072),
    ("right_foot", (-0.096002, -0.118506, -0.749202),
     (0.124000, -0.118506, -0.749202), 0.055),
)

#: Pelvis origin in the env frame; add to pelvis-frame FK to get env coords.
PELVIS_IN_ENV = np.array([0.0, 0.0, 0.80])

# ------------------------------------------------------------------- assets --
ASSETS_DIR: Path = (Path(__file__).resolve().parent.parent / "assets").resolve()
URDF_PATH: Path = ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
DRUM_LAYOUT_PATH: Path = ASSETS_DIR / "drum_kit_layout.yaml"
DRUM_GOAL_POSES_PATH: Path = ASSETS_DIR / "drum_goal_poses.json"


def default_urdf_path() -> Path:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"vendored G1 URDF missing at {URDF_PATH}")
    return URDF_PATH


def default_layout_path() -> Path:
    if not DRUM_LAYOUT_PATH.exists():
        raise FileNotFoundError(f"vendored drum layout missing at {DRUM_LAYOUT_PATH}")
    return DRUM_LAYOUT_PATH
