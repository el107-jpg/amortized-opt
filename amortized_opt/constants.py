"""Geometry / control-grid constants for the amortized optimizer (standalone).

Vendored from ``source/g1_drummer/g1_drummer/control/trajopt/problem.py``
(2026-08-10). Nothing here imports the repo package; the asset paths point at
``amortized/assets/`` so the folder runs with ``UNITREE_MODEL_DIR`` unset.
"""

from __future__ import annotations

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
G1_DRUM_POSTURE_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "left_shoulder_yaw_joint": (-0.9, 0.8),
    "right_shoulder_yaw_joint": (-0.8, 0.9),
    "left_wrist_roll_joint": (-0.2, 1.0),
    "right_wrist_roll_joint": (-1.0, 0.2),
}


def ready_joint_pos(joint_names: tuple[str, ...] | list[str] = G1_UPPER_BODY_JOINTS) -> np.ndarray:
    return np.array([G1_READY_JOINT_OFFSETS.get(n, 0.0) for n in joint_names], dtype=float)


# ------------------------------------------------------------------- sticks --
STICK_LENGTH = 0.40  # m
STICK_RADIUS = 0.008  # m
TIP_RADIUS = 0.014  # m
STICK_MOUNT_POS_LEFT = (0.2, -0.06, 0.0)
STICK_MOUNT_POS_RIGHT = (0.2, 0.06, 0.0)
STICK_MOUNT_ROT_LEFT_WXYZ = (0.656636, 0.163718, 0.717789, -0.163718)
STICK_MOUNT_ROT_RIGHT_WXYZ = (0.656636, -0.163718, 0.717789, 0.163718)
#: Arc coordinate of the striking tip along the stick axis.
STICK_TIP_SIGMA = STICK_LENGTH / 2.0
#: Contact span back from the tip that counts as a shaft hit (cymbals/hihat).
SHAFT_HIT_SPAN = STICK_LENGTH / 2.0
SHAFT_HIT_SAMPLES = 4
#: Distal stick length NOT treated as colliding shaft (torch_fk.SHAFT_TIP_CLEAR).
SHAFT_TIP_CLEAR: float = 0.18

# --------------------------------------------------------- body / self-coll --
#: Torso capsule + head sphere in the waist_pitch (torso) joint frame.
TORSO_CAPSULE = ((0.0, 0.0, 0.05), (0.0, 0.0, 0.32), 0.12)
HEAD_SPHERE = ((0.02, 0.0, 0.52), 0.11)
WRIST_RADIUS = 0.0225

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
