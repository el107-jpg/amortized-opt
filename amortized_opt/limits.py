"""Actuator / joint limit constants for the 17-DoF G1 upper body (standalone).

Every number below is the RUNTIME value the repo's helpers actually return, not
the summary table in CLAUDE.md (they differ: live damping is 2x the table).

Provenance, per symbol:

* ``ISAAC_G1_EFFORT_LIMITS_NM``   -> trajopt/problem.py::ISAAC_G1_EFFORT_LIMITS_NM
* ``ISAAC_G1_VELOCITY_LIMITS_RADS`` -> trajopt/problem.py::ISAAC_G1_VELOCITY_LIMITS_RADS
* ``ISAAC_G1_STIFFNESS`` / ``ISAAC_G1_DAMPING`` -> problem.py, BASE (pre-scale)
* ``G1_PD_STIFFNESS_SCALE`` = 1.0, ``G1_PD_DAMPING_SCALE`` = 4.0 -> problem.py,
  mirroring ``G1DrumEnvCfg.pd_{stiffness,damping}_scale``. So runtime gains are
  arms Kp 40 / Kd 4.0, waist Kp 200 / Kd 20.0. CLAUDE.md's Kp 40 / Kd 1.0 is the
  BASE (pre-scale) value. **These three files must agree** -- the conformance
  test ``test_limits_and_spline.py::test_limits_match_repo`` asserts it to 0.0,
  because an offline plant that does not model the robot is worse than none.
* ``ISAAC_G1_ARMATURE_DEFAULT`` = 0.01 -> problem.py::isaac_armature (flat).
* ``derive_accel_limits`` -> safety/limits.py::derive_accel_limits, reimplemented
  against Pinocchio directly (the original took a ``trajopt.robot.G1UpperBody``).
"""

from __future__ import annotations

import numpy as np

from .constants import G1_DRUM_POSTURE_LIMITS_RAD, G1_UPPER_BODY_JOINTS, default_urdf_path

# ------------------------------------------------------------------ effort --
ISAAC_G1_EFFORT_LIMITS_NM: dict[str, float] = {
    "waist_yaw_joint": 88.0,
    "waist_pitch_joint": 25.0,
    "waist_roll_joint": 25.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}

# ---------------------------------------------------------------- velocity --
ISAAC_G1_VELOCITY_LIMITS_RADS: dict[str, float] = {
    "waist_yaw_joint": 32.0,
    "waist_pitch_joint": 30.0,
    "waist_roll_joint": 30.0,
    "left_shoulder_pitch_joint": 37.0,
    "left_shoulder_roll_joint": 37.0,
    "left_shoulder_yaw_joint": 37.0,
    "left_elbow_joint": 37.0,
    "left_wrist_roll_joint": 37.0,
    "left_wrist_pitch_joint": 22.0,
    "left_wrist_yaw_joint": 22.0,
    "right_shoulder_pitch_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "right_elbow_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "right_wrist_pitch_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
}
G1_MAX_JOINT_VELOCITY_RADS: float = 37.0

# ------------------------------------------------------------------ PD gains --
ISAAC_G1_STIFFNESS: dict[str, float] = {
    "waist_yaw_joint": 200.0,
    "waist_pitch_joint": 40.0,
    "waist_roll_joint": 40.0,
    **{
        n: 40.0
        for n in G1_UPPER_BODY_JOINTS
        if n not in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
    },
}
ISAAC_G1_DAMPING: dict[str, float] = {
    "waist_yaw_joint": 5.0,
    "waist_pitch_joint": 5.0,
    "waist_roll_joint": 5.0,
    **{
        n: 1.0
        for n in G1_UPPER_BODY_JOINTS
        if n not in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
    },
}
# 2026-08-12 (section 9): raised from 1.0 / 2.0.  The closed-loop gain sweep
# (`scripts/eval_gain_sweep.py`, 2 clips x 2 seeds, ki=0, the fixed per-joint
# lead) put this cell at 62/98 crossings and 55/98 strikes with the impact speed
# IN BAND, against 36/98 and 21/98 at the old gains -- the largest single move
# any control-side change has produced in this project.  Damping is the term
# doing the work: under the FOH chain the torque is Kp*e + Kd*(ff*qd_cmd - qd),
# so Kd multiplies the VELOCITY FEED-FORWARD and is what delivers commanded
# stroke speed.  The env's own note reached the same conclusion independently
# ("under FOH, Kd(vel_target - qd) is the STROKE ENGINE"; halving it once cost
# ~20x discovery speed) and explicitly left room to raise it.
# Damping ratio: zeta ~ Kd / sqrt(Kp), so doubling both takes the arms from the
# env's stated zeta ~ 0.4 to ~0.57 -- better damped, not overdamped.
# STIFFNESS IS DELIBERATELY NOT RAISED.  A 2x stiffness (arms Kp 80) was tried
# and reverted the same hour: `simulate_pd`'s explicit substep integrator is
# only CONDITIONALLY stable, and Kp 80 crosses the boundary at the default
# PD_SUBSTEPS = 5.  Measured, 512 rollouts of a held 50 mrad step with the usual
# gain randomization:
#
#     Kp 40 (this config), S=5   ->  max|q| 9.0e-2      correct
#     Kp 80,               S=5   ->  max|q| 2.6e+22     silently wrong, NOT NaN
#     Kp 80,               S=10  ->  max|q| 7.6e-2      correct
#     Kp 160,              S=5   ->  512/512 NaN
#
# The 2.6e22 is the dangerous one: it is finite, so no divergence guard fires.
# The closed-loop MPC path survives Kp 80 (smooth, slew-limited commands from the
# measured state), which is why the section-9 sweep's kp x2 / x4 rows returned
# plausible-looking numbers -- but `fit_cmd_lead.py`, the roller and the BPTT
# term all roll open loop, and those blow up.  A conditionally stable plant is
# not a default.  To raise stiffness, raise PD_SUBSTEPS to 10 FIRST (2x the cost
# of every pd_sim call, which is the training hot path) and re-measure.
# On the evidence that is numerically valid (the kp x1 rows), kd x2 alone is
# 60/98 crossings and 50/98 in the speed band against the old 36 and 21 -- inside
# noise of the kp x2 / kd x2 cell's 62 and 55, at none of the risk.
G1_PD_STIFFNESS_SCALE: float = 1.0
G1_PD_DAMPING_SCALE: float = 4.0

# ----------------------------------------------------------------- armature --
ISAAC_G1_ARMATURE_DEFAULT: float = 0.01

#: Conservative fractions of the hard limits the cost hinges should use
#: (plan: torque 80%, velocity 70%, joint position 80%).
ETA_TORQUE: float = 0.80
ETA_VELOCITY: float = 0.70
ETA_POSITION: float = 0.80


def _lookup(table: dict[str, float], joint_names) -> np.ndarray:
    out = np.empty(len(joint_names), dtype=float)
    for i, name in enumerate(joint_names):
        if name not in table:
            raise KeyError(f"no entry for joint {name!r}")
        out[i] = table[name]
    return out


def isaac_effort_limits(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """Per-joint effort limits (N*m) in ``joint_names`` order."""
    return _lookup(ISAAC_G1_EFFORT_LIMITS_NM, joint_names)


def isaac_velocity_limits(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """Per-joint speed ratings (rad/s) in ``joint_names`` order."""
    return _lookup(ISAAC_G1_VELOCITY_LIMITS_RADS, joint_names)


def isaac_stiffness(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """BASE stiffness (actuator cfg value BEFORE the env's scale)."""
    return _lookup(ISAAC_G1_STIFFNESS, joint_names)


def isaac_damping(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """BASE damping (actuator cfg value BEFORE the env's scale)."""
    return _lookup(ISAAC_G1_DAMPING, joint_names)


def isaac_runtime_stiffness(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """Stiffness the plant actually runs (base x scale)."""
    return isaac_stiffness(joint_names) * G1_PD_STIFFNESS_SCALE


def isaac_runtime_damping(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """Damping the plant actually runs (base x scale) -- 2x the CLAUDE.md table."""
    return isaac_damping(joint_names) * G1_PD_DAMPING_SCALE


def isaac_armature(joint_names=G1_UPPER_BODY_JOINTS) -> np.ndarray:
    """Reflected rotor inertia (kg*m^2), flat 0.01 across every group."""
    return np.full(len(joint_names), ISAAC_G1_ARMATURE_DEFAULT, dtype=float)


# ------------------------------------------------------------ position box --
def urdf_position_limits(
    joint_names=G1_UPPER_BODY_JOINTS,
    *,
    urdf_path=None,
    limit_margin: float = 0.0,
    posture_limits: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """(q_lower, q_upper) from the vendored URDF, in ``joint_names`` order.

    Mirrors ``trajopt/robot.py``'s default branch: URDF limits shrunk by
    ``limit_margin``, then narrowed by ``G1_DRUM_POSTURE_LIMITS_RAD``.
    Parsed with stdlib XML so this does not require Pinocchio.
    """
    import xml.etree.ElementTree as ET

    path = str(urdf_path) if urdf_path is not None else str(default_urdf_path())
    root = ET.parse(path).getroot()
    lo_t: dict[str, float] = {}
    hi_t: dict[str, float] = {}
    for el in root.findall("joint"):
        lim = el.find("limit")
        if lim is None:
            continue
        name = el.get("name", "")
        if lim.get("lower") is not None:
            lo_t[name] = float(lim.get("lower"))
        if lim.get("upper") is not None:
            hi_t[name] = float(lim.get("upper"))
    lo = np.array([lo_t[n] for n in joint_names], dtype=float) + float(limit_margin)
    hi = np.array([hi_t[n] for n in joint_names], dtype=float) - float(limit_margin)
    if posture_limits:
        for name, (plo, phi) in G1_DRUM_POSTURE_LIMITS_RAD.items():
            if name in joint_names:
                i = list(joint_names).index(name)
                lo[i] = max(lo[i], plo)
                hi[i] = min(hi[i], phi)
    return lo, hi


# --------------------------------------------------------------- accel box --
def derive_accel_limits(
    joint_names=G1_UPPER_BODY_JOINTS,
    *,
    urdf_path=None,
    n_samples: int = 200,
    eta_tau: float = 0.8,
    a_cap: float = 30.0,
    a_floor: float = 1.0,
    seed: int = 0,
    tau_limit: np.ndarray | None = None,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Per-joint conservative |qdd| bounds (rad/s^2), optimizer order.

    Same formula as ``safety/limits.py::derive_accel_limits``:

        qdd_i = eta_tau * (tau_lim_i - max_q |g_i(q)|) / max_q |M_ii(q)|

    clipped into ``[a_floor, a_cap]``. The original took a Pinocchio-backed
    ``G1UpperBody``; here the reduced model is built inline from the vendored
    URDF so the function is standalone. Requires Pinocchio (build-time only).
    """
    import pinocchio as pin

    model, opt2pin = build_reduced_model(joint_names, urdf_path=urdf_path)
    data = model.createData()
    n = len(joint_names)
    if q_bounds is None:
        lo, hi = urdf_position_limits(joint_names, urdf_path=urdf_path)
    else:
        lo, hi = (np.asarray(q_bounds[0], float), np.asarray(q_bounds[1], float))
    if tau_limit is None:
        tau_limit = isaac_effort_limits(joint_names)

    rng = np.random.default_rng(seed)
    zero_q = np.zeros(model.nq)
    zero_v = np.zeros(model.nv)
    M_diag_max = np.zeros(n)
    g_max = np.zeros(n)
    for _ in range(max(int(n_samples), 1)):
        q_opt = rng.uniform(lo, hi)
        q = zero_q.copy()
        q[opt2pin] = q_opt
        M = np.asarray(pin.crba(model, data, q))  # crba fills the upper triangle
        M_diag_max = np.maximum(M_diag_max, np.abs(np.diag(M)[opt2pin]))
        g = np.asarray(pin.rnea(model, data, q, zero_v, zero_v))[opt2pin]
        g_max = np.maximum(g_max, np.abs(g))
    headroom = np.clip(np.asarray(tau_limit, float) - g_max, 0.0, None)
    qdd = float(eta_tau) * headroom / np.clip(M_diag_max, 1e-6, None)
    return np.clip(qdd, float(a_floor), float(a_cap))


def build_reduced_model(joint_names=G1_UPPER_BODY_JOINTS, *, urdf_path=None):
    """(pinocchio model, opt2pin index array) for the 17 upper-body joints.

    Mirrors ``trajopt/robot.py::G1UpperBody.__init__``: lock every other joint at
    neutral, then map optimizer index -> ``idx_q``. Pinocchio is a build-time
    dependency only; the torch modules never import it.
    """
    import pinocchio as pin

    path = str(urdf_path) if urdf_path is not None else str(default_urdf_path())
    full = pin.buildModelFromUrdf(path)
    keep = {int(full.getJointId(n)) for n in joint_names}
    missing = [n for n in joint_names if int(full.getJointId(n)) <= 0]
    if missing:
        raise ValueError(f"URDF {path} lacks joints: {missing}")
    lock = [j for j in range(1, full.njoints) if j not in keep]
    try:
        model = pin.buildReducedModel(full, lock, pin.neutral(full))
    except Exception as exc:  # old pybind wheels lack vector<unsigned long>
        if "std::vector<unsigned long" not in str(exc):
            raise
        model = full
    opt2pin = np.array(
        [model.joints[model.getJointId(n)].idx_q for n in joint_names], dtype=np.int64
    )
    return model, opt2pin
