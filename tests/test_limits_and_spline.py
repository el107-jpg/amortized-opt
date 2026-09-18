"""Vendored limit constants and the quintic seed machinery."""

from __future__ import annotations

import numpy as np
import pytest

from amortized_opt import constants as C
from amortized_opt import limits as L
from amortized_opt.spline import joint_path_spline, peak_rate, quintic_time_scaling, retime_joint_path
from conftest import import_repo


def test_scale_soft_wrist_effort_is_scoring_only():
    names = C.G1_UPPER_BODY_JOINTS
    base = L.isaac_effort_limits(names)
    same = L.scale_soft_wrist_effort(base, names, scale=1.0)
    assert np.allclose(same, base)
    scaled = L.scale_soft_wrist_effort(base, names, scale=2.0)
    for i, n in enumerate(names):
        if n in L.SOFT_WRIST_EFFORT_JOINTS:
            assert scaled[i] == pytest.approx(2.0 * base[i])
            assert base[i] == pytest.approx(5.0)
        else:
            assert scaled[i] == pytest.approx(base[i])
    # Plant table unchanged.
    assert np.allclose(L.isaac_effort_limits(names), base)


def test_control_grid_matches_resolved_decision():
    assert C.CONTROL_HZ == 50.0 and C.CONTROL_DT == 0.02
    assert C.PD_HZ == 250.0 and abs(C.PD_DT - 1 / 250) < 1e-15
    # PD_SUBSTEPS is NOT ``CONTROL_DT / PD_DT`` any more.  It was, while the two
    # meant the same thing; since 2026-08-12 it is an INTEGRATION-ACCURACY knob
    # for ``simulate_pd``'s explicit recursion, decoupled from the rate Isaac
    # actually steps at, and it had to be raised when the runtime damping went to
    # Kd 4.0 (section 9 -- at S = 5 a held 50 mrad step reached 3.3e23 rad,
    # FINITE, so nothing caught it).  What must still hold is that it is an
    # integer multiple of the control/PD ratio, so every PD tick lands on a
    # substep boundary.
    ratio = round(C.CONTROL_DT / C.PD_DT)
    assert ratio == 5
    assert C.PD_SUBSTEPS >= ratio and C.PD_SUBSTEPS % ratio == 0


def test_limits_match_repo_runtime_values(capsys):
    prob = import_repo("trajopt.problem")
    names = C.G1_UPPER_BODY_JOINTS
    assert tuple(prob.G1_UPPER_BODY_JOINTS) == names
    pairs = {
        "effort": (L.isaac_effort_limits(names), prob.isaac_effort_limits(names)),
        "velocity": (L.isaac_velocity_limits(names), prob.isaac_velocity_limits(names)),
        "stiffness": (L.isaac_stiffness(names), prob.isaac_stiffness(names)),
        "damping": (L.isaac_damping(names), prob.isaac_damping(names)),
        "rt_stiffness": (L.isaac_runtime_stiffness(names), prob.isaac_runtime_stiffness(names)),
        "rt_damping": (L.isaac_runtime_damping(names), prob.isaac_runtime_damping(names)),
        "armature": (L.isaac_armature(names), prob.isaac_armature(names)),
    }
    worst = {k: float(np.abs(a - b).max()) for k, (a, b) in pairs.items()}
    with capsys.disabled():
        print("\n[limits vs repo] " + ", ".join(f"{k}={v:.1e}" for k, v in worst.items()))
    assert all(v == 0.0 for v in worst.values())


def test_runtime_gains_are_the_scaled_table():
    """The CLAUDE.md table is the PRE-scale value; the plant runs the scales.

    Since 2026-08-12 (section 9) those are 2x on stiffness and 4x on damping, so
    the arms run Kp 80 / Kd 4.0 and the waist Kp 400 / Kd 20.0.  This test is
    deliberately written against the scale CONSTANTS rather than against frozen
    numbers -- the point it defends is that ``*_runtime_*`` applies the scale at
    all, not any particular value of it.  The frozen numbers live in
    ``test_limits_match_repo``, which compares against the repo original.
    """
    names = C.G1_UPPER_BODY_JOINTS
    assert np.allclose(L.isaac_runtime_damping(names),
                       L.G1_PD_DAMPING_SCALE * L.isaac_damping(names))
    assert np.allclose(L.isaac_runtime_stiffness(names),
                       L.G1_PD_STIFFNESS_SCALE * L.isaac_stiffness(names))
    i = names.index("left_elbow_joint")
    assert L.isaac_damping(names)[i] == 1.0
    assert L.isaac_runtime_damping(names)[i] == L.G1_PD_DAMPING_SCALE
    j = names.index("waist_yaw_joint")
    assert L.isaac_damping(names)[j] == 5.0
    assert L.isaac_runtime_damping(names)[j] == 5.0 * L.G1_PD_DAMPING_SCALE


def test_position_limits_match_repo(capsys):
    """URDF box + posture narrowing, same as trajopt/robot.py's default branch."""
    pytest.importorskip("pinocchio")
    prob = import_repo("trajopt.problem")
    robot_mod = import_repo("trajopt.robot")
    cfg = prob.TrajOptConfig()
    rob = robot_mod.G1UpperBody(cfg)
    lo, hi = L.urdf_position_limits(limit_margin=cfg.limit_margin)
    e = max(float(np.abs(lo - rob.q_lower).max()), float(np.abs(hi - rob.q_upper).max()))
    with capsys.disabled():
        print(f"\n[position box vs repo] max abs err = {e:.3e} rad")
    assert e <= 1e-12


def test_derive_accel_limits_matches_repo(capsys):
    pytest.importorskip("pinocchio")
    prob = import_repo("trajopt.problem")
    robot_mod = import_repo("trajopt.robot")
    safety_limits = import_repo("safety.limits")
    cfg = prob.TrajOptConfig()
    rob = robot_mod.G1UpperBody(cfg)
    ref = safety_limits.derive_accel_limits(rob, n_samples=64)
    ours = L.derive_accel_limits(
        n_samples=64,
        tau_limit=np.asarray(rob.tau_limit, float),
        q_bounds=(rob.q_lower, rob.q_upper),
    )
    e = float(np.abs(ref - ours).max())
    with capsys.disabled():
        print(f"\n[derive_accel_limits vs repo] max abs err = {e:.3e} rad/s^2 "
              f"(range {ours.min():.2f}..{ours.max():.2f})")
    assert e <= 1e-9


# ------------------------------------------------------------------- spline --
def test_spline_matches_repo_primitives(capsys):
    ref = import_repo("trajopt.spline")
    rng = np.random.default_rng(0)
    Q = np.cumsum(rng.normal(size=(9, 17)) * 0.1, axis=0)
    sa, la = ref.joint_path_spline(Q)
    sb, lb = joint_path_spline(Q)
    u = np.linspace(0.0, la, 41)
    e_path = float(np.abs(np.asarray(sa(u)) - np.asarray(sb(u))).max())

    ua = ref.quintic_time_scaling(0.5, la, 0.3, 0.1, 2.0, 0.0)
    ub = quintic_time_scaling(0.5, lb, 0.3, 0.1, 2.0, 0.0)
    t = np.linspace(0.0, 0.5, 51)
    e_time = float(np.abs(ua(t) - ub(t)).max())
    e_rate = abs(ref.peak_rate(ua) - peak_rate(ub))
    with capsys.disabled():
        print(f"\n[spline vs repo] path {e_path:.3e}, timing {e_time:.3e}, peak_rate {e_rate:.3e}")
    assert abs(la - lb) == 0.0 and e_path == 0.0 and e_time == 0.0 and e_rate == 0.0


def test_retime_endpoints_and_grid():
    rng = np.random.default_rng(1)
    Q = np.cumsum(rng.normal(size=(6, 17)) * 0.1, axis=0)
    n = 25
    X, rate = retime_joint_path(Q, n, C.CONTROL_DT, tip_speed=3.0, tip_gain=0.4)
    assert X.shape == (n, 17)
    assert np.allclose(X[0], Q[0]) and np.allclose(X[-1], Q[-1])
    assert np.isfinite(rate) and rate > 0.0


def test_retime_degenerate_path_holds_pose():
    Q = np.zeros((4, 17))
    X, rate = retime_joint_path(Q, 10, C.CONTROL_DT)
    assert X.shape == (10, 17) and np.allclose(X, 0.0) and rate == 0.0


def test_isaac_rl_pd_gains_match_trainer():
    """isaac_rl / isaac_rl_ppo must not quietly ship a different Kd than pd_sim."""
    from isaac_rl.env_cfg import DrumRLEnvCfg
    from isaac_rl_ppo.cfg import EnvConfig
    from isaac_rl_ppo.env_cfg import DrumRLEnvCfg as PpoDrumRLEnvCfg

    want_kd = float(L.G1_PD_DAMPING_SCALE)
    want_kp = float(L.G1_PD_STIFFNESS_SCALE)
    assert DrumRLEnvCfg().pd_damping_scale == want_kd
    assert DrumRLEnvCfg().pd_stiffness_scale == want_kp
    assert PpoDrumRLEnvCfg().pd_damping_scale == want_kd
    assert PpoDrumRLEnvCfg().pd_stiffness_scale == want_kp
    assert EnvConfig().pd_damping_scale == want_kd
    assert want_kd == 4.0
    assert want_kp == 1.0
