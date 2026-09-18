"""``waist_upright_ease_wp``: seeds recover upright instead of holding the lean.

The starts sample waist pitch/roll in a posture box, and the old seed held the
measured value for the whole horizon -- so every phase-1 imitation target
taught "keep whatever lean you started with", the prior the phase-2 upright
well then had to fight.  The ease replaces the hold with a quintic smoothstep
to upright, arriving at rest; OFF must remain bit-for-bit the old behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.limits import ETA_VELOCITY, isaac_velocity_limits
from amortized_opt.warmstart import GoalPoseLibrary, warmstart_trajectory

NAMES = tuple(G1_UPPER_BODY_JOINTS)
PITCH = NAMES.index("waist_pitch_joint")
ROLL = NAMES.index("waist_roll_joint")
YAW = NAMES.index("waist_yaw_joint")


@pytest.fixture(scope="module")
def library():
    return GoalPoseLibrary(joint_names=NAMES)


def _leaned_start() -> tuple[np.ndarray, np.ndarray]:
    q0 = np.zeros(len(NAMES))
    q0[PITCH], q0[ROLL], q0[YAW] = 0.30, -0.20, 0.40
    return q0, np.zeros(len(NAMES))


def _build(library, **kw):
    q0, v0 = _leaned_start()
    return q0, warmstart_trajectory(
        q0, v0, horizon=25, goals={"left": None, "right": None},
        library=library, dt=CONTROL_DT, joint_names=NAMES, **kw
    )


def test_off_holds_the_measured_lean(library):
    """The pre-existing behaviour: pitch/roll constant at the sampled pose."""
    q0, ws = _build(library)
    traj = np.asarray(ws.init_traj, float)
    assert np.allclose(traj[:, PITCH], q0[PITCH])
    assert np.allclose(traj[:, ROLL], q0[ROLL])


def test_ease_recovers_upright_and_holds(library):
    # ``init_traj`` rows are waypoints 1..H (the measured q0 is row -1 of the
    # implied grid), so a 5-waypoint ease holds from row index 4 on.
    q0, ws = _build(library, waist_upright_ease_wp=5)
    traj = np.asarray(ws.init_traj, float)
    v_lim = isaac_velocity_limits(NAMES)
    for col in (PITCH, ROLL):
        # first step leaves the measured pose smoothly, toward upright
        assert abs(traj[0, col]) < abs(q0[col])
        assert abs(traj[0, col] - q0[col]) <= ETA_VELOCITY * v_lim[col] * CONTROL_DT
        assert np.allclose(traj[4:, col], 0.0, atol=1e-12)
        # monotone recovery, no overshoot past upright
        d = np.diff(np.concatenate([[q0[col]], traj[:5, col]]))
        assert np.all(d <= 1e-12) if q0[col] > 0 else np.all(d >= -1e-12)
    # the yaw ramp is untouched by the ease (no aim targets -> hold)
    assert traj[0, YAW] == pytest.approx(q0[YAW])


def test_ease_is_velocity_admissible_at_the_worst_lean(library):
    """A full-box lean (0.52 rad) must stretch the ease under the speed rating."""
    q0 = np.zeros(len(NAMES))
    q0[PITCH] = 0.52
    ws = warmstart_trajectory(
        q0, np.zeros(len(NAMES)), horizon=25, goals={"left": None, "right": None},
        library=library, dt=CONTROL_DT, joint_names=NAMES,
        waist_upright_ease_wp=5,
    )
    traj = np.asarray(ws.init_traj, float)
    v_lim = isaac_velocity_limits(NAMES)
    steps = np.abs(np.diff(traj[:, PITCH]))
    assert steps.max() <= ETA_VELOCITY * v_lim[PITCH] * CONTROL_DT + 1e-9
    assert traj[-1, PITCH] == pytest.approx(0.0)


def test_explicit_setpoint_is_the_ease_target(library):
    """``waist_setpoint`` (the planned hold, e.g. the MPC loop's previous
    command) becomes the ease target instead of upright."""
    sp = np.array([0.0, 0.10, -0.05])  # (yaw, pitch, roll) in waist order
    q0, ws = _build(library, waist_upright_ease_wp=5, waist_setpoint=sp)
    traj = np.asarray(ws.init_traj, float)
    # first waypoint has left the measured pose toward the setpoint
    assert q0[PITCH] > traj[0, PITCH] > 0.10
    assert np.allclose(traj[4:, PITCH], 0.10)
    assert np.allclose(traj[4:, ROLL], -0.05)


def test_roller_ease_matches_the_numpy_seed(library):
    """The GPU regoal seed (``vel_ramp_seed`` + the stretch rule + the single
    correction round, the construction ``MPCRoller._rebuild_waist_from_aim``
    uses) must reproduce the numpy seed's waist block exactly: the roller is
    the training distribution and the numpy path is what the deployed
    generator runs.  Nonzero start rates exercise the velocity matching;
    ``v0 = 0`` covers ``IsaacMPC.reset``'s rest-to-rest ramp."""
    import torch

    from amortized_opt.train_cost import vel_ramp_seed

    q0, _v = _leaned_start()
    v0 = np.zeros(len(NAMES))
    v0[PITCH], v0[ROLL] = -0.8, 0.5
    ws = warmstart_trajectory(
        q0, v0, horizon=25, goals={"left": None, "right": None},
        library=library, dt=CONTROL_DT, joint_names=NAMES,
        waist_upright_ease_wp=5,
    )
    traj = np.asarray(ws.init_traj, float)  # rows are waypoints 1..H
    H = traj.shape[0]
    wc = [i for i, n in enumerate(NAMES) if n.startswith("waist_")]
    v_lim = isaac_velocity_limits(NAMES)

    q0w = torch.tensor(q0[wc], dtype=torch.float64)[None]
    term = q0w.clone()
    pr = [j for j, c in enumerate(wc) if c != YAW]
    term[:, pr] = 0.0
    step = torch.tensor(
        ETA_VELOCITY * v_lim[wc] * CONTROL_DT, dtype=torch.float64
    )[None].clamp_min(1e-9)
    cap = step / CONTROL_DT
    v0w = torch.zeros_like(q0w)
    v0w[:, pr] = torch.maximum(
        torch.minimum(torch.tensor(v0[wc], dtype=torch.float64)[None], cap), -cap
    )[:, pr]
    need = torch.ceil(1.875 * (term - q0w).abs() / step)
    L = need.clamp(5.0, float(H))
    sub = vel_ramp_seed(q0w, v0w, term, L, H, CONTROL_DT)
    d0 = (sub[:, 0, :] - q0w).abs().unsqueeze(1)
    peak = torch.cat([d0, (sub[:, 1:] - sub[:, :-1]).abs()], 1).amax(1)
    ratio = peak / step
    over = ratio > 1.0
    L = torch.where(over, (L * ratio).ceil().clamp(1.0, float(H)), L)
    sub = vel_ramp_seed(q0w, v0w, term, L, H, CONTROL_DT)[0].numpy()
    assert np.allclose(sub, traj[:, wc], atol=1e-12)


def test_velocity_matched_ease_continues_the_measured_rate(library):
    """A seed rebuilt mid-recovery CONTINUES at the measured rate instead of
    braking to zero slope -- the fix for the per-tick re-anchor stall (the
    zero-slope ease progressed only ~s(1/k) of the remaining lean per tick)."""
    q0 = np.zeros(len(NAMES))
    q0[PITCH] = 0.30
    v0 = np.zeros(len(NAMES))
    v0[PITCH] = -1.0  # already recovering toward upright
    ws = warmstart_trajectory(
        q0, v0, horizon=25, goals={"left": None, "right": None},
        library=library, dt=CONTROL_DT, joint_names=NAMES,
        waist_upright_ease_wp=5,
    )
    traj = np.asarray(ws.init_traj, float)
    step0 = traj[0, PITCH] - q0[PITCH]
    # p'(0) = v0 exactly: at a long window the first discrete step converges
    # to v0*dt (at short windows quintic curvature adds to it)
    from amortized_opt.warmstart import _boundary_quintic

    fine = _boundary_quintic(q0[PITCH], v0[PITCH], 0.0, 50, 60, CONTROL_DT)
    assert fine[1] - fine[0] == pytest.approx(v0[PITCH] * CONTROL_DT, rel=2e-2)
    # the seed CONTINUES the recovery: its first step outruns what the
    # re-anchored zero-slope ease would take from the same state
    u1 = 1.0 / 5.0
    zero_slope_step = (u1 ** 3 * (10 - 15 * u1 + 6 * u1 * u1)) * (0.0 - q0[PITCH])
    assert step0 < zero_slope_step < 0.0
    # still arrives at rest, upright, and holds
    assert traj[-1, PITCH] == pytest.approx(0.0, abs=1e-12)
    assert abs(traj[-1, PITCH] - traj[-2, PITCH]) < 1e-9
    # a rest start is bit-identical to the pre-velocity-matched smoothstep
    q0r, wsr = _build(library, waist_upright_ease_wp=5)
    u = np.clip(np.arange(1, 26) / 5.0, 0.0, 1.0)
    s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
    ref = q0r[PITCH] * (1.0 - s)
    assert np.allclose(np.asarray(wsr.init_traj, float)[:, PITCH], ref, atol=1e-12)


def test_velocity_matched_ease_is_admissible_with_an_adverse_rate(library):
    """Full-box lean plus the rate cap pointing AWAY from upright: the window
    rule + the correction round must keep every step inside the rating."""
    v_lim = isaac_velocity_limits(NAMES)
    q0 = np.zeros(len(NAMES))
    q0[PITCH] = 0.52
    v0 = np.zeros(len(NAMES))
    v0[PITCH] = ETA_VELOCITY * v_lim[PITCH]  # leaning further, at the cap
    ws = warmstart_trajectory(
        q0, v0, horizon=25, goals={"left": None, "right": None},
        library=library, dt=CONTROL_DT, joint_names=NAMES,
        waist_upright_ease_wp=5,
    )
    traj = np.asarray(ws.init_traj, float)
    steps = np.abs(np.diff(np.concatenate([[q0[PITCH]], traj[:, PITCH]])))
    assert steps.max() <= ETA_VELOCITY * v_lim[PITCH] * CONTROL_DT + 1e-9
    assert traj[-1, PITCH] == pytest.approx(0.0, abs=1e-9)


def test_ease_coexists_with_an_active_hand(library):
    """The ease with a real goal in the same build: the arm loop's ``peak``
    bookkeeping must survive the ease branch (regression: a local in the ease
    loop shadowed the ``peak`` dict and every ACTIVE-hand build crashed --
    which is exactly the combination the ease tests' idle-only builds never
    exercised, and the combination every real pool generates)."""
    q0, v0 = _leaned_start()
    ws = warmstart_trajectory(
        q0, v0, horizon=25, goals={"left": ("snare", 0.4), "right": None},
        library=library, dt=CONTROL_DT, joint_names=NAMES,
        waist_upright_ease_wp=5,
    )
    traj = np.asarray(ws.init_traj, float)
    assert "left" in ws.peak_rate  # the arm bookkeeping survived
    assert np.allclose(traj[10:, PITCH], 0.0, atol=1e-9)  # ease still landed


def test_apex_builder_threads_the_ease(library):
    from amortized_opt.warmstart_apex import apex_warmstart_trajectory

    q0, v0 = _leaned_start()
    ws = apex_warmstart_trajectory(
        q0, v0, horizon=25, goals={"left": None, "right": None},
        library=library, apex_arm={"left": None, "right": None},
        dt=CONTROL_DT, joint_names=NAMES, waist_upright_ease_wp=5,
    )
    traj = np.asarray(ws.target, float)
    assert np.allclose(traj[4:, PITCH], 0.0, atol=1e-12)
