"""Phase-1 quintic warm start and its input noise."""

from __future__ import annotations

import numpy as np
import pytest

from amortized_opt import constants as C
from amortized_opt.limits import urdf_position_limits
from amortized_opt.rhythm import DrumKit
from amortized_opt.waist_aim import aim_targets_from_layout
from amortized_opt.warmstart import (
    GoalPoseLibrary,
    InitNoise,
    arm_slice,
    correlated_noise,
    warmstart_trajectory,
)


@pytest.fixture(scope="module")
def library():
    return GoalPoseLibrary()


def test_library_covers_every_eligible_hand_drum_pair(library, capsys):
    kit = DrumKit.from_layout()
    missing = []
    for hand in ("left", "right"):
        for d in kit.eligible(hand):
            if not library.has(hand, kit.names[d]):
                missing.append((hand, kit.names[d]))
    with capsys.disabled():
        print(f"\n[goal-pose library] {len(library.pairs)} (hand, drum) pairs, "
              f"yaw grid {len(library.yaw_grid_rad)} steps; "
              f"missing eligible pairs={missing}")
    # The library is missing exactly the two no-crossover extremes, which are
    # not eligible for that hand anyway.
    assert not missing
    assert not library.has("left", "floor_tom")
    assert not library.has("right", "hihat")


def test_library_selection_is_closest_in_yaw_among_valid_entries(library):
    for hand, drum in library.pairs:
        rows = library.entries(hand, drum)
        valid = [
            e for e in rows
            if e.tip_err != e.tip_err
            or (e.tip_err <= library.max_tip_err and e.wrist_headroom >= library.min_wrist_headroom)
        ]
        pool = valid or rows
        for target in (-1.5, -0.5, 0.0, 0.5, 1.5):
            got = library.select(hand, drum, target)
            assert got in pool
            if valid:
                best = min(abs(e.waist_yaw - target) for e in pool)
                assert abs(got.waist_yaw - target) <= best + 1e-12
    with pytest.raises(KeyError):
        library.select("left", "floor_tom", 0.0)


def test_terminal_waypoint_matches_the_library_pose(library, capsys):
    q0 = C.ready_joint_pos()
    v0 = np.zeros(C.N_DOF)
    worst = 0.0
    cases = 0
    for lh, rh, tl, tr in (
        ("hihat", "snare", 0.40, 0.26),
        ("tom2", "tom1", 0.60, 0.34),
        ("cymbal_left", "floor_tom", 0.80, 0.48),
        ("snare", "cymbal_right", 0.24, 0.62),
    ):
        ws = warmstart_trajectory(
            q0, v0, horizon=60,
            goals={"left": (lh, tl), "right": (rh, tr)},
            library=library,
            aim_targets=aim_targets_from_layout({"left": lh, "right": rh}),
        )
        for hand, drum in (("left", lh), ("right", rh)):
            k = ws.hit_index[hand]
            assert k == round({"left": tl, "right": tr}[hand] / C.CONTROL_DT)
            want = library.select(hand, drum, ws.yaw_target).arm_q
            worst = max(worst, float(np.abs(ws.q[k, arm_slice(hand)] - want).max()))
            # ...and the arm HOLDS after its hit rather than drifting.
            assert np.allclose(ws.q[k:, arm_slice(hand)], ws.q[k, arm_slice(hand)])
            cases += 1
    with capsys.disabled():
        print(f"\n[warmstart terminal, {cases} arm-cases] max |q_terminal - library| = {worst:.2e} rad")
    assert worst <= 1e-9


def test_hit_lands_on_the_20ms_grid(library):
    q0 = C.ready_joint_pos()
    for tth in (0.13, 0.207, 0.4, 0.55, 0.833):
        ws = warmstart_trajectory(
            q0, np.zeros(C.N_DOF), horizon=80,
            goals={"left": None, "right": ("snare", tth)},
            library=library,
        )
        k = ws.hit_index["right"]
        assert abs(k * C.CONTROL_DT - tth) <= C.CONTROL_DT / 2 + 1e-12


def test_start_pose_and_velocity_are_honoured(library):
    rng = np.random.default_rng(0)
    q0 = C.ready_joint_pos() + rng.normal(0.0, 0.1, size=C.N_DOF)
    v0 = rng.normal(0.0, 0.5, size=C.N_DOF)
    ws = warmstart_trajectory(
        q0, v0, horizon=50,
        goals={"left": ("hihat", 0.4), "right": ("tom1", 0.3)},
        library=library,
        aim_targets=aim_targets_from_layout({"left": "hihat", "right": "tom1"}),
    )
    assert np.allclose(ws.q[0], q0)
    lo, hi = urdf_position_limits()
    assert (ws.q[1:] >= lo - 1e-12).all() and (ws.q[1:] <= hi + 1e-12).all()
    assert np.isfinite(ws.q).all()


def test_waist_is_yaw_only_and_reaches_the_aim(library):
    q0 = C.ready_joint_pos()
    q0[C.G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")] = 0.07
    q0[C.G1_UPPER_BODY_JOINTS.index("waist_roll_joint")] = -0.04
    ws = warmstart_trajectory(
        q0, np.zeros(C.N_DOF), horizon=50,
        goals={"left": ("hihat", 0.5), "right": ("snare", 0.4)},
        library=library,
        aim_targets=aim_targets_from_layout({"left": "hihat", "right": "snare"}),
    )
    pi = C.G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")
    ri = C.G1_UPPER_BODY_JOINTS.index("waist_roll_joint")
    yi = C.G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
    assert np.allclose(ws.q[:, pi], q0[pi]) and np.allclose(ws.q[:, ri], q0[ri])
    assert abs(ws.q[-1, yi] - ws.yaw_target) < 1e-9
    assert abs(ws.yaw_target) > 1e-3  # the aim law actually fired


def test_idle_hand_holds_its_pose(library):
    rng = np.random.default_rng(2)
    q0 = C.ready_joint_pos() + rng.normal(0.0, 0.05, size=C.N_DOF)
    ws = warmstart_trajectory(
        q0, np.zeros(C.N_DOF), horizon=30,
        goals={"left": None, "right": ("snare", 0.3)}, library=library,
    )
    idx = arm_slice("left")
    assert np.allclose(ws.q[:, idx], q0[idx])
    assert ws.hit_index["left"] == -1


def test_noise_is_zero_mean_and_temporally_correlated(capsys):
    rng = np.random.default_rng(0)
    noise = InitNoise()
    x = noise.sample((4000, C.N_DOF), rng)
    mean = float(x.mean())
    sd = float(x.std())
    lag1 = float(np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(C.N_DOF)]))
    lag5 = float(np.mean([np.corrcoef(x[:-5, j], x[5:, j])[0, 1] for j in range(C.N_DOF)]))
    with capsys.disabled():
        print(f"\n[init noise] mean={mean:+.4f} rad, sd={sd:.4f} rad, "
              f"lag-1 autocorr={lag1:.3f}, lag-5={lag5:.3f}")
    assert abs(mean) < 0.005
    assert lag1 > 0.5
    assert lag5 < lag1  # it decays -- it is not a constant offset
    assert 0.9 * np.hypot(noise.sigma_corr, noise.sigma_white) < sd
    assert sd < 1.1 * np.hypot(noise.sigma_corr, noise.sigma_white)


def test_pure_ou_noise_has_the_analytic_autocorrelation(capsys):
    rng = np.random.default_rng(1)
    for tau in (2.0, 5.0, 10.0):
        x = correlated_noise((20000, 4), rng, sigma=0.1, tau_steps=tau)
        got = float(np.mean([np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(4)]))
        want = float(np.exp(-1.0 / tau))
        with capsys.disabled():
            print(f"\n[OU tau={tau}] lag-1 autocorr {got:.4f} (analytic {want:.4f}); "
                  f"sd {float(x.std()):.4f} (target 0.1)")
        assert abs(got - want) < 0.02
        assert abs(float(x.std()) - 0.1) < 0.005


def test_noise_options(library):
    rng = np.random.default_rng(3)
    n = InitNoise(pin_start=True)
    x = n.sample((30, C.N_DOF), rng)
    assert np.allclose(x[0], 0.0)
    tap = InitNoise(taper=True, pin_start=False)
    y = tap.sample((30, C.N_DOF), rng, hit_index={"left": 10, "right": 20})
    li, ri = arm_slice("left"), arm_slice("right")
    assert np.allclose(y[10:, li], 0.0)
    assert np.allclose(y[20:, ri], 0.0)
    assert not np.allclose(y[5, li], 0.0)


def test_noised_seed_stays_in_the_joint_box(library):
    rng = np.random.default_rng(4)
    ws = warmstart_trajectory(
        C.ready_joint_pos(), np.zeros(C.N_DOF), horizon=50,
        goals={"left": ("hihat", 0.4), "right": ("snare", 0.3)}, library=library,
    )
    bounds = urdf_position_limits()
    out = InitNoise(sigma_corr=0.3, sigma_white=0.1).apply(
        ws.init_traj, rng, hit_index=ws.hit_index, q_bounds=bounds
    )
    assert (out >= bounds[0] - 1e-12).all() and (out <= bounds[1] + 1e-12).all()
    assert not np.allclose(out, ws.init_traj)
