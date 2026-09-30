"""``ik.correct_tip``: re-aim a library terminal at the drum that is actually there.

``GoalPoseLibrary`` is keyed on ``(hand, drum, waist_yaw)`` and stores a JOINT
pose, so it knows nothing about where ``KitRandomization`` has moved the drum or
what stick is fitted.  The seed's terminal therefore drifts off the head -- a
median 12-15 cm at the shipped jitter -- and in phase 1 that drift is
UNLEARNABLE: the context carries the randomized geometry while the imitation
target ignores it.

These pin the correction: it converges, it touches only the one arm it is given,
it stays inside the joint box, and it is a no-op at zero steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import G1_UPPER_BODY_JOINTS as JOINTS, STICK_LENGTH
from amortized_opt.fk import StickSpec
from amortized_opt.ik import (
    adjust_arm_terminal,
    arm_joint_indices,
    correct_quintic_init_arms,
    correct_tip,
    tip_jacobian,
    tip_position,
)
from amortized_opt.limits import urdf_position_limits
from amortized_opt.rhythm import KitRandomization
from amortized_opt.waist_aim import aim_yaw
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, warmstart_trajectory

DT = torch.float64
_YAW = JOINTS.index("waist_yaw_joint")


@pytest.fixture(scope="module")
def rig():
    return (A.ChainFK(device="cpu", dtype=DT), A.DrumKit.from_layout(),
            GoalPoseLibrary(), urdf_position_limits(JOINTS))


def _batch(rig, hand, *, per_drum_xy=0.04, n_kits=25, seed=0):
    """Library terminals paired with the aim point of a RANDOMIZED kit."""
    _fk, kit0, lib, _qb = rig
    rng = np.random.default_rng(seed)
    Q, AIM = [], []
    for _ in range(n_kits):
        ks = kit0.randomized(rng, KitRandomization(per_drum_xy=per_drum_xy))
        for drum in lib.drums(hand):
            aim = ks.positions[ks.names.index(drum)]
            yt = float(aim_yaw({hand: np.asarray(aim, float)}, yaw_current=0.0))
            e = lib.select(hand, drum, yt)
            q = np.zeros(len(JOINTS))
            q[_YAW] = yt
            q[arm_joint_indices(hand, JOINTS)] = e.arm_q
            Q.append(q)
            AIM.append(aim)
    return (torch.as_tensor(np.stack(Q), dtype=DT),
            torch.as_tensor(np.stack(AIM), dtype=DT))


def _err(fk, q, aim, hand):
    with torch.no_grad():
        return (aim - tip_position(fk, q, hand, frame="env")).norm(dim=-1).numpy()


@pytest.mark.parametrize("hand", ["left", "right"])
def test_it_converges(rig, hand, capsys):
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, hand)
    e0 = _err(fk, q, aim, hand)
    e1 = _err(fk, correct_tip(fk, q, aim, hand, steps=3, q_bounds=qb), aim, hand)
    with capsys.disabled():
        print(f"\n[{hand}] tip err p50 {np.median(e0)*100:.2f} -> {np.median(e1)*100:.2f} cm, "
              f"p90 {np.percentile(e0,90)*100:.2f} -> {np.percentile(e1,90)*100:.2f} cm")
    assert np.median(e1) < 0.01                      # sub-centimetre median
    assert np.median(e1) < 0.1 * np.median(e0)       # and a >10x improvement


def test_more_jitter_costs_almost_nothing_once_corrected(rig):
    """The whole point: seed quality decouples from how hard the kit is jittered."""
    fk, _kit, _lib, qb = rig
    med = {}
    for pdx in (0.04, 0.10):
        q, aim = _batch(rig, "left", per_drum_xy=pdx)
        med[pdx] = np.median(_err(fk, correct_tip(fk, q, aim, "left", steps=3,
                                                  q_bounds=qb), aim, "left"))
    assert med[0.10] < 0.01
    assert med[0.10] < 3.0 * med[0.04] + 1e-3


@pytest.mark.parametrize("hand", ["left", "right"])
def test_only_that_arm_moves(rig, hand):
    """The waist is shared and already set by the yaw ramp -- it must not move."""
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, hand, n_kits=6)
    out = correct_tip(fk, q, aim, hand, steps=3, q_bounds=qb)
    other = "right" if hand == "left" else "left"
    assert torch.equal(out[:, _YAW], q[:, _YAW])
    for j in ("waist_pitch_joint", "waist_roll_joint"):
        assert torch.equal(out[:, JOINTS.index(j)], q[:, JOINTS.index(j)])
    oc = arm_joint_indices(other, JOINTS)
    assert torch.equal(out[:, oc], q[:, oc])
    assert not torch.equal(out[:, arm_joint_indices(hand, JOINTS)],
                           q[:, arm_joint_indices(hand, JOINTS)])


def test_stays_inside_the_joint_box(rig):
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, "right", n_kits=10)
    out = correct_tip(fk, q, aim, "right", steps=5, q_bounds=qb)
    lo, hi = (torch.as_tensor(b, dtype=DT) for b in qb)
    cols = arm_joint_indices("right", JOINTS)
    assert bool((out[:, cols] >= lo[cols] - 1e-12).all())
    assert bool((out[:, cols] <= hi[cols] + 1e-12).all())


def test_zero_steps_is_identity(rig):
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, "left", n_kits=3)
    assert torch.equal(correct_tip(fk, q, aim, "left", steps=0, q_bounds=qb), q)


def test_batching_does_not_change_the_answer(rig):
    """Batched is 200x cheaper, so it must give the same result as one at a time."""
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, "left", n_kits=4)
    full = correct_tip(fk, q, aim, "left", steps=2, q_bounds=qb)
    for i in (0, 3, len(q) - 1):
        one = correct_tip(fk, q[i:i+1], aim[i:i+1], "left", steps=2, q_bounds=qb)
        assert torch.allclose(one[0], full[i], atol=1e-10)


def test_shape_errors_are_reported(rig):
    fk, _kit, _lib, _qb = rig
    q, aim = _batch(rig, "left", n_kits=2)
    with pytest.raises(ValueError, match="must be"):
        correct_tip(fk, q[0], aim, "left")
    with pytest.raises(ValueError, match="batch mismatch"):
        correct_tip(fk, q, aim[:-1], "left")


# ------------------------------------------------------- randomized sticks --
_SPECS = [
    None,
    StickSpec("left", tilt=0.06),
    StickSpec("left", roll=-0.08, length=STICK_LENGTH - 0.03),
    StickSpec("left", tilt=-0.03, roll=0.05, length=STICK_LENGTH + 0.05),
]


def test_sigma_does_not_drop_the_stick(rig):
    """Regression: ``sigma=`` used to silently discard ``stick=``.

    The contact point at ``sigma = tip_sigma`` IS the tip of the fitted stick,
    so the two calls must agree -- and both must differ from the nominal-mount
    point (the old bug returned that instead, 8.9 cm off at this spec).
    """
    fk, _kit, _lib, _qb = rig
    q, _aim = _batch(rig, "left", n_kits=1)
    sp = _SPECS[3]
    with torch.no_grad():
        p_sig = tip_position(fk, q, "left", sigma=sp.tip_sigma, stick=sp)
        p_tip = tip_position(fk, q, "left", stick=sp)
        p_nom = tip_position(fk, q, "left", sigma=sp.tip_sigma)
    assert torch.allclose(p_sig, p_tip, atol=1e-12)
    assert float((p_sig - p_nom).norm(dim=-1).min()) > 0.01


def test_per_row_sticks_match_per_spec_calls(rig):
    """One batched call with per-row sticks == one call per spec, row for row."""
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, "left", n_kits=1)
    q, aim = q[: len(_SPECS)], aim[: len(_SPECS)]
    with torch.no_grad():
        pos = tip_position(fk, q, "left", stick=list(_SPECS))
    jac = tip_jacobian(fk, q, "left", stick=list(_SPECS))
    fix = correct_tip(fk, q, aim, "left", steps=2, q_bounds=qb, stick=list(_SPECS))
    for i, sp in enumerate(_SPECS):
        with torch.no_grad():
            p1 = tip_position(fk, q[i : i + 1], "left", stick=sp)
        J1 = tip_jacobian(fk, q[i : i + 1], "left", stick=sp)
        f1 = correct_tip(fk, q[i : i + 1], aim[i : i + 1], "left", steps=2,
                         q_bounds=qb, stick=sp)
        assert torch.allclose(pos[i], p1[0], atol=1e-12)
        assert torch.allclose(jac[i], J1[0], atol=1e-12)
        assert torch.allclose(fix[i], f1[0], atol=1e-10)


def test_per_row_correction_aims_each_row_own_stick(rig):
    """Each row converges measured WITH ITS OWN stick, in one batched call."""
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, "left", n_kits=1)
    q, aim = q[: len(_SPECS)], aim[: len(_SPECS)]
    out = correct_tip(fk, q, aim, "left", steps=3, q_bounds=qb, stick=list(_SPECS))
    with torch.no_grad():
        err = (aim - tip_position(fk, out, "left", frame="env",
                                  stick=list(_SPECS))).norm(dim=-1).numpy()
    assert np.median(err) < 0.01


def test_per_row_stick_batch_mismatch_is_reported(rig):
    fk, _kit, _lib, _qb = rig
    q, _aim = _batch(rig, "left", n_kits=1)
    with pytest.raises(ValueError, match="per-row stick batch"):
        tip_position(fk, q, "left", stick=[None, StickSpec("left")])


# ------------------------------------------- quintic-init one-shot (steps=1) --
@pytest.mark.parametrize("hand", ["left", "right"])
def test_one_step_full_arm_waist_frozen(rig, hand):
    """One Newton step, full 7 arm DoF: waist bit-identical, other arm untouched."""
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, hand, n_kits=6)
    out = adjust_arm_terminal(fk, q, aim, hand, steps=1, q_bounds=qb)
    other = "right" if hand == "left" else "left"
    assert torch.equal(out[:, _YAW], q[:, _YAW])
    for j in ("waist_pitch_joint", "waist_roll_joint"):
        assert torch.equal(out[:, JOINTS.index(j)], q[:, JOINTS.index(j)])
    oc = arm_joint_indices(other, JOINTS)
    assert torch.equal(out[:, oc], q[:, oc])
    assert not torch.equal(out[:, arm_joint_indices(hand, JOINTS)],
                           q[:, arm_joint_indices(hand, JOINTS)])


def test_one_step_reduces_lean_miss(rig):
    """Library arm on a leaned (0.2 rad pitch) waist: 1-step cuts env-frame error."""
    fk, _kit, lib, qb = rig
    q, aim = _batch(rig, "left", n_kits=8, seed=3)
    pitch_i = JOINTS.index("waist_pitch_joint")
    q = q.clone()
    q[:, pitch_i] = 0.2
    e0 = _err(fk, q, aim, "left")
    e1 = _err(fk, adjust_arm_terminal(fk, q, aim, "left", steps=1, q_bounds=qb),
              aim, "left")
    assert np.median(e1) < np.median(e0)


def test_one_step_beats_raw_library_median(rig):
    """1-step DLS is a seed neighborhood (~5 cm), not a 3-step strike."""
    fk, _kit, _lib, qb = rig
    q, aim = _batch(rig, "left")
    e0 = _err(fk, q, aim, "left")
    e1 = _err(fk, adjust_arm_terminal(fk, q, aim, "left", steps=1, q_bounds=qb),
              aim, "left")
    assert np.median(e1) < 0.5 * np.median(e0) or np.median(e1) < 0.08


@pytest.mark.parametrize("moving", [False, True])
def test_quintic_rebuild_uses_corrected_terminal(rig, moving):
    """``correct_quintic_init_arms`` writes ``terminal_arm`` and still starts at q0."""
    fk, kit, lib, qb = rig
    hand = "left"
    drum = lib.drums(hand)[0]
    aim = kit.positions[kit.names.index(drum)]
    yt = float(aim_yaw({hand: np.asarray(aim, float)}, yaw_current=0.0))
    e = lib.select(hand, drum, yt)
    n = len(JOINTS)
    q0 = np.zeros(n)
    q0[_YAW] = 0.0
    q0[arm_joint_indices(hand, JOINTS)] = e.arm_q * 0.3
    v0 = np.linspace(-0.3, 0.4, n) if moving else np.zeros(n)
    H, k_hit = 20, 10
    ws = warmstart_trajectory(
        q0, v0, horizon=H,
        goals={hand: (drum, k_hit * 0.02), "right": None},
        library=lib, aim_targets={hand: aim}, dt=0.02,
        joint_names=JOINTS, q_bounds=qb,
    )
    init = torch.as_tensor(np.asarray(ws.init_traj, float), dtype=DT)[None]
    q0t = torch.as_tensor(q0, dtype=DT)[None]
    hit = torch.tensor([[k_hit, -1]])
    aims = torch.zeros(1, 2, 3, dtype=DT)
    aims[0, 0] = torch.as_tensor(aim, dtype=DT)
    out = correct_quintic_init_arms(
        fk, init, q0t, hit, aims, steps=1, q_bounds=qb, joint_names=JOINTS,
        qd0=torch.as_tensor(v0, dtype=DT)[None] if moving else None,
    )
    acols = arm_slice(hand, JOINTS)
    q_seed = torch.zeros(1, n, dtype=DT)
    q_seed[0, :3] = init[0, k_hit - 1, :3]
    q_seed[0, acols] = init[0, k_hit - 1, acols]
    want = adjust_arm_terminal(
        fk, q_seed, aims[:, 0], hand, steps=1, q_bounds=qb,
    )[0, acols]
    assert torch.allclose(out[0, k_hit - 1, acols], want, atol=1e-8)
    assert not torch.allclose(out[0, k_hit - 1, acols], init[0, k_hit - 1, acols])
    # Rebuilt strike seed: knot 0 of the path is q0 (init is knots 1..H).
    from amortized_opt.strike_seed import accelerating_path_torch
    rebuilt = accelerating_path_torch(
        q0t[:, acols], torch.as_tensor(v0[acols], dtype=DT)[None], out[:, k_hit - 1, acols],
        torch.tensor([k_hit]), H,
    ).clamp(torch.as_tensor(qb[0], dtype=DT)[acols], torch.as_tensor(qb[1], dtype=DT)[acols])
    assert torch.allclose(out[0, :, acols], rebuilt[0], atol=1e-8)
