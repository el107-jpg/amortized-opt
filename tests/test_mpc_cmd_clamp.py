"""The executed MPC command must be clamped into the joint box.

Prerequisite fix A gave ``MPCGenerator`` three guards against the waist wind-up
loop, all of them on the *seed*.  None of them bounded the command a LEARNED
planner emits, and ``run_episode`` slew-limited the executed command but never
clamped it.  That made this loop the only place in the stack where a planner
could integrate its own command outside the joint box:
:class:`~amortized_opt.train_cost.MPCRoller` -- the loop the amortized model is
trained against -- has always clamped, and so does the plant (Isaac's implicit
actuators saturate a position target at the joint limit, as does hardware).

Measured on ``runs/phase2_main/last.pt`` before the fix: all three eval clips
tripped the divergence guard after 89 / 30 / 72 ticks, with the COMMAND leaving
the box one tick before the measured pose did and ``waist_roll`` walking to
-1.02 rad against a +/-0.52 rad limit -- at 3 % of the joint's speed rating, so
the slew limiter never saw anything wrong.  After the fix all three run the full
250 ticks and the hit count goes 11 / 1 / 3 -> 23 / 19 / 13.
"""

from __future__ import annotations

import numpy as np
import pytest

from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator
from amortized_opt.rhythm import DrumKit
from amortized_opt.starts import StartSampler


def _out_of_box(x, lo, hi):
    return float(np.maximum(lo[None, :] - x, x - hi[None, :]).max(initial=0.0))


def _runaway_planner(gen, *, joint: str, per_tick: float = 0.03):
    """A planner that walks one joint out of its box, slowly and admissibly.

    Deliberately gentle: ``per_tick`` rad per 20 ms is a few percent of every
    joint's speed rating, so the slew limiter is not the guard under test.
    """
    j = gen.names.index(joint)
    H, n = gen.cfg.horizon, gen.n

    def plan(req):
        traj = np.repeat(np.asarray(req.q, float)[None, :], H, axis=0)
        traj[:, j] -= per_tick * np.arange(1, H + 1)
        return traj

    return plan


@pytest.fixture(scope="module")
def kit():
    return DrumKit.from_layout()


@pytest.fixture(scope="module")
def start(kit):
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=25))
    s = StartSampler(kit=kit, library=gen.library, fk=gen.fk, rnea=gen.rnea,
                     sdf=gen.sdf).sample(1, seed=0)
    return s.q[0].cpu().numpy(), s.qd[0].cpu().numpy()


@pytest.fixture(scope="module")
def clip(kit):
    from amortized_opt.rhythm import default_rcc_dir, load_rcc_pool

    return load_rcc_pool(default_rcc_dir("eval"), kit)[0]


@pytest.mark.parametrize("joint", ["waist_roll_joint", "left_shoulder_yaw_joint"])
def test_clamp_keeps_command_in_box(kit, start, clip, joint):
    """With the clamp ON, a runaway planner cannot leave the box at all."""
    q0, qd0 = start
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=25, clamp_cmd_to_box=True))
    res = gen.run_episode(clip, q0, qd0, seed=0,
                          planner=_runaway_planner(gen, joint=joint), max_steps=120)
    lo, hi = gen.q_bounds
    assert res.cmd_hist.shape[0] == 120, "episode must not diverge"
    assert not res.diverged
    assert _out_of_box(res.cmd_hist, lo, hi) <= 1e-9


def _future_runaway_planner(gen, *, joint: str, offset: float = 1.5):
    """Writes future knots outside the box; first knot stays at the measured pose.

    The command clamp only sees ``traj[0]``, so without ``clamp_init_to_box``
    the shifted tail becomes next tick's seed.
    """
    j = gen.names.index(joint)
    H, n = gen.cfg.horizon, gen.n

    def plan(req):
        traj = np.repeat(np.asarray(req.q, float)[None, :], H, axis=0)
        lo, hi = gen.q_bounds
        traj[1:, j] = lo[j] - float(offset)
        return traj

    return plan


def test_clamp_init_keeps_shifted_plan_in_box(kit, start, clip):
    """Next-tick init is in the box even if the planner's tail is not."""
    q0, qd0 = start
    gen = MPCGenerator(
        kit=kit, config=MPCGenConfig(horizon=25, clamp_cmd_to_box=True,
                                     clamp_init_to_box=True, freeze_wp=0),
    )
    captured = []

    inner = _future_runaway_planner(gen, joint="waist_roll_joint")

    def plan(req):
        captured.append(np.asarray(req.init_traj, float).copy())
        return inner(req)

    res = gen.run_episode(clip, q0, qd0, seed=0, planner=plan, max_steps=40)
    lo, hi = gen.q_bounds
    assert res.cmd_hist.shape[0] == 40
    assert not res.diverged
    assert _out_of_box(res.cmd_hist, lo, hi) <= 1e-9
    # tick 0 is the quintic seed; every later seed is a clipped shift
    for init in captured[1:]:
        assert _out_of_box(init, lo, hi) <= 1e-9


def test_without_init_clamp_shifted_plan_leaves_box(kit, start, clip):
    """The regression: clamp_init OFF lets the illegal tail become the seed."""
    q0, qd0 = start
    gen = MPCGenerator(
        kit=kit, config=MPCGenConfig(horizon=25, clamp_cmd_to_box=True,
                                     clamp_init_to_box=False, freeze_wp=0),
    )
    captured = []
    inner = _future_runaway_planner(gen, joint="waist_roll_joint")

    def plan(req):
        captured.append(np.asarray(req.init_traj, float).copy())
        return inner(req)

    gen.run_episode(clip, q0, qd0, seed=0, planner=plan, max_steps=8)
    lo, hi = gen.q_bounds
    assert len(captured) >= 2
    assert _out_of_box(captured[1], lo, hi) > 0.5


@pytest.mark.parametrize("joint", ["waist_roll_joint"])
def test_without_clamp_the_command_runs_away(kit, start, clip, joint):
    """The regression itself: clamp OFF reproduces the divergence."""
    q0, qd0 = start
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=25, clamp_cmd_to_box=False))
    res = gen.run_episode(clip, q0, qd0, seed=0,
                          planner=_runaway_planner(gen, joint=joint), max_steps=120)
    lo, hi = gen.q_bounds
    assert _out_of_box(res.cmd_hist, lo, hi) > 0.1
    assert res.diverged, "the guard should catch the runaway when nothing clamps"


def test_clamp_does_not_perturb_an_admissible_planner(kit, start, clip):
    """A planner already inside the box gets a bit-identical episode."""
    q0, qd0 = start
    out = []
    for on in (True, False):
        gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=25, clamp_cmd_to_box=on))
        res = gen.run_episode(clip, q0, qd0, seed=0, max_steps=60)  # quintic stand-in
        out.append(res)
    lo, hi = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=25)).q_bounds
    assert _out_of_box(out[0].cmd_hist, lo, hi) <= 1e-9
    np.testing.assert_allclose(out[0].cmd_hist, out[1].cmd_hist, atol=1e-12)
    np.testing.assert_allclose(out[0].q_hist, out[1].q_hist, atol=1e-12)
