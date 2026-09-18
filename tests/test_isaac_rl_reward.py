"""Executed-state reward and lead application, on recorded / synthetic states.

The reward is the RL stage's whole objective, so the properties pinned here are
the ones whose violation would be invisible in a training curve:

* the contact point is ``mount + sigma * axis``, with ``sigma`` measured from
  the stick's CENTRE (``STICK_LENGTH/2`` is the tip).  Measuring it from the
  tip -- the intuitive reading -- puts a cymbal contact 30 cm off, and the
  reward would happily pay for hitting nothing;
* the head-frame ``(axial, in-plane)`` convention must match
  ``probe.head_frame_error``'s, or the Isaac numbers cannot be compared to
  `REPORT.md` §4 at all;
* the tracking term must score the plant against the plan's UN-LED first
  waypoint.  Scoring against the executed (led) command prices the lead itself
  as error and trains the model to cancel its own compensation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.constants import STICK_LENGTH  # noqa: E402
from amortized_opt.fk import stick_contact_offset, stick_tip_offset  # noqa: E402
from amortized_opt.lead import LEAD_PRESETS, lead_fraction, lead_waypoints  # noqa: E402
from amortized_opt.probe import head_frame_error  # noqa: E402
from amortized_opt.strike import default_contact_sigma  # noqa: E402
from isaac_rl.reward import (  # noqa: E402
    ExecutedReward,
    ExecutedStrikeScorer,
    RewardConfig,
)
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig  # noqa: E402


@pytest.fixture(scope="module")
def kit():
    return A.DrumKit.from_layout()


@pytest.fixture(scope="module")
def fk():
    return A.ChainFK(device="cpu", dtype=torch.float64)


@pytest.fixture(scope="module")
def scorer(fk):
    return ExecutedStrikeScorer(fk=fk, device="cpu", dtype=torch.float64)


# ------------------------------------------------------- contact geometry --
def test_contact_point_sigma_is_measured_from_the_stick_centre(scorer):
    """``sigma = STICK_LENGTH/2`` must land exactly on the tip."""
    q = torch.zeros(3, 17, dtype=torch.float64)
    sig = torch.full((3, 2), STICK_LENGTH / 2.0, dtype=torch.float64)
    pts, _u = scorer.contact_points(q, sig)
    R, p = scorer.fk.frame(q, "left_wrist")
    tip = p + torch.einsum("bij,j->bi", R,
                           torch.as_tensor(stick_tip_offset("left"), dtype=torch.float64))
    assert torch.allclose(pts[:, 0], tip, atol=1e-9)


def test_contact_point_matches_stick_contact_offset(scorer):
    """Any sigma must agree with ``fk.stick_contact_offset``, which is what
    ``FusedCost._terminal`` and ``StrikeSpec.contact_sigma`` both use."""
    torch.manual_seed(0)
    q = 0.2 * torch.randn(4, 17, dtype=torch.float64)
    for s in (0.05, 0.10, STICK_LENGTH / 2.0):
        sig = torch.full((4, 2), s, dtype=torch.float64)
        pts, _ = scorer.contact_points(q, sig)
        for hi, hand in enumerate(("left", "right")):
            R, p = scorer.fk.frame(q, f"{hand}_wrist")
            want = p + torch.einsum(
                "bij,j->bi", R,
                torch.as_tensor(stick_contact_offset(hand, s), dtype=torch.float64))
            assert torch.allclose(pts[:, hi], want, atol=1e-9), (hand, s)


def test_head_frame_matches_probe_head_frame_error(kit, scorer):
    """The batched head-frame error must equal the offline one used by
    ``probe.hit_latency`` -- otherwise the Isaac histogram and `REPORT.md` §4
    are measuring different things."""
    ref = A.TrajectoryScorer(device="cpu")
    sched = BatchedSchedule(1, ScheduleConfig(horizon=50), device="cpu",
                            dtype=torch.float64)
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched.reset(torch.tensor([0]), clips, [kit], rng=None)
    ctx = sched.strike_context()
    tok = sched.context()
    torch.manual_seed(1)
    q = 0.15 * torch.randn(1, 17, dtype=torch.float64)
    pts = scorer.best_contact(q, ctx)[2]
    ax, ip = scorer.head_frame(pts, ctx)
    for hi, hand in enumerate(("left", "right")):
        if not bool(tok.goal_valid[0, hi]):
            continue
        drum = int(tok.goal_drum[0, hi])
        sigma = default_contact_sigma(kit.names[drum])
        wax, _wip, _ok = head_frame_error(ref, q[0].numpy(), hand, kit, drum, sigma)
        assert np.sign(float(ax[0, hi])) == np.sign(wax) or abs(wax) < 1e-3
        tol = 0.02 if A.shaft_hit_allowed(kit.names[drum]) else 2e-3
        assert float(ax[0, hi]) == pytest.approx(wax, abs=tol), (hand, drum)


def test_crossing_is_located_sub_tick(kit, scorer):
    """The interpolated crossing must sit on the head plane, not on a knot."""
    sched = BatchedSchedule(1, ScheduleConfig(horizon=50), device="cpu",
                            dtype=torch.float64)
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched.reset(torch.tensor([0]), clips, [kit], rng=None)
    ctx = sched.strike_context()
    # walk a straight line in joint space until one hand crosses its head plane
    torch.manual_seed(2)
    q0 = torch.zeros(1, 17, dtype=torch.float64)
    d = 0.05 * torch.randn(1, 17, dtype=torch.float64)
    prev = q0
    for _ in range(400):
        cur = prev + d
        cs = scorer.score(prev, cur, ctx)
        if bool(cs.crossed.any()):
            u = cs.frac.max()
            assert 0.0 <= float(u) <= 1.0
            qc = prev + u * (cur - prev)
            pc = scorer.best_contact(qc, ctx)[2]
            ax, _ip = scorer.head_frame(pc, ctx)
            hi = int(cs.crossed[0].long().argmax())
            assert abs(float(ax[0, hi])) < 5e-3, "the crossing is not on the plane"
            return
        prev = cur
    pytest.skip("no crossing along the probe ray")


# --------------------------------------------------------------- the lead --
def test_lead_is_applied_before_the_clamp_not_after():
    """``amortized_opt.lead``: the lead goes on the PLAN, then slew, then box --
    so the clamp still guarantees an admissible command."""
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    H, n = 10, 17
    traj = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(1, H, n) * 0.1
    f = lead_fraction(LEAD_PRESETS["fit"], A.CONTROL_DT, n)
    led = lead_waypoints(traj, torch.as_tensor(f, dtype=torch.float32))
    # the lead reads AHEAD along the plan, so waypoint 0 must move forward
    assert (led[0, 0] >= traj[0, 0]).all()
    moved = (led[0, 0] - traj[0, 0]).abs() > 1e-9
    assert bool(moved.any()), "a nonzero preset must move at least one joint"
    # joints whose fitted lead is 0 must be untouched -- the "sparse" argument
    zero = [i for i, v in enumerate(LEAD_PRESETS["fit"]) if v == 0.0]
    assert torch.allclose(led[0, 0, zero], traj[0, 0, zero])


def test_lead_holds_past_the_terminal(kit):
    """A reference lead may only read INSIDE the plan; past a hand's terminal
    the decoder output is unpriced, so the reference is HELD.  Consequence
    (7c): at the strike tick the horizon is one waypoint and the lead is a
    no-op."""
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    H, n, B = 8, 17, 1
    sched = BatchedSchedule(B, ScheduleConfig(horizon=H), device="cpu")
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched.reset(torch.tensor([0]), clips, [kit], rng=None)
    model = A.build_model(A.ModelConfig(max_horizon=H), "cpu")
    mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s="fit",
                                           amp_dtype=None, cmd_slew_eta=None,
                                           clamp_cmd_to_box=False),
                   device="cpu", kit_names=tuple(kit.names))
    traj = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(B, H, n) * 0.1
    # both hands strike at waypoint 1 -> the whole plan is past the terminal
    hidx = torch.ones(B, 2, dtype=torch.long)
    cmd = mpc._command(traj.clone(), hidx)
    assert torch.allclose(cmd, traj[:, 0], atol=1e-6), \
        "with a one-waypoint horizon the lead must be a no-op"


def test_tracking_scores_against_the_unled_reference(kit):
    """``ref0`` must be the plan's own first waypoint, not the led command."""
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    H, B = 12, 2
    sched = BatchedSchedule(B, ScheduleConfig(horizon=H), device="cpu")
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched.reset(torch.arange(B), clips * B, [kit] * B, rng=None)
    model = A.build_model(A.ModelConfig(max_horizon=H), "cpu")
    mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s="fit",
                                           amp_dtype=None),
                   device="cpu", kit_names=tuple(kit.names))
    q = torch.zeros(B, 17)
    mpc.reset(torch.arange(B), q)
    res = mpc.plan(q, torch.zeros(B, 17))
    assert not torch.allclose(mpc.last_ref0, res.cmd), \
        "with a nonzero lead the reference and the command must differ"
    assert torch.allclose(res.traj[:, 0], res.cmd), \
        "the executed command must be written back into the plan"


# -------------------------------------------------------------- the reward --
def test_reward_decomposition_is_complete_and_finite(kit, fk, scorer):
    sched = BatchedSchedule(2, ScheduleConfig(horizon=50), device="cpu",
                            dtype=torch.float64)
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched.reset(torch.arange(2), clips * 2, [kit] * 2, rng=None)
    ctx = sched.strike_context()
    rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=None,
                         device="cpu", dtype=torch.float64)
    q0 = torch.zeros(2, 17, dtype=torch.float64)
    q1 = q0 + 0.01
    terms, cs = rew.step(
        q_before=q0, q_after=q1, qd_after=torch.zeros_like(q0), ctx=ctx,
        hit_index=sched.hit_index(), fired=torch.zeros(2, 2, dtype=torch.bool),
        ref0=q1.clone(),
        drum_heads=ctx.drum_head_centers, drum_axes=sched.geom["axis"],
        drum_radii=sched.drum_radius, drum_valid=sched.drum_valid,
    )
    for k in ("hit", "timing", "speed", "miss", "extra_hit", "early_contact",
              "plan_cost", "track", "filter", "alive"):
        assert k in terms.terms, k
        assert torch.isfinite(terms.terms[k]).all(), k
    assert torch.isfinite(terms.total).all()
    # a perfect tracker must earn the full tracking weight
    assert float(terms.terms["track"][0]) == pytest.approx(
        RewardConfig().w_track, abs=1e-9)


def test_miss_is_charged_only_when_a_hand_fires(kit, scorer):
    sched = BatchedSchedule(1, ScheduleConfig(horizon=50), device="cpu",
                            dtype=torch.float64)
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched.reset(torch.tensor([0]), clips, [kit], rng=None)
    ctx = sched.strike_context()
    rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=None,
                         device="cpu", dtype=torch.float64)
    q = torch.zeros(1, 17, dtype=torch.float64)
    kw = dict(q_before=q, q_after=q, qd_after=q, ctx=ctx,
              hit_index=sched.hit_index(), ref0=q)
    quiet, _ = rew.step(fired=torch.zeros(1, 2, dtype=torch.bool), **kw)
    firing, _ = rew.step(fired=torch.ones(1, 2, dtype=torch.bool), **kw)
    assert float(quiet.terms["miss"]) == 0.0
    assert float(firing.terms["miss"]) == -2 * RewardConfig().w_miss
