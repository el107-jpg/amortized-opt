"""The waist aim law is FUTURE-aware: one law, every producer, continuous.

``waist_aim.aim_yaw`` used to read a hand's currently active goal and nothing
else, which made the commanded yaw a function of a set that churns at beat rate.
Two failure modes were measured on ``gmd_eval_000006`` with
``runs/phase2_19m_ctxv2/ckpt_10k.pt`` (drums at +37..+60 deg):

* the right hand's snare goal crossed the horizon cap at ~1 Hz, so the aim
  flipped between "both hands -> +37 deg" and "left only -> hold" -- a square
  wave, 131 deg peak-to-peak on a momentum-rich torso -- even though the SET of
  upcoming hits never changed, only which label (goal vs future[0]) one of them
  carried;
* the hold branch returned the MEASURED yaw verbatim after an overshoot (aim
  +63.2 = yaw_meas with the drums at +37), so every hold ratified wherever the
  torso had run off to.

The fix weights each of a hand's upcoming hits by TIME TO HIT alone (see the
module docstring of :mod:`amortized_opt.waist_aim`), so relabelling moves no
weight.  This file pins the properties that makes true, and the numbers below
are what it measured when written (2026-08-14, default ``WaistAimConfig``, the
``_spread_clip`` groove):

    horizon-cap crossing   aim jump 1.2e-06 rad   was 0.787 rad
    hit fires              aim jump 0.0525 rad    was 1.337 rad

1.2e-06 rad is the floor of the measurement, not a residual: the two probes sit
2 us apart in time, and the same probe at an instant where NOTHING is relabelled
moves the aim by 3.3e-06.  The crossing is exact.

The 0.05 rad left at a hit firing is real but bounded -- it is the THIRD future
hit becoming visible for the first time and entering the window at the far end.
It is smaller than the law's own 0.09 rad deadband, so it cannot move the
commanded yaw of a torso that is tracking its command at all.
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
from amortized_opt.rhythm import (  # noqa: E402
    Hit,
    RhythmClip,
    aim_inputs_from_snapshot,
    build_context_inputs,
    goal_snapshot,
)
from amortized_opt.waist_aim import (  # noqa: E402
    WaistAimConfig,
    aim_demand,
    aim_inward,
    aim_yaw,
    gate_open,
)

DT = A.CONTROL_DT
#: Horizon cap for the continuity sweeps.  Deliberately NOT a multiple of the
#: clip's 0.5 s spacing, so a cap crossing never lands on the same instant as
#: some hand's strike and the two events can be measured apart.
CAP_S = 0.35
EPS_T = 1e-6                     # how far either side of an event we look


@pytest.fixture(scope="module")
def kit():
    return A.DrumKit.from_layout()


def _clip(kit, spec: dict[str, list[tuple[float, str]]], duration: float = 12.0):
    """A synthetic clip from ``{hand: [(time, drum_name), ...]}``."""
    hits = [
        Hit(time=float(t), raw_time=float(t), drum=kit.index(name), drum_name=name,
            hand=hand, velocity=1.0)
        for hand, times in spec.items()
        for t, name in times
    ]
    hits.sort(key=lambda h: h.time)
    return RhythmClip(name="synthetic", hits=hits, duration=duration)


def _spread_clip(kit):
    """Left on the far-left hihat, right alternating snare / floor_tom.

    Deliberately the hardest layout in the kit for the torso: the right hand's
    two drums sit on opposite sides of its reach cone (snare 66 deg inward,
    floor_tom 44 deg outboard), so any weighting error shows up as a large yaw
    swing rather than as rounding.
    """
    return _clip(kit, {
        "left": [(t, "hihat") for t in np.arange(0.4, 8.0, 0.5)],
        "right": [
            (t, "snare" if i % 2 == 0 else "floor_tom")
            for i, t in enumerate(np.arange(0.5, 8.0, 0.5))
        ],
    })


def _snap(kit, clip, t_now, *, cap=CAP_S, num_future=3):
    return goal_snapshot(clip, kit, t_now, num_future=num_future, rng=None,
                         hit_radius_frac=0.0, horizon_cap=cap)


def _aim_at(kit, clip, t_now, *, yaw, tau, cap=CAP_S, cfg=None):
    """The law's answer at one instant, through the shared snapshot helper."""
    snap = _snap(kit, clip, t_now, cap=cap)
    targets, tth, futures = aim_inputs_from_snapshot(snap)
    cfg = cfg or WaistAimConfig(future_tau=tau)
    return float(aim_yaw(targets, cfg=cfg, yaw_current=yaw,
                         goal_tth=tth, futures=futures))


# ------------------------------------------------- 1. the A/B switch is exact --
def _grid_cases(kit):
    """(targets, goal_tth, futures) rows spanning the shapes the law sees."""
    p = {name: np.asarray(kit.positions_in("pelvis")[kit.index(name)], float)
         for name in ("snare", "hihat", "tom1", "floor_tom", "cymbal_right")}
    rows = [
        ({}, {}, {}),
        ({"left": p["hihat"]}, {"left": 0.3}, {}),
        ({"right": p["snare"]}, {"right": 0.05}, {}),
        ({"left": p["hihat"], "right": p["snare"]}, {"left": 0.2, "right": 0.4}, {}),
    ]
    futs = {
        "left": [(p["tom1"], 0.6), (p["cymbal_right"], 1.1), (p["hihat"], 1.6)],
        "right": [(p["floor_tom"], 0.9), (p["snare"], 1.4)],
    }
    for targets, tth, _ in list(rows):
        rows.append((targets, tth, futs))
    # ...and a hand that has ONLY futures (its goal was demoted by the cap)
    rows.append(({"left": p["hihat"]}, {"left": 0.25},
                 {"left": [], "right": futs["right"]}))
    return rows


def test_future_tau_zero_is_bit_exact_with_the_old_law(kit):
    """``future_tau = 0`` ignores the futures BIT for bit -- the A/B switch.

    The comparison is against the same call with no future arguments at all,
    which is the untouched pre-2026-08-14 arithmetic: same iteration order, same
    weights, same divisions.  Equality is exact, not ``allclose`` -- a law that
    is only *nearly* the old one cannot be used to attribute an A/B result.
    """
    off = WaistAimConfig(future_tau=0.0)
    worst = 0.0
    for targets, tth, futures in _grid_cases(kit):
        for yaw in (-1.2, -0.4, -0.09, 0.0, 0.09, 0.4, 1.2):
            want = aim_yaw(targets, cfg=off, yaw_current=yaw)
            got = aim_yaw(targets, cfg=off, yaw_current=yaw,
                          goal_tth=tth, futures=futures)
            assert got == want, (targets.keys(), yaw, got, want)
            assert gate_open(targets, cfg=off, yaw_current=yaw,
                             goal_tth=tth, futures=futures) == \
                gate_open(targets, cfg=off, yaw_current=yaw)
            iw = aim_inward(targets, cfg=off, yaw_current=yaw,
                            goal_tth=tth, futures=futures)
            assert iw == aim_inward(targets, cfg=off, yaw_current=yaw)
            worst = max(worst, abs(got - want))
        assert aim_demand(targets, cfg=off, goal_tth=tth, futures=futures) == \
            aim_demand(targets, cfg=off)
    assert worst == 0.0


def test_a_caller_that_passes_no_futures_is_unaffected(kit):
    """The default (future_tau = 1) changes nothing for a futures-free caller.

    The eval harness plans single-hit cases and has no schedule to hand; it must
    keep the exact yaw it planned before this change.
    """
    on, off = WaistAimConfig(), WaistAimConfig(future_tau=0.0)
    for targets, tth, _f in _grid_cases(kit):
        for yaw in (-0.4, 0.0, 0.7):
            assert aim_yaw(targets, cfg=on, yaw_current=yaw) == \
                aim_yaw(targets, cfg=off, yaw_current=yaw)


def test_context_token_matches_the_law_on_the_same_snapshot(kit):
    """``build_context_inputs`` is the law, not a second copy of it."""
    clip = _spread_clip(kit)
    for tau in (0.0, 1.0):
        cfg = WaistAimConfig(future_tau=tau)
        for step in range(0, 120, 7):
            t_now = step * DT + 1e-6
            snap = _snap(kit, clip, t_now)
            ctx = build_context_inputs(kit, snap, num_future=3, yaw_now=0.35,
                                       waist_cfg=cfg, dtype=torch.float64)
            targets, tth, futures = aim_inputs_from_snapshot(snap)
            want = aim_yaw(targets, cfg=cfg, yaw_current=0.35,
                           goal_tth=tth, futures=futures)
            assert float(ctx.aim_yaw[0]) == want


# --------------------------------------------- 2. continuity at a relabelling --
def _event_jumps(kit, clip, event: str, *, tau, yaws=(-0.4, 0.0, 0.6)):
    """Max |aim(t-) - aim(t+)| over every ``event`` instant in the clip.

    ``"cap"`` is a hit crossing the horizon cap -- goal <-> future[0], the
    relabelling that produced the measured square wave.  ``"fire"`` is a hit
    being struck, which promotes future[0] for real.

    Both sweeps only look at events with the window POPULATED on either side:
    a cap crossing that lands on some hand's strike is that strike's event and
    is measured there, and the last hits of a clip are skipped because a hand
    running out of music has genuinely lost its whole target set -- no
    weighting can make "this hand stops playing" continuous, and the old law
    jumps there too.
    """
    times = sorted(float(h.time) for h in clip.hits)
    per_hand = {
        hand: sorted(float(h.time) for h in clip.hits if h.hand == hand)
        for hand in ("left", "right")
    }
    worst = 0.0
    for h in clip.hits:
        t_star = float(h.time) - (CAP_S if event == "cap" else 0.0)
        if t_star <= 0.2:
            continue
        if event == "cap" and any(abs(t_star - t) < 1e-9 for t in times):
            continue
        if event == "fire":
            later = [t for t in per_hand[h.hand] if t > float(h.time) + 1e-9]
            if len(later) < 3:
                continue
        for yaw in yaws:
            a = _aim_at(kit, clip, t_star - EPS_T, yaw=yaw, tau=tau)
            b = _aim_at(kit, clip, t_star + EPS_T, yaw=yaw, tau=tau)
            worst = max(worst, abs(a - b))
    return worst


def _quiet_drift(kit, clip, *, tau, yaws=(-0.4, 0.0, 0.6)):
    """The same +/-EPS_T probe at instants where no label moves.

    The two evaluations of an event are 2 * ``EPS_T`` apart in time, so their
    weights differ by that much whatever happens; this is how big that is.
    """
    times = sorted(float(h.time) for h in clip.hits)
    worst = 0.0
    for t0, t1 in zip(times, times[1:]):
        mid = 0.5 * (t0 + t1)
        if mid <= 0.2 or any(abs(mid - CAP_S - t) < 1e-3 for t in times):
            continue
        for yaw in yaws:
            a = _aim_at(kit, clip, mid - EPS_T, yaw=yaw, tau=tau)
            b = _aim_at(kit, clip, mid + EPS_T, yaw=yaw, tau=tau)
            worst = max(worst, abs(a - b))
    return worst


def test_horizon_cap_crossing_does_not_move_the_aim(kit):
    """THE HEADLINE: a goal crossing the cap moves the aim by exactly nothing.

    The cap only relabels -- ``goal_snapshot`` demotes the hit to ``future[0]``
    and shifts the rest -- and the merged list is capped as a whole at
    ``future_max``, so the two spellings are the SAME weighted target set.  The
    old law rebuilt its aim from a set that had just lost its heaviest member,
    which is the 131 deg square wave.
    """
    clip = _spread_clip(kit)
    now = _event_jumps(kit, clip, "cap", tau=1.0)
    then = _event_jumps(kit, clip, "cap", tau=0.0)
    # The control: the same 2 us of clock at instants where NOTHING is
    # relabelled.  The crossing has to be indistinguishable from it -- what is
    # left in both is the weights ageing by 2 us, not the label moving.
    drift = _quiet_drift(kit, clip, tau=1.0)
    print(f"\ncap crossing: future-aware {now:.3e} rad (quiet-clock control "
          f"{drift:.3e}), current-goals-only {then:.3f} rad")
    assert now <= max(drift, 1e-12) * 1.5, (now, drift)
    assert now < 1e-5, now
    assert then > 0.3, then          # the failure this change exists to remove


def test_a_hit_firing_moves_the_aim_less_than_the_deadband(kit):
    """A fire is a real set change; it must still stay inside the deadband.

    The consumed hit leaves with the weight its approach window has already
    decayed to (zero), and its successors keep theirs, so all that changes is
    the far end: the third future hit becomes visible for the first time.  What
    is left is below ``cfg.deadband``, i.e. too small to move the commanded yaw
    of a torso that is tracking its command.
    """
    clip = _spread_clip(kit)
    cfg = WaistAimConfig()
    now = _event_jumps(kit, clip, "fire", tau=1.0)
    then = _event_jumps(kit, clip, "fire", tau=0.0)
    print(f"\nhit fires: future-aware {now:.4f} rad, current-goals-only "
          f"{then:.3f} rad (deadband {cfg.deadband})")
    assert now < cfg.deadband, now
    assert then > 0.4, then


# ------------------------------------------- 3. a demoted hand still gets a vote --
def test_a_capped_goal_still_pulls_the_aim(kit):
    """{left hihat active, right snare capped} aims at BOTH, not just the left.

    This is the square wave seen from the other side: under the old law the
    right hand vanished from the aim the moment its goal left the horizon cap,
    and the torso unwound to the left-hand-only answer until the hit ticked back
    into range.
    """
    clip = _clip(kit, {"left": [(0.30, "hihat"), (1.30, "hihat")],
                       "right": [(0.90, "snare"), (1.90, "snare")]})
    t_now = 0.20                     # left tth 0.10 (active), right 0.70 > cap
    snap = _snap(kit, clip, t_now)
    assert snap.goal["left"].active and not snap.goal["right"].active
    targets, tth, futures = aim_inputs_from_snapshot(snap)
    assert [f[1] for f in futures["right"]][:1] == [pytest.approx(0.70)]

    yaw = 0.0
    got = float(aim_yaw(targets, cfg=WaistAimConfig(), yaw_current=yaw,
                        goal_tth=tth, futures=futures))
    left_only = float(aim_yaw(targets, cfg=WaistAimConfig(future_tau=0.0),
                              yaw_current=yaw))
    both = float(aim_yaw(
        {**targets, "right": snap.future["right"][0].position - A.PELVIS_IN_ENV},
        cfg=WaistAimConfig(future_tau=0.0), yaw_current=yaw,
    ))
    print(f"\ncapped-goal aim: future-aware {got:+.3f}, left-only {left_only:+.3f}, "
          f"both-as-goals {both:+.3f}")
    assert abs(got - left_only) > 0.05, (got, left_only)
    assert abs(got - both) < abs(got - left_only)


# --------------------------------------------------------------- 4. the rest --
def test_a_real_rest_still_holds_the_current_yaw(kit):
    """No goal AND no future anywhere = hold; that is the only hold left.

    The hold branch is where the ratification came from, so it has to keep
    existing (a torso with nothing to play should stay where it is) and it has
    to stop firing anywhere else.
    """
    clip = _clip(kit, {"left": [(0.30, "hihat")], "right": [(0.50, "snare")]},
                 duration=6.0)
    for tau in (0.0, 1.0):
        cfg = WaistAimConfig(future_tau=tau)
        snap = _snap(kit, clip, 2.0)          # past every hit in the clip
        targets, tth, futures = aim_inputs_from_snapshot(snap)
        assert not targets and not any(futures.values())
        for yaw in (-1.0, 0.0, 0.55):
            assert aim_yaw(targets, cfg=cfg, yaw_current=yaw,
                           goal_tth=tth, futures=futures) == pytest.approx(yaw)
    # ...and a hand whose goal is capped is NOT a rest under the new law: it
    # still votes, so the aim is not the single-hand answer the old law gave.
    snap = _snap(kit, clip, 0.05)
    targets, tth, futures = aim_inputs_from_snapshot(snap)
    assert futures["right"], "the capped hit must survive as a future"
    with_future = aim_yaw(targets, cfg=WaistAimConfig(), yaw_current=0.0,
                          goal_tth=tth, futures=futures)
    without = aim_yaw(targets, cfg=WaistAimConfig(future_tau=0.0), yaw_current=0.0)
    assert with_future != pytest.approx(without), (with_future, without)


# ------------------------------------------------------ 5. deployment parity --
@pytest.mark.parametrize("tree", ["isaac_rl", "isaac_rl_ppo"])
def test_deployment_schedule_aims_like_the_training_producer(kit, tree):
    """``BatchedSchedule.context`` and ``build_context_inputs`` agree on the aim.

    The two trees re-derive the tokens on the GPU instead of going through
    ``goal_snapshot``, so each carries its own copy of the anchor rule -- and
    the aim law needs FROM-NOW times while the tokens are hit-relative.  A
    rewrite that is nearly right is the worst outcome here (same argument as
    ``tests/test_isaac_rl_schedule.py``); an aim that disagrees with the pool's
    is a checkpoint conditioned on a token it never saw in training.
    """
    mod = pytest.importorskip(f"{tree}.schedule")
    clip = _spread_clip(kit)
    cap_wp = int(round(CAP_S / DT))
    cfg = mod.ScheduleConfig(horizon=50, num_drums=8, num_future=3,
                             horizon_cap_wp=cap_wp, hit_radius_frac=0.0)
    s = mod.BatchedSchedule(1, cfg, device="cpu", dtype=torch.float64)
    s.reset(torch.arange(1), [clip], [kit], rng=None)
    wcfg = WaistAimConfig()
    yaw = 0.35
    worst = 0.0
    for step in range(80):
        snap = _snap(kit, clip, step * DT + 1e-6)
        want = build_context_inputs(kit, snap, num_future=3, yaw_now=yaw,
                                    waist_cfg=wcfg, dtype=torch.float64)
        got = s.context(yaw_now=torch.tensor([yaw], dtype=torch.float64),
                        waist_cfg=wcfg)
        d = abs(float(got.aim_yaw[0]) - float(want.aim_yaw[0]))
        worst = max(worst, d)
        assert d < 1e-9, (step, float(got.aim_yaw[0]), float(want.aim_yaw[0]))
        s.advance()
    print(f"\n{tree} aim parity: worst {worst:.2e} rad")


# ---------------------------------------------------------- 6. the knob works --
def test_tau_moves_the_weight_between_now_and_later(kit):
    """Small tau -> the imminent hit owns the aim; large tau -> the future does.

    The two limits are the ones that make ``future_tau`` a usable knob: at
    ``tau -> 0`` the law must degrade to the current-goal answer (so an A/B is
    a smooth family, not a cliff), and at large tau the upcoming hits must
    actually move the aim.
    """
    # left holds the hihat; the right hand's goal is on the snare but its next
    # three hits are all the floor_tom, on the far side of its reach cone
    clip = _clip(kit, {
        "left": [(0.30, "hihat"), (1.50, "hihat")],
        "right": [(0.40, "snare"), (0.80, "floor_tom"), (1.20, "floor_tom"),
                  (1.60, "floor_tom")],
    })
    snap = _snap(kit, clip, 0.10, cap=1.0)
    targets, tth, futures = aim_inputs_from_snapshot(snap)
    assert snap.goal["right"].drum_name == "snare"
    yaw = 0.0
    raw = dict(engage_in=None, engage_out=None, deadband=0.0)  # the LAW alone
    goal_only = float(aim_yaw(targets, cfg=WaistAimConfig(future_tau=0.0, **raw),
                              yaw_current=yaw))
    seq = [
        float(aim_yaw(targets, cfg=WaistAimConfig(future_tau=tau, **raw),
                      yaw_current=yaw, goal_tth=tth, futures=futures))
        for tau in (0.02, 0.2, 1.0, 5.0)
    ]
    print(f"\ngoal-only {goal_only:+.4f} -> tau 0.02/0.2/1/5: "
          + " ".join(f"{v:+.4f}" for v in seq))
    assert abs(seq[0] - goal_only) < 1e-3, seq[0]      # tau -> 0 is the old aim
    assert abs(seq[-1] - goal_only) > 0.05, seq[-1]    # the futures do vote
    # monotone in tau: every step moves further from the goal-only answer
    dist = [abs(v - goal_only) for v in seq]
    assert dist == sorted(dist), dist
