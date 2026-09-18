# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VENDORED from ``source/g1_drummer/g1_drummer/control/waist_aim.py`` (2026-08-10).

Two deltas from the original, both mechanical: the lazy dual-path
``trajopt.spline`` import becomes a plain ``from .spline import ...``, and
:func:`aim_targets_from_layout` is added so a caller can get per-hand aim
targets out of the vendored ``assets/drum_kit_layout.yaml`` without the repo's
``DrumLayout``. The law, the gate and every constant are unchanged --
``tests/test_waist_aim_ik.py`` asserts agreement with the repo original to
1e-8.

Waist-yaw aiming: turn the torso so the arms face the drums they must hit.

Scope is deliberately narrow -- **yaw only**. Pitch and roll stay on their
setpoint, so the torso turns but never leans; leaning changes the shoulder
height and the whole reachability picture the strike IK was tuned against,
while a pure yaw only re-parameterizes azimuth.

Two pieces, both pure numpy (no Pinocchio / torch), so the RL env and the
offline teacher can share them:

1. :func:`aim_yaw` -- where the torso *should* point given the upcoming strike
   target(s). Handles one hand or both; with both it is a weighted average of
   the per-hand demands (see the law below).
2. :func:`yaw_reference` -- the time trajectory that gets there: a quintic ramp
   from the current yaw, landing ``settle_wp`` waypoints *before* the strike and
   holding. A still torso at contact means the arm owns all the tip speed and
   the stick's FK offset is not sliding underneath the hit.

The law
-------
For hand ``h`` with shoulder ``s_h`` (in the pelvis frame, rotating with yaw)
and target ``p_h``, the shoulder->target direction has azimuth ``az_h``. A
drummer wants that direction to sit a little outboard of straight ahead in the
*torso* frame -- ``az_pref`` (+ for the left hand, - for the right). The
residual ``e_h = az_h - psi - az_pref_h`` is what a yaw of ``psi`` should
cancel, so minimizing

    sum_h w_h * e_h(psi)^2  +  lambda * (sum_h w_h) * psi^2

gives, to first order in ``psi``,

    psi* = mean_w(az_h(0) - az_pref_h) / (1 + lambda)

i.e. exactly the **weighted average of the per-hand azimuth demands**, damped by
``lambda``. The damping is what makes the torso face the drums *somewhat*
rather than squaring up to them: ``lambda = 0`` commits fully, ``lambda = 1``
turns half way. It is normalized by ``sum_h w_h`` so the commitment does not
change when only one hand is striking.

The shoulders themselves rotate with the torso, so ``az_h`` depends on ``psi``;
:func:`aim_yaw` runs a few fixed-point sweeps to account for that (the
correction is small -- the shoulder offset is ~0.1 m against a ~0.5 m reach --
but it costs nothing).

The gate: an asymmetric reach cone per hand
-------------------------------------------
The law above turns *a little bit* for *every* drum, which makes the torso a
partner in every strike -- the arm leans on it for azimuth it could have found
by itself, and each of those turns is base motion the whole stack has to be
pinned against for no benefit. So the law is gated rather than rewritten: the
gate decides *whether* the torso moves at all, and only once it opens does the
law above decide how far.

An arm's reach cone is not symmetric about its own preferred direction. Swinging
**outward** -- away from the body, the direction the shoulder opens -- is cheap
and stays cheap a long way round. Swinging **inward**, across the chest, runs out
much sooner: the far shoulder, the other arm and the torso itself are all in the
way. So one threshold cannot describe it, and the gate uses two:

    inward_h  =  +(az_h - psi_now - az_pref_h)   RIGHT hand   (inward = CCW)
                 -(az_h - psi_now - az_pref_h)   LEFT hand    (inward = CW)

    turn the torso  <=>  inward_h > engage_in    (default 40 deg)
                    or   inward_h < -engage_out  (default 40 deg)

``inward_h`` is this hand's target azimuth **in the torso frame the robot is
actually in right now** (``psi_now`` = the currently commanded yaw), measured
from that hand's preferred outboard direction ``az_pref_h`` and signed so that
positive is always across the body regardless of which hand it is.

**The test is RELATIVE, and that is the point.** What a gate has to answer is
"must I turn to play this", which depends on where the torso is pointing, not on
where centre happens to be. A torso already turned toward the snare does not
have to unwind to play the toms beside it -- under an absolute (centred) test it
would, on every beat, because the gate would keep re-asking "is this drum 40 deg
off CENTRE" and keep commanding 0 in between. So when the gate is shut the torso
**holds where it is**; it does not spring back to centre.

Ratcheting is not a risk despite the relative test, because the commanded yaw is
an ABSOLUTE damped target recomputed from the beat (the law above), never an
increment on the current one. The torso can only ever sit where some beat put
it, and every engaged beat re-derives that from scratch.

The gate is **per hand, ORed**: any hand whose drum falls outside its own cone
turns the torso, because that hand is the one that cannot reach. This is a
deliberate change from gating on the two-hand *average*, which let a hand at 70
deg inward be cancelled by a comfortable partner and leave the torso still for a
reach that genuinely needed it.

The two bounds are separate knobs because the asymmetry is real, even though
both currently sit at 40 deg. At that setting the shipped kit turns the torso for
six of its twelve hand-drums: the three cross-body reaches (``left/cymbal_right``
66.5 deg inward, ``right/snare`` 66.1, ``right/cymbal_left`` 61.9), two more that
clear the inward bound only just (``right/tom2`` 43.8, ``left/tom1`` 41.5), and
``right/floor_tom``, the one drum past the outward bound (43.9 deg out). It holds
still for the other six, including ``left/hihat`` (34.8 deg OUT -- the far-left
drum, but the left arm reaches it outward, which is cheap).

Note how close three of those six sit to the threshold: at 40 deg the bounds cut
through a cluster (41.5, 43.8, 43.9), so a small kit-layout edit will flip them.
Widening the inward bound past ~44 deg returns to just the cross-body three.

Set ``engage_in``/``engage_out`` to None to disable the gate and fire the law on
every beat, as it did originally.

Aiming at the FUTURE, not just at this beat (2026-08-14)
--------------------------------------------------------
Everything above computes the aim from the hands' *currently active* goals. That
made the commanded yaw a function of a set that churns at beat rate, and two
failure modes were measured on ``gmd_eval_000006`` (drums at +37..+60 deg) with
``runs/phase2_19m_ctxv2/ckpt_10k.pt``:

* **square-waving.** The right hand's snare goal dropped in and out of the
  horizon cap at ~1 Hz, so the target set flipped between {left, right} ->
  +37 deg and {left} -> "no turn needed, hold". A heavy torso chasing that
  square wave oscillated 131 deg peak-to-peak. The set never actually changed --
  only which *label* (goal vs future[0]) the snare hit carried.
* **hold ratification.** A per-tick trace caught the hold branch returning the
  MEASURED yaw verbatim after an overshoot (aim = +63.2 = yaw_meas while the
  drums sat at +37). Every hold phase legitimizes wherever the torso ran off to,
  so an overshoot is never commanded back.

So the aim is now taken over each hand's whole upcoming schedule -- its active
goal AND its next ``future_max`` hits -- with a weight per target that depends
only on TIME TO HIT, never on the active/future label. Both failures follow from
that one change: relabelling moves no weight (so the cap crossing is continuous,
`tests/test_future_aim.py`), and a hand whose goal is capped still votes (so the
"no targets -> hold" branch now fires only at a genuine rest, where holding is
the right answer and there is nothing to ratify against).

The weights are an APPROACH WINDOW, not a point discount: hit ``i`` owns the
interval ``[t_{i-1}, t_i]`` of the future, i.e. the stretch of time during which
the torso is on its way to it, and the weight is the exponential kernel
``exp(-s / future_tau)`` integrated over that interval (normalized per hand, so
a hand's total vote is its ``hand_weights`` entry no matter how many hits it
carries).  Three properties, all wanted:

* the intervals TILE the future, so a hit that is consumed (its ``t`` reaches 0)
  leaves with the weight it has already decayed to -- zero -- and its successor
  inherits the window continuously. A point discount ``exp(-t_i / tau)`` does
  the opposite: it gives the imminent hit the LARGEST weight and drops it at
  the moment it fires, which is the same square wave one beat further on;
* relative weights are all that matter, so a hand with a single pending hit
  aims fully at it however close it is (the normalization gives it 1.0);
* ``future_tau`` is the knob between the two regimes: small -> the imminent hit
  owns nearly the whole window and the law reduces to the current-goal one;
  large -> the next few hits share the vote and the torso parks between them.

``future_tau = 0`` (or None) restores the current-goals-only law BIT-EXACTLY --
it is the A/B switch, and ``tests/test_future_aim.py`` pins the equality on a
grid of scenarios.

The per-hand cap ``future_max`` counts the WHOLE merged list, goal included, and
that is deliberate: ``goal_snapshot`` hands out ``goal + num_future`` targets
when the goal is active but ``num_future`` when the cap demoted it, so counting
the merged list at ``num_future`` is what makes the two spellings of the same
instant the same target set. Counting only the futures would leave a 4-vs-3
mismatch, and the demotion would jump the aim again.

The gate and the deadband are applied ONCE, to the aggregate: the per-hand
inward angle is the same weighted mean of that hand's targets, so the gate still
asks "does THIS hand need the torso" (per hand, ORed) rather than being opened
by any single far-off hit in the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Shoulder-pitch joint origins in the pelvis frame, XY only (Z is irrelevant to
# an azimuth). From assets/robots/_generated/g1_29dof_rev_1_0.urdf:
# {left,right}_shoulder_pitch_joint sit at (0.0039563, +/-0.10022) in the torso
# frame, and the waist chain adds no XY offset above the yaw axis.
_SHOULDER_XY: dict[str, tuple[float, float]] = {
    "left": (0.0039563, 0.10022),
    "right": (0.0039563, -0.10022),
}

# waist_yaw_joint URDF range is +/-2.618 rad (+/-150 deg); +/-100 deg (200 deg
# total swing) leans on more of that range to help reach hard-to-hit drums
# (e.g. the hi-hat, see scripts/gen_drum_goal_poses.py's wrist-headroom
# diagnostics) without spending the arm's own axial joints as hard, while
# still leaving 50 deg of margin to the hard stop on each side.
_DEFAULT_YAW_LIMIT = 1.74533  # rad (100 deg, ~200 deg total)

# How far ACROSS THE BODY a hand reaches before it needs the torso. Inward is
# the expensive direction -- the far shoulder, the other arm and the chest are
# all in the way.
_DEFAULT_ENGAGE_IN = np.deg2rad(40.0)

# ...and how far OUTBOARD, away from the body, where the shoulder opens freely.
# Kept a separate knob from the inward bound because the asymmetry is real, even
# while both sit at 40 deg. At 40 the only drum past it is right/floor_tom
# (43.9 deg out).
_DEFAULT_ENGAGE_OUT = np.deg2rad(40.0)


@dataclass
class WaistAimConfig:
    """Tunables for :func:`aim_yaw`."""

    # --- gate: the per-hand reach cone ----------------------------------------
    # How far a hand may reach ACROSS THE BODY (rad) before the torso helps.
    # Measured per hand from that hand's own ``az_pref`` direction and signed so
    # positive is always inward -- see the module docstring. The gate is ORed
    # over the striking hands: whichever one falls outside its cone is the one
    # that cannot reach, and it turns the torso regardless of how comfortable
    # its partner is. ``None`` disables the inward half of the gate.
    engage_in: float | None = _DEFAULT_ENGAGE_IN
    # ...and how far OUTBOARD, the cheap direction, hence the looser bound.
    # ``None`` disables the outward half.
    engage_out: float | None = _DEFAULT_ENGAGE_OUT
    # --- the aim law (runs when the gate fires, or always if it is off) ------
    # Preferred shoulder->target azimuth in the torso frame, per hand (rad).
    # Positive is outboard-left. The left hand wants its target a bit to the
    # left of straight ahead and the right hand a bit to the right, which is
    # where the elbow has room.
    az_pref: float = np.deg2rad(20.0)
    # Damping lambda in the law above. The torso is NOT meant to square up to
    # the drum -- it gives the arm a better starting azimuth and the arm covers
    # the rest, which is both what a drummer does and what keeps the strike IK
    # inside the posture the rest of the stack was tuned for. lambda = 0 would
    # face the drum outright; 0.3 turns ~77% of the demand, 0.7 turns ~59%,
    # 1.0 turns half. Lowered 0.7 -> 0.3 (2026-08-04) so the torso carries more
    # of the azimuth demand on the hard reaches (hihat/floor_tom) -- the yaw
    # clamp was never the binding constraint, this fraction was.
    damping: float = 0.3
    # Hard clamp on the commanded yaw (rad).
    yaw_limit: float = _DEFAULT_YAW_LIMIT
    # A CHANGE threshold, not a demand threshold: ``_apply_deadband`` tests
    # |psi - yaw_current| -- how far the commanded yaw is from where the torso
    # already sits -- and holds ``yaw_current`` while that stays below this
    # (rad).  Without it the yaw twitches by a degree or two every beat, which
    # reads as noise in the dataset and buys no reach.
    deadband: float = 0.09  # ~5 deg
    # Per-hand weights when both hands strike. Equal by default: the commanded
    # yaw is the plain average of the two demands.  With futures in play this is
    # each HAND's total vote, split across its own upcoming hits -- so a hand
    # playing a roll does not outvote a hand playing whole notes.
    hand_weights: dict[str, float] = field(
        default_factory=lambda: {"left": 1.0, "right": 1.0}
    )
    # --- future awareness (see the module docstring) --------------------------
    # Time constant (s) of the exponential kernel integrated over each hit's
    # APPROACH WINDOW.  1.0 s is ~2-4 beats at drumming tempi: long enough that
    # a goal crossing the horizon cap changes no weight (the square wave) and
    # short enough that the imminent hit still owns most of the vote.
    # ``0`` or ``None`` = the pre-2026-08-14 law, current goals only, bit for
    # bit -- the A/B escape hatch, not a tuning value.
    future_tau: float | None = 1.0
    # How many targets per hand may vote, COUNTING THE CURRENT GOAL.  3 matches
    # the ``num_future`` every snapshot/schedule carries, which is what makes
    # the goal-active and goal-demoted spellings of one instant identical sets;
    # see the module docstring.
    future_max: int = 3
    # Shoulder XY offsets in the pelvis frame; override to match a re-rigged robot.
    shoulder_xy: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_SHOULDER_XY)
    )
    # Fixed-point sweeps accounting for the shoulders rotating with the torso.
    refine_iters: int = 3


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def engage_from_deg(deg: float) -> float | None:
    """One cone half-angle in radians, or ``None`` when that half is disabled.

    The knob is exposed in degrees because that is how the gate is reasoned
    about ("don't turn for anything inside 40 degrees across the body").
    ``>= 180`` means no angle can ever exceed it, which is the same thing as
    switching that half off, so spell it that way rather than leaving a
    threshold nothing can cross.
    """
    d = float(deg)
    return None if not np.isfinite(d) or d >= 180.0 else float(np.deg2rad(d))


#: Inward is across the body: counter-clockwise (toward +y, the robot's left)
#: for the right hand, clockwise for the left. This flips the per-hand residual
#: so ``inward > 0`` means the same thing on both sides.
_INWARD_SIGN: dict[str, float] = {"left": -1.0, "right": +1.0}


# --------------------------------------------------------------------------- #
# The reach gate: measured arm-only reachability, per (hand, drum, torso yaw)
# --------------------------------------------------------------------------- #


def _shoulder_azimuths(
    entries: list[tuple[str, np.ndarray, float]], cfg: WaistAimConfig, psi: float
) -> list[tuple[str, float, float]]:
    """``(hand, weight, world azimuth)`` per target with the torso yawed ``psi``.

    The torso-frame azimuth is this less ``psi``; keeping the world value is
    what lets the gate write its constraint as an interval in ``psi``.
    """
    c, s = np.cos(psi), np.sin(psi)
    out: list[tuple[str, float, float]] = []
    for hand, p, w in entries:
        sx, sy = cfg.shoulder_xy[hand]
        shoulder = np.array([c * sx - s * sy, s * sx + c * sy])
        d = p[:2] - shoulder
        if float(np.hypot(*d)) < 1e-6:
            continue
        out.append((hand, w, float(np.arctan2(d[1], d[0]))))
    return out


def _active_targets(
    targets: dict[str, np.ndarray], cfg: WaistAimConfig
) -> dict[str, np.ndarray]:
    return {
        h: np.asarray(p, float).reshape(3)
        for h, p in targets.items()
        if p is not None and h in cfg.shoulder_xy
    }


def _window_weights(times: np.ndarray, tau: float) -> np.ndarray:
    """Approach-window weights for ONE hand's time-ordered hits, summing to 1.

    Hit ``i`` owns ``[t_{i-1}, t_i]`` (the first owns ``[0, t_0]``) and its
    weight is ``exp(-s / tau)`` integrated over that interval; the ``tau``
    factor of the integral cancels in the normalization and is dropped. The
    intervals tile ``[0, t_last]``, which is what makes a hit's weight fall
    continuously to zero as it is consumed -- see the module docstring.
    """
    t = np.maximum(np.asarray(times, dtype=float).reshape(-1), 0.0)
    if t.size == 0:
        return t
    a = np.empty_like(t)
    a[0] = 0.0
    a[1:] = t[:-1]
    a = np.minimum(a, t)            # a clip that is out of order votes 0, not <0
    w = np.exp(-a / float(tau)) - np.exp(-t / float(tau))
    s = float(w.sum())
    if not np.isfinite(s) or s <= 1e-15:
        # Every hit is at (or before) now: there is no window left to integrate
        # over.  The earliest is the one being played -- give it the hand's
        # whole vote rather than dividing by zero.
        w = np.zeros_like(t)
        w[0] = 1.0
        s = 1.0
    return w / s


def _aim_entries(
    targets: dict[str, np.ndarray],
    cfg: WaistAimConfig,
    *,
    goal_tth: dict[str, float] | None = None,
    futures: dict[str, object] | None = None,
) -> tuple[list[tuple[str, np.ndarray, float]], float]:
    """``[(hand, target, weight)]`` and the total weight the demand divides by.

    ``futures[hand]`` is that hand's upcoming hits beyond its current goal as
    ``(position, time_to_hit)`` pairs, both from NOW, time-ordered.
    ``goal_tth[hand]`` is the same clock for the entry in ``targets``. With no
    futures (or ``cfg.future_tau`` off) this returns exactly one entry per
    active hand at its ``hand_weights`` value -- the pre-2026-08-14 law, whose
    arithmetic downstream is then unchanged bit for bit.
    """
    goal = _active_targets(targets, cfg)
    tau = float(cfg.future_tau or 0.0)
    if tau <= 0.0 or not futures:
        entries = [
            (h, p, float(cfg.hand_weights.get(h, 1.0))) for h, p in goal.items()
        ]
        return entries, sum(w for _h, _p, w in entries)

    if goal and goal_tth is None:
        raise ValueError(
            "aim_yaw: futures were given without goal_tth -- the current goal's "
            "own time to hit is what places it in the weighting window"
        )
    order = list(goal) + [
        h for h in futures if h not in goal and h in cfg.shoulder_xy
    ]
    entries: list[tuple[str, np.ndarray, float]] = []
    for hand in order:
        items: list[tuple[np.ndarray, float]] = []
        if hand in goal:
            items.append((goal[hand], float((goal_tth or {}).get(hand, 0.0))))
        for p, t in futures.get(hand) or ():
            if p is None:
                continue
            items.append((np.asarray(p, dtype=float).reshape(3), float(t)))
        if not items:
            continue
        # Stable sort, so a future sharing the goal's instant still ranks after
        # it; the merged list is capped as a WHOLE (module docstring).
        items.sort(key=lambda pt: max(pt[1], 0.0))
        items = items[: max(int(cfg.future_max), 1)]
        hw = float(cfg.hand_weights.get(hand, 1.0))
        w = _window_weights(np.array([t for _p, t in items]), tau)
        entries.extend(
            (hand, p, hw * float(wi)) for (p, _t), wi in zip(items, w)
        )
    return entries, sum(w for _h, _p, w in entries)


def _demand(
    entries: list[tuple[str, np.ndarray, float]],
    w_sum: float,
    cfg: WaistAimConfig,
    psi: float,
) -> float:
    """Weighted-average azimuth demand (rad) at torso yaw ``psi``.

    This is ``demand`` in the module docstring's law -- what a yaw of ``psi``
    is trying to cancel -- evaluated with the shoulders rotated by ``psi``.
    """
    if w_sum <= 0.0:
        return 0.0
    total = 0.0
    for hand, w, az in _shoulder_azimuths(entries, cfg, psi):
        az_pref = cfg.az_pref if hand == "left" else -cfg.az_pref
        total += w * _wrap_pi(az - az_pref)
    return total / w_sum


def aim_demand(
    targets: dict[str, np.ndarray],
    *,
    cfg: WaistAimConfig | None = None,
    goal_tth: dict[str, float] | None = None,
    futures: dict[str, object] | None = None,
) -> float:
    """The net weighted azimuth demand (rad) the aim LAW damps.

    Evaluated at a **centred** torso (``psi = 0``). This is what decides how far
    the torso turns once the gate opens; :func:`aim_inward` is what decides
    whether it opens. Exposed so a caller can log both.
    """
    cfg = cfg or WaistAimConfig()
    entries, w_sum = _aim_entries(targets, cfg, goal_tth=goal_tth, futures=futures)
    return _demand(entries, w_sum, cfg, psi=0.0)


def aim_inward(
    targets: dict[str, np.ndarray],
    *,
    cfg: WaistAimConfig | None = None,
    yaw_current: float = 0.0,
    goal_tth: dict[str, float] | None = None,
    futures: dict[str, object] | None = None,
) -> dict[str, float]:
    """Per-hand inward angle (rad) **as seen from the torso's current yaw**.

    The gate's variable, and it is RELATIVE: the bearing of the target in the
    torso frame the robot is actually in right now
    (``az_h(yaw_current) - yaw_current``), measured from that hand's preferred
    outboard direction. ``inward > 0`` means that far ACROSS THE BODY,
    ``inward < 0`` that far outboard; same sense for both hands (see
    :data:`_INWARD_SIGN`), so the two cone bounds read alike on either side.

    **Pass ``yaw_current``.** Left at 0 this asks "how far off CENTRE is the
    drum", which is a different and, for a gate, wrong question -- it makes the
    torso unwind and re-turn between beats instead of staying where the last
    reach left it.

    Returned per hand rather than averaged because the gate is per hand: one
    arm's inability to reach is not cancelled by the other arm being comfortable.
    With futures in play it is that hand's OWN weighted mean over its upcoming
    targets -- the aggregate the gate is applied to once (module docstring), so
    a single far-off hit in the window cannot open the gate by itself.
    """
    cfg = cfg or WaistAimConfig()
    entries, _w_sum = _aim_entries(targets, cfg, goal_tth=goal_tth, futures=futures)
    return _inward_of(entries, cfg, yaw_current)


def _inward_of(
    entries: list[tuple[str, np.ndarray, float]],
    cfg: WaistAimConfig,
    yaw_current: float,
) -> dict[str, float]:
    """:func:`aim_inward` on already-built entries (see it for the semantics)."""
    psi = float(yaw_current)
    per_hand: dict[str, list[tuple[float, float]]] = {}
    for hand, w, az in _shoulder_azimuths(entries, cfg, psi):
        az_pref = cfg.az_pref if hand == "left" else -cfg.az_pref
        per_hand.setdefault(hand, []).append((w, _wrap_pi(az - psi - az_pref)))
    out: dict[str, float] = {}
    for hand, items in per_hand.items():
        if len(items) == 1:
            # A single-target hand takes its residual UNDIVIDED: ``(w*x)/w`` is
            # not the identity in floating point, and the futures-off path has
            # to reproduce the old gate bit for bit.
            resid = items[0][1]
        else:
            den = sum(w for w, _x in items)
            resid = (
                sum(w * x for w, x in items) / den if den > 0.0 else 0.0
            )
        out[hand] = _INWARD_SIGN[hand] * resid
    return out


def gate_open(
    targets: dict[str, np.ndarray],
    *,
    cfg: WaistAimConfig | None = None,
    yaw_current: float = 0.0,
    goal_tth: dict[str, float] | None = None,
    futures: dict[str, object] | None = None,
) -> bool:
    """Does any striking hand need the torso to move FROM WHERE IT IS?

    True as soon as ONE hand's aggregate target leaves its cone as seen from
    ``yaw_current`` -- ``inward > engage_in`` (across the body) or
    ``inward < -engage_out`` (outboard). With both bounds disabled this is
    always True and the law fires on every beat.
    """
    cfg = cfg or WaistAimConfig()
    entries, _w_sum = _aim_entries(targets, cfg, goal_tth=goal_tth, futures=futures)
    return _gate_open_of(entries, cfg, yaw_current)


def _gate_open_of(
    entries: list[tuple[str, np.ndarray, float]],
    cfg: WaistAimConfig,
    yaw_current: float,
) -> bool:
    """:func:`gate_open` on already-built entries, so the caller that has them
    (``aim_yaw``) does not assemble the weighted schedule twice per solve."""
    lo = cfg.engage_out
    hi = cfg.engage_in
    if hi is None and lo is None:
        return True
    for inward in _inward_of(entries, cfg, yaw_current).values():
        if hi is not None and inward > float(hi) + 1e-12:
            return True
        if lo is not None and inward < -float(lo) - 1e-12:
            return True
    return False


def _apply_deadband(psi: float, yaw_current: float, cfg: WaistAimConfig) -> float:
    """Clamp ``psi`` to the yaw limit and hold ``yaw_current`` if it is close."""
    psi = float(np.clip(psi, -cfg.yaw_limit, cfg.yaw_limit))
    if abs(psi - float(yaw_current)) < float(cfg.deadband):
        return float(np.clip(yaw_current, -cfg.yaw_limit, cfg.yaw_limit))
    return psi


def aim_yaw(
    targets: dict[str, np.ndarray],
    *,
    cfg: WaistAimConfig | None = None,
    yaw_current: float = 0.0,
    goal_tth: dict[str, float] | None = None,
    futures: dict[str, object] | None = None,
) -> float:
    """Commanded waist yaw (rad) for the upcoming strike target(s).

    ``targets`` maps ``"left"`` / ``"right"`` to the (3,) world/base-frame tip
    target for that hand. Pass one entry for a single-handed beat and both for a
    two-handed one -- with both, the result is the weighted average of the two
    hands' demands (see the module docstring). Entries whose value is None are
    treated as idle and skipped.

    The torso holds still unless some striking hand's target leaves its reach
    cone (:func:`gate_open`); when one does, the full damped law runs and the
    weighted average serves both hands.

    ``yaw_current`` enters in three places, not just the deadband.  It is the
    GATE's reference frame (:func:`gate_open` asks "must I turn from where I
    am", so which side of the engage bounds a target falls on depends on it);
    it is the value RETURNED (clamped to ``cfg.yaw_limit``) whenever the law
    does not run -- no active hand, zero weights, or a shut gate all hold the
    torso where it is; and it is the deadband's hold value: a commanded yaw
    within ``cfg.deadband`` of it returns ``yaw_current`` unchanged, so a
    rhythm that keeps asking for roughly the same aim does not produce a
    jittering torso.

    ``goal_tth`` (per hand, seconds from now) and ``futures`` (per hand, an
    ordered sequence of ``(position, time_to_hit)`` for the hits BEYOND the
    current goal) turn on the future-aware aggregate -- see the module
    docstring.  Every producer in the stack passes them; ``cfg.future_tau = 0``
    is the switch that ignores them.  Note the semantic change they carry: the
    "no targets -> hold ``yaw_current``" branch below now fires only when the
    hands have no upcoming hits AT ALL (a rest), because a hand whose goal the
    horizon cap demoted still votes through its futures.  That is the point --
    the hold branch was ratifying overshoots, returning the measured yaw as the
    command every time the target set momentarily emptied.
    """
    cfg = cfg or WaistAimConfig()
    entries, w_sum = _aim_entries(targets, cfg, goal_tth=goal_tth, futures=futures)
    if not entries:
        return float(np.clip(yaw_current, -cfg.yaw_limit, cfg.yaw_limit))

    if w_sum <= 0.0:
        return float(np.clip(yaw_current, -cfg.yaw_limit, cfg.yaw_limit))

    # Gate from WHERE THE TORSO IS, and hold it there when the gate is shut.
    # The question is "must I turn to play this", which depends on the
    # orientation the robot is actually in: a torso already turned toward the
    # snare does not have to come back to centre to play the toms beside it.
    # Ratcheting is not a risk despite the relative test, because the commanded
    # yaw below is an ABSOLUTE damped target recomputed from the beat, never an
    # increment on the current one -- the torso can only sit where some beat put
    # it, and every engaged beat re-derives that from scratch.
    if not _gate_open_of(entries, cfg, yaw_current):
        return float(np.clip(yaw_current, -cfg.yaw_limit, cfg.yaw_limit))

    # Fixed point of psi = demand(psi) / (1 + lambda), NOT psi += residual(psi):
    # the latter converges to residual == 0, i.e. the torso squared up to the
    # drum, with lambda changing only how fast it gets there. Anchoring the
    # damping against psi = 0 (home) is what makes the turn partial and leaves
    # the rest of the azimuth for the arm.
    psi = 0.0
    for _ in range(max(int(cfg.refine_iters), 1)):
        demand = _demand(entries, w_sum, cfg, psi)
        psi = float(
            np.clip(demand / (1.0 + float(cfg.damping)), -cfg.yaw_limit, cfg.yaw_limit)
        )

    return _apply_deadband(psi, yaw_current, cfg)


def yaw_reference(
    yaw_start: float,
    yaw_target: float,
    n_waypoints: int,
    *,
    settle_wp: int = 3,
    yawd_start: float = 0.0,
    max_move_wp: int | None = None,
) -> np.ndarray:
    """Quintic yaw ramp over ``n_waypoints`` (= ``N + 1``) waypoints.

    **Leads, not trails.** The ramp starts at index 0, with rate ``yawd_start``,
    and reaches ``yaw_target`` with zero rate and acceleration as fast as
    ``max_move_wp`` allows -- then HOLDS for the rest of the horizon, well
    before ``n_waypoints - 1 - settle_wp``, so "settled before contact" is
    satisfied by finishing early rather than by arriving late. The torso
    commits to its aim immediately and is already stationary for nearly the
    whole plan, so the arm's own trajectory is solved against a settled frame
    for almost its entire horizon instead of a still-rotating one near contact.

    ``max_move_wp`` bounds how long the ramp itself takes (waypoints), which is
    what makes it fast: a small value forces a steep quintic that reaches the
    target in a handful of waypoints rather than spending the whole pre-settle
    span on it. It is still clamped so the ramp never runs past the settle
    point even on the shortest horizons. ``None`` spends the entire pre-settle
    span on the ramp -- slower, and the old (trailing) default.

    ``yawd_start`` is in rad per *waypoint index* (i.e. already multiplied by
    ``dt``); see :func:`waist_reference_block`, which converts from rad/s.
    Always honoured now, since the ramp always starts at index 0 -- there is no
    held-still lead-in during which an incoming rate would need discarding.
    """
    n = int(n_waypoints)
    if n < 2:
        raise ValueError(f"need at least 2 waypoints, got {n}")
    k_end = max(n - 1 - max(int(settle_wp), 0), 1)
    k_move_end = k_end if max_move_wp is None else min(max(int(max_move_wp), 1), k_end)
    span = max(k_move_end, 1)
    # Vendored spline (no Pinocchio at import time), so no lazy/dual-path
    # dance is needed here -- unlike the repo original.
    from .spline import quintic_time_scaling

    s = quintic_time_scaling(
        float(span),
        float(yaw_target) - float(yaw_start),
        float(yawd_start),
        0.0,
        0.0,
        0.0,
    )
    k = np.arange(n, dtype=float)
    # Clip into [0, span] so waypoints past the move window evaluate to the
    # held target exactly rather than to a quintic extrapolation (which is not
    # monotone past its interval).
    out = float(yaw_start) + s(np.clip(k, 0.0, float(span)))
    out[k_move_end:] = float(yaw_target)
    return np.asarray(out, float)


def waist_reference_block(
    joint_names: tuple[str, ...] | list[str],
    yaw_start: float,
    yaw_target: float,
    n_waypoints: int,
    *,
    setpoint: np.ndarray | None = None,
    dt: float = 0.02,
    settle_wp: int = 3,
    yaw_rate_start: float = 0.0,
    max_move_s: float | None = None,
) -> np.ndarray:
    """Full ``(n_waypoints, n_waist)`` waist reference in optimizer joint order.

    Yaw follows :func:`yaw_reference`; pitch and roll hold ``setpoint`` (zeros
    by default) because this aimer only rotates -- it never leans. The column
    order follows the ``waist_*`` entries of ``joint_names``, so it drops
    straight into ``TrajOptConfig.waist_reference``.

    ``max_move_s`` caps how long the ramp itself takes, in seconds (converted
    to waypoints here); see :func:`yaw_reference`, which leads with it rather
    than trailing -- the torso commits fast, at the start of the horizon, then
    holds.
    """
    waist = [n for n in joint_names if str(n).startswith("waist_")]
    if not waist:
        raise ValueError("joint_names contains no waist joints")
    sp = np.zeros(len(waist)) if setpoint is None else np.asarray(setpoint, float).reshape(len(waist))
    yaw = yaw_reference(
        yaw_start,
        yaw_target,
        n_waypoints,
        settle_wp=settle_wp,
        yawd_start=float(yaw_rate_start) * float(dt),
        max_move_wp=(
            None
            if max_move_s is None
            else max(int(round(float(max_move_s) / float(dt))), 1)
        ),
    )
    ref = np.tile(sp, (int(n_waypoints), 1))
    for j, name in enumerate(waist):
        if name == "waist_yaw_joint":
            ref[:, j] = yaw
    return ref


def waist_start_velocity(reference: np.ndarray, dt: float) -> np.ndarray:
    """Waist rows of ``v_start`` implied by a reference block.

    Block 1 of the QP pins ``(q_1 - q_0) / dt == v_start`` as a hard equality
    while the reference pins ``q_0`` and ``q_1`` outright, so a ``v_start`` whose
    waist rows disagree with the reference makes the QP primal-infeasible on the
    first iteration. Always source those rows from here.
    """
    ref = np.asarray(reference, float)
    if ref.ndim != 2 or ref.shape[0] < 2:
        raise ValueError(f"reference must be (n_waypoints>=2, n_waist), got {ref.shape}")
    return (ref[1] - ref[0]) / float(dt)


# Fixed [yaw, pitch, roll] column order the controller operates on internally,
# independent of any caller's full 17-DoF joint-name list.
_WAIST_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)


class WaistYawController:
    """Standalone runtime version of the retarget-and-ramp law that
    ``play_control_theory.py --waist_aim`` runs coupled to full-body trajopt
    (``TrajOptAgent._update_waist_aim`` / ``_waist_cmd_at`` / ``_command_waist``).

    Pure numpy, no trajopt / Pinocchio / QP dependency: it only ever calls
    :func:`aim_yaw` (a closed-form azimuth law) and :func:`waist_reference_block`
    (a quintic spline). That is what makes it usable somewhere trajopt is not --
    e.g. driving a waist-only dataset (``scripts/gen_waist_dataset.py``) without
    paying for a single arm QP solve, since the arms never need to be planned at
    all for the torso schedule to be correct.

    Call pattern, once per control step ``step_now``:

    1. :meth:`update` -- decide whether the upcoming beat(s) now want a
       different aim than the one currently commanded and, if so, rebuild the
       ramp schedule anchored at the CURRENTLY COMMANDED yaw (not the measured
       one) -- matching ``_update_waist_aim``'s reasoning that the schedule
       must stay a single continuous function of step.
    2. :meth:`command_at` -- read this step's ``[yaw, pitch, roll]`` command.

    Splitting the two (rather than one ``step()``) matches the trajopt agent's
    own shape, where other work (there: forcing an arm replan) happens between
    deciding a new aim and reading this tick's command.
    """

    def __init__(
        self,
        cfg: WaistAimConfig | None = None,
        *,
        dt: float,
        aim_window_s: float = 1.0,
        aim_move_s: float = 0.2,
        settle_wp: int = 3,
        ready_waist: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg or WaistAimConfig()
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.aim_window_s = float(aim_window_s)
        self.aim_move_s = float(aim_move_s)
        self.settle_wp = max(int(settle_wp), 0)
        self.ready_waist = (
            np.zeros(3, dtype=float)
            if ready_waist is None
            else np.asarray(ready_waist, float).reshape(3).copy()
        )
        self._sched: np.ndarray | None = None
        self._sched_start = 0
        self._target = 0.0

    def reset(self) -> None:
        """Drop the schedule; the next :meth:`update` starts from ``ready_waist``."""
        self._sched = None
        self._sched_start = 0
        self._target = 0.0

    def command_at(self, step: int) -> np.ndarray:
        """Commanded ``[yaw, pitch, roll]`` (3,) at absolute step ``step``.

        Holds the last commanded aim past the end of the schedule -- the torso
        does not spring back to centre between beats (see :func:`aim_yaw`'s own
        module doc on why the target is absolute, not incremental).
        """
        if self._sched is None:
            return self.ready_waist.copy()
        k = int(np.clip(int(step) - self._sched_start, 0, self._sched.shape[0] - 1))
        return np.asarray(self._sched[k], dtype=float)

    def update(
        self,
        step_now: int,
        targets: dict[str, np.ndarray],
        t_to_next_beat: float | None = None,
        *,
        goal_tth: dict[str, float] | None = None,
        futures: dict[str, object] | None = None,
    ) -> bool:
        """Retarget if the pending beat(s) now want a different aim.

        ``targets`` is the same per-hand strike-target mapping :func:`aim_yaw`
        takes (pelvis-frame tip target, one entry per active hand).
        ``t_to_next_beat`` is seconds from now to the earliest pending strike
        across hands (``None`` if nothing is pending); it gates *when* a decision
        is made (``aim_window_s``), exactly as
        ``TrajOptAgent._update_waist_aim`` does -- independent of ``aim_move_s``,
        which only bounds how fast the ramp itself runs once one is built.

        ``goal_tth`` / ``futures`` are :func:`aim_yaw`'s future-aware inputs,
        passed straight through; omit them and this controller aims at the
        pending beat alone, whatever ``cfg.future_tau`` says.

        Returns True when the schedule was rebuilt.
        """
        yaw_now = float(self.command_at(step_now)[0])
        yaw_new = float(aim_yaw(targets, cfg=self.cfg, yaw_current=yaw_now,
                                goal_tth=goal_tth, futures=futures))
        # aim_yaw's own deadband returns yaw_current unchanged on a small
        # demand, so this also absorbs per-beat jitter that would otherwise
        # rebuild the schedule every tick.
        if self._sched is not None and abs(yaw_new - self._target) < 1e-6:
            return False
        if t_to_next_beat is not None and t_to_next_beat > self.aim_window_s:
            return False
        n_wp = max(
            int(round((t_to_next_beat if t_to_next_beat else 0.4) / self.dt)),
            self.settle_wp + 2,
        )
        self._sched = waist_reference_block(
            _WAIST_JOINT_NAMES,
            yaw_start=yaw_now,
            yaw_target=yaw_new,
            n_waypoints=n_wp + 1,
            setpoint=self.ready_waist,
            dt=self.dt,
            settle_wp=self.settle_wp,
            max_move_s=self.aim_move_s,
        )
        self._sched_start = int(step_now)
        self._target = yaw_new
        return True


# --------------------------------------------------------------------------- #
# Vendored addition: aim targets straight out of the drum-kit layout.
# --------------------------------------------------------------------------- #
def aim_targets_from_layout(
    hand_drums: dict[str, str],
    *,
    layout_path=None,
    frame: str = "pelvis",
) -> dict[str, np.ndarray]:
    """Per-hand (3,) aim targets for :func:`aim_yaw` from the layout YAML.

    ``hand_drums`` maps ``"left"``/``"right"`` to a drum name (``None`` for an
    idle hand). ``frame`` is ``"pelvis"`` (layout minus ``PELVIS_IN_ENV``, the
    frame the shoulder offsets live in) or ``"env"``. Only the XY components
    matter to the aim law, and those are identical in both frames -- the
    argument exists so a caller mixing this with FK output cannot get the Z
    convention wrong by accident.
    """
    import yaml

    from .constants import PELVIS_IN_ENV, default_layout_path

    path = layout_path if layout_path is not None else default_layout_path()
    with open(path) as f:
        layout = yaml.safe_load(f)
    drums = layout["drums"]
    off = PELVIS_IN_ENV if frame == "pelvis" else np.zeros(3)
    if frame not in ("pelvis", "env"):
        raise ValueError(f"frame must be 'pelvis' or 'env', got {frame!r}")
    out: dict[str, np.ndarray] = {}
    for hand, name in hand_drums.items():
        if name is None:
            continue
        if name not in drums:
            raise KeyError(f"drum {name!r} not in {path} ({sorted(drums)})")
        out[hand] = np.asarray(drums[name]["position"], dtype=float) - off
    return out
