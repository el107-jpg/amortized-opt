"""Drum-strike geometry: where on a head to aim, and what counts as a hit.

Vendored from ``source/g1_drummer/g1_drummer/control/trajopt/problem.py``
(2026-08-10).  Nothing here imports the repo; the only dependency is numpy plus
``amortized_opt.constants``.

The rules encoded here are *tuned*, not derived -- they came out of the repo's
RL hit detector and months of trajopt tuning -- so this is a port, not a
reimplementation.  ``tests/test_strike_conformance.py`` asserts every function
matches the repo original to 1e-12 on randomized inputs (and skips when the repo
is absent). The hi-hat aim has a deliberate 20 mm inset; its legal head region
still matches the original.

Two consumers:

* the section-2 evaluation harness (:mod:`amortized_opt.eval_harness`), which
  needs to know where the contact point is and whether it landed on the head;
* the section-3 differentiable terminal cost, which needs the same aim point,
  the same half-plane cut and the same impact-velocity weighting.

Semantic summary (see the individual docstrings for the reasoning):

* **Cymbals and the hi-hat** may be sounded by the stick SHAFT, so hit
  membership is the playing-shaft segment
  (:func:`shaft_hit_sigma_range`, bead back to the 5 cm fallback) against the
  legal head region -- not a single shoulder point.  ``contact_sigma`` is still
  the diagnostic default (the shoulder).  They are aimed off-centre toward the
  near edge, keep the *full* head radius (no rim buffer), and only the near
  semicircle is legal.
* **Toms** are tip-only pads that borrow the cymbals' off-centre *geometry*:
  aimed 45 % of the rim-buffered radius toward the drummer, near semicircle only.
* **Everything else** (the snare) is a tip-only pad aimed dead centre, hit disk
  = head radius minus the rim buffer, whole disk legal.
* Pad strikes require the terminal stick elevation to sit in a two-sided band;
  cymbals and the hi-hat do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    SHAFT_HIT_SAMPLES,
    SHAFT_HIT_SPAN,
    STICK_TIP_SIGMA,
)

__all__ = [
    "CYMBAL_CONTACT_SIGMA_FALLBACK",
    "CYMBAL_CONTACT_SIGMA_MIN",
    "CYMBAL_EDGE_FRACTION",
    "HIHAT_RIM_FRACTION",
    "HIHAT_AIM_INSET_M",
    "HIHAT_SHOULDER_SIGMA",
    "STICK_ELEV_MAX_DEG",
    "STICK_ELEV_MIN_DEG",
    "STICK_TIP_SIGMA",
    "STRIKE_AXIAL_HEIGHT",
    "STRIKE_SURFACE_OFFSET",
    "StrikeSpec",
    "TOM_NEAR_FRACTION",
    "contact_sigma_candidates",
    "default_contact_sigma",
    "shaft_hit_sigma_range",
    "is_cymbal_name",
    "is_hihat_name",
    "is_tom_name",
    "shaft_hit_allowed",
    "stick_elevation_from_surface",
    "stick_elevation_ok",
    "strike_aim_on_head",
    "strike_axial_limits",
    "strike_disk_center_offset",
    "strike_elevation_required",
    "strike_half_plane_dir",
    "strike_outward_normal",
    "strike_velocity_weight",
]


# --------------------------------------------------------------- constants --

#: Where on the stick a CYMBAL (hi-hat included) is struck by default: the
#: shoulder, 10 cm back from the bead.  A crash sounds from the shoulder, not
#: the bead.
HIHAT_SHOULDER_SIGMA = 0.10

#: Furthest back along the stick a cymbal strike may normally contact.
CYMBAL_CONTACT_SIGMA_MIN = HIHAT_SHOULDER_SIGMA

#: One probe behind the shoulder, offered only after the whole band has failed.
CYMBAL_CONTACT_SIGMA_FALLBACK = 0.05

#: Aim point on a cymbal, as a fraction of the head radius out from the centre
#: along the facing direction (0 = bell, 1 = rim).
CYMBAL_EDGE_FRACTION = 0.75

#: Near-edge aim with room for the stick capsule to enter through the top.
HIHAT_RIM_FRACTION = 1.0
HIHAT_AIM_INSET_M = 0.020

#: Same idea for the TOMS, as a fraction of the *usable* (rim-buffered) radius
#: out from the centre toward the drummer.
TOM_NEAR_FRACTION = 0.45

#: Aim inset along ``-axis``, past the geometric head plane.  10 mm (was 5 mm)
#: so the aim sits on the INNER face of :data:`STRIKE_AXIAL_HEIGHT`.
STRIKE_SURFACE_OFFSET = 0.01

#: Height of the legal hit cylinder along the head normal.  The cylinder occupies
#: the interior of the drum: outer face on the geometric head surface, inner
#: face ``STRIKE_AXIAL_HEIGHT`` into the drum (the aim plane, when this equals
#: :data:`STRIKE_SURFACE_OFFSET`).
STRIKE_AXIAL_HEIGHT = 0.01

#: Two-sided stick-elevation band at the strike, degrees.  Mirrors
#: ``TrajOptConfig.stick_elev_{min,max}_deg``.  Sign-blind: the requirement is
#: ``|elev| in [min, max]``, which excludes both the in-plane graze (0) and
#: driving straight down the head normal (90).
STICK_ELEV_MIN_DEG = 5.0
STICK_ELEV_MAX_DEG = 45.0


def _unit3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------ drum classes --


def is_cymbal_name(name: str | None) -> bool:
    """True for the cymbals: ``cymbal_*`` plus the HI-HAT.

    The hi-hat is a pair of cymbals and is played the same way -- on the bow and
    the edge, with the stick free to arrive at whatever angle the approach wants
    -- so it shares every cymbal rule here.
    """
    if name is None:
        return False
    n = str(name).lower()
    return "cymbal" in n or "hihat" in n or "hi_hat" in n or "hi-hat" in n


def is_hihat_name(name: str | None) -> bool:
    """True for the HI-HAT only, as distinct from the crash cymbals.

    It sits low and nearly flat with a second cymbal directly beneath it, so the
    face is a cramped target while the EDGE is wide open -- hence its own aim
    fraction (:data:`HIHAT_RIM_FRACTION`).
    """
    if name is None:
        return False
    n = str(name).lower()
    return "hihat" in n or "hi_hat" in n or "hi-hat" in n


def is_tom_name(name: str | None) -> bool:
    """True for the toms (``tom1`` / ``tom2`` / ``floor_tom`` / any alias).

    A tom is still a pad: it keeps the rim buffer, the elevation band and the
    tip-only hit rule.  The only thing it borrows from the cymbals is the
    off-centre AIM, toward the drummer.  It no longer borrows the semicircle
    cut -- see :func:`strike_half_plane_dir` for why that was wrong.
    """
    if name is None:
        return False
    return "tom" in str(name).lower()


def shaft_hit_allowed(name: str | None) -> bool:
    """Whether this drum may be sounded by the stick SHAFT, not just the tip.

    True for the cymbals and hi-hat, mirroring ``shaft_hit: true`` in the layout
    YAML -- the flag the RL hit detector reads.  A membrane drum is tip-only: a
    shaft laid across a snare head is not a hit.
    """
    return is_cymbal_name(name)


def strike_elevation_required(name: str | None) -> bool:
    """Whether a strike on this drum must land in the stick-elevation band.

    PADS ONLY -- the snare and the toms.  A pad is a membrane struck from above
    with the bead, so "how steeply is the stick coming in" is a real
    requirement: the band rejects both the in-plane graze and the near-normal
    tunnelling.  A cymbal is not played that way.  It is struck on the bow or
    the edge with the SHAFT (:func:`shaft_hit_allowed` is True for exactly the
    cymbals, and :func:`default_contact_sigma` gives them the shoulder rather
    than the bead), so the approach angle is whatever the swing wants and a
    band on it is a constraint with no physical referent.

    Until 2026-08-14 this returned True unconditionally, which mattered less
    than it looks: ``elev_ok`` was computed and reported but was NOT part of
    ``CaseScore.accepted``, so nothing was ever gated on it.  Now that it gates,
    the predicate has to be right -- otherwise turning the gate on would fail
    every cymbal case for violating a band that does not apply to cymbals.
    """
    return not is_cymbal_name(name)


# ------------------------------------------------------------ contact point --


def default_contact_sigma(name: str | None) -> float:
    """Contact point a strike on ``name`` uses when it does not name one.

    The bead everywhere except the cymbals (hi-hat included), which default to
    the shoulder because a crash/edge articulation is a shaft hit.  Hit
    *membership* on those drums is the playing-shaft capsule
    (:func:`shaft_hit_sigma_range`); this scalar is the diagnostic / aim
    default.  A caller that specifically wants a bead strike on a cymbal must
    pass ``contact_sigma=STICK_TIP_SIGMA`` explicitly.
    """
    if is_cymbal_name(name):
        return HIHAT_SHOULDER_SIGMA
    return STICK_TIP_SIGMA


def shaft_hit_sigma_range(name: str | None) -> tuple[float, float] | None:
    """Playing-shaft ``(sigma_lo, sigma_hi)`` for a shaft-hittable drum, else None.

    Pads (snare, toms) are tip-only.  Cymbals and the hi-hat may be sounded by
    any point from the bead back to :data:`CYMBAL_CONTACT_SIGMA_FALLBACK`
    (5 cm from the mount, 15 cm from the bead).  The grip (``sigma`` below
    that) does not count.  Cost, eval, probe and the RL scorer all share this
    span so a shaft that spans the rim cannot miss every discrete sample.
    """
    if not shaft_hit_allowed(name):
        return None
    return (float(CYMBAL_CONTACT_SIGMA_FALLBACK), float(STICK_TIP_SIGMA))


def contact_sigma_candidates(
    name: str | None,
    *,
    span: float = SHAFT_HIT_SPAN,
    n_samples: int = SHAFT_HIT_SAMPLES,
) -> tuple[float, ...]:
    """Contact arc coordinates worth trying for a strike on ``name``, best first.

    Tip-only drums get ``(STICK_TIP_SIGMA,)``.  Shaft-hittable drums get the
    bead..shoulder segment, ordered around that drum's own default, with
    :data:`CYMBAL_CONTACT_SIGMA_FALLBACK` appended as a last resort.

    Every returned value is filtered FROM the RL detector's probe list rather
    than computed independently, so the set stays a strict subset of what the
    env can score.
    """
    tip = STICK_TIP_SIGMA
    if not shaft_hit_allowed(name) or n_samples <= 0 or span <= 0.0:
        return (tip,)
    step = float(span) / int(n_samples)
    probes = (tip,) + tuple(tip - step * (i + 1) for i in range(int(n_samples)))
    band = [s for s in probes if s >= CYMBAL_CONTACT_SIGMA_MIN - 1e-9]
    if not band:
        return (tip,)
    default = default_contact_sigma(name)
    band.sort(key=lambda s: (abs(s - default), -s))
    tail = tuple(
        s
        for s in probes
        if abs(s - CYMBAL_CONTACT_SIGMA_FALLBACK) < 1e-9
        and s < CYMBAL_CONTACT_SIGMA_MIN - 1e-9
    )
    return tuple(band) + tail


# ------------------------------------------------------------------- aiming --


def strike_half_plane_dir(
    center: np.ndarray,
    axis: np.ndarray,
    *,
    name: str | None = None,
) -> np.ndarray | None:
    """In-plane facing direction ``t`` for a half-disk strike region, or None.

    None for every head whose whole (rim-buffered) disk is playable: the snare,
    and -- since 2026-08-14 -- the TOMS.  For cymbals, ``t`` points from the
    head centre back toward the drummer, so ``(p - c).t >= 0`` keeps the strike
    on the near semicircle.

    Why the toms lost theirs.  A cymbal is a thin plate on a stand: reaching
    over it to strike the far side means putting the stick through the stand or
    the drummer's own arm, so the cut is real geometry.  A tom is a drum head
    facing up and slightly toward the player; its far side is as strikeable as
    its near side, and the cut was inherited from the cymbal aim geometry rather
    than measured.  Toms keep everything else that made them pads -- the 1 cm
    rim buffer, the elevation band, the tip-only rule, and the near-side aim
    (:func:`_strike_aim_offset` still offsets it) -- so only the LEGAL REGION
    grows, from a half disk to the full rim-buffered disk.

    NOTE this changes the acceptance test: ``ok_hit`` on a tom is strictly
    easier than before, so tom acceptance is not comparable across this change.

    The robot stands at x = y = 0 in both the env and the pelvis frame, so this
    is frame-independent and callers may pass a centre in either.
    """
    if not is_cymbal_name(name):
        return None
    c = np.asarray(center, float)
    n = _unit3(np.asarray(axis, float))

    toward = np.array([-c[0], -c[1], 0.0])
    if float(np.linalg.norm(toward)) > 1e-6:
        t = toward - (toward @ n) * n
        if float(np.linalg.norm(t)) > 1e-9:
            return t / float(np.linalg.norm(t))

    down = np.array([0.0, 0.0, -1.0])
    t = down - (down @ n) * n
    tn = float(np.linalg.norm(t))
    if tn < 1e-9:
        pref = np.array([-1.0, 0.0, 0.0])
        t = pref - (pref @ n) * n
        tn = float(np.linalg.norm(t))
    return t / max(tn, 1e-12)


def _strike_aim_offset(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    name: str | None,
    rim_buffer: float,
) -> np.ndarray | None:
    """In-plane vector from the head centre to the strike aim, or None.

    Single source of truth for the off-centre aim so :func:`strike_aim_on_head`
    and :func:`strike_disk_center_offset` cannot disagree about where the legal
    disk sits relative to the point the objective pulls to.
    """
    cymbal = is_cymbal_name(name)
    if is_hihat_name(name):
        frac = min(HIHAT_RIM_FRACTION,
                   max(0.0, 1.0-HIHAT_AIM_INSET_M/max(float(radius), 1e-9)))
    elif cymbal:
        frac = CYMBAL_EDGE_FRACTION
    elif is_tom_name(name):
        frac = TOM_NEAR_FRACTION
    else:
        return None
    t = strike_half_plane_dir(center, axis, name=name)
    if t is None or frac <= 0.0:
        return None
    # Cymbals are played on the edge, so they measure against the full head.
    # Pads aim within the rim-buffered head they are actually allowed to hit.
    r_use = (
        float(radius)
        if cymbal
        else max(float(radius) - max(float(rim_buffer), 0.0), 1e-3)
    )
    return frac * r_use * t


def strike_axial_limits(
    *,
    surface_offset: float = STRIKE_SURFACE_OFFSET,
    height: float = STRIKE_AXIAL_HEIGHT,
) -> tuple[float, float]:
    """Aim-frame ``(lo, hi)`` along the outward head normal.

    ``axial = (p_contact - aim) · n_out``.  Positive is short of the aim
    (toward / past the geometric surface).  The default band is the interior
    1 cm slab from the aim (inner face) to the head surface (outer face).
    """
    hi = float(surface_offset)
    lo = hi - float(height)
    return lo, hi


def strike_aim_on_head(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    name: str | None = None,
    surface_offset: float = STRIKE_SURFACE_OFFSET,
    rim_buffer: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Cartesian strike aim and in-plane hit-disk radius for a drum head.

    * pads (snare): head centre inset ``surface_offset`` along ``-axis``, disk =
      head radius minus ``rim_buffer``;
    * cymbals: aimed out toward the near edge, **full** head radius, no buffer;
    * toms: aimed toward the drummer within the rim-buffered head.

    ``rim_buffer`` is shared with :func:`strike_disk_center_offset` -- pass the
    same value to both or the disk drifts off the head.
    """
    c = np.asarray(center, float)
    n = _unit3(np.asarray(axis, float))
    r = float(radius)
    buf = max(float(rim_buffer), 0.0)
    aim = c - float(surface_offset) * n
    off = _strike_aim_offset(c, n, r, name=name, rim_buffer=buf)
    if off is not None:
        aim = aim + off
    if is_cymbal_name(name):
        return aim, max(r, 1e-3)
    return aim, max(r - buf, 1e-3)


def strike_disk_center_offset(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    name: str | None = None,
    rim_buffer: float = 0.01,
) -> np.ndarray | None:
    """Vector from the strike aim to the head centre, or None when they coincide.

    The hit disk is centred on the *aim*; cymbals and toms aim off-centre, so the
    disk bounds must be shifted back by this offset to keep the legal region on
    the head:  ``|p - c| <= R``  becomes  ``|(p - aim) - d| <= R``.
    """
    off = _strike_aim_offset(
        np.asarray(center, float),
        np.asarray(axis, float),
        float(radius),
        name=name,
        rim_buffer=rim_buffer,
    )
    return None if off is None else -off


# ------------------------------------------------------- impact velocity ----


def strike_velocity_weight(
    v_star: np.ndarray,
    *,
    w_strike_velocity: float = 1.0,
    a_normal: float = 1.0,
    a_tangential: float = 1.0,
    drum: str | None = None,
) -> np.ndarray:
    """3x3 weight ``W`` for the impact-velocity residual of one strike.

    ``cost = (v_contact - v*)' W (v_contact - v*)``, with::

        W = w * (a_n * n n^T + a_t * (I - n n^T)),   n = v* / ||v*||

    so ``a_n`` prices the descent-speed match and ``a_t`` prices motion parallel
    to the head surface.

    ``drum`` is accepted and currently IGNORED: every drum, cymbals included,
    takes the same pair.  A cymbal used to get ``a_t = 0.5`` so a crash could
    arrive as a glancing sweep; it was removed because a near-in-plane arrival is
    the one failure mode a thin plate cannot tolerate.  The parameter stays in
    the signature so re-introducing a per-drum split is a one-line change.

    A degenerate ``v*`` (zero commanded speed) has no approach axis, so this
    falls back to isotropic ``a_t`` -- a target asking the contact point to be at
    rest wants ALL motion penalized.

    The repo signature is ``strike_velocity_weight(cfg, v_star, drum=...)``; the
    three numbers read off ``cfg`` are explicit keyword arguments here so the
    module stays standalone.
    """
    del drum  # no per-drum split: cymbals are plunged like pads (see above)
    w = float(w_strike_velocity)
    a_n = float(a_normal)
    a_t = float(a_tangential)
    v = np.asarray(v_star, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    if speed <= 1e-9 or not np.isfinite(speed):
        return (w * a_t) * np.eye(3)
    if a_n == a_t:  # isotropic -- keep it exactly diagonal
        return (w * a_n) * np.eye(3)
    n = v / speed
    nn = np.outer(n, n)
    return w * (a_n * nn + a_t * (np.eye(3) - nn))


# ---------------------------------------------------------------- elevation --


def stick_elevation_from_surface(stick_axis: np.ndarray, normal: np.ndarray) -> float:
    """Signed stick elevation (rad) off the drum surface.

    ``asin(clip(-stick.n, -1, 1))``: ``+pi/2`` means the stick points along
    ``-n`` (straight into the head), ``-pi/2`` along ``+n``.  ``stick_axis`` is
    the handle->tip unit direction.
    """
    n = _unit3(np.asarray(normal, float))
    d = _unit3(np.asarray(stick_axis, float))
    return float(np.arcsin(np.clip(-float(d @ n), -1.0, 1.0)))


def stick_elevation_ok(
    stick_axis: np.ndarray,
    normal: np.ndarray,
    *,
    elev_min_deg: float = STICK_ELEV_MIN_DEG,
    elev_max_deg: float = STICK_ELEV_MAX_DEG,
) -> bool:
    """True if ``|elevation|`` sits in ``[min, max]`` degrees.

    Sign-blind, matching ``robot.stick_elevation_ok``: the G1 stick mount often
    has ``stick . n > 0`` with the tip on the pad, and the band still rejects
    both the in-plane graze and near-normal tunnelling.
    """
    elev = abs(stick_elevation_from_surface(stick_axis, normal))
    return (
        np.deg2rad(float(elev_min_deg)) - 1e-3
        <= elev
        <= np.deg2rad(float(elev_max_deg)) + 1e-3
    )


# --------------------------------------------------------------- StrikeSpec --


@dataclass
class StrikeSpec:
    """One drum strike: at ``time`` the given hand's contact point should reach
    ``position`` moving with ``velocity``.

    Field-for-field the repo's ``trajopt.problem.StrikeSpec`` minus the two
    solver-only members (``cached_goal_q``, which is a seed hint, and nothing
    else).  Kept as a plain dataclass so it serializes into the harness's case
    files without dragging the solver in.

    * ``drum`` names the obstacle whose avoidance rows are suspended for the
      striking stick around the strike waypoint -- you cannot both avoid a drum
      and hit it.
    * ``hit_radius`` is the legal in-plane disk radius (``strike_aim_on_head``
      returns the right one per drum).
    * ``half_plane_dir`` cuts the disk to the half facing that in-plane unit
      direction (cymbals, toms).  None = whole disk.
    * ``disk_center_offset`` is the in-plane vector from ``position`` to the head
      centre, for strikes that aim off-centre.  None = disk centred on the aim.
    * ``contact_sigma`` is the diagnostic / aim point ON THE STICK, as an arc
      coordinate from the stick's centre (``STICK_TIP_SIGMA`` = the bead).
      Pads must land that point (the bead) in the disk.  Cymbals and the hi-hat
      use the playing-shaft capsule (:func:`shaft_hit_sigma_range`); velocity
      is applied at the closest point on that capsule.
    """

    time: float
    hand: str  # "left" | "right"
    position: np.ndarray
    velocity: np.ndarray
    drum: str | None = None
    hit_radius: float | None = None
    half_plane_dir: np.ndarray | None = None
    disk_center_offset: np.ndarray | None = None
    contact_sigma: float | None = None

    def waypoint(self, dt: float) -> int:
        return int(round(self.time / dt))

    def resolved_hit_radius(self, cfg_hit_radius: float = 0.08) -> float:
        if self.hit_radius is not None and np.isfinite(float(self.hit_radius)):
            return float(self.hit_radius)
        return float(cfg_hit_radius)

    def resolved_contact_sigma(self) -> float:
        """``contact_sigma``, defaulting per drum, clamped to the stick."""
        s = self.contact_sigma
        if s is None or not np.isfinite(float(s)):
            return default_contact_sigma(self.drum)
        return float(np.clip(float(s), -STICK_TIP_SIGMA, STICK_TIP_SIGMA))

    def is_shaft_contact(self) -> bool:
        """True when the contact sits behind the bead, so the shaft geom is
        allowed to touch the target drum."""
        return self.resolved_contact_sigma() < STICK_TIP_SIGMA - 1e-9


def strike_outward_normal(strike: StrikeSpec) -> np.ndarray:
    """Unit vector pointing OFF the struck head, i.e. ``-v*/||v*||``.

    The commanded strike velocity points *into* the head, so its negation points
    away.  Falls back to +Z for a degenerate (zero) strike velocity.
    """
    v = np.asarray(strike.velocity, float).reshape(3)
    vn = float(np.linalg.norm(v))
    return -v / vn if vn > 1e-9 else np.array([0.0, 0.0, 1.0])


def strike_from_drum(
    *,
    time: float,
    hand: str,
    name: str,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    speed: float,
    contact_sigma: float | None = None,
    surface_offset: float = STRIKE_SURFACE_OFFSET,
    rim_buffer: float = 0.01,
) -> StrikeSpec:
    """Build the canonical :class:`StrikeSpec` for one drum, hand and speed.

    Exactly the construction every bench in the repo repeats inline (aim +
    hit radius + half plane + disk offset, velocity ``-speed * axis``), so the
    harness and the section-3 cost cannot drift apart on it.

    ``center`` / ``axis`` must be in the same frame the trajectory is scored in
    (the harness uses the pelvis frame, matching ``drums_from_layout``'s default).
    """
    c = np.asarray(center, float)
    n = _unit3(np.asarray(axis, float))
    p_hit, hit_r = strike_aim_on_head(
        c, n, float(radius), name=name, surface_offset=surface_offset, rim_buffer=rim_buffer
    )
    return StrikeSpec(
        time=float(time),
        hand=str(hand),
        position=p_hit,
        velocity=-float(speed) * n,
        drum=str(name),
        hit_radius=float(hit_r) if np.isfinite(hit_r) else 0.08,
        half_plane_dir=strike_half_plane_dir(c, n, name=name),
        disk_center_offset=strike_disk_center_offset(
            c, n, float(radius), name=name, rim_buffer=rim_buffer
        ),
        contact_sigma=contact_sigma,
    )
