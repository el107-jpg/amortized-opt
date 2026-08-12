"""Drum-strike geometry: where on a head to aim, and what counts as a hit.

Vendored from ``source/g1_drummer/g1_drummer/control/trajopt/problem.py``
(2026-08-10).  Nothing here imports the repo; the only dependency is numpy plus
``amortized_opt.constants``.

The rules encoded here are *tuned*, not derived -- they came out of the repo's
RL hit detector and months of trajopt tuning -- so this is a port, not a
reimplementation.  ``tests/test_strike_conformance.py`` asserts every function
matches the repo original to 1e-12 on randomized inputs (and skips when the repo
is absent).

Two consumers:

* the section-2 evaluation harness (:mod:`amortized_opt.eval_harness`), which
  needs to know where the contact point is and whether it landed on the head;
* the section-3 differentiable terminal cost, which needs the same aim point,
  the same half-plane cut and the same impact-velocity weighting.

Semantic summary (see the individual docstrings for the reasoning):

* **Cymbals and the hi-hat** may be sounded by the stick SHAFT, so the contact
  point sits ``contact_sigma`` back from the bead (default: the shoulder,
  10 cm).  They are aimed off-centre toward the near edge, keep the *full* head
  radius (no rim buffer), and only the near semicircle is legal.
* **Toms** are tip-only pads that borrow the cymbals' off-centre *geometry*:
  aimed 45 % of the rim-buffered radius toward the drummer, near semicircle only.
* **Everything else** (the snare) is a tip-only pad aimed dead centre, hit disk
  = head radius minus the rim buffer, whole disk legal.
* Every drum requires the terminal stick elevation to sit in a two-sided band.
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
    "HIHAT_SHOULDER_SIGMA",
    "STICK_ELEV_MAX_DEG",
    "STICK_ELEV_MIN_DEG",
    "STICK_TIP_SIGMA",
    "StrikeSpec",
    "TOM_NEAR_FRACTION",
    "contact_sigma_candidates",
    "default_contact_sigma",
    "is_cymbal_name",
    "is_hihat_name",
    "is_tom_name",
    "shaft_hit_allowed",
    "stick_elevation_from_surface",
    "stick_elevation_ok",
    "strike_aim_on_head",
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

#: Same, for the HI-HAT, which is aimed at the rim itself rather than the bow.
HIHAT_RIM_FRACTION = 1.0

#: Same idea for the TOMS, as a fraction of the *usable* (rim-buffered) radius
#: out from the centre toward the drummer.
TOM_NEAR_FRACTION = 0.45

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
    *geometry* of the aim -- off-centre toward the drummer, far semicircle cut.
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

    True for EVERY drum, cymbals and hi-hat included.  Kept as a predicate
    because the call sites read as "does this drum have a terminal pose
    requirement", and because a future layout may answer no.
    """
    del name  # every drum, for now
    return True


# ------------------------------------------------------------ contact point --


def default_contact_sigma(name: str | None) -> float:
    """Contact point a strike on ``name`` uses when it does not name one.

    The bead everywhere except the cymbals (hi-hat included), which default to
    the shoulder because a crash/edge articulation is a shaft hit.  A caller
    that specifically wants a bead strike on a cymbal must pass
    ``contact_sigma=STICK_TIP_SIGMA`` explicitly.
    """
    if is_cymbal_name(name):
        return HIHAT_SHOULDER_SIGMA
    return STICK_TIP_SIGMA


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

    None for heads with no half-plane (the snare -- the whole head is playable).
    For cymbals and TOMS, ``t`` points from the head centre back toward the
    drummer, so ``(p - c).t >= 0`` keeps the strike on the near semicircle.

    The robot stands at x = y = 0 in both the env and the pelvis frame, so this
    is frame-independent and callers may pass a centre in either.
    """
    if not (is_cymbal_name(name) or is_tom_name(name)):
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
        frac = HIHAT_RIM_FRACTION
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


def strike_aim_on_head(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    name: str | None = None,
    surface_offset: float = 0.005,
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
    * ``contact_sigma`` is the point ON THE STICK that must land in the disk, as
      an arc coordinate from the stick's centre (``STICK_TIP_SIGMA`` = the bead).
      Legal behind the bead on the cymbals only.  The velocity requirement is
      applied at that same point, which makes a shoulder hit harder in speed
      exactly as it is easier in position.
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
    surface_offset: float = 0.005,
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
