"""Reference-lead compensation for the position-controlled command path.

The problem (measured, section 7c)
----------------------------------

The acceptance grid scores a *plan*: it asks whether the waypoint the planner
nominates for the hit instant sits on the drum head.  Nothing in the plan, the
cost or the grid asks whether a position-controlled servo can BE there at that
instant, and at a 2 m/s approach it cannot: the only torque source is
``kp * (q_cmd - q) + kd * (qd_cmd - qd)``, so accelerating the arm REQUIRES a
standing position error of roughly ``M(q) qdd / kp``.  Rolling the champion's
grid plans through the PD sim with no replanning leaves the stick a median
**143 mm** short of the head at its own terminal knot, with a peak joint
tracking error of 152 mrad, while the plan itself is 9 mm off.  Closed-loop
replanning recovers most of it (36 mm), because a one-waypoint step from the
measured state is a much smaller demand -- but not all of it.

The fix, from the plan doc's "Lag adjustment"
---------------------------------------------

Command a point slightly AHEAD on the planned trajectory::

    q_cmd(t) = q_ref(t + delta)

so the servo's own delay lands the plant on ``q_ref(t)``.  Because the plan is a
waypoint sequence on the ``CONTROL_DT`` grid, "ahead" is a linear interpolation
between consecutive waypoints, which makes SUB-TICK deltas well defined --
:func:`lead_waypoints` is the whole implementation.

Two properties matter and are what make this preferable to shifting the goal's
time-to-hit (the ``MPCGenConfig.hit_lead_s`` knob, kept but secondary):

* the WHOLE trajectory keeps its timing, so the approach profile, the impact
  speed and the two hands' relative phase are unchanged -- only the command is
  advanced.  A goal shift re-solves the whole plan against an earlier terminal,
  which also moves everything that was correct.
* the velocity feed-forward stays consistent for free.  The FOH plant derives
  its rate target from ``(x_i - x_{i-1}) / dt`` over the EXECUTED commands, so
  leading every command by the same delta leads the rate target by the same
  delta.  Leading only the position would feed the servo a led position against
  an un-led rate, which is a different (and wrong) controller.

What it cannot do
-----------------

A lead cancels *delay*.  It does not cancel *attenuation*: a servo whose closed
loop rolls off below the strike's frequency content reaches a smaller peak
velocity no matter when it is asked to.  Of the 152 mrad transient the lead
recovers the part that is a pure phase shift; the rest is the training
continuation's job (make the planner ask for motions this servo can produce).

Per-joint delta
---------------

The lag is ``M(q) qdd / kp``, and both ``M`` and ``kp`` vary by two orders of
magnitude across the 17 joints (``kp`` 200 at ``waist_yaw`` vs 40 elsewhere;
effective inertia from the shoulder to the wrist likewise), so one delta cannot
be right for all of them.  :func:`fit_per_joint_lead` identifies it numerically
against the exact servo model this package owns -- see
``scripts/fit_cmd_lead.py``.
"""

from __future__ import annotations

import numpy as np
import torch

#: Per-joint reference leads (seconds, ``G1_UPPER_BODY_JOINTS`` order) fitted by
#: ``scripts/fit_cmd_lead.py`` against the exact PD model, on the 162-case
#: ``standard`` grid with randomized gains, minimizing joint tracking RMS over
#: the 6 knots around the strike.
#:
#: ``"fit"``    -- the raw per-joint argmin over a 0..50 ms sweep.
#: ``"sparse"`` -- the same fit keeping ONLY the joints whose lead buys >= 10 %
#:                of their tracking RMS, capped at 50 ms.  Four numbers instead
#:                of seventeen, and each is a joint where the fit has a real
#:                minimum rather than a flat floor; the rest of the vector is a
#:                fit to noise.
#: ``"wide"``   -- the argmin over a 0..150 ms sweep.  The waist and the left
#:                shoulder run into the narrow sweep's ceiling, so this is what
#:                they actually want; it is kept for the record and is NOT
#:                recommended (leading 5 waypoints into a 25-waypoint plan
#:                extrapolates a long way past anything the cost priced).
LEAD_PRESETS: dict[str, tuple[float, ...]] = {
    "fit": (
        0.0025, 0.05, 0.05,
        0.05, 0.0075, 0.0, 0.015, 0.0, 0.0025, 0.0,
        0.0, 0.005, 0.0, 0.015, 0.0025, 0.0, 0.0,
    ),
    # Re-fit 2026-08-12 against the CURRENT plant (arms Kd 4.0, PD_SUBSTEPS 10).
    # ``"fit"`` above was identified at Kd 2.0 / S=5 and is kept for provenance:
    # every number in sections 7b-8 was produced with it.  Strike-window RMS at
    # the new gains: waist 38.4 -> 19.5 mrad, elbow 25.6 -> 13.5, shoulder
    # 29.2 -> 27.9, wrist 3.4 -> 3.3.
    "fit_kd4": (
        0.0025, 0.05, 0.05,
        0.015, 0.01, 0.0, 0.015, 0.0, 0.0, 0.0,
        0.005, 0.0075, 0.0025, 0.0125, 0.0025, 0.0, 0.0,
    ),
    "sparse": (
        0.0, 0.05, 0.05,
        0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0,
    ),
    "wide": (
        0.0025, 0.08, 0.10,
        0.10, 0.12, 0.0, 0.015, 0.0, 0.0025, 0.0,
        0.0, 0.005, 0.0, 0.015, 0.0025, 0.0, 0.0,
    ),
}


def resolve_lead(spec):
    """``None`` / a float / a preset NAME / a sequence -> a lead specification.

    A ``{"schedule": [[...], ...]}`` dict passes through untouched; see
    :func:`is_schedule`.
    """
    if isinstance(spec, str):
        if spec in ("", "none", "off"):
            return 0.0
        if spec in LEAD_PRESETS:
            return LEAD_PRESETS[spec]
        return float(spec)
    return spec


def is_schedule(spec) -> bool:
    """True for a HORIZON-DEPENDENT lead, ``{"schedule": (n_h, n_dof) table}``.

    The fixed presets answer "how far ahead should this joint's command read?"
    with one number per joint.  ``fit_lead_schedule.py`` measures that the answer
    moves by an order of magnitude along the approach, so a schedule indexes the
    same question by ``h``, the number of waypoints REMAINING to that hand's hit
    (row 0 = the strike waypoint).  Row ``min(h, n_h - 1)`` is used, so the table
    only has to cover the region where the lead actually varies.
    """
    return isinstance(spec, dict) and "schedule" in spec


def schedule_fraction(spec, dt: float, n_dof: int) -> np.ndarray:
    """``{"schedule": table}`` -> an ``(n_h, n_dof)`` array of TICK fractions."""
    tab = np.asarray(spec["schedule"], dtype=float)
    if tab.ndim != 2 or tab.shape[1] != n_dof:
        raise ValueError(f"lead schedule must be (n_h, {n_dof}), got {tab.shape}")
    if np.any(tab < 0.0):
        raise ValueError("a NEGATIVE lead would command the past; got "
                         f"{tab.min():.4g} s")
    return tab / float(dt)


__all__ = ["LEAD_PRESETS", "is_schedule", "lead_fraction", "lead_waypoints",
           "lead_waypoints_np", "resolve_lead", "schedule_fraction"]


def lead_fraction(lead_s, dt: float, n_dof: int, device=None, dtype=None):
    """Normalize a scalar / per-joint lead in SECONDS to a ``(n,)`` tick fraction.

    Returns ``None`` when the lead is identically zero, which every call site
    uses to take the untouched fast path.
    """
    if lead_s is None:
        return None
    arr = np.asarray(lead_s, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(n_dof, float(arr[0]))
    if arr.size != n_dof:
        raise ValueError(f"lead must be scalar or ({n_dof},), got {arr.shape}")
    # the sign check comes FIRST: an all-negative vector would otherwise take
    # the "nothing to do" fast path and silently be ignored
    if np.any(arr < 0.0):
        raise ValueError("a NEGATIVE lead would command the past; got "
                         f"{arr.min():.4g} s")
    if not np.any(arr > 0.0):
        return None
    f = arr / float(dt)
    if device is None:
        return f
    return torch.as_tensor(f, device=device, dtype=dtype)


def lead_waypoints(traj: torch.Tensor, frac) -> torch.Tensor:
    """``(B, H, n)`` plan -> the same plan sampled ``frac`` waypoints ahead.

    Waypoint ``i`` of ``traj`` is ``q_ref((i + 1) * dt)``; the result's waypoint
    ``i`` is ``q_ref((i + 1) * dt + frac * dt)``, by linear interpolation, with
    the last waypoint HELD beyond the end of the plan (the planner has said
    nothing about what happens after its terminal, and holding is the only
    choice that cannot invent motion).

    ``frac`` is a ``(n,)`` tensor or a float and may exceed 1, so a lead longer
    than one control period is a whole-waypoint shift plus a remainder.
    Differentiable in ``traj``; the gather is index-only.
    """
    if frac is None:
        return traj
    B, H, n = traj.shape
    f = torch.as_tensor(frac, device=traj.device, dtype=traj.dtype).reshape(1, 1, -1)
    f = f.expand(1, 1, n)
    m = torch.floor(f)                       # whole waypoints
    r = f - m                                # remainder in [0, 1)
    idx = torch.arange(H, device=traj.device).view(1, H, 1)
    i0 = (idx + m.long()).clamp(0, H - 1).expand(B, H, n)
    i1 = (idx + m.long() + 1).clamp(0, H - 1).expand(B, H, n)
    a = traj.gather(1, i0)
    b = traj.gather(1, i1)
    return a + r * (b - a)


def lead_waypoints_np(traj: np.ndarray, frac) -> np.ndarray:
    """Numpy twin of :func:`lead_waypoints` for the host-side MPC loop."""
    if frac is None:
        return traj
    T = torch.as_tensor(np.asarray(traj, float))[None]
    out = lead_waypoints(T, torch.as_tensor(np.asarray(frac, float)))
    return out[0].numpy()
