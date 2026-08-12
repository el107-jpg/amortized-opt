"""Joint-space path spline + quintic timing law (the warm-start seed).

Vendored verbatim from ``control/trajopt/spline.py`` except for
``retime_joint_path``, whose signature took a Pinocchio-backed
``trajopt.robot.G1UpperBody``. Here the three things it read off that object
(``v_start``/``a_start``, the joint box for clamping, and the tip Jacobian gain)
are explicit arguments, so the module is standalone. The math is unchanged.

The seed is two independent pieces:

* **Path** -- a C2 cubic through the joint poses, parameterized by joint-space
  arc length ``u in [0, L]`` (radians). Under that parameterization
  ``|dq/du| = 1``, so ``||qdot|| = udot`` exactly and the timing law bounds
  joint speed directly.
* **Timing** -- a scalar quintic ``u(t): [0, T] -> [0, L]`` with prescribed
  speed and acceleration at both ends.

The seed is ``path(u(t))`` -- a pure resampling, so re-timing cannot move the
path off a curve that was already collision-checked.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

__all__ = [
    "joint_path_spline",
    "peak_rate",
    "quintic_time_scaling",
    "retime_joint_path",
]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def joint_path_spline(Q: np.ndarray):
    """C2 cubic through joint poses ``Q`` (m x n), parameterized by arc length.

    Returns ``(spline, length)``. The parameter is cumulative joint-space chord
    length in radians. Coincident poses are dropped (duplicate parameter values
    are rejected by ``make_interp_spline``); the degree drops for very short
    paths so a 2- or 3-knot path still builds.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] < 2:
        raise ValueError(f"need at least 2 joint poses, got {Q.shape}")
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    keep = np.concatenate([[True], np.diff(u) > 1e-9])
    u, Q = u[keep], Q[keep]
    if u.size < 2:
        raise ValueError("joint path collapsed to a single pose")
    k = min(u.size - 1, 3)
    return make_interp_spline(u, Q, k=k), float(u[-1])


def quintic_time_scaling(T: float, s_end: float, sd0: float, sdd0: float, sd1: float, sdd1: float):
    """Scalar quintic ``s(t)`` on ``[0, T]`` with ``s(0)=0`` and ``s(T)=s_end``.

    Endpoint rates are prescribed: ``s'(0)=sd0``, ``s''(0)=sdd0``, ``s'(T)=sd1``,
    ``s''(T)=sdd1``. Six coefficients, six conditions. The peak rate is above the
    average (see :func:`peak_rate`); the caller must clip into ``[0, s_end]``
    since a quintic meeting aggressive endpoint rates is not guaranteed monotone.
    """
    T = float(T)
    if T <= 0:
        raise ValueError(f"timing horizon must be positive, got {T}")
    a0, a1, a2 = 0.0, float(sd0), 0.5 * float(sdd0)
    A = np.array(
        [[T**3, T**4, T**5],
         [3 * T**2, 4 * T**3, 5 * T**4],
         [6 * T, 12 * T**2, 20 * T**3]],
        dtype=float,
    )
    rhs = np.array(
        [float(s_end) - (a0 + a1 * T + a2 * T**2),
         float(sd1) - (a1 + 2 * a2 * T),
         float(sdd1) - 2 * a2],
        dtype=float,
    )
    a3, a4, a5 = np.linalg.solve(A, rhs)
    c = np.array([a0, a1, a2, a3, a4, a5], dtype=float)[::-1]

    def s(t: np.ndarray) -> np.ndarray:
        return np.polyval(c, np.asarray(t, dtype=float))

    s.coeffs = c  # type: ignore[attr-defined]  # high-order first, for np.polyval
    s.horizon = T  # type: ignore[attr-defined]
    return s


def peak_rate(s) -> float:
    """Max ``|s'(t)|`` on ``[0, T]`` for a law from :func:`quintic_time_scaling`.

    Under joint arc-length parameterization this is the peak ``||qdot||``.
    """
    c = np.asarray(s.coeffs, dtype=float)
    T = float(s.horizon)
    sd = np.polyder(c)
    crit = [r.real for r in np.roots(np.polyder(sd))
            if abs(r.imag) < 1e-9 and 0.0 <= r.real <= T]
    ts = np.array([0.0, T, *crit], dtype=float)
    return float(np.abs(np.polyval(sd, ts)).max())


def retime_joint_path(
    Q: np.ndarray,
    n_waypoints: int,
    dt: float,
    *,
    v_start: np.ndarray | None = None,
    a_start: np.ndarray | None = None,
    v_limit: np.ndarray | None = None,
    q_lower: np.ndarray | None = None,
    q_upper: np.ndarray | None = None,
    tip_speed: float | None = None,
    tip_gain: float | None = None,
) -> tuple[np.ndarray, float]:
    """Walk the joint-space path spline under a quintic timing law.

    ``Q`` is *where* (a dense joint polyline); this decides *when*.

    * start -- ``v_start``/``a_start`` projected on the path tangent (clamped at
      0: a backward projection makes the quintic reverse off the path start).
    * end -- ``u'(T) = tip_speed / tip_gain`` where ``tip_gain`` is
      ``||J(q_end) @ tangent_end||`` (m of tip per rad of path). Both must be
      given, else the law ends at rest.

    Returns ``(X, peak_rate)``: the ``(n_waypoints, n)`` resampled path and the
    seed's peak ``||qdot||`` so an infeasible retiming is visible rather than
    silently clipped (the original stashed this on ``rob._last_seed_peak_rate``).
    """
    Q = np.asarray(Q, dtype=float)
    n_waypoints = int(n_waypoints)
    if float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1))) < 1e-9:
        return np.repeat(Q[-1][None, :], n_waypoints, axis=0), 0.0

    path, length = joint_path_spline(Q)
    tangent = path.derivative(1)
    t0 = _unit(np.asarray(tangent(0.0), float))
    tL = _unit(np.asarray(tangent(length), float))

    sd0 = sdd0 = 0.0
    if v_start is not None:
        v = np.asarray(v_start, float)
        if v_limit is not None:
            v = np.clip(v, -np.asarray(v_limit, float), np.asarray(v_limit, float))
        sd0 = max(float(v @ t0), 0.0)
        if a_start is not None:
            sdd0 = float(np.asarray(a_start, float) @ t0)

    sd1 = 0.0
    if tip_speed is not None and tip_speed > 1e-6 and tip_gain is not None:
        sd1 = float(tip_speed) / max(float(tip_gain), 1e-6)

    T = dt * (n_waypoints - 1)
    u = quintic_time_scaling(T, length, sd0, sdd0, sd1, 0.0)
    try:
        rate = float(peak_rate(u))
    except Exception:
        rate = float("nan")
    u_t = np.clip(u(np.arange(n_waypoints, dtype=float) * dt), 0.0, length)
    X = np.asarray(path(u_t), dtype=float)
    if q_lower is not None and q_upper is not None:
        X = np.clip(X, np.asarray(q_lower, float), np.asarray(q_upper, float))
    # Endpoints are exact by construction; re-pin against spline round-off.
    X[0] = Q[0]
    X[-1] = Q[-1]
    return X, rate
