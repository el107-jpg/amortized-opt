"""Pure helpers for the one-off MPC replay boundary diagnostic.

The helpers in this module do not load checkpoints, build the robot, or mutate
training state.  They only transform supplied tensors and summarize them; the
executable probe lives in :mod:`scripts.probe_replay_boundary`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def make_c1_roller_plan_transform(bridge_knots: int):
    """Build the diagnostic roller-wide C1 projection, or ``None`` for zero.

    The returned callback matches :class:`MPCRoller`'s diagnostic hook.  It
    receives the measured state and live-element mask from the same tick as the
    decoded plan, ensuring projection happens before command execution and
    before the projected plan is shifted into replay.
    """
    knots = int(bridge_knots)
    if knots < 0:
        raise ValueError("bridge_knots must be >= 0")
    if knots == 0:
        return None

    def transform(*, traj, q0, v0, elem_mask, dt):
        return c1_bridge_waypoints(
            traj, q0, v0, dt=float(dt), bridge_knots=knots,
            elem_mask=elem_mask,
        )

    return transform


def c1_bridge_waypoints(
    waypoints: torch.Tensor,
    q0: torch.Tensor,
    v0: torch.Tensor,
    *,
    dt: float,
    bridge_knots: int = 3,
    elem_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a diagnostic C1 bridge from ``(q0, v0)`` into ``waypoints``.

    For each live joint, a cubic Hermite curve joins the measured boundary to
    the original waypoint at ``min(bridge_knots, horizon)``.  Its terminal
    derivative is the original plan's incoming finite-difference velocity.
    The join waypoint and every later waypoint are preserved exactly.  A
    one-segment horizon cannot be repaired while also preserving its only
    terminal waypoint, so it is deliberately left unchanged.

    This is a counterfactual for diagnosis, not a proposed deployment policy.
    """
    if waypoints.ndim != 3 or q0.ndim != 2 or v0.ndim != 2:
        raise ValueError("expected waypoints (B,H,D), q0/v0 (B,D)")
    if waypoints.shape[0] != q0.shape[0] or q0.shape != v0.shape:
        raise ValueError("batch/state shapes do not agree")
    if waypoints.shape[2] != q0.shape[1]:
        raise ValueError("waypoint/state joint counts do not agree")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if bridge_knots < 1:
        raise ValueError("bridge_knots must be >= 1")

    B, H, D = waypoints.shape
    if H == 0:
        return waypoints.clone()
    if elem_mask is None:
        lengths = torch.full(
            (B, D), H, device=waypoints.device, dtype=torch.long,
        )
    else:
        if elem_mask.shape != waypoints.shape:
            raise ValueError("elem_mask must have the same shape as waypoints")
        lengths = (elem_mask > 0).sum(dim=1).to(torch.long)

    join = lengths.clamp(min=1, max=min(int(bridge_knots), H))
    end_idx = (join - 1).unsqueeze(1)
    q_end = waypoints.gather(1, end_idx).squeeze(1)
    prev_idx = (join - 2).clamp_min(0).unsqueeze(1)
    q_prev_wp = waypoints.gather(1, prev_idx).squeeze(1)
    q_prev = torch.where(join > 1, q_prev_wp, q0)
    v_end = (q_end - q_prev) / float(dt)

    k = torch.arange(
        1, H + 1, device=waypoints.device, dtype=waypoints.dtype,
    ).view(1, H, 1)
    join_f = join.to(waypoints.dtype).unsqueeze(1)
    u = (k / join_f).clamp(0.0, 1.0)
    u2, u3 = u * u, u * u * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    duration = join_f * float(dt)
    bridge = (
        h00 * q0.unsqueeze(1)
        + h10 * duration * v0.unsqueeze(1)
        + h01 * q_end.unsqueeze(1)
        + h11 * duration * v_end.unsqueeze(1)
    )

    # Strictly before the preserved join.  Dead and one-segment elements stay
    # byte-for-byte equal to the input.
    replace = (k < join_f) & (k <= lengths.unsqueeze(1)) & (lengths.unsqueeze(1) > 1)
    return torch.where(replace, bridge, waypoints)


def boundary_kinematics(
    q0: torch.Tensor,
    v0: torch.Tensor,
    waypoints: torch.Tensor,
    *,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Per-row and per-joint boundary mismatch under knot-FD semantics."""
    if waypoints.ndim != 3 or waypoints.shape[1] < 1:
        raise ValueError("waypoints must have shape (B,H,D) with H >= 1")
    if q0.shape != v0.shape or waypoints[:, 0].shape != q0.shape:
        raise ValueError("q0, v0, and waypoint boundary shapes do not agree")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    delta = waypoints[:, 0] - q0
    velocity = delta / float(dt)
    acceleration = (velocity - v0) / float(dt)
    D = max(int(delta.shape[1]), 1)
    return {
        "delta": delta,
        "implied_velocity": velocity,
        "implied_acceleration": acceleration,
        "q0_first_l2_rad": delta.square().sum(dim=1).sqrt(),
        "q0_first_rms_rad": (delta.square().sum(dim=1) / D).sqrt(),
        "q0_first_abs_max_rad": delta.abs().amax(dim=1),
        "implied_velocity_abs_max_rads": velocity.abs().amax(dim=1),
        "implied_acceleration_abs_max_rads2": acceleration.abs().amax(dim=1),
    }


def localize_torque_excess(
    tau: torch.Tensor,
    hard_limit: torch.Tensor,
    mask: torch.Tensor,
    *,
    segment_indices: Mapping[str, Sequence[int]] | None = None,
    windows: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Localize hard-limit torque excess by knot, joint, and actuator segment.

    ``first_1`` includes boundary knot 0.  All reductions are per row.  The
    returned tensors retain the input device and are suitable for later JSON
    conversion or differentiable-free analysis.
    """
    if tau.ndim != 3:
        raise ValueError("tau must have shape (B,K,D)")
    if hard_limit.shape != tau.shape or mask.shape != tau.shape:
        raise ValueError("hard_limit and mask must match tau")
    m = mask.to(tau.dtype)
    over = (tau.abs() - hard_limit).clamp_min(0.0) * m
    abs_tau = tau.abs() * m
    B, K, D = tau.shape
    flat = over.reshape(B, K * D)
    peak = flat.amax(dim=1)
    arg = flat.argmax(dim=1)
    total = over.sum(dim=(1, 2))

    out: dict[str, Any] = {
        "excess": over,
        "row_peak_excess_nm": peak,
        "row_peak_abs_tau_nm": abs_tau.reshape(B, K * D).amax(dim=1),
        "row_peak_knot": torch.div(arg, D, rounding_mode="floor"),
        "row_peak_joint": arg.remainder(D),
        "row_excess_sum_nm": total,
        "by_joint_peak_excess_nm": over.amax(dim=1),
        "windows": {},
    }
    knot = torch.arange(K, device=tau.device).view(1, K, 1)
    for n0 in windows:
        n = max(0, min(int(n0), K))
        first_mask = (knot < n).to(tau.dtype) * m
        rest_mask = (knot >= n).to(tau.dtype) * m

        def reduced(which: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            vals = over * which
            vmax = vals.amax(dim=(1, 2))
            denom = which.sum(dim=(1, 2)).clamp_min(1.0)
            return vmax, vals.sum(dim=(1, 2)) / denom

        fmax, fmean = reduced(first_mask)
        rmax, rmean = reduced(rest_mask)
        fsum = (over * first_mask).sum(dim=(1, 2))
        out["windows"][f"first_{n0}"] = {
            "peak_excess_nm": fmax,
            "mean_excess_nm": fmean,
            "excess_sum_fraction": torch.where(
                total > 0.0, fsum / total.clamp_min(1e-30), torch.zeros_like(total),
            ),
        }
        out["windows"][f"rest_after_{n0}"] = {
            "peak_excess_nm": rmax,
            "mean_excess_nm": rmean,
        }

    by_segment: dict[str, torch.Tensor] = {}
    if segment_indices:
        for name, indices in segment_indices.items():
            idx = torch.as_tensor(indices, device=tau.device, dtype=torch.long)
            if idx.numel() == 0:
                by_segment[name] = tau.new_zeros(B)
            else:
                by_segment[name] = over.index_select(2, idx).amax(dim=(1, 2))
    out["by_segment_peak_excess_nm"] = by_segment
    return out


def finite_stats(values: torch.Tensor | Sequence[float]) -> dict[str, float | int | None]:
    """JSON-ready finite-value descriptive statistics."""
    x = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {
            "n": 0, "mean": None, "p50": None, "p90": None, "p99": None,
            "min": None, "max": None,
        }
    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99], dtype=x.dtype))
    return {
        "n": int(x.numel()),
        "mean": float(x.mean()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def pearsonr_finite(
    x: torch.Tensor | Sequence[float],
    y: torch.Tensor | Sequence[float],
) -> dict[str, float | int | None]:
    """Pearson correlation after pairwise finite filtering."""
    a = torch.as_tensor(x).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(y).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("correlation inputs must have equal shape")
    keep = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.numel())
    if n < 2:
        return {"n": n, "r": None}
    ac, bc = a - a.mean(), b - b.mean()
    den = ac.square().sum().sqrt() * bc.square().sum().sqrt()
    if not bool(torch.isfinite(den)) or float(den) <= 0.0:
        return {"n": n, "r": None}
    return {"n": n, "r": float((ac * bc).sum() / den)}


def gradient_cosine(
    a: Sequence[torch.Tensor | None],
    b: Sequence[torch.Tensor | None],
) -> dict[str, float | bool | None]:
    """Cosine and norms for two gradient tuples without flattening them."""
    if len(a) != len(b):
        raise ValueError("gradient tuples must have equal length")
    dot = torch.zeros((), dtype=torch.float64)
    aa = torch.zeros((), dtype=torch.float64)
    bb = torch.zeros((), dtype=torch.float64)
    for ga, gb in zip(a, b):
        if ga is not None:
            x = ga.detach().to(device="cpu", dtype=torch.float64)
            aa = aa + x.square().sum()
        if gb is not None:
            y = gb.detach().to(device="cpu", dtype=torch.float64)
            bb = bb + y.square().sum()
        if ga is not None and gb is not None:
            x = ga.detach().to(device="cpu", dtype=torch.float64)
            y = gb.detach().to(device="cpu", dtype=torch.float64)
            dot = dot + (x * y).sum()
    na, nb = aa.sqrt(), bb.sqrt()
    if float(na) == 0.0 or float(nb) == 0.0:
        cosine = None
    else:
        cosine = float(dot / (na * nb))
    return {
        "norm_a": float(na),
        "norm_b": float(nb),
        "dot": float(dot),
        "cosine": cosine,
        "conflict": bool(cosine is not None and cosine < 0.0),
    }
