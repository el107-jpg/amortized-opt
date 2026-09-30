"""Analytic velocity-matched strike seeds, shared by numpy and GPU callers.

Acceleration rises linearly from zero over the first 10% of the strike
window, then stays constant until contact. Arrival velocity is free. After
contact acceleration tapers to zero over the same interval, then the path
coasts; this tail is padding, not a contact/rebound model. Callers retain
their existing masks and joint-limit handling.
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS

ACCEL_RAMP_FRACTION = 0.1


def _integrated_acceleration(u, ramp_fraction):
    """Double integral of a unit acceleration plateau (numpy or torch)."""
    r = float(ramp_fraction)
    if not 0.0 < r <= 1.0:
        raise ValueError("ramp_fraction must be in (0, 1]")
    t = u.clip(0.0, 1.0)
    x = t.clip(0.0, r)
    y = (t - r).clip(0.0, None)
    before = x**3 / (6.0 * r) + 0.5 * r * y + 0.5 * y**2
    # Continuous position, velocity AND acceleration through the hit.
    z = (u - 1.0).clip(0.0, r)
    w = (u - 1.0 - r).clip(0.0, None)
    after = (1.0 - 0.5 * r) * z + 0.5 * z**2 - z**3 / (6.0 * r) + w
    return (before + after) / (0.5 - 0.5 * r + r * r / 6.0)


def accelerating_path_numpy(
    q0, v0, terminal, hit_steps: int, horizon: int, dt: float = CONTROL_DT,
    *, ramp_fraction: float = ACCEL_RAMP_FRACTION,
):
    """One arm, rows 0..horizon, matching the actual initial velocity.

    From rest the joint-space path is straight. An off-path initial velocity
    necessarily bends it. If a joint would overshoot at its initial velocity,
    braking can be unavoidable. No velocity/torque limits are promised.
    """
    if hit_steps < 1 or dt <= 0:
        raise ValueError("hit_steps and dt must be positive")
    q0, v0, terminal = (np.asarray(x, dtype=float) for x in (q0, v0, terminal))
    delta = terminal - q0
    V = v0 * (hit_steps * dt)
    u = np.arange(horizon + 1, dtype=float) / hit_steps
    ease = _integrated_acceleration(u, ramp_fraction)
    return q0 + u[:, None] * V + ease[:, None] * (delta - V)


def accelerating_path_torch(
    q0, v0, terminal, hit_steps, horizon: int, dt: float = CONTROL_DT,
    *, ramp_fraction: float = ACCEL_RAMP_FRACTION,
):
    """Batched one-arm equivalent, returning rows 1..horizon, (B,H,n)."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    delta = terminal - q0
    L = hit_steps.reshape(-1).to(q0).clamp_min(1)
    V = v0 * L[:, None] * float(dt)
    u = torch.arange(1, horizon + 1, device=q0.device, dtype=q0.dtype)[None, :] / L[:, None]
    ease = _integrated_acceleration(u, ramp_fraction)
    return (
        q0[:, None, :] + u[:, :, None] * V[:, None, :]
        + ease[:, :, None] * (delta - V)[:, None, :]
    )


def accelerating_terminal_velocity_numpy(
    q0, v0, terminal, hit_steps: int, dt: float = CONTROL_DT,
    *, ramp_fraction: float = ACCEL_RAMP_FRACTION,
):
    """Exact velocity at contact for :func:`accelerating_path_numpy`."""
    r = float(ramp_fraction)
    T = float(hit_steps) * float(dt)
    q0, v0, terminal = (np.asarray(x, dtype=float) for x in (q0, v0, terminal))
    den = 0.5 - 0.5 * r + r * r / 6.0
    ease_prime = (1.0 - 0.5 * r) / den
    return v0 + ease_prime * (terminal - q0 - v0 * T) / T


def accelerating_terminal_velocity_torch(
    q0, v0, terminal, hit_steps, dt: float = CONTROL_DT,
    *, ramp_fraction: float = ACCEL_RAMP_FRACTION,
):
    """Batched torch equivalent of :func:`accelerating_terminal_velocity_numpy`."""
    r = float(ramp_fraction)
    T = hit_steps.reshape(-1, 1).to(q0).clamp_min(1) * float(dt)
    den = 0.5 - 0.5 * r + r * r / 6.0
    ease_prime = (1.0 - 0.5 * r) / den
    return v0 + ease_prime * (terminal - q0 - v0 * T) / T


def accelerating_apex_path_numpy(
    q0, v0, apex, terminal, apex_steps: int, hit_steps: int, horizon: int,
    dt: float = CONTROL_DT, *, ramp_fraction: float = ACCEL_RAMP_FRACTION,
):
    """Two low-acceleration segments joined at ``apex``.

    The second segment starts with the exact arrival velocity of the first.
    Position and velocity therefore agree at the apex; each segment independently
    uses the same zero-start-acceleration/ramped-constant-acceleration law.
    """
    ka, kh = int(apex_steps), int(hit_steps)
    if not 1 <= ka < kh:
        raise ValueError(f"need 1 <= apex_steps < hit_steps, got {ka}, {kh}")
    first = accelerating_path_numpy(
        q0, v0, apex, ka, ka, dt, ramp_fraction=ramp_fraction,
    )
    va = accelerating_terminal_velocity_numpy(
        q0, v0, apex, ka, dt, ramp_fraction=ramp_fraction,
    )
    second = accelerating_path_numpy(
        apex, va, terminal, kh - ka, horizon - ka, dt,
        ramp_fraction=ramp_fraction,
    )
    return np.concatenate([first, second[1:]], axis=0)


def accelerating_apex_path_torch(
    q0, v0, apex, terminal, apex_steps, hit_steps, horizon: int,
    dt: float = CONTROL_DT, *, ramp_fraction: float = ACCEL_RAMP_FRACTION,
):
    """Batched, variable-timing version of :func:`accelerating_apex_path_numpy`.

    Returns rows 1..``horizon`` with shape ``(B,H,n)``.
    """
    ka = apex_steps.reshape(-1).to(q0).clamp_min(1)
    kh = hit_steps.reshape(-1).to(q0)
    k2 = (kh - ka).clamp_min(1)
    first = accelerating_path_torch(
        q0, v0, apex, ka, horizon, dt, ramp_fraction=ramp_fraction,
    )
    va = accelerating_terminal_velocity_torch(
        q0, v0, apex, ka, dt, ramp_fraction=ramp_fraction,
    )
    t = torch.arange(1, horizon + 1, device=q0.device, dtype=q0.dtype)[None, :]
    rel = (t - ka[:, None]).clamp_min(0.0)
    u = rel / k2[:, None]
    V = va * k2[:, None] * float(dt)
    ease = _integrated_acceleration(u, ramp_fraction)
    second = (
        apex[:, None, :] + u[:, :, None] * V[:, None, :]
        + ease[:, :, None] * (terminal - apex - V)[:, None, :]
    )
    return torch.where((t <= ka[:, None]).unsqueeze(-1), first, second)


def quintic_seed(q0, v0, terminal, lengths, horizon: int, dt: float = CONTROL_DT):
    """Legacy velocity-matched rest-at-target timing for waist/idle motion."""
    k = torch.arange(1, horizon + 1, device=q0.device, dtype=q0.dtype).view(1, horizon, 1)
    L = lengths.clamp_min(1).to(q0)
    u = (k / L.unsqueeze(1)).clamp(0.0, 1.0)
    d = (terminal - q0).unsqueeze(1)
    V = (v0 * L * float(dt)).unsqueeze(1)
    u3 = u * u * u
    return (q0.unsqueeze(1) + V * u + (10.0 * d - 6.0 * V) * u3
            + (8.0 * V - 15.0 * d) * u3 * u
            + (6.0 * d - 3.0 * V) * u3 * u * u)


def strike_seed(
    q0, v0, terminal, lengths, horizon: int, dt: float = CONTROL_DT,
    *, active, joint_names=G1_UPPER_BODY_JOINTS,
):
    """Full-body seed: accelerating active arms, settling waist/idle arms.

    Each active arm must have a common strike length across its joints.
    """
    out = quintic_seed(q0, v0, terminal, lengths, horizon, dt)
    for h, hand in enumerate(("left", "right")):
        cols = [i for i, name in enumerate(joint_names) if name.startswith(hand + "_")]
        if not cols:
            continue
        sub = accelerating_path_torch(
            q0[:, cols], v0[:, cols], terminal[:, cols], lengths[:, cols[0]], horizon, dt,
        )
        out[:, :, cols] = torch.where(active[:, h, None, None].bool(), sub, out[:, :, cols])
    return out
