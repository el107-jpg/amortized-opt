"""Varied physical starting poses for assignment collection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


@torch.no_grad()
def pose_clearance(cost, q):
    """Existing analytic SDF, both arms, with no strike/departure exemptions.

    Check cross-stick clearance even when its training weight is zero.
    This is the existing collision proxy, not a mesh collision certificate.
    """
    B = len(q)
    from amortized_opt.cost import StrikeContext
    from amortized_opt.constants import STICK_TIP_SIGMA
    scalar, vector = q.new_zeros(B, 2), q.new_zeros(B, 2, 3)
    ctx = StrikeContext(hit_index=torch.zeros(B, 2, device=q.device, dtype=torch.long),
        active=scalar, target_drum=torch.full((B, 2), -1, device=q.device, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, device=q.device, dtype=torch.long),
        p_star=vector, v_star=vector, n_out=vector, hit_radius=scalar,
        disk_offset=vector, half_plane_dir=vector, has_half_plane=scalar,
        contact_sigma=q.new_full((B, 2), STICK_TIP_SIGMA), shaft_allowed=scalar,
        start_shaft_allowed=scalar, elev_required=scalar)
    diag = {}
    cost._collision(cost.fk.forward(q), B, 1, ctx,
                    {h: torch.ones(B, 1, device=q.device) for h in ("left", "right")},
                    {}, {}, {}, diag, {}, None)
    return torch.minimum(diag["collision_min_sdf"], diag["collision_stick_min_sdf"])


class StartingPoseSampler:
    """Mix exact ready poses, perturbed ready poses and executed pose anchors."""

    def __init__(self, anchors, ready, lower, upper, *, clearance,
                 names, ready_fraction=.2, anchor_fraction=.6,
                 arm_std=.08, yaw_std=.15, lean_std=.02, margin=.003):
        self.ready = np.asarray(ready, dtype=np.float32)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.anchors = np.asarray(anchors, dtype=np.float32)
        if self.anchors.ndim != 2 or self.anchors.shape[1:] != self.ready.shape:
            raise ValueError("starting pose anchors must match joint order")
        if not (0 <= ready_fraction <= 1 and 0 <= anchor_fraction <= 1-ready_fraction):
            raise ValueError("invalid starting pose mixture")
        self.ready_fraction, self.anchor_fraction = ready_fraction, anchor_fraction
        self.clearance, self.margin = clearance, margin
        self.std = np.array([yaw_std if n == "waist_yaw_joint" else
                             lean_std if n.startswith("waist_") else arm_std for n in names])
        self.lean = [i for i, n in enumerate(names) if n in ("waist_pitch_joint", "waist_roll_joint")]
        valid = self.valid(self.anchors)
        self.anchors = self.anchors[valid]
        if not len(self.anchors) or not self.valid(self.ready[None])[0]:
            raise ValueError("no collision-clear pose anchors or invalid nominal ready pose")

    def valid(self, poses):
        finite = np.isfinite(poses).all(1)
        joint_ok = ((poses >= self.lower) & (poses <= self.upper)).all(1)
        upright = (np.abs(poses[:, self.lean]) <= .08).all(1)
        result = finite & joint_ok & upright
        ids = np.flatnonzero(result)
        if len(ids):
            result[ids] &= np.asarray(self.clearance(poses[ids])) >= self.margin
        return result

    def sample(self, rng, size):
        mode = rng.random(size)
        ready_rows = mode < self.ready_fraction
        anchor_rows = mode >= 1-self.anchor_fraction
        poses = np.tile(self.ready, (size, 1))
        poses[anchor_rows] = self.anchors[rng.integers(len(self.anchors), size=int(anchor_rows.sum()))]
        base = poses.copy()
        accepted = ready_rows.copy()
        proposed = rejected = 0
        for _ in range(12):
            ids = np.flatnonzero(~accepted)
            if not len(ids):
                break
            noise = np.clip(rng.normal(size=(len(ids), len(self.ready))), -2.5, 2.5)*self.std
            candidate = base[ids]+noise
            valid = self.valid(candidate)
            poses[ids[valid]] = candidate[valid]
            accepted[ids[valid]] = True
            proposed += len(ids); rejected += int((~valid).sum())
        fallback = ~accepted
        poses[fallback] = base[fallback]  # Already checked before sampling.
        return poses.astype(np.float32), dict(
            ready_fraction=float(ready_rows.mean()), anchor_fraction=float(anchor_rows.mean()),
            fallback_fraction=float(fallback.mean()), rejection_fraction=rejected/max(1, proposed),
            yaw_std_rad=float(poses[:, 0].std()),
            arm_std_mean_rad=float(poses[:, 3:].std(0).mean()),
            max_lean_rad=float(np.abs(poses[:, self.lean]).max()))
