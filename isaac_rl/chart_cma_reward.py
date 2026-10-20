"""Bounded proximity credit for missed, scheduled drum hits."""
import math

import torch


def distance_to_hit_region(points, centers, normals, radii, half_plane_dirs):
    """Euclidean distance to a planar disk, optionally cut through its center.

    A zero half-plane vector means the full disk. Inputs use a common frame.
    Projection onto the half-plane cone followed by radial clipping is the
    exact projection onto this disk/half-disk intersection.
    """
    normal = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    offset = points-centers
    axial = (offset*normal).sum(-1, keepdim=True)
    planar = offset-axial*normal
    half = half_plane_dirs-half_plane_dirs.mul(normal).sum(-1, keepdim=True)*normal
    half = half / half.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    planar = planar+(-(planar*half).sum(-1, keepdim=True)).clamp_min(0)*half
    scale = (radii.unsqueeze(-1)/planar.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
    closest = centers+planar*scale
    return (points-closest).norm(dim=-1)


def proximity_credit(closest_distance, credited, valid, *, weight=.25, scale_m=.05):
    """Actual hits retain 1; only missed hits receive capped partial credit."""
    partial = weight*torch.exp(-closest_distance/scale_m)
    partial = torch.where(valid & ~credited, partial, torch.zeros_like(partial))
    quality = torch.where(valid & credited, torch.ones_like(partial), partial)
    return partial, quality


class NearMissTracker:
    def __init__(self, times, hands, valid, centers, normals, radii, half_plane_dirs,
                 *, window_s=.08, weight=.25, scale_m=.05):
        if (not all(math.isfinite(v) for v in (window_s, weight, scale_m))
                or not 0 <= weight < 1 or window_s <= 0 or scale_m <= 0):
            raise ValueError('Proximity weight must be in [0,1); scale/window must be positive')
        self.times, self.hands, self.valid = times, hands.long(), valid
        self.centers, self.normals, self.radii = centers, normals, radii
        self.half = half_plane_dirs
        self.window_s, self.weight, self.scale_m = window_s, weight, scale_m
        self.closest = torch.full_like(times, torch.inf)

    @torch.no_grad()
    def update(self, time_s, tips, live):
        # tips[B, hand, xyz]: only the hand assigned to each requested hit counts.
        selected = tips.gather(1, self.hands[..., None].expand(-1, -1, 3))
        distance = distance_to_hit_region(selected, self.centers, self.normals, self.radii, self.half)
        active = self.valid & live[:, None] & ((self.times-time_s).abs() <= self.window_s+1e-6)
        self.closest = torch.minimum(self.closest, torch.where(active, distance, torch.inf))

    def credits(self, credited):
        return proximity_credit(self.closest, credited, self.valid,
            weight=self.weight, scale_m=self.scale_m)
