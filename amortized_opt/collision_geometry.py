"""Shared arm proxies and conservative clearance between motion samples."""
from __future__ import annotations

import torch

from .constants import FOREARM_RADIUS, UPPER_ARM_RADIUS


def arm_capsules(frames, batch, samples, hand):
    """Shoulder/elbow/wrist frame origins; same proxies in pose and trajectory."""
    points = {name: frames[f"{hand}_{name}"][1].reshape(batch, samples, 3)
              for name in ("shoulder", "elbow", "wrist")}
    return (("upper", points["shoulder"], points["elbow"], UPPER_ARM_RADIUS),
            ("forearm", points["elbow"], points["wrist"], FOREARM_RADIUS))


def temporal_clearance_bound(distance, mask, speed_bound, dt):
    """Lower bound on clearance over each preceding sample interval.

    ``speed_bound[:, i]`` bounds relative surface speed throughout interval
    [i-1,i], including rotation. Signed convex distance is Lipschitz in surface
    motion. Either endpoint is at most dt/2 from the closest sampled endpoint,
    so min(endpoint clearances)-speed*dt/2 is a conservative interval bound.
    Only intervals with BOTH endpoints scored are tightened: contact exemption
    boundaries do not inherit an unqualified earlier contact.

    This may conservatively price an unresolved interval; it is not a measured
    penetration or a claim of exact PhysX continuous collision detection.
    """
    while speed_bound.ndim < distance.ndim:
        speed_bound = speed_bound.unsqueeze(-1)
    interval = torch.minimum(distance[:, 1:], distance[:, :-1])-speed_bound[:, 1:]*(dt/2)
    valid = (mask[:, 1:] > .5) & (mask[:, :-1] > .5)
    return torch.cat((distance[:, :1], torch.where(valid, interval, distance[:, 1:])), dim=1)
