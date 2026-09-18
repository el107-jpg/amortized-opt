"""Two-dimensional task-time frontier primitives.

These routines contain no actor or physics calls.  They define the atomic
two-hand timing semantics used by both shadow calibration and the later
exact-verified assist sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True, order=True)
class TimingPair:
    """An indivisible pair of left/right hit indices.

    ``-1`` denotes an inactive hand.  Active hits use the repository's
    one-based waypoint convention.
    """

    left: int
    right: int

    def validate(self, active: tuple[bool, bool], horizon: int) -> "TimingPair":
        for value, enabled, hand in zip(
            (self.left, self.right), active, ("left", "right"),
        ):
            if enabled and not 1 <= int(value) <= int(horizon):
                raise ValueError(
                    f"active {hand} hit {value} outside [1, {int(horizon)}]"
                )
            if not enabled and int(value) != -1:
                raise ValueError(f"inactive {hand} hit must be -1, got {value}")
        return self

    def as_tuple(self) -> tuple[int, int]:
        return int(self.left), int(self.right)


def sample_directions(
    active: torch.Tensor,
    *,
    generator: torch.Generator,
    axis_probability: float = 0.10,
) -> torch.Tensor:
    """Sample positive timing-compression directions for each scenario.

    For two active hands the mixture is left-axis, right-axis, and a random
    first-quadrant direction.  Each direction is infinity-norm normalized so
    ``alpha`` is measured in approximately one timing knot on its dominant
    hand.  A one-hand task is exactly one-dimensional.
    """
    if active.ndim != 2 or active.shape[1] != 2:
        raise ValueError("active must have shape (B, 2)")
    p = float(axis_probability)
    if not 0.0 <= p <= 0.5:
        raise ValueError("axis_probability must be in [0, 0.5]")
    b = int(active.shape[0])
    u = torch.rand(b, generator=generator)
    theta = (math.pi / 2.0) * torch.rand(b, generator=generator)
    random_d = torch.stack([theta.cos(), theta.sin()], dim=1)
    random_d = random_d / random_d.amax(dim=1, keepdim=True).clamp_min(1e-8)
    direction = torch.where(
        (u < p).unsqueeze(1), torch.tensor([1.0, 0.0]), random_d,
    )
    direction = torch.where(
        ((u >= p) & (u < 2.0 * p)).unsqueeze(1),
        torch.tensor([0.0, 1.0]), direction,
    )
    a = active.detach().cpu().bool()
    direction = direction * a.float()
    only_left = a[:, 0] & ~a[:, 1]
    only_right = ~a[:, 0] & a[:, 1]
    neither = ~a.any(dim=1)
    direction[only_left] = torch.tensor([1.0, 0.0])
    direction[only_right] = torch.tensor([0.0, 1.0])
    direction[neither] = 0.0
    return direction.to(device=active.device)


def directional_pairs(
    active: torch.Tensor,
    directions: torch.Tensor,
    *,
    horizon: int,
    step: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return anchor-to-aggressive integer timing pairs along each ray.

    Returns ``(pairs, alpha)`` with shapes ``(B, K, 2)`` and ``(K,)``.  The
    generous anchor is always column zero and the final column reaches the
    shortest representable active time on at least the dominant hand.
    """
    if active.ndim != 2 or active.shape[1] != 2 or directions.shape != active.shape:
        raise ValueError("active and directions must both have shape (B, 2)")
    h, stride = int(horizon), int(step)
    if h < 1 or stride < 1:
        raise ValueError("horizon and step must be positive")
    alpha = torch.arange(0, h, stride, device=active.device, dtype=torch.float32)
    if int(alpha[-1]) != h - 1:
        alpha = torch.cat([alpha, alpha.new_tensor([float(h - 1)])])
    raw = float(h) - alpha.view(1, -1, 1) * directions.float().unsqueeze(1)
    pairs = raw.round().long().clamp(1, h)
    pairs = torch.where(
        active.bool().unsqueeze(1), pairs, torch.full_like(pairs, -1),
    )
    return pairs, alpha


def first_exit_indices(
    manageable: torch.Tensor,
    *,
    collar: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locate the first failure connected to each ray's generous anchor.

    Returns ``(selected, last_pass, first_fail)``.  ``first_fail == K`` means
    the entire tested ray passed.  A failed anchor has no certified pass and
    therefore selects index zero for fallback handling.
    """
    if manageable.ndim != 2:
        raise ValueError("manageable must have shape (B, K)")
    b, k = manageable.shape
    m = manageable.bool()
    fail = ~m
    has_fail = fail.any(dim=1)
    first_fail = torch.where(
        has_fail, fail.float().argmax(dim=1),
        torch.full((b,), k, device=m.device, dtype=torch.long),
    )
    anchor_ok = m[:, 0]
    last_pass = (first_fail - 1).clamp(0, k - 1)
    last_pass = torch.where(anchor_ok, last_pass, torch.zeros_like(last_pass))
    selected = (last_pass + int(collar)).clamp(0, k - 1)
    selected = torch.where(anchor_ok, selected, torch.zeros_like(selected))
    return selected, last_pass, first_fail


def anchor_connected_component(
    feasible: torch.Tensor,
    anchor: tuple[int, int],
) -> torch.Tensor:
    """Four-neighbour feasible component containing ``anchor``."""
    if feasible.ndim != 2:
        raise ValueError("feasible grid must be two-dimensional")
    rows, cols = feasible.shape
    ai, aj = int(anchor[0]), int(anchor[1])
    if not (0 <= ai < rows and 0 <= aj < cols):
        raise ValueError("anchor index is outside the grid")
    src = feasible.detach().cpu().bool()
    out = torch.zeros_like(src)
    if not bool(src[ai, aj]):
        return out.to(feasible.device)
    stack = [(ai, aj)]
    out[ai, aj] = True
    while stack:
        i, j = stack.pop()
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < rows and 0 <= nj < cols and bool(src[ni, nj]) and not bool(out[ni, nj]):
                out[ni, nj] = True
                stack.append((ni, nj))
    return out.to(feasible.device)


def pareto_frontier_mask(feasible: torch.Tensor) -> torch.Tensor:
    """Mask feasible cells not dominated by a no-slower feasible pair.

    Grid indices are assumed to increase with time, so a cell is dominated if
    another feasible cell has both indices less than or equal and at least one
    strictly less.
    """
    if feasible.ndim != 2:
        raise ValueError("feasible grid must be two-dimensional")
    src = feasible.detach().cpu().bool()
    out = torch.zeros_like(src)
    points = src.nonzero(as_tuple=False)
    for point in points:
        i, j = int(point[0]), int(point[1])
        dominated = src[: i + 1, : j + 1].clone()
        dominated[i, j] = False
        if not bool(dominated.any()):
            out[i, j] = True
    return out.to(feasible.device)


def interval_boundary_loss(
    prediction: torch.Tensor,
    alpha_pass: torch.Tensor,
    alpha_fail: torch.Tensor,
) -> torch.Tensor:
    """Zero inside an exact pass/fail bracket, quadratic outside it."""
    if prediction.shape != alpha_pass.shape or prediction.shape != alpha_fail.shape:
        raise ValueError("boundary prediction and interval endpoints must match")
    if torch.any(alpha_pass > alpha_fail):
        raise ValueError("alpha_pass must not exceed alpha_fail")
    return (
        F.relu(alpha_pass - prediction).square()
        + F.relu(prediction - alpha_fail).square()
    ).mean()

