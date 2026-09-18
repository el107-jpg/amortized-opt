"""Differentiable analytic signed-distance penalty over the drum scene.

Collision geometry matches :class:`~amortized_opt.cost.FusedCost` and
:class:`~amortized_opt.eval_harness.TrajectoryScorer`: each hand is a stick
capsule (butt→tip) plus a wrist sphere, scored against every drum body /
under-volume and against the torso capsule / head sphere.  A target-drum
exemption covers the stick capsule (a hit is contact); the wrist is never
exempted.

The original repo class was a tip/shaft/wrist **point cloud**.  This module
kept that form for bit-match tests; those tests now skip.  Callers -- start
sampling, MPC screening, idle-home, the Isaac safety filter -- share this
capsule path so they cannot accept a pose the cost would reject for a shaft
through a pad.

``forward(..., frames=...)`` accepts a frame dict from a shared
:class:`~amortized_opt.fk.ChainFK` pass.  Passing ``q`` alone keeps the
internal-FK convenience path.

The layout is loaded from ``amortized/assets/drum_kit_layout.yaml`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    HEAD_SPHERE,
    PELVIS_IN_ENV,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
    WRIST_RADIUS,
    default_layout_path,
)
from .fk import ChainFK, stick_shaft_offsets, stick_tip_offset

_EPS = 1.0e-9
#: Floor for 1-D parameter solves (s along a segment).  ``_EPS**2`` (1e-18)
#: makes ``s = q / den`` overflow float32 grads on a degenerate (zero-length
#: or edge-on) segment and those NaNs show up in ``FusedCost`` on pads, whose
#: playing shaft collapses to a point.
_DEN = 1.0e-8
#: Floor under ``sqrt(discriminant)`` in :func:`_quad_roots_01`.  A double root
#: (stick tangent to a drum wall or the legal hit rim) has ``Δ = 0``; ``sqrt``
#: is finite there but ``d(sqrt)/dΔ`` is inf, and those NaNs reach ``FusedCost``.
_DISC_EPS = 1.0e-12


def _unit_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


@dataclass
class DrumSceneSpec:
    """Drum-kit obstacle description.

    ``head_centers`` are the striking surfaces (layout ``position``); the solid
    drum hangs ``height`` below along ``-axis``. Each drum also gets an
    under-drum no-go cylinder of depth ``under_depth`` hung ``under_clearance``
    below the head volume.

    ``frame``: "env" (layout coordinates) or "pelvis" (layout - PELVIS_IN_ENV).
    """

    names: list[str]
    head_centers: np.ndarray  # (D, 3)
    axes: np.ndarray  # (D, 3) unit head normals
    radii: np.ndarray  # (D,)
    heights: np.ndarray  # (D,)
    frame: str = "env"
    under_depth: float = 0.08
    under_clearance: float = 0.01

    @classmethod
    def from_layout(
        cls,
        layout_path: str | Path | None = None,
        *,
        frame: str = "env",
        under_depth: float = 0.08,
        under_clearance: float = 0.01,
    ) -> "DrumSceneSpec":
        import yaml

        path = Path(layout_path) if layout_path is not None else default_layout_path()
        with open(path) as f:
            layout = yaml.safe_load(f)
        # Order by the layout's drum `id` so row i == drum id i (matches the
        # target_drum_* exemption indices).
        items = list(layout["drums"].items())
        if all("id" in spec for _n, spec in items):
            items.sort(key=lambda kv: int(kv[1]["id"]))
        names, centers, axes, radii, heights = [], [], [], [], []
        for name, spec in items:
            names.append(str(name))
            centers.append(np.asarray(spec["position"], dtype=float))
            axes.append(_unit_np(np.asarray(spec.get("normal", [0.0, 0.0, 1.0]), dtype=float)))
            radii.append(float(spec["radius"]))
            heights.append(float(spec["height"]))
        head = np.stack(centers, axis=0)
        if frame == "pelvis":
            head = head - PELVIS_IN_ENV[None, :]
        elif frame != "env":
            raise ValueError(f"frame must be 'env' or 'pelvis', got {frame!r}")
        return cls(
            names=names,
            head_centers=head,
            axes=np.stack(axes, axis=0),
            radii=np.asarray(radii, dtype=float),
            heights=np.asarray(heights, dtype=float),
            frame=frame,
            under_depth=float(under_depth),
            under_clearance=float(under_clearance),
        )

    @property
    def num_drums(self) -> int:
        return len(self.names)


def point_cylinder_sdf(
    x: torch.Tensor,
    center: torch.Tensor,
    axis: torch.Tensor,
    radius: torch.Tensor,
    half_height: torch.Tensor,
) -> torch.Tensor:
    """Signed distance of points to solid capped cylinders (differentiable a.e.)."""
    u = x - center
    z = (u * axis).sum(dim=-1)
    perp = u - z.unsqueeze(-1) * axis
    rho = torch.sqrt((perp * perp).sum(dim=-1) + _EPS**2)
    d_rho = rho - radius
    d_z = z.abs() - half_height
    out = torch.sqrt(d_rho.clamp_min(0.0) ** 2 + d_z.clamp_min(0.0) ** 2 + _EPS**2) - _EPS
    inside = torch.maximum(d_rho, d_z).clamp_max(0.0)
    return out + inside


def point_capsule_sdf(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, radius: float
) -> torch.Tensor:
    ab = b - a
    denom = (ab * ab).sum(dim=-1).clamp_min(_EPS**2)
    t = (((x - a) * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    d = x - closest
    return torch.sqrt((d * d).sum(dim=-1) + _EPS**2) - radius


def point_sphere_sdf(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    d = x - center
    return torch.sqrt((d * d).sum(dim=-1) + _EPS**2) - radius


def segment_segment_sdf(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    ra: float | torch.Tensor = 0.0,
    rb: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Capsule-capsule signed distance between segments ``[a0,a1]``/``[b0,b1]``.

    ``|closest(A, B)| - ra - rb``, i.e. the gap between the two capsule
    SURFACES: negative means the capsules interpenetrate.  Every input
    broadcasts on the leading dims and carries the 3-vector in the last one, so
    a ``(B, S, 3)`` pair of sticks costs one call.

    The math is the textbook clamped closest-point-between-segments (Ericson,
    *Real-Time Collision Detection* §5.1.9) with its endpoint fix-up written as
    two plain coordinate-descent steps, which is what makes it branch-free.
    With ``d1 = a1 - a0``, ``d2 = b1 - b0``, ``r = a0 - b0``::

        A = d1.d1   E = d2.d2   B = d1.d2   C = d1.r   F = d2.r
        den = A*E - B*B                        (>= 0, zero iff parallel)
        s = clamp((B*F - C*E) / den, 0, 1)      [den > eps; else s = 0]
        t = clamp((B*s + F) / E, 0, 1)          best t for that s
        s = clamp((B*t - C) / A, 0, 1)          best s for that t
        d = |(a0 + s*d1) - (b0 + t*d2)| - ra - rb

    The two clamped re-solves ARE Ericson's ``t<0 -> s = clamp(-C/A)`` /
    ``t>1 -> s = clamp((B-C)/A)`` branches -- substitute ``t = 0`` / ``t = 1``
    into the third line and they fall out -- and at an interior optimum each is
    a no-op, because ``s`` already minimizes the distance for that ``t``.
    Running the third line unconditionally is what makes the DEGENERATE cases
    right as well: when ``b`` collapses to a point, ``den`` and ``t`` are both
    zero and it is the only line that finds the foot of the perpendicular on
    ``a``.  Dropping it (the literal transcription of the branchy original)
    returns ``|a0 - b0|`` instead, and the function is then not even symmetric
    in its two arguments.

    DEGENERACIES.  ``den``, ``A`` and ``E`` are clamped away from zero *before*
    any division, so parallel segments (``den -> 0``) and zero-length ones
    (``A`` or ``E`` ``-> 0``) take the clamped path rather than dividing by
    zero -- and the unselected branch of the one :func:`torch.where` is finite
    too, which is what keeps the BACKWARD pass NaN-free (``where`` propagates a
    NaN gradient from the branch it did not take).  A zero-length segment is
    therefore a legal sphere: pass ``a0 == a1`` to price a wrist ball against a
    stick.  The final norm is ``sqrt(|d|^2 + eps^2)``, so touching axes (exact
    zero distance) stay differentiable, exactly as :func:`point_capsule_sdf`
    does it.
    """
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    A = (d1 * d1).sum(dim=-1)
    E = (d2 * d2).sum(dim=-1)
    B = (d1 * d2).sum(dim=-1)
    C = (d1 * r).sum(dim=-1)
    F = (d2 * r).sum(dim=-1)
    A_s = A.clamp_min(_EPS**2)
    E_s = E.clamp_min(_EPS**2)
    den = A * E - B * B
    den_s = den.clamp_min(_EPS**2)
    # Interior solution; parallel (den ~ 0) falls back to s = 0 and lets the
    # t-clamp below place the pair, which is the standard degenerate branch.
    s_int = ((B * F - C * E) / den_s).clamp(0.0, 1.0)
    s = torch.where(den > _EPS, s_int, torch.zeros_like(s_int))
    t = ((B * s + F) / E_s).clamp(0.0, 1.0)
    s = ((B * t - C) / A_s).clamp(0.0, 1.0)
    diff = (a0 + s.unsqueeze(-1) * d1) - (b0 + t.unsqueeze(-1) * d2)
    dist = torch.sqrt((diff * diff).sum(dim=-1) + _EPS**2)
    return dist - ra - rb


def _match_lead(x, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast ``x`` to the leading shape of a 3-vector ``ref`` ``(..., 3)``."""
    lead = ref.shape[:-1]
    if not torch.is_tensor(x):
        # ``new_tensor`` from a Python float is a CUDA host sync.
        x = torch.full(
            (),
            float(x) if np.ndim(x) == 0 else float(np.asarray(x, dtype=float).reshape(-1)[0]),
            device=ref.device,
            dtype=ref.dtype,
        )
    else:
        x = x.to(device=ref.device, dtype=ref.dtype)
    while x.ndim < len(lead):
        x = x.unsqueeze(0)
    return x.expand(*lead) if x.shape != lead else x


def hit_region_dist(
    p: torch.Tensor,
    p_star: torch.Tensor,
    n_out: torch.Tensor,
    hit_R: torch.Tensor | float,
    disk_offset: torch.Tensor,
    half_plane_dir: torch.Tensor,
    has_half_plane: torch.Tensor | float,
    axial_lo: torch.Tensor | float,
    axial_hi: torch.Tensor | float,
    *,
    margin: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hypot distance of points ``p`` to the legal hit cylinder.

    ``dist = hypot(r_disk, r_hp, r_ax)`` with the same residuals
    :meth:`~amortized_opt.cost.FusedCost._terminal` and
    :meth:`~amortized_opt.eval_harness.TrajectoryScorer.score` share.
    ``dist == 0`` iff the point is in the (optionally expanded) region.
    Returns ``(dist, axial, d_in)``.
    """
    p, p_star, n_out, disk_offset, half_plane_dir = torch.broadcast_tensors(
        p, p_star, n_out, disk_offset, half_plane_dir
    )
    hit_R = _match_lead(hit_R, p)
    has_hp = _match_lead(has_half_plane, p)
    lo = _match_lead(axial_lo, p)
    hi = _match_lead(axial_hi, p)
    m = _match_lead(margin, p)
    e = p - p_star
    axial = (e * n_out).sum(dim=-1)
    d_in = e - axial.unsqueeze(-1) * n_out - disk_offset
    rho = torch.sqrt((d_in * d_in).sum(dim=-1) + _EPS**2)
    r_disk = (rho - (hit_R + m)).clamp_min(0.0)
    hp = (d_in * half_plane_dir).sum(dim=-1)
    r_hp = (-hp - m).clamp_min(0.0) * has_hp
    r_ax = (axial - (hi + m)).clamp_min(0.0) + ((lo - m) - axial).clamp_min(0.0)
    dist = torch.sqrt(r_disk * r_disk + r_hp * r_hp + r_ax * r_ax + _EPS**2) - _EPS
    return dist, axial, d_in


def _quad_roots_01(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``A s^2 + B s + C = 0`` on ``[0, 1]``.  Invalid roots are 0 with valid=0.

    Unused branches are evaluated on dummy well-conditioned coefficients so a
    degenerate (A=B=0) segment cannot inject 0/0 NaNs into the backward pass.
    The dummy is ``s^2 - 1 = 0`` (``Δ = 4``), not ``s^2 = 0``: ``torch.where``
    still backprops through the unselected branch, and ``sqrt(0)`` is inf there.
    Live discriminants are floored at ``_DISC_EPS`` for the same reason -- a
    grazing (double) root is a legal geometry, not a dummy.
    """
    quad = (B * B - 4.0 * A * C >= 0.0) & (A.abs() > _EPS)
    lin = (A.abs() <= _EPS) & (B.abs() > _EPS)
    A_q = torch.where(quad, A, torch.ones_like(A))
    B_q = torch.where(quad, B, torch.zeros_like(B))
    C_q = torch.where(quad, C, -torch.ones_like(C))
    disc = (B_q * B_q - 4.0 * A_q * C_q).clamp_min(_DISC_EPS)
    den = (2.0 * A_q).abs().clamp_min(_DEN) * torch.where(
        A_q >= 0.0, torch.ones_like(A_q), -torch.ones_like(A_q)
    )
    sqrt_d = torch.sqrt(disc)
    s_p = (-B_q + sqrt_d) / den
    s_m = (-B_q - sqrt_d) / den
    B_l = torch.where(lin, B, torch.ones_like(B))
    C_l = torch.where(lin, C, torch.zeros_like(C))
    s_lin = -C_l / B_l
    s_p = torch.where(lin, s_lin, s_p).clamp(-10.0, 10.0)
    s_m = torch.where(lin, s_lin, s_m).clamp(-10.0, 10.0)
    v_p = ((quad & (s_p >= 0.0) & (s_p <= 1.0)) | (lin & (s_p >= 0.0) & (s_p <= 1.0))).to(A.dtype)
    v_m = (quad & (s_m >= 0.0) & (s_m <= 1.0) & ~lin).to(A.dtype)
    s_p = torch.where(v_p > 0.5, s_p, torch.zeros_like(s_p))
    s_m = torch.where(v_m > 0.5, s_m, torch.zeros_like(s_m))
    return s_p, s_m, v_p, v_m


def _interior_01(
    numer: torch.Tensor, denom: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = -numer/denom`` on ``(0, 1)`` (open), invalid -> ``(0, 0)``."""
    live = denom.abs() > _EPS
    den = torch.where(live, denom, torch.ones_like(denom))
    num = torch.where(live, numer, torch.zeros_like(numer))
    s = (-num / den).clamp(-10.0, 10.0)
    valid = (live & (s > 0.0) & (s < 1.0)).to(numer.dtype)
    s = torch.where(valid > 0.5, s, torch.zeros_like(s))
    return s, valid


def segment_hit_region(
    a: torch.Tensor,
    b: torch.Tensor,
    p_star: torch.Tensor,
    n_out: torch.Tensor,
    hit_R: torch.Tensor | float,
    disk_offset: torch.Tensor,
    half_plane_dir: torch.Tensor,
    has_half_plane: torch.Tensor | float,
    axial_lo: torch.Tensor | float,
    axial_hi: torch.Tensor | float,
    *,
    margin: torch.Tensor | float = 0.0,
    return_candidates: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Exact min of :func:`hit_region_dist` along the segment ``[a, b]``.

    ``dist(s)`` is convex.  The min is the best of O(1) critical points:
    endpoints, closest in-plane to the disk centre, ``||d_in|| = R`` (0-2
    roots), the half-plane crossing, and the two axial-slab faces.  Grip /
    stick radius are NOT folded in; hit membership is the same hypot metric
    as a single point, minimized over the playing-shaft continuum.

    Returns ``(dist, s, p_contact)`` with ``p_contact = a + s (b - a)``,
    ``s`` in ``[0, 1]``.  Every input broadcasts on the leading dims.

    With ``return_candidates=True`` also returns ``(cand_s, cand_dist,
    cand_valid)`` each ``(..., C)`` so a training head can softmin over the
    same points the hard min uses.
    """
    a, b, p_star, n_out, disk_offset, half_plane_dir = torch.broadcast_tensors(
        a, b, p_star, n_out, disk_offset, half_plane_dir
    )
    hit_R_t = _match_lead(hit_R, a)
    has_hp = _match_lead(has_half_plane, a)
    lo = _match_lead(axial_lo, a)
    hi = _match_lead(axial_hi, a)
    mrg = _match_lead(margin, a)
    d = b - a
    e0 = a - p_star
    ax0 = (e0 * n_out).sum(dim=-1)
    ax1 = (d * n_out).sum(dim=-1)
    q0 = e0 - ax0.unsqueeze(-1) * n_out - disk_offset
    q1 = d - ax1.unsqueeze(-1) * n_out
    q00 = (q0 * q0).sum(dim=-1)
    q01 = (q0 * q1).sum(dim=-1)
    q11 = (q1 * q1).sum(dim=-1)
    R_eff = hit_R_t + mrg

    z = torch.zeros_like(ax0)
    o = torch.ones_like(ax0)
    # Closest in-plane is a clamped endpoint projection: always on the segment.
    live = q11 > 1.0e-6
    q11_s = torch.where(live, q11, torch.ones_like(q11))
    q01_s = torch.where(live, q01, torch.zeros_like(q01))
    s_rho_cl = torch.where(
        live, (-q01_s / (q11_s + _DEN)).clamp(0.0, 1.0), torch.zeros_like(q01)
    )
    s_circ_p, s_circ_m, v_circ_p, v_circ_m = _quad_roots_01(
        q11, 2.0 * q01, q00 - R_eff * R_eff
    )
    hp0 = (q0 * half_plane_dir).sum(dim=-1)
    hp1 = (q1 * half_plane_dir).sum(dim=-1)
    # Expanded half-plane boundary is ``hp = -margin``.
    s_hp, v_hp = _interior_01(hp0 + mrg, hp1)
    v_hp = v_hp * has_hp
    s_lo, v_lo = _interior_01(ax0 - (lo - mrg), ax1)
    s_hi, v_hi = _interior_01(ax0 - (hi + mrg), ax1)

    cand_s = torch.stack(
        [z, o, s_rho_cl, s_circ_p, s_circ_m, s_hp, s_lo, s_hi], dim=-1
    )
    cand_valid = torch.stack(
        [o, o, o, v_circ_p, v_circ_m, v_hp, v_lo, v_hi], dim=-1
    )
    p = a.unsqueeze(-2) + cand_s.unsqueeze(-1) * d.unsqueeze(-2)
    cand_dist, _, _ = hit_region_dist(
        p,
        p_star.unsqueeze(-2),
        n_out.unsqueeze(-2),
        hit_R_t.unsqueeze(-1),
        disk_offset.unsqueeze(-2),
        half_plane_dir.unsqueeze(-2),
        has_hp.unsqueeze(-1),
        lo.unsqueeze(-1),
        hi.unsqueeze(-1),
        margin=mrg.unsqueeze(-1),
    )
    big = 1.0e3
    scored = cand_dist + big * (1.0 - cand_valid)
    dist = scored.amin(dim=-1)
    idx = scored.argmin(dim=-1)
    s = cand_s.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    p_contact = a + s.unsqueeze(-1) * d
    if return_candidates:
        return dist, s, p_contact, cand_s, cand_dist, cand_valid
    return dist, s, p_contact


def segment_cylinder_sdf(
    a0: torch.Tensor,
    a1: torch.Tensor,
    center: torch.Tensor,
    axis: torch.Tensor,
    radius: torch.Tensor | float,
    half_height: torch.Tensor | float,
    ra: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Signed distance from capsule ``[a0, a1]`` (radius ``ra``) to a capped cylinder.

    Min of :func:`point_cylinder_sdf` along the segment, minus ``ra``.  Critical
    points: endpoints, closest approach to the axis, the two cap planes, and
    ``rho = R`` on the side wall.  Denominators are clamped like
    :func:`segment_segment_sdf`.
    """
    a0, a1, center, axis = torch.broadcast_tensors(a0, a1, center, axis)
    radius = _match_lead(radius, a0)
    half_height = _match_lead(half_height, a0)
    d = a1 - a0
    u0 = a0 - center
    z0 = (u0 * axis).sum(dim=-1)
    z1 = (d * axis).sum(dim=-1)
    q0 = u0 - z0.unsqueeze(-1) * axis
    q1 = d - z1.unsqueeze(-1) * axis
    q00 = (q0 * q0).sum(dim=-1)
    q01 = (q0 * q1).sum(dim=-1)
    q11 = (q1 * q1).sum(dim=-1)
    live = q11 > 1.0e-6
    q11_s = torch.where(live, q11, torch.ones_like(q11))
    q01_s = torch.where(live, q01, torch.zeros_like(q01))
    s_rho = torch.where(
        live, (-q01_s / (q11_s + _DEN)).clamp(0.0, 1.0), torch.zeros_like(q01)
    )
    z = torch.zeros_like(z0)
    o = torch.ones_like(z0)
    s_cap_p, v_cap_p = _interior_01(z0 - half_height, z1)
    s_cap_m, v_cap_m = _interior_01(z0 + half_height, z1)
    s_side_p, s_side_m, v_side_p, v_side_m = _quad_roots_01(
        q11, 2.0 * q01, q00 - radius * radius
    )
    cand_s = torch.stack(
        [z, o, s_rho, s_cap_p, s_cap_m, s_side_p, s_side_m], dim=-1
    )
    cand_valid = torch.stack(
        [o, o, o, v_cap_p, v_cap_m, v_side_p, v_side_m], dim=-1
    )
    p = a0.unsqueeze(-2) + cand_s.unsqueeze(-1) * d.unsqueeze(-2)
    sdf = point_cylinder_sdf(
        p,
        center.unsqueeze(-2),
        axis.unsqueeze(-2),
        radius.unsqueeze(-1),
        half_height.unsqueeze(-1),
    )
    big = 1.0e3
    scored = sdf + big * (1.0 - cand_valid)
    return scored.amin(dim=-1) - ra


@dataclass
class SdfPenaltyDetails:
    """Diagnostics from one :class:`SdfPenalty` call (all detached)."""

    min_sdf: torch.Tensor  # (B,)
    pen_drum: torch.Tensor  # (B,)
    pen_under: torch.Tensor  # (B,)
    pen_self: torch.Tensor  # (B,)
    extras: dict = field(default_factory=dict)


class SdfPenalty:
    """hinge(margin - SDF)^p summed over collision pairs, batched + differentiable.

    Stick capsule + wrist sphere, same pairs as :class:`~amortized_opt.cost.FusedCost`.
    """

    def __init__(
        self,
        scene: DrumSceneSpec | None = None,
        *,
        fk: ChainFK | None = None,
        urdf_path: str | Path | None = None,
        margin: float = 0.01,
        power: int = 2,
        clamp_depth: float | None = None,
        shaft_samples: int = 4,
        include_self: bool = True,
        w_drum: float = 1.0,
        w_under: float = 1.0,
        w_self: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.scene = scene if scene is not None else DrumSceneSpec.from_layout()
        scene = self.scene
        self.margin = float(margin)
        self.power = int(power)
        self.clamp_depth = None if clamp_depth is None else float(clamp_depth)
        self.include_self = bool(include_self)
        self.w_drum = float(w_drum)
        self.w_under = float(w_under)
        self.w_self = float(w_self)
        self.device = torch.device(device)
        self.dtype = dtype
        self.fk = fk if fk is not None else ChainFK(urdf_path, device=self.device, dtype=dtype)

        D = scene.num_drums
        t = lambda arr: torch.tensor(
            np.asarray(arr, dtype=float).tolist(), device=self.device, dtype=dtype
        )
        self.head_centers = t(scene.head_centers)  # (D, 3)
        self.axes = t(scene.axes)  # (D, 3)
        self.radii = t(scene.radii)  # (D,)
        self.half_heights = t(scene.heights / 2.0)  # (D,)
        self.under_half = float(scene.under_depth) / 2.0
        self.under_half_t = t(np.full(D, float(scene.under_depth) / 2.0))  # (D,)
        self.under_clearance = float(scene.under_clearance)
        self.num_drums = D
        off = PELVIS_IN_ENV if scene.frame == "env" else np.zeros(3)
        self.frame_offset = t(off)  # (3,)

        # Stick endpoints, wrist-local: [tip, shaft*S, wrist].  Row 0 is the
        # bead and row 1 is the first shaft sample, which
        # ``stick_shaft_offsets`` places at the butt, so [row 1 -> row 0] is
        # the whole stick capsule.  The wrist is a separate sphere.
        pts = {}
        for hand in ("left", "right"):
            local = np.concatenate(
                [
                    stick_tip_offset(hand)[None, :],
                    stick_shaft_offsets(hand, shaft_samples),
                    np.zeros((1, 3)),
                ],
                axis=0,
            )
            pts[hand] = t(local)  # (P, 3)
        self.local_points = pts
        self.n_points = int(pts["left"].shape[0])
        self.tip_index = 0
        self.stick_r = float(max(TIP_RADIUS, STICK_RADIUS))
        self.wrist_r = float(WRIST_RADIUS)
        a, b, r = TORSO_CAPSULE
        self.torso_a = t(a)
        self.torso_b = t(b)
        self.torso_r = float(r)
        hc, hr = HEAD_SPHERE
        self.head_c = t(hc)
        self.head_r = float(hr)

    # ------------------------------------------------------------------ core --
    def _cylinder_centers(self, head_centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ax = self.axes.to(head_centers.dtype)
        hh = self.half_heights.to(head_centers.dtype)
        body = head_centers - hh.unsqueeze(-1) * ax
        under_top = head_centers - (2.0 * hh + self.under_clearance).unsqueeze(-1) * ax
        under = under_top - self.under_half * ax
        return body, under

    def _hinge(self, sdf: torch.Tensor) -> torch.Tensor:
        h = (self.margin - sdf).clamp_min(0.0)
        if self.clamp_depth is not None:
            h = h.clamp_max(self.clamp_depth)
        return h if self.power == 1 else h.pow(self.power)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        q: torch.Tensor | None = None,
        *,
        frames: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        drum_head_centers: torch.Tensor | None = None,  # (B, D, 3), scene frame
        drum_radii: torch.Tensor | None = None,  # (B, D), per-sample head radii
        target_drum_left: torch.Tensor | None = None,  # (B,) long, -1 = none
        target_drum_right: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SdfPenaltyDetails]:
        """Per-row penalty (B,). Differentiable w.r.t. ``q`` / ``frames``.

        Pass ``frames`` (the output of :meth:`ChainFK.forward`) to reuse a shared
        FK pass; otherwise ``q`` (B, 17) is run through the internal FK.

        ``drum_head_centers`` / ``drum_radii`` override the scene's baked-in
        geometry per sample.  A screen run against a domain-randomized kit must
        pass BOTH whenever the kit randomizes both, or it accepts poses that sit
        inside a drum that is really there (see
        :attr:`~amortized_opt.cost.StrikeContext.drum_head_centers`).
        """
        if frames is None:
            if q is None:
                raise ValueError("pass either q or frames")
            if q.dim() != 2:
                raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
            frames = self.fk.forward(q)
        ref = frames["left_wrist"][1]  # (B, 3): dtype/device/batch reference
        dtype, dev = ref.dtype, ref.device
        B = ref.shape[0]
        off = self.frame_offset.to(dtype)
        D = self.num_drums

        heads = self.head_centers.to(dtype) if drum_head_centers is None else drum_head_centers
        body_c, under_c = self._cylinder_centers(heads)
        if body_c.dim() == 2:
            body_c = body_c.unsqueeze(0)
            under_c = under_c.unsqueeze(0)
        ax = self.axes.to(dtype).view(1, D, 3)
        half_h = self.half_heights.to(dtype).view(1, D)
        under_half = self.under_half_t.to(dtype).view(1, D)
        rad = (
            self.radii.to(dtype).view(1, D) if drum_radii is None
            else drum_radii.to(dtype)
        )

        pen_drum = ref.new_zeros(B)
        pen_under = ref.new_zeros(B)
        pen_self = ref.new_zeros(B)
        min_sdf = ref.new_full((B,), float("inf"))

        targets = {"left": target_drum_left, "right": target_drum_right}
        #: Per-hand clearance, so a caller can ask "how close is the IDLE arm"
        #: without re-running the pass with the other hand removed.  Only filled
        #: when ``return_details``; the combined ``min_sdf`` is unchanged.
        per_hand: dict[str, torch.Tensor] = {}
        for hand, frame_name in (("left", "left_wrist"), ("right", "right_wrist")):
            R, p = frames[frame_name]
            local = self.local_points[hand]
            pts = ChainFK.points_world(R, p, local.to(dtype)) + off
            tip, butt, wrist = pts[:, 0], pts[:, 1], pts[:, -1]

            cap_b = segment_cylinder_sdf(
                butt.unsqueeze(1), tip.unsqueeze(1),
                body_c, ax, rad, half_h, ra=self.stick_r,
            )
            cap_u = segment_cylinder_sdf(
                butt.unsqueeze(1), tip.unsqueeze(1),
                under_c, ax, rad, under_half, ra=self.stick_r,
            )
            wr = wrist.unsqueeze(1)
            sdf_wb = point_cylinder_sdf(wr, body_c, ax, rad, half_h) - self.wrist_r
            sdf_wu = point_cylinder_sdf(wr, under_c, ax, rad, under_half) - self.wrist_r

            # Stick capsule vs the TARGET drum is exempted: a hit is contact.
            # The wrist sphere is never exempted.  Branch-free one-hot.
            keep_cap = ref.new_ones(B, D)
            tgt = targets[hand]
            if tgt is not None:
                tgt = tgt.to(device=dev, dtype=torch.long).clamp(-1, D - 1)
                onehot = F.one_hot(tgt.clamp(min=0), D).to(dtype) * (tgt >= 0).to(dtype).unsqueeze(-1)
                keep_cap = keep_cap - onehot
            hinge_d = self._hinge(cap_b) * keep_cap + self._hinge(sdf_wb)
            hinge_u = self._hinge(cap_u) * keep_cap + self._hinge(sdf_wu)
            pen_drum = pen_drum + hinge_d.sum(dim=1)
            pen_under = pen_under + hinge_u.sum(dim=1)

            cap_b_m = cap_b.masked_fill(keep_cap <= 0.5, float("inf"))
            cap_u_m = cap_u.masked_fill(keep_cap <= 0.5, float("inf"))
            hand_min = torch.minimum(
                torch.minimum(cap_b_m.amin(dim=1), cap_u_m.amin(dim=1)),
                torch.minimum(sdf_wb.amin(dim=1), sdf_wu.amin(dim=1)),
            )
            min_sdf = torch.minimum(min_sdf, hand_min)

            if self.include_self:
                Rt, pt = frames["torso"]
                a = pt + (Rt @ self.torso_a.to(dtype).view(1, 3, 1)).squeeze(-1) + off
                b = pt + (Rt @ self.torso_b.to(dtype).view(1, 3, 1)).squeeze(-1) + off
                hc = pt + (Rt @ self.head_c.to(dtype).view(1, 3, 1)).squeeze(-1) + off
                s_t = segment_segment_sdf(butt, tip, a, b, self.stick_r, self.torso_r)
                s_h = segment_segment_sdf(butt, tip, hc, hc, self.stick_r, self.head_r)
                s_wt = point_capsule_sdf(wrist, a, b, self.torso_r) - self.wrist_r
                s_wh = point_sphere_sdf(wrist, hc, self.head_r) - self.wrist_r
                sdf_st = torch.minimum(s_t, s_h)
                sdf_wr = torch.minimum(s_wt, s_wh)
                pen_self = pen_self + self._hinge(sdf_st) + self._hinge(sdf_wr)
                min_sdf = torch.minimum(min_sdf, torch.minimum(sdf_st, sdf_wr))
                hand_min = torch.minimum(hand_min, torch.minimum(sdf_st, sdf_wr))
            per_hand[hand] = hand_min

        pen = self.w_drum * pen_drum + self.w_under * pen_under + self.w_self * pen_self
        if not return_details:
            return pen
        details = SdfPenaltyDetails(
            min_sdf=min_sdf.detach(),
            pen_drum=pen_drum.detach(),
            pen_under=pen_under.detach(),
            pen_self=pen_self.detach(),
            extras={f"min_sdf_{h}": v.detach() for h, v in per_hand.items()},
        )
        return pen, details


@torch.no_grad()
def q0_collision_clear(
    sdf: SdfPenalty,
    q: torch.Tensor,
    *,
    drum_head_centers: torch.Tensor | None = None,
    drum_radii: torch.Tensor | None = None,
    exempt_drum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row start-pose screen: stick + wrist ``min_sdf > 0``.

    Same geometry as Pool A ``_sdf_clear`` / :class:`StartSampler`.  ``exempt_drum``
    is ``(B, 2)`` long (``-1`` = none): the stick capsule vs that hand's drum is
    not a collision -- a just-struck rest, matching the cost's departure
    exemption.  The wrist is never exempted.
    """
    tgt_l = tgt_r = None
    if exempt_drum is not None:
        ex = exempt_drum.to(device=q.device, dtype=torch.long)
        tgt_l, tgt_r = ex[:, 0], ex[:, 1]
    D = int(sdf.num_drums)
    if drum_head_centers is not None and drum_head_centers.shape[-2] != D:
        drum_head_centers = drum_head_centers[..., :D, :]
    if drum_radii is not None and drum_radii.shape[-1] != D:
        drum_radii = drum_radii[..., :D]
    if tgt_l is not None:
        tgt_l = tgt_l.where(tgt_l < D, torch.full_like(tgt_l, -1))
        tgt_r = tgt_r.where(tgt_r < D, torch.full_like(tgt_r, -1))
    _pen, det = sdf.forward(
        q,
        drum_head_centers=drum_head_centers,
        drum_radii=drum_radii,
        target_drum_left=tgt_l,
        target_drum_right=tgt_r,
        return_details=True,
    )
    return det.min_sdf > 0.0
