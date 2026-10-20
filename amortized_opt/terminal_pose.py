"""Amortized terminal-pose solver: one upper-body q from scene + start state.

Trained by backprop through a differentiable kinematic cost.  No behavioural
cloning and no IK teacher.  The network predicts a single pose ``q`` of shape
``(B, 17)`` in :data:`~amortized_opt.constants.G1_UPPER_BODY_JOINTS` order
(including waist yaw/pitch/roll).  Start position, velocity-induced coast
displacement, and time-to-hit select a nearby reachable IK branch.

Feature vector (the only network input), with ``D`` drums from the layout::

    F = 9*D + 20 + 2*N_DOF + 2
      = D*(pos_pelvis 3 + axis 3 + radius 1)
        + 2*(one_hot D + active 1 + stick_axis 3 + stick_origin 3
             + tip_sigma 1 + shaft_lo 1 + is_cymbal 1)
        + normalized_q_start N_DOF + normalized_qd_times_T N_DOF
        + time_to_hit 2

On the default 7-drum kit, ``F = 119``.

Loss (objectives fixed-weight; constraints PHR-priced)::

    L = w_joint*joint + w_start*start + w_motion*motion
        + w_com*com + w_elev_drum*elev_drum
        + w_elev_cymbal*elev_cymbal + w_shaft*shaft
        + w_waist*waist + w_elev_floor*elev_floor
        + λ_hit*v_hit + (pen_hit/2)*v_hit^2
        + λ_col*v_col + (pen_col/2)*v_col^2

``waist`` is a normal penalty on yaw versus the circular mean of the active
drums' azimuths, with standard deviation 30°.  Collision follows
:meth:`amortized_opt.cost.FusedCost._collision`: stick capsule and wrist
sphere versus drums, under-volumes, torso, head and the fixed legs, plus the
cross-hand stick/wrist pairs.  Upper-arm and forearm capsules join those
same pair tests.  The stick capsule versus the target drum stays exempt;
samples of that stick that lie outside the hit disk do not.  The collision
residual is penetration beyond 5 mm: ``v_col = relu(-sdf - 0.005)``.
Pad sticks must clear the head by at least 5° (positive elevation, into
the head).  Cymbals and the hi-hat are not in that floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import limits as _limits
from .constants import (
    G1_UPPER_BODY_JOINTS,
    N_DOF,
    PELVIS_IN_ENV,
    SHAFT_HIT_SAMPLES,
    STICK_LENGTH,
    STICK_RADIUS,
    TIP_RADIUS,
    UPPER_ARM_RADIUS,
    FOREARM_RADIUS,
    ready_joint_pos,
)
from .cost import FusedCost
from .fk import (
    ChainFK,
    StickRandomization,
    stick_axis_local,
    stick_mount_transform,
    stick_shaft_offsets,
)
from .lagrangian import ALConfig, AugmentedLagrangian
from .limits import urdf_position_limits
from .phase2_data import strike_geometry_table
from .rhythm import DrumKit, KitRandomization
from .sdf import (
    SdfPenalty,
    fixed_capsule_self_sdf,
    hit_region_dist,
    point_capsule_sdf,
    point_cylinder_sdf,
    point_sphere_sdf,
    segment_cylinder_sdf,
    segment_hit_region,
    segment_segment_sdf,
)
from .target_contact import v4_static_probe_distance
from .capsule_contact import capsule_static_distance
from .constants import SHAFT_HIT_SPAN
from .convex_distance import capsule_cylinder_signed_distance
from .strike import (
    CYMBAL_CONTACT_SIGMA_FALLBACK,
    HIHAT_SHOULDER_SIGMA,
    STICK_TIP_SIGMA,
    is_cymbal_name,
    strike_axial_limits,
)

__all__ = [
    "FEATURE_DIM_PER_DRUM",
    "HANDS",
    "IDLE_HAND_FRAC",
    "TerminalBatch",
    "TerminalPoseAL",
    "TerminalPoseConfig",
    "TerminalPoseCost",
    "TerminalPoseNet",
    "TerminalPoseResult",
    "aggregate_collision_violation",
    "collision_violation",
    "feature_dim",
    "joint_time_to_hit",
    "pack_features",
    "sample_terminal_batch",
    "target_waist_yaw",
]

HANDS: tuple[str, str] = ("left", "right")
IDLE_HAND_FRAC: float = 0.15
#: Per-drum block in :func:`pack_features` (pos 3 + axis 3 + radius 1).
FEATURE_DIM_PER_DRUM: int = 7
#: Per-hand extras beyond the one-hot (active + stick_axis + origin + sigmas + cym).
_HAND_SCALAR_DIM: int = 10
_START_FEATURE_DIM: int = 2 * N_DOF + 2
_SOFTMIN_TEMP: float = 0.02
#: Penetration deeper than this is the collision residual.  5 mm of overlap
#: is allowed; clearance is not required.
_COLLISION_ALLOW_M: float = 0.005
_ARM_MASSES: tuple[float, float, float, float] = (2.095, 0.600, 0.824, 0.100)
_ELEV_DRUM_TARGET_DEG: float = 35.0
_ELEV_DRUM_MIN_DEG: float = 5.0
_ELEV_CYMBAL_TARGET_Z: float = 0.50
#: URDF shoulder collision cylinders are r=0.03; elbow and wrist links are
#: meshes.  These are conservative capsules in the same spirit as
#: ``FIXED_LOWER_BODY_CAPSULES``.  :class:`~amortized_opt.cost.FusedCost`
#: scores only the stick and the wrist.
_UPPER_ARM_RADIUS: float = UPPER_ARM_RADIUS
_FOREARM_RADIUS: float = FOREARM_RADIUS
_STICK_DRUM_SAMPLES: int = 8
_WAIST_YAW_SIGMA_DEG: float = 30.0


def feature_dim(num_drums: int) -> int:
    """Start-conditioned feature size for a ``D``-drum kit."""
    d = int(num_drums)
    return FEATURE_DIM_PER_DRUM * d + 2 * (d + _HAND_SCALAR_DIM) + _START_FEATURE_DIM


# ================================================================== batch --


@dataclass
class TerminalBatch:
    """One minibatch of terminal-pose tasks.

    ``features`` is the only tensor the network reads; build it with
    :func:`pack_features` so the cost cannot disagree with the net.
    """

    features: torch.Tensor       # (B, F)
    drum_pos: torch.Tensor       # (B, D, 3) env-frame head centres
    drum_axis: torch.Tensor      # (B, D, 3) unit head normals
    drum_radius: torch.Tensor    # (B, D)
    drum_index: torch.Tensor     # (B, 2) long; -1 = idle
    active: torch.Tensor         # (B, 2) bool
    stick_axis: torch.Tensor     # (B, 2, 3) wrist-frame handle->tip
    stick_origin: torch.Tensor   # (B, 2, 3) wrist-frame mount (sigma=0)
    tip_sigma: torch.Tensor      # (B, 2)
    shaft_lo: torch.Tensor       # (B, 2) inner legal shaft end
    is_cymbal: torch.Tensor      # (B, 2) bool; cymbals and hi-hat
    q_start: torch.Tensor        # (B, 17) measured starting position
    qd_start: torch.Tensor       # (B, 17) measured starting velocity
    time_to_hit: torch.Tensor    # (B, 2) seconds; zero for an idle hand
    drum_names: tuple[str, ...] = ()


def pack_features(
    drum_pos: torch.Tensor,
    drum_axis: torch.Tensor,
    drum_radius: torch.Tensor,
    drum_index: torch.Tensor,
    active: torch.Tensor,
    stick_axis: torch.Tensor,
    stick_origin: torch.Tensor,
    tip_sigma: torch.Tensor,
    shaft_lo: torch.Tensor,
    is_cymbal: torch.Tensor,
    q_start: torch.Tensor,
    qd_start: torch.Tensor,
    time_to_hit: torch.Tensor,
) -> torch.Tensor:
    """Build the network feature vector from the structured batch fields.

    Layout (concatenated)::

        for each drum d:
            (drum_pos[d] - PELVIS_IN_ENV), unit drum_axis[d], drum_radius[d]
        for each hand h in (left, right):
            one_hot(drum_index[h]) or zeros if idle,
            active[h], stick_axis[h], stick_origin[h],
            tip_sigma[h], shaft_lo[h], is_cymbal[h]
        normalized start q, normalized qd*T, normalized per-hand time-to-hit
    """
    B, D, _ = drum_pos.shape
    pelvis = drum_pos.new_tensor(PELVIS_IN_ENV.tolist()).view(1, 1, 3)
    pos_p = drum_pos - pelvis
    ax = drum_axis / drum_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    drum_block = torch.cat(
        [pos_p, ax, drum_radius.unsqueeze(-1)], dim=-1
    ).reshape(B, D * FEATURE_DIM_PER_DRUM)

    hands: list[torch.Tensor] = []
    act_f = active.to(dtype=drum_pos.dtype)
    cym_f = is_cymbal.to(dtype=drum_pos.dtype)
    for hi in range(2):
        idx = drum_index[:, hi]
        one = F.one_hot(idx.clamp(min=0), D).to(dtype=drum_pos.dtype)
        one = one * (idx >= 0).to(dtype=drum_pos.dtype).unsqueeze(-1)
        hands.append(
            torch.cat(
                [
                    one,
                    act_f[:, hi : hi + 1],
                    stick_axis[:, hi],
                    stick_origin[:, hi],
                    tip_sigma[:, hi : hi + 1],
                    shaft_lo[:, hi : hi + 1],
                    cym_f[:, hi : hi + 1],
                ],
                dim=-1,
            )
        )
    lo, hi = urdf_position_limits(G1_UPPER_BODY_JOINTS)
    mid = drum_pos.new_tensor(0.5 * (lo + hi)).view(1, -1)
    half = drum_pos.new_tensor(0.5 * (hi - lo)).clamp_min(1e-3).view(1, -1)
    q0 = (q_start.to(dtype=drum_pos.dtype) - mid) / half
    joint_t = joint_time_to_hit(time_to_hit, active)
    coast = qd_start.to(dtype=drum_pos.dtype) * joint_t / half
    # Two seconds is the largest ordinary phase-2 horizon (100 * 20 ms).
    tth = time_to_hit.to(dtype=drum_pos.dtype) / 2.0
    return torch.cat([drum_block, *hands, q0, coast, tth], dim=-1)


def joint_time_to_hit(
    time_to_hit: torch.Tensor,
    active: torch.Tensor,
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
) -> torch.Tensor:
    """Expand per-hand hit times to one time per joint."""
    t = time_to_hit
    act = active.bool()
    inf = torch.full_like(t, float("inf"))
    waist_t = torch.where(act, t, inf).amin(dim=-1)
    waist_t = torch.where(torch.isfinite(waist_t), waist_t, torch.zeros_like(waist_t))
    cols = []
    for name in joint_names:
        if name.startswith("left_"):
            cols.append(torch.where(act[:, 0], t[:, 0], torch.zeros_like(t[:, 0])))
        elif name.startswith("right_"):
            cols.append(torch.where(act[:, 1], t[:, 1], torch.zeros_like(t[:, 1])))
        else:
            cols.append(waist_t)
    return torch.stack(cols, dim=-1)


# ================================================================== net --


class TerminalPoseNet(nn.Module):
    """MLP mapping features to an upper-body pose.

    ``depth`` is the number of hidden SiLU layers.  The default, two layers
    of width 128, is 34,065 parameters with the start-conditioned features.
    """

    def __init__(
        self,
        in_dim: int | None = None,
        *,
        width: int = 128,
        depth: int = 2,
        optnet: bool = False,
        n_constraints: int = 4,
        centered_qp: bool = False,
        mix_width: int | None = None,
        qp_out: bool = False,
        skip_dim: int = 8,
        clamp_knee: float = 0.8,
        num_drums: int | None = None,
    ):
        super().__init__()
        if in_dim is None:
            d = int(num_drums) if num_drums is not None else DrumKit.from_layout().num_drums
            in_dim = feature_dim(d)
        if int(depth) < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.optnet = bool(optnet)
        self.n_constraints = int(n_constraints)
        self.centered_qp = bool(centered_qp)
        self.mix_width = None if mix_width is None else int(mix_width)
        self.qp_out = bool(qp_out)
        self.skip_dim = int(skip_dim)
        self.clamp_knee = float(clamp_knee)
        if self.qp_out and self.mix_width is not None:
            raise ValueError("qp_out ends at the solve, so mix_width must be omitted")
        head_in = self.width
        if self.optnet:
            from .terminal_pose_optnet import CenteredQuadraticLayer, JointQuadraticLayer

            # ``n_constraints`` is both the inequality count and the equality count.
            self.pre = nn.Linear(self.in_dim, width)
            if self.centered_qp:
                self.qp = CenteredQuadraticLayer(width, n=N_DOF)
            else:
                self.qp = JointQuadraticLayer(
                    width, n=N_DOF, n_eq=self.n_constraints, n_ineq=self.n_constraints,
                )
            if self.mix_width is not None or self.qp_out:
                # Two SiLU maps write the QP.  The solve is the pose, unless
                # a mixer residual is attached after it.
                self.enc = nn.Linear(width, width)
            if self.mix_width is not None:
                if self.mix_width < 1:
                    raise ValueError(f"mix_width must be >= 1, got {self.mix_width}")
                if not 1 <= self.skip_dim <= self.width:
                    raise ValueError(
                        f"skip_dim must be in [1, {self.width}], got {self.skip_dim}"
                    )
                # Eight channels of the second hidden map skip past the solve
                # and mix with the 17 joint values.
                self.mix = nn.Linear(N_DOF + self.skip_dim, self.mix_width)
                head_in = self.mix_width
            elif not self.qp_out:
                # One hidden map writes the QP.  The second map reads only
                # the joint solution.
                self.post = nn.Linear(N_DOF, width)
            self.hidden = nn.ModuleList()
        else:
            hidden: list[nn.Linear] = [nn.Linear(self.in_dim, width)]
            for _ in range(self.depth - 1):
                hidden.append(nn.Linear(width, width))
            self.hidden = nn.ModuleList(hidden)
        if not self.qp_out:
            self.head = nn.Linear(head_in, N_DOF)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        # Drumming joint box, same default as FusedCost (posture-narrowed URDF).
        lo, hi = urdf_position_limits(G1_UPPER_BODY_JOINTS)
        ready = ready_joint_pos(G1_UPPER_BODY_JOINTS)
        self.register_buffer("q_lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("q_hi", torch.as_tensor(hi, dtype=torch.float32))
        self.register_buffer(
            "q_half",
            torch.as_tensor(0.5 * (hi - lo), dtype=torch.float32).clamp_min(1e-3),
        )
        self.register_buffer("q_ready", torch.as_tensor(ready, dtype=torch.float32))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.optnet and self.qp_out:
            h1 = F.silu(self.pre(features))
            h2 = F.silu(self.enc(h1))
            z = self.qp(h2, self.q_lo, self.q_hi, self.q_ready)
            return self._soft_clip(z)
        if self.optnet and self.mix_width is not None:
            h1 = F.silu(self.pre(features))
            h2 = F.silu(self.enc(h1))
            z = self.qp(h2, self.q_lo, self.q_hi, self.q_ready)
            skipped = h2[:, : self.skip_dim]
            mixed = F.silu(self.mix(torch.cat([z, skipped], dim=-1)))
            raw = self.head(mixed)
            qh = self.q_half.to(dtype=raw.dtype)
            # Residual on the QP solution.  A zero head leaves that solution.
            q = z + qh * torch.tanh(raw)
            return self._soft_clip(q)
        if self.optnet:
            h = F.silu(self.pre(features))
            z = self.qp(h, self.q_lo, self.q_hi, self.q_ready)
            h = F.silu(self.post(z))
            raw = self.head(h)
            qh = self.q_half.to(dtype=raw.dtype)
            # Residual on the QP solution.  A zero head leaves that solution.
            q = z + qh * torch.tanh(raw)
            return self._soft_clip(q)
        h = features
        for layer in self.hidden:
            h = F.silu(layer(h))
        raw = self.head(h)
        qh = self.q_half.to(dtype=raw.dtype)
        q = self.q_ready.to(dtype=raw.dtype) + qh * torch.tanh(raw)
        return self._soft_clip(q)

    def _soft_clip(self, q: torch.Tensor) -> torch.Tensor:
        """Identity inside ``clamp_knee`` × half-range; tanh tail to URDF limits."""
        lo = self.q_lo.to(dtype=q.dtype).view(1, -1)
        hi = self.q_hi.to(dtype=q.dtype).view(1, -1)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        knee = float(self.clamp_knee) * half
        band = (half - knee).clamp_min(1e-6)
        d = q - mid
        core = d.clamp(-knee, knee)
        return mid + core + band * torch.tanh((d - core) / band)


# ================================================================ sampler --


def _default_kit_rand() -> KitRandomization:
    return KitRandomization(
        per_drum_xy=0.04,
        global_xy=0.075,
        global_z=0.035,
        per_drum_tilt=0.05,
        per_drum_radius=0.05,
    )


def _default_stick_rand() -> StickRandomization:
    return StickRandomization(tilt=0.15, roll=0.15, length=0.02)


def sample_terminal_batch(
    n: int,
    *,
    rng: np.random.Generator | None = None,
    kit: DrumKit | None = None,
    kit_rand: KitRandomization | None = None,
    stick_rand: StickRandomization | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    idle_frac: float = IDLE_HAND_FRAC,
    q_start: torch.Tensor | np.ndarray | None = None,
    qd_start: torch.Tensor | np.ndarray | None = None,
    time_to_hit: torch.Tensor | np.ndarray | None = None,
) -> TerminalBatch:
    """Draw terminal-pose tasks, optionally paired with supplied start states."""
    rng = rng if rng is not None else np.random.default_rng()
    base = kit if kit is not None else DrumKit.from_layout()
    kr = kit_rand if kit_rand is not None else _default_kit_rand()
    sr = stick_rand if stick_rand is not None else _default_stick_rand()
    names = tuple(base.names)
    D = base.num_drums
    eligible: list[list[int]] = []
    for hand in HANDS:
        eligible.append(base.eligible(hand))

    pos = np.zeros((n, D, 3), dtype=np.float64)
    axes = np.zeros((n, D, 3), dtype=np.float64)
    rad = np.zeros((n, D), dtype=np.float64)
    drum_index = np.full((n, 2), -1, dtype=np.int64)
    active = np.zeros((n, 2), dtype=bool)
    stick_ax = np.zeros((n, 2, 3), dtype=np.float64)
    stick_or = np.zeros((n, 2, 3), dtype=np.float64)
    tip_s = np.zeros((n, 2), dtype=np.float64)
    shaft_lo = np.zeros((n, 2), dtype=np.float64)
    is_cym = np.zeros((n, 2), dtype=bool)

    for b in range(n):
        row_kit = base.randomized(rng, kr)
        pos[b] = row_kit.positions
        axes[b] = row_kit.axes
        rad[b] = row_kit.radii
        for hi, hand in enumerate(HANDS):
            sp = sr.sample(rng, hand)
            _R, p_m = stick_mount_transform(hand, sp)
            stick_or[b, hi] = p_m
            stick_ax[b, hi] = stick_axis_local(hand, sp)
            tip_s[b, hi] = float(sp.tip_sigma)
            # Shaft legal from 5 cm past the mount (CYMBAL_CONTACT_SIGMA_FALLBACK),
            # not 5 cm back from the tip; clamp below the bead.
            lo = float(CYMBAL_CONTACT_SIGMA_FALLBACK)
            shaft_lo[b, hi] = min(lo, float(sp.tip_sigma) - 1e-3)
            if rng.random() < float(idle_frac) or not eligible[hi]:
                continue
            di = int(rng.choice(eligible[hi]))
            drum_index[b, hi] = di
            active[b, hi] = True
            is_cym[b, hi] = bool(is_cymbal_name(names[di]))

    def t(arr, *, long: bool = False) -> torch.Tensor:
        if long:
            return torch.as_tensor(arr, device=device, dtype=torch.long)
        return torch.as_tensor(arr, device=device, dtype=dtype)

    drum_pos = t(pos)
    drum_axis = t(axes)
    drum_radius = t(rad)
    di_t = t(drum_index, long=True)
    act_t = torch.as_tensor(active, device=device, dtype=torch.bool)
    sa = t(stick_ax)
    so = t(stick_or)
    ts = t(tip_s)
    sl = t(shaft_lo)
    ic = torch.as_tensor(is_cym, device=device, dtype=torch.bool)

    if q_start is None:
        ready = ready_joint_pos(G1_UPPER_BODY_JOINTS)
        lo, hi = urdf_position_limits(G1_UPPER_BODY_JOINTS)
        q0_np = np.clip(
            ready[None, :] + rng.normal(0.0, 0.18, size=(n, N_DOF)),
            lo[None, :], hi[None, :],
        )
        q0 = t(q0_np)
    else:
        q0 = torch.as_tensor(q_start, device=device, dtype=dtype)
    if qd_start is None:
        vlim = _limits.isaac_velocity_limits(G1_UPPER_BODY_JOINTS)
        qd0 = t(0.10 * vlim[None, :] * rng.normal(size=(n, N_DOF)))
    else:
        qd0 = torch.as_tensor(qd_start, device=device, dtype=dtype)
    if time_to_hit is None:
        # Phase 2 includes a 25% short-horizon family (1--5 waypoints); the
        # remaining strikes span the ordinary 2 s horizon.
        steps = rng.integers(1, 101, size=(n, 2))
        short = rng.random((n, 2)) < 0.25
        steps = np.where(short, rng.integers(1, 6, size=(n, 2)), steps)
        tth = t(0.02 * steps) * act_t.to(dtype)
    else:
        tth = torch.as_tensor(time_to_hit, device=device, dtype=dtype)
        tth = tth * act_t.to(dtype)
    if q0.shape != (n, N_DOF) or qd0.shape != (n, N_DOF):
        raise ValueError(f"q_start and qd_start must have shape ({n}, {N_DOF})")
    if tth.shape != (n, 2):
        raise ValueError(f"time_to_hit must have shape ({n}, 2)")
    feats = pack_features(
        drum_pos, drum_axis, drum_radius, di_t, act_t, sa, so, ts, sl, ic,
        q0, qd0, tth,
    )
    return TerminalBatch(
        features=feats,
        drum_pos=drum_pos,
        drum_axis=drum_axis,
        drum_radius=drum_radius,
        drum_index=di_t,
        active=act_t,
        stick_axis=sa,
        stick_origin=so,
        tip_sigma=ts,
        shaft_lo=sl,
        is_cymbal=ic,
        q_start=q0,
        qd_start=qd0,
        time_to_hit=tth,
        drum_names=names,
    )


# ================================================================== cost --


def target_waist_yaw(
    drum_pos: torch.Tensor,
    drum_index: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Circular mean of the active drums' azimuths, in the pelvis frame.

    Azimuth is ``atan2(y, x)`` of the head centre.  Idle hands drop out.  A
    row with no active hand returns 0.
    """
    pelvis = drum_pos.new_tensor(PELVIS_IN_ENV.tolist()).view(1, 1, 3)
    pos = drum_pos - pelvis
    az = torch.atan2(pos[..., 1], pos[..., 0])
    B, D = az.shape
    idx = drum_index.clamp(min=0)
    gathered = az.gather(1, idx)
    m = active.to(dtype=az.dtype)
    s = (torch.sin(gathered) * m).sum(dim=-1)
    c = (torch.cos(gathered) * m).sum(dim=-1)
    mu = torch.atan2(s, c)
    return torch.where(m.sum(dim=-1) > 0.5, mu, torch.zeros(B, device=az.device, dtype=az.dtype))


def _wrap_angle(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


@dataclass
class TerminalPoseConfig:
    """Objective weights for :class:`TerminalPoseCost` (AL λ/pen come separately)."""

    w_joint: float = 10.0
    w_start: float = 10.0
    w_motion: float = 0.02
    accel_scale: float = 30.0
    w_com: float = 1.0
    w_elev_drum: float = 1.0
    w_elev_floor: float = 25.0
    elev_drum_min_deg: float = _ELEV_DRUM_MIN_DEG
    w_elev_cymbal: float = 0.5
    w_shaft: float = 1.0
    w_waist: float = 1.0
    waist_yaw_sigma_deg: float = _WAIST_YAW_SIGMA_DEG
    collision_allow_m: float = _COLLISION_ALLOW_M
    collision_topk: int = 1
    softmin_temp: float = _SOFTMIN_TEMP
    elev_drum_target_deg: float = _ELEV_DRUM_TARGET_DEG
    elev_cymbal_target_z: float = _ELEV_CYMBAL_TARGET_Z
    #: Pose-only geometry surrogate. Archived role checkpoints omit this and
    #: therefore keep their original near-half-disk/penetrated-slab objective.
    contact_scoring_mode: str = 'legacy'
    physics_loss_version: str = 'legacy'


@dataclass
class TerminalPoseResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    violations: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


class TerminalPoseCost:
    """Differentiable kinematic terminal-pose cost (one FK pass per call)."""

    def __init__(
        self,
        cfg: TerminalPoseConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        fk: ChainFK | None = None,
        sdf: SdfPenalty | None = None,
        drum_names: tuple[str, ...] | None = None,
    ):
        self.cfg = cfg or TerminalPoseConfig()
        if self.cfg.physics_loss_version not in ('legacy', 'geometry_v2'):
            raise ValueError('Unknown terminal-pose physics loss version')
        self._capsule_distance = (capsule_cylinder_signed_distance
            if self.cfg.physics_loss_version == 'geometry_v2' else segment_cylinder_sdf)
        if self.cfg.contact_scoring_mode not in ('legacy','isaac_v4','isaac_v5'):
            raise ValueError('Unknown terminal-pose contact scoring mode')
        self.device = torch.device(device)
        self.dtype = dtype
        self.fk = fk if fk is not None else ChainFK(
            joint_names=G1_UPPER_BODY_JOINTS,
            frames=dict(FusedCost.FRAMES),
            device=self.device,
            dtype=dtype,
        )
        self.sdf = sdf if sdf is not None else SdfPenalty(
            fk=self.fk, device=self.device, dtype=dtype
        )
        names = drum_names if drum_names is not None else tuple(DrumKit.from_layout().names)
        self.drum_names = tuple(names)
        lo, hi = urdf_position_limits(G1_UPPER_BODY_JOINTS)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        self.arm_masses = torch.as_tensor(
            _ARM_MASSES, device=self.device, dtype=dtype
        )
        self.axial_lo, self.axial_hi = strike_axial_limits()
        self.butt_sigma = -0.5 * float(STICK_LENGTH)
        mount=stick_mount_transform('left')[1]
        axis=stick_axis_local('left')
        shaft=np.asarray(stick_shaft_offsets('left',SHAFT_HIT_SAMPLES))
        fractions=np.r_[1.,((shaft-mount)@axis)/float(STICK_TIP_SIGMA)]
        self.probe_sigma_fractions=torch.as_tensor(fractions,device=self.device,dtype=dtype)
        self.probe_radii=torch.as_tensor([TIP_RADIUS,*([STICK_RADIUS]*SHAFT_HIT_SAMPLES)],
                                          device=self.device,dtype=dtype)
        self.yaw_i = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")

    def __call__(self, q, batch, weights=None) -> TerminalPoseResult:
        return self.forward(q, batch, weights=weights)

    def forward(
        self,
        q: torch.Tensor,
        batch: TerminalBatch,
        weights: dict[str, float] | None = None,
    ) -> TerminalPoseResult:
        cfg = self.cfg
        B = q.shape[0]
        dtype = q.dtype
        frames = self.fk.forward(q)
        act = batch.active.to(dtype=dtype)
        n_act = act.sum(dim=-1).clamp_min(1.0)

        # ---- hit disk ------------------------------------------------------
        pelvis = q.new_tensor(PELVIS_IN_ENV.tolist()).view(1, 1, 3)
        pos_p = batch.drum_pos.to(dtype) - pelvis
        ax_d = batch.drum_axis.to(dtype)
        rad = batch.drum_radius.to(dtype)
        names = batch.drum_names or self.drum_names
        geo = strike_geometry_table(pos_p, ax_d, rad, names)
        if cfg.contact_scoring_mode in ('isaac_v4','isaac_v5'):
            edge=torch.as_tensor([is_cymbal_name(name) for name in names],
                                 device=q.device,dtype=torch.bool)
            geo['has_hp']=torch.where(edge[None],torch.zeros_like(geo['has_hp']),
                                      geo['has_hp'])

        hit_sum = q.new_zeros(B)
        hit_dist_aux = q.new_zeros(B, 2)
        elev_pad_sum = q.new_zeros(B)
        elev_floor_sum = q.new_zeros(B)
        n_pad = q.new_zeros(B)
        elev_cym_sum = q.new_zeros(B)
        n_cym = q.new_zeros(B)
        shaft_sum = q.new_zeros(B)
        stick_elev_deg = q.new_zeros(B, 2)
        stick_z = q.new_zeros(B, 2)
        shaft_s_aux = q.new_zeros(B, 2)
        axis_w_store = q.new_zeros(B, 2, 3)
        origin_w_store = q.new_zeros(B, 2, 3)

        ar = torch.arange(B, device=q.device)
        for hi, hand in enumerate(HANDS):
            R, p = frames[f"{hand}_wrist"]
            ax_l = batch.stick_axis[:, hi].to(dtype).unsqueeze(-1)
            or_l = batch.stick_origin[:, hi].to(dtype).unsqueeze(-1)
            axis_w = (R @ ax_l).squeeze(-1)
            origin_w = p + (R @ or_l).squeeze(-1)
            tip_s = batch.tip_sigma[:, hi].to(dtype)
            shaft_lo = batch.shaft_lo[:, hi].to(dtype)
            cym = batch.is_cymbal[:, hi]
            # Pads: degenerate bead segment.  Cymbals/hi-hat: real shaft
            # sigma in [shaft_lo, tip_sigma] (outer ~15 cm on a nominal stick).
            sig_lo = torch.where(cym,
                tip_s-SHAFT_HIT_SPAN if cfg.contact_scoring_mode=='isaac_v5' else shaft_lo,
                tip_s)
            sig_hi = tip_s
            a = origin_w + sig_lo.unsqueeze(-1) * axis_w
            b = origin_w + sig_hi.unsqueeze(-1) * axis_w
            axis_w_store[:, hi] = axis_w
            origin_w_store[:, hi] = origin_w

            di = batch.drum_index[:, hi]
            safe = di.clamp(min=0)
            p_star = geo["disk_center"][ar, safe]
            n_out = geo["axis"][ar, safe]
            hit_r = geo["hit_radius"][ar, safe]
            hp = geo["hp_dir"][ar, safe]
            has_hp = geo["has_hp"][ar, safe]
            disk_off = torch.zeros_like(p_star)

            dmin, _s_hard, _pc, cand_s, cand_dist, cand_valid = segment_hit_region(
                a, b, p_star, n_out, hit_r, disk_off, hp, has_hp,
                self.axial_lo, self.axial_hi,
                return_candidates=True,
            )
            v4_probe_dist=None
            if cfg.contact_scoring_mode in ('isaac_v4','isaac_v5'):
                if cfg.contact_scoring_mode=='isaac_v5':
                    low_sigma=tip_s-SHAFT_HIT_SPAN
                    physical_mount=origin_w+low_sigma[:,None]*axis_w
                    physical_tip=origin_w+tip_s[:,None]*axis_w
                    v4_probe_dist,_,_,fractions,_=capsule_static_distance(
                        physical_mount,physical_tip,pos_p[ar,safe],n_out,
                        rad[ar,safe],torch.full_like(rad[ar,safe],.1))
                    sigmas=low_sigma[:,None]+fractions*SHAFT_HIT_SPAN
                else:
                    sigmas=tip_s[:,None]*self.probe_sigma_fractions[None]
                    physical_probes=origin_w[:,None]+sigmas[...,None]*axis_w[:,None]
                    v4_probe_dist,_,_=v4_static_probe_distance(
                        physical_probes,pos_p[ar,safe],n_out,rad[ar,safe],
                        self.probe_radii)
                dmin=torch.where(cym,v4_probe_dist.amin(-1),dmin)
            m = act[:, hi]
            hit_sum = hit_sum + m * dmin
            hit_dist_aux[:, hi] = dmin.detach()

            # Soft contact parameter for the shaft preference only; hit residual
            # stays the hard minimum distance above.
            logits = -(cand_dist + 1e3 * (1.0 - cand_valid)) / float(cfg.softmin_temp)
            wgt = torch.softmax(logits, dim=-1)
            s_soft = (wgt * cand_s).sum(dim=-1)
            if v4_probe_dist is not None:
                v4_wgt=torch.softmax(-v4_probe_dist/float(cfg.softmin_temp),dim=-1)
                v4_sigma=(v4_wgt*sigmas).sum(-1)
                v4_fraction=((v4_sigma-shaft_lo)/(tip_s-shaft_lo).clamp_min(1e-6)).clamp(0.,1.)
                s_soft=torch.where(cym,v4_fraction,s_soft)
            shaft_s_aux[:, hi] = s_soft.detach()

            # Pad elevation vs head normal (target 35°).
            cosang = (-(axis_w * n_out).sum(-1)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            elev = torch.asin(cosang)
            elev_t = q.new_tensor(float(np.deg2rad(cfg.elev_drum_target_deg)))
            pad_m = m * (~cym).to(dtype)
            elev_pad_sum = elev_pad_sum + pad_m * (elev - elev_t).square()
            floor = pad_elevation_shortfall(elev, cfg.elev_drum_min_deg)
            elev_floor_sum = elev_floor_sum + pad_m * floor.square()
            n_pad = n_pad + pad_m
            stick_elev_deg[:, hi] = torch.rad2deg(elev).detach()

            # Cymbal/hi-hat: prefer tip-up world-z and shoulder contact sigma.
            az = axis_w[..., 2]
            stick_z[:, hi] = az.detach()
            cym_m = m * cym.to(dtype)
            elev_cym_sum = elev_cym_sum + cym_m * (
                az - float(cfg.elev_cymbal_target_z)
            ).square()
            n_cym = n_cym + cym_m
            denom_s = (tip_s - shaft_lo).clamp_min(1e-6)
            s_pref = (
                (float(HIHAT_SHOULDER_SIGMA) - shaft_lo) / denom_s
            ).clamp(0.0, 1.0)
            shaft_sum = shaft_sum + cym_m * (s_soft - s_pref).square()

        v_hit = hit_sum / n_act

        # ---- collision (FusedCost pairs, plus arm capsules) -----------------
        v_col, min_sdf = self._collision(
            frames, batch, axis_w_store, origin_w_store, q, geo,
        )

        # ---- waist yaw: normal well about the mean drum azimuth ------------
        mu = target_waist_yaw(batch.drum_pos.to(dtype), batch.drum_index, batch.active)
        sigma = q.new_tensor(float(np.deg2rad(cfg.waist_yaw_sigma_deg))).clamp_min(1e-3)
        yaw_err = _wrap_angle(q[:, self.yaw_i] - mu)
        n_yaw = act.sum(dim=-1).clamp(max=1.0)
        waist = n_yaw * 0.5 * (yaw_err / sigma).square()

        # ---- objectives ----------------------------------------------------
        over = torch.maximum(
            self.q_lo.to(dtype) - q, q - self.q_hi.to(dtype)
        ).clamp_min(0.0)
        joint = over.square().mean(dim=-1)

        half = (0.5 * (self.q_hi - self.q_lo)).to(dtype).clamp_min(1e-3)
        q_start = batch.q_start.to(dtype)
        qd_start = batch.qd_start.to(dtype)
        start = ((q - q_start) / half).square().mean(dim=-1)
        joint_t = joint_time_to_hit(batch.time_to_hit.to(dtype), batch.active)
        coast_error = q - q_start - qd_start * joint_t
        moving = joint_t > 0.0
        # Exact plateau acceleration of the 10% ramp used by strike_seed,
        # normalized by the repository's 30 rad/s^2 training scale.  Smooth-L1
        # prevents the deliberately impossible 1--5 waypoint cases dominating.
        ramp_integral = 0.5 - 0.5 * 0.1 + 0.1 * 0.1 / 6.0
        denom = (
            float(cfg.accel_scale) * ramp_integral * joint_t.square()
        ).clamp_min(1e-6)
        accel_ratio = coast_error / denom
        motion_elem = F.smooth_l1_loss(
            accel_ratio, torch.zeros_like(accel_ratio), reduction="none", beta=1.0,
        )
        motion = (motion_elem * moving.to(dtype)).sum(dim=-1) / moving.sum(dim=-1).clamp_min(1)

        com_sum = q.new_zeros(B)
        com_z_aux = q.new_zeros(B, 2)
        Rt, pt = frames["torso"]
        masses = self.arm_masses.to(dtype)
        mass_sum = masses.sum()
        for hi, hand in enumerate(HANDS):
            shoulder = frames[f"{hand}_shoulder"][1]
            elbow = frames[f"{hand}_elbow"][1]
            Rw, wrist = frames[f"{hand}_wrist"]
            upper = 0.65 * shoulder + 0.35 * elbow
            forearm = 0.5 * (elbow + wrist)
            or_l = batch.stick_origin[:, hi].to(dtype)
            hand_com = wrist + torch.einsum("bij,bj->bi", Rw, or_l * 0.25)
            stick_com = wrist + torch.einsum("bij,bj->bi", Rw, or_l)
            arm_com = (
                masses[0] * upper
                + masses[1] * forearm
                + masses[2] * hand_com
                + masses[3] * stick_com
            ) / mass_sum
            torso_com = torch.einsum("bji,bj->bi", Rt, arm_com - pt)
            z = torso_com[..., 2]
            com_z_aux[:, hi] = z.detach()
            com_sum = com_sum + act[:, hi] * z
        com = com_sum / n_act

        elev_drum = elev_pad_sum / n_pad.clamp_min(1.0)
        elev_floor = elev_floor_sum / n_pad.clamp_min(1.0)
        elev_cymbal = elev_cym_sum / n_cym.clamp_min(1.0)
        shaft = shaft_sum / n_cym.clamp_min(1.0)

        terms: dict[str, torch.Tensor] = {
            "joint": float(cfg.w_joint) * joint,
            "start": float(cfg.w_start) * start,
            "motion": float(cfg.w_motion) * motion,
            "com": float(cfg.w_com) * com,
            "elev_drum": float(cfg.w_elev_drum) * elev_drum,
            "elev_floor": float(cfg.w_elev_floor) * elev_floor,
            "elev_cymbal": float(cfg.w_elev_cymbal) * elev_cymbal,
            "shaft": float(cfg.w_shaft) * shaft,
            "waist": float(cfg.w_waist) * waist,
            "hit": q.new_zeros(B),
            "collision": q.new_zeros(B),
        }
        # PHR: λ v + (pen/2) v² on live residuals (mirror FusedCost).
        for g, v in (("hit", v_hit), ("collision", v_col)):
            if weights is None:
                lam, pen = 1.0, 0.0
            elif g not in weights and f"pen_{g}" not in weights:
                terms[g] = q.new_zeros(B)
                continue
            else:
                lam = float(weights[g]) if g in weights else 0.0
                pen = float(weights.get(f"pen_{g}", 0.0) or 0.0)
            priced = lam * v
            if pen != 0.0:
                priced = priced + 0.5 * pen * v.clamp_min(0.0).square()
            terms[g] = priced

        total = sum(terms.values())
        return TerminalPoseResult(
            total=total,
            terms=terms,
            violations={"hit": v_hit.detach(), "collision": v_col.detach()},
            aux={
                "hit_dist": hit_dist_aux,
                "stick_elev_deg": stick_elev_deg,
                "stick_z": stick_z,
                "shaft_s": shaft_s_aux,
                "arm_com_z": com_z_aux,
                "joint_hinge": over.detach(),
                "start_rmse_rad": (q - q_start).square().mean(dim=-1).sqrt().detach(),
                "accel_ratio_rms": (
                    (accel_ratio.square() * moving.to(dtype)).sum(dim=-1)
                    / moving.sum(dim=-1).clamp_min(1)
                ).sqrt().detach(),
                "min_sdf": min_sdf.detach(),
                "waist_yaw_target": mu.detach(),
                "waist_yaw_err_deg": torch.rad2deg(yaw_err).detach(),
            },
        )

    def _collision(
        self,
        frames: dict[str, tuple[torch.Tensor, torch.Tensor]],
        batch: TerminalBatch,
        axis_w: torch.Tensor,
        origin_w: torch.Tensor,
        q: torch.Tensor,
        geo: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drum, body, and cross-arm clearance, matching ``FusedCost._collision``.

        Stick capsule and wrist sphere use that checker's pairs and the same
        target-drum exemption (the wrist is never exempt).  Upper-arm and
        forearm capsules join every pair, including the target drum.  Stick
        samples that fall outside the hit disk are also scored against the
        target drum, so the shaft may occupy the disk and nowhere else on
        that head.  When ``collision_topk > 1``, the constraint keeps the full
        worst residual and adds a smaller mean of the remaining selected
        contacts; ``min_sdf`` remains the exact hard minimum for diagnostics.
        """
        sdf = self.sdf
        B = q.shape[0]
        D = batch.drum_pos.shape[1]
        dtype = q.dtype
        off = sdf.frame_offset.to(dtype)
        heads = batch.drum_pos.to(dtype)
        ax = batch.drum_axis.to(dtype)
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rad = batch.drum_radius.to(dtype)
        hh = sdf.half_heights.to(dtype)
        if hh.numel() != D:
            # Scene drum count should match the layout; truncate/pad defensively.
            hh = hh[:D] if hh.numel() > D else F.pad(hh, (0, D - hh.numel()), value=float(hh[-1]))
        hh = hh.view(1, D)
        under_half = sdf.under_half_t.to(dtype)
        if under_half.numel() != D:
            under_half = (
                under_half[:D] if under_half.numel() > D
                else F.pad(under_half, (0, D - under_half.numel()), value=float(sdf.under_half))
            )
        under_half = under_half.view(1, D)
        body_c = heads - hh.unsqueeze(-1) * ax
        under_top = heads - (2.0 * hh + float(sdf.under_clearance)).unsqueeze(-1) * ax
        under_c = under_top - under_half.unsqueeze(-1) * ax

        tip_s = batch.tip_sigma.to(dtype)
        butt_s = tip_s.new_full(tip_s.shape, float(self.butt_sigma))
        tip_w = origin_w + tip_s.unsqueeze(-1) * axis_w
        butt_w = origin_w + butt_s.unsqueeze(-1) * axis_w
        tip_w = tip_w + off
        butt_w = butt_w + off

        min_sdf = q.new_full((B,), float("inf"))
        collision_sdfs: list[torch.Tensor] = []
        stick_r = float(sdf.stick_r)
        wrist_r = float(sdf.wrist_r)
        upper_r = float(_UPPER_ARM_RADIUS)
        fore_r = float(_FOREARM_RADIUS)
        big = q.new_tensor(1.0e3)
        ar = torch.arange(B, device=q.device)
        arms: dict[str, dict[str, torch.Tensor]] = {}

        def collect(dist: torch.Tensor) -> torch.Tensor:
            collision_sdfs.append(dist.reshape(B, -1))
            return dist

        def fold(cur: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
            while dist.dim() > cur.dim():
                dist = dist.amin(dim=-1)
            return torch.minimum(cur, dist)

        def vs_bank(p0, p1, radius, bank_a, bank_b, bank_r) -> torch.Tensor:
            return segment_segment_sdf(
                p0.unsqueeze(1), p1.unsqueeze(1),
                bank_a.unsqueeze(0), bank_b.unsqueeze(0),
                radius, bank_r.view(1, -1),
            )

        Rt, pt = frames["torso"]
        torso_a = pt + (Rt @ sdf.torso_a.to(dtype).view(1, 3, 1)).squeeze(-1) + off
        torso_b = pt + (Rt @ sdf.torso_b.to(dtype).view(1, 3, 1)).squeeze(-1) + off
        head_c = pt + (Rt @ sdf.head_c.to(dtype).view(1, 3, 1)).squeeze(-1) + off
        lower_a = sdf.lower_a.to(dtype) + off
        lower_b = sdf.lower_b.to(dtype) + off
        lower_r = sdf.lower_r.to(dtype)

        for hi, hand in enumerate(HANDS):
            _Rw, p = frames[f"{hand}_wrist"]
            wrist = p + off
            shoulder = frames[f"{hand}_shoulder"][1] + off
            elbow = frames[f"{hand}_elbow"][1] + off
            tip = tip_w[:, hi]
            butt = butt_w[:, hi]
            arms[hand] = {
                "shoulder": shoulder, "elbow": elbow, "wrist": wrist,
                "butt": butt, "tip": tip,
            }

            cap_b = self._capsule_distance(
                butt.unsqueeze(1), tip.unsqueeze(1),
                body_c, ax, rad, hh, ra=stick_r,
            )
            cap_u = self._capsule_distance(
                butt.unsqueeze(1), tip.unsqueeze(1),
                under_c, ax, rad, under_half, ra=stick_r,
            )
            wr = wrist.unsqueeze(1)
            sdf_wb = point_cylinder_sdf(wr, body_c, ax, rad, hh) - wrist_r
            sdf_wu = point_cylinder_sdf(wr, under_c, ax, rad, under_half) - wrist_r

            # Stick vs own target drum is contact, matching FusedCost.  The
            # wrist, and both arm capsules, are never exempt.
            keep = q.new_ones(B, D)
            tgt = batch.drum_index[:, hi]
            onehot = F.one_hot(tgt.clamp(min=0), D).to(dtype) * (tgt >= 0).to(dtype).unsqueeze(-1)
            keep = keep - onehot
            cap_b_m = cap_b.masked_fill(keep <= 0.5, float("inf"))
            cap_u_m = cap_u.masked_fill(keep <= 0.5, float("inf"))
            collect(cap_b_m)
            collect(cap_u_m)
            collect(sdf_wb)
            collect(sdf_wu)
            hand_min = torch.minimum(
                torch.minimum(cap_b_m.amin(dim=1), cap_u_m.amin(dim=1)),
                torch.minimum(sdf_wb.amin(dim=1), sdf_wu.amin(dim=1)),
            )
            for p0, p1, radius in (
                (shoulder, elbow, upper_r),
                (elbow, wrist, fore_r),
            ):
                arm_head = collect(self._capsule_distance(
                    p0.unsqueeze(1), p1.unsqueeze(1), body_c, ax, rad, hh, ra=radius,
                ))
                hand_min = fold(hand_min, arm_head)
                arm_under = collect(self._capsule_distance(
                    p0.unsqueeze(1), p1.unsqueeze(1), under_c, ax, rad, under_half, ra=radius,
                ))
                hand_min = fold(hand_min, arm_under)

            # Stick samples outside the hit disk still collide with the target.
            if geo is not None:
                n_s = int(_STICK_DRUM_SAMPLES)
                ts = torch.linspace(0.0, 1.0, n_s, device=q.device, dtype=dtype)
                butt_p = butt - off
                tip_p = tip - off
                samples_p = butt_p.unsqueeze(1) + ts.view(1, n_s, 1) * (tip_p - butt_p).unsqueeze(1)
                samples_e = samples_p + off
                safe = tgt.clamp(min=0)
                valid = (tgt >= 0).unsqueeze(1)
                bc = body_c[ar, safe]
                uc = under_c[ar, safe]
                axi = ax[ar, safe]
                radi = rad[ar, safe]
                hhi = hh.expand(B, D)[ar, safe]
                uhi = under_half.expand(B, D)[ar, safe]

                def _pts(center, half):
                    return point_cylinder_sdf(
                        samples_e, center.unsqueeze(1), axi.unsqueeze(1),
                        radi.unsqueeze(1), half.unsqueeze(1),
                    ) - stick_r

                dist, _axial, _din = hit_region_dist(
                    samples_p,
                    geo["disk_center"][ar, safe].unsqueeze(1),
                    geo["axis"][ar, safe].unsqueeze(1),
                    geo["hit_radius"][ar, safe].unsqueeze(1),
                    torch.zeros_like(bc).unsqueeze(1),
                    geo["hp_dir"][ar, safe].unsqueeze(1),
                    geo["has_hp"][ar, safe].unsqueeze(1),
                    self.axial_lo, self.axial_hi,
                )
                if self.cfg.contact_scoring_mode in ('isaac_v4','isaac_v5'):
                    probe_radii=torch.full((n_s,),stick_r,device=q.device,dtype=dtype)
                    if self.cfg.contact_scoring_mode=='isaac_v5':
                        probe_radii[:-1]=STICK_RADIUS
                    v4_dist,_,_=v4_static_probe_distance(
                        samples_p,heads[ar,safe]-off,
                        geo['axis'][ar,safe],rad[ar,safe],probe_radii)
                    if self.cfg.contact_scoring_mode=='isaac_v5':
                        # ts=0 is the handle butt; with stick-length variation
                        # the distal 20 cm need not start at exactly ts=.5.
                        tip_sigma=batch.tip_sigma[:,hi].to(dtype)
                        playing_start=tip_sigma/(tip_sigma+STICK_LENGTH/2)
                        v4_dist=v4_dist.masked_fill(ts[None]<playing_start[:,None],
                                                     float('inf'))
                    dist=torch.where(batch.is_cymbal[:,hi,None],v4_dist,dist)
                outside = (dist > 1e-4) & valid
                for cyl in (_pts(bc, hhi), _pts(uc, uhi)):
                    gated = torch.where(outside, cyl, big)
                    collect(gated)
                    hand_min = torch.minimum(hand_min, gated.amin(dim=1))

            s_t = segment_segment_sdf(butt, tip, torso_a, torso_b, stick_r, float(sdf.torso_r))
            s_h = segment_segment_sdf(butt, tip, head_c, head_c, stick_r, float(sdf.head_r))
            s_wt = point_capsule_sdf(wrist, torso_a, torso_b, float(sdf.torso_r)) - wrist_r
            s_wh = point_sphere_sdf(wrist, head_c, float(sdf.head_r)) - wrist_r
            s_lower_st, s_lower_wr = fixed_capsule_self_sdf(
                butt, tip, wrist, lower_a, lower_b, lower_r,
                stick_radius=stick_r, wrist_radius=wrist_r,
            )
            for dist in (s_t, s_h, s_wt, s_wh, s_lower_st, s_lower_wr):
                collect(dist)
            sdf_st = torch.minimum(torch.minimum(s_t, s_h), s_lower_st)
            sdf_wr = torch.minimum(torch.minimum(s_wt, s_wh), s_lower_wr)
            hand_min = torch.minimum(hand_min, torch.minimum(sdf_st, sdf_wr))
            # Forearm against the whole body.  The upper arm starts inside the
            # torso capsule at the shoulder, so it is checked against the head
            # and the legs only.
            for p0, p1, radius, torso in (
                (elbow, wrist, fore_r, True),
                (shoulder, elbow, upper_r, False),
            ):
                if torso:
                    arm_torso = collect(segment_segment_sdf(
                        p0, p1, torso_a, torso_b, radius, float(sdf.torso_r),
                    ))
                    hand_min = fold(hand_min, arm_torso)
                elif self.cfg.physics_loss_version == 'geometry_v2':
                    # Match trajectory checks for nonadjacent same-hand parts;
                    # the forearm/wrist connected junction remains excluded.
                    own_stick = collect(segment_segment_sdf(p0, p1, butt, tip, radius, stick_r))
                    own_wrist = collect(segment_segment_sdf(p0, p1, wrist, wrist, radius, wrist_r))
                    hand_min = fold(fold(hand_min, own_stick), own_wrist)
                arm_head = collect(segment_segment_sdf(
                    p0, p1, head_c, head_c, radius, float(sdf.head_r),
                ))
                hand_min = fold(hand_min, arm_head)
                arm_lower = collect(vs_bank(p0, p1, radius, lower_a, lower_b, lower_r))
                hand_min = fold(hand_min, arm_lower)
            min_sdf = torch.minimum(min_sdf, hand_min)

        # Cross-hand: FusedCost's stick/wrist block, plus both arm capsules.
        left, right = arms["left"], arms["right"]
        rs, rw = stick_r, wrist_r
        cross_parts = [
            segment_segment_sdf(left["butt"], left["tip"], right["butt"], right["tip"], rs, rs),
            segment_segment_sdf(left["wrist"], left["wrist"], right["butt"], right["tip"], rw, rs),
            segment_segment_sdf(right["wrist"], right["wrist"], left["butt"], left["tip"], rw, rs),
            segment_segment_sdf(left["wrist"], left["wrist"], right["wrist"], right["wrist"], rw, rw),
        ]
        for dist in cross_parts:
            collect(dist)
        cross = torch.stack(cross_parts, dim=-1).amin(dim=-1)
        left_segs = (
            (left["shoulder"], left["elbow"], upper_r),
            (left["elbow"], left["wrist"], fore_r),
        )
        right_segs = (
            (right["shoulder"], right["elbow"], upper_r),
            (right["elbow"], right["wrist"], fore_r),
        )
        for a0, a1, ra in left_segs:
            for b0, b1, rb in right_segs:
                dist = collect(segment_segment_sdf(a0, a1, b0, b1, ra, rb))
                cross = torch.minimum(cross, dist)
            dist = collect(segment_segment_sdf(
                a0, a1, right["butt"], right["tip"], ra, rs,
            ))
            cross = torch.minimum(cross, dist)
            dist = collect(segment_segment_sdf(
                a0, a1, right["wrist"], right["wrist"], ra, rw,
            ))
            cross = torch.minimum(cross, dist)
        for b0, b1, rb in right_segs:
            dist = collect(segment_segment_sdf(
                b0, b1, left["butt"], left["tip"], rb, rs,
            ))
            cross = torch.minimum(cross, dist)
            dist = collect(segment_segment_sdf(
                b0, b1, left["wrist"], left["wrist"], rb, rw,
            ))
            cross = torch.minimum(cross, dist)
        min_sdf = torch.minimum(min_sdf, cross)

        all_sdf = torch.cat(collision_sdfs, dim=-1)
        v_col = aggregate_collision_violation(
            all_sdf,
            allow_m=self.cfg.collision_allow_m,
            topk=self.cfg.collision_topk,
        )
        return v_col, min_sdf


def pad_elevation_shortfall(elev_rad: torch.Tensor, min_deg: float = _ELEV_DRUM_MIN_DEG) -> torch.Tensor:
    """Radians by which a pad stick is flatter than ``min_deg`` above the head.

    Elevation is signed: positive points into the head.  The floor is
    ``elev >= min_deg``, so a stick lying on the head or pointing up off it
    pays.  Zero at and above the floor.
    """
    floor = elev_rad.new_tensor(float(np.deg2rad(min_deg)))
    return (floor - elev_rad).clamp_min(0.0)


def collision_violation(min_sdf: torch.Tensor, allow_m: float = _COLLISION_ALLOW_M) -> torch.Tensor:
    """Penetration beyond ``allow_m`` metres.  Zero when overlap is shallower."""
    return (-min_sdf - float(allow_m)).clamp_min(0.0)


def aggregate_collision_violation(
    signed_distances: torch.Tensor,
    *,
    allow_m: float = _COLLISION_ALLOW_M,
    topk: int = 8,
) -> torch.Tensor:
    """Worst residual plus a smaller mean of other penetrating contacts.

    Unlike a single hard minimum, this sends gradient through several
    simultaneous contacts.  The leading residual retains its full magnitude,
    so adding contacts cannot weaken the original collision constraint.
    """
    if signed_distances.ndim != 2:
        raise ValueError("signed_distances must have shape (B, pairs)")
    k = min(max(int(topk), 1), int(signed_distances.shape[1]))
    residual = collision_violation(signed_distances, allow_m)
    worst = residual.topk(k, dim=-1).values
    primary = worst[:, 0]
    if k == 1:
        return primary
    secondary = worst[:, 1:]
    count = (secondary > 0).sum(dim=-1).clamp_min(1).to(worst.dtype)
    return primary + 0.25 * secondary.sum(dim=-1) / count


def TerminalPoseAL(
    *,
    lam_init: float = 1.0,
    rho_hit: float = 40.0,
    rho_collision: float = 80.0,
    pen_hit: float = 1.0e3,
    pen_collision: float = 1.0e4,
    pen_max_hit: float = 1.0e4,
    pen_max_collision: float = 1.0e5,
    pen_grow: float = 1.0,
    pen_shrink: float = 1.2,
    pen_patience: int = 10,
    improve_tol: float = 0.9,
    ema: float = 0.9,
    lam_max: float = 100.0,
    target: float = 0.0,
) -> AugmentedLagrangian:
    """Augmented Lagrangian with groups ``(\"hit\", \"collision\")`` only.

    Penalties are sized so ``0.5 * pen * V^2`` is about 1 at a 1 cm residual.
    The caps match the trajectory optimizer (hit 10^4, collision 10^5) and
    ``pen_grow`` is 1, so a chronic residual cannot multiply them up to 10^12.
    ``lam_max`` is 100, the same safety clip as that optimizer.
    """
    groups = ("hit", "collision")
    return AugmentedLagrangian(
        ALConfig(
            groups=groups,
            lam_init=float(lam_init),
            rho={"hit": float(rho_hit), "collision": float(rho_collision)},
            pen={"hit": float(pen_hit), "collision": float(pen_collision)},
            pen_max={"hit": float(pen_max_hit), "collision": float(pen_max_collision)},
            pen_grow=float(pen_grow),
            pen_shrink=float(pen_shrink),
            pen_patience=int(pen_patience),
            improve_tol=float(improve_tol),
            ema=float(ema),
            lam_max=float(lam_max),
            target=float(target),
        )
    )
