"""Tokenizer / context builder for the amortized-optimizer transformer.

Section 4 of ``docs/amortized_optimizer_plan.md``. Pure tensor plumbing: no
learned parameters live here (the input projections belong to :mod:`model`),
no device->host syncs, no data-dependent Python branching, static shapes at a
fixed ``max_horizon``.

Three token families, matching the plan's "Input tokens" section:

* **position-context token** -- one token carrying all 17 DoF position AND
  velocity of the current state (34 values);
* **context tokens** -- one per drum (position / height / radius / tilt), plus
  a current-goal token per arm and ``num_future`` future-hit tokens per arm.
  Encoded once per context change and cached (:class:`ContextInputs`);
* **initialization tokens** -- up to ``max_horizon`` waypoints, each the full
  17-DoF joint position of the current trajectory guess, evenly spaced at
  ``CONTROL_DT``.

Temporal encoding is TWO Fourier encodings per waypoint, one per hand, each
anchored at that hand's terminal (hit) waypoint: the encoded quantity is
``t_hit_hand - t_waypoint`` in seconds, so a token sitting exactly on a hand's
hit gets the canonical zero-phase code regardless of its integer index.

Masking is **per element** (resolved decision 2): within a waypoint token the
right-arm joints are masked by the right hand's horizon, the left-arm joints by
the left hand's, and the waist by the LONGER of the two. A token past both
horizons is fully masked and is additionally dropped from attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF

#: Auxiliary FK points predicted per waypoint, in head order.
FK_POINT_NAMES: tuple[str, ...] = (
    "left_tip",
    "left_wrist",
    "left_elbow",
    "right_tip",
    "right_wrist",
    "right_elbow",
)
N_FK_POINTS: int = len(FK_POINT_NAMES)

_EPS_T = 1e-9


# ------------------------------------------------------------------ groups --
def joint_group_masks(joint_names=G1_UPPER_BODY_JOINTS) -> dict[str, np.ndarray]:
    """(waist, left, right) 0/1 selector rows over ``joint_names``.

    Derived from the names rather than assumed, so a reordering of
    ``G1_UPPER_BODY_JOINTS`` cannot silently mis-mask an arm.
    """
    waist = np.array([n.startswith("waist_") for n in joint_names], dtype=np.float64)
    left = np.array([n.startswith("left_") for n in joint_names], dtype=np.float64)
    right = np.array([n.startswith("right_") for n in joint_names], dtype=np.float64)
    total = waist + left + right
    if not np.all(total == 1.0):
        bad = [n for n, t in zip(joint_names, total) if t != 1.0]
        raise ValueError(f"joints not classifiable as waist/left/right: {bad}")
    return {"waist": waist, "left": left, "right": right}


#: Index of the FK points belonging to each hand (same order as FK_POINT_NAMES).
FK_POINT_HAND: tuple[str, ...] = tuple(n.split("_")[0] for n in FK_POINT_NAMES)


# ----------------------------------------------------------------- fourier --
def fourier_frequencies(n_bands: int, t_min: float, t_max: float) -> np.ndarray:
    """Angular frequencies for ``n_bands`` geometrically spaced periods."""
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    if n_bands == 1:
        periods = np.array([t_max], dtype=np.float64)
    else:
        periods = np.geomspace(float(t_min), float(t_max), int(n_bands))
    return 2.0 * math.pi / periods


def fourier_dim(n_bands: int) -> int:
    """Width of one hand's temporal code: sin/cos per band plus a linear ramp."""
    return 2 * int(n_bands) + 1


def fourier_time_features(t: torch.Tensor, freqs: torch.Tensor, t_scale: float) -> torch.Tensor:
    """(..., ) seconds -> (..., 2K+1) terminal-anchored temporal code.

    ``t`` is time-to-hit: positive before the strike, exactly 0 on it, negative
    after. At ``t == 0`` the code is the zero-phase vector
    ``[0]*K + [1]*K + [0]`` for every band, which is what anchors the encoding
    at the terminal waypoint instead of the first one.
    """
    x = t.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(x), torch.cos(x), (t / t_scale).unsqueeze(-1)], dim=-1)


def zero_phase_features(n_bands: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """The exact code :func:`fourier_time_features` emits at ``t == 0``."""
    k = int(n_bands)
    out = torch.zeros(fourier_dim(k), device=device, dtype=dtype)
    out[k : 2 * k] = 1.0
    return out


# ------------------------------------------------------------------- masks --
@dataclass
class WaypointMasks:
    """Per-hand horizons resolved onto the fixed waypoint grid.

    ``tau_*`` is time-to-hit per waypoint (seconds, positive before the strike).
    ``elem_mask`` is 1.0 where a joint element of a waypoint is inside its
    owning hand's horizon; ``token_mask`` is True where a token is inside the
    longer horizon (identical to the waist row of ``elem_mask``).
    """

    tau_left: torch.Tensor  # (B, H)
    tau_right: torch.Tensor  # (B, H)
    tau0_left: torch.Tensor  # (B,)  time-to-hit at the current state
    tau0_right: torch.Tensor  # (B,)
    elem_mask: torch.Tensor  # (B, H, N_DOF) float
    token_mask: torch.Tensor  # (B, H) bool
    point_mask: torch.Tensor  # (B, H, N_FK_POINTS) float
    #: ``(B, 2)`` seconds: each hand's OWN scheduled strike, ``[:, 0]`` = left.
    #: This is the current-goal countdown, carried on the masks so it reaches
    #: the DECODER (``ModelConfig.dec_tth_feats``) without any producer changing
    #: its call signature -- every masks producer in the tree goes through
    #: :func:`build_waypoint_masks`.  It moved off the encoder because a from-now
    #: countdown ticks down every 20 ms and forced the deployment MPC to
    #: re-encode the context every tick (``isaac_rl/README.md`` DECISION 2),
    #: which is what let robot state echo back into the encoder.
    #:
    #: It is NOT ``tau0_*``: those are the RESOLVED horizons, so an idle hand's
    #: ``tau0`` is the other hand's horizon, borrowed for masking.  ``hit_tth``
    #: is exactly 0 on a hand with no scheduled strike, and an active hand's
    #: hit is never nearer than one waypoint, so 0 is unambiguously "idle".
    hit_tth: torch.Tensor  # (B, 2)

    def to(self, device) -> "WaypointMasks":
        return WaypointMasks(**{k: getattr(self, k).to(device) for k in self.__dataclass_fields__})


#: Constant selector rows, cached per (device, dtype) so the hot path does no
#: host->device copies (and no syncs).
_SEL_CACHE: dict[tuple, torch.Tensor] = {}


def _group_selector(joint_names: tuple[str, ...], device, dtype) -> torch.Tensor:
    key = ("g", joint_names, str(device), dtype)
    t = _SEL_CACHE.get(key)
    if t is None:
        g = joint_group_masks(joint_names)
        t = torch.as_tensor(
            np.stack([g["waist"], g["left"], g["right"]], axis=0), device=device, dtype=dtype
        )
        _SEL_CACHE[key] = t
    return t


def _point_selector(device, dtype) -> torch.Tensor:
    key = ("p", FK_POINT_HAND, str(device), dtype)
    t = _SEL_CACHE.get(key)
    if t is None:
        t = torch.as_tensor(
            [1.0 if h == "left" else 0.0 for h in FK_POINT_HAND], device=device, dtype=dtype
        )
        _SEL_CACHE[key] = t
    return t


def waypoint_times(
    max_horizon: int, dt: float, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Times of the waypoint tokens, ``dt .. max_horizon*dt`` (the current
    state is a separate token at t = 0)."""
    return (torch.arange(1, int(max_horizon) + 1, device=device, dtype=dtype)) * float(dt)


def build_waypoint_masks(
    hit_time_left: torch.Tensor,
    hit_time_right: torch.Tensor,
    *,
    max_horizon: int,
    dt: float = CONTROL_DT,
    active_left: torch.Tensor | None = None,
    active_right: torch.Tensor | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    eps: float = 1e-6,
) -> WaypointMasks:
    """Resolve two per-hand hit times (seconds from now) into element masks.

    An *inactive* hand (no scheduled strike) has no terminal of its own; its
    elements are predicted over the other hand's horizon, and if neither hand
    is active the whole grid is used. That keeps an idle arm under the model's
    control instead of dropping it from the loss.

    ``hit_tth`` reports the times as PASSED for an active hand and exactly 0 for
    an inactive one.  The gate is here rather than left to the consumer because
    the callers disagree about what they pass for an idle hand -- the deployment
    MPCs pass ``hit_index.clamp_min(0) * dt`` (0) while ``MPCRoller.masks``
    passes ``effective_hits`` (the BORROWED horizon) -- and the decoder, which
    is the only consumer, has no activity mask of its own to gate with.  Ungated
    the same idle hand would carry a different decoder input in training than at
    deployment, which is the silent distribution bug this whole seam exists to
    avoid.
    """
    tl = hit_time_left
    tr = hit_time_right
    ones = torch.ones_like(tl, dtype=torch.bool)
    al = ones if active_left is None else active_left.to(torch.bool)
    ar = ones if active_right is None else active_right.to(torch.bool)

    full = torch.full_like(tl, float(max_horizon) * float(dt))
    both = torch.maximum(torch.where(al, tl, -full), torch.where(ar, tr, -full))
    neither = ~(al | ar)
    both = torch.where(neither, full, both)
    hl = torch.where(al, tl, both)
    hr = torch.where(ar, tr, both)

    t = waypoint_times(max_horizon, dt, device=tl.device, dtype=tl.dtype)  # (H,)
    tau_l = hl.unsqueeze(-1) - t  # (B, H)
    tau_r = hr.unsqueeze(-1) - t

    valid_l = (tau_l >= -eps).to(tl.dtype)
    valid_r = (tau_r >= -eps).to(tl.dtype)
    valid_any = torch.maximum(valid_l, valid_r)

    sel = _group_selector(tuple(joint_names), tl.device, tl.dtype)  # (3, N)
    elem = (
        valid_any.unsqueeze(-1) * sel[0]
        + valid_l.unsqueeze(-1) * sel[1]
        + valid_r.unsqueeze(-1) * sel[2]
    )

    hand_is_left = _point_selector(tl.device, tl.dtype)
    point = valid_l.unsqueeze(-1) * hand_is_left + valid_r.unsqueeze(-1) * (1.0 - hand_is_left)

    return WaypointMasks(
        tau_left=tau_l,
        tau_right=tau_r,
        tau0_left=hl,
        tau0_right=hr,
        elem_mask=elem,
        token_mask=valid_any > 0.5,
        point_mask=point,
        hit_tth=torch.stack(
            [torch.where(al, tl, torch.zeros_like(tl)),
             torch.where(ar, tr, torch.zeros_like(tr))], dim=-1
        ),
    )


def pack_state(q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """(B, 17) + (B, 17) -> the (B, 34) position-context token payload."""
    return torch.cat([q, qd], dim=-1)


# ----------------------------------------------------------------- context --
#: Per-drum geometry token payload width: position(3) radius height axis(3) tilt.
DRUM_FEAT_DIM: int = 9
#: Goal / future-hit token payload width (see :func:`context_features`).
GOAL_RAW_DIM: int = 18  # hit pos(3), hit pos rel head(3), vel(3), speed, geom(8)
#: IDLE-CYLINDER token payload width: packed torso-frame safety cylinder.
#: ``center(3) + axis(3) + radius + half_height`` -- see
#: :class:`~amortized_opt.idle_cylinder.IdleCylinder.pack`.  Same width for
#: both hands; a busy hand's row is zeroed via ``idle_valid``.
IDLE_RAW_DIM: int = 8
#: Stick token payload: axis (3) + tip offset (3) + length (1).
STICK_RAW_DIM: int = 7
#: Arm joints per hand -- kept for callers that still reason about arm width.
N_ARM_JOINTS: int = 7


@dataclass
class ContextInputs:
    """Everything the context encoder sees, batched and padded to fixed sizes.

    Drums are padded to ``num_drums`` rows; ``drum_valid`` drops the padding
    from attention. Goal / future rows are indexed ``[:, 0]`` = left hand,
    ``[:, 1]`` = right hand.
    """

    drum_pos: torch.Tensor  # (B, D, 3)
    drum_axis: torch.Tensor  # (B, D, 3) unit head normal (tilt)
    drum_radius: torch.Tensor  # (B, D)
    drum_height: torch.Tensor  # (B, D)
    drum_valid: torch.Tensor  # (B, D) bool

    goal_drum: torch.Tensor  # (B, 2) long, index into the drum rows (>=0)
    goal_pos: torch.Tensor  # (B, 2, 3) hit position on the head
    goal_vel: torch.Tensor  # (B, 2, 3) commanded hit velocity
    goal_tth: torch.Tensor  # (B, 2) time-to-hit, seconds (relative)
    goal_valid: torch.Tensor  # (B, 2) bool

    future_drum: torch.Tensor  # (B, 2, F) long
    future_pos: torch.Tensor  # (B, 2, F, 3)
    future_vel: torch.Tensor  # (B, 2, F, 3)
    #: HIT-RELATIVE future timing: the gap from this hand's CURRENT goal strike
    #: to future hit ``j``, seconds -- not a time-to-hit from now.  A from-now
    #: quantity ticks down every 20 ms and forces the deployment MPC to re-encode
    #: the whole context every tick (``isaac_rl/README.md`` DECISION 2), which is
    #: what let robot state echo back into the encoder and ran the waist away in
    #: closed loop.  The gap is CONSTANT inside a goal segment, it is the
    #: musically meaningful number ("how long to relocate after this hit"), and
    #: it is what pose selection at the strike actually needs.  A hand with NO
    #: active current goal has no strike to measure from and falls back to
    #: anchoring at now, so ``future_gap[..., 0]`` is then its plain
    #: time-to-hit -- which is exactly what the IDLE token reads it as.
    future_gap: torch.Tensor  # (B, 2, F)
    future_valid: torch.Tensor  # (B, 2, F) bool

    # ---- the IDLE arm's "goal" ---------------------------------------------
    #: A hand with no scheduled strike has no goal token -- its row is zeroed and
    #: masked out of the encoder entirely, so the only thing the context says
    #: about it is *absence*.  These fields give it a goal of its own: the
    #: torso-frame safety cylinder its tip must stay inside (see
    #: :mod:`~amortized_opt.idle_cylinder`).  ``idle_valid`` is normally
    #: ``~goal_valid``, but it is carried explicitly rather than derived so a
    #: caller with no cylinder simply leaves it False and the token is absent.
    idle_cyl: torch.Tensor = None         # (B, 2, 8) packed IdleCylinder
    idle_valid: torch.Tensor = None       # (B, 2) bool
    #: THE STICK.  Per-hand geometry of the stick actually fitted this sample
    #: (:class:`~amortized_opt.fk.StickSpec`).  Constant across a run until
    #: ``StickRandomization`` is switched on; carried in the context because a
    #: randomized stick the model is not TOLD about is unlearnable noise -- the
    #: same trap the drum jitter fell into.  ``stick_valid`` is False when the
    #: caller has no per-sample stick, and the token is then masked out.
    stick_axis: torch.Tensor = None       # (B, 2, 3) handle->tip, wrist frame
    stick_tip: torch.Tensor = None        # (B, 2, 3) tip offset, wrist frame
    stick_length: torch.Tensor = None     # (B, 2) metres
    stick_valid: torch.Tensor = None      # (B, 2) bool
    #: THE WAIST-YAW SET-POINT INPUTS (``ModelConfig.waist_sp_head``).
    #: ``yaw_now`` is the waist yaw at the moment the encoder runs; ``aim_yaw``
    #: is the analytic :func:`~amortized_opt.waist_aim.aim_yaw` command computed
    #: from the ACTIVE current goals at that yaw.  The head predicts a residual
    #: ON TOP of ``aim_yaw``, so the untrained model is exactly the analytic
    #: law.  ``aim_valid`` False means the caller supplied neither (an old pool
    #: or a call site without robot state); the head then falls back to
    #: ``yaw_now``, i.e. "hold where you are".
    yaw_now: torch.Tensor = None          # (B,) rad
    aim_yaw: torch.Tensor = None          # (B,) rad
    aim_valid: torch.Tensor = None        # (B,) bool

    def __post_init__(self) -> None:
        """Materialize the idle block when a caller omits it.

        ``CONTEXT_FIELDS`` is derived from these dataclass fields and the data
        pipeline iterates it with ``getattr(ctx, f)``, so every field has to be a
        real tensor.  Defaulting to zeros + ``idle_valid = False`` is exactly
        "this context carries no idle-cylinder target", which is what a caller
        without a cylinder means -- and it keeps every pre-existing call site
        (and every pool cached before the field existed) working unchanged.
        """
        ref = self.drum_pos
        B = int(ref.shape[0])
        f = dict(device=ref.device, dtype=ref.dtype)
        if self.idle_cyl is None:
            self.idle_cyl = torch.zeros(B, 2, IDLE_RAW_DIM, **f)
        if self.idle_valid is None:
            self.idle_valid = torch.zeros(B, 2, device=ref.device, dtype=torch.bool)
        if self.stick_axis is None:
            self.stick_axis = torch.zeros(B, 2, 3, **f)
        if self.stick_tip is None:
            self.stick_tip = torch.zeros(B, 2, 3, **f)
        if self.stick_length is None:
            self.stick_length = torch.zeros(B, 2, **f)
        if self.stick_valid is None:
            self.stick_valid = torch.zeros(B, 2, device=ref.device, dtype=torch.bool)
        if self.yaw_now is None:
            self.yaw_now = torch.zeros(B, **f)
        if self.aim_yaw is None:
            self.aim_yaw = torch.zeros(B, **f)
        if self.aim_valid is None:
            self.aim_valid = torch.zeros(B, device=ref.device, dtype=torch.bool)

    @property
    def batch(self) -> int:
        return int(self.drum_pos.shape[0])

    @property
    def num_drums(self) -> int:
        return int(self.drum_pos.shape[1])

    @property
    def num_future(self) -> int:
        return int(self.future_pos.shape[2])

    def to(self, device) -> "ContextInputs":
        return ContextInputs(
            **{k: getattr(self, k).to(device) for k in self.__dataclass_fields__}
        )

    @classmethod
    def zeros(
        cls,
        batch: int,
        *,
        num_drums: int,
        num_future: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> "ContextInputs":
        f = dict(device=device, dtype=dtype)
        b = dict(device=device, dtype=torch.bool)
        return cls(
            drum_pos=torch.zeros(batch, num_drums, 3, **f),
            drum_axis=torch.zeros(batch, num_drums, 3, **f),
            drum_radius=torch.zeros(batch, num_drums, **f),
            drum_height=torch.zeros(batch, num_drums, **f),
            drum_valid=torch.zeros(batch, num_drums, **b),
            goal_drum=torch.zeros(batch, 2, device=device, dtype=torch.long),
            goal_pos=torch.zeros(batch, 2, 3, **f),
            goal_vel=torch.zeros(batch, 2, 3, **f),
            goal_tth=torch.zeros(batch, 2, **f),
            goal_valid=torch.zeros(batch, 2, **b),
            future_drum=torch.zeros(batch, 2, num_future, device=device, dtype=torch.long),
            future_pos=torch.zeros(batch, 2, num_future, 3, **f),
            future_vel=torch.zeros(batch, 2, num_future, 3, **f),
            future_gap=torch.zeros(batch, 2, num_future, **f),
            future_valid=torch.zeros(batch, 2, num_future, **b),
            idle_cyl=torch.zeros(batch, 2, IDLE_RAW_DIM, **f),
            idle_valid=torch.zeros(batch, 2, **b),
            stick_axis=torch.zeros(batch, 2, 3, **f),
            stick_tip=torch.zeros(batch, 2, 3, **f),
            stick_length=torch.zeros(batch, 2, **f),
            stick_valid=torch.zeros(batch, 2, **b),
        )


def drum_tensors_from_scene(
    scene,
    *,
    batch: int = 1,
    num_drums: int | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Padded drum-geometry tensors from a :class:`~amortized_opt.sdf.DrumSceneSpec`.

    Row ``i`` is drum id ``i`` (``DrumSceneSpec.from_layout`` already sorts by
    the layout's ``id``), so ``goal_drum`` indices line up with the SDF's
    ``target_drum_*`` exemption indices.
    """
    d = int(scene.num_drums)
    D = d if num_drums is None else int(num_drums)
    if D < d:
        raise ValueError(f"num_drums={D} cannot hold {d} drums")
    pos = torch.zeros(D, 3, device=device, dtype=dtype)
    axis = torch.zeros(D, 3, device=device, dtype=dtype)
    radius = torch.zeros(D, device=device, dtype=dtype)
    height = torch.zeros(D, device=device, dtype=dtype)
    valid = torch.zeros(D, device=device, dtype=torch.bool)
    pos[:d] = torch.as_tensor(np.asarray(scene.head_centers), device=device, dtype=dtype)
    axis[:d] = torch.as_tensor(np.asarray(scene.axes), device=device, dtype=dtype)
    radius[:d] = torch.as_tensor(np.asarray(scene.radii), device=device, dtype=dtype)
    height[:d] = torch.as_tensor(np.asarray(scene.heights), device=device, dtype=dtype)
    valid[:d] = True
    rep = lambda x: x.unsqueeze(0).expand(batch, *x.shape).contiguous()  # noqa: E731
    return {
        "drum_pos": rep(pos),
        "drum_axis": rep(axis),
        "drum_radius": rep(radius),
        "drum_height": rep(height),
        "drum_valid": rep(valid),
    }


def _tilt_angle(axis: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between a head normal and +Z, safe at the poles."""
    return torch.acos(axis[..., 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def drum_features(ctx: ContextInputs) -> torch.Tensor:
    """(B, D, DRUM_FEAT_DIM) geometry payload, zeroed on padded rows."""
    feat = torch.cat(
        [
            ctx.drum_pos,
            ctx.drum_radius.unsqueeze(-1),
            ctx.drum_height.unsqueeze(-1),
            ctx.drum_axis,
            _tilt_angle(ctx.drum_axis).unsqueeze(-1),
        ],
        dim=-1,
    )
    return feat * ctx.drum_valid.unsqueeze(-1).to(feat.dtype)


def _gather_drum_geometry(ctx: ContextInputs, idx: torch.Tensor) -> torch.Tensor:
    """Geometry (pos3, radius, height, axis3) of ``idx``, shaped like ``idx`` + (8,)."""
    geom = torch.cat(
        [
            ctx.drum_pos,
            ctx.drum_radius.unsqueeze(-1),
            ctx.drum_height.unsqueeze(-1),
            ctx.drum_axis,
        ],
        dim=-1,
    )  # (B, D, 8)
    flat = idx.reshape(idx.shape[0], -1).clamp(min=0)
    out = torch.gather(geom, 1, flat.unsqueeze(-1).expand(-1, -1, geom.shape[-1]))
    return out.reshape(*idx.shape, geom.shape[-1])


def goal_features(
    ctx: ContextInputs,
    pos: torch.Tensor,
    vel: torch.Tensor,
    drum: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """(..., GOAL_RAW_DIM) payload shared by the goal and future-hit tokens."""
    geom = _gather_drum_geometry(ctx, drum)
    rel = pos - geom[..., 0:3]
    speed = vel.norm(dim=-1, keepdim=True)
    feat = torch.cat([pos, rel, vel, speed, geom], dim=-1)
    return feat * valid.unsqueeze(-1).to(feat.dtype)


def idle_features(ctx: ContextInputs) -> torch.Tensor:
    """``(B, 2, IDLE_RAW_DIM)`` idle-cylinder payload, zeroed on hands that are busy.

    The idle counterpart of :func:`goal_features`: where a goal token says "put
    the stick HERE, moving THAT fast, at time t", the idle token says "keep the
    tip inside this torso-frame cylinder".  Zeroed (and, in the encoder, masked
    out) for any hand that has a scheduled strike, so a hand can never be handed
    both a strike target and a park target at once.
    """
    return ctx.idle_cyl * ctx.idle_valid.unsqueeze(-1).to(ctx.idle_cyl.dtype)


def stick_features(ctx: ContextInputs) -> torch.Tensor:
    """``(B, 2, STICK_RAW_DIM)`` stick payload, zeroed where invalid.

    Axis and tip offset are in the WRIST frame, which is where they are
    constant for a given stick -- expressing them in the world would make the
    token a function of the arm pose and tell the model nothing about the
    hardware.  Length is carried explicitly even though it is implied by the
    tip offset: it is the quantity that changes the lever arm, and the decoder
    should not have to infer it through a norm.
    """
    feat = torch.cat(
        [ctx.stick_axis, ctx.stick_tip, ctx.stick_length.unsqueeze(-1)], dim=-1
    )
    return feat * ctx.stick_valid.unsqueeze(-1).to(feat.dtype)
