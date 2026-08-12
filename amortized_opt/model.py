"""Transformer for the amortized optimizer: context encoder + trajectory decoder.

Section 4 of ``docs/amortized_optimizer_plan.md``.

* :class:`ContextEncoder` -- 2-3 bidirectional layers over the ~15-30 context
  tokens (drum geometry, per-arm current goal, per-arm future hits, with type
  and arm embeddings). Runs **once per context change**; its output is cached
  and consumed by the decoder through cross-attention.
* :class:`TrajectoryDecoder` -- 6 pre-LN RMSNorm layers (d=384, 8 heads,
  head_dim 48, SwiGLU FFN 4x) with bidirectional self-attention over
  ``[position-context token] + max_horizon waypoint tokens`` and cross-attention
  to the cached context. Emits ONE 17-DoF trajectory (resolved decision 2) plus
  the auxiliary heads.
* :class:`AmortizedOptimizer` -- the iterative interface. One call = one
  refinement iteration; feed the output back in as ``init_traj`` for the next.

Everything on the hot path is static-shape, branch-free and sync-free so
``torch.compile(fullgraph=True)`` works and CUDA graphs can be captured.

DECISIONS NEEDING REVIEW (conservative + reversible choices the plan leaves
open; all are single-line config changes):

1. ``fourier_bands = 8`` over geometrically spaced periods 0.02 s .. 4.0 s, plus
   a linear ramp -> 17 dims per hand, 34 per waypoint token.
2. ``num_future = 3`` future-hit tokens per arm (16 context tokens total with 8
   drum slots, inside the plan's 15-30).
3. The position-context token joins the decoder's self-attention sequence at
   index 0 (it is always visible; heads read only the waypoint tokens).
4. The torque head's target is defined as the per-segment max |tau| from
   RNEA + PD along the spline segment ENDING at that waypoint, in N*m.
5. An idle hand (no scheduled strike) borrows the other hand's horizon so its
   joints stay inside the mask instead of dropping out of the loss.
6. Heads emit physical units through fixed scale buffers (joint box, velocity
   and effort limits from ``limits.py``; accel 30 rad/s^2, tip pos 1 m, tip vel
   10 m/s).
7. SwiGLU hidden width is ``ffn_mult * d`` (i.e. a literal 4x, three matrices),
   not the 2/3-scaled variant.
8. The encoder uses the same SwiGLU/RMSNorm block as the decoder.
9. Drum identity is carried by a learned ``num_drums`` embedding on the drum,
   goal and future tokens, on top of the raw geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF, ready_joint_pos
from .limits import isaac_effort_limits, isaac_velocity_limits, urdf_position_limits
from .tokenizer import (
    DRUM_FEAT_DIM,
    FK_POINT_NAMES,  # noqa: F401  (re-exported: the fk head's point order)
    GOAL_RAW_DIM,
    N_FK_POINTS,
    ContextInputs,
    WaypointMasks,
    IDLE_RAW_DIM,
    drum_features,
    fourier_dim,
    fourier_frequencies,
    fourier_time_features,
    goal_features,
    idle_features,
)

#: Context token type ids (also the row order the encoder builds).
#: ``TYPE_IDLE`` is APPEND-ONLY at the end so the first three embedding rows of
#: an existing checkpoint keep their meaning.
TYPE_DRUM, TYPE_GOAL, TYPE_FUTURE, TYPE_IDLE = 0, 1, 2, 3
N_TOKEN_TYPES = 4


@dataclass
class ModelConfig:
    """Shapes and widths. Depth / width / heads are all configurable; the
    defaults are the plan's (encoder 2L d=256, decoder 6L d=384 h=8)."""

    n_dof: int = N_DOF
    max_horizon: int = 100
    dt: float = CONTROL_DT
    num_drums: int = 8
    num_future: int = 3

    d_ctx: int = 256
    ctx_layers: int = 2
    ctx_heads: int = 8
    ctx_ffn_mult: float = 4.0

    d_model: int = 384
    #: 3 decoder layers (10.8 M total) is the section-8 ablation's sweet spot:
    #: 63 % grid acceptance at 1.71x faster inference than the 6-layer / 19.3 M
    #: decoder, with no sign of underfitting.  The champion checkpoint
    #: (``runs/phase2b_main/best.pt``) is a 6-layer model and still loads --
    #: every eval / deploy entry point rebuilds this config from the
    #: checkpoint's stored ``model_cfg``, so the default only governs NEW runs.
    dec_layers: int = 3
    dec_heads: int = 8
    dec_ffn_mult: float = 4.0

    fourier_bands: int = 8
    fourier_t_min: float = 0.02
    fourier_t_max: float = 4.0

    #: "direct" -> the head emits the trajectory; "residual" -> it emits a
    #: correction added to ``init_traj`` (plan variation; zero-initialised so
    #: the untrained model is the identity refinement).
    output_mode: str = "direct"
    predict_aux: bool = True
    dropout: float = 0.0

    #: THE IDLE ARM'S OWN GOAL TOKEN.  Adds two context rows (one per hand),
    #: carrying the park posture from :mod:`~amortized_opt.idle_home`,
    #: valid exactly on the hands with NO scheduled strike.  Without it the only
    #: thing the context says about an idle hand is *absence* -- its goal token is
    #: zeroed and masked out of attention -- so "where should this arm be" has to
    #: be inferred from the seed alone.
    #:
    #: OFF makes the encoder bit-identical to the pre-2026-08-12 model, including
    #: the ``type_emb`` row count, so existing checkpoints load strictly.  ON adds
    #: one ``idle_proj`` and one ``type_emb`` row -- a checkpoint trained without
    #: it will report those as missing keys and they start random.
    idle_home_token: bool = True

    #: Output scaling (physical units come out of the heads directly).
    accel_scale: float = 30.0
    fk_pos_scale: float = 1.0
    fk_vel_scale: float = 10.0

    #: Joint NAMES the decoder does not control.  Their trajectory columns are
    #: replaced -- after the head, before anything downstream sees them -- by the
    #: READY pose (``constants.G1_READY_JOINT_OFFSETS``, which is 0 for the two
    #: waist lean joints), and their velocity / acceleration / torque aux rows
    #: are zeroed to match.  This is a HARDER statement than any cost weight:
    #: there is no price to trade against, the DOF simply is not an actuator the
    #: planner owns, and no gradient reaches the head through it.
    #:
    #: Applied inside :class:`TrajectoryDecoder`, so every consumer -- training,
    #: the roller, the closed-loop probe, ``mpc_gen``, the Isaac player -- gets
    #: it for free, and it round-trips in the checkpoint's ``model_cfg``.
    #:
    #: NOTE the plant is not frozen, only the COMMAND: a scenario whose start
    #: state has waist pitch/roll off zero still has to travel back, over one
    #: control period, and pays the velocity / torque / box terms for doing so.
    frozen_joints: tuple[str, ...] = ()
    #: Waypoints over which a frozen column travels from the MEASURED start pose
    #: to the ready pose before holding.  A step (``1``) is physically the wrong
    #: request -- see the note in :meth:`TrajectoryDecoder.forward`.
    frozen_ramp_wp: int = 5

    def __post_init__(self) -> None:
        # tuple-ise: a checkpoint round-trips this through JSON/dict as a list,
        # and a bare string would silently freeze one column per CHARACTER.
        if isinstance(self.frozen_joints, str):
            self.frozen_joints = (self.frozen_joints,)
        self.frozen_joints = tuple(self.frozen_joints)
        unknown = [n for n in self.frozen_joints if n not in G1_UPPER_BODY_JOINTS]
        if unknown:
            raise ValueError(f"frozen_joints not in the joint order: {unknown}")
        if self.d_model % self.dec_heads:
            raise ValueError("d_model must divide dec_heads")
        if self.d_ctx % self.ctx_heads:
            raise ValueError("d_ctx must divide ctx_heads")
        if self.output_mode not in ("direct", "residual"):
            raise ValueError(f"output_mode must be direct|residual, got {self.output_mode!r}")

    @property
    def dec_head_dim(self) -> int:
        return self.d_model // self.dec_heads

    @property
    def fourier_width(self) -> int:
        return fourier_dim(self.fourier_bands)

    @property
    def n_context_tokens(self) -> int:
        return self.num_drums + 2 + 2 * self.num_future + (2 if self.idle_home_token else 0)

    @property
    def n_token_types(self) -> int:
        """Rows of ``type_emb``.  Pinned to 3 when the idle token is off so the
        embedding SHAPE matches every checkpoint written before it existed."""
        return N_TOKEN_TYPES if self.idle_home_token else N_TOKEN_TYPES - 1


#: Architecture flags that a checkpoint's stored ``model_cfg`` may not carry,
#: because the field was added to :class:`ModelConfig` after it was written,
#: mapped to a state-dict key that PROVES whether the trained weights have the
#: feature.
#:
#: Append to this whenever a new field changes a parameter SHAPE.  A field that
#: only changes behaviour (``frozen_joints``, ``output_mode``) does not belong
#: here: those round-trip in ``model_cfg`` and have safe defaults.
ARCH_FROM_WEIGHTS: dict[str, str] = {
    "idle_home_token": "encoder.idle_proj.weight",
}


def model_config_from_checkpoint(ck: dict, **overrides) -> ModelConfig:
    """A checkpoint's :class:`ModelConfig`, reconciled against its own weights.

    **Use this instead of ``ModelConfig(**ck["model_cfg"])``.**  That expression
    fills any field the stored config lacks from the CURRENT default, so a
    feature that defaults ON reconstructs an architecture the checkpoint never
    had; the strict load that follows then dies on a shape mismatch.  That is
    exactly what happened on 2026-08-12, when ``idle_home_token`` began
    defaulting to True and widened ``encoder.type_emb`` from 3 rows to 4 --
    every checkpoint written before that date became unloadable by every entry
    point in the tree at once.

    Trusting the WEIGHTS over the stored config (:data:`ARCH_FROM_WEIGHTS`) is
    the only thing that stays correct while the package is edited underneath a
    long run, which is the normal condition here.  Keys the installed
    ``ModelConfig`` no longer has are dropped with a note rather than raising
    ``TypeError``: a drop that really did move a parameter shape still fails,
    loudly, at ``load_state_dict``.

    ``overrides`` win over both, for a caller that is deliberately changing the
    architecture (``train_phase2.py`` switching the idle token ON over a warm
    start that predates it, and then reconciling the tensors itself).
    """
    stored = dict(ck["model_cfg"])
    sd = ck.get("model") or {}
    fields = ModelConfig.__dataclass_fields__
    for field, proof in ARCH_FROM_WEIGHTS.items():
        if field not in stored and field in fields:
            stored[field] = proof in sd
    unknown = [k for k in stored if k not in fields]
    for k in unknown:
        print(f"[model_cfg] checkpoint field {k!r} is no longer in ModelConfig; ignored")
        stored.pop(k)
    stored.update(overrides)
    return ModelConfig(**stored)


# ------------------------------------------------------------------ blocks --
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float):
        super().__init__()
        hidden = int(round(mult * d))
        self.w12 = nn.Linear(d, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Attention(nn.Module):
    """Multi-head attention on torch SDPA. ``mask`` is a bool key mask
    (B, S_kv), True = attend."""

    def __init__(self, d: int, heads: int, kv_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        kv = d if kv_dim is None else kv_dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(kv, d, bias=False)
        self.v_proj = nn.Linear(kv, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.h, self.dh).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(kv))
        v = self._split(self.v_proj(kv))
        attn_mask = None if mask is None else mask[:, None, None, :]
        p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
        b, _, s, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, s, self.h * self.dh))


class EncoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, mult: float, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, heads, dropout=dropout)
        self.ffn = SwiGLU(d, mult)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, mask)
        return x + self.ffn(self.n2(x))


class DecoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, mult: float, kv_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.self_attn = Attention(d, heads, dropout=dropout)
        self.cross_attn = Attention(d, heads, kv_dim=kv_dim, dropout=dropout)
        self.ffn = SwiGLU(d, mult)

    def forward(
        self,
        x: torch.Tensor,
        self_mask: torch.Tensor | None,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.self_attn(h, h, self_mask)
        x = x + self.cross_attn(self.n2(x), memory, memory_mask)
        return x + self.ffn(self.n3(x))


# ----------------------------------------------------------------- encoder --
@dataclass
class ContextEncoding:
    """Cached context-encoder output. Recompute only when the context changes."""

    memory: torch.Tensor  # (B, S_ctx, d_ctx)
    mask: torch.Tensor  # (B, S_ctx) bool

    def to(self, device) -> "ContextEncoding":
        return ContextEncoding(self.memory.to(device), self.mask.to(device))

    def expand(self, batch: int) -> "ContextEncoding":
        """Broadcast a batch-1 encoding across ``batch`` decoder samples."""
        return ContextEncoding(
            self.memory.expand(batch, -1, -1), self.mask.expand(batch, -1)
        )


class ContextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_ctx
        self.drum_proj = nn.Linear(DRUM_FEAT_DIM, d)
        self.goal_proj = nn.Linear(GOAL_RAW_DIM + cfg.fourier_width, d)
        if cfg.idle_home_token:
            # Its own projection, not ``goal_proj``: the payload is a joint-space
            # park pose, not a hit point + velocity, and sharing a linear map
            # between two different physical quantities is how you get one to
            # blur the other.
            self.idle_proj = nn.Linear(IDLE_RAW_DIM + cfg.fourier_width, d)
        self.type_emb = nn.Embedding(cfg.n_token_types, d)
        self.arm_emb = nn.Embedding(2, d)
        self.drum_id_emb = nn.Embedding(cfg.num_drums, d)
        self.future_rank_emb = nn.Embedding(max(cfg.num_future, 1), d)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d, cfg.ctx_heads, cfg.ctx_ffn_mult, cfg.dropout) for _ in range(cfg.ctx_layers)]
        )
        self.norm = RMSNorm(d)
        freqs = fourier_frequencies(cfg.fourier_bands, cfg.fourier_t_min, cfg.fourier_t_max)
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32), persistent=False)

    def forward(self, ctx: ContextInputs) -> ContextEncoding:
        cfg = self.cfg
        B, D, Fh = ctx.batch, ctx.num_drums, ctx.num_future
        if D != cfg.num_drums or Fh != cfg.num_future:
            raise ValueError(f"context shape (D={D}, F={Fh}) != config ({cfg.num_drums}, {cfg.num_future})")
        dev, dt = ctx.drum_pos.device, ctx.drum_pos.dtype
        freqs = self.freqs.to(dt)

        drum_idx = torch.arange(D, device=dev)
        drum_tok = (
            self.drum_proj(drum_features(ctx))
            + self.type_emb.weight[TYPE_DRUM]
            + self.drum_id_emb(drum_idx)
        )

        goal_raw = goal_features(ctx, ctx.goal_pos, ctx.goal_vel, ctx.goal_drum, ctx.goal_valid)
        goal_t = fourier_time_features(ctx.goal_tth, freqs, cfg.fourier_t_max)
        goal_tok = (
            self.goal_proj(torch.cat([goal_raw, goal_t], dim=-1))
            + self.type_emb.weight[TYPE_GOAL]
            + self.arm_emb(torch.arange(2, device=dev))
            + self.drum_id_emb(ctx.goal_drum.clamp(min=0))
        )

        fut_raw = goal_features(
            ctx, ctx.future_pos, ctx.future_vel, ctx.future_drum, ctx.future_valid
        )
        fut_t = fourier_time_features(ctx.future_tth, freqs, cfg.fourier_t_max)
        fut_tok = (
            self.goal_proj(torch.cat([fut_raw, fut_t], dim=-1))
            + self.type_emb.weight[TYPE_FUTURE]
            + self.arm_emb(torch.arange(2, device=dev))[:, None, :]
            + self.drum_id_emb(ctx.future_drum.clamp(min=0))
            + self.future_rank_emb(torch.arange(Fh, device=dev))
        )
        fut_tok = fut_tok.reshape(B, 2 * Fh, cfg.d_ctx)

        toks = [drum_tok, goal_tok, fut_tok]
        masks = [ctx.drum_valid, ctx.goal_valid, ctx.future_valid.reshape(B, 2 * Fh)]

        if cfg.idle_home_token:
            # The idle hand's goal: hold THIS pose.  Same shape of statement as a
            # goal token -- a target plus a time -- so it gets the same arm
            # embedding (which hand) and its own type embedding (what kind of
            # target).  The time is the wait until this hand's next scheduled
            # hit, taken from its first valid future token: that is what makes
            # "park, but be ready" actionable rather than "park forever".  A hand
            # with no future at all gets ``fourier_t_max``, i.e. "indefinitely".
            fut_t_first = torch.where(
                ctx.future_valid[:, :, 0], ctx.future_tth[:, :, 0],
                torch.full_like(ctx.future_tth[:, :, 0], cfg.fourier_t_max),
            )
            idle_t = fourier_time_features(fut_t_first, freqs, cfg.fourier_t_max)
            idle_tok = (
                self.idle_proj(torch.cat([idle_features(ctx), idle_t], dim=-1))
                + self.type_emb.weight[TYPE_IDLE]
                + self.arm_emb(torch.arange(2, device=dev))
            )
            toks.append(idle_tok)
            masks.append(ctx.idle_valid)

        x = torch.cat(toks, dim=1)
        mask = torch.cat(masks, dim=1)
        # A row with no valid key at all would make SDPA emit NaN; fall back to
        # "everything visible" branch-free (the token payloads are zeroed anyway).
        mask = mask | ~mask.any(dim=1, keepdim=True)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)
        return ContextEncoding(self.norm(x) * mask.unsqueeze(-1).to(x.dtype), mask)


# ----------------------------------------------------------------- decoder --
@dataclass
class TrajectoryOutput:
    """One refinement iteration. Masked elements are exactly zero in every
    tensor; the masks are returned so the training code can exclude them."""

    traj: torch.Tensor  # (B, H, 17)
    vel: torch.Tensor  # (B, H, 17)
    acc: torch.Tensor  # (B, H, 17)
    torque: torch.Tensor  # (B, H, 17) per-segment max |tau| reaching the waypoint
    fk_pos: torch.Tensor  # (B, H, 6, 3) FK_POINT_NAMES order
    fk_vel: torch.Tensor  # (B, H, 6, 3)
    elem_mask: torch.Tensor  # (B, H, 17) float
    point_mask: torch.Tensor  # (B, H, 6) float
    token_mask: torch.Tensor  # (B, H) bool

    def as_tuple(self):
        return (self.traj, self.vel, self.acc, self.torque, self.fk_pos, self.fk_vel)


class TrajectoryDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, n = cfg.d_model, cfg.n_dof
        fw = cfg.fourier_width
        self.wp_proj = nn.Linear(2 * n + 2 * fw + 1, d)
        self.state_proj = nn.Linear(2 * n + 2 * fw, d)
        self.state_emb = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d, cfg.dec_heads, cfg.dec_ffn_mult, cfg.d_ctx, cfg.dropout)
                for _ in range(cfg.dec_layers)
            ]
        )
        self.norm = RMSNorm(d)
        self.head_traj = nn.Linear(d, n)
        self.head_vel = nn.Linear(d, n)
        self.head_acc = nn.Linear(d, n)
        self.head_torque = nn.Linear(d, n)
        self.head_fk_pos = nn.Linear(d, N_FK_POINTS * 3)
        self.head_fk_vel = nn.Linear(d, N_FK_POINTS * 3)
        if cfg.output_mode == "residual":
            nn.init.zeros_(self.head_traj.weight)
            nn.init.zeros_(self.head_traj.bias)

        lo, hi = urdf_position_limits(posture_limits=False)
        self.register_buffer("q_center", torch.as_tensor((lo + hi) / 2.0, dtype=torch.float32))
        self.register_buffer(
            "q_half", torch.as_tensor((hi - lo) / 2.0, dtype=torch.float32).clamp_min(1e-3)
        )
        self.register_buffer(
            "v_scale", torch.as_tensor(isaac_velocity_limits(), dtype=torch.float32)
        )
        self.register_buffer(
            "tau_scale", torch.as_tensor(isaac_effort_limits(), dtype=torch.float32)
        )
        freqs = fourier_frequencies(cfg.fourier_bands, cfg.fourier_t_min, cfg.fourier_t_max)
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32), persistent=False)

        # FROZEN COLUMNS (``ModelConfig.frozen_joints``).  ``free_cols`` is 1 on
        # the columns the decoder owns, 0 on the rest; ``hold_q`` is the ready
        # pose the pinned columns settle at.  Both are non-persistent: they are
        # derived from ``cfg``, which the checkpoint already stores, so old
        # checkpoints keep loading with ``strict=True`` and a config change needs
        # no re-save.
        free = torch.ones(n, dtype=torch.float32)
        hold = torch.as_tensor(ready_joint_pos(G1_UPPER_BODY_JOINTS), dtype=torch.float32)
        for nm in cfg.frozen_joints:
            free[G1_UPPER_BODY_JOINTS.index(nm)] = 0.0
        self.n_frozen = int((free == 0.0).sum().item())
        self.register_buffer("free_cols", free, persistent=False)
        self.register_buffer("hold_q", hold, persistent=False)

    # -- the fully tensor-in / tensor-out path (this is what gets compiled) --
    def forward(
        self,
        init_traj: torch.Tensor,  # (B, H, 17)
        state: torch.Tensor,  # (B, 34) = [q, qd]
        tau_left: torch.Tensor,  # (B, H)
        tau_right: torch.Tensor,  # (B, H)
        tau0_left: torch.Tensor,  # (B,)
        tau0_right: torch.Tensor,  # (B,)
        elem_mask: torch.Tensor,  # (B, H, 17)
        token_mask: torch.Tensor,  # (B, H) bool
        point_mask: torch.Tensor,  # (B, H, 6)
        memory: torch.Tensor,  # (B, S, d_ctx)
        memory_mask: torch.Tensor,  # (B, S) bool
    ):
        cfg = self.cfg
        dt = init_traj.dtype
        n = cfg.n_dof
        freqs = self.freqs.to(dt)
        qc, qh = self.q_center.to(dt), self.q_half.to(dt)

        q_n = (init_traj - qc) / qh * elem_mask
        tok_f = token_mask.to(dt).unsqueeze(-1)
        wp_feat = torch.cat(
            [
                q_n,
                elem_mask,
                fourier_time_features(tau_left, freqs, cfg.fourier_t_max),
                fourier_time_features(tau_right, freqs, cfg.fourier_t_max),
                tok_f,
            ],
            dim=-1,
        )
        wp = self.wp_proj(wp_feat * tok_f)

        st_feat = torch.cat(
            [
                (state[:, :n] - qc) / qh,
                state[:, n:] / self.v_scale.to(dt),
                fourier_time_features(tau0_left, freqs, cfg.fourier_t_max),
                fourier_time_features(tau0_right, freqs, cfg.fourier_t_max),
            ],
            dim=-1,
        )
        st = (self.state_proj(st_feat) + self.state_emb).unsqueeze(1)

        x = torch.cat([st, wp], dim=1)
        self_mask = torch.cat(
            [torch.ones_like(token_mask[:, :1]), token_mask], dim=1
        )  # the state token is always visible
        for blk in self.blocks:
            x = blk(x, self_mask, memory, memory_mask)
        h = self.norm(x)[:, 1:]

        raw = self.head_traj(h)
        if cfg.output_mode == "residual":
            traj = init_traj + qh * raw
        else:
            traj = qc + qh * raw
        traj = traj * elem_mask

        vel = self.head_vel(h) * self.v_scale.to(dt) * elem_mask
        acc = self.head_acc(h) * cfg.accel_scale * elem_mask
        torque = self.head_torque(h) * self.tau_scale.to(dt) * elem_mask
        pm = point_mask.unsqueeze(-1)
        b, hh = h.shape[0], h.shape[1]
        fk_pos = self.head_fk_pos(h).view(b, hh, N_FK_POINTS, 3) * cfg.fk_pos_scale * pm
        fk_vel = self.head_fk_vel(h).view(b, hh, N_FK_POINTS, 3) * cfg.fk_vel_scale * pm

        # ---- frozen columns: the decoder does not own these joints ----------
        # Applied AFTER the elem_mask so padded waypoints stay exactly zero (the
        # rest of the stack relies on that).  ``free_cols`` is all-ones in the
        # default configuration, so this whole block is skipped on every run that
        # does not use the flag.
        #
        # The pinned columns RAMP from the measured start pose to the ready pose
        # over ``frozen_ramp_wp`` waypoints and hold, rather than jumping there
        # on waypoint 1.  This is not cosmetic: the starts sample the waist in a
        # posture box (measured over the pool, waist pitch/roll are 0.09 rad rms
        # with tails at the full URDF +-0.52), so a step would demand up to
        # 0.52 rad inside one 20 ms control period.  The peak RATE would survive
        # -- 26 rad/s against a 37 rad/s limit -- but the acceleration at the
        # first knot would not, and it lands straight on ``torque_peak``, which
        # is the statistic the acceptance gate thresholds on.  A frozen run would
        # then fail the gate for a reason that has nothing to do with the waist.
        #
        # The ramp is the quintic smoothstep (zero first AND second derivative at
        # both ends), the same shape ``waist_aim.yaw_reference`` uses, so the
        # departure from the measured pose is smooth in acceleration too.
        if self.n_frozen:
            free = self.free_cols.to(dt)
            pin = 1.0 - free
            q_start = state[:, :n].unsqueeze(1)                    # (B, 1, n)
            hold = self.hold_q.to(dt).view(1, 1, -1)               # (1, 1, n)
            H_ = traj.shape[1]
            i = torch.arange(1, H_ + 1, device=traj.device, dtype=dt).view(1, -1, 1)
            u = (i / float(max(cfg.frozen_ramp_wp, 1))).clamp(0.0, 1.0)
            s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)        # quintic smoothstep
            pinned = q_start + (hold - q_start) * s                # (B, H, n)
            traj = traj * free + (pinned * pin) * elem_mask
            # The aux heads are deliberately NOT masked: during the ramp these
            # joints really are moving, so their velocity / acceleration / torque
            # labels are non-zero and a zeroed prediction would be an
            # irreducible loss floor rather than a statement about the freeze.

        return traj, vel, acc, torque, fk_pos, fk_vel


# ------------------------------------------------------------------- model --
class AmortizedOptimizer(nn.Module):
    """Context encoder + trajectory decoder with the iterative interface.

    ``encode_context`` runs once per context change; ``forward`` is one
    refinement iteration, and refinement is feeding ``out.traj`` back in as
    ``init_traj``.
    """

    def __init__(self, cfg: ModelConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig(**kwargs)
        self.encoder = ContextEncoder(self.cfg)
        self.decoder = TrajectoryDecoder(self.cfg)

    def encode_context(self, ctx: ContextInputs) -> ContextEncoding:
        return self.encoder(ctx)

    def forward(
        self,
        init_traj: torch.Tensor,
        state: torch.Tensor,
        context: ContextEncoding,
        masks: WaypointMasks,
    ) -> TrajectoryOutput:
        traj, vel, acc, torque, fk_pos, fk_vel = self.decoder(
            init_traj,
            state,
            masks.tau_left,
            masks.tau_right,
            masks.tau0_left,
            masks.tau0_right,
            masks.elem_mask,
            masks.token_mask,
            masks.point_mask,
            context.memory,
            context.mask,
        )
        return TrajectoryOutput(
            traj=traj,
            vel=vel,
            acc=acc,
            torque=torque,
            fk_pos=fk_pos,
            fk_vel=fk_vel,
            elem_mask=masks.elem_mask,
            point_mask=masks.point_mask,
            token_mask=masks.token_mask,
        )

    def refine(
        self,
        init_traj: torch.Tensor,
        state: torch.Tensor,
        context: ContextEncoding,
        masks: WaypointMasks,
        iterations: int = 1,
    ) -> TrajectoryOutput:
        """``iterations`` weight-shared passes; the MPC runs exactly one."""
        out = self(init_traj, state, context, masks)
        for _ in range(int(iterations) - 1):
            out = self(out.traj, state, context, masks)
        return out


def param_counts(model: AmortizedOptimizer) -> dict[str, int]:
    """Encoder / decoder-body / head parameter counts."""
    head_prefixes = ("head_",)
    enc = sum(p.numel() for p in model.encoder.parameters())
    heads = sum(
        p.numel()
        for n, p in model.decoder.named_parameters()
        if n.startswith(head_prefixes)
    )
    dec = sum(p.numel() for p in model.decoder.parameters()) - heads
    return {"encoder": enc, "decoder": dec, "heads": heads, "total": enc + dec + heads}
