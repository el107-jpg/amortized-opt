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
4. The torque head's target is set by the trainer, in N*m, and has two
   definitions (``aux_targets(..., torque_mode=...)``):
   ``"rnea_segmax"`` -- the default and what every checkpoint before
   2026-08-16 learned -- is the per-segment max ``|tau_rnea|`` over the spline
   segment ENDING at that waypoint, i.e. the plan's own OPEN-LOOP dynamics;
   ``"tracking"`` is ``|tau_rnea + tau_fb|`` AT that waypoint, the torque the
   PD servo actually demands to track the plan, which is the quantity the cost
   prices through ``w_track_torque``.  The head itself is agnostic -- it emits
   ``n`` numbers scaled by the effort limits -- so a checkpoint's torque head
   only means what the run that fit it chose.
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

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF, ready_joint_pos
from .fk import ChainFK, stick_tip_offset
from .limits import isaac_effort_limits, isaac_velocity_limits, urdf_position_limits
from .rnea import BatchedRNEA
from .waist_aim import _DEFAULT_YAW_LIMIT as _WAIST_SP_LIMIT
from .tokenizer import (
    DRUM_FEAT_DIM,
    FK_POINT_NAMES,  # noqa: F401  (re-exported: the fk head's point order)
    GOAL_RAW_DIM,
    N_FK_POINTS,
    ContextInputs,
    WaypointMasks,
    IDLE_RAW_DIM,
    STICK_RAW_DIM,
    drum_features,
    fourier_dim,
    fourier_frequencies,
    fourier_time_features,
    goal_features,
    idle_features,
    stick_features,
)

#: Context token type ids (also the row order the encoder builds).
#: ``TYPE_IDLE`` is APPEND-ONLY at the end so the first three embedding rows of
#: an existing checkpoint keep their meaning.
TYPE_DRUM, TYPE_GOAL, TYPE_FUTURE, TYPE_IDLE, TYPE_STICK = 0, 1, 2, 3, 4
N_TOKEN_TYPES = 5


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
    #: Clamp the emitted trajectory to the URDF position box, STRAIGHT-THROUGH:
    #: the forward pass returns the clamped value, the backward pass sees the
    #: identity.  ``head_traj`` is a bare ``nn.Linear``, so nothing structural
    #: has ever bounded the plan -- only the soft ``joint_box`` hinge, and on
    #: runs/phase2_19m_waist_sp_d that hinge lost: 94 % of logged steps had a
    #: peak waist pitch/roll past the URDF limit of 0.520 rad (worst 0.685) with
    #: ``lam_joint_box`` pinned at its ceiling and 18.3 % of the objective spent.
    #: The plans were therefore INADMISSIBLE -- ``MPCGenConfig.clamp_cmd_to_box``
    #: is True and the plant saturates, so deployment executed something the
    #: cost never scored.
    #:
    #: A SOFT clip, and the two failed alternatives are why.  Plain ``tanh`` to
    #: the limit saturates at the model's own optimum (gradient 0.02 at 99 % of
    #: the limit, a 50x attenuation) and can never come back off the rail.  A
    #: hard straight-through clamp keeps full gradient but puts CORNERS in the
    #: position trajectory, and ``physics_mode="knot_fd"`` differentiates that
    #: trajectory twice: resuming runs/phase2_19m_waist_sp_d under a hard clamp
    #: spiked d_jerk 142.6M -> 1846.1M, c_smooth 58 -> 688 and d_tau_excess
    #: 15 -> 162 at step 50, on IDENTICAL weights, before any learning.  A hard
    #: clamp is right for a single command (``MPCGenConfig.clamp_cmd_to_box``,
    #: one tick, no derivatives) and wrong for a differentiated path.
    #:
    #: So: identity inside ``clamp_knee`` x the half-range -- gradient exactly 1
    #: where the model actually operates -- and a C1 ``tanh`` bend to the limit
    #: beyond it.  No corners, no saturation in the operating region, and the
    #: emitted plan is bounded by the box.  Behaviour-only, so it round-trips in
    #: ``model_cfg`` and is NOT in ``ARCH_FROM_WEIGHTS``.
    clamp_traj_to_box: bool = False
    #: Fraction of the box half-range that stays EXACTLY linear.  0.8 matches
    #: ``CostConfig.eta_position``, so the soft bend begins where the joint-box
    #: hinge starts charging and the two agree about where "near the limit" is.
    clamp_knee: float = 0.8
    #: Run only the trajectory readout and its physical-unit unprojection in
    #: explicit float32.  Transformer blocks and auxiliary heads remain under
    #: the caller's autocast policy.  This is behaviour-only and defaults OFF,
    #: so checkpoints written before the option remain bit-compatible.
    traj_head_fp32: bool = False
    predict_aux: bool = True
    dropout: float = 0.0

    #: THE IDLE ARM'S OWN GOAL TOKEN.  Adds two context rows (one per hand),
    #: carrying the packed torso-frame idle safety cylinder from
    #: :mod:`~amortized_opt.idle_cylinder` (``IDLE_RAW_DIM = 8``), valid exactly
    #: on the hands with NO scheduled strike.  Without it the only thing the
    #: context says about an idle hand is *absence* -- its goal token is zeroed
    #: and masked out of attention -- so "where should this tip stay" has to be
    #: inferred from the seed alone.
    #:
    #: OFF makes the encoder bit-identical to the pre-2026-08-12 model, including
    #: the ``type_emb`` row count, so existing checkpoints load strictly.  ON adds
    #: one ``idle_proj`` and one ``type_emb`` row -- a checkpoint trained without
    #: it will report those as missing keys and they start random.  A checkpoint
    #: trained on the old 10-D joint+tip idle payload will not load into the
    #: current 8-D cylinder ``idle_proj`` (shape mismatch).
    idle_home_token: bool = True

    #: THE STICK TOKEN.  Two more context rows (one per hand) carrying the
    #: fitted stick's axis, tip offset and length
    #: (:func:`~amortized_opt.tokenizer.stick_features`).  Needed exactly when
    #: ``StickRandomization`` is on: a stick the model is not told about is
    #: unlearnable noise, the same trap the drum jitter fell into before
    #: ``drum_features`` was varied.
    #:
    #: OFF by default, and APPEND-ONLY like ``idle_home_token`` -- the row it
    #: adds is the LAST ``type_emb`` row, so a checkpoint trained without it
    #: keeps every earlier row's meaning and loads through
    #: :func:`~amortized_opt.model.model_config_from_checkpoint`.
    stick_token: bool = False

    #: THE LEARNED WAIST-YAW SET-POINT (``docs/waist_sp_head_plan.md``).  A
    #: readout on the encoder predicts ONE held yaw per goal-segment as a
    #: zero-initialized residual on the analytic
    #: :func:`~amortized_opt.waist_aim.aim_yaw` carried in
    #: ``ContextInputs.aim_yaw`` -- so the untrained head IS the analytic law,
    #: which is the whole distillation.  The set-point is appended to the
    #: cross-attention memory as one extra token (its own projection and bias
    #: parameter, no ``type_emb`` row), and returned on
    #: :class:`ContextEncoding` for the phase-2 anchor / tether terms.
    #:
    #: OFF is bit-identical to the model without the feature (strict checkpoint
    #: load); ON adds ``waist_sp_hidden`` / ``waist_sp_out`` / ``waist_sp_proj``
    #: / ``waist_sp_emb``, reconciled through :data:`ARCH_FROM_WEIGHTS`.
    waist_sp_head: bool = False

    #: THE LEARNED WAIST PITCH/ROLL SET-POINT (``docs/waist_pr_sp_head_plan.md``).
    #: Sibling of ``waist_sp_head``, NOT a widening of it -- ``waist_sp_out``
    #: stays ``(d/2, 1)`` so every yaw-head checkpoint loads strictly.
    #:
    #: A readout on the encoder predicts ONE held ``(pitch, roll)`` per goal
    #: segment as a zero-initialized residual on upright (0, 0).  There is no
    #: analytic lean law (``waist_aim`` is yaw-only).  The untrained head IS
    #: upright.  Inputs are the pooled context tokens only -- no measured lean,
    #: no ``yaw_now`` -- so the set-point is constant inside a goal segment and
    #: cannot form the closed-loop integrator measured on pitch/roll (wp1
    #: copying ``q0`` onto the URDF rail).
    #:
    #: The conditioning token is appended BEFORE the yaw token so
    #: ``substitute_waist_sp``'s ``memory[:, :-1]`` (yaw is last) stays valid.
    #: Payload is ``[sp_pitch, sp_roll]`` only; measured lean already lives on
    #: the decoder state token.
    #:
    #: After encode, the quintic seed eases pitch/roll to this set-point the
    #: same way it ramps yaw to ``aim_yaw`` (``seed_waist_pr_to_sp``).
    #:
    #: OFF is bit-identical; ON adds ``waist_pr_sp_*``, reconciled through
    #: :data:`ARCH_FROM_WEIGHTS`.
    waist_pr_sp_head: bool = False

    #: THE CURRENT-GOAL COUNTDOWN, IN THE DECODER.  Widens the state token's
    #: projection by one Fourier block per hand, fed from
    #: ``WaypointMasks.hit_tth``.  Pairs with ``enc_goal_tth`` below: together
    #: they MOVE the countdown from the encoder to the decoder.
    #:
    #: Why it has to move at all: the countdown is the last from-now input the
    #: encoder had, and a from-now input ticks down every 20 ms, so the
    #: deployment MPC had to re-encode the whole context every tick
    #: (``isaac_rl/README.md`` DECISION 2) -- which is the path by which the
    #: robot's own state echoed back into the encoder and ran the waist away in
    #: closed loop.  The decoder already runs every tick and already reads the
    #: per-waypoint ``tau_*`` codes, so the countdown costs it nothing.
    #:
    #: It is NOT redundant with ``tau0_*``, which the state token already
    #: carries: ``tau0`` is the RESOLVED horizon, so an idle hand's ``tau0`` is
    #: the other hand's, borrowed for masking.  ``hit_tth`` is 0 exactly when a
    #: hand has no strike, which is a distinction the decoder previously could
    #: not make at all.
    dec_tth_feats: bool = True

    #: Whether the ENCODER's goal token still embeds ``ContextInputs.goal_tth``.
    #: OFF (the new default) is the point of ``dec_tth_feats``: with it off,
    #: every encoder input is constant inside a goal segment BY CONSTRUCTION, so
    #: the deployment cache in ``isaac_rl.mpc.IsaacMPC`` is exact rather than an
    #: approximation, and the learned waist set-point is structurally one value
    #: per segment.  ``goal_tth`` itself STAYS on ``ContextInputs`` -- masks,
    #: strike bookkeeping and the schedule all still need the countdown; only
    #: the embedding drops it.
    #:
    #: Implemented by feeding the goal token a ZERO time instead of removing the
    #: block, which is exact in two senses.  Numerically: a constant input to a
    #: linear map is a bias shift, so it is what "omit the block" means, with no
    #: parameter shape moving and no weight sharing broken with the future
    #: tokens (which need the same projection for ``future_gap``).  Semantically:
    #: ``future_gap`` measures from the CURRENT goal's strike, so the current
    #: goal's own gap is exactly 0 -- the goal token gets the same zero-phase
    #: terminal anchor every other hit-relative token gets at its strike.
    #:
    #: Because no shape moves, a checkpoint that predates the field is
    #: recognised by the field's ABSENCE from its stored ``model_cfg``
    #: (:data:`ARCH_LEGACY_DEFAULTS`) rather than from the weights.
    enc_goal_tth: bool = False

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

    #: ROTARY POSITION on decoder SELF-ATTENTION.  The waypoint sequence is a
    #: uniform 50 Hz grid ``[state] + H knots`` with NO other positional code
    #: (Fourier ``tau_*`` is time-to-hit, terminal-anchored, and a weak neighbor
    #: signal).  Llama-style RoPE on Q/K makes signed knot distance ``s - t`` a
    #: structural part of the attention score: neighbors are easy, later vs
    #: earlier is distinguished, and the full softmax over the horizon is
    #: unchanged -- distant knots (the strike from t=0, the other hand) stay
    #: visible.  Cross-attention and the encoder are NOT rotated: those tokens
    #: are not a time grid.
    #:
    #: OFF is bit-identical to every checkpoint written before the flag (no
    #: parameter, ``persistent=False`` buffer).  ON over a warm start is a
    #: behaviour change at the same shapes, so ``strict=True`` still loads.
    dec_rope: bool = False
    #: RoPE base period.  10000 is the Llama default; at H <= 100 the slowest
    #: planes barely rotate, which is what keeps long-range content intact.
    rope_base: float = 10000.0

    #: ANALYTIC PLANT FEATURES ON THE DECODER TOKENS.  Cartesian tip/wrist/elbow
    #: and knot-RNEA of the *current iterate* (``init_traj``) on each waypoint
    #: -- and the state token at t=0 -- so the decoder can see the quantities
    #: ``FusedCost`` actually prices, instead of inferring FK / inverse dynamics
    #: into ``h``.  SDF is deliberately omitted: drum geometry is already in
    #: the encoder, and with world-frame ``p`` on the token the FFN can subtract.
    #:
    #: Computed under ``no_grad`` from a hold-filled copy of ``init_traj``
    #: (masked joints keep their last valid knot, so a parked arm stays in
    #: Cartesian after its horizon).  A separate zero-init Linear per family
    #: (``wp_plant_proj`` / ``state_plant_proj``) adds them to the token, so
    #: (a) ``wp_proj`` / ``state_proj`` keep their trained shapes when the flag
    #: is switched on over a warm start, and (b) the untrained head is a no-op
    #: -- same trick as ``output_mode="residual"``.
    #:
    #: OFF is bit-identical to every checkpoint written before the flag.
    dec_plant_feats: bool = False

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
        if self.dec_rope and self.dec_head_dim % 2:
            raise ValueError(
                f"dec_rope needs an even head dim, got {self.dec_head_dim} "
                f"(d_model={self.d_model}, dec_heads={self.dec_heads})"
            )

    @property
    def dec_head_dim(self) -> int:
        return self.d_model // self.dec_heads

    @property
    def fourier_width(self) -> int:
        return fourier_dim(self.fourier_bands)

    @property
    def plant_feat_dim(self) -> int:
        """Width of the concatenated Cartesian + RNEA plant block (per token)."""
        return N_FK_POINTS * 3 + self.n_dof

    @property
    def n_context_tokens(self) -> int:
        return (self.num_drums + 2 + 2 * self.num_future
                + (2 if self.idle_home_token else 0)
                + (2 if self.stick_token else 0)
                + (1 if self.waist_pr_sp_head else 0)
                + (1 if self.waist_sp_head else 0))

    @property
    def n_token_types(self) -> int:
        """Rows of ``type_emb``.  Pinned to 3 when the idle token is off so the
        embedding SHAPE matches every checkpoint written before it existed."""
        if self.stick_token:
            return N_TOKEN_TYPES
        return N_TOKEN_TYPES - 1 if self.idle_home_token else N_TOKEN_TYPES - 2


#: Architecture flags that a checkpoint's stored ``model_cfg`` may not carry,
#: because the field was added to :class:`ModelConfig` after it was written,
#: mapped to a state-dict key that PROVES whether the trained weights have the
#: feature.
#:
#: Append to this whenever a new field changes a parameter SHAPE.  A field that
#: only changes behaviour (``frozen_joints``, ``output_mode``, ``dec_rope``)
#: does not belong here: those round-trip in ``model_cfg`` and have safe
#: defaults.  ``dec_rope`` also sits in :data:`ARCH_LEGACY_DEFAULTS` so a
#: config written before the field stays OFF if someone later defaults it ON.
ARCH_FROM_WEIGHTS: dict[str, str] = {
    "idle_home_token": "encoder.idle_proj.weight",
    "stick_token": "encoder.stick_proj.weight",
    "waist_sp_head": "encoder.waist_sp_out.weight",
    "waist_pr_sp_head": "encoder.waist_pr_sp_out.weight",
    "dec_plant_feats": "decoder.wp_plant_proj.weight",
}

#: The same job for a flag that widens a projection that ALREADY EXISTS, so the
#: proof is the weight's SHAPE rather than its presence.  Value is
#: ``(state-dict key, off_width(stored_cfg))``: the flag is True iff the stored
#: input width exceeds the width the feature is OFF at.  Kept apart from
#: :data:`ARCH_FROM_WEIGHTS` so the presence rule -- which every flag before
#: ``dec_tth_feats`` uses -- stays a one-line lookup.
ARCH_FROM_SHAPES: dict[str, tuple] = {
    "dec_tth_feats": (
        "decoder.state_proj.weight",
        lambda s: 2 * _stored(s, "n_dof") + 2 * fourier_dim(_stored(s, "fourier_bands")),
    ),
}

#: Flags whose DEFAULT flipped after checkpoints existed and whose old value no
#: weight can prove, because the feature moved no parameter shape.  A stored
#: config that lacks the key was written before the flip, so the absence IS the
#: proof: reconstruct with the value listed here rather than with the current
#: default.  Only ever correct for a field whose value is written into every
#: checkpoint from the moment it exists (``asdict(model_cfg)`` does that).
ARCH_LEGACY_DEFAULTS: dict[str, object] = {
    "enc_goal_tth": True,
    "dec_rope": False,
    "traj_head_fp32": False,
}


def _stored(stored: dict, name: str):
    """``stored[name]``, falling back to the field's declared default."""
    if name in stored:
        return stored[name]
    return ModelConfig.__dataclass_fields__[name].default


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
    for field, (proof, off_width) in ARCH_FROM_SHAPES.items():
        if field not in stored and field in fields:
            w = sd.get(proof)
            stored[field] = False if w is None else int(w.shape[1]) > off_width(stored)
    for field, legacy in ARCH_LEGACY_DEFAULTS.items():
        if field not in stored and field in fields:
            stored[field] = legacy
    unknown = [k for k in stored if k not in fields]
    for k in unknown:
        print(f"[model_cfg] checkpoint field {k!r} is no longer in ModelConfig; ignored")
        stored.pop(k)
    stored.update(overrides)
    return ModelConfig(**stored)


# ------------------------------------------------------------------ plant --
def hold_fill_waypoints(
    q0: torch.Tensor,
    waypoints: torch.Tensor,
    elem_mask: torch.Tensor,
) -> torch.Tensor:
    """Knot positions ``(B, H+1, n)``: ``q0`` then waypoints, masked joints held.

    Past a joint's horizon ``elem_mask`` is 0 and ``waypoints`` is zeroed, which
    would put a parked arm at the origin in FK.  Forward-fill from the last
    valid knot (or ``q0``) so later tokens still see the parked pose in
    Cartesian -- the same hold the spline uses after a hit.
    """
    x0 = q0.unsqueeze(1)
    x = torch.cat([x0, waypoints], dim=1)
    m = torch.cat([torch.ones_like(x0), elem_mask.to(dtype=x.dtype)], dim=1)
    idx = torch.arange(x.shape[1], device=x.device).view(1, -1, 1).expand_as(x)
    last = torch.where(m > 0.5, idx, torch.zeros_like(idx))
    return x.gather(1, last.cummax(dim=1).values)


# ------------------------------------------------------------------ blocks --
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Llama half-split: ``(x1, x2) -> (-x2, x1)`` along the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """``(head_dim/2,)`` inverse frequencies for Llama-style RoPE."""
    if head_dim % 2:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    return 1.0 / (
        float(base) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K ``(B, heads, S, dh)``; ``cos``/``sin`` broadcast to that shape."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


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
    (B, S_kv), True = attend.

    ``rope`` rotates Q and K by token index (Llama half-split) BEFORE SDPA.
    Only decoder self-attention turns this on: the sequence is the 50 Hz
    waypoint grid.  Cross-attention / encoder leave it off.  The inv-freq
    table is a non-persistent buffer, so a checkpoint's state_dict is the
    same with the flag on or off.
    """

    def __init__(
        self,
        d: int,
        heads: int,
        kv_dim: int | None = None,
        dropout: float = 0.0,
        rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        kv = d if kv_dim is None else kv_dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(kv, d, bias=False)
        self.v_proj = nn.Linear(kv, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout
        self.rope = bool(rope)
        if self.rope:
            self.register_buffer(
                "rope_inv_freq", rope_inv_freq(self.dh, rope_base), persistent=False
            )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.h, self.dh).transpose(1, 2)

    def _rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Positions are 0..S-1 on THIS sequence (state token at 0, then knots).
        S = q.shape[2]
        t = torch.arange(S, device=q.device, dtype=torch.float32)
        freqs = torch.outer(t, self.rope_inv_freq.to(dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=q.dtype).view(1, 1, S, -1)
        sin = emb.sin().to(dtype=q.dtype).view(1, 1, S, -1)
        return apply_rope(q, k, cos, sin)

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(kv))
        v = self._split(self.v_proj(kv))
        if self.rope:
            q, k = self._rope(q, k)
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
    def __init__(
        self,
        d: int,
        heads: int,
        mult: float,
        kv_dim: int,
        dropout: float = 0.0,
        rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.self_attn = Attention(
            d, heads, dropout=dropout, rope=rope, rope_base=rope_base
        )
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
    #: The learned waist-yaw set-point / its residual on the analytic aim
    #: (``ModelConfig.waist_sp_head``); None when the head is off.  The
    #: set-point rides the encoding because it is a function of the context
    #: alone -- cached and recomputed exactly when the context changes.
    #: ``waist_yaw_res`` is the POST-clamp effective residual ``sp - base``
    #: (the applied one, which the tether prices), not the raw MLP output.
    waist_yaw_sp: torch.Tensor | None = None  # (B,)
    waist_yaw_res: torch.Tensor | None = None  # (B,)
    #: Pitch/roll set-point (``ModelConfig.waist_pr_sp_head``); None when off.
    #: ``(B, 2)`` in ``(pitch, roll)`` order.  Residual is the set-point itself
    #: (base is upright 0).  Must not depend on measured lean -- see the flag.
    waist_pr_sp: torch.Tensor | None = None  # (B, 2)
    waist_pr_res: torch.Tensor | None = None  # (B, 2)

    def to(self, device) -> "ContextEncoding":
        opt = lambda v: None if v is None else v.to(device)  # noqa: E731
        return ContextEncoding(
            self.memory.to(device), self.mask.to(device),
            opt(self.waist_yaw_sp), opt(self.waist_yaw_res),
            opt(self.waist_pr_sp), opt(self.waist_pr_res),
        )

    def expand(self, batch: int) -> "ContextEncoding":
        """Broadcast a batch-1 encoding across ``batch`` decoder samples."""
        def opt(v):
            if v is None:
                return None
            return v.expand(batch, *([-1] * (v.ndim - 1))) if v.ndim > 1 else v.expand(batch)
        return ContextEncoding(
            self.memory.expand(batch, -1, -1), self.mask.expand(batch, -1),
            opt(self.waist_yaw_sp), opt(self.waist_yaw_res),
            opt(self.waist_pr_sp), opt(self.waist_pr_res),
        )


class ContextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_ctx
        self.drum_proj = nn.Linear(DRUM_FEAT_DIM, d)
        self.goal_proj = nn.Linear(GOAL_RAW_DIM + cfg.fourier_width, d)
        if cfg.idle_home_token:
            # Its own projection, not ``goal_proj``: the payload is a packed
            # torso-frame cylinder (center/axis/radius/half_height), not a hit
            # point + velocity, and sharing a linear map between two different
            # physical quantities is how you get one to blur the other.
            self.idle_proj = nn.Linear(IDLE_RAW_DIM + cfg.fourier_width, d)
        if cfg.stick_token:
            # Its own projection: the payload is hardware geometry, not a
            # target, and it carries no time -- so no fourier block either.
            self.stick_proj = nn.Linear(STICK_RAW_DIM, d)
        if cfg.waist_sp_head:
            # The set-point readout: pooled tokens + [yaw_now, aim base] -> a
            # residual ON the analytic aim.  ``waist_sp_out`` is ZERO-INIT so
            # the untrained head is exactly ``aim_yaw`` -- the distillation is
            # the parameterization, not a loss (same trick as
            # ``output_mode="residual"``).  The conditioning token gets its own
            # projection and bias parameter (like the decoder's ``state_emb``)
            # rather than a ``type_emb`` row, so no embedding shape moves.
            self.waist_sp_hidden = nn.Linear(d + 2, d // 2)
            self.waist_sp_out = nn.Linear(d // 2, 1)
            nn.init.zeros_(self.waist_sp_out.weight)
            nn.init.zeros_(self.waist_sp_out.bias)
            self.waist_sp_proj = nn.Linear(3, d)
            self.waist_sp_emb = nn.Parameter(torch.zeros(d))
        if cfg.waist_pr_sp_head:
            # Residual on upright, pooled tokens ONLY -- no measured lean, no
            # yaw_now.  Zero-init final linear so the untrained head is (0, 0).
            # Token payload is the two set-points; the decoder already has q
            # on the state token.  Appended BEFORE the yaw token.
            self.waist_pr_sp_hidden = nn.Linear(d, d // 2)
            self.waist_pr_sp_out = nn.Linear(d // 2, 2)
            nn.init.zeros_(self.waist_pr_sp_out.weight)
            nn.init.zeros_(self.waist_pr_sp_out.bias)
            self.waist_pr_sp_proj = nn.Linear(2, d)
            self.waist_pr_sp_emb = nn.Parameter(torch.zeros(d))
            lo, hi = urdf_position_limits(posture_limits=False)
            names = G1_UPPER_BODY_JOINTS
            pi = names.index("waist_pitch_joint")
            ri = names.index("waist_roll_joint")
            lim = torch.tensor(
                [min(-float(lo[pi]), float(hi[pi])),
                 min(-float(lo[ri]), float(hi[ri]))],
                dtype=torch.float32,
            )
            self.register_buffer("waist_pr_limit", lim, persistent=False)
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
        # ``enc_goal_tth`` off: the goal token's time is the constant 0 -- its
        # own hit-relative gap, since ``future_gap`` anchors at exactly this
        # strike.  That is what leaves EVERY encoder input constant inside a
        # goal segment; the countdown itself reaches the model through
        # ``ModelConfig.dec_tth_feats`` on the decoder's state token.
        goal_tth = ctx.goal_tth if cfg.enc_goal_tth else torch.zeros_like(ctx.goal_tth)
        goal_t = fourier_time_features(goal_tth, freqs, cfg.fourier_t_max)
        goal_tok = (
            self.goal_proj(torch.cat([goal_raw, goal_t], dim=-1))
            + self.type_emb.weight[TYPE_GOAL]
            + self.arm_emb(torch.arange(2, device=dev))
            + self.drum_id_emb(ctx.goal_drum.clamp(min=0))
        )

        fut_raw = goal_features(
            ctx, ctx.future_pos, ctx.future_vel, ctx.future_drum, ctx.future_valid
        )
        # The same Fourier time encoding the goal token uses, applied to the
        # HIT-RELATIVE gap (``ContextInputs.future_gap``): a gap lives in the
        # same 0..~5 s range as the time-to-hit it replaced, so the band scales
        # are unchanged -- only what the number MEANS moved.
        fut_t = fourier_time_features(ctx.future_gap, freqs, cfg.fourier_t_max)
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
            # The idle hand's goal: keep the tip inside THIS cylinder.  Same
            # shape of statement as a goal token -- a target plus a time -- so it
            # gets the same arm embedding (which hand) and its own type embedding
            # (what kind of target).  The time is the wait until this hand's next
            # scheduled hit, taken from its first valid future token: that is
            # what makes "park, but be ready" actionable rather than "park
            # forever".  A hand with no future at all gets ``fourier_t_max``,
            # i.e. "indefinitely".  ``future_gap`` is anchored at the current
            # goal's strike -- but an IDLE hand has no current goal, so its
            # anchor IS now and slot 0's gap is literally the wait.
            # ``idle_valid == ~goal_valid`` is what makes that hold on every row
            # this branch can reach.
            fut_t_first = torch.where(
                ctx.future_valid[:, :, 0], ctx.future_gap[:, :, 0],
                torch.full_like(ctx.future_gap[:, :, 0], cfg.fourier_t_max),
            )
            idle_t = fourier_time_features(fut_t_first, freqs, cfg.fourier_t_max)
            idle_tok = (
                self.idle_proj(torch.cat([idle_features(ctx), idle_t], dim=-1))
                + self.type_emb.weight[TYPE_IDLE]
                + self.arm_emb(torch.arange(2, device=dev))
            )
            toks.append(idle_tok)
            masks.append(ctx.idle_valid)

        if cfg.stick_token:
            stick_tok = (
                self.stick_proj(stick_features(ctx))
                + self.type_emb.weight[TYPE_STICK]
                + self.arm_emb(torch.arange(2, device=dev))
            )
            toks.append(stick_tok)
            masks.append(ctx.stick_valid)

        x = torch.cat(toks, dim=1)
        mask = torch.cat(masks, dim=1)
        # A row with no valid key at all would make SDPA emit NaN; fall back to
        # "everything visible" branch-free (the token payloads are zeroed anyway).
        mask = mask | ~mask.any(dim=1, keepdim=True)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)
        mem = self.norm(x) * mask.unsqueeze(-1).to(x.dtype)
        yaw_sp = yaw_res = None
        pr_sp = pr_res = None
        if not (cfg.waist_pr_sp_head or cfg.waist_sp_head):
            return ContextEncoding(mem, mask)

        mf = mask.unsqueeze(-1).to(mem.dtype)
        pooled = (mem * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)

        # Pitch/roll token FIRST so the yaw token stays last and
        # ``substitute_waist_sp``'s ``memory[:, :-1]`` still drops only yaw.
        if cfg.waist_pr_sp_head:
            # Residual on upright.  Reported residual is the POST-clamp
            # set-point (base is 0), so a clamp at the URDF rail is what the
            # tether prices -- not a raw MLP output the trajectory never saw.
            h_pr = F.silu(self.waist_pr_sp_hidden(pooled))
            raw_pr = self.waist_pr_sp_out(h_pr)
            lim = self.waist_pr_limit.to(mem.dtype)
            pr_sp = raw_pr.clamp(-lim, lim)
            pr_res = pr_sp
            tok_pr = self.waist_pr_sp_proj(pr_sp) + self.waist_pr_sp_emb
            mem = torch.cat([mem, tok_pr.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)

        if cfg.waist_sp_head:
            # ---- the learned waist-yaw set-point --------------------------------
            # ``base`` is the analytic aim when the caller supplied one, else "hold
            # where you are"; the head emits a residual on top and the clamp is the
            # same +/-100 deg policy limit ``waist_aim`` commands within.  The
            # residual REPORTED (``waist_yaw_res``, what the tether prices) is the
            # POST-clamp effective one, ``sp - base``: when the clamp binds, that
            # is the residual actually applied, and penalizing the raw MLP output
            # instead would tax a correction the set-point never carried.  With a
            # zero-initialized head this is still exactly 0 wherever ``base``
            # itself sits inside the clamp -- which the default ``WaistAimConfig``
            # guarantees, since ``aim_yaw`` clamps to the same +/-1.74533 rad.
            # (Only a custom ``yaw_limit`` above the module clamp can put an
            # untrained ``sp`` off the analytic law, and then ``sp - base`` is the
            # honest report of that clamping.)
            yaw_now = ctx.yaw_now.to(mem.dtype)
            # ``.to(bool)``: replay/pool pipelines may round-trip the mask as 0/1
            # float; ``torch.where`` requires a bool condition.
            base = torch.where(
                ctx.aim_valid.to(torch.bool), ctx.aim_yaw.to(mem.dtype), yaw_now
            )
            h = F.silu(
                self.waist_sp_hidden(
                    torch.cat(
                        [pooled, yaw_now.unsqueeze(-1), base.unsqueeze(-1)], dim=-1
                    )
                )
            )
            res = self.waist_sp_out(h).squeeze(-1)
            yaw_sp = (base + res).clamp(-_WAIST_SP_LIMIT, _WAIST_SP_LIMIT)
            yaw_res = yaw_sp - base
            tok = self.waist_sp_proj(
                torch.stack([yaw_sp, yaw_now, yaw_sp - yaw_now], dim=-1)
            ) + self.waist_sp_emb
            mem = torch.cat([mem, tok.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)

        return ContextEncoding(
            mem, mask,
            waist_yaw_sp=yaw_sp, waist_yaw_res=yaw_res,
            waist_pr_sp=pr_sp, waist_pr_res=pr_res,
        )


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
        self._fk: ChainFK | None = None
        self._rnea: BatchedRNEA | None = None
        d, n = cfg.d_model, cfg.n_dof
        fw = cfg.fourier_width
        self.wp_proj = nn.Linear(2 * n + 2 * fw + 1, d)
        # ``dec_tth_feats`` adds one Fourier block per hand for the current-goal
        # countdown -- a wider projection, no extra token.
        self.state_proj = nn.Linear(2 * n + 2 * fw + (2 * fw if cfg.dec_tth_feats else 0), d)
        self.state_emb = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d, cfg.dec_heads, cfg.dec_ffn_mult, cfg.d_ctx, cfg.dropout,
                    rope=cfg.dec_rope, rope_base=cfg.rope_base,
                )
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
        # ``persistent=False``: these are recomputed from the URDF at every
        # construction, so keeping them out of the state_dict is what lets
        # ``clamp_traj_to_box`` be switched on over a checkpoint written before
        # it existed -- a persistent buffer would be a missing key and fail the
        # strict load, which is exactly what happened the first time.
        self.register_buffer("q_lo_box", torch.as_tensor(lo, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("q_hi_box", torch.as_tensor(hi, dtype=torch.float32),
                             persistent=False)
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

        # ANALYTIC PLANT FEATURES (``ModelConfig.dec_plant_feats``).  Separate
        # zero-init Linears so ``wp_proj`` / ``state_proj`` do not change shape
        # -- a warm start that predates the flag loads strictly, and the
        # untrained addition is a no-op.  FK / RNEA are NOT parameters: they
        # are rebuilt from the URDF whenever the flag is on, and moved with
        # the module via :meth:`_apply`.
        if cfg.dec_plant_feats:
            self.wp_plant_proj = nn.Linear(cfg.plant_feat_dim, d)
            self.state_plant_proj = nn.Linear(cfg.plant_feat_dim, d)
            nn.init.zeros_(self.wp_plant_proj.weight)
            nn.init.zeros_(self.wp_plant_proj.bias)
            nn.init.zeros_(self.state_plant_proj.weight)
            nn.init.zeros_(self.state_plant_proj.bias)
            self._fk = ChainFK(device="cpu", dtype=torch.float32)
            self._rnea = BatchedRNEA.from_urdf(device="cpu", dtype=torch.float32)
            self.register_buffer(
                "_tip_left",
                torch.as_tensor(stick_tip_offset("left"), dtype=torch.float32).view(1, 3),
                persistent=False,
            )
            self.register_buffer(
                "_tip_right",
                torch.as_tensor(stick_tip_offset("right"), dtype=torch.float32).view(1, 3),
                persistent=False,
            )

    def _apply(self, fn):
        out = super()._apply(fn)
        self._sync_plant_device()
        return out

    def _sync_plant_device(self) -> None:
        if self._fk is None or self._rnea is None:
            return
        # Always float32 on the module's device: autocast must not demote RNEA.
        dev = self.q_center.device
        self._fk.to(dev, torch.float32)
        self._rnea = self._rnea.to(dev, torch.float32)

    def _ee_points(self, q: torch.Tensor) -> torch.Tensor:
        """``(B, 6, 3)`` tip/wrist/elbow both hands, :data:`FK_POINT_NAMES` order."""
        frames = self._fk.forward(q)
        cols: dict[str, torch.Tensor] = {}
        for hand, tip in (("left", self._tip_left), ("right", self._tip_right)):
            R, p = frames[f"{hand}_wrist"]
            cols[f"{hand}_tip"] = ChainFK.points_world(R, p, tip.to(dtype=q.dtype))[:, 0]
            cols[f"{hand}_wrist"] = p
            cols[f"{hand}_elbow"] = frames[f"{hand}_elbow"][1]
        return torch.stack([cols[n] for n in FK_POINT_NAMES], dim=1)

    @torch.compiler.disable
    def _plant_tokens(
        self,
        init_traj: torch.Tensor,
        state: torch.Tensor,
        elem_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detached plant features of the current iterate.

        Returns ``(wp, state)`` of widths ``(B, H, plant_feat_dim)`` and
        ``(B, plant_feat_dim)``: flattened Cartesian ``p`` then ``tau / tau_lim``.

        Kept out of ``torch.compile``: FK/RNEA are Python-loop plants (the same
        ones ``compile_cost`` already leaves eager), and inductor/triton
        currently fails to compile this RNEA kernel.
        """
        with torch.no_grad():
            cfg = self.cfg
            n = cfg.n_dof
            dt = float(cfg.dt)
            q0 = state[:, :n].float()
            v0 = state[:, n:].float()
            x = hold_fill_waypoints(q0, init_traj.float(), elem_mask.float())
            B, T, _ = x.shape
            qf = x.reshape(B * T, n)
            pts = self._ee_points(qf).reshape(B, T, N_FK_POINTS, 3)
            qd = torch.cat([v0.unsqueeze(1), (x[:, 1:] - x[:, :-1]) / dt], dim=1)
            qdd = torch.cat(
                [(qd[:, 1:] - qd[:, :-1]) / dt, (qd[:, -1:] - qd[:, -2:-1]) / dt], dim=1
            )
            tau = self._rnea.rnea(
                qf, qd.reshape(B * T, n), qdd.reshape(B * T, n)
            ).reshape(B, T, n)
            tau_n = tau / self.tau_scale.float().clamp_min(1e-3).view(1, 1, -1)
            fk_n = pts / float(cfg.fk_pos_scale)
            feat = torch.cat([fk_n.reshape(B, T, -1), tau_n], dim=-1)
            return feat[:, 1:], feat[:, 0]

    # -- the fully tensor-in / tensor-out path (this is what gets compiled) --
    def forward(
        self,
        init_traj: torch.Tensor,  # (B, H, 17)
        state: torch.Tensor,  # (B, 34) = [q, qd]
        tau_left: torch.Tensor,  # (B, H)
        tau_right: torch.Tensor,  # (B, H)
        tau0_left: torch.Tensor,  # (B,)
        tau0_right: torch.Tensor,  # (B,)
        hit_tth: torch.Tensor,  # (B, 2) each hand's own strike, 0 = idle
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

        st_parts = [
            (state[:, :n] - qc) / qh,
            state[:, n:] / self.v_scale.to(dt),
            fourier_time_features(tau0_left, freqs, cfg.fourier_t_max),
            fourier_time_features(tau0_right, freqs, cfg.fourier_t_max),
        ]
        if cfg.dec_tth_feats:
            # The current-goal countdown the encoder no longer sees
            # (``ModelConfig.dec_tth_feats``).  Same Fourier bands as ``tau0_*``:
            # it is the same physical quantity on the same 0..~5 s range, so the
            # two codes are directly comparable to the network.
            st_parts.append(fourier_time_features(hit_tth[:, 0], freqs, cfg.fourier_t_max))
            st_parts.append(fourier_time_features(hit_tth[:, 1], freqs, cfg.fourier_t_max))
        st_feat = torch.cat(st_parts, dim=-1)
        st = (self.state_proj(st_feat) + self.state_emb).unsqueeze(1)

        if cfg.dec_plant_feats:
            # Observations of the current iterate, not a path through the plant:
            # the cost of the EMITTED traj is the teacher.  fp32 outside autocast
            # so RNEA does not run in bf16.
            amp = torch.amp.autocast
            device_type = "cuda" if init_traj.is_cuda else "cpu"
            with torch.no_grad(), amp(device_type=device_type, enabled=False):
                plant_wp, plant_st = self._plant_tokens(init_traj, state, elem_mask)
            plant_wp = plant_wp.to(dtype=dt) * tok_f
            wp = wp + self.wp_plant_proj(plant_wp)
            st = st + self.state_plant_proj(plant_st.to(dtype=dt)).unsqueeze(1)

        x = torch.cat([st, wp], dim=1)
        self_mask = torch.cat(
            [torch.ones_like(token_mask[:, :1]), token_mask], dim=1
        )  # the state token is always visible
        for blk in self.blocks:
            x = blk(x, self_mask, memory, memory_mask)
        h = self.norm(x)[:, 1:]

        if cfg.traj_head_fp32:
            # Autocast casts Linear inputs/weights back to its low precision
            # even when ``h.float()`` is passed, so disable it around the
            # complete trajectory readout/unprojection.  ``F.linear`` with
            # float views of the parameters preserves gradients to the original
            # parameters.  Keep construction, clamp, masking and frozen-column
            # replacement in this block's fp32 data path as well.
            device_type = "cuda" if h.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                raw = F.linear(
                    h.float(),
                    self.head_traj.weight.float(),
                    None if self.head_traj.bias is None else self.head_traj.bias.float(),
                )
                qh_traj = self.q_half.float()
                qc_traj = self.q_center.float()
                init_traj_out = init_traj.float()
                if cfg.output_mode == "residual":
                    traj = init_traj_out + qh_traj * raw
                else:
                    traj = qc_traj + qh_traj * raw
        else:
            raw = self.head_traj(h)
            if cfg.output_mode == "residual":
                traj = init_traj + qh * raw
            else:
                traj = qc + qh * raw
        if cfg.clamp_traj_to_box:
            # SOFT clip -- see ``clamp_traj_to_box`` for why it is not a hard one.
            # Identity inside the knee, a smooth C1 bend to the limit outside it.
            clamp_device_type = "cuda" if traj.is_cuda else "cpu"
            clamp_ctx = (
                torch.amp.autocast(
                    device_type=clamp_device_type, enabled=False
                )
                if cfg.traj_head_fp32
                else nullcontext()
            )
            with clamp_ctx:
                traj_dt = traj.dtype
                lo_b = self.q_lo_box.to(traj_dt).view(1, 1, -1)
                hi_b = self.q_hi_box.to(traj_dt).view(1, 1, -1)
                mid = 0.5 * (lo_b + hi_b)
                half = 0.5 * (hi_b - lo_b)
                knee = float(cfg.clamp_knee) * half
                band = (half - knee).clamp_min(1e-6)
                d = traj - mid
                core = d.clamp(-knee, knee)
                traj = mid + core + band * torch.tanh((d - core) / band)
        traj = traj * elem_mask.to(traj.dtype)

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
            traj_dt = traj.dtype
            free = self.free_cols.to(traj_dt)
            pin = 1.0 - free
            q_start = state[:, :n].to(traj_dt).unsqueeze(1)       # (B, 1, n)
            hold = self.hold_q.to(traj_dt).view(1, 1, -1)         # (1, 1, n)
            H_ = traj.shape[1]
            i = torch.arange(
                1, H_ + 1, device=traj.device, dtype=traj_dt
            ).view(1, -1, 1)
            u = (i / float(max(cfg.frozen_ramp_wp, 1))).clamp(0.0, 1.0)
            s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)        # quintic smoothstep
            pinned = q_start + (hold - q_start) * s                # (B, H, n)
            traj = traj * free + (pinned * pin) * elem_mask.to(traj_dt)
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
            masks.hit_tth,
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
