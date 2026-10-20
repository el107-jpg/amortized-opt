"""Phase-1 warm-start training: losses, the step function and the loop.

Section 6 of ``docs/amortized_optimizer_plan.md``, under resolved decisions 5
and 6:

* phase 1 is a **short** warm start -- 20 k steps, hard cap, early-stopped on a
  val plateau.  Its job is to put the decoder in a trajectory-producing regime
  and stand the auxiliary heads up, nothing more; the quintic/waist-aim targets
  are deliberately non-optimal, so training them to convergence would only
  overfit the thing phase 2 exists to beat;
* the speed levers are all on: bf16 autocast, batch 512+, ``torch.compile`` over
  the forward *and* the losses, and horizon bucketing so the compiler sees three
  static shapes instead of one padded-to-100 shape.

The step function is deliberately a free function over ``(module, tensors)`` so
section 7 can wrap ``module`` in DDP and reuse it untouched.

Losses
------
Everything is masked by the per-element masks from
:func:`~amortized_opt.tokenizer.build_waypoint_masks` and reduced as a mean over
*valid* elements, so a 25-waypoint sample and a 100-waypoint sample contribute
equally (the plan's normalization note).

* trajectory -- **Huber** (``beta = 0.1 rad``) on the raw joint error.  Huber
  rather than MSE because the library terminal poses jump between waist-yaw grid
  entries: a handful of samples per batch legitimately need a ~1 rad correction,
  and under MSE those few dominate the gradient of all the rest.
* auxiliary heads -- Huber on the residual **normalized by a physical scale**
  (velocity rating, 30 rad/s^2, effort limit, 10 cm, 1 m/s), so one weight
  (``aux_weight``, default 0.1) is meaningful across all five.
"""

from __future__ import annotations

import json
import os
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .limits import ETA_VELOCITY, isaac_effort_limits, isaac_velocity_limits
from .model import AmortizedOptimizer, ModelConfig
from .tokenizer import joint_group_masks
from .warmstart import waist_slice
from .warmstart_data import NoiseConfig, ScenarioConfig, WarmstartPool
from .tensorboard_log import MetricsTensorboard

__all__ = [
    "METRIC_NAMES",
    "LossScales",
    "LossWeights",
    "TrainConfig",
    "batch_sampler",
    "forward_loss",
    "apply_waist_follow",
    "waist_identity_gain",
    "waist_follow_probe",
    "load_checkpoint",
    "make_step",
    "save_checkpoint",
    "train",
    "evaluate",
]

#: Order of the metric vector the compiled loss returns.
METRIC_NAMES: tuple[str, ...] = (
    "loss",
    "loss_traj",
    "loss_vel",
    "loss_acc",
    "loss_torque",
    "loss_fk_pos",
    "loss_fk_vel",
    "rmse_traj_rad",
    "rmse_vel_rads",
    "rmse_acc_rads2",
    "rmse_torque_nm",
    #: Torque RMSE in units of EACH JOINT'S OWN rating -- the normalized-space
    #: RMSE, before any scale is multiplied back in.  ``rmse_torque_nm`` above
    #: undoes the scale in a rating-AVERAGED sense (``mean(scale^2)``), so it is
    #: dominated by the 88 N*m waist and says almost nothing about the 5 N*m
    #: wrists, which are the joints that saturate.  This one weights every
    #: joint by its own motor.
    "rmse_torque_frac",
    #: ...and the same error restricted to the four wrist pitch/yaw joints,
    #: back in N*m against their own 5 N*m rating.
    "rmse_torque_wrist_nm",
    "rmse_fk_pos_m",
    "rmse_fk_vel_mps",
    "n_elem",
)


# =============================================================== weights =====
@dataclass
class LossWeights:
    traj: float = 1.0
    #: One weight for every auxiliary head (they are all scale-normalized).
    aux: float = 0.1
    vel: float = 1.0
    acc: float = 1.0
    torque: float = 1.0
    fk_pos: float = 1.0
    fk_vel: float = 1.0
    #: Huber transition, in the units each term is measured in.
    huber_traj: float = 0.1
    huber_aux: float = 0.1
    #: Weak encoder-head mimic (yaw → aim, P/R → 0).  Only trains residual
    #: MLPs; the decoder token is stopgrad.  0 keeps phase-1 bit-identical
    #: when gain is also 0.
    mimic: float = 0.0
    #: Identity Jacobian: ``(∂ q_hold / ∂ s − 1)²`` on the frozen hold
    #: window, so the decoder passes the encoder set-point through.  0 is off.
    gain: float = 0.0


class LossScales:
    """Per-term normalizers, as device buffers."""

    def __init__(
        self,
        device,
        dtype: torch.dtype = torch.float32,
        *,
        joint_names=G1_UPPER_BODY_JOINTS,
        accel: float = 30.0,
        fk_pos: float = 0.10,
        fk_vel: float = 1.0,
    ):
        t = lambda a: torch.as_tensor(np.asarray(a, float), device=device, dtype=dtype)
        self.vel = t(isaac_velocity_limits(joint_names)).view(1, 1, -1)
        self.torque = t(isaac_effort_limits(joint_names)).view(1, 1, -1)
        #: The four 5 N*m wrist pitch/yaw motors -- the ONLY joints that
        #: actually saturate, and the ones a rating-averaged N*m figure hides
        #: behind the 88 N*m waist.  Used for the ``rmse_torque_wrist_nm``
        #: diagnostic; not part of any loss.
        self.wrist_mask = t(
            [1.0 if re.search(r"wrist_(pitch|yaw)", n) else 0.0 for n in joint_names]
        ).view(1, 1, -1)
        self.acc = float(accel)
        self.fk_pos = float(fk_pos)
        self.fk_vel = float(fk_vel)


# ================================================================ losses =====
def _masked_huber(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(huber, squared-error sum, valid count) with ``mask`` applied.

    Both tensors are multiplied by the mask *before* the residual, so a masked
    element cannot influence the loss no matter what it holds.
    """
    d = (pred - target) * mask
    n = mask.sum().clamp_min(1.0)
    absd = d.abs()
    hub = torch.where(absd < beta, 0.5 * d * d / beta, absd - 0.5 * beta)
    return hub.sum() / n, (d * d).sum() / n, n


def forward_loss(
    module: torch.nn.Module,
    init: torch.Tensor,
    state: torch.Tensor,
    context,
    masks,
    target: torch.Tensor,
    aux_vel: torch.Tensor,
    aux_acc: torch.Tensor,
    aux_torque: torch.Tensor,
    aux_fk_pos: torch.Tensor,
    aux_fk_vel: torch.Tensor,
    scales: LossScales,
    weights: LossWeights,
    enc=None,
    aux_elem_mask=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One refinement iteration + every masked loss.  Returns ``(loss, metrics)``.

    ``metrics`` is a 1-D tensor in :data:`METRIC_NAMES` order.  Everything here
    is static-shape and branch-free so ``torch.compile`` sees one graph per
    horizon bucket.

    Optional ``enc`` skips the in-graph encode (follow mix encodes eagerly).
    Optional ``aux_elem_mask`` replaces ``elem_mask`` on vel/acc/torque only,
    so iv-row waist aux can be dropped without silencing the traj Huber.
    """
    if enc is None:
        enc = module.encode_context(context) if hasattr(module, "encode_context") else \
            module.module.encode_context(context)
    out = module(init, state, enc, masks)

    em = masks.elem_mask.to(target.dtype)
    em_aux = em if aux_elem_mask is None else aux_elem_mask.to(target.dtype)
    pm = masks.point_mask.to(target.dtype).unsqueeze(-1).expand_as(aux_fk_pos)

    l_traj, se_traj, _n = _masked_huber(out.traj, target, em, weights.huber_traj)
    l_vel, se_vel, _ = _masked_huber(
        out.vel / scales.vel, aux_vel / scales.vel, em_aux, weights.huber_aux
    )
    l_acc, se_acc, _ = _masked_huber(
        out.acc / scales.acc, aux_acc / scales.acc, em_aux, weights.huber_aux
    )
    l_tau, se_tau, n_elem = _masked_huber(
        out.torque / scales.torque, aux_torque / scales.torque, em_aux, weights.huber_aux
    )
    # Wrist-only torque error, in N*m against the wrists' own 5 N*m rating.
    # Same residual, a different reduction -- diagnostic only, no gradient.
    wm = em * scales.wrist_mask
    d_tau_w = (out.torque - aux_torque).detach() * wm
    se_tau_w = (d_tau_w * d_tau_w).sum() / wm.sum().clamp_min(1.0)
    l_fp, se_fp, _ = _masked_huber(
        out.fk_pos / scales.fk_pos, aux_fk_pos / scales.fk_pos, pm, weights.huber_aux
    )
    l_fv, se_fv, _ = _masked_huber(
        out.fk_vel / scales.fk_vel, aux_fk_vel / scales.fk_vel, pm, weights.huber_aux
    )

    loss = weights.traj * l_traj + weights.aux * (
        weights.vel * l_vel
        + weights.acc * l_acc
        + weights.torque * l_tau
        + weights.fk_pos * l_fp
        + weights.fk_vel * l_fv
    )

    # Physical-unit RMSEs (diagnostics only; the squared errors above are in
    # normalized units, so undo the scale in the mean-square sense).
    vs = (scales.vel * scales.vel).mean()
    ts = (scales.torque * scales.torque).mean()
    metrics = torch.stack(
        [
            loss.detach(),
            l_traj.detach(),
            l_vel.detach(),
            l_acc.detach(),
            l_tau.detach(),
            l_fp.detach(),
            l_fv.detach(),
            se_traj.detach().sqrt(),
            (se_vel.detach() * vs).sqrt(),
            se_acc.detach().sqrt() * scales.acc,
            (se_tau.detach() * ts).sqrt(),
            se_tau.detach().sqrt(),          # rmse_torque_frac (normalized space)
            se_tau_w.sqrt(),                 # rmse_torque_wrist_nm
            se_fp.detach().sqrt() * scales.fk_pos,
            se_fv.detach().sqrt() * scales.fk_vel,
            n_elem.detach(),
        ]
    )
    return loss, metrics


def make_step(
    module: torch.nn.Module,
    scales: LossScales,
    weights: LossWeights,
    *,
    compile_mode: str | None = "default",
    fullgraph: bool = False,
):
    """A closure over ``(module, scales, weights)`` returning ``(loss, metrics)``.

    ``torch.compile`` is applied to the whole forward + loss, not just the
    module, so the masked reductions fuse into the decoder's epilogue.  Three
    horizon buckets means three specializations; nothing else about the call is
    dynamic (``dynamic=False`` makes a fourth shape a hard recompile rather than
    a silent fall-back to a dynamic kernel).
    """

    def fn(init, state, context, masks, target, av, aa, at, afp, afv):
        return forward_loss(
            module, init, state, context, masks, target, av, aa, at, afp, afv,
            scales, weights,
        )

    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode, fullgraph=fullgraph, dynamic=False)


def unpack(batch: dict):
    """Batch dict -> the positional tensors :func:`make_step` wants."""
    a = batch["aux"]
    return (
        batch["init"], batch["state"], batch["context"], batch["masks"], batch["target"],
        a["vel"], a["acc"], a["torque"], a["fk_pos"], a["fk_vel"],
    )


def _core(module):
    return module.module if hasattr(module, "module") else module


def _heads_on(module) -> bool:
    cfg = _core(module).cfg
    return bool(cfg.waist_sp_head or cfg.waist_pr_sp_head)


def _follow_active(cfg: TrainConfig, weights: LossWeights, module) -> bool:
    return _heads_on(module) and (
        float(weights.mimic) > 0.0 or float(weights.gain) > 0.0
    )


def _elem_huber(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.1):
    d = pred - target
    a = d.abs()
    return torch.where(a < beta, 0.5 * d * d / beta, a - 0.5 * beta).mean()


def make_decode_step(
    module: torch.nn.Module,
    scales: LossScales,
    weights: LossWeights,
    *,
    compile_mode: str | None = "default",
    fullgraph: bool = False,
):
    """Like :func:`make_step` but takes a precomputed ``enc`` and aux mask."""

    def fn(init, state, enc, masks, target, av, aa, at, afp, afv, aux_em):
        return forward_loss(
            module, init, state, None, masks, target, av, aa, at, afp, afv,
            scales, weights, enc=enc, aux_elem_mask=aux_em,
        )

    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode, fullgraph=fullgraph, dynamic=False)


def apply_waist_follow(module, batch, enc, cfg: TrainConfig, *, generator=None):
    """Mimic + stopgrad token substitute.  Does not touch ``init`` or ``target``.

    The decoder sees the announced set-point; residual MLPs train only through
    ``l_mimic``.  Identity Jacobian is :func:`waist_identity_gain` (separate
    decode, frozen hold mask).  ``generator`` is unused (kept so old call
    sites still type-check).
    """
    from .train_cost import substitute_waist_setpoints

    del generator
    ctx = batch["context"]
    device = enc.memory.device
    dtype = enc.memory.dtype
    yaw_now = ctx.yaw_now.to(device=device, dtype=dtype)
    base_yaw = torch.where(
        ctx.aim_valid.to(device=device, dtype=torch.bool),
        ctx.aim_yaw.to(device=device, dtype=dtype),
        yaw_now,
    )

    l_mimic = enc.memory.new_zeros(())
    if enc.waist_yaw_sp is not None:
        l_mimic = l_mimic + _elem_huber(enc.waist_yaw_sp, base_yaw.detach())
    if enc.waist_pr_sp is not None:
        l_mimic = l_mimic + _elem_huber(
            enc.waist_pr_sp, torch.zeros_like(enc.waist_pr_sp)
        )

    yaw_use = None if enc.waist_yaw_sp is None else enc.waist_yaw_sp.detach()
    pr_use = None if enc.waist_pr_sp is None else enc.waist_pr_sp.detach()
    enc_dec = substitute_waist_setpoints(
        module, enc,
        yaw_sp=yaw_use, pr_sp=pr_use,
        yaw_now=yaw_now if yaw_use is not None else None,
        base_yaw=base_yaw if yaw_use is not None else None,
    )
    return enc_dec, batch["target"], batch["masks"].elem_mask, l_mimic, {"n_iv": 0}


def _index_encoding(enc, sl):
    opt = lambda v: None if v is None else v[sl]  # noqa: E731
    return type(enc)(
        enc.memory[sl], enc.mask[sl],
        opt(enc.waist_yaw_sp), opt(enc.waist_yaw_res),
        opt(enc.waist_pr_sp), opt(enc.waist_pr_res),
    )


def _encoding_float(enc):
    """fp32 copy so substitute + VJP do not run in bf16."""
    opt = lambda v: None if v is None else v.float()  # noqa: E731
    return type(enc)(
        enc.memory.detach().float(), enc.mask,
        opt(None if enc.waist_yaw_sp is None else enc.waist_yaw_sp.detach()),
        opt(None if enc.waist_yaw_res is None else enc.waist_yaw_res.detach()),
        opt(None if enc.waist_pr_sp is None else enc.waist_pr_sp.detach()),
        opt(None if enc.waist_pr_res is None else enc.waist_pr_res.detach()),
    )


def _index_masks(masks, sl):
    kw = {}
    for name in masks.__dataclass_fields__:
        t = getattr(masks, name)
        kw[name] = t[sl] if torch.is_tensor(t) else t
    return type(masks)(**kw)


def _hold_mask(elem_mask: torch.Tensor, col: int, hold_wp: int) -> torch.Tensor:
    """Frozen hold window: last ``hold_wp`` valid knots.  Independent of ``s``."""
    em = elem_mask[..., col]
    idx = torch.arange(em.shape[1], device=em.device, dtype=em.dtype).view(1, -1)
    H_w = (em * (idx + 1.0)).amax(-1).clamp_min(1.0)
    start = (H_w - float(hold_wp)).clamp_min(0.0)
    return (idx >= start.unsqueeze(-1)) & (idx < H_w.unsqueeze(-1)) & (em > 0.5)


def _hold_mean(series: torch.Tensor, hold_m: torch.Tensor) -> torch.Tensor:
    n = hold_m.float().sum(-1).clamp_min(1.0)
    return (series * hold_m.float()).sum(-1) / n


def waist_identity_gain(module, batch, enc, cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    """``mean((∂q_hold/∂s − 1)²)`` on a batch slice.  Leaf ``s`` is stopgrad(sp).

    Hold mask does not depend on ``s``.  Encoder residual MLPs are not in
    this graph.  Runs eager fp32 (no autocast) so the VJP is well-defined.
    """
    from .train_cost import substitute_waist_setpoints

    core = _core(module)
    names = list(G1_UPPER_BODY_JOINTS)
    ctx = batch["context"]
    B = enc.memory.shape[0]
    frac = float(cfg.waist_gain_frac)
    n = B if frac >= 1.0 else max(int(frac * B), 1) if frac > 0.0 else B
    n = min(n, B)
    sl = slice(0, n)

    enc_s = _encoding_float(_index_encoding(enc, sl))
    masks = _index_masks(batch["masks"], sl)
    init = batch["init"][sl].float()
    state = batch["state"][sl].float()
    yaw_now = ctx.yaw_now[sl].float()
    base_yaw = torch.where(ctx.aim_valid[sl], ctx.aim_yaw[sl].float(), yaw_now)

    yaw_leaf = pr_leaf = None
    if enc_s.waist_yaw_sp is not None:
        yaw_leaf = enc_s.waist_yaw_sp.detach().float().requires_grad_(True)
    if enc_s.waist_pr_sp is not None:
        pr_leaf = enc_s.waist_pr_sp.detach().float().requires_grad_(True)
    if yaw_leaf is None and pr_leaf is None:
        z = enc.memory.new_zeros(())
        return z, {"n": 0, "gain_yaw": float("nan"), "gain_pr": float("nan")}

    enc_g = substitute_waist_setpoints(
        module, enc_s,
        yaw_sp=yaw_leaf, pr_sp=pr_leaf,
        yaw_now=yaw_now if yaw_leaf is not None else None,
        base_yaw=base_yaw if yaw_leaf is not None else None,
    )
    # Flash / mem-efficient SDPA have no double backward; the identity
    # Jacobian needs create_graph=True, so force the math kernel.
    sdp = torch.nn.attention.sdpa_kernel
    backend = torch.nn.attention.SDPBackend.MATH
    with torch.autocast(device_type="cuda", enabled=False), sdp(backend):
        out = core(init, state, enc_g, masks)
    em = masks.elem_mask
    hold_wp = int(cfg.waist_hold_wp)
    terms: list[torch.Tensor] = []
    info: dict = {"n": n}

    if yaw_leaf is not None:
        yi = names.index("waist_yaw_joint")
        hm = _hold_mask(em, yi, hold_wp)
        hold = _hold_mean(out.traj[..., yi], hm)
        g = torch.autograd.grad(
            hold, yaw_leaf, torch.ones_like(hold), create_graph=True, retain_graph=True,
        )[0]
        terms.append((g - 1.0).square().mean())
        info["gain_yaw"] = float(g.detach().mean())
    else:
        info["gain_yaw"] = float("nan")

    if pr_leaf is not None:
        pi = names.index("waist_pitch_joint")
        ri = names.index("waist_roll_joint")
        hm = _hold_mask(em, pi, hold_wp)
        hold_p = _hold_mean(out.traj[..., pi], hm)
        hold_r = _hold_mean(out.traj[..., ri], hm)
        gp = torch.autograd.grad(
            hold_p, pr_leaf, torch.ones_like(hold_p), create_graph=True, retain_graph=True,
        )[0]
        gr = torch.autograd.grad(
            hold_r, pr_leaf, torch.ones_like(hold_r), create_graph=True, retain_graph=True,
        )[0]
        terms.append(
            (gp[:, 0] - 1.0).square().mean() + gp[:, 1].square().mean()
            + gr[:, 0].square().mean() + (gr[:, 1] - 1.0).square().mean()
        )
        info["gain_pr"] = float(0.5 * (gp[:, 0].detach().mean() + gr[:, 1].detach().mean()))
    else:
        info["gain_pr"] = float("nan")

    loss = terms[0] if len(terms) == 1 else (terms[0] + terms[1]) * 0.5
    return loss, info


# ============================================================== sampling =====
def batch_sampler(
    pool: WarmstartPool,
    batch_size: int,
    *,
    rng: np.random.Generator,
    steps: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """Yield ``(bucket, index tensor)`` pairs.

    Buckets are drawn in proportion to their population, so a step is
    bucket-homogeneous (one compiled shape) while the *distribution* over
    horizons is preserved.  With ``steps=None`` this is one epoch over every
    bucket.
    """
    sizes = {b: pool.size(b) for b in pool.buckets}
    total = sum(sizes.values())
    if total == 0:
        raise ValueError("empty pool")
    dev = pool.device
    if steps is None:
        order = []
        for b, n in sizes.items():
            idx = rng.permutation(n) if shuffle else np.arange(n)
            nb = n // batch_size if drop_last else math.ceil(n / batch_size)
            for i in range(nb):
                order.append((b, idx[i * batch_size : (i + 1) * batch_size]))
        if shuffle:
            rng.shuffle(order)
        for b, ix in order:
            yield int(b), torch.as_tensor(np.asarray(ix), device=dev, dtype=torch.long)
        return

    keys = list(sizes)
    p = np.asarray([sizes[b] for b in keys], dtype=float)
    p = p / p.sum()
    for _ in range(int(steps)):
        b = keys[int(rng.choice(len(keys), p=p))]
        ix = rng.integers(0, sizes[b], size=batch_size)
        yield int(b), torch.as_tensor(ix, device=dev, dtype=torch.long)


# ================================================================ config =====
@dataclass
class TrainConfig:
    """Phase-1 hyper-parameters.  ``max_steps`` is the plan's HARD cap."""

    max_steps: int = 20_000
    batch_size: int = 512
    lr: float = 3e-4
    min_lr_frac: float = 0.05
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    grad_clip: float = 1.0

    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    compile_mode: str | None = "default"
    compile_fullgraph: bool = False
    tf32: bool = True

    eval_every: int = 500
    eval_batches: int = 12
    log_every: int = 50
    ckpt_every: int = 2_000

    #: Early stopping on the val plateau (still capped by ``max_steps``).
    patience_evals: int = 8
    min_delta: float = 1e-4

    seed: int = 0
    device: str = "cuda:1"
    tensorboard: bool = True

    #: WHICH torque the aux torque head is regressed onto.  ``False`` (default)
    #: is the RNEA segment max every phase-1 checkpoint before 2026-08-16 was
    #: trained on; ``True`` is the DEMANDED PD tracking torque, the quantity
    #: :attr:`~amortized_opt.cost.CostConfig.w_track_torque` prices.  The label
    #: itself is baked into the pool
    #: (:class:`~amortized_opt.warmstart_data.WarmstartPool.aux_torque_tracking`);
    #: this field records the choice in the checkpoint and :func:`train`
    #: refuses to run if the two disagree, so a checkpoint can never claim one
    #: definition while its head learned the other.
    aux_torque_tracking: bool = False

    #: Removed: jitter + rewritten-target mix.  Non-zero is a hard error;
    #: use :attr:`LossWeights.gain` (identity Jacobian) instead.
    waist_follow_frac: float = 0.0
    waist_follow_jitter: float = 0.3
    waist_pr_follow_jitter: float = 0.1
    #: Frozen hold window (waypoints) for the identity Jacobian and the probe.
    waist_hold_wp: int = 10
    #: Fraction of the batch that pays for ``∂q_hold/∂s`` (operating point,
    #: no token noise).  1 = every row.
    waist_gain_frac: float = 1.0

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / max(self.warmup_steps, 1)
        t = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.lr * (self.min_lr_frac + (1.0 - self.min_lr_frac) * cos)


# =========================================================== checkpoints =====
def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    scenario_cfg: ScenarioConfig | None = None,
    noise_cfg: NoiseConfig | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": int(step),
            "model": core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_cfg": asdict(cfg),
            "model_cfg": asdict(model_cfg),
            "scenario_cfg": asdict(scenario_cfg) if scenario_cfg is not None else None,
            "noise_cfg": asdict(noise_cfg) if noise_cfg is not None else None,
            "extra": dict(extra or {}),
            "torch_rng": torch.get_rng_state(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location=None,
) -> dict:
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return ck


def build_model(cfg: ModelConfig, device, dtype: torch.dtype = torch.float32):
    return AmortizedOptimizer(cfg).to(device=device, dtype=dtype)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay off for norms / biases / embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or name.endswith("state_emb") or "emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=tuple(cfg.betas),
    )


# =============================================================== the loop ====
@torch.no_grad()
def evaluate(
    step_fn,
    pool: WarmstartPool,
    batch_size: int,
    *,
    n_batches: int,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
    device: str = "cuda",
    hold_init: bool = False,
) -> dict[str, float]:
    """Deterministic pass over ``n_batches`` val minibatches."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=pool.device)
    gen.manual_seed(int(seed))
    acc = np.zeros(len(METRIC_NAMES))
    m = 0
    for b, idx in batch_sampler(pool, batch_size, rng=rng, steps=int(n_batches)):
        batch = pool.batch(b, idx, generator=gen, hold_init=hold_init)
        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with ctx:
            _loss, metrics = step_fn(*unpack(batch))
        acc += metrics.float().cpu().numpy()
        m += 1
    return dict(zip(METRIC_NAMES, (acc / max(m, 1)).tolist()))


def train(
    *,
    model: torch.nn.Module,
    train_pool: WarmstartPool,
    val_pool: WarmstartPool,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    weights: LossWeights | None = None,
    run_dir: str | Path,
    scenario_cfg: ScenarioConfig | None = None,
    noise_cfg: NoiseConfig | None = None,
    log: bool = True,
) -> dict:
    """The phase-1 loop.  Returns a summary dict; writes checkpoints + JSONL."""
    for name, pool in (("train_pool", train_pool), ("val_pool", val_pool)):
        got = bool(getattr(pool, "aux_torque_tracking", False))
        if got != bool(cfg.aux_torque_tracking):
            raise ValueError(
                f"{name}.aux_torque_tracking={got} but "
                f"TrainConfig.aux_torque_tracking={cfg.aux_torque_tracking}; the "
                "torque label is baked into the pool, so these must agree"
            )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")

    weights = weights or LossWeights()
    scales = LossScales(device)
    amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.amp_dtype)
    opt = build_optimizer(model, cfg)
    step_fn = make_step(
        model, scales, weights,
        compile_mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph,
    )
    # Validation runs the EAGER function on purpose: `model.eval()` and
    # `no_grad` are both dynamo guards, so reusing the compiled step would add
    # three more specializations (one per bucket) for 12 batches every 500
    # steps.  Eager val costs ~1 s per eval; the extra compiles cost ~100 s.
    eval_fn = make_step(model, scales, weights, compile_mode=None)

    if float(cfg.waist_follow_frac) > 0.0:
        raise ValueError(
            "waist_follow_frac is removed; set LossWeights.gain so "
            "∂q_hold/∂s = 1 (no token jitter, no rewritten target)"
        )
    if (weights.mimic > 0.0 or weights.gain > 0.0) and not _heads_on(model):
        raise ValueError(
            "mimic / gain require ModelConfig.waist_sp_head or waist_pr_sp_head"
        )
    follow = _follow_active(cfg, weights, model)
    decode_fn = (
        make_decode_step(
            model, scales, weights,
            compile_mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph,
        )
        if follow else None
    )

    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=train_pool.device)
    gen.manual_seed(int(cfg.seed) + 7919)

    metrics_path = run_dir / "metrics.jsonl"
    fh = open(metrics_path, "a")
    tb = MetricsTensorboard(run_dir / "tb", enabled=cfg.tensorboard)
    if cfg.tensorboard and os.environ.get("TB_AUTO_REGISTER", "1") != "0":
        registry = Path(__file__).resolve().parent.parent / "runs/tensorboard_active"
        registry.mkdir(exist_ok=True)
        link = registry / run_dir.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to((run_dir / "tb").resolve())

    def emit(rec: dict) -> None:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        tb.log_record(dict(rec, event="eval") if rec.get("event") == "val" else rec)

    emit({
        "event": "start",
        "train_cfg": asdict(cfg),
        "model_cfg": asdict(model_cfg),
        "weights": asdict(weights),
        "scenario_cfg": asdict(scenario_cfg) if scenario_cfg else None,
        "noise_cfg": asdict(noise_cfg) if noise_cfg else None,
        "train_pool": train_pool.distribution(),
        "val_pool": val_pool.distribution(),
    })

    best = float("inf")
    best_step = 0
    since_improve = 0
    bucket_counts: dict[int, int] = {b: 0 for b in train_pool.buckets}
    run_acc = np.zeros(len(METRIC_NAMES))
    run_n = 0
    t_start = time.perf_counter()
    t_last = t_start
    step = 0
    stopped = "max_steps"

    sampler = batch_sampler(train_pool, cfg.batch_size, rng=rng, steps=cfg.max_steps)
    for step, (bucket, idx) in enumerate(sampler):
        lr = cfg.lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = train_pool.batch(bucket, idx, generator=gen)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        ctx = (
            torch.autocast(device_type="cuda", dtype=amp)
            if amp is not None
            else torch.autocast(device_type="cuda", enabled=False)
        )
        g_info: dict = {}
        with ctx:
            if follow:
                enc = (
                    model.encode_context(batch["context"])
                    if hasattr(model, "encode_context")
                    else model.module.encode_context(batch["context"])
                )
                enc_d, target, aux_em, l_mimic, _info = apply_waist_follow(
                    model, batch, enc, cfg, generator=gen,
                )
                a = batch["aux"]
                loss, metrics = decode_fn(
                    batch["init"], batch["state"], enc_d, batch["masks"], target,
                    a["vel"], a["acc"], a["torque"], a["fk_pos"], a["fk_vel"], aux_em,
                )
                if weights.mimic > 0.0:
                    loss = loss + weights.mimic * l_mimic
            else:
                loss, metrics = step_fn(*unpack(batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Second backward: identity Jacobian.  Must not share the live Huber
        # graph (a 1024-row create_graph decode on top of BC OOM'd 32 GB).
        if follow and weights.gain > 0.0:
            l_gain, g_info = waist_identity_gain(model, batch, enc, cfg)
            (weights.gain * l_gain).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        run_acc += metrics.float().detach().cpu().numpy()
        run_n += 1

        if (step + 1) % cfg.log_every == 0:
            now = time.perf_counter()
            rec = {"event": "train", "step": step + 1, "lr": lr,
                   "grad_norm": float(gnorm), "steps_per_s": cfg.log_every / (now - t_last),
                   "elapsed_s": now - t_start}
            rec.update({k: float(v) for k, v in zip(METRIC_NAMES, run_acc / max(run_n, 1))})
            if g_info:
                rec["gain_yaw"] = g_info.get("gain_yaw")
                rec["gain_pr"] = g_info.get("gain_pr")
            emit(rec)
            if log:
                print(
                    f"[{step+1:6d}] loss {rec['loss']:.4f} traj {rec['rmse_traj_rad']*1000:7.1f} mrad "
                    f"tau {rec['rmse_torque_nm']:6.2f} Nm "
                    f"({rec['rmse_torque_frac']:.3f} rat, "
                    f"wrist {rec['rmse_torque_wrist_nm']:5.2f} Nm) "
                    f"fk {rec['rmse_fk_pos_m']*100:5.2f} cm "
                    f"| {rec['steps_per_s']:.1f} it/s",
                    flush=True,
                )
            run_acc[:] = 0.0
            run_n = 0
            t_last = now

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.max_steps:
            model.eval()
            vm = evaluate(
                eval_fn, val_pool, cfg.batch_size, n_batches=cfg.eval_batches,
                seed=cfg.seed + 1, amp_dtype=amp, device=cfg.device,
            )
            model.train()
            emit({"event": "val", "step": step + 1,
                  "elapsed_s": time.perf_counter() - t_start, **vm})
            if log:
                print(
                    f"    val loss {vm['loss']:.4f} traj {vm['rmse_traj_rad']*1000:7.1f} mrad",
                    flush=True,
                )
            if vm["loss"] < best - cfg.min_delta:
                best, best_step, since_improve = vm["loss"], step + 1, 0
                save_checkpoint(
                    run_dir / "best.pt", model=model, optimizer=opt, step=step + 1,
                    cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
                    noise_cfg=noise_cfg, extra={"val": vm},
                )
            else:
                since_improve += 1
                if since_improve >= cfg.patience_evals:
                    stopped = "plateau"
                    emit({"event": "early_stop", "step": step + 1, "best": best,
                          "best_step": best_step})
                    if log:
                        print(f"    early stop at {step+1} (best {best:.4f} @ {best_step})")
                    break

        if (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(
                run_dir / "last.pt", model=model, optimizer=opt, step=step + 1,
                cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
                noise_cfg=noise_cfg,
            )

    total = time.perf_counter() - t_start
    save_checkpoint(
        run_dir / "last.pt", model=model, optimizer=opt, step=step + 1, cfg=cfg,
        model_cfg=model_cfg, scenario_cfg=scenario_cfg, noise_cfg=noise_cfg,
    )
    summary = {
        "event": "done",
        "steps": step + 1,
        "wall_s": total,
        "steps_per_s": (step + 1) / max(total, 1e-9),
        "best_val": best,
        "best_step": best_step,
        "stopped": stopped,
        "bucket_counts": bucket_counts,
        "recompiles": int(
            torch._dynamo.utils.counters["frames"]["ok"]
            if "frames" in torch._dynamo.utils.counters else 0
        ),
    }
    emit(summary)
    fh.close()
    tb.close()
    return summary


# ========================================================== diagnostics ======
@torch.no_grad()
def imitation_report(
    model: torch.nn.Module,
    pool: WarmstartPool,
    *,
    batch_size: int = 256,
    n_batches: int = 12,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
    hold_init: bool = False,
) -> dict:
    """Per-joint-group RMSE, terminal error and aux-head accuracy on ``pool``."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=pool.device)
    gen.manual_seed(int(seed))
    groups = joint_group_masks(pool.cfg.joint_names)
    gsum = {k: 0.0 for k in groups}
    gcnt = {k: 0.0 for k in groups}
    term_se, term_n = 0.0, 0.0
    term_max = 0.0
    aux_se = {k: 0.0 for k in ("vel", "acc", "torque", "fk_pos", "fk_vel")}
    aux_ss = {k: 0.0 for k in aux_se}
    aux_n = {k: 0.0 for k in aux_se}
    was_training = model.training
    model.eval()
    for b, idx in batch_sampler(pool, batch_size, rng=rng, steps=int(n_batches)):
        batch = pool.batch(b, idx, generator=gen, hold_init=hold_init)
        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with ctx:
            enc = model.encode_context(batch["context"])
            out = model(batch["init"], batch["state"], enc, batch["masks"])
        em = batch["masks"].elem_mask.float()
        d = (out.traj.float() - batch["target"].float()) * em
        for k, sel in groups.items():
            s = torch.as_tensor(sel, device=d.device, dtype=d.dtype).view(1, 1, -1)
            gsum[k] += float((d * d * s).sum())
            gcnt[k] += float((em * s).sum())
        # terminal waypoint of each active hand
        for hi, hand in enumerate(("left", "right")):
            k = batch["hit"][:, hi]
            act = batch["active"][:, hi]
            cols = torch.as_tensor(
                np.nonzero(groups[hand] > 0.5)[0], device=d.device, dtype=torch.long
            )
            gi = (k.clamp(1, d.shape[1]) - 1).view(-1, 1, 1).expand(-1, 1, d.shape[2])
            row = d.gather(1, gi).squeeze(1).index_select(1, cols)
            w = act.float().unsqueeze(-1)
            term_se += float(((row * row) * w).sum())
            term_n += float(w.sum() * cols.numel())
            if float(w.sum()):
                term_max = max(term_max, float((row.abs() * w).max()))
        pm = batch["masks"].point_mask.float().unsqueeze(-1)
        for key, pred, tgt, mask in (
            ("vel", out.vel, batch["aux"]["vel"], em),
            ("acc", out.acc, batch["aux"]["acc"], em),
            ("torque", out.torque, batch["aux"]["torque"], em),
            ("fk_pos", out.fk_pos, batch["aux"]["fk_pos"], pm.expand_as(out.fk_pos)),
            ("fk_vel", out.fk_vel, batch["aux"]["fk_vel"], pm.expand_as(out.fk_vel)),
        ):
            e = (pred.float() - tgt.float()) * mask
            aux_se[key] += float((e * e).sum())
            aux_ss[key] += float(((tgt.float() * mask) ** 2).sum())
            aux_n[key] += float(mask.sum())
    if was_training:
        model.train()
    rep = {
        "rmse_by_group_rad": {
            k: math.sqrt(gsum[k] / max(gcnt[k], 1.0)) for k in groups
        },
        "terminal_rmse_rad": math.sqrt(term_se / max(term_n, 1.0)),
        "terminal_max_rad": term_max,
    }
    for k in aux_se:
        rmse = math.sqrt(aux_se[k] / max(aux_n[k], 1.0))
        rms = math.sqrt(aux_ss[k] / max(aux_n[k], 1.0))
        rep[f"aux_{k}_rmse"] = rmse
        rep[f"aux_{k}_rel"] = rmse / max(rms, 1e-12)
    return rep


@torch.no_grad()
def waist_follow_probe(
    model: torch.nn.Module,
    pool: WarmstartPool,
    *,
    batch_size: int = 64,
    n_batches: int = 8,
    seed: int = 0,
    yaw_deltas: tuple[float, ...] = (0.30, -0.30, 0.15, -0.15),
) -> dict:
    """Causal token-read check: decode original init against ``sp+Δ``.

    Never used as a training or val loss.  Reports ``corr(Δ, hold−aim)``,
    gain, and hold RMSE to ``sp_use`` vs aim.  A decoder that merely clones
    the quintic scores corr ≈ 0; one that reads the token tracks Δ.
    """
    from .train_cost import substitute_waist_setpoints, waist_sp_bound

    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=pool.device)
    gen.manual_seed(int(seed) + 17)
    core = _core(model)
    if not core.cfg.waist_sp_head:
        raise ValueError("waist_follow_probe requires waist_sp_head")
    names = list(G1_UPPER_BODY_JOINTS)
    yaw_i = names.index("waist_yaw_joint")
    bound = float(waist_sp_bound())
    was_training = model.training
    model.eval()
    deltas, hold_minus_aim, err_sp, err_aim = [], [], [], []
    n_iv_hold = 0
    for b, idx in batch_sampler(pool, batch_size, rng=rng, steps=int(n_batches)):
        batch = pool.batch(b, idx, generator=gen, hold_init=False)
        enc = core.encode_context(batch["context"])
        ctx = batch["context"]
        yaw_now = ctx.yaw_now.to(enc.waist_yaw_sp.dtype)
        aim = torch.where(ctx.aim_valid.to(torch.bool), ctx.aim_yaw.to(yaw_now.dtype), yaw_now)
        em = batch["masks"].elem_mask
        B, H = em.shape[:2]
        # cycle signed deltas across the batch
        dlt = torch.as_tensor(
            [yaw_deltas[i % len(yaw_deltas)] for i in range(B)],
            device=enc.waist_yaw_sp.device, dtype=enc.waist_yaw_sp.dtype,
        )
        sp_use = (enc.waist_yaw_sp + dlt).clamp(-bound, bound)
        enc_d = substitute_waist_setpoints(
            core, enc, yaw_sp=sp_use, yaw_now=yaw_now, base_yaw=aim,
        )
        out = core(batch["init"], batch["state"], enc_d, batch["masks"])
        idx_t = torch.arange(H, device=em.device, dtype=em.dtype).view(1, H)
        H_w = (em[..., yaw_i] * (idx_t + 1.0)).amax(-1).clamp_min(1.0)
        q0y = batch["q0"][:, yaw_i]
        step = float(ETA_VELOCITY) * float(isaac_velocity_limits(names)[yaw_i]) * float(CONTROL_DT)
        need = torch.ceil(1.875 * (sp_use - q0y).abs() / max(step, 1e-9))
        hi = H_w.clamp_min(1.0)
        lo = torch.minimum(torch.full_like(hi, 10.0), hi)
        kstar = need.clamp(min=lo, max=hi)
        t = torch.arange(H, device=em.device, dtype=em.dtype).view(1, H)
        hold_m = (t >= (kstar - 1.0).unsqueeze(-1)) & (em[..., yaw_i] > 0.5)
        n = hold_m.float().sum(-1).clamp_min(1.0)
        hold = (out.traj[..., yaw_i] * hold_m.float()).sum(-1) / n
        ok = (H_w >= 2.0) & (hold_m.any(-1))
        if not bool(ok.any()):
            continue
        n_iv_hold += int(ok.sum())
        deltas.append(dlt[ok].detach().cpu())
        hold_minus_aim.append((hold - aim)[ok].detach().cpu())
        err_sp.append((hold - sp_use)[ok].abs().detach().cpu())
        err_aim.append((hold - aim)[ok].abs().detach().cpu())
    if was_training:
        model.train()
    if not deltas:
        return {"n": 0, "corr": 0.0, "gain": 0.0,
                "rmse_to_sp": float("nan"), "rmse_to_aim": float("nan")}
    d = torch.cat(deltas).numpy()
    y = torch.cat(hold_minus_aim).numpy()
    corr = float(np.corrcoef(d, y)[0, 1]) if d.size > 2 else 0.0
    var = float(np.var(d))
    gain = float(np.cov(d, y)[0, 1] / var) if var > 1e-12 else 0.0
    return {
        "n": int(n_iv_hold),
        "corr": corr if np.isfinite(corr) else 0.0,
        "gain": gain if np.isfinite(gain) else 0.0,
        "rmse_to_sp": float(torch.cat(err_sp).pow(2).mean().sqrt()),
        "rmse_to_aim": float(torch.cat(err_aim).pow(2).mean().sqrt()),
        "mean_abs_delta": float(np.abs(d).mean()),
    }
