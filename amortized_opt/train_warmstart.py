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
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import G1_UPPER_BODY_JOINTS
from .limits import isaac_effort_limits, isaac_velocity_limits
from .model import AmortizedOptimizer, ModelConfig
from .tokenizer import joint_group_masks
from .warmstart_data import NoiseConfig, ScenarioConfig, WarmstartPool

__all__ = [
    "METRIC_NAMES",
    "LossScales",
    "LossWeights",
    "TrainConfig",
    "batch_sampler",
    "forward_loss",
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """One refinement iteration + every masked loss.  Returns ``(loss, metrics)``.

    ``metrics`` is a 1-D tensor in :data:`METRIC_NAMES` order.  Everything here
    is static-shape and branch-free so ``torch.compile`` sees one graph per
    horizon bucket.
    """
    enc = module.encode_context(context) if hasattr(module, "encode_context") else \
        module.module.encode_context(context)
    out = module(init, state, enc, masks)

    em = masks.elem_mask.to(target.dtype)
    pm = masks.point_mask.to(target.dtype).unsqueeze(-1).expand_as(aux_fk_pos)

    l_traj, se_traj, _n = _masked_huber(out.traj, target, em, weights.huber_traj)
    l_vel, se_vel, _ = _masked_huber(
        out.vel / scales.vel, aux_vel / scales.vel, em, weights.huber_aux
    )
    l_acc, se_acc, _ = _masked_huber(
        out.acc / scales.acc, aux_acc / scales.acc, em, weights.huber_aux
    )
    l_tau, se_tau, n_elem = _masked_huber(
        out.torque / scales.torque, aux_torque / scales.torque, em, weights.huber_aux
    )
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

    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=train_pool.device)
    gen.manual_seed(int(cfg.seed) + 7919)

    metrics_path = run_dir / "metrics.jsonl"
    fh = open(metrics_path, "a")

    def emit(rec: dict) -> None:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

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
        with ctx:
            loss, metrics = step_fn(*unpack(batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
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
            emit(rec)
            if log:
                print(
                    f"[{step+1:6d}] loss {rec['loss']:.4f} traj {rec['rmse_traj_rad']*1000:7.1f} mrad "
                    f"tau {rec['rmse_torque_nm']:6.2f} Nm fk {rec['rmse_fk_pos_m']*100:5.2f} cm "
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
