"""Isolated LoRA imitation trainer for optimized trajectory teachers.

This module deliberately has no server hooks.  It consumes exact tensor
snapshots produced by trajectory optimization, freezes a phase-2 actor, and
fits only rank-4 decoder LoRA matrices.  Source checkpoints and teacher bundles
are read-only; the sole output is an adapter-only checkpoint in an explicit
job directory.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from amortized_opt.cost import StrikeContext
from amortized_opt.model import AmortizedOptimizer, model_config_from_checkpoint
from amortized_opt.tokenizer import ContextInputs, WaypointMasks, build_waypoint_masks
from isaac_rl_ppo.actor import apply_lora

__all__ = [
    "ILTrainConfig",
    "ILTrainResult",
    "InferenceComparison",
    "TeacherBundle",
    "apply_il_adapter",
    "collate_training_records",
    "compare_saved_adapter",
    "load_backbone",
    "load_teacher_bundle",
    "save_teacher_bundle",
    "teacher_bundle_from_records",
    "train_il_adapter",
]


BUNDLE_FORMAT = "strike_pose_teacher_v1"
ADAPTER_FORMAT = "strike_pose_il_lora_v1"
LORA_TARGETS = ("o_proj", "w12", "w3")
_CTX_FIELDS = tuple(field.name for field in fields(ContextInputs))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _safe_torch_load(path: Path, *, map_location="cpu", allow_unsafe: bool = False):
    try:
        # Production phase-2 checkpoints retain tensor-valued sampling state in
        # these repository dataclasses, plus NumPy scalar RNG/config values.
        # Explicitly allowlisting only those known inert containers preserves
        # weights-only loading without requiring arbitrary pickle execution.
        numpy_core = getattr(np, "_core", None)
        if numpy_core is None:  # NumPy < 2.0
            numpy_core = np.core
        safe_types = [
            numpy_core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
            type(np.dtype(np.int64)),
            type(np.dtype(np.uint32)),
            ContextInputs,
            WaypointMasks,
            StrikeContext,
        ]
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not allow_unsafe:
            raise ValueError(
                f"safe tensor-only load failed for {path}; pass allow_unsafe=True "
                "only for a trusted local checkpoint"
            ) from exc
        return torch.load(path, map_location=map_location, weights_only=False)


def _context_dict(context: ContextInputs) -> dict[str, torch.Tensor]:
    return {name: getattr(context, name).detach().cpu() for name in _CTX_FIELDS}


def _index_context(context: ContextInputs, index: torch.Tensor) -> ContextInputs:
    return ContextInputs(**{
        name: getattr(context, name).index_select(0, index)
        for name in _CTX_FIELDS
    })


@dataclass
class TeacherBundle:
    """Exact actor inputs paired with an optimized waypoint trajectory.

    ``hit_index`` uses the repository's 1-based waypoint convention; inactive
    hands may contain ``-1`` and are gated by ``active``. ``teacher_mask`` is
    the authoritative imitation mask and normally equals the serialized
    scenario's ``WaypointMasks.elem_mask``. Optional ``strike`` tensors are
    preserved and batch-indexed for a caller-supplied differentiable physics
    loss; this module never guesses missing strike geometry.
    """

    init: torch.Tensor                 # (N, H, 17)
    state: torch.Tensor                # (N, 34), q0 then qd0
    context: ContextInputs | Mapping[str, torch.Tensor]
    hit_index: torch.Tensor            # (N, 2), 1-based
    active: torch.Tensor               # (N, 2) bool/0-1
    teacher_q: torch.Tensor            # (N, H, 17)
    teacher_mask: torch.Tensor         # (N, H, 17), nonnegative
    strike: Mapping[str, torch.Tensor | None] | None = None
    scenario_ids: tuple[str, ...] | list[str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.context, Mapping):
            self.context = ContextInputs(**dict(self.context))
        if self.scenario_ids is None:
            self.scenario_ids = tuple(f"scenario_{i:06d}" for i in range(self.init.shape[0]))
        else:
            self.scenario_ids = tuple(str(value) for value in self.scenario_ids)
        self.validate()

    @classmethod
    def from_records(cls, records: list[Mapping]) -> "TeacherBundle":
        """Collate ``TrajectoryScenario.training_bundle`` records.

        This is the narrow handoff from :mod:`trajectory_opt`: records retain
        their exact :class:`ContextInputs` and ``StrikeContext`` tensors rather
        than flattening geometry into a second, drifting schema.
        """
        if not records:
            raise ValueError("at least one teacher record is required")

        def tensor(name: str) -> torch.Tensor:
            values = [record[name] for record in records]
            if not all(torch.is_tensor(value) for value in values):
                raise ValueError(f"record field {name!r} must contain tensors")
            return torch.cat(values, dim=0)

        contexts = [record["context"] for record in records]
        if not all(isinstance(value, ContextInputs) for value in contexts):
            raise ValueError("record context must be ContextInputs")
        context = ContextInputs(**{
            name: torch.cat([getattr(value, name) for value in contexts], dim=0)
            for name in _CTX_FIELDS
        })
        strikes = [record.get("strike") for record in records]
        strike = None
        if any(value is not None for value in strikes):
            if not all(value is not None and hasattr(value, "__dict__") for value in strikes):
                raise ValueError("record strike values must all be StrikeContext-like objects")
            names = tuple(strikes[0].__dict__)
            strike = {}
            for name in names:
                values = [getattr(value, name) for value in strikes]
                if all(value is None for value in values):
                    strike[name] = None
                elif any(value is None for value in values):
                    raise ValueError(f"strike field {name!r} is inconsistently absent")
                else:
                    strike[name] = torch.cat(values, dim=0)
        ids = [str(record.get("scenario_id") or "") for record in records]
        return cls(
            init=tensor("init"), state=tensor("state"), context=context,
            hit_index=tensor("hit_index"), active=tensor("active"),
            teacher_q=tensor("teacher_q"), teacher_mask=tensor("teacher_mask"),
            strike=strike, scenario_ids=ids,
        )

    @property
    def size(self) -> int:
        return int(self.init.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.init.shape[1])

    def validate(self) -> None:
        if self.init.ndim != 3 or self.init.shape[-1] != 17:
            raise ValueError("init must have shape (N, H, 17)")
        n, horizon, dof = self.init.shape
        expected = (n, horizon, dof)
        if tuple(self.teacher_q.shape) != expected:
            raise ValueError(f"teacher_q must have shape {expected}")
        if tuple(self.teacher_mask.shape) != expected:
            raise ValueError(f"teacher_mask must have shape {expected}")
        if tuple(self.state.shape) != (n, 34):
            raise ValueError(f"state must have shape ({n}, 34)")
        if tuple(self.hit_index.shape) != (n, 2) or tuple(self.active.shape) != (n, 2):
            raise ValueError(f"hit_index and active must have shape ({n}, 2)")
        if self.context.batch != n:
            raise ValueError(f"context batch {self.context.batch} does not match N={n}")
        if len(self.scenario_ids) != n or len(set(self.scenario_ids)) != n \
                or any(not value for value in self.scenario_ids):
            raise ValueError("scenario_ids must contain N unique, nonempty strings")
        if n < 1 or horizon < 1:
            raise ValueError("teacher bundle may not be empty")
        for name, tensor in (
            ("init", self.init), ("state", self.state),
            ("teacher_q", self.teacher_q), ("teacher_mask", self.teacher_mask),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if bool((self.teacher_mask < 0).any()) or not bool((self.teacher_mask > 0).any()):
            raise ValueError("teacher_mask must be nonnegative and contain supervision")
        active = self.active.bool()
        if bool((active & ((self.hit_index < 1) | (self.hit_index > horizon))).any()):
            raise ValueError("active hit_index values must be in [1, H]")
        for name in _CTX_FIELDS:
            value = getattr(self.context, name)
            if not torch.is_tensor(value) or value.shape[0] != n:
                raise ValueError(f"context.{name} must be a tensor with leading size N")
        if self.strike is not None:
            for name, value in self.strike.items():
                if value is not None and (
                    not torch.is_tensor(value) or value.shape[0] != n
                ):
                    raise ValueError(f"strike.{name} must be None or a tensor with leading size N")

    def to(self, device: torch.device | str) -> "TeacherBundle":
        dev = torch.device(device)
        strike = None if self.strike is None else {
            name: None if value is None else value.to(dev)
            for name, value in self.strike.items()
        }
        return TeacherBundle(
            init=self.init.to(dev), state=self.state.to(dev), context=self.context.to(dev),
            hit_index=self.hit_index.to(dev), active=self.active.to(dev),
            teacher_q=self.teacher_q.to(dev), teacher_mask=self.teacher_mask.to(dev),
            strike=strike, scenario_ids=self.scenario_ids,
        )

    def index_select(self, index: torch.Tensor) -> "TeacherBundle":
        strike = None if self.strike is None else {
            name: None if value is None else value.index_select(0, index)
            for name, value in self.strike.items()
        }
        return TeacherBundle(
            init=self.init.index_select(0, index),
            state=self.state.index_select(0, index),
            context=_index_context(self.context, index),
            hit_index=self.hit_index.index_select(0, index),
            active=self.active.index_select(0, index),
            teacher_q=self.teacher_q.index_select(0, index),
            teacher_mask=self.teacher_mask.index_select(0, index),
            strike=strike,
            # Sampling is with replacement, so retain provenance while keeping
            # this derived mini-bundle's identifiers locally unique.
            scenario_ids=tuple(
                f"{self.scenario_ids[int(i)]}#batch{j}"
                for j, i in enumerate(index.detach().cpu())
            ),
        )

    def masks(self, *, dt: float) -> WaypointMasks:
        hit_s = self.hit_index.clamp_min(0).to(self.init.dtype) * float(dt)
        return build_waypoint_masks(
            hit_s[:, 0], hit_s[:, 1], max_horizon=self.horizon, dt=float(dt),
            active_left=self.active[:, 0], active_right=self.active[:, 1],
        )

    def payload(self) -> dict:
        return {
            "format": BUNDLE_FORMAT,
            "init": self.init.detach().cpu(),
            "state": self.state.detach().cpu(),
            "context": _context_dict(self.context),
            "hit_index": self.hit_index.detach().cpu(),
            "active": self.active.detach().cpu(),
            "teacher_q": self.teacher_q.detach().cpu(),
            "teacher_mask": self.teacher_mask.detach().cpu(),
            "scenario_ids": list(self.scenario_ids),
            "strike": None if self.strike is None else {
                name: None if value is None else value.detach().cpu()
                for name, value in self.strike.items()
            },
        }


def save_teacher_bundle(path: str | Path, bundle: TeacherBundle) -> Path:
    """Atomically save a tensor-only bundle suitable for a later server job."""
    out = Path(path)
    _atomic_torch_save(bundle.payload(), out)
    return out


def teacher_bundle_from_records(records: list[Mapping]) -> TeacherBundle:
    """Public functional spelling for server/job code."""
    return TeacherBundle.from_records(records)


def collate_training_records(records: list[Mapping]) -> TeacherBundle:
    """Collate a list of ``TrajectoryScenario.training_bundle`` dictionaries."""
    return TeacherBundle.from_records(records)


def load_teacher_bundle(
    path: str | Path, *, map_location="cpu", allow_unsafe: bool = False,
) -> TeacherBundle:
    raw = _safe_torch_load(Path(path), map_location=map_location, allow_unsafe=allow_unsafe)
    if not isinstance(raw, dict) or raw.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"not a {BUNDLE_FORMAT} bundle: {path}")
    return TeacherBundle(
        init=raw["init"], state=raw["state"], context=raw["context"],
        hit_index=raw["hit_index"], active=raw["active"],
        teacher_q=raw["teacher_q"], teacher_mask=raw["teacher_mask"],
        strike=raw.get("strike"), scenario_ids=raw.get("scenario_ids"),
    )


@dataclass
class ILTrainConfig:
    steps: int = 100
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 0.0
    rank: int = 4
    alpha: float = 8.0
    lora_dropout: float = 0.0
    teacher_weight: float = 1.0
    distill_weight: float = 0.05
    physics_weight: float = 0.0
    huber_beta: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    adapter_name: str = "adapter.pt"
    allow_unsafe_checkpoint: bool = False

    def validate(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.lr <= 0 or self.rank < 1 or self.alpha <= 0:
            raise ValueError("lr, rank and alpha must be positive")
        if self.huber_beta <= 0 or self.grad_clip <= 0:
            raise ValueError("huber_beta and grad_clip must be positive")
        if min(self.teacher_weight, self.distill_weight, self.physics_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.teacher_weight == self.distill_weight == self.physics_weight == 0:
            raise ValueError("at least one loss weight must be nonzero")
        if Path(self.adapter_name).name != self.adapter_name:
            raise ValueError("adapter_name must be a filename, not a path")


@dataclass
class InferenceComparison:
    base: torch.Tensor
    teacher: torch.Tensor
    adapted: torch.Tensor
    teacher_rmse_base_rad: float
    teacher_rmse_adapted_rad: float
    adapted_delta_rmse_rad: float


@dataclass
class ILTrainResult:
    adapter_path: Path
    lora_paths: tuple[str, ...]
    trainable_parameters: int
    history: list[dict[str, float]]
    comparison: InferenceComparison


def load_backbone(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    allow_unsafe: bool = False,
) -> tuple[AmortizedOptimizer, dict, str]:
    """Safely reconstruct and strictly load one phase-2 actor checkpoint."""
    path = Path(checkpoint).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"backbone checkpoint is not a file: {path}")
    digest = _sha256(path)
    raw = _safe_torch_load(path, map_location="cpu", allow_unsafe=allow_unsafe)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict) \
            or not isinstance(raw.get("model_cfg"), dict):
        raise ValueError("phase-2 checkpoint must contain model and model_cfg dictionaries")
    cfg = model_config_from_checkpoint(raw)
    model = AmortizedOptimizer(cfg)
    model.load_state_dict(raw["model"], strict=True)
    model.to(device)
    model.eval()
    return model, raw, digest


def _freeze_and_apply(model: AmortizedOptimizer, cfg: ILTrainConfig) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    paths = tuple(apply_lora(
        model.decoder, rank=cfg.rank, alpha=cfg.alpha,
        targets=LORA_TARGETS, dropout=cfg.lora_dropout,
    ))
    if not paths:
        raise RuntimeError("decoder LoRA target set matched no linear layers")
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(".lora_a" not in n and ".lora_b" not in n for n in trainable):
        raise RuntimeError(f"LoRA isolation failed; unexpected trainables: {trainable}")
    return paths


def _adapter_state(model: AmortizedOptimizer) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }


def _normalized_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    weight = mask.to(dtype=prediction.dtype)
    error = (prediction - target) / scale.view(1, 1, -1)
    absolute = error.abs()
    value = torch.where(
        absolute < float(beta),
        0.5 * error.square() / float(beta),
        absolute - 0.5 * float(beta),
    )
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _infer(model: AmortizedOptimizer, bundle: TeacherBundle) -> torch.Tensor:
    masks = bundle.masks(dt=model.cfg.dt)
    with torch.no_grad():
        encoded = model.encode_context(bundle.context)
        return model(bundle.init * masks.elem_mask, bundle.state, encoded, masks).traj.detach()


def _comparison(
    base: torch.Tensor, adapted: torch.Tensor, bundle: TeacherBundle,
) -> InferenceComparison:
    mask = bundle.teacher_mask.to(base.dtype)
    n = mask.sum().clamp_min(1.0)
    rmse = lambda value: float(((value.square() * mask).sum() / n).sqrt().cpu())  # noqa: E731
    return InferenceComparison(
        base=base.detach().cpu(), teacher=bundle.teacher_q.detach().cpu(),
        adapted=adapted.detach().cpu(),
        teacher_rmse_base_rad=rmse(base - bundle.teacher_q),
        teacher_rmse_adapted_rad=rmse(adapted - bundle.teacher_q),
        adapted_delta_rmse_rad=rmse(adapted - base),
    )


PhysicsLoss = Callable[[torch.Tensor, TeacherBundle, WaypointMasks], torch.Tensor]


def train_il_adapter(
    backbone_checkpoint: str | Path,
    bundle: TeacherBundle | str | Path,
    job_dir: str | Path,
    *,
    config: ILTrainConfig | None = None,
    physics_loss: PhysicsLoss | None = None,
) -> ILTrainResult:
    """Fit and atomically save an adapter without mutating the source actor.

    A physics term is intentionally dependency-injected.  The optimizer can
    supply a closure over its exact ``FusedCost`` and serialized strike context;
    silently reconstructing an incomplete physical scene here would be worse
    than omitting that term.  Setting ``physics_weight > 0`` without the
    callback is therefore an error.
    """
    cfg = config or ILTrainConfig()
    cfg.validate()
    source = Path(backbone_checkpoint).resolve(strict=True)
    output_dir = Path(job_dir).resolve(strict=False)
    if output_dir == source.parent:
        raise ValueError("job_dir must not be the source checkpoint directory")
    adapter_path = output_dir / cfg.adapter_name
    if adapter_path == source:
        raise ValueError("adapter output may not overwrite the source checkpoint")
    if adapter_path.exists():
        raise FileExistsError(f"adapter output already exists: {adapter_path}")
    if cfg.physics_weight > 0 and physics_loss is None:
        raise ValueError("physics_weight > 0 requires a physics_loss callback")

    teacher = (
        load_teacher_bundle(bundle, map_location=cfg.device)
        if isinstance(bundle, (str, Path)) else bundle.to(cfg.device)
    )
    model, raw_checkpoint, source_hash = load_backbone(
        source, device=cfg.device, allow_unsafe=cfg.allow_unsafe_checkpoint,
    )
    if teacher.init.shape[-1] != model.cfg.n_dof:
        raise ValueError("teacher joint width does not match the backbone")
    if teacher.context.num_drums != model.cfg.num_drums \
            or teacher.context.num_future != model.cfg.num_future:
        raise ValueError("teacher context shape does not match the backbone")
    actor_masks = teacher.masks(dt=model.cfg.dt)
    if bool(((teacher.teacher_mask > 0) & (actor_masks.elem_mask <= 0)).any()):
        raise ValueError("teacher_mask supervises elements outside the actor horizon mask")

    base = _infer(model, teacher)
    lora_paths = _freeze_and_apply(model, cfg)
    initial = _infer(model, teacher)
    if not torch.equal(base, initial):
        raise RuntimeError("zero-initialized LoRA changed the backbone at step zero")

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    generator = torch.Generator(device=teacher.init.device).manual_seed(cfg.seed)
    scale = model.decoder.q_half.detach().to(teacher.init.device).clamp_min(1e-3)
    history: list[dict[str, float]] = []
    model.eval()  # gradients stay enabled; this merely disables transformer dropout
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    metrics_path = output_dir / "metrics.jsonl"
    predictions_path = output_dir / "predictions.pt"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("job_dir already contains IL training outputs")
    _atomic_json({
        "status": "running", "step": 0, "steps": int(cfg.steps),
        "checkpoint": str(source), "bundle_size": teacher.size,
    }, status_path)

    with metrics_path.open("x", encoding="utf-8") as metrics_file:
        for step in range(cfg.steps):
            count = min(cfg.batch_size, teacher.size)
            index = torch.randint(
                teacher.size, (count,), generator=generator, device=teacher.init.device,
            )
            batch = teacher.index_select(index)
            masks = batch.masks(dt=model.cfg.dt)
            with torch.no_grad():
                encoded = model.encode_context(batch.context)
            output = model(batch.init * masks.elem_mask, batch.state, encoded, masks).traj
            teacher_loss = _normalized_huber(
                output, batch.teacher_q, batch.teacher_mask, scale, cfg.huber_beta,
            )
            base_batch = base.index_select(0, index)
            distill_loss = _normalized_huber(
                output, base_batch, masks.elem_mask, scale, cfg.huber_beta,
            )
            physics = output.new_zeros(())
            if physics_loss is not None and cfg.physics_weight > 0:
                physics = physics_loss(output, batch, masks)
                if physics.ndim != 0 or not torch.isfinite(physics):
                    raise ValueError("physics_loss must return one finite scalar tensor")
            total = (
                cfg.teacher_weight * teacher_loss
                + cfg.distill_weight * distill_loss
                + cfg.physics_weight * physics
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite LoRA gradient")
            optimizer.step()
            record = {
                "step": float(step + 1), "loss": float(total.detach()),
                "teacher_loss": float(teacher_loss.detach()),
                "distill_loss": float(distill_loss.detach()),
                "physics_loss": float(physics.detach()),
                "grad_norm": float(grad_norm.detach()),
            }
            history.append(record)
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            metrics_file.flush()
            _atomic_json({
                "status": "running", "step": step + 1, "steps": int(cfg.steps),
                "checkpoint": str(source), "bundle_size": teacher.size,
            }, status_path)

    adapted = _infer(model, teacher)
    comparison = _comparison(base, adapted, teacher)
    state = _adapter_state(model)
    if not state or any("lora_" not in name for name in state):
        raise RuntimeError("adapter checkpoint would contain no LoRA tensors")
    payload = {
        "format": ADAPTER_FORMAT,
        # Existing scripts/play_isaac.py consumes this adapter-only convention.
        "actor": state,
        "backbone_ckpt": str(source),
        "backbone_sha256": source_hash,
        "iteration": int(cfg.steps),
        "cfg": {"actor": {
            "lora_rank": int(cfg.rank), "lora_alpha": float(cfg.alpha),
            "lora_targets": list(LORA_TARGETS), "lora_decoder_only": True,
            "lora_dropout": float(cfg.lora_dropout),
        }},
        "il_train_cfg": asdict(cfg),
        "metrics": {
            "teacher_rmse_base_rad": comparison.teacher_rmse_base_rad,
            "teacher_rmse_adapted_rad": comparison.teacher_rmse_adapted_rad,
            "adapted_delta_rmse_rad": comparison.adapted_delta_rmse_rad,
            "final_loss": history[-1]["loss"],
        },
        "source_step": raw_checkpoint.get("step"),
    }
    _atomic_torch_save(payload, adapter_path)
    _atomic_torch_save({
        "format": "strike_pose_il_predictions_v1",
        "base": {
            scenario_id: comparison.base[i]
            for i, scenario_id in enumerate(teacher.scenario_ids)
        },
        "teacher": {
            scenario_id: comparison.teacher[i]
            for i, scenario_id in enumerate(teacher.scenario_ids)
        },
        "adapted": {
            scenario_id: comparison.adapted[i]
            for i, scenario_id in enumerate(teacher.scenario_ids)
        },
    }, predictions_path)
    if _sha256(source) != source_hash:
        raise RuntimeError("source checkpoint changed during isolated IL training")
    _atomic_json({
        "status": "complete", "step": int(cfg.steps), "steps": int(cfg.steps),
        "adapter": adapter_path.name, "predictions": predictions_path.name,
        "metrics": payload["metrics"], "bundle_size": teacher.size,
    }, status_path)
    return ILTrainResult(
        adapter_path=adapter_path, lora_paths=lora_paths,
        trainable_parameters=sum(p.numel() for p in parameters),
        history=history, comparison=comparison,
    )


def apply_il_adapter(model: AmortizedOptimizer, adapter: str | Path) -> tuple[str, ...]:
    """Apply one saved adapter to an already-loaded matching backbone."""
    raw = _safe_torch_load(Path(adapter), map_location="cpu")
    if not isinstance(raw, dict) or raw.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"not a {ADAPTER_FORMAT} checkpoint: {adapter}")
    actor_cfg = raw["cfg"]["actor"]
    cfg = ILTrainConfig(
        rank=int(actor_cfg["lora_rank"]), alpha=float(actor_cfg["lora_alpha"]),
        lora_dropout=float(actor_cfg.get("lora_dropout", 0.0)),
    )
    paths = _freeze_and_apply(model, cfg)
    named = dict(model.named_parameters())
    adapter_state = raw.get("actor") or {}
    expected = {name for name in named if name.endswith((".lora_a", ".lora_b"))}
    if set(adapter_state) != expected:
        raise ValueError(
            f"adapter tensor keys do not match model: missing={sorted(expected-set(adapter_state))[:4]} "
            f"unexpected={sorted(set(adapter_state)-expected)[:4]}"
        )
    with torch.no_grad():
        for name, value in adapter_state.items():
            parameter = named[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(f"adapter shape mismatch at {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return paths


def compare_saved_adapter(
    backbone_checkpoint: str | Path,
    adapter_checkpoint: str | Path,
    bundle: TeacherBundle | str | Path,
    *,
    device: str = "cpu",
    allow_unsafe_checkpoint: bool = False,
) -> InferenceComparison:
    """Run the explicit Base / Teacher / Adapted comparison used by the UI."""
    teacher = (
        load_teacher_bundle(bundle, map_location=device)
        if isinstance(bundle, (str, Path)) else bundle.to(device)
    )
    model, _raw, digest = load_backbone(
        backbone_checkpoint, device=device, allow_unsafe=allow_unsafe_checkpoint,
    )
    raw_adapter = _safe_torch_load(Path(adapter_checkpoint), map_location="cpu")
    if raw_adapter.get("backbone_sha256") != digest:
        raise ValueError("adapter was trained against a different backbone checkpoint")
    base = _infer(model, teacher)
    apply_il_adapter(model, adapter_checkpoint)
    adapted = _infer(model, teacher)
    return _comparison(base, adapted, teacher)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit an isolated decoder-only LoRA adapter to optimized teachers."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=0.05)
    parser.add_argument(
        "--allow-unsafe-checkpoint", action="store_true",
        help="allow pickle loading only for a trusted local checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    cfg = ILTrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=args.device,
        teacher_weight=args.teacher_weight, distill_weight=args.distill_weight,
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
    )
    try:
        result = train_il_adapter(
            args.checkpoint, args.bundle, output_dir, config=cfg,
        )
    except Exception as exc:
        _atomic_json({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        }, status_path)
        raise
    print(
        json.dumps({
            "adapter": str(result.adapter_path),
            "teacher_rmse_base_rad": result.comparison.teacher_rmse_base_rad,
            "teacher_rmse_adapted_rad": result.comparison.teacher_rmse_adapted_rad,
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
