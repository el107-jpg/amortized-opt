#!/usr/bin/env python
"""Compare critic widths and decaying/replaced functional anchor policies."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from amortized_opt.monotonic_critic import (
    MonotonicConfidenceConfig,
    MonotonicConfidenceTrainer,
)
from scripts.ablate_monotonic_confidence_critic import calibrate_monotonic_rule
from scripts.evaluate_monotonic_critic_ood import (
    evaluate,
    load_rows,
    load_rows_with_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates-per-batch", type=int, default=50)
    parser.add_argument("--stream-batch", type=int, default=256)
    parser.add_argument("--anchor-capacity", type=int, default=2048)
    parser.add_argument("--anchor-half-life-rows", type=int, default=55_296)
    parser.add_argument("--max-false-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def compact_metrics(
    trainer: MonotonicConfidenceTrainer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rule: dict,
    initial_prediction: torch.Tensor | None = None,
) -> dict[str, object]:
    result, _ = evaluate(
        trainer, x, y,
        residual_rule=rule, joint_rule=rule,
    )
    decision = result["decisions"]["joint_with_support_tempered_confidence"]
    out: dict[str, object] = {
        "rows": result["rows"],
        "mae_normalized": result["mae_normalized"],
        "constraint_sign_accuracy": result["constraint_sign_accuracy"],
        "confidence_brier": result["confidence_brier"],
        "confidence_correct_auc": result["confidence_correct_auc"],
        "confidence_mean": result["confidence"]["mean"],
        "support_mean": result["support"]["mean"],
        "decision": decision,
    }
    if initial_prediction is not None:
        with torch.no_grad():
            current = trainer.predict(x.to(trainer.device)).mean(dim=0).cpu()
        drift = (current - initial_prediction).abs()
        out["prediction_drift_mae"] = float(drift.mean())
        out["prediction_drift_p95"] = float(torch.quantile(drift, 0.95))
    return out


def split_original(
    reference: str, report: dict,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    x, y, steps, _cfg = load_rows_with_metadata(reference)
    unique = steps.unique(sorted=True)
    holdout_n = int(report["holdout_actor_steps"])
    calibration_n = int(report["calibration_actor_steps"])
    calibration_start = unique[-holdout_n - calibration_n]
    holdout_start = unique[-holdout_n]
    masks = (
        steps < calibration_start,
        (steps >= calibration_start) & (steps < holdout_start),
        steps >= holdout_start,
    )
    return tuple((x[mask], y[mask]) for mask in masks)


def train_base(
    *,
    width: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    source_cfg: dict,
    device: torch.device,
    stream_batch: int,
    updates_per_batch: int,
    seed: int,
) -> tuple[dict, dict[str, float]]:
    target_keys = (
        "terminal_limit", "collision_sdf_limit", "joint_limit", "torque_limit",
        "terminal_scale", "collision_scale", "joint_scale", "torque_scale",
    )
    target_kwargs = {key: source_cfg[key] for key in target_keys if key in source_cfg}
    cfg = MonotonicConfidenceConfig(
        num_heads=5, hidden_dim=int(width), depth=2, knots=8,
        batch_size=int(stream_batch), updates_per_observation=int(updates_per_batch),
        replay_capacity=int(train_x.shape[0]), inertia_weight=0.5,
        background_confidence_weight=0.05, randomized_prior_scale=0.1,
        anchor_capacity=0, **target_kwargs,
    )
    trainer = MonotonicConfidenceTrainer(
        int(train_x.shape[1]), cfg, device=device, seed=seed,
    )
    started = time.perf_counter()
    actor_step = 0
    for start in range(0, len(train_x), int(stream_batch)):
        stop = min(start + int(stream_batch), len(train_x))
        actor_step += 1
        trainer.observe_targets(
            train_x[start:stop], train_y[start:stop], actor_step=actor_step,
        )
    elapsed = time.perf_counter() - started
    trainable = sum(p.numel() for p in trainer.live.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.live.parameters())
    return copy.deepcopy(trainer.state_dict()), {
        "seconds": elapsed,
        "updates": float(trainer.updates),
        "trainable_parameters": float(trainable),
        "total_parameters": float(total),
    }


def load_base(saved: dict, device: torch.device) -> MonotonicConfidenceTrainer:
    cfg = MonotonicConfidenceConfig(**saved["config"])
    trainer = MonotonicConfidenceTrainer(
        int(saved["input_dim"]), cfg, device=device, seed=0,
    )
    trainer.load_state_dict(copy.deepcopy(saved))
    return trainer


def enable_policy(
    trainer: MonotonicConfidenceTrainer,
    policy: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    capacity: int,
    half_life_rows: int,
) -> None:
    if policy == "none":
        return
    kwargs = {
        "capacity": capacity,
        "half_life_rows": half_life_rows,
        "batch_size": 128,
        "exact_weight": 0.25,
        "inertia_weight": 1.0,
        "drift_tolerance": 0.05,
        "match_radius": 0.20,
    }
    if policy == "fixed":
        kwargs.update(
            replacement=False, slow_fraction=1.0,
            half_life_rows=1_000_000_000,
        )
    elif policy == "elastic":
        kwargs.update(replacement=True, slow_fraction=0.0)
    elif policy == "two_tier":
        kwargs.update(
            replacement=True, slow_fraction=0.20,
            slow_half_life_multiplier=5.0,
        )
    else:
        raise ValueError(f"unknown anchor policy {policy!r}")
    trainer.enable_anchors(train_x, train_y, **kwargs)


def run_continuation(
    *,
    saved: dict,
    width: int,
    policy: str,
    train: tuple[torch.Tensor, torch.Tensor],
    calibration: tuple[torch.Tensor, torch.Tensor],
    holdout: tuple[torch.Tensor, torch.Tensor],
    streams: list[tuple[str, torch.Tensor, torch.Tensor]],
    device: torch.device,
    stream_batch: int,
    updates_per_batch: int,
    anchor_capacity: int,
    anchor_half_life_rows: int,
    max_false_rate: float,
) -> dict[str, object]:
    trainer = load_base(saved, device)
    trainer.cfg.updates_per_observation = int(updates_per_batch)
    train_x, train_y = train
    anchor_n = min(1024, len(train_x) // 2)
    anchors = {
        "oldest_original": (train_x[:anchor_n], train_y[:anchor_n]),
        "newest_original": (train_x[-anchor_n:], train_y[-anchor_n:]),
    }
    initial_prediction = {
        name: trainer.predict(x.to(device)).mean(dim=0).cpu()
        for name, (x, _y) in anchors.items()
    }
    enable_policy(
        trainer, policy, train_x, train_y,
        capacity=anchor_capacity, half_life_rows=anchor_half_life_rows,
    )

    def snapshot(stage: str) -> dict[str, object]:
        rule = calibrate_monotonic_rule(
            trainer, calibration[0], calibration[1],
            max_false_rate=max_false_rate, use_confidence=True,
        )
        return {
            "stage": stage,
            "updates": int(trainer.updates),
            "rule": rule,
            "anchors": {
                name: compact_metrics(
                    trainer, x, y, rule=rule,
                    initial_prediction=initial_prediction[name],
                )
                for name, (x, y) in anchors.items()
            },
            "holdout": compact_metrics(
                trainer, holdout[0], holdout[1], rule=rule,
            ),
            "anchor_memory": (
                None if trainer.anchors is None else trainer.anchors.diagnostics()
            ),
        }

    stages = [snapshot("initial")]
    actor_step = int(trainer.recent.actor_step[:len(trainer.recent)].max())
    started = time.perf_counter()
    for stream_name, x, y in streams:
        for start in range(0, len(x), int(stream_batch)):
            stop = min(start + int(stream_batch), len(x))
            actor_step += 1
            trainer.observe_targets(x[start:stop], y[start:stop], actor_step=actor_step)
        stages.append(snapshot(stream_name))
    elapsed = time.perf_counter() - started
    final_rule = stages[-1]["rule"]
    shifted = {
        name: compact_metrics(trainer, x, y, rule=final_rule)
        for name, x, y in streams
    }
    return {
        "width": width,
        "policy": policy,
        "continuation_seconds": elapsed,
        "stages": stages,
        "final_shifted_sets": shifted,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    report = json.loads(Path(args.calibration_report).read_text())
    original_ref = args.cache + "::original"
    train, calibration, holdout = split_original(original_ref, report)
    _x, _y, _steps, source_cfg = load_rows_with_metadata(original_ref)
    streams = [
        (name, *load_rows(args.cache + "::" + name))
        for name in ("same_generator", "readiness_cross_hand", "legacy_task_regime")
    ]
    plan = (
        {32: ("none", "two_tier")}
        if args.quick else
        {
            128: ("two_tier",),
            256: ("none", "fixed", "elastic", "two_tier"),
            512: ("two_tier",),
        }
    )
    arms: dict[str, object] = {}
    base_training: dict[str, object] = {}
    for width, policies in plan.items():
        saved, base_stats = train_base(
            width=width, train_x=train[0], train_y=train[1],
            source_cfg=source_cfg, device=device,
            stream_batch=args.stream_batch,
            updates_per_batch=args.updates_per_batch,
            seed=args.seed,
        )
        base_training[str(width)] = base_stats
        for policy in policies:
            name = f"width{width}_{policy}"
            arms[name] = run_continuation(
                saved=saved, width=width, policy=policy,
                train=train, calibration=calibration, holdout=holdout,
                streams=streams, device=device,
                stream_batch=args.stream_batch,
                updates_per_batch=args.updates_per_batch,
                anchor_capacity=args.anchor_capacity,
                anchor_half_life_rows=args.anchor_half_life_rows,
                max_false_rate=args.max_false_rate,
            )
        del saved
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "cache": args.cache,
        "rows": {
            "train": len(train[0]), "calibration": len(calibration[0]),
            "holdout": len(holdout[0]),
            "continuation": sum(len(x) for _name, x, _y in streams),
        },
        "config": vars(args),
        "base_training": base_training,
        "arms": arms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("ELASTIC_ANCHOR_CAPACITY_RESULT=" + json.dumps({
        "output": str(output), "arms": list(arms),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
