#!/usr/bin/env python
"""Stress-test critic retention after replay eviction under task shifts."""

from __future__ import annotations

import argparse
import copy
import json
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
    parse_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--original-dataset", required=True)
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=CHECKPOINT",
    )
    parser.add_argument("--anchor-rows", type=int, default=1024)
    parser.add_argument("--stream-batch", type=int, default=256)
    parser.add_argument("--updates-per-batch", type=int, default=50)
    parser.add_argument(
        "--inertia-weight", action="append", type=float, default=None,
        help="inertia weight to evaluate; repeat for a sweep",
    )
    parser.add_argument(
        "--max-false-rate", type=float, default=None,
        help="override the protected recalibration false-feasible budget",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def anchor_metrics(
    trainer: MonotonicConfidenceTrainer,
    features: torch.Tensor,
    targets: torch.Tensor,
    initial_prediction: torch.Tensor,
    *,
    residual_rule: dict,
    joint_rule: dict,
) -> dict[str, object]:
    result, _ = evaluate(
        trainer, features, targets,
        residual_rule=residual_rule, joint_rule=joint_rule,
    )
    current = trainer.predict(features.to(trainer.device)).mean(dim=0).cpu()
    delta = current - initial_prediction
    result["prediction_drift_mae"] = float(delta.abs().mean())
    result["prediction_drift_p95"] = float(torch.quantile(delta.abs(), 0.95))
    return result


def make_trainer(
    saved: dict, *, device: str, inertia_weight: float,
) -> MonotonicConfidenceTrainer:
    cfg = MonotonicConfidenceConfig(**saved["config"])
    trainer = MonotonicConfidenceTrainer(
        int(saved["input_dim"]), cfg, device=device, seed=0,
    )
    trainer.load_state_dict(copy.deepcopy(saved))
    trainer.cfg.inertia_weight = float(inertia_weight)
    return trainer


def run_arm(
    *,
    saved: dict,
    device: str,
    inertia_weight: float,
    anchors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    calibration: tuple[torch.Tensor, torch.Tensor],
    streams: list[tuple[str, torch.Tensor, torch.Tensor]],
    residual_rule: dict,
    joint_rule: dict,
    stream_batch: int,
    updates_per_batch: int,
    max_false_rate: float,
) -> dict[str, object]:
    trainer = make_trainer(
        saved, device=device, inertia_weight=inertia_weight,
    )
    trainer.cfg.updates_per_observation = int(updates_per_batch)
    initial_prediction = {
        name: trainer.predict(x.to(trainer.device)).mean(dim=0).cpu()
        for name, (x, _y) in anchors.items()
    }

    def snapshot() -> tuple[dict[str, object], dict, dict[str, dict]]:
        fixed = {
            name: anchor_metrics(
                trainer, x, y, initial_prediction[name],
                residual_rule=residual_rule, joint_rule=joint_rule,
            )
            for name, (x, y) in anchors.items()
        }
        adaptive_rule = calibrate_monotonic_rule(
            trainer, calibration[0], calibration[1],
            max_false_rate=max_false_rate, use_confidence=True,
        )
        adaptive = {}
        for name, (x, y) in anchors.items():
            metrics, _ = evaluate(
                trainer, x, y,
                residual_rule=residual_rule, joint_rule=adaptive_rule,
            )
            adaptive[name] = metrics["decisions"][
                "joint_with_support_tempered_confidence"
            ]
        return fixed, adaptive_rule, adaptive

    initial_fixed, initial_rule, initial_adaptive = snapshot()
    stages: list[dict[str, object]] = [{
        "stage": "initial", "updates": int(trainer.updates),
        "replay_size": len(trainer.recent), "anchors": initial_fixed,
        "adaptive_calibration_rule": initial_rule,
        "adaptive_anchor_decisions": initial_adaptive,
    }]
    actor_step = int(trainer.recent.actor_step[:len(trainer.recent)].max())
    for stream_name, x, y in streams:
        for start in range(0, len(x), int(stream_batch)):
            stop = min(start + int(stream_batch), len(x))
            actor_step += 1
            trainer.observe_targets(x[start:stop], y[start:stop], actor_step=actor_step)
        fixed, adaptive_rule, adaptive = snapshot()
        stages.append({
            "stage": stream_name,
            "rows_consumed": int(len(x)),
            "updates": int(trainer.updates),
            "replay_size": len(trainer.recent),
            "anchors": fixed,
            "adaptive_calibration_rule": adaptive_rule,
            "adaptive_anchor_decisions": adaptive,
        })
    return {
        "inertia_weight": float(inertia_weight),
        "updates_per_batch": int(updates_per_batch),
        "stages": stages,
    }


def main() -> None:
    args = parse_args()
    saved = torch.load(
        args.critic_checkpoint, map_location="cpu", weights_only=False,
    )
    report = json.loads(Path(args.calibration_report).read_text())
    x, y, steps, _cfg = load_rows_with_metadata(args.original_dataset)
    unique_steps = steps.unique(sorted=True)
    holdout_steps = int(report["holdout_actor_steps"])
    calibration_steps = int(report["calibration_actor_steps"])
    calibration_start = unique_steps[-holdout_steps - calibration_steps]
    holdout_start = unique_steps[-holdout_steps]
    train = steps < calibration_start
    calibration_mask = (steps >= calibration_start) & (steps < holdout_start)
    train_x, train_y = x[train], y[train]
    count = min(int(args.anchor_rows), len(train_x) // 2)
    anchors = {
        "oldest_original": (train_x[:count], train_y[:count]),
        "newest_original": (train_x[-count:], train_y[-count:]),
    }
    streams = []
    for value in args.dataset:
        name, path = parse_dataset(value)
        sx, sy = load_rows(path)
        streams.append((name, sx, sy))

    inertia = float(saved["config"]["inertia_weight"])
    weights = args.inertia_weight or [inertia, 0.0]
    unique_weights = []
    for weight in weights:
        if weight < 0.0:
            raise ValueError("inertia weights must be nonnegative")
        if weight not in unique_weights:
            unique_weights.append(weight)

    def arm_name(weight: float) -> str:
        return "inertia_" + f"{weight:g}".replace(".", "p")

    max_false_rate = (
        float(args.max_false_rate)
        if args.max_false_rate is not None
        else float(report["config"]["max_false_feasible_rate"])
    )
    if not 0.0 <= max_false_rate < 1.0:
        raise ValueError("max false-feasible rate must be in [0, 1)")

    result = {
        "critic_checkpoint": args.critic_checkpoint,
        "original_dataset": args.original_dataset,
        "original_replay_capacity": int(saved["config"]["replay_capacity"]),
        "original_replay_size": int(saved["recent"]["size"]),
        "anchor_rows": count,
        "stream_rows": sum(len(x) for _name, x, _y in streams),
        "stream_order": [name for name, _x, _y in streams],
        "adaptive_calibration_max_false_rate": max_false_rate,
        "arms": {
            arm_name(weight): run_arm(
                saved=saved, device=args.device, inertia_weight=weight,
                anchors=anchors, streams=streams,
                calibration=(x[calibration_mask], y[calibration_mask]),
                residual_rule=report["residual_calibration_rule"],
                joint_rule=report["confidence_calibration_rule"],
                stream_batch=args.stream_batch,
                updates_per_batch=args.updates_per_batch,
                max_false_rate=max_false_rate,
            )
            for weight in unique_weights
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MONOTONIC_FORGETTING_RESULT=" + json.dumps({
        "output": str(destination),
        "stream_rows": result["stream_rows"],
        "arms": list(result["arms"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
