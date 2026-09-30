#!/usr/bin/env python
"""Compare the default MLP and a parameter-matched local-expert critic."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from amortized_opt.monotonic_critic import (
    MonotonicConfidenceConfig,
    MonotonicConfidenceTrainer,
)
from scripts.ablate_elastic_anchors_and_capacity import (
    compact_metrics,
    enable_policy,
    split_original,
)
from scripts.ablate_monotonic_confidence_critic import calibrate_monotonic_rule
from scripts.evaluate_monotonic_critic_ood import (
    load_rows,
    load_rows_with_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--updates-per-batch", type=int, default=50)
    parser.add_argument("--stream-batch", type=int, default=256)
    parser.add_argument("--max-false-rate", type=float, default=0.005)
    parser.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run_arm(
    *,
    architecture: str,
    seed: int,
    train: tuple[torch.Tensor, torch.Tensor],
    calibration: tuple[torch.Tensor, torch.Tensor],
    holdout: tuple[torch.Tensor, torch.Tensor],
    streams: list[tuple[str, torch.Tensor, torch.Tensor]],
    source_cfg: dict,
    device: torch.device,
    stream_batch: int,
    updates_per_batch: int,
    max_false_rate: float,
) -> dict[str, object]:
    target_keys = (
        "terminal_limit", "collision_sdf_limit", "joint_limit", "torque_limit",
        "terminal_scale", "collision_scale", "joint_scale", "torque_scale",
    )
    target_kwargs = {key: source_cfg[key] for key in target_keys if key in source_cfg}
    train_x, train_y = train
    cfg = MonotonicConfidenceConfig(
        num_heads=3,
        hidden_dim=256,
        depth=2,
        knots=8,
        trunk_architecture=architecture,
        expert_count=8,
        expert_top_k=2,
        expert_hidden_dim=80,
        batch_size=int(stream_batch),
        updates_per_observation=int(updates_per_batch),
        replay_capacity=int(train_x.shape[0]),
        inertia_weight=0.5,
        background_confidence_weight=0.05,
        randomized_prior_scale=0.1,
        anchor_capacity=0,
        **target_kwargs,
    )
    trainer = MonotonicConfidenceTrainer(
        int(train_x.shape[1]), cfg, device=device, seed=seed,
    )
    trainable = sum(p.numel() for p in trainer.live.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.live.parameters())

    actor_step = 0
    started = time.perf_counter()
    for start in range(0, len(train_x), int(stream_batch)):
        stop = min(start + int(stream_batch), len(train_x))
        actor_step += 1
        trainer.observe_targets(
            train_x[start:stop], train_y[start:stop], actor_step=actor_step,
        )
    base_seconds = time.perf_counter() - started

    anchor_n = min(1024, len(train_x) // 2)
    anchors = {
        "oldest_original": (train_x[:anchor_n], train_y[:anchor_n]),
        "newest_original": (train_x[-anchor_n:], train_y[-anchor_n:]),
    }
    initial_prediction = {
        name: trainer.predict(x.to(device)).mean(dim=0).cpu()
        for name, (x, _target) in anchors.items()
    }
    enable_policy(
        trainer, "two_tier", train_x, train_y,
        capacity=2048, half_life_rows=55_296,
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
            "anchor_memory": trainer.anchors.diagnostics(),
            "routing": trainer.routing_statistics(holdout[0]),
        }

    stages = [snapshot("initial")]
    started = time.perf_counter()
    for stream_name, x, y in streams:
        for start in range(0, len(x), int(stream_batch)):
            stop = min(start + int(stream_batch), len(x))
            actor_step += 1
            trainer.observe_targets(x[start:stop], y[start:stop], actor_step=actor_step)
        stages.append(snapshot(stream_name))
    continuation_seconds = time.perf_counter() - started
    final_rule = stages[-1]["rule"]
    shifted = {
        name: compact_metrics(trainer, x, y, rule=final_rule)
        for name, x, y in streams
    }
    return {
        "architecture": architecture,
        "seed": seed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "base_seconds": base_seconds,
        "continuation_seconds": continuation_seconds,
        "stages": stages,
        "final_shifted_sets": shifted,
    }


def _value(run: dict, path: tuple[str, ...]) -> float:
    value: object = run
    for key in path:
        if key == "final_stage":
            value = value["stages"][-1]  # type: ignore[index]
        else:
            value = value[key]  # type: ignore[index]
    return float(value)


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    paths = {
        "oldest_drift_mae": (
            "final_stage", "anchors", "oldest_original", "prediction_drift_mae",
        ),
        "oldest_drift_p95": (
            "final_stage", "anchors", "oldest_original", "prediction_drift_p95",
        ),
        "holdout_mae": ("final_stage", "holdout", "mae_normalized"),
        "holdout_confidence_auc": (
            "final_stage", "holdout", "confidence_correct_auc",
        ),
        "holdout_false_feasible": (
            "final_stage", "holdout", "decision", "false_feasible_rate",
        ),
        "holdout_coverage": (
            "final_stage", "holdout", "decision", "coverage",
        ),
        "same_generator_false_feasible": (
            "final_shifted_sets", "same_generator", "decision", "false_feasible_rate",
        ),
        "same_generator_coverage": (
            "final_shifted_sets", "same_generator", "decision", "coverage",
        ),
        "readiness_false_feasible": (
            "final_shifted_sets", "readiness_cross_hand", "decision",
            "false_feasible_rate",
        ),
        "readiness_coverage": (
            "final_shifted_sets", "readiness_cross_hand", "decision", "coverage",
        ),
    }
    result: dict[str, object] = {
        "runs": len(runs),
        "trainable_parameters": runs[0]["trainable_parameters"],
        "total_parameters": runs[0]["total_parameters"],
    }
    for name, path in paths.items():
        values = [_value(run, path) for run in runs]
        result[name] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    for name in ("base_seconds", "continuation_seconds"):
        values = [float(run[name]) for run in runs]
        result[name] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
        }
    routing = [run["stages"][-1]["routing"] for run in runs]  # type: ignore[index]
    if routing[0] is not None:
        result["routing"] = {
            key: statistics.mean(float(row[key]) for row in routing)
            for key in routing[0]
        }
    return result


def main() -> None:
    args = parse_args()
    seeds = args.seed or ([17] if args.quick else [17, 29, 43])
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
    if args.quick:
        train = (train[0][:1024], train[1][:1024])
        calibration = (calibration[0][:512], calibration[1][:512])
        holdout = (holdout[0][:512], holdout[1][:512])
        streams = [(name, x[:256], y[:256]) for name, x, y in streams]

    runs: dict[str, list[dict[str, object]]] = {
        "mlp": [], "local_experts": [],
    }
    for seed in seeds:
        for architecture in runs:
            runs[architecture].append(run_arm(
                architecture=architecture,
                seed=seed,
                train=train,
                calibration=calibration,
                holdout=holdout,
                streams=streams,
                source_cfg=source_cfg,
                device=device,
                stream_batch=args.stream_batch,
                updates_per_batch=args.updates_per_batch,
                max_false_rate=args.max_false_rate,
            ))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "cache": args.cache,
        "config": {**vars(args), "seeds": seeds},
        "rows": {
            "train": len(train[0]),
            "calibration": len(calibration[0]),
            "holdout": len(holdout[0]),
            "continuation": sum(len(x) for _name, x, _y in streams),
        },
        "runs": runs,
        "summary": {
            architecture: summarize(architecture_runs)
            for architecture, architecture_runs in runs.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("LOCAL_EXPERT_ABLATION_RESULT=" + json.dumps({
        "output": str(output), "seeds": seeds, "summary": result["summary"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
