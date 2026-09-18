#!/usr/bin/env python
"""Calibrate a conservative timing-critic decision rule on future labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from amortized_opt.violation_critic import (
    ViolationCriticConfig,
    ViolationCriticTrainer,
    calibrate_manageability_rule,
    conservative_score_from_prediction,
    manageability_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--min-actor-step", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--false-manageable-rate", type=float, default=0.02)
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=(0.8,),
        help="head-score quantiles to compare; fixed 0.8 avoids tuning noise by default",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_rows(state: dict, min_actor_step: int):
    recent = state["extra"]["timing_critic"]["recent"]
    size = int(recent["size"])
    steps = recent["actor_step"][:size].long()
    keep = steps >= int(min_actor_step)
    return (
        recent["features"][:size][keep].float(),
        recent["targets"][:size][keep].float(),
        steps[keep],
    )


def _config(state: dict) -> ViolationCriticConfig:
    raw = state["extra"]["timing_critic"].get("config", {})
    valid = ViolationCriticConfig.__dataclass_fields__
    return ViolationCriticConfig(**{key: value for key, value in raw.items() if key in valid})


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.calibration_fraction) < 1.0:
        raise ValueError("calibration-fraction must be in (0, 1)")
    predictor_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False,
    )
    observation_checkpoint = torch.load(
        args.observations, map_location="cpu", weights_only=False,
    )
    predictor_state = predictor_checkpoint["extra"]["timing_critic"]
    cfg = _config(predictor_checkpoint)
    features, targets, actor_steps = _load_rows(
        observation_checkpoint, args.min_actor_step,
    )
    unique_steps = actor_steps.unique(sorted=True)
    split = max(1, min(
        int(unique_steps.numel()) - 1,
        round(float(unique_steps.numel()) * float(args.calibration_fraction)),
    ))
    calibration_steps = unique_steps[:split]
    calibration_mask = actor_steps <= calibration_steps[-1]
    validation_mask = ~calibration_mask

    trainer = ViolationCriticTrainer(
        int(predictor_state["input_dim"]), cfg,
        device=args.device, seed=1234,
    )
    trainer.load_state_dict(predictor_state)
    prediction = trainer.predict(features.to(args.device), ema=True, physical=False).cpu()
    fit = calibrate_manageability_rule(
        prediction[:, calibration_mask], targets[calibration_mask],
        quantiles=tuple(float(value) for value in args.quantiles),
        max_false_manageable_rate=float(args.false_manageable_rate),
    )
    selected = fit["selected"]
    validation_score = conservative_score_from_prediction(
        prediction[:, validation_mask], quantile=selected["quantile"],
    )
    validation = manageability_metrics(
        validation_score, targets[validation_mask],
        threshold=selected["threshold"],
    )
    validation_candidates = []
    for candidate in fit["candidates"]:
        candidate_score = conservative_score_from_prediction(
            prediction[:, validation_mask], quantile=candidate["quantile"],
        )
        validation_candidates.append({
            "quantile": candidate["quantile"],
            **manageability_metrics(
                candidate_score, targets[validation_mask],
                threshold=candidate["threshold"],
            ),
        })
    uncalibrated_score = conservative_score_from_prediction(
        prediction[:, validation_mask], quantile=cfg.conservative_quantile,
    )
    uncalibrated = manageability_metrics(
        uncalibrated_score, targets[validation_mask], threshold=0.0,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "observations": str(args.observations),
        "min_actor_step": int(args.min_actor_step),
        "actor_step_split": {
            "calibration": [int(calibration_steps[0]), int(calibration_steps[-1])],
            "validation": [int(unique_steps[split]), int(unique_steps[-1])],
        },
        "calibration": fit,
        "validation": validation,
        "validation_candidates": validation_candidates,
        "uncalibrated_validation": uncalibrated,
        "recommended_config": {
            "conservative_quantile": selected["quantile"],
            "decision_threshold": selected["threshold"],
        },
    }
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).with_name("critic_calibration.json")
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("CRITIC_CALIBRATION_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
