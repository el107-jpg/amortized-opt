#!/usr/bin/env python
"""Exact directional-proposal ablation for frozen timing critics and actor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from amortized_opt.train_cost import phase2_config_from_checkpoint, train_phase2
from amortized_opt.violation_critic import ViolationCriticConfig
from scripts.run_timing_critic_trial import (
    configure_run,
    load_model,
    load_pool,
    scenario_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--pool-cache", required=True)
    parser.add_argument("--critic-ablation", required=True)
    parser.add_argument("--critic-state-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--threshold-mode", choices=("calibrated", "zero"),
        default="calibrated",
    )
    return parser.parse_args()


def _frozen_state(state: dict) -> dict:
    frozen = copy.deepcopy(state)
    frozen["live"] = copy.deepcopy(frozen["ema"])
    for group in frozen["optimizer"]["param_groups"]:
        group["lr"] = 0.0
    # Every arm gets the identical proposal stream.  MC masks have their own
    # generator and therefore cannot move this stream forward.
    proposal_generator = torch.Generator(device="cpu")
    proposal_generator.manual_seed(0x51A7E)
    frozen["generator_state"] = proposal_generator.get_state()
    return frozen


def _summarize_metrics(path: Path, min_step: int) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "train" or int(row.get("step", -1)) <= min_step:
            continue
        count += 1
        roller = row.get("roller", {})
        for key, value in roller.items():
            if key.startswith("timing_"):
                totals[key] = totals.get(key, 0.0) + float(value)
    proposed = totals.get("timing_assist_proposed", 0.0)
    accepted = totals.get("timing_assist_accepted", 0.0)
    pred_ok = totals.get("timing_selected_pred_manageable", 0.0)
    false_ok = totals.get("timing_selected_false_manageable", 0.0)
    totals.update({
        "logged_steps": float(count),
        "verified_proposal_yield": accepted / max(proposed, 1.0),
        "predicted_manageable_precision": (
            (pred_ok - false_ok) / max(pred_ok, 1.0)
        ),
        "false_fraction_among_predicted_manageable": (
            false_ok / max(pred_ok, 1.0)
        ),
    })
    return totals


def main() -> None:
    args = parse_args()
    actor_checkpoint = torch.load(
        args.actor_checkpoint, map_location="cpu", weights_only=False,
    )
    ablation = json.loads(Path(args.critic_ablation).read_text())
    scenario_cfg = scenario_from_checkpoint(actor_checkpoint)
    start_step = int(actor_checkpoint["step"])
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = {}

    for arm, offline in ablation["arms"].items():
        state = torch.load(
            Path(args.critic_state_dir) / f"{arm}.pt",
            map_location="cpu", weights_only=False,
        )
        valid = ViolationCriticConfig.__dataclass_fields__
        tc = ViolationCriticConfig(**{
            key: value for key, value in state["config"].items() if key in valid
        })
        tc.mode = "control"
        tc.updates_per_step = 1
        tc.min_assist_samples = 1
        tc.conservative_quantile = float(
            offline["calibration"]["selected"]["quantile"]
        )
        tc.decision_threshold = (
            0.0 if args.threshold_mode == "zero" else float(
                offline["calibration"]["selected"]["threshold"]
            )
        )

        cfg = phase2_config_from_checkpoint(actor_checkpoint, device=args.device)
        cfg.timing_critic = tc
        cfg.lr = 0.0
        cfg.grad_term_every = 0
        configure_run(
            cfg, max_steps=start_step + int(args.steps),
            batch=int(args.batch), log_every=1,
        )
        model, model_cfg = load_model(actor_checkpoint, args.device)
        pool = load_pool(Path(args.pool_cache), scenario_cfg, args.device)
        arm_dir = root / arm
        train_phase2(
            model=model,
            pool=pool,
            cfg=cfg,
            model_cfg=model_cfg,
            run_dir=arm_dir,
            scenario_cfg=scenario_cfg,
            al_state=actor_checkpoint["al"],
            opt_state=actor_checkpoint["optimizer"],
            timing_critic_state=_frozen_state(state),
            start_step=start_step,
        )
        result = _summarize_metrics(arm_dir / "metrics.jsonl", start_step)
        result.update({
            "quantile": tc.conservative_quantile,
            "decision_threshold": tc.decision_threshold,
            "randomized_prior_scale": tc.randomized_prior_scale,
            "bootstrap_probability": tc.bootstrap_probability,
            "num_heads": tc.num_heads,
            "dropout_probability": tc.dropout_probability,
            "mc_dropout_samples": tc.mc_dropout_samples,
            "head_architecture_mix": tc.head_architecture_mix,
            "replay_timescale_diversity": bool(
                offline.get("replay_timescale_diversity", False)
            ),
        })
        results[arm] = result
        print(f"[{arm}] " + json.dumps(result, sort_keys=True), flush=True)
        del model, pool
        torch.cuda.empty_cache()

    report = {
        "actor_checkpoint": args.actor_checkpoint,
        "critic_ablation": args.critic_ablation,
        "steps": int(args.steps),
        "batch": int(args.batch),
        "threshold_mode": args.threshold_mode,
        "arms": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("TIMING_FRONTIER_ABLATION=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
