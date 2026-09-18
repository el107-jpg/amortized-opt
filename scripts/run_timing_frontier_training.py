#!/usr/bin/env python
"""Resume actor training with exact-verified critic timing collocations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from amortized_opt.train_cost import phase2_config_from_checkpoint, train_phase2
from amortized_opt.violation_critic import (
    CRITIC_FORMAT_VERSION,
    ViolationCriticConfig,
)
from scripts.run_timing_critic_trial import (
    configure_run,
    load_model,
    load_pool,
    scenario_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--critic-state", required=True)
    parser.add_argument("--pool-cache", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--critic-updates", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--decision-threshold", type=float, default=0.0)
    parser.add_argument(
        "--mode", choices=("control", "shadow"), default="control",
        help="Use exact-verified timing replacements or a matched no-replacement baseline.",
    )
    return parser.parse_args()


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / max(len(values), 1)


def _summarize(path: Path, start_step: int) -> dict:
    rows = [
        row for row in map(json.loads, path.read_text().splitlines())
        if row.get("event") == "train" and int(row.get("step", 0)) > start_step
    ]
    split = max(1, len(rows) // 2)

    def block(part: list[dict]) -> dict:
        roller_totals: dict[str, float] = {}
        for row in part:
            for key, value in row.get("roller", {}).items():
                if key.startswith("timing_"):
                    roller_totals[key] = roller_totals.get(key, 0.0) + float(value)
        proposed = roller_totals.get("timing_assist_proposed", 0.0)
        accepted = roller_totals.get("timing_assist_accepted", 0.0)
        predicted = roller_totals.get("timing_selected_pred_manageable", 0.0)
        false = roller_totals.get("timing_selected_false_manageable", 0.0)
        return {
            "steps": len(part),
            "loss": _mean(part, "loss"),
            "v_terminal": _mean(part, "v_terminal"),
            "v_collision": _mean(part, "v_collision"),
            "v_torque_peak": _mean(part, "v_torque_peak"),
            "v_joint_box_peak": _mean(part, "v_joint_box_peak"),
            "proposal_yield": accepted / max(proposed, 1.0),
            "predicted_manageable_precision": (
                predicted - false
            ) / max(predicted, 1.0),
            **roller_totals,
        }

    return {
        "logged_steps": len(rows),
        "first_half": block(rows[:split]),
        "second_half": block(rows[split:]),
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(torch.device(args.device).index or 0)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    critic_state = torch.load(
        args.critic_state, map_location="cpu", weights_only=False,
    )
    if int(critic_state.get("format_version", -1)) != CRITIC_FORMAT_VERSION:
        raise SystemExit(
            "critic checkpoint uses the retired initializer/provenance feature "
            "schema; collect a fresh shadow critic before control training"
        )
    valid = ViolationCriticConfig.__dataclass_fields__
    critic_cfg = ViolationCriticConfig(**{
        key: value for key, value in critic_state["config"].items() if key in valid
    })
    critic_cfg.mode = args.mode
    critic_cfg.replay_timescale_diversity = True
    critic_cfg.min_assist_samples = 1
    critic_cfg.updates_per_step = int(args.critic_updates)
    critic_cfg.decision_threshold = float(args.decision_threshold)
    # Mark the imported ablation state with the production training semantics;
    # older state files predate this config key but have compatible weights.
    critic_state = copy.deepcopy(critic_state)
    critic_state["config"]["replay_timescale_diversity"] = True

    cfg = phase2_config_from_checkpoint(source, device=args.device)
    cfg.timing_critic = critic_cfg
    start_step = int(source["step"])
    configure_run(
        cfg, max_steps=start_step + int(args.steps),
        batch=int(args.batch), log_every=int(args.log_every),
    )
    scenario_cfg = scenario_from_checkpoint(source)
    pool = load_pool(Path(args.pool_cache), scenario_cfg, args.device)
    model, model_cfg = load_model(source, args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[timing-frontier-train] {args.mode} mode; "
        f"exact verification {'ON' if args.mode == 'control' else 'OFF'}; "
        f"heads={critic_cfg.num_heads}; dropout={critic_cfg.dropout_probability:g}; "
        f"MC={critic_cfg.mc_dropout_samples}; replay windows=25%/50%/100%; "
        f"threshold={critic_cfg.decision_threshold:g}",
        flush=True,
    )
    summary = train_phase2(
        model=model, pool=pool, cfg=cfg, model_cfg=model_cfg,
        run_dir=run_dir, scenario_cfg=scenario_cfg,
        al_state=source["al"], opt_state=source["optimizer"],
        timing_critic_state=critic_state, start_step=start_step,
    )
    report = {
        "source": args.source,
        "critic_state": args.critic_state,
        "start_step": start_step,
        "final_step": int(summary["steps"]),
        "configuration": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "critic_updates": int(args.critic_updates),
            "mode": args.mode,
            "decision_threshold": float(args.decision_threshold),
            "replay_windows": [0.25, 0.50, 1.0],
        },
        "training": _summarize(run_dir / "metrics.jsonl", start_step),
    }
    (run_dir / "timing_frontier_training.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("TIMING_FRONTIER_TRAINING=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
