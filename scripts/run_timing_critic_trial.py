#!/usr/bin/env python
"""Train the timing critic online, then evaluate on future actor-only labels."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import amortized_opt as A
from amortized_opt.model import model_config_from_checkpoint
from amortized_opt.phase2_data import Phase2Pool, Phase2ScenarioConfig
from amortized_opt.rhythm import KitRandomization
from amortized_opt.train_cost import phase2_config_from_checkpoint, train_phase2
from amortized_opt.violation_critic import (
    VIOLATION_NAMES,
    ViolationCriticTrainer,
    conservative_score_from_prediction,
    robust_inverse,
)
from amortized_opt.waist_aim import WaistAimConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="/tmp/timing-actor-only-smoke-8wdaniyb/last.pt",
    )
    parser.add_argument(
        "--pool-cache",
        default=(
            "runs/p2pool_diagnostic_safe_downweight_300_"
            "mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz"
        ),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--train-updates", type=int, default=1_000)
    parser.add_argument("--holdout-steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--critic-updates", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def scenario_from_checkpoint(checkpoint: dict) -> Phase2ScenarioConfig:
    raw = dict(checkpoint["scenario_cfg"])
    if isinstance(raw.get("kit_randomization"), dict):
        raw["kit_randomization"] = KitRandomization(**raw["kit_randomization"])
    if isinstance(raw.get("waist_cfg"), dict):
        raw["waist_cfg"] = WaistAimConfig(**raw["waist_cfg"])
    valid = Phase2ScenarioConfig.__dataclass_fields__
    return Phase2ScenarioConfig(**{k: v for k, v in raw.items() if k in valid})


def load_pool(
    path: Path, scenario_cfg: Phase2ScenarioConfig, device: str
) -> Phase2Pool:
    archive = np.load(path, allow_pickle=True)
    buckets: dict[int, dict[str, np.ndarray]] = {}
    for key in archive.files:
        if key.startswith("_"):
            continue
        bucket, field = key.split("/", 1)
        buckets.setdefault(int(bucket), {})[field] = archive[key]
    stats = json.loads(str(archive["_stats"]))
    return Phase2Pool(buckets, cfg=scenario_cfg, device=device, stats=stats)


def load_model(checkpoint: dict, device: str):
    model_cfg = model_config_from_checkpoint(checkpoint)
    model = A.build_model(model_cfg, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, model_cfg


def configure_run(cfg, *, max_steps: int, batch: int, log_every: int) -> None:
    cfg.max_steps = int(max_steps)
    cfg.batch_size = int(batch)
    cfg.eval_every = 0
    cfg.log_every = int(log_every)
    cfg.tensorboard = False
    cfg.ckpt_every = 999_999
    cfg.ckpt_last_every = 999_999
    cfg.ckpt_keep_every = 999_999


@torch.no_grad()
def score_holdout(
    predictor_state: dict,
    observation_state: dict,
    cfg,
    *,
    min_actor_step: int,
    device: str,
) -> dict[str, float]:
    recent = observation_state["recent"]
    size = int(recent["size"])
    keep = recent["actor_step"][:size].long() >= int(min_actor_step)
    features = recent["features"][:size][keep].to(device, torch.float32)
    targets = recent["targets"][:size][keep].to(device, torch.float32)
    trainer = ViolationCriticTrainer(
        int(predictor_state["input_dim"]), cfg, device=device, seed=1234
    )
    trainer.load_state_dict(predictor_state)
    prediction = trainer.predict(features, ema=True, physical=False)
    mean = prediction.mean(dim=0)
    pred_physical = robust_inverse(mean, cfg.scales)
    true_physical = robust_inverse(targets, cfg.scales)
    conservative = conservative_score_from_prediction(
        prediction, quantile=cfg.conservative_quantile,
    )
    true_score = targets.amax(dim=-1)
    truth = true_score <= 0.0
    guess = conservative <= float(cfg.decision_threshold)
    tp = (truth & guess).sum().float()
    fp = ((~truth) & guess).sum().float()
    result = {
        "n": float(features.shape[0]),
        "true_manageable_rate": float(truth.float().mean()),
        "pred_manageable_rate": float(guess.float().mean()),
        "manageable_recall": float(tp / truth.sum().clamp_min(1)),
        "manageable_precision": float(tp / (tp + fp).clamp_min(1)),
        "false_manageable_rate": float(((~truth) & guess).float().mean()),
        "disagreement": float(prediction.std(dim=0, unbiased=False).mean()),
        "decision_threshold": float(cfg.decision_threshold),
        "conservative_quantile": float(cfg.conservative_quantile),
    }
    for index, name in enumerate(VIOLATION_NAMES):
        pred = pred_physical[:, index]
        true = true_physical[:, index]
        denom = pred.std(unbiased=False) * true.std(unbiased=False)
        corr = ((pred - pred.mean()) * (true - true.mean())).mean()
        result[f"mae_{name}"] = float((pred - true).abs().mean())
        result[f"corr_{name}"] = float(corr / denom.clamp_min(1e-12))
        result[f"true_pass_{name}"] = float((true <= 0.0).float().mean())
        result[f"pred_pass_{name}"] = float((pred <= 0.0).float().mean())
    return result


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(torch.device(args.device).index or 0)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    scenario_cfg = scenario_from_checkpoint(source)
    pool = load_pool(Path(args.pool_cache), scenario_cfg, args.device)
    model, model_cfg = load_model(source, args.device)
    start_step = int(source["step"])
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(tempfile.mkdtemp(prefix="timing-actor-only-long-", dir="/tmp"))
    )

    cfg = phase2_config_from_checkpoint(source, device=args.device)
    if args.critic_updates is not None:
        cfg.timing_critic.updates_per_step = int(args.critic_updates)
    configure_run(
        cfg,
        max_steps=start_step + int(args.train_updates),
        batch=args.batch,
        log_every=args.log_every,
    )
    train_phase2(
        model=model,
        pool=pool,
        cfg=cfg,
        model_cfg=model_cfg,
        run_dir=run_dir,
        scenario_cfg=scenario_cfg,
        al_state=source["al"],
        opt_state=source["optimizer"],
        timing_critic_state=source["extra"]["timing_critic"],
        start_step=start_step,
    )

    trained = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    predictor_state = copy.deepcopy(trained["extra"]["timing_critic"])
    del model
    torch.cuda.empty_cache()

    # Collect later actor labels without moving the critic. Starting live at
    # EMA and restoring a zero-LR optimizer makes both live and EMA stationary.
    frozen_state = copy.deepcopy(predictor_state)
    frozen_state["live"] = copy.deepcopy(frozen_state["ema"])
    for group in frozen_state["optimizer"]["param_groups"]:
        group["lr"] = 0.0
    holdout_start = int(trained["step"])
    holdout_dir = run_dir / "holdout"
    model, model_cfg = load_model(trained, args.device)
    holdout_cfg = phase2_config_from_checkpoint(trained, device=args.device)
    # The holdout optimizer has zero LR; one call is enough to collect each
    # actor batch without multiplying pointless frozen critic work.
    holdout_cfg.timing_critic.updates_per_step = 1
    configure_run(
        holdout_cfg,
        max_steps=holdout_start + int(args.holdout_steps),
        batch=args.batch,
        log_every=args.log_every,
    )
    train_phase2(
        model=model,
        pool=pool,
        cfg=holdout_cfg,
        model_cfg=model_cfg,
        run_dir=holdout_dir,
        scenario_cfg=scenario_cfg,
        al_state=trained["al"],
        opt_state=trained["optimizer"],
        timing_critic_state=frozen_state,
        start_step=holdout_start,
    )
    held = torch.load(
        holdout_dir / "last.pt", map_location="cpu", weights_only=False
    )
    critic_cfg = phase2_config_from_checkpoint(
        trained, device=args.device
    ).timing_critic
    result = {
        "run_dir": str(run_dir),
        "source_step": start_step,
        "trained_through_step": holdout_start,
        "final_step": int(held["step"]),
        "weight_changing_critic_updates": int(predictor_state["updates"]),
        "holdout": score_holdout(
            predictor_state,
            held["extra"]["timing_critic"],
            critic_cfg,
            min_actor_step=holdout_start,
            device=args.device,
        ),
    }
    (run_dir / "critic_holdout.json").write_text(json.dumps(result, indent=2))
    print("CRITIC_TRIAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
