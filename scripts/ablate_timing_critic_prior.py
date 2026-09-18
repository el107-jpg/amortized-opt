#!/usr/bin/env python
"""Offline timing-critic uncertainty ablations on one fixed replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from amortized_opt.violation_critic import (
    ViolationCriticConfig,
    ViolationCriticTrainer,
    calibrate_manageability_rule,
    conservative_score_from_prediction,
    critic_loss_per_head_row,
    manageability_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--holdout-checkpoint", required=True)
    parser.add_argument("--min-holdout-step", type=int, required=True)
    parser.add_argument("--updates", type=int, default=4_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--false-manageable-rate", type=float, default=0.02)
    parser.add_argument(
        "--suite", choices=("prior", "dropout", "diversity"), default="prior",
        help="uncertainty variants to compare",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--state-dir", default=None,
        help="optional directory for the freshly trained critic arm states",
    )
    return parser.parse_args()


def _cfg(state: dict, **overrides) -> ViolationCriticConfig:
    raw = dict(state["extra"]["timing_critic"].get("config", {}))
    valid = ViolationCriticConfig.__dataclass_fields__
    raw = {key: value for key, value in raw.items() if key in valid}
    raw.update(overrides)
    return ViolationCriticConfig(**raw)


def _holdout_rows(state: dict, min_step: int):
    recent = state["extra"]["timing_critic"]["recent"]
    size = int(recent["size"])
    steps = recent["actor_step"][:size].long()
    keep = steps >= int(min_step)
    return (
        recent["features"][:size][keep].float(),
        recent["targets"][:size][keep].float(),
        steps[keep],
    )


def _auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    pos = score[positive]
    neg = score[~positive]
    if not int(pos.numel()) or not int(neg.numel()):
        return float("nan")
    pair = pos[:, None] - neg[None, :]
    return float(((pair > 0).float() + 0.5 * (pair == 0).float()).mean())


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = a.std(unbiased=False) * b.std(unbiased=False)
    return float(
        (((a - a.mean()) * (b - b.mean())).mean() / denom.clamp_min(1e-12))
    )


def _window_pools(dataset: dict, cfg: ViolationCriticConfig):
    """Precompute target-stratified newest-25%, newest-50%, and all pools."""
    size = int(dataset["size"])
    steps = dataset["actor_step"][:size].long()
    target_score = dataset["targets"][:size].float().amax(dim=-1)
    target_class = torch.where(
        target_score <= 0.0,
        torch.zeros_like(target_score, dtype=torch.long),
        torch.where(
            target_score <= float(cfg.replay_boundary_band),
            torch.ones_like(target_score, dtype=torch.long),
            torch.full_like(target_score, 2, dtype=torch.long),
        ),
    )
    unique = steps.unique(sorted=True)
    windows = []
    for fraction in (0.25, 0.50, 1.0):
        count = max(1, int(round(float(unique.numel()) * fraction)))
        eligible = steps >= unique[-count]
        windows.append(tuple(
            (eligible & (target_class == cls)).nonzero(as_tuple=True)[0]
            for cls in range(3)
        ))
    return tuple(windows)


def _sample_window(
    dataset: dict,
    pools,
    cfg: ViolationCriticConfig,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fractions = torch.tensor([
        cfg.replay_manageable_fraction,
        cfg.replay_near_failure_fraction,
        1.0 - cfg.replay_manageable_fraction - cfg.replay_near_failure_fraction,
    ])
    available = torch.tensor([int(pool.numel()) > 0 for pool in pools])
    probabilities = fractions * available.float()
    if float(probabilities.sum()) <= 0.0:
        probabilities = available.float()
    probabilities /= probabilities.sum()
    count = min(int(cfg.batch_size), sum(int(pool.numel()) for pool in pools))
    classes = torch.multinomial(
        probabilities, count, replacement=True, generator=generator,
    )
    pieces = []
    for cls, pool in enumerate(pools):
        n = int((classes == cls).sum())
        if n:
            draw = torch.randint(0, int(pool.numel()), (n,), generator=generator)
            pieces.append(pool.index_select(0, draw))
    index = torch.cat(pieces)
    index = index.index_select(
        0, torch.randperm(int(index.numel()), generator=generator),
    )
    return (
        dataset["features"].index_select(0, index).to(device, torch.float32),
        dataset["targets"].index_select(0, index).to(device, torch.float32),
    )


def _update_timescale_diverse(
    trainer: ViolationCriticTrainer,
    dataset: dict,
    windows,
) -> None:
    """Train each head on a different replay age while sharing one optimizer."""
    batches = [
        _sample_window(
            dataset, pools, trainer.cfg,
            generator=trainer.generator, device=trainer.device,
        )
        for pools in windows
    ]
    seed = int(torch.randint(
        0, 2**31 - 1, (1,), generator=trainer.mc_generator,
    ))
    devices = [trainer.device.index or 0] if trainer.device.type == "cuda" else []
    numerators = []
    denominators = []
    with torch.random.fork_rng(devices=devices):
        if trainer.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        for head, (features, target) in zip(trainer.live.heads, batches):
            per_row, _, _ = critic_loss_per_head_row(
                head(features).unsqueeze(0), target, trainer.cfg,
            )
            mask = torch.rand(
                per_row.shape, generator=trainer.generator, device="cpu",
            ).to(trainer.device) < float(trainer.cfg.bootstrap_probability)
            if not bool(mask.any()):
                mask[0, 0] = True
            numerators.append((per_row * mask).sum())
            denominators.append(mask.sum())
    loss = torch.stack(numerators).sum() / torch.stack(denominators).sum().clamp_min(1)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.optimizer.step()
    trainer._update_ema()
    trainer.updates += 1


@torch.no_grad()
def _score(
    trainer: ViolationCriticTrainer,
    features: torch.Tensor,
    targets: torch.Tensor,
    steps: torch.Tensor,
    *,
    max_false_manageable_rate: float,
) -> dict:
    prediction = trainer.predict(
        features.to(trainer.device), ema=True, physical=False,
    ).cpu()
    head_score = prediction.amax(dim=-1)
    disagreement = head_score.std(dim=0, unbiased=False)
    mean_score = prediction.mean(dim=0).amax(dim=-1)
    true_score = targets.amax(dim=-1)
    truth = true_score <= 0.0
    q_score = conservative_score_from_prediction(prediction, quantile=0.8)
    guess = q_score <= 0.0
    wrong = guess != truth
    abs_score_error = (mean_score - true_score).abs()
    by_model = prediction.reshape(
        int(trainer.cfg.mc_dropout_samples), int(trainer.cfg.num_heads),
        prediction.shape[1], prediction.shape[2],
    ).mean(dim=0)
    member_abs_score_error = (
        by_model.amax(dim=-1) - true_score.unsqueeze(0)
    ).abs().mean(dim=1)

    unique_steps = steps.unique(sorted=True)
    split = int(unique_steps.numel()) // 2
    calibration_mask = steps <= unique_steps[split - 1]
    validation_mask = ~calibration_mask
    calibration = calibrate_manageability_rule(
        prediction[:, calibration_mask], targets[calibration_mask],
        quantiles=(0.8,),
        max_false_manageable_rate=max_false_manageable_rate,
    )
    rule = calibration["selected"]
    validation_score = conservative_score_from_prediction(
        prediction[:, validation_mask], quantile=rule["quantile"],
    )
    return {
        "uncalibrated": manageability_metrics(q_score, targets, threshold=0.0),
        "calibration": calibration,
        "validation": manageability_metrics(
            validation_score, targets[validation_mask],
            threshold=rule["threshold"],
        ),
        "mean_head_score_disagreement": float(disagreement.mean()),
        "disagreement_correct": float(disagreement[~wrong].mean()),
        "disagreement_wrong": float(disagreement[wrong].mean()),
        "uncertainty_wrong_auc": _auc(disagreement, wrong),
        "uncertainty_abs_error_correlation": _correlation(
            disagreement, abs_score_error,
        ),
        "mean_abs_score_error": float(abs_score_error.mean()),
        "member_mean_abs_score_error": [
            float(value) for value in member_abs_score_error
        ],
    }


def main() -> None:
    args = parse_args()
    training = torch.load(
        args.training_checkpoint, map_location="cpu", weights_only=False,
    )
    holdout = torch.load(
        args.holdout_checkpoint, map_location="cpu", weights_only=False,
    )
    dataset = training["extra"]["timing_critic"]["recent"]
    features, targets, steps = _holdout_rows(holdout, args.min_holdout_step)
    if args.suite == "diversity":
        common = {
            "num_heads": 3, "dropout_probability": 0.10,
            "mc_dropout_samples": 5, "randomized_prior_scale": 0.0,
        }
        arms = (
            ("homogeneous", {**common,
                "head_architecture_mix": "homogeneous"}, False),
            ("heterogeneous", {**common,
                "head_architecture_mix": "heterogeneous"}, False),
            ("timescale", {**common,
                "head_architecture_mix": "homogeneous"}, True),
            ("heterogeneous_timescale", {**common,
                "head_architecture_mix": "heterogeneous"}, True),
        )
    elif args.suite == "dropout":
        # Match roughly the same inference budget: 16 stochastic functions for
        # one head versus 3*5=15 for the small ensemble.  p=0.10 is deliberately
        # modest because this critic is only two hidden layers deep.
        raw_arms = (
            ("baseline_5_head", {
                "num_heads": 5, "dropout_probability": 0.0,
                "mc_dropout_samples": 1,
            }),
            ("dropout_1_head_16_mc", {
                "num_heads": 1, "dropout_probability": 0.10,
                "mc_dropout_samples": 16,
            }),
            ("dropout_3_head_5_mc", {
                "num_heads": 3, "dropout_probability": 0.10,
                "mc_dropout_samples": 5,
            }),
        )
        arms = tuple((name, overrides, False) for name, overrides in raw_arms)
    else:
        raw_arms = (
            ("baseline", {
                "randomized_prior_scale": 0.0,
                "bootstrap_probability": 0.8,
            }),
            ("prior_025", {
                "randomized_prior_scale": 0.25,
                "bootstrap_probability": 0.8,
            }),
            ("prior_025_bootstrap_05", {
                "randomized_prior_scale": 0.25,
                "bootstrap_probability": 0.5,
            }),
        )
        arms = tuple((name, overrides, False) for name, overrides in raw_arms)
    results = {}
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides, timescale_diversity in arms:
        cfg = _cfg(training, **overrides)
        trainer = ViolationCriticTrainer(
            int(dataset["input_dim"]), cfg,
            device=args.device, seed=args.seed,
        )
        trainer.recent.load_state_dict(dataset)
        windows = _window_pools(dataset, cfg) if timescale_diversity else None
        started = time.monotonic()
        for update in range(int(args.updates)):
            if windows is None:
                trainer._update_once()
            else:
                _update_timescale_diverse(trainer, dataset, windows)
            if (update + 1) % 500 == 0:
                print(f"[{name}] {update + 1}/{args.updates}", flush=True)
        result = _score(
            trainer, features, targets, steps,
            max_false_manageable_rate=float(args.false_manageable_rate),
        )
        result.update({
            "num_heads": cfg.num_heads,
            "dropout_probability": cfg.dropout_probability,
            "mc_dropout_samples": cfg.mc_dropout_samples,
            "head_architecture_mix": cfg.head_architecture_mix,
            "replay_timescale_diversity": timescale_diversity,
            "randomized_prior_scale": cfg.randomized_prior_scale,
            "bootstrap_probability": cfg.bootstrap_probability,
            "trainable_parameters": sum(
                p.numel() for p in trainer.live.parameters() if p.requires_grad
            ),
            "per_head_parameters": [
                sum(p.numel() for p in head.parameters() if p.requires_grad)
                for head in trainer.live.heads
            ],
            "updates": int(args.updates),
            "elapsed_s": time.monotonic() - started,
        })
        results[name] = result
        if state_dir is not None:
            torch.save(trainer.state_dict(), state_dir / f"{name}.pt")
        print(f"[{name}] " + json.dumps(result, sort_keys=True), flush=True)
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "training_checkpoint": args.training_checkpoint,
        "holdout_checkpoint": args.holdout_checkpoint,
        "min_holdout_step": int(args.min_holdout_step),
        "suite": args.suite,
        "training_rows": int(dataset["size"]),
        "holdout_rows": int(features.shape[0]),
        "arms": results,
        "state_dir": str(state_dir) if state_dir is not None else None,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("TIMING_PRIOR_ABLATION=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
