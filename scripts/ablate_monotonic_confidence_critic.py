#!/usr/bin/env python
"""Online-style evaluation of the monotonic confidence critic.

The input checkpoint must contain the bounded timing-critic replay produced by
an actor label-collection run.  Rows are restored to chronological order,
scored before being learned, and the newest actor steps are held out entirely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from amortized_opt.critic_task_sampler import GlobalCriticTaskSampler
from amortized_opt.monotonic_critic import (
    MonotonicConfidenceConfig,
    MonotonicConfidenceTrainer,
    expected_calibration_error,
)
from amortized_opt.violation_critic import (
    VIOLATION_NAMES,
    ViolationCriticConfig,
    ViolationCriticTrainer,
    calibrate_manageability_rule,
    robust_inverse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stream-batch", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    # Leave transfer margin for threshold drift between protected calibration
    # memory and the live stream. A 0.5% calibration target stayed below 1%
    # on both evicted and retained anchors in the replay-turnover stress test.
    parser.add_argument("--max-false-feasible-rate", type=float, default=0.005)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--knots", type=int, default=8)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--inertia-weight", type=float, default=0.50)
    parser.add_argument("--prior-scale", type=float, default=0.10)
    parser.add_argument("--no-baseline", action="store_true")
    return parser.parse_args()


def chronological_rows(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    state = checkpoint["extra"]["timing_critic"]
    replay = state["recent"]
    n = int(replay["size"])
    capacity = int(replay.get("capacity", n))
    cursor = int(replay.get("cursor", n))
    if n == capacity and cursor:
        order = torch.cat((torch.arange(cursor, n), torch.arange(0, cursor)))
    else:
        order = torch.arange(n)
    features = replay["features"][:n].float().index_select(0, order)
    targets = replay["targets"][:n].float().index_select(0, order)
    steps = replay["actor_step"][:n].long().index_select(0, order)
    # Actor-step order is authoritative if a checkpoint was assembled from
    # multiple pending batches around a ring wrap.
    stable = torch.argsort(steps, stable=True)
    return (
        features.index_select(0, stable),
        targets.index_select(0, stable),
        steps.index_select(0, stable),
        dict(state.get("config", {})),
    )


def experiment_rows(
    reference: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, int]:
    """Load a collector checkpoint or a compact ``CACHE::DATASET`` ref."""
    value = str(reference)
    if "::" in value:
        path_value, dataset = value.rsplit("::", 1)
        cache = torch.load(path_value, map_location="cpu", weights_only=False)
        if cache.get("format") != "monotonic_critic_actor_labels":
            raise ValueError(f"{path_value} is not a monotonic critic data cache")
        if dataset not in cache["datasets"]:
            raise KeyError(f"dataset {dataset!r} is not in {path_value}")
        row = cache["datasets"][dataset]
        return (
            row["features"].float(), row["targets"].float(),
            row["actor_step"].long(), dict(row.get("source_config", {})),
            int(row.get("source_actor_step", -1)),
        )
    checkpoint = torch.load(value, map_location="cpu", weights_only=False)
    features, targets, steps, config = chronological_rows(checkpoint)
    return features, targets, steps, config, int(checkpoint.get("step", -1))


def binary_auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    score = score.float().flatten().cpu()
    positive = positive.bool().flatten().cpu()
    pos = score[positive]
    neg = score[~positive]
    if not int(pos.numel()) or not int(neg.numel()):
        return float("nan")
    # Rank statistic without materializing the full positive/negative matrix.
    order = torch.argsort(score, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float32)
    return float(
        (ranks[positive].sum() - pos.numel() * (pos.numel() + 1) / 2)
        / max(pos.numel() * neg.numel(), 1)
    )


@torch.no_grad()
def evaluate_monotonic(
    trainer: MonotonicConfidenceTrainer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    residual_threshold: float = 0.0,
    confidence_threshold: float = 0.5,
) -> dict[str, float]:
    device = trainer.device
    x = features.to(device)
    target = targets.to(device)
    member = trainer.predict(x)
    prediction = member.mean(dim=0)
    member_confidence = trainer.confidence(x)
    confidence = member_confidence.mean(dim=0)
    correct = (prediction <= 0.0) == (target <= 0.0)
    true_safe = target.amax(dim=-1) <= 0.0
    conservative_score = torch.quantile(
        member.amax(dim=-1).float(), 0.8, dim=0,
    )
    pred_safe = conservative_score <= 0.0
    row_confidence = confidence.amin(dim=-1)
    accepted = (
        conservative_score <= float(residual_threshold)
    ) & (row_confidence >= float(confidence_threshold))
    tp = (pred_safe & true_safe).sum()
    fp = (pred_safe & ~true_safe).sum()
    result = {
        "rows": float(x.shape[0]),
        "mae_normalized": float((prediction - target).abs().mean()),
        "correct_rate": float(correct.float().mean()),
        "confidence_mean": float(confidence.mean()),
        "confidence_brier": float((confidence - correct.float()).square().mean()),
        "confidence_ece": float(expected_calibration_error(confidence, correct)),
        "confidence_correct_auc": binary_auc(confidence, correct),
        "support_mean": float(trainer.support_probability(x).mean()),
        "true_feasible": float(true_safe.float().mean()),
        "pred_feasible": float(pred_safe.float().mean()),
        "feasible_precision": float(tp / (tp + fp).clamp_min(1)),
        "feasible_recall": float(tp / true_safe.sum().clamp_min(1)),
        "false_feasible": float((pred_safe & ~true_safe).float().mean()),
        "confident_coverage": float(accepted.float().mean()),
        "confident_false_feasible": float((accepted & ~true_safe).float().mean()),
        "confident_precision": float(
            (accepted & true_safe).sum() / accepted.sum().clamp_min(1)
        ),
        "confident_recall": float(
            (accepted & true_safe).sum() / true_safe.sum().clamp_min(1)
        ),
        "decision_residual_threshold": float(residual_threshold),
        "decision_confidence_threshold": float(confidence_threshold),
        "ensemble_disagreement": float(member.std(dim=0, unbiased=False).mean()),
    }
    physical_error = (
        robust_inverse(prediction, trainer.cfg.scales)
        - robust_inverse(target, trainer.cfg.scales)
    ).abs().mean(dim=0)
    for name, value in zip(VIOLATION_NAMES, physical_error):
        result[f"mae_{name}"] = float(value)
    return result


@torch.no_grad()
def evaluate_baseline(
    trainer: ViolationCriticTrainer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    decision_threshold: float = 0.0,
) -> dict[str, float]:
    x = features.to(trainer.device)
    target = targets.to(trainer.device)
    member = trainer.predict(x)
    prediction = member.mean(dim=0)
    true_safe = target.amax(dim=-1) <= 0.0
    conservative_score = torch.quantile(
        member.amax(dim=-1).float(), 0.8, dim=0,
    )
    pred_safe = conservative_score <= float(decision_threshold)
    tp = (pred_safe & true_safe).sum()
    fp = (pred_safe & ~true_safe).sum()
    result = {
        "rows": float(x.shape[0]),
        "mae_normalized": float((prediction - target).abs().mean()),
        "true_feasible": float(true_safe.float().mean()),
        "pred_feasible": float(pred_safe.float().mean()),
        "feasible_precision": float(tp / (tp + fp).clamp_min(1)),
        "feasible_recall": float(tp / true_safe.sum().clamp_min(1)),
        "false_feasible": float((pred_safe & ~true_safe).float().mean()),
        "ensemble_disagreement": float(member.std(dim=0, unbiased=False).mean()),
        "decision_threshold": float(decision_threshold),
    }
    physical_error = (
        robust_inverse(prediction, trainer.cfg.scales)
        - robust_inverse(target, trainer.cfg.scales)
    ).abs().mean(dim=0)
    for name, value in zip(VIOLATION_NAMES, physical_error):
        result[f"mae_{name}"] = float(value)
    return result


@torch.no_grad()
def calibrate_monotonic_rule(
    trainer: MonotonicConfidenceTrainer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_false_rate: float,
    use_confidence: bool,
) -> dict[str, float]:
    x = features.to(trainer.device)
    target = targets.to(trainer.device)
    member = trainer.predict(x)
    residual_score = torch.quantile(member.amax(dim=-1).float(), 0.8, dim=0)
    confidence = trainer.confidence(x).mean(dim=0).amin(dim=-1)
    truth = target.amax(dim=-1) <= 0.0
    residual_candidates = torch.cat((
        residual_score[residual_score <= 0.0].unique(sorted=True),
        residual_score.new_tensor([min(-1.0e-6, float(residual_score.min()) - 1.0e-6)]),
    )).unique(sorted=True)
    if use_confidence:
        confidence_candidates = torch.quantile(
            confidence,
            torch.linspace(0.0, 1.0, 51, device=confidence.device),
        ).unique(sorted=True)
    else:
        confidence_candidates = confidence.new_tensor([-1.0])
    best = None
    n = max(int(truth.numel()), 1)
    for residual_threshold in residual_candidates:
        residual_ok = residual_score <= residual_threshold
        for confidence_threshold in confidence_candidates:
            accepted = residual_ok & (confidence >= confidence_threshold)
            tp = int((accepted & truth).sum())
            fp = int((accepted & ~truth).sum())
            false_rate = fp / n
            if false_rate > float(max_false_rate) + 1.0e-12:
                continue
            precision = tp / max(tp + fp, 1)
            candidate = (
                tp, precision, int(accepted.sum()),
                float(residual_threshold), float(confidence_threshold), false_rate,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return {
            "residual_threshold": float(residual_score.min()) - 1.0,
            "confidence_threshold": 1.0,
            "calibration_recall": 0.0,
            "calibration_precision": 0.0,
            "calibration_false_feasible": 0.0,
        }
    tp, precision, _count, residual_threshold, confidence_threshold, false_rate = best
    return {
        "residual_threshold": residual_threshold,
        "confidence_threshold": confidence_threshold,
        "calibration_recall": tp / max(int(truth.sum()), 1),
        "calibration_precision": precision,
        "calibration_false_feasible": false_rate,
    }


@torch.no_grad()
def monotonicity_check(trainer, features: torch.Tensor) -> dict[str, float]:
    x = features[: min(512, len(features))].to(trainer.device).clone()
    low = x.clone()
    high = x.clone()
    low[:, -2:] = torch.where(low[:, -2:] > 0.0, low[:, -2:].clamp(max=0.5), low[:, -2:])
    high[:, -2:] = torch.where(
        low[:, -2:] > 0.0,
        (low[:, -2:] + 0.25).clamp(max=trainer.cfg.max_tth_s),
        low[:, -2:],
    )
    low_prediction = trainer.predict(low)
    high_prediction = trainer.predict(high)
    increase = high_prediction - low_prediction
    return {
        "violation_fraction": float((increase > 1.0e-6).float().mean()),
        "largest_increase": float(increase.max()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    features, targets, steps, source_cfg, source_step = experiment_rows(
        args.dataset_checkpoint,
    )
    unique_steps = steps.unique(sorted=True)
    if unique_steps.numel() < 2:
        raise ValueError("dataset needs at least two actor steps")
    holdout_steps = max(1, int(round(
        float(unique_steps.numel()) * float(args.holdout_fraction)
    )))
    calibration_steps = max(1, int(round(
        float(unique_steps.numel()) * float(args.calibration_fraction)
    )))
    if holdout_steps + calibration_steps >= int(unique_steps.numel()):
        raise ValueError("calibration and holdout fractions leave no training steps")
    holdout_start = unique_steps[-holdout_steps]
    calibration_start = unique_steps[-holdout_steps - calibration_steps]
    train_mask = steps < calibration_start
    calibration_mask = (steps >= calibration_start) & (steps < holdout_start)
    holdout_mask = steps >= holdout_start
    train_x, train_y, train_steps = (
        features[train_mask], targets[train_mask], steps[train_mask]
    )
    holdout_x, holdout_y = features[holdout_mask], targets[holdout_mask]
    calibration_x, calibration_y = (
        features[calibration_mask], targets[calibration_mask]
    )

    target_kwargs = {
        key: source_cfg[key]
        for key in (
            "terminal_limit", "collision_sdf_limit", "joint_limit", "torque_limit",
            "terminal_scale", "collision_scale", "joint_scale", "torque_scale",
        )
        if key in source_cfg
    }
    cfg = MonotonicConfidenceConfig(
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        knots=args.knots,
        batch_size=args.stream_batch,
        updates_per_observation=args.updates_per_batch,
        replay_capacity=max(1024, int(train_x.shape[0])),
        background_confidence_weight=args.background_weight,
        inertia_weight=args.inertia_weight,
        randomized_prior_scale=args.prior_scale,
        **target_kwargs,
    )
    trainer = MonotonicConfidenceTrainer(
        int(features.shape[1]), cfg, device=device, seed=args.seed,
    )

    baseline = None
    if not args.no_baseline:
        baseline_cfg = ViolationCriticConfig(
            mode="shadow", num_heads=args.num_heads,
            hidden_dim=args.hidden_dim, depth=args.depth,
            batch_size=args.stream_batch,
            updates_per_step=args.updates_per_batch,
            recent_capacity=max(1024, int(train_x.shape[0])),
            audit_capacity=0,
            **target_kwargs,
        )
        baseline = ViolationCriticTrainer(
            int(features.shape[1]), baseline_cfg,
            device=device, seed=args.seed,
        )

    initial = evaluate_monotonic(trainer, holdout_x, holdout_y)
    timeline = []
    stream_batch = int(args.stream_batch)
    for start in range(0, int(train_x.shape[0]), stream_batch):
        stop = min(start + stream_batch, int(train_x.shape[0]))
        batch_x = train_x[start:stop]
        batch_y = train_y[start:stop]
        actor_step = int(train_steps[stop - 1])
        prequential = trainer.prequential_metrics(batch_x, batch_y)
        trainer.observe_targets(batch_x, batch_y, actor_step=actor_step)
        if baseline is not None:
            baseline.recent.add(batch_x, batch_y, actor_step=actor_step)
            for _ in range(int(args.updates_per_batch)):
                baseline._update_once()
        if not timeline or stop == int(train_x.shape[0]) or len(timeline) < 20:
            timeline.append({
                "rows_seen": stop,
                "actor_step": actor_step,
                **prequential,
            })

    raw_final = evaluate_monotonic(
        trainer, holdout_x, holdout_y,
        residual_threshold=0.0, confidence_threshold=-1.0,
    )
    residual_rule = calibrate_monotonic_rule(
        trainer, calibration_x, calibration_y,
        max_false_rate=args.max_false_feasible_rate,
        use_confidence=False,
    )
    joint_rule = calibrate_monotonic_rule(
        trainer, calibration_x, calibration_y,
        max_false_rate=args.max_false_feasible_rate,
        use_confidence=True,
    )
    residual_calibrated = evaluate_monotonic(
        trainer, holdout_x, holdout_y,
        residual_threshold=residual_rule["residual_threshold"],
        confidence_threshold=-1.0,
    )
    confidence_calibrated = evaluate_monotonic(
        trainer, holdout_x, holdout_y,
        residual_threshold=joint_rule["residual_threshold"],
        confidence_threshold=joint_rule["confidence_threshold"],
    )
    anchor_n = min(stream_batch, int(train_x.shape[0]))
    remembered = evaluate_monotonic(trainer, train_x[:anchor_n], train_y[:anchor_n])
    sampler = GlobalCriticTaskSampler(max_tth_s=cfg.max_tth_s)
    ood = sampler.sample(
        min(2048, int(holdout_x.shape[0])),
        generator=torch.Generator().manual_seed(args.seed + 901),
    )
    ood_confidence = float(trainer.confidence(ood.to(device)).mean())

    result: dict[str, object] = {
        "dataset_checkpoint": args.dataset_checkpoint,
        "source_step": source_step,
        "rows": int(features.shape[0]),
        "train_rows": int(train_x.shape[0]),
        "holdout_rows": int(holdout_x.shape[0]),
        "calibration_rows": int(calibration_x.shape[0]),
        "train_actor_steps": int(train_steps.unique().numel()),
        "calibration_actor_steps": int(steps[calibration_mask].unique().numel()),
        "holdout_actor_steps": int(steps[holdout_mask].unique().numel()),
        "config": vars(args),
        "initial_holdout": initial,
        "final_holdout": raw_final,
        "residual_calibration_rule": residual_rule,
        "residual_calibrated_holdout": residual_calibrated,
        "confidence_calibration_rule": joint_rule,
        "confidence_calibrated_holdout": confidence_calibrated,
        "remembered_early_stream": remembered,
        "ood_confidence_mean": ood_confidence,
        "monotonicity": monotonicity_check(trainer, holdout_x),
        "timeline": timeline,
    }
    if baseline is not None:
        baseline_calibration_prediction = baseline.predict(
            calibration_x.to(device), ema=True, physical=False,
        ).cpu()
        baseline_rule = calibrate_manageability_rule(
            baseline_calibration_prediction,
            calibration_y,
            quantiles=(0.8,),
            max_false_manageable_rate=args.max_false_feasible_rate,
        )["selected"]
        result["baseline_calibration_rule"] = baseline_rule
        result["baseline_holdout"] = evaluate_baseline(
            baseline, holdout_x, holdout_y,
            decision_threshold=float(baseline_rule["threshold"]),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save(trainer.state_dict(), output.with_suffix(".pt"))
    print("MONOTONIC_CRITIC_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
