#!/usr/bin/env python
"""Evaluate a trained monotonic critic on exact-labeled distribution shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from amortized_opt.monotonic_critic import (
    MonotonicConfidenceConfig,
    MonotonicConfidenceTrainer,
    expected_calibration_error,
)
from scripts.ablate_monotonic_confidence_critic import (
    binary_auc,
    chronological_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--id-dataset", required=True)
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=CHECKPOINT",
        help="exact-labeled evaluation checkpoint; repeat for multiple shifts",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows_with_metadata(
    reference: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load a collector checkpoint or ``CACHE::DATASET`` compact-cache ref."""
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
        )
    checkpoint = torch.load(value, map_location="cpu", weights_only=False)
    return chronological_rows(checkpoint)


def load_rows(reference: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    features, targets, _steps, _cfg = load_rows_with_metadata(reference)
    return features, targets


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--dataset must have the form NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError("dataset name cannot be empty")
    return name, Path(path)


def actor_unchanged(source: dict, collected: dict) -> dict[str, float | bool]:
    source_model = source["model"]
    collected_model = collected["model"]
    changed = 0
    maximum = 0.0
    for name, reference in source_model.items():
        candidate = collected_model[name]
        difference = (candidate.cpu() - reference.cpu()).abs()
        local = float(difference.max()) if difference.numel() else 0.0
        maximum = max(maximum, local)
        changed += int(local != 0.0)
    return {
        "unchanged": changed == 0,
        "changed_tensors": changed,
        "max_abs_change": maximum,
    }


def decision_metrics(accepted: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    accepted = accepted.bool()
    truth = truth.bool()
    tp = int((accepted & truth).sum())
    fp = int((accepted & ~truth).sum())
    n = max(int(truth.numel()), 1)
    return {
        "accepted": float(accepted.sum()),
        "coverage": float(accepted.float().mean()),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "precision": tp / max(tp + fp, 1),
        "false_discovery_rate": fp / max(tp + fp, 1),
        "recall": tp / max(int(truth.sum()), 1),
        "false_feasible_rate": fp / n,
    }


def quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu().flatten()
    q = torch.quantile(value, torch.tensor([0.05, 0.50, 0.95]))
    return {
        "mean": float(value.mean()),
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
    }


@torch.no_grad()
def evaluate(
    trainer: MonotonicConfidenceTrainer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    residual_rule: dict,
    joint_rule: dict,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    x = features.to(trainer.device)
    target = targets.to(trainer.device)
    member = trainer.predict(x)
    prediction = member.mean(dim=0)
    score = torch.quantile(member.amax(dim=-1).float(), 0.8, dim=0)
    support = trainer.support_probability(x)
    confidence_element = trainer.confidence(x).mean(dim=0)
    learned_element = trainer.confidence(
        x, temper_by_support=False,
    ).mean(dim=0)
    confidence = confidence_element.amin(dim=-1)
    learned_confidence = learned_element.amin(dim=-1)
    correct = (prediction <= 0.0) == (target <= 0.0)
    truth = target.amax(dim=-1) <= 0.0

    residual_threshold = float(residual_rule["residual_threshold"])
    joint_residual = float(joint_rule["residual_threshold"])
    joint_confidence = float(joint_rule["confidence_threshold"])
    raw = score <= 0.0
    residual = score <= residual_threshold
    joint_margin = score <= joint_residual
    joint_untempered = joint_margin & (learned_confidence >= joint_confidence)
    joint = joint_margin & (confidence >= joint_confidence)

    optimism = (target - prediction).clamp_min(0.0)
    wrong = ~correct
    result: dict[str, object] = {
        "rows": int(x.shape[0]),
        "true_feasible_rate": float(truth.float().mean()),
        "mae_normalized": float((prediction - target).abs().mean()),
        "optimistic_error_mean": float(optimism.mean()),
        "optimistic_error_p95": float(torch.quantile(optimism.flatten(), 0.95)),
        "constraint_sign_accuracy": float(correct.float().mean()),
        "confidence_brier": float(
            (confidence_element - correct.float()).square().mean()
        ),
        "confidence_ece": float(expected_calibration_error(
            confidence_element, correct,
        )),
        "confidence_correct_auc": binary_auc(confidence_element, correct),
        "confidence": quantiles(confidence),
        "learned_confidence_without_support": quantiles(learned_confidence),
        "support": quantiles(support),
        "mean_confidence_on_correct_constraints": float(
            confidence_element[correct].mean()
        ),
        "mean_confidence_on_incorrect_constraints": (
            float(confidence_element[wrong].mean()) if bool(wrong.any()) else 0.0
        ),
        "decisions": {
            "raw_zero_boundary": decision_metrics(raw, truth),
            "calibrated_residual_only": decision_metrics(residual, truth),
            "joint_margin_without_confidence": decision_metrics(joint_margin, truth),
            "joint_with_learned_confidence_only": decision_metrics(
                joint_untempered, truth,
            ),
            "joint_with_support_tempered_confidence": decision_metrics(joint, truth),
        },
    }
    return result, {
        "support": support.detach().cpu(),
        "confidence": confidence.detach().cpu(),
        "learned_confidence": learned_confidence.detach().cpu(),
    }


def main() -> None:
    args = parse_args()
    saved = torch.load(
        args.critic_checkpoint, map_location="cpu", weights_only=False,
    )
    cfg = MonotonicConfidenceConfig(**saved["config"])
    trainer = MonotonicConfidenceTrainer(
        int(saved["input_dim"]), cfg, device=args.device, seed=0,
    )
    trainer.load_state_dict(saved)
    report = json.loads(Path(args.calibration_report).read_text())
    residual_rule = report["residual_calibration_rule"]
    joint_rule = report["confidence_calibration_rule"]

    id_x, id_y = load_rows(args.id_dataset)
    holdout_rows = int(report["holdout_rows"])
    id_x, id_y = id_x[-holdout_rows:], id_y[-holdout_rows:]
    id_result, id_score = evaluate(
        trainer, id_x, id_y,
        residual_rule=residual_rule, joint_rule=joint_rule,
    )

    source = torch.load(
        report["dataset_checkpoint"], map_location="cpu", weights_only=False,
    )
    datasets: dict[str, object] = {}
    for value in args.dataset:
        name, path = parse_dataset(value)
        x, y = load_rows(path)
        result, score = evaluate(
            trainer, x, y,
            residual_rule=residual_rule, joint_rule=joint_rule,
        )
        result["checkpoint"] = str(path)
        if "::" in str(path):
            result["actor"] = {"source": "cached_verified_labels"}
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            result["actor"] = actor_unchanged(source, checkpoint)
        result["shift_detection_auc"] = {
            "one_minus_support": binary_auc(
                torch.cat((1.0 - score["support"], 1.0 - id_score["support"])),
                torch.cat((
                    torch.ones_like(score["support"]),
                    torch.zeros_like(id_score["support"]),
                )),
            ),
            "one_minus_tempered_confidence": binary_auc(
                torch.cat((1.0 - score["confidence"], 1.0 - id_score["confidence"])),
                torch.cat((
                    torch.ones_like(score["confidence"]),
                    torch.zeros_like(id_score["confidence"]),
                )),
            ),
            "one_minus_learned_confidence": binary_auc(
                torch.cat((
                    1.0 - score["learned_confidence"],
                    1.0 - id_score["learned_confidence"],
                )),
                torch.cat((
                    torch.ones_like(score["learned_confidence"]),
                    torch.zeros_like(id_score["learned_confidence"]),
                )),
            ),
        }
        datasets[name] = result

    output = {
        "critic_checkpoint": args.critic_checkpoint,
        "calibration_report": args.calibration_report,
        "id_dataset": args.id_dataset,
        "rules": {"residual": residual_rule, "joint": joint_rule},
        "id_holdout": id_result,
        "datasets": datasets,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("MONOTONIC_OOD_RESULT=" + json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
