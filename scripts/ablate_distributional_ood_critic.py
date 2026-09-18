#!/usr/bin/env python
"""Compare softmax/IsoMax, ensembles, OOD exposure, and calibration."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from amortized_opt.distributional_critic import (
    DistributionalEnsemble,
    MANAGEABLE_BINS,
    N_BINS,
    categorical_ordinal_loss,
    distributional_targets,
    high_entropy_loss,
    normalized_entropy,
    probabilities,
)
from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.limits import (
    ETA_VELOCITY,
    isaac_velocity_limits,
    urdf_position_limits,
)
from amortized_opt.rhythm import DrumKit, head_plane_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-checkpoint", required=True)
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--unknown-fraction", type=float, default=0.20)
    parser.add_argument("--unknown-weight", type=float, default=0.05)
    parser.add_argument(
        "--exposure-mode", choices=("static", "global"), default="static",
        help="static replayed near-OOD points or fresh global tasks every update",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--selected-only", action="store_true",
        help="run only the selected three-head Softmax plain/OE comparison",
    )
    parser.add_argument(
        "--arms", choices=("both", "plain", "oe"), default="both",
        help="select plain and/or entropy-exposed arms",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-output", default=None,
        help=(
            "optional checkpoint path; requires exactly one arm and stores "
            "its weights, calibration temperature, and feature contract"
        ),
    )
    return parser.parse_args()


def _split_by_time(recent: dict) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    n = int(recent["size"])
    x = recent["features"][:n].float()
    y = recent["targets"][:n].float()
    step = recent["actor_step"][:n].long()
    unique = step.unique(sorted=True)
    if int(unique.numel()) < 10:
        raise ValueError("dataset needs at least ten distinct actor steps")
    a = unique[max(1, int(0.70 * len(unique))) - 1]
    b = unique[max(2, int(0.85 * len(unique))) - 1]
    masks = (step <= a, (step > a) & (step <= b), step > b)
    return tuple((x[m], y[m]) for m in masks)


def _feature_groups(width: int) -> tuple[slice, ...]:
    n_dof, rem = divmod(int(width) - 8, 2)
    if rem or n_dof <= 0:
        raise ValueError(f"unexpected minimal critic width {width}")
    return (
        slice(0, n_dof),
        slice(n_dof, 2 * n_dof),
        slice(2 * n_dof, 2 * n_dof + 6),
        slice(2 * n_dof + 6, 2 * n_dof + 8),
    )


def make_unknown(
    reference: torch.Tensor,
    n: int,
    *,
    generator: torch.Generator,
    mode: str,
    magnitude_scale: float = 1.0,
) -> torch.Tensor:
    """Generate structured near-OOD points without leaving feature units."""
    groups = _feature_groups(reference.shape[1])
    base_idx = torch.randint(0, len(reference), (int(n),), generator=generator)
    out = reference.index_select(0, base_idx).clone()
    if mode == "factorized":
        # Each factor is individually in-distribution but their combination is
        # not an observed state/task pairing.
        for group in groups:
            idx = torch.randint(0, len(reference), (int(n),), generator=generator)
            out[:, group] = reference.index_select(0, idx)[:, group]
        return out
    if mode != "perturbed":
        raise ValueError(f"unknown OOD mode {mode!r}")

    which = torch.randint(0, len(groups), (int(n),), generator=generator)
    if not math.isfinite(float(magnitude_scale)) or magnitude_scale < 0.0:
        raise ValueError("magnitude_scale must be finite and nonnegative")
    magnitudes = tuple(
        float(magnitude_scale) * value for value in (0.14, 0.16, 0.12, 0.20)
    )
    for group_i, (group, magnitude) in enumerate(zip(groups, magnitudes)):
        rows = (which == group_i).nonzero(as_tuple=True)[0]
        if not int(rows.numel()):
            continue
        width = out[:, group].shape[1]
        direction = torch.randn(
            len(rows), width, generator=generator,
        )
        direction /= direction.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        out[rows, group] += float(magnitude) * direction
    # TTH remains a time in seconds; negative values would accidentally encode
    # an inactive hand and turn the perturbation into a different convention.
    out[:, groups[-1]] = out[:, groups[-1]].clamp(0.0, 1.25)
    return out


class GlobalTaskSampler:
    """Fresh Monte-Carlo samples from the bounded physical task domain.

    Every hand draws uniformly from its legally reachable drums plus one idle
    choice.  Thus single-hand tasks are rare as a consequence of the task
    alphabet rather than a separately tuned mode probability.  Samples are
    emitted directly in the minimal critic feature convention.
    """

    def __init__(self, *, max_tth_s: float = 1.0, min_active_tth_s: float = 0.08):
        self.joint_names = tuple(G1_UPPER_BODY_JOINTS)
        q_lo, q_hi = urdf_position_limits(self.joint_names, posture_limits=True)
        qd_hi = ETA_VELOCITY * isaac_velocity_limits(self.joint_names)
        self.q_lo = torch.as_tensor(q_lo, dtype=torch.float32) / math.pi
        self.q_hi = torch.as_tensor(q_hi, dtype=torch.float32) / math.pi
        self.qd_hi = torch.as_tensor(qd_hi, dtype=torch.float32) / 10.0
        self.max_tth_s = float(max_tth_s)
        self.min_active_tth_s = float(min_active_tth_s)
        if not 0.0 < self.min_active_tth_s <= self.max_tth_s:
            raise ValueError("global TTH bounds must satisfy 0 < min <= max")
        self.min_tth_wp = int(math.ceil(self.min_active_tth_s / CONTROL_DT))
        self.max_tth_wp = int(math.floor(self.max_tth_s / CONTROL_DT))

        kit = DrumKit.from_layout()
        self.positions = torch.as_tensor(kit.positions, dtype=torch.float32)
        self.radii = torch.as_tensor(kit.radii, dtype=torch.float32)
        basis = [head_plane_basis(axis) for axis in kit.axes]
        self.basis_u = torch.as_tensor(
            np.stack([row[0] for row in basis]), dtype=torch.float32,
        )
        self.basis_v = torch.as_tensor(
            np.stack([row[1] for row in basis]), dtype=torch.float32,
        )
        self.eligible = (
            torch.as_tensor(kit.eligible("left"), dtype=torch.long),
            torch.as_tensor(kit.eligible("right"), dtype=torch.long),
        )

    @staticmethod
    def _disk(
        n: int, radius: torch.Tensor | float, *, generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.sqrt(torch.rand(int(n), generator=generator))
        r = r * torch.as_tensor(radius, dtype=torch.float32)
        theta = 2.0 * math.pi * torch.rand(int(n), generator=generator)
        return r * torch.cos(theta), r * torch.sin(theta)

    def sample(self, n: int, *, generator: torch.Generator) -> torch.Tensor:
        n = int(n)
        q = self.q_lo + torch.rand(n, len(self.q_lo), generator=generator) * (
            self.q_hi - self.q_lo
        )
        qd = (
            2.0 * torch.rand(n, len(self.qd_hi), generator=generator) - 1.0
        ) * self.qd_hi
        goals = torch.zeros(n, 2, 3)
        tth = torch.zeros(n, 2)
        # One randomized kit per task row.  Both hands must see the same global
        # translation and the same per-drum placement jitter.
        num_drums = int(self.positions.shape[0])
        kit_x, kit_y = self._disk(
            n * num_drums, 0.04, generator=generator,
        )
        kit_x = kit_x.reshape(n, num_drums)
        kit_y = kit_y.reshape(n, num_drums)
        global_x, global_y = self._disk(n, 0.075, generator=generator)
        global_z = (
            2.0 * torch.rand(n, generator=generator) - 1.0
        ) * 0.035
        for hand in range(2):
            eligible = self.eligible[hand]
            # The final category is idle, with exactly the same probability as
            # each reachable drum.
            choice = torch.randint(
                0, len(eligible) + 1, (n,), generator=generator,
            )
            active = choice < len(eligible)
            rows = active.nonzero(as_tuple=True)[0]
            if not int(rows.numel()):
                continue
            drum = eligible.index_select(0, choice.index_select(0, rows))
            pos = self.positions.index_select(0, drum).clone()

            # Random legal head contact, per-drum horizontal placement jitter,
            # and one global kit offset.  The nominal head-plane basis retains
            # tilted-drum geometry without requiring an actor/context encoder.
            hit_r = 0.6 * self.radii.index_select(0, drum)
            du, dv = self._disk(len(rows), hit_r, generator=generator)
            pos += du[:, None] * self.basis_u.index_select(0, drum)
            pos += dv[:, None] * self.basis_v.index_select(0, drum)
            pos[:, 0] += kit_x[rows, drum] + global_x[rows]
            pos[:, 1] += kit_y[rows, drum] + global_y[rows]
            pos[:, 2] += global_z[rows]
            goals[rows, hand] = pos
            tth[rows, hand] = CONTROL_DT * torch.randint(
                self.min_tth_wp, self.max_tth_wp + 1,
                (len(rows),), generator=generator,
            ).float()
        return torch.cat((q, qd, goals.reshape(n, 6), tth), dim=1)


def _auc(positive_score: torch.Tensor, negative_score: torch.Tensor) -> float:
    """Probability that an OOD score exceeds an ID score."""
    pos = positive_score.float().cpu()
    neg = negative_score.float().cpu()
    pair = pos[:, None] - neg[None, :]
    return float(((pair > 0).float() + 0.5 * (pair == 0).float()).mean())


def _ece(probability: torch.Tensor, truth: torch.Tensor, bins: int = 10) -> float:
    total = max(int(probability.numel()), 1)
    value = probability.new_zeros(())
    for i in range(int(bins)):
        lo, hi = i / bins, (i + 1) / bins
        mask = (probability >= lo) & (
            probability <= hi if i == bins - 1 else probability < hi
        )
        if bool(mask.any()):
            value += mask.float().sum() / total * (
                probability[mask].mean() - truth[mask].float().mean()
            ).abs()
    return float(value)


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Finite Pearson correlation, used only as an uncertainty diagnostic."""
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1.0e-12:
        return 0.0
    return float((x * y).sum() / denominator)


def _multiclass_ece(
    probability: torch.Tensor, target: torch.Tensor, bins: int = 10,
) -> float:
    confidence, prediction = probability.max(dim=-1)
    return _ece(confidence, prediction == target, bins=bins)


def _selective_metrics(
    probability: torch.Tensor,
    entropy: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    """Measure whether low entropy really selects more reliable predictions."""
    prediction = probability.argmax(dim=-1)
    error = (prediction != target).float()
    picked = probability.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll = -picked.clamp_min(1.0e-8).log()
    order = entropy.argsort()
    ordered_error = error.index_select(0, order)
    cumulative_risk = ordered_error.cumsum(0) / torch.arange(
        1, len(error) + 1, device=error.device,
    )
    rows: dict[str, dict[str, float]] = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        keep = max(1, int(math.ceil(float(coverage) * len(order))))
        index = order[:keep]
        rows[f"{coverage:.1f}"] = {
            "bin_accuracy": float(1.0 - error.index_select(0, index).mean()),
            "nll": float(nll.index_select(0, index).mean()),
            "mean_abs_bin_error": float(
                (prediction.index_select(0, index) - target.index_select(0, index))
                .abs().float().mean()
            ),
        }
    incorrect = error.bool()
    uncertainty_auc = (
        _auc(entropy[incorrect], entropy[~incorrect])
        if bool(incorrect.any()) and bool((~incorrect).any())
        else float("nan")
    )
    return {
        "aurc": float(cumulative_risk.mean()),
        "entropy_error_auc": uncertainty_auc,
        "coverage": rows,
    }


@torch.no_grad()
def _prediction(model, x: torch.Tensor, temperature=1.0) -> dict[str, torch.Tensor]:
    logits, distance = model(x)
    member_p = probabilities(logits, temperature)
    mean_p = member_p.mean(dim=0)
    distribution_h = normalized_entropy(mean_p)
    predictive_h = distribution_h.mean(dim=-1)
    member_distribution_h = normalized_entropy(member_p)
    member_h = member_distribution_h.mean(dim=-1)
    js = predictive_h - member_h.mean(dim=0)
    result = {
        "logits": logits,
        "probability": mean_p,
        "distribution_entropy": distribution_h,
        "entropy": predictive_h,
        "final_entropy": distribution_h[:, -1],
        "disagreement": js,
    }
    if distance is not None:
        # Average over independent heads and the five diagnostic distributions.
        result["min_distance"] = distance.mean(dim=(0, 2))
    return result


def fit_temperature(model, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fit one positive temperature per diagnostic distribution."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    log_t = torch.zeros(5, device=x.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(300):
        p = probabilities(logits, log_t.exp()).mean(dim=0)
        picked = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -picked.clamp_min(1e-8).log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return log_t.detach().exp().clamp(0.05, 20.0)


def _id_metrics(prediction: dict, target: torch.Tensor) -> dict[str, object]:
    p = prediction["probability"]
    picked = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    one_hot = F.one_hot(target, num_classes=N_BINS).to(p.dtype)
    predicted_bin = p.argmax(dim=-1)
    bin_error = (predicted_bin - target).abs().float()
    pred_cdf = p.cumsum(dim=-1)[..., :-1]
    true_cdf = one_hot.cumsum(dim=-1)[..., :-1]
    true_safe = target[:, -1] < MANAGEABLE_BINS
    p_safe = p[:, -1, :MANAGEABLE_BINS].sum(dim=-1)
    pred_safe = p_safe >= 0.5
    tp = (pred_safe & true_safe).sum().float()
    fp = (pred_safe & ~true_safe).sum().float()
    final_nll = -picked[:, -1].clamp_min(1e-8).log()
    final_entropy = prediction["final_entropy"]
    result: dict[str, object] = {
        "bin_accuracy": float((p.argmax(dim=-1) == target).float().mean()),
        "final_bin_accuracy": float((predicted_bin[:, -1] == target[:, -1]).float().mean()),
        "nll": float(-picked.clamp_min(1e-8).log().mean()),
        "final_nll": float(final_nll.mean()),
        "categorical_brier": float((p - one_hot).square().sum(dim=-1).mean()),
        "ranked_probability_score": float(
            (pred_cdf - true_cdf).square().mean()
        ),
        "mean_abs_bin_error": float(bin_error.mean()),
        "final_mean_abs_bin_error": float(bin_error[:, -1].mean()),
        "final_confidence_ece": _multiclass_ece(p[:, -1], target[:, -1]),
        "safe_brier": float((p_safe - true_safe.float()).square().mean()),
        "safe_ece": _ece(p_safe, true_safe),
        "safe_accuracy": float((pred_safe == true_safe).float().mean()),
        "safe_auc": _auc(p_safe[true_safe], p_safe[~true_safe]),
        "safe_recall": float(tp / true_safe.sum().clamp_min(1)),
        "safe_precision": float(tp / (tp + fp).clamp_min(1)),
        "false_safe_rate": float((pred_safe & ~true_safe).float().mean()),
        "mean_entropy": float(prediction["entropy"].mean()),
        "mean_final_entropy": float(final_entropy.mean()),
        "mean_disagreement": float(prediction["disagreement"].mean()),
        "final_entropy_nll_correlation": _correlation(final_entropy, final_nll),
        "final_entropy_bin_error_correlation": _correlation(
            final_entropy, bin_error[:, -1],
        ),
    }
    result["selective_final"] = _selective_metrics(
        p[:, -1], final_entropy, target[:, -1],
    )
    return result


def _ood_metrics(id_pred: dict, ood_pred: dict) -> dict[str, float]:
    def standardized(score: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (score - reference.mean()) / reference.std(unbiased=False).clamp_min(1.0e-6)

    id_entropy_z = standardized(id_pred["entropy"], id_pred["entropy"])
    ood_entropy_z = standardized(ood_pred["entropy"], id_pred["entropy"])
    id_disagreement_z = standardized(
        id_pred["disagreement"], id_pred["disagreement"],
    )
    ood_disagreement_z = standardized(
        ood_pred["disagreement"], id_pred["disagreement"],
    )
    out = {
        "entropy_auc": _auc(ood_pred["entropy"], id_pred["entropy"]),
        "disagreement_auc": _auc(
            ood_pred["disagreement"], id_pred["disagreement"],
        ),
        # Fixed rules with scales estimated only from ID predictions.  No OOD
        # labels or per-dataset mixture weight are used to tune the scores.
        "combined_sum_auc": _auc(
            ood_entropy_z + ood_disagreement_z,
            id_entropy_z + id_disagreement_z,
        ),
        "combined_or_auc": _auc(
            torch.maximum(ood_entropy_z, ood_disagreement_z),
            torch.maximum(id_entropy_z, id_disagreement_z),
        ),
        "id_entropy": float(id_pred["entropy"].mean()),
        "ood_entropy": float(ood_pred["entropy"].mean()),
    }
    if "min_distance" in id_pred:
        out["min_distance_auc"] = _auc(
            ood_pred["min_distance"], id_pred["min_distance"],
        )
        out["id_min_distance"] = float(id_pred["min_distance"].mean())
        out["ood_min_distance"] = float(ood_pred["min_distance"].mean())
    return out


def train_arm(
    *,
    kind: str,
    heads: int,
    exposure: bool,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    unknown: torch.Tensor | None,
    global_sampler: GlobalTaskSampler | None,
    updates: int,
    batch: int,
    unknown_fraction: float,
    unknown_weight: float,
    lr: float,
    seed: int,
) -> tuple[DistributionalEnsemble, float]:
    torch.manual_seed(int(seed))
    model = DistributionalEnsemble(
        train_x.shape[1], num_heads=heads, kind=kind,
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 91)
    unknown_generators = [
        torch.Generator(device="cpu").manual_seed(int(seed) + 10_091 + i)
        for i in range(int(heads))
    ]
    target_bin = distributional_targets(train_y)
    unknown_n = max(1, int(round(float(batch) * float(unknown_fraction))))
    started = time.perf_counter()
    model.train()
    for _ in range(int(updates)):
        losses = []
        for head_i, head in enumerate(model.heads):
            index = torch.randint(0, len(train_x), (int(batch),), generator=generator)
            logits, _ = head(train_x.index_select(0, index.to(train_x.device)))
            loss = categorical_ordinal_loss(
                logits.unsqueeze(0),
                target_bin.index_select(0, index.to(target_bin.device)),
            )
            if exposure:
                if global_sampler is not None:
                    unknown_batch = global_sampler.sample(
                        unknown_n, generator=unknown_generators[head_i],
                    ).to(train_x.device)
                else:
                    if unknown is None:
                        raise RuntimeError("static exposure requires unknown data")
                    uindex = torch.randint(
                        0, len(unknown), (unknown_n,), generator=generator,
                    )
                    unknown_batch = unknown.index_select(
                        0, uindex.to(unknown.device),
                    )
                unknown_logits, _ = head(unknown_batch)
                loss = loss + float(unknown_weight) * high_entropy_loss(
                    unknown_logits.unsqueeze(0)
                )
            losses.append(loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), time.perf_counter() - started


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    checkpoint = torch.load(
        args.dataset_checkpoint, map_location="cpu", weights_only=False,
    )
    recent = checkpoint["extra"]["timing_critic"]["recent"]
    (train_x, train_y), (cal_x, cal_y), (test_x, test_y) = _split_by_time(recent)
    train_x, train_y = train_x.to(device), train_y.to(device)
    cal_x, cal_y = cal_x.to(device), cal_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    global_sampler = GlobalTaskSampler() if args.exposure_mode == "global" else None
    train_unknown = None
    if global_sampler is None:
        unknown_gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
        train_unknown = torch.cat((
            make_unknown(train_x.cpu(), len(train_x) // 2, generator=unknown_gen, mode="factorized"),
            make_unknown(train_x.cpu(), len(train_x) // 2, generator=unknown_gen, mode="perturbed"),
        )).to(device)
    test_gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + 2)
    ood_sets = {
        "factorized": make_unknown(
            test_x.cpu(), len(test_x), generator=test_gen, mode="factorized",
        ).to(device),
        "perturbed": make_unknown(
            test_x.cpu(), len(test_x), generator=test_gen, mode="perturbed",
        ).to(device),
    }
    # Use identical reference rows, feature groups, and perturbation directions
    # at every radius so the entropy profile measures distance rather than a
    # different Monte-Carlo draw.  Scale 0 is a permuted ID control.
    for scale in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        radial_gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + 4)
        ood_sets[f"radial_{scale:g}"] = make_unknown(
            test_x.cpu(), len(test_x), generator=radial_gen,
            mode="perturbed", magnitude_scale=scale,
        ).to(device)
    global_test_gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + 3)
    global_test_sampler = global_sampler or GlobalTaskSampler()
    ood_sets["global"] = global_test_sampler.sample(
        len(test_x), generator=global_test_gen,
    ).to(device)

    arms = {}
    saved_arm = None
    kinds = ("softmax",) if args.selected_only else ("softmax", "isomax")
    head_counts = (3,) if args.selected_only else (1, 3)
    for kind in kinds:
        for heads in head_counts:
            exposures = {
                "both": (False, True), "plain": (False,), "oe": (True,),
            }[args.arms]
            for exposure in exposures:
                name = f"{kind}_{heads}h_{'oe' if exposure else 'plain'}"
                model, train_seconds = train_arm(
                    kind=kind, heads=heads, exposure=exposure,
                    train_x=train_x, train_y=train_y,
                    unknown=train_unknown, global_sampler=global_sampler,
                    updates=args.updates,
                    batch=args.batch, unknown_fraction=args.unknown_fraction,
                    unknown_weight=args.unknown_weight,
                    lr=args.lr,
                    seed=args.seed + (0 if kind == "softmax" else 10_000) + heads,
                )
                cal_target = distributional_targets(cal_y)
                test_target = distributional_targets(test_y)
                temperature = fit_temperature(model, cal_x, cal_target)
                raw_id = _prediction(model, test_x)
                calibrated_id = _prediction(model, test_x, temperature)
                result = {
                    "kind": kind,
                    "heads": heads,
                    "outlier_exposure": exposure,
                    "train_seconds": train_seconds,
                    "temperature": [float(x) for x in temperature],
                    "id_raw": _id_metrics(raw_id, test_target),
                    "id_calibrated": _id_metrics(calibrated_id, test_target),
                    "ood": {},
                }
                for ood_name, ood_x in ood_sets.items():
                    result["ood"][ood_name] = {
                        "raw": _ood_metrics(raw_id, _prediction(model, ood_x)),
                        "calibrated": _ood_metrics(
                            calibrated_id, _prediction(model, ood_x, temperature),
                        ),
                    }
                arms[name] = result
                if args.model_output is not None:
                    if saved_arm is not None:
                        raise ValueError(
                            "--model-output requires an invocation that runs exactly one arm"
                        )
                    saved_arm = {
                        "format_version": 1,
                        "name": name,
                        "input_dim": int(train_x.shape[1]),
                        "kind": kind,
                        "heads": int(heads),
                        "hidden_dim": 128,
                        "depth": 2,
                        "embedding_dim": 64,
                        "model": model.state_dict(),
                        "temperature": temperature.detach().cpu(),
                        "dataset_checkpoint": args.dataset_checkpoint,
                        "training": {
                            "updates": int(args.updates),
                            "batch": int(args.batch),
                            "lr": float(args.lr),
                            "unknown_fraction": float(args.unknown_fraction),
                            "unknown_weight": float(args.unknown_weight),
                            "exposure_mode": args.exposure_mode,
                            "seed": int(args.seed),
                        },
                    }
                print(f"[{name}] " + json.dumps(result, sort_keys=True), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    report = {
        "dataset_checkpoint": args.dataset_checkpoint,
        "rows": {
            "train": len(train_x), "calibration": len(cal_x), "test": len(test_x),
            "unknown_train": (
                "fresh_per_update" if global_sampler is not None
                else len(train_unknown)
            ),
        },
        "updates": int(args.updates),
        "batch": int(args.batch),
        "lr": float(args.lr),
        "unknown_fraction": float(args.unknown_fraction),
        "unknown_weight": float(args.unknown_weight),
        "exposure_mode": args.exposure_mode,
        "selected_only": bool(args.selected_only),
        "arms_run": args.arms,
        "arms": arms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.model_output is not None:
        if saved_arm is None:
            raise RuntimeError("no distributional critic arm was trained")
        model_output = Path(args.model_output)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_arm, model_output)
    print("DISTRIBUTIONAL_OOD_ABLATION=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
