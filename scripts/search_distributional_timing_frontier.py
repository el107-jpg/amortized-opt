#!/usr/bin/env python
"""Propose two-hand collocations with a saved distributional timing critic.

The critic is deliberately only a proposal mechanism.  Unknown tasks fall
back to the generous timing anchor so that the exact actor/physics pass can
turn them into useful labels without trusting an extrapolated frontier.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from amortized_opt.distributional_critic import (
    DistributionalEnsemble,
    MANAGEABLE_BINS,
    distributional_risk,
    distributional_targets,
    normalized_entropy,
    probabilities,
)
from amortized_opt.timing_curriculum import (
    anchor_connected_component,
    pareto_frontier_mask,
)
from amortized_opt.violation_critic import build_critic_features
from amortized_opt.violation_critic import residuals_from_gate, robust_transform
from amortized_opt.ik import correct_quintic_init_arms
from amortized_opt.train_cost import (
    _gate_from_ct,
    _index_batch,
    build_costs,
    effective_hits,
    group_lengths,
    phase2_config_from_checkpoint,
    retime_batch_timing,
)
try:
    from scripts.ablate_distributional_ood_critic import (
        GlobalTaskSampler,
        _split_by_time,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
    from ablate_distributional_ood_critic import (  # type: ignore[no-redef]
        GlobalTaskSampler,
        _split_by_time,
    )

try:
    from scripts.run_timing_critic_trial import (
        load_model, load_pool, scenario_from_checkpoint,
    )
except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
    from run_timing_critic_trial import (  # type: ignore[no-redef]
        load_model, load_pool,
        scenario_from_checkpoint,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument(
        "--task-source", choices=("global", "pool"), default="global",
        help="global bounded exploration or random tasks from the actor pool",
    )
    parser.add_argument("--pool-cache", default=None)
    parser.add_argument(
        "--exact-verify", action="store_true",
        help="run selected pool frontiers through the frozen actor and physics",
    )
    parser.add_argument(
        "--exact-verify-all", action="store_true",
        help="also verify fallback anchor probes (diagnostic; more expensive)",
    )
    parser.add_argument(
        "--decision-rule", choices=("risk", "legacy"), default="risk",
    )
    parser.add_argument(
        "--acceptable-normalized-violation", type=float, default=0.2,
        help="largest worst signed-asinh residual considered close enough",
    )
    parser.add_argument(
        "--frontier-backoff-knots", type=int, default=0,
        help="diagnostic: move selected frontier timings toward the anchor",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--minimum-safe-precision", type=float, default=0.50,
        help=(
            "held-out proposal precision; 0.5 is intentional because every "
            "candidate still requires exact actor/physics verification"
        ),
    )
    parser.add_argument("--uncertainty-quantile", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def prediction(model, features: torch.Tensor, temperature: torch.Tensor) -> dict:
    logits, _ = model(features)
    member = probabilities(logits, temperature)
    mean = member.mean(dim=0)
    distribution_entropy = normalized_entropy(mean)
    member_entropy = normalized_entropy(member).mean(dim=-1)
    predictive_entropy = distribution_entropy.mean(dim=-1)
    disagreement = predictive_entropy - member_entropy.mean(dim=0)
    return {
        "member_probability": member,
        "member_safe": member[..., -1, :MANAGEABLE_BINS].sum(dim=-1),
        "safe": mean[:, -1, :MANAGEABLE_BINS].sum(dim=-1),
        "entropy": predictive_entropy,
        "final_entropy": distribution_entropy[:, -1],
        "disagreement": disagreement,
    }


def calibrate_safe_threshold(
    safe_probability: torch.Tensor,
    truth: torch.Tensor,
    *,
    minimum_precision: float,
) -> dict[str, float]:
    """Choose the recall-maximizing threshold at a requested precision."""
    p = safe_probability.detach().float().cpu()
    y = truth.detach().bool().cpu()
    thresholds = torch.cat((p.unique(sorted=True), p.new_tensor([1.0 + 1e-6])))
    best = None
    for threshold in thresholds:
        selected = p >= threshold
        count = int(selected.sum())
        if count == 0:
            continue
        tp = int((selected & y).sum())
        precision = tp / count
        recall = tp / max(int(y.sum()), 1)
        if precision + 1e-12 < float(minimum_precision):
            continue
        candidate = (recall, count, -float(threshold), precision)
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), precision, recall, count)
    if best is None:
        return {
            "threshold": 1.0 + 1e-6,
            "precision": 1.0,
            "recall": 0.0,
            "selected": 0.0,
        }
    return {
        "threshold": best[1],
        "precision": best[2],
        "recall": best[3],
        "selected": float(best[4]),
    }


def calibrate_risk_threshold(
    risk: torch.Tensor,
    truth: torch.Tensor,
    *,
    minimum_precision: float,
) -> dict[str, float]:
    """Choose the maximum-coverage upper risk bound at requested precision."""
    value = risk.detach().float().cpu()
    y = truth.detach().bool().cpu()
    best = None
    for threshold in value.unique(sorted=True):
        selected = value <= threshold
        count = int(selected.sum())
        tp = int((selected & y).sum())
        precision = tp / max(count, 1)
        recall = tp / max(int(y.sum()), 1)
        if precision + 1e-12 < float(minimum_precision):
            continue
        candidate = (recall, count, float(threshold), precision)
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), precision, recall, count)
    if best is None:
        return {
            "threshold": float("-inf"), "precision": 1.0,
            "recall": 0.0, "selected": 0.0,
        }
    return {
        "threshold": best[1], "precision": best[2],
        "recall": best[3], "selected": float(best[4]),
    }


def _load_critic(path: Path, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = DistributionalEnsemble(
        int(saved["input_dim"]),
        num_heads=int(saved["heads"]),
        kind=str(saved["kind"]),
        hidden_dim=int(saved.get("hidden_dim", 128)),
        depth=int(saved.get("depth", 2)),
        embedding_dim=int(saved.get("embedding_dim", 64)),
    ).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    temperature = torch.as_tensor(saved["temperature"], device=device).float()
    return saved, model, temperature


def _grid_features(
    base: torch.Tensor,
    active: torch.Tensor,
    *,
    horizon: int,
    dt: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    times = torch.arange(1, int(horizon) + 1, device=base.device).float() * float(dt)
    left = times if bool(active[0]) else times.new_zeros(1)
    right = times if bool(active[1]) else times.new_zeros(1)
    ll, rr = torch.meshgrid(left, right, indexing="ij")
    pairs = torch.stack((ll.reshape(-1), rr.reshape(-1)), dim=1)
    features = base.expand(len(pairs), -1).clone()
    features[:, -2:] = pairs
    return features, (len(left), len(right))


@torch.no_grad()
def exact_verify_frontiers(
    *,
    rows: list[dict],
    physical: dict,
    pool,
    actor_checkpoint: dict,
    device: torch.device,
    acceptable_limit: float,
    verify_all: bool = False,
) -> dict[str, float]:
    """Attach frozen-actor physical diagnostics to frontier proposals."""
    selected = [
        i for i, row in enumerate(rows)
        if verify_all or row["proposal_kind"] == "frontier"
    ]
    if not selected:
        return {
            "evaluated": 0.0, "exact_feasible": 0.0,
            "within_requested_range": 0.0, "within_repairable_range": 0.0,
        }
    index = torch.as_tensor(selected, device=device, dtype=torch.long)
    proposal = _index_batch(physical, index)
    hit = torch.as_tensor(
        [rows[i]["time_to_hit_knots"] for i in selected],
        device=device, dtype=torch.long,
    )
    retime_batch_timing(proposal, hit, pool)

    actor, _ = load_model(actor_checkpoint, str(device))
    actor.eval()
    cfg = phase2_config_from_checkpoint(actor_checkpoint, device=str(device))
    cfg.compile_model = False
    cfg.compile_cost = False
    cost = build_costs(
        [int(proposal["bucket"])], cfg.cost, device=device,
    )[int(proposal["bucket"])]
    proposal["init"] = correct_quintic_init_arms(
        cost.fk, proposal["init"], proposal["q0"], proposal["hit"],
        proposal["context"].goal_pos, active=proposal["active"], steps=1,
        q_bounds=(pool.q_lo, pool.q_hi), joint_names=cost.joint_names,
        qd0=proposal["v0"], dt=float(pool.cfg.dt),
    )
    encoding = actor.encode_context(proposal["context"])
    hit_eff = effective_hits(
        proposal["hit"], proposal["active"], int(proposal["bucket"]),
    )
    strike = replace(proposal["strike"], hit_index=hit_eff)
    decoded = actor(
        proposal["init"], proposal["state"], encoding, proposal["masks"],
    )
    ct = cost.forward(
        proposal["q0"], decoded.traj, proposal["v0"], strike,
        lengths=group_lengths(hit_eff), weights=None,
    )
    residual = residuals_from_gate(_gate_from_ct(ct), cfg.timing_critic)
    normalized = robust_transform(residual, cfg.timing_critic.scales)
    worst = normalized.amax(dim=1)
    exact = worst <= 0.0
    close = worst <= float(acceptable_limit)
    repairable = worst <= 0.5
    diagnostics = {
        name: ct.diagnostics[name].detach().float().cpu()
        for name in (
            "terminal_region_dist", "collision_min_sdf",
            "joint_box_excess_max", "torque_excess_max",
        )
    }
    for local, original in enumerate(selected):
        rows[original]["exact"] = {
            "worst_normalized_residual": float(worst[local]),
            "feasible": bool(exact[local]),
            "within_requested_range": bool(close[local]),
            "within_repairable_range": bool(repairable[local]),
            "normalized_residuals": normalized[local].cpu().tolist(),
            **{name: float(value[local]) for name, value in diagnostics.items()},
        }
    n = len(selected)
    return {
        "evaluated": float(n),
        "exact_feasible": float(exact.sum()),
        "exact_feasible_rate": float(exact.float().mean()),
        "within_requested_range": float(close.sum()),
        "within_requested_range_rate": float(close.float().mean()),
        "within_repairable_range": float(repairable.sum()),
        "within_repairable_range_rate": float(repairable.float().mean()),
        "mean_worst_normalized_residual": float(worst.mean()),
        "median_worst_normalized_residual": float(worst.median()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum_safe_precision <= 1.0:
        raise ValueError("--minimum-safe-precision must be in (0, 1]")
    if not 0.0 < args.uncertainty_quantile < 1.0:
        raise ValueError("--uncertainty-quantile must be in (0, 1)")
    device = torch.device(args.device)
    saved, model, temperature = _load_critic(Path(args.critic), device)

    dataset = torch.load(
        saved["dataset_checkpoint"], map_location="cpu", weights_only=False,
    )
    recent = dataset["extra"]["timing_critic"]["recent"]
    _, (cal_x, cal_y), _ = _split_by_time(recent)
    cal_x = cal_x.to(device)
    cal_pred = prediction(model, cal_x, temperature)
    cal_y = cal_y.to(device)
    cal_truth = distributional_targets(cal_y)[:, -1] < MANAGEABLE_BINS
    cal_close_truth = cal_y.amax(dim=1) <= float(
        args.acceptable_normalized_violation
    )

    # Requiring every independently initialized head to agree is the
    # conservative decision statistic.  Entropy + JS disagreement is used as
    # a separate familiarity test, standardized only on calibration data.
    cal_conservative = cal_pred["member_safe"].amin(dim=0)
    legacy_calibration = calibrate_safe_threshold(
        cal_conservative, cal_truth,
        minimum_precision=float(args.minimum_safe_precision),
    )
    cal_risk, cal_risk_parts = distributional_risk(
        cal_pred["member_probability"],
        acceptable_limit=float(args.acceptable_normalized_violation),
    )
    risk_calibration = calibrate_risk_threshold(
        cal_risk, cal_close_truth,
        minimum_precision=float(args.minimum_safe_precision),
    )
    calibration = (
        risk_calibration if args.decision_rule == "risk" else legacy_calibration
    )
    entropy_mean = cal_pred["entropy"].mean()
    entropy_std = cal_pred["entropy"].std(unbiased=False).clamp_min(1e-6)
    disagreement_mean = cal_pred["disagreement"].mean()
    disagreement_std = cal_pred["disagreement"].std(unbiased=False).clamp_min(1e-6)
    cal_unknown = (
        (cal_pred["entropy"] - entropy_mean) / entropy_std
        + (cal_pred["disagreement"] - disagreement_mean) / disagreement_std
    )
    unknown_limit = torch.quantile(cal_unknown, float(args.uncertainty_quantile))

    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    if args.task_source == "global":
        sampler = GlobalTaskSampler(max_tth_s=float(args.horizon) * float(args.dt))
        tasks = sampler.sample(int(args.tasks), generator=generator).to(device)
    else:
        if args.pool_cache is None:
            raise ValueError("--task-source pool requires --pool-cache")
        source_checkpoint = torch.load(
            saved["dataset_checkpoint"], map_location="cpu", weights_only=False,
        )
        scenario = scenario_from_checkpoint(source_checkpoint)
        pool = load_pool(Path(args.pool_cache), scenario, str(device))
        if int(args.horizon) not in pool.buckets:
            raise ValueError(
                f"horizon {args.horizon} is not in pool buckets {pool.buckets}"
            )
        idx = pool.sample_idx(int(args.horizon), int(args.tasks), generator=generator)
        physical = pool.batch(int(args.horizon), idx, generator=generator, noise=False)
        tasks = build_critic_features(
            q0=physical["q0"], qd0=physical["v0"],
            goal_pos=physical["context"].goal_pos,
            goal_tth=physical["context"].goal_tth,
        ).to(device)
    randomizer = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
    rows = []
    for task_i, base in enumerate(tasks):
        active = base[-2:] > 0.0
        grid, shape = _grid_features(
            base, active, horizon=int(args.horizon), dt=float(args.dt),
        )
        pred = prediction(model, grid, temperature)
        risk, risk_parts = distributional_risk(
            pred["member_probability"],
            acceptable_limit=float(args.acceptable_normalized_violation),
        )
        conservative = pred["member_safe"].amin(dim=0)
        unknown_score = (
            (pred["entropy"] - entropy_mean) / entropy_std
            + (pred["disagreement"] - disagreement_mean) / disagreement_std
        )
        trusted = unknown_score <= unknown_limit
        if args.decision_rule == "risk":
            manageable = (
                risk <= float(risk_calibration["threshold"])
            ).reshape(shape)
        else:
            manageable = (
                (conservative >= float(legacy_calibration["threshold"])) & trusted
            ).reshape(shape)
        component = anchor_connected_component(
            manageable, (shape[0] - 1, shape[1] - 1),
        )
        frontier = pareto_frontier_mask(component)
        points = frontier.nonzero(as_tuple=False)
        fallback = None
        if not bool(manageable[-1, -1]):
            fallback = "untrusted_or_failed_anchor"
        elif not int(points.shape[0]):
            fallback = "no_anchor_connected_frontier"

        if fallback is None:
            choice = int(torch.randint(
                0, len(points), (1,), generator=randomizer,
            ))
            ij = points[choice]
            if int(args.frontier_backoff_knots) > 0:
                ij = torch.minimum(
                    ij + int(args.frontier_backoff_knots),
                    torch.tensor(
                        [shape[0] - 1, shape[1] - 1], device=ij.device,
                    ),
                )
        else:
            ij = torch.tensor(
                [shape[0] - 1, shape[1] - 1], device=device,
            )
        flat = int(ij[0]) * shape[1] + int(ij[1])
        chosen = grid[flat]
        q_width = (int(saved["input_dim"]) - 8) // 2
        rows.append({
            "task": task_i,
            "q0": [float(x * math.pi) for x in chosen[:q_width].cpu()],
            "qd0": [float(x * 10.0) for x in chosen[q_width:2*q_width].cpu()],
            "goal_pos": chosen[2*q_width:2*q_width+6].reshape(2, 3).cpu().tolist(),
            "active": [bool(x) for x in active.cpu()],
            "time_to_hit": [float(x) for x in chosen[-2:].cpu()],
            "time_to_hit_knots": [
                int(round(float(x) / float(args.dt))) if bool(a) else -1
                for x, a in zip(chosen[-2:].cpu(), active.cpu())
            ],
            "proposal_kind": "frontier" if fallback is None else "anchor_probe",
            "fallback_reason": fallback,
            "frontier_points": int(points.shape[0]),
            "conservative_safe_probability": float(conservative[flat]),
            "mean_safe_probability": float(pred["safe"][flat]),
            "entropy": float(pred["entropy"][flat]),
            "disagreement": float(pred["disagreement"][flat]),
            "unknown_score": float(unknown_score[flat]),
            "trusted": bool(trusted[flat]),
            "distributional_risk": float(risk[flat]),
            "risk_expected": float(risk_parts["expected"][flat]),
            "risk_cvar": float(risk_parts["cvar"][flat]),
            "risk_epistemic": float(risk_parts["epistemic"][flat]),
        })

    frontier_count = sum(row["proposal_kind"] == "frontier" for row in rows)
    exact_summary = None
    if args.exact_verify or args.exact_verify_all:
        if args.task_source != "pool":
            raise ValueError("--exact-verify currently requires --task-source pool")
        exact_summary = exact_verify_frontiers(
            rows=rows, physical=physical, pool=pool,
            actor_checkpoint=dataset, device=device,
            acceptable_limit=float(args.acceptable_normalized_violation),
            verify_all=bool(args.exact_verify_all),
        )
    report = {
        "critic": str(args.critic),
        "tasks": int(args.tasks),
        "seed": int(args.seed),
        "task_source": args.task_source,
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "decision_rule": args.decision_rule,
        "acceptable_normalized_violation": float(
            args.acceptable_normalized_violation
        ),
        "frontier_backoff_knots": int(args.frontier_backoff_knots),
        "calibration": {
            **calibration,
            "minimum_precision": float(args.minimum_safe_precision),
            "uncertainty_quantile": float(args.uncertainty_quantile),
            "unknown_limit": float(unknown_limit),
        },
        "risk_calibration": risk_calibration,
        "legacy_calibration": legacy_calibration,
        "summary": {
            "frontier_proposals": frontier_count,
            "anchor_probes": len(rows) - frontier_count,
            "trusted_selected": sum(row["trusted"] for row in rows),
        },
        "exact_verification": exact_summary,
        "admission": "requires_exact_actor_physics_verification",
        "collocations": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("TIMING_FRONTIER_SEARCH=" + json.dumps(report["summary"], sort_keys=True))
    print("TIMING_FRONTIER_CALIBRATION=" + json.dumps(report["calibration"], sort_keys=True))


if __name__ == "__main__":
    main()
