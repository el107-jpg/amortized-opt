#!/usr/bin/env python
"""Compare phase-2 checkpoints around a loss spike on one fixed pool batch.

This is a read-only, one-off diagnostic.  It separates four signals:

* model parameter movement, overall and by module;
* decoder trajectory movement on identical, noise-free pool rows;
* each trajectory's cost under its checkpoint's own objective and under one
  frozen common objective (cost config, schedule point, AL/PID/PHR state);
* Adam moment jumps and alignment of the observed parameter delta with the
  bias-corrected effective update encoded in either checkpoint.

Example:

    python scripts/diagnose_loss_spike.py \\
      --checkpoints runs/run/step_041900.pt runs/run/step_042000.pt \\
      --pool-cache runs/p2pool_rhythm_bellman_nf1_ls_mix.npz \\
      --bucket 100 --batch-size 64 --device cuda:0 \\
      --common-objective 0 --json-out runs/run/spike_41900_42000.json

``--common-objective`` is a checkpoint index (default 0) or one of the paths
listed in ``--checkpoints``.  Pool noise and dropout are disabled.  The same
stored rows, initialization, masks, and context are used for every model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import amortized_opt as A  # noqa: E402
from train_phase2 import _load_npz_buckets  # noqa: E402
from amortized_opt.checkpoint_diagnostics import (  # noqa: E402
    AdamAnalysis,
    adjacent_adam_report,
    analyze_adam_state,
    tensor_mapping_delta_report,
)
from amortized_opt.fk import StickRandomization  # noqa: E402
from amortized_opt.idle_cylinder import derive_cylinder  # noqa: E402
from amortized_opt.lagrangian import AugmentedLagrangian  # noqa: E402
from amortized_opt.phase2_data import (  # noqa: E402
    Phase2Noise,
    Phase2Pool,
    Phase2ScenarioConfig,
)
from amortized_opt.rhythm import KitRandomization  # noqa: E402
from amortized_opt.train_cost import (  # noqa: E402
    _dataclass_from_mapping,
    _wtensors,
    build_costs,
    effective_hits,
    group_lengths,
    migrate_al_state,
    phase2_config_from_checkpoint,
    resolve_idle_cylinder_library,
)
from amortized_opt.waist_aim import WaistAimConfig  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="two or more phase-2 checkpoints in chronological order",
    )
    p.add_argument("--pool-cache", required=True, help="fixed phase-2 pool npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--bucket", type=int, default=None, help="horizon bucket (default: largest)"
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="fixed row-selection seed")
    p.add_argument(
        "--common-objective",
        default="0",
        help="checkpoint index/path whose entire objective is frozen (default: 0)",
    )
    p.add_argument(
        "--module-depth",
        type=int,
        default=2,
        help="parameter path components in module aggregates (default: 2)",
    )
    p.add_argument(
        "--stage-a-closed",
        action="store_true",
        help="evaluate schedule with the legacy stage-A gate closed (default: open)",
    )
    p.add_argument("--json-out", default=None, help="write complete JSON report")
    args = p.parse_args(argv)
    if len(args.checkpoints) < 2:
        p.error("--checkpoints needs at least two paths")
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.module_depth < 1:
        p.error("--module-depth must be positive")
    return args


def _scenario_from_checkpoint(
    ck: dict,
    buckets: tuple[int, ...],
    arrays: Mapping[str, np.ndarray],
) -> Phase2ScenarioConfig:
    """Rebuild nested scenario config and bind it to the actual cache shapes."""
    raw = dict(ck.get("scenario_cfg") or {})
    nested = (
        ("kit_randomization", KitRandomization),
        ("waist_cfg", WaistAimConfig),
        ("stick_rand", StickRandomization),
    )
    for key, cls in nested:
        if isinstance(raw.get(key), dict):
            raw[key] = _dataclass_from_mapping(cls, raw[key])
    valid = {f.name for f in fields(Phase2ScenarioConfig)}
    raw = {k: v for k, v in raw.items() if k in valid}
    raw["buckets"] = tuple(int(x) for x in buckets)
    raw["horizon"] = max(buckets)
    if "joint_names" in raw:
        raw["joint_names"] = tuple(raw["joint_names"])
    drum = arrays.get("drum_pos")
    future = arrays.get("future_pos")
    if drum is not None:
        raw["num_drums"] = int(drum.shape[1])
    if future is not None:
        raw["num_future"] = int(future.shape[2])
    return Phase2ScenarioConfig(**raw)


def _resolve_common(spec: str, paths: list[Path]) -> int:
    try:
        idx = int(spec)
    except ValueError:
        target = Path(spec).expanduser().resolve()
        matches = [i for i, path in enumerate(paths) if path.resolve() == target]
        if not matches:
            raise ValueError(
                "--common-objective path must also appear in --checkpoints"
            )
        return matches[0]
    if idx < 0:
        idx += len(paths)
    if not 0 <= idx < len(paths):
        raise ValueError(
            f"--common-objective index {spec} outside [0, {len(paths) - 1}]"
        )
    return idx


def _fixed_indices(size: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.choice(int(size), size=int(count), replace=count > size).astype(np.int64)


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        x = value.detach().cpu().contiguous().numpy()
        digest.update(str(x.dtype).encode())
        digest.update(np.asarray(x.shape, dtype=np.int64).tobytes())
        digest.update(x.tobytes())
    return digest.hexdigest()


def _number(value):
    if torch.is_tensor(value):
        x = value.detach().cpu()
        return float(x.item()) if x.numel() == 1 else x.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_stats(value: torch.Tensor) -> dict[str, float | int | None]:
    x = value.detach().float().reshape(-1)
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {
            "count": int(x.numel()),
            "nonfinite": int(x.numel()),
            "mean": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(x.numel()),
        "nonfinite": int(x.numel() - finite.numel()),
        "mean": float(finite.mean().item()),
        "p50": float(torch.quantile(finite, 0.5).item()),
        "p90": float(torch.quantile(finite, 0.9).item()),
        "max": float(finite.max().item()),
    }


def _cost_report(ct) -> dict:
    return {
        "total": _safe_stats(ct.total),
        "terms": {k: _safe_stats(v) for k, v in sorted(ct.terms.items())},
        "violations": {
            k: _safe_stats(v) for k, v in sorted(ct.violations.items())
        },
    }


def _trajectory_delta(
    before: torch.Tensor,
    after: torch.Tensor,
    mask: torch.Tensor,
    joint_names: tuple[str, ...],
) -> dict:
    d = (after.detach().float() - before.detach().float()) * mask.detach().float()
    valid = mask.detach().float()
    n = valid.sum().clamp_min(1.0)
    report = {
        "l2": float(d.square().sum().sqrt().item()),
        "rms": float((d.square().sum() / n).sqrt().item()),
        "mean_abs": float(d.abs().sum().div(n).item()),
        "max_abs": float(d.abs().max().item()),
        "valid_elements": int(valid.sum().item()),
        "by_joint_rms": {},
    }
    denom = valid.sum(dim=(0, 1)).clamp_min(1.0)
    rms = (d.square().sum(dim=(0, 1)) / denom).sqrt()
    report["by_joint_rms"] = {
        name: float(rms[i].item()) for i, name in enumerate(joint_names)
    }
    return report


def _load_al(cfg, ck: dict) -> AugmentedLagrangian:
    al = AugmentedLagrangian(cfg.al)
    state = migrate_al_state(ck.get("al"))
    if state:
        al.load_state_dict(state)
    return al


@dataclass
class _Objective:
    cfg: object
    cost: object
    al: AugmentedLagrangian
    weights: dict
    metadata: dict


def _install_idle_cylinder(cost, cfg, pool, kit) -> None:
    idle_lib = resolve_idle_cylinder_library(
        cfg.resolved_idle_cylinder_path(), joint_names=pool.cfg.joint_names
    )
    if idle_lib is None:
        return
    try:
        cylinder = derive_cylinder(kit, cost.fk)
    except Exception:
        cylinder = idle_lib.cylinder
    cost.set_idle_cylinder(cylinder)


def _make_objective(
    ck: dict,
    *,
    bucket: int,
    device,
    pool,
    kit,
    stage_a: bool,
) -> _Objective:
    cfg = phase2_config_from_checkpoint(
        ck, device=str(device), compile_model=False
    )
    cfg.compile_mode = None
    cfg.compile_cost = False
    cfg.compile_aux = False
    step = int(ck.get("step") or 0)
    al = _load_al(cfg, ck)
    cost = build_costs(
        (bucket,), cfg.cost, device=device, dtype=torch.float32
    )[bucket]
    _install_idle_cylinder(cost, cfg, pool, kit)
    schedule = cfg.schedule.multipliers(step, stage_a)
    weights = _wtensors(
        schedule,
        al.weights(),
        device,
        torch.float32,
        sched=cfg.schedule,
        joint_names=cost.joint_names,
        step=step,
        stage_a=stage_a,
        pen=al.pen,
    )
    metadata = {
        "checkpoint_step": step,
        "al_state_present": ck.get("al") is not None,
        "al_integrator": dict(sorted(al.lam.items())),
        "al_used": dict(sorted(al.weights().items())),
        "phr_penalty": dict(sorted(al.pen.items())),
        "schedule_multipliers": {
            k: float(v) for k, v in sorted(schedule.items())
        },
        "effective_cost_weights": {
            k: _number(v) for k, v in sorted(weights.items())
        },
        "cost_config": asdict(cfg.cost),
        "stage_a_open": bool(stage_a),
    }
    return _Objective(cfg, cost, al, weights, metadata)


def _head_references(cfg, enc, horizon: int) -> dict[str, torch.Tensor]:
    refs: dict[str, torch.Tensor] = {}
    if cfg.waist_sp_from_head:
        if enc.waist_yaw_sp is None:
            raise RuntimeError(
                "checkpoint objective requests waist_sp_from_head but model has no head"
            )
        refs["waist_yaw_ref"] = enc.waist_yaw_sp.float().unsqueeze(1).expand(
            -1, horizon
        )
    if cfg.waist_pr_sp_from_head:
        if enc.waist_pr_sp is None:
            raise RuntimeError(
                "checkpoint objective requests waist_pr_sp_from_head but model has no head"
            )
        refs["waist_pr_ref"] = enc.waist_pr_sp.float().unsqueeze(1).expand(
            -1, horizon, -1
        )
    return refs


def _strike_with_refs(strike, refs: Mapping[str, torch.Tensor]):
    return replace(strike, **dict(refs)) if refs else strike


def _model_parameter_order(model) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def _label(path: Path, ck: dict, index: int) -> str:
    return f"{index}:{path.name}@{int(ck.get('step') or 0)}"


def _mean_map(section: dict) -> dict[str, float]:
    return {
        key: float(stats["mean"])
        for key, stats in section.items()
        if stats.get("mean") is not None
    }


def _delta_map(before: dict, after: dict) -> dict[str, float]:
    keys = set(before) | set(after)
    return {
        key: float(after.get(key, 0.0) - before.get(key, 0.0))
        for key in sorted(keys)
    }


def _print_values(title: str, values: dict[str, float], *, nonzero_only=False) -> None:
    items = [
        (key, value)
        for key, value in values.items()
        if not nonzero_only or abs(value) > 0.0
    ]
    text = " ".join(f"{key}={value:.5g}" for key, value in items)
    print(f"    {title}: {text or '(all zero)'}")


def _print_human(report: dict) -> None:
    batch = report["batch"]
    common = report["common_objective"]
    print(
        f"[batch] pool={batch['pool_cache']} bucket={batch['bucket']} "
        f"B={batch['batch_size']} seed={batch['seed']} "
        f"digest={batch['tensor_sha256'][:12]}"
    )
    print(
        f"[common objective] {common['label']} "
        f"(cost config + schedule + AL/PID/PHR frozen)"
    )
    for ck in report["checkpoints"]:
        own = ck["fixed_batch"]["own_objective"]
        frozen = ck["fixed_batch"]["common_objective"]
        print(
            f"\n[{ck['label']}] own={own['total']['mean']:.6g} "
            f"common={frozen['total']['mean']:.6g} "
            f"repricing={ck['fixed_batch']['own_minus_common_total_mean']:+.6g}"
        )
        _print_values("own terms", _mean_map(own["terms"]), nonzero_only=True)
        _print_values(
            "own violations", _mean_map(own["violations"]), nonzero_only=True
        )
        _print_values("common terms", _mean_map(frozen["terms"]), nonzero_only=True)
        _print_values(
            "common violations",
            _mean_map(frozen["violations"]),
            nonzero_only=True,
        )
        adam = ck["optimizer_adam"]
        if adam["available"]:
            moment = adam["exp_avg"]["overall"]
            eff = adam["effective_direction"]["overall"]
            update = adam["proposed_update"]["overall"]
            print(
                f"    Adam: coverage={adam['coverage']:.1%} "
                f"steps={adam['step']['min']:g}..{adam['step']['max']:g} "
                f"||m||={moment['l2']:.6g} ||d_eff||={eff['l2']:.6g} "
                f"||u_saved||={update['l2']:.6g} "
                f"v_zero={adam['second_moment']['zero_fraction']:.2%}"
            )
        else:
            print("    Adam: unavailable")

    for pair in report["pairs"]:
        pd = pair["parameter_delta"]["overall"]
        td = pair["trajectory_delta"]
        print(f"\n[pair] {pair['before']} -> {pair['after']}")
        print(
            f"    params: ||Δθ||={pd['delta_l2']:.6g} "
            f"rms={pd['delta_rms']:.6g} rel={pd['relative_delta_l2']:.3g}"
        )
        modules = sorted(
            pair["parameter_delta"]["by_module"].items(),
            key=lambda item: item[1]["delta_l2"],
            reverse=True,
        )
        _print_values(
            "module ||Δθ||",
            {name: values["delta_l2"] for name, values in modules},
            nonzero_only=True,
        )
        print(
            f"    decoder traj: rms={td['rms']:.6g} "
            f"max={td['max_abs']:.6g}"
        )
        print(
            f"    frozen-common total Δ="
            f"{pair['common_objective_total_mean_delta']:+.6g}"
        )
        _print_values(
            "frozen-common term Δ",
            pair["common_objective_term_mean_delta"],
            nonzero_only=True,
        )
        _print_values(
            "frozen-common violation Δ",
            pair["common_objective_violation_mean_delta"],
            nonzero_only=True,
        )
        adam_pair = pair["optimizer_adam"]
        dm = adam_pair["exp_avg_delta"]["overall"]
        dv = adam_pair["exp_avg_sq_delta"]["overall"]
        before_align = adam_pair["before_effective_update_alignment"]["overall"]
        after_align = adam_pair["after_effective_update_alignment"]["overall"]
        print(
            f"    Adam moments: ||Δm||={dm['delta_l2']:.6g} "
            f"rel={dm['relative_delta_l2']:.3g} "
            f"||Δv||={dv['delta_l2']:.6g} rel={dv['relative_delta_l2']:.3g}"
        )
        print(
            "    Adam effective-update cosine with observed Δθ: "
            f"before={before_align['cosine']} after={after_align['cosine']}"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = [Path(p).expanduser() for p in args.checkpoints]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"checkpoint(s) not found: {missing}")
    pool_path = Path(args.pool_cache).expanduser()
    if not pool_path.is_file():
        raise SystemExit(f"pool cache not found: {pool_path}")
    try:
        common_idx = _resolve_common(args.common_objective, paths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device.index or 0)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    for path, ck in zip(paths, checkpoints):
        if not isinstance(ck.get("model"), Mapping):
            raise SystemExit(f"{path} is not a phase-2 model checkpoint")

    buckets, stats = _load_npz_buckets(pool_path)
    bucket_keys = tuple(sorted(int(x) for x in buckets))
    bucket = int(args.bucket if args.bucket is not None else max(bucket_keys))
    if bucket not in buckets:
        raise SystemExit(f"bucket {bucket} not in pool {bucket_keys}")

    common_ck = checkpoints[common_idx]
    scenario = _scenario_from_checkpoint(
        common_ck, bucket_keys, buckets[bucket]
    )
    pool = Phase2Pool(
        buckets,
        cfg=scenario,
        noise=Phase2Noise(),
        device=device,
        dtype=torch.float32,
        stats=stats,
    )
    common_train_cfg = phase2_config_from_checkpoint(
        common_ck, device=str(device), compile_model=False
    )
    # StrikeContext carries one randomized-stick point per shaft sample.  Bind
    # the pool materialization to the checkpoint objective before making the
    # fixed batch (the current recipe uses the default, but older configs need
    # not have done so).
    pool.n_shaft = int(common_train_cfg.cost.shaft_samples)
    indices_np = _fixed_indices(pool.size(bucket), args.batch_size, args.seed)
    indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
    batch = pool.batch(bucket, indices, noise=False)
    hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
    lengths = group_lengths(hit_eff)
    base_strike = replace(batch["strike"], hit_index=hit_eff)
    batch_digest = _tensor_digest(batch["q0"], batch["v0"], batch["init"])

    model_cfgs = [A.model_config_from_checkpoint(ck) for ck in checkpoints]
    for path, model_cfg in zip(paths, model_cfgs):
        if int(model_cfg.max_horizon) < bucket:
            raise SystemExit(
                f"{path}: model max_horizon={model_cfg.max_horizon} < bucket={bucket}"
            )
        if int(model_cfg.num_drums) != scenario.num_drums:
            raise SystemExit(
                f"{path}: model num_drums={model_cfg.num_drums} != "
                f"pool {scenario.num_drums}"
            )
        if int(model_cfg.num_future) != scenario.num_future:
            raise SystemExit(
                f"{path}: model num_future={model_cfg.num_future} != "
                f"pool {scenario.num_future}"
            )

    stage_a = not bool(args.stage_a_closed)
    kit = A.DrumKit.from_layout()
    objectives: list[_Objective | None] = [None] * len(checkpoints)
    common_objective = _make_objective(
        common_ck,
        bucket=bucket,
        device=device,
        pool=pool,
        kit=kit,
        stage_a=stage_a,
    )
    objectives[common_idx] = common_objective
    common_shaft = int(common_objective.cfg.cost.shaft_samples)
    for i, ck in enumerate(checkpoints):
        cfg = phase2_config_from_checkpoint(
            ck, device=str(device), compile_model=False
        )
        if int(cfg.cost.shaft_samples) != common_shaft:
            raise SystemExit(
                "checkpoints use different cost.shaft_samples; one fixed "
                "StrikeContext cannot represent both objectives"
            )

    trajectories: list[torch.Tensor | None] = [None] * len(checkpoints)
    parameter_orders: list[list[str] | None] = [None] * len(checkpoints)
    adam: list[AdamAnalysis | None] = [None] * len(checkpoints)
    checkpoint_reports: list[dict | None] = [None] * len(checkpoints)
    common_refs: dict[str, torch.Tensor] | None = None

    # Evaluate the common checkpoint first so head-owned reference trajectories,
    # if present in a legacy run, are also frozen with the common objective.
    eval_order = [common_idx] + [i for i in range(len(checkpoints)) if i != common_idx]
    for i in eval_order:
        ck, path, model_cfg = checkpoints[i], paths[i], model_cfgs[i]
        model = A.build_model(model_cfg, device)
        model.load_state_dict(ck["model"])
        model.eval()
        order = _model_parameter_order(model)
        parameter_orders[i] = order
        parameter_map = {
            name: ck["model"][name] for name in order if name in ck["model"]
        }
        adam_i = analyze_adam_state(
            ck.get("optimizer"),
            order,
            parameter_map,
            module_depth=args.module_depth,
        )
        adam[i] = adam_i

        own_objective = objectives[i]
        if own_objective is None:
            own_objective = _make_objective(
                ck,
                bucket=bucket,
                device=device,
                pool=pool,
                kit=kit,
                stage_a=stage_a,
            )
            objectives[i] = own_objective

        with torch.inference_mode():
            enc = model.encode_context(batch["context"])
            out = model(batch["init"], batch["state"], enc, batch["masks"])
            traj = out.traj.float()
            own_refs = _head_references(own_objective.cfg, enc, bucket)
            if i == common_idx:
                common_refs = {
                    key: value.detach().clone() for key, value in own_refs.items()
                }
            assert common_refs is not None
            own_ct = own_objective.cost.forward(
                batch["q0"],
                traj,
                batch["v0"],
                _strike_with_refs(base_strike, own_refs),
                lengths=lengths,
                elem_mask=batch["masks"].elem_mask,
                weights=own_objective.weights,
            )
            common_ct = common_objective.cost.forward(
                batch["q0"],
                traj,
                batch["v0"],
                _strike_with_refs(base_strike, common_refs),
                lengths=lengths,
                elem_mask=batch["masks"].elem_mask,
                weights=common_objective.weights,
            )
            own_report = _cost_report(own_ct)
            frozen_report = _cost_report(common_ct)
            trajectories[i] = traj.detach().cpu()

        own_mean = own_report["total"]["mean"]
        common_mean = frozen_report["total"]["mean"]
        checkpoint_reports[i] = {
            "label": _label(path, ck, i),
            "path": str(path),
            "step": int(ck.get("step") or 0),
            "model_config": asdict(model_cfg),
            "own_objective": own_objective.metadata,
            "fixed_batch": {
                "own_objective": own_report,
                "common_objective": frozen_report,
                "own_minus_common_total_mean": (
                    None
                    if own_mean is None or common_mean is None
                    else float(own_mean - common_mean)
                ),
            },
            "optimizer_adam": adam_i.report,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    parameter_maps = [
        {
            name: ck["model"][name]
            for name in order
            if name in ck["model"]
        }
        for ck, order in zip(checkpoints, parameter_orders)
    ]
    mask_cpu = batch["masks"].elem_mask.detach().cpu()
    joint_names = tuple(pool.cfg.joint_names)
    pairs: list[dict] = []
    for i in range(len(checkpoints) - 1):
        j = i + 1
        param_delta = tensor_mapping_delta_report(
            parameter_maps[i],
            parameter_maps[j],
            module_depth=args.module_depth,
        )
        traj_delta = _trajectory_delta(
            trajectories[i], trajectories[j], mask_cpu, joint_names
        )
        before_common = checkpoint_reports[i]["fixed_batch"]["common_objective"]
        after_common = checkpoint_reports[j]["fixed_batch"]["common_objective"]
        before_total = before_common["total"]["mean"]
        after_total = after_common["total"]["mean"]
        adam_pair = adjacent_adam_report(
            parameter_maps[i],
            parameter_maps[j],
            adam[i],
            adam[j],
            module_depth=args.module_depth,
        )
        pairs.append(
            {
                "before": checkpoint_reports[i]["label"],
                "after": checkpoint_reports[j]["label"],
                "step_delta": int(checkpoint_reports[j]["step"])
                - int(checkpoint_reports[i]["step"]),
                "parameter_delta": param_delta,
                "trajectory_delta": traj_delta,
                "common_objective_total_mean_delta": (
                    None
                    if before_total is None or after_total is None
                    else float(after_total - before_total)
                ),
                "common_objective_term_mean_delta": _delta_map(
                    _mean_map(before_common["terms"]),
                    _mean_map(after_common["terms"]),
                ),
                "common_objective_violation_mean_delta": _delta_map(
                    _mean_map(before_common["violations"]),
                    _mean_map(after_common["violations"]),
                ),
                "optimizer_adam": adam_pair,
            }
        )

    report = {
        "schema_version": 1,
        "diagnostic": "phase2_loss_spike_checkpoint_comparison",
        "batch": {
            "pool_cache": str(pool_path),
            "bucket": bucket,
            "batch_size": int(args.batch_size),
            "pool_bucket_size": pool.size(bucket),
            "seed": int(args.seed),
            "noise": False,
            "indices": indices_np.tolist(),
            "tensor_sha256": batch_digest,
        },
        "common_objective": {
            "checkpoint_index": common_idx,
            "label": checkpoint_reports[common_idx]["label"],
            **common_objective.metadata,
        },
        "checkpoints": checkpoint_reports,
        "pairs": pairs,
    }
    _print_human(report)
    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"\n[json] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
