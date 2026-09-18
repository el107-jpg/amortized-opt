#!/usr/bin/env python
"""Probe whether shifted MPC replay plans disagree with their plant boundary.

This is a diagnostic counterfactual, not a training or deployment change.  It
primes the real ``MPCRoller`` and replay ring, samples shifted rows, and compares
three plans against the same measured ``q0/qd0`` and strike context:

* ``shifted_init``: the harvested, shifted parent plan;
* ``current_decode``: the checkpoint's one-pass decoder output;
* ``decode_c1_bridge``: that output with a short C1 boundary bridge.

Example:

    python scripts/probe_replay_boundary.py \\
      --ckpt runs/rhythm_bellman_nf1_ls_full_rnea_fresh/best.pt \\
      --pool-cache runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \\
      --device cuda:0 --prime-roller-ticks 40 --rows 64 \\
      --json-out runs/rhythm_bellman_nf1_ls_full_rnea_fresh/replay_boundary_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from train_phase2 import _load_npz_buckets
from amortized_opt.idle_cylinder import derive_cylinder
from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.limits import available_torque_torch
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool, Phase2ScenarioConfig
from amortized_opt.replay_boundary_diag import (
    boundary_kinematics,
    c1_bridge_waypoints,
    finite_stats,
    gradient_cosine,
    localize_torque_excess,
    make_c1_roller_plan_transform,
    pearsonr_finite,
)
from amortized_opt.rhythm import KitRandomization
from amortized_opt.train_cost import (
    MPCRoller,
    ReplayPool,
    _dataclass_from_mapping,
    _harvest_replay_viol_score,
    _wtensors,
    build_costs,
    effective_hits,
    group_lengths,
    mpc_init_tie_loss,
    migrate_al_state,
    phase2_config_from_checkpoint,
    resolve_idle_cylinder_library,
    seed_waist_pr_to_sp,
)
from amortized_opt.waist_aim import WaistAimConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool-cache", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bucket", type=int, default=None,
                   help="replay horizon bucket (default: largest roller bucket)")
    p.add_argument("--rows", type=int, default=64,
                   help="distinct shifted replay rows to inspect")
    p.add_argument("--prime-roller-ticks", type=int, default=40)
    p.add_argument("--replay-capacity", type=int, default=8192,
                   help="probe-only ring capacity per bucket")
    p.add_argument("--min-roller-age", type=int, default=1,
                   help="exclude unshifted spawn rows (default: age >= 1)")
    p.add_argument("--bridge-knots", type=int, default=3)
    p.add_argument(
        "--roller-output-bridge-knots", type=int, default=0,
        help=(
            "diagnostic-only C1 projection applied to every roller decode "
            "before plant execution and replay shifting (default: 0/off)"
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=None,
                   help="schedule/AL step (default: checkpoint step)")
    p.add_argument("--skip-parameter-gradients", action="store_true",
                   help="still report output-space torque/tie cosine")
    p.add_argument("--json-out", default=None)
    p.add_argument("--quiet", action="store_true",
                   help="write --json-out without echoing the full report")
    return p.parse_args(argv)


def _scenario_from_ckpt(ck: dict) -> Phase2ScenarioConfig:
    raw = dict(ck.get("scenario_cfg") or {})
    if isinstance(raw.get("waist_cfg"), dict):
        raw["waist_cfg"] = _dataclass_from_mapping(WaistAimConfig, raw["waist_cfg"])
    if isinstance(raw.get("kit_randomization"), dict):
        raw["kit_randomization"] = _dataclass_from_mapping(
            KitRandomization, raw["kit_randomization"],
        )
    valid = {f.name for f in fields(Phase2ScenarioConfig)}
    return Phase2ScenarioConfig(**{k: v for k, v in raw.items() if k in valid})


def _load_al(cfg, ck: dict) -> AugmentedLagrangian:
    al = AugmentedLagrangian(cfg.al)
    state = ck.get("al")
    if not state:
        return al
    al.load_state_dict(migrate_al_state(state))
    fresh = AugmentedLagrangian(cfg.al)
    for attr in (
        "rho", "lam_max", "lam_min", "target", "kp", "kd", "pen_min", "pen_max",
    ):
        setattr(al, attr, dict(getattr(fresh, attr)))
    al.cfg = fresh.cfg
    if not state.get("pen"):
        al.pen = dict(fresh.pen)
    for g in al.groups:
        al.lam[g] = float(min(max(al.lam[g], al.lam_min[g]), al.lam_max[g]))
        al.pen[g] = float(min(max(al.pen[g], al.pen_min[g]), al.pen_max[g]))
    return al


def _scalar_float(x: torch.Tensor | float | int, default: float = 0.0) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _stats_by_age(
    age: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    bands = (
        ("age_0", 0, 0),
        ("age_1_25", 1, 25),
        ("age_26_50", 26, 50),
        ("age_51_100", 51, 100),
        ("age_101_150", 101, 150),
        ("age_151_200", 151, 200),
        ("age_201_plus", 201, None),
    )
    out = {}
    for name, lo, hi in bands:
        take = age >= lo
        if hi is not None:
            take &= age <= hi
        out[name] = finite_stats(values[take])
    return out


def _roller_bucket_rows(roller: MPCRoller, bucket: int) -> torch.Tensor:
    """Repeat ``MPCRoller.harvest``'s exact slot selection for metadata."""
    hit = roller.state["hit"]
    k_max = hit.max(dim=1).values
    buckets = tuple(int(b) for b in roller.pool.buckets if int(b) <= roller.H)
    lo = 0 if int(bucket) == min(buckets) else max(b for b in buckets if b < int(bucket))
    return ((k_max > lo) & (k_max <= int(bucket))).nonzero(as_tuple=True)[0]


@torch.no_grad()
def _prime_replay_with_scores(
    roller: MPCRoller,
    replay: ReplayPool,
    pool: Phase2Pool,
    costs: dict[int, Any],
    weights: dict[str, Any],
    n_ticks: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, float]]:
    """Prime replay and retain probe-only roller age/tick ring columns."""
    age_ring = {
        b: torch.zeros(replay.capacity, device=pool.device, dtype=torch.long)
        for b in replay.data
    }
    tick_ring = {
        b: torch.full(
            (replay.capacity,), -1, device=pool.device, dtype=torch.long,
        )
        for b in replay.data
    }
    acc: dict[str, float] = {}
    for tick in range(int(n_ticks)):
        mark = getattr(roller, "_post_hit_fresh", None)
        if mark is not None:
            acc["post_hit_attempted"] = (
                acc.get("post_hit_attempted", 0.0) + float(mark.sum())
            )
        harvested = roller.harvest()
        for b, cols0 in harvested.items():
            if b not in replay.data:
                continue
            slots = _roller_bucket_rows(roller, b)
            if int(slots.numel()) != int(cols0["q0"].shape[0]):
                raise RuntimeError("probe metadata selection drifted from MPCRoller.harvest")
            cols = dict(cols0)
            post_hit = cols.get("_post_hit_fresh")
            if post_hit is not None:
                acc["post_hit_accepted"] = (
                    acc.get("post_hit_accepted", 0.0) + float(post_hit.sum())
                )
            score = _harvest_replay_viol_score(cols, b, pool, costs[b], weights)
            cols["viol_score"] = score

            pos = replay._pos[b]
            ring = (
                torch.arange(score.numel(), device=pool.device, dtype=torch.long) + pos
            ) % replay.capacity
            age_ring[b].index_copy_(0, ring, roller.state["age"].index_select(0, slots))
            tick_ring[b].index_fill_(0, ring, int(tick))
            replay.add(b, cols)
            acc["harvest_viol_score_sum"] = (
                acc.get("harvest_viol_score_sum", 0.0) + float(score.sum())
            )
            acc["harvest_rows"] = acc.get("harvest_rows", 0.0) + float(score.numel())
        for key, value in roller.tick().items():
            acc[key] = acc.get(key, 0.0) + float(value)
    return age_ring, tick_ring, acc


def _knot_dynamics(
    cost,
    plan: torch.Tensor,
    q0: torch.Tensor,
    v0: torch.Tensor,
    lengths: torch.Tensor,
    weights: dict[str, Any],
):
    """Reproduce FusedCost's knot-FD torque and hard-limit construction."""
    sp = cost.spline.build(q0, plan, v0, lengths)
    q, qd, qdd = sp.knot_finite_differences(v0)
    B, K, D = q.shape
    qf, qdf, qddf = (
        x.reshape(B * K, D) for x in (q, qd, qdd)
    )
    cfg = cost.cfg
    blend_default = 1.0 if cfg.torque_physics == "full" else 0.0
    blend = _scalar_float(weights.get("torque_full_blend", blend_default), blend_default)
    if cfg.torque_physics == "full" or blend >= 1.0 - 1e-6:
        tau = cost.rnea.rnea(qf, qdf, qddf)
    else:
        simple = cost._torque_level_a(
            qf, qddf, frozen=(cfg.torque_physics == "diag_grav_frozen"),
        )
        if blend <= 1e-6:
            tau = simple
        else:
            full = cost.rnea.rnea(qf, qdf, qddf)
            tau = blend * full + (1.0 - blend) * simple
    tau = tau.reshape(B, K, D)
    hard = available_torque_torch(
        qd,
        cost.tau_hard.view(1, 1, D),
        cost.v_hard.view(1, 1, D),
        eta=1.0,
        saturation_scale=cfg.torque_saturation_scale,
        derate=cfg.torque_derate,
    ).expand_as(qd)
    knot = torch.arange(K, device=q.device).view(1, K, 1)
    mask = (knot <= sp.lengths.unsqueeze(1)).to(q.dtype)
    return sp, q, qd, qdd, tau, hard, mask


def _variant(
    name: str,
    plan: torch.Tensor,
    *,
    cost,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike,
    lengths: torch.Tensor,
    weights: dict[str, Any],
    segment_indices: dict[str, list[int]],
    ct=None,
) -> dict[str, Any]:
    if ct is None:
        with torch.no_grad():
            ct = cost.forward(q0, plan, v0, strike, lengths=lengths, weights=weights)
    plan_eval = plan.detach()
    sp, _q, qd, qdd, tau, hard, mask = _knot_dynamics(
        cost, plan_eval, q0, v0, lengths, weights,
    )
    kin = boundary_kinematics(q0, v0, plan_eval, dt=float(cost.dt))
    start_acc = 2.0 * sp.coeffs[:, 0, :, 2]
    offsets = torch.linspace(
        0.0, float(cost.dt), 5, device=plan.device, dtype=plan.dtype,
    )
    _qs, qds, qdds, _qddds = sp.eval_offsets(offsets)
    loc = localize_torque_excess(
        tau, hard, mask, segment_indices=segment_indices,
    )
    scalars = {
        **{k: v for k, v in kin.items() if v.ndim == 1},
        "spline_start_velocity_error_abs_max_rads": (
            sp.m[:, 0] - v0
        ).abs().amax(dim=1),
        "spline_start_acceleration_abs_max_rads2": start_acc.abs().amax(dim=1),
        "spline_first_segment_velocity_abs_max_rads": qds[:, 0].abs().amax(dim=(1, 2)),
        "spline_first_segment_acceleration_abs_max_rads2": (
            qdds[:, 0].abs().amax(dim=(1, 2))
        ),
        "knot_fd_first_acceleration_abs_max_rads2": qdd[:, 0].abs().amax(dim=1),
        "torque_peak_excess_nm": loc["row_peak_excess_nm"],
        "torque_peak_abs_nm": loc["row_peak_abs_tau_nm"],
        "hit_distance_m": ct.diagnostics["contact_pos_err"],
        "hit_region_excess_m": ct.diagnostics["terminal_region_dist"],
        "fused_torque_excess_max_nm": ct.diagnostics["torque_excess_max"],
    }
    for window, vals in loc["windows"].items():
        for key, value in vals.items():
            scalars[f"torque_{window}_{key}"] = value
    return {
        "name": name,
        "plan": plan,
        "ct": ct,
        "kinematics": kin,
        "sp": sp,
        "qd": qd,
        "qdd": qdd,
        "tau": tau,
        "hard_limit": hard,
        "mask": mask,
        "localization": loc,
        "scalars": scalars,
    }


def _winner_counts(
    values: torch.Tensor,
    names: list[str],
    *,
    live: torch.Tensor | None = None,
) -> dict[str, int]:
    vals = values.detach().cpu().long()
    if live is not None:
        vals = vals[live.detach().cpu().bool()]
    if vals.numel() == 0:
        return {}
    count = torch.bincount(vals, minlength=len(names))
    return {name: int(count[i]) for i, name in enumerate(names) if int(count[i])}


def _aggregate_variant(
    data: dict[str, Any],
    joint_names: list[str],
    joint_to_segment: list[str],
) -> dict[str, Any]:
    loc = data["localization"]
    peak_live = loc["row_peak_excess_nm"] > 0.0
    out: dict[str, Any] = {
        "scalars": {k: finite_stats(v) for k, v in data["scalars"].items()},
        "torque_violation_rows": {
            "n": int(peak_live.sum()),
            "fraction": float(peak_live.float().mean()),
        },
        "torque_window_violation_rows": {
            name: {
                "n": int((values["peak_excess_nm"] > 0.0).sum()),
                "fraction": float((values["peak_excess_nm"] > 0.0).float().mean()),
            }
            for name, values in loc["windows"].items()
        },
        "torque_by_joint_peak_excess_nm": {
            joint_names[j]: finite_stats(loc["by_joint_peak_excess_nm"][:, j])
            for j in range(len(joint_names))
        },
        "torque_by_segment_peak_excess_nm": {
            name: finite_stats(values)
            for name, values in loc["by_segment_peak_excess_nm"].items()
        },
        "peak_joint_counts": {},
        "peak_segment_counts": {},
        "peak_knot_counts": {},
        "torque_by_knot_peak_excess_nm": {},
    }
    live = peak_live
    out["peak_joint_counts"] = _winner_counts(
        loc["row_peak_joint"], joint_names, live=live,
    )
    seg_names = list(dict.fromkeys(joint_to_segment))
    seg_id = torch.as_tensor(
        [seg_names.index(joint_to_segment[int(j)]) for j in loc["row_peak_joint"]],
        dtype=torch.long,
    )
    out["peak_segment_counts"] = _winner_counts(seg_id, seg_names, live=live)
    knots = loc["row_peak_knot"].detach().cpu().long()
    knots = knots[live.detach().cpu()]
    for k in torch.unique(knots).tolist():
        out["peak_knot_counts"][str(int(k))] = int((knots == int(k)).sum())
    over = loc["excess"]
    for k in range(over.shape[1]):
        out["torque_by_knot_peak_excess_nm"][str(k)] = finite_stats(
            over[:, k].amax(dim=1),
        )
    return out


def _paired_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Report ``b - a`` and the fraction of rows improved (lower is better)."""
    keys = (
        "q0_first_abs_max_rad",
        "implied_acceleration_abs_max_rads2",
        "spline_start_acceleration_abs_max_rads2",
        "knot_fd_first_acceleration_abs_max_rads2",
        "torque_peak_excess_nm",
        "torque_first_1_peak_excess_nm",
        "torque_first_3_peak_excess_nm",
        "torque_first_5_peak_excess_nm",
        "hit_distance_m",
    )
    out = {}
    for key in keys:
        x, y = a["scalars"][key], b["scalars"][key]
        delta = y - x
        out[key] = {
            "delta_b_minus_a": finite_stats(delta),
            "fraction_lower_in_b": float((y < x).float().mean()),
            "fraction_equal": float((y == x).float().mean()),
        }
    return out


def _row_variant(
    data: dict[str, Any],
    row: int,
    joint_names: list[str],
    joint_to_segment: list[str],
) -> dict[str, Any]:
    loc = data["localization"]
    ji = int(loc["row_peak_joint"][row])
    by_joint = loc["by_joint_peak_excess_nm"][row]
    has_excess = bool(loc["row_peak_excess_nm"][row] > 0.0)
    return {
        **{k: float(v[row].detach()) for k, v in data["scalars"].items()},
        "peak_torque_excess_knot": (
            int(loc["row_peak_knot"][row]) if has_excess else None
        ),
        "peak_torque_excess_joint": joint_names[ji] if has_excess else None,
        "peak_torque_excess_segment": joint_to_segment[ji] if has_excess else None,
        "torque_by_joint_peak_excess_nm": {
            joint_names[j]: float(by_joint[j].detach())
            for j in range(len(joint_names))
        },
        "torque_by_segment_peak_excess_nm": {
            name: float(values[row].detach())
            for name, values in loc["by_segment_peak_excess_nm"].items()
        },
    }


def _gradient_report(
    model,
    dec,
    ct,
    batch,
    cfg,
    *,
    skip_parameters: bool,
) -> dict[str, Any]:
    torque_loss = float(cfg.w_cost) * ct.terms["torque"].mean()
    tie_loss_raw, tie_count = mpc_init_tie_loss(
        dec.traj, batch["seed"], batch["masks"].elem_mask, batch["mpc_init"],
    )
    tie_loss = float(cfg.w_mpc_init_tie) * tie_loss_raw
    gt_out = torch.autograd.grad(
        torque_loss, dec.traj, retain_graph=True, allow_unused=False,
    )[0]
    gi_out = torch.autograd.grad(
        tie_loss, dec.traj, retain_graph=True, allow_unused=False,
    )[0]
    out = {
        "weights": {
            "w_cost": float(cfg.w_cost),
            "w_mpc_init_tie": float(cfg.w_mpc_init_tie),
        },
        "losses": {
            "weighted_torque_mean": float(torque_loss.detach()),
            "unweighted_mpc_init_tie": float(tie_loss_raw.detach()),
            "weighted_mpc_init_tie": float(tie_loss.detach()),
            "mpc_init_tie_elements": int(tie_count.detach()),
        },
        "output_space": gradient_cosine([gt_out], [gi_out]),
    }
    a = gt_out.detach().float().flatten(1)
    b = gi_out.detach().float().flatten(1)
    dot = (a * b).sum(dim=1)
    den = a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt()
    row_cos = torch.where(
        den > 0.0, dot / den.clamp_min(1e-30), torch.full_like(den, float("nan")),
    )
    finite = torch.isfinite(row_cos)
    out["output_space_per_row"] = {
        "cosine": finite_stats(row_cos),
        "n_defined": int(finite.sum()),
        "fraction_conflicting": (
            float((row_cos[finite] < 0.0).float().mean()) if bool(finite.any()) else None
        ),
        "values": [
            (float(x) if math.isfinite(float(x)) else None)
            for x in row_cos.detach().cpu()
        ],
    }
    if skip_parameters:
        out["parameter_space"] = {"skipped": True}
        return out

    core = model.module if hasattr(model, "module") else model
    named = [(name, p) for name, p in core.named_parameters() if p.requires_grad]
    params = [p for _name, p in named]
    gt = torch.autograd.grad(
        torque_loss, params, retain_graph=True, allow_unused=True,
    )
    gi = torch.autograd.grad(
        tie_loss, params, retain_graph=False, allow_unused=True,
    )
    report = {"all_trainable": gradient_cosine(gt, gi)}
    for prefix in ("decoder.", "encoder."):
        take = [i for i, (name, _p) in enumerate(named) if name.startswith(prefix)]
        report[prefix[:-1]] = gradient_cosine(
            [gt[i] for i in take], [gi[i] for i in take],
        )
    out["parameter_space"] = report
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.rows < 1 or args.prime_roller_ticks < 1:
        raise SystemExit("--rows and --prime-roller-ticks must be positive")
    if args.replay_capacity < args.rows:
        raise SystemExit("--replay-capacity must be at least --rows")
    if args.roller_output_bridge_knots < 0:
        raise SystemExit("--roller-output-bridge-knots must be >= 0")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA requested but unavailable: {dev}")
        torch.cuda.set_device(dev.index or 0)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    step = int(args.step if args.step is not None else ck.get("step") or 0)
    cfg = phase2_config_from_checkpoint(ck, device=str(dev), compile_model=False)
    scen = _scenario_from_ckpt(ck)
    buckets, stats = _load_npz_buckets(args.pool_cache)
    pool = Phase2Pool(
        buckets, cfg=scen, noise=Phase2Noise(), device=str(dev), stats=stats,
    )
    eligible_buckets = [b for b in pool.buckets if b <= cfg.roller_horizon]
    bucket = int(args.bucket if args.bucket is not None else max(eligible_buckets))
    if bucket not in eligible_buckets:
        raise SystemExit(
            f"bucket {bucket} is not in roller-compatible pool buckets {eligible_buckets}",
        )

    model_cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(model_cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()

    costs = build_costs(pool.buckets, cfg.cost, device=str(dev), dtype=torch.float32)
    base = costs[max(costs)]
    al = _load_al(cfg, ck)
    weights = _wtensors(
        cfg.schedule.multipliers(step, False),
        al.weights(), dev, pool.dtype,
        sched=cfg.schedule, joint_names=pool.cfg.joint_names,
        step=step, stage_a=False, pen=al.pen,
    )

    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    idle_lib = resolve_idle_cylinder_library(
        cfg.resolved_idle_cylinder_path(), joint_names=pool.cfg.joint_names,
    )
    idle_cyl = None
    if idle_lib is not None:
        try:
            idle_cyl = derive_cylinder(kit, base.fk)
        except Exception:
            idle_cyl = idle_lib.cylinder
        for cost in costs.values():
            if hasattr(cost, "set_idle_cylinder"):
                cost.set_idle_cylinder(idle_cyl if idle_cyl is not None else idle_lib)

    want_waist_ref = cfg.cost.w_waist_aim > 0.0 and not cfg.waist_sp_from_head
    want_waist_pr_ref = bool(cfg.waist_pr_ema) and not cfg.waist_pr_sp_from_head
    # A tiny positive constructor value allocates the score column.  Sampling
    # below is explicit and unweighted, so this does not downweight any row.
    replay = ReplayPool(
        pool, min(int(cfg.replay_capacity), int(args.replay_capacity)),
        buckets=eligible_buckets,
        recency_bias=cfg.recency_bias, viol_downweight=1e-12,
        proficiency=False, waist_ref=want_waist_ref,
        waist_pr_ref=want_waist_pr_ref,
    )
    roller_clips = None
    if "clip_i" in next(iter(pool.data.values())):
        from amortized_opt.rhythm_chain_data import roller_clip_list
        roller_clips = roller_clip_list(kit, library=library, cfg=pool.cfg)

    from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig

    sim = BatchedPDSim(
        PDSimConfig(
            dt=pool.cfg.dt,
            saturate_effort=bool(cfg.plant_saturate_effort),
        ),
        device=str(dev), dtype=torch.float32, rnea=base.rnea,
    )
    roller = MPCRoller(
        pool, model=model, sim=sim, rnea=base.rnea, fk=base.fk,
        horizon=cfg.roller_horizon, slots=cfg.roller_slots,
        chain_frac=cfg.roller_chain_frac, bounce=cfg.bounce,
        device=str(dev), dtype=torch.float32, seed=int(args.seed),
        amp_dtype=None, regoal=cfg.roller_regoal, cmd_lead_s=cfg.cmd_lead_s,
        library=library, kit=kit, yaw_bins=cfg.roller_yaw_bins,
        max_age=cfg.roller_max_age, idle_library=idle_lib,
        idle_cylinder=idle_cyl, idle_settle_wp=cfg.idle_settle_wp,
        rest_ticks=cfg.roller_rest_ticks if idle_lib is not None else 0,
        waist_ref=want_waist_ref, n_shaft=costs[min(costs)].n_points - 2,
        waist_upright_ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
        clips=roller_clips, waist_pr_ema=want_waist_pr_ref,
        waist_pr_ema_alpha=float(cfg.waist_pr_ema_alpha),
        waist_pr_ema_leak=float(cfg.waist_pr_ema_leak),
        decoded_plan_transform=make_c1_roller_plan_transform(
            int(args.roller_output_bridge_knots)
        ),
    )
    age_ring, tick_ring, prime_stats = _prime_replay_with_scores(
        roller, replay, pool, costs, weights, args.prime_roller_ticks,
    )
    n_replay = replay.size(bucket)
    ring_idx_all = torch.arange(n_replay, device=dev)
    ring_score = replay.data[bucket]["viol_score"].index_select(0, ring_idx_all)
    ring_age = age_ring[bucket].index_select(0, ring_idx_all)
    ring_tick = tick_ring[bucket].index_select(0, ring_idx_all)
    eligible = torch.arange(n_replay, device=dev)[
        age_ring[bucket][:n_replay] >= int(args.min_roller_age)
    ]
    if eligible.numel() < args.rows:
        raise SystemExit(
            f"only {eligible.numel()} bucket-{bucket} rows have roller age >= "
            f"{args.min_roller_age}; requested {args.rows}. Increase "
            "--prime-roller-ticks, reduce --rows, or choose another --bucket.",
        )
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(args.seed) + 7919)
    take = torch.randperm(eligible.numel(), device=dev, generator=gen)[: args.rows]
    ring_idx = eligible.index_select(0, take)
    batch = replay.batch(bucket, ring_idx, generator=gen, noise=False)
    batch["mpc_init"] = torch.ones(args.rows, device=dev, dtype=torch.bool)
    harvest_score = replay.data[bucket]["viol_score"].index_select(0, ring_idx)
    roller_age = age_ring[bucket].index_select(0, ring_idx)
    harvest_tick = tick_ring[bucket].index_select(0, ring_idx)

    hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
    lengths = group_lengths(hit_eff)
    strike = replace(batch["strike"], hit_index=hit_eff)
    enc = model.encode_context(batch["context"])
    input_init = batch["init"]
    if cfg.waist_sp_from_head:
        strike = replace(
            strike, waist_yaw_ref=enc.waist_yaw_sp.float().unsqueeze(1).expand(-1, bucket),
        )
    if cfg.waist_pr_sp_from_head:
        strike = replace(
            strike,
            waist_pr_ref=enc.waist_pr_sp.float().unsqueeze(1).expand(-1, bucket, -1),
        )
    if enc.waist_pr_sp is not None:
        input_init = seed_waist_pr_to_sp(
            input_init, batch["q0"], batch["v0"], enc.waist_pr_sp,
            joint_names=base.joint_names, dt=float(pool.cfg.dt),
            ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
        )

    dec = model(input_init, batch["state"], enc, batch["masks"])
    ct_decode = costs[bucket].forward(
        batch["q0"], dec.traj, batch["v0"], strike,
        lengths=lengths, weights=weights,
    )
    shifted = batch["seed"] * batch["masks"].elem_mask
    repaired = c1_bridge_waypoints(
        dec.traj.detach(), batch["q0"], batch["v0"],
        dt=float(pool.cfg.dt), bridge_knots=int(args.bridge_knots),
        elem_mask=batch["masks"].elem_mask,
    )
    segment_indices = {
        name: idx.detach().cpu().tolist() for name, idx in costs[bucket].c_seg
    }
    joint_names = list(costs[bucket].joint_names)
    joint_to_segment = [""] * len(joint_names)
    for name, indices in segment_indices.items():
        for j in indices:
            joint_to_segment[j] = name

    variants = {
        "shifted_init": _variant(
            "shifted_init", shifted, cost=costs[bucket],
            q0=batch["q0"], v0=batch["v0"], strike=strike, lengths=lengths,
            weights=weights, segment_indices=segment_indices,
        ),
        "current_decode": _variant(
            "current_decode", dec.traj, cost=costs[bucket],
            q0=batch["q0"], v0=batch["v0"], strike=strike, lengths=lengths,
            weights=weights, segment_indices=segment_indices, ct=ct_decode,
        ),
        "decode_c1_bridge": _variant(
            "decode_c1_bridge", repaired, cost=costs[bucket],
            q0=batch["q0"], v0=batch["v0"], strike=strike, lengths=lengths,
            weights=weights, segment_indices=segment_indices,
        ),
    }
    current_torque = variants["current_decode"]["scalars"]["torque_peak_excess_nm"]
    current_hit_distance = variants["current_decode"]["scalars"]["hit_distance_m"]
    slot_ticks = max(int(args.prime_roller_ticks) * int(roller.R), 1)
    post_hit_attempted = float(prime_stats.get("post_hit_attempted", 0.0))
    post_hit_accepted = float(prime_stats.get("post_hit_accepted", 0.0))
    fixed_policy = {
        "ticks": int(args.prime_roller_ticks),
        "slots": int(roller.R),
        "slot_ticks": slot_ticks,
        "reset_count": int(prime_stats.get("reset", 0.0)),
        "reset_fraction_per_slot_tick": (
            float(prime_stats.get("reset", 0.0)) / slot_ticks
        ),
        "diverged_count": int(prime_stats.get("diverged", 0.0)),
        "diverged_fraction_per_slot_tick": (
            float(prime_stats.get("diverged", 0.0)) / slot_ticks
        ),
        "regoal_count": int(prime_stats.get("regoal", 0.0)),
        "post_hit_attempted": int(post_hit_attempted),
        "post_hit_accepted": int(post_hit_accepted),
        "post_hit_acceptance_fraction": (
            post_hit_accepted / post_hit_attempted
            if post_hit_attempted > 0.0 else None
        ),
        "post_hit_sdf_rejected": int(prime_stats.get("post_hit_sdf", 0.0)),
    }
    grads = _gradient_report(
        model, dec, ct_decode, batch, cfg,
        skip_parameters=bool(args.skip_parameter_gradients),
    )

    signals = {
        "shifted_q0_first_abs_max_rad":
            variants["shifted_init"]["scalars"]["q0_first_abs_max_rad"],
        "harvest_viol_score": harvest_score,
        "decode_q0_first_abs_max_rad":
            variants["current_decode"]["scalars"]["q0_first_abs_max_rad"],
        "decode_first1_torque_excess_nm":
            variants["current_decode"]["scalars"]["torque_first_1_peak_excess_nm"],
        "decode_torque_peak_excess_nm":
            variants["current_decode"]["scalars"]["torque_peak_excess_nm"],
        "decode_hit_distance_m":
            variants["current_decode"]["scalars"]["hit_distance_m"],
        "decode_hit_region_excess_m":
            variants["current_decode"]["scalars"]["hit_region_excess_m"],
    }
    corr_pairs = (
        ("shifted_q0_first_abs_max_rad", "harvest_viol_score"),
        ("shifted_q0_first_abs_max_rad", "decode_q0_first_abs_max_rad"),
        ("shifted_q0_first_abs_max_rad", "decode_first1_torque_excess_nm"),
        ("shifted_q0_first_abs_max_rad", "decode_torque_peak_excess_nm"),
        ("shifted_q0_first_abs_max_rad", "decode_hit_distance_m"),
        ("shifted_q0_first_abs_max_rad", "decode_hit_region_excess_m"),
        ("harvest_viol_score", "decode_torque_peak_excess_nm"),
        ("harvest_viol_score", "decode_hit_distance_m"),
    )
    correlations = {
        f"{a}__vs__{b}": pearsonr_finite(signals[a], signals[b])
        for a, b in corr_pairs
    }
    row_cos = grads["output_space_per_row"]["values"]
    defined_cos = torch.tensor(
        [float("nan") if x is None else x for x in row_cos], dtype=torch.float64,
    )
    correlations["shifted_q0_first_abs_max_rad__vs__torque_tie_output_cosine"] = (
        pearsonr_finite(signals["shifted_q0_first_abs_max_rad"], defined_cos)
    )

    rows = []
    for i in range(args.rows):
        rows.append({
            "row": i,
            "replay_ring_index": int(ring_idx[i]),
            "roller_age": int(roller_age[i]),
            "harvest_tick": int(harvest_tick[i]),
            "harvest_viol_score": float(harvest_score[i]),
            "hit": [int(x) for x in batch["hit"][i].detach().cpu()],
            "active": [bool(x) for x in batch["active"][i].detach().cpu()],
            "torque_tie_output_cosine": row_cos[i],
            "variants": {
                name: _row_variant(data, i, joint_names, joint_to_segment)
                for name, data in variants.items()
            },
        })

    report = {
        "meta": {
            "ckpt": str(args.ckpt),
            "checkpoint_step": int(ck.get("step") or 0),
            "probe_step": step,
            "pool_cache": str(args.pool_cache),
            "device": str(dev),
            "dtype": str(pool.dtype),
            "bucket": bucket,
            "rows": int(args.rows),
            "prime_roller_ticks": int(args.prime_roller_ticks),
            "probe_replay_capacity": int(replay.capacity),
            "min_roller_age": int(args.min_roller_age),
            "bridge_knots": int(args.bridge_knots),
            "bridge_base": "current_decode",
            "roller_output_bridge_knots": int(args.roller_output_bridge_knots),
            "roller_output_bridge_stage": (
                "every decode before plant command and replay shift"
                if args.roller_output_bridge_knots > 0 else "off"
            ),
            "roller_seed": int(args.seed),
            "bridge_warning": (
                "falsification-only diagnostic; does not alter the training recipe"
            ),
            "first_window_convention": "first_1 includes measured boundary knot 0",
            "physics_mode": str(cfg.cost.physics_mode),
            "torque_physics": str(cfg.cost.torque_physics),
            "torque_derate": str(cfg.cost.torque_derate),
            "w_mpc_init_tie": float(cfg.w_mpc_init_tie),
            "model_output_mode": str(model_cfg.output_mode),
            "replay_sizes": replay.distribution(),
            "eligible_shifted_rows": int(eligible.numel()),
            "prime_stats": prime_stats,
        },
        "aggregate": {
            "requested_summary": {
                "fixed_policy": fixed_policy,
                "ring": {
                    "rows": n_replay,
                    "harvest_viol_score": finite_stats(ring_score),
                    "roller_age": finite_stats(ring_age),
                    "harvest_tick": finite_stats(ring_tick),
                    "harvest_viol_score_by_age": _stats_by_age(
                        ring_age, ring_score,
                    ),
                    "age_vs_harvest_viol_score": pearsonr_finite(
                        ring_age, ring_score,
                    ),
                },
                "sampled_current_decode": {
                    "rows": int(args.rows),
                    "torque_violation_rows": {
                        "n": int((current_torque > 0.0).sum()),
                        "fraction": float((current_torque > 0.0).float().mean()),
                    },
                    "torque_peak_excess_nm": finite_stats(current_torque),
                    "hit_distance_m": finite_stats(current_hit_distance),
                    "roller_age": finite_stats(roller_age),
                    "torque_peak_excess_by_age": _stats_by_age(
                        roller_age, current_torque,
                    ),
                    "age_vs_torque_peak_excess": pearsonr_finite(
                        roller_age, current_torque,
                    ),
                },
            },
            "harvest_viol_score": finite_stats(harvest_score),
            "harvest_score_vs_recomputed_shifted_abs_error_nm": finite_stats(
                (
                    harvest_score
                    - variants["shifted_init"]["scalars"]["torque_peak_excess_nm"]
                ).abs(),
            ),
            "roller_age": finite_stats(roller_age),
            "variants": {
                name: _aggregate_variant(data, joint_names, joint_to_segment)
                for name, data in variants.items()
            },
            "comparisons": {
                "shifted_init_to_current_decode": _paired_comparison(
                    variants["shifted_init"], variants["current_decode"],
                ),
                "current_decode_to_decode_c1_bridge": _paired_comparison(
                    variants["current_decode"], variants["decode_c1_bridge"],
                ),
            },
            "correlations": correlations,
            "torque_vs_mpc_init_tie_gradients": grads,
        },
        "rows": rows,
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    if not args.quiet:
        print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
