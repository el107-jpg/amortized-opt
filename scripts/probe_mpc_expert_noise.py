#!/usr/bin/env python
"""Frozen decoder recovery from corrupted, demonstrably feasible MPC plans.

Select distinct parent/offset pairs with a feasible reference AND feasible clean
decoder output. Keep state, context and timing fixed, perturb only init, and
report distances and loss of feasibility. This is a local reference-region
probe, not an identification of a global expert trajectory manifold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from amortized_opt.phase2_data import Phase2Pool
from amortized_opt.train_cost import (
    _cat_phase2_batches, _index_batch, _wtensors, build_costs,
    phase2_config_from_checkpoint,
)
from amortized_opt.warmstart_data import ou_kernel
from probe_mpc_plan_drift import BANDS, _stats, evaluate, make_ring, masked_rms
from probe_replay_boundary import _load_al, _scenario_from_ckpt


def corrupt_reference(reference, mask, correlated, white, sigma, lo, hi, anchored):
    """Match fresh-seed OU + white noise and optional five-knot envelope.

    Zero sigma is the clean control; positive sigma also enables 0.01 rad white
    noise. Directions are reused across sigma levels for paired comparisons.
    """
    if sigma == 0:
        return reference.clone()
    noisy = (reference + float(sigma) * correlated + .01 * white).clamp(lo, hi)
    if anchored:
        h = reference.shape[1]
        u = torch.arange(1, h + 1, device=reference.device,
                         dtype=reference.dtype).clamp_max(5) / 5
        envelope = (u.square() * (3 - 2 * u))[None, :, None]
        noisy = reference + envelope * (noisy - reference)
    return noisy * mask


@torch.no_grad()
def select_feasible_references(ring, bucket, model, cost, weights, gen,
                               count, microbatch, max_draws):
    parts = []
    seen = set()
    selected = examined = input_ok_n = both_ok_n = 0
    for draw in range(max_draws):
        idx = ring.sample_idx(bucket, 1024, generator=gen)
        candidates = ring.batch(bucket, idx, generator=gen)
        for start in range(0, 1024, microbatch):
            take = torch.arange(start, min(start + microbatch, 1024),
                                device=ring.device)
            batch = _index_batch(candidates, take)
            initial = evaluate(cost, batch, batch["init"], weights)
            examined += len(take)
            input_ok_n += int(initial["hard_ok"].sum())
            feasible = np.flatnonzero(initial["hard_ok"])
            if not len(feasible):
                continue
            batch = _index_batch(batch, torch.as_tensor(feasible, device=ring.device))
            out = model(batch["init"], batch["state"],
                        model.encode_context(batch["context"]), batch["masks"])
            metrics = evaluate(cost, batch, out.traj, weights)
            both_ok_n += int(metrics["hard_ok"].sum())
            parent_ids = batch["probe_parent_id"].cpu().numpy()
            nearest = torch.where(batch["active"], batch["hit"], bucket).min(1).values
            offset = (batch["probe_parent_nearest"] - nearest).cpu().numpy()
            accept = []
            for row in np.flatnonzero(metrics["hard_ok"]):
                key = (int(parent_ids[row]), int(offset[row]))
                if key not in seen:
                    seen.add(key)
                    accept.append(int(row))
                    selected += 1
                if selected >= count:
                    break
            if accept:
                parts.append(_index_batch(batch, torch.tensor(accept, device=ring.device)))
            if selected >= count:
                break
        print(f"[expert-noise] selection: {selected}/{count} distinct feasible "
              f"references from {examined} candidate draws", flush=True)
        if selected >= count:
            break
    if selected < count:
        raise RuntimeError(f"Only {selected} feasible distinct pairs after {examined} draws")
    return _cat_phase2_batches(parts), dict(
        examined=examined, input_feasible_draws=input_ok_n,
        input_and_clean_output_feasible_draws=both_ok_n,
        selected=selected, distinct_parent_offset_pairs=len(seen))


def summary(values, selection):
    report = {"n": int(selection.sum())}
    for stage in ("input", "output", "baseline_output"):
        for flag in ("hard_ok", "on_head", "within_5mm", "geometric_ok"):
            report[f"{stage}_{flag}_rate"] = float(values[stage][flag][selection].mean())
        for diag in ("terminal_region_dist", "collision_min_sdf", "torque_excess_max",
                     "velocity_excess_max", "joint_box_excess_max", "contact_speed_err",
                     "cost"):
            report[f"{stage}_{diag}"] = _stats(values[stage][diag][selection])
    for key in ("input_reference_rms", "output_reference_rms",
                "baseline_output_reference_rms", "output_change_rms",
                "first_waypoint_change_rms"):
        report[key] = _stats(values[key][selection])
    report["paired_output_gain"] = _stats(
        values["output_change_rms"][selection] /
        np.maximum(values["input_reference_rms"][selection], 1e-12))
    out = values["output"]
    report["terminal_failure_rate"] = float((out["terminal_region_dist"][selection] > 1e-6).mean())
    report["collision_failure_rate"] = float((out["collision_min_sdf"][selection] < 0).mean())
    report["torque_failure_rate"] = float((out["torque_excess_max"][selection] > 1e-6).mean())
    report["joint_box_failure_rate"] = float((out["joint_box_excess_max"][selection] > 1e-6).mean())
    return report


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-draws", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0, .02, .05, .1, .2, .4, .8, 1.2])
    ap.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args(argv)
    if min(args.rows, args.batch, args.max_draws, *args.iterations) <= 0 or min(args.sigmas) < 0:
        ap.error("Counts must be positive and sigmas nonnegative")
    start = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = phase2_config_from_checkpoint(ck, device=str(device), compile_model=False)
    scenario = _scenario_from_ckpt(ck)
    pool = Phase2Pool({}, cfg=scenario, device=device)
    state = ck["extra"]["timing_critic"]["primary_mpc_ring"]
    bucket = max(int(h) for h in state["data"])
    costs = build_costs([bucket], cfg.cost, device=device)
    cost = costs[bucket]
    al = _load_al(cfg, ck)
    step = int(ck["step"])
    weights = _wtensors(cfg.schedule.multipliers(step, True), al.weights(),
                        device, pool.dtype, sched=cfg.schedule,
                        joint_names=scenario.joint_names, step=step,
                        stage_a=True, pen=al.pen)
    model = A.build_model(A.model_config_from_checkpoint(ck), device)
    model.load_state_dict(ck["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    ring = make_ring(state, cfg, pool, costs, device)
    stored = ring.data[bucket]
    stored["probe_parent_id"] = torch.arange(ring.size(bucket), device=device)
    stored["probe_parent_nearest"] = torch.where(
        stored["active"], stored["hit"], bucket).min(1).values
    batch, selection_info = select_feasible_references(
        ring, bucket, model, cost, weights, gen, args.rows, args.batch, args.max_draws)
    n = len(batch["q0"])
    nearest = torch.where(batch["active"], batch["hit"], bucket).min(1).values.cpu().numpy()
    conditions = [(mode, sigma) for mode in ("ou_white", "anchored_ou_white")
                  for sigma in args.sigmas]
    accumulated = {(mode, sigma, it): [] for mode, sigma in conditions for it in args.iterations}
    arrays = {"nearest_hit": nearest}
    for start_row in range(0, n, args.batch):
        idx = torch.arange(start_row, min(start_row + args.batch, n), device=device)
        bt = _index_batch(batch, idx)
        reference = bt["init"]
        mask = bt["masks"].elem_mask.to(reference.dtype)
        encoded = model.encode_context(bt["context"])
        baseline_outputs = {}
        baseline_metrics = {}
        current = reference
        for iteration in range(1, max(args.iterations) + 1):
            current = model(current, bt["state"], encoded, bt["masks"]).traj
            if iteration in args.iterations:
                baseline_outputs[iteration] = current.clone()
                baseline_metrics[iteration] = evaluate(cost, bt, current, weights)
        eps = torch.randn(reference.shape, device=device, generator=gen)
        correlated = torch.einsum("ij,bjd->bid", ou_kernel(bucket, 5, device, reference.dtype), eps)
        white = torch.randn(reference.shape, device=device, generator=gen)
        for mode, sigma in conditions:
            initial = corrupt_reference(reference, mask, correlated, white, sigma,
                                        pool.q_lo, pool.q_hi, mode == "anchored_ou_white")
            input_metrics = evaluate(cost, bt, initial, weights)
            input_distance = masked_rms(initial - reference, mask).cpu().numpy()
            current = initial
            for iteration in range(1, max(args.iterations) + 1):
                current = model(current, bt["state"], encoded, bt["masks"]).traj
                if iteration in args.iterations:
                    values = dict(
                        input=input_metrics, output=evaluate(cost, bt, current, weights),
                        baseline_output=baseline_metrics[iteration],
                        input_reference_rms=input_distance,
                        output_reference_rms=masked_rms(current-reference, mask).cpu().numpy(),
                        baseline_output_reference_rms=masked_rms(
                            baseline_outputs[iteration]-reference, mask).cpu().numpy(),
                        output_change_rms=masked_rms(
                            current-baseline_outputs[iteration], mask).cpu().numpy(),
                        first_waypoint_change_rms=(
                            (current[:, 0]-baseline_outputs[iteration][:, 0]).square().mean(1)
                        ).sqrt().cpu().numpy())
                    accumulated[(mode, sigma, iteration)].append(values)
        print(f"[expert-noise] sweep: {min(start_row+args.batch,n)}/{n}", flush=True)
    reports = []
    for (mode, sigma, iteration), chunks in accumulated.items():
        values = {}
        for key, value in chunks[0].items():
            if isinstance(value, dict):
                values[key] = {k: np.concatenate([chunk[key][k] for chunk in chunks]) for k in value}
            else:
                values[key] = np.concatenate([chunk[key] for chunk in chunks])
        groups = {"all": np.ones(n, dtype=bool)}
        groups.update({key: (nearest >= lo) & (nearest <= hi) for key, (lo, hi) in BANDS.items()})
        groups = {key: take for key, take in groups.items() if take.any()}
        reports.append(dict(mode=mode, sigma_corr=sigma, iterations=iteration,
                            groups={key: summary(values, take) for key, take in groups.items()}))
        prefix = f"{mode}/{sigma}/{iteration}"
        for key, value in values.items():
            if isinstance(value, dict):
                arrays.update({f"{prefix}/{key}/{k}": v for k, v in value.items()})
            else:
                arrays[f"{prefix}/{key}"] = value
    for key, value in batch.items():
        if torch.is_tensor(value):
            arrays[f"reference/{key}"] = value.cpu().numpy()
        elif hasattr(value, "__dataclass_fields__"):
            for field in value.__dataclass_fields__:
                v = getattr(value, field)
                if torch.is_tensor(v):
                    arrays[f"reference/{key}/{field}"] = v.cpu().numpy()
    payload = dict(checkpoint=str(args.ckpt.resolve()), step=step, seed=args.seed,
                   elapsed_s=time.monotonic()-start, selection=selection_info,
                   design=dict(frozen_model=True, plan_only=True, amp=False,
                               state_noise=False, freeze=False, sigma_white=.01,
                               ou_tau_steps=5, joint_box_clipped=True,
                               reference_predicate="input AND one-pass clean output pass terminal/collision/joint-box/torque gate",
                               manifold_proxy="distance to feasible reference for the same task and loss of gate feasibility"),
                   reports=reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    print(f"[expert-noise] wrote {args.out}; elapsed {payload['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
