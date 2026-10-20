#!/usr/bin/env python
"""Paired plan-only drift sweeps on training-style MPC suffix collocations.

State, context, hit indices and masks are held fixed across every sweep.
No plant, freeze window, optimizer update, or critic admission is applied.
Two sources are compared: freshly decoded training scenarios and the saved
training MPC parent ring. Admission is not assumed to imply feasibility.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from probe_replay_boundary import _load_al, _scenario_from_ckpt
from train_phase2 import _load_npz_buckets
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool
from amortized_opt.train_cost import (
    _TrajectorySuffixMPCParentRing, _apply_terminal_initializer,
    _cat_phase2_batches, _gate_from_ct, _hard_feasibility_ok_from_gate,
    _index_batch, _wtensors, build_costs, effective_hits, group_lengths,
    phase2_config_from_checkpoint,
)
from amortized_opt.warmstart_data import ou_kernel

BANDS = {"h1_5": (1, 5), "h6_10": (6, 10),
         "h11_25": (11, 25), "h26_49": (26, 49)}
DIAGNOSTICS = ("terminal_region_dist", "contact_pos_err", "contact_speed_err",
               "contact_dir_cos", "collision_min_sdf", "torque_excess_max",
               "velocity_excess_max", "joint_box_excess_max",
               "waist_yawd_max", "smooth_jerk_integral")


def masked_rms(x, mask):
    return ((x.square() * mask).sum((1, 2)) /
            mask.sum((1, 2)).clamp_min(1)).sqrt()


def unit_drift(init, mask, mode, generator):
    """Per-row unit RMS directions; zero outside each joint's valid horizon."""
    b, h, d = init.shape
    if mode in ("ou", "anchored_ou"):
        eps = torch.randn(b, h, d, device=init.device, generator=generator)
        kernel = ou_kernel(h, 5.0, init.device, init.dtype)
        drift = torch.einsum("ij,bjd->bid", kernel, eps)
        if mode == "anchored_ou":
            u = torch.arange(1, h + 1, device=init.device).clamp_max(5) / 5.0
            drift = drift * (u.square() * (3 - 2 * u))[None, :, None]
    elif mode == "smooth_bias":
        bias = torch.randn(b, 1, d, device=init.device, generator=generator)
        u = (torch.arange(1, h + 1, device=init.device)[None, :, None] /
             mask.sum(1, keepdim=True).clamp_min(1)).clamp_max(1)
        drift = bias * u.square() * (3 - 2 * u)
    else:
        raise ValueError(mode)
    drift = drift * mask
    return drift / masked_rms(drift, mask).clamp_min(1e-12)[:, None, None]


def phase_lag(init, q0, mask, ticks):
    """Delay the input path by fractional knots, keeping the goal clock fixed."""
    b, h, d = init.shape
    path = torch.cat([q0[:, None], init], 1)
    t = (torch.arange(1, h + 1, device=init.device, dtype=init.dtype)
         - float(ticks)).clamp(0, h)
    lo = t.floor().long()
    hi = (lo + 1).clamp_max(h)
    u = (t - lo)[None, :, None]
    out = (1 - u) * path[:, lo] + u * path[:, hi]
    return out * mask


def _stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"mean": None, "p50": None, "p90": None}
    return dict(mean=float(x.mean()), p50=float(np.median(x)),
                p90=float(np.percentile(x, 90)))


def _rate(x, take):
    return float(np.asarray(x)[take].mean()) if np.any(take) else None


def summarize(inp, out, baseline_in, baseline_out, take, drift, output_drift):
    """Keep unconditional rates separate from preservation and recovery."""
    result = {"n": int(take.sum()), "input_drift_rad": _stats(drift[take]),
              "output_drift_rad": _stats(output_drift[take]),
              "input_on_head_rate": _rate(inp["on_head"], take),
              "output_on_head_rate": _rate(out["on_head"], take),
              "input_within_5mm_rate": _rate(inp["within_5mm"], take),
              "output_within_5mm_rate": _rate(out["within_5mm"], take),
              "input_geometric_rate": _rate(inp["geometric_ok"], take),
              "output_geometric_rate": _rate(out["geometric_ok"], take),
              "input_hard_rate": _rate(inp["hard_ok"], take),
              "output_hard_rate": _rate(out["hard_ok"], take)}
    for flag in ("on_head", "within_5mm", "geometric_ok", "hard_ok"):
        good_init = take & baseline_in[flag]
        good_output = take & baseline_out[flag]
        damaged = good_init & ~inp[flag]
        result[f"clean_input_{flag}_n"] = int(good_init.sum())
        result[f"clean_output_{flag}_n"] = int(good_output.sum())
        result[f"preserve_clean_input_{flag}_rate"] = _rate(out[flag], good_init)
        result[f"preserve_clean_output_{flag}_rate"] = _rate(out[flag], good_output)
        result[f"damaged_{flag}_n"] = int(damaged.sum())
        result[f"repair_damaged_{flag}_rate"] = _rate(out[flag], damaged)
        lost = good_init & ~out[flag]
        result[f"lost_clean_input_{flag}_region_dist"] = _stats(
            out["terminal_region_dist"][lost])
        result[f"lost_clean_input_{flag}_cost_improved_rate"] = _rate(
            out["cost"] < inp["cost"], lost)
    result["cost_improved_rate"] = _rate(out["cost"] < inp["cost"], take)
    result["cost_gain"] = _stats((inp["cost"] - out["cost"])[take])
    result["output_cost_delta_vs_clean"] = _stats(
        (out["cost"] - baseline_out["cost"])[take])
    for key in DIAGNOSTICS:
        result[f"input_{key}"] = _stats(inp[key][take])
        result[f"output_{key}"] = _stats(out[key][take])
    nonzero = take & (drift > 1e-8)
    result["paired_sensitivity"] = _stats(
        output_drift[nonzero] / drift[nonzero])
    return result


@torch.no_grad()
def evaluate(cost, batch, traj, weights):
    hit = effective_hits(batch["hit"], batch["active"], traj.shape[1])
    ct = cost.forward(batch["q0"], traj, batch["v0"],
                      replace(batch["strike"], hit_index=hit),
                      lengths=group_lengths(hit), weights=weights)
    gate = _gate_from_ct(ct)
    finite = torch.isfinite(gate[:, 4:8]).all(1)
    on_head = finite & (gate[:, 4] <= 1e-6)
    geometric = on_head & (gate[:, 5] >= 0) & (gate[:, 6] <= 1e-6)
    values = {k: ct.diagnostics[k] for k in DIAGNOSTICS}
    values.update(cost=ct.total, on_head=on_head, geometric_ok=geometric,
                  within_5mm=finite & (gate[:, 4] <= .005),
                  hard_ok=_hard_feasibility_ok_from_gate(gate))
    return {k: v.cpu().numpy() for k, v in values.items()}


def make_ring(state, cfg, pool, costs, device):
    ring = _TrajectorySuffixMPCParentRing(
        device=device, pool=pool, costs=costs, q_noise=0, qd_noise=0,
        capacity=cfg.timing_mpc_parent_capacity,
        terminal_peak_frac=cfg.terminal_suffix_peak_frac,
        terminal_long_frac=cfg.terminal_suffix_long_frac,
        terminal_low_frac=cfg.terminal_suffix_low_frac,
        terminal_peak_wp=cfg.terminal_suffix_peak_wp,
        terminal_tail_tau=cfg.terminal_suffix_tail_tau)
    if state is not None:
        ring.load_state_dict(state)
    return ring


@torch.no_grad()
def collocations(ring, bucket, rows_per_band, gen):
    pieces = {name: [] for name in BANDS}
    counts = {name: 0 for name in BANDS}
    for _ in range(50):
        idx = ring.sample_idx(bucket, max(256, rows_per_band * 4), generator=gen)
        batch = ring.batch(bucket, idx, generator=gen)
        hit = torch.where(batch["active"], batch["hit"], bucket).min(1).values
        for name, (lo, hi) in BANDS.items():
            n = rows_per_band - counts[name]
            take = ((hit >= lo) & (hit <= hi)).nonzero(as_tuple=True)[0][:n]
            if n > 0 and len(take):
                pieces[name].append(_index_batch(batch, take))
                counts[name] += len(take)
        if all(n == rows_per_band for n in counts.values()):
            break
    if any(n < rows_per_band for n in counts.values()):
        raise RuntimeError(f"insufficient suffix support: {counts}")
    return _cat_phase2_batches([p for ps in pieces.values() for p in ps])


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pool-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows-per-band", type=int, default=64)
    ap.add_argument("--fresh-parents", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--amplitudes", type=float, nargs="+",
                    default=[0, .005, .01, .02, .04, .08, .16, .30])
    ap.add_argument("--phase-ticks", type=float, nargs="+",
                    default=[0, .25, .5, 1, 2, 3, 5])
    ap.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args(argv)
    if min(args.rows_per_band, args.batch, args.fresh_parents) <= 0:
        ap.error("row counts and batch must be positive")
    if min(args.iterations) < 1 or min(args.amplitudes + args.phase_ticks) < 0:
        ap.error("iterations must be positive and perturbations nonnegative")
    start = time.monotonic()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = phase2_config_from_checkpoint(ck, device=str(dev), compile_model=False)
    scen = _scenario_from_ckpt(ck)
    model = A.build_model(A.model_config_from_checkpoint(ck), dev)
    model.load_state_dict(ck["model"])
    model.eval()
    # Read the pool on CPU, then move only sampled fresh cases to the GPU.
    arrays, stats = _load_npz_buckets(args.pool_cache)
    ring_state = ck["extra"]["timing_critic"]["primary_mpc_ring"]
    bucket = max(int(h) for h in ring_state["data"])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(arrays[bucket]["q0"]), args.fresh_parents, replace=False)
    small = {k: v[idx] for k, v in arrays[bucket].items()}
    del arrays
    pool = Phase2Pool({bucket: small}, cfg=scen, noise=Phase2Noise(),
                      device=dev, stats=stats)
    costs = build_costs([bucket], cfg.cost, device=dev)
    cost = costs[bucket]
    al = _load_al(cfg, ck)
    step = int(ck["step"])
    weights = _wtensors(cfg.schedule.multipliers(step, True), al.weights(),
                        dev, pool.dtype, sched=cfg.schedule,
                        joint_names=scen.joint_names, step=step,
                        stage_a=True, pen=al.pen)
    saved = make_ring(ring_state, cfg, pool, costs, dev)
    fresh = make_ring(None, cfg, pool, costs, dev)
    for lo in range(0, args.fresh_parents, args.batch):
        take = torch.arange(lo, min(lo + args.batch, args.fresh_parents), device=dev)
        batch = pool.batch(bucket, take, generator=gen, noise=True)
        _apply_terminal_initializer(
            model, batch, pool=pool, generator=gen,
            start_anchored_noise=cfg.terminal_init_noise,
            noise_ramp_steps=cfg.terminal_init_noise_ramp_steps)
        traj = model(batch["init"], batch["state"],
                     model.encode_context(batch["context"]), batch["masks"]).traj
        fresh.refresh(bucket, batch, traj,
                      hard_ok=torch.ones(len(take), device=dev, dtype=torch.bool))
    reports = []
    all_arrays = {}
    source_info = {}
    for source, ring in (("fresh_training_parent", fresh),
                         ("saved_training_mpc_parent", saved)):
        batch = collocations(ring, bucket, args.rows_per_band, gen)
        n = len(batch["q0"])
        hit = torch.where(batch["active"], batch["hit"], bucket).min(1).values
        hit_np = hit.cpu().numpy()
        source_info[source] = {"n": n, "parents_available": ring.size(bucket)}
        all_arrays[f"{source}/hit"] = hit_np
        for key in ("q0", "v0", "init", "active"):
            all_arrays[f"{source}/{key}"] = batch[key].cpu().numpy()
        for key in batch["context"].__dataclass_fields__:
            value = getattr(batch["context"], key)
            if torch.is_tensor(value):
                all_arrays[f"{source}/context/{key}"] = value.cpu().numpy()
        conditions = [(mode, a, "rad") for mode in
                      ("ou", "anchored_ou", "smooth_bias") for a in args.amplitudes]
        conditions += [("phase_lag", t, "ticks") for t in args.phase_ticks]
        conditions += [("fresh_terminal_reset", 0, "reset")]
        source_rows = {cond: [] for cond in conditions}
        for lo in range(0, n, args.batch):
            take = torch.arange(lo, min(lo + args.batch, n), device=dev)
            bt = _index_batch(batch, take)
            clean = bt["init"]
            mask = bt["masks"].elem_mask.to(clean.dtype)
            enc = model.encode_context(bt["context"])
            base_input = evaluate(cost, bt, clean, weights)
            clean_outputs = {}
            base_outputs = {}
            current = clean
            for iteration in range(1, max(args.iterations) + 1):
                current = model(current, bt["state"], enc, bt["masks"]).traj
                if iteration in args.iterations:
                    clean_outputs[iteration] = current.clone()
                    base_outputs[iteration] = evaluate(cost, bt, current, weights)
            directions = {mode: unit_drift(clean, mask, mode, gen) for mode in
                          ("ou", "anchored_ou", "smooth_bias")}
            for mode, amplitude, unit in conditions:
                if mode == "phase_lag":
                    init = phase_lag(clean, bt["q0"], mask, amplitude)
                elif mode == "fresh_terminal_reset":
                    init = model.initial_trajectory(bt["context"], bt["state"],
                                                    bt["masks"]) * mask
                else:
                    init = clean + amplitude * directions[mode]
                init = init.clamp(pool.q_lo, pool.q_hi) * mask
                drift = masked_rms(init - clean, mask).cpu().numpy()
                inp = evaluate(cost, bt, init, weights)
                current = init
                rows = {}
                for iteration in range(1, max(args.iterations) + 1):
                    current = model(current, bt["state"], enc, bt["masks"]).traj
                    if iteration in args.iterations:
                        rows[iteration] = dict(
                            inp=inp, out=evaluate(cost, bt, current, weights),
                            baseline_in=base_input, baseline_out=base_outputs[iteration],
                            drift=drift, output_drift=masked_rms(
                                current - clean_outputs[iteration], mask).cpu().numpy())
                source_rows[(mode, amplitude, unit)].append(rows)
            print(f"[drift] {source}: {min(lo + args.batch, n)}/{n} rows", flush=True)
        for (mode, amplitude, unit), chunks in source_rows.items():
            for iteration in args.iterations:
                merged = {}
                for key in chunks[0][iteration]:
                    if isinstance(chunks[0][iteration][key], dict):
                        merged[key] = {k: np.concatenate([c[iteration][key][k] for c in chunks])
                                       for k in chunks[0][iteration][key]}
                    else:
                        merged[key] = np.concatenate([c[iteration][key] for c in chunks])
                groups = {"all": np.ones(n, dtype=bool)}
                groups.update({name: (hit_np >= low) & (hit_np <= high)
                               for name, (low, high) in BANDS.items()})
                record = dict(source=source, mode=mode, amplitude=amplitude,
                              amplitude_unit=unit, iterations=iteration,
                              groups={name: summarize(**merged, take=selection)
                                      for name, selection in groups.items()})
                reports.append(record)
                prefix = f"{source}/{mode}/{amplitude}/{iteration}"
                for key, value in merged.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            all_arrays[f"{prefix}/{key}/{k}"] = v
                    else:
                        all_arrays[f"{prefix}/{key}"] = value
    payload = dict(checkpoint=str(args.ckpt.resolve()), step=step, seed=args.seed,
                   device=str(dev), elapsed_s=time.monotonic() - start,
                   design=dict(plan_only=True, state_noise=False, amp=False,
                               freeze=False, balanced_horizon_bands=BANDS,
                               input_joint_box_clamp=True,
                               source_info=source_info,
                               fresh_training_init_noise=cfg.terminal_init_noise),
                   reports=reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **all_arrays)
    print(f"[drift] wrote {args.out}; elapsed {payload['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
