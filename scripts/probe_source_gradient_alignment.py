#!/usr/bin/env python
"""Measure actor-gradient alignment across fresh, MPC, and Bellman sources."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A  # noqa: E402
from probe_replay_boundary import _load_al, _scenario_from_ckpt  # noqa: E402
from train_phase2 import _load_npz_buckets  # noqa: E402
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool  # noqa: E402
from amortized_opt.train_cost import (  # noqa: E402
    _VerifiedSuffixRing,
    _VoxelMPCParentRing,
    _apply_terminal_initializer,
    _index_batch,
    _suffix_distill_loss,
    _wtensors,
    build_costs,
    effective_hits,
    group_lengths,
    make_suffix_step,
    mpc_init_tie_loss,
    phase2_config_from_checkpoint,
    shift_plan,
    suffix_offsets,
)


def _grads(loss: torch.Tensor, params: list[torch.nn.Parameter]):
    values = torch.autograd.grad(loss, params, allow_unused=True)
    return tuple(
        None if value is None else value.detach().float().cpu()
        for value in values
    )


def _add(*vectors):
    out = []
    for values in zip(*vectors):
        live = [value for value in values if value is not None]
        out.append(None if not live else sum(live[1:], live[0].clone()))
    return tuple(out)


def _alignment(a, b, take: list[int]):
    dot = na = nb = 0.0
    for i in take:
        ga, gb = a[i], b[i]
        if ga is None or gb is None:
            continue
        ga = ga.double()
        gb = gb.double()
        dot += float((ga * gb).sum())
        na += float(ga.square().sum())
        nb += float(gb.square().sum())
    den = math.sqrt(na * nb)
    return {
        "cosine": dot / den if den > 0 else None,
        "dot": dot,
        "norm_a": math.sqrt(na),
        "norm_b": math.sqrt(nb),
        "norm_b_over_a": math.sqrt(nb / na) if na > 0 else None,
    }


def _prepare_primary(model, core, batch, pool, cost, weights, cfg, gen):
    bucket = int(batch["init"].shape[1])
    batch = _index_batch(batch, torch.arange(batch["q0"].shape[0], device=pool.device))
    if "mpc_init" not in batch:
        batch["mpc_init"] = torch.zeros(
            batch["q0"].shape[0], device=pool.device, dtype=torch.bool,
        )
    enc = model.encode_context(batch["context"])
    _apply_terminal_initializer(
        core, batch, pool=pool, generator=gen,
        start_anchored_noise=bool(cfg.terminal_init_noise),
        noise_ramp_steps=int(cfg.terminal_init_noise_ramp_steps),
    )
    hit = effective_hits(batch["hit"], batch["active"], bucket)
    lengths = group_lengths(hit)
    strike = replace(batch["strike"], hit_index=hit)
    dec = model(batch["init"], batch["state"], enc, batch["masks"])
    ct = cost.forward(
        batch["q0"], dec.traj, batch["v0"], strike,
        lengths=lengths, weights=weights,
    )
    return batch, dec, ct


def _primary_grad(model, core, batch, pool, cost, weights, cfg, gen, params):
    _batch, _dec, ct = _prepare_primary(
        model, core, batch, pool, cost, weights, cfg, gen,
    )
    return _grads(float(cfg.w_cost) * ct.total.mean(), params)


def _mpc_grads(model, core, batch, pool, cost, weights, cfg, gen, params):
    batch, dec, ct = _prepare_primary(
        model, core, batch, pool, cost, weights, cfg, gen,
    )
    primary = float(cfg.w_cost) * ct.total.mean()
    tie_raw, _ = mpc_init_tie_loss(
        dec.traj, batch["seed"], batch["masks"].elem_mask, batch["mpc_init"],
    )
    tie = float(cfg.w_mpc_init_tie) * tie_raw
    gp = torch.autograd.grad(primary, params, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(tie, params, allow_unused=True)
    to_cpu = lambda gs: tuple(  # noqa: E731
        None if g is None else g.detach().float().cpu() for g in gs
    )
    return to_cpu(gp), to_cpu(gt), {
        "primary_loss": float(primary.detach()),
        "tie_raw": float(tie_raw.detach()),
        "tie_weighted": float(tie.detach()),
    }


def _fresh_batch(pool, bucket: int, rows: int, gen):
    idx = pool.sample_idx(
        bucket, rows, generator=gen, short_wp=0, short_boost=1.0,
        kind_fracs=None,
    )
    batch = pool.batch(bucket, idx, generator=gen, noise=True)
    batch["mpc_init"] = torch.zeros(rows, device=pool.device, dtype=torch.bool)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool-cache", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = phase2_config_from_checkpoint(ck, device=str(dev), compile_model=False)
    scen = _scenario_from_ckpt(ck)
    buckets, stats = _load_npz_buckets(args.pool_cache)
    pool = Phase2Pool(
        buckets, cfg=scen, noise=Phase2Noise(), device=str(dev), stats=stats,
    )
    bucket = max(pool.buckets)
    model_cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(model_cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    core = model.module if hasattr(model, "module") else model
    named = [(name, p) for name, p in core.named_parameters() if p.requires_grad]
    params = [p for _name, p in named]
    groups = {
        "all": list(range(len(named))),
        "encoder": [i for i, (name, _p) in enumerate(named) if name.startswith("encoder.")],
        "decoder": [i for i, (name, _p) in enumerate(named) if name.startswith("decoder.")],
        "decoder_backbone": [
            i for i, (name, _p) in enumerate(named)
            if name.startswith("decoder.") and not name.startswith("decoder.head_")
        ],
        "trajectory_head": [
            i for i, (name, _p) in enumerate(named)
            if name.startswith("decoder.head_traj.")
        ],
    }

    costs = build_costs(pool.buckets, cfg.cost, device=str(dev), dtype=torch.float32)
    cost = costs[bucket]
    al = _load_al(cfg, ck)
    step = int(ck.get("step", 0))
    weights = _wtensors(
        cfg.schedule.multipliers(step, True), al.weights(), dev, pool.dtype,
        sched=cfg.schedule, joint_names=pool.cfg.joint_names,
        step=step, stage_a=True, pen=al.pen,
    )

    extra = ck.get("extra") or {}
    mpc_ring = _VoxelMPCParentRing(device=dev)
    mpc_ring.load_state_dict(extra["timing_critic"]["primary_mpc_ring"])
    suffix_ring = _VerifiedSuffixRing(
        capacity=int(cfg.suffix_replay_capacity), device=dev,
    )
    suffix_ring.load_state_dict(extra["suffix_replay"])
    suffix_fn = make_suffix_step(model, cost, cfg, compile_mode=None)

    reports = []
    for seed in range(args.seeds):
        gen = torch.Generator(device=dev)
        gen.manual_seed(104729 + seed)
        fresh_a = _fresh_batch(pool, bucket, args.rows, gen)
        fresh_b = _fresh_batch(pool, bucket, args.rows, gen)
        idx = mpc_ring.sample_idx(bucket, args.rows, generator=gen)
        mpc = mpc_ring.batch(bucket, idx, generator=gen, noise=False)
        mpc["mpc_init"] = torch.ones(args.rows, device=dev, dtype=torch.bool)
        suffix, fill = suffix_ring.sample(
            bucket, args.rows, min_parents=int(cfg.suffix_replay_min_parents),
            generator=gen,
        )
        if suffix is None:
            raise RuntimeError("verified suffix replay is empty")
        offs = suffix_offsets(
            suffix["hit"], suffix["active"], bucket,
            generator=gen, min_offset=1,
        )

        gf_a = _primary_grad(
            model, core, fresh_a, pool, cost, weights, cfg, gen, params,
        )
        gf_b = _primary_grad(
            model, core, fresh_b, pool, cost, weights, cfg, gen, params,
        )
        gm_primary, gm_tie, losses = _mpc_grads(
            model, core, mpc, pool, cost, weights, cfg, gen, params,
        )
        bellman_raw, tie_elems, q0_s, v0_s, child_init = _suffix_distill_loss(
            suffix, offs, bucket=bucket, encode_step=model.encode_context,
            suffix_fn=suffix_fn, pool=pool, microbatch=int(cfg.suffix_microbatch),
        )
        hit_child = torch.where(
            suffix["active"].bool(), suffix["hit"] - offs.unsqueeze(1),
            suffix["hit"],
        )
        masks_child = pool.masks_for(
            hit_child.clamp_min(0), suffix["active"], bucket,
        )
        shifted, valid = shift_plan(suffix["parent_traj"], offs)
        shifted_mask, _ = shift_plan(suffix["masks"].elem_mask, offs)
        tie_mask = (
            masks_child.elem_mask.to(shifted.dtype)
            * shifted_mask.to(shifted.dtype)
            * valid.unsqueeze(-1).to(shifted.dtype)
        )
        with torch.no_grad():
            child = model(
                child_init, torch.cat([q0_s, v0_s], dim=-1),
                model.encode_context(suffix["context"]), masks_child,
            ).traj
            residual = (child - shifted).abs()[tie_mask.bool()].float()
            residual_stats = {
                "elements": int(tie_elems),
                "mean_abs_rad": float(residual.mean()),
                "p50_abs_rad": float(residual.median()),
                "p90_abs_rad": float(torch.quantile(residual, 0.9)),
                "fraction_quadratic_huber": float((residual < 0.02).float().mean()),
                "offset_mean_wp": float(offs.float().mean()),
                "offset_p50_wp": float(offs.float().median()),
            }
        bellman = float(cfg.w_suffix_tie) * float(fill) * bellman_raw
        gb = _grads(bellman, params)
        gm = _add(gm_primary, gm_tie)
        greplay = _add(gm_primary, gm_tie, gb)

        vectors = {
            "fresh_a": gf_a,
            "fresh_b": gf_b,
            "mpc_primary": gm_primary,
            "mpc_tie": gm_tie,
            "mpc_combined": gm,
            "bellman": gb,
            "mpc_plus_bellman": greplay,
        }
        pairs = (
            ("fresh_a", "fresh_b"),
            ("fresh_a", "mpc_primary"),
            ("fresh_a", "mpc_tie"),
            ("fresh_a", "mpc_combined"),
            ("fresh_a", "bellman"),
            ("fresh_a", "mpc_plus_bellman"),
            ("mpc_primary", "mpc_tie"),
            ("mpc_primary", "bellman"),
            ("mpc_tie", "bellman"),
        )
        alignment = {
            f"{a}__{b}": {
                group: _alignment(vectors[a], vectors[b], take)
                for group, take in groups.items()
            }
            for a, b in pairs
        }
        reports.append({
            "seed": seed,
            "losses": {
                **losses,
                "bellman_raw": float(bellman_raw.detach()),
                "bellman_weighted": float(bellman.detach()),
                "suffix_fill": float(fill),
                "bellman_residual": residual_stats,
            },
            "alignment": alignment,
        })
        print(
            f"seed={seed} fresh/fresh={alignment['fresh_a__fresh_b']['all']['cosine']:+.3f} "
            f"fresh/mpc={alignment['fresh_a__mpc_combined']['all']['cosine']:+.3f} "
            f"fresh/bellman={alignment['fresh_a__bellman']['all']['cosine']:+.3f} "
            f"fresh/all-replay={alignment['fresh_a__mpc_plus_bellman']['all']['cosine']:+.3f}",
            flush=True,
        )

    summary = {}
    for pair in reports[0]["alignment"]:
        summary[pair] = {}
        for group in groups:
            vals = [r["alignment"][pair][group]["cosine"] for r in reports]
            vals = [v for v in vals if v is not None]
            ratios = [r["alignment"][pair][group]["norm_b_over_a"] for r in reports]
            ratios = [v for v in ratios if v is not None]
            summary[pair][group] = {
                "cosine_mean": float(np.mean(vals)),
                "cosine_min": float(np.min(vals)),
                "cosine_max": float(np.max(vals)),
                "norm_b_over_a_mean": float(np.mean(ratios)),
            }
    out = {
        "meta": {
            "checkpoint": str(Path(args.ckpt).resolve()),
            "checkpoint_step": step,
            "pool_cache": str(Path(args.pool_cache).resolve()),
            "rows_per_source": args.rows,
            "seeds": args.seeds,
            "bucket": bucket,
            "weights": {
                "w_cost": float(cfg.w_cost),
                "w_mpc_init_tie": float(cfg.w_mpc_init_tie),
                "w_suffix_tie": float(cfg.w_suffix_tie),
            },
            "note": "Full-fp32 actor gradients; fresh/fresh is the sampling-noise baseline.",
        },
        "summary": summary,
        "seeds": reports,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
