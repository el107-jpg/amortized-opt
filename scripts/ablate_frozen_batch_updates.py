#!/usr/bin/env python
"""Controlled early-actor ablation of repeated updates on frozen collocations.

All arms start with the same checkpoint/model/AdamW moments and consume the
same ordered bank of independently sampled Phase-2 batches. An arm performs
K newly evaluated forward/backward/optimizer passes on each bank batch before
advancing. Physics + auxiliary-head loss is identical across arms; curriculum
selection, AL multipliers, and loss schedule are frozen at the checkpoint.
This isolates repeat count, not the complete online MPC/critic training loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from probe_row_grad_stratify import _load_al, _scenario_from_ckpt
from train_phase2 import _load_npz_buckets
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool
from amortized_opt.train_cost import (
    LossScales,
    _detach_batch,
    _sample_fresh_batch,
    _wtensors,
    build_costs,
    build_optimizer,
    correct_quintic_init_arms,
    effective_hits,
    eval_subset,
    group_lengths,
    make_cost_step,
    phase2_config_from_checkpoint,
    resolve_idle_cylinder_library,
    rolling_eval,
)
from amortized_opt.idle_cylinder import derive_cylinder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pool-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    p.add_argument("--unique-batches", type=int, default=8)
    p.add_argument("--bank-in", type=Path,
                   help="load an existing frozen bank instead of sampling again")
    p.add_argument("--exposures-per-batch", type=int,
                   help="give every bank batch this many total updates; "
                        "K changes grouping, not compute or data exposure")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--heldout-size", type=int, default=64)
    p.add_argument("--bucket", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval-grid", action="store_true",
                   help="run the fixed 81-case rolling eval after each arm")
    p.add_argument("--score-every-updates", type=int,
                   help="score training and heldout batches only when this many "
                        "additional updates have completed (and at the end)")
    return p.parse_args()


def _freeze_batch(pool, cost, cfg, bucket, bs, gen):
    batch = _sample_fresh_batch(
        pool, bucket, bs, generator=gen,
        kind_fracs=cfg.effective_kind_fracs(pool), live_post_hit=None,
        short_wp=cfg.short_wp, short_boost=cfg.short_boost, noise=True,
    )
    # The normal training path applies this deterministic IK seed correction
    # once after selecting timings. It must be frozen too, not rerun per arm.
    active = batch["active"] & ~batch["mpc_init"].unsqueeze(-1)
    with torch.no_grad():
        batch["init"] = correct_quintic_init_arms(
            cost.fk, batch["init"], batch["q0"], batch["hit"],
            batch["context"].goal_pos, active=active, steps=1,
            q_bounds=(pool.q_lo, pool.q_hi), joint_names=cost.joint_names,
            qd0=batch["v0"], dt=float(pool.cfg.dt),
        )
    return _detach_batch(batch)


def _loss_inputs(batch, bucket):
    hit = effective_hits(batch["hit"], batch["active"], bucket)
    return replace(batch["strike"], hit_index=hit), group_lengths(hit)


@torch.no_grad()
def _score_batch(model, cost, batch, bucket, weights):
    was_training = model.training
    model.eval()
    try:
        enc = model.encode_context(batch["context"])
        traj = model(batch["init"], batch["state"], enc, batch["masks"]).traj
        strike, lengths = _loss_inputs(batch, bucket)
        ct = cost.forward(batch["q0"], traj, batch["v0"], strike,
                          lengths=lengths, weights=weights)
        return {
            "cost": float(ct.total.mean()),
            "terminal": float((ct.terms["terminal"] + ct.terms["terminal_aim"]).mean()),
            "contact_mm": 1000.0 * float(ct.diagnostics["contact_pos_err"].mean()),
            "torque": float(ct.terms["torque"].mean()),
            "collision": float(ct.terms["collision"].mean()),
            "joint_box": float(ct.terms["joint_box"].mean()),
        }
    finally:
        model.train(was_training)


def _grid_cases(cfg, pool):
    scorer = A.TrajectoryScorer(
        device="cpu",
        track_torque=bool(
            cfg.cost.w_track_torque > 0.0 or cfg.cost.w_track_torque_peak > 0.0
        ),
        soft_wrist_effort_scale=float(cfg.cost.soft_wrist_effort_scale),
    )
    cases = eval_subset(A.generate_grid(
        seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
        start_above_drums=("snare", "floor_tom"),
        start_not_under_drums=True, start_clearance_m=0.02,
    ), 81)
    return cases, scorer, A.DrumKit.from_layout(), A.GoalPoseLibrary()


def main():
    args = parse_args()
    if not args.repeats or any(k < 1 for k in args.repeats):
        raise SystemExit("--repeats must contain positive integers")
    if args.unique_batches < 1 or args.batch_size < 1 or args.lr <= 0.0:
        raise SystemExit("batch count, batch size, and learning rate must be positive")
    if args.exposures_per_batch is not None and args.exposures_per_batch < 1:
        raise SystemExit("--exposures-per-batch must be positive")
    if args.score_every_updates is not None and args.score_every_updates < 1:
        raise SystemExit("--score-every-updates must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = phase2_config_from_checkpoint(ck, device="cpu", compile_model=False)
    cfg.compile_cost = False
    cfg.compile_aux = False
    cfg.grad_balance = ""
    cfg.curvature_direct_frac = 0.0
    cfg.dls_terminal_frac = 0.0
    cfg.torque_grad_cap = 0.0
    if cfg.waist_sp_from_head or cfg.waist_pr_sp_from_head:
        raise SystemExit("this fixed-bank ablation requires no encoder waist SP heads")
    buckets, stats = _load_npz_buckets(str(args.pool_cache))
    pool = Phase2Pool(
        buckets, cfg=_scenario_from_ckpt(ck), noise=Phase2Noise(),
        device="cpu", stats=stats,
    )
    if args.bucket not in pool.buckets:
        raise SystemExit(f"bucket {args.bucket} not in {pool.buckets}")
    cost = build_costs([args.bucket], cfg.cost, device="cpu",
                       dtype=torch.float32)[args.bucket]
    idle_lib = resolve_idle_cylinder_library(
        cfg.resolved_idle_cylinder_path(), joint_names=pool.cfg.joint_names,
    )
    if idle_lib is not None:
        try:
            cyl = derive_cylinder(A.DrumKit.from_layout(), cost.fk)
        except Exception:
            cyl = idle_lib.cylinder
        cost.set_idle_cylinder(cyl)
    al = _load_al(cfg, ck)
    start_step = int(ck.get("step") or 0)
    weights = _wtensors(
        cfg.schedule.multipliers(start_step, False), al.weights(), device,
        torch.float32, sched=cfg.schedule, joint_names=cost.joint_names,
        step=start_step, stage_a=False, pen=al.pen,
    )
    if args.bank_in is not None:
        frozen_data = torch.load(args.bank_in, map_location="cpu", weights_only=False)
        bank, heldout = frozen_data["batches"], frozen_data["heldout"]
        if len(bank) != args.unique_batches:
            raise SystemExit("--unique-batches does not match --bank-in")
        if any(int(b["q0"].shape[0]) != args.batch_size for b in bank):
            raise SystemExit("--batch-size does not match --bank-in")
        if int(heldout["q0"].shape[0]) != args.heldout_size:
            raise SystemExit("--heldout-size does not match --bank-in")
    else:
        gen = torch.Generator(device="cpu").manual_seed(args.seed + 7919)
        bank = [_freeze_batch(pool, cost, cfg, args.bucket, args.batch_size, gen)
                for _ in range(args.unique_batches)]
        heldout = _freeze_batch(pool, cost, cfg, args.bucket, args.heldout_size, gen)
    model_cfg = A.model_config_from_checkpoint(ck)
    cases = scorer = kit = library = None
    if args.eval_grid:
        cases, scorer, kit, library = _grid_cases(cfg, pool)
    result = {
        "checkpoint": str(args.ckpt), "checkpoint_step": start_step,
        "pool_cache": str(args.pool_cache), "seed": args.seed,
        "unique_batches": args.unique_batches, "batch_size": args.batch_size,
        "heldout_size": args.heldout_size, "bucket": args.bucket,
        "learning_rate": args.lr, "repeats": args.repeats,
        "exposures_per_batch": args.exposures_per_batch,
        "score_every_updates": args.score_every_updates,
        "loss_scope": (
            "checkpoint FusedCost + auxiliary heads, frozen AL/schedule, "
            "AdamW with checkpoint moments and fixed LR; no SaLSa, "
            "MPC/replay, timing critic, TC/BPTT/suffix, DLS, torque cap, or curvature"
        ),
        "arms": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Persist exact sampled rows once, including their noise and corrected init.
    bank_path = args.bank_in or args.out.with_suffix(".bank.pt")
    if args.bank_in is None:
        torch.save({"batches": bank, "heldout": heldout}, bank_path)
    result["bank_path"] = str(bank_path)
    for k in args.repeats:
        torch.manual_seed(args.seed)
        model = A.build_model(model_cfg, device)
        model.load_state_dict(ck["model"])
        model.train()
        optimizer = build_optimizer(model, cfg)
        # load_state_dict can alias CPU moment tensors when dtype/device already
        # match. Without the deep copy, earlier arms silently change the later
        # arms' starting Adam moments even though their model weights reset.
        optimizer.load_state_dict(copy.deepcopy(ck["optimizer"]))
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        step_fn = make_cost_step(model, cost, LossScales(device, torch.float32),
                                 cfg, compile_mode=None)
        history = []
        updates = 0
        next_score_update = args.score_every_updates
        begun = time.perf_counter()
        train_elapsed_s = 0.0
        visits = []
        if args.exposures_per_batch is None:
            visits = [(bi, k) for bi in range(len(bank))]
        else:
            exposures = [0] * len(bank)
            while min(exposures) < args.exposures_per_batch:
                for bi in range(len(bank)):
                    inner_count = min(
                        k, args.exposures_per_batch - exposures[bi],
                    )
                    if inner_count:
                        visits.append((bi, inner_count))
                        exposures[bi] += inner_count
        for visit_i, (bi, inner_count) in enumerate(visits):
            frozen = bank[bi]
            batch = _detach_batch(frozen)
            score_this = (
                next_score_update is None
                or updates + inner_count >= next_score_update
                or visit_i + 1 == len(visits)
            )
            before = (_score_batch(model, cost, batch, args.bucket, weights)
                      if score_this else None)
            strike, lengths = _loss_inputs(batch, args.bucket)
            train_begun = time.perf_counter()
            for inner in range(inner_count):
                optimizer.zero_grad(set_to_none=True)
                enc = model.encode_context(batch["context"])
                loss, _metrics, _traj, _gate = step_fn(
                    batch["init"], batch["state"], batch["q0"],
                    batch["v0"], enc, batch["masks"], strike, lengths,
                    weights, torch.zeros((), device=device),
                )
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip,
                )
                if not torch.isfinite(loss) or not torch.isfinite(gnorm):
                    raise RuntimeError(f"nonfinite loss/grad: K={k} batch={bi} inner={inner}")
                optimizer.step()
                updates += 1
            train_elapsed_s += time.perf_counter() - train_begun
            if not score_this:
                continue
            after = _score_batch(model, cost, batch, args.bucket, weights)
            held = _score_batch(model, cost, heldout, args.bucket, weights)
            if next_score_update is not None:
                while next_score_update <= updates:
                    next_score_update += args.score_every_updates
            row = {
                "batch_index": bi, "visit_index": visit_i,
                "inner_steps": inner_count, "updates": updates,
                "batch_before": before, "batch_after": after,
                "heldout": held, "elapsed_s": time.perf_counter() - begun,
            }
            history.append(row)
            print(
                f"[K={k} visit={visit_i+1}/{len(visits)} bank={bi+1}/{len(bank)} "
                f"updates={updates}] "
                f"train {before['cost']:.2f}->{after['cost']:.2f} "
                f"heldout {held['cost']:.2f} contact {held['contact_mm']:.1f} mm "
                f"elapsed={row['elapsed_s']:.1f}s", flush=True,
            )
        grid = None
        if args.eval_grid:
            grid = rolling_eval(
                model, cases, scorer, device=device, kit=kit, library=library,
                buckets=pool.buckets, cost=cost,
                num_drums=model_cfg.num_drums,
                num_future=model_cfg.num_future,
            )
            print(f"[K={k} eval] {grid['accepted']}/{grid['n']} "
                  f"hit={grid['ok_hit']:.3f} contact={grid['contact_pos_err_mm_p50']:.1f} mm",
                  flush=True)
        result["arms"][str(k)] = {
            "updates": updates, "history": history, "grid": grid,
            "elapsed_s": time.perf_counter() - begun,
            "train_elapsed_s": train_elapsed_s,
        }
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
