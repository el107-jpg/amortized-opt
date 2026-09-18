#!/usr/bin/env python
"""Vector torque-gradient attribution on a training-like batch.

Answers: do high-violation replay rows reinforce or cancel the ACTUAL
batch-mean torque gradient?

    python scripts/probe_row_grad_stratify.py \\
        --ckpt runs/rhythm_bellman_nf1_ls_full_rnea_replay_dw/last.pt \\
        --pool-cache runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \\
        --device cuda:0 --prime-roller-ticks 200 --use-replay --batch-size 64

Repeat representative draws explicitly with different seeds, for example::

    for seed in 0 1 2; do
      python scripts/probe_row_grad_stratify.py ... --batch-size 64 \\
          --seed "$seed" --json-out "attribution_seed${seed}.json"
    done

Loads checkpoint AL state, optionally primes the MPC roller to fill replay,
draws one bucket-homogeneous batch, and reports exact vector-summed gradient
metrics for harvest/current-violation quantiles and top-tail counterfactuals.
Every row in the sampled diagnostic batch is included; ``--batch-size`` does
not claim to reproduce a larger configured training batch.  Gradients are
reduced groupwise rather than kept as a per-row tensor bank.

This remains pure torque-gradient attribution.  A checkpoint stores Adam
moments, but the staged executed direction also needs the next full clipped
training gradient (cost + aux + TC/BPTT/suffix/ties) and line-search state.
It cannot be reconstructed reliably from this torque-only diagnostic batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from train_phase2 import _load_npz_buckets
from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool, Phase2ScenarioConfig
from amortized_opt.train_cost import (
    MPCRoller,
    ReplayPool,
    _dataclass_from_mapping,
    _sample_fresh_batch,
    _wtensors,
    build_costs,
    correct_quintic_init_arms,
    effective_hits,
    forward_torque_viol_per_row,
    group_lengths,
    make_encode_step,
    migrate_al_state,
    phase2_config_from_checkpoint,
    prime_replay_from_roller,
    resolve_idle_cylinder_library,
    seed_waist_pr_to_sp,
)
from amortized_opt.torque_grad_attribution import (
    measure_batch_torque_gradient_attribution,
    replay_sampling_probabilities,
    summarize_replay_sampling,
)
from amortized_opt.waist_aim import WaistAimConfig
from amortized_opt.rhythm import KitRandomization
from amortized_opt.idle_cylinder import derive_cylinder


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool-cache", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bucket", type=int, default=None,
                   help="horizon bucket (default: largest in pool)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None,
                   help="diagnostic sample size (default: configured training batch)")
    p.add_argument("--step", type=int, default=None,
                   help="training step for schedule/replay_frac (default: ckpt step)")
    p.add_argument("--prime-roller-ticks", type=int, default=200,
                   help="MPC roller ticks before sampling replay (0 = skip)")
    p.add_argument("--use-replay", action="store_true",
                   help="force a replay batch (requires primed replay)")
    p.add_argument("--grad-rows", type=int, default=None,
                   help="deprecated; exact vector attribution always uses the full batch")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--top-fracs", type=float, nargs="+", default=(0.10, 0.25),
                   help="top violation fractions to remove counterfactually")
    p.add_argument("--json-out", default=None)
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
    al_state = ck.get("al")
    if not al_state:
        return al
    al.load_state_dict(migrate_al_state(al_state))
    fresh = AugmentedLagrangian(cfg.al)
    al.rho = dict(fresh.rho)
    al.lam_max = dict(fresh.lam_max)
    al.lam_min = dict(fresh.lam_min)
    al.target = dict(fresh.target)
    al.kp = dict(fresh.kp)
    al.kd = dict(fresh.kd)
    al.pen_min = dict(fresh.pen_min)
    al.pen_max = dict(fresh.pen_max)
    al.cfg = fresh.cfg
    if not (al_state or {}).get("pen"):
        al.pen = dict(fresh.pen)
    for g in al.groups:
        al.lam[g] = float(min(max(al.lam[g], al.lam_min[g]), al.lam_max[g]))
        al.pen[g] = float(min(max(al.pen[g], al.pen_min[g]), al.pen_max[g]))
    return al


def _resolve_batch_sizes(cfg, bucket: int, override: int | None) -> tuple[int, int]:
    """Return configured and sampled diagnostic sizes without mutating cfg."""
    configured = int(cfg.batch_for(bucket))
    diagnostic = configured if override is None else int(override)
    if diagnostic <= 0:
        raise ValueError(
            f"diagnostic batch size must be positive, got {diagnostic}"
        )
    return configured, diagnostic


def _prepare_batch(model, encode_step, pool, replay, costs, cfg, al, bucket, step,
                   gen, rng, *, use_replay: bool, base,
                   diagnostic_batch_size: int | None = None):
    device = pool.device
    dtype = pool.dtype
    configured_bs, bs = _resolve_batch_sizes(
        cfg, bucket, diagnostic_batch_size
    )
    stage_a = False
    replay_loc = None
    viol_harvest = None

    if use_replay:
        if bucket not in replay.data or replay.size(bucket) < bs:
            raise RuntimeError(
                f"replay bucket {bucket} has {replay.size(bucket)} rows, need {bs}; "
                f"raise --prime-roller-ticks"
            )
        idx = replay.sample_idx(
            bucket, bs, generator=gen,
            short_wp=cfg.short_wp,
            short_boost=cfg.short_boost,
            viol_downweight=cfg.replay_viol_downweight,
            viol_uniform_mix=cfg.replay_viol_uniform_mix,
            stratify=bool(cfg.replay_proficiency),
            frac_prof=float(cfg.replay_frac_prof),
            frac_new=float(cfg.replay_frac_new),
            ema_nmin=int(cfg.frontier_ema_nmin),
            ema_prof=float(cfg.frontier_ema_prof),
        )
        replay_loc = idx
        batch = replay.batch(bucket, idx, generator=gen, noise=False)
        batch["replay_loc"] = idx
        batch["mpc_init"] = torch.ones(bs, device=device, dtype=torch.bool)
        if "viol_score" in replay.data[bucket]:
            viol_harvest = replay.data[bucket]["viol_score"][idx].detach().float().cpu().numpy()
    else:
        batch = _sample_fresh_batch(
            pool, bucket, bs, generator=gen,
            kind_fracs=cfg.effective_kind_fracs(pool),
            live_post_hit=None, short_wp=cfg.short_wp,
            short_boost=cfg.short_boost, noise=True,
        )
        batch["mpc_init"] = torch.zeros(bs, device=device, dtype=torch.bool)

    hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
    lengths = group_lengths(hit_eff)
    strike = __import__("dataclasses").replace(batch["strike"], hit_index=hit_eff)
    w = _wtensors(
        cfg.schedule.multipliers(step, stage_a), al.weights(), device, dtype,
        sched=cfg.schedule, joint_names=base.joint_names,
        step=step, stage_a=stage_a, pen=al.pen,
    )

    amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.amp_dtype)
    ctxm = (
        torch.autocast(device_type="cuda", dtype=amp)
        if amp is not None and str(device).startswith("cuda")
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with ctxm:
        enc = encode_step(batch["context"])
        if cfg.waist_sp_from_head:
            sp = enc.waist_yaw_sp.float()
            strike = __import__("dataclasses").replace(
                strike, waist_yaw_ref=sp.unsqueeze(1).expand(-1, bucket),
            )
        if cfg.waist_pr_sp_from_head:
            pr = enc.waist_pr_sp.float()
            strike = __import__("dataclasses").replace(
                strike, waist_pr_ref=pr.unsqueeze(1).expand(-1, bucket, -1),
            )
        if enc.waist_pr_sp is not None:
            batch["init"] = seed_waist_pr_to_sp(
                batch["init"], batch["q0"], batch["v0"], enc.waist_pr_sp,
                joint_names=base.joint_names,
                dt=float(getattr(pool.cfg, "dt", 0.02)),
                ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
            )
        if not use_replay:
            fresh_active = batch["active"] & ~batch["mpc_init"].unsqueeze(-1)
            batch["init"] = correct_quintic_init_arms(
                costs[bucket].fk, batch["init"], batch["q0"], batch["hit"],
                batch["context"].goal_pos,
                active=fresh_active, steps=1,
                q_bounds=(pool.q_lo, pool.q_hi),
                joint_names=base.joint_names, qd0=batch["v0"],
                dt=float(getattr(pool.cfg, "dt", 0.02)),
            )

    meta = {
        "bucket": bucket,
        # ``batch_size`` is retained for readers of the first diagnostic
        # schema; the two explicit fields prevent a B=64 probe being mistaken
        # for the configured production B=512 training step.
        "batch_size": bs,
        "configured_training_batch_size": configured_bs,
        "diagnostic_sampled_batch_size": bs,
        "batch_size_overridden": bs != configured_bs,
        "use_replay": use_replay,
        "harvest_viol_score_available": viol_harvest is not None,
        "replay_frac_at_step": float(cfg.replay_frac(step)),
        "viol_downweight": float(cfg.replay_viol_downweight),
        "viol_sampling_beta": float(cfg.replay_viol_downweight),
        "viol_uniform_mix": float(cfg.replay_viol_uniform_mix),
        "replay_proficiency": bool(cfg.replay_proficiency),
    }
    if viol_harvest is not None:
        meta["viol_harvest_p50"] = float(np.percentile(viol_harvest, 50))
        meta["viol_harvest_p90"] = float(np.percentile(viol_harvest, 90))
        meta["viol_harvest_max"] = float(viol_harvest.max())
    if replay_loc is not None:
        meta["replay_loc_min"] = int(replay_loc.min())
        meta["replay_loc_max"] = int(replay_loc.max())
    return batch, enc, strike, lengths, w, meta, viol_harvest, replay_loc


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    step = int(args.step if args.step is not None else ck.get("step") or 0)
    cfg = phase2_config_from_checkpoint(ck, device=str(dev), compile_model=False)
    model_cfg = A.model_config_from_checkpoint(ck)
    scen = _scenario_from_ckpt(ck)

    buckets, stats = _load_npz_buckets(args.pool_cache)
    pool = Phase2Pool(
        buckets, cfg=scen, noise=Phase2Noise(), device=str(dev), stats=stats,
    )
    bucket = int(args.bucket or max(pool.buckets))
    if bucket not in pool.buckets:
        raise SystemExit(f"bucket {bucket} not in pool {pool.buckets}")

    model = A.build_model(model_cfg, dev)
    model.load_state_dict(ck["model"])
    model.train()

    costs = build_costs(pool.buckets, cfg.cost, device=str(dev), dtype=torch.float32)
    base = costs[max(costs)]
    al = _load_al(cfg, ck)
    encode_step = make_encode_step(model, compile_mode=None)

    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    idle_path = cfg.resolved_idle_cylinder_path()
    idle_lib = resolve_idle_cylinder_library(
        idle_path, joint_names=pool.cfg.joint_names,
    )
    idle_cyl = None
    if idle_lib is not None:
        try:
            idle_cyl = derive_cylinder(kit, base.fk)
        except Exception:
            idle_cyl = idle_lib.cylinder
        for c in costs.values():
            if hasattr(c, "set_idle_cylinder"):
                c.set_idle_cylinder(idle_cyl if idle_cyl is not None else idle_lib)

    want_waist_ref = cfg.cost.w_waist_aim > 0.0 and not cfg.waist_sp_from_head
    want_waist_pr_ref = bool(cfg.waist_pr_ema) and not cfg.waist_pr_sp_from_head
    replay = ReplayPool(
        pool, cfg.replay_capacity,
        buckets=[b for b in pool.buckets if b <= cfg.roller_horizon],
        recency_bias=cfg.recency_bias,
        viol_downweight=float(cfg.replay_viol_downweight),
        viol_uniform_mix=float(cfg.replay_viol_uniform_mix),
        proficiency=bool(cfg.replay_proficiency),
        ema_alpha=float(cfg.frontier_ema_alpha),
        waist_ref=want_waist_ref,
        waist_pr_ref=want_waist_pr_ref,
    )

    roller_clips = None
    src0 = next(iter(pool.data.values()))
    if "clip_i" in src0:
        from amortized_opt.rhythm_chain_data import roller_clip_list as _roller_clips

        roller_clips = _roller_clips(kit, library=library, cfg=pool.cfg)

    from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig

    sim = BatchedPDSim(
        PDSimConfig(dt=pool.cfg.dt, saturate_effort=bool(cfg.plant_saturate_effort)),
        device=str(dev), dtype=torch.float32, rnea=base.rnea,
    )
    roller = MPCRoller(
        pool, model=model, sim=sim, rnea=base.rnea, fk=base.fk,
        horizon=cfg.roller_horizon, slots=cfg.roller_slots,
        chain_frac=cfg.roller_chain_frac, bounce=cfg.bounce,
        device=str(dev), dtype=torch.float32, seed=cfg.seed + 1301,
        amp_dtype=None,
        regoal=cfg.roller_regoal, cmd_lead_s=cfg.cmd_lead_s,
        library=library, kit=kit,
        yaw_bins=cfg.roller_yaw_bins, max_age=cfg.roller_max_age,
        idle_library=idle_lib, idle_cylinder=idle_cyl,
        idle_settle_wp=cfg.idle_settle_wp,
        rest_ticks=cfg.roller_rest_ticks if idle_lib is not None else 0,
        waist_ref=want_waist_ref,
        n_shaft=costs[min(costs)].n_points - 2,
        waist_upright_ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
        clips=roller_clips,
        waist_pr_ema=want_waist_pr_ref,
        waist_pr_ema_alpha=float(cfg.waist_pr_ema_alpha),
        waist_pr_ema_leak=float(cfg.waist_pr_ema_leak),
    )

    prime_stats = {}
    if args.prime_roller_ticks > 0:
        prime_stats = prime_replay_from_roller(
            roller, replay, pool, costs, cfg, al, step,
            args.prime_roller_ticks,
        )
        print(f"[prime] {args.prime_roller_ticks} ticks -> "
              f"replay sizes {{{', '.join(f'{b}:{replay.size(b)}' for b in replay.data)}}}",
              flush=True)

    gen = torch.Generator(device=str(dev))
    gen.manual_seed(int(args.seed) + 7919)
    rng = np.random.default_rng(args.seed + 31)

    batch, enc, strike, lengths, w, meta, viol_harvest, replay_loc = _prepare_batch(
        model, encode_step, pool, replay, costs, cfg, al, bucket, step,
        gen, rng, use_replay=bool(args.use_replay), base=base,
        diagnostic_batch_size=args.batch_size,
    )

    sampling_rows = {}
    if replay_loc is not None:
        sampling_diag = replay_sampling_probabilities(
            replay, bucket, int(meta["batch_size"]),
            short_wp=cfg.short_wp,
            short_boost=cfg.short_boost,
            viol_downweight=cfg.replay_viol_downweight,
            viol_uniform_mix=cfg.replay_viol_uniform_mix,
            stratify=bool(cfg.replay_proficiency),
            frac_prof=float(cfg.replay_frac_prof),
            frac_new=float(cfg.replay_frac_new),
            ema_nmin=int(cfg.frontier_ema_nmin),
            ema_prof=float(cfg.frontier_ema_prof),
        )
        for key in (
            "raw_weight",
            "expected_draw_count",
            "expected_draw_fraction",
            "inclusion_probability",
        ):
            sampling_rows[key] = sampling_diag[key].index_select(0, replay_loc)
        meta["sampling"] = summarize_replay_sampling(
            sampling_diag, replay_loc,
        )

    labels = {}
    if viol_harvest is not None:
        labels["replay_viol_score"] = viol_harvest

    amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.amp_dtype)

    def fresh_torque_rows():
        # ``enc`` was created once in _prepare_batch and its graph has not been
        # consumed.  The decoder and cost are both fresh here, under the same
        # autocast policy as the training step.
        ctxm = (
            torch.autocast(device_type="cuda", dtype=amp)
            if amp is not None and dev.type == "cuda"
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with ctxm:
            dec = model(batch["init"], batch["state"], enc, batch["masks"])
            ct = costs[bucket].forward(
                batch["q0"], dec.traj, batch["v0"], strike,
                lengths=lengths, weights=w,
            )
        return (
            float(cfg.w_cost) * ct.terms["torque"],
            {"forward_torque_viol": forward_torque_viol_per_row(ct)},
        )

    report = measure_batch_torque_gradient_attribution(
        fresh_torque_rows,
        (p for p in model.parameters() if p.requires_grad),
        labels=labels,
        sampling=sampling_rows,
        n_bins=int(args.n_bins),
        top_fracs=tuple(float(x) for x in args.top_fracs),
    )
    report["meta"] = {
        **meta,
        "ckpt": str(args.ckpt),
        "step": step,
        "prime_stats": prime_stats,
        "attribution": (
            "exact for sampled diagnostic batch; weighted torque rows divided "
            "by diagnostic batch mean"
        ),
        "gradient_scope": "pure weighted torque term",
        "adam_executed_direction": {
            "available": False,
            "reason": (
                "checkpoint moments are insufficient: staging the next executed "
                "Adam direction requires the next full clipped training gradient "
                "and line-search state, not this torque-only diagnostic gradient"
            ),
        },
        "legacy_grad_rows_arg": args.grad_rows,
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
