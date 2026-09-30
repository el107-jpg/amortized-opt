#!/usr/bin/env python
"""Phase-1 warm start, APEX VARIANT: same recipe as ``train_phase1.py``, apex targets.

    python scripts/train_phase1_apex.py --device cuda:1 --steps 20000 --batch 1024 \
        --train-scenarios 560000 --val-scenarios 53000 --workers 64

Identical in every hyper-parameter to the ``phase1_main`` run (20 000-step hard
cap, batch 1024, bf16, ``torch.compile("default")``, buckets 25/50/100, aux
weight 0.1, lr 3e-4 with 500 warm-up steps).  The ONLY difference is the
imitation target:

* ``phase1_main`` -- target = straight-line quintic to a library terminal pose,
  init = that target + OU/white noise;
* here          -- a frozen start-aware MLP predicts the terminal pose; the
  init uses the low-acceleration direct path, and the target uses the same law
  on start -> apex and apex -> terminal, plus the existing OU/white noise mix.

Scenario counts default higher than phase 1's 450 k / 40 k because the apex
detour is longer and so trips the impossible-transition filter more often
(~45 % drop vs ~31 %); the defaults land the pool at roughly the same SIZE as
``phase1_main``'s so the two runs see the same amount of data.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.train_warmstart import (  # noqa: E402
    LossWeights,
    TrainConfig,
    imitation_report,
    train,
)
from amortized_opt.warmstart_apex import ApexConfig, generate_apex_pool  # noqa: E402
from amortized_opt.warmstart_data import NoiseConfig, ScenarioConfig  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=20_000, help="HARD cap (plan decision 5)")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--train-scenarios", type=int, default=560_000)
    p.add_argument("--val-scenarios", type=int, default=53_000)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--buckets", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=None, help="decoder width (default 384)")
    p.add_argument("--dec-layers", type=int, default=None, help="decoder depth (default 3)")
    p.add_argument("--d-ctx", type=int, default=None, help="context encoder width (default 256)")
    p.add_argument("--ctx-layers", type=int, default=None, help="context encoder depth (default 2)")
    p.add_argument("--ctx-heads", type=int, default=None, help="context encoder attention heads (default 8)")
    p.add_argument(
        "--no-ctx-self-attn", action="store_true",
        help="omit encoder self-attention but retain its per-token FFN",
    )
    p.add_argument("--dec-heads", type=int, default=None, help="decoder attention heads (default 8)")
    p.add_argument(
        "--dec-cross-attn-layers", type=int, nargs="+", default=None,
        metavar="LAYER",
        help="decoder layers that retain context cross-attention (default all); "
             "e.g. --dec-cross-attn-layers 0 omits it from every later layer",
    )
    p.add_argument("--dec-rope", action="store_true",
                   help="rotary position on decoder SELF-attention (see "
                        "ModelConfig.dec_rope); off by default")
    p.add_argument("--dec-plant-feats", action="store_true",
                   help="analytic Cartesian (tip/wrist/elbow) + knot-RNEA of "
                        "init_traj on decoder tokens (see "
                        "ModelConfig.dec_plant_feats); off by default")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-kit-randomization", action="store_true")
    # ---- the idle arm (same flags, same meanings, as train_phase1.py) ------
    p.add_argument("--no-idle-home", action="store_true",
                   help="target the idle arm at its START pose instead of the "
                        "park posture, and leave the IDLE context token empty")
    p.add_argument("--idle-home", default=None,
                   help="path to an idle-home table (default: "
                        "assets/idle_home_poses.json when it exists)")
    p.add_argument("--idle-settle-wp", type=int, default=10)
    p.add_argument("--rest-frac", type=float, default=0.25,
                   help="fraction of BOTH-hands-idle draws kept instead of "
                        "rejected.  0 reproduces the old distribution")
    p.add_argument("--waist-upright-ease-wp", type=int, default=None,
                   help="seed AND target pitch/roll ease from the sampled lean "
                        "to upright (0 rad) over this many waypoints, then "
                        "hold (try 5).  Default None keeps the old "
                        "hold-the-measured-lean data bit for bit -- which is "
                        "the 'stay leaned' prior the phase-2 upright well has "
                        "to fight")
    p.add_argument("--hit-time-jitter-s", type=float, default=0.0,
                   help="uniform +- jitter on each clip hit's time at snapshot "
                        "time, seconds (unison-coherent; see goal_snapshot)")
    p.add_argument("--hit-loudness-jitter", type=float, default=0.0,
                   help="uniform +- jitter on each hit's 0-1 loudness before "
                        "the speed mapping")
    p.add_argument("--waist-aim-future-tau", type=float, default=1.0,
                   help="time constant (s) of the FUTURE-aware waist aim law "
                        "(WaistAimConfig.future_tau): the imitation target's "
                        "yaw ramp and the aim_yaw token both aim at each hand's "
                        "active goal AND its next hits, weighted by time to "
                        "hit.  0 restores the current-goals-only law bit for "
                        "bit.  Must match the phase-2 setting -- phase 2 warm "
                        "starts from this checkpoint's head")
    p.add_argument("--waist-sp-head", action="store_true",
                   help="learned waist-yaw set-point head on the encoder "
                        "(docs/waist_sp_head_plan.md).  Phase 1 needs NO loss "
                        "change: the head is a zero-init residual on the "
                        "analytic aim_yaw the imitation targets' yaw ramp "
                        "already follows, and it trains through the decoder's "
                        "conditioning token")
    p.add_argument("--waist-pr-sp-head", action="store_true",
                   help="learned waist pitch/roll set-point head on the encoder "
                        "(docs/waist_pr_sp_head_plan.md).  Zero-init residual "
                        "on upright; trains through the conditioning token.  "
                        "Pair with --waist-upright-ease-wp so phase-1 targets "
                        "are not 'keep the measured lean'")
    p.add_argument("--waist-follow-frac", type=float, default=0.0,
                   help="REMOVED. Non-zero exits; use --w-waist-gain")
    p.add_argument("--waist-follow-jitter", type=float, default=0.3)
    p.add_argument("--waist-pr-follow-jitter", type=float, default=0.1)
    p.add_argument("--waist-hold-wp", type=int, default=10)
    p.add_argument("--w-waist-mimic", type=float, default=0.0,
                   help="Huber weight of encoder set-point vs aim / upright "
                        "(try 0.1 with heads)")
    p.add_argument("--w-waist-gain", type=float, default=0.0,
                   help="identity Jacobian (∂ hold / ∂ sp − 1)² (try 1.0 with heads). "
                        "No token jitter, no rewritten target")
    p.add_argument("--waist-gain-frac", type=float, default=1.0,
                   help="fraction of each batch used for the Jacobian VJP")
    p.add_argument("--no-idle-token", action="store_true",
                   help="keep the idle-arm targets but drop the encoder's IDLE "
                        "context token")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out", default=None, help="run directory (default runs/phase1_apex)")
    p.add_argument("--pool-device", default=None)
    p.add_argument(
        "--terminal-head-ckpt",
        default="runs/terminal_pose/start_collision_20260926/start_k1_10k.pt",
        help="pretrained start-aware terminal MLP (loaded frozen; gradients stop at its output)",
    )
    # apex knobs
    p.add_argument("--apex-height", type=float, default=0.30)
    p.add_argument("--apex-radius", type=float, default=0.15)
    p.add_argument("--apex-half-height", type=float, default=0.10)
    p.add_argument("--apex-iters", type=int, default=60)
    p.add_argument("--apex-seeds", type=int, default=3)
    p.add_argument("--terminal-mode", default="nearest_apex",
                   choices=["nearest_apex", "nearest_yaw"])
    # init channel (Eric, 2026-08-10 update 1)
    p.add_argument("--init-mode", default="straight",
                   choices=["mix", "straight", "apex"],
                   help="decoder input channel (default: terminal-head direct path; "
                        "mix/apex remain available as diagnostics)")
    p.add_argument("--init-mix", type=float, default=0.5,
                   help="fraction of samples initialized from the apex path")
    p.add_argument("--white-noise", action="store_true",
                   help="i.i.d. Gaussian inits at matched variance (OU off). "
                        "Implemented but NOT part of the sanctioned experiment "
                        "-- Eric cancelled the white-noise arm on 2026-08-10.")
    p.add_argument("--aux-torque-tracking", action="store_true",
                   help="train head_torque on the DEMANDED PD tracking torque "
                        "|tau_rnea + tau_fb| (what w_track_torque prices) instead "
                        "of the RNEA segment max. OFF by default: it redefines "
                        "the aux target, so every earlier checkpoint's torque "
                        "head means the other thing.")
    return p.parse_args(argv)


def _fmt_stats(name: str, pool) -> str:
    s = pool.stats
    pr = np.asarray(s.get("peak_rate_ratio", []), dtype=float)
    pi = np.asarray(s.get("peak_rate_ratio_init", []), dtype=float)
    ao = np.asarray(s.get("apex_outside_m", []), dtype=float)
    dz = np.asarray(s.get("apex_tip_dz_m", []), dtype=float)
    q = lambda a, p: (float(np.percentile(a, p)) if a.size else float("nan"))  # noqa: E731
    solved = max(int(s.get("apex_solved", 0)), 1)
    return (
        f"[data] {name}: {len(pool)} samples, buckets {pool.distribution()}; "
        f"accepted {s['accepted']}/{s['requested']} "
        f"(drop {100 * (1 - s['accepted'] / max(s['requested'], 1)):.1f}%: "
        f"|qd| {s['reject_velocity']}, idle {s['reject_idle']}); "
        f"dual-arm {s['dual_active']} (asym {s['asymmetric']})\n"
        f"[apex] {name}: solved {s.get('apex_solved', 0)} "
        f"(no room {s.get('apex_no_room', 0)}, dropped for missing the cylinder "
        f"{s.get('apex_rejected_outside', 0)}); collision-free "
        f"{s.get('apex_collision_free', 0)}/{solved} "
        f"({100 * s.get('apex_collision_free', 0) / solved:.1f}%); "
        f"cylinder miss p50={1000 * q(ao, 50):.2f} mm p90={1000 * q(ao, 90):.1f} mm, "
        f"<=1 cm {float((ao <= 1e-2).mean()) if ao.size else float('nan'):.3f}; "
        f"apex height p10/p50/p90 = {q(dz, 10):.3f}/{q(dz, 50):.3f}/{q(dz, 90):.3f} m; "
        f"terminal moved off the yaw pick "
        f"{s.get('terminal_moved', 0)}/{max(s.get('terminal_seen', 1), 1)}\n"
        f"[rate] {name}: peak|qd|/limit -- apex target p50={q(pr, 50):.2f} "
        f"p99={q(pr, 99):.2f} | straight init p50={q(pi, 50):.2f} p99={q(pi, 99):.2f}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.out) if args.out else root / "runs" / "phase1_apex"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}", flush=True)

    scen = ScenarioConfig(
        waist_cfg=A.WaistAimConfig(future_tau=float(args.waist_aim_future_tau)),
        horizon=int(args.horizon),
        buckets=tuple(args.buckets),
        randomize_kit=not args.no_kit_randomization,
        idle_home_path=None if args.no_idle_home else (args.idle_home or "default"),
        idle_settle_wp=int(args.idle_settle_wp),
        rest_frac=float(args.rest_frac),
        waist_upright_ease_wp=args.waist_upright_ease_wp,
        hit_time_jitter_s=float(args.hit_time_jitter_s),
        hit_loudness_jitter=float(args.hit_loudness_jitter),
    )
    print(f"[data] waist aim: future_tau={scen.waist_cfg.future_tau} s "
          f"(0 = current goals only)", flush=True)
    acfg = ApexConfig(
        height=float(args.apex_height), radius=float(args.apex_radius),
        half_height=float(args.apex_half_height), iters=int(args.apex_iters),
        n_seeds=int(args.apex_seeds), terminal_mode=str(args.terminal_mode),
    )
    noise = NoiseConfig()
    pool_dev = args.pool_device or args.device

    # Encoder and frozen terminal MLP are the two pre-decoder branches.  Build
    # them before the pool so the terminal head can replace the old library
    # terminal before apex auxiliary labels are generated.
    model_cfg = A.ModelConfig(
        max_horizon=int(args.horizon), num_drums=scen.num_drums,
        num_future=scen.num_future,
        terminal_head=True,
        idle_home_token=(scen.idle_home_path is not None) and not args.no_idle_token,
        waist_sp_head=bool(args.waist_sp_head),
        waist_pr_sp_head=bool(args.waist_pr_sp_head),
        dec_cross_attn_layers=(
            None if args.dec_cross_attn_layers is None
            else tuple(args.dec_cross_attn_layers)
        ),
        ctx_self_attn=not bool(args.no_ctx_self_attn),
    )
    for attr in (
        "d_model", "dec_layers", "dec_heads", "d_ctx", "ctx_layers", "ctx_heads"
    ):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(model_cfg, attr, val)
    if args.dec_rope:
        model_cfg.dec_rope = True
    if args.dec_plant_feats:
        model_cfg.dec_plant_feats = True
    model = A.build_model(model_cfg, dev)
    terminal_ckpt = Path(args.terminal_head_ckpt)
    if not terminal_ckpt.is_absolute():
        terminal_ckpt = root / terminal_ckpt
    model.load_terminal_head(terminal_ckpt, map_location=dev)
    print(f"[terminal] frozen head <- {terminal_ckpt}", flush=True)
    print(f"[model] {A.param_counts(model)}", flush=True)
    pool_initializer = model.terminal_initializer
    if torch.device(pool_dev) != dev:
        pool_initializer = copy.deepcopy(pool_initializer).to(pool_dev)

    fk = A.ChainFK(device=pool_dev, dtype=torch.float32)
    rnea = A.BatchedRNEA.from_urdf(with_armature=True, device=pool_dev, dtype=torch.float32)

    t0 = time.perf_counter()
    print(f"[data] generating {args.train_scenarios} APEX train scenarios "
          f"({args.workers} workers) ...", flush=True)
    train_pool = generate_apex_pool(
        args.train_scenarios, seed=args.seed, cfg=scen, apex_cfg=acfg, noise=noise,
        split="train", workers=args.workers,
        threads_per_worker=int(args.threads_per_worker),
        device=pool_dev, fk=fk, rnea=rnea,
        init_mode=str(args.init_mode), init_mix=float(args.init_mix),
        white_noise=bool(args.white_noise),
        aux_torque_tracking=bool(args.aux_torque_tracking),
        terminal_initializer=pool_initializer,
    )
    print(f"[data] generating {args.val_scenarios} APEX val scenarios ...", flush=True)
    val_pool = generate_apex_pool(
        args.val_scenarios, seed=args.seed + 991_1993, cfg=scen, apex_cfg=acfg,
        noise=noise, split="eval", workers=max(args.workers // 4, 1),
        threads_per_worker=int(args.threads_per_worker),
        device=pool_dev, fk=fk, rnea=rnea,
        init_mode=str(args.init_mode), init_mix=float(args.init_mix),
        white_noise=bool(args.white_noise),
        aux_torque_tracking=bool(args.aux_torque_tracking),
        terminal_initializer=pool_initializer,
    )
    t_data = time.perf_counter() - t0
    for name, pool in (("train", train_pool), ("val", val_pool)):
        print(_fmt_stats(name, pool), flush=True)
    print(f"[data] {t_data:.1f} s", flush=True)

    cfg = TrainConfig(
        max_steps=int(args.steps),
        batch_size=int(args.batch),
        lr=float(args.lr),
        warmup_steps=int(args.warmup),
        amp_dtype="float32" if args.no_amp else "bfloat16",
        compile_mode=None if args.no_compile else "default",
        eval_every=int(args.eval_every),
        patience_evals=int(args.patience),
        seed=int(args.seed),
        device=str(args.device),
        aux_torque_tracking=bool(args.aux_torque_tracking),
        waist_hold_wp=int(args.waist_hold_wp),
        waist_gain_frac=float(args.waist_gain_frac),
    )
    if float(args.waist_follow_frac) > 0.0:
        raise SystemExit(
            "--waist-follow-frac is removed; use --w-waist-gain (identity Jacobian)"
        )
    if float(args.w_waist_gain) > 0.0 and not (args.waist_sp_head or args.waist_pr_sp_head):
        raise SystemExit("--w-waist-gain > 0 requires --waist-sp-head and/or --waist-pr-sp-head")
    if float(args.w_waist_mimic) > 0.0 and not (args.waist_sp_head or args.waist_pr_sp_head):
        raise SystemExit("--w-waist-mimic > 0 requires a waist set-point head")
    weights = LossWeights(
        aux=float(args.aux_weight),
        mimic=float(args.w_waist_mimic),
        gain=float(args.w_waist_gain),
    )

    summary = train(
        model=model, train_pool=train_pool, val_pool=val_pool, cfg=cfg,
        model_cfg=model_cfg, weights=weights, run_dir=run_dir,
        scenario_cfg=scen, noise_cfg=noise,
    )
    summary["variant"] = "apex"
    summary["terminal_head_checkpoint"] = str(terminal_ckpt)
    summary["terminal_head_stop_gradient"] = True
    summary["init_mode"] = str(args.init_mode)
    summary["init_mix"] = float(args.init_mix)
    summary["white_noise"] = bool(args.white_noise)
    summary["apex_cfg"] = {k: v for k, v in acfg.__dict__.items()}
    summary["data_gen_s"] = t_data
    summary["train_pool"] = train_pool.distribution()
    summary["val_pool"] = val_pool.distribution()
    summary["train_stats"] = {
        k: (float(np.percentile(v, 99)) if isinstance(v, np.ndarray) and v.size else v)
        for k, v in train_pool.stats.items()
        if not isinstance(v, dict)
    }

    best = run_dir / "best.pt"
    if best.exists():
        A.load_checkpoint(best, model=model, map_location=dev)
    amp = None if args.no_amp else torch.bfloat16
    # "straight_init" is the trained regime (matches deployment); the other two
    # are robustness probes.
    for tag, kw in (
        ("mix_init", {"init_mode": "mix"}),
        ("straight_init", {"init_mode": "straight"}),
        ("apex_init", {"init_mode": "apex"}),
        ("hold_init", {"hold_init": True}),
    ):
        rep = _report(model, val_pool, cfg, amp, kw)
        summary[f"imitation_{tag}"] = rep
        print(f"[val:{tag}] " + json.dumps(
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rep.items()}
        ), flush=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] {summary['steps']} steps in {summary['wall_s']:.0f} s "
          f"({summary['steps_per_s']:.2f} it/s), best val {summary['best_val']:.4f} "
          f"@ step {summary['best_step']} ({summary['stopped']})", flush=True)
    print(f"[done] buckets used: {summary['bucket_counts']}", flush=True)
    return 0


def _report(model, pool, cfg, amp, kw):
    """``imitation_report`` with an ``init_mode`` forced on the pool's batches."""
    if "init_mode" in kw:
        mode = kw["init_mode"]
        orig = pool.batch

        def patched(bucket, idx, *, generator=None, hold_init=False, init_mode=None):
            return orig(bucket, idx, generator=generator, hold_init=hold_init,
                        init_mode=init_mode or mode)

        pool.batch = patched
        try:
            return imitation_report(
                model, pool, batch_size=min(512, cfg.batch_size), n_batches=16,
                seed=1234, amp_dtype=amp,
            )
        finally:
            pool.batch = orig
    return imitation_report(
        model, pool, batch_size=min(512, cfg.batch_size), n_batches=16,
        seed=1234, amp_dtype=amp, hold_init=bool(kw.get("hold_init")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
