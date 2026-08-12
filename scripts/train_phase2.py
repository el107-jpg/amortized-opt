#!/usr/bin/env python
"""Merged phase-2/3 run: direct cost descent + chained BPTT (section 7).

    python scripts/train_phase2.py --device cuda:1 --steps 60000 \
        --init runs/phase1_main/best.pt --scenarios 300000

Everything lands in ``amortized/runs/<name>/``: ``metrics.jsonl`` (one record
per log / eval / event), ``best.pt`` (highest rolling grid acceptance),
``last.pt`` and ``summary.json``.  Checkpoints carry the augmented-Lagrangian
state, so a resume continues the multipliers rather than restarting them.

Resolved decision 6: bf16 autocast, batch >= 512, ``torch.compile``, horizon
bucketing, rolling eval every 2 k steps with the plan's stage-A gates.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def configure_inductor() -> None:
    """Settings under which inductor DOES compile this stack on this toolchain.

    The phase-2 chapter recorded ``torch.compile`` as unusable here: inductor
    emitted kernels the conda Triton (3.3.0) could not parse against the
    prebundled Isaac torch (2.7.0+cu128) -- ``'constexpr' has no attribute
    'is_block'`` on the ``triton_poi_fused_stack_*`` kernels -- and then retried
    **every autotune config**, so a compile that should take ~2 minutes ran past
    13 with no first step.

    Re-measured 2026-08-10 with autotuning and inductor's own cudagraphs turned
    off explicitly: the decoder compiles in 19 s for a 3.4x fwd+bwd speedup
    (42.9 -> 12.5 ms at batch 256 x H = 50) and ``FusedCost`` -- the FK/RNEA
    stack the chapter blamed -- compiles in 157 s for 3.3x (212 -> 64 ms), under
    bf16 autocast, with ``suppress_errors`` OFF.  So the failure was the autotune
    retry storm, not the codegen; these are the settings that avoid it.

    ``backend="cudagraphs"`` was also tried and does NOT work: its outputs are
    overwritten by the next invocation and the eager backward then raises, even
    with ``cudagraph_mark_step_begin`` before every call.
    """
    import torch

    torch._inductor.config.max_autotune = False
    torch._inductor.config.max_autotune_pointwise = False
    torch._inductor.config.max_autotune_gemm = False
    torch._inductor.config.coordinate_descent_tuning = False
    torch._inductor.config.triton.cudagraphs = False
    # One compiled unit per (bucket, batch shape) across the cost / tc / bptt
    # steps is already ~10 graphs; the default limit of 8 silently drops back to
    # eager partway through the run.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256


import amortized_opt as A  # noqa: E402
from amortized_opt.lead import resolve_lead  # noqa: E402
from amortized_opt.phase2_data import (  # noqa: E402
    Phase2Noise,
    Phase2ScenarioConfig,
    generate_cost_pool,
)
from amortized_opt.train_cost import (  # noqa: E402
    Phase2Config,
    WeightSchedule,
    eval_subset,
    load_phase2_checkpoint,
    phase2_al_config,
    phase2_cost_config,
    train_phase2,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--init", default="runs/phase1_main/best.pt",
                   help="phase-1 checkpoint to start from ('' = random init)")
    p.add_argument("--resume", default=None, help="phase-2 checkpoint to continue")
    p.add_argument("--scenarios", type=int, default=250_000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--buckets", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--eval-like-frac", type=float, default=0.30)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--compile-cost", action="store_true",
                   help="also compile the physics (broken on this Triton -- see "
                        "Phase2Config.compile_cost)")
    # model architecture overrides
    p.add_argument("--d-model", type=int, default=None, help="decoder width (default 384)")
    p.add_argument("--dec-layers", type=int, default=None, help="decoder depth (default 3)")
    p.add_argument("--d-ctx", type=int, default=None, help="encoder width (default 256)")
    p.add_argument("--ctx-layers", type=int, default=None, help="encoder depth (default 2)")
    p.add_argument("--dec-ffn-mult", type=float, default=None,
                   help="decoder FFN multiplier (default 4.0)")
    p.add_argument("--no-aux-heads", action="store_true",
                   help="disable auxiliary heads (vel/acc/torque/fk)")
    # objective knobs (the pilot race varies these)
    p.add_argument("--w-aux", type=float, default=None)
    p.add_argument("--w-tc", type=float, default=None)
    p.add_argument("--tc-frac", type=float, default=None)
    p.add_argument("--rho-scale", type=float, default=1.0)
    p.add_argument("--ramp-start", type=int, default=None)
    p.add_argument("--ramp-end", type=int, default=None)
    p.add_argument("--bptt-start-frac", type=float, default=None)
    p.add_argument("--bptt-max-frac", type=float, default=None)
    p.add_argument("--n-sub", type=int, default=1)
    p.add_argument("--w-torque", type=float, default=None,
                   help="override phase2_cost_config's torque price (default 20)")
    p.add_argument("--w-torque-peak", type=float, default=None,
                   help="price on the SOFT-MAX half of the torque term, which is "
                        "what the acceptance gate checks (default 2.0)")
    p.add_argument("--torque-peak-ramp", type=int, default=1000,
                   help="steps over which the peak half ramps 0 -> 1")
    p.add_argument("--w-smooth-jerk", type=float, default=None,
                   help="override the restored jerk weight (default 1e-4)")
    p.add_argument("--w-collision", type=float, default=None)
    p.add_argument("--w-collision-peak", type=float, default=None,
                   help="price on the SMOOTH-MAX half of the collision term -- "
                        "the reduction ok_collision's zero-tolerance check "
                        "actually corresponds to.  The mean is diluted by "
                        "~2100-4200 (knot, point, drum) pairs.  OFF by default; "
                        "try ~500 (the hinge is in metres, ~0.03 at a 27 mm "
                        "penetration).  Priced by its own AL group")
    p.add_argument("--collision-peak-ramp", type=int, default=1000,
                   help="steps over which the collision peak half ramps 0 -> 1")
    # --- section 7d: keep the waist upright (pitch/roll only, yaw stays free) --
    p.add_argument("--waist-pr-box", type=float, default=None,
                   help="shrink the joint box on waist_pitch/waist_roll to this "
                        "half-width in rad (URDF is 0.52).  waist_yaw is NOT "
                        "touched.  Off by default; flows through the existing "
                        "joint_box AL group")
    p.add_argument("--w-waist-upright", type=float, default=None,
                   help="quadratic upright well on waist pitch/roll, rad^-2 "
                        "(CostConfig.w_waist_upright).  Yaw is excluded")
    p.add_argument("--waist-ramp", type=int, default=300,
                   help="steps over which BOTH waist terms ramp 0 -> 1; ramping "
                        "the box is what keeps lam_joint_box from spiking")
    p.add_argument("--waist-full", action="store_true",
                   help="waist terms at FULL weight from step 0 (sets "
                        "weight_schedule.waist_start = -1, waist_ramp = 1).  "
                        "Use when the warm start was already trained with waist "
                        "terms, so there is no multiplier shock to absorb")
    p.add_argument("--w-waist-aim", type=float, default=None,
                   help="waist-aim IL penalty weight, rad^-2 "
                        "(CostConfig.w_waist_aim).  Penalises yaw deviation from "
                        "the waist_aim reference trajectory, and pitch/roll "
                        "deviation from zero, over the WHOLE horizon on every "
                        "batch (fresh pool + replay), not just in the seed")
    p.add_argument("--w-waist-yaw-vel", type=float, default=None,
                   help="price on mean waist-yaw RATE^2, (rad/s)^-2 "
                        "(CostConfig.w_waist_yaw_vel).  Stops the planner buying "
                        "impact speed by winding up the torso.  A steady 1 rad/s "
                        "turn pays exactly this weight")
    p.add_argument("--freeze-waist-pr", action="store_true",
                   help="remove waist_pitch / waist_roll from the CONTROLLER: "
                        "the decoder's columns for them are pinned to the ready "
                        "pose (0 rad) and their aux rows zeroed, so no cost "
                        "weight can trade them away.  waist_yaw stays free.  "
                        "Stored in model_cfg, so eval / deploy inherit it")
    # --- section 7d leg 2: the strike-masked tau^2 effort term ----------------
    p.add_argument("--w-effort", type=float, default=None,
                   help="price on mean tau^2 (N^2 m^2), faded out around each "
                        "hit.  An objective, not an AL group.  Expect 1e-4..1e-2")
    p.add_argument("--effort-fade-wp", type=int, default=None,
                   help="waypoints before a hit over which the effort term is "
                        "exactly OFF (it ramps back over the same span again)")
    p.add_argument("--effort-ramp", type=int, default=300,
                   help="steps over which w_effort ramps 0 -> 1")
    # ---- the idle arm ------------------------------------------------------
    p.add_argument("--no-idle-home", action="store_true",
                   help="disable the idle-arm home pose entirely: no park seed, "
                        "no rest stretches in the roller, and w_idle_home / "
                        "w_idle_vel stay exact zeros")
    p.add_argument("--idle-home", default=None,
                   help="path to an idle-home table (default: "
                        "assets/idle_home_poses.json when it exists)")
    p.add_argument("--no-idle-token", action="store_true",
                   help="drop the idle arm's own CONTEXT token (the model then "
                        "learns the park pose from the seed alone).  Also the "
                        "way to load a pre-2026-08-12 checkpoint strictly")
    p.add_argument("--w-idle-home", type=float, default=None,
                   help="price on (q_arm - home)^2 for a hand with no strike "
                        "(default 1.0; needs a home table to do anything)")
    p.add_argument("--w-idle-vel", type=float, default=None,
                   help="price on the idle arm's terminal-weighted qd^2 -- the "
                        "'arrive stopped and stay stopped' half (default 0.02)")
    p.add_argument("--idle-ramp", type=int, default=300,
                   help="steps over which both idle weights ramp in")
    p.add_argument("--idle-settle-wp", type=int, default=10,
                   help="waypoints the retraction eases over before holding home")
    p.add_argument("--rest-ticks", type=int, default=25,
                   help="ticks a BOTH-hands-idle roller slot keeps simulating "
                        "before respawn (0 = delete rests, the old behaviour)")
    p.add_argument("--rest-frac", type=float, default=0.25,
                   help="fraction of naturally both-idle scenario draws kept in "
                        "the fresh pool instead of rejected")
    p.add_argument("--lr-horizon", type=int, default=None,
                   help="denominator of the LR cosine, in steps (default: "
                        "--steps).  Set it to the PREDECESSOR run's budget to "
                        "continue on its LR trajectory under a shorter step cap")
    p.add_argument("--no-roller-regoal", action="store_true",
                   help="reproduce the old respawn-on-hit roller (no in-place "
                        "per-hand re-goal, no time-to-hit ageing)")
    p.add_argument("--stage-a-gate", action="store_true",
                   help="hold the weight ramp at lo_limits until the stage-A "
                        "metrics pass (see WeightSchedule.gate_stage_a)")
    p.add_argument("--stage-a-open", action="store_true",
                   help="seed the stage-A gate OPEN (a resume from a checkpoint "
                        "that already passed it)")
    p.add_argument("--short-wp", type=int, default=8,
                   help="waypoints defining the final-approach band that is "
                        "oversampled in the fresh pool and the MPC replay")
    p.add_argument("--short-boost", type=float, default=1.0,
                   help="sampling-weight multiplier for that band (1 = uniform)")
    p.add_argument("--probe-every", type=int, default=0,
                   help="steps between CLOSED-LOOP probes; 0 = off.  best.pt is "
                        "gated on the probe not regressing (see train_phase2)")
    p.add_argument("--probe-clips", type=int, default=1)
    p.add_argument("--probe-steps", type=int, default=250)
    p.add_argument("--probe-lead", type=float, default=0.0,
                   help="MPCGenConfig.hit_lead_s used by the probe")
    p.add_argument("--cmd-lead", default="0.0",
                   help="reference lead executed by the training roller / TC / "
                        "BPTT plant: a float in seconds, or a preset name from "
                        "amortized_opt.lead.LEAD_PRESETS ('fit'/'sparse'/'wide'). "
                        "MUST match the deployed MPCGenConfig.cmd_lead_s")
    p.add_argument("--probe-cmd-lead", default=None,
                   help="reference lead used by the closed-loop probe; defaults "
                        "to --cmd-lead")
    p.add_argument("--probe-freeze", type=int, default=0,
                   help="MPCGenConfig.freeze_wp used by the probe")
    p.add_argument("--plateau-evals", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; stops cleanly at the next step")
    p.add_argument("--target-acceptance", type=float, default=None,
                   help="rolling-eval acceptance that ends the run early "
                        "(default 0.60; raise it to spend the whole budget)")
    # bookkeeping
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-cases", type=int, default=81)
    p.add_argument("--ckpt-every", type=int, default=2_000)
    p.add_argument("--roller-slots", type=int, default=256)
    p.add_argument("--roller-horizon", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--pool-cache", default=None,
                   help="npz of a previously generated pool (reused if present)")
    return p.parse_args(argv)


def load_or_make_pool(args, scen, noise, device):
    cache = Path(args.pool_cache) if args.pool_cache else None
    if cache is not None and cache.exists():
        from amortized_opt.phase2_data import Phase2Pool

        z = np.load(cache, allow_pickle=True)
        buckets: dict[int, dict[str, np.ndarray]] = {}
        for k in z.files:
            if k.startswith("_"):
                continue
            b, name = k.split("/", 1)
            buckets.setdefault(int(b), {})[name] = z[k]
        stats = json.loads(str(z["_stats"])) if "_stats" in z.files else {}
        print(f"[data] reusing {cache}", flush=True)
        return Phase2Pool(buckets, cfg=scen, noise=noise, device=device, stats=stats)

    t0 = time.perf_counter()
    print(f"[data] generating {args.scenarios} scenarios ({args.workers} workers)...",
          flush=True)
    pool = generate_cost_pool(
        args.scenarios, seed=args.seed, cfg=scen, noise=noise, split="train",
        workers=args.workers, device=device,
    )
    print(f"[data] {time.perf_counter() - t0:.1f} s -> {len(pool)} samples "
          f"{pool.distribution()}", flush=True)
    s = pool.stats
    pr = np.asarray(s.get("peak_rate_ratio", []), float)
    print(f"[data] accepted {s['accepted']}/{s['requested']} "
          f"(|qd| reject {s['reject_velocity']}, idle {s['reject_idle']}, "
          f"box-start {s.get('reject_box_start', 0)}); "
          f"eval-like {s['eval_like']}, dual-arm {s['dual_active']} "
          f"(asym {s['asymmetric']}), single-idle {s.get('single_idle', 0)}, "
          f"rest {s.get('rest', 0)}; peak|qd|/limit p50={np.percentile(pr, 50):.2f} "
          f"p99={np.percentile(pr, 99):.2f}", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for b, d in pool.data.items():
            for k, v in d.items():
                flat[f"{b}/{k}"] = v.detach().cpu().numpy()
        stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in pool.stats.items()}
        np.savez(cache, _stats=np.array(json.dumps(stats)), **flat)
    return pool


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if not args.no_compile:
        configure_inductor()

    root = Path(__file__).resolve().parent.parent
    name = args.name or f"phase2_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out) if args.out else root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}", flush=True)

    # The idle-arm table is resolved ONCE and threaded to all three consumers
    # (the pool's seeds, the roller, the cost) so they cannot disagree.
    idle_home_path = None if args.no_idle_home else (args.idle_home or "default")
    scen = Phase2ScenarioConfig(
        horizon=int(args.horizon), buckets=tuple(args.buckets),
        eval_like_frac=float(args.eval_like_frac),
        idle_home_path=idle_home_path,
        idle_settle_wp=int(args.idle_settle_wp),
        rest_frac=float(args.rest_frac),
    )
    noise = Phase2Noise()
    pool = load_or_make_pool(args, scen, noise, dev)

    # ---- model -------------------------------------------------------------
    # ``model_config_from_checkpoint`` reconstructs the architecture the weights
    # ACTUALLY have, not the one today's defaults would build -- so this is the
    # checkpoint as trained.  The deliberate change (switching the idle token on
    # over a warm start that predates it) is applied below, explicitly, and the
    # tensors are reconciled with it.
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_cfg = A.model_config_from_checkpoint(ck)
    elif args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model_cfg = A.model_config_from_checkpoint(ck)
    else:
        ck = None
        model_cfg = A.ModelConfig(max_horizon=int(args.horizon),
                                  num_drums=scen.num_drums, num_future=scen.num_future)
    # apply CLI overrides
    overridden = False
    for attr in ("d_model", "dec_layers", "d_ctx", "ctx_layers", "dec_ffn_mult"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(model_cfg, attr, val)
            overridden = True
    if getattr(args, "no_aux_heads", False):
        model_cfg.predict_aux = False
        overridden = True
    # The idle token is a context-encoder shape, so it has to be decided BEFORE
    # the model is built -- and it is meaningless without a home table.
    want_token = (not args.no_idle_token) and idle_home_path is not None
    if want_token != getattr(model_cfg, "idle_home_token", False):
        model_cfg.idle_home_token = want_token
        overridden = True
    if args.freeze_waist_pr:
        # NOT an architecture change -- no parameter shapes move, so this does
        # NOT set ``overridden`` and the warm start still loads with
        # ``strict=True``.  It only pins two output columns.
        model_cfg.frozen_joints = ("waist_pitch_joint", "waist_roll_joint")
        print(f"[model] frozen joints: {model_cfg.frozen_joints}", flush=True)
    # A checkpoint written before the idle token existed has neither
    # ``encoder.idle_proj`` nor the 4th ``type_emb`` row, so a strict load fails
    # on a shape mismatch.  Detect it and fall back to a REPORTED non-strict
    # load rather than silently disabling the token or silently random-initing.
    if ck is not None and model_cfg.idle_home_token:
        te = ck["model"].get("encoder.type_emb.weight")
        if te is not None and te.shape[0] != model_cfg.n_token_types:
            print(f"[init] checkpoint predates the idle-home token "
                  f"(type_emb {tuple(te.shape)}); its 2 new context params start "
                  f"random -- use --no-idle-token to load it strictly", flush=True)
            overridden = True

    model = A.build_model(model_cfg, dev)
    if ck is not None:
        if overridden:
            # ``strict=False`` forgives MISSING and UNEXPECTED keys but NOT a
            # shape mismatch on a key that exists in both -- it still raises.
            # ``encoder.type_emb.weight`` is exactly that case when the idle
            # token is switched on over a checkpoint that predates it: (3, 256)
            # against (4, 256).  Copy the overlapping block (the DRUM / GOAL /
            # FUTURE rows, which ARE trained) and leave the rest at its random
            # init, rather than dropping the whole tensor and throwing three
            # trained embeddings away with the one new row.
            sd = dict(ck["model"])
            own = model.state_dict()
            for k, v in list(sd.items()):
                w = own.get(k)
                if w is None or w.shape == v.shape:
                    continue
                if w.dim() != v.dim():
                    print(f"[init] {k}: rank {tuple(v.shape)} != {tuple(w.shape)}, "
                          f"dropped (random init)", flush=True)
                    sd.pop(k)
                    continue
                sl = tuple(slice(0, min(a, b)) for a, b in zip(w.shape, v.shape))
                new = w.clone()
                new[sl] = v[sl].to(new.dtype)
                sd[k] = new
                print(f"[init] {k}: {tuple(v.shape)} -> {tuple(w.shape)}, copied "
                      f"{tuple(s.stop for s in sl)} (remainder random)", flush=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[init] {len(missing)} missing keys (random init): "
                      f"{missing[:3]}...", flush=True)
            if unexpected:
                print(f"[init] {len(unexpected)} unused ckpt keys: "
                      f"{unexpected[:3]}...", flush=True)
        else:
            model.load_state_dict(ck["model"])
        print(f"[init] {args.resume or args.init} @ step {ck.get('step')}", flush=True)
    print(f"[model] {A.param_counts(model)}", flush=True)

    cfg = Phase2Config(
        max_steps=int(args.steps), batch_size=int(args.batch), lr=float(args.lr),
        amp_dtype="float32" if args.no_amp else "bfloat16",
        compile_mode=None if args.no_compile else args.compile_mode,
        compile_cost=bool(args.compile_cost), compile_aux=bool(args.compile_cost),
        al=phase2_al_config(rho_scale=float(args.rho_scale)),
        roller_regoal=not args.no_roller_regoal,
        cost=phase2_cost_config(
            n_sub=int(args.n_sub),
            **({} if args.w_torque is None else {"w_torque": float(args.w_torque)}),
            **({} if args.w_torque_peak is None
               else {"w_torque_peak": float(args.w_torque_peak)}),
            **({} if args.w_smooth_jerk is None
               else {"w_smooth_jerk": float(args.w_smooth_jerk)}),
            **({} if args.w_collision is None else {
                "w_collision_drum": float(args.w_collision),
                "w_collision_under": float(args.w_collision),
                "w_collision_self": float(args.w_collision)}),
            **({} if args.w_collision_peak is None
               else {"w_collision_peak": float(args.w_collision_peak)}),
            **({} if args.waist_pr_box is None
               else {"waist_pr_box": float(args.waist_pr_box)}),
            **({} if args.w_waist_upright is None
               else {"w_waist_upright": float(args.w_waist_upright)}),
            **({} if args.w_waist_aim is None
               else {"w_waist_aim": float(args.w_waist_aim)}),
            **({} if args.w_waist_yaw_vel is None
               else {"w_waist_yaw_vel": float(args.w_waist_yaw_vel)}),
            **({} if args.w_effort is None else {"w_effort": float(args.w_effort)}),
            **({} if args.effort_fade_wp is None
               else {"effort_fade_wp": int(args.effort_fade_wp)}),
            **({} if args.w_idle_home is None
               else {"w_idle_home": float(args.w_idle_home)}),
            **({} if args.w_idle_vel is None
               else {"w_idle_vel": float(args.w_idle_vel)}),
        ),
        idle_home_path=idle_home_path,
        idle_settle_wp=int(args.idle_settle_wp),
        roller_rest_ticks=int(args.rest_ticks),
        eval_every=int(args.eval_every), ckpt_every=int(args.ckpt_every),
        short_wp=int(args.short_wp), short_boost=float(args.short_boost),
        cmd_lead_s=resolve_lead(args.cmd_lead),
        probe_every=int(args.probe_every),
        roller_slots=int(args.roller_slots), roller_horizon=int(args.roller_horizon),
        seed=int(args.seed), device=str(args.device),
    )
    if args.w_aux is not None:
        cfg.w_aux = float(args.w_aux)
    if args.w_tc is not None:
        cfg.w_tc = float(args.w_tc)
    if args.tc_frac is not None:
        cfg.tc_frac = float(args.tc_frac)
    if args.ramp_start is not None:
        cfg.schedule = replace(cfg.schedule, ramp_start=int(args.ramp_start))
    if args.ramp_end is not None:
        cfg.schedule = replace(cfg.schedule, ramp_end=int(args.ramp_end))
    if args.target_acceptance is not None:
        cfg.target_acceptance = float(args.target_acceptance)
    if args.plateau_evals is not None:
        cfg.plateau_evals = int(args.plateau_evals)
    if args.lr_horizon is not None:
        cfg.lr_horizon = int(args.lr_horizon)
    cfg.schedule = replace(
        cfg.schedule,
        torque_peak_ramp=int(args.torque_peak_ramp),
        collision_peak_ramp=int(args.collision_peak_ramp),
        waist_ramp=int(args.waist_ramp),
        effort_ramp=int(args.effort_ramp),
        idle_ramp=int(args.idle_ramp),
        gate_stage_a=bool(args.stage_a_gate),
    )
    if args.bptt_start_frac is not None:
        cfg.bptt_start_frac = float(args.bptt_start_frac)
    if args.bptt_max_frac is not None:
        cfg.bptt_max_frac = float(args.bptt_max_frac)

    # ---- rolling-eval grid --------------------------------------------------
    scorer = A.TrajectoryScorer(device="cpu")
    cases = eval_subset(A.generate_grid(seed=0, grid="standard", scorer=scorer),
                        int(args.eval_cases))
    print(f"[eval] rolling grid: {len(cases)} cases", flush=True)
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()

    # ---- closed-loop probe ---------------------------------------------------
    probe_fn = None
    if args.probe_every > 0:
        from amortized_opt.probe import closed_loop_probe

        probe_clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.probe_clips]
        probe_scorer = A.TrajectoryScorer(device="cpu")
        print(f"[probe] every {args.probe_every} steps on "
              f"{[c.name for c in probe_clips]} "
              f"(lead {args.probe_lead*1000:.0f} ms, freeze {args.probe_freeze} wp)",
              flush=True)

        def probe_fn(m, _clips=probe_clips, _sc=probe_scorer):  # noqa: F811
            r = closed_loop_probe(
                m, kit=kit, scorer=_sc, clips=_clips, device=dev,
                num_drums=model_cfg.num_drums, num_future=model_cfg.num_future,
                horizon=int(args.buckets[-1]), steps=int(args.probe_steps),
                seed=0, amp_dtype=None if args.no_amp else torch.bfloat16,
                lead_s=float(args.probe_lead), freeze_wp=int(args.probe_freeze),
                cmd_lead_s=resolve_lead(
                    args.probe_cmd_lead if args.probe_cmd_lead is not None
                    else args.cmd_lead),
                record_drift=False,
            )
            r.pop("_hits", None)
            return r

    al_state = opt_state = None
    if args.resume:
        al_state = ck.get("al")
        opt_state = ck.get("optimizer")
        print(f"[resume] AL {'yes' if al_state else 'no'}, "
              f"optimizer {'yes' if opt_state else 'no'}", flush=True)

    summary = train_phase2(
        model=model, pool=pool, cfg=cfg, model_cfg=model_cfg, run_dir=run_dir,
        scenario_cfg=scen, eval_cases=cases, scorer=scorer, kit=kit, library=library,
        al_state=al_state, opt_state=opt_state, probe_fn=probe_fn,
        stage_a_init=bool(args.stage_a_open), max_hours=args.max_hours,
    )
    summary["args"] = vars(args)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] {summary['steps']} steps in {summary['wall_s']:.0f} s "
          f"({summary['steps_per_s']:.2f} it/s), best acceptance "
          f"{summary['best_acceptance']:.3f} @ {summary['best_step']} "
          f"({summary['stopped']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
