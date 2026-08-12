#!/usr/bin/env python
"""RL-2: the episodic parameter-space-noise fine-tune, in Isaac.

The plan's scheme, assembled (``docs/amortized_optimizer_plan.md``, "Final Fine
Tune in RL via IsaacSim", revised 2026-08-10) -- the update math lives in
``isaac_rl/psn.py`` (Isaac-free, unit-tested); this file owns the Isaac loop.

One iteration
-------------

1. **Rollout** ``--ticks`` control ticks over ``--envs`` envs with the per-env
   FiLM perturbation attached.  One epsilon per env is held across every replan
   of a hit segment and redrawn at the fire event that ends it, in antithetic
   pairs across env index parity.  Each tick buffers everything the update
   needs to *re-run the model on that state*: the context tokens, the strike
   rows, the masked init, the measured state, and the trajectory the perturbed
   policy actually emitted (the distillation target).

2. **Credit.** The per-tick executed reward accumulates into the env's open
   segment; the fire event closes it into an ``(epsilon, return)`` sample.
   Segments still open when the window ends are dropped.

3. **Update.**  Two rules, both ending in a weight update through the same
   optimizer so they are directly A/B-able:

   * ``distill`` -- advantage-weighted regression of theta onto the buffered
     trajectories of the top-quantile segments.
   * ``nes`` -- the rank-shaped antithetic NES direction in FiLM coordinates,
     applied as a batch-independent FiLM offset (``ParameterNoise.set_mean``),
     used to generate targets on the *same* buffered states, which are then
     regressed onto exactly as above.  This is what turns a subspace direction
     into a weight update without shipping a FiLM layer at deployment.

   Both carry the analytic ``FusedCost`` on the model's own output as a
   simultaneous loss (the plan's "constraint anchor") and the auxiliary heads
   throughout, trunk included, with an EXECUTED-state anchor on waypoint 0.

4. **Sigma.**  Plappert-style adaptation on a physical distance: clean vs
   perturbed plan from the same state, terminal tip deviation in mm.

5. **Probe.**  Every ``--probe-every`` iterations the UNPERTURBED policy runs
   on fixed held-out clips and seeds; ``best.pt`` is selected on that closed-loop
   probe (miss rate, then |latency| p50), never on the training return.

Device recipe (mandatory on this box -- Warp initialises on the visible cuda:0
whatever ``--device`` says):

    CUDA_VISIBLE_DEVICES=1 python isaac_rl/scripts/train_psn.py --headless \
        --device cuda:0 --envs 128 --ticks 60 --iters 200 --out runs/rl2_main
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse(argv=None):
    from isaaclab.app import AppLauncher as _AL
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "runs/phase2b_main/best.pt"))
    p.add_argument("--out", default=str(ROOT / "runs/rl2_main"))
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--ticks", type=int, default=60, help="rollout ticks per iteration")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--hours", type=float, default=0.0,
                   help="wall-clock budget; stop after it even if iters remain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--probe-clips", type=int, default=3,
                   help="held-out clips reserved for the probe (taken from the END "
                        "of the eval pool, never trained on)")
    p.add_argument("--lead", default=str(ROOT / "runs/rl1b/lead_isaac.json"))
    p.add_argument("--lead-key", default="best_strike_s")

    # ---- exploration ----
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--noise-mode", default="film", choices=("film", "lora"))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--target-mm", type=float, default=3.0)
    p.add_argument("--sigma-every", type=int, default=1,
                   help="iterations between adaptive-sigma probes")
    p.add_argument("--clean-frac", type=float, default=0.125,
                   help="fraction of envs run UNPERTURBED every iteration "
                        "(the plan's periodic unperturbed eval episodes)")

    # ---- update ----
    p.add_argument("--update", default="distill", choices=("distill", "nes", "both"))
    p.add_argument("--quantile", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--warmup", type=int, default=20,
                   help="linear LR warmup over this many gradient steps")
    p.add_argument("--steps-per-iter", type=int, default=8)
    p.add_argument("--minibatch", type=int, default=192)
    p.add_argument("--w-bc", type=float, default=1.0)
    p.add_argument("--w-cost", type=float, default=1.0)
    p.add_argument("--w-aux", type=float, default=1.0)
    p.add_argument("--w-aux-exec", type=float, default=0.5)
    p.add_argument("--nes-step", type=float, default=0.5)

    # ---- reward ----
    p.add_argument("--w-filter", type=float, default=0.25)
    p.add_argument("--w-idle-home", type=float, default=0.2)
    p.add_argument("--cost-every", type=int, default=1,
                   help="evaluate the analytic FusedCost reward term every k ticks "
                        "(throughput lever; the term is scaled by k so the segment "
                        "return keeps its units)")

    # ---- plant / safety / DR ----
    p.add_argument("--safety", default="box", choices=("off", "box", "qp"))
    p.add_argument("--safety-sdf-steps", type=int, default=2)
    p.add_argument("--no-target-exempt", action="store_true",
                   help="reproduce the RL-1b confound: do NOT exempt the target drum")
    p.add_argument("--randomize-kit", action="store_true")
    p.add_argument("--lag-substeps", default="0,0",
                   help="residual command lag range in PHYSICS substeps, 'lo,hi'")
    p.add_argument("--pd-noise", type=float, default=0.0,
                   help="per-env log-normal sigma on the PD stiffness/damping")
    p.add_argument("--restitution", default="",
                   help="'lo,hi' restitution range for the stick material")
    p.add_argument("--friction", default="", help="'lo,hi' friction range")

    # ---- bookkeeping ----
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--probe-ticks", type=int, default=250)
    p.add_argument("--resume", default="")
    _AL.add_app_launcher_args(p)
    return p.parse_args(argv)


def _pair(s, cast=float):
    if not s:
        return None
    a, b = str(s).split(",")
    return (cast(a), cast(b))


def main(argv=None) -> int:  # noqa: C901 -- one linear driver, deliberately
    args = parse(argv)
    from isaaclab.app import AppLauncher
    _app = AppLauncher(args).app

    import numpy as np
    import torch
    import torch.nn.functional as F

    import amortized_opt as A
    from amortized_opt.train_cost import group_lengths
    from amortized_opt.train_warmstart import LossScales, _masked_huber
    from amortized_opt.warmstart_data import _fk_points, aux_targets
    from isaac_rl.env import make_env, try_build_rnea
    from isaac_rl.env_cfg import CONTROL_DT, DrumRLEnvCfg
    from isaac_rl.loop import DrumMPCLoop
    from isaac_rl.measure import latency_report
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    from isaac_rl.perturb import (AdaptiveSigmaConfig, NoiseConfig, ParameterNoise,
                                  deviation_report)
    from isaac_rl.psn import PSNConfig, SegmentBuffer, TickBuffer, nes_direction
    from isaac_rl.reward import ExecutedReward, ExecutedStrikeScorer, RewardConfig
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (out_dir / "metrics.jsonl").open("a")

    def emit(rec: dict) -> None:
        jsonl.write(json.dumps(rec, default=float) + "\n")
        jsonl.flush()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32
    B = int(args.envs)
    H_TICKS = int(args.ticks)

    lead, lead_src = args.lead, "preset"
    if isinstance(lead, str) and lead.endswith(".json") and Path(lead).is_file():
        lead = tuple(float(x) for x in json.loads(Path(lead).read_text())[args.lead_key])
        lead_src = f"{args.lead}[{args.lead_key}]"

    # ============================================================ the stack ==
    rl_cfg = DrumRLEnvCfg(
        num_envs=B, device=args.device, seed=args.seed, record=False,
        lag_substeps=tuple(int(x) for x in args.lag_substeps.split(",")),
        restitution_range=_pair(args.restitution),
        friction_range=_pair(args.friction),
    )
    env = make_env(rl_cfg)
    env.reset()
    if args.pd_noise > 0 and hasattr(env, "randomize_pd_gains"):
        env.randomize_pd_gains(float(args.pd_noise), seed=args.seed)
    if (rl_cfg.restitution_range or rl_cfg.friction_range) and \
            hasattr(env, "randomize_contact_material"):
        env.randomize_contact_material(seed=args.seed)
    q, qd = env.read_state()
    q0_home = q.clone()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    H = int((ck.get("scenario_cfg") or {}).get("horizon", 50))
    if args.resume:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        print(f"[rl2] resumed weights from {args.resume}", flush=True)

    kit = A.DrumKit.from_layout()
    pool = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    n_probe = max(1, min(int(args.probe_clips), len(pool) - 1))
    probe_clips = pool[-n_probe:]                 # HELD OUT -- never trained on
    train_clips = pool[:-n_probe][: max(int(args.clips), 1)]
    print(f"[rl2] {len(train_clips)} train clips, {len(probe_clips)} held-out "
          f"probe clips (of a pool of {len(pool)})", flush=True)

    scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future,
                          randomize_kit=bool(args.randomize_kit))
    sched = BatchedSchedule(B, scfg, device=dev, dtype=dtype)
    rng = np.random.default_rng(args.seed)
    all_ids = torch.arange(B, device=dev)

    def _reset(ids: torch.Tensor, *, clips=train_clips) -> None:
        n = int(ids.numel())
        if n == 0:
            return
        pick = [clips[int(rng.integers(len(clips)))] for _ in range(n)]
        sched.reset(ids, pick, [kit] * n, rng=rng)
        env.reset_to(ids, q0_home[ids])
        cur, _ = env.read_state()
        mpc.reset(ids, cur)
        if safety is not None:
            safety.reset(ids, cur)

    mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s=lead),
                   device=dev, dtype=dtype, kit_names=tuple(kit.names))
    fk = A.ChainFK(device=dev, dtype=dtype)
    rnea = try_build_rnea(dev, dtype)
    if rnea is None:
        raise RuntimeError("no RNEA in-process: the analytic anchor and the "
                           "gravity feedforward would both be silently off")
    cost = A.FusedCost(A.CostConfig(), rnea=rnea, max_horizon=H, device=dev, dtype=dtype)
    scorer = ExecutedStrikeScorer(fk=fk, dt=CONTROL_DT, device=dev, dtype=dtype)
    rcfg = RewardConfig(w_filter=float(args.w_filter),
                        w_idle_home=float(args.w_idle_home))
    rew = ExecutedReward(rcfg, scorer=scorer, cost=cost, device=dev, dtype=dtype)
    W_PLAN_COST = float(rcfg.w_plan_cost)

    safety = None
    if args.safety != "off":
        from isaac_rl.safety import BatchedSafetyFilter, SafetyConfig
        safety = BatchedSafetyFilter(
            SafetyConfig(mode=args.safety, sdf_steps=int(args.safety_sdf_steps)),
            device=dev, dtype=dtype)
        safety.reset(all_ids, q)

    noise = ParameterNoise.build(
        model, B,
        NoiseConfig(mode=args.noise_mode, rank=args.rank, sigma=float(args.sigma),
                    seed=args.seed),
        device=dev, dtype=dtype)
    noise.adaptive.cfg = AdaptiveSigmaConfig(target_mm=float(args.target_mm))
    # the plan's "periodic unperturbed eval episodes": a fixed slice of envs runs
    # the clean policy every iteration, which also gives the antithetic estimator
    # a same-batch reference.  Whole antithetic PAIRS, or the pairing rots.
    n_clean = 2 * int(round(0.5 * float(args.clean_frac) * B))
    clean_envs = torch.arange(B - n_clean, B, device=dev) if n_clean else None
    if clean_envs is not None:
        noise.set_sigma(0.0, clean_envs)
    noise.attach()

    mpc.reset(all_ids, q)
    _reset(all_ids)
    q, qd = env.read_state()
    # the loop does NOT own the noise: the trainer must read the epsilon that was
    # in force BEFORE the redraw that the fire event triggers.
    loop = DrumMPCLoop(schedule=sched, mpc=mpc, reward=rew, plant=env.plant,
                       noise=None, safety=safety, record=False,
                       safety_target_exempt=not args.no_target_exempt)

    pcfg = PSNConfig(update=args.update, quantile=args.quantile, beta=args.beta,
                     w_bc=args.w_bc, w_cost=args.w_cost, w_aux=args.w_aux,
                     w_aux_exec=args.w_aux_exec, nes_step=args.nes_step,
                     lr=args.lr, steps_per_iter=args.steps_per_iter,
                     minibatch=args.minibatch)
    opt = torch.optim.AdamW(model.parameters(), lr=pcfg.lr,
                            weight_decay=pcfg.weight_decay)
    gstep = 0

    def _set_lr() -> float:
        # Linear warmup.  AdamW's first steps move every coordinate by ~lr
        # regardless of gradient magnitude, so an un-warmed start is the largest
        # single perturbation the whole run applies -- and it lands before any
        # of the return statistics have stabilised.
        f = min(1.0, (gstep + 1) / max(int(args.warmup), 1))
        for g in opt.param_groups:
            g["lr"] = pcfg.lr * f
        return pcfg.lr * f

    scales = LossScales(dev, dtype)
    # the tip's arc coordinate along the stick axis from the MOUNT (= the
    # stick's centre), which is what ExecutedStrikeScorer.contact_points wants.
    TIP_SIGMA = float(np.dot(A.stick_tip_offset('left')
                             - A.stick_mount_transform('left')[1],
                             A.stick_axis_local('left')))

    STRIKE_KEYS = ("hit_index", "active", "target_drum", "start_drum", "p_star",
                   "v_star", "n_out", "hit_radius", "disk_offset",
                   "half_plane_dir", "has_half_plane", "contact_sigma",
                   "shaft_allowed", "start_shaft_allowed", "elev_required",
                   "drum_head_centers")
    CTX_KEYS = tuple(A.ContextInputs.__dataclass_fields__)
    STRIKE_KEYS = tuple(k for k in STRIKE_KEYS
                        if k in A.StrikeContext.__dataclass_fields__)

    def _flatten(prefix, obj, keys):
        out = {}
        for k in keys:
            v = getattr(obj, k, None)
            if v is None:
                continue
            out[f"{prefix}{k}"] = v.detach()
        return out

    def _rebuild(prefix, cls, row, keys):
        """Rebuild a dataclass from buffered rows, restoring fp32 on the way.

        The buffer stores half precision for the radian-scale rows; every
        consumer downstream (the encoder's Linear, FusedCost) is fp32, and a
        Half/Float mismatch there is a hard error rather than a silent cast.
        """
        kw = {}
        for k in keys:
            v = row.get(f"{prefix}{k}")
            if v is None:
                continue
            kw[k] = v.float() if v.dtype == torch.float16 else v
        return cls(**kw)

    # ================================================================ probe ==
    @torch.no_grad()
    def probe(tag: str, ticks: int) -> dict:
        """UNPERTURBED closed loop on the HELD-OUT clips.  The selection metric."""
        noise.detach()
        prng = np.random.default_rng(12345)
        psched = BatchedSchedule(B, scfg, device=dev, dtype=dtype)
        psched.reset(all_ids, [probe_clips[i % len(probe_clips)] for i in range(B)],
                     [kit] * B, rng=prng,
                     clip_ids=[i % len(probe_clips) for i in range(B)])
        pmpc = IsaacMPC(model, psched, MPCConfig(horizon=H, cmd_lead_s=lead),
                        device=dev, dtype=dtype, kit_names=tuple(kit.names))
        prew = ExecutedReward(rcfg, scorer=scorer, cost=cost, device=dev, dtype=dtype)
        psafe = None
        if safety is not None:
            from isaac_rl.safety import BatchedSafetyFilter, SafetyConfig
            psafe = BatchedSafetyFilter(
                SafetyConfig(mode=args.safety, sdf_steps=int(args.safety_sdf_steps)),
                device=dev, dtype=dtype)
        env.reset_to(all_ids, q0_home)
        pq, pqd = env.read_state()
        pmpc.reset(all_ids, pq)
        if psafe is not None:
            psafe.reset(all_ids, pq)
        ploop = DrumMPCLoop(schedule=psched, mpc=pmpc, reward=prew, plant=env.plant,
                            safety=psafe, record=True,
                            safety_target_exempt=not args.no_target_exempt)
        t0 = time.perf_counter()
        pq, pqd, plog = ploop.rollout(pq, pqd, int(ticks))
        rep = latency_report(ploop, kit, device="cpu")
        o = rep["overall"]
        res = {"tag": tag, "n": o["n"], "miss_rate": o["miss_rate"],
               "crossed_in_disk": o["crossed_in_disk"],
               "lat_abs_p50_ms": o["lat_abs_p50_ms"],
               "lat_abs_p90_ms": o["lat_abs_p90_ms"],
               "speed_in_band": o["speed_in_band"],
               "hist_ms": o["hist_ms"],
               "collision": float(plog.get("raw/plan.collision", float("nan"))),
               "filter_interventions": ploop.stats.filter_interventions,
               "probe_wall_s": time.perf_counter() - t0}
        # restore the training state
        env.reset_to(all_ids, q0_home)
        noise.attach()
        return res

    # ============================================================ the loop ===
    best = {"miss_rate": float("inf"), "lat_abs_p50_ms": float("inf"), "iter": -1}
    t_start = time.perf_counter()
    base = probe("baseline", int(args.probe_ticks))
    print(f"[rl2] BASELINE probe: n={base['n']} miss {100*base['miss_rate']:.1f}% "
          f"in-disk {base['crossed_in_disk']} p50 {base['lat_abs_p50_ms']:.1f} ms "
          f"speed {base['speed_in_band']}", flush=True)
    emit({"kind": "probe", "iter": -1, **base})
    torch.save({"model": model.state_dict(), "model_cfg": ck["model_cfg"],
                "scenario_cfg": ck.get("scenario_cfg"), "probe": base, "iter": -1},
               out_dir / "best.pt")
    best.update(miss_rate=base["miss_rate"], lat_abs_p50_ms=base["lat_abs_p50_ms"],
                iter=-1)

    _reset(all_ids)
    q, qd = env.read_state()
    cap = B * (H_TICKS + 2)

    for it in range(int(args.iters)):
        if args.hours > 0 and (time.perf_counter() - t_start) > args.hours * 3600:
            print(f"[rl2] wall-clock budget reached at iter {it}", flush=True)
            break

        # ------------------------------------------------------- rollout ----
        tb = TickBuffer(cap, device=dev)
        sb = SegmentBuffer(B, device=dev)
        model.eval()
        roll_t0 = time.perf_counter()
        agg: dict[str, list[float]] = {}
        for t in range(H_TICKS):
            ctx_in = sched.context()
            strike = sched.strike_context()
            q_before, qd_before = q, qd
            # the warm start the model is ABOUT to be handed (IsaacMPC.plan does
            # ``init = self.init * masks.elem_mask``), captured before the tick
            # mutates it in ``commit``.
            init_snap = mpc.init.detach().clone()
            # analytic-cost throttle: the FusedCost reward term is the single
            # most expensive per-tick item, so it runs every ``cost_every``
            # ticks with its weight scaled by the same factor -- the segment
            # return keeps its units, at k-fold coarser sampling.
            k_cost = max(int(args.cost_every), 1)
            cost_on = (t % k_cost) == 0
            rew.cost = cost if cost_on else None
            rew.cfg.w_plan_cost = W_PLAN_COST * k_cost
            q, qd, terms, res, cs = loop.tick(q, qd)

            # The distillation TARGET is the plan the (perturbed) policy
            # emitted, not ``res.traj``: ``IsaacMPC.plan`` overwrites row 0 with
            # the LED, slewed, box-clamped command.  Regressing onto that would
            # train the model to emit its own lead compensation twice.
            target = res.traj.detach().clone()
            target[:, 0] = mpc.last_ref0.detach()
            row = {
                "init": init_snap,
                "state": torch.cat([q_before, qd_before], dim=-1).detach(),
                "traj": target,
                "elem_mask": res.elem_mask.detach(),
                "hidx": res.hit_index.detach(),
                "q_after": q.detach(), "qd_after": qd.detach(),
                "qd_before": qd_before.detach(),
            }
            row.update(_flatten("ctx.", ctx_in, CTX_KEYS))
            row.update(_flatten("st.", strike, STRIKE_KEYS))
            lo, _hi = tb.append(row)
            sb.add(terms.total, lo,
                   hits=terms.counts.get("scheduled_hits"),
                   misses=terms.counts.get("missed"))
            for k, v in terms.log().items():
                agg.setdefault(k, []).append(v)

            fired_env = (terms.counts["fired"] > 0)
            if bool(fired_env.any()):
                ids = fired_env.nonzero(as_tuple=True)[0]
                sb.close(ids, noise.epsilon())
                noise.redraw(ids)
                if clean_envs is not None:      # redraw() rewrites whole pairs
                    noise.set_sigma(0.0, clean_envs)
            done = sched.done()
            if bool(done.any()):
                _reset(done.nonzero(as_tuple=True)[0])
                q, qd = env.read_state()
        roll_s = time.perf_counter() - roll_t0

        # ------------------------------------------------ credit + update ----
        rows, rw, ret, eps = sb.harvest(pcfg)
        n_seg = len(sb.closed)
        model.train()
        noise.detach()                       # replay minibatches != n_envs
        upd_t0 = time.perf_counter()
        losses: dict[str, list[float]] = {}

        def _forward(idx: torch.Tensor, *, target: torch.Tensor | None = None,
                     weights: torch.Tensor | None = None) -> torch.Tensor:
            row = tb.gather(idx)
            ctx = _rebuild("ctx.", A.ContextInputs, row, CTX_KEYS)
            st = _rebuild("st.", A.StrikeContext, row, STRIKE_KEYS)
            st.hit_index = st.hit_index.long()
            st.target_drum = st.target_drum.long()
            st.start_drum = st.start_drum.long()
            em = row["elem_mask"].float()
            # masks from the RAW per-hand hit index (-1 = idle).  ``st.hit_index``
            # is ``effective_hits`` -- an idle hand has already borrowed the
            # other's horizon there, and feeding that back through ``masks``
            # would silently widen the idle arm's element mask.
            masks = mpc.masks(row["hidx"].long())
            masks.elem_mask = em
            state = row["state"].float()
            q0 = state[:, :A.N_DOF]
            v0 = torch.zeros_like(q0)
            init = row["init"].float() * em
            enc = model.encode_context(ctx)
            out = model(init, state, enc, masks)

            loss = out.traj.new_zeros(())
            tgt = target if target is not None else row["traj"].float()
            tgt = tgt * em
            w = weights if weights is not None else torch.ones(idx.numel(), device=dev)
            wv = w.view(-1, 1, 1)
            bc = F.huber_loss(out.traj, tgt, reduction="none",
                              delta=pcfg.bc_huber_beta)
            bc = (bc * em * wv).sum() / (em * wv).sum().clamp_min(1.0)
            loss = loss + pcfg.w_bc * bc
            losses.setdefault("bc", []).append(float(bc))

            # The analytic term is a CONSTRAINT ANCHOR, not an objective, and
            # the distinction is not pedantic here.  The champion sits at a
            # minimum of the phase-2 objective -- which was an augmented
            # Lagrangian with learned per-group multipliers (``ckpt['al']``) on
            # top of a ramped weight schedule -- not at a minimum of a default
            # ``CostConfig``.  Descending the default total instead pulls the
            # model off its own optimum from step one: measured, 4 AdamW steps
            # at lr 1e-5 with ``w_cost * total`` took the closed-loop probe from
            # 78 % to 98 % miss while the BC term was still 2e-4.
            #
            # So the term is a one-sided hinge against the cost of the very
            # trajectory being imitated: "do not be WORSE on the analytic cost
            # than the plan you are distilling".  It is exactly zero at the
            # start (the target is the model's own output at sigma -> 0), which
            # also gives the whole update the right failure mode -- with no
            # reward signal it degenerates to a no-op instead of to drift.
            lengths = group_lengths(st.hit_index).to(out.traj.dtype)
            ct = cost(q0, out.traj, v0, st, lengths=lengths, elem_mask=em)
            with torch.no_grad():
                ct_ref = cost(q0, tgt, v0, st, lengths=lengths, elem_mask=em)
            c = (ct.total - ct_ref.total).clamp_min(0.0).mean()
            loss = loss + pcfg.w_cost * c
            losses.setdefault("cost_excess", []).append(float(c))
            losses.setdefault("cost_abs", []).append(float(ct.total.mean()))

            with torch.no_grad():
                lab = aux_targets(q0, out.traj.detach().float(), v0, em,
                                  builder=cost.spline, fk=cost.fk, rnea=cost.rnea,
                                  dt=cost.dt)
            # EXACTLY the phase-2 aux loss (``train_cost.make_cost_step._aux``):
            # same normalizers, same 0.1 Huber beta, same masks.  Keeping it
            # identical is what makes the reported torque correlation comparable
            # to the 0.743 the champion was measured at.
            pmf = masks.point_mask.to(out.traj.dtype).unsqueeze(-1).expand_as(out.fk_pos)
            hb = 0.1
            a = (_masked_huber(out.vel / scales.vel, lab["vel"] / scales.vel, em, hb)[0]
                 + _masked_huber(out.acc / scales.acc, lab["acc"] / scales.acc, em, hb)[0]
                 + _masked_huber(out.torque / scales.torque,
                                 lab["torque"] / scales.torque, em, hb)[0]
                 + _masked_huber(out.fk_pos / scales.fk_pos,
                                 lab["fk_pos"] / scales.fk_pos, pmf, hb)[0]
                 + _masked_huber(out.fk_vel / scales.fk_vel,
                                 lab["fk_vel"] / scales.fk_vel, pmf, hb)[0])
            losses.setdefault("aux_self", []).append(float(a))

            # ---- the EXECUTED anchor, on waypoint 0 --------------------------
            qa = row["q_after"].float()
            va = row["qd_after"].float()
            vb = row["qd_before"].float()
            with torch.no_grad():
                acc_e = (va - vb) / CONTROL_DT
                tau_e = cost.rnea.rnea(qa, va, acc_e).abs()
                pts_a = _fk_points(cost.fk, qa)
                pts_b = _fk_points(cost.fk, q0)
                fkv_e = (pts_a - pts_b) / CONTROL_DT
            sv, stq = scales.vel.squeeze(0), scales.torque.squeeze(0)   # (1, n)
            ae = F.huber_loss(out.vel[:, 0] / sv, va / sv, delta=hb)
            ae = ae + F.huber_loss(out.torque[:, 0] / stq, tau_e / stq, delta=hb)
            ae = ae + F.huber_loss(out.fk_pos[:, 0] / scales.fk_pos,
                                   pts_a / scales.fk_pos, delta=hb)
            ae = ae + F.huber_loss(out.fk_vel[:, 0] / scales.fk_vel,
                                   fkv_e / scales.fk_vel, delta=hb)
            losses.setdefault("aux_exec", []).append(float(ae))
            loss = loss + pcfg.w_aux * (a + pcfg.w_aux_exec * ae)

            with torch.no_grad():   # the torque-head correlation trajectory
                pt = out.torque.detach().reshape(-1).float()
                lt = lab["torque"].reshape(-1).float()
                pt = pt - pt.mean(); lt = lt - lt.mean()
                corr = float((pt * lt).sum() /
                             (pt.norm() * lt.norm()).clamp_min(1e-12))
                losses.setdefault("torque_corr", []).append(corr)
                pe = out.torque[:, 0].detach().reshape(-1).float()
                te = tau_e.reshape(-1).float()
                pe = pe - pe.mean(); te = te - te.mean()
                losses.setdefault("torque_corr_exec", []).append(
                    float((pe * te).sum() / (pe.norm() * te.norm()).clamp_min(1e-12)))
            return loss

        nes_norm = 0.0
        if rows.numel() > 0:
            arms = (["distill", "nes"] if pcfg.update == "both" else [pcfg.update])
            for arm in arms:
                d = None
                if arm == "nes":
                    # The NES direction lives in FiLM coordinates.  It is turned
                    # into a WEIGHT update by evaluating the shifted policy on
                    # the buffered states and regressing onto that -- so nothing
                    # at deployment has to carry a FiLM layer.
                    sigma_live = float(args.sigma) * noise.adaptive.scale
                    d = nes_direction(eps.float(), ret, sigma=sigma_live,
                                      step=pcfg.nes_step)
                    nes_norm = float(d.norm())
                perm = torch.randperm(rows.numel(), device=dev)
                for step in range(pcfg.steps_per_iter):
                    s0 = (step * pcfg.minibatch) % max(rows.numel(), 1)
                    sel = perm[s0: s0 + pcfg.minibatch]
                    if sel.numel() < 8:
                        continue
                    idx, w = rows[sel], rw[sel]
                    tgt = None
                    if d is not None:
                        noise.set_mean(d)
                        with torch.no_grad():
                            model.eval()
                            r = tb.gather(idx)
                            ctx = _rebuild("ctx.", A.ContextInputs, r, CTX_KEYS)
                            em = r["elem_mask"].float()
                            m = mpc.masks(r["hidx"].long())
                            m.elem_mask = em
                            e = model.encode_context(ctx)
                            tgt = model(r["init"].float() * em, r["state"].float(),
                                        e, m).traj.detach()
                            model.train()
                        noise.set_mean(None)
                    loss = _forward(idx, target=tgt, weights=w)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        pcfg.grad_clip)
                    lr_now = _set_lr()
                    opt.step()
                    gstep += 1
                    losses.setdefault("total", []).append(float(loss))
                    losses.setdefault("grad_norm", []).append(float(gn))
                    losses.setdefault("lr", []).append(lr_now)
        upd_s = time.perf_counter() - upd_t0
        model.eval()
        noise.attach()

        # ------------------------------------------------- adaptive sigma ----
        dev_mm = float("nan")
        rep: dict = {}
        n_active_hands = 0
        if (it % max(int(args.sigma_every), 1)) == 0:
            with torch.no_grad():
                hidx = sched.hit_index()
                masks = mpc.masks(hidx)
                enc = model.encode_context(sched.context())
                init = mpc.init * masks.elem_mask
                state = torch.cat([q, qd], dim=-1)
                noise.detach()
                clean = model(init, state, enc, masks).traj
                noise.attach()
                pert = model(init, state, enc, masks).traj
                # Per-hand tip deviation at THAT HAND'S OWN terminal knot,
                # averaged over ACTIVE hands only.  Two traps, both measured:
                # taking ``max`` over the two hands prices an idle hand at the
                # other's terminal, and an env with no scheduled strike at all
                # falls back to knot 0 -- where the plan is pinned one control
                # period from the measured pose and the deviation is ~0 by
                # construction.  Averaging those in reported 0.00 mm against a
                # live 23 mrad plan deviation, which would have driven the
                # adaptive scale up until the trajectory degraded.
                sig = torch.full((B, 2), TIP_SIGMA, device=dev, dtype=dtype)
                act = (hidx >= 1)
                dsum = clean.new_zeros(())
                dcnt = clean.new_zeros(())
                for h in (0, 1):
                    kh = hidx[:, h].clamp(1, H) - 1
                    gi = kh.view(-1, 1, 1).expand(-1, 1, clean.shape[-1])
                    qc = clean.gather(1, gi).squeeze(1)
                    qp = pert.gather(1, gi).squeeze(1)
                    tc, _ = scorer.contact_points(qc, sig)
                    tp, _ = scorer.contact_points(qp, sig)
                    d = (tp[:, h] - tc[:, h]).norm(dim=-1) * act[:, h]
                    dsum = dsum + d.sum()
                    dcnt = dcnt + act[:, h].sum()
                dev_mm = float(1000.0 * dsum / dcnt.clamp_min(1))
                n_active_hands = int(dcnt)
                rep = deviation_report(clean, pert, elem_mask=masks.elem_mask,
                                       hit_index=hidx.clamp_min(0))
                rep["dev_rms_mrad"] = 1000.0 * float(rep["rms_rad"])
            noise.observe_deviation(dev_mm)

        # ------------------------------------------------------- logging -----
        rec = {
            "kind": "iter", "iter": it,
            "wall_s": time.perf_counter() - t_start,
            "roll_s": roll_s, "upd_s": upd_s,
            "env_fps": B * H_TICKS / max(roll_s, 1e-9),
            "n_segments": n_seg,
            "n_rows_kept": int(rows.numel()),
            "ret_mean": float(ret.mean()) if ret.numel() else float("nan"),
            "ret_std": float(ret.std()) if ret.numel() > 1 else 0.0,
            "ret_max": float(ret.max()) if ret.numel() else float("nan"),
            "sigma_scale": noise.adaptive.scale,
            "sigma_live": float(args.sigma) * noise.adaptive.scale,
            "tip_dev_mm": dev_mm,
            "dev_rms_mrad": float(rep.get("dev_rms_mrad", float("nan"))),
            "n_active_hands": n_active_hands,
            "nes_norm": nes_norm,
            "contact_share": float(rep.get("contact_share", float("nan"))),
            "wrist_share": float(rep.get("wrist_share", float("nan"))),
        }
        rec.update({f"loss/{k}": float(np.mean(v)) for k, v in losses.items() if v})
        rec.update({k: float(np.mean(v)) for k, v in agg.items()})
        emit(rec)
        print(f"[rl2 it{it:03d}] seg {n_seg:4d} ret {rec['ret_mean']:+7.2f}"
              f"+-{rec['ret_std']:5.2f} | sigma x{rec['sigma_scale']:.3f} "
              f"dev {dev_mm:5.2f} mm/{rec['dev_rms_mrad']:5.1f} mrad | "
              f"bc {rec.get('loss/bc', float('nan')):.4f} "
              f"dcost {rec.get('loss/cost_excess', float('nan')):.3f} "
              f"tauR {rec.get('loss/torque_corr', float('nan')):.3f} "
              f"| {rec['env_fps']:.0f} fps ({roll_s:.0f}s roll / {upd_s:.0f}s upd)",
              flush=True)

        # ---------------------------------------------------------- probe ----
        if (it + 1) % max(int(args.probe_every), 1) == 0:
            p = probe(f"iter{it}", int(args.probe_ticks))
            emit({"kind": "probe", "iter": it, **p})
            better = (p["miss_rate"] < best["miss_rate"] - 1e-9) or (
                abs(p["miss_rate"] - best["miss_rate"]) <= 1e-9
                and p["lat_abs_p50_ms"] < best["lat_abs_p50_ms"])
            print(f"[rl2 it{it:03d}] PROBE miss {100*p['miss_rate']:.1f}% "
                  f"in-disk {p['crossed_in_disk']} p50 {p['lat_abs_p50_ms']:.1f} ms "
                  f"speed {p['speed_in_band']}  {'** BEST **' if better else ''}",
                  flush=True)
            torch.save({"model": model.state_dict(), "model_cfg": ck["model_cfg"],
                        "scenario_cfg": ck.get("scenario_cfg"), "probe": p,
                        "iter": it}, out_dir / "last.pt")
            if better:
                best.update(miss_rate=p["miss_rate"],
                            lat_abs_p50_ms=p["lat_abs_p50_ms"], iter=it)
                torch.save({"model": model.state_dict(),
                            "model_cfg": ck["model_cfg"],
                            "scenario_cfg": ck.get("scenario_cfg"),
                            "probe": p, "iter": it}, out_dir / "best.pt")
            _reset(all_ids)
            q, qd = env.read_state()

    final = probe("final", int(args.probe_ticks))
    emit({"kind": "probe", "iter": "final", **final})
    torch.save({"model": model.state_dict(), "model_cfg": ck["model_cfg"],
                "scenario_cfg": ck.get("scenario_cfg"), "probe": final,
                "iter": "final"}, out_dir / "last.pt")
    (out_dir / "summary.json").write_text(json.dumps(
        {"args": vars(args), "lead_source": lead_src, "baseline": base,
         "final": final, "best": best,
         "wall_s": time.perf_counter() - t_start}, indent=2, default=float))
    print(f"\n[rl2] baseline miss {100*base['miss_rate']:.1f}% -> "
          f"final {100*final['miss_rate']:.1f}% (best {100*best['miss_rate']:.1f}% "
          f"at iter {best['iter']})", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
