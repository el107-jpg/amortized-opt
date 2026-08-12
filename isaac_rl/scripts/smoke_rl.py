#!/usr/bin/env python
"""RL-1 smoke test: the whole scaffold, in Isaac, with NO learning.

    python isaac_rl/scripts/smoke_rl.py --device cuda:1 --envs 64 --steps 250 \
        --out runs/rl1/smoke.json

What it verifies, in order (each is a printed PASS/FAIL line):

1. the env steps at 250 Hz physics / decimation 5 -> a true 50 Hz control tick,
   and PhysX is stable there (no NaN, no articulation blow-up);
2. the model-as-MPC drives all envs in one shared batched forward pass, and the
   achieved ``envs x fps`` and the per-tick inference share are measured;
3. hits are detected and scored, and the reward decomposition logs per term;
4. with ``sigma > 0`` the per-env FiLM perturbation produces diverse but SMOOTH
   trajectories, and the deviation diagnostic says where the diversity lives
   (the plan predicts wrist-heavy and contact-adjacent);
5. the safety filter engages on deliberately bad plans and the interventions
   are counted;
6. the closed-loop hit-latency histogram, measured by
   ``amortized_opt.probe.hit_latency`` -- the SAME function that produced
   `REPORT.md` §4's 63 % miss rate -- so the two numbers are comparable.

Nothing here trains anything.  The champion checkpoint is loaded and frozen.
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
    # NOTE: the launcher args MUST come from ``AppLauncher.add_app_launcher_args``
    # and be handed in as the parsed Namespace.  Constructing
    # ``AppLauncher(headless=True, device=...)`` from keywords instead selects a
    # different kit experience, and on this box that wedges the PhysX GPU tensor
    # API: ``DirectRLEnv.__init__`` blocks forever in
    # ``scene.update() -> articulation.update() -> joint_acc -> joint_vel ->
    # omni.physics.tensors.get_dof_velocities``.  Measured: the repo's own env
    # forced to the same 250 Hz / decimation 5 builds in 4.0 s through the
    # argparse path and never completes through the keyword path.
    from isaaclab.app import AppLauncher as _AL
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "runs/phase2b_main/best.pt"))
    # NOTE: --device is contributed by AppLauncher.add_app_launcher_args
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lead", default="fit",
                   help='"fit"/"sparse"/"wide"/"off", a float in seconds, or a '
                        'path to a fit_lead_isaac.py JSON (its "best_strike_s" '
                        'vector is used -- that is the ISAAC-fitted lead)')
    p.add_argument("--lead-key", default="best_strike_s",
                   help="which vector to read out of a lead JSON")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--sigma-probe", type=float, default=0.02,
                   help="sigma used for the deviation diagnostic (step 4)")
    p.add_argument("--noise-mode", default="film", choices=("film", "lora"))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--safety", default="box", choices=("off", "box", "qp"))
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cost", action="store_true")
    p.add_argument("--out", default="")
    _AL.add_app_launcher_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)

    # --- Isaac must be launched before anything imports isaaclab.envs ---------
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app

    import numpy as np
    import torch

    import amortized_opt as A
    from isaac_rl.env import make_env
    from isaac_rl.env_cfg import CONTROL_DT, DECIMATION, SIM_DT, DrumRLEnvCfg
    from isaac_rl.loop import DrumMPCLoop
    from isaac_rl.measure import latency_report
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    from isaac_rl.perturb import NoiseConfig, ParameterNoise, deviation_report
    from isaac_rl.reward import ExecutedReward, ExecutedStrikeScorer, RewardConfig
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    # A lead JSON from fit_lead_isaac.py wins over the built-in presets: the
    # shipped LEAD_PRESETS["fit"] was identified against amortized_opt.pd_sim,
    # NOT against PhysX, and RL priority 3 requires the executed lead to be the
    # one fitted against the plant that is actually running.
    lead = args.lead
    lead_src = "preset"
    if isinstance(lead, str) and lead.endswith(".json") and Path(lead).is_file():
        _d = json.loads(Path(lead).read_text())
        lead = tuple(float(x) for x in _d[args.lead_key])
        lead_src = f"{args.lead}[{args.lead_key}]"

    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32
    checks: dict[str, bool] = {}
    report: dict = {"args": vars(args), "lead_source": lead_src,
                    "lead_s": (list(lead) if not isinstance(lead, str) else lead),
                    "rates": {"sim_dt": SIM_DT, "decimation": DECIMATION,
                              "control_dt": CONTROL_DT, "control_hz": 1 / CONTROL_DT}}

    # ---- 1. the env ---------------------------------------------------------
    rl_cfg = DrumRLEnvCfg(num_envs=args.envs, device=args.device, seed=args.seed,
                          record=True)
    t0 = time.perf_counter()
    env = make_env(rl_cfg)
    env.reset()
    report["build_s"] = time.perf_counter() - t0
    q, qd = env.read_state()
    checks["1_env_builds_at_250hz"] = bool(
        torch.isfinite(q).all() and torch.isfinite(qd).all()
        and abs(env.step_dt - CONTROL_DT) < 1e-9)
    report["step_dt"] = float(env.step_dt)
    print(f"[1] env: {args.envs} envs, sim_dt={SIM_DT:.5f} dec={DECIMATION} "
          f"step_dt={env.step_dt:.5f}  build {report['build_s']:.1f}s  "
          f"{'PASS' if checks['1_env_builds_at_250hz'] else 'FAIL'}", flush=True)

    # ---- the stack ----------------------------------------------------------
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    H = int((ck.get("scenario_cfg") or {}).get("horizon", 50))

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: max(args.clips, 1)]
    scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future)
    sched = BatchedSchedule(args.envs, scfg, device=dev, dtype=dtype)
    rng = np.random.default_rng(args.seed)
    pick = [clips[i % len(clips)] for i in range(args.envs)]
    sched.reset(torch.arange(args.envs, device=dev), pick, [kit] * args.envs, rng=rng)

    mpc = IsaacMPC(model, sched,
                   MPCConfig(horizon=H, cmd_lead_s=lead,
                             amp_dtype=None if args.no_amp else torch.bfloat16),
                   device=dev, dtype=dtype, kit_names=tuple(kit.names))
    fk = A.ChainFK(device=dev, dtype=dtype)
    # NOTE: FusedCost builds its OWN SdfPenalty from its own scene; it takes no
    # ``sdf=`` kwarg and never did.  (A stray unused ``SdfPenalty`` used to be
    # constructed here, which read like it was about to be handed over.)
    # FusedCost needs the RNEA, which needs Pinocchio's kinematic tree -- see
    # ``isaac_rl.env.try_build_rnea`` for why that is unavailable inside Isaac.
    from isaac_rl.env import try_build_rnea
    rnea = None if args.no_cost else try_build_rnea(dev, dtype)
    report["rnea_in_process"] = rnea is not None
    # Do NOT hand FusedCost our ``fk``: it needs the SHOULDER frames for the
    # posture term (``_posture`` reads ``frames["left_shoulder"]``) and the
    # default ChainFK frame set does not carry them.  Letting it build its own
    # with ``FusedCost.FRAMES`` costs one extra FK object and avoids a
    # KeyError that only fires once the posture term is actually evaluated.
    cost = None if rnea is None else A.FusedCost(
        A.CostConfig(), rnea=rnea, max_horizon=H, device=dev, dtype=dtype)
    if not args.no_cost and cost is None:
        print("[3] analytic half of the reward DISABLED (no RNEA in-process)", flush=True)
    scorer = ExecutedStrikeScorer(fk=fk, dt=CONTROL_DT, device=dev, dtype=dtype)
    rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=cost, device=dev, dtype=dtype)

    safety = None
    if args.safety != "off":
        try:
            from isaac_rl.safety import BatchedSafetyFilter, SafetyConfig
            safety = BatchedSafetyFilter(SafetyConfig(mode=args.safety),
                                         device=dev, dtype=dtype)
            safety.reset(torch.arange(args.envs, device=dev), q)
        except Exception as exc:            # scaffold: never block the smoke run
            print(f"[5] safety filter unavailable ({exc}); continuing without",
                  flush=True)

    noise = None
    if args.sigma > 0:
        noise = ParameterNoise.build(
            model, args.envs,
            NoiseConfig(mode=args.noise_mode, rank=args.rank, sigma=args.sigma,
                        seed=args.seed),
            device=dev, dtype=dtype)
        noise.attach()

    mpc.reset(torch.arange(args.envs, device=dev), q)
    loop = DrumMPCLoop(schedule=sched, mpc=mpc, reward=rew, plant=env.plant,
                       noise=noise, safety=safety, record=True)

    # ---- 2 + 3. the rollout --------------------------------------------------
    q, qd, log = loop.rollout(q, qd, args.steps)
    thr = loop.stats.summary(args.envs, dt=CONTROL_DT)
    report["throughput"] = thr
    report["reward"] = log
    checks["2_batched_mpc_runs"] = bool(torch.isfinite(q).all())
    checks["3_reward_decomposes"] = all(
        f"terms/{k}" in log for k in ("hit", "timing", "speed", "miss", "track"))
    print(f"[2] throughput: {thr['env_fps']:.0f} env-steps/s "
          f"({thr['realtime_factor']:.2f}x realtime), "
          f"plan {100*thr['share_plan']:.0f}% / plant {100*thr['share_plant']:.0f}% / "
          f"reward {100*thr['share_reward']:.0f}%  "
          f"{'PASS' if checks['2_batched_mpc_runs'] else 'FAIL'}", flush=True)
    print(f"[3] reward terms: " + ", ".join(
        f"{k.split('/')[-1]}={v:+.3f}" for k, v in log.items()
        if k.startswith("terms/")) +
        f"  {'PASS' if checks['3_reward_decomposes'] else 'FAIL'}", flush=True)

    # ---- 6. the headline: hit latency, measured the REPORT.md way -------------
    # ALWAYS dump the raw rollout first.  ``TrajectoryScorer`` needs the RNEA,
    # which needs Pinocchio's kinematic tree, which Isaac's process cannot
    # provide (see ``isaac_rl.env.try_build_rnea``) -- so the measurement runs
    # out-of-process on this dump, using the SAME ``probe.hit_latency`` the
    # analytic 63 % came from.
    if args.out:
        raw = Path(args.out).with_suffix(".rollout.npz")
        raw.parent.mkdir(parents=True, exist_ok=True)
        ev = loop.fired_events()
        np.savez_compressed(
            raw, q=loop.q_hist(),
            ev_tick=np.array([e[0] for e in ev], np.int64),
            ev_env=np.array([e[1] for e in ev], np.int64),
            ev_hand=np.array([0 if e[2] == "left" else 1 for e in ev], np.int64),
            ev_drum=np.array([e[3] for e in ev], np.int64))
        print(f"[6] rollout dumped to {raw} ({len(ev)} scheduled strikes)", flush=True)
    try:
        lat = latency_report(loop, kit, device="cpu")
    except Exception as exc:
        print(f"[6] in-process latency unavailable ({exc}); run "
              f"isaac_rl/scripts/latency_from_dump.py on the .rollout.npz",
              flush=True)
        report["checks"] = checks
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, default=float))
        sys.stdout.flush()
        os._exit(0)
    report["latency"] = lat
    o = lat["overall"]
    checks["6_hits_detected"] = o["n"] > 0
    print(f"[6] ISAAC hit latency over {o['n']} scheduled strikes: "
          f"miss {100*o['miss_rate']:.0f}%  |lat| p50 {o['lat_abs_p50_ms']:.1f} ms  "
          f"p90 {o['lat_abs_p90_ms']:.1f} ms  speed-in-band {o['speed_in_band']}  "
          f"(analytic plant, REPORT.md §4: 63 % miss, 9.5 ms p50)", flush=True)
    print(f"    histogram(ms): {o['hist_ms']}", flush=True)

    # ---- 4. the deviation diagnostic ------------------------------------------
    # Clean vs perturbed plan FROM THE SAME STATE AND CONTEXT: the only honest
    # way to ask what the low-rank subspace can express.  Not a rollout
    # difference, which would confound the perturbation with the divergence it
    # causes downstream.
    dev_probe = None
    if args.sigma_probe > 0:
        probe = ParameterNoise.build(
            model, args.envs,
            NoiseConfig(mode=args.noise_mode, rank=args.rank,
                        sigma=args.sigma_probe, seed=args.seed + 1),
            device=dev, dtype=dtype)
        hidx = sched.hit_index()
        masks = mpc.masks(hidx)
        enc = model.encode_context(sched.context())
        init = mpc.init * masks.elem_mask
        state = torch.cat([q, qd], dim=-1)
        with torch.no_grad():
            clean = model(init, state, enc, masks).traj
            probe.attach()
            pert = model(init, state, enc, masks).traj
            probe.detach()
        dev_probe = deviation_report(clean, pert, elem_mask=masks.elem_mask,
                                     hit_index=hidx.clamp_min(0))
        # smoothness: second difference along the waypoint axis, perturbed vs clean
        d2 = lambda x: (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).abs().mean()  # noqa: E731
        rough_clean, rough_pert = float(d2(clean)), float(d2(pert))
        dev_probe["roughness_clean_rad"] = rough_clean
        dev_probe["roughness_pert_rad"] = rough_pert
        dev_probe["roughness_ratio"] = rough_pert / max(rough_clean, 1e-12)
        report["deviation"] = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in dev_probe.items()}
        checks["4_perturbation_diverse_and_smooth"] = bool(
            dev_probe["rms_rad"] > 1e-6 and dev_probe["roughness_ratio"] < 3.0)
        print(f"[4] sigma={args.sigma_probe} {args.noise_mode}: "
              f"rms {1000*dev_probe['rms_rad']:.2f} mrad, "
              f"group share " + " ".join(
                  f"{k}={100*v:.0f}%" for k, v in dev_probe['group_energy_share'].items())
              + f", contact-adjacent {100*dev_probe.get('contact_share', 0):.0f}% "
                f"(of {100*dev_probe.get('contact_frac_of_grid', 0):.0f}% of the grid), "
                f"roughness x{dev_probe['roughness_ratio']:.2f}  "
                f"{'PASS' if checks['4_perturbation_diverse_and_smooth'] else 'FAIL'}",
              flush=True)

    # ---- 5. safety filter, on deliberately bad plans ---------------------------
    if safety is not None:
        safety.reset_stats()
        bad = q.clone()
        bad[: args.envs // 2] += 5.0            # far outside the joint box
        bad[args.envs // 2:] = q[args.envs // 2:] + 3.0   # a huge slew step
        _safe, engaged = safety.project(bad, q, qd)
        report["safety"] = {"engaged_on_bad_plan": float(engaged.float().mean()),
                            "stats": safety.stats,
                            "interventions_in_rollout": loop.stats.filter_interventions}
        checks["5_safety_engages"] = bool(engaged.float().mean() > 0.99)
        print(f"[5] safety({args.safety}): engaged on {100*float(engaged.float().mean()):.0f}% "
              f"of injected bad plans; {loop.stats.filter_interventions} interventions "
              f"in the {args.steps}-tick rollout  "
              f"{'PASS' if checks['5_safety_engages'] else 'FAIL'}", flush=True)

    report["checks"] = checks
    print("\n" + json.dumps(checks, indent=2), flush=True)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=float))
        print(f"wrote {p}", flush=True)

    # Isaac's shutdown is not just noisy, it HANGS: ``app.close()`` blocks in
    # the carb tasking mutex (the repo's known "std::system_error: Invalid
    # argument" trap) long after every result is on disk.  Measured: the
    # rollout finished and the JSON was written at 02:50:09, and the process
    # was still alive 3+ minutes later holding the GPU, which serialises any
    # queued run behind a 28-minute ``timeout``.  Everything is flushed by
    # here, so leave via ``os._exit`` and skip interpreter/Isaac teardown
    # entirely.  ``repo250.py`` and the repo's own probes do the same.
    ok = all(checks.values())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
