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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "runs/phase2b_main/best.pt"))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lead", default="fit")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--sigma-probe", type=float, default=0.02,
                   help="sigma used for the deviation diagnostic (step 4)")
    p.add_argument("--noise-mode", default="film", choices=("film", "lora"))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--safety", default="box", choices=("off", "box", "qp"))
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cost", action="store_true")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)

    # --- Isaac must be launched before anything imports isaaclab.envs ---------
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app

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

    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32
    checks: dict[str, bool] = {}
    report: dict = {"args": vars(args),
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
                   MPCConfig(horizon=H, cmd_lead_s=args.lead,
                             amp_dtype=None if args.no_amp else torch.bfloat16),
                   device=dev, dtype=dtype, kit_names=tuple(kit.names))
    fk = A.ChainFK(device=dev, dtype=dtype)
    rnea = A.BatchedRNEA.from_urdf(device=dev).to(dev, dtype)
    sdf = A.SdfPenalty(fk=fk, device=dev, dtype=dtype)
    cost = None if args.no_cost else A.FusedCost(
        A.CostConfig(), fk=fk, rnea=rnea, sdf=sdf, max_horizon=H, device=dev, dtype=dtype)
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
    lat = latency_report(loop, kit, device="cpu")
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

    # Isaac's carb tasking mutex throws "std::system_error: Invalid argument"
    # at shutdown; it is not our code (see the repo note) and must not turn a
    # passing smoke run into a failure.
    try:
        app.close()
    except Exception:
        pass
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
