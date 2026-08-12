#!/usr/bin/env python
"""Isaac-FREE reference driver: the same loop, with ``BatchedPDSim`` as plant.

This exists for two reasons.

1. **It is the control experiment for the Isaac numbers.**  The headline of the
   RL-1 report is the closed-loop hit-latency histogram measured in Isaac
   against `REPORT.md` §4's analytic 63 % miss rate.  Those two were produced by
   different loops (``MPCGenerator.run_episode``, host-side, one env) so a
   difference could be the plant OR the loop.  Running THIS loop against the
   analytic plant isolates it: same schedule, same MPC, same reward, same
   measurement, only the plant differs.

2. It is how the batched machinery gets exercised in CI without Isaac.

    python isaac_rl/scripts/ref_loop.py --device cuda:1 --envs 64 --steps 250
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, frozen_minv, mass_matrix  # noqa: E402
from isaac_rl.loop import DrumMPCLoop  # noqa: E402
from isaac_rl.measure import latency_report  # noqa: E402
from isaac_rl.mpc import IsaacMPC, MPCConfig  # noqa: E402
from isaac_rl.perturb import NoiseConfig, ParameterNoise  # noqa: E402
from isaac_rl.reward import ExecutedReward, ExecutedStrikeScorer, RewardConfig  # noqa: E402
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig  # noqa: E402


def build(args, device, dtype=torch.float32):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    H = int((ck.get("scenario_cfg") or {}).get("horizon", 50))

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future, hit_lead_s=args.hit_lead)
    sched = BatchedSchedule(args.envs, scfg, device=device, dtype=dtype)
    rng = np.random.default_rng(args.seed)
    pick = [clips[i % len(clips)] for i in range(args.envs)]
    sched.reset(torch.arange(args.envs, device=device), pick, [kit] * args.envs,
                rng=rng, clip_ids=[i % len(clips) for i in range(args.envs)])

    mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s=args.lead,
                                           amp_dtype=None if args.no_amp else torch.bfloat16),
                   device=device, dtype=dtype, kit_names=tuple(kit.names))
    fk = A.ChainFK(device=device, dtype=dtype)
    rnea = A.BatchedRNEA.from_urdf(device=device).to(device, dtype)
    sdf = A.SdfPenalty(fk=fk, device=device, dtype=dtype)
    cost = A.FusedCost(A.CostConfig(), fk=fk, rnea=rnea, sdf=sdf, max_horizon=H,
                       device=device, dtype=dtype) if not args.no_cost else None
    scorer = ExecutedStrikeScorer(fk=fk, dt=A.CONTROL_DT, device=device, dtype=dtype)
    rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=cost, device=device, dtype=dtype)
    return model, kit, clips, sched, mpc, rew, rnea, fk


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "runs/phase2b_main/best.pt"))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lead", default="fit")
    p.add_argument("--hit-lead", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--noise-mode", default="film")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cost", action="store_true")
    p.add_argument("--strike-impedance", action="store_true",
                   help="soften elbow+wrist Kp/Kd around each hit")
    p.add_argument("--imp-kp", type=float, default=0.5)
    p.add_argument("--imp-kd", type=float, default=0.7)
    p.add_argument("--imp-window", type=int, default=3,
                   help="ramp ticks before AND after the hit")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32

    model, kit, clips, sched, mpc, rew, rnea, fk = build(args, dev, dtype)
    sim = BatchedPDSim(PDSimConfig(dt=A.CONTROL_DT), device=dev, dtype=dtype, rnea=rnea)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    kp, kd, ki = sim.sample_gains(args.envs, generator=gen)

    def plant(prev_cmd, cmd, q, qd, gain_scale=None):
        Minv = torch.linalg.inv(mass_matrix(rnea, q))
        q_next, qd_next, _int = sim.step(prev_cmd, cmd, q, qd, kp=kp, kd=kd, ki=ki, Minv=Minv,
                                         gain_scale=gain_scale)
        return q_next, qd_next

    noise = None
    if args.sigma > 0:
        noise = ParameterNoise.build(model, args.envs,
                                     NoiseConfig(mode=args.noise_mode, sigma=args.sigma,
                                                 seed=args.seed),
                                     device=dev, dtype=dtype)
        noise.attach()

    sampler = A.StartSampler(kit=kit, library=A.GoalPoseLibrary(), fk=fk, rnea=rnea,
                             sdf=A.SdfPenalty(fk=fk, device=dev, dtype=dtype))
    q0 = torch.as_tensor(
        np.stack([A.GoalPoseLibrary().home_q if hasattr(A.GoalPoseLibrary(), "home_q")
                  else np.zeros(17) for _ in range(args.envs)]),
        device=dev, dtype=dtype)
    qd0 = torch.zeros_like(q0)
    mpc.reset(torch.arange(args.envs, device=dev), q0)

    imp = None
    if args.strike_impedance:
        from amortized_opt.impedance import StrikeImpedance, StrikeImpedanceConfig
        imp = StrikeImpedance(
            StrikeImpedanceConfig(kp_scale=args.imp_kp, kd_scale=args.imp_kd,
                                  before_wp=args.imp_window, after_wp=args.imp_window),
            batch=args.envs, device=dev)
    loop = DrumMPCLoop(schedule=sched, mpc=mpc, reward=rew, plant=plant,
                       noise=noise, impedance=imp, record=True)
    q, qd, log = loop.rollout(q0, qd0, args.steps)

    stats = loop.stats.summary(args.envs)
    rep = latency_report(loop, kit, device="cpu")
    out = {"args": vars(args), "throughput": stats, "reward": log, "latency": rep}
    print(json.dumps(out, indent=2, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
