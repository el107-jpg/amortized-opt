#!/usr/bin/env python
"""RL-2 work item 0: settle the safety-filter confound before training anything.

RL-1b measured 87 % miss (ff ON, Isaac-fitted lead) and flagged that the QP
backstop engaged on **26-34 %** of env-ticks in that run against **0 %** in the
ff-OFF run, so an unknown share of the degradation is the filter and not the
plant.  This script runs the SAME smoke protocol several times in one Isaac
process, from the same seed and the same start state, and separates:

    off             no filter at all -- the unconfounded plant
    box_noexempt    what RL-1b actually ran: box + slew + 2 SDF clearance steps,
                    with NO target-drum exemption
    box_exempt      the same filter, given ``target_drum`` so each hand's TIP is
                    exempt from the drum it is aiming at
    box_nosdf       box + slew only (sdf_steps=0): the hard, unambiguous limits

and, on every tick of every armed run, audits whether the intervention was
*physically justified*:

* **box** and **slew** are hard machine limits (URDF/posture position box,
  per-joint speed rating).  A hit there means the raw command genuinely asked
  for something the robot cannot do, and the correction is justified by
  construction.  We record how deep the violation was, in rad.
* **sdf** is a *margin*, and margins are fitted.  We record the raw command's
  penetration depth (m, summed over violated pairs) BOTH with and without the
  target-drum exemption.  If the penetration is entirely the striking tip
  against its own target drum, the filter is fighting the task, not the
  geometry, and the intervention is NOT justified.

Usage (note the device recipe -- Warp initialises on the visible cuda:0
regardless of --device, so mask at the driver level):

    CUDA_VISIBLE_DEVICES=1 python isaac_rl/scripts/safety_ab.py \
        --device cuda:0 --envs 32 --steps 200 \
        --lead runs/rl1b/lead_isaac.json --out runs/rl2/safety_ab.json
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

ARMS = ("off", "box_noexempt", "box_exempt", "box_nosdf")


def parse(argv=None):
    from isaaclab.app import AppLauncher as _AL
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "runs/phase2b_main/best.pt"))
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--lead", default=str(ROOT / "runs/rl1b/lead_isaac.json"))
    p.add_argument("--lead-key", default="best_strike_s")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--out", default="")
    _AL.add_app_launcher_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    from isaaclab.app import AppLauncher
    _app = AppLauncher(args).app

    import numpy as np
    import torch

    import amortized_opt as A
    from isaac_rl.env import make_env, try_build_rnea
    from isaac_rl.env_cfg import CONTROL_DT, DrumRLEnvCfg
    from isaac_rl.loop import DrumMPCLoop
    from isaac_rl.measure import latency_report
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    from isaac_rl.reward import ExecutedReward, ExecutedStrikeScorer, RewardConfig
    from isaac_rl.safety import BatchedSafetyFilter, SafetyConfig
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32
    B = int(args.envs)

    lead, lead_src = args.lead, "preset"
    if isinstance(lead, str) and lead.endswith(".json") and Path(lead).is_file():
        lead = tuple(float(x) for x in json.loads(Path(lead).read_text())[args.lead_key])
        lead_src = f"{args.lead}[{args.lead_key}]"

    # ------------------------------------------------------------- the stack --
    rl_cfg = DrumRLEnvCfg(num_envs=B, device=args.device, seed=args.seed, record=True)
    env = make_env(rl_cfg)
    env.reset()
    q0, qd0 = env.read_state()
    q0, qd0 = q0.clone(), qd0.clone()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    H = int((ck.get("scenario_cfg") or {}).get("horizon", 50))

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: max(args.clips, 1)]
    fk = A.ChainFK(device=dev, dtype=dtype)
    rnea = try_build_rnea(dev, dtype)
    cost = None if rnea is None else A.FusedCost(
        A.CostConfig(), rnea=rnea, max_horizon=H, device=dev, dtype=dtype)
    scorer = ExecutedStrikeScorer(fk=fk, dt=CONTROL_DT, device=dev, dtype=dtype)

    # An INDEPENDENT SdfPenalty, used only to audit -- never to correct.
    scene = A.DrumSceneSpec.from_layout()
    audit_sdf = A.SdfPenalty(scene, margin=SafetyConfig.margin, power=1,
                             include_self=True, device=dev, dtype=dtype)
    q_lo, q_hi = A.urdf_position_limits(A.G1_UPPER_BODY_JOINTS,
                                        limit_margin=0.0, posture_limits=True)
    q_lo = torch.as_tensor(np.asarray(q_lo, np.float32), device=dev)
    q_hi = torch.as_tensor(np.asarray(q_hi, np.float32), device=dev)
    dq_max = torch.as_tensor(
        np.asarray(A.isaac_velocity_limits(A.G1_UPPER_BODY_JOINTS) * CONTROL_DT,
                   np.float32), device=dev)

    # ------------------------------------------------------- the audit shim --
    class AuditFilter:
        """Delegates to the real filter; measures what it was asked to fix."""

        def __init__(self, filt):
            self.f = filt
            self.acc = {k: 0.0 for k in (
                "n_calls", "n_env_ticks", "engaged",
                "box_depth_rad", "slew_depth_rad",
                "pen_raw_m", "pen_raw_exempt_m", "pen_after_m",
                "corr_rad_sum", "corr_rad_max", "n_pen_raw", "n_pen_raw_exempt")}

        def reset(self, ids, q):
            self.f.reset(ids, q)

        def project(self, cmd, q, qd, *, target_drum=None):
            with torch.no_grad():
                tl = tr = None
                if target_drum is not None:
                    t = torch.as_tensor(target_drum, device=cmd.device).long()
                    tl, tr = (t[:, 0], t[:, 1]) if t.dim() == 2 else (t, t)
                pen_raw = audit_sdf(cmd)
                pen_ex = audit_sdf(cmd, target_drum_left=tl, target_drum_right=tr) \
                    if tl is not None else pen_raw
                box_d = (cmd - cmd.clamp(q_lo, q_hi)).abs().amax(dim=-1)
                prev = self.f._prev_cmd
                slew_d = torch.zeros_like(box_d) if prev is None else \
                    ((cmd - prev).abs() - dq_max).clamp_min(0).amax(dim=-1)
            safe, engaged = self.f.project(cmd, q, qd, target_drum=target_drum)
            with torch.no_grad():
                a = self.acc
                a["n_calls"] += 1
                a["n_env_ticks"] += cmd.shape[0]
                a["engaged"] += float(engaged.sum())
                a["box_depth_rad"] += float(box_d.sum())
                a["slew_depth_rad"] += float(slew_d.sum())
                a["pen_raw_m"] += float(pen_raw.sum())
                a["pen_raw_exempt_m"] += float(pen_ex.sum())
                a["n_pen_raw"] += float((pen_raw > 1e-4).sum())
                a["n_pen_raw_exempt"] += float((pen_ex > 1e-4).sum())
                a["pen_after_m"] += float(audit_sdf(safe).sum())
                corr = (safe - cmd).abs().amax(dim=-1)
                a["corr_rad_sum"] += float(corr.sum())
                a["corr_rad_max"] = max(a["corr_rad_max"], float(corr.max()))
            return safe, engaged

        def report(self):
            a = dict(self.acc)
            n = max(a["n_env_ticks"], 1)
            return {
                **a,
                "engaged_frac": a["engaged"] / n,
                "mean_box_depth_rad": a["box_depth_rad"] / n,
                "mean_slew_depth_rad": a["slew_depth_rad"] / n,
                "mean_pen_raw_m": a["pen_raw_m"] / n,
                "mean_pen_raw_exempt_m": a["pen_raw_exempt_m"] / n,
                "mean_pen_after_m": a["pen_after_m"] / n,
                "frac_pen_raw": a["n_pen_raw"] / n,
                "frac_pen_raw_exempt": a["n_pen_raw_exempt"] / n,
                "mean_corr_rad": a["corr_rad_sum"] / n,
                "filter_stats": self.f.stats,
            }

    # ------------------------------------------------------------- one arm ----
    def run(arm: str) -> dict:
        rng = np.random.default_rng(args.seed)
        scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                              num_future=mcfg.num_future)
        sched = BatchedSchedule(B, scfg, device=dev, dtype=dtype)
        pick = [clips[i % len(clips)] for i in range(B)]
        sched.reset(torch.arange(B, device=dev), pick, [kit] * B, rng=rng,
                    clip_ids=[i % len(clips) for i in range(B)])
        mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s=lead),
                       device=dev, dtype=dtype, kit_names=tuple(kit.names))
        rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=cost,
                             device=dev, dtype=dtype)

        env.reset_to(torch.arange(B, device=dev), q0, qd0)
        q, qd = env.read_state()
        mpc.reset(torch.arange(B, device=dev), q)

        safety, exempt = None, True
        if arm != "off":
            # "default" rather than "reduce-overhead": three filter instances
            # and an eager audit SdfPenalty share this process, and CUDA-graph
            # capture interleaved with eager autograd is not a risk worth
            # taking for a diagnostic whose throughput does not matter.
            scfg_s = SafetyConfig(mode="box", sdf_compile="default",
                                  sdf_steps=0 if arm == "box_nosdf" else 2)
            filt = BatchedSafetyFilter(scfg_s, device=dev, dtype=dtype)
            filt.reset(torch.arange(B, device=dev), q)
            safety = AuditFilter(filt)
            exempt = arm != "box_noexempt"

        loop = DrumMPCLoop(schedule=sched, mpc=mpc, reward=rew, plant=env.plant,
                           safety=safety, record=True,
                           safety_target_exempt=exempt)
        t0 = time.perf_counter()
        q, qd, log = loop.rollout(q, qd, args.steps)
        wall = time.perf_counter() - t0
        out = {"arm": arm, "wall_s": wall,
               "throughput": loop.stats.summary(B, dt=CONTROL_DT),
               "reward": log,
               "interventions": loop.stats.filter_interventions}
        if safety is not None:
            out["audit"] = safety.report()
        try:
            out["latency"] = latency_report(loop, kit, device="cpu")
        except Exception as exc:
            out["latency_error"] = repr(exc)
        return out

    results = {}
    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        # "off#2" repeats the "off" arm under a distinct key -- repeats are how
        # the noise floor of this whole comparison gets measured.
        arm = spec.split("#", 1)[0]
        print(f"\n===== arm: {spec} =====", flush=True)
        r = run(arm)
        r["spec"] = spec
        results[spec] = r
        o = r.get("latency", {}).get("overall")
        if o:
            print(f"[{spec}] n={o['n']} miss {100*o['miss_rate']:.0f}%  "
                  f"|lat| p50 {o['lat_abs_p50_ms']:.1f} p90 {o['lat_abs_p90_ms']:.1f} ms  "
                  f"in-disk {o['crossed_in_disk']}  "
                  f"speed-in-band {o['speed_in_band']}  hist {o['hist_ms']}", flush=True)
        if "audit" in r:
            a = r["audit"]
            print(f"[{spec}] engaged {100*a['engaged_frac']:.1f}%  "
                  f"box depth {1000*a['mean_box_depth_rad']:.3f} mrad  "
                  f"slew depth {1000*a['mean_slew_depth_rad']:.3f} mrad  "
                  f"raw pen {1000*a['mean_pen_raw_m']:.2f} mm "
                  f"(exempt {1000*a['mean_pen_raw_exempt_m']:.2f} mm)  "
                  f"rows pen {100*a['frac_pen_raw']:.1f}% "
                  f"(exempt {100*a['frac_pen_raw_exempt']:.1f}%)  "
                  f"mean corr {1000*a['mean_corr_rad']:.2f} mrad", flush=True)
        print(f"[{spec}] interventions {r['interventions']}  "
              f"fps {r['throughput']['env_fps']:.0f}", flush=True)

    report = {"args": vars(args), "lead_source": lead_src, "arms": results}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {p}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'arm':<14} {'in-disk':>8} {'miss':>7} {'p50ms':>7} {'p90ms':>7} "
          f"{'speed':>7} {'engaged':>8}")
    for spec, r in results.items():
        o = r.get("latency", {}).get("overall", {})
        eng = r.get("audit", {}).get("engaged_frac", 0.0)
        print(f"{spec:<14} {o.get('crossed_in_disk', '?'):>8} "
              f"{100*o.get('miss_rate', float('nan')):>6.0f}% "
              f"{o.get('lat_abs_p50_ms', float('nan')):>7.1f} "
              f"{o.get('lat_abs_p90_ms', float('nan')):>7.1f} "
              f"{o.get('speed_in_band', '?'):>7} {100*eng:>7.1f}%")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
