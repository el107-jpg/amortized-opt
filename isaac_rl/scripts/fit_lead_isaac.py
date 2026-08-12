#!/usr/bin/env python
"""Re-fit the per-joint reference lead against the ISAAC actuator, not the servo model.

`REPORT.md` §8, RL priority 3:

    The per-joint lead is the *control-side* compensation and it is a fit to one
    servo model (§7 E.7). ... Re-fit the vector against Isaac's actuator before
    the run -- 15 s of GPU time.

``scripts/fit_cmd_lead.py`` identifies ``delta`` by sweeping it and rolling 162
grid plans through ``amortized_opt.pd_sim.BatchedPDSim``, an analytic FOH plant
with a frozen effective mass matrix.  That plant is a *model*; PhysX with
``IdealPDActuator``, armature, self-collision and a 4 ms substep is not the same
operator, and the whole point of the RL stage is that the difference matters.

This script is the same identification with the plant swapped:

* build the same 162-case ``standard`` grid and the same champion plans;
* for each ``delta`` in the sweep, command ``q_ref(t + delta)`` -- through the
  identical admissibility chain (``lead_waypoints`` -> slew -> box) -- to a
  batch of Isaac envs, one env per case, in OPEN LOOP;
* take the per-joint argmin of ``|q_act - q_ref|`` RMS over the knots around
  the strike.

One env per case and one sweep value per rollout, so the whole sweep is
``len(deltas)`` rollouts of ``K`` control ticks at ``B = 162`` -- a few thousand
physics steps, which is the "15 s of GPU time" the report budgeted.

    python isaac_rl/scripts/fit_lead_isaac.py --device cuda:1 --out runs/rl1/lead_isaac.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse(argv=None):
    # --device is contributed by AppLauncher.add_app_launcher_args, and the
    # launcher MUST be built from the parsed Namespace -- the keyword form
    # selects a different kit experience (see smoke_rl.py).
    from isaaclab.app import AppLauncher as _AL
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(ROOT / "runs/phase2b_main/best.pt"))
    p.add_argument("--grid", default="standard")
    p.add_argument("--max-cases", type=int, default=162)
    p.add_argument("--deltas", type=float, nargs="*", default=None,
                   help="seconds; default 0..0.05 in 2.5 ms steps (the shipped sweep)")
    p.add_argument("--strike-window", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default="runs/rl1/lead_isaac.json")
    _AL.add_app_launcher_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app

    import numpy as np
    import torch

    import amortized_opt as A
    from amortized_opt.lead import LEAD_PRESETS, lead_waypoints
    from isaac_rl.env import make_env
    from isaac_rl.env_cfg import CONTROL_DT, DrumRLEnvCfg

    NAMES = A.G1_UPPER_BODY_JOINTS
    GROUPS = {
        "waist": [i for i, n in enumerate(NAMES) if n.startswith("waist_")],
        "shoulder": [i for i, n in enumerate(NAMES) if "shoulder" in n],
        "elbow": [i for i, n in enumerate(NAMES) if "elbow" in n],
        "wrist": [i for i, n in enumerate(NAMES) if "wrist" in n],
    }
    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32
    deltas = args.deltas or list(np.arange(0.0, 0.0501, 0.0025))

    # ---- the same plans fit_cmd_lead.py uses --------------------------------
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50)))
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    # TrajectoryScorer builds its own RNEA; inside Isaac that needs the frozen
    # tree (isaac_rl.rnea_cache).
    from isaac_rl.rnea_cache import rnea_from_cache_if_needed
    with rnea_from_cache_if_needed():
        scorer = A.TrajectoryScorer(device="cpu")
        cases = A.generate_grid(seed=args.seed, grid=args.grid, scorer=scorer)
    if args.max_cases and len(cases) > args.max_cases:
        cases = cases[:: max(len(cases) // args.max_cases, 1)][: args.max_cases]
    print(f"[fit-isaac] {len(cases)} cases, deltas "
          f"{deltas[0]*1000:.1f}..{deltas[-1]*1000:.1f} ms", flush=True)
    plans, _ = A.batched_model_plans(
        model, cases, device=str(dev), kit=kit, library=library, buckets=buckets,
        num_drums=mcfg.num_drums, num_future=mcfg.num_future, iterations=1,
        init="quintic", amp_dtype=None if args.no_amp else torch.bfloat16)

    B = len(cases)
    K = max(int(c.N) for c in cases)
    n = len(NAMES)
    Q = np.zeros((B, K + 1, n)); M = np.zeros((B, K + 1)); hit = np.zeros(B, np.int64)
    for i, c in enumerate(cases):
        qq = np.asarray(plans[int(c.case_id)], float)
        N = int(c.N)
        Q[i, : N + 1] = qq
        Q[i, N + 1:] = qq[-1]
        M[i, : N + 1] = 1.0
        hit[i] = int(np.clip(c.strike.waypoint(c.dt), 1, N))
    t = lambda a, d=dtype: torch.as_tensor(np.asarray(a), device=dev, dtype=d)  # noqa: E731
    Q = t(Q); mask = t(M); hitT = torch.as_tensor(hit, device=dev)
    q0 = t(np.stack([c.q_start for c in cases]))
    v0 = t(np.stack([c.qdot_start for c in cases]))
    q_lo, q_hi = A.urdf_position_limits(NAMES)
    q_lo, q_hi = t(q_lo), t(q_hi)
    step_max = t(A.isaac_velocity_limits(NAMES) * CONTROL_DT)

    # ---- the Isaac plant ------------------------------------------------------
    env = make_env(DrumRLEnvCfg(num_envs=B, device=args.device, seed=args.seed,
                                episode_length_s=60.0))
    env.reset()

    ref = Q[:, 1:]                                     # q_ref at dt .. K dt
    m_all = mask[:, 1:]
    kidx = torch.arange(K, device=dev).view(1, -1)
    m_str = m_all * (kidx >= (hitT.view(-1, 1) - int(args.strike_window))).to(dtype) \
        * (kidx <= hitT.view(-1, 1)).to(dtype)

    @torch.no_grad()
    def rollout(delta: float):
        """Open-loop: command ``q_ref(t + delta)`` for K ticks, record ``q_act``."""
        f = torch.full((n,), float(delta) / CONTROL_DT, device=dev, dtype=dtype)
        led = lead_waypoints(ref, f)
        env.reset_to(torch.arange(B, device=dev), q0, v0)
        prev = q0.clone()
        acts = []
        q, qd = env.read_state()
        for i in range(K):
            cmd = prev + torch.clamp(led[:, i] - prev, -step_max, step_max)
            cmd = torch.clamp(cmd, q_lo, q_hi)
            q, qd = env.plant(prev, cmd, q, qd)
            prev = cmd
            acts.append(q)
        return torch.stack(acts, dim=1)                # (B, K, n)

    res = {}
    for d in deltas:
        qa = rollout(float(d))
        err = (qa - ref) ** 2
        rms_all = torch.sqrt((err * m_all.unsqueeze(-1)).sum((0, 1))
                             / m_all.sum().clamp_min(1.0))
        rms_str = torch.sqrt((err * m_str.unsqueeze(-1)).sum((0, 1))
                             / m_str.sum().clamp_min(1.0))
        res[float(d)] = (rms_all.cpu().numpy(), rms_str.cpu().numpy())
        print(f"  delta {1000*d:5.1f} ms  all {1000*float(rms_all.mean()):7.2f} mrad  "
              f"strike {1000*float(rms_str.mean()):7.2f} mrad", flush=True)

    out = {"ckpt": args.ckpt, "plant": "isaac", "n_cases": B,
           "deltas_s": [float(x) for x in deltas],
           "strike_window": args.strike_window,
           "sim_dt": env.cfg.sim.dt, "decimation": env.cfg.decimation,
           "analytic_fit_s": list(LEAD_PRESETS["fit"])}
    ds = np.array(sorted(res))
    for tag, key in (("all", 0), ("strike", 1)):
        curves = np.stack([res[d][key] for d in ds])
        best = ds[curves.argmin(axis=0)]
        out[f"best_{tag}_s"] = [float(x) for x in best]
        out[f"rms_at_0_{tag}_mrad"] = [float(1000 * x) for x in curves[0]]
        out[f"rms_at_best_{tag}_mrad"] = [float(1000 * curves[:, j].min())
                                          for j in range(curves.shape[1])]
        print(f"\n=== ISAAC, window: {tag} ===")
        print(f"{'joint':<28}{'isaac ms':>10}{'analytic ms':>13}"
              f"{'rms@0':>9}{'rms@best':>10}{'gain %':>8}")
        for j, nm in enumerate(NAMES):
            r0, rb = 1000 * curves[0, j], 1000 * curves[:, j].min()
            print(f"{nm:<28}{1000*best[j]:>10.1f}{1000*LEAD_PRESETS['fit'][j]:>13.1f}"
                  f"{r0:>9.1f}{rb:>10.1f}{100*(1-rb/max(r0,1e-9)):>8.0f}")
        for g, idx in GROUPS.items():
            print(f"  group {g:<10} isaac median {1000*np.median(best[idx]):6.1f} ms   "
                  f"analytic {1000*np.median([LEAD_PRESETS['fit'][i] for i in idx]):6.1f} ms")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}", flush=True)
    # Isaac's ``app.close()`` HANGS in the carb tasking mutex long after the
    # results are on disk (measured: the JSON was written and the process was
    # still holding the GPU minutes later, serialising any queued run behind
    # the ``timeout``).  Everything is flushed; leave without teardown.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
