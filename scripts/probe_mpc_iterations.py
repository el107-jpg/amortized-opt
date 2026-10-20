#!/usr/bin/env python
"""Paired closed-loop one/two-pass MPC with every-plan feasibility auditing.

Uses the deployed IsaacMPC scheduler, encoder cache, terminal initializer,
warm-start shift, lead/slew/box path. Plant is analytic PD or an ideal tracker;
this script does not launch Isaac or change model weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, frozen_minv
from amortized_opt.train_cost import (
    _index_batch, _wtensors, apply_bounce, build_costs, phase2_config_from_checkpoint,
)
from isaac_rl.mpc import IsaacMPC, MPCConfig
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig
from isaac_rl.reward import ExecutedStrikeScorer
from probe_mpc_plan_drift import BANDS, _stats, evaluate
from probe_replay_boundary import _load_al, _scenario_from_ckpt


def score_events(events, history, scorer, device):
    """Score pre-bounce plant geometry at the hit tick and +/-3 ticks."""
    outcomes = []
    q = torch.stack(history)
    for event in events:
        tick, env, hand = event["tick"], event["env"], event["hand"]
        offsets = list(range(-3, 4))
        offsets = [off for off in offsets if 0 <= tick+off < len(q)]
        indices = torch.tensor([tick+off for off in offsets], device=device)
        ctx = _index_batch({"strike": event["strike"]},
                           torch.zeros(len(offsets), device=device, dtype=torch.long))["strike"]
        poses = q[indices, env]
        dist, _sigma, point = scorer.best_contact(poses, ctx)
        axial, lateral = scorer.head_frame(point, ctx)
        on_head = dist[:, hand] <= 1e-6
        zero = offsets.index(0)
        rows = []
        for k, offset in enumerate(offsets):
            rows.append(dict(offset_ticks=offset, in_region=bool(on_head[k]),
                             region_excess_mm=float(dist[k, hand])*1000,
                             axial_mm=float(axial[k, hand])*1000,
                             lateral_beyond_mm=float(lateral[k, hand])*1000))
        outcomes.append(dict(env=env, hand=hand, tick=tick, drum=event["drum"],
                             at_tick_in_region=bool(on_head[zero]),
                             within_60ms_in_region=bool(on_head.any()),
                             samples=rows))
    return dict(scheduled=len(outcomes),
                at_tick_in_region=sum(e["at_tick_in_region"] for e in outcomes),
                within_60ms_in_region=sum(e["within_60ms_in_region"] for e in outcomes),
                events=outcomes)


def summarize_plans(values, hit, live):
    groups = {"all_active": live}
    groups.update({name: live & (hit >= lo) & (hit <= hi)
                   for name, (lo, hi) in BANDS.items()})
    result = {}
    for name, take in groups.items():
        if not take.any():
            continue
        report = dict(n=int(take.sum()))
        for stage, metrics in values.items():
            report[stage] = {
                key: float(metrics[key][take].mean()) for key in
                ("hard_ok", "on_head", "within_5mm", "geometric_ok")}
            report[stage].update({key: _stats(metrics[key][take]) for key in
                ("terminal_region_dist", "collision_min_sdf", "torque_excess_max",
                 "velocity_excess_max", "cost")})
            report[stage]["failed_plans"] = int((~metrics["hard_ok"][take]).sum())
            report[stage]["terminal_failed"] = int((metrics["terminal_region_dist"][take] > 1e-6).sum())
            report[stage]["collision_failed"] = int((metrics["collision_min_sdf"][take] < 0).sum())
            report[stage]["torque_failed"] = int((metrics["torque_excess_max"][take] > 1e-6).sum())
        if "pass2" in values:
            before = values["pass1"]["hard_ok"]
            after = values["pass2"]["hard_ok"]
            report["pass2_repaired"] = int((take & ~before & after).sum())
            report["pass2_damaged"] = int((take & before & ~after).sum())
        result[name] = report
    return result


@torch.no_grad()
def run(model, cfg, scenario, clips, kit, cost, weights, *, device, iterations,
        freeze_wp, steps, seed, plant, amp, saturate_effort, cmd_lead_s):
    b = len(clips)
    horizon = model.cfg.max_horizon
    schedule = BatchedSchedule(b, ScheduleConfig(
        horizon=horizon, num_drums=model.cfg.num_drums,
        num_future=model.cfg.num_future), device=device)
    schedule.reset(torch.arange(b, device=device), clips, [kit]*b,
                   rng=np.random.default_rng(seed))
    mpc = IsaacMPC(model, schedule, MPCConfig(
        horizon=horizon, iterations=iterations, freeze_wp=freeze_wp,
        freeze_waist=True, amp_dtype=amp,
        cmd_lead_s=cmd_lead_s, waist_upright_ease_wp=scenario.waist_upright_ease_wp),
        device=device, kit_names=kit.names)
    sim = BatchedPDSim(PDSimConfig(saturate_effort=saturate_effort), device=device)
    q = torch.as_tensor(ready_joint_pos(), device=device, dtype=torch.float32)[None].expand(b,-1).clone()
    qd = torch.zeros_like(q)
    mpc.reset(torch.arange(b, device=device), q)
    bounce_gen = torch.Generator(device=device).manual_seed(seed+1000)
    pass_outputs = []
    hook = model.register_forward_hook(lambda _module, _inputs, output:
                                        pass_outputs.append(output.traj.detach().float().clone()))
    # The hook observes the actual refinement passes; it does not run extras.
    metrics = {f"pass{i}": [] for i in range(1, iterations+1)}
    metrics["committed"] = []
    hit_history, active_history, q_history, frozen_history = [], [], [q.clone()], []
    events = []
    latencies = []
    termination = None
    start = time.monotonic()
    scorer = ExecutedStrikeScorer(fk=cost.fk, device=device)
    try:
        for tick in range(steps):
            pass_outputs.clear()
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            before = time.perf_counter()
            result = mpc.plan(q, qd)
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            latencies.append(1000*(time.perf_counter()-before))
            if len(pass_outputs) != iterations:
                raise RuntimeError(f"Expected {iterations} network passes, observed {len(pass_outputs)}")
            strike = schedule.strike_context()
            batch = dict(q0=q, v0=qd, hit=result.hit_index,
                         active=result.active, strike=strike)
            repeated = torch.arange(b, device=device).repeat(iterations+1)
            combined = evaluate(cost, _index_batch(batch, repeated),
                                torch.cat(pass_outputs+[result.traj]), weights)
            for i, stage in enumerate(metrics):
                metrics[stage].append({key: value[i*b:(i+1)*b]
                                      for key, value in combined.items()})
            hit_history.append(result.hit_index.clone())
            active_history.append(result.active.clone())
            frozen_history.append(result.frozen.clone())
            previous = q
            if plant == "perfect_tracking":
                q = result.cmd.clone()
                qd = (q-previous)/mpc.dt
            else:
                minv = frozen_minv(cost.rnea, q[:, None])[:, 0]
                q, qd, _integral = sim.step(mpc.prev_cmd, result.cmd, q, qd, Minv=minv)
            finite_env = torch.isfinite(q).all(1) & torch.isfinite(qd).all(1)
            if not finite_env.all():
                termination = dict(reason="nonfinite analytic plant state", tick=tick+1,
                                   environments=(~finite_env).nonzero(as_tuple=True)[0].tolist())
                print(f"[iterations] {plant} freeze={freeze_wp} x{iterations}: {termination}", flush=True)
                break
            # Keep the pre-bounce pose for scheduled-strike scoring.
            q_history.append(q.clone())
            fired = schedule.advance()
            for env, hand in fired.nonzero().tolist():
                selected = _index_batch({"strike": strike},
                                        torch.tensor([env], device=device))["strike"]
                events.append(dict(env=env, hand=hand, tick=tick+1,
                                   drum=int(strike.target_drum[env, hand]), strike=selected))
            for hand in (0, 1):
                ids = fired[:, hand].nonzero(as_tuple=True)[0]
                if len(ids):
                    qb, vb = apply_bounce(q[ids], qd[ids], fk=cost.fk, rnea=cost.rnea,
                        hand_index=torch.full((len(ids),), hand, device=device, dtype=torch.long),
                        normal=strike.n_out[ids, hand], cfg=cfg.bounce, generator=bounce_gen)
                    q[ids], qd[ids] = qb, vb
            mpc.commit(result.cmd, q, fired, qd=qd)
            if (tick+1) % 50 == 0:
                print(f"[iterations] {plant} freeze={freeze_wp} x{iterations}: {tick+1}/{steps}", flush=True)
    finally:
        hook.remove()
    values = {stage: {key: np.concatenate([item[key] for item in rows])
                       for key in rows[0]} for stage, rows in metrics.items()}
    hit = torch.stack(hit_history)
    active = torch.stack(active_history)
    nearest = torch.where(active, hit, horizon).min(2).values.cpu().numpy().reshape(-1)
    live = active.any(2).cpu().numpy().reshape(-1)
    run_report = dict(plant=plant, iterations=iterations, freeze_wp=freeze_wp,
        steps=len(latencies), requested_steps=steps, executed_steps=len(q_history)-1,
        termination=termination, clips=[c.name for c in clips], elapsed_s=time.monotonic()-start,
        plan_latency_ms=_stats(latencies), frozen_env_ticks=int(torch.stack(frozen_history).sum()),
        plans=summarize_plans(values, nearest, live),
        strikes=score_events(events, q_history, scorer, device))
    arrays = dict(hit=hit.cpu().numpy(), active=active.cpu().numpy(),
                  q=torch.stack(q_history).cpu().numpy(),
                  frozen=torch.stack(frozen_history).cpu().numpy())
    for stage, data in values.items():
        for key, value in data.items():
            arrays[f"{stage}/{key}"] = value
    return run_report, arrays


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clip-index", type=int, default=10)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--first-hit-s", type=float, default=1.)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--iterations", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--freeze-wp", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--plants", nargs="+", choices=["pd", "perfect_tracking"],
                    default=["pd", "perfect_tracking"])
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--saturate-effort", action="store_true")
    ap.add_argument("--cmd-lead-s", choices=["saved", "off", "fit"], default="saved")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = phase2_config_from_checkpoint(ck, device=str(device), compile_model=False)
    scenario = _scenario_from_ckpt(ck)
    model = A.build_model(A.model_config_from_checkpoint(ck), device)
    model.load_state_dict(ck["model"])
    model.eval()
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[args.clip_index:args.clip_index+args.clips]
    clips = [A.pad_first_hit_to(c, args.first_hit_s)[0] for c in clips]
    if len(clips) != args.clips:
        ap.error("Not enough evaluation clips at requested index")
    cost = build_costs([model.cfg.max_horizon], cfg.cost, device=device)[model.cfg.max_horizon]
    al = _load_al(cfg, ck)
    step = int(ck["step"])
    cmd_lead_s = cfg.cmd_lead_s if args.cmd_lead_s == "saved" else (
        0.0 if args.cmd_lead_s == "off" else "fit")
    weights = _wtensors(cfg.schedule.multipliers(step, True), al.weights(), device,
        torch.float32, sched=cfg.schedule, joint_names=scenario.joint_names,
        step=step, stage_a=True, pen=al.pen)
    reports, arrays = [], {}
    for freeze_wp in args.freeze_wp:
        for plant in args.plants:
            for iterations in args.iterations:
                report, values = run(model, cfg, scenario, clips, kit, cost, weights,
                    device=device, iterations=iterations, freeze_wp=freeze_wp,
                    steps=args.steps, seed=args.seed, plant=plant,
                    amp=None if args.fp32 else torch.bfloat16,
                    saturate_effort=args.saturate_effort, cmd_lead_s=cmd_lead_s)
                reports.append(report)
                arrays.update({f"{plant}/freeze{freeze_wp}/x{iterations}/{key}": value
                               for key, value in values.items()})
                summary = report["plans"]["all_active"]
                print(f"[iterations] {plant} freeze={freeze_wp} x{iterations}: "
                      f"raw feasible={summary[f'pass{iterations}']['hard_ok']:.3f}, "
                      f"committed feasible={summary['committed']['hard_ok']:.3f}, "
                      f"strikes={report['strikes']['at_tick_in_region']}/{report['strikes']['scheduled']}",
                      flush=True)
    payload = dict(checkpoint=str(args.ckpt.resolve()), step=step,
                   design=dict(seed=args.seed, starts="ready pose, zero velocity",
                               first_hit_s=args.first_hit_s, precision="fp32" if args.fp32 else "bf16",
                               cmd_lead_s=cmd_lead_s, saturate_effort=args.saturate_effort,
                               bounce=True, native_isaac_mpc=True,
                               pd_inertia="coupled RNEA mass matrix refreshed every tick",
                               feasible="terminal/collision/joint-box/torque gate",
                               velocity_diagnostic_enabled=bool(
                                   cfg.cost.w_velocity > 0 or cfg.cost.w_velocity_peak > 0),
                               committed="after freeze, lead, slew and box; first row is executed command"),
                   runs=reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
