#!/usr/bin/env python3
"""Paired graph versus 4/6-group chart assignment on full decoder splines."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.ik import arm_joint_indices
from amortized_opt.limits import available_torque_torch
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.assignment_search import hit_groups
from isaac_rl.min_time_assignment import MinimumTimeChart, graph_actions, receding_assignments, window_scores
from isaac_rl.random_rhythms import RandomRhythmConfig, RandomRhythmSampler
from isaac_rl.rnea_cache import load_cached
from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
from isaac_rl.trajectory_qp import cost_for_horizon
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance, profile
from scripts.fill_transition_table_gaps import context

METHODS = ("graph", "chart_4_groups", "chart_6_groups")


def write_json(path, value):
    temp = Path(str(path)+".tmp")
    temp.write_text(json.dumps(value, indent=2, default=float)+"\n")
    temp.replace(path)


def setup(checkpoint, device, physics="knot_fd"):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = A.CostConfig(**ck["train_cfg"]["cost"])
    cfg.physics_mode = physics
    cfg.n_sub = 4
    cfg.w_velocity = max(1., cfg.w_velocity)
    cfg.w_joint_box = max(1., cfg.w_joint_box)
    cost = A.FusedCost(cfg, max_horizon=100, device=device, rnea=load_cached(device=device))
    model = A.build_model(A.model_config_from_checkpoint(ck), device)
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit, horizon=100,
        waist_upright_ease_wp=(ck.get("scenario_cfg") or {}).get("waist_upright_ease_wp"))
    return cost, kit, generator


@torch.no_grad()
def ready_arrivals(checkpoint, device, chart, out):
    """Actual ready-state arrivals, using the chart's gate and both yaw hints."""
    cost, kit, generator = setup(checkpoint, device, "spline")
    cost.calibration_active_hand_only = True
    allowed = chart.allowed.diagonal(axis1=1, axis2=2)
    hands, drums = np.where(allowed)
    hands = np.tile(hands, 2)
    drums = np.tile(drums, 2)
    B = len(hands)
    q = torch.as_tensor(ready_joint_pos(), device=device, dtype=torch.float32)[None].expand(B, -1)
    v = torch.zeros_like(q)
    best = np.full((2, kit.num_drums), np.inf)
    allowance = SmallViolationAllowance(collision_depth_m=.005,
        terminal_distance_m=.01, joint_excess_rad=np.deg2rad(5), torque_average_limit_Nm=5.)
    begin = time.monotonic()
    for N in range(1, 101):
        for h in (0, 1):
            ids = np.flatnonzero(hands == h)
            targets, ctx, lengths = context(kit, h, [-1]*len(ids), drums[ids], N, device)
            # Keep the two candidate batches separate, matching the original
            # chart's fixed-yaw and analytic-yaw input channels.
            half = len(ids)//2
            plans = []
            for yaw_hint in (0, 1):
                plan = generator.predict(q[ids[:half]], v[ids[:half]],
                    torch.tensor(targets[:half, None], device=device),
                    q.new_full((half, 1), N*.02), q.new_full((half, 1, 2), 2.),
                    torch.ones(half, 1, device=device, dtype=torch.bool), 0,
                    return_plan=True, waist_yaw=q[ids[:half], 0] if yaw_hint == 0 else None)
                path = plan["traj"][:, :N].clone()
                idle = arm_joint_indices(("left", "right")[1-h])
                path[:, :, idle] = q[ids[:half]][:, None, idle]
                plans.append(path)
            traj = torch.cat(plans)
            dense = cost_for_horizon(cost, N)
            dense.cfg = copy.copy(cost.cfg)
            dense.cfg.n_sub = 32
            values = profile(dense, q[ids], v[ids], traj, ctx, lengths, allowance=allowance)
            for j in np.flatnonzero(values["average_success"]):
                best[h, drums[ids[j]]] = min(best[h, drums[ids[j]]], N*.02)
        if N % 20 == 0:
            write_json(out/"progress.json", dict(stage="ready_calibration", durations=N,
                supported=int(np.isfinite(best).sum()), elapsed_s=time.monotonic()-begin))
            print(f"Ready calibration {N}/100 durations", flush=True)
    np.save(out/"ready_initial_minimum_s.npy", best)
    return best


def pack_physical(clips, device):
    groups = [hit_groups(c) for c in clips]
    G = max(map(len, groups))
    B = len(clips)
    drums = np.full((B, G, 2), -1, np.int64)
    speeds = np.zeros((B, G, 2), np.float32)
    times = np.zeros((B, G), np.float32)
    valid = np.zeros((B, G), bool)
    for b, gs in enumerate(groups):
        times[b] = gs[-1][0].time
        for g, rows in enumerate(gs):
            valid[b, g] = True
            times[b, g] = rows[0].time
            for j, hit in enumerate(rows):
                drums[b, g, j] = hit.drum
                speeds[b, g, j] = 1+2*hit.velocity
    return {k: torch.as_tensor(v, device=device) for k, v in
            dict(drums=drums, speeds=speeds, times=times, valid=valid).items()}


class IntegralEnvironment(DecoderScheduleEnvironment):
    def _prefix_metrics(self, spline, v, ctx, steps):
        metrics = super()._prefix_metrics(spline, v, ctx, steps)
        sample = spline.subsample(self.cost.cfg.n_sub)
        N = int(steps.max())*self.cost.cfg.n_sub+1
        q, qd, qdd = (x[:, :N] for x in (sample.q, sample.qd, sample.qdd))
        B, T, D = q.shape
        tau = self.cost.rnea.rnea(q.reshape(-1, D), qd.reshape(-1, D), qdd.reshape(-1, D)).reshape(B, T, D)
        available = available_torque_torch(qd, self.cost.tau_hard.view(1, 1, D),
            self.cost.v_hard.view(1, 1, D), eta=1.,
            saturation_scale=self.cost.cfg.torque_saturation_scale, derate=self.cost.cfg.torque_derate)
        stop = steps*self.cost.cfg.n_sub
        index = torch.arange(T, device=q.device)[None]
        weight = (index <= stop[:, None]).to(q.dtype)
        weight[:, 0] *= .5
        weight[torch.arange(B, device=q.device), stop] *= .5
        weight *= spline.dt/self.cost.cfg.n_sub
        integral = ((tau.abs()-available).clamp_min(0).sum(-1)*weight).sum(-1)
        metrics["dense_torque_integral_Nms"] = integral
        metrics["dense_torque_average_Nm"] = integral/(steps*spline.dt)
        return metrics


@torch.no_grad()
def worker(args):
    torch.set_num_threads(2)
    problem = torch.load(args.out/"problems.pt", weights_only=False)
    owners = np.arange(len(problem["clips"]))[args.rank::len(args.devices)]
    clips = [problem["clips"][i] for i in owners]
    cost, kit, generator = setup(problem["checkpoint"], args.device)
    table = pack_physical(clips, args.device)
    G = table["valid"].shape[1]
    candidates = torch.tensor(problem["actions"][owners, :, :G], device=args.device)
    draws = torch.tensor(problem["draws"][owners, :, :G], device=args.device)
    q = torch.tensor(ready_joint_pos(), device=args.device, dtype=torch.float32)[None].expand(len(clips), -1)
    env = IntegralEnvironment(generator, chunk=args.chunk)
    started = time.monotonic()
    results = env.score(table, q, torch.zeros_like(q), candidates, draws)
    torch.save(dict(owners=owners, results={k: v.cpu() for k, v in results.items()}),
               args.out/f"w{args.rank}_results.pt")
    write_json(args.out/f"w{args.rank}_done.json", dict(status="complete", windows=len(clips),
        elapsed_s=time.monotonic()-started))


def sample_clips(count, rng, kit, groups_per_clip):
    recorded = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    sampler = RandomRhythmSampler(kit, RandomRhythmConfig(horizon_s=4., max_hits=32,
                                                         tempo_min=100, tempo_max=180))
    result, sources = [], []
    rejected = 0
    attempts = 0
    while len(result) < count:
        natural = len(result) < count//2
        clip = recorded[int(rng.integers(len(recorded)))] if natural else sampler.sample(rng, 1)[0]
        attempts += 1
        if attempts > count*100:
            raise RuntimeError("Could not sample enough eligible complete rhythms")
        from itertools import groupby
        groups = [list(g) for _,g in groupby(sorted(clip.hits, key=lambda h: (h.time, h.drum)),
                                             key=lambda h: h.time)]
        if len(groups) < groups_per_clip:
            continue
        start = int(rng.integers(len(groups)-groups_per_clip+1)) if natural else 0
        selected = groups[start:start+groups_per_clip]
        if any(len(g) > 2 for g in selected):
            rejected += 1
            continue
        # Reject unsupported duplicate-floor pairs, never remove one strike.
        if any(len(g) == 2 and all(h.drum_name == "floor_tom" for h in g) for g in selected):
            rejected += 1
            continue
        gaps = np.diff([g[0].time for g in selected])
        if (gaps > 2.).any() or (gaps < .02-1e-7).any():
            rejected += 1
            continue
        offset = 1.-selected[0][0].time
        hits = [replace(h, time=round(h.time+offset, 8), raw_time=h.raw_time+offset)
                for g in selected for h in g]
        result.append(replace(clip, name=f"{clip.name}@group{start}", hits=hits, duration=hits[-1].time))
        sources.append("recorded_eval" if natural else "procedural")
    return result, sources, rejected


def summarize(args, problem, parts):
    owners = np.concatenate([p["owners"] for p in parts])
    order = np.argsort(owners)
    result = {k: torch.cat([p["results"][k] for p in parts])[order]
              for k in parts[0]["results"]}
    report = dict(status="complete", windows=len(problem["clips"]),
        groups_per_clip=args.groups, bounce_draws=args.draws, methods={}, comparisons={},
        table=str(args.table.resolve()), checkpoint=problem["checkpoint"],
        environment="Actual frozen decoder spline prefixes, perfect tracking, analytic rebound; no Isaac",
        assignment_state="Last scheduled drum/time per hand; ready-state arrivals calibrated from measured q/qd",
        physics="Full-body 5 ms collisions/continuous-velocity checks; knot-FD runtime torque plus dense spline torque integrals",
        unsupported_table_edges="Penalized as unknown; not treated as impossible or excluded",
        assignment_horizon="Next four/six complete hit groups; commit first group and replan",
        ready_supported=int(np.isfinite(problem["initial"]).sum()),
        graph_fallbacks=problem["fallbacks"], rejected_windows=problem["rejected"])
    if "conditional_chart" in problem:
        report["conditional_chart"] = problem["conditional_chart"]
    requested = result["requested"].sum(-1)
    durations = np.array([c.duration for c in problem["clips"]])[:, None]
    for j, name in enumerate(METHODS):
        req = int(requested[:, j].sum())
        hit = result["hits"][:, j].sum(-1)
        report["methods"][name] = dict(hits=int(hit.sum()), requested=req,
            hit_rate=float(hit.sum()/req),
            strict_valid_hit_rate=float(result["strict_valid_hits"][:, j].sum()/req),
            score_mean=float(result["score"][:, j].sum(-1).mean()),
            collision_peak_mm_mean=float(result["collision_m"][:, j].amax(-1).mean()*1000),
            knot_torque_peak_ratio_mean=float(result["torque_peak_ratio"][:, j].amax(-1).mean()),
            dense_torque_average_Nm_mean=float((result["dense_torque_integral_Nms"][:, j].sum(-1).numpy()/durations).mean()),
            choices_changed_vs_graph=float((problem["actions"][:, j] != problem["actions"][:, 0]).mean()))
    rng = np.random.default_rng(args.seed+99)
    bootstrap = rng.integers(0, args.windows, (2000, args.windows))
    hits = result["hits"].sum(-1).float().mean(-1).numpy()
    req = requested.float().mean(-1).numpy()
    score = result["score"].sum(-1).float().mean(-1).numpy()
    for a, b in ((1, 0), (2, 0), (2, 1)):
        name = METHODS[a]+"_minus_"+METHODS[b]
        differences = (hits[:, a]-hits[:, b])[bootstrap].sum(-1)/req[:, a][bootstrap].sum(-1)
        report["comparisons"][name] = dict(hit_rate_gain=float((hits[:, a]-hits[:, b]).sum()/req[:, a].sum()),
            paired_episode_bootstrap_95ci=np.quantile(differences, [.025, .975]).tolist(),
            score_gain_mean=float((score[:, a]-score[:, b]).mean()),
            episodes_better_score=int((score[:, a] > score[:, b]+1e-5).sum()),
            episodes_worse_score=int((score[:, a] < score[:, b]-1e-5).sum()))
    for source in sorted(set(problem["sources"])):
        mask = np.array(problem["sources"]) == source
        report[source] = {name:dict(hit_rate=float(hits[mask, j].sum()/req[mask, j].sum()),
            score_mean=float(score[mask, j].mean())) for j, name in enumerate(METHODS)}
    report["search_latency_ms"] = problem["search_latency_ms"]
    torch.save(result, args.out/"rollouts.pt")
    write_json(args.out/"summary.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    labels = ["Graph", "Chart: next 4", "Chart: next 6"]
    for axis, key, title in zip(ax, ("hit_rate", "strict_valid_hit_rate", "score_mean"),
                              ("Actual hits (%)", "Strict valid hits (%)", "Physical rollout score")):
        values = [report["methods"][n][key]*(100 if "rate" in key else 1) for n in METHODS]
        bars = axis.bar(labels, values, color=["#888888", "#3288bd", "#66c2a5"])
        axis.bar_label(bars, fmt="%.1f", padding=3)
        axis.set_ylim((0, max(values)*1.18) if min(values) >= 0 else (min(values)*1.18, 0))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Matched decoder rollouts: {args.windows} episodes × {args.draws} rebound draws")
    fig.tight_layout()
    fig.savefig(args.out/"comparison.png", dpi=180)
    plt.close(fig)
    write_json(args.out/"run.json", report)
    print(json.dumps(report), flush=True)


@torch.no_grad()
def parent(args):
    args.out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    meta = json.loads((args.table.parent/"config.json").read_text())
    checkpoint = meta["checkpoint"]
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != meta["checkpoint_sha256"]:
        raise ValueError("Chart's frozen checkpoint has changed")
    write_json(args.out/"config.json", dict(table=str(args.table.resolve()),
        checkpoint=checkpoint, windows=args.windows, groups=args.groups, draws=args.draws,
        seed=args.seed, devices=args.devices, training=False, chart_constraints=meta))
    write_json(args.out/"run.json", dict(status="running", stage="ready_calibration"))
    torch.set_num_threads(2)
    chart = MinimumTimeChart.load(args.table)
    if args.conditional_chart is not None:
        from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
        chart = ConditionalMinimumTimeChart.load(args.conditional_chart, chart)
        config_path = args.out/"config.json"
        config = json.loads(config_path.read_text())
        config["conditional_chart"] = dict(path=str(args.conditional_chart.resolve()),
                                          **chart.description())
        write_json(config_path, config)
    if args.ready_initial is None:
        initial = ready_arrivals(checkpoint, args.devices[0], chart, args.out)
    else:
        previous = json.loads((args.ready_initial.parent/"config.json").read_text())
        if previous["checkpoint"] != checkpoint or previous["table"] != str(args.table.resolve()):
            raise ValueError("Ready calibration must use the same chart and checkpoint")
        initial = np.load(args.ready_initial)
        if initial.shape != (2, len(chart.names)) or np.isnan(initial).any():
            raise ValueError("Invalid ready-arrival calibration")
        np.save(args.out/"ready_initial_minimum_s.npy", initial)
    kit = A.DrumKit.from_layout()
    rng = np.random.default_rng(args.seed)
    clips, sources, rejected = sample_clips(args.windows, rng, kit, args.groups)
    actions, details = [], []
    times = {n: [] for n in METHODS[1:]}
    fallbacks = 0
    for c in clips:
        groups = hit_groups(c)
        graph, fallback = graph_actions(groups)
        fallbacks += fallback
        row = [graph]
        for horizon, name in zip((4, 6), METHODS[1:]):
            started = time.perf_counter()
            selected, diag = receding_assignments(c, chart, horizon, initial)
            times[name].append((time.perf_counter()-started)*1000/len(groups))
            row.append(selected)
        actions.append(np.stack(row))
        proxy = {name: {k: float(v[0]) for k, v in window_scores(groups, a[None], chart,
                    initial_drums=(-1, -1), last_times=(0., 0.), initial_minimum_s=initial).items()}
                 for name, a in zip(METHODS, row)}
        details.append(dict(name=c.name, hits=[dict(time=h.time, drum=h.drum_name) for h in c.hits],
                            actions={name: a.tolist() for name, a in zip(METHODS, row)}, proxy=proxy))
    problem = dict(clips=clips, sources=sources, rejected=rejected, checkpoint=checkpoint,
        initial=initial, actions=np.stack(actions), fallbacks=fallbacks,
        draws=rng.uniform(.2, .8, (len(clips), args.draws, args.groups, 2)).astype(np.float32),
        search_latency_ms={n: dict(per_group_mean=float(np.mean(t)), per_group_p95=float(np.quantile(t,.95)))
                           for n,t in times.items()})
    if args.conditional_chart is not None:
        problem["conditional_chart"] = chart.description()
    torch.save(problem, args.out/"problems.pt")
    write_json(args.out/"assignments.json", details)
    children = []
    for rank, device in enumerate(args.devices):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--rank", str(rank),
               "--device", device, "--out", str(args.out), "--table", str(args.table),
               "--devices", *args.devices, "--chunk", str(args.chunk)]
        with (args.out/f"w{rank}.log").open("w") as log:
            children.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")))
    write_json(args.out/"run.json", dict(status="running", stage="physical_rollouts",
        worker_pids=[p.pid for p in children], windows=args.windows))
    try:
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("Rollout worker failed; inspect w*.log")
            write_json(args.out/"progress.json", dict(stage="physical_rollouts",
                completed_workers=len(list(args.out.glob("w*_done.json"))),
                workers=len(children), elapsed_s=time.monotonic()-begin))
            time.sleep(3)
        if any(p.returncode for p in children):
            raise RuntimeError("Rollout worker failed")
        parts = [torch.load(args.out/f"w{i}_results.pt", weights_only=False) for i in range(len(children))]
        summarize(args, problem, parts)
    except BaseException:
        for p in children:
            if p.poll() is None:
                p.terminate()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=ROOT/"runs/transition_decoder_torque_integral_20261005_2240/collision_5mm/transition_table.npz")
    parser.add_argument("--conditional-chart", type=Path,
                        help="optional parabolic planning overlay; base table and ready calibration remain unchanged")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--ready-initial", type=Path,
                        help="reuse measured ready arrivals from a matching completed probe")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.windows < len(args.devices) or args.groups < 6 or min(args.draws, args.chunk) < 1:
        parser.error("Need positive draws/chunk, at least six groups, and one episode per worker")
    args.out = args.out.resolve()
    try:
        worker(args) if args.worker else parent(args)
    except BaseException as exc:
        if args.out.exists() and not args.worker:
            write_json(args.out/"run.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
