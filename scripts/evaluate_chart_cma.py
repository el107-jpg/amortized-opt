#!/usr/bin/env python3
"""Fresh recorded-clip Isaac comparison: learned chart, initial chart, graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from isaac_rl.chart_cma import assigned_clip, schedule_key
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart, receding_assignments, graph_actions
from isaac_rl.assignment_search import hit_groups
from scripts.train_chart_cma import (parse_args as train_args, sample_panel, setup_isaac,
    write_json, load_configured_overlay)


def load_manifest_panel(path, groups, chart, used_files):
    """Load the exact coverage-selected windows, checking recording exclusion."""
    import amortized_opt as A
    from collections import Counter, defaultdict
    manifest = json.loads(path.read_text())
    kit = A.DrumKit.from_layout()
    used_sources = set()
    for file in used_files:
        source = json.loads(file.read_text()).get("metadata", {}).get("source_midi")
        if source:
            used_sources.add(Path(source).resolve())
    panel, files, source_intervals = [], set(), defaultdict(list)
    for entry in manifest["clips"]:
        file = (path.parent/entry["file"]).resolve()
        if file in files or file in used_files:
            raise ValueError("Repeated panel file or overlap with search clips")
        metadata = json.loads(file.read_text())["metadata"]
        source = Path(metadata["source_midi"]).resolve()
        if source in used_sources:
            raise ValueError("Fixed panel recording overlaps search training/validation")
        clip = A.load_rcc(file, kit)
        rows = hit_groups(clip)
        times = [r[0].time for r in rows]
        if (len(rows) != groups or any(len(r) > 2 or not chart.choices(r) for r in rows)
                or any(len({h.drum for h in r}) != len(r) for r in rows)
                or abs(times[0]-1.) > 1e-8
                or (np.diff(times) > 1.+1e-8).any()
                or (np.diff(times) < .02-1e-8).any()):
            raise ValueError("Fixed panel has an incompatible hit-group window")
        start = int(metadata["source_group_start"])
        if any(abs(start-s) < groups for s in source_intervals[source]):
            raise ValueError("Fixed panel source windows overlap")
        source_intervals[source].append(start)
        panel.append(clip)
        files.add(file)
    summary = manifest["summary"]
    counts = Counter(h.drum_name for c in panel for h in c.hits)
    if (not panel or len(panel) != summary["scenarios"]
            or sum(counts.values()) != summary["requested"]
            or any(counts[n] != v for n, v in summary["drum_counts"].items())):
        raise ValueError("Fixed panel differs from its declared drum counts")
    return panel, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    panel_source = parser.add_mutually_exclusive_group(required=True)
    panel_source.add_argument("--test-dir", type=Path)
    panel_source.add_argument("--panel-manifest", type=Path,
        help="use every saved window exactly once, without random resampling")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--learned-overlay", type=Path, help="immutable snapshot; overrides --chart-selection")
    parser.add_argument("--plan-correction", type=Path, help="explicit learned plan-cost correction on the selected base overlay")
    parser.add_argument("--initial-overlay", type=Path, help="optional matched reference chart, instead of run initialization")
    parser.add_argument("--initial-label", default="Starting chart")
    parser.add_argument("--chart-selection", choices=("training", "validation"), default="training",
        help="chart to evaluate when no snapshot is specified (default: current training chart)")
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if min(args.scenarios, args.envs) < 1:
        parser.error("Positive scenario and environment capacities required")
    cfg = json.loads((args.run/"config.json").read_text())
    training = cfg["args"]
    import torch
    import amortized_opt as A
    base = MinimumTimeChart.load(training["table"])
    initial = (ConditionalMinimumTimeChart.load(args.initial_overlay, base) if args.initial_overlay else
        load_configured_overlay(argparse.Namespace(**training), base))
    learned_path = (args.learned_overlay if args.learned_overlay is not None else
        args.run/("last_chart.npz" if args.chart_selection == "training" else "best_chart.npz"))
    learned = ConditionalMinimumTimeChart.load(learned_path, base)
    if args.plan_correction:
        from isaac_rl.plan_preferences import PreferenceChart
        learned = PreferenceChart.load(args.plan_correction, learned, learned_path)
    ready = np.load(training["ready_initial"])
    source = Path(cfg["checkpoint"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != cfg["checkpoint_sha256"]:
        raise ValueError("Frozen decoder differs from search")
    train_dir = Path(training["train_dir"]) if training["train_dir"] else A.default_rcc_dir("train")
    val_dir = Path(training["validation_dir"]) if training["validation_dir"] else A.default_rcc_dir("eval")
    used = {p.resolve() for d in (train_dir, val_dir) for p in d.glob("*.json")}
    evaluation_groups = training['groups']
    if (args.plan_correction or cfg.get('algorithm') == 'Direct transition-time plan preferences') and args.panel_manifest:
        # The preference scorer learns four/six-group choices; a fixed music
        # test can contain longer complete clips and use receding inference.
        evaluation_groups = json.loads(args.panel_manifest.read_text())['summary']['groups_per_window']
    if args.panel_manifest is not None:
        panel, test_files = load_manifest_panel(args.panel_manifest, evaluation_groups, initial, used)
        args.scenarios = len(panel)
    else:
        test_files = {p.resolve() for p in args.test_dir.glob("*.json")}
        if test_files & used:
            raise ValueError("Final test source clips overlap search training/validation")
        clips = A.load_rcc_pool(args.test_dir, A.DrumKit.from_layout())
        if not clips:
            raise ValueError("No recorded test rhythms")
        rng = np.random.default_rng(args.seed)
        panel = sample_panel(clips, args.scenarios, training["groups"], rng, initial)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    names = ("learned_chart", "initial_chart", "graph")
    gather = np.empty((3, len(panel)), int)
    episodes, keys, choices = [], {}, []
    for k, chart in enumerate((learned, initial, None)):
        selected = []
        for j, clip in enumerate(panel):
            actions = (graph_actions(hit_groups(clip))[0] if chart is None else
                receding_assignments(clip, chart, training["horizon"], ready)[0])
            selected.append(actions.tolist())
            physical = assigned_clip(clip, actions)
            key = schedule_key(physical, j)
            if key not in keys:
                keys[key] = len(episodes); episodes.append(physical)
            gather[k, j] = keys[key]
        choices.append(selected)
    # Isaac sizes beat buffers from the seed rhythm. A fixed panel can have
    # different request counts despite equal group counts; choose the largest.
    kit = A.DrumKit.from_layout()
    seed_clip = max(test_files, key=lambda file: len(A.load_rcc(file, kit).hits))
    write_json(args.out/"config.json", dict(run=str(args.run.resolve()), test_files=sorted(map(str, test_files)),
        scenarios=args.scenarios, seed=args.seed, envs=args.envs, source_checkpoint=str(source),
        horizon=training["horizon"], groups=evaluation_groups, training_groups=training['groups'], training=False,
        learned_overlay=str(learned_path.resolve()),
        plan_correction=str(args.plan_correction.resolve()) if args.plan_correction else None,
        plan_correction_sha256=hashlib.sha256(args.plan_correction.read_bytes()).hexdigest() if args.plan_correction else None,
        initial_overlay=str(args.initial_overlay.resolve()) if args.initial_overlay else training['overlay'],
        initial_label=args.initial_label,
        chart_selection=learned.metadata.get("parameter_search", {}).get("selection", args.chart_selection),
        panel_manifest=str(args.panel_manifest.resolve()) if args.panel_manifest else None,
        panel_manifest_sha256=hashlib.sha256(args.panel_manifest.read_bytes()).hexdigest() if args.panel_manifest else None,
        independent_source_recording_split_verified=args.panel_manifest is not None,
        isaac_seed_clip=str(seed_clip),
        independent_source_clip_split_verified=True,
        physics_hz=training.get('physics_hz', 250.), control_hz=50.))
    torch.save(dict(panel=panel, gather=gather, actions=choices), args.out/"panel.pt")
    runtime = train_args(["--out", str(args.out), "--envs", str(args.envs), "--device", args.device])
    for key in ("groups", "decode_chunk", "torque_weight", "joint_weight", "collision_weight", "collision_stride"):
        setattr(runtime, key, training[key])
    runtime.groups = evaluation_groups
    runtime.near_miss_weight = training.get('near_miss_weight', 0.)
    runtime.near_miss_scale_m = training.get('near_miss_scale_m', .05)
    runtime.near_miss_window_s = training.get('near_miss_window_s', .08)
    runtime.collision_chunk = training.get('collision_chunk', runtime.decode_chunk)
    runtime.plan_diagnostics = training.get('plan_diagnostics', True)
    runtime.physics_hz = training.get('physics_hz', 250.)
    runtime.seed = args.seed
    launcher = env = None
    try:
        write_json(args.out/"run.json", dict(status="running", unique_rollouts=len(episodes)))
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        launcher, env, evaluator = setup_isaac(runtime, source, checkpoint, seed_clip)
        def progress(offset, total, tick, ticks):
            write_json(args.out/"progress.json", dict(episode_offset=offset, episodes=total, tick=tick, ticks=ticks))
            print(f"[test] episodes={offset}/{total} tick={tick}/{ticks}", flush=True)
        reports = evaluator.evaluate(episodes, progress)
        write_json(args.out/"episodes.json", reports)
        metrics = {}
        for k, name in enumerate(names):
            rows = [reports[i] for i in gather[k]]
            n = sum(r["requested"] for r in rows); hits = sum(r["hits"] for r in rows)
            metrics[name] = dict(hits=hits, requested=n, hit_rate=hits/n,
                mean_episode_near_miss_bonus=float(np.mean([r.get('near_miss_bonus', 0.) for r in rows])),
                mean_episode_hit_quality=float(np.mean([r.get('hit_quality', r['hit_rate']) for r in rows])),
                by_drum={drum: dict(
                    requested=sum(r["by_drum"][drum]["requested"] for r in rows),
                    hits=sum(r["by_drum"][drum]["hits"] for r in rows),
                    misses=sum(r["by_drum"][drum]["misses"] for r in rows))
                    for drum in rows[0]["by_drum"]},
                mean_episode_hit_rate=float(np.mean([r["hit_rate"] for r in rows])),
                mean_reward=float(np.mean([r["reward"] for r in rows])),
                torque_demand_excess_average_Nm=float(np.mean([r["torque_demand_excess_average_Nm"] for r in rows])),
                measured_geometry_penetration_average_m=float(np.mean([r["measured_geometry_penetration_average_m"] for r in rows])))
        comparisons = {}
        ids_by_source = {}
        source_by_clip = {file.stem: json.loads(file.read_text())["metadata"]["source_midi"]
            for file in test_files} if args.panel_manifest else {}
        for j, c in enumerate(panel):
            key = source_by_clip[c.name] if args.panel_manifest else c.name.split("@cma_group")[0]
            ids_by_source.setdefault(key, []).append(j)
        clusters = list(ids_by_source.values())
        draw_rng = np.random.default_rng(args.seed+1)
        for a, b in ((0, 1), (0, 2)):
            samples = []
            for _ in range(2000):
                ids = np.concatenate([clusters[i] for i in draw_rng.integers(0, len(clusters), len(clusters))])
                delta = sum(reports[gather[a, i]]["hits"]-reports[gather[b, i]]["hits"] for i in ids)
                total = sum(reports[gather[a, i]]["requested"] for i in ids)
                samples.append(delta/total)
            comparisons[names[a]+"_minus_"+names[b]] = dict(
                hit_rate_gain=metrics[names[a]]["hit_rate"]-metrics[names[b]]["hit_rate"],
                source_cluster_bootstrap_95ci=np.quantile(samples, [.025, .975]).tolist(),
                changed_group_fraction=float(np.mean(np.asarray(choices[a]) != np.asarray(choices[b]))))
        report = dict(status="complete", scenarios=len(panel), unique_source_clips=len(clusters),
            methods=metrics, comparisons=comparisons, actual_isaac_contacts=True,
            bootstrap_cluster_unit="source_recording" if args.panel_manifest else "source_clip",
            source_clip_holdout=True, nominal_ready_starts=True)
        write_json(args.out/"summary.json", report); write_json(args.out/"run.json", report)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(["CMA chart", args.initial_label, "Graph"], [100*metrics[n]["hit_rate"] for n in names],
                      color=["#3288bd", "#66c2a5", "#888888"])
        ax.bar_label(bars, fmt="%.1f%%", padding=4); ax.set_ylim(0, 100)
        ax.set_ylabel("Isaac credited hit rate (%)"); ax.set_title(f"Fresh recorded test clips · {len(panel)} windows")
        fig.tight_layout(); fig.savefig(args.out/"comparison.png", dpi=180); plt.close(fig)
        print("[test-result] "+json.dumps(report), flush=True)
    except BaseException as exc:
        write_json(args.out/"run.json", dict(status="failed", error=str(exc)))
        raise
    finally:
        if env is not None: env.close()
        if launcher is not None: launcher.app.close()


if __name__ == "__main__":
    main()
