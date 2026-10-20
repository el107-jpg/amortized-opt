#!/usr/bin/env python3
"""Replay saved chart/graph assignments exactly and record per-drum contacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from isaac_rl.chart_cma import assigned_clip, schedule_key
from scripts.train_chart_cma import parse_args as train_args, setup_isaac, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    cfg = json.loads((args.test/"config.json").read_text())
    run = json.loads((Path(cfg["run"])/"config.json").read_text())
    checkpoint = Path(cfg["source_checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != run["checkpoint_sha256"]:
        raise ValueError("Frozen decoder checkpoint differs from the original test")
    saved = torch.load(args.test/"panel.pt", map_location="cpu", weights_only=False)
    panel = saved["panel"]
    names = ("learned_chart", "initial_chart", "graph")
    episodes, keys = [], {}
    gather = np.empty((len(names), len(panel)), dtype=int)
    for k in range(len(names)):
        for j, clip in enumerate(panel):
            physical = assigned_clip(clip, np.asarray(saved["actions"][k][j]))
            key = schedule_key(physical, j)
            if key not in keys:
                keys[key] = len(episodes)
                episodes.append(physical)
            gather[k, j] = keys[key]
    if not np.array_equal(gather, saved["gather"]):
        raise ValueError("Rebuilt saved assignments differ from the original cache mapping")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    runtime = train_args(["--out", str(args.out), "--envs", str(cfg["envs"]), "--device", args.device])
    for key in ("groups", "decode_chunk", "torque_weight", "joint_weight", "collision_weight", "collision_stride"):
        setattr(runtime, key, run["args"][key])
    runtime.near_miss_weight = run['args'].get('near_miss_weight', 0.)
    runtime.near_miss_scale_m = run['args'].get('near_miss_scale_m', .05)
    runtime.near_miss_window_s = run['args'].get('near_miss_window_s', .08)
    runtime.collision_chunk = run['args'].get('collision_chunk', runtime.decode_chunk)
    runtime.plan_diagnostics = run['args'].get('plan_diagnostics', True)
    runtime.physics_hz = cfg.get('physics_hz', 250.)
    runtime.seed = cfg["seed"]
    write_json(args.out/"config.json", dict(original_test=str(args.test.resolve()),
        replay_saved_assignments=True, seed=runtime.seed, envs=runtime.envs,
        checkpoint_sha256=run["checkpoint_sha256"], training=False, device=args.device))
    launcher = env = None
    try:
        write_json(args.out/"run.json", dict(status="running", unique_rollouts=len(episodes)))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        launcher, env, evaluator = setup_isaac(runtime, checkpoint, state, Path(cfg["test_files"][0]))
        def progress(offset, total, tick, ticks):
            write_json(args.out/"progress.json", dict(episode_offset=offset, episodes=total, tick=tick, ticks=ticks))
            print(f"[replay] episodes={offset}/{total} tick={tick}/{ticks}", flush=True)
        reports = evaluator.evaluate(episodes, progress)
        write_json(args.out/"episodes.json", reports)
        original = json.loads((args.test/"summary.json").read_text())
        methods = {}
        for k, name in enumerate(names):
            rows = [reports[i] for i in gather[k]]
            by_drum = {drum: {field: sum(r["by_drum"][drum][field] for r in rows)
                for field in ("requested", "hits", "misses")} for drum in rows[0]["by_drum"]}
            hits, requested = sum(r["hits"] for r in rows), sum(r["requested"] for r in rows)
            methods[name] = dict(hits=hits, requested=requested, hit_rate=hits/requested,
                by_drum=by_drum, original_hits=original["methods"][name]["hits"],
                matches_original_total=hits == original["methods"][name]["hits"])
        old_reports = json.loads((args.test/"episodes.json").read_text())
        changed = sum(a["hits"] != b["hits"] for a, b in zip(reports, old_reports))
        result = dict(status="complete", scenarios=len(panel), methods=methods,
            episode_totals_differing_from_original=changed, actual_isaac_contacts=True)
        write_json(args.out/"summary.json", result)
        write_json(args.out/"run.json", result)
        print("[replay-result] "+json.dumps(result), flush=True)
    except BaseException:
        traceback.print_exc()
        write_json(args.out/"run.json", dict(status="failed", traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
