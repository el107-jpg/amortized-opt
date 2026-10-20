#!/usr/bin/env python3
"""Disposable Isaac lane/memory probe; no CMA updates or training fitness."""
import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from scripts.train_chart_cma import (parse_args as train_args, sample_panel,
    setup_isaac, write_json, load_configured_overlay)
from isaac_rl.min_time_assignment import MinimumTimeChart, graph_actions
from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ticks', type=int, default=10)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--decode-chunk', type=int, default=64)
    p.add_argument('--collision-chunk', type=int)
    p.add_argument('--plan-diagnostics', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--clone-kit-from-source', action='store_true')
    p.add_argument('--decoder-checkpoint', type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    runtime = train_args(['--out', str(args.out), '--envs', str(args.envs),
        '--device', args.device, '--decode-chunk', str(args.decode_chunk)])
    runtime.seed = 20261030
    runtime.clone_kit_from_source = args.clone_kit_from_source
    runtime.collision_chunk = args.collision_chunk or args.decode_chunk
    runtime.plan_diagnostics = args.plan_diagnostics
    base = MinimumTimeChart.load(runtime.table)
    chart = load_configured_overlay(runtime, base)
    kit = A.DrumKit.from_layout()
    pool = A.load_rcc_pool(A.default_rcc_dir('train'), kit)
    panel = sample_panel(pool, args.envs, 8, np.random.default_rng(runtime.seed), chart)
    clips = [assigned_clip(c, graph_actions(hit_groups(c))[0]) for c in panel]
    source_path = args.decoder_checkpoint or Path('runs/phase2_clean_lines_fromp1_20261001/last.pt')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    seed_clip = max(A.default_rcc_dir('train').glob('*.json'), key=lambda f: f.stat().st_size)
    write_json(args.out/'run.json', dict(status='starting', training=False, envs=args.envs))
    launcher = env = None
    try:
        launcher, env, evaluator = setup_isaac(runtime, source_path, source, seed_clip)
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        first = stage.GetPrimAtPath('/World/envs/env_0/DrumKit')
        last = stage.GetPrimAtPath(f'/World/envs/env_{args.envs-1}/DrumKit')
        if not first.IsValid() or not last.IsValid():
            raise ValueError('Replicated drum kit missing from the scene')
        if sorted(str(p.GetName()) for p in first.GetChildren()) != sorted(str(p.GetName()) for p in last.GetChildren()):
            raise ValueError('Replicated kit differs from its source')
        report = evaluator.capacity_probe(clips, args.ticks)
        report.update(status='complete', training=False, device=args.device,
            decode_chunk=args.decode_chunk, clone_kit_from_source=args.clone_kit_from_source,
            collision_chunk=runtime.collision_chunk, plan_diagnostics=runtime.plan_diagnostics,
            replicated_kit_children_verified=True)
        write_json(args.out/'summary.json', report)
        write_json(args.out/'run.json', report)
        print('[capacity] '+str(report), flush=True)
    except BaseException:
        traceback.print_exc()
        write_json(args.out/'run.json', dict(status='failed', error=traceback.format_exc(), training=False))
        raise
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == '__main__':
    main()
