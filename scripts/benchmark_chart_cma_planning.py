#!/usr/bin/env python3
"""CPU-only serial/spawned hand-search benchmark with exact decision parity."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.train_chart_cma import parse_args, load_configured_overlay
from isaac_rl.chart_cma import ChartCMA, plan_independent_population, planning_worker_ready, schedule_key
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler
from isaac_rl.min_time_assignment import MinimumTimeChart
from amortized_opt.rhythm import DrumKit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', type=int, default=1024)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    settings = parse_args(['--out', str(args.run), '--resume', str(args.run/'last.pkl'), '--dry-run'])
    base = MinimumTimeChart.load(settings.table)
    chart = load_configured_overlay(settings, base)
    optimizer, _ = ChartCMA.load(args.run/'last.pkl', chart)
    samples = optimizer.ask()[:args.cases]
    panel = ProceduralChartRhythmSampler(DrumKit.from_layout(), 8, chart).sample(args.cases, np.random.default_rng(771))
    ready = np.load(settings.ready_initial)
    stamp = time.monotonic()
    serial = plan_independent_population(optimizer, samples, panel, ready, settings.horizon)
    serial_s = time.monotonic()-stamp
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for job in [pool.submit(planning_worker_ready) for _ in range(args.workers)]:
            job.result()
        stamp = time.monotonic()
        parallel = plan_independent_population(optimizer, samples, panel, ready, settings.horizon, executor=pool)
        parallel_s = time.monotonic()-stamp
    np.testing.assert_array_equal(serial['gather'], parallel['gather'])
    assert [schedule_key(c, i) for i, c in enumerate(serial['episodes'])] == [
        schedule_key(c, i) for i, c in enumerate(parallel['episodes'])]
    report = dict(cases=args.cases, workers=args.workers, serial_s=serial_s,
        parallel_s=parallel_s, speedup=serial_s/parallel_s, exact_decision_parity=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
