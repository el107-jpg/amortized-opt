#!/usr/bin/env python3
"""Draw real decoder splines for four failure types from a transition audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import amortized_opt as A
from amortized_opt.constants import STICK_TIP_SIGMA
from amortized_opt.ik import arm_joint_indices, tip_position
from amortized_opt.rhythm import head_plane_basis, strike_region
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.transition_calibration import evaluate_trials


def ring(center, normal, radius):
    a, b = head_plane_basis(normal)
    theta = np.linspace(0, 2*np.pi, 100)
    return center+radius*(np.cos(theta)[:, None]*a+np.sin(theta)[:, None]*b)


def draw_disk(ax, center, normal, radius, color, alpha):
    points = ring(center, normal, radius)
    ax.add_collection3d(Poly3DCollection([points], facecolor=color, edgecolor=color,
                                        linewidth=.8, alpha=alpha))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run = args.audit.resolve()
    out = run/'failure_examples'; out.mkdir(exist_ok=True)
    config = json.loads((run/'config.json').read_text())
    with np.load(run/'deadline_audit.npz') as saved:
        keys = saved['keys'].copy()
        col = int(np.flatnonzero(np.isclose(saved['deadlines_s'], 1.))[0])
        values = {name: saved[name][:, col].copy() for name in saved.files
                  if name not in ('keys', 'valid_start', 'deadlines_s')}
        never_passed = ~saved['success'].any(-1)
    with (run/'full_range_minima.csv').open() as f:
        bounded = np.array([r['departure_posture_valid'] == 'True' for r in csv.DictReader(f)])
    checks = dict(region=values['terminal_region_dist'] <= 1e-6,
        collision=values['collision_min_sdf'] >= -1e-6,
        torque=values['torque_excess_max'] <= 1e-6,
        joints=values['joint_box_excess_max'] <= 1e-6,
        velocity=values['velocity_excess_max'] <= 1e-6,
        direction=values['contact_dir_cos'] >= .5,
        elevation=(values['stick_elev_deg'] >= 5)&(values['stick_elev_deg'] <= 45))
    magnitudes = dict(region=values['terminal_region_dist']*1000,
        collision=np.maximum(-values['collision_min_sdf'], 0)*1000,
        torque=values['torque_excess_max'], joints=np.rad2deg(values['joint_box_excess_max']))
    kit = A.DrumKit.from_layout()
    pads = [i for i, name in enumerate(kit.names) if name in ('snare', 'tom1', 'tom2', 'floor_tom')]
    selected = []
    for kind in magnitudes:
        eligible = bounded & never_passed & ~checks[kind] & np.isin(keys[:, 2], pads)
        other_failures = sum(~ok for name, ok in checks.items() if name != kind)
        eligible &= other_failures == other_failures[eligible].min()
        candidates = np.flatnonzero(eligible)
        target = np.median(magnitudes[kind][eligible])
        index = int(candidates[np.argmin(np.abs(magnitudes[kind][candidates]-target))])
        selected.append((kind, index))
    print(json.dumps(dict(stage='selected', keys=[keys[i].tolist() for _, i in selected])), flush=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(checkpoint), device)
        cfg = A.CostConfig(**checkpoint['train_cfg']['cost'])
        cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cfg.n_sub = config['collocation_substeps']
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(checkpoint['model'])
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(checkpoint.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    fig = plt.figure(figsize=(16, 13), facecolor='white')
    titles = dict(region='Misses the head', collision='Reaches head; collides',
                  torque='Reaches head; exceeds torque', joints='Exceeds joint bounds')
    reports, paths = [], {}
    for panel, (kind, index) in enumerate(selected):
        h, src, dst, y, s = map(int, keys[index])
        hand = ('left', 'right')[h]; yaw = config['yaw_centers'][y]
        with np.load(Path(config['source_calibration'])/'samples'/f'h{h}_d{src}_y{y}.npz') as raw:
            q = torch.as_tensor(raw['q'][s:s+1], device=device)
            v = torch.as_tensor(raw['v'][s:s+1], device=device)
        result = evaluate_trials(generator, q, v, h, [src], [dst], [yaw], np.array([[1.]]),
            hit_depth_m=config['hit_depth_mm']/1000,
            require_hit_speed=config['impact_speed_required'],
            allow_early_target_top_entry=config.get('allow_early_target_top_entry', False))
        for name, actual in result.items():
            # Single-example GPU kernels can differ slightly from the audit's
            # 64-example batches; require close values and identical hit gates.
            np.testing.assert_allclose(actual[0, 0], values[name][index], rtol=1e-3, atol=1e-4)
        assert not result['success'][0, 0]
        assert result['geometric'][0, 0] == values['geometric'][index]
        drums = torch.full((1, 1, 2), -1, device=device, dtype=torch.long); drums[0, 0, h] = dst
        plan = generator.predict(q, v, drums, q.new_ones(1, 1), q.new_full((1, 1, 2), 2.),
                                 torch.ones(1, 1, device=device, dtype=torch.bool), 0,
                                 return_plan=True, waist_yaw=[yaw])
        traj = plan['traj'].clone()
        idle = arm_joint_indices(('left', 'right')[1-h]); traj[:, :, idle] = q[:, None, idle]
        lengths = torch.full_like(plan['lengths'], 50)
        samples = cost.spline.build(q, traj, v, lengths).subsample(8).q[0, :401]
        tip = tip_position(cost.fk, samples, hand, frame='env').cpu().numpy()
        butt = tip_position(cost.fk, samples, hand, sigma=-STICK_TIP_SIGMA,
                            frame='env').cpu().numpy()
        ax = fig.add_subplot(2, 2, panel+1, projection='3d')
        for drum, (center, normal, radius) in enumerate(zip(kit.positions, kit.axes, kit.radii)):
            color = '#95a0ad' if drum != dst else '#167d53'
            draw_disk(ax, center, normal, radius, color, .10 if drum != dst else .22)
            bottom = ring(center-normal*kit.heights[drum], normal, radius)
            ax.plot(*bottom.T, color=color, alpha=.18, lw=.6)
            if drum in (src, dst):
                ax.text(*(center+normal*.025), kit.names[drum].replace('_', ' '),
                        fontsize=10, color=color, ha='center')
        aim, disk, radius, _ = strike_region(kit, dst)
        outer = disk+kit.axes[dst]*.01
        lower = outer-kit.axes[dst]*(config['hit_depth_mm']/1000)
        top_ring, low_ring = ring(outer, kit.axes[dst], radius), ring(lower, kit.axes[dst], radius)
        draw_disk(ax, outer, kit.axes[dst], radius, '#23a770', .28)
        ax.plot(*low_ring.T, color='#23a770', alpha=.6, lw=1)
        for j in range(0, len(top_ring), 16):
            ax.plot(*np.stack((top_ring[j], low_ring[j])).T, color='#23a770', alpha=.45, lw=.7)
        color = '#2868cb' if h == 0 else '#d2691e'
        ax.plot(*tip.T, color=color, lw=2.5)
        ax.scatter(*tip[0], color=color, s=55, marker='o', depthshade=False)
        ax.scatter(*tip[-1], color='#c52d3b', s=90, marker='x', linewidth=3, depthshade=False)
        ax.scatter(*aim, color='#167d53', s=130, marker='*', depthshade=False)
        for j in (0, 100, 200, 300, 400):
            ax.plot(*np.stack((butt[j], tip[j])).T, color=color, lw=1.2,
                    alpha=.24 if j != 400 else .85)
        metric = dict(region=f"Head-region miss: {magnitudes['region'][index]:.0f} mm",
                      collision=f"Collision penetration: {magnitudes['collision'][index]:.0f} mm",
                      torque=f"Torque excess: {magnitudes['torque'][index]:.1f} N m",
                      joints=f"Joint-bound excess: {magnitudes['joints'][index]:.1f} deg")[kind]
        failures = [name for name, passed in checks.items() if not passed[index]]
        ax.set_title(f"{titles[kind]}\n{hand.title()}: {kit.names[src].replace('_', ' ')} → "
                     f"{kit.names[dst].replace('_', ' ')}", fontsize=14, loc='left', pad=14)
        ax.text2D(.02, .98, f"{metric}\nFails: {', '.join(failures)}\nYaw {yaw:+.1f} rad · start {s}",
                  transform=ax.transAxes, fontsize=11, va='top',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=5))
        scene = np.concatenate((kit.positions, tip, butt))
        low, high = scene.min(0)-.06, scene.max(0)+.09
        ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_zlim(low[2], high[2])
        ax.set_box_aspect(high-low); ax.view_init(elev=25, azim=135)
        ax.set_xlabel('x (m)', labelpad=6); ax.set_ylabel('y (m)', labelpad=6)
        ax.set_zlabel('Height above floor (m)', labelpad=10)
        ax.tick_params(labelsize=8)
        paths[f'case{panel}_tip_env'] = tip; paths[f'case{panel}_butt_env'] = butt
        reports.append(dict(kind=kind, key=keys[index].tolist(), hand=hand,
            source=kit.names[src], destination=kit.names[dst], deadline_s=1., yaw_rad=yaw,
            start_index=s, failures=failures, start_within_posture_bounds=True,
            no_accepted_duration_in_full_sweep=True,
            diagnostics={name: float(result[name][0, 0]) for name in result},
            tip_start_env=tip[0].tolist(), tip_end_env=tip[-1].tolist()))
    fig.suptitle('Four failed decoder plans at a 1-second deadline', fontsize=22, y=.985)
    fig.text(.5, .953, 'Posture-valid starts; none succeeds at any tested duration (20 ms–2 s). '
             '70 mm depth; no impact-speed requirement.', ha='center', fontsize=12)
    legend = [Line2D([0], [0], color='#2868cb', lw=2.5, label='Left stick-tip path'),
              Line2D([0], [0], color='#d2691e', lw=2.5, label='Right stick-tip path'),
              Line2D([0], [0], color='#555', marker='o', ls='', label='Start'),
              Line2D([0], [0], color='#c52d3b', marker='x', ls='', label='End at deadline'),
              Line2D([0], [0], color='#167d53', marker='*', ls='', label='Target aim / green legal volume')]
    fig.legend(handles=legend, loc='lower center', ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, .025))
    fig.subplots_adjust(left=.015, right=.97, top=.90, bottom=.075, wspace=.03, hspace=.20)
    fig.savefig(out/'failing_plans.png', dpi=150, facecolor='white')
    plt.close(fig)
    (out/'examples.json').write_text(json.dumps(dict(coordinate_frame='environment',
        path='Actual decoder cubic spline sampled every 2.5 ms; stick tip and shaft poses',
        checkpoint=config['checkpoint'], examples=reports), indent=2))
    np.savez_compressed(out/'paths.npz', **paths)
    print(json.dumps(dict(image=str(out/'failing_plans.png'), examples=reports)), flush=True)


if __name__ == '__main__':
    main()
