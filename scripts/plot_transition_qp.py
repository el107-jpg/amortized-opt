#!/usr/bin/env python3
"""Plot the saved four-example QP comparison using actual cubic spline FK."""
import argparse
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
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.ik import tip_position
from amortized_opt.rhythm import DrumKit
from isaac_rl.rnea_cache import build_or_load
from plot_transition_failures import draw_disk


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=Path, required=True)
    args = p.parse_args()
    folder = args.smoke/'w0'
    results = np.load(folder/'deadline_050.npz')
    kit = DrumKit.from_layout()
    cost = FusedCost(CostConfig(), max_horizon=50, rnea=build_or_load(device='cpu'))
    fig = plt.figure(figsize=(15, 12))
    panel = 0
    for hand in (0, 1):
        data = np.load(folder/f'paths_h{hand}.npz')
        q, v = torch.from_numpy(data['q']), torch.from_numpy(data['v'])
        lengths = torch.full((len(q), 3), 50)
        paths = {}
        for name in ('before', 'after'):
            sp = cost.spline.build(q, torch.from_numpy(data[name]), v, lengths)
            samples = sp.subsample(8).q
            paths[name] = tip_position(cost.fk, samples.reshape(-1, 17),
                ('left', 'right')[hand], frame='env').reshape(len(q), -1, 3).numpy()
        for i, key in enumerate(data['keys']):
            match = np.flatnonzero((results['keys'] == key).all(-1))[0]
            _, src, dst, yaw, start = key
            ax = fig.add_subplot(2, 2, panel+1, projection='3d'); panel += 1
            for d, (center, normal, radius) in enumerate(zip(kit.positions, kit.axes, kit.radii)):
                draw_disk(ax, center, normal, radius, '#18875a' if d == dst else '#83919f',
                          .25 if d == dst else .09)
                if d in (src, dst): ax.text(*(center+normal*.03), kit.names[d].replace('_', ' '), fontsize=9)
            for name, color in (('before', '#dd7949'), ('after', '#2364bb')):
                path = paths[name][i]
                ax.plot(*path.T, color=color, lw=2.4)
                ax.scatter(*path[-1], color=color, marker='x', s=55)
            ax.scatter(*paths['before'][i, 0], color='#333', marker='o', s=35)
            old = [results['before_'+x][match] for x in
                   ('terminal_region_dist', 'collision_min_sdf', 'torque_excess_max', 'joint_box_excess_max')]
            new = [results['after_'+x][match] for x in
                   ('terminal_region_dist', 'collision_min_sdf', 'torque_excess_max', 'joint_box_excess_max')]
            text = (f'Head miss: {old[0]*1000:.0f} → {new[0]*1000:.0f} mm\n'
                    f'Penetration: {max(-old[1],0)*1000:.1f} → {max(-new[1],0)*1000:.1f} mm\n'
                    f'Torque excess: {old[2]:.1f} → {new[2]:.1f} Nm\n'
                    f'Joint excess: {np.rad2deg(old[3]):.3f} → {np.rad2deg(new[3]):.3f}°')
            ax.text2D(.02, .98, text, transform=ax.transAxes, va='top', fontsize=10,
                      bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
            outcome = 'ACCEPTED' if results['after_success'][match] else 'Still fails'
            ax.set_title(f"{('Left', 'Right')[hand]}: {kit.names[src]} → {kit.names[dst]}\n{outcome}",
                         fontsize=13, loc='left', pad=10)
            scene = np.concatenate((kit.positions, paths['before'][i], paths['after'][i]))
            lo, hi = scene.min(0)-.05, scene.max(0)+.1
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi-lo); ax.view_init(elev=25, azim=135)
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('Height above floor (m)')
            ax.tick_params(labelsize=8)
    fig.suptitle('Frozen generator plans before and after up to five local QP steps', fontsize=18)
    fig.legend(handles=[Line2D([0],[0],color='#dd7949',lw=2,label='Original decoder plan'),
                        Line2D([0],[0],color='#2364bb',lw=2,label='QP repaired plan')],
               loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(top=.91, bottom=.06, left=.01, right=.97, hspace=.2)
    output = args.smoke/'qp_before_after.png'; fig.savefig(output, dpi=150); plt.close(fig)
    print(str(output.resolve()))


if __name__ == '__main__': main()
