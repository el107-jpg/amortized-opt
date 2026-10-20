#!/usr/bin/env python3
"""Create a flat conditional planning overlay; optionally verify saved-panel parity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from amortized_opt.rhythm import DrumKit
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart, lateral_proxy
from isaac_rl.min_time_assignment import MinimumTimeChart, receding_assignments


def verify_panel(base, conditional, panel):
    import torch
    problem = torch.load(panel, map_location='cpu', weights_only=False)
    initial = np.asarray(problem['initial'])
    groups = changed = historical_changes = 0
    max_cost_difference = 0.
    for clip_index, clip in enumerate(problem['clips']):
        for horizon in (4, 6):
            old, old_diag = receding_assignments(clip, base, horizon, initial)
            new, new_diag = receding_assignments(clip, conditional, horizon, initial)
            np.testing.assert_array_equal(new, old)
            if 'actions' in problem:
                # Archived assignments used now-removed calibration penalties.
                # Scalar/flat-overlay parity must hold under the current score;
                # equality to the old objective is deliberately not required.
                historical_changes += int(np.count_nonzero(new !=
                    problem['actions'][clip_index, (1 if horizon == 4 else 2), :len(new)]))
            groups += len(old)
            changed += int(np.count_nonzero(new != old))
            for x, y in zip(old_diag, new_diag):
                assert x == y
                max_cost_difference = max(max_cost_difference, abs(x['proxy_cost']-y['proxy_cost']))
    return dict(panel=str(Path(panel).resolve()), episodes=len(problem['clips']),
                horizons=[4, 6], group_assignments_checked=groups, changed_assignments=changed,
                max_proxy_cost_difference=max_cost_difference, exact_diagnostics_parity=True,
                changes_from_archived_penalty_score=historical_changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parity-panel', type=Path)
    parser.add_argument('--minimum-s', type=float, default=.02)
    parser.add_argument('--from-overlay', type=Path, help='preserve existing learned curves when initializing unknown cells')
    parser.add_argument('--initialize-unknown', action='store_true')
    parser.add_argument('--unknown-center-s', type=float, default=.3)
    parser.add_argument('--unknown-half-width-s', type=float, default=.05)
    parser.add_argument('--unknown-seed', type=int, default=20261006)
    args = parser.parse_args()
    if args.parity_panel and (args.initialize_unknown or args.from_overlay):
        parser.error('Scalar parity only applies to a flat overlay without unknown guesses')
    base = MinimumTimeChart.load(args.table)
    kit = DrumKit.from_layout()
    if tuple(kit.names) != base.names:
        raise ValueError('Layout drum order does not match base chart')
    conditional = ConditionalMinimumTimeChart(base, lateral_proxy(kit),
        minimum_s=args.minimum_s, source=str(args.table.resolve()),
        metadata=dict(initialization='all_three_anchors_equal_base_time',
            tuning_scope='finite_eligible_cells_only',
            ready_arrivals='separate_scalar_calibration',
            timing_interpretation='planning_correction_not_measured_feasibility_certificate'))
    if args.from_overlay:
        conditional = ConditionalMinimumTimeChart.load(args.from_overlay, base)
    if args.initialize_unknown:
        conditional = conditional.with_initialized_unknown(center_s=args.unknown_center_s,
            half_width_s=args.unknown_half_width_s, seed=args.unknown_seed)
    conditional.save(args.out)
    report = conditional.description()
    if args.parity_panel:
        report['parity'] = verify_panel(base, conditional, args.parity_panel)
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
