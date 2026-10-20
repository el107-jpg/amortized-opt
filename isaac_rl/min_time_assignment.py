"""Receding hand assignment from a completed, yaw-pooled minimum-time chart."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from amortized_opt.hand_assign import LIMB_LEFT, LIMB_RIGHT, assign_simultaneous, kit_graph_from_layout
from isaac_rl.assignment_search import hit_groups
from isaac_rl.transition_assignment import AssignmentScoreConfig, slack_cost


@dataclass
class MinimumTimeChart:
    names: tuple
    times: np.ndarray
    support: np.ndarray
    trials: np.ndarray
    allowed: np.ndarray

    @classmethod
    def load(cls, path, statistic="minimum_s"):
        with np.load(Path(path), allow_pickle=False) as data:
            allowed = data["allowed"].copy()
            if not data["completed"][allowed].all():
                raise ValueError("Assignment probe requires a completed chart")
            return cls(tuple(data["drum_names"]), data[statistic].copy(),
                       data["support"].copy(), data["trials"].copy(), allowed)

    def choices(self, group):
        return [a for a in (0, 1) if all(
            self.allowed[(a+j) % 2, hit.drum, hit.drum]
            for j, hit in enumerate(group))]

    def minimum(self, hand, source, destination, proxy=0.):
        return self.times[hand, source, destination].copy()


def graph_actions(groups, *, initial_drums=(-1, -1), graph=None):
    """Original graph preferences, including its historical hi-hat hand ban."""
    graph = graph or kit_graph_from_layout()
    parked = {limb: None if drum < 0 else graph.names[drum]
              for limb, drum in zip((LIMB_LEFT, LIMB_RIGHT), initial_drums)}
    actions = []
    fallbacks = 0
    for i, rows in enumerate(groups):
        names = [hit.drum_name for hit in rows]
        mapping = dict(assign_simultaneous(names, parked, graph))
        if len(mapping) != len(names):
            action = i % 2
            fallbacks += 1
        else:
            action = int(mapping[names[0]] == LIMB_RIGHT)
        actions.append(action)
        for j, hit in enumerate(rows):
            parked[(LIMB_LEFT, LIMB_RIGHT)[(action+j) % 2]] = hit.drum_name
    return np.asarray(actions), fallbacks


def window_scores(groups, actions, chart, *, initial_drums, last_times,
                  initial_minimum_s, cfg=AssignmentScoreConfig(),
                  committed_history=None, origin_time=0., return_features=False,
                  return_timing_design=False):
    """Timing-slack score only; historical support/missing-data penalties are unused."""
    actions = np.asarray(actions, int)
    C = len(actions)
    drums = np.tile(initial_drums, (C, 1))
    previous = np.tile(last_times, (C, 1)).astype(float)
    total = np.zeros(C)
    missing_count = np.zeros(C, int)
    deficit_count = np.zeros(C, int)
    collect_features = return_features or return_timing_design or hasattr(chart, 'adjust_plan_scores')
    features = None
    timing_rows = []
    if collect_features:
        mask = chart.trainable_mask
        feature_index = np.full(mask.shape, -1, int)
        feature_index[mask] = np.arange(int(mask.sum()))*3
        features = np.zeros((C, int(mask.sum())*3))
        feature_normalizer = max(sum(len(rows) for rows in groups), 1)
    timelines = None
    if getattr(chart, "requires_proxy", False) or collect_features:
        from isaac_rl.conditional_min_time import candidate_target_intervals
        timelines = candidate_target_intervals(groups, actions, initial_drums,
            last_times, committed_history, origin_time)
    for g, rows in enumerate(groups):
        for member, hit in enumerate(rows):
            hands = (actions[:, g]+member) % 2
            ids = np.arange(C)
            sources = drums[ids, hands]
            allowed = chart.allowed[hands, sources.clip(0), hit.drum]
            proxy = np.zeros(C)
            if timelines is not None:
                from isaac_rl.conditional_min_time import interval_proxy
                # Constructed before scoring: paired hits are order independent,
                # and later target deadlines can contribute to an earlier edge.
                for hand in (0, 1):
                    values = interval_proxy(timelines[1-hand], previous[ids, hands],
                        np.full(C, hit.time), chart.proxy_values, chart.idle_proxy)
                    proxy[hands == hand] = values[hands == hand]
            minimum = chart.minimum(hands, sources.clip(0), hit.drum, proxy)
            if collect_features:
                offset = feature_index[hands, sources.clip(0), hit.drum]
                valid = (sources >= 0) & (offset >= 0)
                z = proxy[valid]
                basis = np.stack((z*(z-1)/2, 1-z*z, z*(z+1)/2), axis=1)
                for anchor in range(3):
                    np.add.at(features, (ids[valid], offset[valid]+anchor),
                              basis[:, anchor]/feature_normalizer)
            unknown = sources < 0
            minimum[unknown] = initial_minimum_s[hands[unknown], hit.drum]
            missing = ~np.isfinite(minimum)
            span = hit.time-previous[ids, hands]
            # Unknown calibration remains marked unknown. It contributes no
            # fabricated duration or arbitrary missing-data surcharge.
            costs = np.where(missing, 0., slack_cost(span-np.where(missing, 0, minimum), cfg))
            if return_timing_design:
                edge_indices = np.full((C, 3), -1, int)
                edge_basis = np.zeros((C, 3))
                edge_indices[valid] = offset[valid, None]+np.arange(3)
                edge_basis[valid] = basis
                fixed_cost = costs.copy()
                fixed_cost[valid] = 0.
                timing_rows.append((edge_indices, edge_basis, span.copy(), fixed_cost, valid.copy()))
            total += costs
            total[~allowed] = np.inf
            missing_count += missing
            deficit_count += (~missing) & (span < minimum-1e-8)
            drums[ids, hands] = hit.drum
            previous[ids, hands] = hit.time
    if hasattr(chart, 'adjust_plan_scores'):
        total = chart.adjust_plan_scores(total, features)
    result = dict(total=total, missing_edges=missing_count, deficient_edges=deficit_count)
    if return_features:
        result['features'] = features
    if return_timing_design:
        result['timing_design'] = {name: np.stack([row[i] for row in timing_rows], axis=1)
            for i, name in enumerate(('indices', 'basis', 'span', 'fixed_cost', 'variable'))}
    return result


def receding_assignments(clip, chart, horizon, initial_minimum_s, *,
                         initial_drums=(-1, -1), last_times=(0., 0.),
                         committed_history=None, origin_time=0.):
    """Exhaust all <=2**horizon choices, commit one group, then search again.

    Like the old graph, planning carries the last *scheduled* drum per hand.
    Physical evaluation must independently account for execution failures.
    Ready-state arrival estimates are used until a hand has a scheduled hit;
    an arbitrary ready pose is never relabeled as a nearby drum departure.
    """
    if horizon not in (4, 6):
        raise ValueError("This comparison supports four or six upcoming groups")
    groups = hit_groups(clip)
    sources = list(initial_drums)
    times = list(last_times)
    history = ([[], []] if committed_history is None else
               [list(events) for events in committed_history])
    # Legacy callers know only the last scheduled target. Seed that state;
    # callers with longer history can supply it to preserve earlier overlaps.
    for hand in (0, 1):
        if not history[hand] and sources[hand] >= 0:
            history[hand].append((times[hand], sources[hand]))
    selected = []
    diagnostics = []
    graph = kit_graph_from_layout()
    for start in range(len(groups)):
        window = groups[start:start+horizon]
        choices = [chart.choices(rows) for rows in window]
        if any(not c for c in choices):
            raise ValueError("Hit group has no eligible hand assignment")
        bank = np.asarray(list(product(*choices)))
        scores = window_scores(window, bank, chart, initial_drums=sources,
                               last_times=times, initial_minimum_s=initial_minimum_s,
                               committed_history=history, origin_time=origin_time)
        # Preserve graph choices on exact proxy ties, rather than favoring bit0.
        seed, _ = graph_actions(window, initial_drums=sources, graph=graph)
        distance = (bank != seed).sum(-1)
        best = int(np.lexsort((np.arange(len(bank)), distance, scores["total"]))[0])
        action = int(bank[best, 0])
        diagnostics.append(dict(groups=len(window), candidates=len(bank),
                                proxy_cost=float(scores["total"][best]),
                                missing_edges=int(scores["missing_edges"][best]),
                                deficient_edges=int(scores["deficient_edges"][best])))
        selected.append(action)
        for j, hit in enumerate(groups[start]):
            hand = (action+j) % 2
            sources[hand], times[hand] = hit.drum, hit.time
            history[hand].append((hit.time, hit.drum))
    return np.asarray(selected), diagnostics
