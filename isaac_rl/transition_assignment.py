"""Cheap joint hand assignment using empirically calibrated transition times.

Times are means of sampled successful minima, not physical lower bounds.
Missing samples remain visible through support counts and risk penalties.
The shared waist timeline uses each hand's NEXT assigned goal, including idle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np

from isaac_rl.assignment_search import hit_groups


def minimum_statistics(durations, successful):
    """One minimum per starting state, then average; never average all trials.

    Last dimension contains randomly searched durations. Unsupported starts
    contribute to failure_rate, not a fabricated finite minimum.
    """
    durations = np.asarray(durations, float)
    successful = np.asarray(successful, bool)
    minima = np.where(successful, durations, np.inf).min(-1)
    live = np.isfinite(minima)
    count = live.sum(-1)
    mean = np.divide(np.where(live, minima, 0).sum(-1), count,
                     out=np.full(count.shape, np.inf), where=count > 0)
    safe_mean = np.where(count > 0, mean, 0)
    residual = np.where(live, minima, safe_mean[..., None])-safe_mean[..., None]
    variance = np.divide((residual**2).sum(-1), count,
                         out=np.zeros(count.shape, float), where=count > 0)
    return dict(mean=mean, std=np.sqrt(variance), count=count,
                trials=np.full(count.shape, minima.shape[-1], int), minima=minima)


@dataclass(frozen=True)
class AssignmentScoreConfig:
    comfortable_slack_s: float = .10
    slack_scale_s: float = .02
    slack_weight: float = 10.
    deficit_weight: float = 100.
    support_weight: float = 0.
    missing_cost: float = 0.
    yaw_turn_weight: float = .1
    yaw_speed_weight: float = .1
    yaw_angle_scale_rad: float = .3
    yaw_speed_scale_rad_s: float = 1.

    def __post_init__(self):
        values = np.asarray(list(self.__dict__.values()), float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('score parameters must be finite and nonnegative')
        if min(self.slack_scale_s, self.yaw_angle_scale_rad,
               self.yaw_speed_scale_rad_s) <= 0:
            raise ValueError('score scales must be positive')


def slack_cost(slack, cfg=AssignmentScoreConfig()):
    """Squared softplus shoulder and a strong penalty below estimated minimum.

    At comfortable_slack (default 100 ms) the shoulder is almost flat: the
    softplus argument is -5. At zero slack it is 0. Beyond the minimum it
    grows steeply, with an additional quadratic deficit cost.
    """
    s = np.asarray(slack, float)
    x = (cfg.comfortable_slack_s-5*cfg.slack_scale_s-s)/cfg.slack_scale_s
    return (cfg.slack_weight*np.logaddexp(0, x)**2+
            cfg.deficit_weight*(np.maximum(-s, 0)/cfg.slack_scale_s)**2)


@dataclass
class TransitionTable:
    drum_names: tuple[str, ...]
    yaw_centers: np.ndarray
    minimum_s: np.ndarray  # hand, source, destination, yaw
    support: np.ndarray
    trials: np.ndarray
    pair_yaw: np.ndarray  # ordered left/right next goals; last index = idle
    pair_supported: np.ndarray
    metadata: dict

    def __post_init__(self):
        D, Y = len(self.drum_names), len(self.yaw_centers)
        if Y < 1 or not np.isfinite(self.yaw_centers).all() or (np.diff(self.yaw_centers) <= 0).any():
            raise ValueError('yaw centers must be finite and strictly increasing')
        shape = (2, D, D, Y)
        if any(np.shape(a) != shape for a in (self.minimum_s, self.support, self.trials)):
            raise ValueError(f'transition arrays must have shape {shape}')
        if self.pair_yaw.shape != (D+1, D+1) or self.pair_supported.shape != (D+1, D+1):
            raise ValueError('pair tables must include an idle row and column')
        if not np.isfinite(self.pair_yaw).all():
            raise ValueError('pair yaw must be finite, including unsupported fallbacks')
        if ((self.support < 0) | (self.support > self.trials)).any():
            raise ValueError('invalid transition support')
        if ((self.minimum_s < 0) | np.isnan(self.minimum_s)).any():
            raise ValueError('times must be nonnegative; unsupported entries use infinity')
        if 'yaw_region_edges' in self.metadata:
            edges = np.asarray(self.metadata['yaw_region_edges'], float)
            if (edges.shape != (Y+1,) or not np.isfinite(edges).all()
                    or (np.diff(edges) <= 0).any()
                    or (self.yaw_centers <= edges[:-1]).any()
                    or (self.yaw_centers >= edges[1:]).any()):
                raise ValueError('Yaw region edges must bracket each representative center')

    def save(self, path):
        np.savez_compressed(path, drum_names=self.drum_names, yaw_centers=self.yaw_centers,
            minimum_s=self.minimum_s, support=self.support, trials=self.trials,
            pair_yaw=self.pair_yaw, pair_supported=self.pair_supported,
            metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(tuple(data['drum_names']), *(data[k].copy() for k in
                ('yaw_centers', 'minimum_s', 'support', 'trials', 'pair_yaw', 'pair_supported')),
                json.loads(str(data['metadata'])))

    def yaw_index(self, yaw):
        return int(self.yaw_indices(float(yaw)))

    def yaw_indices(self, yaws):
        """Use explicit region boundaries when present; preserve old point tables."""
        yaws = np.asarray(yaws, float)
        if not np.isfinite(yaws).all():
            raise ValueError('Yaw must be finite')
        if 'yaw_region_edges' in self.metadata:
            edges = np.asarray(self.metadata['yaw_region_edges'], float)
            return np.clip(np.searchsorted(edges, yaws, side='right')-1,
                           0, len(self.yaw_centers)-1)
        return np.abs(yaws[..., None]-self.yaw_centers).argmin(-1)

    def allowed_drums(self):
        """Metadata-defined hand eligibility, with an always-allowed idle slot."""
        allowed = np.ones((2, len(self.drum_names)+1), bool)
        for hand, name in enumerate(('left', 'right')):
            for drum in self.metadata.get('forbidden_drums_by_hand', {}).get(name, []):
                if drum not in self.drum_names:
                    raise ValueError(f'Unknown forbidden drum: {drum}')
                allowed[hand, self.drum_names.index(drum)] = False
        return allowed

    def allowed_groups(self):
        allowed = self.allowed_drums()[:, :-1]
        return np.broadcast_to(allowed[:, :, None, None] & allowed[:, None, :, None],
                               self.minimum_s.shape)


def score_assignment(groups, actions, table, *, yaw_now, initial_drums=(-1, -1),
                     initial_minimum_s=None, cfg=AssignmentScoreConfig()):
    """Score one complete schedule. Unknown initial drums require live estimates.

    initial_minimum_s is (2,D,Y), calibrated from the actual initial q/qd.
    Never map an arbitrary ready/moving arm onto a drum-departure table.
    """
    D = len(table.drum_names)
    if len(groups) != len(actions) or not groups:
        raise ValueError('one binary assignment required per nonempty hit group')
    times = np.asarray([g[0].time for g in groups], float)
    if not np.isfinite(times).all() or times[0] <= 0 or (np.diff(times) <= 0).any():
        raise ValueError('hit times must be finite, positive and strictly increasing')
    if not np.isfinite(yaw_now) or any(d < -1 or d >= D for d in initial_drums):
        raise ValueError('invalid initial state')
    allowed = table.allowed_drums()
    if any(d >= 0 and not allowed[h, d] for h, d in enumerate(initial_drums)):
        raise ValueError('initial drum violates hand eligibility; use measured-state estimates for an arbitrary pose')
    if any(d < 0 for d in initial_drums):
        if initial_minimum_s is None or np.shape(initial_minimum_s) != (2, D, len(table.yaw_centers)):
            raise ValueError('unknown initial drums require measured-state initial_minimum_s')
    assigned = np.full((len(groups), 2), -1, int)
    for k, (rows, action) in enumerate(zip(groups, actions)):
        if action not in (0, 1) or not 1 <= len(rows) <= 2:
            raise ValueError('binary choices support one or two simultaneous hits')
        for member, hit in enumerate(rows):
            if not 0 <= hit.drum < D:
                raise ValueError('drum index outside calibrated table')
            assigned[k, (int(action)+member) % 2] = hit.drum
    forbidden_hits = int((~allowed[np.arange(2), np.where(assigned < 0, D, assigned)]).sum())
    # Each interval's pair consists of both hands' next assigned goals.
    next_goals = np.full_like(assigned, D)
    goals = np.full(2, D, int)
    for k in range(len(groups)-1, -1, -1):
        goals = np.where(assigned[k] >= 0, assigned[k], goals)
        next_goals[k] = goals
    yaws = table.pair_yaw[next_goals[:, 0], next_goals[:, 1]].copy()
    idle = (next_goals == D).all(-1)
    previous_yaw = yaw_now
    for k in range(len(yaws)):
        if idle[k]:
            yaws[k] = previous_yaw
        previous_yaw = yaws[k]
    dt = np.diff(np.r_[0., times])
    changes = np.diff(np.r_[yaw_now, yaws])
    yaw_cost = float(np.sum(cfg.yaw_turn_weight*(changes/cfg.yaw_angle_scale_rad)**2+
        cfg.yaw_speed_weight*(changes/dt/cfg.yaw_speed_scale_rad_s)**2))
    pair_missing = int((~table.pair_supported[next_goals[:, 0], next_goals[:, 1]]).sum())
    timing = risk = 0.
    edges = []
    previous_time = np.zeros(2)
    previous_drum = np.asarray(initial_drums).copy()
    starts = np.r_[0., times[:-1]]
    for k, deadline in enumerate(times):
        for hand in range(2):
            destination = assigned[k, hand]
            if destination < 0:
                continue
            span = deadline-previous_time[hand]
            overlap = np.maximum(0., np.minimum(times, deadline)-np.maximum(starts, previous_time[hand]))
            average_yaw = float(overlap@yaws/span)
            yi = table.yaw_index(average_yaw)
            source = previous_drum[hand]
            if source < 0:
                minimum = float(initial_minimum_s[hand, destination, yi])
                failure = 0.
            else:
                key = (hand, source, destination, yi)
                minimum = float(table.minimum_s[key])
                failure = 1-float(table.support[key])/max(int(table.trials[key]), 1)
            missing = not np.isfinite(minimum)
            edge_cost = cfg.missing_cost if missing else float(slack_cost(span-minimum, cfg))
            timing += edge_cost
            risk += cfg.support_weight*failure
            edges.append(dict(hand=hand, source=int(source), destination=int(destination),
                available_s=float(span), minimum_s=None if missing else minimum,
                slack_s=None if missing else float(span-minimum), average_yaw_rad=average_yaw,
                yaw_index=yi, cost=edge_cost, sampled_failure_fraction=failure))
            previous_time[hand], previous_drum[hand] = deadline, destination
    total = timing+risk+yaw_cost+cfg.missing_cost*pair_missing
    if forbidden_hits:
        total = np.inf
    return dict(total=float(total), timing_cost=timing, support_cost=risk, yaw_cost=yaw_cost,
        forbidden_hits=forbidden_hits,
        missing_pairs=pair_missing, missing_edges=sum(e['minimum_s'] is None for e in edges),
        actions=list(map(int, actions)), assigned_drums=assigned.tolist(),
        interval_pair_goals=next_goals.tolist(), interval_yaw_rad=yaws.tolist(), edges=edges)


def score_assignments(groups, actions, table, *, yaw_now, initial_drums=(-1, -1),
                      initial_minimum_s=None, cfg=AssignmentScoreConfig()):
    """Vectorized scores for a complete candidate bank; no generator calls."""
    actions = np.asarray(actions, int)
    if actions.ndim != 2 or actions.shape[1] != len(groups) or not len(actions):
        raise ValueError('actions must be a nonempty candidate-by-group array')
    if not np.isin(actions, (0, 1)).all():
        raise ValueError('assignments must be binary')
    # Share scalar validation and maintain one readable diagnostic reference.
    score_assignment(groups, actions[0], table, yaw_now=yaw_now,
        initial_drums=initial_drums, initial_minimum_s=initial_minimum_s, cfg=cfg)
    C, G = actions.shape
    D = len(table.drum_names)
    times = np.asarray([g[0].time for g in groups])
    starts = np.r_[0., times[:-1]]
    dt = times-starts
    assigned = np.full((C, G, 2), -1, int)
    batch = np.arange(C)
    for g, rows in enumerate(groups):
        for member, hit in enumerate(rows):
            assigned[batch, g, (actions[:, g]+member) % 2] = hit.drum
    forbidden_hits = (~table.allowed_drums()[np.arange(2),
        np.where(assigned < 0, D, assigned)]).sum(axis=(1, 2))
    next_goals = np.full_like(assigned, D)
    goals = np.full((C, 2), D, int)
    for g in range(G-1, -1, -1):
        goals = np.where(assigned[:, g] >= 0, assigned[:, g], goals)
        next_goals[:, g] = goals
    yaws = table.pair_yaw[next_goals[..., 0], next_goals[..., 1]]
    changes = np.diff(np.concatenate((np.full((C, 1), yaw_now), yaws), 1), axis=1)
    yaw = np.sum(cfg.yaw_turn_weight*(changes/cfg.yaw_angle_scale_rad)**2+
                 cfg.yaw_speed_weight*(changes/dt/cfg.yaw_speed_scale_rad_s)**2, 1)
    pair_missing = (~table.pair_supported[next_goals[..., 0], next_goals[..., 1]]).sum(1)
    timing = np.zeros(C); risk = np.zeros(C); missing_edges = np.zeros(C, int)
    previous_time = np.zeros((C, 2))
    previous_drum = np.tile(initial_drums, (C, 1))
    for g, deadline in enumerate(times):
        for hand in range(2):
            live = assigned[:, g, hand] >= 0
            ids = np.flatnonzero(live)
            if not len(ids):
                continue
            destination = assigned[ids, g, hand]
            span = deadline-previous_time[ids, hand]
            overlap = np.maximum(0., np.minimum(times, deadline)[None]-
                np.maximum(starts[None], previous_time[ids, hand, None]))
            average = np.sum(overlap*yaws[ids], 1)/span
            yi = table.yaw_indices(average)
            source = previous_drum[ids, hand]
            minimum = table.minimum_s[hand, source.clip(0), destination, yi].copy()
            failure = 1-table.support[hand, source.clip(0), destination, yi]/np.maximum(
                table.trials[hand, source.clip(0), destination, yi], 1)
            initial = source < 0
            if initial.any():
                minimum[initial] = initial_minimum_s[hand, destination[initial], yi[initial]]
                failure[initial] = 0.
            missing = ~np.isfinite(minimum)
            # Use finite placeholders to avoid evaluating a loss at -infinity.
            cost = slack_cost(span-np.where(missing, 0., minimum), cfg)
            timing[ids] += np.where(missing, cfg.missing_cost, cost)
            risk[ids] += cfg.support_weight*failure
            missing_edges[ids] += missing
            previous_time[ids, hand] = deadline
            previous_drum[ids, hand] = destination
    total = np.where(forbidden_hits > 0, np.inf, timing+risk+yaw+cfg.missing_cost*pair_missing)
    return dict(total=total, forbidden_hits=forbidden_hits,
                timing_cost=timing, support_cost=risk, yaw_cost=yaw,
                missing_pairs=pair_missing, missing_edges=missing_edges)


def search_assignments(clip, table, *, max_groups=10, detailed_candidates=False, **kwargs):
    """Exact proxy search respecting table hand eligibility, with winner diagnostics.

    Candidate rows normally contain compact scores. Request full edge diagnostics
    explicitly for offline analysis; this is substantially slower.
    """
    groups = hit_groups(clip)
    if len(groups) > max_groups:
        raise ValueError('window exceeds search capacity; cap upstream without splitting pairs')
    allowed = table.allowed_drums()
    choices = []
    for rows in groups:
        if not 1 <= len(rows) <= 2 or any(not 0 <= hit.drum < len(table.drum_names) for hit in rows):
            raise ValueError('invalid hit group for calibrated table')
        choices.append([a for a in (0, 1)
                        if all(allowed[(a+m) % 2, hit.drum] for m, hit in enumerate(rows))])
    if any(not c for c in choices):
        raise ValueError('no assignment satisfies drum-to-hand eligibility')
    actions = np.asarray(list(product(*choices)), int)
    scores = score_assignments(groups, actions, table, **kwargs)
    selected = int(scores['total'].argmin())
    best = score_assignment(groups, actions[selected], table, **kwargs)
    if detailed_candidates:
        rows = [score_assignment(groups, action, table, **kwargs) for action in actions]
    else:
        rows = [dict(actions=action.tolist(), **{k: float(v[i]) for k, v in scores.items()})
                for i, action in enumerate(actions)]
    return best, rows
