"""Opt-in planning corrections, not new measured minimum-time certificates."""
import hashlib
import json
from pathlib import Path

import numpy as np

from isaac_rl.min_time_assignment import MinimumTimeChart


FORMAT_VERSION = 2
PROXY_RULE = 'last_two_overlapping_hit_targets_parked_tail_extended_v1'


def chart_fingerprint(chart):
    digest = hashlib.sha256(json.dumps(list(chart.names)).encode())
    for value in (chart.times, chart.support, chart.trials, chart.allowed):
        value = np.ascontiguousarray(value)
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def lateral_proxy(kit):
    y = np.asarray(kit.positions, float)[:, 1]
    if np.ptp(y) == 0:
        return np.zeros_like(y)
    return 2*(y-y.min())/np.ptp(y)-1


class ConditionalMinimumTimeChart(MinimumTimeChart):
    """Three time anchors per measured or explicitly initialized eligible cell.

    Ready-state arrival calibration remains scalar and is never replaced with a
    chart edge. Base times/support retain their measured semantics. Optional
    guesses for unknown cells are marked separately and can be optimized.
    """
    def __init__(self, base, proxy_values, *, anchors=None, minimum_s=.02,
                 idle_proxy=0., source='', metadata=None, initialized_unknown=None):
        super().__init__(base.names, base.times.copy(), base.support.copy(),
                         base.trials.copy(), base.allowed.copy())
        self.proxy_values = np.asarray(proxy_values, float).copy()
        if (self.proxy_values.shape != (len(self.names),) or
                not np.isfinite(self.proxy_values).all() or
                (np.abs(self.proxy_values) > 1).any()):
            raise ValueError('Proxy values must be finite, one per named drum, in [-1, 1]')
        self.minimum_s = float(minimum_s)
        self.idle_proxy = float(idle_proxy)
        if not np.isfinite(self.minimum_s) or self.minimum_s <= 0:
            raise ValueError('The control-grid time floor must be positive')
        if not np.isfinite(self.idle_proxy) or abs(self.idle_proxy) > 1:
            raise ValueError('Idle proxy must be finite and in [-1, 1]')
        self.source = str(source)
        self.metadata = dict(metadata or {})
        self._initialized_unknown = (np.zeros_like(self.allowed) if initialized_unknown is None
                                     else np.asarray(initialized_unknown, bool).copy())
        if (self._initialized_unknown.shape != self.times.shape or
                (self._initialized_unknown & ~(self.allowed & np.isposinf(self.times))).any()):
            raise ValueError('Only unknown eligible cells may be explicitly initialized')
        self._trainable = self.allowed & (np.isfinite(self.times) | self._initialized_unknown)
        self._anchors = np.repeat(self.times[..., None], 3, axis=-1)
        self.set_anchors(self._anchors if anchors is None else anchors)

    @property
    def anchors(self):
        return self._anchors.copy()

    @property
    def trainable_mask(self):
        return self._trainable.copy()

    @property
    def initialized_unknown_mask(self):
        # Locally pickled optimizers from format v1 have no explicit mask.
        return getattr(self, '_initialized_unknown', np.zeros_like(self.allowed)).copy()

    def with_initialized_unknown(self, *, center_s=.3, half_width_s=.05, seed=0):
        """Copy the overlay, adding flat guesses drawn uniformly near 300 ms.

        Existing curves are preserved. All three new anchors initially share
        one draw, so there is no assumed dependence on the other hand yet.
        """
        if (not np.isfinite([center_s, half_width_s]).all() or half_width_s < 0
                or center_s-half_width_s < self.minimum_s):
            raise ValueError('Unknown-cell initialization range must meet the positive time floor')
        mask = self.allowed & np.isposinf(self.times) & ~self._trainable
        anchors = self.anchors
        values = np.random.default_rng(seed).uniform(center_s-half_width_s,
            center_s+half_width_s, int(mask.sum()))
        anchors[mask] = values[:, None]
        metadata = dict(self.metadata)
        if mask.any():
            metadata['unknown_cell_initialization'] = dict(distribution='uniform',
                center_s=center_s, half_width_s=half_width_s, seed=int(seed),
                added_cells=int(mask.sum()), flat_three_anchor_curves=True,
                interpretation='planning_guesses_not_measured_minimum_times')
            metadata['tuning_scope'] = 'measured_and_explicitly_initialized_eligible_cells'
        return type(self)(self, self.proxy_values, anchors=anchors,
            minimum_s=self.minimum_s, idle_proxy=self.idle_proxy, source=self.source,
            metadata=metadata, initialized_unknown=self.initialized_unknown_mask | mask)

    @property
    def requires_proxy(self):
        a = self._anchors[self._trainable]
        return bool(np.any(a[:, 0] != a[:, 1]) or np.any(a[:, 1] != a[:, 2]))

    def parameter_mapping(self):
        return [dict(hand=('left', 'right')[h], source=self.names[s],
                     destination=self.names[d], anchor=anchor,
                     initially_unknown=bool(self.initialized_unknown_mask[h, s, d]))
                for h, s, d in np.argwhere(self._trainable)
                for anchor in ('left', 'center', 'right')]

    def get_parameters(self):
        return self._anchors[self._trainable].reshape(-1).copy()

    def set_parameters(self, parameters):
        parameters = np.asarray(parameters, float)
        if parameters.shape != (int(self._trainable.sum())*3,):
            raise ValueError('Wrong conditional chart parameter-vector shape')
        anchors = self._anchors.copy()
        anchors[self._trainable] = parameters.reshape(-1, 3)
        self.set_anchors(anchors)

    def set_anchors(self, anchors):
        anchors = np.asarray(anchors, float)
        if anchors.shape != self.times.shape+(3,):
            raise ValueError('Expected three anchors for every chart cell')
        frozen = np.repeat(self.times[~self._trainable, None], 3, axis=-1)
        if not np.array_equal(anchors[~self._trainable], frozen):
            raise ValueError('Unknown and ineligible cells must retain base values')
        a = anchors[self._trainable]
        if not np.isfinite(a).all():
            raise ValueError('Trainable anchors must be finite')
        L, M, R = a.T
        b, c = (R-L)/2, (L+R)/2-M
        vertex = np.zeros_like(c)
        np.divide(-b, 2*c, out=vertex, where=c != 0)
        interior = (c > 0) & (np.abs(vertex) < 1)
        low = np.minimum(L, R)
        low = np.minimum(low, np.where(interior, M+b*vertex+c*vertex**2, np.inf))
        if (low < self.minimum_s-1e-12).any():
            raise ValueError('Quadratic violates the positive time floor inside [-1, 1]')
        self._anchors = anchors.copy()  # Commit only after every validation passes.

    def minimum(self, hand, source, destination, proxy=0.):
        z = np.asarray(proxy, float)
        if not np.isfinite(z).all() or (np.abs(z) > 1).any():
            raise ValueError('Do not extrapolate conditional curves outside [-1, 1]')
        h, s, d, z = np.broadcast_arrays(hand, source, destination, z)
        result = np.array(self.times[h, s, d], copy=True)
        mask = self._trainable[h, s, d]
        a = self._anchors[h[mask], s[mask], d[mask]]
        L, M, R = a.T
        result[mask] = M+(R-L)/2*z[mask]+((L+R)/2-M)*z[mask]**2
        return result

    def description(self):
        return dict(format_version=FORMAT_VERSION, source=self.source,
                    base_sha256=chart_fingerprint(self), proxy_rule=PROXY_RULE,
                    proxy_kind='normalized_drum_lateral_y_not_waist_angle',
                    parked_tail_rule='extends_final_actual_target_without_extra_slot',
                    drum_names=list(self.names), proxy_values=self.proxy_values.tolist(),
                    idle_proxy=self.idle_proxy, minimum_s=self.minimum_s,
                    trainable_cells=int(self._trainable.sum()),
                    initialized_unknown_cells=int(self.initialized_unknown_mask.sum()),
                    parameters=len(self.get_parameters()), metadata=self.metadata)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as output:
            np.savez_compressed(output, anchors=self._anchors,
                                initialized_unknown=self.initialized_unknown_mask,
                                metadata_json=json.dumps(self.description()))

    @classmethod
    def load(cls, path, base):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json']))
            if (meta['format_version'] not in (1, FORMAT_VERSION) or
                    meta['proxy_rule'] != PROXY_RULE or
                    tuple(meta['drum_names']) != base.names or
                    meta['base_sha256'] != chart_fingerprint(base)):
                raise ValueError('Conditional overlay is incompatible with this base chart')
            if meta['format_version'] == FORMAT_VERSION and 'initialized_unknown' not in data:
                raise ValueError('Conditional overlay is missing its unknown-initialization mask')
            return cls(base, meta['proxy_values'], anchors=data['anchors'],
                       minimum_s=meta['minimum_s'], idle_proxy=meta['idle_proxy'],
                       source=meta['source'], metadata=meta.get('metadata'),
                       initialized_unknown=data['initialized_unknown'] if meta['format_version'] == FORMAT_VERSION else None)


def candidate_target_intervals(groups, actions, initial_drums, last_times,
                               committed_history=None, origin_time=0.):
    """Return two hand timelines, each as arrays (start, end, target) of shape CxM.

    History is two chronological lists of (time, drum) scheduled events. Without
    history, each last drum/time is a seed event, approximating earlier history
    as movement toward that last target from origin_time. The receding planner
    supplies full history. The final target interval extends to infinity while
    parked, without consuming another one of the last-two-hit slots. Unknown
    parked state has target -1 and contributes the explicit neutral idle proxy.
    """
    actions = np.asarray(actions, int)
    C = len(actions)
    history = [[], []] if committed_history is None else committed_history
    if len(history) != 2:
        raise ValueError('Committed history must contain one event list per hand')
    result = []
    for hand in range(2):
        seed = list(history[hand])
        if not seed and initial_drums[hand] >= 0:
            seed = [(float(last_times[hand]), int(initial_drums[hand]))]
        if any(t < origin_time for t, _ in seed) or any(
                seed[i][0] > seed[i+1][0] for i in range(len(seed)-1)):
            raise ValueError('History must be chronological and at or after origin_time')
        if seed and (seed[-1][0] != last_times[hand] or seed[-1][1] != initial_drums[hand]):
            raise ValueError('Committed history must end at the supplied last hand state')
        starts, ends, targets = [], [], []
        previous = np.full(C, origin_time, float)
        parked = np.full(C, -1, int)
        last_event = np.full(C, -1, int)
        for deadline, drum in seed:
            starts.append(previous.copy()); ends.append(np.full(C, deadline))
            targets.append(np.full(C, drum, int))
            previous[:] = deadline
            parked[:] = drum
            last_event[:] = len(starts)-1
        for group_index, rows in enumerate(groups):
            for member, hit in enumerate(rows):
                live = (actions[:, group_index]+member) % 2 == hand
                # Zero-width rows for candidates where this hand is not assigned.
                starts.append(previous.copy())
                ends.append(np.where(live, hit.time, previous))
                targets.append(np.full(C, hit.drum, int))
                previous = np.where(live, hit.time, previous)
                parked = np.where(live, hit.drum, parked)
                last_event[live] = len(starts)-1
        # Extend the last actual hit target while parked. A separate held tail
        # would spuriously displace the penultimate hit in the final-two rule.
        for index in range(len(ends)):
            ends[index][last_event == index] = np.inf
        starts.append(previous.copy())
        ends.append(np.where(last_event < 0, np.inf, previous))
        targets.append(parked)
        result.append(tuple(np.stack(x, axis=1) for x in (starts, ends, targets)))
    return result


def interval_proxy(intervals, start, end, proxy_values, idle_proxy=0.):
    """Duration-weight only the final two positive-overlap target intervals."""
    lo, hi, targets = intervals
    start, end = np.broadcast_arrays(start, end)
    overlaps = np.maximum(0., np.minimum(hi, end[:, None])-np.maximum(lo, start[:, None]))
    values = np.where(targets >= 0, np.asarray(proxy_values)[targets.clip(0)], idle_proxy)
    C = len(lo)
    weights = np.zeros((C, 2))
    latest = np.full((C, 2), idle_proxy, float)
    for column in range(lo.shape[1]):
        live = overlaps[:, column] > 0
        weights[live, 0] = weights[live, 1]
        weights[live, 1] = overlaps[live, column]
        latest[live, 0] = latest[live, 1]
        latest[live, 1] = values[live, column]
    denominator = weights.sum(-1)
    answer = np.full(C, idle_proxy, float)
    np.divide((weights*latest).sum(-1), denominator, out=answer, where=denominator > 0)
    return np.clip(answer, -1., 1.)
