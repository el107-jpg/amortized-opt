"""Complete recorded rhythm windows for independent CMA offspring."""
from dataclasses import replace
from itertools import groupby

import numpy as np
from amortized_opt.rhythm import Hit, RhythmClip


class IndependentRhythmSampler:
    """Sample fresh windows with balanced target-drum exposure, preserving hits."""
    def __init__(self, clips, groups, chart):
        self.clips = clips
        self.groups = int(groups)
        self.grouped = []
        records, counts = [], []
        for source, clip in enumerate(clips):
            rows = [list(g) for _, g in groupby(sorted(clip.hits,
                key=lambda h: (h.time, h.drum)), key=lambda h: h.time)]
            self.grouped.append(rows)
            n = len(rows)
            if n < groups:
                continue
            starts = np.arange(n-groups+1)
            invalid = np.array([len(g) > 2 or not chart.choices(g) for g in rows], int)
            prefix = np.r_[0, invalid.cumsum()]
            valid = prefix[starts+groups]-prefix[starts] == 0
            gaps = np.diff([g[0].time for g in rows])
            bad_gaps = (gaps > 1.+1e-8) | (gaps < .02-1e-8)
            prefix = np.r_[0, bad_gaps.cumsum()]
            valid &= prefix[starts+groups-1]-prefix[starts] == 0
            per_group = np.array([np.bincount([h.drum for h in g], minlength=len(chart.names)) for g in rows])
            prefix = np.vstack((np.zeros((1, len(chart.names)), int), per_group.cumsum(0)))
            for start in starts[valid]:
                records.append((source, int(start)))
                counts.append(prefix[start+groups]-prefix[start])
        if not records:
            raise ValueError('No eligible independent training windows')
        self.records = np.asarray(records, int)
        self.counts = np.asarray(counts, int)
        self.by_drum = [np.flatnonzero(self.counts[:, d]) for d in range(len(chart.names))]
        if any(not len(ids) for ids in self.by_drum):
            raise ValueError('Independent training needs windows covering every drum')
        self.last_targets = None

    def sample(self, count, rng):
        # Stratification balances exposure, while window selection and pairing
        # with parameter samples remain independently randomized.
        targets = np.arange(count) % len(self.by_drum)
        rng.shuffle(targets)
        self.last_targets = targets.copy()
        panel = []
        for target in targets:
            ids = self.by_drum[int(target)]
            index = int(ids[int(rng.integers(len(ids)))])
            source, start = self.records[index]
            clip = self.clips[source]
            selected = self.grouped[source][start:start+self.groups]
            shift = 1.-selected[0][0].time
            hits = [replace(h, time=round(h.time+shift, 8), raw_time=h.raw_time+shift)
                for group in selected for h in group]
            panel.append(replace(clip, name=f'{clip.name}@cma_group{start}',
                hits=hits, duration=hits[-1].time+.3))
        return panel


class ProceduralChartRhythmSampler:
    """Fresh random combinations with explicit request-pair/timing coverage.

    Hands are placeholders until the chart assigns them. Coverage guarantees
    concern requested combinations; actual hand transitions are logged too.
    """
    speed_ranges = ((.02, .12), (.12, .32), (.32, .90))

    def __init__(self, kit, groups, chart, *, pair_probability=.35,
                 first_hit_policy='fixed_one_second'):
        if groups < 3 or not 0 <= pair_probability <= 1:
            raise ValueError('Procedural coverage needs >=3 groups and a valid pair probability')
        if tuple(kit.names) != chart.names:
            raise ValueError('Procedural kit and chart drum order differ')
        self.kit, self.groups, self.chart = kit, int(groups), chart
        self.pair_probability = pair_probability
        if first_hit_policy not in ('fixed_one_second','tier_random'):
            raise ValueError('Unknown first-hit timing policy')
        self.first_hit_policy = first_hit_policy
        self.pairs = [(a, b) for a in range(kit.num_drums) for b in range(a+1, kit.num_drums)]
        self.last_strata = None
        self.summary = dict(source='procedural_random_combinations', groups=groups,
            pair_probability=pair_probability, first_hit_s=1. if first_hit_policy=='fixed_one_second' else None,
            first_hit_policy=first_hit_policy, grid_s=.02,
            speed_ranges_s=[list(x) for x in self.speed_ranges],
            ordered_request_pairs=kit.num_drums**2, simultaneous_pairs=len(self.pairs),
            assigned_hands='chosen by chart, not predetermined')

    def sample(self, count, rng, minimum_plan_choices=1):
        if minimum_plan_choices not in (1, 2):
            raise ValueError('Minimum eligible plan choices must be one or two')
        n = self.kit.num_drums
        levels = np.arange(count) % 3
        rng.shuffle(levels)
        codes = np.empty(count, int)
        pair_ids = np.empty(count, int)
        for level in range(3):
            ids = np.flatnonzero(levels == level)
            ordered = np.arange(len(ids)) % (n*n)
            rng.shuffle(ordered)
            codes[ids] = level*n*n+ordered
            values = np.arange(len(ids)) % len(self.pairs)
            rng.shuffle(values)
            pair_ids[ids] = values
        self.last_strata = levels.copy()
        clips = []
        for i, code in enumerate(codes):
            level = int(levels[i])
            source, destination = divmod(int(code % (n*n)), n)
            anchor = int(rng.integers(self.groups-1))
            available = [g for g in range(self.groups) if g not in (anchor, anchor+1)]
            pair_slot = int(rng.choice(available))
            lo, hi = self.speed_ranges[level]
            low_tick, high_tick = int(round(lo/.02)), int(round(hi/.02))
            first_tick = 50 if self.first_hit_policy=='fixed_one_second' else int(rng.integers(low_tick,high_tick+1))
            hits, grouped_rows = [], []
            tick = first_tick
            for group in range(self.groups):
                if group == anchor:
                    drums = [source]
                elif group == anchor+1:
                    drums = [destination]
                elif group == pair_slot:
                    drums = list(self.pairs[int(pair_ids[i])])
                elif rng.random() < self.pair_probability:
                    drums = rng.choice(n, 2, replace=False).tolist()
                else:
                    drums = [int(rng.integers(n))]
                drums.sort()
                time_s = round(tick*.02, 8)
                rows = [Hit(time_s, time_s, int(d), self.kit.names[d],
                    ('left', 'right')[j], float(rng.uniform(.2, 1.))) for j, d in enumerate(drums)]
                if not self.chart.choices(rows):
                    raise ValueError('Procedural group has no allowed hand assignment')
                hits.extend(rows)
                grouped_rows.append(rows)
                tick += int(rng.integers(low_tick, high_tick+1))
            if minimum_plan_choices == 2 and all(len(self.chart.choices(rows)) == 1 for rows in grouped_rows):
                # Preserve the forced ordered-request anchor and pair coverage.
                # Only the independent random group is redrawn if the entire
                # window otherwise has no genuine runner-up (e.g. all floor).
                free = [g for g in range(self.groups) if g not in (anchor, anchor+1, pair_slot)]
                if not free:
                    raise ValueError('No free group available for a distinct runner-up')
                g = free[0]
                for _ in range(128):
                    drums = sorted(rng.choice(n, len(grouped_rows[g]), replace=False).tolist())
                    replacement = [replace(h, drum=d, drum_name=self.kit.names[d])
                                   for h, d in zip(grouped_rows[g], drums)]
                    if len(self.chart.choices(replacement)) == 2:
                        grouped_rows[g] = replacement
                        hits = [h for rows in grouped_rows for h in rows]
                        break
                else:
                    raise ValueError('Could not generate a distinct second eligible plan')
            clips.append(RhythmClip(f'procedural_cma_{level}_{int(rng.integers(2**63)):016x}',
                hits, hits[-1].time+.3))
        return clips

    def seed_payload(self):
        # Buffer allocation only. The evaluator replaces every environment's
        # beats with its independently generated offspring rhythm after reset.
        groups = []
        for i in range(self.groups):
            groups.append(dict(time=1.+i*.2, targets=[
                dict(drum=self.kit.names[0], drum_id=0, limb='left_hand', velocity=.5),
                dict(drum=self.kit.names[-1], drum_id=len(self.kit.names)-1, limb='right_hand', velocity=.5)]))
        return dict(metadata=dict(source='procedural_scene_buffer_seed'), groups=groups)


def planned_transition_coverage(clips, num_drums):
    """Requested hand/drum/transition counts under the actual chart choices."""
    requests = np.zeros((2, num_drums), int)
    transitions = np.zeros((2, num_drums, num_drums), int)
    for clip in clips:
        previous = [-1, -1]
        for hit in sorted(clip.hits, key=lambda h: (h.time, h.drum)):
            hand = int(hit.hand == 'right')
            requests[hand, hit.drum] += 1
            if previous[hand] >= 0:
                transitions[hand, previous[hand], hit.drum] += 1
            previous[hand] = hit.drum
    return dict(planned_hand_drum_counts=requests.tolist(),
        planned_hand_transition_counts=transitions.tolist(),
        planned_hand_drum_combinations=int(np.count_nonzero(requests)),
        planned_hand_transition_cells=int(np.count_nonzero(transitions)))
