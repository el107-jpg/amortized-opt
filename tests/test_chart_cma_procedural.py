from itertools import groupby
import json

import numpy as np
import pytest

from amortized_opt.rhythm import DrumKit
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler
from isaac_rl.chart_cma import stratified_fitness_ranks
from isaac_rl.min_time_assignment import MinimumTimeChart


def sampler(groups=8):
    kit = DrumKit.from_layout()
    allowed = np.ones((2, 7, 7), bool)
    allowed[0, 6, :] = allowed[0, :, 6] = False
    chart = MinimumTimeChart(tuple(kit.names), np.full((2, 7, 7), .3),
        np.ones((2, 7, 7)), np.ones((2, 7, 7)), allowed)
    return ProceduralChartRhythmSampler(kit, groups, chart)


def test_procedural_coverage_all_request_pairs_simultaneous_pairs_and_tiers():
    s = sampler()
    clips = s.sample(4096, np.random.default_rng(72))
    tiers = s.last_strata
    counts = np.bincount(tiers, minlength=3)
    assert counts.max()-counts.min() <= 1
    singles = [set() for _ in range(3)]
    pairs = [set() for _ in range(3)]
    for clip, tier in zip(clips, tiers):
        rows = [list(g) for _, g in groupby(clip.hits, key=lambda h: h.time)]
        assert len(rows) == 8 and rows[0][0].time == 1.
        lo, hi = s.speed_ranges[tier]
        gaps = np.diff([g[0].time for g in rows])
        assert (gaps >= lo-1e-7).all() and (gaps <= hi+1e-7).all()
        assert clip.duration == pytest.approx(clip.hits[-1].time+.3)
        for i, group in enumerate(rows):
            assert len(group) in [1, 2]
            assert len({h.drum for h in group}) == len(group)
            assert s.chart.choices(group)
            assert all(.2 <= h.velocity <= 1. for h in group)
            if len(group) == 2:
                pairs[tier].add(tuple(sorted(h.drum for h in group)))
            if i+1 < len(rows) and len(group) == len(rows[i+1]) == 1:
                singles[tier].add((group[0].drum, rows[i+1][0].drum))
    assert all(len(x) == 49 for x in singles)
    assert all(len(x) == 21 for x in pairs)
    assert len({c.name for c in clips}) == 4096


def test_procedural_rng_reproducibility_and_fresh_sequences():
    s = sampler()
    a = s.sample(32, np.random.default_rng(2))
    assert a == s.sample(32, np.random.default_rng(2))
    assert a != s.sample(32, np.random.default_rng(3))
    payload = s.seed_payload()
    assert len(payload['groups']) == 8
    assert sum(len(g['targets']) for g in payload['groups']) == 16
    assert payload['metadata']['source'] == 'procedural_scene_buffer_seed'


def test_random_first_deadline_covers_all_tiers_and_preserves_requests():
    previous=sampler(groups=4)
    s=ProceduralChartRhythmSampler(previous.kit,4,previous.chart,first_hit_policy='tier_random')
    clips=s.sample(4096,np.random.default_rng(72),minimum_plan_choices=2)
    first=[set() for _ in range(3)];singles=[set() for _ in range(3)];pairs=[set() for _ in range(3)]
    for clip,tier in zip(clips,s.last_strata):
        rows=[list(g) for _,g in groupby(clip.hits,key=lambda h:h.time)]
        times=np.array([g[0].time for g in rows]);lo,hi=s.speed_ranges[tier]
        intervals=np.diff(np.r_[0.,times])
        assert (intervals>=lo-1e-7).all() and (intervals<=hi+1e-7).all()
        assert np.allclose(times/.02,np.round(times/.02))
        assert np.prod([len(s.chart.choices(g)) for g in rows])>=2
        first[tier].add(times[0])
        for i,group in enumerate(rows):
            if len(group)==2:pairs[tier].add(tuple(h.drum for h in group))
            if i+1<len(rows) and len(group)==len(rows[i+1])==1:
                singles[tier].add((group[0].drum,rows[i+1][0].drum))
    assert all(len(values)>1 for values in first)
    assert all(len(values)==49 for values in singles)
    assert all(len(values)==21 for values in pairs)
    assert s.summary['first_hit_policy']=='tier_random' and s.summary['first_hit_s'] is None
    assert clips==s.sample(4096,np.random.default_rng(72),minimum_plan_choices=2)


def test_top_two_sampler_has_distinct_plans_and_keeps_forced_coverage():
    s = sampler(groups=4)
    clips = s.sample(4096, np.random.default_rng(72), minimum_plan_choices=2)
    singles, pairs = [set() for _ in range(3)], [set() for _ in range(3)]
    for clip, tier in zip(clips, s.last_strata):
        rows = [list(g) for _, g in groupby(clip.hits, key=lambda h: h.time)]
        assert np.prod([len(s.chart.choices(g)) for g in rows]) >= 2
        for i, group in enumerate(rows):
            if len(group) == 2:
                pairs[tier].add(tuple(sorted(h.drum for h in group)))
            if i+1 < len(rows) and len(group) == len(rows[i+1]) == 1:
                singles[tier].add((group[0].drum, rows[i+1][0].drum))
    assert all(len(values) == 49 for values in singles)
    assert all(len(values) == 21 for values in pairs)


def test_speed_ranking_gives_hard_and_easy_tiers_equal_rank_range_and_ties():
    fitness = np.array([10., 20., 30., 0., .1, .2, 7., 7., 7., 99.])
    tiers = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1])
    valid = np.ones(10, bool); valid[-1] = False
    ranks = stratified_fitness_ranks(fitness, tiers, valid)
    np.testing.assert_array_equal(ranks[:3], [0., .5, 1.])
    np.testing.assert_array_equal(ranks[3:6], [0., .5, 1.])
    np.testing.assert_array_equal(ranks[6:9], [.5, .5, .5])
    assert np.isinf(ranks[-1])


def test_source_switch_on_resume_is_explicit_and_ranking_tracks_source(tmp_path):
    from scripts.train_chart_cma import parse_args
    (tmp_path/'config.json').write_text(json.dumps({'args': {'sampling': 'independent'}}))
    base = ['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl')]
    a = parse_args(base)
    assert a.rhythm_source == 'recorded' and a.fitness_ranking == 'global'
    a = parse_args(base+['--rhythm-source', 'procedural'])
    assert a.rhythm_source == 'procedural' and a.fitness_ranking == 'speed'
    a = parse_args(['--out', str(tmp_path/'fresh')])
    assert a.rhythm_source == 'procedural' and a.fitness_ranking == 'speed'
