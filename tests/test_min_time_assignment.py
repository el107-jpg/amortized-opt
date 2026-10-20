from itertools import product
from dataclasses import replace

import numpy as np
import pytest

from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from isaac_rl.min_time_assignment import MinimumTimeChart, receding_assignments, window_scores, graph_actions
from isaac_rl.assignment_search import hit_groups
from scripts.probe_min_time_chart_choices import pack_physical


def chart():
    kit = DrumKit.from_layout()
    rng = np.random.default_rng(231)
    times = rng.uniform(.02, .6, (2, 7, 7))
    allowed = np.ones_like(times, bool)
    allowed[0, 6, :] = False
    allowed[0, :, 6] = False
    return MinimumTimeChart(tuple(kit.names), times, np.full_like(times, 8, int),
                            np.full_like(times, 8, int), allowed)


def clip(n=12):
    names = DrumKit.from_layout().names
    hits = [Hit((i+1)*.16, (i+1)*.16, i % 6, names[i % 6], "left", .5) for i in range(n)]
    return RhythmClip("test", hits, hits[-1].time)


@pytest.mark.parametrize("horizon", [4, 6])
def test_receding_search_keeps_full_phrase_and_each_local_exhaustive_winner(horizon):
    c = clip()
    t = chart()
    initial = np.full((2, 7), .1)
    actions, diag = receding_assignments(c, t, horizon, initial)
    groups = hit_groups(c)
    assert len(actions) == len(groups) == 12
    assert diag[0]["candidates"] == 2**horizon
    last_drums, last_times = [-1, -1], [0., 0.]
    for i, a in enumerate(actions):
        window = groups[i:i+horizon]
        bank = np.array(list(product(*[t.choices(g) for g in window])))
        scores = window_scores(window, bank, t, initial_drums=last_drums,
                               last_times=last_times, initial_minimum_s=initial)["total"]
        assert scores[bank[:, 0] == a].min() == scores.min()
        last_drums[a], last_times[a] = groups[i][0].drum, groups[i][0].time


def test_idle_hand_keeps_time_since_its_last_scheduled_hit():
    t = chart()
    t.times[:] = .5
    t.times[0, 0, 2] = .3
    h = Hit(.4, .4, 2, t.names[2], "left", .5)
    scores = window_scores([[h]], [[0], [1]], t, initial_drums=[0, 1],
        last_times=[.02, 0.], initial_minimum_s=np.full((2, 7), .1))
    assert scores["deficient_edges"].tolist() == [0, 1]
    assert scores["total"][0] < scores["total"][1]


def test_missing_edge_is_unknown_without_a_historical_penalty():
    t = chart()
    t.times[0, 0, 2] = np.inf
    t.support[0, 0, 2] = 0
    h = Hit(.4, .4, 2, t.names[2], "left", .5)
    score = window_scores([[h]], [[0]], t, initial_drums=[0, 1],
        last_times=[0., 0.], initial_minimum_s=np.full((2, 7), .1))
    assert score["missing_edges"].tolist() == [1]
    assert np.isfinite(score["total"]).all()
    assert score['total'].tolist() == [0.]


def test_historical_support_counts_and_penalty_weights_cannot_change_ranking():
    from isaac_rl.transition_assignment import AssignmentScoreConfig
    t = chart()
    h = Hit(.4, .4, 2, t.names[2], 'left', .5)
    kwargs = dict(initial_drums=[0, 1], last_times=[0., 0.], initial_minimum_s=np.full((2, 7), .1))
    baseline = window_scores([[h]], [[0], [1]], t, **kwargs)
    t.support[0] = 0; t.support[1] = 100; t.trials[:] = 1
    altered = window_scores([[h]], [[0], [1]], t,
        cfg=AssignmentScoreConfig(support_weight=100000., missing_cost=1000000.), **kwargs)
    np.testing.assert_array_equal(baseline['total'], altered['total'])


def test_missing_ready_calibration_has_no_surcharge_and_remains_unknown():
    t = chart(); h = Hit(1., 1., 0, t.names[0], 'left', .5)
    ready = np.full((2, 7), .1); ready[1, 0] = np.inf
    scored = window_scores([[h]], [[0], [1]], t, initial_drums=(-1, -1),
        last_times=(0., 0.), initial_minimum_s=ready)
    assert scored['missing_edges'].tolist() == [0, 1]
    assert scored['total'][1] == 0
    assert scored['total'][0] < 1e-20


def test_complete_pair_and_long_window_are_kept():
    c = clip()
    last = replace(c.hits[-1], drum=6, drum_name="floor_tom")
    c.hits.append(last)
    t = chart()
    actions, diag = receding_assignments(c, t, 6, np.full((2, 7), .1))
    assert actions[-1] == 0  # first member left, floor-tom member right
    assert len(graph_actions(hit_groups(c))[0]) == 12
    packed = pack_physical([c], "cpu")
    assert packed["valid"].sum() == 12
    assert (packed["drums"] >= 0).sum() == 13
    assert packed["times"].max() > 1
