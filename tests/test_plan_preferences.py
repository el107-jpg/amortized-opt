from itertools import product

import numpy as np
import pytest

from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart, receding_assignments, window_scores
from isaac_rl.plan_preferences import (PlanPreferenceLearner, PreferenceChart,
    normalize_plan_scores, select_context_plans)
from scripts.train_plan_preferences import build_batch, summarize_rewards, export_correction


def chart():
    names = ('snare', 'hihat', 'floor_tom')
    allowed = np.ones((2, 3, 3), bool)
    allowed[0, :, 2] = False; allowed[0, 2, :] = False
    times = np.where(allowed, .2, np.inf)
    base = MinimumTimeChart(names, times, np.where(allowed, 8, 0), np.full_like(times, 8), allowed)
    return ConditionalMinimumTimeChart(base, [0., 1., -1.])


def clip():
    names = ('snare', 'hihat', 'floor_tom')
    hits = [Hit(1.+i*.2, 1.+i*.2, i%3, names[i%3], 'left', .8) for i in range(4)]
    return RhythmClip('test', hits, 1.9)


def test_zero_correction_preserves_receding_choices_and_times():
    base = chart(); ready = np.full((2, 3), .3)
    original = base.get_parameters().copy()
    wrapped = PreferenceChart(base, np.zeros(len(original)))
    np.testing.assert_array_equal(receding_assignments(clip(), base, 4, ready)[0],
                                  receding_assignments(clip(), wrapped, 4, ready)[0])
    np.testing.assert_array_equal(base.get_parameters(), original)
    np.testing.assert_array_equal(wrapped.minimum(0, 0, 1, .5), base.minimum(0, 0, 1, .5))


def test_correction_gradients_are_plan_features_and_ready_has_no_fake_edge():
    from isaac_rl.assignment_search import hit_groups
    base = chart(); groups = hit_groups(clip()); ready = np.full((2, 3), .3)
    bank = np.array(list(product(*[base.choices(g) for g in groups])))
    score = window_scores(groups, bank, base, initial_drums=(-1, -1), last_times=(0., 0.),
        initial_minimum_s=ready, return_features=True)
    dimensions = len(base.get_parameters())
    theta = np.linspace(-.2, .2, dimensions)
    wrapped = PreferenceChart(base, theta)
    values = window_scores(groups, bank, wrapped, initial_drums=(-1, -1), last_times=(0., 0.),
        initial_minimum_s=ready)['total']
    np.testing.assert_allclose(values, normalize_plan_scores(score['total'])+score['features']@theta)
    for index in [0, dimensions//2, dimensions-1]:
        trial = theta.copy(); trial[index] += 1e-6
        shifted = window_scores(groups, bank, PreferenceChart(base, trial), initial_drums=(-1, -1),
            last_times=(0., 0.), initial_minimum_s=ready)['total']
        np.testing.assert_allclose((shifted-values)/1e-6, score['features'][:, index], atol=1e-9)
    first = window_scores(groups[:1], bank[:, :1], base, initial_drums=(-1, -1), last_times=(0., 0.),
        initial_minimum_s=ready, return_features=True)
    assert not first['features'].any()


def preference_data(n=80, equal=False):
    features = np.tile(np.array([[[1., 1.], [0., 0.]]]), (n, 1, 1))
    baseline = np.tile([.8, 0.], (n, 1))
    rewards = np.tile([.5, .5] if equal else [1., 0.], (n, 1))
    return features, baseline, rewards


def test_winning_plan_moves_cost_down_and_covariance_learns_correlated_direction():
    learner = PlanPreferenceLearner(2, temperature=1.)
    f, b, r = preference_data()
    learner.add_preferences(f, b, r)
    info = learner.fit()
    assert info['preference_loss_after'] < info['preference_loss_before']
    assert b[0, 0]+f[0, 0]@learner.mean < b[0, 1]+f[0, 1]@learner.mean
    relevant = np.array([1., 1.])/np.sqrt(2)
    irrelevant = np.array([1., -1.])/np.sqrt(2)
    assert relevant@learner.covariance@relevant < .5
    np.testing.assert_allclose(irrelevant@learner.covariance@irrelevant, 1., atol=1e-10)
    assert learner.covariance[0, 1] < 0
    assert learner.uncertainty(np.zeros(2)) == 0


def test_near_ties_weakly_teach_indifference_without_arbitrary_winner():
    learner = PlanPreferenceLearner(2, temperature=1.)
    f, b, r = preference_data(equal=True)
    result = learner.add_preferences(f, b, r)
    assert result['near_tie_pairs'] == len(f)
    np.testing.assert_allclose(learner.targets, .5)
    assert learner.weights.sum() < len(f)*.1
    learner.fit()
    assert abs((b[0]+f[0]@learner.mean)[0]-(b[0]+f[0]@learner.mean)[1]) < .8
    assert np.linalg.eigvalsh(learner.covariance).min() < 1.


def test_same_replay_is_not_counted_again_into_covariance():
    learner = PlanPreferenceLearner(2, temperature=1.)
    learner.add_preferences(*preference_data())
    learner.fit(max_steps=20)
    covariance = learner.covariance.copy()
    learner.fit(max_steps=20)
    np.testing.assert_allclose(learner.covariance, covariance, atol=1e-8)


def test_pair_weights_per_context_are_bounded_and_feature_aliases_skipped():
    learner = PlanPreferenceLearner(2)
    f = np.array([[[0., 0.], [1., 0.], [0., 1.], [0., 0.]]])
    result = learner.add_preferences(f, np.zeros((1, 4)), np.array([[0., .3, .6, .9]]))
    assert result['unidentifiable_pairs'] == 1
    assert learner.weights.sum() <= 1.+1e-12


def test_context_plans_include_greedy_and_graph_keep_eligibility_and_commit_windows():
    base = chart(); learner = PlanPreferenceLearner(len(base.get_parameters()))
    selected = select_context_plans(clip(), base, learner, np.full((2, 3), .3), np.random.default_rng(8),
        candidates=4, selection='thompson')
    assert len(selected['episodes']) == 4 and selected['distinct_selected'] == 4
    assert selected['greedy_slot'] == 0
    assert 0 <= selected['graph_slot'] < 4
    for candidate in selected['episodes']:
        assert len(candidate.hits) == len(clip().hits)
        assert all(hit.hand == 'right' for hit in candidate.hits if hit.drum_name == 'floor_tom')


def test_candidate_major_gpu_layout_and_reward_gather_are_matched():
    base = chart(); learner = PlanPreferenceLearner(len(base.get_parameters()))
    panel = [clip(), clip()]
    panel[1].name = 'second'
    batch = build_batch(panel, base, learner, np.full((2, 3), .3), np.random.default_rng(3), 4, 1., 'thompson')
    assert [c.name for c in batch['episodes']] == ['test', 'second']*4
    reports = [dict(reward=float(i), hit_rate=float(i)/10) for i in range(8)]
    rewards, _ = summarize_rewards(batch, reports)
    np.testing.assert_array_equal(rewards, [[0, 2, 4, 6], [1, 3, 5, 7]])


def test_checkpoint_restores_covariance_replay_and_exploration_rng(tmp_path):
    learner = PlanPreferenceLearner(2)
    learner.add_preferences(*preference_data())
    learner.fit()
    rng = np.random.default_rng(17)
    learner.save(tmp_path/'last.pkl', rng=rng.bit_generator.state, pending='pending plans')
    loaded, extra = PlanPreferenceLearner.load(tmp_path/'last.pkl')
    restored_rng = np.random.default_rng(); restored_rng.bit_generator.state = extra['rng']
    np.testing.assert_array_equal(learner.sample(rng), loaded.sample(restored_rng))
    np.testing.assert_array_equal(learner.features, loaded.features)
    assert extra['pending'] == 'pending plans'


def test_exported_correction_loads_for_inference_and_rejects_wrong_base(tmp_path):
    base = chart(); learner = PlanPreferenceLearner(len(base.get_parameters()))
    learner.mean[:] = np.linspace(-.1, .1, learner.dimensions)
    frozen = tmp_path/'base.npz'; base.save(frozen)
    correction = tmp_path/'correction.npz'
    export_correction(correction, frozen, learner, tmp_path/'decoder.pt')
    restored = PreferenceChart.load(correction, base, frozen)
    np.testing.assert_array_equal(restored.coefficients, learner.mean)
    assert restored.metadata['parameter_search']['selection'] == 'current_preference_MAP'
    wrong = tmp_path/'wrong.npz'; wrong.write_bytes(b'different chart')
    with pytest.raises(ValueError, match='different frozen base'):
        PreferenceChart.load(correction, base, wrong)


def test_top_two_are_exact_current_search_ranks_without_sampling(monkeypatch):
    from isaac_rl.assignment_search import hit_groups
    from isaac_rl.min_time_assignment import graph_actions
    base = chart(); learner = PlanPreferenceLearner(len(base.get_parameters()))
    learner.mean[:] = np.random.default_rng(42).normal(size=learner.dimensions)
    ready = np.full((2, 3), .3)
    def forbidden(*args, **kwargs):
        raise AssertionError('Deterministic top-two selection must not sample covariance')
    monkeypatch.setattr(learner, 'sample', forbidden)
    rng = np.random.default_rng(8)
    before = rng.bit_generator.state
    result = select_context_plans(clip(), base, learner, ready, rng)
    assert result['selected_search_ranks'] == [1, 2]
    assert result['graph_slot'] is None
    assert rng.bit_generator.state == before
    groups = hit_groups(clip())
    bank = np.array(list(product(*[base.choices(g) for g in groups])))
    values = window_scores(groups, bank, PreferenceChart(base, learner.mean),
        initial_drums=(-1, -1), last_times=(0., 0.), initial_minimum_s=ready)['total']
    graph, _ = graph_actions(groups)
    order = np.lexsort((np.arange(len(bank)), (bank != graph).sum(1), values))
    np.testing.assert_array_equal(result['actions'], bank[order[:2]])
    alternate = select_context_plans(clip(), base, learner, ready, np.random.default_rng(99))
    np.testing.assert_array_equal(alternate['actions'], result['actions'])


def test_top_two_metrics_compare_runner_up_without_graph_labels():
    base = chart(); learner = PlanPreferenceLearner(len(base.get_parameters()))
    batch = build_batch([clip(), clip()], base, learner, np.full((2, 3), .3), np.random.default_rng(4), 2, 1.)
    assert batch['graph_slots'] is None
    rewards, metrics = summarize_rewards(batch, [dict(reward=x, hit_rate=x) for x in [.2, .3, .4, .1]])
    np.testing.assert_allclose(rewards, [[.2, .4], [.3, .1]])
    assert 'mean_graph_reward' not in metrics
    assert metrics['second_beats_greedy_fraction'] == .5
    assert metrics['mean_second_reward'] == .25
