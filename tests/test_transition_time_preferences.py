import pickle
from itertools import product

import numpy as np

from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl.assignment_search import hit_groups
from isaac_rl.min_time_assignment import MinimumTimeChart, window_scores
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.transition_time_preferences import (TransitionTimePreferenceLearner,
    timing_scores_and_jacobian, curve_minima, build_time_batch, select_time_context,
    unpack_time_reports)


def chart():
    base = MinimumTimeChart(('snare', 'hihat'), np.full((2, 2, 2), .25),
        np.full((2, 2, 2), 8), np.full((2, 2, 2), 8), np.ones((2, 2, 2), bool))
    return ConditionalMinimumTimeChart(base, [-.5, .9])


def clip():
    hits = [Hit(1.+i*.13, 1.+i*.13, i%2, ('snare', 'hihat')[i%2], 'left', .7) for i in range(4)]
    return RhythmClip('matched', hits, hits[-1].time+.3)


def test_timing_replay_exactly_matches_original_chart_scores_and_derivatives():
    base = chart(); ready = np.full((2, 2), .15)
    groups = hit_groups(clip()); bank = np.array(list(product((0, 1), repeat=4)))
    scores = window_scores(groups, bank, base, initial_drums=(-1, -1), last_times=(0., 0.),
        initial_minimum_s=ready, return_timing_design=True)
    values, jacobian = timing_scores_and_jacobian(base.get_parameters(), scores['timing_design'])
    np.testing.assert_allclose(values, scores['total'], rtol=1e-12, atol=1e-10)
    original = base.get_parameters()
    for index in range(len(original)):
        plus, minus = original.copy(), original.copy()
        plus[index] += 1e-7; minus[index] -= 1e-7
        cplus, cminus = chart(), chart()
        cplus.set_parameters(plus); cminus.set_parameters(minus)
        high = window_scores(groups, bank, cplus, initial_drums=(-1, -1), last_times=(0., 0.), initial_minimum_s=ready)['total']
        low = window_scores(groups, bank, cminus, initial_drums=(-1, -1), last_times=(0., 0.), initial_minimum_s=ready)['total']
        np.testing.assert_allclose((high-low)/2e-7, jacobian[:, index], rtol=1e-6, atol=1e-5)


def test_actual_time_updates_favor_observed_second_plan_and_are_bounded_in_ms():
    base = chart(); model = TransitionTimePreferenceLearner(base, max_update_ms=10.)
    batch = build_time_batch([clip()]*24, base, np.full((2, 2), .15))
    model.add_preferences(batch['design'], np.tile([0., 1.], (24, 1)))
    before = model.parameters_s.copy()
    logits_before, _ = model.logits_and_jacobian(model.mean)
    result = model.fit()
    logits_after, _ = model.logits_and_jacobian(model.mean)
    assert result['changed_time_anchors'] > 0
    assert np.mean(logits_after) < np.mean(logits_before)
    assert result['preference_loss_after'] < result['preference_loss_before']
    assert np.abs(model.parameters_s-before).max() <= .01000000001
    np.testing.assert_allclose(model.current_chart().get_parameters(), model.parameters_s)
    assert curve_minima(model.parameters_s.reshape(-1, 3)).min() >= .02-1e-12


def test_logit_gradients_are_correct_in_standardized_seconds_coordinates():
    model = TransitionTimePreferenceLearner(chart())
    batch = build_time_batch([clip()], model.current_chart(), np.full((2, 2), .15))
    model.add_preferences(batch['design'], np.array([[.2, .8]]))
    mean = np.linspace(-.02, .02, len(model.mean))
    logits, jacobian = model.logits_and_jacobian(mean)
    for index in range(len(mean)):
        high, low = mean.copy(), mean.copy()
        high[index] += 1e-6; low[index] -= 1e-6
        difference = (model.logits_and_jacobian(high)[0]-model.logits_and_jacobian(low)[0])/2e-6
        np.testing.assert_allclose(difference, jacobian[:, index], rtol=1e-5, atol=1e-7)


def test_curve_projection_handles_interior_floor_and_per_update_trust_bound():
    base = chart(); model = TransitionTimePreferenceLearner(base, max_update_ms=300.)
    proposed = model.mean.copy()
    # Convex parabola with an interior value below20 ms.
    proposed[:3] = (np.array([.05, .001, .05])-model.initial_s[:3])/model.prior_std_s
    trial = model.feasible_trial(proposed, model.mean)
    assert trial is not None
    parameters = model.initial_s+trial*model.prior_std_s
    assert curve_minima(parameters.reshape(-1, 3)).min() >= .02-1e-12
    model.template.set_parameters(parameters)


def test_ready_arrival_is_fixed_and_not_a_fictitious_transition_parameter():
    base = chart()
    groups = hit_groups(clip())[:1]
    result = window_scores(groups, np.array([[0], [1]]), base,
        initial_drums=(-1, -1), last_times=(0., 0.), initial_minimum_s=np.full((2, 2), .15),
        return_timing_design=True)
    assert not result['timing_design']['variable'].any()
    _, jacobian = timing_scores_and_jacobian(base.get_parameters(), result['timing_design'])
    assert not jacobian.any()


def test_export_and_checkpoint_contain_real_time_anchors_and_timing_covariance(tmp_path):
    from scripts.train_transition_time_preferences import export_chart
    model = TransitionTimePreferenceLearner(chart())
    batch = build_time_batch([clip()]*8, model.current_chart(), np.full((2, 2), .15))
    model.add_preferences(batch['design'], np.tile([0., 1.], (8, 1))); model.fit()
    export_chart(tmp_path/'last_chart.npz', model)
    exported = ConditionalMinimumTimeChart.load(tmp_path/'last_chart.npz', chart())
    np.testing.assert_allclose(exported.get_parameters(), model.parameters_s)
    assert exported.metadata['parameter_search']['parameter_units'] == 'seconds'
    rng = np.random.default_rng(5)
    model.save(tmp_path/'last.pkl', rng=rng.bit_generator.state, pending=batch)
    restored, extra = TransitionTimePreferenceLearner.load(tmp_path/'last.pkl')
    np.testing.assert_array_equal(restored.parameters_s, model.parameters_s)
    np.testing.assert_array_equal(restored.covariance, model.covariance)
    assert len(extra['pending']['episodes']) == 16
    assert np.sqrt(np.diag(restored.covariance)).max() <= .05+1e-12


def test_top_two_selection_uses_original_time_scorer_without_cost_overlay():
    base = chart(); ready = np.full((2, 2), .15)
    selected = select_time_context(clip(), base, ready)
    assert selected['selected_search_ranks'] == [1, 2]
    assert selected['scores'][0] <= selected['scores'][1]
    assert not hasattr(base, 'adjust_plan_scores')


def test_top_eight_are_unique_and_match_full_chart_order():
    base = chart(); ready = np.full((2, 2), .15)
    full = select_time_context(clip(), base, ready, candidates=16)
    selected = select_time_context(clip(), base, ready, candidates=8)
    assert selected['selected_search_ranks'] == list(range(1, 9))
    assert len({tuple(a) for a in selected['actions']}) == 8
    np.testing.assert_array_equal(selected['actions'], full['actions'][:8])
    np.testing.assert_array_equal(selected['actions'][:2], select_time_context(clip(), base, ready)['actions'])


def test_variable_candidate_banks_execute_and_learn_only_legal_slots(monkeypatch):
    import isaac_rl.transition_time_preferences as preferences
    base = chart(); ready = np.full((2, 2), .15)
    contexts = [select_time_context(clip(), base, ready, candidates=k) for k in (2, 8)]
    iterator = iter(contexts)
    monkeypatch.setattr(preferences, 'select_time_context', lambda *args: next(iterator))
    batch = build_time_batch([clip(), clip()], base, ready, candidates=8)
    assert len(batch['episodes']) == 10
    np.testing.assert_array_equal(batch['valid'].sum(1), [2, 8])
    np.testing.assert_array_equal(batch['report_indices'][0], [0, 2, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(batch['report_indices'][1], [1, 3, 4, 5, 6, 7, 8, 9])
    reports = [dict(reward=float(i), hit_rate=i/10.) for i in range(10)]
    rewards, hits, valid = unpack_time_reports(batch, reports)
    assert np.isnan(rewards[0, 2:]).all()
    np.testing.assert_array_equal(rewards[1], [1, 3, 4, 5, 6, 7, 8, 9])
    model = TransitionTimePreferenceLearner(base)
    metrics = model.add_ranked_preferences(batch['design'], rewards, valid)
    assert metrics['new_pairs'] == 29  # One pair plus all28 comparisons of8 plans.
    np.testing.assert_allclose(metrics['pair_weight_sum'], 2.)
    # Full-strength observations still sum to one independent rhythm per bank.
    np.testing.assert_allclose(model.weights.sum(), 2.)
    assert np.isfinite(model.targets).all()


def test_all_candidate_preferences_are_invariant_to_slot_permutation():
    base = chart(); ready = np.full((2, 2), .15)
    batch = build_time_batch([clip()]*4, base, ready, candidates=8)
    rewards = np.tile([.3, .2, .8, .4, .7, .1, .6, .5], (4, 1))
    model = TransitionTimePreferenceLearner(base)
    model.add_ranked_preferences(batch['design'], rewards, batch['valid'])
    perm = np.array([4, 7, 2, 0, 3, 6, 1, 5])
    reordered = TransitionTimePreferenceLearner(base)
    reordered.add_ranked_preferences({k: v[:, perm] for k, v in batch['design'].items()},
        rewards[:, perm], batch['valid'][:, perm])
    mean = np.linspace(-.02, .02, len(model.mean))
    np.testing.assert_allclose(model.objective(mean), reordered.objective(mean))
    for a, b in zip(model.derivatives(mean), reordered.derivatives(mean)):
        np.testing.assert_allclose(a, b, atol=1e-10)
    assert len(model.targets) == 4*28
    assert (model.targets > .5).any() and (model.targets < .5).any()


def test_old_pending_top_two_reports_remain_resume_compatible():
    batch = build_time_batch([clip()]*3, chart(), np.full((2, 2), .15))
    del batch['report_indices']; del batch['valid']
    rewards, hits, valid = unpack_time_reports(batch, [dict(reward=i, hit_rate=i/10) for i in range(6)])
    np.testing.assert_array_equal(rewards, [[0, 3], [1, 4], [2, 5]])
    assert valid.all()


def test_update_cap_override_preserves_real_timings_covariance_and_replay(tmp_path):
    model = TransitionTimePreferenceLearner(chart(), max_update_ms=10.)
    batch = build_time_batch([clip()]*8, model.current_chart(), np.full((2, 2), .15))
    model.add_preferences(batch['design'], np.tile([0., 1.], (8, 1))); model.fit()
    model.save(tmp_path/'last.pkl', pending=batch)
    restored, _ = TransitionTimePreferenceLearner.load(tmp_path/'last.pkl')
    restored.set_max_update_ms(50.)
    np.testing.assert_array_equal(restored.parameters_s, model.parameters_s)
    np.testing.assert_array_equal(restored.covariance, model.covariance)
    np.testing.assert_array_equal(restored.design['span'], model.design['span'])
    np.testing.assert_array_equal(restored.targets, model.targets)
    assert restored.max_update_s == .05 and model.max_update_s == .01


def test_shared_plan_costs_cannot_change_preferences_or_gradients():
    import copy
    model = TransitionTimePreferenceLearner(chart())
    batch = build_time_batch([clip()]*3, model.current_chart(), np.full((2, 2), .15))
    model.add_preferences(batch['design'], np.tile([.2, .8], (3, 1)))
    original_logits, original_jacobian = model.logits_and_jacobian(model.mean)
    shifted = copy.deepcopy(model)
    # A common additive penalty changes neither ranking nor preference signal.
    shifted.design['fixed_cost'][:, :, 0] += 1000.
    shifted.reference_scales = None
    logits, jacobian = shifted.logits_and_jacobian(shifted.mean)
    np.testing.assert_allclose(logits, original_logits, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian, original_jacobian, rtol=1e-12, atol=1e-12)


def test_identical_timing_function_has_zero_preference_gradient():
    model = TransitionTimePreferenceLearner(chart())
    batch = build_time_batch([clip()], model.current_chart(), np.full((2, 2), .15))
    for key in batch['design']:
        batch['design'][key][:, 1] = batch['design'][key][:, 0]
    # Fixed advantage independent of times must not induce timing updates.
    batch['design']['fixed_cost'][:, 1, 0] += 1.
    model.add_preferences(batch['design'], np.array([[.1, .9]]))
    _, jacobian = model.logits_and_jacobian(model.mean)
    assert not jacobian.any()
    before = model.parameters_s.copy(); model.fit()
    np.testing.assert_array_equal(model.parameters_s, before)


def test_legacy_replay_penalties_are_removed_on_resume_without_changing_times(tmp_path):
    from isaac_rl.transition_time_preferences import TIMING_SCORING_VERSION
    model = TransitionTimePreferenceLearner(chart())
    batch = build_time_batch([clip()], model.current_chart(), np.full((2, 2), .15))
    model.add_preferences(batch['design'], np.array([[.2, .8]]))
    # Simulate v1 known-edge support surcharges and a missing ready constant.
    model.design['fixed_cost'][model.design['variable']] = 7.5
    model.design['fixed_cost'][0, 1, 0] = 1000.
    good_ready = model.design['fixed_cost'][0, 0, 0]
    del model.scoring_version
    model.save(tmp_path/'legacy.pkl', pending=batch)
    restored, extra = TransitionTimePreferenceLearner.load(tmp_path/'legacy.pkl')
    np.testing.assert_array_equal(restored.parameters_s, model.parameters_s)
    np.testing.assert_array_equal(restored.targets, model.targets)
    np.testing.assert_array_equal(restored.covariance, model.covariance)
    assert not restored.design['fixed_cost'][restored.design['variable']].any()
    assert restored.design['fixed_cost'][0, 1, 0] == 0
    assert restored.design['fixed_cost'][0, 0, 0] == good_ready
    assert restored.scoring_version == TIMING_SCORING_VERSION
    assert extra['pending'] is None and extra['historical_penalties_removed_on_resume']
