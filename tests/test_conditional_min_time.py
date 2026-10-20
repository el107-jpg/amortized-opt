from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from isaac_rl.conditional_min_time import (ConditionalMinimumTimeChart,
    candidate_target_intervals, interval_proxy, lateral_proxy)
from isaac_rl.min_time_assignment import MinimumTimeChart, window_scores, receding_assignments


def base_chart():
    kit = DrumKit.from_layout()
    times = np.full((2, 7, 7), .3)
    allowed = np.ones_like(times, bool)
    allowed[0, 6, :] = allowed[0, :, 6] = False
    times[1, 1, 2] = np.inf
    return MinimumTimeChart(tuple(kit.names), times, np.where(np.isfinite(times), 8, 0),
                            np.full_like(times, 8, int), allowed)


def overlay():
    base = base_chart()
    return ConditionalMinimumTimeChart(base, np.linspace(-1, 1, 7))


def hit(t, drum):
    return Hit(t, t, drum, DrumKit.from_layout().names[drum], 'left', .5)


def test_three_anchor_interpolation_and_vector_roundtrip():
    c = overlay()
    a = c.anchors
    a[0, 0, 1] = [.7, .3, .5]
    c.set_anchors(a)
    np.testing.assert_allclose(c.minimum(0, 0, 1, [-1, 0, 1, .5]), [.7, .3, .5, .325])
    params = c.get_parameters()
    assert len(params) == len(c.parameter_mapping()) == c.trainable_mask.sum()*3
    c.set_parameters(params)
    np.testing.assert_array_equal(c.get_parameters(), params)
    with pytest.raises(ValueError, match='extrapolate'):
        c.minimum(0, 0, 1, 1.01)


def test_positive_anchors_with_negative_interior_rejected_atomically():
    c = overlay()
    before = c.get_parameters()
    params = before.copy()
    params[:3] = [10, .021, .021]  # Interior negative although every anchor exceeds floor.
    with pytest.raises(ValueError, match='floor'):
        c.set_parameters(params)
    np.testing.assert_array_equal(c.get_parameters(), before)
    params[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        c.set_parameters(params)
    np.testing.assert_array_equal(c.get_parameters(), before)


def test_masks_unknown_ineligible_and_external_mutation():
    c = overlay()
    assert not c.trainable_mask[1, 1, 2]
    assert not c.trainable_mask[0, 6].any()
    assert np.isinf(c.minimum(1, 1, 2, .5))
    a = c.anchors
    a[0, 6, 1] = [1, 1, 1]
    with pytest.raises(ValueError, match='retain'):
        c.set_anchors(a)
    a[:] = 0
    assert (c.get_parameters() > 0).all()


def test_initialize_unknown_preserves_known_curves_and_measured_evidence(tmp_path):
    original = overlay()
    params = original.get_parameters()*1.2
    original.set_parameters(params)
    expanded = original.with_initialized_unknown(seed=19)
    assert expanded.initialized_unknown_mask.sum() == 1
    assert expanded.trainable_mask.sum() == original.trainable_mask.sum()+1
    assert len(expanded.get_parameters()) == len(params)+3
    np.testing.assert_array_equal(expanded.anchors[original.trainable_mask], original.anchors[original.trainable_mask])
    for field in ('times', 'support', 'trials', 'allowed'):
        np.testing.assert_array_equal(getattr(expanded, field), getattr(original, field))
    guess = expanded.anchors[1, 1, 2]
    assert .25 <= guess[0] <= .35
    np.testing.assert_array_equal(guess, np.full(3, guess[0]))
    assert np.isinf(original.minimum(1, 1, 2))
    assert expanded.minimum(1, 1, 2) == guess[0]
    assert not expanded.trainable_mask[0, 6].any()
    np.testing.assert_array_equal(original.with_initialized_unknown(seed=19).anchors, expanded.anchors)
    np.testing.assert_array_equal(expanded.with_initialized_unknown(seed=999).anchors, expanded.anchors)
    assert sum(row['initially_unknown'] for row in expanded.parameter_mapping()) == 3
    path = tmp_path/'expanded.npz'
    expanded.save(path)
    restored = ConditionalMinimumTimeChart.load(path, base_chart())
    np.testing.assert_array_equal(restored.anchors, expanded.anchors)
    np.testing.assert_array_equal(restored.initialized_unknown_mask, expanded.initialized_unknown_mask)
    assert restored.minimum(1, 1, 2, .5) == guess[0]


def test_initialized_unknown_scoring_has_finite_guess_and_retains_measurement_provenance():
    c = overlay().with_initialized_unknown(seed=3)
    scores = window_scores([[hit(.4, 2)]], [[1]], c,
        initial_drums=(0, 1), last_times=(0., 0.), initial_minimum_s=np.full((2, 7), .1))
    assert np.isfinite(scores['total']).all()
    assert scores['missing_edges'].tolist() == [0]
    assert c.support[1, 1, 2] == 0  # The initialization did not invent successful trials.


def test_invalid_unknown_initialization_and_mask_rejected():
    c = overlay()
    for options in [dict(half_width_s=-.01), dict(center_s=.01), dict(center_s=np.nan)]:
        with pytest.raises(ValueError, match='floor'):
            c.with_initialized_unknown(**options)
    mask = np.zeros_like(c.allowed)
    mask[0, 0, 0] = True  # Already measured.
    with pytest.raises(ValueError, match='unknown eligible'):
        ConditionalMinimumTimeChart(base_chart(), c.proxy_values, initialized_unknown=mask)
    mask[:] = False
    mask[0, 6, 0] = True  # Ineligible.
    with pytest.raises(ValueError, match='unknown eligible'):
        ConditionalMinimumTimeChart(base_chart(), c.proxy_values, initialized_unknown=mask)


def test_legacy_format_one_overlay_still_loads(tmp_path):
    c = overlay()
    metadata = c.description()
    metadata['format_version'] = 1
    path = tmp_path/'legacy.npz'
    np.savez(path, anchors=c.anchors, metadata_json=json.dumps(metadata))
    loaded = ConditionalMinimumTimeChart.load(path, base_chart())
    np.testing.assert_array_equal(loaded.get_parameters(), c.get_parameters())
    assert not loaded.initialized_unknown_mask.any()


def test_constant_shifted_curve_uses_parameters_without_proxy_construction(monkeypatch):
    c = overlay()
    a = c.anchors
    a[0, 0, 2] = [.6, .6, .6]
    c.set_anchors(a)
    assert not c.requires_proxy
    import isaac_rl.conditional_min_time as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Constant curves should skip timeline construction')
    monkeypatch.setattr(module, 'candidate_target_intervals', forbidden)
    from isaac_rl.transition_assignment import slack_cost
    scores = window_scores([[hit(.4, 2)]], [[0]], c, initial_drums=(0, 0),
                           last_times=(0., 0.), initial_minimum_s=np.full((2, 7), .1))
    np.testing.assert_allclose(scores['total'], [slack_cost(.4-.6)])
    assert scores['deficient_edges'].tolist() == [1]


def test_initial_ready_pose_arrivals_stay_separate_from_conditional_edge():
    c = overlay()
    c.set_parameters(c.get_parameters()*4)
    scores = window_scores([[hit(.4, 2)]], [[0]], c, initial_drums=(-1, -1),
                           last_times=(0., 0.), initial_minimum_s=np.full((2, 7), .1))
    from isaac_rl.transition_assignment import slack_cost
    np.testing.assert_allclose(scores['total'], [slack_cost(.4-.1)])
    assert scores['deficient_edges'].tolist() == [0]


def test_save_load_checks_base_and_order(tmp_path):
    c = overlay()
    params = c.get_parameters()*.8
    c.set_parameters(params)
    path = tmp_path/'conditional.npz'
    c.save(path)
    loaded = ConditionalMinimumTimeChart.load(path, base_chart())
    np.testing.assert_array_equal(loaded.get_parameters(), params)
    np.testing.assert_array_equal(loaded.proxy_values, c.proxy_values)
    bad = base_chart()
    bad.support[0, 0, 0] = 7
    with pytest.raises(ValueError, match='incompatible'):
        ConditionalMinimumTimeChart.load(path, bad)
    bad = base_chart()
    bad.names = bad.names[::-1]
    with pytest.raises(ValueError, match='incompatible'):
        ConditionalMinimumTimeChart.load(path, bad)


def proxy(groups, actions, hand, start, end, initial=(-1, -1), times=(0., 0.), history=None):
    timelines = candidate_target_intervals(groups, np.array(actions), initial, times, history)
    c = overlay()
    return interval_proxy(timelines[hand], np.full(len(actions), start),
                          np.full(len(actions), end), c.proxy_values)


def test_clipped_unequal_duration_and_future_target_deadline():
    groups = [[hit(.2, 0)], [hit(1., 6)]]
    # Both right targets: [-.1,.2] clips first to .1; second deadline is after .5.
    z = proxy(groups, [[1, 1]], 1, .1, .5)
    np.testing.assert_allclose(z, [.5])  # (-1*.1 + 1*.3)/.4


def test_more_than_two_intervals_only_final_two_retained():
    groups = [[hit(.1, 0)], [hit(.3, 3)], [hit(.6, 6)]]
    np.testing.assert_allclose(proxy(groups, [[1, 1, 1]], 1, 0., .6), [.6])


def test_parked_tail_extends_last_target_without_displacing_previous_hit():
    groups = [[hit(.2, 0)], [hit(.6, 6)]]
    np.testing.assert_allclose(proxy(groups, [[1, 1]], 1, 0., .8), [.5])


def test_parked_tail_after_three_actual_targets_retains_only_last_two():
    groups = [[hit(.2, 0)], [hit(.4, 3)], [hit(.6, 6)]]
    np.testing.assert_allclose(proxy(groups, [[1, 1, 1]], 1, 0., .8), [2/3])


def test_committed_history_before_root_window_preserves_overlap():
    history = [[(.1, 0)], [(.2, 6), (.5, 0), (.8, 6)]]
    groups = [[hit(1., 2)]]
    z = proxy(groups, [[0]], 1, .1, 1., initial=(0, 6), times=(.1, .8), history=history)
    # Final two targets are snare [.2,.5] and floor [.5,1], including parking.
    np.testing.assert_allclose(z, [.25])
    # This also verifies committed targets cannot disappear just because the
    # evaluated hand's prior deadline precedes several committed other hits.
    z = proxy(groups, [[0]], 1, .1, .7, initial=(0, 6), times=(.1, .8), history=history)
    np.testing.assert_allclose(z, [(-.3+.2)/.5])


def test_idle_unknown_and_parked_state():
    groups = [[hit(.5, 2)]]
    np.testing.assert_array_equal(proxy(groups, [[0]], 1, .1, .5), [0.])
    np.testing.assert_array_equal(proxy(groups, [[0]], 1, .1, .5,
                                 initial=(0, 6), times=(0., 0.)), [1.])


def test_simultaneous_members_have_symmetric_candidate_timelines():
    groups = [[hit(.4, 0), hit(.4, 6)]]
    timelines = candidate_target_intervals(groups, np.array([[0], [1]]), (3, 3), (0., 0.))
    c = overlay()
    left = interval_proxy(timelines[1], np.zeros(2), np.full(2, .4), c.proxy_values)
    right = interval_proxy(timelines[0], np.zeros(2), np.full(2, .4), c.proxy_values)
    np.testing.assert_array_equal(left, [1., -1.])
    np.testing.assert_array_equal(right, [-1., 1.])


def test_nonflat_candidate_specific_proxy_changes_winner():
    b = base_chart()
    c = ConditionalMinimumTimeChart(b, np.linspace(-1, 1, 7))
    anchors = c.anchors
    anchors[0, 0, 2] = [.7, .4, .1]
    anchors[1, 0, 2] = [.7, .4, .1]
    c.set_anchors(anchors)
    groups = [[hit(.25, 2)], [hit(.5, 0)]]
    args = dict(initial_drums=(0, 6), last_times=(0., 0.),
                initial_minimum_s=np.full((2, 7), .1))
    bank = [[0, 1], [0, 0]]
    old = window_scores(groups, bank, b, **args)['total']
    new = window_scores(groups, bank, c, **args)['total']
    assert old[0] < old[1]
    assert new[0] > new[1]  # Other hand heading left versus parked at floor.


@pytest.mark.parametrize('horizon', [4, 6])
def test_flat_exact_scores_choices_and_no_dropped_groups(horizon):
    b = base_chart()
    c = ConditionalMinimumTimeChart(b, lateral_proxy(DrumKit.from_layout()))
    groups = [[hit(.3, 0), hit(.3, 6)], [hit(.6, 1)], [hit(.9, 3)]]
    args = dict(initial_drums=(-1, -1), last_times=(0., 0.),
                initial_minimum_s=np.full((2, 7), .1))
    for key, value in window_scores(groups, [[0, 0, 1], [0, 1, 0]], b, **args).items():
        np.testing.assert_array_equal(value,
            window_scores(groups, [[0, 0, 1], [0, 1, 0]], c, **args)[key])
    clip = RhythmClip('test', [h for g in groups for h in g], .9)
    old, old_diag = receding_assignments(clip, b, horizon, args['initial_minimum_s'])
    new, new_diag = receding_assignments(clip, c, horizon, args['initial_minimum_s'])
    np.testing.assert_array_equal(old, new)
    assert old_diag == new_diag
    assert len(new) == 3


def test_saved_real_panel_flat_parity():
    root = Path(__file__).resolve().parents[1]
    base_path = root/'runs/transition_decoder_torque_integral_20261005_2240/collision_5mm/transition_table.npz'
    panel = root/'runs/min_time_chart_vs_graph_4_6_groups_20261006/problems.pt'
    if not (base_path.exists() and panel.exists()):
        pytest.skip('Local saved panel unavailable')
    from scripts.init_conditional_min_time_chart import verify_panel
    b = MinimumTimeChart.load(base_path)
    c = ConditionalMinimumTimeChart(b, lateral_proxy(DrumKit.from_layout()))
    report = verify_panel(b, c, panel)
    assert report['episodes'] == 128
    assert report['changed_assignments'] == 0
    assert report['group_assignments_checked'] == 3072
    assert report['changes_from_archived_penalty_score'] > 0
