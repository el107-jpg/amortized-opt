import numpy as np
from isaac_rl.transition_time_search import coarse_horizons, fine_horizons


def test_scan_covers_below_failed_reference_and_longer_fallbacks():
    coarse = coarse_horizons()
    assert coarse == [50, 40, 30, 20, 10, 1, 60, 70, 80, 90, 100]
    success = np.isin(coarse, [30, 70])  # failure at 1 s does not censor .6 s
    assert fine_horizons(coarse, success) == list(range(21, 30))


def test_refinement_scans_whole_interval_without_monotonic_assumption():
    coarse = coarse_horizons()
    assert fine_horizons(coarse, np.isin(coarse, [10])) == list(range(2, 10))
    assert fine_horizons(coarse, np.isin(coarse, [1])) == []
    assert fine_horizons(coarse, np.zeros(len(coarse), bool)) == []


def test_stride_one_evaluates_all_100_deadlines_including_nonmonotonic_islands():
    coarse=coarse_horizons(stride=1)
    assert coarse[0]==50
    assert sorted(coarse)==list(range(1,101))
    assert fine_horizons(coarse,np.isin(coarse,[7,43,81]))==[]


def test_decoder_cache_discards_repaired_metrics_and_success():
    from scripts.search_transition_minimum_times import decoder_rows
    rows = dict(keys=np.zeros((1, 5), int), before_success=np.array([False]),
        after_success=np.array([True]), before_terminal_region_dist=np.array([.02]),
        after_terminal_region_dist=np.array([0.]), after_accepted_steps=np.array([3]))
    raw = decoder_rows(rows)
    assert not raw['after_success'][0]
    assert raw['after_terminal_region_dist'][0] == .02
    assert 'after_accepted_steps' not in raw


def test_decoder_only_database_never_uses_baseline_qp_witnesses(tmp_path):
    from types import SimpleNamespace
    from isaac_rl.transition_assignment import TransitionTable
    from scripts.search_transition_minimum_times import publish, POLICY
    names = ('snare', 'hihat', 'tom1', 'tom2', 'cymbal_left', 'cymbal_right', 'floor_tom')
    shape = (2, 7, 7, 4)
    table = TransitionTable(names, np.array([-.9, -.3, .3, .9]), np.full(shape, np.inf),
        np.zeros(shape, int), np.zeros(shape, int), np.zeros((8, 8)),
        np.ones((8, 8), bool), {'forbidden_drums_by_hand': POLICY})
    baseline = dict(keys=np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]),
        before_success=np.array([False, True]), after_success=np.array([True, True]))
    args = SimpleNamespace(out=tmp_path, devices=['cuda:0'], stride=10,
        baseline=tmp_path/'baseline', decoder_only=True)
    summary = publish(args, {'samples': 1, 'qp': {'iterations': 5}}, table,
        baseline, np.ones(2, bool), True)
    assert summary['decoder_supported_starts'] == 1
    assert 'qp_supported_starts' not in summary
    result = TransitionTable.load(tmp_path/'database/transition_table_decoder.npz')
    assert np.isinf(result.minimum_s[0, 0, 0, 0])
    assert result.minimum_s[1, 0, 0, 0] == 1.
    assert result.metadata['qp']['iterations'] == 0
    assert not (tmp_path/'database/transition_table_qp.npz').exists()
    assert 'qp_mean' not in (tmp_path/'database/transition_table.csv').read_text()


def test_region_boundaries_and_round_trip_preserve_scalar_and_batched_lookup(tmp_path):
    from isaac_rl.transition_assignment import TransitionTable
    shape = (2, 1, 1, 3)
    t = TransitionTable(('snare',), np.array([-.6, 0., .6]), np.full(shape, np.inf),
        np.zeros(shape, int), np.zeros(shape, int), np.zeros((2, 2)), np.ones((2, 2), bool),
        {'yaw_region_edges': [-.9, -.3, .3, .9]})
    yaws = np.array([-.9, -.31, -.3, 0., .299, .3, .9])
    expected = [0, 0, 1, 1, 1, 2, 2]
    np.testing.assert_array_equal(t.yaw_indices(yaws), expected)
    assert [t.yaw_index(y) for y in yaws] == expected
    t.save(tmp_path/'table.npz')
    np.testing.assert_array_equal(TransitionTable.load(tmp_path/'table.npz').yaw_indices(yaws), expected)
