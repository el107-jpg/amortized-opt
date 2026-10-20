"""Physical exposure balance, rhythm preservation and reproducible resume."""
import numpy as np
import pytest
from dataclasses import replace

from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl.assignment_sampling import BalancedWindowSampler, capped_softmax


def clip(name, drums):
    return RhythmClip(name, [Hit(i*.1, i*.1, d, str(d), "right", .8)
                            for i, d in enumerate(drums)], 4.)


def test_balances_actual_hit_counts_including_accompanying_drums():
    # The common drum accompanies rare drums. Equal probabilities of clips
    # would still heavily overrepresent drum zero because clip densities differ.
    clips = [clip("common", [0]*20), clip("rare1", [0, 1, 1, 1]),
             clip("rare2", [0, 2, 2, 2, 2, 2])]
    sampler = BalancedWindowSampler(clips, 3)
    np.testing.assert_allclose(sampler.expected_shares, np.full(3, 1/3), atol=.002)
    _, windows = sampler.sample(np.random.default_rng(0), 20000)
    counts = np.bincount([h.drum for w in windows for h in w.hits], minlength=3)
    np.testing.assert_allclose(counts/counts.sum(), np.full(3, 1/3), atol=.015)


def test_preserves_simultaneous_pairs_gaps_velocities_and_source_configs():
    source = clip("pair", [0, 1, 0, 1])
    source.hits[1] = replace(source.hits[1], time=source.hits[0].time,
                             raw_time=source.hits[0].raw_time)
    sampler = BalancedWindowSampler([source], 2)
    indexes, windows = sampler.sample(np.random.default_rng(1), 1)
    assert indexes == [0]
    out = windows[0]
    assert len(out.hits) == len(source.hits)
    assert out.hits[0].time == out.hits[1].time == 1.
    np.testing.assert_allclose(np.diff([h.time for h in out.hits]),
                               np.diff([h.time for h in source.hits]))
    assert [(h.drum, h.hand, h.velocity) for h in out.hits] == [
        (h.drum, h.hand, h.velocity) for h in source.hits]
    assert source.hits[0].time == 0  # Source not mutated.


def test_new_sampler_and_restored_rng_reproduce_next_batch():
    clips = [clip("a", [0]*12), clip("b", [1]*8)]
    sampler = BalancedWindowSampler(clips, 2, span_s=.5)
    rng = np.random.default_rng(4)
    sampler.sample(rng, 20)
    saved = rng.bit_generator.state
    indexes, windows = sampler.sample(rng, 20)
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    other = BalancedWindowSampler(clips, 2, span_s=.5)
    indexes2, windows2 = other.sample(restored, 20)
    assert indexes == indexes2
    assert windows == windows2


def test_missing_drum_is_reported_instead_of_fabricating_training_data():
    with pytest.raises(ValueError, match="every drum"):
        BalancedWindowSampler([clip("a", [0, 0])], 2)


def test_deduplication_compares_notes_and_gaps_not_source_name():
    one = clip("one", [0, 1, 1, 1])
    two = clip("two", [0, 1, 1, 1])
    three = clip("three", [0, 0, 0, 1])
    s = BalancedWindowSampler([one, two, three], 2, deduplicate=True)
    assert len(s.candidates) == 2
    assert {i for i, _ in s.candidates} == {0, 2}
    np.testing.assert_allclose(s.expected_shares, [.5, .5], atol=.002)


def test_probability_cap_handles_extreme_weights_and_limits_concentration():
    np.testing.assert_allclose(capped_softmax(np.array([1000., 1001., -1000.]), .4), [.4, .4, .2])
    s = BalancedWindowSampler([clip("a", [0]*20), clip("b", [0, 1, 1, 1]),
                               clip("c", [0, 2, 2, 2, 2, 2])], 3, max_probability=.4)
    assert s.probabilities.sum() == pytest.approx(1)
    assert s.probabilities.max() <= .4
    assert s.summary["effective_windows"] >= 2.5
