import numpy as np

from amortized_opt.rhythm import DrumKit
from isaac_rl.random_rhythms import RandomRhythmSampler


def test_procedural_rhythms_are_reproducible_and_preserve_physical_limits():
    sampler = RandomRhythmSampler(DrumKit.from_layout())
    a = sampler.sample(np.random.default_rng(42), 256)
    b = sampler.sample(np.random.default_rng(42), 256)
    assert a == b
    assert len({c.name for c in a}) == len(a)
    seen_pairs = False
    for clip in a:
        assert 1 <= len(clip.hits) <= 10
        assert .02 <= clip.hits[0].time <= .16
        assert clip.hits[-1].time <= 1.
        groups = {}
        for h in clip.hits:
            assert abs(h.time/.02-round(h.time/.02)) < 1e-6
            assert .2 <= h.velocity <= 1.
            groups.setdefault(h.time, []).append(h.drum)
        for drums in groups.values():
            assert 1 <= len(drums) <= 2
            assert len(set(drums)) == len(drums)
            seen_pairs |= len(drums) == 2
    assert seen_pairs
    assert len({c.hits[0].time for c in a}) == 8


def test_procedural_drums_have_balanced_coverage_and_fresh_seed_changes_rhythms():
    kit = DrumKit.from_layout()
    sampler = RandomRhythmSampler(kit)
    a = sampler.sample(np.random.default_rng(123), 4096)
    counts = np.bincount([h.drum for c in a for h in c.hits], minlength=kit.num_drums)
    assert np.max(np.abs(counts/counts.sum()-1/kit.num_drums)) < .012
    assert sampler.sample(np.random.default_rng(124), 4) != a[:4]


def test_random_schedule_bank_preserves_pairs_and_padding_and_covers_short_schedules():
    from scripts.train_random_rhythm_critic import random_bank
    kit = DrumKit.from_layout()
    clips = RandomRhythmSampler(kit).sample(np.random.default_rng(2), 64)
    bank, valid, seeds = random_bank(clips, kit, np.random.default_rng(3), 16)
    for i, clip in enumerate(clips):
        groups = len({h.time for h in clip.hits})
        rows = bank[i, valid[i]]
        assert len(rows) == min(16, 2**groups)
        assert len(set(map(tuple, rows))) == len(rows)
        assert np.all(rows[:, groups:] == 0)
        assert np.array_equal(rows[0], seeds[i])
        assert np.all((rows == 0) | (rows == 1))
