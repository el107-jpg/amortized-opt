"""The paired-hand phase-2 family includes crossed legal assignments."""

import numpy as np
import pytest

from amortized_opt.phase2_data import (
    Phase2Pool, Phase2ScenarioConfig, add_empty_schedule_provenance,
    pair_combinations,
)
from amortized_opt.rhythm import DrumKit
from amortized_opt.train_cost import Phase2Config


def test_pair_family_enumerates_every_eligible_ordered_assignment():
    kit = DrumKit.from_layout()
    distinct, same = pair_combinations(kit)
    actual = set(distinct) | {(d, d) for d in same}
    expected = {(left, right)
                for left in kit.eligible("left")
                for right in kit.eligible("right")}
    assert actual == expected
    assert len(actual) == 42
    assert (kit.index("snare"), kit.index("hihat")) in actual
    assert all(left != kit.index("floor_tom") for left, _ in actual)


def test_synthetic_pairs_receive_aligned_empty_schedule_columns():
    fields = dict(
        q0=np.zeros((3, 17), np.float32),
        future_valid=np.zeros((3, 2, 1), np.bool_),
        goal_valid=np.ones((3, 2), np.bool_),
        goal_tth=np.full((3, 2), 0.6, np.float32),
    )
    add_empty_schedule_provenance({50: fields})
    assert all(len(value) == 3 for value in fields.values())
    assert fields["sched_valid"].shape == (3, 2, 2)
    assert not fields["sched_valid"].any()
    assert np.all(fields["clip_i"] == -1)
    assert np.all(fields["hand_hit_time"] == 0.6)


def test_pool_rejects_misaligned_columns_before_cuda_sampling():
    cfg = Phase2ScenarioConfig(horizon=25, buckets=(25,))
    with pytest.raises(ValueError, match="mismatched columns"):
        Phase2Pool({25: {
            "q0": np.zeros((3, 17), np.float32),
            "sched_valid": np.zeros((2, 2, 2), np.bool_),
        }}, cfg=cfg)


def test_fresh_training_mix_draws_paired_rows():
    class Pool:
        def kind_counts(self):
            return {3: 100, 4: 100, 6: 100}

    cfg = Phase2Config(pair_frac=0.3, rhythm_idle_frac=0.3,
                       post_hit_frac=0.6, far_fetched_frac=0.1)
    fracs = cfg.effective_kind_fracs(Pool(), log_fn=None)
    assert fracs is not None
    assert fracs[3] > 0
    assert abs(sum(fracs.values()) - 1) < 1e-12
