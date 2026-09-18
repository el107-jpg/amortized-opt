"""Replay sampling downweights high harvested torque violations."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.train_cost import (
    Phase2Config,
    ReplayPool,
    _replay_sampling_probs,
    _replay_viol_score_telemetry,
)


def _minimal_replay(
    *, viol_downweight: float = 1.0, viol_uniform_mix: float = 0.0,
) -> ReplayPool:
    replay = object.__new__(ReplayPool)
    replay.recency_bias = 0.0
    replay.viol_downweight = float(viol_downweight)
    replay.viol_uniform_mix = float(viol_uniform_mix)
    replay.capacity = 16
    replay.device = torch.device("cpu")
    replay.dtype = torch.float32
    m = 8
    replay.data = {
        25: {
            "viol_score": torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]),
            "hit": torch.full((16, 2), 5),
            "active": torch.ones(16, 2, dtype=torch.bool),
        }
    }
    replay._n = {25: m}
    replay._pos = {25: m}
    replay._order = {25: torch.arange(16)}
    return replay


def test_replay_sample_downweights_high_viol_score():
    replay = _minimal_replay(viol_downweight=1.0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    picks = replay.sample_idx(25, 512, generator=gen, viol_downweight=1.0)
    assert picks is not None
    low = int((picks < 4).sum())
    high = int((picks >= 4).sum())
    assert low > high * 3


def test_replay_viol_downweight_off_skips_weighting():
    replay = _minimal_replay(viol_downweight=0.0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    picks = replay.sample_idx(25, 64, generator=gen, viol_downweight=0.0)
    assert picks is not None
    assert int(picks.min()) >= 0 and int(picks.max()) < 8


def test_uniform_mix_probability_has_analytical_floor():
    weights = torch.tensor([9.0, 1.0, 0.0, float("nan")])
    gamma = 0.2
    probs = _replay_sampling_probs(weights, gamma)
    expected = 0.8 * torch.tensor([0.9, 0.1, 0.0, 0.0]) + 0.2 / 4
    assert torch.allclose(probs, expected)
    assert float(probs.min()) == pytest.approx(gamma / 4)
    assert float(probs.sum()) == pytest.approx(1.0)


def test_gamma_zero_is_bit_identical_to_legacy_multinomial():
    replay = _minimal_replay(viol_downweight=0.3)
    weights = replay._row_weights(25, replay.size(25), viol_downweight=0.3)
    expected_gen = torch.Generator(device="cpu").manual_seed(17)
    actual_gen = torch.Generator(device="cpu").manual_seed(17)
    expected = torch.multinomial(
        weights, 128, replacement=True, generator=expected_gen,
    )
    actual = replay.sample_idx(
        25, 128, generator=actual_gen,
        viol_downweight=0.3, viol_uniform_mix=0.0,
    )
    assert torch.equal(actual, expected)


def test_gamma_one_is_uniform_and_draws_with_replacement():
    probs = _replay_sampling_probs(torch.tensor([100.0, 2.0, 0.0]), 1.0)
    assert torch.equal(probs, torch.full((3,), 1.0 / 3.0))

    replay = _minimal_replay(viol_uniform_mix=1.0)
    gen = torch.Generator(device="cpu").manual_seed(4)
    picks = replay.sample_idx(25, 128, generator=gen)
    assert picks.numel() == 128
    assert picks.unique().numel() <= replay.size(25)


@pytest.mark.parametrize("gamma", [0.0, 0.25])
def test_exponential_underflow_falls_back_to_finite_uniform(gamma):
    replay = _minimal_replay(
        viol_downweight=1.0, viol_uniform_mix=gamma,
    )
    replay.data[25]["viol_score"][:8] = 1.0e30
    first_gen = torch.Generator(device="cpu").manual_seed(9)
    second_gen = torch.Generator(device="cpu").manual_seed(9)
    first = replay.sample_idx(25, 64, generator=first_gen)
    second = replay.sample_idx(25, 64, generator=second_gen)
    assert torch.equal(first, second)
    assert int(first.min()) >= 0 and int(first.max()) < 8


def test_uniform_mix_composes_after_recency_short_and_violation_weights():
    replay = _minimal_replay(viol_downweight=0.1, viol_uniform_mix=0.3)
    replay.recency_bias = 2.0
    replay.data[25]["hit"][:8, 0] = torch.arange(1, 9)
    replay.data[25]["active"][:8, 1] = False
    raw = replay._row_weights(
        25, 8, short_wp=3, short_boost=4.0, viol_downweight=0.1,
    )
    probs = _replay_sampling_probs(raw, 0.3)
    expected = 0.7 * raw / raw.sum() + 0.3 / 8
    assert torch.allclose(probs, expected)
    assert bool((probs > 0.0).all())
    expected_gen = torch.Generator(device="cpu").manual_seed(21)
    actual_gen = torch.Generator(device="cpu").manual_seed(21)
    expected_draw = torch.multinomial(
        expected, 64, replacement=True, generator=expected_gen,
    )
    actual_draw = replay.sample_idx(
        25, 64, generator=actual_gen,
        short_wp=3, short_boost=4.0,
        viol_downweight=0.1, viol_uniform_mix=0.3,
    )
    assert torch.equal(actual_draw, expected_draw)


def test_phase2_config_uniform_mix_default_and_range():
    assert Phase2Config().replay_viol_uniform_mix == 0.0
    assert Phase2Config(replay_viol_uniform_mix=1.0).replay_viol_uniform_mix == 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Phase2Config(replay_viol_uniform_mix=-0.01)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Phase2Config(replay_viol_uniform_mix=1.01)


def test_stored_score_telemetry_reports_ring_and_selected_quantiles():
    replay = _minimal_replay()
    selected = [replay.data[25]["viol_score"][torch.tensor([0, 4, 4])]]
    out = _replay_viol_score_telemetry(replay, selected)
    assert out["whole_ring"]["n"] == 8
    assert out["whole_ring"]["p50"] == pytest.approx(5.0)
    assert out["selected"]["n"] == 3
    assert out["selected"]["p50"] == pytest.approx(10.0)
