"""Exact vector-gradient attribution and replay probability diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from amortized_opt.torque_grad_attribution import (
    measure_batch_torque_gradient_attribution,
    replay_sampling_probabilities,
    summarize_replay_sampling,
)
from amortized_opt.train_cost import ReplayPool


def test_vector_attribution_keeps_batch_mean_scaling_and_cancellation():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
    calls = 0

    def fresh_forward():
        nonlocal calls
        calls += 1
        p0, p1 = parameter.unbind()
        rows = torch.stack((p0, -2.0 * p0, p1, 2.0 * p0 + p1))
        return rows, {"forward_violation": torch.arange(4.0)}

    report = measure_batch_torque_gradient_attribution(
        fresh_forward,
        [parameter],
        labels={"harvest_viol_score": torch.arange(4.0)},
        sampling={"inclusion_probability": torch.tensor([0.9, 0.8, 0.7, 0.6])},
        n_bins=2,
        top_fracs=(0.25,),
    )

    assert calls == 1
    assert report["batch_mean_denominator"] == 4
    total = report["actual_batch_gradient"]
    assert total["gradient_norm"] == pytest.approx(math.sqrt(5.0) / 4.0)

    quantiles = report["labels"]["forward_violation"]["quantiles"]
    bottom, top = quantiles
    assert bottom["gradient_norm"] == pytest.approx(0.25)
    assert bottom["signed_contribution_fraction"] == pytest.approx(-0.2)
    assert bottom["cancellation_ratio"] == pytest.approx(1.0 / 3.0)
    assert top["signed_contribution_fraction"] == pytest.approx(1.2)
    assert (
        report["labels"]["forward_violation"][
            "quantile_signed_contribution_fraction_sum"
        ]
        == pytest.approx(1.0)
    )
    assert bottom["sampling"]["inclusion_probability"]["mean"] == pytest.approx(0.85)

    counterfactual = report["labels"]["forward_violation"]["tails"][
        "top_25pct"
    ]["counterfactual_excluding"]
    assert counterfactual["mean_denominator"] == 3
    assert counterfactual["gradient_norm"] == pytest.approx(math.sqrt(2.0) / 3.0)
    assert counterfactual["cosine_with_total"] == pytest.approx(1.0 / math.sqrt(10.0))
    assert counterfactual["signed_contribution_fraction"] == pytest.approx(4.0 / 15.0)


def _minimal_replay() -> ReplayPool:
    replay = object.__new__(ReplayPool)
    replay.recency_bias = 0.0
    replay.viol_downweight = 1.0
    replay.capacity = 4
    replay.device = torch.device("cpu")
    replay.dtype = torch.float32
    replay.data = {
        25: {
            "viol_score": torch.tensor(
                [0.0, math.log(2.0), math.log(4.0), math.log(8.0)]
            ),
            "hit": torch.full((4, 2), 5),
            "active": torch.ones(4, 2, dtype=torch.bool),
            "ema_m": torch.tensor([0.8, 0.4, 0.5, 0.5]),
            "ema_n": torch.tensor([3, 3, 0, 0]),
        }
    }
    replay._n = {25: 4}
    replay._pos = {25: 4}
    replay._order = {25: torch.arange(4)}
    return replay


def test_replay_sampling_probabilities_match_weighted_draws():
    replay = _minimal_replay()
    diag = replay_sampling_probabilities(
        replay, 25, 10, viol_downweight=1.0,
    )
    expected_probability = torch.tensor([1.0, 0.5, 0.25, 0.125])
    expected_probability /= expected_probability.sum()

    assert diag["expected_draw_count"].sum() == pytest.approx(10.0)
    assert torch.allclose(
        diag["expected_draw_fraction"], expected_probability, atol=1e-7
    )
    assert torch.allclose(
        diag["inclusion_probability"],
        1.0 - (1.0 - expected_probability).pow(10),
        atol=1e-7,
    )


def test_replay_sampling_probabilities_include_stratified_quotas():
    replay = _minimal_replay()
    diag = replay_sampling_probabilities(
        replay,
        25,
        10,
        viol_downweight=0.0,
        stratify=True,
        frac_prof=0.3,
        frac_new=0.2,
        ema_nmin=2,
        ema_prof=0.7,
    )

    expected = torch.tensor([3.0 + 10.0 / 3.0, 5.0 / 3.0, 1.0, 1.0])
    assert torch.allclose(diag["expected_draw_count"], expected, atol=1e-6)
    assert [stage["name"] for stage in diag["stages"]] == [
        "proficient",
        "new",
        "flex",
    ]

    summary = summarize_replay_sampling(diag, torch.tensor([0, 2, 2]))
    assert summary["unique_selected_rows"] == 2
    assert summary["selected_batch_rows"]["raw_weight"]["mean"] == pytest.approx(1.0)


def test_uniform_mix_is_local_to_each_proficiency_stratum():
    replay = _minimal_replay()
    diag = replay_sampling_probabilities(
        replay,
        25,
        10,
        viol_downweight=0.0,
        viol_uniform_mix=1.0,
        stratify=True,
        frac_prof=0.3,
        frac_new=0.2,
        ema_nmin=2,
        ema_prof=0.7,
    )

    # 3 proficient draws from {0}; 2 new draws uniformly from {2,3};
    # 5 flex draws uniformly from trusted {0,1}.
    assert torch.allclose(
        diag["expected_draw_count"],
        torch.tensor([5.5, 2.5, 1.0, 1.0]),
    )
    assert diag["viol_sampling_beta"] == 0.0
    assert diag["viol_uniform_mix"] == 1.0
    assert all(stage["uniform_mix_gamma"] == 1.0 for stage in diag["stages"])

    first_gen = torch.Generator(device="cpu").manual_seed(13)
    second_gen = torch.Generator(device="cpu").manual_seed(13)
    kw = dict(
        stratify=True, frac_prof=0.3, frac_new=0.2,
        ema_nmin=2, ema_prof=0.7, viol_uniform_mix=1.0,
    )
    first = replay.sample_idx(25, 10, generator=first_gen, **kw)
    second = replay.sample_idx(25, 10, generator=second_gen, **kw)
    assert torch.equal(first, second)
    assert torch.equal(first[:3], torch.zeros(3, dtype=torch.long))
    assert bool(((first[3:5] == 2) | (first[3:5] == 3)).all())
    assert bool(((first[5:] == 0) | (first[5:] == 1)).all())
