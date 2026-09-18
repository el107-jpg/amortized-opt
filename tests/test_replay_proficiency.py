"""Replay ring EMA proficiency and stratified sampling."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.train_cost import ReplayPool, _proficiency_ok_from_cost


def _minimal_replay(
    *, proficiency: bool = True, ema_alpha: float = 0.25,
    device: str | torch.device = "cpu",
) -> ReplayPool:
    device = torch.device(device)
    replay = object.__new__(ReplayPool)
    replay.recency_bias = 0.0
    replay.viol_downweight = 0.0
    replay.proficiency = proficiency
    replay.ema_alpha = ema_alpha
    replay.capacity = 16
    replay.device = device
    replay.dtype = torch.float32
    m = 8
    d = {
        "hit": torch.full((16, 2), 5, device=device),
        "active": torch.ones(16, 2, dtype=torch.bool, device=device),
    }
    if proficiency:
        d["ema_m"] = torch.tensor(
            [0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2] + [0.5] * 8,
            device=device,
        )
        d["ema_n"] = torch.tensor(
            [5, 5, 5, 5, 5, 5, 5, 5] + [0] * 8, device=device,
        )
    replay.data = {25: d}
    replay._n = {25: m}
    replay._pos = {25: m}
    replay._order = {25: torch.arange(16, device=device)}
    return replay


def test_replay_ema_update():
    replay = _minimal_replay()
    before = float(replay.data[25]["ema_m"][0])
    replay.ema_update(25, torch.tensor([0]), torch.tensor([1.0]))
    after = float(replay.data[25]["ema_m"][0])
    assert abs(after - (0.75 * before + 0.25)) < 1e-5
    assert int(replay.data[25]["ema_n"][0]) == 6


def test_replay_ema_repeated_index_uses_mean_label_and_full_count():
    replay = _minimal_replay()
    before_m = replay.data[25]["ema_m"].clone()
    before_n = replay.data[25]["ema_n"].clone()

    replay.ema_update(
        25, torch.tensor([2, 2, 2, 2]), torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )

    expected = 0.75 * before_m[2] + 0.25 * 0.5
    assert replay.data[25]["ema_m"][2] == pytest.approx(float(expected))
    assert int(replay.data[25]["ema_n"][2]) == int(before_n[2]) + 4
    keep = torch.arange(16) != 2
    assert torch.equal(replay.data[25]["ema_m"][keep], before_m[keep])
    assert torch.equal(replay.data[25]["ema_n"][keep], before_n[keep])


def test_replay_ema_duplicate_aggregation_is_permutation_invariant():
    first = _minimal_replay()
    second = _minimal_replay()
    idx = torch.tensor([3, 1, 3, 6, 1, 3, 6, 1])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    permutation = torch.tensor([7, 3, 0, 5, 2, 6, 4, 1])

    first.ema_update(25, idx, labels)
    second.ema_update(25, idx[permutation], labels[permutation])

    assert torch.equal(first.data[25]["ema_m"], second.data[25]["ema_m"])
    assert torch.equal(first.data[25]["ema_n"], second.data[25]["ema_n"])


def test_replay_ema_unique_indices_match_per_row_update():
    replay = _minimal_replay(ema_alpha=0.4)
    before_m = replay.data[25]["ema_m"].clone()
    before_n = replay.data[25]["ema_n"].clone()
    idx = torch.tensor([6, 0, 4, 2])
    labels = torch.tensor([0.0, 1.0, 0.25, 0.75])

    replay.ema_update(25, idx, labels)

    expected_m = before_m.clone()
    expected_n = before_n.clone()
    expected_m[idx] = 0.6 * before_m[idx] + 0.4 * labels
    expected_n[idx] += 1
    assert torch.equal(replay.data[25]["ema_m"], expected_m)
    assert torch.equal(replay.data[25]["ema_n"], expected_n)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_replay_ema_keeps_storage_device_and_accepts_cpu_inputs(device):
    replay = _minimal_replay(device=device)
    replay.ema_update(
        25, torch.tensor([0, 0, 5]), torch.tensor([1.0, 0.0, 1.0]),
    )

    assert replay.data[25]["ema_m"].device.type == device
    assert replay.data[25]["ema_n"].device.type == device
    assert replay.data[25]["ema_m"][0].item() == pytest.approx(
        0.75 * 0.9 + 0.25 * 0.5,
    )
    assert replay.data[25]["ema_n"][0].item() == 7
    assert replay.data[25]["ema_n"][5].item() == 6


def test_replay_add_resets_ema():
    replay = _minimal_replay()
    replay._pos[25] = 0
    slot = 0
    replay.data[25]["ema_m"][slot] = 0.99
    replay.data[25]["ema_n"][slot] = 9
    cols = {
        "q0": torch.zeros(1, 29),
        "qd0": torch.zeros(1, 29),
        "hit": torch.full((1, 2), 3),
        "active": torch.ones(1, 2, dtype=torch.bool),
        "init": torch.zeros(1, 25, 29),
        "drum": torch.zeros(1, 2, dtype=torch.long),
        "drum_pos": torch.zeros(1, 2, 3),
        "drum_radius": torch.ones(1, 2),
        "future_gap": torch.zeros(1, 2, 2),
        "future_drum": torch.zeros(1, 2, 2, dtype=torch.long),
        "start_drum": torch.zeros(1, 2, dtype=torch.long),
    }
    replay.add(25, cols)
    assert float(replay.data[25]["ema_m"][slot]) == 0.5
    assert int(replay.data[25]["ema_n"][slot]) == 0


def test_replay_stratified_prefers_proficient():
    replay = _minimal_replay()
    replay.data[25]["ema_n"][4:8] = 0
    replay.data[25]["ema_m"][4:8] = 0.5
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    picks = replay.sample_idx(
        25, 512, generator=gen, stratify=True,
        frac_prof=0.5, frac_new=0.25, ema_nmin=2, ema_prof=0.70,
    )
    assert picks is not None
    prof = int((picks < 4).sum())
    new = int(((picks >= 4) & (picks < 8)).sum())
    assert prof > 200
    assert new > 50


def test_replay_stratify_off_matches_uniform():
    replay = _minimal_replay()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1)
    picks = replay.sample_idx(25, 64, generator=gen, stratify=False)
    assert picks is not None
    assert int(picks.min()) >= 0 and int(picks.max()) < 8


def test_proficiency_ok_includes_torque_cap():
    class _Ct:
        violations = {"torque_peak": torch.tensor([1.0, 12.0])}
        diagnostics = {
            "terminal_region_dist": torch.tensor([0.01, 0.01]),
            "collision_min_sdf": torch.tensor([0.0, 0.0]),
            "torque_excess_max": torch.tensor([1.0, 12.0]),
        }

    ok = _proficiency_ok_from_cost(_Ct(), viol_cap=8.0)
    assert bool(ok[0])
    assert not bool(ok[1])
