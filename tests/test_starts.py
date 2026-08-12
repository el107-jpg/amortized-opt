"""Start-state sampler: validity, mix, eligibility, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt import constants as C
from amortized_opt.ik import tip_jacobian
from amortized_opt.starts import KIND_DRUM, KIND_NEUTRAL, StartConfig, StartSampler

N_SAMPLES = 1000


@pytest.fixture(scope="module")
def sampler():
    return StartSampler()


@pytest.fixture(scope="module")
def batch(sampler):
    return sampler.sample(N_SAMPLES, seed=0)


def test_no_collisions_and_no_joint_limit_violations(sampler, batch, capsys):
    v = sampler.violations(batch)
    with capsys.disabled():
        print(f"\n[starts, n={N_SAMPLES}] joint-box excess max={v['joint_box'].max():.2e} rad, "
              f"velocity excess max={v['velocity'].max():.2e} rad/s, "
              f"min clearance={v['min_sdf'].min():.2e} m "
              f"(median {np.median(v['min_sdf']):.3f})")
    assert v["joint_box"].max() == 0.0
    assert v["velocity"].max() == 0.0
    assert (v["min_sdf"] >= 0.0).all()


def test_drum_proximal_fraction(sampler, batch, capsys):
    frac = float((batch.kind == KIND_DRUM).double().mean())
    tries = batch.tries.double()
    with capsys.disabled():
        print(f"\n[starts] drum-proximal fraction={frac:.3f} "
              f"(requested {sampler.cfg.drum_fraction}); "
              f"draws per accepted sample mean={float(tries.mean()):.2f} max={int(tries.max())}")
    assert frac >= 0.85
    # ...and the rest really are neutral starts, not a silent third category.
    assert set(batch.kind.tolist()) <= {KIND_DRUM, KIND_NEUTRAL}


def test_both_hands_are_placed_independently(batch, capsys):
    on = batch.drum[batch.kind == KIND_DRUM]
    assert bool((on >= 0).all()), "a drum start must place BOTH hands"
    same = float((on[:, 0] == on[:, 1]).double().mean())
    pairs = {(int(a), int(b)) for a, b in on.tolist()}
    with capsys.disabled():
        print(f"\n[starts] distinct (left, right) drum pairs={len(pairs)}; "
              f"same-drum fraction={same:.3f}")
    assert len(pairs) >= 20  # 6 x 6 eligible combinations, minus rejections


def test_hand_eligibility_is_respected(sampler, batch):
    """No-crossover rule: left never plays the min-y drum, right never the max-y."""
    elig = {h: set(sampler.kit.eligible(h)) for h in ("left", "right")}
    lib = sampler.library
    bad = []
    for i in range(len(batch)):
        for hi, hand in enumerate(("left", "right")):
            d = int(batch.drum[i, hi])
            if d < 0:
                continue
            if d not in elig[hand] or not lib.has(hand, sampler.kit.names[d]):
                bad.append((i, hand, sampler.kit.names[d]))
    assert not bad, bad[:5]
    # The rule must actually exclude something on this kit.
    assert len(elig["left"]) == len(elig["right"]) == sampler.kit.num_drums - 1
    assert elig["left"] != elig["right"]


def test_departure_velocity_leaves_the_head(sampler, batch, capsys):
    """The tip velocity must point AWAY from the drum, within the noise cone."""
    m = (batch.kind == KIND_DRUM).nonzero().flatten()[:200]
    q, qd = batch.q[m], batch.qd[m]
    stats = {}
    for hi, hand in enumerate(("left", "right")):
        J = tip_jacobian(sampler.fk, q, hand)
        v = torch.einsum("bij,bj->bi", J, qd)
        axis = torch.as_tensor(
            sampler.kit.axes[batch.drum[m, hi].cpu().numpy()], dtype=q.dtype
        )
        speed = v.norm(dim=-1)
        cos = (v * axis).sum(-1) / speed.clamp_min(1e-9)
        stats[hand] = (float(speed.mean()), float(cos.mean()), float((cos > 0).double().mean()))
    with capsys.disabled():
        for h, (s, c, f) in stats.items():
            print(f"\n[starts departure {h}] mean tip speed={s:.2f} m/s, "
                  f"mean cos(v, head normal)={c:.3f}, outward fraction={f:.3f}")
    for _h, (s, c, f) in stats.items():
        assert 0.2 <= s <= 2.1
        assert c > 0.8 and f > 0.99


def test_neutral_starts_sit_near_the_ready_pose(batch):
    m = batch.kind == KIND_NEUTRAL
    if not bool(m.any()):
        pytest.skip("no neutral samples in this draw")
    ready = torch.as_tensor(C.ready_joint_pos(), dtype=batch.q.dtype)
    d = (batch.q[m] - ready).abs().max(dim=-1).values
    assert float(d.max()) < 1.0


def test_seeded_batches_are_identical(sampler):
    a = sampler.sample(256, seed=42)
    b = sampler.sample(256, seed=42)
    c = sampler.sample(256, seed=43)
    assert torch.equal(a.q, b.q) and torch.equal(a.qd, b.qd)
    assert torch.equal(a.drum, b.drum) and torch.equal(a.kind, b.kind)
    assert not torch.equal(a.q, c.q)


def test_drum_fraction_zero_gives_only_neutral_starts():
    s = StartSampler(config=StartConfig(drum_fraction=0.0))
    b = s.sample(64, seed=0)
    assert bool((b.kind == KIND_NEUTRAL).all())
    assert bool((b.drum < 0).all())
