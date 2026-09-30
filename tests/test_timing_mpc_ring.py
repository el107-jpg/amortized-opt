from dataclasses import dataclass
from types import SimpleNamespace

import torch

from amortized_opt.train_cost import (
    _TrajectorySuffixMPCParentRing,
    _VoxelMPCParentRing,
    effective_hits,
    terminal_mixture_suffix_offsets,
)


@dataclass
class _Masks:
    elem_mask: torch.Tensor


@dataclass
class _Context:
    token: torch.Tensor
    goal_tth: torch.Tensor
    yaw_now: torch.Tensor


@dataclass
class _Strike:
    hit_index: torch.Tensor
    waist_yaw_ref: torch.Tensor | None = None
    waist_pr_ref: torch.Tensor | None = None


def _batch(n=2, horizon=5, dof=2):
    q0 = torch.arange(n * dof, dtype=torch.float32).reshape(n, dof)
    return {
        "bucket": horizon,
        "q0": q0,
        "v0": -q0,
        "state": torch.cat([q0, -q0], dim=-1),
        "init": torch.zeros(n, horizon, dof),
        "seed": torch.zeros(n, horizon, dof),
        "masks": _Masks(torch.ones(n, horizon, dof)),
        "mpc_init": torch.zeros(n, dtype=torch.bool),
        "al_ok": torch.ones(n, dtype=torch.bool),
    }


def test_terminal_mixture_peaks_at_freeze_and_keeps_long_support():
    rows, horizon = 50_000, 50
    hit = torch.full((rows, 2), -1, dtype=torch.long)
    hit[:, 0] = horizon
    active = torch.zeros(rows, 2, dtype=torch.bool)
    active[:, 0] = True
    offsets = terminal_mixture_suffix_offsets(
        hit, active, horizon,
        generator=torch.Generator().manual_seed(17),
        peak_frac=0.55, long_frac=0.42, low_frac=0.03,
        peak_wp=5, tail_tau=8.0,
    )
    remaining = horizon - offsets
    hist = torch.bincount(remaining, minlength=horizon).float() / rows
    assert int(hist[1:].argmax()) + 1 == 5
    assert 0.03 < float(hist[1:5].sum()) < 0.06
    assert 0.06 < float(hist[5]) < 0.085
    assert 0.32 < float(hist[20:].sum()) < 0.39
    assert int(offsets.min()) >= 1


def test_critic_timing_ring_advances_state_timing_and_initialization():
    horizon, dof = 5, 2
    batch = _batch()
    batch["hit"] = torch.tensor([[4, 3], [5, -1]])
    batch["active"] = torch.tensor([[True, True], [True, False]])
    batch["context"] = _Context(
        token=torch.arange(2, dtype=torch.float32).unsqueeze(1),
        goal_tth=torch.tensor([[0.08, 0.06], [0.10, 0.0]]),
        yaw_now=torch.tensor([-1.0, -1.0]),
    )
    batch["strike"] = _Strike(
        hit_index=batch["hit"].clone(),
        waist_yaw_ref=torch.arange(2 * horizon, dtype=torch.float32).reshape(2, horizon),
    )
    traj = torch.arange(2 * horizon * dof, dtype=torch.float32).reshape(
        2, horizon, dof,
    )

    class _Spline:
        def __init__(self, parent):
            self.parent = parent

        def eval_offsets(self, _end):
            qd = (self.parent + 100.0).unsqueeze(2)
            z = torch.zeros_like(qd)
            return z, qd, z, z

    class _SplineFactory:
        def from_masks(self, _q0, parent, _v0, _mask):
            return _Spline(parent)

    class _Pool:
        q_lo = torch.full((dof,), -1_000.0)
        q_hi = torch.full((dof,), 1_000.0)
        cfg = SimpleNamespace(
            dt=0.02,
            joint_names=("waist_yaw_joint", "joint_1"),
        )

        def masks_for(self, hit, active, bucket):
            assert bucket == horizon
            return _Masks(torch.ones(active.shape[0], horizon, dof))

    cost = SimpleNamespace(spline=_SplineFactory(), dt=0.02)
    ring = _TrajectorySuffixMPCParentRing(
        device="cpu", pool=_Pool(), costs={horizon: cost},
        q_noise=0.0, qd_noise=0.0, capacity=3,
    )
    assert ring.refresh(
        horizon, batch, traj, hard_ok=torch.ones(2, dtype=torch.bool),
        generator=torch.Generator().manual_seed(7),
    ) == 2

    # Refresh stores one full parent, without committing to a suffix offset.
    stored = ring.data[horizon]
    torch.testing.assert_close(stored["_parent_traj"], traj)
    assert "seed" not in stored and "init" not in stored
    torch.testing.assert_close(stored["q0"], batch["q0"])
    torch.testing.assert_close(stored["hit"], batch["hit"])

    sampled = ring.batch(
        horizon, torch.arange(2),
        generator=torch.Generator().manual_seed(7),
    )
    offs = batch["hit"][:, 0] - sampled["hit"][:, 0]
    assert bool((offs >= 1).all())
    nearest = batch["hit"].masked_fill(~batch["active"], horizon).min(1).values
    assert bool((offs < nearest).all())
    km1 = (offs - 1).view(2, 1, 1).expand(2, 1, dof)
    expected_q0 = traj.gather(1, km1)[:, 0]
    torch.testing.assert_close(sampled["q0"], expected_q0)
    torch.testing.assert_close(sampled["v0"], expected_q0 + 100.0)
    torch.testing.assert_close(
        sampled["state"], torch.cat([expected_q0, expected_q0 + 100.0], dim=-1),
    )
    torch.testing.assert_close(sampled["context"].yaw_now, expected_q0[:, 0])
    torch.testing.assert_close(
        sampled["context"].goal_tth,
        torch.where(
            batch["active"], sampled["hit"].float() * 0.02,
            torch.zeros_like(sampled["hit"], dtype=torch.float32),
        ),
    )
    for row in range(2):
        k = int(offs[row])
        expected = torch.cat(
            [traj[row, k:], traj[row, -1:].expand(k, -1)], dim=0,
        )
        torch.testing.assert_close(sampled["init"][row], expected)
    torch.testing.assert_close(sampled["seed"], sampled["init"])
    torch.testing.assert_close(
        sampled["strike"].hit_index,
        effective_hits(sampled["hit"], batch["active"], horizon),
    )
    expected_ref = torch.stack([
        torch.cat([
            batch["strike"].waist_yaw_ref[row, int(offs[row]):],
            batch["strike"].waist_yaw_ref[row, -1:].expand(int(offs[row])),
        ])
        for row in range(2)
    ])
    torch.testing.assert_close(sampled["strike"].waist_yaw_ref, expected_ref)

    # Repeated draws of one stored parent independently choose offsets. The
    # parent itself remains unchanged, so no suffix is stored per waypoint.
    repeated = ring.batch(
        horizon, torch.zeros(64, dtype=torch.long),
        generator=torch.Generator().manual_seed(19),
    )
    repeated_offs = batch["hit"][0, 0] - repeated["hit"][:, 0]
    assert repeated_offs.unique().numel() > 1
    torch.testing.assert_close(ring.data[horizon]["_parent_traj"], traj)
    torch.testing.assert_close(ring.data[horizon]["hit"], batch["hit"])

    # New certified parents append to the bounded FIFO instead of replacing
    # the whole pool. With capacity 3, the oldest of four total rows drops.
    newer = traj + 1_000.0
    assert ring.refresh(
        horizon, batch, newer, hard_ok=torch.ones(2, dtype=torch.bool),
    ) == 2
    assert ring.size(horizon) == 3
    expected_parents = torch.cat([traj[1:], newer], dim=0)
    torch.testing.assert_close(
        ring.data[horizon]["_parent_traj"], expected_parents,
    )
    # A step with no new certified plans leaves reusable parents available.
    assert ring.refresh(
        horizon, batch, newer, hard_ok=torch.zeros(2, dtype=torch.bool),
    ) == 0
    assert ring.size(horizon) == 3


def test_critic_timing_ring_drops_nonadvancing_and_legacy_rows():
    batch = _batch(n=1, horizon=4)
    batch.update({
        "hit": torch.tensor([[1, -1]]),
        "active": torch.tensor([[True, False]]),
    })
    ring = object.__new__(_TrajectorySuffixMPCParentRing)
    ring.device = torch.device("cpu")
    ring.data = {}
    ring.generation = {}
    ring.pool = None
    ring.costs = {}
    ring.q_noise = 0.0
    ring.qd_noise = 0.0
    ring.last_advance_offset_mean = 0.0
    ring.last_advance_seconds_mean = 0.0
    ring.dropped_legacy_rows = 0
    assert ring.refresh(
        4, batch, torch.ones(1, 4, 2), hard_ok=torch.ones(1, dtype=torch.bool),
    ) == 0
    assert ring.size(4) == 0

    legacy_batch = _batch(n=1, horizon=4)
    legacy = _VoxelMPCParentRing(device="cpu")
    legacy.refresh(
        4, legacy_batch, torch.ones(1, 4, 2),
        hard_ok=torch.ones(1, dtype=torch.bool),
    )
    ring.load_state_dict(legacy.state_dict())
    assert len(ring) == 0
    assert ring.dropped_legacy_rows == 1
