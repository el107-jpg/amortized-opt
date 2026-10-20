"""BPTT baseline and line-search probe must price identical sampled chains."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from amortized_opt import train_cost as tc
from amortized_opt.objective_replay import ReplayIncompleteError, ReplayRecord


@dataclass
class _Strike:
    hit_index: torch.Tensor
    waist_yaw_ref: torch.Tensor | None
    n_out: torch.Tensor


class _Pool:

    def size(self, bucket):
        return 10

    def batch(self, bucket, idx, *, generator, noise):
        n = idx.numel()
        q0 = torch.rand(n, tc.N_DOF, generator=generator)
        hit = torch.full((n, 2), 2, dtype=torch.long)
        active = torch.ones(n, 2, dtype=torch.bool)
        return {
            "q0": q0, "v0": q0 * .1,
            "init": q0[:, None].expand(-1, bucket, -1).clone(),
            "state": q0, "context": q0, "masks": None,
            "hit": hit, "active": active,
            "strike": _Strike(hit, q0[:, 0], torch.ones(n, 2, 3)),
            "seed": q0[:, None].expand(-1, bucket, -1).clone(),
        }


class _Sim:
    def sample_gains(self, n, *, generator):
        kp = torch.rand(n, tc.N_DOF, generator=generator)
        kd = torch.rand(n, tc.N_DOF, generator=generator)
        return kp, kd, None


def test_bptt_replay_uses_frozen_batches_gains_and_bounce(monkeypatch):
    monkeypatch.setattr(tc, "encode", lambda _model, context: context)
    monkeypatch.setattr(tc, "per_element_lengths", lambda hit, names: torch.full(
        (hit.shape[0], tc.N_DOF), 2, dtype=torch.long))
    pool = _Pool()
    pool.cfg = SimpleNamespace(joint_names=tuple(f"joint{i}" for i in range(tc.N_DOF)))
    cfg = SimpleNamespace(bptt_batch=4, bounce=SimpleNamespace(
        restitution=(.2, .8), q_noise=.02, qd_noise=.05, tip_vel_noise=.03))
    def bptt_fn(init1, state1, q0, v0, enc1, masks1, s1, len1, hit1,
                term2, enc2, masks2, s2, len2, elen2, w, kp, kd,
                hand, normal, restitution, noise_q, noise_v, noise_tip):
        value = (q0.mean() + term2.mean() + enc2.mean() + kp.mean() + kd.mean()
                 + restitution.mean() + noise_q.mean() + noise_v.mean()
                 + noise_tip.mean() + w["anchor"])
        return value, value

    gen = torch.Generator().manual_seed(74)
    rec = ReplayRecord()
    args = (pool, SimpleNamespace(data={}), bptt_fn, _Sim(), cfg, 4, "cpu", torch.float32,
            gen, None, {"anchor": torch.tensor(1.)})
    baseline = tc._bptt_term(*args, model=None, replay_record=rec)
    assert rec.bptt is not None
    assert rec.samples.bptt_ki is None
    before = gen.get_state().clone()
    replayed = tc._bptt_term(*args, model=None, replay_record=rec, replay_only=True)
    assert torch.equal(before, gen.get_state())
    assert torch.equal(baseline, replayed)
    assert not torch.equal(baseline, tc._bptt_term(*args, model=None))

    rec.samples.bptt_kp = None
    with pytest.raises(ReplayIncompleteError, match="gains and bounce noise"):
        tc._bptt_term(*args, model=None, replay_record=rec, replay_only=True)
