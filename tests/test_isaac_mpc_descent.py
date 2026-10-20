from types import SimpleNamespace

import torch

from isaac_rl.mpc import IsaacMPC, MPCConfig


class _QuadraticCost:
    def __init__(self, target):
        self.target = target

    def __call__(self, _q, traj, _qd, _strike, **_kwargs):
        target = torch.as_tensor(self.target, device=traj.device, dtype=traj.dtype)
        return SimpleNamespace(total=(traj - target).square().flatten(1).sum(1))


def _mpc(target=0.0):
    mpc = object.__new__(IsaacMPC)
    mpc.cfg = MPCConfig(
        descent_accept=True,
        descent_alphas=(1.0, 0.5, 0.25, 0.125),
        descent_min_abs=1e-6,
        descent_min_rel=0.0,
    )
    mpc.descent_cost = _QuadraticCost(target)
    mpc.sched = SimpleNamespace(
        strike_context=lambda: SimpleNamespace(
            hit_index=torch.ones(1, 2, dtype=torch.long)
        )
    )
    mpc._descent_alphas = mpc.cfg.descent_alphas
    mpc.descent_trials = 0
    mpc.descent_rejects = 0
    mpc.last_descent_alpha = torch.ones(1)
    return mpc


def _call(mpc, init, proposal):
    init = torch.full((1, 2, 1), float(init))
    proposal = torch.full_like(init, float(proposal))
    q = torch.zeros(1, 1)
    qd = torch.zeros_like(q)
    return mpc._descent_accept(init, proposal, q, qd, torch.ones_like(init))


def test_accepts_full_decoder_step_when_it_decreases_cost():
    accepted, diag = _call(_mpc(target=0.0), init=1.0, proposal=0.0)
    assert torch.equal(accepted, torch.zeros_like(accepted))
    assert diag["descent_alpha"].item() == 1.0
    assert diag["descent_accepted_cost"].item() < diag["descent_base_cost"].item()


def test_backtracks_to_largest_decreasing_blend():
    # J(0.5)=0.5.  Full -> -2 and half -> -0.75 are worse; quarter -> -0.125
    # is the first decreasing candidate.
    accepted, diag = _call(_mpc(target=0.0), init=0.5, proposal=-2.0)
    assert diag["descent_alpha"].item() == 0.25
    assert torch.allclose(accepted, torch.full_like(accepted, -0.125))


def test_rejects_every_step_when_warm_start_is_already_optimal():
    mpc = _mpc(target=0.0)
    accepted, diag = _call(mpc, init=0.0, proposal=1.0)
    assert torch.equal(accepted, torch.zeros_like(accepted))
    assert diag["descent_alpha"].item() == 0.0
    assert mpc.descent_rejects == 1
