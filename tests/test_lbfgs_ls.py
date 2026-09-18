"""L-BFGS line-search adapter smoke tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from amortized_opt.lbfgs_ls import LbfgsLineSearchAdapter
from amortized_opt.line_search import LineSearchConfig
from amortized_opt.line_search_step import init_line_search_runtime


def test_lbfgs_quadratic_descent():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.zero_()
    opt = AdamW(model.parameters(), lr=1.0e-3)
    adapter = LbfgsLineSearchAdapter(opt, eta_ref=1.0e-3, history_size=5)

    target = torch.tensor([[3.0, -2.0]])
    x = torch.tensor([[1.0, 1.0]])

    loss0 = None
    for step in range(8):
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = (y - target).pow(2).mean()
        if loss0 is None:
            loss0 = float(loss.item())
        loss.backward()
        adapter.snapshot_theta0()
        g_old = [p.grad.detach().clone() for p in adapter._params]
        st = adapter.stage_from_q(use_param_grad_as_q=True)
        assert st.stats.probe_valid
        eps = 0.5
        adapter.write_candidate(eps, st.p_exec, into_params=True)
        adapter.commit(st, params_already_written=True)
        opt.zero_grad(set_to_none=True)
        y2 = model(x)
        loss2 = (y2 - target).pow(2).mean()
        loss2.backward()
        g_new = [p.grad.detach().clone() for p in adapter._params]
        s = adapter.displacement_from_theta0()
        y = [(gn - go) for go, gn in zip(g_old, g_new)]
        adapter.push_pair(s, y)

    loss_f = float(((model(x) - target).pow(2).mean()).item())
    assert loss_f < loss0


def test_lbfgs_rejects_q_direction_uphill_for_raw_objective():
    p = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    opt = AdamW([p], lr=1e-3, weight_decay=0.0)
    p.grad = torch.tensor([-1.0], dtype=p.dtype)  # transformed optimizer q
    adapter = LbfgsLineSearchAdapter(opt, eta_ref=1e-3)
    adapter.snapshot_theta0()
    staged = adapter.stage_from_q(
        g_objective=[torch.tensor([1.0], dtype=p.dtype)],
    )
    assert staged.stats.s_probe > 0.0
    assert staged.stats.s_probe_q < 0.0
    assert staged.stats.a < 0.0
    assert not staged.stats.probe_valid


def test_lbfgs_torque_cap_requires_replayable_history_transform():
    model = nn.Linear(1, 1)
    opt = AdamW(model.parameters(), lr=1e-3)
    phase = SimpleNamespace(
        lr=1e-3,
        optimizer="lbfgs",
        lbfgs_history=5,
        torque_grad_cap=0.2,
        grad_clip=1.0,
        cost=SimpleNamespace(torque_physics="full"),
        line_search_cfg=LineSearchConfig(),
    )
    with pytest.raises(ValueError, match="does not support torque_grad_cap"):
        init_line_search_runtime(phase2_cfg=phase, optimizer=opt)
