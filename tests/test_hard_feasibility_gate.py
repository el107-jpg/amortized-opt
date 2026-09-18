from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from amortized_opt.train_cost import _hard_feasibility_ok_from_cost


def _ct(**updates):
    diagnostics = {
        "terminal_region_dist": torch.tensor([0.0, 0.0]),
        "collision_min_sdf": torch.tensor([0.01, 0.01]),
        "joint_box_excess_max": torch.tensor([0.0, 0.0]),
        "torque_excess_max": torch.tensor([0.0, 0.0]),
    }
    diagnostics.update(updates)
    return SimpleNamespace(diagnostics=diagnostics)


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("terminal_region_dist", 1.0e-3),
        ("collision_min_sdf", -1.0e-3),
        ("joint_box_excess_max", 1.0e-3),
        ("torque_excess_max", 1.0e-3),
    ],
)
def test_each_hard_constraint_can_reject_a_witness(name, bad):
    values = torch.tensor([0.0, bad])
    if name == "collision_min_sdf":
        values[0] = 0.01
    ok = _hard_feasibility_ok_from_cost(_ct(**{name: values}))
    assert ok.tolist() == [True, False]


def test_nonfinite_diagnostic_never_passes():
    ok = _hard_feasibility_ok_from_cost(
        _ct(torque_excess_max=torch.tensor([float("nan"), float("inf")]))
    )
    assert not bool(ok.any())


def test_gate_has_explicit_numerical_tolerances():
    ct = _ct(
        terminal_region_dist=torch.tensor([2.0e-5, 2.0e-5]),
        joint_box_excess_max=torch.tensor([2.0e-5, 2.0e-5]),
        torque_excess_max=torch.tensor([2.0e-5, 2.0e-5]),
    )
    assert not bool(_hard_feasibility_ok_from_cost(ct).any())
    assert bool(
        _hard_feasibility_ok_from_cost(
            ct, terminal_eps=3.0e-5, joint_eps=3.0e-5, torque_eps=3.0e-5,
        ).all()
    )


def test_missing_required_diagnostic_fails_loudly():
    ct = _ct()
    del ct.diagnostics["joint_box_excess_max"]
    with pytest.raises(KeyError, match="joint_box_excess_max"):
        _hard_feasibility_ok_from_cost(ct)

