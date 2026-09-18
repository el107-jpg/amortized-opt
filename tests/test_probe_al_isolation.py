from types import SimpleNamespace

import torch

from amortized_opt.train_cost import (
    DIAG_KEYS,
    TERM_KEYS,
    VIOL_KEYS,
    _voxel_mask_al_weights,
    cost_tuple,
)


def test_probe_rows_get_zero_dual_and_fixed_penalty():
    weights = {
        "terminal": torch.tensor(7.0),
        "collision": torch.tensor(5.0),
        "pen_terminal": torch.tensor(11.0),
        "smooth": torch.tensor(3.0),
    }
    out = _voxel_mask_al_weights(
        weights, torch.tensor([True, False]), probe_penalty=2.0,
    )

    assert torch.equal(out["terminal"], torch.tensor([7.0, 0.0]))
    assert torch.equal(out["collision"], torch.tensor([5.0, 0.0]))
    assert torch.equal(out["pen_terminal"], torch.tensor([11.0, 2.0]))
    assert out["smooth"].ndim == 0 and float(out["smooth"]) == 3.0
    assert torch.equal(out["_al_ok"], torch.tensor([1.0, 0.0]))


def test_cost_tuple_violation_means_exclude_probe_rows():
    batch = 2
    terms = {name: torch.zeros(batch) for name in TERM_KEYS}
    violations = {
        name: torch.tensor([1.0, 101.0]) for name in VIOL_KEYS
    }
    diagnostics = {name: torch.zeros(batch) for name in DIAG_KEYS}
    diagnostics.update({
        "terminal_region_dist": torch.zeros(batch),
        "joint_box_excess_max": torch.zeros(batch),
        "collision_pen_max": torch.zeros(batch),
        "collision_min_sdf": torch.zeros(batch),
        "torque_excess_max": torch.zeros(batch),
    })
    ct = SimpleNamespace(
        total=torch.zeros(batch), terms=terms, violations=violations,
        diagnostics=diagnostics, aux_labels=None,
    )

    class FakeCost:
        def forward(self, *args, **kwargs):
            return ct

    dummy = torch.zeros(batch, 1)
    _total, _terms, viol, _diag, gate = cost_tuple(
        FakeCost(), dummy, dummy, dummy, None, None,
        {"_al_ok": torch.tensor([1.0, 0.0])},
    )

    assert torch.equal(viol, torch.ones(len(VIOL_KEYS)))
    assert gate.shape == (batch, 8)
