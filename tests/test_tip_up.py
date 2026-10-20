"""Downward stick-tip ReLU, exempt on the snare and floor tom."""

from __future__ import annotations

import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.train_cost import DIAG_KEYS, DIAG_NAMES, TERM_KEYS

DT = 0.02
H = 12
JOINT = {name: i for i, name in enumerate(G1_UPPER_BODY_JOINTS)}


def _ctx(drums: tuple[int, int]) -> StrikeContext:
    dtype = torch.float64
    return StrikeContext(
        hit_index=torch.full((1, 2), H, dtype=torch.long),
        active=torch.ones(1, 2, dtype=dtype),
        target_drum=torch.tensor([list(drums)], dtype=torch.long),
        start_drum=torch.full((1, 2), -1, dtype=torch.long),
        p_star=torch.full((1, 2, 3), 0.30, dtype=dtype),
        v_star=torch.zeros(1, 2, 3, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=dtype).view(1, 1, 3).expand(1, 2, 3).contiguous(),
        hit_radius=torch.full((1, 2), 0.08, dtype=dtype),
        disk_offset=torch.zeros(1, 2, 3, dtype=dtype),
        half_plane_dir=torch.zeros(1, 2, 3, dtype=dtype),
        has_half_plane=torch.zeros(1, 2, dtype=dtype),
        contact_sigma=torch.full((1, 2), 0.20, dtype=dtype),
        shaft_allowed=torch.zeros(1, 2, dtype=dtype),
        start_shaft_allowed=torch.zeros(1, 2, dtype=dtype),
        elev_required=torch.ones(1, 2, dtype=dtype),
    )


def _cost(**kw) -> FusedCost:
    kw.setdefault("w_tip_up", 10.0)
    return FusedCost(
        CostConfig(**kw),
        max_horizon=H, device="cpu", dtype=torch.float64, dt=DT,
    )


def _downward_pose(cost: FusedCost) -> torch.Tensor:
    q = torch.as_tensor(ready_joint_pos(G1_UPPER_BODY_JOINTS), dtype=torch.float64).view(1, -1)
    best, best_z = q, -1.0
    for delta in (-1.2, -0.6, 0.0, 0.6, 1.2):
        trial = q.clone()
        trial[0, JOINT["left_shoulder_pitch_joint"]] += delta
        trial[0, JOINT["right_shoulder_pitch_joint"]] += delta
        wp = trial.unsqueeze(1).repeat(1, H, 1)
        tom = cost.drum_index["tom1"]
        out = cost.forward(trial, wp, torch.zeros_like(trial), _ctx((tom, tom)))
        z = float(out.raw["tip_up"])
        if z > best_z:
            best, best_z = trial, z
    if best_z <= 1e-4:
        raise AssertionError("no shoulder pitch put the stick tip below horizontal")
    return best


def test_metric_names_include_the_tip_up_term():
    assert TERM_KEYS[-1] == "tip_up"
    assert DIAG_NAMES[DIAG_KEYS.index("tip_up_down")] == "d_tip_up"


def test_downward_tip_is_penalized_except_on_snare_and_floor_tom():
    cost = _cost()
    snare = cost.drum_index["snare"]
    floor = cost.drum_index["floor_tom"]
    tom = cost.drum_index["tom1"]
    q = _downward_pose(cost)
    wp = q.unsqueeze(1).repeat(1, H, 1)
    qd = torch.zeros_like(q)

    high = cost.forward(q, wp, qd, _ctx((tom, tom)))
    low = cost.forward(q, wp, qd, _ctx((snare, floor)))
    assert float(high.raw["tip_up"]) > 1e-3
    assert float(low.raw["tip_up"]) == 0.0
    assert float(high.terms["tip_up"]) == 10.0 * float(high.raw["tip_up"])

    off = _cost(w_tip_up=0.0)
    silent = off.forward(q, wp, qd, _ctx((tom, tom)))
    assert float(silent.terms["tip_up"]) == 0.0
    assert float(silent.diagnostics["tip_up_down"]) > 1e-3
