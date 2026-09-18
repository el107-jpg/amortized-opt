"""``velocity_peak`` -- smooth-max ReLU priced, hard max for AL."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import CONSTRAINT_GROUPS
from amortized_opt.train_cost import phase2_al_config, phase2_cost_config

DT = 0.02
H = 12


def _ctx(B, *, kl, kr):
    ones = torch.ones(B, 2, dtype=torch.float64)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=ones.clone(),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=torch.float64),
        v_star=torch.full((B, 2, 3), -1.0, dtype=torch.float64),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=torch.float64),
        disk_offset=torch.zeros(B, 2, 3, dtype=torch.float64),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=torch.float64),
        has_half_plane=torch.zeros(B, 2, dtype=torch.float64),
        contact_sigma=torch.full((B, 2), 0.20, dtype=torch.float64),
        shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        start_shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        elev_required=ones.clone(),
    )


def test_group_and_phase2_defaults():
    assert "velocity_peak" in CONSTRAINT_GROUPS
    assert CONSTRAINT_GROUPS[-1] == "velocity_peak"
    cfg = phase2_al_config()
    assert cfg.target["velocity_peak"] == 0.0
    assert phase2_cost_config().w_velocity_peak == pytest.approx(2.0)


# left_wrist_pitch: hard limit 22 rad/s — dump a jump that exceeds it.
_WPITCH = 8


def _fast_wp(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    # Δq ≈ 2 rad / 0.02 s ≈ 100 rad/s ≫ 22
    wp[:, :, _WPITCH] = torch.linspace(0, 20.0, H).view(1, H).expand(B, H)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    return q0, wp, v0


def test_zero_weight_is_noop_on_peak_raw():
    cost = FusedCost(CostConfig(w_velocity_peak=0.0), max_horizon=H,
                     device="cpu", dtype=torch.float64, dt=DT)
    B = 2
    q0, wp, v0 = _fast_wp(B)
    kl = torch.full((B,), H, dtype=torch.long)
    out = cost.forward(q0, wp, v0, _ctx(B, kl=kl, kr=kl))
    assert float(out.raw["velocity_peak"].abs().max()) == 0.0
    assert out.raw["velocity_peak"].requires_grad is False


def test_both_velocity_weights_off_skips_locator():
    """w_velocity = w_velocity_peak = 0: no priced term, no segment live."""
    cost = FusedCost(CostConfig(w_velocity=0.0, w_velocity_peak=0.0),
                     max_horizon=H, device="cpu", dtype=torch.float64, dt=DT)
    B = 2
    q0, wp, v0 = _fast_wp(B)
    kl = torch.full((B,), H, dtype=torch.long)
    out = cost.forward(q0, wp, v0, _ctx(B, kl=kl, kr=kl))
    assert float(out.terms["velocity"].abs().max()) == 0.0
    assert float(out.raw["velocity"].abs().max()) == 0.0
    assert float(out.violations["velocity"].abs().max()) == 0.0
    assert float(out.diagnostics["velocity_max"].abs().max()) == 0.0
    assert out.raw["velocity"].requires_grad is False
    assert out.terms["velocity"].requires_grad is False
    for k, v in out.violations.items():
        if k.startswith("velocity"):
            assert float(v.abs().max()) == 0.0, k

def test_smooth_peak_ge_max_and_scales_term():
    cost = FusedCost(CostConfig(w_velocity_peak=1.0, w_velocity=0.0),
                     max_horizon=H, device="cpu", dtype=torch.float64, dt=DT)
    B = 1
    q0, wp, v0 = _fast_wp(B)
    kl = torch.full((B,), H, dtype=torch.long)
    out = cost.forward(q0, wp, v0, _ctx(B, kl=kl, kr=kl))
    assert float(out.diagnostics["velocity_excess_max"].max()) > 0.0
    assert float(out.raw["velocity_peak"].max()) >= float(
        out.violations["velocity_peak"].max()
    ) - 1e-12
    assert torch.equal(
        out.violations["velocity_peak"], out.diagnostics["velocity_excess_max"]
    )
    # Peak product is λ×w×smooth_max; segment λ×live still rides ``terms["velocity"]``.
    peak = cost.cfg.w_velocity_peak * out.raw["velocity_peak"]
    assert float((out.terms["velocity"] - peak).min()) >= -1e-12
    hot = cost.forward(
        q0, wp, v0, _ctx(B, kl=kl, kr=kl),
        weights={"velocity_peak": 3.0},
    )
    assert torch.allclose(
        hot.terms["velocity"] - out.terms["velocity"],
        2.0 * peak,
        atol=1e-12,
    )
