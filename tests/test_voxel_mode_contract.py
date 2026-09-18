"""Voxel mode owns its context and torso paths structurally."""

from dataclasses import dataclass

import pytest
import torch

import amortized_opt as A
from amortized_opt.tokenizer import ContextInputs
from amortized_opt.train_cost import (
    Phase2Config,
    _voxel_current_goal_only,
)
from scripts.train_phase2 import (
    _configure_voxel_model,
    _validate_voxel_cli,
    parse_args,
)


@dataclass
class _StrikeRefs:
    waist_yaw_ref: torch.Tensor | None
    waist_pr_ref: torch.Tensor | None


def _context() -> ContextInputs:
    b, d, f = 2, 7, 1
    return ContextInputs(
        drum_pos=torch.zeros(b, d, 3),
        drum_axis=torch.zeros(b, d, 3),
        drum_radius=torch.ones(b, d),
        drum_height=torch.ones(b, d),
        drum_valid=torch.ones(b, d, dtype=torch.bool),
        goal_drum=torch.zeros(b, 2, dtype=torch.long),
        goal_pos=torch.zeros(b, 2, 3),
        goal_vel=torch.zeros(b, 2, 3),
        goal_tth=torch.ones(b, 2),
        goal_valid=torch.ones(b, 2, dtype=torch.bool),
        future_drum=torch.full((b, 2, f), 3, dtype=torch.long),
        future_pos=torch.ones(b, 2, f, 3),
        future_vel=torch.ones(b, 2, f, 3),
        future_gap=torch.ones(b, 2, f),
        future_valid=torch.ones(b, 2, f, dtype=torch.bool),
        aim_yaw=torch.ones(b),
        aim_valid=torch.ones(b, dtype=torch.bool),
    )


def test_voxel_batch_masks_future_and_removes_all_waist_refs():
    ctx = _context()
    active = torch.tensor([[True, False], [False, False]])
    ctx.idle_valid[:] = False
    ctx.idle_cyl[:] = 1.0
    batch = {
        "context": ctx,
        "strike": _StrikeRefs(torch.ones(2, 25), torch.ones(2, 25, 2)),
        "active": active,
        "kind": torch.tensor([1, 2]),
    }
    _voxel_current_goal_only(batch)
    ctx = batch["context"]
    assert not ctx.future_valid.any()
    assert not ctx.future_pos.any()
    assert not ctx.future_vel.any()
    assert not ctx.future_gap.any()
    assert not ctx.aim_valid.any()
    assert torch.equal(ctx.goal_valid, active)
    assert torch.equal(ctx.idle_valid, ~active)
    assert not bool(ctx.goal_pos[~active].any())
    assert not bool(ctx.goal_vel[~active].any())
    assert not bool(ctx.goal_tth[~active].any())
    assert not bool(ctx.idle_cyl[active].any())
    assert not bool(batch["kind"].any())
    assert batch["strike"].waist_yaw_ref is None
    assert batch["strike"].waist_pr_ref is None


def test_old_checkpoint_architecture_is_stripped_before_voxel_model_build():
    cfg = A.ModelConfig(
        d_model=64, d_ctx=64, dec_layers=1, ctx_layers=1,
        dec_heads=4, ctx_heads=4, waist_sp_head=True,
        waist_pr_sp_head=True,
    )
    assert _configure_voxel_model(cfg, enabled=True)
    model = A.build_model(cfg, "cpu")
    assert not model.cfg.waist_sp_head
    assert not model.cfg.waist_pr_sp_head
    assert not any("waist_sp_" in name for name, _ in model.named_parameters())
    assert not any("waist_pr_sp_" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"waist_sp_from_head": True},
        {"waist_pr_sp_from_head": True},
        {"w_waist_sp_tether": 1.0},
        {"w_waist_pr_sp_hold": 1.0},
        {"w_waist_sp_smooth": 1.0},
        {"waist_pr_ema": True},
    ],
)
def test_voxel_config_rejects_setpoint_and_reference_couplings(kwargs):
    with pytest.raises(ValueError, match="decoder-owned torso"):
        Phase2Config(voxel_pool=True, **kwargs)


def test_voxel_cli_defaults_are_current_goal_only_and_head_free():
    args = parse_args(["--voxel-pool", "--num-future", "1"])
    _validate_voxel_cli(args)
    assert args.voxel_pool
    assert not args.waist_sp_head
    assert not args.waist_pr_sp_head
    assert not args.waist_pr_ema
    assert args.voxel_frac_mpc == pytest.approx(0.50)
    assert args.voxel_frac_pool == pytest.approx(0.30)
    assert args.voxel_frac_boundary == pytest.approx(0.15)
    assert args.voxel_frac_probe == pytest.approx(0.05)
    assert args.voxel_probe_capacity == 256
    assert args.voxel_probe_refresh_every == 10
    assert args.voxel_probe_shorter_layers == 1
    assert args.voxel_probe_longer_layers == 4
    assert args.voxel_probe_timing_step == 1
    assert args.voxel_probe_timing_frac == pytest.approx(0.5)
    assert args.voxel_bootstrap_stride == 4
    assert args.voxel_posthit_anchor_tth_s == pytest.approx(1.0)
    assert args.voxel_rest_gap_s == pytest.approx(1.0)
    assert args.voxel_strike_speed_min == pytest.approx(1.0)
    assert args.voxel_strike_speed_max == pytest.approx(3.0)


@pytest.mark.parametrize(
    "option",
    ["--waist-sp-head", "--waist-pr-sp-head", "--waist-pr-ema", "--line-search"],
)
def test_voxel_cli_rejects_legacy_couplings(option):
    args = parse_args(["--voxel-pool", option])
    with pytest.raises(SystemExit, match="incompatible options"):
        _validate_voxel_cli(args)
