"""Frozen terminal-head integration with the trajectory optimizer."""

import torch

from amortized_opt.constants import N_DOF
from amortized_opt.model import AmortizedOptimizer, ModelConfig, model_config_from_checkpoint
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks, pack_state
from amortized_opt.train_cost import _apply_terminal_initializer


def _batch(B=3, H=40):
    ctx = ContextInputs.zeros(B, num_drums=7, num_future=3)
    ctx.drum_pos[..., 2] = 0.8
    ctx.drum_axis[..., 2] = 1.0
    ctx.drum_radius.fill_(0.15)
    ctx.drum_valid[:] = True
    ctx.goal_valid[:] = True
    ctx.goal_drum[:, 0] = 0
    ctx.goal_drum[:, 1] = 2
    ctx.goal_tth[:, 0] = 0.4
    ctx.goal_tth[:, 1] = 0.6
    masks = build_waypoint_masks(
        ctx.goal_tth[:, 0], ctx.goal_tth[:, 1], max_horizon=H,
        active_left=ctx.goal_valid[:, 0], active_right=ctx.goal_valid[:, 1],
    )
    q0 = torch.zeros(B, N_DOF)
    v0 = torch.linspace(-0.3, 0.4, N_DOF).expand(B, -1).clone()
    return ctx, pack_state(q0, v0), masks


def test_pretrained_terminal_head_is_frozen_and_seed_hits_prediction(tmp_path):
    model = AmortizedOptimizer(ModelConfig(
        max_horizon=40, num_drums=7, terminal_head=True,
        d_ctx=32, ctx_heads=4, d_model=32, dec_heads=4, dec_layers=1,
    ))
    ckpt = tmp_path / "terminal.pt"
    torch.save({
        "model": model.terminal_initializer.head.state_dict(),
        "optnet": False, "in_dim": 119, "width": 128, "depth": 2,
    }, ckpt)
    model.load_terminal_head(ckpt)
    ctx, state, masks = _batch()
    state.requires_grad_(True)
    terminal = model.terminal_initializer.terminal_pose(ctx, state, masks)
    seed = model.initial_trajectory(ctx, state, masks)
    assert not terminal.requires_grad
    assert not seed.requires_grad
    assert all(not p.requires_grad for p in model.terminal_initializer.parameters())
    lo = model.terminal_initializer.head.q_lo.view(1, 1, -1)
    hi = model.terminal_initializer.head.q_hi.view(1, 1, -1)
    violation = torch.maximum(lo - seed, seed - hi).clamp_min(0) * masks.elem_mask
    assert float(violation.max()) == 0.0
    # left hits at waypoint 20, right at waypoint 30 (rows are zero based).
    torch.testing.assert_close(seed[:, 19, 3:10], terminal[:, 3:10])
    torch.testing.assert_close(seed[:, 29, 10:17], terminal[:, 10:17])


def test_checkpoint_architecture_detection_finds_terminal_head():
    model = AmortizedOptimizer(ModelConfig(
        max_horizon=10, num_drums=7, terminal_head=True,
        d_ctx=32, ctx_heads=4, d_model=32, dec_heads=4, dec_layers=1,
    ))
    stored = {k: v for k, v in model.cfg.__dict__.items() if k != "terminal_head"}
    cfg = model_config_from_checkpoint({"model_cfg": stored, "model": model.state_dict()})
    assert cfg.terminal_head is True


def test_phase2_reseeds_only_fresh_rows():
    model = AmortizedOptimizer(ModelConfig(
        max_horizon=40, num_drums=7, terminal_head=True,
        d_ctx=32, ctx_heads=4, d_model=32, dec_heads=4, dec_layers=1,
    ))
    ctx, state, masks = _batch(B=2)
    old = torch.full((2, 40, N_DOF), 9.0)
    batch = {
        "context": ctx, "state": state, "masks": masks,
        "init": old.clone(), "mpc_init": torch.tensor([False, True]),
    }
    expected = model.initial_trajectory(ctx, state, masks)
    assert _apply_terminal_initializer(model, batch)
    torch.testing.assert_close(batch["init"][0], expected[0])
    torch.testing.assert_close(batch["init"][1], old[1])


def test_seed_from_terminal_keeps_the_supplied_pose():
    model = AmortizedOptimizer(ModelConfig(
        max_horizon=40, num_drums=7, terminal_head=True,
        d_ctx=32, ctx_heads=4, d_model=32, dec_heads=4, dec_layers=1,
    ))
    ctx, state, masks = _batch()
    terminal = torch.zeros(3, N_DOF)
    terminal[:, 3:10] = 0.2
    terminal[:, 10:17] = -0.15
    seed = model.terminal_initializer.seed_from_terminal(ctx, state, masks, terminal)
    torch.testing.assert_close(seed[:, 19, 3:10], terminal[:, 3:10])
    torch.testing.assert_close(seed[:, 29, 10:17], terminal[:, 10:17])
    moved = state.clone()
    moved[:, 4] += 0.25
    redrawn = model.terminal_initializer.seed_from_terminal(ctx, moved, masks, terminal)
    assert not torch.allclose(seed[:, 0], redrawn[:, 0])
    torch.testing.assert_close(redrawn[:, 19, 3:10], terminal[:, 3:10])


def test_phase2_terminal_noise_is_start_anchored_and_fresh_only():
    model = AmortizedOptimizer(ModelConfig(
        max_horizon=40, num_drums=7, terminal_head=True,
        d_ctx=32, ctx_heads=4, d_model=32, dec_heads=4, dec_layers=1,
    ))
    ctx, state, masks = _batch(B=2)
    old = torch.full((2, 40, N_DOF), 9.0)
    batch = {
        "context": ctx, "state": state, "masks": masks,
        "init": old.clone(), "mpc_init": torch.tensor([False, True]),
    }

    class _UnitNoisePool:
        def noised_init(self, seed, bucket, *, generator=None):
            assert bucket == 40
            return seed + 1.0

    clean = model.initial_trajectory(ctx, state, masks)
    assert _apply_terminal_initializer(
        model, batch, pool=_UnitNoisePool(), start_anchored_noise=True,
        noise_ramp_steps=5,
    )
    # smoothstep(1/5)=0.104; it reaches full amplitude at knot five.
    torch.testing.assert_close(
        batch["init"][0, 0], clean[0, 0] + 0.104,
        rtol=0.0, atol=1e-6,
    )
    torch.testing.assert_close(batch["init"][0, 4], clean[0, 4] + 1.0)
    torch.testing.assert_close(batch["init"][1], old[1])
