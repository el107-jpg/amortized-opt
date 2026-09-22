"""Focused CPU tests for the isolated optimized-teacher LoRA trainer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import numpy as np
import torch

from amortized_opt.cost import StrikeContext
from amortized_opt.model import AmortizedOptimizer, ModelConfig
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
from tools.strike_pose_editor.il_train import (
    ADAPTER_FORMAT,
    ILTrainConfig,
    TeacherBundle,
    collate_training_records,
    compare_saved_adapter,
    load_teacher_bundle,
    save_teacher_bundle,
    train_il_adapter,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strike_context(n: int) -> StrikeContext:
    zeros_2 = torch.zeros(n, 2)
    zeros_23 = torch.zeros(n, 2, 3)
    return StrikeContext(
        hit_index=torch.ones(n, 2, dtype=torch.long),
        active=torch.ones(n, 2),
        target_drum=torch.zeros(n, 2, dtype=torch.long),
        start_drum=torch.full((n, 2), -1, dtype=torch.long),
        p_star=zeros_23.clone(),
        v_star=zeros_23.clone(),
        n_out=zeros_23.clone(),
        hit_radius=zeros_2.clone(),
        disk_offset=zeros_23.clone(),
        half_plane_dir=zeros_23.clone(),
        has_half_plane=zeros_2.clone(),
        contact_sigma=zeros_2.clone(),
        shaft_allowed=zeros_2.clone(),
        start_shaft_allowed=zeros_2.clone(),
        elev_required=zeros_2.clone(),
    )


def _fixture(tmp_path: Path, *, n: int = 4, horizon: int = 6):
    torch.manual_seed(4)
    cfg = ModelConfig(
        max_horizon=horizon,
        num_drums=2,
        num_future=1,
        d_ctx=32,
        ctx_layers=1,
        ctx_heads=4,
        ctx_ffn_mult=2.0,
        d_model=32,
        dec_layers=1,
        dec_heads=4,
        dec_ffn_mult=2.0,
        idle_home_token=False,
        dec_tth_feats=True,
        enc_goal_tth=False,
        dec_plant_feats=False,
    )
    model = AmortizedOptimizer(cfg).eval()
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": model.state_dict(), "model_cfg": asdict(cfg), "step": 7,
            # Real phase-2 checkpoints include these otherwise unrelated
            # sampling records. The trainer must still load them safely.
            "extra": {
                "numpy_scalar": np.float64(1.25),
                "masks": build_waypoint_masks(
                    torch.tensor([horizon * cfg.dt]), torch.zeros(1),
                    max_horizon=horizon, dt=cfg.dt,
                    active_left=torch.ones(1, dtype=torch.bool),
                    active_right=torch.zeros(1, dtype=torch.bool),
                ),
                "context": ContextInputs.zeros(1, num_drums=2, num_future=1),
            },
        },
        checkpoint,
    )

    q0 = 0.02 * torch.randn(n, 17)
    qd0 = torch.zeros_like(q0)
    init = q0[:, None, :].expand(n, horizon, 17).contiguous()
    init = init + 0.01 * torch.randn_like(init)
    state = torch.cat([q0, qd0], dim=-1)
    context = ContextInputs.zeros(n, num_drums=2, num_future=1)
    context.drum_valid[:, 0] = True
    context.drum_axis[:, 0, 2] = 1.0
    context.drum_radius[:, 0] = 0.15
    context.drum_height[:, 0] = 0.05
    context.goal_drum[:, 0] = 0
    context.goal_pos[:, 0] = torch.tensor([0.45, 0.20, 0.25])
    context.goal_vel[:, 0] = torch.tensor([0.0, 0.0, -1.0])
    context.goal_valid[:, 0] = True
    context.goal_tth[:, 0] = horizon * cfg.dt
    hit_index = torch.tensor([[horizon, -1]]).expand(n, 2).clone()
    active = torch.tensor([[True, False]]).expand(n, 2).clone()
    masks = build_waypoint_masks(
        context.goal_tth[:, 0], torch.zeros(n), max_horizon=horizon, dt=cfg.dt,
        active_left=active[:, 0], active_right=active[:, 1],
    )
    with torch.no_grad():
        base = model(init * masks.elem_mask, state, model.encode_context(context), masks).traj
    teacher = base.clone()
    teacher[:, :, 3] += 0.08
    bundle = TeacherBundle(
        init=init,
        state=state,
        context=context,
        hit_index=hit_index,
        active=active,
        teacher_q=teacher,
        teacher_mask=masks.elem_mask,
        strike={"dummy": torch.arange(n).view(n, 1)},
        scenario_ids=tuple(f"optimized-{i}" for i in range(n)),
    )
    return checkpoint, bundle


def test_teacher_bundle_tensor_only_round_trip_and_validation(tmp_path):
    _checkpoint, bundle = _fixture(tmp_path)
    path = save_teacher_bundle(tmp_path / "teachers.pt", bundle)
    loaded = load_teacher_bundle(path)
    assert loaded.scenario_ids == bundle.scenario_ids
    assert torch.equal(loaded.init, bundle.init)
    assert torch.equal(loaded.teacher_q, bundle.teacher_q)
    assert torch.equal(loaded.strike["dummy"], bundle.strike["dummy"])
    assert set(loaded.payload()) >= {
        "format", "init", "state", "context", "hit_index", "active",
        "teacher_q", "teacher_mask", "strike", "scenario_ids",
    }

    records = []
    for i in range(bundle.size):
        one = bundle.index_select(torch.tensor([i]))
        records.append({
            "scenario_id": bundle.scenario_ids[i],
            "init": one.init,
            "state": one.state,
            "context": one.context,
            "strike": _strike_context(1),
            "hit_index": one.hit_index,
            "active": one.active,
            "teacher_q": one.teacher_q,
            "teacher_mask": one.teacher_mask,
        })
    collated = collate_training_records(records)
    assert collated.scenario_ids == bundle.scenario_ids
    assert torch.equal(collated.teacher_q, bundle.teacher_q)
    assert collated.strike is not None
    assert collated.strike["p_star"].shape == (bundle.size, 2, 3)

    with pytest.raises(ValueError, match="teacher_q"):
        TeacherBundle(
            init=bundle.init,
            state=bundle.state,
            context=bundle.context,
            hit_index=bundle.hit_index,
            active=bundle.active,
            teacher_q=bundle.teacher_q[:, :-1],
            teacher_mask=bundle.teacher_mask,
        )


def test_quick_cpu_training_is_isolated_and_improves_teacher_fit(tmp_path):
    checkpoint, bundle = _fixture(tmp_path)
    before = _hash(checkpoint)
    bundle_path = save_teacher_bundle(tmp_path / "teachers.pt", bundle)
    job = tmp_path / "session" / "jobs" / "quick_il"
    result = train_il_adapter(
        checkpoint,
        bundle_path,
        job,
        config=ILTrainConfig(
            steps=60,
            batch_size=bundle.size,
            lr=1e-2,
            teacher_weight=1.0,
            distill_weight=0.01,
            seed=3,
            device="cpu",
        ),
    )

    assert _hash(checkpoint) == before
    assert result.adapter_path == job / "adapter.pt"
    assert result.adapter_path.is_file()
    assert result.trainable_parameters > 0
    assert result.lora_paths
    assert all(path.endswith(("o_proj", "w12", "w3")) for path in result.lora_paths)
    assert result.comparison.teacher_rmse_adapted_rad \
        < result.comparison.teacher_rmse_base_rad

    raw = torch.load(result.adapter_path, map_location="cpu", weights_only=True)
    assert raw["format"] == ADAPTER_FORMAT
    assert "model" not in raw and "model_cfg" not in raw and "optimizer" not in raw
    assert raw["actor"]
    assert all(key.endswith((".lora_a", ".lora_b")) for key in raw["actor"])
    assert raw["cfg"]["actor"]["lora_rank"] == 4
    assert raw["cfg"]["actor"]["lora_alpha"] == 8.0

    status = json.loads((job / "status.json").read_text())
    assert status["status"] == "complete" and status["step"] == 60
    assert len((job / "metrics.jsonl").read_text().splitlines()) == 60
    predictions = torch.load(job / "predictions.pt", weights_only=True)
    assert set(predictions["base"]) == set(bundle.scenario_ids)
    assert set(predictions["adapted"]) == set(bundle.scenario_ids)
    assert predictions["base"][bundle.scenario_ids[0]].shape == bundle.teacher_q.shape[1:]
    assert predictions["adapted"][bundle.scenario_ids[0]].shape == bundle.teacher_q.shape[1:]

    replay = compare_saved_adapter(checkpoint, result.adapter_path, bundle)
    torch.testing.assert_close(replay.base, result.comparison.base)
    torch.testing.assert_close(replay.adapted, result.comparison.adapted)
    assert replay.teacher_rmse_adapted_rad == pytest.approx(
        result.comparison.teacher_rmse_adapted_rad
    )


def test_physics_callback_is_explicit_and_output_path_is_confined(tmp_path):
    checkpoint, bundle = _fixture(tmp_path, n=2)
    with pytest.raises(ValueError, match="physics_loss"):
        train_il_adapter(
            checkpoint,
            bundle,
            tmp_path / "missing_physics",
            config=ILTrainConfig(steps=1, physics_weight=1.0),
        )
    with pytest.raises(ValueError, match="source checkpoint directory"):
        train_il_adapter(
            checkpoint,
            bundle,
            checkpoint.parent,
            config=ILTrainConfig(steps=1),
        )

    calls = []

    def physics_loss(trajectory, batch, masks):
        calls.append((trajectory.shape, batch.size, masks.elem_mask.shape))
        return 0.001 * trajectory.square().mean()

    result = train_il_adapter(
        checkpoint,
        bundle,
        tmp_path / "with_physics",
        config=ILTrainConfig(steps=2, batch_size=2, physics_weight=0.2),
        physics_loss=physics_loss,
    )
    assert result.adapter_path.is_file()
    assert len(calls) == 2
    assert all(row["physics_loss"] >= 0.0 for row in result.history)
