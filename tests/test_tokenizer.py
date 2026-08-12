"""Tokenizer: joint grouping, the terminal-anchored Fourier code, per-element
horizon masks, and the context payloads."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.tokenizer import (
    DRUM_FEAT_DIM,
    FK_POINT_NAMES,
    GOAL_RAW_DIM,
    ContextInputs,
    build_waypoint_masks,
    drum_features,
    drum_tensors_from_scene,
    fourier_dim,
    fourier_frequencies,
    fourier_time_features,
    goal_features,
    joint_group_masks,
    pack_state,
    waypoint_times,
    zero_phase_features,
)

H = 100


def _freqs(bands=8):
    return torch.as_tensor(fourier_frequencies(bands, 0.02, 4.0), dtype=torch.float64)


# ------------------------------------------------------------------ groups --
def test_joint_groups_match_the_vendored_ordering():
    g = joint_group_masks()
    assert g["waist"].sum() == 3 and g["left"].sum() == 7 and g["right"].sum() == 7
    assert list(np.flatnonzero(g["waist"])) == [0, 1, 2]
    assert list(np.flatnonzero(g["left"])) == list(range(3, 10))
    assert list(np.flatnonzero(g["right"])) == list(range(10, 17))
    assert len(G1_UPPER_BODY_JOINTS) == N_DOF == 17


# ----------------------------------------------------------------- fourier --
def test_zero_phase_at_the_hit():
    f = _freqs()
    t = torch.zeros(5, dtype=torch.float64)
    enc = fourier_time_features(t, f, 4.0)
    ref = zero_phase_features(8, dtype=torch.float64)
    assert enc.shape == (5, fourier_dim(8))
    assert torch.equal(enc, ref.expand_as(enc))


def test_encoding_tracks_time_not_index():
    """Shift the hit time by one waypoint: the code shifts with the horizon."""
    f = _freqs()
    hit = torch.tensor([1.0], dtype=torch.float64)
    m0 = build_waypoint_masks(hit, hit, max_horizon=H, dt=CONTROL_DT)
    m1 = build_waypoint_masks(hit + CONTROL_DT, hit + CONTROL_DT, max_horizon=H, dt=CONTROL_DT)
    e0 = fourier_time_features(m0.tau_left, f, 4.0)
    e1 = fourier_time_features(m1.tau_left, f, 4.0)
    # token j of the shifted problem == token j-1 of the original
    assert torch.allclose(e1[0, 1:], e0[0, :-1], atol=1e-12)
    # the hit waypoint carries the zero-phase code in both
    k0 = int(round(1.0 / CONTROL_DT)) - 1
    ref = zero_phase_features(8, dtype=torch.float64)
    assert torch.allclose(e0[0, k0], ref, atol=1e-12)
    assert torch.allclose(e1[0, k0 + 1], ref, atol=1e-12)
    assert not torch.allclose(e1[0, k0], ref, atol=1e-6)


def test_encoding_is_signed_around_the_hit():
    f = _freqs()
    hit = torch.tensor([0.5], dtype=torch.float64)
    m = build_waypoint_masks(hit, hit, max_horizon=H, dt=CONTROL_DT)
    enc = fourier_time_features(m.tau_left, f, 4.0)
    k = int(round(0.5 / CONTROL_DT)) - 1
    assert enc[0, k - 1, -1] > 0 > enc[0, k + 1, -1]  # linear ramp changes sign


# ------------------------------------------------------------------- masks --
def test_per_element_mask_unequal_horizons():
    dt = CONTROL_DT
    t_l = torch.tensor([1.0])  # 50 waypoints
    t_r = torch.tensor([0.4])  # 20 waypoints
    m = build_waypoint_masks(t_l, t_r, max_horizon=H, dt=dt)
    e = m.elem_mask[0]
    times = waypoint_times(H, dt)
    for j, t in enumerate(times.tolist()):
        want_l = float(t <= 1.0 + 1e-6)
        want_r = float(t <= 0.4 + 1e-6)
        assert e[j, 3:10].tolist() == [want_l] * 7, j
        assert e[j, 10:17].tolist() == [want_r] * 7, j
        assert e[j, 0:3].tolist() == [max(want_l, want_r)] * 3, j
    assert m.token_mask[0].sum().item() == 50
    assert m.elem_mask[0, :, 3:10].sum().item() == 50 * 7
    assert m.elem_mask[0, :, 10:17].sum().item() == 20 * 7
    # FK points follow their own hand
    assert m.point_mask[0, 30, [0, 1, 2]].tolist() == [1.0, 1.0, 1.0]  # left
    assert m.point_mask[0, 30, [3, 4, 5]].tolist() == [0.0, 0.0, 0.0]  # right
    assert m.tau0_left.item() == pytest.approx(1.0)
    assert m.tau0_right.item() == pytest.approx(0.4)


def test_token_mask_equals_the_waist_row():
    m = build_waypoint_masks(
        torch.tensor([0.3, 1.2]), torch.tensor([0.9, 0.1]), max_horizon=H, dt=CONTROL_DT
    )
    assert torch.equal(m.token_mask.float(), m.elem_mask[..., 0])


def test_inactive_hand_borrows_the_other_horizon():
    t = torch.tensor([0.6])
    m = build_waypoint_masks(
        t,
        torch.tensor([0.0]),
        max_horizon=H,
        dt=CONTROL_DT,
        active_left=torch.tensor([True]),
        active_right=torch.tensor([False]),
    )
    assert m.elem_mask[0, :, 10:17].sum().item() == 30 * 7  # right follows left
    assert m.tau0_right.item() == pytest.approx(0.6)
    both_off = build_waypoint_masks(
        t,
        t,
        max_horizon=H,
        dt=CONTROL_DT,
        active_left=torch.tensor([False]),
        active_right=torch.tensor([False]),
    )
    assert both_off.token_mask.all()  # the full grid stays live


def test_horizon_beyond_the_grid_saturates():
    m = build_waypoint_masks(torch.tensor([9.0]), torch.tensor([9.0]), max_horizon=H, dt=CONTROL_DT)
    assert m.elem_mask.min().item() == 1.0


def test_pack_state():
    q = torch.randn(4, N_DOF)
    qd = torch.randn(4, N_DOF)
    assert pack_state(q, qd).shape == (4, 2 * N_DOF)


# ----------------------------------------------------------------- context --
def test_context_payload_shapes_and_zeroing():
    ctx = ContextInputs.zeros(3, num_drums=8, num_future=3)
    ctx.drum_pos.uniform_(-1, 1)
    ctx.drum_axis[..., 2] = 1.0
    ctx.drum_radius.fill_(0.15)
    ctx.drum_height.fill_(0.1)
    ctx.drum_valid[:, :7] = True
    ctx.goal_valid[:] = True
    ctx.goal_pos.uniform_(-1, 1)
    ctx.goal_vel.uniform_(-3, 3)
    ctx.future_valid[:, :, 0] = True

    df = drum_features(ctx)
    assert df.shape == (3, 8, DRUM_FEAT_DIM)
    assert torch.equal(df[:, 7], torch.zeros_like(df[:, 7]))  # padded drum zeroed

    gf = goal_features(ctx, ctx.goal_pos, ctx.goal_vel, ctx.goal_drum, ctx.goal_valid)
    assert gf.shape == (3, 2, GOAL_RAW_DIM)
    ff = goal_features(ctx, ctx.future_pos, ctx.future_vel, ctx.future_drum, ctx.future_valid)
    assert ff.shape == (3, 2, 3, GOAL_RAW_DIM)
    assert torch.equal(ff[:, :, 1:], torch.zeros_like(ff[:, :, 1:]))  # invalid rows zeroed
    # the gathered geometry is the target drum's row
    # layout: pos(3) rel(3) vel(3) speed(1) geom(8) -> geom.pos at 10:13
    assert torch.allclose(gf[:, :, 10:13], ctx.drum_pos[:, 0].unsqueeze(1).expand(3, 2, 3))
    assert torch.allclose(gf[:, :, 3:6], ctx.goal_pos - ctx.drum_pos[:, 0].unsqueeze(1))
    assert len(FK_POINT_NAMES) == 6


def test_drum_tensors_from_scene_pad_and_order():
    from amortized_opt.sdf import DrumSceneSpec

    scene = DrumSceneSpec.from_layout(frame="pelvis")
    d = drum_tensors_from_scene(scene, batch=2, num_drums=8)
    assert d["drum_pos"].shape == (2, 8, 3)
    assert d["drum_valid"][0].tolist() == [True] * scene.num_drums + [False] * (
        8 - scene.num_drums
    )
    assert np.allclose(d["drum_pos"][0, : scene.num_drums].numpy(), scene.head_centers, atol=1e-6)
