"""CPU smoke / contract tests for amortized_opt.terminal_pose + its trainer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("amortized_opt.terminal_pose")

import amortized_opt.terminal_pose as TP  # noqa: E402
from amortized_opt.constants import PELVIS_IN_ENV, STICK_TIP_SIGMA  # noqa: E402
from amortized_opt.limits import urdf_position_limits  # noqa: E402
from amortized_opt.phase2_data import strike_geometry_table  # noqa: E402
from amortized_opt.sdf import segment_hit_region  # noqa: E402
from amortized_opt.strike import (  # noqa: E402
    CYMBAL_CONTACT_SIGMA_FALLBACK,
    strike_axial_limits,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_train_mod():
    import sys

    path = ROOT / "scripts" / "train_terminal_pose.py"
    name = "train_terminal_pose_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses looks up the defining module in sys.modules at class creation.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_net(**kw):
    return TP.TerminalPoseNet(**kw)


def _make_cost(**kw):
    return TP.TerminalPoseCost(TP.TerminalPoseConfig(), **kw)


def test_parameter_count_in_band():
    n = int(_make_net().n_parameters())
    assert 20_000 <= n <= 45_000, f"n_parameters={n}"


def test_sample_and_forward_q_in_urdf_box():
    rng = np.random.default_rng(0)
    batch = TP.sample_terminal_batch(4, rng=rng, device="cpu", dtype=torch.float32)
    net = _make_net()
    q = net(batch.features)
    assert q.shape == (4, 17)
    assert torch.isfinite(q).all()
    lo, hi = urdf_position_limits()
    lo_t = torch.as_tensor(lo, dtype=q.dtype)
    hi_t = torch.as_tensor(hi, dtype=q.dtype)
    assert (q >= lo_t - 1e-4).all()
    assert (q <= hi_t + 1e-4).all()


def test_v4_terminal_pose_contact_geometry_backpropagates():
    batch=TP.sample_terminal_batch(4,rng=np.random.default_rng(907),
                                   device='cpu',dtype=torch.float32)
    cost=TP.TerminalPoseCost(TP.TerminalPoseConfig(contact_scoring_mode='isaac_v4'),
                             device='cpu')
    q=batch.q_start.detach().clone().requires_grad_(True)
    result=cost(q,batch)
    assert torch.isfinite(result.total).all()
    result.total.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_v5_terminal_pose_capsule_contact_geometry_backpropagates():
    batch=TP.sample_terminal_batch(4,rng=np.random.default_rng(908),
                                   device='cpu',dtype=torch.float32)
    cost=TP.TerminalPoseCost(TP.TerminalPoseConfig(contact_scoring_mode='isaac_v5'),
                             device='cpu')
    q=batch.q_start.detach().clone().requires_grad_(True)
    result=cost(q,batch)
    assert torch.isfinite(result.total).all()
    result.total.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_features_include_start_state_and_time_to_hit():
    rng = np.random.default_rng(1)
    batch = TP.sample_terminal_batch(3, rng=rng, device="cpu", dtype=torch.float32)
    feat = batch.features
    assert feat.ndim == 2
    assert feat.shape[0] == 3
    # Prefer an explicit FEATURE_DIM if present; else feature_dim(D).
    feat_dim = getattr(TP, "FEATURE_DIM", None)
    if feat_dim is not None:
        assert feat.shape[-1] == int(feat_dim)
    elif hasattr(TP, "feature_dim"):
        d = batch.drum_pos.shape[1]
        assert feat.shape[-1] == int(TP.feature_dim(d))
    assert batch.q_start.shape == (3, 17)
    assert batch.qd_start.shape == (3, 17)
    assert batch.time_to_hit.shape == (3, 2)
    assert torch.isfinite(batch.q_start).all()
    assert torch.isfinite(batch.qd_start).all()
    assert torch.isfinite(batch.time_to_hit).all()


def test_collision_topk_sends_gradient_to_multiple_contacts():
    sdf = torch.tensor([[-0.020, -0.015, -0.001, 0.1]], requires_grad=True)
    loss = TP.aggregate_collision_violation(sdf, allow_m=0.005, topk=2).sum()
    loss.backward()
    assert sdf.grad is not None
    assert int((sdf.grad.abs() > 0).sum()) == 2


def test_start_and_velocity_aware_motion_terms_have_correct_minima():
    rng = np.random.default_rng(22)
    batch = TP.sample_terminal_batch(
        4, rng=rng, device="cpu", dtype=torch.float32, idle_frac=0.0,
    )
    cost = _make_cost(device="cpu")
    at_start = cost(batch.q_start, batch, None)
    joint_t = TP.joint_time_to_hit(batch.time_to_hit, batch.active)
    at_coast = cost(batch.q_start + batch.qd_start * joint_t, batch, None)
    assert torch.allclose(at_start.terms["start"], torch.zeros(4), atol=1e-8)
    assert torch.allclose(at_coast.terms["motion"], torch.zeros(4), atol=1e-8)


def test_features_change_with_start_position():
    rng = np.random.default_rng(23)
    batch = TP.sample_terminal_batch(2, rng=rng, device="cpu", dtype=torch.float32)
    shifted = TP.pack_features(
        batch.drum_pos, batch.drum_axis, batch.drum_radius,
        batch.drum_index, batch.active, batch.stick_axis, batch.stick_origin,
        batch.tip_sigma, batch.shaft_lo, batch.is_cymbal,
        batch.q_start + 0.1, batch.qd_start, batch.time_to_hit,
    )
    assert not torch.equal(batch.features, shifted)


def test_cost_backward_nonzero_grad():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    net = _make_net()
    cost = _make_cost()
    al = TP.TerminalPoseAL()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    nonzero = False
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        batch = TP.sample_terminal_batch(4, rng=rng, device="cpu", dtype=torch.float32)
        q = net(batch.features)
        out = cost(q, batch, al.cost_weights())
        out.total.mean().backward()
        for p in net.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if torch.isfinite(g).all() and float(g.abs().sum()) > 0.0:
                nonzero = True
                break
        if nonzero:
            break
        opt.step()
    assert nonzero, "expected a finite nonzero decoder gradient within a few steps"


def test_hit_residual_is_disk_distance():
    rng = np.random.default_rng(3)
    cost = _make_cost()
    batch = TP.sample_terminal_batch(4, rng=rng, device="cpu", dtype=torch.float32)
    net = _make_net()
    with torch.no_grad():
        q = net(batch.features)
        out = cost(q, batch)
    hit = out.violations["hit"]
    assert hit.shape == (4,)
    assert torch.isfinite(hit).all()
    assert (hit >= -1e-8).all()

    # Geometry convention: segment on the aim point -> dist ~ 0; 0.5 m off -> large.
    names = ["snare"]
    pos = torch.tensor([[[0.35, 0.0, 0.75]]], dtype=torch.float64)
    axis = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    radius = torch.tensor([[0.08]], dtype=torch.float64)
    pelvis = pos - torch.as_tensor(PELVIS_IN_ENV, dtype=pos.dtype).view(1, 1, 3)
    tab = strike_geometry_table(pelvis, axis, radius, names)
    ax_lo, ax_hi = strike_axial_limits()
    aim = tab["disk_center"][0, 0]
    n_out = tab["axis"][0, 0]
    a = aim.clone()
    dist0, _, _ = segment_hit_region(
        a, a,
        tab["disk_center"][0, 0],
        n_out,
        tab["hit_radius"][0, 0],
        torch.zeros(3, dtype=a.dtype),
        tab["hp_dir"][0, 0],
        tab["has_hp"][0, 0],
        ax_lo, ax_hi,
    )
    assert float(dist0) < 1e-5

    a_off = aim + torch.tensor([0.5, 0.0, 0.0], dtype=aim.dtype)
    dist1, _, _ = segment_hit_region(
        a_off, a_off,
        tab["disk_center"][0, 0],
        n_out,
        tab["hit_radius"][0, 0],
        torch.zeros(3, dtype=a.dtype),
        tab["hp_dir"][0, 0],
        tab["has_hp"][0, 0],
        ax_lo, ax_hi,
    )
    assert float(dist1) > 0.3


def test_cymbal_shaft_window_outer_15cm():
    assert float(STICK_TIP_SIGMA) == pytest.approx(0.20, abs=1e-9)
    assert float(CYMBAL_CONTACT_SIGMA_FALLBACK) == pytest.approx(0.05, abs=1e-9)

    from amortized_opt.fk import StickRandomization

    # Zero stick randomization -> handle-fixed tip_sigma == STICK_TIP_SIGMA.
    rng = np.random.default_rng(4)
    tip = shaft_lo = None
    for _ in range(40):
        batch = TP.sample_terminal_batch(
            16,
            rng=rng,
            device="cpu",
            dtype=torch.float32,
            stick_rand=StickRandomization(),
        )
        cym = batch.is_cymbal & batch.active
        if not bool(cym.any()):
            continue
        tip = batch.tip_sigma[cym]
        shaft_lo = batch.shaft_lo[cym]
        break
    assert tip is not None and shaft_lo is not None, "no cymbal hand in samples"

    assert torch.allclose(tip, tip.new_full(tip.shape, float(STICK_TIP_SIGMA)), atol=1e-6)
    assert torch.allclose(
        shaft_lo,
        shaft_lo.new_full(shaft_lo.shape, float(CYMBAL_CONTACT_SIGMA_FALLBACK)),
        atol=1e-6,
    )
    legal = tip - shaft_lo
    assert torch.allclose(legal, legal.new_full(legal.shape, 0.15), atol=1e-6)


def test_penalty_stays_at_the_trajectory_cap():
    al = TP.TerminalPoseAL()
    assert al.pen["hit"] == pytest.approx(1.0e3)
    assert al.pen["collision"] == pytest.approx(1.0e4)
    assert al.pen_max["hit"] == pytest.approx(1.0e4)
    assert al.pen_max["collision"] == pytest.approx(1.0e5)
    assert al.lam_max["hit"] == pytest.approx(100.0)
    chronic = {
        "hit": torch.full((8,), 0.01),
        "collision": torch.full((8,), 0.01),
    }
    for _ in range(200):
        al.update(chronic)
    assert al.pen["hit"] <= al.pen_max["hit"] + 1e-6
    assert al.pen["collision"] <= al.pen_max["collision"] + 1e-6
    assert al.pen["hit"] == pytest.approx(1.0e3)
    assert al.lam["hit"] <= 100.0 + 1e-6


def test_al_hit_grows_collision_does_not():
    al = TP.TerminalPoseAL()
    lam0_hit = float(al.lam["hit"])
    lam0_col = float(al.lam["collision"])
    fake = {
        "hit": torch.full((8,), 0.2),
        "collision": torch.zeros(8),
    }
    al.update(fake)
    assert al.lam["hit"] > lam0_hit
    assert al.lam["collision"] <= lam0_col + 1e-12


def test_right_hihat_is_sampled_and_left_floor_tom_is_excluded():
    rng = np.random.default_rng(11)
    batch = TP.sample_terminal_batch(400, rng=rng, device="cpu", dtype=torch.float32, idle_frac=0.0)
    names = list(batch.drum_names)
    floor = names.index("floor_tom")
    hihat = names.index("hihat")
    assert not bool((batch.drum_index[:, 0] == floor).any())
    assert bool((batch.drum_index[:, 1] == hihat).any())
    assert bool(batch.active.all())


def test_waist_yaw_is_normal_about_drum_average():
    cost = _make_cost(device="cpu")
    rng = np.random.default_rng(12)
    batch = TP.sample_terminal_batch(
        6, rng=rng, device="cpu", dtype=torch.float32, idle_frac=0.0,
    )
    mu = TP.target_waist_yaw(batch.drum_pos, batch.drum_index, batch.active)
    q = torch.zeros(6, 17)
    q[:, cost.yaw_i] = mu
    on = cost(q, batch, None)
    q_off = q.clone()
    q_off[:, cost.yaw_i] = mu + torch.deg2rad(torch.tensor(30.0))
    off = cost(q_off, batch, None)
    assert torch.allclose(on.terms["waist"], torch.zeros(6), atol=1e-5)
    # N(0, 30°): one standard deviation costs 1/2 under weight 1.
    assert torch.allclose(off.terms["waist"], torch.full((6,), 0.5), atol=1e-4)


def test_collision_scores_arms_and_cross_hand():
    cost = _make_cost(device="cpu")
    rng = np.random.default_rng(13)
    batch = TP.sample_terminal_batch(4, rng=rng, device="cpu", dtype=torch.float32)
    q = torch.zeros(4, 17, requires_grad=True)
    out = cost(q, batch, None)
    assert torch.isfinite(out.violations["collision"]).all()
    out.total.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_pad_elevation_floor_is_five_degrees():
    elev = torch.deg2rad(torch.tensor([0.0, 4.0, 5.0, 35.0, -10.0]))
    short = TP.pad_elevation_shortfall(elev, 5.0)
    assert float(torch.rad2deg(short[0])) == pytest.approx(5.0)
    assert float(torch.rad2deg(short[1])) == pytest.approx(1.0)
    assert float(short[2]) == pytest.approx(0.0, abs=1e-8)
    assert float(short[3]) == pytest.approx(0.0, abs=1e-8)
    assert float(torch.rad2deg(short[4])) == pytest.approx(15.0)


def test_collision_allows_5mm_penetration():
    sdf = torch.tensor([0.02, 0.0, -0.004, -0.005, -0.008])
    v = TP.collision_violation(sdf, 0.005)
    assert torch.allclose(v[:4], torch.zeros(4), atol=1e-8)
    assert float(v[4]) == pytest.approx(0.003)


def test_line_search_step_runs():
    from amortized_opt.terminal_pose_ls import (
        line_search_update, make_terminal_pose_line_search,
    )

    rng = np.random.default_rng(3)
    batch = TP.sample_terminal_batch(2, rng=rng, device="cpu", dtype=torch.float32)
    net = _make_net()
    cost = _make_cost(device="cpu")
    ls = make_terminal_pose_line_search(net)
    before = net.head.weight.detach().clone()
    info = line_search_update(ls, net, cost, batch, None, step=1, grad_clip=1.0)
    assert info["searched"] == 1.0
    assert info["epsilon"] > 0.0
    assert torch.isfinite(net.head.weight).all()
    assert not torch.equal(before, net.head.weight.detach())


def test_optnet_gpu_coordinate_equalities_match_cvxpy():
    """The unrefined batched factorization is wrong on A = [I 0]."""
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    from qpth.qp import QPFunction, QPSolvers

    from amortized_opt.terminal_pose_optnet import solve_feasible_qp

    torch.manual_seed(0)
    device = "cuda:0"
    B, n, neq = 8, 17, 4
    Q = (0.8 * torch.eye(n, device=device, dtype=torch.float64)).expand(B, n, n).contiguous()
    p = torch.randn(B, n, device=device, dtype=torch.float64, requires_grad=True)
    A = torch.zeros(B, neq, n, device=device, dtype=torch.float64)
    A[:, :, :neq] = torch.eye(neq, device=device, dtype=torch.float64)
    b = 0.2 * torch.randn(B, neq, device=device, dtype=torch.float64)
    G = torch.cat([torch.eye(n), -torch.eye(n)], 0).to(device=device, dtype=torch.float64)
    h = 2 * torch.ones(2 * n, device=device, dtype=torch.float64)
    lower = -1e6 * torch.ones_like(h)
    z = solve_feasible_qp(Q, p, A, b, G, lower, h)
    with torch.no_grad():
        z_cvx = QPFunction(verbose=-1, check_Q_spd=False, solver=QPSolvers.CVXPY)(
            Q, p.detach(), G, h, A, b,
        )
    assert (z - z_cvx).abs().max() < 1e-5
    (z[:, 6] ** 2).sum().backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def test_qp_out_is_the_solve():
    rng = np.random.default_rng(5)
    batch = TP.sample_terminal_batch(2, rng=rng, device="cpu", dtype=torch.float64)
    features = batch.features.double()
    net = TP.TerminalPoseNet(optnet=True, qp_out=True).double()
    q = net(features)
    h2 = torch.nn.functional.silu(net.enc(torch.nn.functional.silu(net.pre(features))))
    z = net.qp(h2, net.q_lo, net.q_hi, net.q_ready)
    assert q.shape == (2, 17)
    assert (q - net._soft_clip(z)).abs().max() < 1e-8
    assert not hasattr(net, "head")
    # Move the linear objective farther from initialization and verify that
    # gradients still cross the exact-solve output path into the encoder.
    net.qp.to_p.bias.data.fill_(0.1)
    q = net(features)
    q.sum().backward()
    assert net.qp.to_p.weight.grad.abs().sum() > 0
    assert net.enc.weight.grad.abs().sum() > 0


def test_qp_mix_starts_at_the_solve_and_skip_learns():
    rng = np.random.default_rng(5)
    batch = TP.sample_terminal_batch(2, rng=rng, device="cpu", dtype=torch.float64)
    features = batch.features.double()
    net = TP.TerminalPoseNet(optnet=True, mix_width=32).double()
    q = net(features)
    h2 = torch.nn.functional.silu(net.enc(torch.nn.functional.silu(net.pre(features))))
    z = net.qp(h2, net.q_lo, net.q_hi, net.q_ready)
    assert q.shape == (2, 17)
    assert (q - net._soft_clip(z)).abs().max() < 1e-8
    torch.nn.init.normal_(net.head.weight, std=0.02)
    q = net(features)
    q.sum().backward()
    assert torch.isfinite(net.qp.to_p.weight.grad).all()
    assert net.qp.to_p.weight.grad.abs().sum() > 0
    assert net.mix.weight.grad.abs().sum() > 0
    assert net.enc.weight.grad.abs().sum() > 0


def test_optnet_layer_learns_the_full_program():
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    batch = TP.sample_terminal_batch(32, rng=rng, device="cpu", dtype=torch.float64)
    net = TP.TerminalPoseNet(optnet=True).double()
    q = net(batch.features.double())
    assert q.shape == (32, 17)
    assert torch.isfinite(q).all()
    # A nonuniform upstream derivative avoids cancellations between joints.
    scale = torch.linspace(0.5, 1.5, q.shape[-1], dtype=q.dtype)
    (q.square().mean() + (q * scale).mean()).backward()
    for name in ("to_chol", "to_p", "to_A", "to_b", "to_G", "to_l", "to_u"):
        layer = getattr(net.qp, name)
        assert layer.weight.grad is not None, name
        assert torch.isfinite(layer.weight.grad).all(), name
        assert layer.weight.grad.abs().sum() > 0, name

    h = torch.nn.functional.silu(net.pre(batch.features.double()))
    _Q, _p, A, b, G, lower, upper, z0 = net.qp.qp_data(h)
    assert torch.all(lower < upper)
    assert torch.allclose(
        (A @ z0.unsqueeze(-1)).squeeze(-1), b, atol=1e-9, rtol=1e-9,
    )
    Gz0 = (G @ z0.unsqueeze(-1)).squeeze(-1)
    assert torch.all(Gz0 > lower)
    assert torch.all(Gz0 < upper)


def test_centered_qp_low_rank_factor_can_leave_initialization():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    batch = TP.sample_terminal_batch(8, rng=rng, device="cpu", dtype=torch.float64)
    net = TP.TerminalPoseNet(optnet=True, centered_qp=True).double()
    # Once the learned preferred pose differs from ready, curvature affects
    # the equality-constrained projection and its factor must receive signal.
    with torch.no_grad():
        net.qp.to_pref.bias.fill_(0.2)
    q = net(batch.features.double())
    q.square().mean().backward()
    grad = net.qp.to_factor.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_no_isaac_import_in_source():
    src = Path(TP.__file__).read_text()
    assert "import isaac" not in src


def test_line_search_train_steps(tmp_path):
    mod = _load_train_mod()
    out = tmp_path / "ls.pt"
    cfg = mod.TrainConfig(
        steps=2,
        batch=4,
        lr=1e-3,
        device="cpu",
        seed=0,
        log_every=1,
        al_every=1,
        grad_clip=1.0,
        line_search=True,
        start_bank_size=0,
        out=str(out),
    )
    ckpt = mod.train_terminal_pose(cfg)
    assert out.is_file()
    assert ckpt["step"] == 2
    assert "optimizer" in ckpt
    assert "rng_state" in ckpt
    assert ckpt["line_search"] is not None
    net = _make_net()
    net.load_state_dict(ckpt["model"])


def test_fixed_line_search_interval_implies_line_search():
    mod = _load_train_mod()
    cfg = mod.parse_args(["--line-search-interval", "25"])
    assert cfg.line_search
    assert cfg.line_search_interval == 25


def test_three_step_train_no_nan(tmp_path):
    mod = _load_train_mod()
    out = tmp_path / "last.pt"
    cfg = mod.TrainConfig(
        steps=3,
        batch=4,
        lr=1e-3,
        device="cpu",
        seed=0,
        log_every=1,
        al_every=1,
        grad_clip=1.0,
        start_bank_size=0,
        out=str(out),
    )
    ckpt = mod.train_terminal_pose(cfg)
    assert out.is_file()
    assert "model" in ckpt
    assert "loss_spec" in ckpt
    net = _make_net()
    net.load_state_dict(ckpt["model"])
    rng = np.random.default_rng(0)
    batch = TP.sample_terminal_batch(4, rng=rng, device="cpu", dtype=torch.float32)
    cost = _make_cost()
    with torch.no_grad():
        loss = cost(net(batch.features), batch).total
    assert torch.isfinite(loss).all()
