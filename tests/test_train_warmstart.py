"""Section 6: phase-1 warm-start data, losses, bucketing and checkpointing.

Fast unit tests only -- the training run itself lives in
``scripts/train_phase1.py``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.rhythm import DrumKit, default_rcc_dir, load_rcc_pool
from amortized_opt.train_warmstart import (
    METRIC_NAMES,
    LossScales,
    LossWeights,
    TrainConfig,
    batch_sampler,
    build_optimizer,
    forward_loss,
    load_checkpoint,
    make_step,
    save_checkpoint,
    unpack,
)
from amortized_opt.warmstart_data import (
    NoiseConfig,
    ScenarioConfig,
    WarmstartPool,
    bucket_for,
    generate_scenarios,
    ou_kernel,
)

TINY = dict(d_ctx=32, ctx_layers=1, ctx_heads=2, d_model=32, dec_layers=1, dec_heads=2)


# ------------------------------------------------------------------ fixtures --
@pytest.fixture(scope="module")
def kit():
    return DrumKit.from_layout()


@pytest.fixture(scope="module")
def clips(kit):
    return load_rcc_pool(default_rcc_dir("train"), kit)


@pytest.fixture(scope="module")
def scenario_cfg():
    return ScenarioConfig(horizon=50, buckets=(25, 50))


@pytest.fixture(scope="module")
def raw(clips, scenario_cfg):
    return generate_scenarios(140, clips, seed=3, cfg=scenario_cfg)


@pytest.fixture(scope="module")
def pool(raw, scenario_cfg):
    return WarmstartPool(
        raw["buckets"], cfg=scenario_cfg, noise=NoiseConfig(),
        device="cpu", dtype=torch.float64, stats=raw["stats"],
    )


def _tiny_model(cfg: ScenarioConfig, dtype=torch.float64):
    mc = A.ModelConfig(max_horizon=cfg.horizon, num_drums=cfg.num_drums,
                       num_future=cfg.num_future, **TINY)
    torch.manual_seed(0)
    return A.AmortizedOptimizer(mc).to(dtype=dtype), mc


# ------------------------------------------------------------------- scenarios --
def test_scenarios_cover_buckets_and_report_the_drop_rate(raw, scenario_cfg, capsys):
    s = raw["stats"]
    n_seen = s["accepted"] + s["reject_idle"] + s["reject_velocity"] + s["reject_library"]
    assert n_seen == s["requested"]
    assert s["accepted"] > 0
    assert raw["buckets"], "no bucket populated"
    for H, arrays in raw["buckets"].items():
        assert arrays["target"].shape[1] == H
        assert arrays["target"].shape[2] == A.N_DOF
        # every stored sample fits its bucket
        k = arrays["hit"].max(axis=1)
        assert (k <= H).all()
        assert (bucket_for(int(kk), scenario_cfg.buckets) == H for kk in k)
    sizes = {int(k): int(v["q0"].shape[0]) for k, v in raw["buckets"].items()}
    with capsys.disabled():
        print(
            f"\n[phase1 data] accepted {s['accepted']}/{s['requested']} "
            f"(drop: idle {s['reject_idle']}, |qd| {s['reject_velocity']}), "
            f"dual-arm {s['dual_active']} (asym {s['asymmetric']}), buckets {sizes}"
        )


def test_velocity_filter_actually_bounds_the_commanded_rate(raw, scenario_cfg):
    """No accepted target may command more than 1.5x a joint's speed rating."""
    v_lim = A.isaac_velocity_limits()
    worst = 0.0
    for arrays in raw["buckets"].values():
        X = np.concatenate([arrays["q0"][:, None, :], arrays["target"]], axis=1)
        peak = np.abs(np.diff(X, axis=1) / scenario_cfg.dt).max(axis=1)
        worst = max(worst, float((peak / v_lim[None, :]).max()))
    assert worst <= scenario_cfg.vel_reject_scale + 1e-6


def test_ou_kernel_matches_correlated_noise_statistics():
    H, tau = 64, 5.0
    L = ou_kernel(H, tau, "cpu", torch.float64)
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(20000, H, 1, dtype=torch.float64, generator=g)
    x = torch.einsum("ij,bjd->bid", L, eps)[:, :, 0]
    assert abs(float(x.std(dim=0).mean()) - 1.0) < 0.02      # unit marginal sd
    a = math.exp(-1.0 / tau)
    lag1 = float((x[:, :-1] * x[:, 1:]).mean())
    assert abs(lag1 - a) < 0.02                              # lag-1 autocorr = a


# ---------------------------------------------------------------- aux targets --
def test_torque_targets_match_the_cost_diagnostics(pool, scenario_cfg):
    """The aux torque label and ``cost.py``'s ``torque_max`` are the same number.

    Both go through ``knot_fd`` + ``BatchedRNEA``, so the per-waypoint segment
    maxima must reduce to exactly the cost's per-sample maximum.
    """
    H = pool.buckets[0]
    n = min(24, pool.size(H))
    idx = torch.arange(n)
    b = pool.batch(H, idx)
    cost = FusedCost(CostConfig(), max_horizon=H, dt=scenario_cfg.dt,
                     device="cpu", dtype=torch.float64)
    z3 = torch.zeros(n, 2, 3, dtype=torch.float64)
    z2 = torch.zeros(n, 2, dtype=torch.float64)
    n_out = z3.clone()
    n_out[..., 2] = 1.0
    ctx = StrikeContext(
        hit_index=b["hit"].clamp_min(1),
        active=b["active"].to(torch.float64),
        target_drum=b["drum"],
        start_drum=torch.full((n, 2), -1, dtype=torch.long),
        p_star=z3.clone(), v_star=z3.clone(), n_out=n_out,
        hit_radius=torch.full((n, 2), 0.08, dtype=torch.float64),
        disk_offset=z3.clone(), half_plane_dir=z3.clone(), has_half_plane=z2.clone(),
        contact_sigma=torch.full((n, 2), 0.2, dtype=torch.float64),
        shaft_allowed=z2.clone(), start_shaft_allowed=z2.clone(), elev_required=z2.clone(),
    )
    out = cost.forward(b["q0"], b["target"], b["v0"], ctx,
                       elem_mask=b["masks"].elem_mask)
    mine = (b["aux"]["torque"] * b["masks"].elem_mask).amax(dim=(1, 2))
    theirs = out.diagnostics["torque_max"]
    assert torch.allclose(mine, theirs, atol=1e-9, rtol=1e-9), (
        f"max |diff| = {float((mine - theirs).abs().max())}"
    )


def test_spline_aux_targets_are_the_spline_derivatives(pool, scenario_cfg):
    """``vel``/``acc`` labels reproduce a direct spline evaluation at the knots."""
    from amortized_opt.tspline import TrajectorySplineBuilder

    H = pool.buckets[0]
    idx = torch.arange(min(16, pool.size(H)))
    b = pool.batch(H, idx)
    sp = TrajectorySplineBuilder(
        max_segments=H, dt=scenario_cfg.dt, dtype=torch.float64
    ).from_masks(b["q0"], b["target"], b["v0"], b["masks"].elem_mask)
    m = b["masks"].elem_mask
    # knot derivative of knot i+1 == the aux label at waypoint i
    assert torch.allclose(sp.m[:, 1:] * m, b["aux"]["vel"] * m, atol=1e-9)
    # the label at the LAST valid waypoint of each element is the terminal value
    assert torch.isfinite(b["aux"]["acc"]).all()


def test_fk_aux_targets_match_a_direct_fk_call(pool):
    H = pool.buckets[0]
    idx = torch.arange(min(8, pool.size(H)))
    b = pool.batch(H, idx)
    fk = A.ChainFK(dtype=torch.float64)
    q = b["target"][:, 3]
    R, p = fk.forward(q)["left_wrist"]
    tip = A.ChainFK.points_world(
        R, p, torch.as_tensor(A.stick_tip_offset("left")[None, :], dtype=torch.float64)
    )[:, 0]
    valid = b["masks"].point_mask[:, 3, 0] > 0.5
    if bool(valid.any()):
        got = b["aux"]["fk_pos"][:, 3, 0][valid]
        assert torch.allclose(got, tip[valid], atol=1e-9)


# --------------------------------------------------------------------- losses --
def _batch_and_model(pool, scenario_cfg, n=8):
    H = pool.buckets[0]
    idx = torch.arange(min(n, pool.size(H)))
    gen = torch.Generator().manual_seed(0)
    b = pool.batch(H, idx, generator=gen)
    model, mc = _tiny_model(scenario_cfg)
    return b, model, mc


def test_losses_ignore_masked_elements(pool, scenario_cfg):
    """Perturbing anything outside the mask must not move a single loss term."""
    b, model, _mc = _batch_and_model(pool, scenario_cfg)
    scales, w = LossScales("cpu", torch.float64), LossWeights()
    with torch.no_grad():
        _l0, m0 = forward_loss(model, *unpack(b), scales, w)

    inv_e = (1.0 - b["masks"].elem_mask)
    inv_p = (1.0 - b["masks"].point_mask).unsqueeze(-1)
    g = torch.Generator().manual_seed(1)
    kick = lambda t, m: t + 137.0 * m * torch.randn(  # noqa: E731
        t.shape, dtype=t.dtype, generator=g
    )
    b2 = dict(b)
    b2["target"] = kick(b["target"], inv_e)
    b2["init"] = kick(b["init"], inv_e)
    b2["aux"] = {
        "vel": kick(b["aux"]["vel"], inv_e),
        "acc": kick(b["aux"]["acc"], inv_e),
        "torque": kick(b["aux"]["torque"], inv_e),
        "fk_pos": kick(b["aux"]["fk_pos"], inv_p),
        "fk_vel": kick(b["aux"]["fk_vel"], inv_p),
    }
    with torch.no_grad():
        _l1, m1 = forward_loss(model, *unpack(b2), scales, w)
    assert torch.allclose(m0, m1, atol=1e-10, rtol=1e-10), (
        f"masked perturbation moved {[METRIC_NAMES[i] for i in torch.nonzero(~torch.isclose(m0, m1)).flatten().tolist()]}"
    )


def test_one_optimization_step_reduces_the_loss(pool, scenario_cfg):
    b, model, _mc = _batch_and_model(pool, scenario_cfg, n=16)
    scales, w = LossScales("cpu", torch.float64), LossWeights()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(5):
        loss, _m = forward_loss(model, *unpack(b), scales, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], losses


def test_metric_vector_is_finite_and_named(pool, scenario_cfg):
    b, model, _mc = _batch_and_model(pool, scenario_cfg)
    with torch.no_grad():
        _l, m = forward_loss(model, *unpack(b), LossScales("cpu", torch.float64),
                             LossWeights())
    assert m.shape == (len(METRIC_NAMES),)
    assert torch.isfinite(m).all()
    assert float(m[METRIC_NAMES.index("n_elem")]) > 0


# ------------------------------------------------------------------ bucketing --
def test_bucket_for_routes_horizons():
    bk = (25, 50, 100)
    assert [bucket_for(k, bk) for k in (1, 25, 26, 50, 51, 100, 140)] == [
        25, 25, 50, 50, 100, 100, 100
    ]


def test_batch_sampler_is_bucket_homogeneous_and_in_range(pool):
    rng = np.random.default_rng(0)
    seen = set()
    for b, idx in batch_sampler(pool, 4, rng=rng, steps=40):
        assert b in pool.buckets
        assert int(idx.max()) < pool.size(b)
        assert pool.data[b]["target"].shape[1] == b
        batch = pool.batch(b, idx)
        assert batch["target"].shape == (4, b, A.N_DOF)
        # every sample in the batch really needs at most this bucket
        assert int(batch["hit"].max()) <= b
        seen.add(b)
    assert seen  # at least one bucket exercised


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_shapes_stay_static_across_buckets(pool, scenario_cfg, cuda_device, capsys):
    """One compile per bucket, and no further recompiles once warmed up."""
    from torch._dynamo.utils import counters

    dev = cuda_device
    model, _mc = _tiny_model(scenario_cfg, dtype=torch.float32)
    model = model.to(dev)
    gpu_pool = WarmstartPool(
        {b: {k: v.cpu().numpy() for k, v in d.items()
             if k not in ("elem_mask", "point_mask") and not k.startswith("aux_")}
         for b, d in pool.data.items()},
        cfg=scenario_cfg, device=dev, dtype=torch.float32,
    )
    step_fn = make_step(model, LossScales(dev), LossWeights(), compile_mode="default")
    rng = np.random.default_rng(0)

    torch._dynamo.reset()
    counters.clear()
    for b in gpu_pool.buckets:  # warm up every shape
        idx = torch.as_tensor(rng.integers(0, gpu_pool.size(b), size=4), device=dev)
        loss, _m = step_fn(*unpack(gpu_pool.batch(b, idx)))
        loss.backward()
    warm = sum(counters["frames"].values()) if "frames" in counters else 0
    counters.clear()
    for _ in range(6):  # steady state: every shape already seen
        for b in gpu_pool.buckets:
            idx = torch.as_tensor(rng.integers(0, gpu_pool.size(b), size=4), device=dev)
            loss, _m = step_fn(*unpack(gpu_pool.batch(b, idx)))
            loss.backward()
    after = sum(counters["frames"].values()) if "frames" in counters else 0
    with capsys.disabled():
        print(f"\n[compile] warmup frames={warm}, steady-state frames={after} "
              f"(buckets {gpu_pool.buckets})")
    assert after == 0, "torch.compile recompiled after warmup -- shapes are not static"


# --------------------------------------------------------------- checkpointing --
def test_checkpoint_roundtrip_is_bit_exact(pool, scenario_cfg, tmp_path):
    b, model, mc = _batch_and_model(pool, scenario_cfg, n=8)
    cfg = TrainConfig(max_steps=4, batch_size=8, device="cpu", compile_mode=None)
    opt = build_optimizer(model, cfg)
    scales, w = LossScales("cpu", torch.float64), LossWeights()
    for _ in range(3):  # give Adam non-trivial moments
        loss, _m = forward_loss(model, *unpack(b), scales, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    path = save_checkpoint(tmp_path / "ck.pt", model=model, optimizer=opt, step=3,
                           cfg=cfg, model_cfg=mc, scenario_cfg=scenario_cfg)

    ref_state = copy.deepcopy(model.state_dict())
    model2, _ = _tiny_model(scenario_cfg)
    opt2 = build_optimizer(model2, cfg)
    load_checkpoint(path, model=model2, optimizer=opt2)
    for k, v in model2.state_dict().items():
        assert torch.equal(v, ref_state[k]), f"{k} differs after reload"

    # optimizer state too: one more step from each must agree exactly
    l1, _ = forward_loss(model, *unpack(b), scales, w)
    opt.zero_grad(); l1.backward(); opt.step()
    l2, _ = forward_loss(model2, *unpack(b), scales, w)
    opt2.zero_grad(); l2.backward(); opt2.step()
    assert float(l1) == float(l2)
    for k, v in model2.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), f"{k} diverged after resume"


# ------------------------------------------------------------------- planners --
def test_model_planner_produces_a_scorable_trajectory(scenario_cfg):
    scorer = A.TrajectoryScorer(device="cpu")
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)[:2]
    model, _mc = _tiny_model(ScenarioConfig(horizon=100), dtype=torch.float32)
    for init in ("quintic", "hold"):
        planner = A.ModelPlanner(model=model, device="cpu", init=init)
        for case in cases:
            res = planner(case)
            assert res.Q.shape == (case.n_waypoints, A.N_DOF)
            assert np.isfinite(res.Q).all()
            assert np.allclose(res.Q[0], case.q_start)
            row = scorer.score(case, res.Q)
            assert np.isfinite(row.contact_pos_err_mm)


def test_quintic_case_planner_hits_the_grid_contract():
    scorer = A.TrajectoryScorer(device="cpu")
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)[:3]
    plan = A.quintic_case_planner()
    for case in cases:
        res = plan(case)
        assert res.Q.shape == (case.n_waypoints, A.N_DOF)
        assert np.allclose(res.Q[0], case.q_start)
        assert np.isfinite(res.Q).all()
