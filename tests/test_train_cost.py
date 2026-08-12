"""Phase-2/3 training machinery (section 7).

Small, fast checks on the pieces the run depends on and that a training curve
alone would not catch: that a step through the *compiled* path actually reduces
the cost, that the temporal-consistency term is exactly zero on a consistent
pair, that the chained-BPTT gradient reaches the first solve, that the bounce
model conserves tangential velocity, and that the augmented-Lagrangian state
survives a save/resume bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig
from amortized_opt.phase2_data import (
    Phase2Pool,
    Phase2ScenarioConfig,
    generate_cost_scenarios,
)
from amortized_opt.train_cost import (
    BounceConfig,
    MPCRoller,
    Phase2Config,
    ReplayPool,
    WeightSchedule,
    _wtensors,
    apply_bounce,
    bounce_tip_velocity,
    build_costs,
    effective_hits,
    group_lengths,
    make_bptt_step,
    encode,
    make_cost_step,
    make_tc_step,
    per_element_lengths,
    phase2_al_config,
    phase2_cost_config,
    ramp_seed,
    temporal_consistency_loss,
)

from conftest import pick_cuda_device

DEV = pick_cuda_device() or "cpu"
HORIZON = 25


@pytest.fixture(scope="module")
def bits():
    if torch.device(DEV).type == "cuda":
        torch.cuda.set_device(torch.device(DEV).index or 0)
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = Phase2ScenarioConfig(horizon=HORIZON, buckets=(HORIZON,))
    raw = generate_cost_scenarios(600, clips, seed=0, cfg=cfg)
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device=DEV)
    model = A.build_model(A.ModelConfig(max_horizon=HORIZON), DEV)
    p2 = Phase2Config(compile_mode=None, batch_size=32, bptt_batch=16, device=DEV)
    costs = build_costs(pool.buckets, p2.cost, device=DEV)
    sim = BatchedPDSim(PDSimConfig(dt=cfg.dt), device=DEV, dtype=torch.float32,
                       rnea=costs[HORIZON].rnea)
    return pool, model, p2, costs, sim


def _batch(pool, n=32, seed=0):
    g = torch.Generator(device=pool.device)
    g.manual_seed(seed)
    b = pool.buckets[0]
    idx = torch.randint(0, pool.size(b), (n,), device=pool.device, generator=g)
    batch = pool.batch(b, idx, generator=g)
    he = effective_hits(batch["hit"], batch["active"], b)
    from dataclasses import replace as dreplace

    batch["strike"] = dreplace(batch["strike"], hit_index=he)
    batch["hit_eff"] = he
    batch["lengths"] = group_lengths(he)
    return batch


# ================================================== 1. a step descends ======
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_one_step_reduces_cost(bits, compile_mode):
    """A single Adam step on the fused cost lowers it on the SAME batch."""
    pool, _model, p2, costs, _sim = bits
    torch.manual_seed(0)
    model = A.build_model(A.ModelConfig(max_horizon=HORIZON), DEV)
    scales = A.LossScales(torch.device(DEV))
    b = pool.buckets[0]
    step_fn = make_cost_step(model, costs[b], scales, p2, compile_mode=compile_mode)
    al = AugmentedLagrangian(phase2_al_config())
    w = _wtensors(WeightSchedule().multipliers(0), al.weights(),
                  torch.device(DEV), torch.float32)
    batch = _batch(pool, 32, seed=1)
    enc = encode(model, batch["context"])
    args = (batch["init"], batch["state"], batch["q0"], batch["v0"],
            enc, batch["masks"], batch["strike"], batch["lengths"], w,
            torch.zeros((), device=DEV))

    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    try:
        loss0, m0, _ = step_fn(*args)
    except Exception as exc:  # noqa: BLE001
        # This box pairs Isaac's prebundled torch with the conda env's Triton;
        # inductor emits kernels that Triton cannot parse ("type object
        # 'constexpr' has no attribute 'is_block'").  That is a toolchain fault,
        # not a fault in the step, and the eager parametrization already covers
        # the behaviour under test.
        if compile_mode is not None and "constexpr" in str(exc):
            pytest.skip(f"inductor/Triton mismatch in this environment: {exc}")
        raise
    opt.zero_grad(set_to_none=True)
    loss0.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        loss1, m1, _ = step_fn(*args)
    assert torch.isfinite(loss0) and torch.isfinite(loss1)
    assert float(loss1) < float(loss0), f"{float(loss0):.4f} -> {float(loss1):.4f}"
    # ...and it is the COST that fell, not only the auxiliary heads.
    assert float(m1[1]) < float(m0[1])


# ============================================ 2. temporal consistency =======
def test_temporal_consistency_is_zero_when_consistent():
    torch.manual_seed(0)
    B, H, n = 4, 12, A.N_DOF
    traj = torch.randn(B, H, n)
    mask = torch.ones(B, H, n)
    shifted = torch.cat([traj[:, 1:], torch.randn(B, 1, n)], dim=1)  # tail is free
    loss, cnt = temporal_consistency_loss(traj, shifted, mask, mask)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    assert float(cnt) == B * (H - 1) * n

    # a real disagreement is priced
    bad = shifted.clone()
    bad[:, 0] += 0.5
    loss2, _ = temporal_consistency_loss(traj, bad, mask, mask)
    assert float(loss2) > 0.0


def test_tc_step_runs_and_is_differentiable(bits):
    pool, model, p2, costs, sim = bits
    b = pool.buckets[0]
    fn = make_tc_step(model, costs[b], sim, p2, compile_mode=None)
    batch = _batch(pool, 16, seed=2)
    g = torch.Generator(device=pool.device)
    g.manual_seed(3)
    masks2 = pool.masks_for((batch["hit"] - 1).clamp_min(0), batch["active"], b)
    kp, kd, _ki = sim.sample_gains(16, generator=g)
    loss, cnt = fn(batch["init"], batch["state"], batch["q0"], batch["v0"],
                   encode(model, batch["context"]), batch["masks"], masks2, kp, kd)
    assert torch.isfinite(loss) and float(cnt) > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g_).all() for g_ in grads)
    model.zero_grad(set_to_none=True)


# ================================================ 3. chained BPTT ===========
def test_bptt_gradient_reaches_the_first_solve(bits):
    """The second solve's cost must move the weights *through* the plant."""
    pool, model, p2, costs, sim = bits
    b = pool.buckets[0]
    fn = make_bptt_step(model, costs[b], sim, p2, compile_mode=None)
    n = 16
    b1, b2 = _batch(pool, n, seed=4), _batch(pool, n, seed=5)
    g = torch.Generator(device=pool.device)
    g.manual_seed(7)
    al = AugmentedLagrangian(phase2_al_config())
    w = _wtensors(WeightSchedule().multipliers(0), al.weights(),
                  torch.device(DEV), torch.float32)

    big = torch.where(b1["active"], b1["hit"].clamp_min(0),
                      torch.full_like(b1["hit"], b + 1))
    k_first = big.min(dim=1).values.clamp(1, b)
    hand = big.argmin(dim=1)
    normal = b1["strike"].n_out.gather(1, hand.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    elen2 = per_element_lengths(b2["hit_eff"], pool.cfg.joint_names)
    term2 = b2["seed"].gather(1, (elen2.clamp(1, b) - 1).unsqueeze(1)).squeeze(1)
    kp, kd, _ki = sim.sample_gains(n, generator=g)
    z3 = torch.zeros(n, 3, device=DEV)
    zN = torch.zeros(n, A.N_DOF, device=DEV)
    e = torch.full((n,), 0.45, device=DEV)

    c1, c2 = fn(b1["init"], b1["state"], b1["q0"], b1["v0"],
                encode(model, b1["context"]), b1["masks"],
                b1["strike"], b1["lengths"], k_first,
                term2, encode(model, b2["context"]), b2["masks"], b2["strike"],
                b2["lengths"], elen2, w, kp, kd, hand, normal, e, zN, zN, z3)
    assert torch.isfinite(c1) and torch.isfinite(c2)

    model.zero_grad(set_to_none=True)
    c2.backward()                      # ONLY the second solve's cost
    named = dict(model.named_parameters())
    # The first solve is what feeds the plant; its gradient path is the decoder
    # trunk, so require a finite, non-zero gradient there.
    trunk = [p for k, p in named.items() if k.startswith("decoder.blocks.0")]
    assert trunk
    total = sum(float(p.grad.abs().sum()) for p in trunk if p.grad is not None)
    assert np.isfinite(total) and total > 0.0
    model.zero_grad(set_to_none=True)


# ==================================================== 4. the bounce =========
def test_bounce_conserves_tangential_velocity():
    torch.manual_seed(0)
    B = 64
    v = torch.randn(B, 3, dtype=torch.float64)
    n = torch.randn(B, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)
    e = torch.rand(B, dtype=torch.float64) * 0.3 + 0.3
    out = bounce_tip_velocity(v, n, e)

    def tangential(x):
        return x - (x * n).sum(-1, keepdim=True) * n

    assert torch.allclose(tangential(out), tangential(v), atol=1e-12)

    # incoming normal component is reflected and scaled by -e; outgoing is left
    vn = (v * n).sum(-1)
    on = (out * n).sum(-1)
    into = vn < 0
    assert torch.allclose(on[into], -e[into] * vn[into], atol=1e-12)
    assert torch.allclose(on[~into], vn[~into], atol=1e-12)


def test_apply_bounce_tracks_the_task_space_target(bits):
    """The joint-space realization reproduces the commanded tip velocity."""
    pool, _model, _p2, costs, _sim = bits
    cost = costs[pool.buckets[0]]
    g = torch.Generator(device=pool.device)
    g.manual_seed(0)
    batch = _batch(pool, 16, seed=8)
    q, qd = batch["q0"], batch["v0"] + 0.5
    hand = torch.zeros(16, dtype=torch.long, device=pool.device)
    normal = torch.zeros(16, 3, device=pool.device)
    normal[:, 2] = 1.0
    cfg = BounceConfig(tip_vel_noise=0.0, q_noise=0.0, qd_noise=0.0)
    q2, qd2 = apply_bounce(q, qd, fk=cost.fk, rnea=cost.rnea, hand_index=hand,
                           normal=normal, cfg=cfg, generator=g)
    assert torch.equal(q2, q)
    from amortized_opt.ik import tip_jacobian

    J = tip_jacobian(cost.fk, q, "left")
    v0 = torch.matmul(J, qd.unsqueeze(-1)).squeeze(-1)
    v1 = torch.matmul(J, qd2.unsqueeze(-1)).squeeze(-1)
    tang = lambda x: x - (x * normal).sum(-1, keepdim=True) * normal  # noqa: E731
    assert torch.allclose(tang(v1), tang(v0), atol=2e-3)
    # ...and the normal component actually reversed where it was incoming
    into = (v0 * normal).sum(-1) < 0
    if bool(into.any()):
        assert torch.all((v1 * normal).sum(-1)[into] >= (v0 * normal).sum(-1)[into])


# ============================================== 5. AL state round-trip ======
def test_al_state_save_resume_is_bit_exact(tmp_path):
    al = AugmentedLagrangian(phase2_al_config(rho_scale=1.7))
    rng = np.random.default_rng(0)
    for _ in range(37):
        al.update({g: float(abs(rng.normal(0.0, 0.1))) for g in al.groups})
    p = al.save(tmp_path / "al.json")
    other = AugmentedLagrangian.load(p, phase2_al_config(rho_scale=1.7))
    assert other.lam == al.lam
    assert other.avg_violation == al.avg_violation
    assert other.steps == al.steps
    # ...and it CONTINUES identically rather than merely restoring
    v = {g: 0.05 for g in al.groups}
    assert al.update(v) == other.update(v)


def test_checkpoint_round_trip_carries_the_multipliers(bits, tmp_path):
    from amortized_opt.train_cost import (
        load_phase2_checkpoint,
        save_phase2_checkpoint,
    )

    _pool, model, p2, _costs, _sim = bits
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    al = AugmentedLagrangian(phase2_al_config())
    al.update({g: 0.3 for g in al.groups})
    mcfg = A.ModelConfig(max_horizon=HORIZON)
    save_phase2_checkpoint(tmp_path / "c.pt", model=model, optimizer=opt, al=al,
                           step=42, cfg=p2, model_cfg=mcfg)
    al2 = AugmentedLagrangian(phase2_al_config())
    ck = load_phase2_checkpoint(tmp_path / "c.pt", al=al2)
    assert ck["step"] == 42
    assert al2.lam == al.lam and al2.steps == al.steps


# ================================================ 6. schedule + roller ======
def test_weight_schedule_ramps_monotonically():
    s = WeightSchedule(ramp_start=100, ramp_end=1_100, lo_limits=0.05, lo_strike=0.1)
    assert s.multipliers(0)["torque"] == pytest.approx(0.05)
    assert s.multipliers(50)["terminal_velocity"] == pytest.approx(0.1)
    assert s.multipliers(5_000)["torque"] == pytest.approx(1.0)
    assert s.multipliers(5_000)["terminal_velocity"] == pytest.approx(1.0)
    xs = [s.multipliers(k)["torque"] for k in range(0, 1_200, 25)]
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))
    # the stage-A groups never move
    for k in (0, 500, 50_000):
        assert s.multipliers(k)["collision"] == 1.0


def test_ramp_seed_hits_the_terminal_and_holds():
    B, n, H = 3, A.N_DOF, 20
    q = torch.zeros(B, n)
    term = torch.ones(B, n)
    lengths = torch.full((B, n), 10, dtype=torch.long)
    out = ramp_seed(q, term, lengths, H)
    assert out.shape == (B, H, n)
    assert torch.allclose(out[:, 9], term)          # arrives at its horizon
    assert torch.allclose(out[:, -1], term)         # ...and holds
    assert torch.all(out[:, 0] < out[:, 5])         # monotone on the way


def test_roller_produces_usable_mpc_contexts(bits):
    pool, model, p2, costs, sim = bits
    base = costs[pool.buckets[0]]
    roller = MPCRoller(
        pool, model=model, sim=sim, rnea=base.rnea, fk=base.fk,
        horizon=HORIZON, slots=48, device=DEV, dtype=torch.float32, seed=0,
    )
    replay = ReplayPool(pool, 512, buckets=list(pool.buckets))
    n_reset = 0
    for _ in range(12):
        for b, cols in roller.harvest().items():
            replay.add(b, cols)
        n_reset += roller.tick()["reset"]
    assert len(replay) > 0
    assert torch.isfinite(roller.state["q0"]).all()
    assert torch.isfinite(roller.state["init"]).all()
    # the replay is drawable and its batches are the same shape as the pool's
    b = pool.buckets[0]
    g = torch.Generator(device=pool.device)
    g.manual_seed(0)
    idx = replay.sample_idx(b, 8, generator=g)
    batch = replay.batch(b, idx, generator=g, noise=False)
    assert batch["init"].shape == (8, b, A.N_DOF)
    assert torch.isfinite(batch["init"]).all()
    assert batch["strike"].drum_head_centers is not None
