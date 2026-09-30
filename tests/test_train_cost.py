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
from amortized_opt.grad_balance import GRAD_BALANCE_GROUPS
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
    default_weight_schedule,
    _sample_fresh_batch,
    _index_batch,
    _run_suffix_step_chunked,
    _VerifiedSuffixRing,
    _split_timing_superbatch,
    _timing_curriculum_group_count,
    _timing_primary_layout,
    _splice_frontier_batch,
    _wtensors,
    apply_bounce,
    bounce_tip_velocity,
    build_costs,
    cost_grad_balance_tuple,
    effective_hits,
    group_lengths,
    make_bptt_step,
    encode,
    make_cost_step,
    make_tc_step,
    measure_torque_directional_curvature,
    mpc_init_tie_loss,
    per_element_lengths,
    phase2_al_config,
    phase2_cost_config,
    ramp_seed,
    stage_a_pass,
    temporal_consistency_loss,
    waist_yaw_mpc_init_tie_loss,
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


def test_timing_superbatch_split_keeps_task_groups_together():
    # Four six-row timing groups plus eight ungrouped MPC rows -> 4 x B8.
    groups = torch.cat((torch.arange(4).repeat_interleave(6),
                        torch.full((8,), -1))).long()
    batch = {
        "q0": torch.arange(32).float().unsqueeze(1),
        "timing_primary_group": groups,
        "timing_primary_kind": torch.zeros(32, dtype=torch.long),
        "mpc_init": groups < 0,
        "bucket": 50,
    }
    gen = torch.Generator().manual_seed(7)
    parts = _split_timing_superbatch(
        batch, 8, 4, generator=gen,
    )
    assert [int(p["q0"].shape[0]) for p in parts] == [8, 8, 8, 8]
    for group in range(4):
        owners = [
            i for i, part in enumerate(parts)
            if bool((part["timing_primary_group"] == group).any())
        ]
        assert len(owners) == 1
        assert int((parts[owners[0]]["timing_primary_group"] == group).sum()) == 6


def test_timing_curriculum_group_ramp_preserves_512_row_layout():
    from amortized_opt.violation_critic import ViolationCriticConfig

    cfg = ViolationCriticConfig(
        representation="distributional", anchor_curriculum=True,
        anchor_curriculum_ramp_steps=2_000,
    )
    base, anchor, boundary, interior, mpc = _timing_primary_layout(512, cfg)
    assert (base, anchor, boundary, interior, mpc) == (40, 40, 40, 160, 272)
    timing_n = anchor + boundary + interior
    per_task = cfg.primary_anchor_per_task + cfg.primary_boundary_per_task + cfg.primary_interior_per_task
    for step, expected_groups in ((7_550, 0), (8_550, 20), (9_550, 40)):
        groups = _timing_curriculum_group_count(
            base_tasks=base, curriculum_open=True,
            curriculum_open_actor_step=7_550, actor_step=step,
            ramp_steps=cfg.anchor_curriculum_ramp_steps,
        )
        independent = timing_n - groups * per_task
        selected = groups * (
            cfg.primary_boundary_per_task + cfg.primary_interior_per_task
        )
        assert groups == expected_groups
        assert groups * per_task + independent == timing_n == 240
        assert timing_n + mpc == 512
        assert selected == expected_groups * 5

    # An old open checkpoint has no opening-step provenance and must not be
    # unexpectedly re-ramped; the disabled setting is likewise full-open.
    assert _timing_curriculum_group_count(
        base_tasks=base, curriculum_open=True,
        curriculum_open_actor_step=None, actor_step=7_550, ramp_steps=2_000,
    ) == base
    assert _timing_curriculum_group_count(
        base_tasks=base, curriculum_open=True,
        curriculum_open_actor_step=7_550, actor_step=7_550, ramp_steps=0,
    ) == base


def test_suffix_microbatch_preserves_weighted_loss_and_row_order(bits):
    """Chunking is a memory schedule, not a change to the suffix objective."""
    pool, model, _p2, _costs, _sim = bits
    batch = _batch(pool, 7, seed=11)
    enc = encode(model, batch["context"])
    traj = batch["init"]
    q0 = batch["q0"]
    v0 = batch["v0"]
    elem_mask = batch["masks"].elem_mask
    offsets = torch.arange(7, device=traj.device)

    def fake_suffix_fn(
        traj_i, q0_i, v0_i, _enc_i, _em_i, _masks_i, offsets_i,
    ):
        # The tie is a batch mean, matching make_suffix_step's contract.
        loss_tie = (q0_i.square().mean(dim=1)).mean()
        return (
            loss_tie,
            torch.as_tensor(
                float(traj_i.numel()), device=traj_i.device,
                dtype=traj_i.dtype,
            ),
            q0_i[:, :1],
            v0_i[:, :1],
            offsets_i[:, None].to(traj_i.dtype),
        )

    args = (
        fake_suffix_fn, traj, q0, v0, enc, elem_mask, batch["masks"],
        offsets,
    )
    full = _run_suffix_step_chunked(*args, microbatch=0)
    chunked = _run_suffix_step_chunked(*args, microbatch=3)
    for expected, actual in zip(full, chunked):
        torch.testing.assert_close(actual, expected)


def test_verified_suffix_ring_is_fifo_and_fixed_batch(bits):
    pool, _model, _p2, _costs, _sim = bits
    batch = _batch(pool, 40, seed=17)
    bucket = int(batch["bucket"])
    ring = _VerifiedSuffixRing(capacity=32, device=batch["q0"].device)
    admitted = ring.add(
        bucket, batch, batch["init"],
        hard_ok=torch.ones(40, device=batch["q0"].device, dtype=torch.bool),
    )
    assert admitted == 40
    assert len(ring) == ring.size(bucket) == 32
    assert ring.data[bucket]["suffix_sequence"].min().item() == 8

    sampled, fill = ring.sample(
        bucket, 16, min_parents=16,
        generator=torch.Generator(device=batch["q0"].device).manual_seed(3),
    )
    assert sampled is not None and sampled["q0"].shape[0] == 16
    assert sampled["suffix_sequence"].unique().numel() == 16
    assert fill == 1.0

    restored = _VerifiedSuffixRing(capacity=32, device=batch["q0"].device)
    restored.load_state_dict(ring.state_dict())
    assert restored.size(bucket) == 32
    torch.testing.assert_close(
        restored.data[bucket]["suffix_sequence"],
        ring.data[bucket]["suffix_sequence"],
    )

    cold = _VerifiedSuffixRing(capacity=32, device=batch["q0"].device)
    cold.add(
        bucket, _index_batch(batch, torch.arange(8, device=batch["q0"].device)),
        batch["init"][:8],
        hard_ok=torch.ones(8, device=batch["q0"].device, dtype=torch.bool),
    )
    sampled, fill = cold.sample(bucket, 16, min_parents=16)
    assert sampled is None and fill == 0.0


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
        loss0, m0, _, _ = step_fn(*args)
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
        loss1, m1, _, _ = step_fn(*args)
    assert torch.isfinite(loss0) and torch.isfinite(loss1)
    assert float(loss1) < float(loss0), f"{float(loss0):.4f} -> {float(loss1):.4f}"
    # ...and it is the COST that fell, not only the auxiliary heads.
    assert float(m1[1]) < float(m0[1])


def test_grad_balance_cost_tuple_matches_named_terms(bits):
    pool, model, _p2, costs, _sim = bits
    batch = _batch(pool, 8, seed=7)
    enc = encode(model, batch["context"])
    out = model(batch["init"], batch["state"], enc, batch["masks"])
    al = AugmentedLagrangian(phase2_al_config())
    weights = _wtensors(
        WeightSchedule().multipliers(0), al.weights(),
        torch.device(DEV), torch.float32,
    )
    payload = cost_grad_balance_tuple(
        costs[pool.buckets[0]], batch["q0"], out.traj, batch["v0"],
        batch["strike"], batch["lengths"], weights,
    )
    direct = costs[pool.buckets[0]].forward(
        batch["q0"], out.traj, batch["v0"], batch["strike"],
        lengths=batch["lengths"], weights=weights,
    )
    assert torch.allclose(payload[0], direct.total)
    expected = torch.stack([
        sum(direct.terms[k].mean() for k in term_keys)
        for term_keys, _al_keys in GRAD_BALANCE_GROUPS.values()
    ])
    assert torch.allclose(payload[5], expected)
    assert torch.allclose(
        payload[6], direct.balance_terms["joint_box_waist"].mean(),
    )


def test_torque_directional_curvature_probe_is_finite(bits):
    pool, model, _p2, costs, _sim = bits
    batch = _batch(pool, 4, seed=13)
    enc = encode(model, batch["context"])
    out = model(batch["init"], batch["state"], enc, batch["masks"])
    al = AugmentedLagrangian(phase2_al_config())
    weights = _wtensors(
        WeightSchedule().multipliers(0), al.weights(),
        torch.device(DEV), torch.float32,
    )
    # Production weights can be row-wise (for example voxel probe rows have
    # zero dual price).  The curvature diagnostic uses only the first two rows
    # and must slice those weights with the trajectory subset.
    weights["collision_peak"] = torch.linspace(
        0.5, 1.0, 4, device=torch.device(DEV),
    )
    cfg = Phase2Config(device=DEV, torque_curvature_rows=2)
    traj_before = out.traj.detach().clone()
    rnea_dtype = costs[pool.buckets[0]].rnea.axis.dtype
    device_type = "cuda" if out.traj.is_cuda else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16):
        metrics = measure_torque_directional_curvature(
            costs[pool.buckets[0]], cfg, traj=out.traj, q0=batch["q0"],
            v0=batch["v0"], strike=batch["strike"], lengths=batch["lengths"],
            weights=weights,
        )
    assert metrics["torque_curve_ok"] == 1.0
    assert np.isfinite(metrics["torque_curve_rho_1p0mrad"])
    assert np.isfinite(metrics["torque_curve_fd_safe_mrad"])
    assert metrics["torque_curve_hvp_ok"] == 1.0
    assert np.isfinite(metrics["torque_curve_dHd"])
    assert metrics["torque_curve_selected_step_mrad"] in (2.0, 1.0, 0.5, 0.25)
    assert metrics["torque_curve_fd_legacy_unreliable"] == 1.0
    for step in ("5p0", "2p0", "1p0", "0p5", "0p25"):
        assert np.isfinite(metrics[f"torque_curve_gradfd_dHd_{step}mrad"])
        assert metrics[f"torque_curve_active_cross_total_{step}mrad"] >= 0.0
    # Diagnostic transaction: caller data and plant precision are untouched.
    assert torch.equal(out.traj.detach(), traj_before)
    assert costs[pool.buckets[0]].rnea.axis.dtype == rnea_dtype


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


def test_mpc_init_tie_masks_fresh_rows_and_padded_tail():
    torch.manual_seed(1)
    B, H, n = 3, 6, A.N_DOF
    init = torch.randn(B, H, n)
    traj = init.clone().requires_grad_(True)
    mask = torch.ones_like(traj)
    mpc = torch.tensor([True, False, True])

    # Fresh row disagreement and the repeated/padded final knot are ignored.
    changed = traj.clone()
    changed[1, :-1] += 2.0
    changed[:, -1] += 3.0
    loss0, count = mpc_init_tie_loss(changed, init, mask, mpc)
    assert float(loss0) == pytest.approx(0.0, abs=1e-12)
    assert float(count) == 2 * (H - 1) * n

    # A disagreement on a marked replay row is priced and differentiates only
    # the current output; stored history remains detached.
    changed = traj.clone()
    changed[0, 0, 0] += 0.5
    loss1, _ = mpc_init_tie_loss(changed, init, mask, mpc)
    assert float(loss1) > 0.0
    loss1.backward()
    assert traj.grad is not None
    assert float(traj.grad[0, 0, 0].abs()) > 0.0
    assert float(traj.grad[1].abs().max()) == 0.0
    assert float(traj.grad[:, -1].abs().max()) == 0.0


def test_waist_yaw_mpc_tie_only_differentiates_yaw():
    B, H, n = 2, 5, A.N_DOF
    yaw = A.G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
    parent = torch.zeros(B, H, n)
    traj = torch.zeros(B, H, n, requires_grad=True)
    mask = torch.ones_like(traj)
    with torch.no_grad():
        traj[0, 0, yaw] = 0.1
        traj[0, 0, yaw + 1] = 3.0
        traj[1, 0, yaw] = 4.0
    loss, count, grad_norm = waist_yaw_mpc_init_tie_loss(
        traj, parent, mask, torch.tensor([True, False]), yaw_index=yaw,
    )
    assert float(loss) > 0.0
    assert float(count) == H - 1
    assert float(grad_norm) > 0.0
    loss.backward()
    assert float(traj.grad[0, 0, yaw].abs()) > 0.0
    assert float(traj.grad[0, :, yaw + 1].abs().max()) == 0.0
    assert float(traj.grad[1].abs().max()) == 0.0


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
    assert other.pen == al.pen
    assert other.err_prev == al.err_prev
    assert other._d == al._d
    assert other._seen == al._seen
    # ...and it CONTINUES identically rather than merely restoring
    v = {g: 0.05 for g in al.groups}
    assert al.update(v) == other.update(v)
    assert al.pen == other.pen
    assert al.err_prev == other.err_prev
    assert al._d == other._d


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
    assert al2.pen == al.pen and al2.err_prev == al.err_prev


def test_checkpoint_round_trip_restores_phase2_rng_streams(tmp_path):
    """Checkpointed global, local torch, and NumPy streams resume exactly."""
    from amortized_opt.train_cost import (
        load_phase2_checkpoint,
        save_phase2_checkpoint,
    )

    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    al = AugmentedLagrangian(phase2_al_config())
    cfg = Phase2Config(max_steps=1)
    mcfg = A.ModelConfig(max_horizon=2)
    streams = {
        "gen": torch.Generator().manual_seed(11),
        "sfx_gen": torch.Generator().manual_seed(12),
        "jit_gen": torch.Generator().manual_seed(13),
        "rng": np.random.default_rng(14),
    }
    old_global = torch.get_rng_state()
    try:
        torch.manual_seed(15)
        # Advance every stream before saving so this checks state, not seeds.
        for stream in streams.values():
            if isinstance(stream, torch.Generator):
                torch.rand(7, generator=stream)
            else:
                stream.random(7)
        save_phase2_checkpoint(
            tmp_path / "rng.pt", model=model, optimizer=opt, al=al, step=3,
            cfg=cfg, model_cfg=mcfg, rng_streams=streams,
        )
        expected_global = torch.rand(5)
        expected_torch = {
            name: torch.rand(5, generator=stream)
            for name, stream in streams.items()
            if isinstance(stream, torch.Generator)
        }
        expected_numpy = streams["rng"].random(5)

        restored = {
            "gen": torch.Generator().manual_seed(101),
            "sfx_gen": torch.Generator().manual_seed(102),
            "jit_gen": torch.Generator().manual_seed(103),
            "rng": np.random.default_rng(104),
        }
        load_phase2_checkpoint(tmp_path / "rng.pt", rng_streams=restored)
        assert torch.equal(torch.rand(5), expected_global)
        for name, stream in restored.items():
            if isinstance(stream, torch.Generator):
                assert torch.equal(torch.rand(5, generator=stream), expected_torch[name])
            else:
                np.testing.assert_array_equal(stream.random(5), expected_numpy)
    finally:
        torch.set_rng_state(old_global)


def test_checkpoint_rng_restore_accepts_legacy_missing_stream_state(tmp_path):
    """Old CPU-only checkpoints restore what they have and never require extras."""
    from amortized_opt.train_cost import load_phase2_checkpoint

    old_global = torch.get_rng_state()
    try:
        torch.manual_seed(22)
        legacy_cpu = torch.get_rng_state().clone()
        expected = torch.rand(4)
        torch.save({"torch_rng": legacy_cpu}, tmp_path / "legacy.pt")
        torch.manual_seed(999)
        streams = {
            "gen": torch.Generator().manual_seed(31),
            "sfx_gen": torch.Generator().manual_seed(32),
            "jit_gen": torch.Generator().manual_seed(33),
            "rng": np.random.default_rng(34),
        }
        load_phase2_checkpoint(tmp_path / "legacy.pt", rng_streams=streams)
        assert torch.equal(torch.rand(4), expected)
    finally:
        torch.set_rng_state(old_global)


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


def test_weight_schedule_default_is_full_at_step_zero():
    m = default_weight_schedule().multipliers(0)
    for k in ("torque", "velocity", "joint_box", "terminal_velocity",
              "waist_yaw_vel", "waist_pr_vel", "torque_peak", "joint_box_peak"):
        assert m[k] == pytest.approx(1.0), k
    # Phase2Config uses the same factory.
    assert Phase2Config().schedule.multipliers(0)["torque"] == pytest.approx(1.0)


def test_weight_schedule_snap_full_is_one_at_step_zero():
    s = WeightSchedule(
        ramp_start=10_000, ramp_end=30_000, lo_limits=0.05, lo_strike=0.1,
        torque_peak_start=0, torque_peak_ramp=1000,
        collision_peak_start=0, collision_peak_ramp=1000,
        waist_start=0, waist_ramp=300,
        effort_start=0, effort_ramp=300,
        idle_start=0, idle_ramp=300,
        tip_crossover_start=0, tip_crossover_ramp=300,
    ).snap_full()
    m = s.multipliers(0)
    for k in ("torque", "velocity", "joint_box", "terminal_velocity",
              "terminal_elev", "torque_peak", "velocity_peak", "collision_peak",
              "joint_box_peak",
              "waist_box", "posture_upright", "waist_aim",
              "waist_pr_terminal",
              "effort", "posture_idle", "tip_crossover"):
        assert m[k] == pytest.approx(1.0), k


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


def test_stage_a_unlocks_at_30_of_81():
    base = dict(n=81, ok_collision=0.90, elbow_down=1.0)
    assert not stage_a_pass({**base, "accepted": 29})
    assert stage_a_pass({**base, "accepted": 30})
    assert not stage_a_pass({**base, "accepted": 81, "ok_collision": 0.5})
    assert not stage_a_pass({**base, "accepted": 81, "elbow_down": 0.5})


def test_phase2_turns_on_precision_terminal_aim():
    assert phase2_cost_config().w_terminal_aim == pytest.approx(40.0)
    assert phase2_cost_config().w_torque * 1.0 == pytest.approx(
        phase2_cost_config().w_joint_box * 0.05
    )
    assert phase2_cost_config().w_torque * 1.0 == pytest.approx(
        phase2_cost_config().w_velocity * 0.5
    )
    assert A.CostConfig().w_terminal_aim == pytest.approx(0.0)
    assert phase2_cost_config().terminal_aim_deadband_m == pytest.approx(0.005)


def test_sample_fresh_batch_draws_live_post_hit(bits):
    from amortized_opt.phase2_data import KIND_POST_HIT

    pool, *_ = bits
    b = pool.buckets[0]
    live = ReplayPool(pool, 64, buckets=[b], recency_bias=0.0)
    n = min(16, pool.size(b))
    cols = {k: v[:n].contiguous() for k, v in pool.data[b].items()}
    cols["kind"] = torch.full((n,), KIND_POST_HIT, device=pool.device, dtype=torch.long)
    live.add(b, cols)
    g = torch.Generator(device=pool.device)
    g.manual_seed(0)
    batch = _sample_fresh_batch(
        pool, b, 8, generator=g,
        kind_fracs={KIND_POST_HIT: 1.0},
        live_post_hit=live, noise=False,
    )
    assert int(batch["kind"].shape[0]) == 8
    assert (batch["kind"] == KIND_POST_HIT).all()
    assert batch["mpc_init"].all()


def test_sample_fresh_batch_concats_pool_and_live(bits):
    """Live post-hit rows omit ``waist_yaw_ref`` under head-mode harvest;
    mixing them with the fresh pool must not crash on ``None`` cat."""
    from amortized_opt.phase2_data import KIND_POST_HIT

    pool, *_ = bits
    b = pool.buckets[0]
    live = ReplayPool(pool, 64, buckets=[b], recency_bias=0.0)
    n = min(16, pool.size(b))
    cols = {k: v[:n].contiguous() for k, v in pool.data[b].items()}
    cols["kind"] = torch.full((n,), KIND_POST_HIT, device=pool.device, dtype=torch.long)
    live.add(b, cols)
    other = next(int(k) for k, c in pool.kind_counts().items() if int(k) != KIND_POST_HIT and c > 0)
    g = torch.Generator(device=pool.device)
    g.manual_seed(1)
    batch = _sample_fresh_batch(
        pool, b, 16, generator=g,
        kind_fracs={other: 0.4, KIND_POST_HIT: 0.6},
        live_post_hit=live, noise=False,
    )
    assert int(batch["q0"].shape[0]) == 16
    assert 0 < int(batch["mpc_init"].sum()) < 16
    assert batch["strike"].waist_yaw_ref is None or torch.is_tensor(
        batch["strike"].waist_yaw_ref
    )


def test_sample_fresh_idle_far_from_cloud_only(bits):
    """Idle/far remainder is cloud collocations; post-hit stays live."""
    from amortized_opt.frontier import FrontierCloud, FrontierConfig, ingest_pool
    from amortized_opt.phase2_data import KIND_FAR_FETCHED, KIND_POST_HIT, KIND_RHYTHM_IDLE

    pool, *_ = bits
    b = pool.buckets[0]
    cloud = FrontierCloud(
        FrontierConfig(enabled=True, traj_stride=10_000, buckets=tuple(pool.buckets)),
        device=torch.device(pool.device),
        q_lo=pool.q_lo, q_hi=pool.q_hi,
    )
    ingest_pool(cloud, pool)
    assert cloud.size(b) >= 16
    g = torch.Generator(device=pool.device)
    g.manual_seed(2)
    all_cloud = _sample_fresh_batch(
        pool, b, 8, generator=g,
        kind_fracs={KIND_RHYTHM_IDLE: 0.3, KIND_FAR_FETCHED: 0.1, KIND_POST_HIT: 0.6},
        live_post_hit=None, noise=False,
        frontier=cloud, frontier_idle_far=True,
    )
    assert int(all_cloud["q0"].shape[0]) == 8
    assert (all_cloud["cid"] > 0).all()
    assert not bool(all_cloud["mpc_init"].any())

    live = ReplayPool(pool, 64, buckets=[b], recency_bias=0.0)
    n = min(16, pool.size(b))
    cols = {k: v[:n].contiguous() for k, v in pool.data[b].items()}
    cols["kind"] = torch.full((n,), KIND_POST_HIT, device=pool.device, dtype=torch.long)
    live.add(b, cols)
    g.manual_seed(3)
    mixed = _sample_fresh_batch(
        pool, b, 32, generator=g,
        kind_fracs={KIND_RHYTHM_IDLE: 0.4, KIND_POST_HIT: 0.6},
        live_post_hit=live, noise=False,
        frontier=cloud, frontier_idle_far=True,
    )
    assert int(mixed["q0"].shape[0]) == 32
    n_cloud = int((mixed["cid"] > 0).sum())
    n_ph = int((mixed["cid"] == 0).sum())
    assert n_cloud > 0 and n_ph > 0
    assert n_cloud + n_ph == 32
    assert torch.equal(mixed["mpc_init"], mixed["cid"] == 0)


def test_splice_frontier_into_replay_prefers_children_when_able(bits):
    """Idle-far replay splice keeps the prefix and stamps cloud cids on the tail."""
    from amortized_opt.frontier import FrontierCloud, FrontierConfig, ingest_pool

    pool, *_ = bits
    b = pool.buckets[0]
    cloud = FrontierCloud(
        FrontierConfig(enabled=True, traj_stride=10_000, buckets=tuple(pool.buckets)),
        device=torch.device(pool.device),
        q_lo=pool.q_lo, q_hi=pool.q_hi,
    )
    ingest_pool(cloud, pool)
    g = torch.Generator(device=pool.device)
    g.manual_seed(4)
    base = _sample_fresh_batch(
        pool, b, 16, generator=g, kind_fracs=None, noise=False,
    )
    prefix = base["q0"][:12].clone()
    out = _splice_frontier_batch(
        base, pool, cloud, b, 16, k=4, generator=g, inv_frac=0.0,
        prefer_children=True,
    )
    assert int(out["q0"].shape[0]) == 16
    assert torch.equal(out["q0"][:12], prefix)
    assert (out["cid"][12:] > 0).all()
    assert (out["cid"][:12] == 0).all()
