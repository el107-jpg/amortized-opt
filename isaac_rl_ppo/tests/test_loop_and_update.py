"""The rollout driver and the PPO update, end to end, WITHOUT Isaac.

The plant is swapped for ``amortized_opt.pd_sim.BatchedPDSim.step``, whose
signature is deliberately identical to the Isaac env's ``plant`` (the sibling
built it that way so the two loops are the same code path).  So everything here
-- the warm-start state machine, the lead/slew/box command path, the reward
wiring, the storage, GAE, and the fused update -- is the code that will run
under Isaac, driven by a different plant.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A

from isaac_rl_ppo.cfg import PpoRunConfig
from isaac_rl_ppo.ppo_loop import (
    PpoStorage,
    PpoUpdater,
    RolloutDriver,
    compute_gae,
    index_dataclass,
    stack_dataclass,
)


@pytest.fixture
def driver(actor, scheduled, mpc, run_cfg, surface):
    """A full :class:`RolloutDriver` on the analytic PD plant."""
    from amortized_opt.cost import FusedCost
    from amortized_opt.pd_sim import BatchedPDSim
    from amortized_opt.train_cost import phase2_cost_config

    sched, _kit, B = scheduled
    fk = A.ChainFK(device="cpu")
    cost = FusedCost(phase2_cost_config(), max_horizon=actor.H, device="cpu")
    scorer = surface.ExecutedStrikeScorer(fk=fk, device="cpu")
    rcfg = surface.RewardConfig()
    rcfg.w_plan_cost = run_cfg.reward_plan_cost_weight
    reward = surface.ExecutedReward(rcfg, scorer=scorer, cost=cost, device="cpu")
    sim = BatchedPDSim(device="cpu")

    drv = RolloutDriver(
        actor=actor, mpc=mpc, schedule=sched, reward=reward,
        plant=sim.step, cfg=run_cfg, fk=fk)
    q = torch.zeros(B, 17)
    mpc.reset(torch.arange(B), q)
    return drv, q, torch.zeros(B, 17), cost


# ================================================ warm start / state machine ==
def test_horizons_decrement_between_ticks(driver):
    """``hit_index`` counts DOWN one waypoint per control tick."""
    drv, q, qd, _cost = driver
    sched = drv.sched

    first = sched.hit_index().clone()
    seen = [first]
    for _ in range(4):
        q, qd, _ = drv.tick(q, qd)
        seen.append(sched.hit_index().clone())

    for a, b in zip(seen[:-1], seen[1:]):
        live = (a > 1) & (b >= 0)
        assert torch.all(b[live] == a[live] - 1), \
            f"horizon must decrement by exactly one waypoint: {a} -> {b}"


def test_warm_start_shifts_by_one_waypoint(driver):
    """``commit`` shifts the plan: next tick's init is this tick's traj[1:]."""
    drv, q, qd, _cost = driver
    res = drv.mpc.plan(q, qd)
    traj = res.traj.clone()
    fired = drv.sched.advance()
    drv.mpc.commit(res.cmd, q, fired)

    assert torch.allclose(drv.mpc.init[:, :-1], traj[:, 1:], atol=0, rtol=0)
    assert torch.allclose(drv.mpc.init[:, -1], traj[:, -1], atol=0, rtol=0)


def test_regoal_on_hit_completion(driver):
    """When a hand fires, ITS arm columns are re-seeded and nothing else is.

    The re-seed ramps that hand's arm to the goal-pose library terminal; the
    other hand and the waist carry straight through.  A re-seed that touched
    the idle hand would silently discard its in-flight plan.
    """
    drv, q, qd, _cost = driver
    mpc, sched = drv.mpc, drv.sched

    fired_any = False
    for _ in range(120):
        res = mpc.plan(q, qd)
        traj = res.traj.clone()
        q, qd = drv._step_plant(mpc.prev_cmd, res.cmd, q, qd)
        fired = sched.advance()
        before = mpc.init.clone()
        mpc.commit(res.cmd, q, fired)

        if bool(fired.any()):
            fired_any = True
            shifted = torch.cat([traj[:, 1:], traj[:, -1:]], dim=1)
            rows = fired.any(dim=1).nonzero(as_tuple=True)[0]
            for b in rows.tolist():
                for h in (0, 1):
                    cols = mpc.arm_cols[h]
                    same = torch.equal(mpc.init[b][:, cols], shifted[b][:, cols])
                    if bool(fired[b, h]) and sched.hit_index()[b, h] >= 1:
                        assert not same, f"hand {h} fired but was not re-seeded"
                    elif not bool(fired[b, h]):
                        assert same, f"idle hand {h} must carry through untouched"
                # the waist always carries through the shift
                assert torch.equal(mpc.init[b][:, mpc.waist_cols],
                                   shifted[b][:, mpc.waist_cols])
            break
        del before
    assert fired_any, "no hit fired in 120 ticks; the fixture clip is wrong"


def test_reset_reseeds_the_plan_and_survives(driver):
    """State survives an env reset: the plan of reset envs is re-seeded."""
    drv, q, qd, _cost = driver
    mpc = drv.mpc
    for _ in range(3):
        q, qd = drv.tick(q, qd)[:2]

    before = mpc.init.clone()
    ids = torch.tensor([0, 2])
    q_new = 0.2 * torch.ones_like(q)
    mpc.reset(ids, q_new)

    assert not torch.equal(mpc.init[0], before[0]), "reset env must be re-seeded"
    assert torch.equal(mpc.init[1], before[1]), "untouched env must not move"
    assert torch.allclose(mpc.prev_cmd[0], q_new[0])
    assert torch.isfinite(mpc.init).all()


def test_executed_command_respects_slew_and_box(driver):
    """The command handed to the plant is admissible even with the noise on."""
    drv, q, qd, _cost = driver
    mpc = drv.mpc
    with torch.no_grad():
        drv.actor.std.fill_(0.9)                   # large exploration
    for _ in range(6):
        prev = mpc.prev_cmd.clone()
        q, qd, _ = drv.tick(q, qd)
        cmd = mpc.prev_cmd                          # commit stored what executed
        assert torch.all(cmd >= mpc.q_lo - 1e-6) and torch.all(cmd <= mpc.q_hi + 1e-6)
        if mpc.step_max is not None:
            assert torch.all((cmd - prev).abs() <= mpc.step_max + 1e-5), \
                "slew bound violated after the exploration offset"


def test_lead_is_not_applied_twice(driver):
    """With sigma = 0 the executed command is EXACTLY the deployed one.

    The Gaussian head must add an offset to the finished (led, slewed, clamped)
    command and nothing else.  If the head or the loop re-applied the lead, this
    would drift -- the failure the coordination note asked to be guarded.
    """
    drv, q, qd, _cost = driver
    with torch.no_grad():
        drv.actor.std.fill_(drv.actor.cfg.noise_std_min)
    mpc = drv.mpc

    res = mpc.plan(q, qd)
    cmd_deployed = res.cmd.clone()
    # replay the same tick through the driver's command path with zero noise
    from isaac_rl_ppo.actor import executed_command
    zero = torch.zeros_like(cmd_deployed)
    cmd_ppo = executed_command(cmd_deployed, zero, prev_cmd=mpc.prev_cmd,
                               step_max=mpc.step_max, q_lo=mpc.q_lo, q_hi=mpc.q_hi)
    assert torch.equal(cmd_deployed, cmd_ppo)


# ==================================================== storage / GAE / update ==
def test_stack_and_index_dataclass_roundtrip(scheduled):
    sched, _kit, B = scheduled
    ctxs = [sched.context() for _ in range(3)]
    flat = stack_dataclass(ctxs)
    assert flat.drum_pos.shape[0] == 3 * B
    picked = index_dataclass(flat, torch.tensor([0, B, 2 * B]))
    assert picked.drum_pos.shape[0] == 3
    assert torch.equal(picked.drum_pos[0], ctxs[0].drum_pos[0])


def test_gae_matches_the_closed_form_on_a_toy_case():
    """No dones, constant reward: A_t is the analytic geometric sum."""
    T, B, gamma, lam = 4, 2, 0.9, 0.8
    rewards = torch.ones(T, B)
    values = torch.zeros(T, B)
    dones = torch.zeros(T, B, dtype=torch.bool)
    last = torch.zeros(B)

    returns, adv = compute_gae(rewards, values, dones, last, gamma=gamma, lam=lam)
    want = torch.zeros(T, B)
    acc = 0.0
    for t in reversed(range(T)):
        acc = 1.0 + gamma * lam * acc
        want[t] = acc
    assert torch.allclose(adv, want, atol=1e-6)
    assert torch.allclose(returns, adv, atol=1e-6)      # values are zero


def test_gae_cuts_the_bootstrap_on_done():
    T, B = 3, 1
    rewards = torch.ones(T, B)
    values = torch.full((T, B), 5.0)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1] = True
    _returns, adv = compute_gae(rewards, values, dones, torch.zeros(B),
                                gamma=0.99, lam=0.95)
    assert float(adv[1]) == pytest.approx(1.0 - 5.0, abs=1e-6)


def test_collect_then_update_runs_and_is_finite(driver, run_cfg):
    """The whole iteration: collect -> GAE -> update.  Nothing NaN, LoRA moves."""
    drv, q, qd, cost = driver
    actor = drv.actor
    B = q.shape[0]
    steps = 6

    storage = PpoStorage(device=torch.device("cpu"), horizon=actor.H, n_dof=17)
    q, qd, logs = drv.collect(q, qd, steps, storage)
    assert storage.num_steps == steps
    assert np.isfinite(logs["reward/total"])

    with torch.no_grad():
        last_obs = drv.privileged_obs(q, qd, drv.mpc.last_ref0,
                                      drv.sched.context(), drv.sched.hit_index())
        last_values = actor.critic(last_obs)
    rewards = torch.stack(storage.rewards, 0)
    values = torch.stack(storage.values, 0)
    dones = torch.stack(storage.dones, 0)
    storage.returns, storage.advantages = compute_gae(
        rewards, values, dones, last_values,
        gamma=run_cfg.ppo.gamma, lam=run_cfg.ppo.lam)
    storage.finalize()
    assert storage.flat["init"].shape == (steps * B, actor.H, 17)

    before = {n: p.detach().clone() for n, p in actor.named_parameters()
              if p.requires_grad}
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=cost)
    out = upd.update(storage)

    for k, v in out.items():
        assert np.isfinite(v), f"{k} = {v}"
    assert out["kl"] >= 0.0
    moved = [n for n, p in actor.named_parameters()
             if p.requires_grad and not torch.equal(p.detach(), before[n])]
    assert any(".lora_" in n for n in moved), "no LoRA parameter moved"
    assert any(n.startswith("critic.") for n in moved), "the critic did not train"
    actor.assert_frozen()
    assert "orth_cos" in out
    assert abs(out["orth_cos"]) <= 1.0 + 1e-5


def test_ratio_is_one_before_any_gradient_step(driver, run_cfg):
    """The recomputed mean must reproduce the collected one.

    This is the single most informative check on the whole replay path: it says
    the stored ``(init, state, ctx, hit_index)`` really do reconstruct the exact
    forward pass collection ran, so the importance ratio starts at 1 and PPO's
    first epoch is unbiased.  A warm start stored one tick out of phase, a mask
    rebuilt from the wrong hit index, or an autocast dtype mismatch between
    collection and update all show up here and essentially nowhere else.
    """
    drv, q, qd, cost = driver
    actor = drv.actor
    storage = PpoStorage(device=torch.device("cpu"), horizon=actor.H, n_dof=17)
    q, qd, _ = drv.collect(q, qd, 4, storage)

    storage.returns = torch.zeros(4, q.shape[0])
    storage.advantages = torch.zeros(4, q.shape[0])
    storage.finalize()

    from isaac_rl_ppo.actor import MpcInputs
    from isaac_rl_ppo.ppo_loop import index_dataclass
    flat = storage.flat
    n = flat["actions"].shape[0]
    sel = torch.arange(n)
    inputs = MpcInputs(init=flat["init"], state=flat["state"],
                       ctx=flat["ctx"], hit_index=flat["hit_index"])
    with torch.no_grad():
        plan = actor.plan(
            inputs, encoding=index_dataclass(flat["encoding"], sel))
        dist = actor.update_distribution(plan.mean)
        logp = dist.log_prob(flat["actions"]).sum(dim=-1)

    mean_err = (plan.mean - flat["mu"]).abs().max()
    assert float(mean_err) < 1e-5, f"recomputed mean differs by {float(mean_err):.2e}"
    ratio = torch.exp(logp - flat["log_probs"])
    assert float((ratio - 1.0).abs().max()) < 1e-4, \
        f"ratio at epoch 0 is {ratio.min():.6f}..{ratio.max():.6f}, expected 1"


def test_adaptive_kl_controller_moves_the_learning_rate(actor, run_cfg):
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=None)
    lr0 = upd.learning_rate

    upd._adapt_lr(run_cfg.ppo.desired_kl * 10.0)          # far too big a step
    assert upd.learning_rate == pytest.approx(lr0 / 1.5)
    upd._adapt_lr(run_cfg.ppo.desired_kl / 100.0)         # far too small
    assert upd.learning_rate == pytest.approx(lr0)

    for _ in range(60):                                    # rail it
        upd._adapt_lr(1.0)
    assert upd.learning_rate == pytest.approx(run_cfg.ppo.lr_min)
    assert upd.lr_floor_hits > 0, "the floor must be counted, not hidden"


def test_first_minibatch_low_kl_does_not_raise_lr(actor, run_cfg):
    """Replay-matched minibatch 0 has KL≈0; raising then ratchets the LR."""
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=None)
    lr0 = upd.learning_rate
    upd._adapt_lr(run_cfg.ppo.desired_kl / 100.0, allow_increase=False)
    assert upd.learning_rate == pytest.approx(lr0)
    upd._adapt_lr(run_cfg.ppo.desired_kl / 100.0, allow_increase=True)
    assert upd.learning_rate == pytest.approx(min(run_cfg.ppo.lr_max, lr0 * 1.5))


def test_micro_batching_matches_a_single_pass(driver, run_cfg):
    """Gradient accumulation is a memory knob, not an algorithmic one."""
    drv, q, qd, cost = driver
    actor = drv.actor
    storage = PpoStorage(device=torch.device("cpu"), horizon=actor.H, n_dof=17)
    q, qd, _ = drv.collect(q, qd, 4, storage)
    T, B = 4, q.shape[0]
    storage.returns = torch.randn(T, B)
    storage.advantages = torch.randn(T, B)
    storage.finalize()

    def grads(micro):
        torch.manual_seed(0)
        cfg = PpoRunConfig(**{**run_cfg.__dict__})
        cfg.ppo.num_learning_epochs = 1
        cfg.ppo.num_mini_batches = 1
        cfg.ppo.micro_batch_size = micro
        cfg.ppo.schedule = "fixed"
        upd = PpoUpdater(actor, cfg, device="cpu", cost=cost)
        actor.zero_grad(set_to_none=True)
        upd._begin_balanced_head_grads()
        idx = torch.arange(storage.flat["actions"].shape[0])
        adv = storage.flat["advantages"]
        mb = idx.shape[0]
        for s in range(0, mb, micro or mb):
            sel = idx[s: s + (micro or mb)]
            upd._chunk_loss(storage.flat, sel, adv,
                            weight=sel.shape[0] / mb)
        upd._finish_balanced_head_grads()
        return {n: p.grad.detach().clone() for n, p in actor.named_parameters()
                if p.requires_grad and p.grad is not None}

    whole = grads(0)
    chunked = grads(4)
    assert set(whole) == set(chunked)
    # Compared RELATIVE TO EACH TENSOR'S OWN SCALE, not elementwise: summing the
    # same terms in a different order is exact only up to fp32 reassociation,
    # and an elementwise rtol would flag the small entries of a tensor whose
    # largest entry is ~2.6e4.  The bound below is ~50x fp32 epsilon.
    worst = max((float((whole[n] - chunked[n]).abs().max()
                       / whole[n].abs().max().clamp_min(1e-12)), n)
                for n in whole)
    assert worst[0] < 1e-5, f"micro-batching changed the gradient: {worst}"


def test_orth_update_trains_lora_std_and_critic(driver, run_cfg):
    """Decoder LoRA, Gaussian std, and the critic all receive a gradient.

    The combine is traj-space: LoRA sees cost + orth(RL), std sees raw PPO,
    the critic sees only the value loss.  A combine that dropped any of those
    three would look like a healthy loss curve and train nothing useful.
    """
    drv, q, qd, cost = driver
    actor = drv.actor
    storage = PpoStorage(device=torch.device("cpu"), horizon=actor.H, n_dof=17)
    q, qd, _ = drv.collect(q, qd, 4, storage)
    T, B = 4, q.shape[0]
    storage.returns = torch.randn(T, B)
    storage.advantages = torch.randn(T, B)
    storage.finalize()

    assert run_cfg.orth.enabled
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=cost)
    actor.zero_grad(set_to_none=True)
    upd._begin_balanced_head_grads()
    idx = torch.arange(storage.flat["actions"].shape[0])
    upd._chunk_loss(storage.flat, idx, storage.flat["advantages"], weight=1.0)
    upd._finish_balanced_head_grads()

    lora = [p for n, p in actor.named_parameters()
            if ".lora_" in n and p.grad is not None
            and float(p.grad.abs().sum()) > 0]
    assert lora, "orth combine reached no LoRA parameter"
    assert actor.std.grad is not None and float(actor.std.grad.abs().sum()) > 0
    assert actor.mean_correction is not None
    mc_hit = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in actor.mean_correction.parameters())
    assert mc_hit, "the mean-correction head received no PPO gradient"
    critic_hit = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in actor.critic.parameters())
    assert critic_hit, "the critic received no value-loss gradient"
    actor.assert_frozen()


def test_scalar_fusion_still_runs_when_orth_disabled(driver, run_cfg):
    """``--no-orth`` is the A/B back to ``surrogate + aux.weight * cost``."""
    drv, q, qd, cost = driver
    actor = drv.actor
    storage = PpoStorage(device=torch.device("cpu"), horizon=actor.H, n_dof=17)
    q, qd, _ = drv.collect(q, qd, 4, storage)
    T, B = 4, q.shape[0]
    storage.returns = torch.randn(T, B)
    storage.advantages = torch.randn(T, B)
    storage.finalize()

    run_cfg.orth.enabled = False
    run_cfg.ppo.num_learning_epochs = 1
    run_cfg.ppo.num_mini_batches = 1
    run_cfg.ppo.schedule = "fixed"
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=cost)
    out = upd.update(storage)
    for k, v in out.items():
        assert np.isfinite(v), f"{k} = {v}"
    assert out["orth_cos"] == 0.0
