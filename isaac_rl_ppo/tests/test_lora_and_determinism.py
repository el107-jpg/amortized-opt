"""LoRA isolation, parameter counts, determinism, and aux-cost gradient flow.

Two of these are correctness *preconditions* rather than quality checks:

* **Determinism.**  The Gaussian head's log-prob is only the true sampling
  density if refinement is exactly deterministic given ``(obs, warm start)``.
  Any stochasticity inside the MPC pass silently invalidates every importance
  ratio in the run.
* **LoRA isolation.**  A fine-tune that quietly trains all 19.3 M backbone
  parameters is invisible in the loss curve.  Only the gradient check and the
  parameter count can see it.
"""

from __future__ import annotations

import pytest
import torch

from isaac_rl_ppo.actor import AmortizedPpoActor, LoRALinear, apply_lora
from isaac_rl_ppo.tests.conftest import make_inputs


# ==================================================================== LoRA ====
def test_lora_wraps_the_intended_matrices(actor):
    """Attention OUT-projections and both FFN matrices, decoder only."""
    kinds = {p.rsplit(".", 1)[-1] for p in actor.lora_paths}
    assert kinds == {"o_proj", "w12", "w3"}
    # 2 layers x (self_attn.o_proj + cross_attn.o_proj + ffn.w12 + ffn.w3)
    assert len(actor.lora_paths) == 8
    # never the encoder
    enc = [n for n, m in actor.model.encoder.named_modules()
           if isinstance(m, LoRALinear)]
    assert enc == [], f"LoRA leaked into the encoder: {enc}"
    # never q/k/v
    assert not any(k in kinds for k in ("q_proj", "k_proj", "v_proj"))


def test_lora_is_identity_at_initialisation(model, run_cfg):
    """B is zero-initialised, so the wrapped model starts EXACTLY at the base.

    This is what lets the fine-tune begin from the pretrained policy's
    behaviour bit for bit -- and it is what makes 'the mean at step 0 is the
    champion's plan' a true statement rather than an approximation.
    """
    torch.manual_seed(0)
    base = model.decoder.blocks[0].ffn.w3
    x = torch.randn(8, base.in_features)           # SwiGLU hidden = ffn_mult * d
    want = base(x).clone()

    apply_lora(model.decoder, rank=4, alpha=16.0, targets=("w3",))
    got = model.decoder.blocks[0].ffn.w3(x)
    assert torch.allclose(want, got, atol=0, rtol=0)


#: The CHAMPION's architecture, as recorded in ``runs/phase2b_main/best.pt``'s
#: own ``model_cfg``.  Written out explicitly rather than taken from
#: ``ModelConfig()``'s defaults because those defaults are library state that
#: other sessions edit -- ``dec_layers`` moved 6 -> 3 mid-build, which would
#: silently change this test's subject from the shipped 19.32 M backbone to a
#: 10.8 M one.  ``train_ppo.py`` is immune by construction: it builds the model
#: from ``ck["model_cfg"]``, i.e. from the checkpoint, never from the defaults.
CHAMPION_ARCH = dict(max_horizon=50, d_ctx=256, ctx_layers=2, ctx_heads=8,
                     d_model=384, dec_layers=6, dec_heads=8, dec_ffn_mult=4.0,
                     num_drums=8, num_future=3)


def test_real_model_param_counts(run_cfg):
    """The counts that actually ship, on the real 19.32 M backbone.

    The tiny fixture model cannot answer this: at d_model = 64 a rank-4 LoRA is
    a large fraction of the layer, whereas at the real d_model = 384 it is
    ~1.7 %.  This is the number the run reports at startup.
    """
    import amortized_opt as A

    torch.manual_seed(0)
    real = A.AmortizedOptimizer(A.ModelConfig(**CHAMPION_ARCH))
    cfg = run_cfg
    cfg.actor.lora_rank = 8
    act = AmortizedPpoActor(real, cfg.actor, cfg.critic, device="cpu")
    c = act.param_counts()

    assert c["frozen"] == pytest.approx(19_320_296, rel=1e-6), c
    assert c["lora"] == pytest.approx(331_776, rel=1e-6), c
    assert c["trainable"] < 1_000_000, c
    assert c["frozen"] > 30 * c["lora"], "LoRA must be a small slice of the backbone"
    # 6 decoder layers x (2 attention o_proj + w12 + w3)
    assert len(act.lora_paths) == 24


def test_trainable_parameter_count_is_sane(actor):
    counts = actor.param_counts()
    assert counts["frozen"] > counts["lora"]
    assert counts["trainable"] < 19_000_000, "trainable must be << 19 M"
    # every trainable tensor is a LoRA factor, the head, or the critic
    for name, p in actor.named_parameters():
        if not p.requires_grad:
            continue
        assert (".lora_a" in name or ".lora_b" in name
                or name in ("std",) or name.startswith("mean_correction.")
                or name.startswith("critic.")), f"unexpected trainable: {name}"


def test_frozen_base_receives_no_gradient(actor, scheduled):
    """After a backward, every frozen backbone parameter has None/zero grad."""
    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B)

    plan = actor.plan(inputs)
    plan.mean.square().sum().backward()

    actor.assert_frozen()                       # raises if any frozen p has grad
    grads = [(n, p.grad) for n, p in actor.model.named_parameters()
             if not p.requires_grad]
    assert grads, "the backbone must actually be frozen"
    assert all(g is None or float(g.abs().sum()) == 0.0 for _n, g in grads)

    # ...and the LoRA factors DID get one, or the test proves nothing
    lora_b = [p for n, p in actor.named_parameters()
              if ".lora_b" in n and p.requires_grad]
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in lora_b)


# ============================================================= determinism ====
def test_plan_is_deterministic(actor, scheduled):
    """Two passes with the same (obs, warm start) give identical means.

    Bit-identical, not 'close': the refinement operator is a deterministic
    feature extractor by requirement, and a tolerance here would hide exactly
    the dropout / sampling / uninitialised-buffer bugs the test exists for.
    """
    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B, seed=7)

    actor.eval()
    with torch.no_grad():
        a = actor.plan(inputs)
        b = actor.plan(inputs)
    assert torch.equal(a.mean, b.mean)
    assert torch.equal(a.traj, b.traj)


def test_plan_is_deterministic_in_train_mode(actor, scheduled):
    """The update runs in train mode; it must be deterministic there too.

    ``lora_dropout`` defaults to 0.0 precisely so that train and eval agree.
    A non-zero dropout would make the recomputed mean differ from the collected
    one and put a bias in every importance ratio.
    """
    assert actor.cfg.lora_dropout == 0.0
    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B, seed=8)

    actor.train()
    with torch.no_grad():
        a = actor.plan(inputs)
        b = actor.plan(inputs)
    assert torch.equal(a.mean, b.mean)


def test_act_inference_equals_the_plan_mean(actor, scheduled):
    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B, seed=9)
    with torch.no_grad():
        assert torch.equal(actor.act_inference(inputs), actor.plan(inputs).mean)


def test_mean_correction_starts_as_the_identity(actor, scheduled):
    """Zero-initialised correction => mean == the plan's own first waypoint."""
    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B, seed=10)
    with torch.no_grad():
        plan = actor.plan(inputs)
    assert actor.mean_correction is not None
    assert torch.equal(plan.mean, plan.raw_first)


# ================================================== aux-cost gradient flow ====
def test_aux_cost_gradients_reach_the_lora_parameters(actor, scheduled, run_cfg):
    """The analytic cost must actually train the subspace it is supposed to.

    Without this the aux term is decoration: it would be computed, logged, and
    have no effect on any weight.
    """
    from amortized_opt.cost import FusedCost
    from amortized_opt.train_cost import phase2_cost_config

    from isaac_rl_ppo.ppo_loop import PpoUpdater

    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B, seed=11)
    strike = sched.strike_context()

    cost = FusedCost(phase2_cost_config(), max_horizon=actor.H, device="cpu")
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=cost)

    plan = actor.plan(inputs)
    aux = upd._aux_cost(plan.traj, plan.elem_mask, inputs.state, strike)
    assert torch.isfinite(aux) and float(aux) > 0.0

    actor.zero_grad(set_to_none=True)
    aux.backward()

    touched = [n for n, p in actor.named_parameters()
               if ".lora_" in n and p.grad is not None
               and float(p.grad.abs().sum()) > 0]
    assert touched, "the aux cost reached no LoRA parameter"
    actor.assert_frozen()
    # it must NOT train the critic
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for p in actor.critic.parameters())


def test_policy_checkpoint_round_trips(actor, scheduled, run_cfg, tmp_path):
    """Save/load restores every trainable tensor exactly.

    ``policy_state_dict`` deliberately stores ONLY the trainable parameters plus
    the critic's normaliser buffers -- the 19.32 M backbone is reloaded from the
    pretrained checkpoint instead, which keeps a resume file at ~7 MB rather
    than ~240 MB.  The consequence is that ``load_state_dict(strict=False)``
    reports the backbone as "missing", so the count of missing keys is expected
    to be large and the load is only correct if ``unexpected`` is EMPTY and
    every trainable tensor comes back bit-identical.  Both are asserted here.
    """
    import amortized_opt as A

    from isaac_rl_ppo.actor import AmortizedPpoActor
    from isaac_rl_ppo.tests.conftest import TINY

    torch.manual_seed(0)
    with torch.no_grad():                       # move off the init values
        actor.std.mul_(0.8)
        for n, p in actor.named_parameters():
            if ".lora_b" in n or n.startswith("critic."):
                p.add_(0.01 * torch.randn_like(p))
    saved = {k: v.clone() for k, v in actor.policy_state_dict().items()}
    torch.save(saved, tmp_path / "policy.pt")

    fresh = AmortizedPpoActor(
        A.AmortizedOptimizer(A.ModelConfig(**TINY)),
        run_cfg.actor, run_cfg.critic, device="cpu")
    missing, unexpected = fresh.load_state_dict(
        torch.load(tmp_path / "policy.pt", weights_only=False), strict=False)

    assert list(unexpected) == [], f"unexpected keys on resume: {unexpected}"
    # every saved (trainable) tensor is restored exactly
    got = fresh.state_dict()
    for k, v in saved.items():
        assert torch.equal(got[k], v), f"{k} did not round-trip"
    # and the "missing" keys are ONLY frozen backbone tensors
    frozen = {n for n, p in fresh.named_parameters() if not p.requires_grad}
    for m in missing:
        assert m in frozen or m.startswith("model."), \
            f"a TRAINABLE tensor was missing from the checkpoint: {m}"


def test_aux_cost_term_selection_rejects_a_typo(actor, scheduled, run_cfg):
    from amortized_opt.cost import FusedCost
    from amortized_opt.train_cost import phase2_cost_config

    from isaac_rl_ppo.ppo_loop import PpoUpdater

    sched, _kit, B = scheduled
    inputs = make_inputs(actor, sched, B, seed=12)
    cost = FusedCost(phase2_cost_config(), max_horizon=actor.H, device="cpu")
    run_cfg.aux.terms = ("collsion",)                      # deliberate typo
    upd = PpoUpdater(actor, run_cfg, device="cpu", cost=cost)
    plan = actor.plan(inputs)
    with pytest.raises(KeyError, match="matched no FusedCost group"):
        upd._aux_cost(plan.traj, plan.elem_mask, inputs.state,
                      sched.strike_context())
