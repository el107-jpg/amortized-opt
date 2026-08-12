"""Per-env parameter-space perturbation: the math, without Isaac.

The properties that matter for the plan's exploration design and that would
fail silently if broken:

* ``sigma = 0`` must reproduce the unperturbed network BIT FOR BIT -- otherwise
  every "clean" eval episode and every antithetic reference is quietly
  perturbed;
* the perturbation must be genuinely PER ENV inside ONE shared forward pass;
* antithetic pairs must stay paired across partial redraws (a hit segment ends
  on one env of a pair but not the other);
* the FiLM and LoRA subspace dimensions must be what the plan budgeted
  (~10k-200k, not ~10M).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from isaac_rl.perturb import (  # noqa: E402
    AdaptiveSigma,
    AdaptiveSigmaConfig,
    NoiseConfig,
    ParameterNoise,
    antithetic_normal,
    deviation_report,
)

H, B = 12, 8


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return A.build_model(A.ModelConfig(max_horizon=H, dec_layers=2, d_model=64,
                                       d_ctx=32, ctx_layers=1, dec_heads=4,
                                       ctx_heads=4), "cpu")


def _inputs(model, batch=B):
    cfg = model.cfg
    ctx = A.ContextInputs.zeros(batch, num_drums=cfg.num_drums,
                                num_future=cfg.num_future)
    ctx.drum_valid[:] = True
    ctx.goal_valid[:] = True
    ctx.goal_tth[:] = 0.2
    t = torch.full((batch,), 0.2)
    masks = A.build_waypoint_masks(t, t, max_horizon=H, dt=cfg.dt)
    init = torch.zeros(batch, H, cfg.n_dof)
    state = torch.zeros(batch, 2 * cfg.n_dof)
    return ctx, masks, init, state


def _forward(model, ctx, masks, init, state):
    with torch.no_grad():
        return model(init, state, model.encode_context(ctx), masks).traj


# ---------------------------------------------------------------- identity --
@pytest.mark.parametrize("mode", ["film", "lora"])
def test_sigma_zero_is_bit_identical(model, mode):
    ctx, masks, init, state = _inputs(model)
    base = _forward(model, ctx, masks, init, state)
    n = ParameterNoise.build(model, B, NoiseConfig(mode=mode, sigma=0.0, rank=4))
    n.attach()
    try:
        out = _forward(model, ctx, masks, init, state)
    finally:
        n.detach()
    assert torch.equal(base, out), "sigma=0 must not touch the network at all"


@pytest.mark.parametrize("mode", ["film", "lora"])
def test_detach_restores_the_model(model, mode):
    ctx, masks, init, state = _inputs(model)
    base = _forward(model, ctx, masks, init, state)
    n = ParameterNoise.build(model, B, NoiseConfig(mode=mode, sigma=0.05, rank=4))
    n.attach()
    n.redraw(None)
    pert = _forward(model, ctx, masks, init, state)
    n.detach()
    assert not torch.allclose(base, pert), "sigma>0 must actually change the output"
    assert torch.equal(base, _forward(model, ctx, masks, init, state))


# ------------------------------------------------------------- per-env-ness --
@pytest.mark.parametrize("mode", ["film", "lora"])
def test_perturbation_is_per_env_in_one_forward(model, mode):
    """Identical inputs in every row, different eps -> different rows out.

    This is the property that makes per-env exploration feasible at 256 envs:
    one set of weights, one forward pass, N different optimizers.
    """
    ctx, masks, init, state = _inputs(model)
    n = ParameterNoise.build(model, B, NoiseConfig(mode=mode, sigma=0.05, rank=4,
                                                   antithetic=False, seed=3))
    n.attach()
    try:
        out = _forward(model, ctx, masks, init, state)
    finally:
        n.detach()
    spread = (out - out.mean(dim=0, keepdim=True)).abs().max()
    assert spread > 1e-6, f"rows are identical -> the eps is not per-env ({spread})"


def test_zero_sigma_env_runs_the_clean_policy(model):
    """Per-env sigma: the way "periodic unperturbed eval episodes" is expressed."""
    ctx, masks, init, state = _inputs(model)
    clean = _forward(model, ctx, masks, init, state)
    n = ParameterNoise.build(model, B, NoiseConfig(mode="film", sigma=0.05, seed=1))
    n.set_sigma(0.0, envs=torch.tensor([0, 1]))
    n.attach()
    try:
        out = _forward(model, ctx, masks, init, state)
    finally:
        n.detach()
    assert torch.allclose(out[:2], clean[:2], atol=1e-6)
    assert not torch.allclose(out[2:], clean[2:], atol=1e-6)


# -------------------------------------------------------------- antithetic --
def test_antithetic_draw_pairs_rows():
    x = antithetic_normal(8, (5,), generator=torch.Generator().manual_seed(0))
    for k in range(4):
        assert torch.equal(x[2 * k], -x[2 * k + 1])


def test_antithetic_odd_batch_leaves_last_row_independent():
    x = antithetic_normal(5, (3,), generator=torch.Generator().manual_seed(0))
    for k in range(2):
        assert torch.equal(x[2 * k], -x[2 * k + 1])
    assert x.shape == (5, 3)


def test_partial_redraw_keeps_pairs_paired(model):
    """A hit segment ends on one env of a pair; the pairing must survive.

    Redrawing only that env would leave ``eps_{2k} != -eps_{2k+1}`` and quietly
    destroy the variance reduction the antithetic estimator was set up for.
    """
    n = ParameterNoise.build(model, 8, NoiseConfig(mode="film", sigma=0.05, seed=2))
    n.redraw(torch.tensor([2]))          # only ONE member of the pair (2, 3)
    e = n.epsilon()
    assert torch.allclose(e[2], -e[3], atol=1e-6)


# ------------------------------------------------------------------- dims ---
def test_subspace_dimension_is_low_rank(model):
    """The plan budgets ~10k-200k effective dims, not the model's ~10M."""
    total = sum(p.numel() for p in model.decoder.parameters())
    film = ParameterNoise.build(model, 2, NoiseConfig(mode="film")).dim
    lora = ParameterNoise.build(model, 2, NoiseConfig(mode="lora", rank=8)).dim
    assert film == 2 * model.cfg.dec_layers * model.cfg.d_model
    assert film < total / 10
    assert lora < total


def test_lora_rank_scales_the_dimension(model):
    d4 = ParameterNoise.build(model, 2, NoiseConfig(mode="lora", rank=4)).dim
    d8 = ParameterNoise.build(model, 2, NoiseConfig(mode="lora", rank=8)).dim
    assert d8 == 2 * d4


def test_encoder_is_never_perturbed(model):
    """Plan item 1: "The context encoder stays unperturbed"."""
    ctx, masks, init, state = _inputs(model)
    with torch.no_grad():
        base = model.encode_context(ctx).memory.clone()
    n = ParameterNoise.build(model, B, NoiseConfig(mode="film", sigma=0.2, seed=5))
    n.attach()
    try:
        with torch.no_grad():
            after = model.encode_context(ctx).memory
    finally:
        n.detach()
    assert torch.equal(base, after)


# --------------------------------------------------------- adaptive sigma ---
def test_adaptive_sigma_moves_toward_the_target():
    a = AdaptiveSigma(AdaptiveSigmaConfig(target_mm=3.0, adapt_rate=1.1, ema=0.0))
    s0 = a.scale
    for _ in range(20):
        a.observe(0.1)               # far UNDER target -> scale must grow
    assert a.scale > s0
    b = AdaptiveSigma(AdaptiveSigmaConfig(target_mm=3.0, adapt_rate=1.1, ema=0.0))
    for _ in range(20):
        b.observe(50.0)              # far OVER target -> scale must shrink
    assert b.scale < 1.0


def test_adaptive_sigma_respects_bounds_and_nan():
    a = AdaptiveSigma(AdaptiveSigmaConfig(adapt_rate=10.0, scale_max=2.0, ema=0.0))
    for _ in range(10):
        a.observe(0.0)
    assert a.scale <= 2.0
    before = a.scale
    a.observe(float("nan"))
    assert a.scale == before


# ------------------------------------------------------------ diagnostics ---
def test_deviation_report_localizes_the_joints():
    """The wrist/contact diagnostic must actually see an injected concentration."""
    torch.manual_seed(0)
    names = A.G1_UPPER_BODY_JOINTS
    wrist = [i for i, n in enumerate(names) if "wrist" in n]
    a = torch.zeros(4, 10, len(names))
    b = a.clone()
    b[:, 6:9, wrist] = 0.01
    rep = deviation_report(a, b, hit_index=torch.full((4, 2), 9))
    assert rep["wrist_share"] > 0.99
    assert rep["contact_share"] > 0.99
    assert rep["by_horizon_rad"][0] == 0.0
