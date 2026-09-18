"""The torque PEAK term: a smooth max alongside the mean torque hinge.

``ok_torque`` is a zero-tolerance max against the hard effort limits, but
``raw["torque"]`` is a mean over ~850 ``(knot, joint)`` elements.  On the
recorded phase-2 failures the median offending case is over on ONE element
(0.27 % of the valid ones) by 0.87 N*m -- a ~0.3 % move in the mean for a
guaranteed rejection.  These tests pin the fix: :func:`amortized_opt.cost.
smooth_max` tracks the true max to within a few percent where the mean is off
by ~1/850, concentrates its gradient on the offender, respects the mask
bit-exactly, and costs nothing when ``w_torque_peak`` is left at its 0 default.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext, smooth_max

# The real shape of the reduction: (max_horizon + 1) knots x 17 joints, of
# which the per-sample horizon mask leaves ~50 x 17 = 850 live.
KNOTS, JOINTS = 60, 17
N_VALID = 850


# ------------------------------------------------------------------ setup --
def _synthetic_ctx(B, device, dtype, *, kl, kr, active=(1.0, 1.0)):
    """Same minimal context ``tests/test_cost.py`` uses."""
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=torch.tensor(active, device=device, dtype=dtype).expand(B, 2).contiguous(),
        target_drum=torch.zeros(B, 2, device=device, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, device=device, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, device=device, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, device=device, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, device=device, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, device=device, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, device=device, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        elev_required=ones,
    )


def _one_violator(true_max: float, *, background: float = 0.0, B: int = 3,
                  dtype=torch.float64):
    """``(h, mask)`` with exactly one materially non-zero element per row.

    ``mask`` leaves ``N_VALID`` elements live, matching the real reduction, and
    the violator sits at a different ``(knot, joint)`` in every row.
    """
    h = torch.full((B, KNOTS, JOINTS), background, dtype=dtype)
    mask = torch.zeros(B, KNOTS, JOINTS, dtype=dtype)
    n_knots = N_VALID // JOINTS  # 50
    mask[:, :n_knots, :] = 1.0
    spots = [(3, 0), (17, 9), (n_knots - 1, JOINTS - 1)][:B]
    for b, (k, j) in enumerate(spots):
        h[b, k, j] = true_max
    return h, mask, float(mask[0].sum())


# --------------------------------------------------- 1. fidelity vs. mean --
@pytest.mark.parametrize("mode", ["pnorm", "lse"])
@pytest.mark.parametrize("true_max", [0.9, 3.0, 12.0])
def test_single_violator_tracks_the_true_max(mode, true_max):
    """One violator in ~850 elements: the smooth max sees it, the mean does not."""
    h, mask, n_valid = _one_violator(true_max)
    assert abs(n_valid - N_VALID) < 1.0

    peak = smooth_max(h, mask, mode, 12.0, 0.05)
    mean = (h * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
    true = (h * mask).amax(dim=(1, 2))

    rel = (peak / true - 1.0).abs()
    assert float(rel.max()) < 0.10, (mode, true_max, float(rel.max()))
    assert bool((peak >= 0.9 * true).all())

    # ... while the MEAN is ~1/n_valid of the thing the gate looks at.
    expect = true / n_valid
    ratio = mean / expect
    assert bool(((ratio > 0.5) & (ratio < 2.0)).all()), float(ratio.max())
    print(f"[{mode} max={true_max}] true {float(true[0]):.4f} "
          f"peak {float(peak[0]):.4f} (x{float(peak[0]/true[0]):.4f}) "
          f"mean {float(mean[0]):.3e} (true/n = {float(expect[0]):.3e})")


@pytest.mark.parametrize("mode", ["pnorm", "lse"])
@pytest.mark.parametrize("background", [0.005, 0.02, 0.1])
def test_the_softplus_floor_does_not_drown_the_peak(mode, background):
    """Uniform background on a synthetic array (the reduction, not the hinge).

    A 0.02 background on a ReLU peak is only from actual ReLU, not a
    softplus floor -- this test feeds ``smooth_max`` a hand-made ``h``.
    """
    h, mask, n_valid = _one_violator(3.0, background=background)
    true = (h * mask).amax(dim=(1, 2))
    peak = smooth_max(h, mask, mode, 12.0, 0.05)
    mean = (h * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
    assert float((peak / true - 1.0).abs().max()) < 0.10
    print(f"[{mode} bg={background}] peak {float(peak[0]):.5f} vs true "
          f"{float(true[0]):.5f}; the mean is {float(mean[0]):.5f}, "
          f"{100 * float(background * (n_valid - 1) / (mean[0] * n_valid)):.1f} % "
          f"of it pure floor")


def test_both_modes_reported_side_by_side():
    """One printed row so the numbers in the commit message are reproducible."""
    h, mask, n_valid = _one_violator(3.0, B=1)
    true = float((h * mask).amax())
    pn = float(smooth_max(h, mask, "pnorm", 12.0, 0.05)[0])
    ls = float(smooth_max(h, mask, "lse", 12.0, 0.05)[0])
    mn = float((h * mask).sum() / mask.sum())
    print(f"n_valid={n_valid:.0f} true_max={true:.6f} pnorm={pn:.6f} "
          f"({pn/true:.4f}x) lse={ls:.6f} ({ls/true:.4f}x) "
          f"mean={mn:.6e} ({mn/true:.6f}x = 1/{true/mn:.0f})")
    assert pn >= true and ls >= true          # both are over-estimates
    assert pn / true < 1.10 and ls / true < 1.10
    assert mn / true < 0.01


# ------------------------------------------- 2. where the gradient lands --
@pytest.mark.parametrize("mode", ["pnorm", "lse"])
def test_gradient_concentrates_on_the_violator(mode):
    h, mask, n_valid = _one_violator(3.0, background=0.02, B=1)
    x = h.clone().requires_grad_(True)
    smooth_max(x, mask, mode, 12.0, 0.05).sum().backward()
    g = x.grad.abs()
    k, j = 3, 0                                # where ``_one_violator`` put it
    frac = float(g[0, k, j] / g.sum())
    assert frac >= 0.90, (mode, frac)

    # The mean spreads the very same gradient budget uniformly.
    y = h.clone().requires_grad_(True)
    ((y * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))).sum().backward()
    gm = y.grad.abs()
    assert float(gm[0, k, j] / gm.sum()) == pytest.approx(1.0 / n_valid, rel=1e-6)
    print(f"[{mode}] gradient on the one violator: {100 * frac:.4f} % "
          f"(mean reduction: {100 / n_valid:.4f} %)")


@pytest.mark.parametrize("mode", ["pnorm", "lse"])
def test_all_zero_row_is_finite_with_finite_gradient(mode):
    """A perfectly feasible row must not produce ``0 ** (1/p)`` infinities."""
    h = torch.zeros(2, KNOTS, JOINTS, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, KNOTS, JOINTS, dtype=torch.float64)
    peak = smooth_max(h, mask, mode, 12.0, 0.05)
    assert torch.isfinite(peak).all()
    peak.sum().backward()
    assert torch.isfinite(h.grad).all()
    if mode == "pnorm":
        assert float(peak.abs().max()) == 0.0

    # ... and so must a row with NO valid elements at all.
    h2 = torch.randn(2, KNOTS, JOINTS, dtype=torch.float64).abs().requires_grad_(True)
    dead = torch.zeros_like(mask)
    p2 = smooth_max(h2, dead, mode, 12.0, 0.05)
    assert torch.isfinite(p2).all() and float(p2.abs().max()) == 0.0
    p2.sum().backward()
    assert torch.isfinite(h2.grad).all()


# ------------------------------------------------------------ 3. masking --
@pytest.mark.parametrize("mode", ["pnorm", "lse"])
def test_masked_elements_contribute_exactly_nothing(mode):
    h, mask, _ = _one_violator(3.0, background=0.02, B=2)
    base = smooth_max(h, mask, mode, 12.0, 0.05)
    h2 = h.clone()
    h2[:, KNOTS - 1, :] = 1.0e4          # far outside every horizon
    assert float((mask[:, KNOTS - 1, :]).abs().max()) == 0.0
    other = smooth_max(h2, mask, mode, 12.0, 0.05)
    assert torch.equal(base, other), (mode, (base - other).abs().max())


# ---------------------------------------------------- 4. through the cost --
FIXTURE = __import__("pathlib").Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def cost_batch():
    """A real batch.

    The RECORDED trajopt solutions when the fixture is present -- that is the
    population the peak term has to behave sanely on, hinge maxima of a few
    N*m rather than the hundreds a joint-space random walk produces.  Falls back
    to a random walk so the file is self-contained.
    """
    npz, cjson = FIXTURE / "trajopt_hpipm_fixture.npz", \
        FIXTURE / "trajopt_hpipm_fixture_cases.json"
    if npz.exists() and cjson.exists():
        import json

        cases = [A.EvalCase.from_dict(d)
                 for d in json.loads(cjson.read_text())["cases"]]
        z = np.load(npz)
        trajs = {int(k.split("_")[1]): np.asarray(z[k], float)
                 for k in z.files if k.startswith("case_")}
        cases = [c for c in cases if c.case_id in trajs][:8]
        H, B = 40, len(cases)
        q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
        v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
        wp = torch.zeros(B, H, N_DOF, dtype=torch.float64)
        for i, c in enumerate(cases):
            Q = trajs[c.case_id]
            q0[i] = torch.as_tensor(Q[0], dtype=torch.float64)
            v0[i] = torch.as_tensor(c.qdot_start, dtype=torch.float64)
            wp[i, : c.N] = torch.as_tensor(Q[1 : c.N + 1], dtype=torch.float64)
            wp[i, c.N :] = torch.as_tensor(Q[c.N], dtype=torch.float64)
        return dict(q0=q0, wp=wp, v0=v0, cases=cases, H=H, B=B)

    torch.manual_seed(0)
    H, B = 30, 6
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.02, dim=1
    )
    v0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.5
    kl = torch.tensor([30, 22, 30, 17, 26, 30])
    kr = torch.tensor([25, 30, 14, 30, 30, 21])
    return dict(q0=q0, wp=wp, v0=v0, kl=kl, kr=kr, H=H, B=B)


def _ctx(b, cost, device="cpu", dtype=torch.float64):
    if "cases" in b:
        return A.strike_context_from_cases(b["cases"], drum_index=cost.drum_index,
                                           device=device, dtype=dtype)
    return _synthetic_ctx(b["B"], device, dtype,
                          kl=b["kl"].to(device), kr=b["kr"].to(device))


def _torque_hard(out):
    """Default-λ=1 sum of per-segment ReLU lives (the hard wall, no peak)."""
    from amortized_opt.lagrangian import ACTUATOR_SEGMENT_NAMES, limit_group_name
    return sum(
        out.violations[limit_group_name("torque", s)]
        for s in ACTUATOR_SEGMENT_NAMES
    )


def _fwd(cfg_kw, cost_batch, **fwd_kw):
    b = cost_batch
    cost = FusedCost(CostConfig(**cfg_kw), max_horizon=b["H"], device="cpu",
                     dtype=torch.float64)
    return cost, cost.forward(b["q0"], b["wp"], b["v0"], _ctx(b, cost), **fwd_kw)


def test_zero_weight_is_bit_identical_to_the_old_term(cost_batch):
    """Default ``w_torque_peak = 0`` is warning + per-segment live, no peak."""
    cost, out = _fwd({}, cost_batch)
    assert cost.cfg.w_torque_peak == 0.0          # the pinned default
    warning = cost.cfg.w_torque * out.raw["torque"]
    hard = _torque_hard(out)
    assert torch.allclose(out.terms["torque"], warning + hard, atol=1e-12)

    _, explicit = _fwd({"w_torque_peak": 0.0}, cost_batch)
    assert torch.equal(explicit.terms["torque"], out.terms["torque"])

    # doubling family λ scales the hard wall, not the warning
    _, doubled = _fwd({}, cost_batch, weights={"torque": 2.0})
    assert torch.allclose(cost.cfg.w_torque * doubled.raw["torque"], warning, atol=1e-12)
    assert torch.allclose(doubled.terms["torque"], warning + 2.0 * hard, atol=1e-12)


def test_peak_output_is_smooth_max_at_least_the_hard_max(cost_batch):
    """``raw["torque_peak"]`` is a smooth max of ReLU excess; AL viol is the max."""
    b = cost_batch
    cost, out = _fwd({"w_torque_peak": 1.0}, cost_batch)
    peak = out.raw["torque_peak"]
    true = out.diagnostics["torque_excess_max"]
    assert peak.shape == (b["B"],)
    assert torch.isfinite(peak).all()
    assert bool((peak >= 0.0).all())
    over = true > 0.0
    if bool(over.any()):
        assert bool((peak[over] >= true[over] - 1e-12).all())
    zero = ~over
    if bool(zero.any()):
        assert float(peak[zero].abs().max()) == 0.0
    assert torch.equal(out.violations["torque_peak"], true)


def test_peak_weight_strictly_raises_the_term(cost_batch):
    _, base = _fwd({}, cost_batch)
    _, hot = _fwd({"w_torque_peak": 1.0}, cost_batch)
    if bool((hot.raw["torque_peak"] > 0).any()):
        assert bool((hot.terms["torque"] >= base.terms["torque"]).all())
        assert bool((hot.terms["torque"] > base.terms["torque"]).any())
    # the curriculum knob multiplies only the peak half (λ_peak * w_peak * smooth_max)
    _, hot2 = _fwd({"w_torque_peak": 1.0}, cost_batch,
                   weights={"torque_peak": 3.0})
    expect = base.terms["torque"] + 3.0 * hot.raw["torque_peak"]
    assert torch.allclose(hot2.terms["torque"], expect, atol=1e-12)


def test_one_adam_step_on_the_peak_lowers_the_worst_element():
    """The whole point: the peak term is a usable descent direction on the MAX ReLU."""
    # A motionless hold is feasible (ReLU peak = 0, no gradient).  Displace
    # one wrist knot so exactly one (knot, joint) element is over the hard
    # effort limit -- the shape the peak group exists for.
    H, B = 12, 4
    wrist_cols = (8, 9, 15, 16)
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    for b in range(B):
        wp[b, 3 + b, wrist_cols[b % len(wrist_cols)]] = 0.10 * (-1.0) ** b
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    cost = FusedCost(CostConfig(w_torque=0.0, w_torque_peak=1.0),
                     max_horizon=H, device="cpu", dtype=torch.float64, dt=0.02)
    ctx = _synthetic_ctx(B, "cpu", torch.float64, kl=kl, kr=kr)
    wp = wp.clone().requires_grad_(True)

    def excess_max():
        with torch.no_grad():
            o = cost.forward(q0, wp, v0, ctx)
        return o.diagnostics["torque_excess_max"].clone()

    before = excess_max()
    assert bool((before > 0).all()), before
    opt = torch.optim.Adam([wp], lr=1.0e-4)
    opt.zero_grad(set_to_none=True)
    out = cost.forward(q0, wp, v0, ctx)
    out.terms["torque"].sum().backward()
    assert torch.isfinite(wp.grad).all()
    assert float(wp.grad.abs().max()) > 0.0
    opt.step()
    after = excess_max()
    print("torque_excess_max before -> after one Adam step: "
          f"{[round(float(x), 3) for x in before]} -> "
          f"{[round(float(x), 3) for x in after]}")
    assert bool((after < before).all()), (before, after)

    opt = torch.optim.Adam([wp], lr=1.0e-4)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        cost.forward(q0, wp, v0, ctx).terms["torque"].sum().backward()
        opt.step()
    far = excess_max()
    print(f"  after 21 steps: mean {float(before.mean()):.4f} -> "
          f"{float(far.mean()):.4f}")
    assert float(far.mean()) < float(before.mean())


# ----------------------------------------------------------- 5. autocast --
@pytest.mark.parametrize("mode", ["pnorm", "lse"])
def test_bf16_input_is_upcast_and_stays_accurate(mode, cuda_device):
    """A bf16 hinge (what autocast hands us) must not wreck the reduction."""
    h32, mask, _ = _one_violator(2.7182818, background=0.017, B=3,
                                 dtype=torch.float32)
    # ... plus a spread of values that bf16's 8 mantissa bits cannot hold
    # exactly, so this measures rounding and not luck.
    g = torch.Generator().manual_seed(0)
    h32 = h32 + torch.rand(h32.shape, generator=g, dtype=torch.float32) * 0.31
    h32, mask = h32.to(cuda_device), mask.to(cuda_device)
    ref = smooth_max(h32, mask, mode, 12.0, 0.05)
    got = smooth_max(h32.to(torch.bfloat16), mask.to(torch.bfloat16), mode, 12.0, 0.05)
    assert got.dtype == torch.float32 and torch.isfinite(got).all()
    rel = ((got - ref).abs() / ref.abs()).max()
    assert float(rel) < 0.05, float(rel)
    print(f"[{mode}] bf16 vs fp32 smooth_max: {float(rel) * 100:.3f} % rel err")


def test_autocast_forward_is_finite_and_close(cost_batch, cuda_device):
    b = cost_batch
    cost = FusedCost(CostConfig(w_torque_peak=1.0), max_horizon=b["H"],
                     device=cuda_device, dtype=torch.float32)
    ctx = _ctx(b, cost, device=cuda_device, dtype=torch.float32)
    args = [b["q0"].to(device=cuda_device, dtype=torch.float32),
            b["wp"].to(device=cuda_device, dtype=torch.float32),
            b["v0"].to(device=cuda_device, dtype=torch.float32)]
    ref = cost.forward(*args, ctx).raw["torque_peak"].detach()
    with torch.autocast("cuda", torch.bfloat16):
        got = cost.forward(*args, ctx).raw["torque_peak"].detach().float()
    assert torch.isfinite(got).all()
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max()
    print(f"autocast(bf16) end-to-end torque_peak rel err: {float(rel) * 100:.3f} %")
    # bf16 RNEA moves |tau| itself, and the hinge is a DIFFERENCE against the
    # effort limit, so this is looser than the pure-reduction check above.
    assert float(rel) < 0.05, float(rel)


# --------------------------------------------------------- 6. validation --
def test_bad_mode_is_rejected():
    with pytest.raises(ValueError, match="torque_peak_mode"):
        CostConfig(torque_peak_mode="softmax").validate()
    with pytest.raises(ValueError, match="torque_peak_p"):
        CostConfig(torque_peak_p=0.5).validate()
    with pytest.raises(ValueError, match="torque_peak_temp"):
        CostConfig(torque_peak_temp=0.0).validate()
    for good in ("pnorm", "lse"):
        assert CostConfig(torque_peak_mode=good).validate().torque_peak_mode == good
    with pytest.raises(ValueError, match="mode"):
        smooth_max(torch.zeros(2, 3, 4), torch.ones(2, 3, 4), "softmax")


def test_smooth_max_rejects_partial_dims():
    with pytest.raises(ValueError, match="trailing dims"):
        smooth_max(torch.zeros(2, 3, 4), torch.ones(2, 3, 4), dims=(1,))


# ------------------------------------------------- the mismatch, measured --
def test_the_mismatch_this_fix_addresses_is_real(cost_batch):
    """Document the ratio: ReLU peak / softplus mean on a batch that trips."""
    _, out = _fwd({"w_torque_peak": 1.0}, cost_batch)
    mean = out.raw["torque"].detach()
    peak = out.raw["torque_peak"].detach()
    cnt = out.counts["torque"]
    print("per-sample  n_valid / mean(softplus) / peak(ReLU) / ratio:")
    for i in range(mean.numel()):
        ratio = float(peak[i] / mean[i]) if float(mean[i]) else 0.0
        print(f"  {int(cnt[i]):4d}  {float(mean[i]):10.4f}  {float(peak[i]):10.4f}  "
              f"{ratio:8.1f}x")
    # feasible rows: ReLU peak is 0, softplus mean is the floor.  Violating
    # rows: peak exceeds the diluted mean.
    over = peak > 0
    if bool(over.any()):
        assert bool((peak[over] > mean[over]).all())
    else:
        assert float(peak.abs().max()) == 0.0
        assert bool((mean > 0).all())


def test_relu_peak_is_zero_on_a_feasible_hold():
    """A gravity-only hold: softplus mean is the floor, ReLU peak is 0."""
    H, B = 12, 2
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    kl = torch.full((B,), H, dtype=torch.long)
    cost = FusedCost(CostConfig(w_torque_peak=1.0), max_horizon=H,
                     device="cpu", dtype=torch.float64, dt=0.02)
    out = cost.forward(q0, wp, v0, _synthetic_ctx(B, "cpu", torch.float64, kl=kl, kr=kl))
    assert float(out.raw["torque"].min()) > 0.0
    assert float(out.raw["torque_peak"].abs().max()) == 0.0
    assert float(out.violations["torque_peak"].abs().max()) == 0.0
    assert torch.equal(out.violations["torque_peak"], out.diagnostics["torque_excess_max"])
