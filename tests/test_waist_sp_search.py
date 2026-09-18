"""Zeroth-order SEARCH DISTILLATION for the waist-yaw set-point head.

``Phase2Config.w_waist_aim`` prices ``||traj_yaw - sp||`` -- a CONSISTENCY
term.  Its gradient pulls the trajectory toward the set-point and the set-point
toward the trajectory, so the pair is free to co-drift; the only outcome
grounding is a deliberately weak tether to the analytic aim law, and on the
2026-08-14 grooves run that was not enough (``waist_sp_res`` ~1.0 rad by step
5 000, ~90 mm of systematic contact miss).  The gradient is also structurally
biased: the real objective for ``sp`` is the ENVELOPE V(psi) = cost after the
arms re-plan around torso yaw psi, and a gradient taken at a frozen trajectory
mostly says "do not break the plan you already have".

So evaluate V directly.  Five claims, each load-bearing:

1. **The candidate set is what it says.**  Symmetric ladder on the head's own
   set-point, reaching exactly ``span``, clamped to the waist-yaw box, with the
   analytic aim always present as the last column.
2. **The target is bounded by the candidates.**  It is a softmin-weighted
   convex mix, so runaway is structurally impossible rather than merely
   penalized -- and the reported V-spread is ``max - min`` of the raw scores.
3. **A candidate evaluation is an honest re-plan.**  Substituting the set-point
   token reproduces a full second ``encode_context`` bit for bit, and the
   candidate score never prices the ``waist_aim`` term the search exists to
   escape.
4. **Off is exactly off.**  At ``waist_sp_search_frac = 0.0`` the code path is
   unreachable, and a short run is bit-identical -- same losses, same
   parameters -- to one built before the knobs existed.
5. **It descends the envelope.**  Rig one candidate to win and the head's
   gradient moves the set-point toward it.
"""

from __future__ import annotations

import json

import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import CONTROL_DT, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.model import AmortizedOptimizer, ModelConfig
from amortized_opt.phase2_data import (
    Phase2Pool,
    Phase2ScenarioConfig,
    generate_cost_scenarios,
)
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks, pack_state
from amortized_opt.train_cost import (
    WAIST_SP_SELF_TERMS,
    Phase2Config,
    _waist_sp_search_term,
    encode,
    group_lengths,
    outcome_only_weights,
    phase2_cost_config,
    substitute_waist_sp,
    train_phase2,
    waist_sp_bound,
    waist_sp_candidates,
    waist_sp_offsets,
    waist_sp_softmin,
)

from conftest import pick_cuda_device

#: The core is pure torch, so every check here runs on CPU when the box has no
#: GPU -- the same fall-back ``tests/test_train_cost.py`` uses.  Nothing is
#: skipped for want of CUDA.
DEV = pick_cuda_device() or "cpu"
H = 20


def _model(seed: int = 0, **kw) -> AmortizedOptimizer:
    torch.manual_seed(seed)
    return AmortizedOptimizer(
        ModelConfig(dec_layers=1, ctx_layers=1, max_horizon=H, waist_sp_head=True, **kw)
    )


def _strike(B: int, ref=None, dtype=torch.float32) -> StrikeContext:
    """The minimal all-active strike context, batched (as in
    ``tests/test_waist_sp_head.py``)."""
    k = torch.full((B,), H - 1, dtype=torch.long)
    return StrikeContext(
        hit_index=torch.stack([k, k], dim=1),
        active=torch.ones(B, 2, dtype=dtype),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, dtype=dtype),
        elev_required=torch.ones(B, 2, dtype=dtype),
        waist_yaw_ref=ref,
    )


def _ctx(B: int = 4, *, aim=None) -> ContextInputs:
    ctx = ContextInputs.zeros(B, num_drums=8, num_future=3)
    ctx.drum_valid[:, :3] = True
    ctx.goal_valid[:] = True
    ctx.goal_pos[:, 0] = torch.tensor([0.4, 0.2, 0.8])
    ctx.goal_pos[:, 1] = torch.tensor([0.3, -0.2, 0.8])
    ctx.goal_tth[:] = 0.4
    ctx.yaw_now = torch.zeros(B)
    ctx.aim_yaw = torch.zeros(B) if aim is None else torch.as_tensor(aim).float()
    ctx.aim_valid = torch.ones(B, dtype=torch.bool)
    return ctx


# ================================================ 1. candidate construction ==
def test_offset_ladder_is_symmetric_and_reaches_the_span():
    offs = waist_sp_offsets(8, 0.8)
    assert offs[0] == 0.0
    assert len(offs) == 7                       # k - 1 slots, the aim takes one
    assert sorted(offs) == sorted(-o for o in offs)      # symmetric as a set
    assert max(offs) == pytest.approx(0.8)
    assert min(offs) == pytest.approx(-0.8)
    # dense near the incumbent: the inner rungs are closer together than the
    # outer ones, which is what keeps the softmin target moving smoothly
    mags = sorted({abs(o) for o in offs})
    assert mags[1] - mags[0] < mags[-1] - mags[-2]
    # an odd k rounds DOWN, it never silently over-spends the compute budget
    assert waist_sp_offsets(9, 0.8) == offs
    # the floor: even a degenerate k still brackets the incumbent
    assert waist_sp_offsets(2, 0.5) == [0.0, 0.5, -0.5]


def test_candidates_carry_the_aim_and_clamp_to_the_waist_box():
    sp = torch.tensor([0.0, 1.5, -1.7, 0.2])
    aim = torch.tensor([0.3, -0.4, 0.0, 9.0])   # last one is out of the box
    cand = waist_sp_candidates(sp, aim, k=8, span=0.8)

    assert cand.shape == (4, 8)                 # k total, aim included
    b = waist_sp_bound()
    assert b == pytest.approx(1.74533)          # head clamp, tighter than URDF
    assert float(cand.abs().max()) <= b + 1e-12

    # the analytic aim is MANDATORY and is the last column (clamped, like the
    # rest) -- the one candidate whose value is known to be defensible
    assert torch.allclose(cand[:, -1], aim.clamp(-b, b))
    # the incumbent is always in the set (offset 0, first column)
    assert torch.allclose(cand[:, 0], sp)
    # unclamped rows span exactly sp +- span
    assert float(cand[0, :-1].max()) == pytest.approx(float(sp[0]) + 0.8)
    assert float(cand[0, :-1].min()) == pytest.approx(float(sp[0]) - 0.8)
    # a clamped row is squashed, never wrapped past the box
    assert float(cand[2].min()) >= -b - 1e-12


# ==================================================== 2. the softmin target ==
def test_softmin_target_is_inside_the_candidate_span():
    g = torch.Generator().manual_seed(0)
    cand = torch.randn(64, 8, generator=g)
    scores = torch.randn(64, 8, generator=g).abs() * 1e4
    star, spread = waist_sp_softmin(cand, scores, 0.2)

    assert torch.all(star >= cand.min(dim=1).values - 1e-6)
    assert torch.all(star <= cand.max(dim=1).values + 1e-6)
    # the V-spread is the RAW score range, the "does the set-point matter"
    # diagnostic -- not the normalized one
    assert torch.allclose(spread, scores.max(1).values - scores.min(1).values)


def test_softmin_anneals_from_argmin_to_the_mean():
    cand = torch.tensor([[0.0, 0.5, -0.5, 1.0]])
    scores = torch.tensor([[10.0, 1.0, 12.0, 11.0]])        # column 1 wins
    hard, _ = waist_sp_softmin(cand, scores, 1e-3)
    soft, _ = waist_sp_softmin(cand, scores, 1e6)
    assert float(hard[0]) == pytest.approx(0.5, abs=1e-3)
    assert float(soft[0]) == pytest.approx(float(cand.mean()), abs=1e-3)


def test_a_flat_envelope_gives_a_finite_target():
    """V-spread exactly 0 means the set-point changes nothing.  The target must
    still be finite (it degenerates to the candidate mean), or a moot head
    would poison the whole loss with NaN."""
    cand = torch.tensor([[0.0, 0.5, -0.5, 1.0]])
    star, spread = waist_sp_softmin(cand, torch.full((1, 4), 7.0), 0.2)
    assert float(spread) == 0.0
    assert torch.isfinite(star).all()
    assert float(star) == pytest.approx(float(cand.mean()), abs=1e-6)


# =============================== 3. the candidate evaluation is an honest one ==
def test_substituted_encoding_matches_a_full_reencode():
    """The set-point token is appended AFTER the encoder blocks, so a candidate
    re-plan is one small Linear rather than a second pass -- but only if the
    substitution reproduces ``ContextEncoder.forward`` exactly.  If that token
    construction ever moves, this fails instead of the search silently scoring
    a stale set-point."""
    m = _model()
    with torch.no_grad():           # a non-trivial head, not the zero-init one
        m.encoder.waist_sp_out.weight.normal_(0.0, 0.05)
        m.encoder.waist_sp_out.bias.fill_(0.1)

    c1, c2 = _ctx(aim=[0.0, 0.2, -0.3, 0.5]), _ctx(aim=[0.7, -0.6, 0.9, -0.1])
    e1, e2 = m.encode_context(c1), m.encode_context(c2)
    assert not torch.equal(e1.waist_yaw_sp, e2.waist_yaw_sp)

    base2 = torch.where(c2.aim_valid, c2.aim_yaw, c2.yaw_now)
    sub = substitute_waist_sp(m, e1, e2.waist_yaw_sp, c2.yaw_now, base2)
    assert torch.equal(sub.memory, e2.memory)
    assert torch.equal(sub.mask, e2.mask)
    assert torch.equal(sub.waist_yaw_sp, e2.waist_yaw_sp)
    assert torch.equal(sub.waist_yaw_res, e2.waist_yaw_res)


def test_outcome_only_weights_zero_the_self_term_and_nothing_else():
    w = {k: torch.tensor(3.0) for k in
         ("terminal", "collision", "torque", "velocity", "joint_box", "posture",
          "smooth", "waist_aim", "waist_yaw_vel", "waist_pr_vel", "effort")}
    w["limit_alpha"] = torch.full((N_DOF,), 0.4)
    out = outcome_only_weights(w)

    assert WAIST_SP_SELF_TERMS == ("waist_aim",)
    assert float(out["waist_aim"]) == 0.0
    assert w["waist_aim"] is not out["waist_aim"]    # the caller's dict is intact
    assert float(w["waist_aim"]) == 3.0
    for k, v in w.items():
        if k not in WAIST_SP_SELF_TERMS:
            assert out[k] is v


def test_outcome_only_weights_silence_the_aim_term_in_the_real_cost():
    """Dict-level zeroing is only meaningful if ``FusedCost`` honours it: the
    priced ``waist_aim`` term must go to exactly 0 while every other term keeps
    the value it had."""
    B = 3
    cost = FusedCost(CostConfig(w_waist_aim=500.0, waist_aim_yaw_only=True),
                     max_horizon=H, device="cpu", dtype=torch.float64)
    yaw_i = cost.joint_names.index("waist_yaw_joint")
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    traj = torch.zeros(B, H, N_DOF, dtype=torch.float64)
    traj[:, :, yaw_i] = 0.4
    strike = _strike(B, torch.zeros(B, H, dtype=torch.float64), torch.float64)
    w = {k: torch.tensor(1.0, dtype=torch.float64) for k in ("waist_aim", "collision")}
    on = cost.forward(q0, traj, v0, strike, weights=w)
    off = cost.forward(q0, traj, v0, strike, weights=outcome_only_weights(w))

    assert float(on.terms["waist_aim"].abs().min()) > 0.0
    assert float(off.terms["waist_aim"].abs().max()) == 0.0
    for k in on.terms:
        if k != "waist_aim":
            assert torch.equal(on.terms[k], off.terms[k])
    # ...and the aim term was the ONLY difference in the total
    assert torch.allclose(on.total - on.terms["waist_aim"], off.total)


# ================================================ 4. off is exactly off ======
@pytest.fixture(scope="module")
def pool():
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    scfg = Phase2ScenarioConfig(horizon=H, buckets=(H,))
    raw = generate_cost_scenarios(120, clips, seed=0, cfg=scfg)
    return Phase2Pool(raw["buckets"], cfg=scfg, device=DEV)


def _p2(**kw) -> Phase2Config:
    """A tiny, deterministic loop: no roller, no eval, no probe, no sub-losses
    -- only the main cost step, which is where the search hooks in.

    ``amp_dtype=None``: bf16 autocast on an untrained decoder whose cost is
    ~1e6 produces a non-finite grad norm every step, and the loop then SKIPS
    the optimizer step -- which would make a parameter comparison vacuous.
    """
    return Phase2Config(
        compile_mode=None, amp_dtype="", batch_size=8, device=DEV, max_steps=3,
        log_every=1,
        eval_every=10 ** 9, ckpt_every=10 ** 9, probe_every=0,
        roller_start=10 ** 9, w_tc=0.0, tc_frac=0.0, w_bptt=0.0, suffix_frac=0.0,
        waist_sp_from_head=True,
        cost=phase2_cost_config(w_waist_aim=500.0, waist_aim_yaw_only=True),
        **kw,
    )


def _short_run(pool, run_dir, **kw):
    mc = ModelConfig(dec_layers=1, ctx_layers=1, max_horizon=H, waist_sp_head=True)
    torch.manual_seed(1234)
    m = AmortizedOptimizer(mc).to(DEV)
    train_phase2(model=m, pool=pool, cfg=_p2(**kw), model_cfg=mc,
                 run_dir=run_dir, log=False)
    recs = [json.loads(ln)
            for ln in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return m, [r for r in recs if r.get("event") == "train"]


def test_frac_zero_is_bit_for_bit_the_pre_change_run(pool, tmp_path):
    """Default-off must be a NO-OP, not merely a small perturbation: no RNG
    draw, no extra loss term, no metric.  The search slice is the batch's
    leading rows precisely so that turning the knobs on cannot advance the
    batch / noise / gain stream an A/B partner is sharing."""
    m_a, rec_a = _short_run(pool, tmp_path / "a")
    # every search knob moved off its default EXCEPT the fraction
    m_b, rec_b = _short_run(
        pool, tmp_path / "b", waist_sp_search_frac=0.0, w_waist_sp_search=25.0,
        waist_sp_search_k=6, waist_sp_search_span=0.3, waist_sp_search_temp=0.05,
    )
    # ...and the mirror case: a fraction with no price on the result
    m_c, rec_c = _short_run(pool, tmp_path / "c", waist_sp_search_frac=0.5,
                            w_waist_sp_search=0.0)

    assert len(rec_a) == 3
    # the run has to have actually TRAINED, or "identical" is vacuous
    assert all(r["grad_norm"] > 0.0 for r in rec_a)
    for rb, rc, ra in zip(rec_b, rec_c, rec_a):
        for k, v in ra.items():
            if isinstance(v, float) and k not in ("steps_per_s", "elapsed_s"):
                assert rb[k] == v, k
                assert rc[k] == v, k
        assert "waist_sp_search_spread" not in rb
        assert "waist_sp_search_spread" not in rc
    sd_a = m_a.state_dict()
    assert not torch.equal(sd_a["encoder.waist_sp_out.weight"],
                           torch.zeros_like(sd_a["encoder.waist_sp_out.weight"]))
    for k, v in sd_a.items():
        assert torch.equal(m_b.state_dict()[k], v), k
        assert torch.equal(m_c.state_dict()[k], v), k


def test_frac_zero_never_reaches_the_search_at_all(pool, tmp_path, monkeypatch):
    """Bit-identical output could in principle come from a search that ran and
    contributed nothing.  Make the search itself explode, and check the default
    run never touches it -- then check the trap is armed by tripping it."""

    def boom(**_kw):
        raise AssertionError("the search ran with the feature off")

    monkeypatch.setattr("amortized_opt.train_cost._waist_sp_search_term", boom)
    _short_run(pool, tmp_path / "off")                     # must not raise

    with pytest.raises(AssertionError, match="the search ran"):
        _short_run(pool, tmp_path / "on", waist_sp_search_frac=0.25,
                   w_waist_sp_search=25.0)


def test_the_search_reports_its_three_diagnostics(pool, tmp_path):
    """With the feature on, the training record carries the head-vs-target gap,
    the target-vs-analytic-aim gap, and the V-spread.  The spread is the one
    that matters: ~0 says the set-point does not change the outcome and the
    whole head is moot."""
    _m, recs = _short_run(pool, tmp_path / "on", waist_sp_search_frac=0.25,
                          w_waist_sp_search=25.0, waist_sp_search_k=4)
    assert recs
    for r in recs:
        assert r["waist_sp_search_delta"] >= 0.0
        assert r["waist_sp_search_aim"] >= 0.0
        # an untrained decoder on a real pool is nowhere near indifferent to
        # where the torso points
        assert r["waist_sp_search_spread"] > 0.0


# ================================ 5. the head follows the winning candidate ==
class _StubCost:
    """Scores a plan by its commanded waist yaw alone: ``(psi - target)^2``.

    Enough of ``FusedCost`` for :func:`_waist_sp_search_term`, which only ever
    reads ``forward(...).total``.  It makes one candidate the obvious winner
    without needing the real objective to happen to prefer it.
    """

    def __init__(self, target: float):
        self.target = float(target)

    def forward(self, q0, traj, v0, strike, *, lengths=None, weights=None):
        assert float(weights["waist_aim"]) == 0.0    # outcome-only, always
        psi = strike.waist_yaw_ref[:, 0].float()
        return type("CT", (), {"total": (psi - self.target).pow(2)})()


def _synthetic_batch(B=4):
    t = torch.full((B,), 0.4)
    g = torch.Generator().manual_seed(3)
    return {
        "context": _ctx(B),
        "masks": build_waypoint_masks(t, t, max_horizon=H, dt=CONTROL_DT),
        "init": (torch.rand(B, H, N_DOF, generator=g) - 0.5) * 0.6,
        "state": pack_state(torch.zeros(B, N_DOF), torch.zeros(B, N_DOF)),
        "q0": torch.zeros(B, N_DOF),
        "v0": torch.zeros(B, N_DOF),
    }


@pytest.mark.parametrize("target", [0.8, -0.8])
def test_the_head_gradient_points_at_the_winning_candidate(target):
    """Rig the envelope so one candidate obviously wins.  The distilled target
    must land on it, and one gradient step on the head must move the set-point
    that way -- the whole point of the term (the analytic gradient, at a frozen
    trajectory, would instead defend the plan the head already has)."""
    B = 4
    m = _model()
    batch = _synthetic_batch(B)
    enc = encode(m, batch["context"])
    assert float(enc.waist_yaw_sp.abs().max()) == 0.0    # zero-init head: sp = aim

    # a hard argmin here, so the assertion is about WHICH candidate won rather
    # than about how far the default temperature travels toward it
    cfg = _p2(waist_sp_search_frac=1.0, w_waist_sp_search=1.0,
              waist_sp_search_k=8, waist_sp_search_span=0.8,
              waist_sp_search_temp=1e-3)
    lengths = group_lengths(torch.full((B, 2), H, dtype=torch.long))
    loss, (d_sp, d_aim, spread) = _waist_sp_search_term(
        model=m, cost=_StubCost(target), cfg=cfg, enc=enc, batch=batch,
        strike=_strike(B), lengths=lengths, bucket=H,
        weights={"waist_aim": torch.tensor(1.0)}, rows=slice(0, B),
    )
    # the +-span rung IS the winner, and the target is on it
    assert d_sp == pytest.approx(abs(target), abs=1e-3)
    assert d_aim == pytest.approx(abs(target), abs=1e-3)
    assert spread > 0.0
    assert float(loss) > 0.0

    # the DEFAULT temperature is softer on purpose: same direction, shorter
    # step, so one mis-scored candidate cannot yank the head to the rail
    soft_cfg = _p2(waist_sp_search_frac=1.0, w_waist_sp_search=1.0)
    _l, (soft_d, _a, _v) = _waist_sp_search_term(
        model=m, cost=_StubCost(target), cfg=soft_cfg, enc=enc, batch=batch,
        strike=_strike(B), lengths=lengths, bucket=H,
        weights={"waist_aim": torch.tensor(1.0)}, rows=slice(0, B),
    )
    assert 0.4 * abs(target) < soft_d < abs(target)

    m.zero_grad()
    loss.backward()
    assert float(m.encoder.waist_sp_out.bias.grad.abs().sum()) > 0.0
    with torch.no_grad():
        for p in m.parameters():
            if p.grad is not None:
                p -= 0.02 * p.grad
    moved = encode(m, batch["context"]).waist_yaw_sp
    assert float(moved.mean()) * target > 0.0            # toward, not away
    # ...and strictly closer to the winner than the zero-init head was
    assert float((moved - target).abs().mean()) < abs(target)


def test_fused_candidate_groups_score_like_serial_ones():
    """Candidates are evaluated several at a time along the batch axis (K
    separate forwards of a quarter-batch leave the GPU idle in launch
    overhead).  The group size falls out of the MAIN batch size, so the same
    rows are scored fused or serially depending on the surrounding batch --
    and the two must agree to float32 batching noise, or the search's answer
    would depend on the fraction knob rather than on the envelope."""
    big = _synthetic_batch(8)                    # rows 0..1 -> 4 candidates/group
    small = {"context": _ctx(2), **{k: v[:2] for k, v in big.items()
                                    if k not in ("context", "masks")}}
    small["masks"] = type(big["masks"])(
        **{f: getattr(big["masks"], f)[:2] for f in big["masks"].__dataclass_fields__}
    )
    m = _model()
    cost = FusedCost(CostConfig(w_waist_aim=500.0, waist_aim_yaw_only=True),
                     max_horizon=H, device="cpu", dtype=torch.float32)
    cfg = _p2(waist_sp_search_frac=1.0, w_waist_sp_search=1.0, waist_sp_search_k=8)
    lengths = group_lengths(torch.full((8, 2), H, dtype=torch.long))
    out = []
    for b, n in ((big, 8), (small, 2)):
        loss, diag = _waist_sp_search_term(
            model=m, cost=cost, cfg=cfg, enc=encode(m, b["context"]), batch=b,
            strike=_strike(n, torch.zeros(n, H)), lengths=lengths[:n], bucket=H,
            weights={"waist_aim": torch.tensor(1.0)}, rows=slice(0, 2),
        )
        out.append((float(loss), *diag))
    assert out[0][3] > 0.0                       # a non-degenerate envelope
    # float32 reductions over a 4x wider batch are not bit-stable; 1e-3 relative
    # is three decades tighter than the effect being measured
    assert out[0] == pytest.approx(out[1], rel=1e-3)


def test_the_target_can_never_leave_the_candidate_span():
    """The bound that makes runaway structurally impossible: whatever the
    scores say, the distilled target is a convex mix of ``sp +- span`` and the
    analytic aim, so one step can move the set-point by at most ``span``."""
    B = 4
    m = _model()
    batch = _synthetic_batch(B)
    enc = encode(m, batch["context"])
    cfg = _p2(waist_sp_search_frac=1.0, w_waist_sp_search=1.0,
              waist_sp_search_span=0.25, waist_sp_search_temp=1e-3)
    lengths = group_lengths(torch.full((B, 2), H, dtype=torch.long))
    # a stub that wants the torso at the far end of the URDF box
    _loss, (d_sp, _d_aim, _v) = _waist_sp_search_term(
        model=m, cost=_StubCost(9.0), cfg=cfg, enc=enc, batch=batch,
        strike=_strike(B), lengths=lengths, bucket=H,
        weights={"waist_aim": torch.tensor(1.0)}, rows=slice(0, B),
    )
    assert d_sp <= 0.25 + 1e-6
