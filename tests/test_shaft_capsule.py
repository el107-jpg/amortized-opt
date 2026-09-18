"""Playing-shaft capsule vs legal hit region: primitive, pads, cost/eval/probe."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import CONTROL_DT, N_DOF, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.eval_harness import AcceptanceThresholds, EvalCase, TrajectoryScorer
from amortized_opt.probe import head_frame_error
from amortized_opt.sdf import hit_region_dist, point_cylinder_sdf, segment_hit_region
from amortized_opt.strike import (
    CYMBAL_CONTACT_SIGMA_FALLBACK,
    STICK_TIP_SIGMA,
    StrikeSpec,
    contact_sigma_candidates,
    shaft_hit_sigma_range,
    strike_axial_limits,
)


def test_sigma_range_pads_vs_cymbals():
    assert shaft_hit_sigma_range("snare") is None
    assert shaft_hit_sigma_range("tom1") is None
    assert shaft_hit_sigma_range("cymbal_left") == (
        CYMBAL_CONTACT_SIGMA_FALLBACK, STICK_TIP_SIGMA,
    )
    assert shaft_hit_sigma_range("hihat") == shaft_hit_sigma_range("cymbal_left")


def test_candidates_unchanged_for_repo_conformance():
    """Discrete sample list stays; the capsule is an additional continuum."""
    assert contact_sigma_candidates("snare") == (STICK_TIP_SIGMA,)
    assert any(abs(s - CYMBAL_CONTACT_SIGMA_FALLBACK) < 1e-9
               for s in contact_sigma_candidates("cymbal_left"))


def test_interior_hit_misses_every_discrete_sample():
    """Endpoints and the 4 SHAFT_SIGMAS sit outside a small disk; the midpoint is in."""
    # p(σ) = (σ - 0.125, 0, 0.005) on σ in [0.05, 0.20]; disk R=0.02 at origin.
    a = torch.tensor([-0.075, 0.0, 0.005])
    b = torch.tensor([0.075, 0.0, 0.005])
    p_star = torch.zeros(3)
    n_out = torch.tensor([0.0, 0.0, 1.0])
    z = torch.zeros(3)
    ax_lo, ax_hi = strike_axial_limits()
    dist, s, p = segment_hit_region(
        a, b, p_star, n_out, 0.02, z, z, 0.0, ax_lo, ax_hi,
    )
    assert float(dist) <= 1e-8
    assert abs(float(s) - 0.5) < 0.05
    assert float(p.norm()) < 0.02

    sigmas = contact_sigma_candidates("cymbal_left")
    for sig in sigmas:
        x = float(sig) - 0.125
        pt = torch.tensor([x, 0.0, 0.005])
        d, _, _ = hit_region_dist(pt, p_star, n_out, 0.02, z, z, 0.0, ax_lo, ax_hi)
        assert float(d) > 1e-4, sig


def test_segment_hit_region_gradients_finite():
    torch.manual_seed(0)
    B = 8
    a = torch.randn(B, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(B, 3, dtype=torch.float64, requires_grad=True)
    p_star = torch.randn(B, 3, dtype=torch.float64)
    n = torch.randn(B, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z = torch.zeros(B, 3, dtype=torch.float64)
    with torch.no_grad():
        n = n.clone()
        n[0] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        a0 = a.detach().clone()
        b0 = b.detach().clone()
        a0[0] = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        b0[0] = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
        b0[1] = a0[1]
    a = a0.requires_grad_(True)
    b = b0.requires_grad_(True)
    dist, s, p = segment_hit_region(
        a, b, p_star, n, 0.08, z, z, 0.0, 0.0, 0.01,
    )
    dist.sum().backward()
    assert torch.isfinite(dist).all()
    assert torch.isfinite(s).all()
    assert torch.isfinite(p).all()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


def test_segment_hit_region_rim_tangent_gradients_are_finite():
    """Playing shaft tangent to the legal hit circle: ``Δ = 0`` at ``s = 0.5``."""
    for dtype in (torch.float32, torch.float64):
        a = torch.tensor([-0.2, 0.08, 0.0], dtype=dtype, requires_grad=True)
        b = torch.tensor([0.2, 0.08, 0.0], dtype=dtype, requires_grad=True)
        p_star = torch.zeros(3, dtype=dtype)
        n_out = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
        z = torch.zeros(3, dtype=dtype)
        dist, s, p = segment_hit_region(
            a, b, p_star, n_out, 0.08, z, z, 0.0, -0.01, 0.02,
        )
        dist.sum().backward()
        assert torch.isfinite(dist).all()
        assert abs(float(s.detach()) - 0.5) < 1e-4
        assert abs(float(dist.detach())) < 1e-5
        assert torch.isfinite(a.grad).all(), dtype
        assert torch.isfinite(b.grad).all(), dtype


def _shaft_span_case(drum: str, *, hit_R: float = 0.02, case_id: int = 0
                     ) -> tuple[EvalCase, np.ndarray]:
    scorer = TrajectoryScorer(device="cpu")
    q = ready_joint_pos()
    lo, hi = shaft_hit_sigma_range("cymbal_left")
    a, _, u = scorer.contact_world(q, "left", lo)
    b, _, _ = scorer.contact_world(q, "left", hi)
    mid = 0.5 * (a + b)
    n = np.cross(u, np.array([0.0, 1.0, 0.0]))
    if float(np.linalg.norm(n)) < 1e-6:
        n = np.cross(u, np.array([1.0, 0.0, 0.0]))
    n = n / np.linalg.norm(n)
    spec = StrikeSpec(
        time=CONTROL_DT, hand="left", position=mid, velocity=-n,
        drum=drum, hit_radius=hit_R, half_plane_dir=None,
        disk_center_offset=None,
        contact_sigma=0.10 if A.shaft_hit_allowed(drum) else STICK_TIP_SIGMA,
    )
    case = EvalCase(
        case_id=int(case_id), grid="capsule", drum=drum,
        drum_index=scorer.drum_index.get(drum, 0),
        hand="left", N=1, dt=CONTROL_DT, strike_speed=1.0,
        q_start=q, qdot_start=np.zeros(N_DOF), strike=spec,
    )
    Q = np.stack([q, q], axis=0)
    return case, Q


def test_cymbal_shaft_span_is_a_hit_pad_is_not():
    thr = AcceptanceThresholds(hit_margin_m=0.0)
    scorer = TrajectoryScorer(device="cpu", thresholds=thr)
    cym, Q = _shaft_span_case("cymbal_left")
    snare, _ = _shaft_span_case("snare")
    snare.q_start = cym.q_start
    r_c = scorer.score(cym, Q)
    r_s = scorer.score(snare, Q)
    assert r_c.on_head
    assert not r_s.on_head


def test_probe_matches_eval_on_shaft_span():
    thr = AcceptanceThresholds(hit_margin_m=0.0)
    scorer = TrajectoryScorer(device="cpu", thresholds=thr)
    kit = A.DrumKit.from_layout()
    cym, Q = _shaft_span_case("cymbal_left")
    r = scorer.score(cym, Q)
    # Probe uses the kit's real head, eval uses the synthetic p_star; both
    # must use the same capsule primitive.  Compare eval against a probe-style
    # call that shares segment_hit_region via head_frame_error on a grid case.
    from amortized_opt.eval_harness import generate_grid
    grid = generate_grid(seed=0, grid="standard", scorer=scorer)
    case = next(c for c in grid if c.drum.startswith("cymbal") or c.drum == "hihat")
    Qg = np.repeat(case.q_start[None, :], case.n_waypoints, axis=0)
    rg = scorer.score(case, Qg)
    drum = int(case.drum_index)
    _ax, _ip, ok = head_frame_error(
        scorer, Qg[int(case.strike.waypoint(case.dt))], case.hand, kit, drum,
        0.10,
    )
    assert bool(ok) == bool(rg.on_head)
    assert r.on_head  # synthetic shaft-span still a cymbal hit


def test_cost_terminal_agrees_with_eval_on_head():
    thr = AcceptanceThresholds(hit_margin_m=0.0)
    scorer = TrajectoryScorer(device="cpu", thresholds=thr)
    cym, Q = _shaft_span_case("cymbal_left", case_id=0)
    snare, Qs = _shaft_span_case("snare", case_id=1)
    cost = FusedCost(CostConfig(), max_horizon=8, device="cpu", dtype=torch.float64)
    cases = [cym, snare]
    ctx = A.strike_context_from_cases(
        cases, drum_index=cost.drum_index, dtype=torch.float64,
    )
    q0 = torch.stack([
        torch.as_tensor(Q[0], dtype=torch.float64),
        torch.as_tensor(Qs[0], dtype=torch.float64),
    ])
    H = cost.max_horizon
    wp = torch.zeros(2, H, N_DOF, dtype=torch.float64)
    wp[0, :] = torch.as_tensor(Q[1], dtype=torch.float64)
    wp[1, :] = torch.as_tensor(Qs[1], dtype=torch.float64)
    v0 = torch.zeros(2, N_DOF, dtype=torch.float64)
    out = cost.forward(q0, wp, v0, ctx)
    dist = out.diagnostics["terminal_region_dist"].numpy()
    r0 = scorer.score(cym, Q)
    r1 = scorer.score(snare, Qs)
    assert r0.on_head and not r1.on_head
    assert dist[0] < 2e-3
    assert dist[1] > 1e-3


def test_stick_capsule_vs_pad_sees_shaft_through_body():
    """A shaft through the snare is a collision even when both endpoints sit outside.

    The old point cloud left ``SHAFT_TIP_CLEAR`` (0.18 m) of distal stick
    unsampled, so a pad graze at mid-shaft scored clean.  Pads are capsules now.
    """
    scorer = TrajectoryScorer(device="cpu")
    i = scorer.drum_index["snare"]
    c = scorer._body_c[i]
    ax = scorer._axes[i]
    t = torch.tensor([1.0, 0.0, 0.0], dtype=c.dtype, device=c.device)
    if abs(float((t * ax).sum())) > 0.9:
        t = torch.tensor([0.0, 1.0, 0.0], dtype=c.dtype, device=c.device)
    t = t - (t * ax).sum() * ax
    t = t / t.norm().clamp_min(1e-9)
    half = 0.25
    butt = (c - half * t).unsqueeze(0)
    tip = (c + half * t).unsqueeze(0)
    body, _under = scorer._stick_drum_sdf(butt, tip)
    assert float(body[0, i]) < 0.0
    d_tip = point_cylinder_sdf(
        tip, c.unsqueeze(0), ax.unsqueeze(0),
        scorer._radii[i], scorer._half_h[i],
    )
    assert float(d_tip) > 0.0
    d_butt = point_cylinder_sdf(
        butt, c.unsqueeze(0), ax.unsqueeze(0),
        scorer._radii[i], scorer._half_h[i],
    )
    assert float(d_butt) > 0.0
