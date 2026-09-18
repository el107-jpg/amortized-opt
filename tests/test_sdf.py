"""SdfPenalty: capsule collision, target-drum exemption, shared-FK path.

The four ``test_sdf_matches_repo_*`` tests used to pin this class bit-for-bit
against the original repo point cloud.  Collision is now the same stick
capsule + wrist sphere that ``FusedCost`` / ``TrajectoryScorer`` use, so those
comparisons would fail by design and are skipped.
"""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.fk import ChainFK
from amortized_opt.sdf import (
    SdfPenalty, _quad_roots_01, point_cylinder_sdf, q0_collision_clear,
    segment_cylinder_sdf,
)


def _q(n: int = 96, dtype=torch.float64, seed: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 17, generator=g, dtype=torch.float64) * 2.0 - 1.0).to(dtype)


_REPO_SKIP = pytest.mark.skip(
    reason="SdfPenalty is capsule collision; the repo original is a point cloud"
)


@_REPO_SKIP
def test_sdf_matches_repo_no_targets():
    pass


@_REPO_SKIP
def test_sdf_matches_repo_with_targets():
    pass


@_REPO_SKIP
def test_sdf_all_targets_set():
    pass


@_REPO_SKIP
def test_sdf_matches_repo_with_jittered_heads():
    pass


def test_shared_fk_path_matches_internal_fk():
    """forward(frames=...) must equal forward(q)."""
    fk = ChainFK(dtype=torch.float64)
    pen = SdfPenalty(fk=fk, dtype=torch.float64)
    q = _q(32, seed=31)
    a = pen.forward(q)
    b = pen.forward(frames=fk.forward(q))
    assert float((a - b).abs().max()) == 0.0


def test_exemption_lowers_penalty():
    """Exempting the target drum can only reduce the penalty."""
    pen = SdfPenalty(dtype=torch.float64)
    q = _q(64, seed=37)
    D = pen.num_drums
    base = pen.forward(q)
    for d in range(D):
        tgt = torch.full((q.shape[0],), d, dtype=torch.long)
        assert bool(
            (pen.forward(q, target_drum_left=tgt, target_drum_right=tgt) <= base + 1e-12).all()
        )


def test_sdfpenalty_pad_shaft_through_body_is_collision():
    """A shaft through the snare is a collision even when both endpoints sit outside."""
    from amortized_opt.sdf import DrumSceneSpec

    scene = DrumSceneSpec.from_layout(frame="pelvis")
    pen = SdfPenalty(scene, dtype=torch.float64, include_self=False)
    names = list(pen.scene.names)
    i = names.index("snare")
    ax = pen.axes[i]
    body = pen.head_centers[i] - pen.half_heights[i] * ax
    t = torch.tensor([1.0, 0.0, 0.0], dtype=body.dtype)
    if abs(float((t * ax).sum())) > 0.9:
        t = torch.tensor([0.0, 1.0, 0.0], dtype=body.dtype)
    t = t - (t * ax).sum() * ax
    t = t / t.norm().clamp_min(1e-9)
    half = 0.25
    butt = (body - half * t).unsqueeze(0)
    tip = (body + half * t).unsqueeze(0)
    cap = segment_cylinder_sdf(
        butt, tip,
        body.unsqueeze(0), ax.unsqueeze(0),
        pen.radii[i], pen.half_heights[i],
        ra=pen.stick_r,
    )
    assert float(cap) < 0.0
    d_tip = point_cylinder_sdf(
        tip, body.unsqueeze(0), ax.unsqueeze(0),
        pen.radii[i], pen.half_heights[i],
    )
    d_butt = point_cylinder_sdf(
        butt, body.unsqueeze(0), ax.unsqueeze(0),
        pen.radii[i], pen.half_heights[i],
    )
    assert float(d_tip) > 0.0 and float(d_butt) > 0.0

    # Public forward: rotate the left wrist so [butt→tip] is that piercing
    # segment; park the right hand far away.
    fk = ChainFK(dtype=torch.float64)
    q = torch.zeros(1, 17, dtype=torch.float64)
    frames = dict(fk.forward(q))
    local_tip = pen.local_points["left"][0]
    local_butt = pen.local_points["left"][1]
    a = local_tip - local_butt
    b = (tip - butt).squeeze(0)
    a_u, b_u = a / a.norm(), b / b.norm()
    v = torch.linalg.cross(a_u, b_u)
    c = (a_u * b_u).sum()
    if float(v.norm()) < 1e-8:
        R = torch.eye(3, dtype=torch.float64)
        if float(c) < 0:
            R = -R
    else:
        vx = torch.tensor([
            [0.0, -float(v[2]), float(v[1])],
            [float(v[2]), 0.0, -float(v[0])],
            [-float(v[1]), float(v[0]), 0.0],
        ], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64) + vx + vx @ vx * ((1.0 - c) / (v.norm() ** 2))
    p_left = tip.squeeze(0) - R @ local_tip
    p_right = p_left + 10.0
    frames["left_wrist"] = (R.unsqueeze(0), p_left.unsqueeze(0))
    frames["right_wrist"] = (R.unsqueeze(0), p_right.unsqueeze(0))
    _, det = pen.forward(frames=frames, return_details=True)
    assert float(det.pen_drum[0]) > 0.0
    assert float(det.min_sdf[0]) < 0.0


def test_q0_collision_clear_ready_and_exemption():
    """Ready is clear; a random box has collisions; exempting a drum can only help."""
    pen = SdfPenalty(dtype=torch.float64)
    ready = torch.as_tensor(
        ready_joint_pos(G1_UPPER_BODY_JOINTS)[None], dtype=torch.float64,
    )
    assert bool(q0_collision_clear(pen, ready)[0])

    g = torch.Generator().manual_seed(0)
    q = torch.rand(64, 17, generator=g, dtype=torch.float64) * 2.0 - 1.0
    clear = q0_collision_clear(pen, q)
    assert bool((~clear).any()), "random joint box should include collisions"
    # Exempt drum 0 on both hands: stick-vs-that-pad no longer counts.
    ex = torch.zeros(64, 2, dtype=torch.long)
    clear_ex = q0_collision_clear(pen, q, exempt_drum=ex)
    assert int(clear_ex.sum()) >= int(clear.sum())
    zeros = torch.zeros(1, 17, dtype=torch.float64)
    # URDF zero parks an arm in a drum under-volume on this layout.
    assert not bool(q0_collision_clear(pen, zeros)[0])


def test_quad_roots_grazing_gradients_are_finite():
    """Double root ``(s-0.5)^2 = 0`` used to NaN ``d(sqrt Δ)/dΔ``."""
    A = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    B = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    C = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    s_p, s_m, v_p, v_m = _quad_roots_01(A, B, C)
    (s_p + s_m).sum().backward()
    assert float(v_p.detach()) == 1.0
    assert abs(float(s_p.detach()) - 0.5) < 1e-5
    assert torch.isfinite(A.grad).all()
    assert torch.isfinite(B.grad).all()
    assert torch.isfinite(C.grad).all()


def test_segment_cylinder_tangent_gradients_are_finite():
    """Stick tangent to the side wall, unique closest point in the interior."""
    for dtype in (torch.float32, torch.float64):
        a0 = torch.tensor([[0.1, -0.2, 0.0]], dtype=dtype, requires_grad=True)
        a1 = torch.tensor([[0.1, 0.2, 0.0]], dtype=dtype, requires_grad=True)
        center = torch.zeros(1, 3, dtype=dtype)
        axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
        sdf = segment_cylinder_sdf(a0, a1, center, axis, 0.1, 0.5, ra=0.0)
        sdf.sum().backward()
        assert torch.isfinite(sdf).all()
        assert abs(float(sdf.detach())) < 1e-5
        assert torch.isfinite(a0.grad).all(), dtype
        assert torch.isfinite(a1.grad).all(), dtype
