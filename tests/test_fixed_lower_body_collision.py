"""Fixed lower-body self-collision coverage for the 17-DoF drummer.

The Isaac drummer welds its legs in a standing pose, while the analytic model
only receives the waist and arm joints.  This regression fixture is legal under
the optimizer's posture box and sends the left arm through the fixed lower body
without touching the moving torso/head proxy.  It therefore proves that the
pelvis/leg capsule bank, rather than the older torso-only term, is what catches
the collision.
"""

from __future__ import annotations

import numpy as np
import torch

from amortized_opt.constants import (
    FIXED_LOWER_BODY_CAPSULES,
    HEAD_SPHERE,
    N_DOF,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
    WRIST_RADIUS,
)
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.eval_harness import TrajectoryScorer
from amortized_opt.fk import ChainFK, stick_shaft_offsets, stick_tip_offset
from amortized_opt.sdf import (
    DrumSceneSpec,
    SdfPenalty,
    fixed_capsule_self_sdf,
    point_capsule_sdf,
    point_sphere_sdf,
    segment_segment_sdf,
)


DTYPE = torch.float64
H = 4

# Found by sampling within urdf_position_limits(posture_limits=True), then
# retained because its old torso/head clearance is positive while its lower-body
# SDF is deeply negative.  Keep all numbers explicit: this is a geometry test,
# not a stochastic search in the suite.
LOWER_ONLY_Q = torch.tensor(
    [[
        0.7112065548444333, 0.48203049836841605, -0.3391394651750871,
        -0.014848176094095056, -0.2594468829288141, 0.6437401730355011,
        0.8442765290069656, 0.6520180661471124, -0.27477798253542396,
        -1.1230760236991513, -1.1940459087780664, 0.8177768753280517,
        -0.5484246796174681, 0.7490907432654521, 0.0380339155622369,
        1.453186060286548, -0.8372277137219934,
    ]],
    dtype=DTYPE,
)


def _ctx() -> StrikeContext:
    ones = torch.ones(1, 2, dtype=DTYPE)
    return StrikeContext(
        hit_index=torch.full((1, 2), H, dtype=torch.long),
        active=ones,
        target_drum=torch.zeros(1, 2, dtype=torch.long),
        start_drum=torch.full((1, 2), -1, dtype=torch.long),
        p_star=torch.full((1, 2, 3), 0.30, dtype=DTYPE),
        v_star=torch.full((1, 2, 3), -1.0, dtype=DTYPE),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE).expand(1, 2, 3).contiguous(),
        hit_radius=torch.full((1, 2), 0.08, dtype=DTYPE),
        disk_offset=torch.zeros(1, 2, 3, dtype=DTYPE),
        half_plane_dir=torch.zeros(1, 2, 3, dtype=DTYPE),
        has_half_plane=torch.zeros(1, 2, dtype=DTYPE),
        contact_sigma=torch.full((1, 2), 0.20, dtype=DTYPE),
        shaft_allowed=torch.zeros(1, 2, dtype=DTYPE),
        start_shaft_allowed=torch.zeros(1, 2, dtype=DTYPE),
        elev_required=ones,
    )


def _left_points_and_frames():
    fk = ChainFK(device="cpu", dtype=DTYPE)
    frames = fk.forward(LOWER_ONLY_Q)
    R, p = frames["left_wrist"]
    local = np.concatenate(
        [
            stick_tip_offset("left")[None, :],
            stick_shaft_offsets("left", 4),
            np.zeros((1, 3)),
        ],
        axis=0,
    )
    pts = ChainFK.points_world(R, p, torch.as_tensor(local, dtype=DTYPE))
    return fk, frames, pts


def test_fixture_is_lower_body_only_not_the_legacy_torso_head_pair():
    """The retained pose specifically isolates the new fixed lower-body bank."""
    _fk, frames, pts = _left_points_and_frames()
    tip, butt, wrist = pts[:, 0], pts[:, 1], pts[:, -1]
    Rt, pt = frames["torso"]
    t = lambda x: torch.as_tensor(x, dtype=DTYPE)  # noqa: E731
    ta = pt + (Rt @ t(TORSO_CAPSULE[0]).view(1, 3, 1)).squeeze(-1)
    tb = pt + (Rt @ t(TORSO_CAPSULE[1]).view(1, 3, 1)).squeeze(-1)
    hc = pt + (Rt @ t(HEAD_SPHERE[0]).view(1, 3, 1)).squeeze(-1)
    legacy = torch.minimum(
        torch.minimum(
            segment_segment_sdf(butt, tip, ta, tb, max(STICK_RADIUS, TIP_RADIUS), TORSO_CAPSULE[2]),
            segment_segment_sdf(butt, tip, hc, hc, max(STICK_RADIUS, TIP_RADIUS), HEAD_SPHERE[1]),
        ),
        torch.minimum(
            point_capsule_sdf(wrist, ta, tb, TORSO_CAPSULE[2]) - WRIST_RADIUS,
            point_sphere_sdf(wrist, hc, HEAD_SPHERE[1]) - WRIST_RADIUS,
        ),
    )
    lower_a = t([x[1] for x in FIXED_LOWER_BODY_CAPSULES])
    lower_b = t([x[2] for x in FIXED_LOWER_BODY_CAPSULES])
    lower_r = t([x[3] for x in FIXED_LOWER_BODY_CAPSULES])
    lower_stick, lower_wrist = fixed_capsule_self_sdf(
        butt, tip, wrist, lower_a, lower_b, lower_r,
        stick_radius=max(STICK_RADIUS, TIP_RADIUS), wrist_radius=WRIST_RADIUS,
    )
    assert float(legacy.min()) > 0.05
    assert float(torch.minimum(lower_stick, lower_wrist).min()) < -0.06


def test_all_analytic_collision_consumers_reject_fixed_leg_penetration():
    """Training cost, start SDF and eval use the same added capsule bank."""
    # Main training objective exposes the lower-specific diagnostic and folds it
    # into collision_self / collision_min_sdf.
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=DTYPE)
    wp = LOWER_ONLY_Q.unsqueeze(1).repeat(1, H, 1)
    out = cost.forward(LOWER_ONLY_Q, wp, torch.zeros_like(LOWER_ONLY_Q), _ctx())
    assert float(out.diagnostics["collision_lower_min_sdf_left"][0]) < -0.06
    assert float(out.diagnostics["collision_min_sdf_left"][0]) < -0.06
    assert float(out.raw["collision_self"][0]) > 0.0

    # Start-pose / pool screen shares SdfPenalty, including an externally
    # supplied frame dictionary (the path that avoids a second FK pass).
    _fk, frames, _pts = _left_points_and_frames()
    sdf = SdfPenalty(DrumSceneSpec.from_layout(frame="pelvis"), fk=_fk, dtype=DTYPE)
    _pen, det = sdf.forward(frames=frames, return_details=True)
    assert float(det.min_sdf[0]) < -0.06
    assert float(det.pen_self[0]) > 0.0

    # The evaluator must make the same plan fail its self-collision gate.
    scorer = TrajectoryScorer(device="cpu", dtype=DTYPE)
    Qt = scorer._t(LOWER_ONLY_Q.numpy())
    pts = scorer._points_world(Qt, "left")
    assert float(scorer._self_sdf(Qt, pts).min()) < -0.06

