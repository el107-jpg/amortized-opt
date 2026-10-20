"""Adversarial geometry, contact-union gradients, coverage and resume checks."""
import math

import numpy as np
import pytest
import torch

from amortized_opt.collision_geometry import temporal_clearance_bound
from amortized_opt.convex_distance import capsule_cylinder_signed_distance
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.target_contact import v4_contact_selection
from amortized_opt.train_cost import phase2_cost_config, phase2_config_from_checkpoint
from amortized_opt.tspline import TrajectorySplineBuilder
from scripts.train_phase2 import _resolve_physics_loss_settings


def test_convex_distance_finds_missed_rim_penetration():
    midpoint = torch.tensor([.156, 0, .056], dtype=torch.float64)
    offset = torch.tensor([.4/math.sqrt(8), 0, -.4/math.sqrt(8)], dtype=torch.float64)
    gap = capsule_cylinder_signed_distance(midpoint+offset, midpoint-offset,
        torch.zeros(3, dtype=torch.float64), torch.tensor([0., 0., 1.], dtype=torch.float64),
        .15, .05, .01)
    assert gap.item() == pytest.approx(math.sqrt(2)*.006-.01, abs=1e-8)


def test_convex_distance_values_and_backward_against_spheres():
    points = torch.tensor([[.2, 0, 0], [0, 0, .07], [0, 0, 0]], dtype=torch.float64,
                          requires_grad=True)
    gap = capsule_cylinder_signed_distance(points, points, torch.zeros_like(points),
        torch.tensor([0., 0., 1.], dtype=torch.float64), .15, .05, .01)
    assert gap.tolist() == pytest.approx([.04, .01, -.06], abs=1e-9)
    gap.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_convex_distance_pose_and_shape_gradients():
    rng = torch.Generator().manual_seed(54)
    a = torch.rand(5, 3, dtype=torch.float64, generator=rng)*.6-.3
    b = a+torch.randn(5, 3, dtype=torch.float64, generator=rng)*.18
    values = (a, b, torch.zeros_like(a), torch.randn(5, 3, dtype=torch.float64, generator=rng),
              torch.full((5,), .15, dtype=torch.float64),
              torch.full((5,), .05, dtype=torch.float64),
              torch.full((5,), .01, dtype=torch.float64))
    assert torch.autograd.gradcheck(capsule_cylinder_signed_distance,
        tuple(x.requires_grad_() for x in values), atol=2e-5, rtol=2e-4)


def test_convex_distance_matches_gjk_epa():
    coal = pytest.importorskip('coal')
    rng = np.random.default_rng(54)
    n = 160
    a = rng.uniform(-.3, .3, (n, 3)); b = a+rng.normal(size=(n, 3))*.18
    k = rng.normal(size=(n, 3)); k /= np.linalg.norm(k, axis=1)[:, None]
    radius = rng.uniform(.08, .2, n); height = rng.uniform(.02, .15, n)
    capsule_radius = rng.uniform(.008, .04, n)
    def rotation(axis):
        axis = axis/np.linalg.norm(axis)
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        x = np.cross(axis, seed); x /= np.linalg.norm(x)
        return np.column_stack((x, np.cross(axis, x), axis))
    request = coal.DistanceRequest(); request.enable_signed_distance = True
    request.gjk_tolerance = 1e-10
    reference = []
    for i in range(n):
        result = coal.DistanceResult()
        coal.distance(coal.Capsule(capsule_radius[i], np.linalg.norm(b[i]-a[i])),
            coal.Transform3s(rotation(b[i]-a[i]), (a[i]+b[i])/2),
            coal.Cylinder(radius[i], 2*height[i]), coal.Transform3s(rotation(k[i]), np.zeros(3)),
            request, result)
        reference.append(result.min_distance)
    values = [torch.tensor(x, dtype=torch.float64) for x in
              (a, b, np.zeros_like(a), k, radius, height, capsule_radius)]
    actual = capsule_cylinder_signed_distance(*values).numpy()
    assert np.max(np.abs(actual-reference)) < 3e-6  # EPA's penetration tolerance
    assert np.array_equal(actual > 0, np.asarray(reference) > 0)


def test_contact_union_valid_probe_is_zero_and_unused_probe_has_no_gradient():
    distance = torch.tensor([[0., .004]], dtype=torch.float64, requires_grad=True)
    weights, residual = v4_contact_selection(distance, torch.zeros_like(distance),
                                            torch.tensor([[2.4]]), dt=.004)
    assert residual.item() == 0
    assert weights.tolist() == [[1., 0.]]
    gradient = torch.autograd.grad(residual.sum(), distance)[0]
    assert gradient.tolist() == [[0., 0.]]


def test_contact_union_requires_quality_from_the_same_probe():
    # The geometrically valid slow probe cannot borrow the distant probe's speed.
    distance = torch.tensor([[0., .004]], requires_grad=True)
    deficit = torch.tensor([[.2, 0.]], requires_grad=True)
    weights, residual = v4_contact_selection(distance, deficit, torch.tensor([[2.4]]), dt=.004)
    assert residual.item() > 0
    assert weights.tolist() == [[1., 0.]]
    residual.sum().backward()
    assert deficit.grad[0, 0] > 0 and deficit.grad[0, 1] == 0


def test_contact_union_distance_gradients_are_nonnegative():
    distance = torch.tensor([[.001, .004]], dtype=torch.float64, requires_grad=True)
    _, residual = v4_contact_selection(distance, torch.zeros_like(distance),
                                       torch.tensor([[2.4]]), dt=.004)
    gradient = torch.autograd.grad(residual.sum(), distance)[0]
    assert (gradient >= 0).all()
    assert gradient[0, 0] > .99 and gradient[0, 1] == 0


def test_temporal_bound_detects_unsampled_sphere_crossing():
    # A sphere flies through the origin; both endpoint surface gaps are +2 mm.
    sdf = torch.tensor([[.002, .002]], requires_grad=True)
    bound = temporal_clearance_bound(sdf, torch.ones_like(sdf), torch.tensor([[0., 6.]]), .004)
    assert bound[0, 1].item() == pytest.approx(-.01)
    (-bound[0, 1]).backward()
    assert (sdf.grad < 0).all()
    exempt = temporal_clearance_bound(sdf, torch.tensor([[0., 1.]]), torch.tensor([[0., 6.]]), .004)
    assert exempt[0, 1].item() == pytest.approx(.002)


def test_foh_holds_shorter_arm_and_cubic_position_extrema():
    builder = TrajectorySplineBuilder(max_segments=3, dtype=torch.float64)
    q = torch.zeros(1, 17, dtype=torch.float64)
    waypoint = torch.zeros(1, 3, 17, dtype=torch.float64)
    waypoint[0, :, 3] = torch.tensor([1., 99., -99.])
    spline = builder.build(q, waypoint, q, torch.tensor([[3, 1, 3]]))
    samples = spline.subsample_foh(5)
    assert samples.q.shape == (1, 16, 17)
    assert torch.all(samples.q[0, 5:, 3] == 1)
    # q(s)=4u(1-u): both endpoint positions 0; midpoint reaches 1 rad.
    spline.coeffs = torch.zeros_like(spline.coeffs)
    spline.coeffs[0, 0, 0, 1] = 4/spline.dt
    spline.coeffs[0, 0, 0, 2] = -4/spline.dt**2
    low, high = spline.segment_position_bounds()
    assert low[0, 0, 0] == 0
    assert high[0, 0, 0].item() == pytest.approx(1.)


def test_new_defaults_and_archived_resume_are_separate_objectives():
    fresh = phase2_cost_config()
    assert fresh.physics_loss_version == 'geometry_v2'
    assert fresh.n_sub == 5 and fresh.collision_temporal_bound
    assert fresh.geometry_motion_mode == 'foh' and fresh.w_stick_stick > 0
    archived = {'train_cfg': {'cost': {'contact_scoring_mode': 'isaac_v4', 'n_sub': 1}}}
    settings = _resolve_physics_loss_settings('auto', archived, resume=True)
    assert settings['physics_loss_version'] == 'legacy' and settings['n_sub'] == 1
    assert settings['geometry_motion_mode'] == 'spline'
    assert not settings['collision_temporal_bound'] and settings['w_stick_stick'] == 0
    restored = phase2_config_from_checkpoint(archived)
    assert restored.cost.physics_loss_version == 'legacy'
    with pytest.raises(ValueError, match='--init'):
        _resolve_physics_loss_settings('geometry_v2', archived, resume=True)
    with pytest.raises(ValueError, match='--init'):
        _resolve_physics_loss_settings('auto', archived, resume=True, n_sub=5)


def test_dense_pd_return_preserves_control_knots():
    from amortized_opt.pd_sim import simulate_pd
    x = torch.tensor([[[0.], [.2], [.1]]], dtype=torch.float64, requires_grad=True)
    args = (x, torch.ones(1, 1, dtype=torch.float64), torch.tensor([40.]),
            torch.tensor([4.]), .02, x[:, 0], torch.zeros(1, 1, dtype=torch.float64))
    knot_q, knot_v, _, dense_q, dense_v = simulate_pd(*args, substeps=5, return_substeps=True)
    assert torch.equal(knot_q, dense_q[:, ::5])
    assert torch.equal(knot_v, dense_v[:, ::5])
    assert dense_q.shape == (1, 11, 1)
    dense_q.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[:, 1:].abs().sum() > 0


def test_forearm_collision_is_priced_even_when_stick_and_wrist_are_clear(monkeypatch):
    from tests.test_cost import _synthetic_ctx
    cfg = phase2_cost_config(n_sub=1, collision_temporal_bound=False, w_stick_stick=0)
    cost = FusedCost(cfg, max_horizon=2, dtype=torch.float64)
    D, S = cost.n_drums, 2
    heads = torch.full((D, 3), 100., dtype=torch.float64); heads[0] = torch.tensor([0., 0., .05])
    axes = torch.tensor([0., 0., 1.], dtype=torch.float64).expand(D, -1)
    monkeypatch.setattr(cost, 'c_heads', heads)
    monkeypatch.setattr(cost, 'c_axes', axes)
    monkeypatch.setattr(cost, 'c_body_c', heads-.05*axes)
    monkeypatch.setattr(cost, 'c_half_h', torch.full((D,), .05, dtype=torch.float64))
    monkeypatch.setattr(cost, 'c_under_c', heads-.3*axes)
    monkeypatch.setattr(cost, 'c_radii', torch.full((D,), .15, dtype=torch.float64))
    monkeypatch.setattr(cost, 'c_lower_a', torch.full_like(cost.c_lower_a, 100.))
    monkeypatch.setattr(cost, 'c_lower_b', torch.full_like(cost.c_lower_b, 100.))
    eye = torch.eye(3, dtype=torch.float64)[None].expand(S, -1, -1)
    frames = {name: (eye, torch.full((S, 3), 40., dtype=torch.float64)) for name in cost.FRAMES}
    for name, position in (('shoulder', [-.4, 0., 0.]), ('elbow', [-.25, 0., 0.]),
                           ('wrist', [.25, 0., 0.])):
        frames[f'left_{name}'] = (eye, torch.tensor(position, dtype=torch.float64)[None].expand(S, -1))
    points = torch.full((S, cost.n_points, 3), 20., dtype=torch.float64)
    points[:, -1] = frames['left_wrist'][1]
    monkeypatch.setattr(cost, '_points_world', lambda frames, hand, ctx:
        points if hand == 'left' else torch.full_like(points, 30.))
    ctx = _synthetic_ctx(1, D, 'cpu', torch.float64, kl=torch.tensor([2]), kr=torch.tensor([2]))
    masks = {'left': torch.ones(1, S, dtype=torch.float64), 'right': torch.zeros(1, S, dtype=torch.float64)}
    diagnostics, capture = {}, {}
    cost._collision(frames, 1, S, ctx, masks, {}, {}, {}, diagnostics, {}, None,
                    constraint_capture=capture)
    assert capture['collision_left_forearm_body'].max() > .03
    assert diagnostics['collision_min_sdf'] < -.03
    legacy = cost.with_config(physics_loss_version='legacy')
    diagnostics = {}
    legacy._collision(frames, 1, S, ctx, masks, {}, {}, {}, diagnostics, {}, None)
    assert diagnostics['collision_min_sdf'] > 0


def test_shorter_arm_stays_in_collision_mask(monkeypatch):
    from tests.test_cost import _synthetic_ctx
    cost = FusedCost(phase2_cost_config(n_sub=1), max_horizon=3, dtype=torch.float64)
    recorded = {}
    original = cost._collision
    def capture(*args, **kwargs):
        recorded.update(args[4])
        return original(*args, **kwargs)
    monkeypatch.setattr(cost, '_collision', capture)
    q = cost.q_ready[None]
    ctx = _synthetic_ctx(1, cost.n_drums, 'cpu', torch.float64,
                         kl=torch.tensor([1]), kr=torch.tensor([3]))
    cost(q, q[:, None].repeat(1, 3, 1), torch.zeros_like(q), ctx)
    assert recorded['left'].tolist() == [[1., 1., 1., 1.]]
    assert torch.equal(recorded['left'], recorded['right'])


@pytest.mark.parametrize('motion', ['foh', 'spline', 'pd'])
def test_corrected_cost_has_finite_backward(motion):
    from tests.test_cost import _synthetic_ctx
    cfg = phase2_cost_config(geometry_motion_mode=motion)
    cost = FusedCost(cfg, max_horizon=3, dtype=torch.float64)
    q = cost.q_ready[None]
    waypoint = q[:, None].repeat(1, 3, 1).clone().requires_grad_()
    with torch.no_grad():
        waypoint[:, 1:, 4] += .03
    ctx = _synthetic_ctx(1, cost.n_drums, 'cpu', torch.float64,
                         kl=torch.tensor([1]), kr=torch.tensor([3]))
    result = cost(q, waypoint, torch.zeros_like(q), ctx)
    result.total.sum().backward()
    assert torch.isfinite(result.total).all() and torch.isfinite(waypoint.grad).all()


@pytest.mark.parametrize('contact,motion', [('isaac_v4','foh'), ('default','foh'), ('isaac_v4','pd')])
def test_corrected_cuda_cost_has_no_host_sync(contact, motion):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from tests.test_cost import _synthetic_ctx
    device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
    options = {} if contact == 'default' else {'contact_scoring_mode': contact}
    cost = FusedCost(phase2_cost_config(geometry_motion_mode=motion, **options),
                     max_horizon=2, device=device)
    q = cost.q_ready[None]
    waypoint = q[:, None].repeat(1, 2, 1).clone().requires_grad_()
    k = torch.full((1,), 2, device=device, dtype=torch.long)
    ctx = _synthetic_ctx(1, cost.n_drums, device, torch.float32, kl=k, kr=k)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode('error')
    try:
        result = cost(q, waypoint, torch.zeros_like(q), ctx)
        result.total.sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.isfinite(waypoint.grad).all()
