import numpy as np
import torch

from amortized_opt.constants import ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.rhythm import DrumKit
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import strike_context, exact_contact_region
from isaac_rl.trajectory_qp import signed_contact_constraints
from scripts.build_unrestricted_transition_chart import aggregate, allowed_pairs, retime


NAMES = ['snare', 'hihat', 'tom1', 'tom2', 'cymbal_left', 'cymbal_right', 'floor_tom']


def test_only_left_floor_tom_is_excluded_and_statistics_preserve_missing_cases():
    allowed = allowed_pairs(NAMES)
    assert allowed.sum() == 85
    assert allowed[1].all()
    assert not allowed[0, 6].any() and not allowed[0, :, 6].any()
    pieces = [dict(hand=np.asarray(1), source=np.asarray(2),
        destination=np.array([6, 6, 6, 1]), minimum_s=np.array([.24, .32, np.inf, np.inf]),
        decoder_minimum_s=np.array([.56, .60, np.inf, np.inf]))]
    result = aggregate(NAMES, pieces)
    assert result['minimum_s'].shape == (2, 7, 7)
    assert result['minimum_s'][1, 2, 6] == .24
    assert result['mean_minimum_s'][1, 2, 6] == .28
    np.testing.assert_allclose(result['std_minimum_s'][1, 2, 6], .08/np.sqrt(2))
    assert np.isnan(result['std_minimum_s'][1, 2, 1])
    assert result['support'][1, 2, 6] == 2 and result['trials'][1, 2, 6] == 3
    assert result['support_fraction'][1, 2, 6] == 2/3
    assert result['all_start_median_s'][1, 2, 6] == .32
    assert np.isinf(result['all_start_p90_s'][1, 2, 6])
    assert np.isinf(result['minimum_s'][1, 2, 1])
    assert result['completed'][1, 2, 1]
    assert not result['completed'][1, 3, 6]
    pieces[0]['completed'] = False
    partial = aggregate(NAMES, pieces)
    assert not partial['completed'][1, 2].any()
    assert partial['minimum_s'][1, 2, 6] == .24
    pieces[0]['minimum_s'][1] = np.inf
    single = aggregate(NAMES, pieces)
    assert single['support'][1, 2, 6] == 1
    assert np.isnan(single['std_minimum_s'][1, 2, 6])


def test_retiming_preserves_endpoint_and_measured_start_position():
    q = torch.tensor([[1., 2.]])
    path = torch.tensor([[[2., 3.], [3., 4.], [4., 5.]]])
    changed = retime(q, path, 2)
    torch.testing.assert_close(changed[:, -1], path[:, -1])
    torch.testing.assert_close(changed[:, 0], torch.tensor([[2.5, 3.5]]))


def test_signed_head_constraints_match_tip_and_shaft_membership_and_keep_gradients():
    torch.set_num_threads(1)
    cost = FusedCost(CostConfig(w_velocity=1., w_joint_box=1.), max_horizon=2,
        dtype=torch.float64, rnea=load_cached(dtype=torch.float64))
    kit = DrumKit.from_layout()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float64)
    for drum in (6, 1):
        ctx = strike_context(kit, [[-1, drum]], [[-1, drum]], [2], 'cpu').to('cpu', torch.float64)
        R, p = cost.fk.frame(q, 'right_wrist')
        base = p+(R@cost.c_mount_p['right'].view(1, 3, 1)).squeeze(-1)
        axis = (R@cost.c_mount_axis['right'].view(1, 3, 1)).squeeze(-1)
        # Place the synthetic disk through the specified contact point, with
        # a radial offset to make the interior signed gradient observable.
        point = base+ctx.contact_sigma[:, 1, None]*axis
        ctx.n_out[:, 1] = q.new_tensor([0., 0., 1.])
        ctx.disk_offset[:, 1] = 0
        ctx.has_half_plane[:, 1] = 0
        ctx.p_star[:, 1] = point+q.new_tensor([.02, 0., 0.])
        trajectory = q[:, None].expand(-1, 2, -1).clone().requires_grad_(True)
        signed = signed_contact_constraints(cost, trajectory, ctx, .07)
        assert exact_contact_region(cost, trajectory, ctx, depth_band_m=.07).item() <= 1e-8
        assert all((value <= 1e-8).all() for value in signed.values())
        grad = torch.autograd.grad(signed['head_radial'].sum(), trajectory)[0]
        assert torch.isfinite(grad).all() and grad.abs().max() > 0
        ctx.p_star[:, 1] += q.new_tensor([2., 0., 0.])
        signed = signed_contact_constraints(cost, trajectory, ctx, .07)
        assert signed['head_radial'][0, 1] > 0
        assert exact_contact_region(cost, trajectory, ctx, depth_band_m=.07).item() > 0
