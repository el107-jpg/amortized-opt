import numpy as np
import torch
import pytest
from types import SimpleNamespace

from amortized_opt.constants import ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.rhythm import DrumKit
from isaac_rl.rnea_cache import build_or_load
from isaac_rl.schedule_decoder import prefix_physics, time_statistics
from isaac_rl.transition_calibration import strike_context


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA replay test')
def test_cuda_replay_updates_inputs_and_preserves_preceding_outputs():
    from isaac_rl.cuda_replay import CudaReplay
    replay = CudaReplay(lambda a,b: {'sum':a+b, 'product':a*b})
    a = torch.arange(8, device='cuda', dtype=torch.float32)
    b = torch.ones_like(a)
    with torch.no_grad():
        first = replay(a,b)
        second = replay(a+10,b*3)
    torch.testing.assert_close(first['sum'], a+b)
    torch.testing.assert_close(first['product'], a*b)
    torch.testing.assert_close(second['sum'], a+10+b*3)
    torch.testing.assert_close(second['product'], (a+10)*b*3)
    # The inference accelerator must preserve gradients by using the normal
    # function when autograd is enabled.
    x = a.clone().requires_grad_()
    replay(x,b)['sum'].sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_pruning_preserves_optimum_with_joint_bounce_mean_and_future_dependent_scores():
    from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
    class SyntheticEnvironment(DecoderScheduleEnvironment):
        def __init__(self, scores, chunk):
            self.scores = scores; self.chunk = chunk
        def rollout(self, packed, *, prune=None):
            B, G, _ = packed['drums'].shape
            bits = (packed['drums'][:, :, 1]>=0).long()
            candidate = (bits*torch.tensor([8, 4, 2, 1])).sum(-1)
            # Draw identity is attached to the physical hit, independent of hand.
            draw = packed['efficiency'][:, 0].sum(-1).long()
            scores = self.scores[candidate, draw]
            alive = torch.ones(B, dtype=torch.bool)
            result = torch.zeros_like(scores); count = torch.zeros(B, dtype=torch.long)
            for g in range(G):
                result[alive, g] = scores[alive, g]; count[alive] += 1
                if prune is not None:alive &= prune(g, result, alive)
            return dict(score=result, evaluated_groups=count)
    # A candidate can have a disastrous first draw but still win on average.
    # Its fourth decision changes the FIRST score: no prefix reuse is valid.
    scores = torch.full((16, 2, 4), -80.)
    scores[0] = 0.
    scores[15, 0] = torch.tensor([-30., 10., 10., 10.])
    scores[15, 1] = torch.tensor([10., 10., 10., 10.])
    bank = ((torch.arange(16)[:, None]//torch.tensor([8, 4, 2, 1]))&1)[None]
    table = dict(drums=torch.tensor([[[0, -1]]*4]), speeds=torch.ones(1, 4, 2),
                 times=torch.tensor([[.1, .2, .3, .4]]), valid=torch.ones(1, 4, dtype=torch.bool))
    draws = torch.zeros(1, 2, 4, 2); draws[:, 1, :, 0] = 1.
    q = torch.zeros(1, 17)
    for chunk in (2, 10, 64):
        env = SyntheticEnvironment(scores, chunk)
        full = env.score(table, q, q, bank, draws)['score'].double().sum(-1).mean(-1)
        pruned = env.search_pruned(table, q, q, bank, draws, torch.tensor([0]))
        torch.testing.assert_close(pruned['best_value'], full.max(-1).values)
        assert pruned['best_index'].item()==15
        assert pruned['saved_group_evaluations']>0
        assert (pruned['upper_bounds'][~pruned['completed']]<pruned['best_value']).all()


def test_pruning_keeps_ties_for_exhaustive_lowest_index_tie_break():
    from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
    class TiedEnvironment(DecoderScheduleEnvironment):
        def __init__(self):self.chunk=4
        def rollout(self, packed, *, prune=None):
            B,G,_ = packed['drums'].shape
            score = torch.zeros(B,G)
            alive = torch.ones(B,dtype=torch.bool); count = torch.zeros(B,dtype=torch.long)
            for g in range(G):
                count[alive]+=1
                if prune is not None:alive &= prune(g,score,alive)
            return dict(score=score,evaluated_groups=count)
    bank = torch.tensor([[[0,0],[0,1],[1,0],[1,1]]])
    table = dict(drums=torch.tensor([[[0,-1],[0,-1]]]), speeds=torch.ones(1,2,2),
        times=torch.tensor([[.1,.2]]),valid=torch.ones(1,2,dtype=torch.bool))
    q=torch.zeros(1,17)
    result=TiedEnvironment().search_pruned(table,q,q,bank,torch.ones(1,2,2,2),torch.tensor([3]))
    assert result['best_index'].item()==0
    assert result['completed'].all()


def test_automatic_search_avoids_incumbent_pass_when_candidates_fit_one_batch():
    from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
    class SingleBatchEnvironment(DecoderScheduleEnvironment):
        def __init__(self):self.chunk=64;self.calls=0
        def rollout(self,packed,*,prune=None):
            assert prune is None
            self.calls+=1
            B,G,_=packed['drums'].shape
            score=(packed['drums'][:,:,1]>=0).float()
            return dict(score=score,evaluated_groups=torch.full((B,),G))
        def search_pruned(self,*args,**kwargs):
            raise AssertionError('Single-batch online search must avoid a separate incumbent pass')
    bank=((torch.arange(16)[:,None]//torch.tensor([8,4,2,1]))&1)[None]
    table=dict(drums=torch.tensor([[[0,-1]]*4]),speeds=torch.ones(1,4,2),
        times=torch.tensor([[.1,.2,.3,.4]]),valid=torch.ones(1,4,dtype=torch.bool))
    env=SingleBatchEnvironment();q=torch.zeros(1,17)
    result=env.search(table,q,q,bank,torch.ones(1,2,4,2),torch.tensor([0]))
    assert env.calls==1
    assert result['best_index'].item()==15
    assert result['strategy']=='exhaustive_single_batch'


def test_decoder_idle_hand_uses_its_borrowed_horizon_for_the_physical_spline():
    from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
    from isaac_rl.schedule_decoder import physical_spline
    cost = FusedCost(CostConfig(), max_horizon=10, device='cpu',
                    rnea=build_or_load(device='cpu'))
    kit = DrumKit.from_layout()
    class Model:
        cfg = SimpleNamespace(num_drums=kit.num_drums, num_future=3)
        def eval(self): return self
        def requires_grad_(self, value): return self
        def initial_trajectory(self, ctx, state, masks):
            return state[:, None, :17]+torch.arange(1, 11)[None, :, None]*.002
        def encode_context(self, ctx): return SimpleNamespace(waist_pr_sp=None)
        def refine(self, seed, state, encoding, masks, iterations):
            return SimpleNamespace(traj=seed*masks.elem_mask)
    generator = DecoderHitPoseGenerator(Model(), cost, kit, horizon=10)
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float32)
    v = torch.zeros_like(q)
    plan = generator.predict(q, v, torch.tensor([[[0, -1]]]), torch.tensor([[.1]]),
        torch.full((1, 1, 2), 2.), torch.ones(1, 1, dtype=torch.bool), 0, return_plan=True)
    torch.testing.assert_close(plan['lengths'], torch.full((1, 3), 5., dtype=q.dtype))
    spline = physical_spline(cost, q, v, plan)
    torch.testing.assert_close(spline.x_held[:, 5], plan['pose'])


def test_variable_prefix_duration_excludes_tail_and_weights_actual_endpoint():
    flags = torch.ones(2, 6, 2, dtype=torch.bool)
    total, longest = time_statistics(flags, torch.tensor([2, 5]), .02)
    torch.testing.assert_close(total, torch.tensor([[.04, .04], [.1, .1]]))
    torch.testing.assert_close(longest, total)


def test_executed_prefix_matches_full_cost_at_full_deadline_and_keeps_original_spline():
    torch.set_num_threads(1)
    cost = FusedCost(CostConfig(n_sub=4, physics_mode='knot_fd', w_velocity=1., w_joint_box=1.),
        max_horizon=8, device='cpu', dtype=torch.float64,
        rnea=build_or_load(device='cpu', dtype=torch.float64))
    kit = DrumKit.from_layout()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float64)
    v = torch.zeros_like(q)
    trajectory = q[:, None].expand(-1, 8, -1).clone()
    trajectory[:, :, 3] += torch.tensor([.005, .01, .02, .04, .08, .12, .18, .2])
    lengths = torch.full((1, 3), 8, dtype=torch.long)
    spline = cost.spline.build(q, trajectory, v, lengths)
    ctx = strike_context(kit, [[0, 2]], [[-1, -1]], [8], 'cpu').to('cpu', torch.float64)
    ctx.target_top_entry_depth_m = .01
    actual = prefix_physics(cost, spline, v, ctx, torch.tensor([8]))
    full = cost(q, trajectory, v, ctx, lengths=lengths)
    collision = -torch.minimum(full.diagnostics['collision_min_sdf'], full.diagnostics['collision_stick_min_sdf']).clamp_max(0)
    torch.testing.assert_close(actual['collision_m'], collision)
    tau_capture = {}
    cost(q, trajectory, v, ctx, lengths=lengths, constraint_profile_capture=tau_capture)
    ratio = (tau_capture['torque_tau'].abs()/tau_capture['torque_available']).amax((1, 2))
    torch.testing.assert_close(actual['torque_peak_ratio'], ratio)
    # An early natural boundary changes the spline's derivatives. The probe
    # must sample its original prefix, including future-dependent derivatives.
    from isaac_rl.trajectory_qp import cost_for_horizon
    shortened = cost_for_horizon(cost, 3).spline.build(q, trajectory[:, :3], v, lengths.clamp_max(3))
    assert not torch.allclose(spline.coeffs[:, :3], shortened.coeffs)
    prefix = prefix_physics(cost, spline, v, ctx, torch.tensor([3]))
    expected = (spline.segment_velocity_max_analytic()[:, :3]/cost.v_hard).amax((1, 2))
    torch.testing.assert_close(prefix['velocity_peak_ratio'], expected)


def test_previous_contact_exemption_expires_instead_of_restarting_at_each_event():
    cost = FusedCost(CostConfig(n_sub=4), max_horizon=4, device='cpu', dtype=torch.float64,
        rnea=build_or_load(device='cpu', dtype=torch.float64))
    kit = DrumKit.from_layout()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float64)
    ctx = strike_context(kit, [[-1, -1]], [[0, 0]], [4], 'cpu').to('cpu', torch.float64)
    captured = {}
    # Capture the departure mask at its use, independent of this ready pose's
    # distance to the source drum. An old contact must have the same collision
    # profile as having no departure exemption at all.
    ctx.start_contact_age_s = q.new_full((1, 2), 1.)
    cost._collision(cost.fk.forward(q), 1, 1, ctx, {h:q.new_ones(1, 1) for h in ('left','right')},
        {}, {}, {}, {}, {}, None, profile_capture=captured)
    ctx.start_drum[:] = -1
    reference = {}
    cost._collision(cost.fk.forward(q), 1, 1, ctx, {h:q.new_ones(1, 1) for h in ('left','right')},
        {}, {}, {}, {}, {}, None, profile_capture=reference)
    for key in captured:
        torch.testing.assert_close(captured[key], reference[key], rtol=0, atol=0)
