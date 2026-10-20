import torch
from amortized_opt.fk import ChainFK,stick_contact_offset
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import contact_jacobian,rebound_velocity,hermite


def test_contact_jacobian_matches_autograd():
    fk=ChainFK()
    q=torch.tensor(ready_joint_pos(),dtype=torch.float64,requires_grad=True)
    for hand in ('left','right'):
        local=q.new_tensor(stick_contact_offset(hand,.15))
        def point(x):
            R,p=fk.frame(x[None],hand+'_wrist')
            return p[0]+R[0]@local
        actual=torch.autograd.functional.jacobian(point,q)
        pos,jac=contact_jacobian(fk,q[None],hand,local[None])
        torch.testing.assert_close(pos[0],point(q),atol=1e-9,rtol=1e-8)
        torch.testing.assert_close(jac[0],actual,atol=1e-9,rtol=1e-8)


def test_rebound_energy_and_missed_separating_contacts():
    v=torch.tensor([[-2.,3.,0.],[-2.,3.,0.],[2.,3.,0.]])
    J=torch.zeros(3,2,3); J[:,0,0]=1
    contacts=torch.tensor([[True,False],[False,False],[True,False]])
    out=rebound_velocity(v,J,torch.eye(3),torch.full((3,2),.25),contacts)
    torch.testing.assert_close(out[0],torch.tensor([1.,3.,0.]),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out[1:],v[1:])
    torch.manual_seed(7)
    v=-torch.rand(32,3)
    J=torch.rand(32,2,3)
    out=rebound_velocity(v,J,torch.eye(3),torch.rand(32,2),torch.ones(32,2,dtype=torch.bool))
    assert torch.isfinite(out).all()
    assert (out.square().sum(-1)<=v.square().sum(-1)+1e-5).all()


def test_connectors_preserve_endpoint_states_and_velocities():
    torch.manual_seed(8)
    q0,v0,q1,v1=torch.randn(4,2,17)
    q,v,_=hermite(q0,v0,q1,v1,torch.tensor([.3,.5]),torch.tensor([[0.,1.],[0.,1.]]))
    torch.testing.assert_close(q[:,0],q0); torch.testing.assert_close(q[:,1],q1)
    torch.testing.assert_close(v[:,0],v0); torch.testing.assert_close(v[:,1],v1)


def test_single_pose_mlp_never_calls_encoder_or_decoder_and_retains_idle_arm():
    from types import SimpleNamespace
    from torch import nn
    from amortized_opt.rhythm import DrumKit
    from amortized_opt.cost import CostConfig,FusedCost
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS
    from amortized_opt.trajectory_surrogate import SingleHitGenerator
    from isaac_rl.rnea_cache import build_or_load
    kit=DrumKit.from_layout()
    calls=[]
    class PoseHead(nn.Module):
        def terminal_pose(self,ctx,state,masks):
            calls.append(ctx.goal_valid.clone())
            return state[:,:17]+.01
    class Model:
        cfg=SimpleNamespace(num_drums=kit.num_drums,num_future=3)
        terminal_initializer=PoseHead()
        def encode_context(self,*args): raise AssertionError('encoder was called')
        def refine(self,*args,**kwargs): raise AssertionError('decoder was called')
    cost=FusedCost(CostConfig(),max_horizon=25,device='cpu',rnea=build_or_load(device='cpu'))
    generator=SingleHitGenerator(Model(),cost,kit,horizon=25)
    q=torch.tensor(ready_joint_pos()[None],dtype=torch.float32)
    drums=torch.tensor([[[0,-1],[1,-1]]])
    pose,v=generator.predict(q,torch.zeros_like(q),drums,torch.tensor([[.5,.9]]),
                             torch.full((1,2,2),2.),torch.ones(1,2,dtype=torch.bool),0)
    assert pose.shape==v.shape==(1,17) and torch.isfinite(v).all()
    assert len(calls)==1 and torch.equal(calls[0],torch.tensor([[True,False]]))
    idle=[j for j,n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith('right_')]
    torch.testing.assert_close(pose[:,idle],q[:,idle])


def test_analytic_schedule_collision_context_works_without_top_entry_override():
    from amortized_opt.cost import CostConfig, FusedCost
    from amortized_opt.rhythm import DrumKit
    from amortized_opt.trajectory_surrogate import TrajectorySurrogate
    from isaac_rl.rnea_cache import build_or_load
    cost=FusedCost(CostConfig(),max_horizon=10,device='cpu',rnea=build_or_load(device='cpu'))
    surrogate=TrajectorySurrogate(cost,DrumKit.from_layout())
    q=torch.as_tensor(ready_joint_pos()[None],dtype=torch.float32)
    result=surrogate.evaluate(q,torch.zeros_like(q),q[:,None],torch.zeros_like(q[:,None]),
        torch.tensor([[[0,-1]]]),torch.tensor([[.1]]),torch.full((1,1,2),2.),
        torch.ones(1,1,dtype=torch.bool),torch.full((1,1,2),.5),return_segments=True)
    assert torch.isfinite(result['score']).all()
    assert result['requested'].item()==1
