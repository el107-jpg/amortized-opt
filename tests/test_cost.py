"""Unit tests for the fused cost: conformance, masking, gradients, syncs.

The heavier baseline validation (hinge separation, noise monotonicity, descent)
lives in ``test_cost_validation.py``; the full report is
``scripts/validate_cost.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import N_DOF, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.train_cost import phase2_cost_config
from amortized_opt.fk import stick_axis_local

_HERE = Path(__file__).resolve().parent
FIXTURE = _HERE / "data" / "trajopt_hpipm_fixture.npz"
FIXTURE_CASES = _HERE / "data" / "trajopt_hpipm_fixture_cases.json"


# ------------------------------------------------------------------ setup --
@pytest.fixture(scope="module")
def fixture_run():
    if not FIXTURE.exists():
        pytest.skip("trajopt fixture missing")
    blob = json.loads(FIXTURE_CASES.read_text())
    cases = [A.EvalCase.from_dict(d) for d in blob["cases"]]
    z = np.load(FIXTURE)
    trajs = {int(k.split("_")[1]): np.asarray(z[k], float) for k in z.files
             if k.startswith("case_")}
    cases = [c for c in cases if c.case_id in trajs]
    return cases, trajs


@pytest.fixture(scope="module")
def cost_cpu():
    return FusedCost(CostConfig(), max_horizon=40, device="cpu", dtype=torch.float64)


def test_target_top_entry_uses_physical_face_and_keeps_side_wrist_other_checks(cost_cpu, monkeypatch):
    cost = cost_cpu
    D, S, P = cost.n_drums, 4, cost.n_points
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    heads = torch.full((D, 3), 20., dtype=torch.float64); heads[0] = tensor([0, 0, 1])
    axes = tensor([0, 0, 1]).expand(D, -1)
    monkeypatch.setattr(cost, 'c_heads', heads)
    monkeypatch.setattr(cost, 'c_axes', axes)
    monkeypatch.setattr(cost, 'c_radii', torch.full((D,), .15, dtype=torch.float64))
    monkeypatch.setattr(cost, 'c_half_h', torch.full((D,), .1, dtype=torch.float64))
    bodies = heads-.1*axes
    monkeypatch.setattr(cost, 'c_body_c', bodies)
    monkeypatch.setattr(cost, 'c_under_c', heads-.25*axes)
    monkeypatch.setattr(cost, 'c_heights_uc', torch.full((D,), .21, dtype=torch.float64))
    monkeypatch.setattr(cost, 'c_under_half', torch.full((D,), .04, dtype=torch.float64))
    tip = tensor([[0, 0, 1.1], [0, 0, 1.02], [0, 0, 1.005], [0, 0, .98]])
    points = tip[:, None].expand(-1, P, -1).clone()
    points[:, 1] = tip+tensor([0, 0, .4])
    points[:, -1] = tensor([0, 0, 1.7])
    idle = torch.full_like(points, 30.)
    monkeypatch.setattr(cost, '_points_world', lambda frames, hand, ctx: points if hand == 'left' else idle)
    frames = {'torso': (torch.eye(3, dtype=torch.float64)[None].expand(S, -1, -1),
                        torch.full((S, 3), 40., dtype=torch.float64))}
    ctx = _synthetic_ctx(1, D, 'cpu', torch.float64,
                         kl=torch.tensor([50]), kr=torch.tensor([50]), active=(1., 0.))
    ctx.p_star[:] = tensor([0, 0, .99])  # aim is 10 mm below physical head
    ctx.n_out[:] = tensor([0, 0, 1]); ctx.hit_radius[:] = .12
    ctx.disk_offset.zero_(); ctx.has_half_plane.zero_()
    ctx.contact_sigma[:] = .2; ctx.shaft_allowed.zero_()
    masks = {'left': torch.ones(1, S, dtype=torch.float64),
             'right': torch.zeros(1, S, dtype=torch.float64)}
    def clearance():
        diag = {}
        cost._collision(frames, 1, S, ctx, masks, {}, {}, {}, diag, {}, None)
        return diag['collision_min_sdf'].item()
    assert clearance() < -.01  # early target body contact under checkpoint law
    ctx.target_top_entry_depth_m = .07
    assert clearance() >= 0
    ctx.p_star[:] = tensor([0, 0, .94])
    assert clearance() >= 0  # collision face is independent of the inward aim
    # The target's wrist sphere always remains checked.
    points[:, -1] = tensor([0, 0, .98])
    assert clearance() < 0
    points[:, -1] = tensor([0, 0, 1.7])
    # A different drum in the same path gets no target exemption.
    bodies[1] = bodies[0]
    assert clearance() < 0
    bodies[1] = heads[1]-.1*axes[1]
    # Entering laterally below the head stays a collision.
    tip[:] = tensor([[.25, 0, .97], [.18, 0, .97], [.12, 0, .97], [.08, 0, .97]])
    points[:, 0] = tip; points[:, 1] = tip+tensor([0, 0, .4])
    assert clearance() < 0
    # A randomized tilt must rotate the physical cylinder and certificate,
    # as well as the terminal normal supplied in the goal tokens.
    rotation = tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    local_tip = tensor([[0, 0, .1], [0, 0, .02], [0, 0, .005], [0, 0, -.02]])
    head = heads[0]
    points[:, 0] = local_tip @ rotation.T + head
    points[:, 1] = (local_tip+tensor([0, 0, .4])) @ rotation.T + head
    points[:, -1] = tensor([0, 0, .7]) @ rotation.T + head
    ctx.drum_head_centers = heads[None]
    ctx.drum_axes = (axes @ rotation.T)[None]
    ctx.n_out[:] = axes[0] @ rotation.T
    assert clearance() >= 0
    ctx.drum_axes = None
    assert clearance() < 0


def test_v4_target_collision_exempts_downward_upper_side_but_not_deep_or_horizontal(cost_cpu,monkeypatch):
    cost=cost_cpu.with_config(contact_scoring_mode='isaac_v4')
    D,S,P=cost.n_drums,4,cost.n_points
    t=lambda value:torch.as_tensor(value,dtype=torch.float64)
    heads=torch.full((D,3),20.,dtype=torch.float64);heads[1]=t([0.,0.,1.])
    axes=t([0.,0.,1.]).expand(D,-1)
    monkeypatch.setattr(cost,'c_heads',heads)
    monkeypatch.setattr(cost,'c_axes',axes)
    monkeypatch.setattr(cost,'c_radii',torch.full((D,),.15,dtype=torch.float64))
    monkeypatch.setattr(cost,'c_half_h',torch.full((D,),.05,dtype=torch.float64))
    monkeypatch.setattr(cost,'c_body_c',heads-.05*axes)
    monkeypatch.setattr(cost,'c_under_c',heads-.25*axes)
    monkeypatch.setattr(cost,'c_heights_uc',torch.full((D,),.21,dtype=torch.float64))
    monkeypatch.setattr(cost,'c_under_half',torch.full((D,),.04,dtype=torch.float64))
    points=torch.zeros(S,P,3,dtype=torch.float64);points[:,-1]=t([0.,0.,3.])
    far=torch.full_like(points,30.)
    monkeypatch.setattr(cost,'_points_world',lambda frames,hand,ctx:points if hand=='left' else far)
    ctx=_synthetic_ctx(1,D,'cpu',torch.float64,kl=torch.tensor([2]),
        kr=torch.tensor([50]),active=(1.,0.))
    ctx.target_drum[:,0]=1;ctx.shaft_allowed[:,0]=1.
    ctx.p_star[:,0]=t([0.,0.,.99]);ctx.n_out[:,0]=t([0.,0.,1.])
    masks={'left':torch.ones(1,S,dtype=torch.float64),
           'right':torch.zeros(1,S,dtype=torch.float64)}
    frames={'torso':(torch.eye(3,dtype=torch.float64)[None].expand(S,-1,-1),
                    torch.full((S,3),40.,dtype=torch.float64))}
    def check(path):
        points[:,:-1]=t(path)[:,None]
        diag={};cost._collision(frames,1,S,ctx,masks,{}, {}, {},diag,{},None)
        return float(diag['collision_min_sdf']),float(diag['v4_upper_side_exempt_samples_left'][0])
    good=check([[.159,0.,1.08],[.159,0.,1.001],[.155,0.,.997],[.154,0.,.996]])
    deep=check([[.159,0.,1.08],[.159,0.,.998],[.155,0.,.994],[.154,0.,.993]])
    horizontal=check([[.159,0.,1.08],[.159,0.,.997],[.155,0.,.997],[.154,0.,.997]])
    top=check([[.10,0.,1.08],[.10,0.,1.02],[.10,0.,.997],[.10,0.,.98]])
    assert good[1]>0 and good[0]>=0
    assert deep[0]<0 and horizontal[0]<0
    assert top[0]>=0


def test_new_phase2_capsule_contact_objective_has_finite_terminal_and_collision_gradients():
    cfg=phase2_cost_config(n_sub=1)
    assert cfg.contact_scoring_mode=='isaac_v5'
    class ZeroRNEA:
        def to(self,*_args):return self
        def rnea(self,q,*_args):return torch.zeros_like(q)
    # This geometry/gradient test does not need Pinocchio's torque model.
    cost=FusedCost(cfg,max_horizon=8,device='cpu',dtype=torch.float32,rnea=ZeroRNEA())
    q0=torch.as_tensor(ready_joint_pos(),dtype=torch.float32)[None]
    waypoints=q0[:,None].expand(1,8,-1).clone().requires_grad_(True)
    ctx=_synthetic_ctx(1,cost.n_drums,'cpu',torch.float32,
        kl=torch.tensor([5]),kr=torch.tensor([8]),active=(1.,0.))
    ctx.target_drum[:,0]=1;ctx.shaft_allowed[:,0]=1.
    ctx.n_out[:,0]=cost.c_axes[1]
    ctx.p_star[:,0]=cost.c_heads[1]-.01*cost.c_axes[1]
    ctx.hit_radius[:,0]=cost.c_radii[1]
    ctx.v_star[:,0]=-2.4*cost.c_axes[1]
    output=cost.forward(q0,waypoints,torch.zeros_like(q0),ctx)
    assert torch.isfinite(output.total).all()
    assert 'v4_upper_side_exempt_samples_left' in output.diagnostics
    assert 'v4_speed_quality_left' in output.diagnostics
    output.total.sum().backward()
    assert waypoints.grad is not None and torch.isfinite(waypoints.grad).all()


def test_v4_terminal_scores_probe_top_and_upper_side_not_legacy_deep_side(cost_cpu,monkeypatch):
    from types import SimpleNamespace
    cost=cost_cpu.with_config(contact_scoring_mode='isaac_v4',w_terminal_aim=0.)
    legacy=cost_cpu.with_config(contact_scoring_mode='legacy',w_terminal_aim=0.)
    for scorer in (cost,legacy):
        heads=scorer.c_heads.clone();heads[1]=0.
        monkeypatch.setattr(scorer,'c_heads',heads)
        monkeypatch.setitem(scorer.c_mount_p,'left',torch.zeros(3,dtype=torch.float64))
        monkeypatch.setitem(scorer.c_mount_axis,'left',torch.tensor([0.,0.,1.],dtype=torch.float64))
        monkeypatch.setitem(scorer.c_mount_p,'right',torch.zeros(3,dtype=torch.float64))
        monkeypatch.setitem(scorer.c_mount_axis,'right',torch.tensor([0.,0.,1.],dtype=torch.float64))
        probes=torch.zeros_like(scorer.c_points['left'])
        probes[0,2]=.2;probes[1:-1,2]=-.2
        monkeypatch.setitem(scorer.c_points,'left',probes)
        def frame(q,_name):
            return torch.eye(3,dtype=q.dtype).expand(len(q),-1,-1),q[:,:3]
        monkeypatch.setattr(scorer.fk,'frame',frame)
    B,D=1,cost.n_drums
    ctx=_synthetic_ctx(B,D,'cpu',torch.float64,kl=torch.tensor([2]),
        kr=torch.tensor([2]),active=(1.,0.))
    ctx.target_drum[:,0]=1;ctx.shaft_allowed[:,0]=1.
    ctx.p_star[:,0]=torch.tensor([0.,0.,-.01],dtype=torch.float64)
    ctx.n_out[:,0]=torch.tensor([0.,0.,1.],dtype=torch.float64)
    ctx.hit_radius[:,0]=.15
    ctx.v_star[:,0]=torch.tensor([0.,0.,-2.4],dtype=torch.float64)
    def score(current_z,previous_z,radial,scorer):
        current=torch.zeros(1,N_DOF,dtype=torch.float64)
        previous=torch.zeros_like(current)
        current[0,:3]=torch.tensor([radial,0.,current_z-.2])
        previous[0,:3]=torch.tensor([radial+.004,0.,previous_z-.2])
        class Spline:
            x=current[:,None]
            max_segments=2
            def knot(self,k):return current if int(k[0])==2 else previous
        raw,viol,live,diag,counts={},{},{},{},{}
        scorer._terminal(Spline(),ctx,raw,viol,live,diag,counts)
        return float(raw['terminal_region']),diag
    upper,upper_diag=score(-.003,.001,.148,cost)
    deep,deep_diag=score(-.006,-.002,.148,cost)
    top,top_diag=score(.01,.03,.10,cost)
    old_deep,_=score(-.006,-.002,.148,legacy)
    assert upper<deep and top<deep
    assert deep>old_deep
    assert upper_diag['v4_speed_quality_left'][0]<.25  # 0.2 m/s normal speed
    assert torch.isfinite(top_diag['v4_top_contact_distance_left']).all()
    ctx.has_half_plane[:,0]=1.
    ctx.half_plane_dir[:,0]=torch.tensor([1.,0.,0.],dtype=torch.float64)
    near,_=score(.01,.03,.10,cost)
    far,_=score(.01,.03,-.10,cost)
    old_far,_=score(.01,.03,-.10,legacy)
    assert far==pytest.approx(near,abs=1e-6)
    assert old_far>far+.01  # Old near-half-disk rule rejects valid far top.
    shaft=cost.c_points['left'].clone()
    shaft[:-1]=torch.tensor([1.,0.,1.],dtype=torch.float64)
    shaft[3]=torch.tensor([0.,0.,.2],dtype=torch.float64)
    monkeypatch.setitem(cost.c_points,'left',shaft)
    shaft_hit,_=score(-.003,.001,.148,cost)
    shaft[3]=torch.tensor([1.,0.,1.],dtype=torch.float64)
    without_shaft,_=score(-.003,.001,.148,cost)
    assert shaft_hit+.01<without_shaft


def test_v5_terminal_repairs_a_hit_between_isaac_shaft_probes(cost_cpu,monkeypatch):
    v4=cost_cpu.with_config(contact_scoring_mode='isaac_v4',w_terminal_aim=0.)
    v5=cost_cpu.with_config(contact_scoring_mode='isaac_v5',w_terminal_aim=0.)
    dtype=torch.float64
    for scorer in (v4,v5):
        heads=scorer.c_heads.clone();heads[1]=torch.tensor([-.025,0.,0.],dtype=dtype)
        monkeypatch.setattr(scorer,'c_heads',heads)
        radii=scorer.c_radii.clone();radii[1]=.009
        monkeypatch.setattr(scorer,'c_radii',radii)
        monkeypatch.setitem(scorer.c_mount_p,'left',torch.zeros(3,dtype=dtype))
        monkeypatch.setitem(scorer.c_mount_axis,'left',torch.tensor([1.,0.,0.],dtype=dtype))
        # v4's archived shaft cloud reaches the wrong, handle-side half.
        cloud=scorer.c_points['left'].clone()
        cloud[0]=torch.tensor([.2,0.,0.],dtype=dtype)
        cloud[1:-1,0]=torch.linspace(-.2,.02,cloud.shape[0]-2,dtype=dtype)
        cloud[1:-1,1:]=0.
        monkeypatch.setitem(scorer.c_points,'left',cloud)
        def frame(q,_name):
            return torch.eye(3,dtype=q.dtype).expand(len(q),-1,-1),q[:,:3]
        monkeypatch.setattr(scorer.fk,'frame',frame)
    ctx=_synthetic_ctx(1,v5.n_drums,'cpu',dtype,kl=torch.tensor([2]),
                       kr=torch.tensor([2]),active=(1.,0.))
    ctx.target_drum[:,0]=1;ctx.shaft_allowed[:,0]=1.
    ctx.p_star[:,0]=torch.tensor([-.025,0.,0.],dtype=dtype)
    ctx.n_out[:,0]=torch.tensor([0.,0.,1.],dtype=dtype)
    ctx.hit_radius[:,0]=.009
    ctx.v_star[:,0]=torch.tensor([0.,0.,-2.4],dtype=dtype)
    previous=torch.zeros(1,N_DOF,dtype=dtype)
    previous[0,:3]=torch.tensor([-.2,0.,.02],dtype=dtype)
    def evaluate(scorer,y):
        current=torch.zeros(1,N_DOF,dtype=dtype)
        current[0,:3]=torch.tensor([-.2,y,.009],dtype=dtype)
        current.requires_grad_(True)
        class Spline:
            x=current[:,None]
            max_segments=2
            def knot(self,k):return current if int(k[0])==2 else previous
        raw,viol,live,diag,counts={},{},{},{},{}
        scorer._terminal(Spline(),ctx,raw,viol,live,diag,counts)
        return current,raw,diag
    _,_,old=evaluate(v4,0.)
    current,raw,new=evaluate(v5,0.)
    assert old['v4_geometric_min_distance_left'][0]>.007
    assert new['v4_geometric_min_distance_left'][0]<1e-6
    moved,repair,_=evaluate(v5,.03)
    repair['terminal_region'].backward()
    assert torch.isfinite(moved.grad).all()
    assert moved.grad[0,1]>0  # gradient descent moves inward toward the head


def _pack(cases, trajs, H, dtype=torch.float64):
    B = len(cases)
    q0 = torch.zeros(B, N_DOF, dtype=dtype)
    v0 = torch.zeros(B, N_DOF, dtype=dtype)
    wp = torch.zeros(B, H, N_DOF, dtype=dtype)
    for i, c in enumerate(cases):
        Q = trajs[c.case_id]
        q0[i] = torch.as_tensor(Q[0], dtype=dtype)
        v0[i] = torch.as_tensor(c.qdot_start, dtype=dtype)
        wp[i, : c.N] = torch.as_tensor(Q[1 : c.N + 1], dtype=dtype)
        wp[i, c.N :] = torch.as_tensor(Q[c.N], dtype=dtype)
    return q0, wp, v0


def _synthetic_ctx(B, D, device, dtype, *, kl, kr, active=(1.0, 1.0)):
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


# ------------------------------------------------------- harness agreement --
def test_torque_matches_the_scorer_exactly(fixture_run, cost_cpu):
    """``physics_mode="knot_fd"`` reproduces ``TrajectoryScorer``'s torque."""
    cases, trajs = fixture_run
    H = cost_cpu.max_horizon
    q0, wp, v0 = _pack(cases, trajs, H)
    ctx = A.strike_context_from_cases(cases, drum_index=cost_cpu.drum_index,
                                      dtype=torch.float64)
    out = cost_cpu.forward(q0, wp, v0, ctx)
    scorer = A.TrajectoryScorer(device="cpu")
    ref_max = np.array([scorer.score(c, trajs[c.case_id]).tau_max_nm for c in cases])
    ref_ex = np.array(
        [scorer.score(c, trajs[c.case_id]).tau_excess_max_nm for c in cases]
    )
    got_max = out.diagnostics["torque_max"].numpy()
    got_ex = out.diagnostics["torque_excess_max"].numpy()
    assert np.abs(got_max - ref_max).max() < 1e-9, np.abs(got_max - ref_max).max()
    assert np.abs(got_ex - ref_ex).max() < 1e-9


def test_contact_geometry_matches_the_scorer(fixture_run, cost_cpu):
    cases, trajs = fixture_run
    q0, wp, v0 = _pack(cases, trajs, cost_cpu.max_horizon)
    ctx = A.strike_context_from_cases(cases, drum_index=cost_cpu.drum_index,
                                      dtype=torch.float64)
    out = cost_cpu.forward(q0, wp, v0, ctx)
    scorer = A.TrajectoryScorer(device="cpu")
    rows = [scorer.score(c, trajs[c.case_id]) for c in cases]
    ref_pos = np.array([r.contact_pos_err_mm for r in rows]) / 1000.0
    ref_elev = np.abs(np.array([r.stick_elev_deg for r in rows]))
    assert np.abs(out.diagnostics["contact_pos_err"].numpy() - ref_pos).max() < 1e-9
    assert np.abs(out.diagnostics["stick_elev_deg"].numpy() - ref_elev).max() < 1e-9


def test_collision_agrees_with_the_scorer_when_substepping_matches(fixture_run):
    """At ``n_sub = 4`` the cost sees the same interpolated path the scorer does.

    The residual gap is the interpolant: the scorer walks a LINEAR interpolation
    between waypoints, the cost walks the C2 spline.
    """
    cases, trajs = fixture_run
    cost = FusedCost(CostConfig(n_sub=4), max_horizon=40, device="cpu",
                     dtype=torch.float64)
    q0, wp, v0 = _pack(cases, trajs, cost.max_horizon)
    ctx = A.strike_context_from_cases(cases, drum_index=cost.drum_index,
                                      dtype=torch.float64)
    out = cost.forward(q0, wp, v0, ctx)
    scorer = A.TrajectoryScorer(device="cpu")
    ref = np.array(
        [scorer.score(c, trajs[c.case_id]).coll_min_sdf_all_m for c in cases]
    )
    got = out.diagnostics["collision_min_sdf"].numpy()
    assert np.abs(got - ref).max() < 0.02, np.abs(got - ref).max()


# ----------------------------------------------------------------- masking --
def test_masked_elements_cannot_change_the_cost():
    torch.manual_seed(0)
    H, B = 24, 6
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.03, dim=1
    )
    v0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.3
    kl = torch.tensor([10, 24, 7, 18, 12, 24])
    kr = torch.tensor([24, 11, 20, 6, 24, 9])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    base = cost.forward(q0, wp, v0, ctx)

    lengths = cost._lengths_from_context(ctx)
    per = lengths[:, cost.spline._group_np]  # (B, n)
    idx = torch.arange(H).view(1, -1, 1)
    beyond = idx >= per.unsqueeze(1)
    wp2 = wp.clone()
    wp2[beyond] = 17.0 * torch.randn(int(beyond.sum()), dtype=torch.float64)
    other = cost.forward(q0, wp2, v0, ctx)

    assert torch.equal(base.total, other.total)
    for k in base.raw:
        assert torch.equal(base.raw[k], other.raw[k]), k


def test_each_hand_lands_on_its_own_hit_index():
    """Asymmetric horizons: the terminal term reads each hand's own knot."""
    torch.manual_seed(1)
    H, B = 30, 4
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.04, dim=1
    )
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    kl = torch.tensor([11, 30, 17, 8])
    kr = torch.tensor([30, 14, 9, 22])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    out = cost.forward(q0, wp, v0, ctx)

    # Recompute the diagnostic by hand, FK at each hand's own knot.
    x = torch.cat([q0.unsqueeze(1), wp], dim=1)
    manual = torch.zeros(B, dtype=torch.float64)
    for hi, (hand, k) in enumerate((("left", kl), ("right", kr))):
        qk = x[torch.arange(B), k]
        R, p = cost.fk.frame(qk, f"{hand}_wrist")
        loc = torch.as_tensor(
            A.stick_contact_offset(hand, 0.20), dtype=torch.float64
        ).view(1, 3, 1)
        pc = p + (R @ loc).squeeze(-1)
        manual = manual + (pc - ctx.p_star[:, hi]).norm(dim=-1)
    assert torch.allclose(out.diagnostics["contact_pos_err"], manual / 2.0, atol=1e-10)

    # And the terminal term is sensitive to each hand's own knot, not the other's.
    for hi, k in enumerate((kl, kr)):
        wp2 = wp.clone()
        rows = torch.arange(B)
        wp2[rows, (k - 1).clamp_min(0)] += 0.05
        d = cost.forward(q0, wp2, v0, ctx).raw["terminal_region"]
        assert not torch.allclose(d, out.raw["terminal_region"])


def test_inactive_hand_contributes_no_terminal_cost():
    torch.manual_seed(2)
    H, B = 20, 3
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.03, dim=1
    )
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    both = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    left_only = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k,
                               active=(1.0, 0.0))
    a = cost.forward(q0, wp, v0, both).raw["terminal_region"]
    b = cost.forward(q0, wp, v0, left_only).raw["terminal_region"]
    assert not torch.allclose(a, b)
    # ... and moving the RIGHT arm cannot change a left-only terminal cost
    wp2 = wp.clone()
    right = cost.spline._group_cols[2]
    wp2[:, :, right] += 0.3
    c = cost.forward(q0, wp2, v0, left_only).raw["terminal_region"]
    assert torch.allclose(b, c, atol=1e-12)


# --------------------------------------------------------------- gradients --
def test_waist_yaw_monotonic_gradient_matches_its_spline_scalar():
    """Internal spline samples contribute on both sides of each increment."""
    H, B = 5, 2
    cost = FusedCost(CostConfig(n_sub=2), max_horizon=H, device="cpu",
                     dtype=torch.float64)
    yaw_i = cost.joint_names.index("waist_yaw_joint")
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = torch.zeros(B, H, N_DOF, dtype=torch.float64)
    wp[0, :, yaw_i] = torch.tensor(
        [-0.22, 0.18, -0.15, 0.8, 0.9], dtype=torch.float64
    )
    wp.requires_grad_(True)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    kl = torch.tensor([3, H])
    kr = torch.tensor([H, H])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    ctx.active[0] = torch.tensor([1.0, 0.0], dtype=torch.float64)
    ctx.active[1] = 0.0
    ctx.waist_yaw_ref = torch.full((B, H), 1.0, dtype=torch.float64)

    def raw_for(x):
        return cost.forward(q0, x, v0, ctx).raw["waist_yaw_monotonic"]

    raw_for(wp).sum().backward()
    grad = wp.grad.detach().clone()
    eps = 1.0e-6
    fd = torch.zeros_like(grad)
    for k in range(H):
        plus = wp.detach().clone()
        minus = wp.detach().clone()
        plus[0, k, yaw_i] += eps
        minus[0, k, yaw_i] -= eps
        fd[0, k, yaw_i] = (raw_for(plus).sum() - raw_for(minus).sum()) / (2 * eps)

    assert raw_for(wp)[1].item() == 0.0  # inactive batch row stays masked
    assert torch.allclose(grad, fd, atol=2e-6, rtol=2e-5)


def test_gradcheck_small():
    torch.manual_seed(3)
    H, B = 6, 2
    cfg = CostConfig(
        w_terminal_aim=1.0, w_smooth_vel=1e-3, w_axial_twist=1e-2,
        contact_mode="softmin",
    )
    cost = FusedCost(cfg, max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = (torch.randn(B, N_DOF, dtype=torch.float64) * 0.2).requires_grad_(True)
    wp = (torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.2).requires_grad_(True)
    v0 = (torch.randn(B, N_DOF, dtype=torch.float64) * 0.2).requires_grad_(True)
    kl = torch.tensor([6, 4])
    kr = torch.tensor([5, 6])
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=kl, kr=kr)
    ctx.shaft_allowed = torch.ones(B, 2, dtype=torch.float64)

    def f(a, b, c):
        return cost.forward(a, b, c, ctx).total

    assert torch.autograd.gradcheck(f, (q0, wp, v0), eps=1e-6, atol=2e-6, rtol=1e-3)


def test_gradients_are_finite_at_training_scale():
    torch.manual_seed(4)
    H, B = 50, 64
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float32)
    q0 = torch.randn(B, N_DOF) * 0.2
    wp = (q0.unsqueeze(1) + torch.cumsum(torch.randn(B, H, N_DOF) * 0.03, 1)
          ).requires_grad_(True)
    v0 = torch.randn(B, N_DOF) * 0.5
    kl = torch.randint(20, H + 1, (B,))
    kr = torch.randint(20, H + 1, (B,))
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float32, kl=kl, kr=kr)
    out = cost.forward(q0, wp, v0, ctx)
    out.total.sum().backward()
    assert torch.isfinite(out.total).all()
    assert torch.isfinite(wp.grad).all()
    assert float(wp.grad.abs().max()) > 0.0


def test_elevation_near_pole_gradients_are_finite():
    """Stick along the head normal used to NaN ``asin'(±1)`` in ``terminal_elev``."""
    H, B = 8, 2
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float32)
    q0 = torch.as_tensor(ready_joint_pos(), dtype=torch.float32).expand(B, -1).contiguous()
    v0 = torch.zeros(B, N_DOF)
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    R, _ = cost.fk.frame(q0, "left_wrist")
    u = (
        R @ torch.as_tensor(stick_axis_local("left"), dtype=torch.float32).view(1, 3, 1)
    ).squeeze(-1)
    for tilt in (0.0, 1e-3):
        wp = q0.unsqueeze(1).expand(B, H, N_DOF).contiguous().clone().requires_grad_(True)
        n = -u + tilt * torch.tensor([1.0, 0.0, 0.0])
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float32, kl=kl, kr=kr)
        ctx.n_out[:, 0] = n
        ctx.n_out[:, 1] = n
        out = cost.forward(q0, wp, v0, ctx)
        out.raw["terminal_elev"].sum().backward()
        assert torch.isfinite(out.raw["terminal_elev"]).all(), tilt
        assert torch.isfinite(wp.grad).all(), tilt
        wp2 = q0.unsqueeze(1).expand(B, H, N_DOF).contiguous().clone().requires_grad_(True)
        out2 = cost.forward(q0, wp2, v0, ctx)
        out2.total.sum().backward()
        assert torch.isfinite(out2.total).all(), tilt
        assert torch.isfinite(wp2.grad).all(), tilt


# ------------------------------------------------------------------ device --
def test_no_device_host_sync(cuda_device):
    H, B = 50, 32
    cost = FusedCost(CostConfig(), max_horizon=H, device=cuda_device,
                     dtype=torch.float32)
    torch.manual_seed(5)
    q0 = (torch.randn(B, N_DOF) * 0.2).to(cuda_device)
    wp = (q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, device=cuda_device) * 0.03, 1)).requires_grad_(True)
    v0 = torch.randn(B, N_DOF, device=cuda_device) * 0.5
    kl = torch.randint(20, H + 1, (B,), device=cuda_device)
    kr = torch.randint(20, H + 1, (B,), device=cuda_device)
    ctx = _synthetic_ctx(B, cost.n_drums, cuda_device, torch.float32, kl=kl, kr=kr)
    cost.forward(q0, wp, v0, ctx).total.sum().backward()  # warm the caches
    wp.grad = None
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = cost.forward(q0, wp, v0, ctx)
        out.total.sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(out.total).all()


def test_cuda_matches_cpu(cuda_device):
    torch.manual_seed(6)
    H, B = 20, 8
    cpu = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float32)
    gpu = FusedCost(CostConfig(), max_horizon=H, device=cuda_device,
                    dtype=torch.float32)
    q0 = torch.randn(B, N_DOF) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(torch.randn(B, H, N_DOF) * 0.03, 1)
    v0 = torch.randn(B, N_DOF) * 0.3
    kl = torch.randint(8, H + 1, (B,))
    kr = torch.randint(8, H + 1, (B,))
    a = cpu.forward(q0, wp, v0,
                    _synthetic_ctx(B, cpu.n_drums, "cpu", torch.float32, kl=kl, kr=kr))
    b = gpu.forward(q0.to(cuda_device), wp.to(cuda_device), v0.to(cuda_device),
                    _synthetic_ctx(B, gpu.n_drums, cuda_device, torch.float32,
                                   kl=kl.to(cuda_device), kr=kr.to(cuda_device)))
    rel = ((a.total - b.total.cpu()).abs() / a.total.abs().clamp_min(1e-6)).max()
    assert float(rel) < 2e-3, float(rel)


# -------------------------------------------------------------- lagrangian --
def test_augmented_lagrangian_grows_then_relaxes(tmp_path):
    al = A.AugmentedLagrangian()
    v = {g: torch.tensor([0.05, 0.05]) for g in A.CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(v)
    grown = dict(al.lam)
    assert all(grown[g] > 1.0 for g in A.CONSTRAINT_GROUPS), grown
    zero = {g: torch.zeros(2) for g in A.CONSTRAINT_GROUPS}
    for _ in range(5):
        al.update(zero)
    assert all(al.lam[g] == grown[g] for g in A.CONSTRAINT_GROUPS)  # no drift at 0

    al2 = A.AugmentedLagrangian(A.ALConfig(decay=0.9))
    for _ in range(5):
        al2.update(v)
    hi = dict(al2.lam)
    for _ in range(20):
        al2.update(zero)
    assert all(al2.lam[g] < hi[g] for g in A.CONSTRAINT_GROUPS)

    p = al.save(tmp_path / "al.json")
    back = A.AugmentedLagrangian.load(p)
    assert back.lam == al.lam and back.steps == al.steps
    assert al.lam["collision"] <= al.cfg.lam_max


def test_lagrangian_weights_reach_the_cost():
    torch.manual_seed(7)
    H, B = 12, 3
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.2
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    base = cost.forward(q0, wp, v0, ctx)
    doubled = cost.forward(q0, wp, v0, ctx, weights={"torque": 2.0})
    warning = cost.cfg.w_torque * base.raw["torque"]
    # warning (cfg.w * softplus mean) is invariant to λ
    assert torch.allclose(cost.cfg.w_torque * doubled.raw["torque"], warning, atol=1e-12)
    # hard half (terms - warning) scales with λ: family key broadcasts onto
    # every torque_* live residual.  Peak is off (w_torque_peak=0).
    from amortized_opt.lagrangian import ACTUATOR_SEGMENT_NAMES, limit_group_name
    hard = sum(
        base.violations[limit_group_name("torque", s)]
        for s in ACTUATOR_SEGMENT_NAMES
    )
    assert torch.allclose(base.terms["torque"] - warning, hard, atol=1e-12)
    assert torch.allclose(doubled.terms["torque"] - warning, 2.0 * hard, atol=1e-12)
    assert torch.allclose(doubled.terms["collision"], base.terms["collision"])


def test_violations_are_zero_when_feasible():
    """The AL is driven by the RAW excess, which vanishes on a feasible path --
    unlike the softplus penalty, which never does."""
    H, B = 10, 2
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = torch.zeros(B, H, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    out = cost.forward(q0, wp, v0, ctx)
    for g in ("torque", "velocity", "joint_box"):
        assert float(out.violations[g].max()) == 0.0, g
        assert float(out.raw[g].min()) > 0.0, g  # softplus floor


def test_terminal_aim_is_not_al_scaled():
    """``lam_terminal`` must not multiply the in-plane aim pull."""
    torch.manual_seed(5)
    H, B = 8, 2
    cfg = CostConfig(
        w_terminal_aim=10.0, w_terminal_region=0.0,
        w_terminal_velocity=0.0, w_terminal_elev=0.0,
    )
    cost = FusedCost(cfg, max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.randn(B, N_DOF, dtype=torch.float64) * 0.2
    wp = q0.unsqueeze(1) + torch.cumsum(
        torch.randn(B, H, N_DOF, dtype=torch.float64) * 0.03, dim=1
    )
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    k = torch.full((B,), H)
    ctx = _synthetic_ctx(B, cost.n_drums, "cpu", torch.float64, kl=k, kr=k)
    a = cost.forward(q0, wp, v0, ctx, weights={"terminal": 1.0})
    b = cost.forward(q0, wp, v0, ctx, weights={"terminal": 4.0})
    assert torch.allclose(a.terms["terminal_aim"], b.terms["terminal_aim"])
    assert float(a.raw["terminal_aim"].mean()) >= 0.0
    assert "terminal_aim" in a.terms
