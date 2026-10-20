"""Repair one dense-audited hi-hat candidate per source, preserving the baseline table."""
import argparse,sys,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch,amortized_opt as A
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import cost_for_horizon,metrics,refine,QPConfig
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',default='cuda:1');p.add_argument('--hand',type=int,choices=[0,1],required=True)
p.add_argument('--checkpoint',default=str(ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt'))
p.add_argument('--terminal-seed',action='store_true',help='repair the generator initialization at a fixed deadline')
p.add_argument('--dls-seed',action='store_true',help='use a smooth four-step terminal correction before QP')
p.add_argument('--duration',type=float,default=1.)
p.add_argument('--fixed-duration',type=float,default=None,
               help='regenerate the decoder seed at this deadline before correction')
p.add_argument('--source',default=None,help='optional source drum filter')
p.add_argument('--sample',type=int,default=None,help='optional departure sample index')
p.add_argument('--iterations',type=int,default=15)
p.add_argument('--deep-backtrack',action='store_true')
p.add_argument('--torque-time-bins',type=int,default=1)
args=p.parse_args()
if args.fixed_duration is not None and not .02 <= args.fixed_duration <= 2.:
 p.error('--fixed-duration must be between 0.02 and 2 seconds')
torch.set_num_threads(2);dev=args.device;kit=A.DrumKit.from_layout();hat=kit.names.index('hihat')
ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
cfg=A.CostConfig(**ck['train_cfg']['cost']);cfg.n_sub=4;cfg.physics_mode='spline';cfg.w_velocity=max(1,cfg.w_velocity);cfg.w_joint_box=max(1,cfg.w_joint_box)
cost=A.FusedCost(cfg,max_horizon=100,device=dev,rnea=load_cached(device=dev));cost.calibration_active_hand_only=True
if args.terminal_seed or args.fixed_duration is not None:
 from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
 model=A.build_model(A.model_config_from_checkpoint(ck),dev);model.load_state_dict(ck['model']);model.eval()
 generator=DecoderHitPoseGenerator(model,cost,kit,horizon=100,
     waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
base=args.audit;out=args.out;out.mkdir(parents=True,exist_ok=True)
rows=json.load(open(base/'cases.json'));done=[]
for h in [args.hand]:
 data=np.load(base/f'closest_h{h}.npz');cases=[r for r in rows if r['hand']==['left','right'][h]]
 for src in sorted(set(r['source'] for r in cases)):
  if args.source and src!=args.source:continue
  candidates=[r for r in cases if r['source']==src and (args.sample is None or r['sample']==args.sample)]
  if not candidates:continue
  r=min(candidates,key=lambda r:r['dense_metrics']['merit'])
  i=next(i for i,x in enumerate(cases) if x is r);N=int(data['lengths'][i]);q=torch.tensor(data['q'][i:i+1],device=dev);v=torch.tensor(data['v'][i:i+1],device=dev);traj=torch.tensor(data['paths'][i:i+1,:N],device=dev)
  target=np.full((1,2),-1);target[:,h]=hat;source=target.copy();source[:,h]=int(data['sources'][i]);ctx=strike_context(kit,target,source,[N],dev,speeds=np.full((1,2),2.));ctx.target_top_entry_depth_m=.07;L=torch.full((1,3),N,device=dev,dtype=torch.long)
  if args.terminal_seed or args.fixed_duration is not None:
   N=int(round((args.fixed_duration if args.fixed_duration is not None else args.duration)/.02))
   with torch.no_grad():
    plan=generator.predict(q,v,torch.tensor(target[:,None],device=dev),q.new_full((1,1),N*.02),
        q.new_full((1,1,2),2.),torch.ones(1,1,device=dev,dtype=torch.bool),0,return_plan=True)
    traj=(model.initial_trajectory(plan['context'],torch.cat((q,v),-1),plan['masks'])
          if args.terminal_seed else plan['traj'])[:,:N].clone()
    from amortized_opt.ik import arm_joint_indices
    idle=arm_joint_indices(('left','right')[1-h]);traj[:,:,idle]=q[:,None,idle]
   ctx=strike_context(kit,target,source,[N],dev,speeds=np.full((1,2),2.));ctx.target_top_entry_depth_m=.07
   L=torch.full((1,3),N,device=dev,dtype=torch.long)
  if args.dls_seed:
   from amortized_opt.train_cost import dls_terminal_teacher
   traj,_,_=dls_terminal_teacher(traj,ctx,cost,steps=4,window=32)
  short=cost_for_horizon(cost,N);dense=copy.copy(short);dense.cfg=copy.copy(short.cfg);dense.cfg.n_sub=32
  before=metrics(dense,q,v,traj,ctx,L,.07,signed_contact=True)
  search=(1.,.5,.25,.125,.0625,.03125,.015625,.0078125) if args.deep_backtrack else QPConfig().line_search
  fixed,_,_,info=refine(short,q,v,traj,ctx,L,h,cfg=QPConfig(iterations=args.iterations,trust_rad=.03,jacobian_chunk=32,signed_contact=True,line_search=search,torque_time_bins=args.torque_time_bins))
  after=metrics(dense,q,v,fixed,ctx,L,.07,signed_contact=True)
  if not after['success'][0]:
   fixed,_,_,second=refine(dense,q,v,fixed,ctx,L,h,cfg=QPConfig(iterations=10,trust_rad=.03,jacobian_chunk=32,signed_contact=True,line_search=search,torque_time_bins=args.torque_time_bins));after=metrics(dense,q,v,fixed,ctx,L,.07,signed_contact=True);info['dense_refine']=second
  summary=dict(hand=['left','right'][h],source=src,sample=r['sample'],duration_s=N*.02,seed='dls_corrected' if args.dls_seed else 'terminal_initialization' if args.terminal_seed else 'decoder',before={k:float(x[0]) for k,x in before.items()},after={k:float(x[0]) for k,x in after.items()},seconds=info['elapsed_s'])
  np.savez_compressed(out/f'h{h}_{src}.npz',q=q.cpu().numpy(),v=v.cpu().numpy(),before=traj.cpu().numpy(),after=fixed.cpu().numpy());(out/f'h{h}_{src}_qp.json').write_text(json.dumps(info,indent=2));done.append(summary);(out/f'summary_h{h}.json').write_text(json.dumps(done,indent=2));print(json.dumps(summary),flush=True)
