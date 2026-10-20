#!/usr/bin/env python3
"""Draw sampled departure poses with URDF arm FK and visible waist orientation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.constants import PELVIS_IN_ENV, STICK_TIP_SIGMA
from amortized_opt.fk import ChainFK
from amortized_opt.ik import arm_joint_indices, tip_position
from amortized_opt.strike import default_contact_sigma, STRIKE_SURFACE_OFFSET
from amortized_opt.rhythm import head_plane_basis
from amortized_opt.trajectory_surrogate import contact_jacobian
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import near_head_starts
from scripts.probe_transition_qp import save_npz, write_json

BODY_FACES = [[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [0,2,6,4], [1,3,7,5]]
BODY_LOCAL = np.array([[x,y,z] for x in (-.075,.075) for y in (-.11,.11) for z in (.04,.29)])
ARM_SUFFIXES = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                'wrist_roll', 'wrist_pitch', 'wrist_yaw']
COLORS = {'left':'#2563eb', 'right':'#d45c17'}


def sample_pending(cost, kit, meta, hand, src):
    rng = np.random.default_rng(np.random.SeedSequence([meta['seed'], hand, src]))
    _, lo, hi = cost._joint_box_hard_limits()
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    columns = list(range(3))+list(arm_joint_indices(('left','right')[hand]))
    qs, vs, diagnostics, tried = [], [], [], 0
    while sum(len(q) for q in qs) < meta['samples_per_source'] and tried < meta['max_start_trials']:
        count = min(32, meta['max_start_trials']-tried)
        yaws = rng.uniform(float(lo[0]), float(hi[0]), count)
        q,v,valid,diag = near_head_starts(cost,kit,('left','right')[hand],src,yaws,count,rng,
                                         ignore_idle_arm=True)
        bounded = ((q[:,columns] >= lo[columns]-1e-6)&(q[:,columns] <= hi[columns]+1e-6)).all(-1)
        take = np.flatnonzero((valid&bounded).cpu().numpy())
        take = take[:meta['samples_per_source']-sum(len(q) for q in qs)]
        qs.append(q[take]); vs.append(v[take])
        diagnostics.append({key:value[take] for key,value in diag.items()})
        tried += count
    return dict(q=torch.cat(qs).cpu().numpy(),v=torch.cat(vs).cpu().numpy(),
                **{key:np.concatenate([d[key] for d in diagnostics]) for key in diagnostics[0]})


def ring(center, normal, radius, count=48):
    x,y = head_plane_basis(normal)
    theta = np.linspace(0,2*np.pi,count)
    return center+radius*(np.cos(theta)[:,None]*x+np.sin(theta)[:,None]*y)


def configure(ax,kit):
    ax.set_xlim(-.25,1.0); ax.set_ylim(-.95,.95); ax.set_zlim(.3,1.42)
    ax.set_box_aspect((1.25,1.9,1.12)); ax.view_init(elev=27,azim=140)
    ax.set_xlabel('Forward x (m)',fontsize=8); ax.set_ylabel('Left y (m)',fontsize=8)
    ax.set_zlabel('Height (m)',fontsize=8); ax.tick_params(labelsize=7)


def draw_kit(ax,kit,src):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for d,(p,n,r) in enumerate(zip(kit.positions,kit.axes,kit.radii)):
        points=ring(p,n,r)
        color='#23875a' if d==src else '#94a3b8'
        ax.add_collection3d(Poly3DCollection([points],facecolor=color,edgecolor=color,
                                            alpha=.28 if d==src else .055,linewidth=.7))
        if d==src:
            ax.text(*(p+n*.03),kit.names[d].replace('_',' '),color=color,fontsize=9)


def draw_pose(ax,pose, *, color=None, idle=True, alpha=1.):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    body=np.asarray(pose['body']); waist=np.asarray(pose['waist'])
    torso_color=color or '#64748b'
    ax.add_collection3d(Poly3DCollection([body[f] for f in BODY_FACES],
        facecolor=torso_color,edgecolor=torso_color,alpha=.1*alpha,linewidth=.7))
    for hand in ('left','right'):
        active=hand==pose['hand']
        if not active and not idle: continue
        shade=(color or COLORS[hand]) if active else '#9ca3af'
        a=np.asarray(pose['arms'][hand]); stick=np.asarray(pose['sticks'][hand])
        ax.plot(*a.T,color=shade,lw=3.5 if active else 1.3,alpha=alpha if active else .35,
                marker='o',markersize=3 if active else 2)
        ax.plot(*np.stack([a[-1],np.mean(stick,0)]).T,color=shade,lw=1,ls=':',alpha=.65*alpha)
        ax.plot(*stick.T,color=shade,lw=3 if active else 1,alpha=alpha if active else .3)
        if active: ax.scatter(*stick[-1],color=shade,s=14,marker='^',depthshade=False)
    forward=np.asarray(pose['forward'])-waist
    ax.quiver(*waist,*forward,color=color or '#ca8a04',arrow_length_ratio=.23,lw=2,alpha=alpha)
    ax.scatter(*waist,color=color or '#ca8a04',s=20,depthshade=False)
    contact=np.asarray(pose['contact']); velocity=np.asarray(pose['contact_velocity'])*.08
    ax.scatter(*contact,color='#b91c8c',s=27,depthshade=False)
    ax.quiver(*contact,*velocity,color='#15803d',arrow_length_ratio=.25,lw=1.5,alpha=alpha)


def chosen(group):
    order=sorted(group,key=lambda p:p['q'][0])
    indexes=sorted(set([0,len(order)//2,len(order)-1]))
    return [order[i] for i in indexes] if order else []


def static_images(out,kit,poses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels=[Line2D([0],[0],color=COLORS['left'],lw=3,label='Left arm'),
            Line2D([0],[0],color=COLORS['right'],lw=3,label='Right arm'),
            Line2D([0],[0],color='#9ca3af',lw=2,label='Idle arm (constraints ignored)'),
            Line2D([0],[0],color='#ca8a04',lw=2,label='Torso forward / waist'),
            Line2D([0],[0],color='#b91c8c',marker='o',ls='',label='Sampled contact point'),
            Line2D([0],[0],color='#15803d',lw=2,label='Departure velocity × 80 ms')]
    for src,name in enumerate(kit.names):
        hands=(1,) if name=='floor_tom' else (0,1)
        fig=plt.figure(figsize=(16,6.5 if len(hands)==1 else 11))
        for row,hand in enumerate(hands):
            group=[p for p in poses if p['source']==src and p['hand']==('left','right')[hand]]
            examples=chosen(group)
            for col in range(3):
                ax=fig.add_subplot(len(hands),3,row*3+col+1,projection='3d')
                if col>=len(examples):
                    ax.set_axis_off()
                    ax.text2D(.2,.5,'Left floor tom excluded' if hand==0 and name=='floor_tom'
                              else 'No valid samples',transform=ax.transAxes)
                    continue
                p=examples[col];draw_kit(ax,kit,src);draw_pose(ax,p);configure(ax,kit)
                yaw,pitch,roll=np.rad2deg(p['q'][:3])
                ax.set_title(f'{p["hand"].title()} · sample {p["sample"]} · yaw {yaw:+.1f}°\n'
                    f'Pitch {pitch:+.1f}° / roll {roll:+.1f}° · outward {p["outward_speed"]:.2f} m/s',fontsize=10)
                if p['origin']=='reconstructed_pending':
                    ax.text2D(.01,.02,'Pending cohort: reconstructed from saved seed',
                              transform=ax.transAxes,fontsize=7,color='#64748b')
        fig.suptitle(f'{name.replace("_"," ").title()} — sampled starting poses'
                     +(' (right only; left excluded)' if name=='floor_tom' else '')+'\n'
                     'Three samples spanning waist yaw; URDF joint geometry, schematic torso',fontsize=17)
        fig.legend(handles=labels,loc='lower center',ncol=3,fontsize=9,frameon=False)
        fig.subplots_adjust(top=.78 if len(hands)==1 else .88,
                            bottom=.13 if len(hands)==1 else .1,
                            left=.01,right=.99,wspace=.03,hspace=.17)
        fig.savefig(out/f'{name}.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(18,11))
    preferred=['left','left','left','right','left','right','right']
    sample_colors=['#2563eb','#ea580c','#9333ea']
    for src,name in enumerate(kit.names):
        ax=fig.add_subplot(2,4,src+1,projection='3d')
        group=[p for p in poses if p['source']==src and p['hand']==preferred[src]]
        examples=chosen(group);draw_kit(ax,kit,src)
        for i,p in enumerate(examples):draw_pose(ax,p,color=sample_colors[i],idle=False,alpha=.8)
        configure(ax,kit)
        ax.set_zlabel('');ax.set_zticklabels([])
        description=' · '.join(f's{p["sample"]}: {np.rad2deg(p["q"][0]):+.0f}°' for p in examples)
        ax.set_title(f'{name.replace("_"," ")} — {preferred[src]}\n{description}',fontsize=10)
        if any(p['origin']=='reconstructed_pending' for p in examples):
            ax.text2D(.02,.02,'Reconstructed pending cohort',transform=ax.transAxes,fontsize=7)
    ax=fig.add_subplot(2,4,8);ax.set_axis_off()
    ax.text(0,.96,'Starting-state sampler\n\n'
        'Three samples: low / middle / high waist yaw\n'
        'Blue = low; orange = middle; purple = high\n'
        'Waist pitch and roll start at zero\n'
        'Target: 2–12 mm above the inset strike plane\n'
        'Strike plane: 10 mm below the physical head\n'
        'Contact point: magenta dot\n'
        'Green arrow: departure velocity × 80 ms\n'
        'Idle arm omitted in this overview\n\n'
        'Detailed per-drum images show both arms.\n'
        'HTML viewer contains every sampled pose.\n'
        'Coordinates: environment frame\n'
        'Pelvis: fixed 0.8 m above the floor\n\n'
        'Torso schematic; arm/shaft positions from FK.',va='top',fontsize=10)
    fig.suptitle('Starting poses across the drum kit — three yaw samples per drum',fontsize=19)
    fig.subplots_adjust(top=.89,bottom=.05,left=.005,right=.995,wspace=.03,hspace=.21)
    fig.savefig(out/'overview.png',dpi=160);plt.close(fig)


def interactive(out,kit,poses):
    from plotly.offline import get_plotlyjs
    data=dict(names=list(kit.names),drums=[dict(center=p.tolist(),ring=ring(p,n,r).tolist())
        for p,n,r in zip(kit.positions,kit.axes,kit.radii)],poses=poses,faces=BODY_FACES)
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Sampled starting poses</title>
<script>__PLOTLY__</script><style>body{margin:0;background:#f8fafc;color:#172033;font:15px system-ui}
header{padding:18px 24px;background:white}h1{font-size:22px;margin:0 0 8px}select,input{margin:0 18px 0 8px}
#scene{height:76vh}#details{padding:12px 24px;white-space:pre-wrap}small{color:#64748b}</style></head>
<body><header><h1>Sampled starting poses</h1><label>Drum<select id="drum"></select></label>
<label>Hand<select id="hand"><option value="left">Left</option><option value="right">Right</option></select></label>
<label>Sample<input id="sample" type="range" min="0" max="7" value="0"><b id="number"></b></label>
<p><small>Drag to rotate; scroll to zoom. Arm joints and sticks use URDF forward kinematics.
Torso is schematic. Magenta = sampled contact; green = departure velocity × 80 ms; gold = torso forward.</small></p>
</header><div id="scene"></div><div id="details"></div><script>
const D=__DATA__,drum=document.getElementById('drum'),hand=document.getElementById('hand'),slider=document.getElementById('sample');
D.names.forEach((n,i)=>drum.add(new Option(n.replaceAll('_',' '),i)));
function line(points,color,width,name,opacity=1,markers=false){return {type:'scatter3d',x:points.map(p=>p[0]),
y:points.map(p=>p[1]),z:points.map(p=>p[2]),mode:markers?'lines+markers':'lines',line:{color,width},
marker:{size:4,color},name,opacity};}
function dot(p,color,size,name){return {type:'scatter3d',x:[p[0]],y:[p[1]],z:[p[2]],mode:'markers',marker:{color,size},name};}
function mesh(points,faces,color,opacity,name){let i=[],j=[],k=[];faces.forEach(f=>{for(let n=1;n<f.length-1;n++){i.push(f[0]);j.push(f[n]);k.push(f[n+1])}});
return {type:'mesh3d',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),i,j,k,color,opacity,name,showlegend:false};}
function draw(){let src=+drum.value;if(src===6)hand.value='right';hand.options[0].disabled=src===6;
let pool=D.poses.filter(p=>p.source===src&&p.hand===hand.value);slider.max=Math.max(0,pool.length-1);
let index=Math.min(+slider.value,pool.length-1);slider.value=Math.max(0,index);let p=pool[index];if(!p)return;
document.getElementById('number').textContent=p.sample;let traces=[];
D.drums.forEach((d,i)=>{let selected=i===src,color=selected?'#23875a':'#94a3b8';let pts=[d.center,...d.ring],faces=[];
for(let n=1;n<pts.length-1;n++)faces.push([0,n,n+1]);traces.push(mesh(pts,faces,color,selected?.45:.1,D.names[i]));
traces.push(line(d.ring,color,selected?4:2,D.names[i],selected?1:.3));});
traces.push(mesh(p.body,D.faces,'#64748b',.3,'Torso'));
for(let h of ['left','right']){let active=h===p.hand,c=active?(h==='left'?'#2563eb':'#d45c17'):'#9ca3af';
traces.push(line(p.arms[h],c,active?9:4,h+' arm',active?1:.4,true));
traces.push(line(p.sticks[h],c,active?8:3,h+' stick',active?1:.4));}
traces.push(line([p.waist,p.forward],'#ca8a04',7,'Torso forward'));
traces.push(dot(p.waist,'#ca8a04',6,'Waist'));traces.push(dot(p.contact,'#b91c8c',7,'Sampled contact'));
let end=p.contact.map((v,i)=>v+p.contact_velocity[i]*.08);traces.push(line([p.contact,end],'#15803d',6,'Departure velocity × 80 ms'));
traces.push({type:'cone',x:[end[0]],y:[end[1]],z:[end[2]],u:[p.contact_velocity[0]],v:[p.contact_velocity[1]],w:[p.contact_velocity[2]],
sizemode:'absolute',sizeref:.035,anchor:'tip',colorscale:[[0,'#15803d'],[1,'#15803d']],showscale:false,showlegend:false});
Plotly.react('scene',traces,{margin:{l:0,r:0,t:0,b:0},uirevision:'camera',scene:{aspectmode:'data',
xaxis:{title:'Forward x (m)',range:[-.25,1]},yaxis:{title:'Left y (m)',range:[-.95,.95]},zaxis:{title:'Height (m)',range:[.3,1.42]},
camera:{eye:{x:-1.6,y:1.4,z:1.1}}},legend:{x:0,y:1},paper_bgcolor:'#f8fafc'}, {responsive:true});
let deg=p.q.slice(0,3).map(x=>(x*180/Math.PI).toFixed(1));
document.getElementById('details').textContent=`${p.hand} · ${D.names[src]} · sample ${p.sample}\nWaist yaw / pitch / roll: ${deg.join('° / ')}°\nContact height above physical head: ${(p.contact_height_m*1000).toFixed(1)} mm · above inset strike plane: ${(p.contact_height_above_strike_plane_m*1000).toFixed(1)} mm\nOutward speed: ${p.outward_speed.toFixed(2)} m/s\nIK error: ${(p.ik_error_m*1000).toFixed(2)} mm · active-arm clearance: ${(p.clearance_m*1000).toFixed(2)} mm\n${p.origin==='saved'?'Saved departure state from the running calibration.':'Pending cohort reconstructed with the running calibration’s seed and sampler settings.'}\nIdle arm is displayed; its constraint violations are ignored in this calibration.`;}
drum.onchange=hand.onchange=slider.oninput=draw;draw();
</script></body></html>'''
    payload=json.dumps(data).replace('</','<\\/')
    (out/'starting_poses.html').write_text(html.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/transition_qp_unrestricted_20261005')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();run=args.run.resolve();out=run/'starting_pose_viz';out.mkdir(exist_ok=True)
    meta=json.loads((run/'config.json').read_text());kit=A.DrumKit.from_layout()
    torch.set_num_threads(2)
    frames=dict(ChainFK.FRAME_JOINTS)
    for hand in ('left','right'):
        frames.update({hand+'_'+s:hand+'_'+s+'_joint' for s in ARM_SUFFIXES})
    fk=ChainFK(frames=frames)
    cost=None;poses=[];saved_groups=0;preview_groups=0
    for hand in (0,1):
        for src,name in enumerate(kit.names):
            if hand==0 and name=='floor_tom':continue
            path=next(run.glob(f'w*/starts_h{hand}_d{src}.npz'),None)
            origin='saved'
            if path is not None:
                with np.load(path) as d: data={k:d[k].copy() for k in d.files}
                saved_groups+=1
            else:
                if cost is None:
                    ck=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
                    cfg=A.CostConfig(**ck['train_cfg']['cost']);cfg.n_sub=4;cfg.physics_mode='spline'
                    cfg.w_velocity=max(cfg.w_velocity,1.);cfg.w_joint_box=max(cfg.w_joint_box,1.)
                    cost=A.FusedCost(cfg,max_horizon=100,device=args.device,rnea=load_cached(device=args.device))
                    cost.calibration_active_hand_only=True
                data=sample_pending(cost,kit,meta,hand,src);origin='reconstructed_pending';preview_groups+=1
                path=out/f'preview_starts_h{hand}_d{src}.npz';save_npz(path,**data)
            q=torch.as_tensor(data['q']);v=torch.as_tensor(data['v'])
            if not len(q):continue
            computed=fk.forward(q)
            computed={key:(R.numpy(),p.numpy()+PELVIS_IN_ENV) for key,(R,p) in computed.items()}
            active=('left','right')[hand];sigma=default_contact_sigma(name)
            contact=tip_position(fk,q,active,sigma=sigma,frame='env').numpy()
            from amortized_opt.fk import stick_contact_offset
            _,J=contact_jacobian(fk,q,active,q.new_tensor(stick_contact_offset(active,sigma)).expand(len(q),-1))
            velocity=torch.einsum('bij,bj->bi',J,v).numpy()
            sticks={h:np.stack([tip_position(fk,q,h,sigma=-STICK_TIP_SIGMA,frame='env').numpy(),
                               tip_position(fk,q,h,frame='env').numpy()],axis=1) for h in ('left','right')}
            for i in range(len(q)):
                R,p=computed['torso'][0][i],computed['torso'][1][i]
                pose=dict(hand=active,source=src,drum=name,sample=i,origin=origin,source_file=str(path),
                    q=q[i].tolist(),v=v[i].tolist(),waist=p.tolist(),forward=(p+R@np.array([.22,0,.06])).tolist(),
                    body=(BODY_LOCAL@R.T+p).tolist(),contact=contact[i].tolist(),
                    contact_velocity=velocity[i].tolist(),
                    contact_height_m=float((contact[i]-kit.positions[src])@kit.axes[src]),
                    contact_height_above_strike_plane_m=float((contact[i]-kit.positions[src])@kit.axes[src]+STRIKE_SURFACE_OFFSET),
                    outward_speed=float(data['outward_speed_m_s'][i]),ik_error_m=float(data['ik_error_m'][i]),
                    clearance_m=float(data['clearance_m'][i]),
                    arms={h:np.stack([computed[h+'_'+s][1][i] for s in ARM_SUFFIXES]).tolist() for h in ('left','right')},
                    sticks={h:sticks[h][i].tolist() for h in ('left','right')})
                # Rendering coordinates must reproduce the exact contact used
                # by sampling, with the environment pelvis offset added once.
                np.testing.assert_allclose(pose['contact'],tip_position(fk,q[i:i+1],active,
                    sigma=sigma,frame='pelvis').numpy()[0]+PELVIS_IN_ENV,atol=1e-6)
                poses.append(pose)
    write_json(out/'poses.json',dict(coordinate_frame='environment',pelvis_in_env=PELVIS_IN_ENV.tolist(),
        torso='Schematic body; joint/stick geometry uses URDF FK',saved_source_groups=saved_groups,
        reconstructed_pending_groups=preview_groups,poses=poses))
    save_npz(out/'poses.npz',q=np.asarray([p['q'] for p in poses]),v=np.asarray([p['v'] for p in poses]),
        hand=np.asarray([0 if p['hand']=='left' else 1 for p in poses]),source=np.asarray([p['source'] for p in poses]),
        sample=np.asarray([p['sample'] for p in poses]))
    static_images(out,kit,poses);interactive(out,kit,poses)
    print(json.dumps(dict(out=str(out),poses=len(poses),saved_groups=saved_groups,
                          pending_reconstructed_groups=preview_groups)),flush=True)


if __name__=='__main__':main()
