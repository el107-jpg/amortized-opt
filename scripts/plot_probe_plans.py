#!/usr/bin/env python3
"""Replay training probes and draw every generated/committed plan with the plant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amortized_opt as A
from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.ik import tip_position
from amortized_opt.lead import resolve_lead
from amortized_opt.probe import closed_loop_probe
from amortized_opt.tspline import TrajectorySplineBuilder


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def plan_geometry(rows, trajectories, scorer, substeps):
    H = trajectories.shape[1]
    tensor = lambda x: torch.as_tensor(x, device=scorer.device, dtype=scorer.dtype)
    sp = TrajectorySplineBuilder(max_segments=H, dt=CONTROL_DT, device=scorer.device,
                                dtype=scorer.dtype).from_masks(
        tensor(np.stack([r['q0'] for r in rows])), tensor(trajectories),
        tensor(np.stack([r['v0'] for r in rows])),
        tensor(np.stack([r['elem_mask'] for r in rows])))
    q = sp.subsample(substeps).q
    result = []
    tips = {h: tip_position(scorer.fk, q.reshape(-1, q.shape[-1]), h, frame='env')
            .reshape(q.shape[0], q.shape[1], 3).cpu().numpy()
            for h in ('left', 'right')}
    for i, row in enumerate(rows):
        obj = {'step': row['step'], 'time_s': row['time_s'], 'hands': {}}
        for h in ('left', 'right'):
            cols = [j for j, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith(h+'_')]
            K = int(np.max(row['elem_mask'][:, cols].sum(0)))
            if not K:
                continue
            kh = int(row['hit'][h])
            path = tips[h][i, :K*substeps+1]
            obj['hands'][h] = {'path': np.round(path, 6).tolist(),
                'endpoint': np.round(tips[h][i, kh*substeps], 6).tolist()
                    if 1 <= kh <= K else None,
                'hit_step': row['step']+kh if kh >= 1 else None,
                'drum': row['target_drum'][h]}
        result.append(obj)
    return result


def replay(run, device, out, substeps, *, perfect_tracking=False, checkpoint_override=None):
    args = json.loads((run/'args.json').read_text())
    events = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    reference = [r for r in events if r.get('event') == 'probe'][-1]
    checkpoint = run/f"step_{reference['step']:06d}.pt"
    if not checkpoint.exists():
        checkpoint = run/'last.pt'
    if checkpoint_override is not None:
        checkpoint = Path(checkpoint_override)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    torch.set_float32_matmul_precision('high' if ck['train_cfg'].get('tf32', True) else 'highest')
    model_cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(model_cfg, device)
    model.load_state_dict(ck['model']); model.eval()
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir('eval'), kit)[:args['probe_clips']]
    scorer = A.TrajectoryScorer(device='cpu', track_torque=False,
                              thresholds=A.AcceptanceThresholds(hit_margin_m=0.0))
    result = closed_loop_probe(model, kit=kit, scorer=scorer, clips=clips, device=device,
        num_drums=model_cfg.num_drums, num_future=model_cfg.num_future,
        horizon=args['buckets'][-1], steps=args['probe_steps'], seed=0,
        amp_dtype=None if args['no_amp'] else torch.bfloat16,
        lead_s=args['probe_lead'], freeze_wp=args['probe_freeze'],
        cmd_lead_s=resolve_lead(args['probe_cmd_lead'] if args['probe_cmd_lead'] is not None else args['cmd_lead']),
        waist_upright_ease_wp=args['waist_upright_ease_wp'],
        saturate_effort=args['plant_saturate_effort'], bounce=not args['no_probe_bounce'],
        record_drift=False, record_rollout=True,
        fresh_line_init=args['terminal_init_all_rows'],
        waist_pr_ema=args['waist_pr_ema'], waist_pr_ema_alpha=args['waist_pr_ema_alpha'],
        waist_pr_ema_leak=args['waist_pr_ema_leak'], perfect_tracking=perfect_tracking)
    episodes = []; arrays = {}
    for ep, rollout in enumerate(result['_rollouts']):
        res = rollout['result']; rows = rollout['committed_plans']
        raw = np.stack([r['traj'] for r in rollout['plans']])
        committed = np.stack([r['traj'] for r in rows])
        assert len(rows) == len(raw) == len(res.cmd_hist)
        np.testing.assert_allclose(committed[:, 0], res.cmd_hist, rtol=0, atol=0)
        if perfect_tracking:
            executed = np.array(res.q_hist[1:], copy=True)
            for (step, _hand, _drum), (q_fire, _qd_fire) in zip(res.hit_steps, res.hit_states):
                executed[int(step)-1] = q_fire
            np.testing.assert_allclose(executed, res.cmd_hist, rtol=0, atol=1e-12)
        actual = {h: np.round(tip_position(scorer.fk, scorer._t(res.q_hist), h, frame='env')
                             .cpu().numpy(), 6).tolist() for h in ('left', 'right')}
        episodes.append({'clip': res.clip, 'actual': actual,
            'generated': plan_geometry(rows, raw, scorer, substeps),
            'committed': plan_geometry(rows, committed, scorer, substeps),
            'hit_steps': safe(res.hit_steps), 'diverged': res.diverged,
            'diverged_step': res.diverged_step,
            'hits': [h for h in result['_hits']], 'ticks': len(res.cmd_hist)})
        for key, value in dict(q=res.q_hist, qd=res.qd_hist, command=res.cmd_hist,
                               generated=raw, committed=committed,
                               elem_mask=np.stack([r['elem_mask'] for r in rows]),
                               q0=np.stack([r['q0'] for r in rows]),
                               v0=np.stack([r['v0'] for r in rows])).items():
            arrays[f'episode_{ep}_{key}'] = value
    np.savez_compressed(out/(run.name+('_perfect' if perfect_tracking else '')+'.npz'), **arrays)
    summary = {k: v for k, v in result.items() if not k.startswith('_')}
    data = {'run': run.name, 'checkpoint': str(checkpoint), 'checkpoint_step': ck['step'],
        'coordinate_frame': 'env',
        'tracking': 'perfect' if perfect_tracking else 'pd',
        'command_lead': args['probe_cmd_lead'] if args['probe_cmd_lead'] is not None else args['cmd_lead'],
        'initialization': 'fresh_line_every_tick' if args['terminal_init_all_rows'] else 'shifted_mpc',
        'episodes': episodes, 'summary': summary,
        'reference_probe': {k: reference.get(k) for k in ('step','hits','in_disk','diverged','ticks')}}
    if perfect_tracking:
        data['matches_logged_probe'] = None
    elif int(ck['step']) == int(reference['step']):
        for key in ('hits', 'in_disk', 'diverged', 'ticks'):
            assert result[key] == reference[key], (run.name, key, result[key], reference[key])
        data['matches_logged_probe'] = True
    else:
        data['matches_logged_probe'] = False
    print(json.dumps({'run': run.name, 'step': ck['step'], 'ticks': result['ticks'],
                      'tracking': data['tracking'],
                      'attempts': result['hits'], 'on_time_hits': result['in_disk'],
                      'plans': sum(len(e['generated']) for e in episodes),
                      'matches_logged_probe': data['matches_logged_probe']}), flush=True)
    del model
    return safe(data), kit


def static_plot(data, kit, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(len(data), 2, figsize=(13, 4.2*len(data)), squeeze=False)
    colors = {'left': '#2563eb', 'right': '#d45c17'}
    for row_i, run in enumerate(data):
        ep = run['episodes'][0]
        for col_i, dim in enumerate((1, 2)):
            ax = axes[row_i,col_i]
            for center, radius, name in zip(kit.positions, kit.radii, kit.names):
                if dim == 1:
                    ax.add_patch(Circle(center[:2], radius, fill=False, color='#555', lw=1))
                else:
                    ax.plot([center[0]-radius, center[0]+radius], [center[2]]*2, color='#555', lw=1)
                ax.text(center[0], center[dim], name, fontsize=6, color='#444', ha='center')
            for h, color in colors.items():
                ends = []
                for plan in ep['generated']:
                    if h not in plan['hands']: continue
                    hand = plan['hands'][h]; p = np.asarray(hand['path'])
                    ax.plot(p[:,0], p[:,dim], color=color, alpha=.14, lw=.65)
                    if hand['endpoint'] is not None: ends.append(hand['endpoint'])
                if ends:
                    ends = np.asarray(ends);ax.scatter(ends[:,0],ends[:,dim],color=color,s=5,alpha=.35)
                p = np.asarray(ep['actual'][h]);ax.plot(p[:,0],p[:,dim],color=color,lw=2.7)
                ax.scatter(p[0,0],p[0,dim],color=color,marker='o',s=24,zorder=5)
                ax.scatter(p[-1,0],p[-1,dim],color=color,marker='X',s=65,zorder=5)
            ax.set_aspect('equal', adjustable='datalim');ax.grid(alpha=.2)
            if dim == 2:
                ax.axhline(0, color='#777', ls=':', lw=.8)
            ax.set(xlabel='x (m, environment)', ylabel=f"{'y' if dim==1 else 'z'} (m, environment)")
            label=run['run'].replace('phase2_','').replace('_fromp1_20261001','')
            ax.set_title(f"{label} · step {run['checkpoint_step']} · {len(ep['generated'])} plans\n"
                f"{run['summary']['in_disk']}/{run['summary']['hits']} on-time · "
                f"{'diverged' if ep['diverged'] else 'completed'} · {'top' if dim==1 else 'side'} view", fontsize=10)
        axes[row_i,0].legend(handles=[Line2D([0],[0],color=colors[h],lw=2.7,label=h+' followed') for h in colors]
            +[Line2D([0],[0],color='#666',lw=.65,alpha=.4,label='every generated plan'),
              Line2D([0],[0],color='#666',marker='.',ls='',label='tip at strike waypoint')],fontsize=7)
    fig.suptitle('Closed-loop training probe: every generated plan and the executed stick-tip path', fontsize=13)
    fig.tight_layout();fig.savefig(out/'all_plans.png',dpi=150);fig.savefig(out/'all_plans.pdf');plt.close(fig)


def attach_hit_requests(data, kit):
    """Match recorded plan goals to the original RCC strike requests."""
    from amortized_opt.rhythm import strike_velocity
    clips = {clip.name: clip for clip in A.load_rcc_pool(A.default_rcc_dir('eval'), kit)}
    for run in data:
        args_path = Path(run['checkpoint']).parent/'args.json'
        args = json.loads(args_path.read_text())
        lead = float(args['probe_lead'])
        episodes = list(run['episodes'])
        for variant in run.get('tracking_variants', {}).values():
            episodes.extend(e for e in variant['episodes'] if not any(e is old for old in episodes))
        for episode in episodes:
            clip = clips[episode['clip']]
            goals = {}
            for mode in ('generated', 'committed'):
                for plan in episode[mode]:
                    for hand, path in plan['hands'].items():
                        if path['hit_step'] is None or path['drum'] < 0:
                            continue
                        candidates = [(i, hit) for i, hit in enumerate(clip.hits)
                                      if hit.hand == hand and hit.drum == path['drum']]
                        time = path['hit_step']*CONTROL_DT + lead
                        index, hit = min(candidates, key=lambda x: abs(x[1].time-time))
                        if abs(hit.time-time) > CONTROL_DT+1e-6:
                            raise ValueError(f'Plan goal does not match scheduled strike: {time}, {hit}')
                        path['request_id'] = index
                        if index not in goals:
                            velocity = strike_velocity(kit, hit.drum, hit.velocity)
                            goals[index] = {'request_id': index, 'hand': hand, 'drum': hit.drum,
                                'drum_name': hit.drum_name, 'scheduled_time_s': hit.time,
                                'loudness': hit.velocity, 'target_speed_m_s': float(np.linalg.norm(velocity)),
                                'velocity_env_m_s': velocity.tolist(), 'hit_step': path['hit_step'],
                                'start_step': plan['step']}
                        goals[index]['start_step'] = min(goals[index]['start_step'], plan['step'])
            requests = sorted(goals.values(), key=lambda x: (x['scheduled_time_s'], x['request_id']))
            for request in requests:
                scored = next((h for h in episode['hits'] if h['hand']==request['hand']
                               and h['k_fire']==request['hit_step']), None)
                request['reached'] = scored is not None
                request['on_time_hit'] = bool(scored and scored['meas'].get('0',{}).get('in_disk'))
                request['end_step'] = min(request['hit_step'], episode['ticks'])
            episode['hit_requests'] = requests


def interactive_plot(data, kit, out):
    from plotly.offline import get_plotlyjs
    attach_hit_requests(data, kit)
    payload = json.dumps({'runs': data, 'kit': {'names':kit.names,'positions':kit.positions.tolist(),
                         'axes':kit.axes.tolist(),'radii':kit.radii.tolist()}}, allow_nan=False)
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Probe plans</title>
<style>body{font:15px system-ui;margin:20px;background:#f5f7fa;color:#202b3b}h1{font-size:23px}label{margin-right:16px}select{padding:6px}#plot{height:72vh}#info{margin:12px 0}input[type=range]{width:350px}.note{color:#596574;font-size:13px}</style>
<script>__PLOTLY__</script></head><body><h1>MPC plans for one strike</h1>
<p>Thick line: followed stick-tip path for this hit. Thin lines: plans aimed at this hit. Dots: planned strike endpoints.</p>
<label>Run <select id="run"></select></label><label>Plans <select id="mode"><option value="generated">Network output</option><option value="committed">After freeze and command lead</option></select></label>
<label>Tracking <select id="tracking"><option value="perfect">Perfect</option><option value="pd">PD controller</option></select></label>
<label><input id="all" type="checkbox" checked>Show every plan for this hit</label>
<p><label>Hit <input id="hit" type="range" min="0" value="0"></label><span id="hitLabel"></span></p>
<div id="request" style="padding:10px 14px;background:white;border-left:4px solid #27804c;border-radius:4px"></div>
<p><label>Highlight tick <input id="tick" type="range" min="0" value="0"></label><span id="tickLabel"></span></p>
<div id="info"></div><div id="plot"></div>
<p class="note">The hit slider selects one hand’s scheduled strike. Perfect tracking executes each controller command exactly; freeze, command lead, and the synthetic post-hit bounce remain enabled. Plans are regenerated from the resulting states. Loudness is normalized to 0–100%; target speed is into the drum along its normal. All geometry uses environment coordinates; the pelvis is at z = 0.80 m. Strike dots show the bead tip; shaft contact on cymbals/hi-hat can differ.</p>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let first=true,activeRequestId=null;
DATA.runs.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.run.replace('phase2_','').replace('_fromp1_20261001','')+' · step '+r.checkpoint_step;$('run').append(o)});$('run').value=DATA.runs.length-1;if(!DATA.runs.at(-1).tracking_variants?.perfect)$('tracking').value='pd';
function line(points,name,color,width,opacity=1,extra={}){return {type:'scatter3d',mode:'lines',x:points.map(p=>p?p[0]:null),y:points.map(p=>p?p[1]:null),z:points.map(p=>p?p[2]:null),name,line:{color,width},opacity,...extra}}
function dots(points,name,color,extra={}){return {type:'scatter3d',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),name,marker:{color,size:3},...extra}}
function draw(reset=false){const r=DATA.runs[+$('run').value];if($('tracking').value==='perfect'&&!r.tracking_variants?.perfect)$('tracking').value='pd';const variant=r.tracking_variants?.[$('tracking').value]||r,e=variant.episodes[0],requests=e.hit_requests;$('hit').max=requests.length-1;$('hit').value=Math.min(+$('hit').value,requests.length-1);const req=requests[+$('hit').value];activeRequestId=req.request_id;const plans=e[$('mode').value].filter(p=>p.hands[req.hand]?.request_id===req.request_id);$('tick').max=plans.length-1;let idx=Math.min(+$('tick').value,plans.length-1);$('tick').value=idx;const selected=plans[idx];$('tickLabel').textContent='tick '+selected.step+' · '+selected.time_s.toFixed(2)+' s · '+Math.max(0,req.scheduled_time_s-selected.time_s).toFixed(2)+' s to hit';$('hitLabel').textContent=(+$('hit').value+1)+' / '+requests.length;$('request').textContent=req.hand.toUpperCase()+' → '+req.drum_name+' · scheduled '+req.scheduled_time_s.toFixed(2)+' s · loudness '+(req.loudness*100).toFixed(1)+'% · target inward speed '+req.target_speed_m_s.toFixed(2)+' m/s';
const traces=[line([[-1,-.8,0],[1,-.8,0],[1,.8,0],[-1,.8,0],[-1,-.8,0]],'environment z = 0','#999',2),dots([[0,0,.8]],'pelvis origin','#333',{marker:{color:'#333',size:6,symbol:'diamond'}})];const kit=DATA.kit;kit.names.forEach((n,i)=>{const c=kit.positions[i],a=kit.axes[i],v=Math.abs(a[2])<.9?[0,0,1]:[1,0,0];let u=[a[1]*v[2]-a[2]*v[1],a[2]*v[0]-a[0]*v[2],a[0]*v[1]-a[1]*v[0]];const len=Math.hypot(...u);u=u.map(x=>x/len);const w=[a[1]*u[2]-a[2]*u[1],a[2]*u[0]-a[0]*u[2],a[0]*u[1]-a[1]*u[0]];const ring=Array.from({length:65},(_,j)=>c.map((x,k)=>x+kit.radii[i]*(Math.cos(j*Math.PI/32)*u[k]+Math.sin(j*Math.PI/32)*w[k])));traces.push(line(ring,n,'#555',3,1,{showlegend:false}));traces.push({...dots([c],n,'#555'),mode:'text',text:[n],textfont:{size:11},showlegend:false})});
traces.push(dots([kit.positions[req.drum]],'requested drum center','#27804c',{marker:{color:'#27804c',size:7,symbol:'diamond'}}));
for(const h of [req.hand]){const color=h==='left'?'#2563eb':'#d45c17';let paths=[],ends=[],hover=[];
if($('all').checked){for(const p of plans){const a=p.hands[h];if(!a)continue;paths.push(...a.path,null);if(a.endpoint){ends.push(a.endpoint);hover.push('tick '+p.step+' → '+(kit.names[a.drum]||'idle')+' at tick '+a.hit_step)}}traces.push(line(paths,h+' every plan',color,2,.23));traces.push(dots(ends,h+' strike tips',color,{text:hover,hovertemplate:'%{text}<extra></extra>',opacity:.5}))}
const followed=e.actual[h].slice(req.start_step,req.end_step+1);traces.push(line(followed,h+' followed for this hit',color,7));traces.push(dots([followed[0]],h+' start',color));traces.push(dots([followed.at(-1)],req.reached?'measured at strike':'rollout ended before hit',color,{marker:{color,size:7,symbol:'x'}}));
if(selected.hands[h]){const a=selected.hands[h];traces.push(line(a.path,h+' selected plan',h==='left'?'#00bfd0':'#e632b3',8));if(a.endpoint)traces.push(dots([a.endpoint],h+' selected strike tip',color,{marker:{color,size:7,symbol:'diamond'}}))}
traces.push(dots([e.actual[h][Math.min(selected.step,e.actual[h].length-1)]],h+' measured at selected tick',color,{marker:{color,size:6}}))}
$('info').textContent=($('tracking').value==='perfect'?'Perfect command tracking':'PD tracking')+' · '+plans.length+' plans for this hit · window '+(req.start_step*.02).toFixed(2)+'–'+(req.end_step*.02).toFixed(2)+' s · '+(req.reached?(req.on_time_hit?'on-time hit':'miss at scheduled tick'):'rollout ended before scheduled hit')+' · '+r.initialization;
Plotly.react('plot',traces,{margin:{l:0,r:0,t:10,b:0},paper_bgcolor:'#f5f7fa',legend:{x:0,y:1},scene:{aspectmode:'data',xaxis:{title:{text:'x (m, environment)'}},yaxis:{title:{text:'y (m, environment)'}},zaxis:{title:{text:'z (m, environment)'}}},uirevision:reset?String($('run').value):'view-'+$('run').value},{responsive:true,displaylogo:false});first=false}
for(const id of ['mode','all','tick'])$(id).addEventListener('input',()=>draw());$('hit').addEventListener('input',()=>{$('tick').value=0;draw()});$('tracking').addEventListener('change',()=>{const r=DATA.runs[+$('run').value],v=r.tracking_variants?.[$('tracking').value]||r;$('hit').max=v.episodes[0].hit_requests.length-1;$('hit').value=Math.max(0,v.episodes[0].hit_requests.findIndex(h=>h.request_id===activeRequestId));$('tick').value=0;draw()});$('run').addEventListener('change',()=>{$('hit').value=0;$('tick').value=0;draw(true)});draw(true);
</script></body></html>'''
    (out/'all_plans.html').write_text(html.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, action='append', required=True)
    ap.add_argument('--device', default='cuda:0');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plan-substeps',type=int,default=2)
    ap.add_argument('--compare-perfect-tracking', action='store_true')
    args = ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=[]
    for run in args.run:
        entry,kit=replay(run.resolve(),torch.device(args.device),args.out,args.plan_substeps)
        if args.compare_perfect_tracking:
            perfect,_=replay(run.resolve(),torch.device(args.device),args.out,args.plan_substeps,
                             perfect_tracking=True,checkpoint_override=entry['checkpoint'])
            entry['tracking_variants']={'pd':{'episodes':entry['episodes'],'summary':entry['summary']},
                                        'perfect':{'episodes':perfect['episodes'],'summary':perfect['summary']}}
        data.append(entry)
    attach_hit_requests(data,kit)
    (args.out/'trajectories.json').write_text(json.dumps(data,allow_nan=False))
    summaries=[{k:v for k,v in run.items() if k!='episodes'} for run in data]
    (args.out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
    static_plot(data,kit,args.out);interactive_plot(data,kit,args.out)
    print(f"Wrote {args.out}/all_plans.html and all_plans.png",flush=True)


if __name__ == '__main__':
    main()
