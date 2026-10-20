#!/usr/bin/env python3
"""Inspect starting yaw/rebound speed versus observed transition-time witnesses."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

NAMES = ['snare', 'hihat', 'tom1', 'tom2', 'cymbal_left', 'cymbal_right', 'floor_tom']
FEATURES = [('yaw_deg', 'Starting waist yaw (degrees)'),
            ('outward_speed_m_s', 'Outward rebound speed (m/s)')]


def linear_fit(x, y):
    """Descriptive fit plus leave-one-out prediction error; guard tiny/constant samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    result = dict(n=len(y), slope=None, intercept=None, r2=None,
                  linear_loo_rmse_ms=None, constant_loo_rmse_ms=None,
                  quadratic_loo_rmse_ms=None)
    if len(y) < 3 or np.ptp(x) < 1e-8:
        return result
    z = x-x.mean()
    slope = np.dot(z, y-y.mean())/np.dot(z, z)
    prediction = y.mean()+slope*z
    result.update(slope=float(slope), intercept=float(y.mean()-slope*x.mean()))
    total = np.sum((y-y.mean())**2)
    if total > 1e-10:
        result['r2'] = float(1-np.sum((prediction-y)**2)/total)
    if len(y) < 4:
        return result
    result['constant_loo_rmse_ms'] = float(np.sqrt(np.mean(((y-y.mean())*len(y)/(len(y)-1))**2)))
    for degree, key in [(1, 'linear_loo_rmse_ms'), (2, 'quadratic_loo_rmse_ms')]:
        if degree == 2 and len(y) < 5:
            continue
        errors = []
        for index in range(len(y)):
            train = np.arange(len(y)) != index
            center, scale = x[train].mean(), x[train].std()
            if scale < 1e-8:
                break
            design = np.vander((x[train]-center)/scale, degree+1, increasing=True)
            if np.linalg.matrix_rank(design) < degree+1:
                break
            weights = np.linalg.lstsq(design, y[train], rcond=None)[0]
            predicted = sum(weights[j]*((x[index]-center)/scale)**j for j in range(degree+1))
            errors.append(float(predicted-y[index]))
        if len(errors) == len(y):
            result[key] = float(np.sqrt(np.mean(np.square(errors))))
    return result


def load_pairs(run):
    pairs = {}
    for hand in range(2):
        for src in range(7):
            for dst in range(7):
                if hand == 0 and (src == 6 or dst == 6):
                    continue
                pairs[hand, src, dst] = dict(hand=('left', 'right')[hand], source=NAMES[src],
                    destination=NAMES[dst], completed=False, records=[])
    for path in sorted(run.glob('w*/h*_source_*.npz')):
        with np.load(path) as d:
            hand, src = int(d['hand']), int(d['source'])
            with np.load(path.parent/f'starts_h{hand}_d{src}.npz') as starts:
                for index, dest in enumerate(d['destination']):
                    sample = int(d['sample'][index])
                    pair = pairs[hand, src, int(dest)]
                    pair['completed'] = bool(d['completed'])
                    record = dict(sample=sample, yaw_deg=float(np.rad2deg(d['q'][index, 0])),
                        outward_speed_m_s=float(starts['outward_speed_m_s'][sample]),
                        selected_for_qp=bool(d['selected_for_qp'][index]),
                        qp_attempts=int(d['qp_attempts'][index].sum()),
                        ik_error_mm=float(starts['ik_error_m'][sample]*1000))
                    for source, target in [('minimum_s', 'minimum_ms'),
                                           ('decoder_minimum_s', 'decoder_minimum_ms')]:
                        value = float(d[source][index]*1000)
                        record[target] = value if np.isfinite(value) else None
                    pair['records'].append(record)
    fits = []
    for pair in pairs.values():
        pair['fits'] = {}
        for measure in ['minimum_ms', 'decoder_minimum_ms']:
            records = [r for r in pair['records'] if r[measure] is not None]
            for feature, _ in FEATURES:
                fit = linear_fit([r[feature] for r in records], [r[measure] for r in records])
                pair['fits'][measure+'_'+feature] = fit
                fits.append(dict(hand=pair['hand'], source=pair['source'], destination=pair['destination'],
                    search_complete=pair['completed'], measure=measure, predictor=feature, **fit))
    return list(pairs.values()), fits


def atomic_text(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def export(out, pairs, fits):
    rows = [dict(hand=p['hand'], source=p['source'], destination=p['destination'],
                 search_complete=p['completed'], **r) for p in pairs for r in p['records']]
    for filename, data in [('per_start.csv', rows), ('pair_linear_fits.csv', fits)]:
        temporary = out/(filename+'.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        temporary.replace(out/filename)
    counts = {n: sum(sum(r['minimum_ms'] is not None for r in p['records']) >= n for p in pairs)
              for n in (1, 4, 5)}
    summary = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        allowed_pairs=len(pairs), saved_start_records=len(rows), pairs_with_at_least_n_successes=counts,
        limitations=['Eight starts per source; successful searches only in time fits.',
            'Only two selected starts per pair receive QP; remaining minima are decoder witnesses.',
            'Unknown minima are unresolved, not physical impossibility.',
            'Arm pose and velocity vary with yaw; associations are not isolated causal effects.',
            'All minima are finite-search witnesses at 20 ms resolution, not certified optimal durations.'],
        fits=fits)
    atomic_text(out/'analysis_summary.json', json.dumps(summary, indent=2, allow_nan=False)+'\n')


def panel(ax, pair, feature, measure='minimum_ms'):
    finite = [r for r in pair['records'] if r[measure] is not None]
    for selected, color, marker in [(False, '#2563eb', 'o'), (True, '#d45c17', 's')]:
        rows = [r for r in finite if r['selected_for_qp'] == selected]
        ax.scatter([r[feature] for r in rows], [r[measure] for r in rows],
                   c=color, marker=marker, s=40, zorder=3)
        for r in rows:
            ax.annotate(str(r['sample']), (r[feature], r[measure]), xytext=(4, 3),
                        textcoords='offset points', fontsize=8)
    unknown = [r for r in pair['records'] if r[measure] is None]
    ax.scatter([r[feature] for r in unknown], [.02]*len(unknown), transform=ax.get_xaxis_transform(),
               c='#9ca3af', marker='x', s=28, clip_on=False)
    fit = pair['fits'][measure+'_'+feature]
    if fit['slope'] is not None and fit['n'] >= 4:
        x = np.array([min(r[feature] for r in finite), max(r[feature] for r in finite)])
        ax.plot(x, fit['intercept']+fit['slope']*x, color='#475569', ls='--', lw=1)
    info = f'{len(finite)}/{len(pair["records"])} successful starts'
    if fit['r2'] is not None and fit['n'] >= 4:
        info += f' · R²={fit["r2"]:.2f}'
    ax.set_title(f'{pair["source"]} → {pair["destination"]}\n{info}'+
                 ('' if pair['completed'] else ' · searching'), fontsize=10)
    ax.set_xlabel(dict(FEATURES)[feature], fontsize=9)
    ax.set_ylabel('Best known duration (ms)', fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.tick_params(labelsize=8)
    if not finite:
        ax.text(.5, .5, 'No finite witness yet' if pair['records'] else 'Pending',
                ha='center', va='center', transform=ax.transAxes, color='#64748b')


def static_plots(out, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = [p for p in pairs if sum(r['minimum_ms'] is not None for r in p['records']) >= 4]
    candidates.sort(key=lambda p: (p['completed'], sum(r['minimum_ms'] is not None for r in p['records'])), reverse=True)
    overview = candidates[:6]
    if overview:
        fig, axes = plt.subplots(len(overview), 2, figsize=(13, 3*len(overview)), squeeze=False,
                                 layout='constrained')
        for row, pair in enumerate(overview):
            for col, (feature, _) in enumerate(FEATURES):
                panel(axes[row, col], pair, feature)
                axes[row, col].set_title(pair['hand'].title()+' · '+axes[row, col].get_title())
        fig.suptitle('Starting state versus observed transition duration\n'
            'Blue circles: no QP; orange squares: selected for QP. Labels: sample index.\n'
            'Grey × at baseline: unresolved, not a measured duration. Lines are descriptive only.', fontsize=12)
        temporary=out/'overview.tmp.png'; fig.savefig(temporary, dpi=160); temporary.replace(out/'overview.png')
        plt.close(fig)
    for hand in ['left', 'right']:
        for source in NAMES:
            group = [p for p in pairs if p['hand'] == hand and p['source'] == source]
            if not group:
                continue
            fig, axes = plt.subplots(len(group), 2, figsize=(13, 2.7*len(group)), squeeze=False,
                                     layout='constrained')
            for row, pair in enumerate(group):
                for col, (feature, _) in enumerate(FEATURES):
                    panel(axes[row, col], pair, feature)
            fig.suptitle(f'{hand.title()} hand departing {source} — all destinations\n'
                'Blue: no QP; orange: selected for QP; grey baseline ×: unresolved; labels: sample index.\n'
                'Successful searches only; 20 ms grid; varying arm configurations and unequal search effort.', fontsize=12)
            target=out/f'{hand}_{source}.png'; temporary=target.with_suffix('.tmp.png')
            fig.savefig(temporary, dpi=130); temporary.replace(target); plt.close(fig)


def interactive(out, pairs):
    from plotly.offline import get_plotlyjs
    html = r'''<!doctype html><html><head><meta charset="utf-8"><title>Transition starting-state relationships</title>
<script>__PLOTLY__</script><style>body{font:15px system-ui;color:#172033;background:#f8fafc;margin:0}
header,#details{padding:18px 24px}header{background:white}h1{font-size:23px;margin:0 0 14px}select{margin:0 16px 8px 6px}
#plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}.plot{height:430px}#details{white-space:pre-wrap}
small{color:#64748b}@media(max-width:1000px){#plots{grid-template-columns:1fr}}</style></head><body>
<header><h1>Starting yaw, rebound velocity and transition duration</h1>
<label>Hand<select id="hand"><option value="left">Left</option><option value="right">Right</option></select></label>
<label>From<select id="source"></select></label><label>To<select id="destination"></select></label>
<label>Time measure<select id="measure"><option value="minimum_ms">Best known (decoder + selected QP)</option>
<option value="decoder_minimum_ms">Decoder only (equal grid for all starts)</option></select></label>
<p>Blue circles: starts not selected for QP; orange squares: starts selected for QP. Labels are sample indices.
Grey × in the separate bottom strip: unresolved time; its vertical position has no duration meaning.</p>
<small>Only eight valid starts per hand/source; other joints and velocities vary too. QP explores only two starts per pair.
Dashed fits are descriptive, not causal. Unknown searches are not proof of infeasibility. Snapshot: __STAMP__.</small></header>
<div id="plots"><div id="yaw" class="plot"></div><div id="speed" class="plot"></div><div id="confounding" class="plot"></div></div>
<div id="details"></div><script>
const pairs=__DATA__,names=__NAMES__,el=id=>document.getElementById(id);
for(const id of ['source','destination'])names.forEach(n=>el(id).add(new Option(n.replaceAll('_',' '),n)));
el('source').value='tom1';el('destination').value='snare';
function hover(r){return `sample ${r.sample}<br>yaw ${r.yaw_deg.toFixed(1)}°<br>outward speed ${r.outward_speed_m_s.toFixed(3)} m/s<br>best ${r.minimum_ms===null?'unknown':r.minimum_ms.toFixed(0)+' ms'}<br>decoder ${r.decoder_minimum_ms===null?'unknown':r.decoder_minimum_ms.toFixed(0)+' ms'}<br>QP attempts ${r.qp_attempts}`;}
function trace(rows,x,y,selected){return {type:'scatter',mode:'markers+text',x:rows.map(r=>r[x]),y:rows.map(r=>r[y]),
text:rows.map(r=>String(r.sample)),textposition:'top right',customdata:rows.map(hover),hovertemplate:'%{customdata}<extra></extra>',
marker:{size:10,color:selected?'#d45c17':'#2563eb',symbol:selected?'square':'circle'},name:selected?'Selected for QP':'No QP selection'};}
function draw(){const hand=el('hand').value;
for(const id of ['source','destination']){el(id).options[6].disabled=hand==='left';if(hand==='left'&&el(id).value==='floor_tom')el(id).value='snare';}
const pair=pairs.find(p=>p.hand===hand&&p.source===el('source').value&&p.destination===el('destination').value),measure=el('measure').value;
const success=pair.records.filter(r=>r[measure]!==null),unknown=pair.records.filter(r=>r[measure]===null);let descriptions=[];
for(const [id,feature,label] of [['yaw','yaw_deg','Starting waist yaw (degrees)'],['speed','outward_speed_m_s','Outward rebound speed (m/s)']]){
let traces=[false,true].map(selected=>trace(success.filter(r=>r.selected_for_qp===selected),feature,measure,selected));
traces.push({type:'scatter',mode:'markers+text',x:unknown.map(r=>r[feature]),y:unknown.map(()=>.5),yaxis:'y2',
marker:{color:'#9ca3af',symbol:'x',size:10},text:unknown.map(r=>String(r.sample)),textposition:'top right',name:'Unresolved',
customdata:unknown.map(hover),hovertemplate:'%{customdata}<extra></extra>'});
const f=pair.fits[measure+'_'+feature];if(f.n>=4&&f.slope!==null){let x=[Math.min(...success.map(r=>r[feature])),Math.max(...success.map(r=>r[feature]))];
traces.push({type:'scatter',mode:'lines',x,y:x.map(v=>f.intercept+f.slope*v),line:{color:'#475569',dash:'dash'},name:'Linear fit'});}
Plotly.react(id,traces,{title:{text:label+'<br>'+success.length+'/'+pair.records.length+' successful starts',font:{size:15}},
margin:{l:60,r:12,b:75,t:65},xaxis:{title:label,anchor:'y2'},yaxis:{title:'Observed minimum duration (ms)',domain:[.15,1],rangemode:'tozero'},
yaxis2:{domain:[0,.08],range:[0,1],showticklabels:false,showgrid:false,zeroline:false},showlegend:false,paper_bgcolor:'#f8fafc'}, {responsive:true});
const fmt=v=>v===null?'not estimated':v.toFixed(1);descriptions.push(`${label}: n=${f.n}, R²=${f.r2===null?'not estimated':f.r2.toFixed(2)}\n  Leave-one-out RMSE (ms): constant ${fmt(f.constant_loo_rmse_ms)}; linear ${fmt(f.linear_loo_rmse_ms)}; quadratic ${fmt(f.quadratic_loo_rmse_ms)}`);}
Plotly.react('confounding',[false,true].map(selected=>trace(pair.records.filter(r=>r.selected_for_qp===selected),'yaw_deg','outward_speed_m_s',selected)),
{title:{text:'Yaw versus departure speed — all starts',font:{size:15}},xaxis:{title:'Starting waist yaw (degrees)'},yaxis:{title:'Outward rebound speed (m/s)'},
margin:{l:60,r:12,b:75,t:65},showlegend:false,paper_bgcolor:'#f8fafc'},{responsive:true});
el('details').textContent=`${hand} · ${pair.source} → ${pair.destination} · ${pair.records.length===0?'pending':pair.completed?'source search completed':'source search still running'}\n\n${descriptions.join('\n\n')}\n\nPrediction error is computed by holding out each start in turn. Lower is better. A linear fit must beat a constant prediction to be useful.\nWith fewer than four successes, prediction error is not estimated. A high training R² alone does not establish linearity.\nBounce velocity means actual contact velocity projected outward along the source drum normal.`;
}for(const id of ['hand','source','destination','measure'])el(id).onchange=draw;draw();
</script></body></html>'''
    html=html.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(pairs,allow_nan=False).replace('</','<\\/'))
    html=html.replace('__NAMES__',json.dumps(NAMES)).replace('__STAMP__',time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))
    atomic_text(out/'relationships.html',html)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=60)
    args=parser.parse_args();run=args.run.resolve();out=run/'starting_state_relationships';out.mkdir(exist_ok=True)
    previous=None
    while True:
        signature=[(str(p),p.stat().st_mtime_ns) for p in sorted(run.glob('w*/h*_source_*.npz'))]
        if signature!=previous:
            pairs,fits=load_pairs(run);export(out,pairs,fits);interactive(out,pairs);static_plots(out,pairs)
            previous=signature
            atomic_text(out/'watch.json',json.dumps(dict(pid=os.getpid(),status='watching' if args.watch else 'rendered')))
            print(f'Updated {out}',flush=True)
        if not args.watch:break
        if json.loads((run/'run.json').read_text()).get('status') in ('complete','failed'):
            atomic_text(out/'watch.json',json.dumps(dict(pid=os.getpid(),status='finished')));break
        time.sleep(max(1,args.poll_seconds))


if __name__=='__main__':
    main()
