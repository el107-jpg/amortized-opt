#!/usr/bin/env python3
"""Render the saved model-free snare-side descent with physical hit geometry."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args()
    run=args.run.resolve()
    summary=json.loads((run/'summary.json').read_text())
    data=np.load(run/'trajectories.npz')
    rows=summary['history']
    step=np.array([r['step'] for r in rows])
    final_step=int(step[-1])
    term_rho=np.array([r['terminal_rho_m'] for r in rows])
    term_z=np.array([r['terminal_axial_m'] for r in rows])
    first_rho=np.array([np.nan if r['first_target_contact_radial_m'] is None
                        else r['first_target_contact_radial_m'] for r in rows])
    first_z=np.array([np.nan if r['first_target_contact_axial_m'] is None
                      else r['first_target_contact_axial_m'] for r in rows])
    physical_radius=.19
    loss_radius=.18
    isaac_radius=.17
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)

    def geometry_background(ax,*,legend=False):
        ax.fill_between([.12,physical_radius],[-.16,-.16],[0,0],
            color='#939393',alpha=.36,label='physical drum' if legend else None)
        ax.fill_between([.12,loss_radius],[-.01,-.01],[0,0],
            color='#50b77c',alpha=.48,label='optimizer hit region' if legend else None)
        ax.axvline(physical_radius,color='#555555',ls=':',lw=1.3)
        ax.axvline(isaac_radius,color='#107139',ls='--',lw=1.4,
                   label='Isaac tip-center radius' if legend else None)
        ax.axhline(0,color='#555555',ls=':',lw=1)
        ax.set_xlim(.14,.23)
        ax.set_xlabel('tip distance from snare axis (m)')
        ax.set_ylabel('height above snare top (m)')

    ax=axes[0,0]
    geometry_background(ax,legend=True)
    ax.plot(term_rho,term_z,color='#2465a4',lw=1.6,label='terminal tip')
    ax.plot(first_rho,first_z,color='#de7a20',lw=1.2,ls='--',
            label='first target contact (4 ms grid)')
    marks=(0,1,5,10,20,40,60,70) if final_step<=70 else (0,1,10,70,final_step)
    for k in marks:
        if k>=len(rows):continue
        ax.scatter(term_rho[k],term_z[k],color='#2465a4',s=22,zorder=5)
        ax.annotate(str(k),(term_rho[k],term_z[k]),xytext=(2,4),
                    textcoords='offset points',fontsize=8)
        if np.isfinite(first_rho[k]):
            ax.scatter(first_rho[k],first_z[k],color='#de7a20',s=16,zorder=5)
    ax.set_ylim(min(-.04,float(np.nanmin(term_z))-.01),.028)
    ax.set_title('Terminal point and first contact during descent')
    ax.legend(fontsize=8,loc='upper left')

    ax=axes[0,1]
    geometry_background(ax)
    color={0:'#2663a1',10:'#e78a25',70:'#8f5ba4',final_step:'#294d37'}
    for k,c in color.items():
        key=f'rho_step_{k}'
        if key not in data:continue
        rho=data[key];z=data[f'axial_step_{k}']
        ax.plot(rho,z,label=f'step {k}',color=c,lw=1.6)
        ax.scatter(rho[-1],z[-1],color=c,s=25)
    plotted_heights=[data[f'axial_step_{k}'] for k in color if f'axial_step_{k}' in data]
    ax.set_ylim(min(-.045,float(min(np.min(z) for z in plotted_heights))-.01),.15)
    ax.set_title('Final approach of the 4 ms sampled tip path')
    ax.legend(fontsize=8)

    ax=axes[1,0]
    ax.plot(step,[r['cost'] for r in rows],label='total',color='#2360a0')
    ax.plot(step,[r['terminal_cost'] for r in rows],label='terminal',color='#ea8523')
    ax.plot(step,[r['collision_cost'] for r in rows],label='collision',color='#238350')
    ax.set(xlabel='projected gradient-descent step',ylabel='weighted loss',
           title='Current full FusedCost, no neural model')
    ax.legend()

    ax=axes[1,1]
    ax.plot(step,[1000*r['terminal_region_m'] for r in rows],color='#2360a0',
            label='terminal-region miss')
    ax.plot(step,[-1000*r['target_tip_min_sdf_m'] for r in rows],color='#ea8523',
            label='maximum target tip overlap')
    ax.axhline(0,color='#555555',ls=':',lw=1)
    ax.set(xlabel='projected gradient-descent step',ylabel='millimetres',
           title='Lower hit error, deeper physical overlap')
    ax.legend()

    fig.suptitle('Snare-side trajectory: direct waypoint gradient descent',fontsize=15)
    output=run/'side_descent_physical_geometry.png'
    fig.savefig(output,dpi=170)
    plt.close(fig)
    print(output)


if __name__=='__main__':main()
