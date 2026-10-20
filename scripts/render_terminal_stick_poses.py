#!/usr/bin/env python3
"""Draw the actual terminal stick capsule at selected descent checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon,Rectangle
import numpy as np
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS,STICK_RADIUS,TIP_RADIUS
from amortized_opt.cost import FusedCost
from amortized_opt.fk import ChainFK,stick_contact_offset
from amortized_opt.sdf import DrumSceneSpec


BASE=ROOT/'runs/snare_side_trajectory_descent_v5_20261013'
SOURCES={
    0:(BASE/'offset25mm_at70/trajectories.npz','initial'),
    10:(BASE/'offset25mm_step10/trajectories.npz','final'),
    70:(BASE/'offset25mm_at70/trajectories.npz','final'),
    250:(BASE/'offset25mm/trajectories.npz','final'),
}
OUTPUT=BASE/'offset25mm/terminal_stick_poses.png'


def draw_capsule(ax,a,b,r,color,alpha=.9,zorder=5):
    direction=b-a
    length=np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])/max(length,1e-12)
    polygon=np.stack((a+r*normal,b+r*normal,b-r*normal,a-r*normal))
    ax.add_patch(Polygon(polygon,closed=True,facecolor=color,edgecolor='none',
                         alpha=alpha,zorder=zorder))
    for point in (a,b):
        ax.add_patch(Circle(point,r,facecolor=color,edgecolor='none',
                            alpha=alpha,zorder=zorder))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,
        help='A descent run with saved q_step_* arrays; default uses archived side case')
    parser.add_argument('--steps',type=int,nargs='+',default=[0,10,70,250])
    cli=parser.parse_args()
    if len(cli.steps)!=4:raise ValueError('This diagram uses exactly four checkpoints')
    run=cli.run.resolve() if cli.run is not None else BASE/'offset25mm'
    if cli.run is None:
        sources=SOURCES
    else:
        saved=np.load(run/'trajectories.npz')
        sources={step:(run/'trajectories.npz',f'q_step_{step}') for step in cli.steps}
        for _,key in sources.values():
            if key not in saved:raise ValueError(f'Missing saved waypoint snapshot {key}')
    q=[]
    for step,(path,key) in sources.items():
        data=np.load(path)
        q.append(data[key][-1])
    q=torch.tensor(np.stack(q),dtype=torch.float32)
    fk=ChainFK(joint_names=G1_UPPER_BODY_JOINTS,frames=dict(FusedCost.FRAMES),
               device='cpu',dtype=q.dtype)
    rotation,origin=fk.frame(q,'right_wrist')
    local=torch.tensor(np.stack([stick_contact_offset('right',sigma)
                                 for sigma in (-.2,0.,.2)]),dtype=q.dtype)
    world=(origin[:,None]+torch.einsum('bij,kj->bki',rotation,local)).numpy()
    scene=DrumSceneSpec.from_layout(frame='pelvis')
    drum=scene.names.index('snare')
    head=np.asarray(scene.head_centers[drum]);axis=np.asarray(scene.axes[drum])
    radius=float(scene.radii[drum]);height=float(scene.heights[drum])
    initial=world[0,2]-head
    radial=initial-initial.dot(axis)*axis
    radial/=np.linalg.norm(radial)
    tangent=np.cross(axis,radial);tangent/=np.linalg.norm(tangent)
    # Coordinates in the actual tilted drum's frame: n is vertical; radial
    # points through the initial terminal tip. Both orthogonal projections are
    # required because the stick is substantially out of the side-view plane.
    relative=world-head[None,None]
    side=np.stack((relative@radial,relative@axis),axis=-1)
    top=np.stack((relative@radial,relative@tangent),axis=-1)
    history=json.loads((run/'summary.json').read_text())['history']

    fig,grid=plt.subplots(2,4,figsize=(17,9))
    fig.subplots_adjust(left=.055,right=.985,bottom=.13,top=.90,
                        hspace=.42,wspace=.30)
    for column,step in enumerate(sources):
        ax=grid[0,column]
        ax.add_patch(Rectangle((-radius,-height),2*radius,height,
            facecolor='#d1d6d9',edgecolor='#56616a',linewidth=1.2,zorder=1))
        ax.plot([-radius,radius],[0,0],color='#384750',lw=2,zorder=2)
        ax.axhline(TIP_RADIUS,color='#3a8a65',lw=1,ls=':',alpha=.8,zorder=1)
        butt,mount,tip=side[column]
        draw_capsule(ax,butt,mount,STICK_RADIUS,'#6f7f90')
        draw_capsule(ax,mount,tip,STICK_RADIUS,'#e69832')
        ax.add_patch(Circle(tip,TIP_RADIUS,facecolor='#d45c54',
            edgecolor='#742824',linewidth=1.0,zorder=7))
        ax.scatter(*mount,c='#273540',s=10,zorder=8)
        ax.set(xlim=(-.27,.34),ylim=(-.21,.25),aspect='equal',
               xlabel='in-plane radius (m)',ylabel='height along head normal (m)',
               title=f'Step {step}: side cross-section')
        ax.grid(alpha=.15)

        ax=grid[1,column]
        ax.add_patch(Circle((0,0),radius,facecolor='#d1d6d9',
            edgecolor='#56616a',linewidth=1.2,zorder=1))
        ax.add_patch(Circle((0,0),radius-.01,facecolor='#a8d8b7',
            edgecolor='none',alpha=.55,zorder=2))
        ax.add_patch(Circle((0,0),radius-.02,facecolor='none',
            edgecolor='#257950',linewidth=1.4,linestyle='--',zorder=3))
        butt,mount,tip=top[column]
        draw_capsule(ax,butt,mount,STICK_RADIUS,'#6f7f90')
        draw_capsule(ax,mount,tip,STICK_RADIUS,'#e69832')
        ax.add_patch(Circle(tip,TIP_RADIUS,facecolor='#d45c54',
            edgecolor='#742824',linewidth=1.0,zorder=7))
        ax.scatter(*mount,c='#273540',s=10,zorder=8)
        state=history[step]
        ax.set(xlim=(-.26,.34),ylim=(-.45,.25),aspect='equal',
               xlabel='radial-plane coordinate (m)',ylabel='perpendicular coordinate (m)',
               title=(f'Top view: tip radius {state["terminal_rho_m"]*100:.1f} cm, '
                      f'height {state["terminal_axial_m"]*100:.1f} cm'))
        ax.grid(alpha=.15)
        sdf=state['target_tip_min_sdf_m']
        overlap=(f'max analytic tip overlap: {-sdf*1000:.1f} mm' if sdf<0
                 else f'min analytic tip clearance: {sdf*1000:.1f} mm')
        ax.text(.02,.02,overlap,
                transform=ax.transAxes,fontsize=9,
                bbox=dict(boxstyle='round,pad=.25',facecolor='white',alpha=.85,edgecolor='none'))

    fig.suptitle('Right-hand terminal stick relative to the snare',fontsize=17,y=.975)
    fig.text(.5,.045,'Gray: proximal shaft · Orange: distal 20 cm playing shaft · '
             'Red: 10 mm bead · Dashed green: Isaac tip-center hit radius (17 cm) · '
             'Green fill: optimizer hit disk (18 cm)',ha='center',fontsize=10)
    output=run/'terminal_stick_poses.png'
    fig.savefig(output,dpi=180)
    plt.close(fig)
    print(output)


if __name__=='__main__':main()
