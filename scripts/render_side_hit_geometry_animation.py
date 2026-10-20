#!/usr/bin/env python3
"""Animate controlled v4 side/top contact paths in a head cross-section."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle,Rectangle
import numpy as np


def main():
    out=ROOT/'runs/hihat_side_geometry_paths_v4_20261009.mp4'
    cases=[
        dict(title='TOP FACE',start=(-50,30),end=(-50,10),verdict='ENTRY: counts if speed/force pass',good=True),
        dict(title='UPPER SIDE, DOWNWARD',start=(19,1),end=(5,-3),verdict='ENTRY: eligible in top 5 mm',good=True),
        dict(title='UPPER SIDE, HORIZONTAL',start=(19,-3),end=(5,-3),verdict='NO ENTRY: incidence graze',good=False),
        dict(title='DEEP SIDE',start=(19,-20),end=(5,-20),verdict='NO ENTRY: below 5 mm band',good=False),
    ]
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for ax,case in zip(axes.flat,cases):
        ax.set_xlim(-80,36);ax.set_ylim(-35,38);ax.set_aspect('equal')
        ax.add_patch(Rectangle((-80,-100),80,100,facecolor='#c8ced4',edgecolor='none'))
        ax.plot([-80,0],[0,0],color='#34495e',lw=3)
        ax.plot([0,0],[-35,0],color='#34495e',lw=3)
        ax.plot([0,0],[-5,0],color='#20b078',lw=7,solid_capstyle='butt')
        ax.axhline(-5,color='#20b078',ls=':',lw=1)
        ax.text(-77,-30,'10 cm physical body\n(top 3.5 cm shown)',fontsize=9,color='#34495e')
        ax.text(2,-7,'5 mm\nside band',fontsize=9,color='#12845a')
        ax.set_title(case['title'],fontweight='bold',fontsize=13)
        ax.set_xlabel('Radial distance from edge (mm)');ax.set_ylabel('Height from top (mm)')
        ax.grid(alpha=.13)
        case['circle']=Circle(case['start'],10,facecolor='#f2ae4e',edgecolor='#8a4a12',lw=2,alpha=.9)
        ax.add_patch(case['circle'])
        case['path'],=ax.plot([],[],color='#d47725',lw=2,alpha=.8)
        case['label']=ax.text(.02,.05,'',transform=ax.transAxes,fontsize=11,
            fontweight='bold',color='#0d8754' if case['good'] else '#b93732',
            bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    fig.suptitle('v4 hi-hat side/top tracker paths — controlled force inputs, not Isaac trajectories',fontsize=15)
    writer=FFMpegWriter(fps=30,metadata={'title':'Controlled v4 hi-hat side-contact qualification'},bitrate=2500)
    with writer.saving(fig,str(out),dpi=135):
        for frame in range(90):
            progress=min(1.,frame/54)
            eased=.5-.5*np.cos(np.pi*progress)
            for case in cases:
                a=np.asarray(case['start'],float);b=np.asarray(case['end'],float)
                point=a+(b-a)*eased
                case['circle'].center=point
                case['path'].set_data([a[0],point[0]],[a[1],point[1]])
                case['label'].set_text(case['verdict'] if progress>=1 else '')
            writer.grab_frame()
    plt.close(fig)
    print(out)


if __name__=='__main__':main()
