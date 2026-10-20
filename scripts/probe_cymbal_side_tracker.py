#!/usr/bin/env python3
"""Exercise the real contact tracker with controlled hi-hat side/top paths."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from isaac_rl.cymbal_contact_scoring import (cymbal_entry_geometry,
    make_side_sliver_tracker)


def upstream_tracker_classes():
    """Load pure-Torch upstream tracker without importing Isaac's app package."""
    folder=Path('/data2/eric/g1-drummer-lab-rythmic/source/g1_drummer/g1_drummer/tasks/direct/g1_drum')
    package=ModuleType('controlled_drum_tracker');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    detector=importlib.import_module(package.__name__+'.hit_detector')
    contact=importlib.import_module(package.__name__+'.contact_hit_detector')
    return detector.GeometricHitDetector,contact.ContactStrikeTracker


def main():
    layout=yaml.safe_load((ROOT/'assets/drum_kit_layout.yaml').read_text())['drums']['hihat']
    radius=float(layout['radius']);height=float(layout['height'])
    normal=torch.as_tensor(layout['normal'],dtype=torch.float32)
    normal=normal/normal.norm()
    tangent=torch.tensor([1.,0.,0.])-normal[0]*normal
    tangent=tangent/tangent.norm()
    Detector,BaseTracker=upstream_tracker_classes()
    kit=SimpleNamespace(radii=[radius],heights=[height],edge_hit=[True])
    Tracker=make_side_sliver_tracker(BaseTracker,kit)
    def position(radial,axial):return radial*tangent+axial*normal
    def run(name,*,previous_radial,current_radial,previous_axial,current_axial,
            tangential_speed,normal_speed,force_N=10.):
        detector=Detector(num_envs=1,num_drums=1,device='cpu',
            hit_radius=torch.tensor([radius-.02]),head_z_tol_above=.017,
            band_z_tol_above=.024,strike_rearm_travel_m=.03,
            head_z_tol_below=.02,head_normal=normal[None],
            num_probes=1,shaft_drums=torch.tensor([True]),stray_min_speed=0.)
        tracker=Tracker(num_envs=1,num_drums=1,device='cpu',detector=detector,
            force_threshold=1.,require_band=True,require_rearm=True,
            speed_window_m=.12,max_tangential_ratio=1.5)
        tracker.reset(torch.tensor([0]))
        centre=torch.zeros(1,3)
        def sample(radial,axial,velocity,force,time):
            point=position(radial,axial).reshape(1,1,1,3).expand(1,2,1,3)
            speed=velocity.reshape(1,1,1,3).expand_as(point)
            forces=torch.zeros(1,2,1,3);forces[0,0,0]=force*tangent
            tracker.sample(point,speed,centre,torch.tensor([time]),forces)
        velocity=-tangential_speed*tangent-normal_speed*normal
        sample(previous_radial,.08,torch.zeros(3),0.,0.)  # Arm above the re-arm threshold.
        sample(previous_radial,previous_axial,velocity,0.,.004)
        sample(current_radial,current_axial,velocity,force_N,.008)
        previous=position(previous_radial,previous_axial).reshape(1,1,1,3).expand(1,2,1,3)
        current=position(current_radial,current_axial).reshape(1,1,1,3).expand(1,2,1,3)
        top,side,inside,through=cymbal_entry_geometry(current,previous,centre,
            normal[None],torch.tensor([radius]),torch.tensor([height]),
            torch.tensor([True]),torch.tensor([True]),torch.tensor([[True]]))
        verdict=dict(path=name,depth_below_top_mm=round(-current_axial*1000,2),
            tangential_speed_m_s=tangential_speed,normal_speed_m_s=normal_speed,
            contact_force_N=force_N,top_entry=bool(top[0,0,0,0]),
            upper_side=bool(side[0,0,0,0]),inside_body=bool(inside[0,0,0,0]),
            tracker_entry=bool(tracker.entry_time[0,0,0]>0),
            contact_qualified=bool(tracker.contact_step[0,0,0]),
            latched_impact_speed_m_s=float(tracker.entry_speed[0,0,0]),
            stats=tracker.stats())
        return verdict
    paths=[
        run('upper side, horizontal swipe',previous_radial=.159,current_radial=.155,
            previous_axial=-.003,current_axial=-.003,tangential_speed=1.,normal_speed=0.),
        run('upper side, diagonal downward strike',previous_radial=.159,current_radial=.155,
            previous_axial=.001,current_axial=-.003,tangential_speed=1.,normal_speed=1.),
        run('upper side at 5 mm, diagonal',previous_radial=.159,current_radial=.155,
            previous_axial=-.001,current_axial=-.005,tangential_speed=1.,normal_speed=1.),
        run('side at 6 mm, diagonal',previous_radial=.159,current_radial=.155,
            previous_axial=-.002,current_axial=-.006,tangential_speed=1.,normal_speed=1.),
        run('side at 30 mm, diagonal',previous_radial=.159,current_radial=.155,
            previous_axial=-.026,current_axial=-.03,tangential_speed=1.,normal_speed=1.),
        run('upper side, diagonal without force',previous_radial=.159,current_radial=.155,
            previous_axial=.001,current_axial=-.003,tangential_speed=1.,normal_speed=1.,force_N=0.),
        run('top face, downward strike',previous_radial=.1,current_radial=.1,
            previous_axial=.03,current_axial=.01,tangential_speed=0.,normal_speed=2.),
    ]
    result=dict(runtime='thick_cymbal_top_entry_side5mm_contact_v4',
        top_normal=normal.tolist(),radius_m=radius,thickness_m=height,
        real_tracker_class=str(BaseTracker),force_threshold_N=1.,
        max_tangential_to_normal_speed_ratio=1.5,
        note='Synthetic probe trajectories and force fed through the actual tracker; no PhysX scene or beat-time reward gate',
        paths=paths)
    out=ROOT/'runs/hihat_side_tracker_paths_v4_20261009.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    for row in paths:print(row['path'],'side',row['upper_side'],
        'force_contact',row['contact_qualified'],'entry',row['tracker_entry'],
        'grazing_rejections',row['stats']['rejected_grazing'])


if __name__=='__main__':main()
