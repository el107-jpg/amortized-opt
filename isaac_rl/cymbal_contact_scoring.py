"""Distinguish top-face cymbal entry from contact on the deep cylinder side."""
from __future__ import annotations

import torch
from amortized_opt.constants import STICK_RADIUS,TIP_RADIUS
from amortized_opt.capsule_contact import capsule_contact_geometry
from amortized_opt.target_contact import (SIDE_CREDIT_DEPTH_M,
    SIDE_CREDIT_ABOVE_M,SIDE_CREDIT_RADIAL_INSET_M,SIDE_CREDIT_RADIAL_EXTRA_M)


def cymbal_entry_geometry(current,previous,drum_pos,normals,radii,heights,edge_drums,
                          has_previous,probe_allowed):
    """Classify per-probe top entry, upper side contact and bounded depth.

    Positions are centre points of the physical tip sphere and shaft capsules.
    A top entry may continue through the body thickness while in a contact
    episode; a side strike qualifies only at the upper 5 mm of the side wall.
    The caller still requires real PhysX force and the existing impact-speed
    and incidence tests before crediting a beat.
    """
    if current.ndim!=4 or previous.shape!=current.shape:
        raise ValueError('Expected [env,hand,probe,xyz] current/previous positions')
    pos=drum_pos if drum_pos.ndim==3 else drum_pos[None]
    normal=normals[None,None,None]
    def coordinates(points):
        delta=points[:,:,:,None]-pos[:,None,None]
        axial=(delta*normal).sum(-1)
        radial=torch.linalg.vector_norm(delta-axial[...,None]*normal,dim=-1)
        return axial,radial
    axial,radial=coordinates(current)
    previous_axial,previous_radial=coordinates(previous)
    probe_radius=current.new_full((current.shape[2],),STICK_RADIUS)
    probe_radius[0]=TIP_RADIUS
    probe_radius=probe_radius[None,None,:,None]
    radius=radii[None,None,None]
    permitted=probe_allowed[None,None] & edge_drums[None,None,None]
    # A sweep crosses the top face over its physical disk. The 6 mm upper
    # allowance accommodates early force generation from PhysX contact offset.
    top_entry=(permitted & has_previous[:,None,None,None]
        & (previous_axial>=probe_radius-.002)
        & (axial<=probe_radius+.006)
        & (previous_axial-axial>.0005)
        & (previous_radial<=radius+.002)
        & (radial<=radius+.002))
    # Side-wall contact point has approximately the probe centre's axial
    # height: do not let the body's remaining 9.5 cm become a playing surface.
    side=(permitted & (axial>=-SIDE_CREDIT_DEPTH_M-1e-6)
        & (axial<=SIDE_CREDIT_ABOVE_M)
        & (radial>=radius-SIDE_CREDIT_RADIAL_INSET_M)
        & (radial<=radius+probe_radius+SIDE_CREDIT_RADIAL_EXTRA_M))
    inside_body=(permitted & (radial<=radius+probe_radius+SIDE_CREDIT_RADIAL_EXTRA_M)
        & (axial>=-heights[None,None,None]-probe_radius-.003))
    through=(permitted & has_previous[:,None,None,None]
        & (previous_axial>probe_radius)
        & (axial<-heights[None,None,None]-probe_radius)
        & (previous_radial<=radius) & (radial<=radius))
    return top_entry,side,inside_body,through


def make_side_sliver_tracker(base_cls,kit):
    """Wrap the upstream contact tracker only for cymbal/hihat qualification."""
    if getattr(base_cls,'_side_sliver_tracker',False):return base_cls
    radii=torch.as_tensor(kit.radii,dtype=torch.float32)
    heights=torch.as_tensor(kit.heights,dtype=torch.float32)
    edge=torch.as_tensor(kit.edge_hit,dtype=torch.bool)

    class SideSliverContactTracker(base_cls):
        _side_sliver_tracker=True

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.num_drums!=len(radii):raise ValueError('Cymbal scoring kit differs from Isaac layout')
            self._radii=radii.to(self.device);self._heights=heights.to(self.device)
            self._edge=edge.to(self.device)
            self._previous_points=torch.zeros((self.num_envs,2,self.detector.num_probes,3),device=self.device)
            self._has_previous=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            shape=(self.num_envs,2,self.detector.num_probes,self.num_drums)
            self._top_recent=torch.zeros(shape,dtype=torch.int8,device=self.device)
            self._top_latched=torch.zeros(shape,dtype=torch.bool,device=self.device)
            self.n_rejected_deep_side=torch.zeros((),dtype=torch.long,device=self.device)
            self.n_top_qualified=torch.zeros((),dtype=torch.long,device=self.device)
            self.n_upper_side_qualified=torch.zeros((),dtype=torch.long,device=self.device)
            self.n_complete_through_sweeps=torch.zeros((),dtype=torch.long,device=self.device)
            self.debug_contact_geometry=False
            self.debug_events=[]

        def reset(self,env_ids):
            super().reset(env_ids)
            self._previous_points[env_ids]=0.
            self._has_previous[env_ids]=False
            self._top_recent[env_ids]=0
            self._top_latched[env_ids]=False
            if self.debug_contact_geometry:self.debug_events=[]

        def sample(self,tip_pos_env,tip_vel,drum_pos,now,contact_force):
            points=tip_pos_env if tip_pos_env.ndim==4 else tip_pos_env[:,:,None]
            before_entry=self.entry_time.clone() if self.debug_contact_geometry else None
            top,side,inside,through=cymbal_entry_geometry(points,self._previous_points,drum_pos,
                self.detector.head_normal,self._radii,self._heights,self._edge,
                self._has_previous,self.detector.probe_allowed)
            self.n_complete_through_sweeps+=through.sum()
            self._top_recent=torch.where(top,torch.full_like(self._top_recent,3),
                (self._top_recent-1).clamp_min(0))
            touching=torch.linalg.vector_norm(contact_force,dim=-1)>self.force_threshold
            self._top_latched=(self._top_latched | (self._top_recent>0)) & touching[:,:,None,:] & self._edge[None,None,None]
            top_qualified_probes=self._top_latched & inside
            top_qualified=top_qualified_probes.any(dim=2)
            upper_side=side.any(dim=2) & touching
            self.n_top_qualified+=(top_qualified & touching).sum()
            self.n_upper_side_qualified+=(upper_side & ~top_qualified).sum()
            self.n_rejected_deep_side+=(touching & self._edge[None,None]
                & ~top_qualified & ~upper_side).sum()
            custom=top_qualified_probes | side
            original_band=self.detector.band
            edge_mask=self._edge[None,None,None]
            contact_band=self.detector.band_z_tol_above
            def band_override(positions,centres,tol=None):
                result=original_band(positions,centres,tol)
                if tol is not None and abs(float(tol)-float(contact_band))<1e-9:
                    return torch.where(edge_mask,custom,result)
                return result
            self.detector.band=band_override
            try:
                super().sample(tip_pos_env,tip_vel,drum_pos,now,contact_force)
                if self.debug_contact_geometry and len(self.debug_events)<128:
                    new=(self.entry_time>before_entry+1e-6)&self._edge[None,None]
                    pos=drum_pos if drum_pos.ndim==3 else drum_pos[None]
                    for env,hand,drum in new.nonzero(as_tuple=False).cpu().tolist():
                        probes=(top_qualified_probes[env,hand,:,drum]
                            | side[env,hand,:,drum]).nonzero(as_tuple=True)[0]
                        if not len(probes):continue
                        probe=int(probes[0]);normal=self.detector.head_normal[drum]
                        center=pos[env,drum] if len(pos)>1 else pos[0,drum]
                        delta=points[env,hand,probe]-center
                        axial=float((delta*normal).sum())
                        radial=float(torch.linalg.vector_norm(delta-axial*normal))
                        self.debug_events.append(dict(time_s=float(now[env]),env=env,hand=hand,drum=drum,
                            probe=probe,entry='top' if bool(top_qualified[env,hand,drum]) else 'upper_side',
                            axial_m=axial,radial_m=radial,head_radius_m=float(self._radii[drum]),
                            force_N=float(torch.linalg.vector_norm(contact_force[env,hand,drum])),
                            latched_impact_speed_m_s=float(self.entry_speed[env,hand,drum])))
            finally:
                self.detector.band=original_band
                self._previous_points.copy_(points)
                self._has_previous.fill_(True)

        def stats(self):
            result=super().stats()
            result.update(top_qualified_substeps=int(self.n_top_qualified),
                upper_side_qualified_substeps=int(self.n_upper_side_qualified),
                rejected_deep_side_substeps=int(self.n_rejected_deep_side),
                complete_through_sweeps=int(self.n_complete_through_sweeps))
            return result

    return SideSliverContactTracker


def make_capsule_tracker(base_cls,kit):
    """PhysX strike tracker with a continuous distal 20-cm playing shaft.

    Pads retain the upstream tip-only detector. Edge drums use the shared
    continuous capsule witness solver, real force, pre-impact normal speed,
    contact onset, and the same backswing re-arm gate. The probe-based v4
    wrapper remains available for archived labels.
    """
    if getattr(base_cls,'_capsule_tracker',False):return base_cls
    radii=torch.as_tensor(kit.radii,dtype=torch.float32)
    heights=torch.as_tensor(kit.heights,dtype=torch.float32)
    edge=torch.as_tensor(kit.edge_hit,dtype=torch.bool)

    class CapsuleContactTracker(base_cls):
        _capsule_tracker=True

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.num_drums!=len(radii):
                raise ValueError('Capsule contact kit differs from Isaac layout')
            self._radii=radii.to(self.device)
            self._heights=heights.to(self.device)
            self._edge=edge.to(self.device)
            E,D=self.num_envs,self.num_drums
            self._capsule_prev_points=torch.zeros(E,2,2,3,device=self.device)
            self._capsule_prev_vel=torch.zeros_like(self._capsule_prev_points)
            self._capsule_pending=torch.ones(E,dtype=torch.bool,device=self.device)
            self._capsule_contact=torch.zeros(E,2,D,dtype=torch.bool,device=self.device)
            self._capsule_armed=torch.zeros_like(self._capsule_contact)
            self._capsule_top_recent=torch.zeros(E,2,D,dtype=torch.int8,device=self.device)
            self._capsule_top_s=torch.zeros(E,2,D,device=self.device)
            self._capsule_top_radius=torch.zeros(E,2,D,device=self.device)
            self._capsule_top_speed=torch.zeros(E,2,D,device=self.device)
            self._capsule_top_latched=torch.zeros_like(self._capsule_contact)
            self.n_top_qualified=torch.zeros((),dtype=torch.long,device=self.device)
            self.n_upper_side_qualified=torch.zeros((),dtype=torch.long,device=self.device)
            self.n_rejected_deep_side=torch.zeros((),dtype=torch.long,device=self.device)
            self.debug_contact_geometry=False
            self.debug_events=[]

        def reset(self,env_ids):
            super().reset(env_ids)
            self._capsule_prev_points[env_ids]=0
            self._capsule_prev_vel[env_ids]=0
            self._capsule_pending[env_ids]=True
            self._capsule_contact[env_ids]=False
            self._capsule_armed[env_ids]=False
            self._capsule_top_recent[env_ids]=0
            self._capsule_top_s[env_ids]=0
            self._capsule_top_radius[env_ids]=0
            self._capsule_top_speed[env_ids]=0
            self._capsule_top_latched[env_ids]=False
            if self.debug_contact_geometry:self.debug_events=[]

        def sample(self,tip_pos_env,tip_vel,drum_pos,now,contact_force):
            points=tip_pos_env if tip_pos_env.ndim==4 else tip_pos_env[:,:,None]
            velocity=tip_vel if tip_vel.ndim==4 else tip_vel[:,:,None]
            if points.shape[2]!=5 or velocity.shape!=points.shape:
                raise ValueError('Capsule tracker requires tip + four distal shaft samples')
            # The final shaft probe is at the wrist mount; the tip is s=1.
            mount,bead=points[:,:,-1],points[:,:,0]
            vmount,vbead=velocity[:,:,-1],velocity[:,:,0]
            # The upstream implementation remains the exact pad tracker. Edge
            # force is masked so its discrete probes cannot create a second hit.
            pad_force=contact_force.masked_fill(self._edge[None,None,:,None],0)
            super().sample(tip_pos_env,tip_vel,drum_pos,now,pad_force)
            force=torch.linalg.vector_norm(contact_force,dim=-1)
            pos=drum_pos if drum_pos.ndim==3 else drum_pos[None].expand(self.num_envs,-1,-1)
            prev_mount=self._capsule_prev_points[:,:,0]
            prev_bead=self._capsule_prev_points[:,:,1]
            pv_mount=self._capsule_prev_vel[:,:,0]
            pv_bead=self._capsule_prev_vel[:,:,1]
            edge_ids=self._edge.nonzero(as_tuple=True)[0]
            edge_count=int(edge_ids.numel())
            if edge_count:
                edge_heads=pos[:,edge_ids][:,None].expand(-1,2,-1,-1)
                edge_normals=self.detector.head_normal[edge_ids][None,None].expand(
                    self.num_envs,2,-1,-1)
                edge_radii=self._radii[edge_ids][None,None].expand(self.num_envs,2,-1)
                edge_heights=self._heights[edge_ids][None,None].expand(self.num_envs,2,-1)
                extended=lambda value:value[:,:,None].expand(-1,-1,edge_count,-1)
                all_s,all_top,all_side,_all_inside,all_speed,_all_current=(
                    capsule_contact_geometry(
                        extended(prev_mount),extended(prev_bead),
                        extended(mount),extended(bead),edge_heads,edge_normals,
                        edge_radii,edge_heights,extended(pv_mount),extended(pv_bead),
                        incidence_ratio=(1e9 if self.max_tangential_ratio is None
                            else self.max_tangential_ratio)))
            for edge_index,drum in enumerate(edge_ids.tolist()):
                head=pos[:,drum,None,:].expand(-1,2,-1)
                normal=self.detector.head_normal[drum].view(1,1,3).expand_as(head)
                radius=self._radii[drum].expand(self.num_envs,2)
                height=self._heights[drum].expand(self.num_envs,2)
                s=all_s[:,:,edge_index]
                top=all_top[:,:,edge_index]
                side=all_side[:,:,edge_index]
                speed=all_speed[:,:,edge_index]
                touching=force[:,:,drum]>self.force_threshold
                best_top=torch.where(top,speed,torch.full_like(speed,-1.)).argmax(-1)
                top_event=top.any(-1)&~self._capsule_pending[:,None]
                chosen_s=s.gather(-1,best_top[...,None]).squeeze(-1)
                chosen_speed=speed.gather(-1,best_top[...,None]).squeeze(-1)
                chosen_radius=torch.where(best_top==s.shape[-1]-1,
                    torch.full_like(chosen_s,TIP_RADIUS),
                    torch.full_like(chosen_s,STICK_RADIUS))
                recent=self._capsule_top_recent[:,:,drum]
                self._capsule_top_recent[:,:,drum]=torch.where(top_event,
                    torch.full_like(recent,3),(recent-1).clamp_min(0))
                self._capsule_top_s[:,:,drum]=torch.where(top_event,chosen_s,
                    self._capsule_top_s[:,:,drum])
                self._capsule_top_radius[:,:,drum]=torch.where(top_event,chosen_radius,
                    self._capsule_top_radius[:,:,drum])
                self._capsule_top_speed[:,:,drum]=torch.where(top_event,chosen_speed,
                    self._capsule_top_speed[:,:,drum])
                latch=self._capsule_top_latched[:,:,drum]
                latch=(latch|(self._capsule_top_recent[:,:,drum]>0))&touching
                self._capsule_top_latched[:,:,drum]=latch
                contact_point=mount+self._capsule_top_s[:,:,drum,None]*(bead-mount)
                n=normal
                delta=contact_point-head
                axial=(delta*n).sum(-1)
                radial=torch.linalg.vector_norm(delta-axial[...,None]*n,dim=-1)
                same_material_inside=(radial<=radius+self._capsule_top_radius[:,:,drum]
                    +SIDE_CREDIT_RADIAL_EXTRA_M)&(axial>=-height-self._capsule_top_radius[:,:,drum]
                    -SIDE_CREDIT_RADIAL_EXTRA_M)
                top_qualified=latch&same_material_inside
                side_speed=torch.where(side,speed,torch.zeros_like(speed)).amax(-1)
                side_qualified=side.any(-1)&touching&~self._capsule_pending[:,None]
                qualified=touching&(top_qualified|side_qualified)
                self.n_top_qualified+=top_qualified.sum()
                self.n_upper_side_qualified+=(side_qualified&~top_qualified).sum()
                self.n_rejected_deep_side+=(touching&~qualified).sum()
                self.n_contacts+=touching.sum()
                self.n_rejected_band+=(touching&~qualified).sum()
                onset=qualified&~self._capsule_contact[:,:,drum]&~self._capsule_pending[:,None]
                if self.require_rearm:
                    entered=onset&self._capsule_armed[:,:,drum]
                    self.n_rejected_rearm+=(onset&~self._capsule_armed[:,:,drum]).sum()
                else:
                    entered=onset
                impact_speed=torch.where(top_qualified,self._capsule_top_speed[:,:,drum],side_speed)
                entered=entered&(impact_speed>1e-6)
                self.n_rejected_grazing+=(onset&(impact_speed<=1e-6)).sum()
                self.n_entries+=entered.sum()
                self.entry_speed[:,:,drum]=torch.where(entered,impact_speed,self.entry_speed[:,:,drum])
                stamp=now[:,None].expand(-1,2)
                self.entry_time[:,:,drum]=torch.where(entered,stamp,self.entry_time[:,:,drum])
                self.events[:,:,drum]|=entered&(impact_speed>=self.detector.stray_min_speed)
                self.contact_step[:,:,drum]|=qualified
                self.peak_force[:,:,drum]=torch.maximum(self.peak_force[:,:,drum],
                    force[:,:,drum]*qualified)
                if self.require_rearm:
                    if self.detector.strike_rearm_axial is None:
                        rearm=~qualified
                    else:
                        z_mount=((mount-head)*n).sum(-1)
                        z_bead=((bead-head)*n).sum(-1)
                        rearm=torch.minimum(z_mount,z_bead)>=self.detector.strike_rearm_axial
                    self._capsule_armed[:,:,drum]=rearm|(
                        self._capsule_armed[:,:,drum]&~entered)
                self._capsule_contact[:,:,drum]=qualified
                if self.debug_contact_geometry and len(self.debug_events)<128:
                    for env,hand in entered.nonzero(as_tuple=False).cpu().tolist():
                        is_top=bool(top_qualified[env,hand])
                        if is_top:
                            fraction=float(self._capsule_top_s[env,hand,drum])
                        else:
                            witness=torch.where(side[env,hand],speed[env,hand],
                                                torch.full_like(speed[env,hand],-1.)).argmax()
                            fraction=float(s[env,hand,witness])
                        witness_point=mount[env,hand]+fraction*(bead[env,hand]-mount[env,hand])
                        witness_delta=witness_point-head[env,hand]
                        witness_axial=float((witness_delta*normal[env,hand]).sum())
                        witness_radial=float(torch.linalg.vector_norm(
                            witness_delta-witness_axial*normal[env,hand]))
                        self.debug_events.append(dict(time_s=float(now[env]),env=env,
                            hand=hand,drum=drum,fraction=fraction,
                            entry='top' if is_top else 'upper_side',
                            axial_m=witness_axial,radial_m=witness_radial,
                            head_radius_m=float(radius[env,hand]),
                            force_N=float(force[env,hand,drum]),
                            latched_impact_speed_m_s=float(impact_speed[env,hand])))
            self._capsule_prev_points[:,:,0].copy_(mount)
            self._capsule_prev_points[:,:,1].copy_(bead)
            self._capsule_prev_vel[:,:,0].copy_(vmount)
            self._capsule_prev_vel[:,:,1].copy_(vbead)
            self._capsule_pending.fill_(False)

        def stats(self):
            result=super().stats()
            result.update(top_qualified_substeps=int(self.n_top_qualified),
                upper_side_qualified_substeps=int(self.n_upper_side_qualified),
                rejected_deep_side_substeps=int(self.n_rejected_deep_side))
            return result

    return CapsuleContactTracker
