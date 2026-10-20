"""Geometric target-head entry exemptions for analytic plan calibration."""
from __future__ import annotations

import torch
from .sdf import segment_hit_region

# Shared physical upper-side band used by the v4 Isaac contact tracker and the
# differentiable cost. Heights are measured along each drum's outward normal.
SIDE_CREDIT_DEPTH_M = .005
SIDE_CREDIT_ABOVE_M = .002
SIDE_CREDIT_RADIAL_INSET_M = .005
SIDE_CREDIT_RADIAL_EXTRA_M = .003
V4_DONE_QUALITY = .25


def v4_contact_selection(distance, quality_deficit, target_speed, *, dt):
    """Select ONE physical probe satisfying geometry and impact quality.

    ``distance`` already includes the top/side union and incidence residual.
    Quality deficit is converted to metres using dt*requested speed. The hard
    minimum is zero iff a single probe satisfies every included requirement;
    it cannot reward moving an unused probe away or mix different probes' hits.
    """
    geometric = (distance-1e-9).clamp_min(0)
    complete = (torch.sqrt(geometric.square()
        +(float(dt)*target_speed*quality_deficit).square()+1e-18)-1e-9).clamp_min(0)
    index = complete.argmin(-1)
    weights = torch.nn.functional.one_hot(index, distance.shape[-1]).to(distance.dtype)
    return weights, complete.amin(-1)


def v4_impact_quality(velocity, outward_normal, target_speed, *,
                      incidence_ratio=1.5):
    """Normal-speed Gamma quality and the downward-incidence hinge.

    The physical scorer uses the *normal* approach speed, not the full vector
    speed. At the requested deadline, timing quality is one; a speed-quality
    value below 0.25 cannot complete the beat. Inputs broadcast across stick
    contact candidates, preserving gradients to both normal and tangent speed.
    """
    n = torch.nn.functional.normalize(outward_normal, dim=-1)
    normal_speed = -(velocity*n).sum(-1)
    tangential = torch.linalg.vector_norm(velocity+normal_speed[..., None]*n,dim=-1)
    ratio = (normal_speed.clamp_min(0)/target_speed.clamp_min(1e-6)).clamp_max(20.)
    quality = ratio*torch.exp(1.-ratio)
    incidence_excess = (tangential-incidence_ratio*normal_speed).clamp_min(0)
    quality_deficit = (V4_DONE_QUALITY-quality).clamp_min(0)
    return normal_speed,tangential,quality,incidence_excess,quality_deficit


def v4_contact_probe_distance(previous, current, head, normal, body_radius,
                              body_height, probe_radii, *, dt,
                              incidence_ratio=1.5):
    """Differentiable metres-to-valid-contact for the v4 top OR upper side.

    This is a trajectory-loss surrogate, not a force simulator. A top contact
    needs an inward sweep over the physical face; the sampled centre may then
    continue through the 10 cm body. A side contact needs its probe centre in
    the upper 5 mm annulus *at impact*. Both require approach within the
    incidence cone. Minimum across branches gives the gradient toward the
    nearer legal mode without classifying all side contact as a hit.
    """
    if previous.shape != current.shape or current.ndim != 3 or dt <= 0:
        raise ValueError('Expected previous/current [batch,probe,xyz]')
    B,P,_=current.shape
    if (head.shape!=(B,3) or normal.shape!=(B,3)
            or body_radius.shape!=(B,) or body_height.shape!=(B,)
            or probe_radii.shape!=(P,)):
        raise ValueError('Physical contact geometry dimensions differ')
    n=torch.nn.functional.normalize(normal,dim=-1)[:,None]
    def coordinates(points):
        delta=points-head[:,None]
        axial=(delta*n).sum(-1)
        radial=torch.linalg.vector_norm(delta-axial[...,None]*n,dim=-1)
        return axial,radial
    z0,r0=coordinates(previous);z1,r1=coordinates(current)
    velocity=(current-previous)/dt
    approach=-(velocity*n).sum(-1)
    tangent=torch.linalg.vector_norm(velocity+approach[...,None]*n,dim=-1)
    angle=(float(dt)*(tangent-incidence_ratio*approach).clamp_min(0)).square()
    R=body_radius[:,None];H=body_height[:,None];rp=probe_radii[None]
    def high(value,limit):return (value-limit).clamp_min(0).square()
    def low(value,limit):return (limit-value).clamp_min(0).square()
    top_sq=(low(z0,rp-.002)+high(z1,rp+.006)
        +low(z1,-H-rp-SIDE_CREDIT_RADIAL_EXTRA_M)
        +high(r0,R+.002)+high(r1,R+.002)
        +low(z0-z1,.0005)+angle)
    side_sq=(low(z1,-SIDE_CREDIT_DEPTH_M)+high(z1,SIDE_CREDIT_ABOVE_M)
        +low(r1,R-SIDE_CREDIT_RADIAL_INSET_M)
        +high(r1,R+rp+SIDE_CREDIT_RADIAL_EXTRA_M)+angle)
    top=torch.sqrt(top_sq+1e-18)
    side=torch.sqrt(side_sq+1e-18)
    return torch.minimum(top,side),top,side


def v4_static_probe_distance(probes, head, normal, body_radius, probe_radii):
    """Pose-only top/upper-side region for the terminal role auxiliary.

    A single pose has no approach direction, contact force or impact speed;
    the trajectory FusedCost handles those. This term only asks that an actual
    tip/shaft probe centre lie at the physical top face or upper side sliver.
    """
    if probes.ndim!=3 or probes.shape[-1]!=3:
        raise ValueError('Expected [batch,probe,xyz] physical stick probes')
    B,P,_=probes.shape
    if (head.shape!=(B,3) or normal.shape!=(B,3)
            or body_radius.shape!=(B,) or probe_radii.shape!=(P,)):
        raise ValueError('Static probe geometry dimensions differ')
    n=torch.nn.functional.normalize(normal,dim=-1)[:,None]
    delta=probes-head[:,None]
    axial=(delta*n).sum(-1)
    radial=torch.linalg.vector_norm(delta-axial[...,None]*n,dim=-1)
    R=body_radius[:,None];rp=probe_radii[None]
    def high(value,limit):return (value-limit).clamp_min(0).square()
    def low(value,limit):return (limit-value).clamp_min(0).square()
    top=torch.sqrt(high(radial,R+.002)+low(axial,rp-.002)
        +high(axial,rp+.006)+1e-18)
    side=torch.sqrt(low(axial,-SIDE_CREDIT_DEPTH_M)
        +high(axial,SIDE_CREDIT_ABOVE_M)
        +low(radial,R-SIDE_CREDIT_RADIAL_INSET_M)
        +high(radial,R+rp+SIDE_CREDIT_RADIAL_EXTRA_M)+1e-18)
    return torch.minimum(top,side),top,side


def target_upper_side_entry(probes, head, normal, body_radius, probe_radii,
                            body_sdf, *, dt, incidence_ratio=1.5,
                            contact_margin=.003):
    """v4 side geometry, downward incidence and a differentiable repair gap.

    ``probes`` is ``(B,S,P,3)`` with tip first and shaft samples afterward;
    wrist samples are excluded. The hard qualification mirrors Isaac's upper
    side band but substitutes near-zero body SDF for unavailable PhysX force.
    The repair gap has metres as its unit, including ``dt`` times the velocity
    incidence violation. It is zero inside the qualified geometric/angle set.
    """
    if probes.ndim != 4 or probes.shape[-1] != 3 or dt <= 0 or incidence_ratio <= 0:
        raise ValueError('Expected batched sampled stick probes and positive timing/incidence')
    B, S, P, _ = probes.shape
    if (head.shape != (B, 3) or normal.shape != (B, 3)
            or body_radius.shape != (B,) or probe_radii.shape != (P,)
            or body_sdf.shape != (B, S)):
        raise ValueError('Side-entry geometry dimensions differ')
    n = torch.nn.functional.normalize(normal, dim=-1)[:, None, None]
    delta = probes-head[:, None, None]
    axial = (delta*n).sum(-1)
    radial = torch.linalg.vector_norm(delta-axial[..., None]*n, dim=-1)
    previous = torch.cat((probes[:, :1], probes[:, :-1]), dim=1)
    velocity = (probes-previous)/dt
    normal_speed = -(velocity*n).sum(-1)
    tangential = torch.linalg.vector_norm(
        velocity+normal_speed[..., None]*n, dim=-1)
    R = body_radius[:, None, None]
    rp = probe_radii[None, None]
    geometry = ((axial >= -SIDE_CREDIT_DEPTH_M-1e-6)
        & (axial <= SIDE_CREDIT_ABOVE_M)
        & (radial >= R-SIDE_CREDIT_RADIAL_INSET_M)
        & (radial <= R+rp+SIDE_CREDIT_RADIAL_EXTRA_M))
    incidence = ((normal_speed > 1e-6)
        & (tangential <= incidence_ratio*normal_speed))
    allowed = (geometry & incidence & (body_sdf <= contact_margin)[..., None]).any(-1)
    axial_gap = ((-SIDE_CREDIT_DEPTH_M-axial).clamp_min(0).square()
        +(axial-SIDE_CREDIT_ABOVE_M).clamp_min(0).square())
    radial_gap = ((R-SIDE_CREDIT_RADIAL_INSET_M-radial).clamp_min(0).square()
        +(radial-R-rp-SIDE_CREDIT_RADIAL_EXTRA_M).clamp_min(0).square())
    angle_gap = (float(dt)*(tangential-incidence_ratio*normal_speed)
                 .clamp_min(0)).square()
    repair = torch.sqrt(axial_gap+radial_gap+angle_gap+1e-18).amin(-1)
    return geometry, incidence, allowed, repair


def target_v4_top_entry_mask(probes, head, normal, body_radius, body_height,
                             probe_radii, body_sdf, *, dt,
                             already_contacting_source, incidence_ratio=1.5,
                             contact_margin=.003):
    """Per-probe top-entry latch for the sampled v4 differentiable trajectory.

    ``body_sdf <= contact_margin`` stands in for unavailable PhysX force.
    A top entry latches for the same proxy contact episode and may continue
    through the physical body's depth; a later side re-entry cannot inherit it.
    Geometry and incidence thresholds match the Isaac tracker at each sample.
    """
    if probes.ndim!=4 or dt<=0:
        raise ValueError('Expected [batch,sample,probe,xyz] path and positive dt')
    B,S,P,_=probes.shape
    if (head.shape!=(B,3) or normal.shape!=(B,3)
            or body_radius.shape!=(B,) or body_height.shape!=(B,)
            or probe_radii.shape!=(P,) or body_sdf.shape!=(B,S)
            or already_contacting_source.shape!=(B,)):
        raise ValueError('Top-entry path/geometry dimensions differ')
    n=torch.nn.functional.normalize(normal,dim=-1)[:,None,None]
    prior=torch.cat((probes[:,:1],probes[:,:-1]),dim=1)
    def coordinates(points):
        delta=points-head[:,None,None]
        axial=(delta*n).sum(-1)
        radial=torch.linalg.vector_norm(delta-axial[...,None]*n,dim=-1)
        return axial,radial
    axial,radial=coordinates(probes)
    prior_axial,prior_radial=coordinates(prior)
    rp=probe_radii[None,None];R=body_radius[:,None,None];H=body_height[:,None,None]
    velocity=(probes-prior)/dt
    approach=-(velocity*n).sum(-1)
    tangential=torch.linalg.vector_norm(velocity+approach[...,None]*n,dim=-1)
    incidence=(approach>1e-6)&(tangential<=incidence_ratio*approach)
    touching=body_sdf<=contact_margin
    top=(touching[...,None] & incidence
        & (prior_axial>=rp-.002) & (axial<=rp+.006)
        & (prior_axial-axial>.0005)
        & (prior_radial<=R+.002) & (radial<=R+.002))
    inside=(radial<=R+rp+SIDE_CREDIT_RADIAL_EXTRA_M)
    inside &= axial>=-H-rp-SIDE_CREDIT_RADIAL_EXTRA_M
    top=torch.cat((top[:,:1] | (already_contacting_source[:,None,None]
        & touching[:,:1,None] & inside[:,:1]),top[:,1:]),dim=1)
    index=torch.arange(S,device=probes.device)[None,:,None]
    last_event=torch.where(top,index,torch.full_like(index,-1)).cummax(1).values
    last_no_contact=torch.where(~touching[...,None],index,
        torch.full_like(index,-1)).cummax(1).values
    latched=(last_event>last_no_contact)&touching[...,None]&inside
    allowed=latched.any(-1)
    first_contact=touching & ~torch.cat((torch.zeros(B,1,device=probes.device,
        dtype=torch.bool),touching[:,:-1]),dim=1)
    _,top_distance,_=v4_contact_probe_distance(prior.reshape(B*S,P,3),
        probes.reshape(B*S,P,3),head[:,None].expand(-1,S,-1).reshape(B*S,3),
        normal[:,None].expand(-1,S,-1).reshape(B*S,3),
        body_radius[:,None].expand(-1,S).reshape(B*S),
        body_height[:,None].expand(-1,S).reshape(B*S),probe_radii,
        dt=dt,incidence_ratio=incidence_ratio)
    repair=torch.where(first_contact & ~allowed,
        top_distance.reshape(B,S,P).amin(-1),torch.zeros_like(body_sdf))
    return allowed,top.any(-1),repair


def target_top_entry_mask(butt, tip, contact_a, contact_b, head, normal,
                          body_radius, body_sdf, *, stick_radius, depth,
                          already_contacting_source, return_repair_distance=False,
                          entry_gate=None):
    """Allow shallow target overlap following entry through its physical top.

    Inputs are batched paths (B,S,3), with per-path head geometry. Interpolate
    the physical shaft's crossing between collocation samples. A new body
    intersection episode requires a fresh top crossing; side re-entry cannot
    inherit an earlier exemption. Once top contact is certified, the bounded
    head-depth continuation is accepted as a proxy for an early impact, and
    the wrist gets no exemption. It is not a real penetration/bounce rollout.

    Returns (exempt, entry_events), both (B,S) boolean. With
    return_repair_distance=True, also returns a differentiable (B,S) distance
    for unapproved episode starts. This is a sampled path check, not continuous
    collision detection or a physical bounce simulation.
    """
    if depth <= 0:
        raise ValueError('target entry depth must be positive')
    B, S, _ = tip.shape
    n = normal[:, None]
    # The collision capsule may contact the skin along any of its shaft.
    # Goal-contact arc restrictions are retained for the depth/terminal gates.
    h_a = ((butt-head[:, None])*n).sum(-1)
    h_b = ((tip-head[:, None])*n).sum(-1)
    height = torch.minimum(h_a, h_b)
    previous_height = torch.cat((height[:, :1], height[:, :-1]), 1)
    descend = previous_height-height
    fraction = ((previous_height-stick_radius)/descend.clamp_min(1e-9)).clamp(0, 1)
    # Head contact may be along a playing-shaft segment, rather than at its
    # lowest endpoint. Test the whole segment at the face and at its swept
    # crossing. This also covers a parallel shaft moving down onto the head.
    # Collision entry concerns the whole physical skin. The smaller goal disk
    # and any playable half-plane still gate terminal hit membership separately.
    def region(a, b, low, high):
        # At height stick_radius the capsule's first skin contact is directly
        # below its centerline. That point need only lie on the physical face;
        # the capsule's full cross-section need not fit inside the head disk.
        radius = body_radius
        return segment_hit_region(a, b, head[:, None], n, radius[:, None],
            torch.zeros_like(n), torch.zeros_like(n), torch.zeros_like(body_radius[:, None]), low, high)
    face_distance, s, face_point = region(butt, tip, stick_radius, stick_radius)
    previous_a = torch.cat((butt[:, :1], butt[:, :-1]), 1)
    previous_b = torch.cat((tip[:, :1], tip[:, :-1]), 1)
    previous_face_point = previous_a+s[..., None]*(previous_b-previous_a)
    prior_face_height = ((previous_face_point-head[:, None])*n).sum(-1)
    face_height = ((face_point-head[:, None])*n).sum(-1)
    face_downward = ((prior_face_height >= stick_radius-1e-6)
                     & (prior_face_height-face_height > 1e-9))
    swept_a = previous_a+fraction[..., None]*(butt-previous_a)
    swept_b = previous_b+fraction[..., None]*(tip-previous_b)
    swept_distance, _, swept_point = region(swept_a, swept_b, stick_radius, stick_radius)
    swept_downward = ((previous_height >= stick_radius-1e-6) & (height <= stick_radius+1e-6)
                      & (descend > 1e-9))
    def fits_physical_head(point):
        v = point-head[:, None]
        radial = torch.linalg.vector_norm(v-(v*n).sum(-1, keepdim=True)*n, dim=-1)
        return radial <= body_radius[:, None]+1e-6
    entry = (((face_distance <= 1e-6) & face_downward & fits_physical_head(face_point))
             | ((swept_distance <= 1e-6) & swept_downward & fits_physical_head(swept_point)))
    if entry_gate is not None:
        if entry_gate.shape != (B, S):
            raise ValueError('Top-entry incidence gate dimensions differ')
        entry &= entry_gate
    legal_depth, _, depth_point = region(contact_a, contact_b, -depth, stick_radius)
    entry[:, 0] = (already_contacting_source & (legal_depth[:, 0] <= 1e-6)
                   & fits_physical_head(depth_point)[:, 0])
    penetrating = body_sdf < -1e-6
    previous_penetrating = torch.cat((torch.zeros(B, 1, device=tip.device, dtype=torch.bool),
                                      penetrating[:, :-1]), 1)
    episode_start = penetrating & ~previous_penetrating
    index = torch.arange(S, device=tip.device)[None].expand(B, -1)
    missing = torch.full_like(index, -S-2)
    last_episode = torch.where(episode_start, index, missing).cummax(1).values
    # A tangency at the preceding sample can register top entry just before
    # the capsule has a nonzero penetration at the next sample.
    previous_entry = torch.cat((torch.zeros(B, 1, device=tip.device, dtype=torch.bool), entry[:, :-1]), 1)
    certified_start = episode_start & (entry | previous_entry)
    last_certified = torch.where(certified_start, index, missing).cummax(1).values
    entered_from_top = (last_certified >= 0) & (last_certified == last_episode)
    allowed = entered_from_top & (legal_depth <= 1e-6)
    if return_repair_distance:
        # The capped-cylinder minimum can be a cap-plane intersection with
        # almost zero derivative. Give SQP an explicit geometric direction
        # toward top entry at the first unapproved overlap of each episode.
        # This guides repair; it does not grant an exemption or change the
        # acceptance mask. Recompute the episode after each nonlinear step.
        needs_entry = episode_start & ~entered_from_top
        distance = torch.where(needs_entry, face_distance, torch.zeros_like(face_distance))
        return allowed, entry, distance
    return allowed, entry
