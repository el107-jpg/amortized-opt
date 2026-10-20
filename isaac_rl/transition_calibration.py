"""Random empirical calibration using frozen decoder paths and existing physics."""
from __future__ import annotations

import numpy as np
import torch
import copy

from amortized_opt.constants import N_DOF, PELVIS_IN_ENV, ready_joint_pos
from amortized_opt.cost import StrikeContext
from amortized_opt.fk import stick_contact_offset
from amortized_opt.ik import arm_joint_indices, correct_tip, tip_position
from amortized_opt.limits import urdf_position_limits, isaac_velocity_limits
from amortized_opt.phase2_data import _strike_row
from amortized_opt.rhythm import sample_hit_point, strike_velocity
from amortized_opt.strike import (default_contact_sigma, shaft_hit_allowed, STICK_ELEV_MIN_DEG,
                                 STICK_ELEV_MAX_DEG, CYMBAL_CONTACT_SIGMA_FALLBACK)
from amortized_opt.constants import STICK_TIP_SIGMA
from amortized_opt.sdf import segment_hit_region
from amortized_opt.trajectory_surrogate import contact_jacobian
from amortized_opt.warmstart import GoalPoseLibrary


def strike_context(kit, targets, sources, ticks, device, *, speeds=None):
    targets, sources = np.asarray(targets), np.asarray(sources)
    B = len(targets)
    rows = [_strike_row(kit, d, sample_hit_point(kit, d), strike_velocity(kit, d, .5))
            for d in range(kit.num_drums)]
    idx = targets.clip(0)
    live = targets >= 0
    tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, device=device, dtype=dtype)
    field = lambda key: tensor(np.asarray([r[key] for r in rows])[idx])
    active = tensor(live)
    velocity = field('v_star')
    if speeds is not None:
        velocity = -field('n_out')*tensor(speeds)[..., None]
    return StrikeContext(hit_index=tensor(np.broadcast_to(np.asarray(ticks)[:, None], (B, 2)).copy(), torch.long),
        active=active, target_drum=tensor(targets, torch.long), start_drum=tensor(sources, torch.long),
        p_star=field('p_star'), v_star=velocity, n_out=field('n_out'), hit_radius=field('hit_radius'),
        disk_offset=field('disk_off'), half_plane_dir=field('hp_dir'), has_half_plane=field('has_hp'),
        contact_sigma=field('sigma'), shaft_allowed=field('shaft'),
        start_shaft_allowed=tensor([[float(shaft_hit_allowed(kit.names[d])) if d >= 0 else 0.
                                    for d in row] for row in sources]), elev_required=field('elev'))


@torch.no_grad()
def static_clearance(cost, kit, q, sources, *, ignore_idle_arm=False):
    """Full collision checks with the existing departure contact exemptions."""
    B = len(q)
    ctx = strike_context(kit, sources, sources, np.full(B, 2), q.device)
    diag = {}
    masks = {h: q.new_ones(B, 1) for h in ('left', 'right')}
    if ignore_idle_arm:
        masks = {h: ctx.active[:, hi, None] for hi, h in enumerate(('left', 'right'))}
    options = {'active_hand_constraints_only': True} if ignore_idle_arm else {}
    cost._collision(cost.fk.forward(q), B, 1, ctx,
        masks, {}, {}, {}, diag, {}, None, **options)
    return torch.minimum(diag['collision_min_sdf'], diag['collision_stick_min_sdf'])


def near_head_starts(cost, kit, hand, source, yaw, samples, rng, *, ignore_idle_arm=False):
    """Random near-head points, joint configurations and outward tip velocities.

    Correct only the source arm; keep waist yaw fixed and the other arm ready.
    Library poses seed IK, including unrestricted pairs absent from the library.
    Invalid starts are recorded and excluded from successful witness counts.
    """
    bounds = urdf_position_limits(posture_limits=False)
    columns = arm_joint_indices(hand)
    library = GoalPoseLibrary()
    yaws = np.broadcast_to(np.asarray(yaw, float), (samples,))
    if not np.isfinite(yaws).all():
        raise ValueError('Starting yaw must be finite, scalar or one value per sample')
    base = np.tile(ready_joint_pos(), (samples, 1))
    base[:, 0] = yaws
    if library.has(hand, kit.names[source]):
        base[:, columns] = np.stack([library.select(hand, kit.names[source], float(y)).arm_q
                                    for y in yaws])
    base[:, columns] += rng.normal(0, .08, (samples, len(columns)))
    base = np.clip(base, *bounds)
    points = np.stack([sample_hit_point(kit, source, rng, hit_radius_frac=.25)
                       +kit.axes[source]*rng.uniform(.002, .012) for _ in range(samples)])
    q = torch.as_tensor(base, device=cost.device, dtype=torch.float32)
    aim = q.new_tensor(points)
    sigma = default_contact_sigma(kit.names[source])
    q = correct_tip(cost.fk, q, aim, hand, steps=30, damping=.025,
                    sigma=sigma, q_bounds=bounds)
    error = (tip_position(cost.fk, q, hand, sigma=sigma, frame='env')-aim).norm(dim=-1)
    local = q.new_tensor(stick_contact_offset(hand, sigma)).expand(samples, -1)
    _, jac = contact_jacobian(cost.fk, q, hand, local)
    J = jac[:, :, columns]
    normal = q.new_tensor(kit.axes[source])
    tangent = q.new_tensor(rng.normal(0, .10, (samples, 3)))
    tangent -= (tangent@normal)[:, None]*normal
    desired = normal[None]*q.new_tensor(rng.uniform(.1, 1.2, samples))[:, None]+tangent
    # Restrict inverse inertia to the moving arm so yaw/other-arm rates stay zero.
    from amortized_opt.ik import inertia_weighted_pinv
    # The actual cost RNEA supplies per-pose effective mass, including armature.
    basis = torch.eye(N_DOF, device=q.device).expand(samples, -1, -1)
    Q = q[:, None].expand(-1, N_DOF, -1).reshape(-1, N_DOF)
    mass = cost.rnea.mass_matrix_torque(Q, basis.reshape(-1, N_DOF)).reshape(samples, N_DOF, N_DOF).transpose(1, 2)
    M = mass[:, columns][:, :, columns]
    rates = (inertia_weighted_pinv(J, M=M, damping=1e-4)@desired[..., None]).squeeze(-1)
    v = torch.zeros_like(q); v[:, columns] = rates
    limits = q.new_tensor(isaac_velocity_limits())
    # Keep requested outward direction when capping unusually large IK rates.
    scale = (limits/v.abs().clamp_min(1e-9)).amin(-1).clamp_max(1)
    v *= scale[:, None]
    realized = torch.einsum('bij,bj->bi', jac, v)
    sources = np.full((samples, 2), -1, int)
    sources[:, 0 if hand == 'left' else 1] = source
    clearance = static_clearance(cost, kit, q, sources, ignore_idle_arm=ignore_idle_arm)
    valid = ((error <= .003) & (clearance >= -1e-6) & ((realized@normal) > .05)
             & torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1))
    return q, v, valid, dict(ik_error_m=error.cpu().numpy(),
        clearance_m=clearance.cpu().numpy(), outward_speed_m_s=(realized@normal).cpu().numpy(),
        sampled_yaw_rad=q[:, 0].cpu().numpy())


def exact_contact_region(cost, traj, ctx, *, depth_band_m=None):
    """Exact closest legal contact at the deadline, without the training softmin.

    Tip-only drums use a point; cymbals/hi-hat use their allowed shaft segment.
    Return the largest active hand's excess so each requested hit must qualify.
    Optional depth extends inward from the existing outer face for calibration
    only. It does not change the checkpoint, cost configuration or collision law.
    """
    B = len(traj)
    axial_lo = cost.cfg.terminal_axial_lo
    if depth_band_m is not None:
        if not np.isfinite(depth_band_m) or depth_band_m <= 0:
            raise ValueError('calibration contact depth must be finite and positive')
        axial_lo = cost.cfg.terminal_axial_hi-float(depth_band_m)
    distance = traj.new_zeros(B)
    batch = torch.arange(B, device=traj.device)
    for h, hand in enumerate(('left', 'right')):
        if not bool((ctx.active[:, h] > .5).any()):
            continue
        k = ctx.hit_index[:, h].clamp(1, traj.shape[1])-1
        q = traj[batch, k]
        R, p = cost.fk.frame(q, hand+'_wrist')
        base = p+(R@cost.c_mount_p[hand].to(q.dtype).view(1, 3, 1)).squeeze(-1)
        local_axis = (cost.c_mount_axis[hand].to(q.dtype).view(1, 3, 1)
                      if ctx.stick_axis_local is None else ctx.stick_axis_local[:, h, :, None])
        axis = (R@local_axis).squeeze(-1)
        low = high = ctx.contact_sigma[:, h]
        if cost.cfg.contact_mode != 'spec':
            tip = (torch.full_like(low, STICK_TIP_SIGMA) if ctx.stick_tip_sigma is None
                   else ctx.stick_tip_sigma[:, h])
            shaft = ctx.shaft_allowed[:, h] > .5
            low = torch.where(shaft, torch.full_like(low, CYMBAL_CONTACT_SIGMA_FALLBACK), low)
            high = torch.where(shaft, tip, high)
            low = torch.minimum(low, high)
        exact, _, _ = segment_hit_region(base+low[:, None]*axis, base+high[:, None]*axis,
            ctx.p_star[:, h], ctx.n_out[:, h], ctx.hit_radius[:, h], ctx.disk_offset[:, h],
            ctx.half_plane_dir[:, h], ctx.has_half_plane[:, h],
            axial_lo, cost.cfg.terminal_axial_hi)
        distance = torch.maximum(distance, exact*ctx.active[:, h])
    return distance


@torch.no_grad()
def evaluate_trials(generator, q, v, hand, sources, destinations, yaws, durations,
                    *, chunk=64, speed=2., hit_depth_m=None, require_hit_speed=True,
                    allow_early_target_top_entry=False, ignore_idle_arm=False):
    """All durations are checked independently; neural feasibility need not be monotone."""
    samples, K = durations.shape
    flat = np.asarray(durations).reshape(-1)
    indexes = np.repeat(np.arange(samples), K)
    results = {}
    trial_cost = generator.cost
    if ignore_idle_arm:
        trial_cost = copy.copy(generator.cost)
        trial_cost.calibration_active_hand_only = True
    for start in range(0, len(flat), chunk):
        ids = indexes[start:start+chunk]
        t = flat[start:start+chunk]
        Q, V = q[ids], v[ids]
        B = len(ids)
        targets = np.full((B, 2), -1, int)
        targets[:, hand] = np.asarray(destinations)[ids]
        source = np.full((B, 2), -1, int)
        source[:, hand] = np.asarray(sources)[ids]
        drums = torch.as_tensor(targets[:, None], device=q.device)
        times = q.new_tensor(t[:, None])
        speeds = q.new_full((B, 1, 2), speed)
        valid = torch.ones(B, 1, device=q.device, dtype=torch.bool)
        plan = generator.predict(Q, V, drums, times, speeds, valid, 0,
            return_plan=True, waist_yaw=np.asarray(yaws)[ids])
        ctx = strike_context(generator.kit, targets, source, np.rint(t/.02).astype(int),
                             q.device, speeds=np.full((B, 2), speed))
        if allow_early_target_top_entry:
            ctx.target_top_entry_depth_m = (hit_depth_m if hit_depth_m is not None else
                generator.cost.cfg.terminal_axial_hi-generator.cost.cfg.terminal_axial_lo)
        # Decoder masking excludes idle joints from its output. Their physical
        # execution holds q0, and collision checks must include that arm for
        # the ENTIRE moving arm's horizon (including cross-stick clearance).
        physical_traj = plan['traj'].clone()
        idle_cols = arm_joint_indices(('left', 'right')[1-hand])
        physical_traj[:, :, idle_cols] = Q[:, None, idle_cols]
        # Spline groups are (waist, left, right), not (left, right). Both
        # physical arms exist throughout this single-hit experiment.
        physical_lengths = torch.as_tensor(np.rint(t/.02), device=q.device,
            dtype=plan['lengths'].dtype)[:, None].expand_as(plan['lengths']).clone()
        ct = trial_cost.forward(Q, physical_traj, V, ctx, lengths=physical_lengths)
        diag = ct.diagnostics
        exact_region = exact_contact_region(generator.cost, physical_traj, ctx,
                                            depth_band_m=hit_depth_m)
        from .trajectory_qp import elevation_feasible
        geometric = ((exact_region <= 1e-6) &
            (diag['contact_dir_cos'] >= .5) & elevation_feasible(diag, ctx))
        if require_hit_speed:
            geometric &= diag['contact_speed_err'] <= .75
        clearance = torch.minimum(diag['collision_min_sdf'], diag['collision_stick_min_sdf'])
        hard = (geometric & (clearance >= -1e-6) & (diag['joint_box_excess_max'] <= 1e-6)
                & (diag['torque_excess_max'] <= 1e-6) & (diag['velocity_excess_max'] <= 1e-6))
        finite = torch.isfinite(ct.total)
        values = dict(success=(hard & finite), geometric=(geometric & finite),
            terminal_region_dist=exact_region, terminal_region_dist_soft=diag['terminal_region_dist'],
            collision_min_sdf=clearance,
            torque_excess_max=diag['torque_excess_max'], velocity_excess_max=diag['velocity_excess_max'],
            joint_box_excess_max=diag['joint_box_excess_max'], contact_speed_err=diag['contact_speed_err'],
            contact_dir_cos=diag['contact_dir_cos'], stick_elev_deg=diag['stick_elev_deg'])
        if allow_early_target_top_entry:
            name = ('left', 'right')[hand]
            values['target_top_entry_exempt_samples'] = diag[f'target_top_entry_exempt_samples_{name}']
            values['target_top_entry_first_time_s'] = diag[f'target_top_entry_first_time_s_{name}']
        for name, value in values.items():
            results.setdefault(name, []).append(value.cpu().numpy())
    return {name: np.concatenate(parts).reshape(samples, K) for name, parts in results.items()}


def random_durations(rng, samples, trials, *, minimum=.08, maximum=1.8):
    """Log-uniform random deadlines plus four broad anchors on the 50 Hz grid."""
    if trials < 4 or samples < 1 or minimum < .02 or maximum > 2. or maximum < minimum:
        raise ValueError('invalid duration search configuration')
    values = np.exp(rng.uniform(np.log(minimum), np.log(maximum), (samples, trials)))
    values[:, :4] = np.geomspace(minimum, maximum, 4)
    return np.clip(np.rint(values/.02)*.02, np.ceil(minimum/.02)*.02,
                   np.floor(maximum/.02)*.02)


def descending_deadlines(*, reference=1., minimum=.08, grid=.02,
                         longer=(1.2, 1.4, 1.6, 1.8)):
    """Reference first, then every shorter grid point, then longer fallback anchors.

    Evaluate every point independently even after failure; neural generation
    has no monotonic feasibility guarantee. Never reuse decoder outputs as
    seeds when calibrating a straight-line initialization model.
    """
    if not 0 < grid <= minimum <= reference <= 2.:
        raise ValueError('invalid descending deadline range')
    lo, hi = int(round(minimum/grid)), int(round(reference/grid))
    if not np.isclose(lo*grid, minimum) or not np.isclose(hi*grid, reference):
        raise ValueError('reference and minimum must be on the control grid')
    extension = np.asarray(longer, float)
    if (not np.isfinite(extension).all() or (extension <= reference).any() or
        (extension > 2.).any() or not np.allclose(extension/grid, np.rint(extension/grid))):
        raise ValueError('longer anchors must lie on the grid above reference and at most 2 seconds')
    return np.r_[np.arange(hi, lo-1, -1)*grid, np.unique(extension)]


def full_range_deadlines():
    """Every supported 50 Hz deadline, 20 ms through 2 s, reference first.

    Each starting state gets all 100 trials, including those failing 1 second.
    The decoder horizon is 100 waypoints; larger durations would be clamped.
    """
    return descending_deadlines(minimum=.02, longer=np.arange(51, 101)*.02)


def calibrate_pair_yaws(cost, kit, rng, *, candidates=24, chunk=128):
    """Random fixed-yaw IK search for comfortable ordered drum pairs, with idle.

    This table measures static reach/clearance, not a dynamic two-arm certificate.
    No hand-eligibility mask; unavailable library poses use ready-pose IK seeds.
    """
    from amortized_opt.waist_aim import aim_yaw, WaistAimConfig
    D = kit.num_drums
    library = GoalPoseLibrary()
    bounds = urdf_position_limits(posture_limits=False)
    cfg = WaistAimConfig(engage_in=None, engage_out=None, deadband=0.)
    poses, sources, preferred = [], [], []
    for left in range(D+1):
        for right in range(D+1):
            goals = {h: sample_hit_point(kit, d)-PELVIS_IN_ENV for h, d in
                     zip(('left', 'right'), (left, right)) if d < D}
            pref = aim_yaw(goals, cfg=cfg, yaw_current=0.) if goals else 0.
            preferred.append(pref)
            yaws = rng.uniform(-1.4, 1.4, candidates)
            yaws[0] = pref
            for yaw in yaws:
                q = ready_joint_pos().copy(); q[0] = yaw
                for h, d in zip(('left', 'right'), (left, right)):
                    if d < D and library.has(h, kit.names[d]):
                        cols = arm_joint_indices(h)
                        q[cols] = library.select(h, kit.names[d], yaw).arm_q
                q[3:] += rng.normal(0, .04, N_DOF-3)
                poses.append(np.clip(q, *bounds))
                sources.append([left if left < D else -1, right if right < D else -1])
    q = torch.as_tensor(np.asarray(poses), device=cost.device, dtype=torch.float32)
    sources = np.asarray(sources)
    good = torch.ones(len(q), device=q.device, dtype=torch.bool)
    for hand, name in enumerate(('left', 'right')):
        ids = np.flatnonzero(sources[:, hand] >= 0)
        for start in range(0, len(ids), chunk):
            take = ids[start:start+chunk]
            drums = sources[take, hand]
            # Contact offset varies between shaft drums and tip drums.
            for drum in np.unique(drums):
                selected = take[drums == drum]
                aim = q.new_tensor(sample_hit_point(kit, int(drum))+kit.axes[drum]*.004).expand(len(selected), -1)
                sigma = default_contact_sigma(kit.names[drum])
                q[selected] = correct_tip(cost.fk, q[selected], aim, name, steps=25,
                    damping=.025, sigma=sigma, q_bounds=bounds)
                err = (tip_position(cost.fk, q[selected], name, sigma=sigma, frame='env')-aim).norm(dim=-1)
                good[selected] &= err <= .003
    for start in range(0, len(q), chunk):
        sl = slice(start, start+chunk)
        good[sl] &= static_clearance(cost, kit, q[sl], sources[sl]) >= -1e-6
    Q = q.detach().cpu().numpy().reshape(D+1, D+1, candidates, N_DOF)
    good = good.cpu().numpy().reshape(D+1, D+1, candidates)
    pref = np.asarray(preferred).reshape(D+1, D+1)
    # Prefer modest arm displacement and the analytic geometric aim.
    objective = .02*((Q[..., 3:]-ready_joint_pos()[3:])**2).sum(-1)+(Q[..., 0]-pref[..., None])**2
    objective = np.where(good, objective, np.inf)
    choice = objective.argmin(-1)
    yaw = np.take_along_axis(Q[..., 0], choice[..., None], axis=-1)[..., 0]
    supported = good.any(-1)
    yaw = np.where(supported, yaw, pref)
    yaw[D, D] = 0.; supported[D, D] = True
    return yaw, supported, dict(candidate_yaws=Q[..., 0], valid=good, objective=objective)
