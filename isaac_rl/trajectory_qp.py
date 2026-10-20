"""Small-budget SQP repair of frozen decoder paths at a fixed hit deadline.

Every waypoint of the active arm and waist is a variable. The idle arm, q0,
v0 and duration are fixed. Each QP linearizes the worst signed constraint per
actuator/obstacle over the entire evaluated path. These are nonsmooth maxima;
we reselect their active points and check the actual nonlinear path after
every step. Failure within this budget is NOT an infeasibility certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import osqp
import scipy.sparse as sparse
import torch

from amortized_opt.ik import arm_joint_indices
from amortized_opt.constants import STICK_TIP_SIGMA
from amortized_opt.strike import CYMBAL_CONTACT_SIGMA_FALLBACK
from amortized_opt.sdf import segment_hit_region
from isaac_rl.transition_calibration import exact_contact_region


@dataclass
class QPConfig:
    iterations: int = 3
    trust_rad: float = .12
    smooth_weight: float = 2.
    slack_weight: float = 1000.
    jacobian_chunk: int = 256
    line_search: tuple[float, ...] = (1., .5, .25, .125)
    signed_contact: bool = False
    torque_time_bins: int = 1


def cost_for_horizon(cost, horizon):
    """Reuse the original plant and identical spline gains without padded work."""
    out = copy.copy(cost)
    out.max_horizon = horizon
    out.spline = copy.copy(cost.spline)
    out.spline.max_segments = horizon
    out.spline._Gx_np = cost.spline._Gx_np[:horizon+1, :horizon+1, :horizon+1]
    out.spline._Gv_np = cost.spline._Gv_np[:horizon+1, :horizon+1]
    out.spline._tables = {}
    return out


def signed_contact_constraints(cost, traj, ctx, depth_m):
    """Signed disk/slab bounds at the current closest legal shaft contact.

    Unlike a clamped distance, these retain gradients inside the strike region.
    Shaft contact is reselected each SQP step; its coordinate is held fixed for
    that step's linearization. The exact region still determines acceptance.
    """
    result = {name: [] for name in ('head_radial', 'head_lo', 'head_hi', 'head_half')}
    batch = torch.arange(len(traj), device=traj.device)
    for h, hand in enumerate(('left', 'right')):
        k = ctx.hit_index[:, h].clamp(1, traj.shape[1])-1
        R, p = cost.fk.frame(traj[batch, k], hand+'_wrist')
        base = p+(R@cost.c_mount_p[hand].to(traj.dtype).view(1, 3, 1)).squeeze(-1)
        local = (cost.c_mount_axis[hand].to(traj.dtype).view(1, 3, 1)
                 if ctx.stick_axis_local is None else ctx.stick_axis_local[:, h, :, None])
        axis = (R@local).squeeze(-1)
        low = high = ctx.contact_sigma[:, h]
        if cost.cfg.contact_mode != 'spec':
            shaft = ctx.shaft_allowed[:, h] > .5
            tip = (torch.full_like(low, STICK_TIP_SIGMA) if ctx.stick_tip_sigma is None
                   else ctx.stick_tip_sigma[:, h])
            low = torch.where(shaft, torch.full_like(low, CYMBAL_CONTACT_SIGMA_FALLBACK), low)
            high = torch.where(shaft, tip, high)
            low = torch.minimum(low, high)
        a, b = base+low[:, None]*axis, base+high[:, None]*axis
        axial_lo = cost.cfg.terminal_axial_hi-depth_m
        with torch.no_grad():
            _, s, _ = segment_hit_region(a, b, ctx.p_star[:, h], ctx.n_out[:, h],
                ctx.hit_radius[:, h], ctx.disk_offset[:, h], ctx.half_plane_dir[:, h],
                ctx.has_half_plane[:, h], axial_lo, cost.cfg.terminal_axial_hi)
        point = a+s.detach()[:, None]*(b-a)
        delta = point-ctx.p_star[:, h]
        axial = (delta*ctx.n_out[:, h]).sum(-1)
        planar = delta-axial[:, None]*ctx.n_out[:, h]-ctx.disk_offset[:, h]
        live = ctx.active[:, h] > .5
        values = (planar.norm(dim=-1)-ctx.hit_radius[:, h], axial_lo-axial,
                  axial-cost.cfg.terminal_axial_hi,
                  -(planar*ctx.half_plane_dir[:, h]).sum(-1))
        for name, value in zip(result, values):
            keep = live & (ctx.has_half_plane[:, h] > .5) if name == 'head_half' else live
            result[name].append(torch.where(keep, value, torch.full_like(value, -1e3)))
    return {name: torch.stack(values, -1) for name, values in result.items()}


def captured_constraints(cost, q, v, traj, ctx, lengths, depth_m, *, signed_contact=False,
                         torque_time_bins=1):
    capture = {}
    profile = {} if torque_time_bins > 1 else None
    terms = cost(q, traj, v, ctx, lengths=lengths, constraint_capture=capture,
                 constraint_profile_capture=profile)
    if profile is not None:
        # A single maximum per actuator can move between distant parts of the
        # path after a step. Keep one bound per time interval so the QP also
        # sees competing peaks, while acceptance still checks the full path.
        excess = torch.where(profile['torque_mask'] > .5,
            profile['torque_tau'].abs()-profile['torque_available'], -1e3)
        capture['torque'] = torch.cat([part.amax(1) for part in
            torch.tensor_split(excess, min(torque_time_bins, excess.shape[1]), dim=1)], -1)
    if signed_contact:
        capture.update(signed_contact_constraints(cost, traj, ctx, depth_m))
    else:
        capture['region'] = exact_contact_region(cost, traj, ctx, depth_band_m=depth_m)[:, None]
    names, parts = [], []
    for name, values in capture.items():
        scale = (.01 if name.startswith(('collision', 'head_')) or name == 'region' else
                 10. if name == 'torque' else 1. if name == 'velocity' else
                 5. if name.startswith('elev') else .1)
        names.extend(f'{name}_{j}' for j in range(values.shape[1]))
        parts.append(values/scale)
    return torch.cat(parts, -1), terms, names


def elevation_feasible(diagnostics, ctx=None):
    """Only active pad strikes require the band; check each hand separately."""
    if ctx is None:
        angle = diagnostics['stick_elev_deg']
        return (angle >= 5.) & (angle <= 45.)
    angles = diagnostics.get('stick_elev_deg_per_hand')
    if angles is None:
        angles = diagnostics['stick_elev_deg'][:, None].expand_as(ctx.active)
    required = (ctx.active > .5) & (ctx.elev_required > .5)
    return (~required | ((angles >= 5.) & (angles <= 45.))).all(-1)


def fully_feasible(terms, region, ctx=None):
    d = terms.diagnostics
    return (torch.isfinite(terms.total) & (region <= 1e-6)
            & (torch.minimum(d['collision_min_sdf'], d['collision_stick_min_sdf']) >= -1e-6)
            & (d['torque_excess_max'] <= 1e-6) & (d['velocity_excess_max'] <= 1e-6)
            & (d['joint_box_excess_max'] <= 1e-6) & (d['contact_dir_cos'] >= .5)
            & elevation_feasible(d, ctx))


def metrics(cost, q, v, traj, ctx, lengths, depth_m, *, signed_contact=False,
            torque_time_bins=1):
    with torch.no_grad():
        g, terms, _ = captured_constraints(cost, q, v, traj, ctx, lengths, depth_m,
                                           signed_contact=signed_contact,
                                           torque_time_bins=torque_time_bins)
        region = exact_contact_region(cost, traj, ctx, depth_band_m=depth_m)
    d = terms.diagnostics
    out = {k: d[k].cpu().numpy() for k in ('torque_excess_max', 'velocity_excess_max',
        'joint_box_excess_max', 'contact_dir_cos', 'stick_elev_deg', 'contact_speed_err')}
    out.update(success=fully_feasible(terms, region, ctx).cpu().numpy(),
        terminal_region_dist=region.cpu().numpy(),
        collision_min_sdf=torch.minimum(d['collision_min_sdf'], d['collision_stick_min_sdf']).cpu().numpy(),
        merit=g.clamp_min(0).square().sum(-1).cpu().numpy())
    return out


def solve_step(jac, g, horizon, dofs, cfg, *, region_rows=None):
    """Convex QP: smooth bounded correction + L1 elastic inequality slacks."""
    n, m = horizon*dofs, len(g)
    # Order variables by waypoint, matching the decoder path. Penalizing the
    # correction's curvature preserves the seed's intended smooth motion.
    if horizon >= 3:
        d2 = sparse.diags([np.ones(horizon-2), -2*np.ones(horizon-2),
                          np.ones(horizon-2)], [0, 1, 2], shape=(horizon-2, horizon))
        d2 = sparse.kron(d2, sparse.eye(dofs), format='csc')
    else:
        d2 = sparse.csc_matrix((0, n))
    P = sparse.block_diag((sparse.eye(n)+cfg.smooth_weight*d2.T@d2,
                          sparse.eye(m)*1e-8), format='csc')
    jac = jac.copy()
    jac[np.abs(jac) < 1e-7] = 0.  # discard negligible far-spline sensitivities
    A = sparse.vstack((sparse.hstack((sparse.csc_matrix(jac), -sparse.eye(m))),
        sparse.hstack((sparse.eye(n), sparse.csc_matrix((n, m)))),
        sparse.hstack((sparse.csc_matrix((m, n)), sparse.eye(m)))), format='csc')
    # A small inward margin for signed inequalities avoids ending on a torque
    # or clearance boundary to floating point accuracy. Region distance has
    # a zero floor: it needs a zero target rather than an impossible negative.
    target = np.full(m, -1e-3)
    if region_rows is not None: target[region_rows] = 0.
    qp = osqp.OSQP()
    qp.setup(P=sparse.triu(P, format='csc'), q=np.r_[np.zeros(n), np.full(m, cfg.slack_weight)],
        A=A, l=np.r_[np.full(m, -np.inf), np.full(n, -cfg.trust_rad), np.zeros(m)],
        u=np.r_[target-g, np.full(n, cfg.trust_rad), np.full(m, np.inf)],
        verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=3000,
        polishing=True, scaled_termination=True, adaptive_rho_interval=50)
    result = qp.solve(raise_error=False)
    # An iteration-limited QP can still propose a useful descent candidate.
    # It is not called converged; its bounded correction must pass the same
    # nonlinear merit/feasibility check as a converged QP.
    usable = result.x is not None and np.isfinite(result.x).all() and result.info.status_val in (1, 2, 7)
    step = (np.clip(result.x[:n], -cfg.trust_rad, cfg.trust_rad).reshape(horizon, dofs)
            if usable else np.zeros((horizon, dofs)))
    return step, dict(
        status=result.info.status, iterations=result.info.iter, usable=usable,
        converged=result.info.status_val in (1, 2),
        primal_residual=float(result.info.prim_res), dual_residual=float(result.info.dual_res),
        max_slack=float(np.max(result.x[n:])) if usable else None)


def refine(cost, q, v, trajectories, ctx, lengths, hand, *, depth_m=.07, cfg=None):
    cfg = cfg or QPConfig()
    if cfg.iterations < 1 or cfg.trust_rad <= 0 or cfg.jacobian_chunk < 1 or cfg.torque_time_bins < 1:
        raise ValueError('QP budget, trust region and Jacobian chunk must be positive')
    columns = list(range(3))+list(arm_joint_indices(('left', 'right')[hand]))
    # All entries in this batch must have the same deadline, so no padded
    # waypoint becomes an optimization variable.
    horizon = trajectories.shape[1]
    if not bool((lengths == horizon).all()):
        raise ValueError('QP refinement requires an unpadded, common physical horizon')
    best = trajectories.detach().clone()
    metric_options = dict(signed_contact=cfg.signed_contact, torque_time_bins=cfg.torque_time_bins)
    before = metrics(cost, q, v, best, ctx, lengths, depth_m, **metric_options)
    accepted = np.zeros(len(q), int)
    stalled = np.zeros(len(q), bool)
    histories = [[] for _ in range(len(q))]
    begin = time.monotonic()
    for iteration in range(cfg.iterations):
        # Feasible seeds are retained; every failed plan gets the same budget.
        current = metrics(cost, q, v, best, ctx, lengths, depth_m, **metric_options)
        if (current['success'] | stalled).all(): break
        with torch.enable_grad():
            x = best[:, :, columns].detach().requires_grad_(True)
            physical = best.index_copy(2, torch.as_tensor(columns, device=q.device), x)
            g, _, names = captured_constraints(cost, q, v, physical, ctx, lengths, depth_m,
                                               **metric_options)
            # Independent trajectories: summing the same row across the batch
            # produces its per-plan gradient without cross-plan terms.
            active = torch.arange(g.shape[1], device=q.device)
            outputs = g[:, active].sum(0)
            chunks = []
            for lo in range(0, len(outputs), cfg.jacobian_chunk):
                eye = torch.eye(len(outputs), device=q.device)[lo:lo+cfg.jacobian_chunk]
                grad = torch.autograd.grad(outputs, x, grad_outputs=eye,
                    is_grads_batched=True, retain_graph=True)[0]
                chunks.append(grad.detach())
            J = torch.cat(chunks).permute(1, 0, 2, 3).reshape(len(q), len(outputs), -1).cpu().numpy()
            G = g[:, active].detach().cpu().numpy()
            region_row = np.asarray([name.startswith(('region_', 'head_entry_')) for name in names])
        step = np.zeros((len(q), horizon, len(columns)), np.float32)
        for b in range(len(q)):
            if current['success'][b] or stalled[b]: continue
            if not np.isfinite(J[b]).all() or not np.isfinite(G[b]).all():
                histories[b].append(dict(iteration=iteration, status='nonfinite_linearization'))
                continue
            step[b], info = solve_step(J[b], G[b], horizon, len(columns), cfg,
                                       region_rows=region_row)
            histories[b].append(dict(iteration=iteration, **info))
            histories[b][-1]['merit_before'] = float(current['merit'][b])
        delta = q.new_tensor(step)
        improved = np.zeros(len(q), bool)
        merit_after = current['merit'].copy()
        success_after = current['success'].copy()
        for alpha in cfg.line_search:
            candidate = best.clone()
            candidate[:, :, columns] = best[:, :, columns]+alpha*delta
            trial = metrics(cost, q, v, candidate, ctx, lengths, depth_m, **metric_options)
            for b in range(len(q)):
                if histories[b] and histories[b][-1]['iteration'] == iteration and not improved[b]:
                    histories[b][-1].setdefault('trials', []).append(dict(
                        alpha=float(alpha), merit=float(trial['merit'][b]),
                        success=bool(trial['success'][b])))
            keep = (~current['success'] & ~stalled & ~improved & np.isfinite(trial['merit']) &
                    (trial['success'] | (trial['merit'] < current['merit']-1e-9)))
            best[keep] = candidate[keep]
            merit_after[keep] = trial['merit'][keep]
            success_after[keep] = trial['success'][keep]
            improved |= keep
            # Later step sizes cannot replace a plan that already accepted a
            # step. Avoid evaluating them once every unresolved plan improved.
            if (current['success'] | stalled | improved).all():
                break
        accepted += improved
        for b in range(len(q)):
            if histories[b] and histories[b][-1]['iteration'] == iteration:
                histories[b][-1]['accepted'] = bool(improved[b])
                histories[b][-1]['merit_after'] = float(merit_after[b])
                histories[b][-1]['strict_success'] = bool(success_after[b])
        # Plans are independent. An unchanged plan has the identical next
        # linearization/step, even when another plan in the batch improved.
        stalled |= ~current['success'] & ~improved
        # With the same linearization and no accepted step, another QP would
        # repeat the identical answer. Log the failure instead of pretending
        # the remaining budget is a certificate.
        if not improved.any(): break
    after = metrics(cost, q, v, best, ctx, lengths, depth_m, **metric_options)
    after['accepted_steps'] = accepted
    return best, before, after, dict(elapsed_s=time.monotonic()-begin,
        variables_per_plan=horizon*len(columns), constraint_rows=len(names) if 'names' in locals() else 0,
        histories=histories)
