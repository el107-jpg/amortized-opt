"""Single-hit generator rollouts with geometric contact and a bounce surrogate.

No Isaac imports or physics stepping. A frozen terminal MLP directly generates
one next-hit pose. Search rolls forward from that pose and its rebound velocity.
Connecting cubic Hermite segments retain approach and rebound velocities. An
optional trajectory decoder adapter provides a baseline for comparison.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from .constants import G1_UPPER_BODY_JOINTS, N_DOF, PELVIS_IN_ENV, ready_joint_pos
from .fk import stick_contact_offset
from .limits import isaac_effort_limits, isaac_velocity_limits, urdf_position_limits
from .phase2_data import _strike_row
from .rhythm import sample_hit_point, strike_velocity
from .sdf import segment_hit_region
from .strike import shaft_hit_sigma_range, strike_axial_limits
from .tokenizer import ContextInputs, build_waypoint_masks
from .train_cost import group_lengths


def contact_jacobian(fk, q, hand, local):
    """Analytic position Jacobian, with gradients through joint positions."""
    name = hand+"_wrist"
    ch, raw = fk._cast(q.dtype)[name], fk._chains[name]
    B, D = q.shape
    R = torch.eye(3, device=q.device, dtype=q.dtype).expand(B, 3, 3)
    p = q.new_zeros(B, 3)
    origins, axes = [], []
    for k, column in enumerate(raw["qidx"]):
        p = p+(R @ ch["p"][k, :, None]).squeeze(-1)
        R = R @ ch["R"][k]
        origins.append(p)
        axes.append((R @ ch["axis"][k, :, None]).squeeze(-1))
        angle = q[:, column, None, None]
        rotation = torch.eye(3, device=q.device, dtype=q.dtype)+torch.sin(angle)*ch["K"][k]+(1-torch.cos(angle))*ch["K2"][k]
        R = R @ rotation
    p = p+(R @ ch["p_tail"][:, None]).squeeze(-1)
    R = R @ ch["R_tail"]
    point = p+(R @ local.unsqueeze(-1)).squeeze(-1)
    cols = [q.new_zeros(B, 3) for _ in range(D)]
    for j, column in enumerate(raw["qidx"]):
        cols[column] = torch.cross(axes[j], point-origins[j], dim=-1)
    return point, torch.stack(cols, -1)


def rebound_velocity(velocity, normal_jacobian, inverse_mass, efficiency, contacts, *, check_errors=True):
    """Frictionless normal impulse; energy retention eta means restitution sqrt(eta).

    Simultaneous contacts are solved together. Misses and separating contacts
    receive no impulse. inverse_mass uses the fixed nominal robot inertia.
    """
    incoming = torch.einsum("bhd,bd->bh", normal_jacobian, velocity)
    live = contacts & (incoming < -1e-5)
    J = normal_jacobian*live[..., None]
    minv_j = torch.einsum("ij,bhj->bih", inverse_mass, J)
    gram = J @ minv_j
    gram = gram+torch.diag_embed((~live).to(gram.dtype))+torch.eye(2, device=gram.device)*1e-7
    change = -(1+efficiency.clamp(0, 1).sqrt())*incoming*live
    impulse = (torch.linalg.solve(gram, change[..., None]) if check_errors else
               torch.linalg.solve_ex(gram, change[..., None], check_errors=False).result)
    result = velocity+(minv_j @ impulse).squeeze(-1)
    # Independent simultaneous restitution coefficients can inject energy
    # through coupled constraints. Bound total kinetic energy in that case.
    mass = (torch.linalg.inv(inverse_mass) if check_errors else
            torch.linalg.inv_ex(inverse_mass, check_errors=False).inverse)
    before = torch.einsum("bi,ij,bj->b", velocity, mass, velocity)
    after = torch.einsum("bi,ij,bj->b", result, mass, result)
    scale = (before/after.clamp_min(1e-10)).clamp_max(1).sqrt()
    return result*scale[:, None]


def hermite(q0, v0, q1, v1, duration, fraction):
    """Position, velocity and acceleration; fraction is (..., samples)."""
    u = fraction[..., None]
    T = duration[..., None, None].clamp_min(.001)
    a, b, c, d = [x.unsqueeze(-2) for x in (q0, v0, q1, v1)]
    q = (2*u**3-3*u**2+1)*a+(u**3-2*u**2+u)*T*b+(-2*u**3+3*u**2)*c+(u**3-u**2)*T*d
    v = ((6*u**2-6*u)*a+(-6*u**2+6*u)*c)/T+(3*u**2-4*u+1)*b+(3*u**2-2*u)*d
    acceleration = ((12*u-6)*a+(-12*u+6)*c)/T.square()+(6*u-4)*b/T+(6*u-2)*d/T
    return q, v, acceleration


class DecoderHitPoseGenerator:
    """One frozen decoder pass, returning the full joint pose at the next hit.

    The other arm can already move toward its next goal. Future assigned hits
    remain encoder context; output is a single pose and approach velocity.
    """
    def __init__(self, model, cost, kit, *, horizon=100, waist_upright_ease_wp=None):
        self.model = model.eval().requires_grad_(False)
        self.cost, self.kit, self.horizon = cost, kit, horizon
        self.waist_upright_ease_wp = waist_upright_ease_wp
        self.rows = [_strike_row(kit, d, sample_hit_point(kit, d), strike_velocity(kit, d, .5))
                     for d in range(kit.num_drums)]

    @torch.no_grad()
    def predict(self, q, v, drums, times, speeds, valid, group, *,
                return_plan=False, waist_yaw=None):
        """Return the next endpoint, or raw decoder path/context for calibration.

        An optional waist target permits controlled yaw-region measurements.
        Raw paths retain decoder masks. Inactive hands borrow the active
        horizon and remain under decoder control, as in deployment.
        """
        B, G, _ = drums.shape
        mc = self.model.cfg
        ctx = ContextInputs.zeros(B, num_drums=mc.num_drums, num_future=mc.num_future, device=q.device)
        n = self.kit.num_drums
        for field, values in (("drum_pos", self.kit.positions), ("drum_axis", self.kit.axes),
                              ("drum_radius", self.kit.radii), ("drum_height", self.kit.heights)):
            getattr(ctx, field)[:, :n] = q.new_tensor(values)
        ctx.drum_valid[:, :n] = True
        aim = q.new_tensor(np.asarray([r["p_star"] for r in self.rows]))+q.new_tensor(PELVIS_IN_ENV)
        axis = q.new_tensor(self.kit.axes)
        now = times.new_zeros(B) if group == 0 else times[:, group-1]
        ids = torch.arange(G, device=q.device)[None].expand(B, -1)
        for h in range(2):
            upcoming = (drums[..., h] >= 0) & valid & (ids >= group)
            indexes = torch.where(upcoming, ids, G).sort(1).values
            index = indexes[:, 0].clamp_max(G-1)
            batch = torch.arange(B, device=q.device)
            d = drums[batch, index, h].clamp_min(0)
            raw_tth = times[batch, index]-now
            live = (indexes[:, 0] < G) & (raw_tth <= self.horizon*.02+1e-6)
            tth = raw_tth.clamp(.02, self.horizon*.02)
            ctx.goal_drum[:, h] = d
            ctx.goal_valid[:, h] = live
            ctx.goal_tth[:, h] = tth*live
            ctx.goal_pos[:, h] = aim[d]*live[:, None]
            ctx.goal_vel[:, h] = -axis[d]*speeds[batch, index, h, None]*live[:, None]
            for j in range(mc.num_future):
                if j >= G:
                    break
                slot = (live.long()+j).clamp_max(G-1)
                candidate = indexes.gather(1, slot[:, None])[:, 0]
                f = candidate.clamp_max(G-1)
                ok = (candidate < G) & (live.long()+j < G)
                fd = drums[batch, f, h].clamp_min(0)
                ctx.future_drum[:, h, j] = fd
                ctx.future_valid[:, h, j] = ok
                ctx.future_pos[:, h, j] = aim[fd]*ok[:, None]
                ctx.future_vel[:, h, j] = -axis[fd]*speeds[batch, f, h, None]*ok[:, None]
                anchor = torch.where(live, times[batch, index], now)
                ctx.future_gap[:, h, j] = (times[batch, f]-anchor)*ok
        # Use the same analytic waist aim as playback, including future votes.
        from isaac_rl.schedule import BatchedSchedule
        schedule = SimpleNamespace(B=B, device=q.device, dtype=q.dtype,
                                   _pelvis=q.new_tensor(PELVIS_IN_ENV))
        BatchedSchedule._fill_aim(schedule, ctx, q[:, 0], ctx.goal_pos, ctx.goal_valid, None)
        if waist_yaw is not None:
            ctx.aim_yaw[:] = torch.as_tensor(waist_yaw, device=q.device,
                                           dtype=q.dtype).reshape(B)
            ctx.aim_valid[:] = True
        masks = build_waypoint_masks(ctx.goal_tth[:, 0], ctx.goal_tth[:, 1], max_horizon=self.horizon,
            dt=.02, active_left=ctx.goal_valid[:, 0], active_right=ctx.goal_valid[:, 1])
        state = torch.cat((q, v), -1)
        seed = self.model.initial_trajectory(ctx, state, masks)
        encoding = self.model.encode_context(ctx)
        if encoding.waist_pr_sp is not None:
            from .warmstart import seed_waist_pr_to_sp
            seed = seed_waist_pr_to_sp(seed, q, v, encoding.waist_pr_sp,
                dt=.02, ease_wp=self.waist_upright_ease_wp)
        out = self.model.refine(seed, state, encoding, masks, iterations=1)
        traj = out.traj.float()
        lengths = self.cost.spline.lengths_from_mask(masks.elem_mask).to(q.dtype)
        spline = self.cost.spline.build(q, traj, v, lengths)
        k = ((times[:, group]-now)/.02).round().long().clamp(1, self.horizon)-1
        batch = torch.arange(B, device=q.device)
        pose = traj[batch, k]
        coefficient = spline.coeffs[batch, k]
        velocity = coefficient[..., 1]+.04*coefficient[..., 2]+.0012*coefficient[..., 3]
        # Elements outside their effective decoder horizon hold their measured
        # pose. Inactive hands borrow a horizon and still have live elements.
        moving = masks.elem_mask[batch, k].bool()
        pose = torch.where(moving, pose, q)
        velocity = torch.where(moving, velocity, torch.zeros_like(v))
        live = valid[:, group]
        pose = torch.where(live[:, None], pose, q)
        velocity = torch.where(live[:, None], velocity, v)
        if return_plan:
            return dict(pose=pose, velocity=velocity, traj=traj, context=ctx,
                        masks=masks, lengths=lengths)
        return pose, velocity


class SingleHitGenerator:
    """Direct single-pose MLP, using the checkpoint's pretrained terminal head.

    Only the next global hit group is active (two arms for a simultaneous hit).
    No encoder/trajectory decoder is called. The physics surrogate constructs
    an approach velocity from the contact Jacobian and requested impact speed.
    """
    def __init__(self, model, cost, kit, *, horizon=100, waist_upright_ease_wp=None):
        if model.terminal_initializer is None:
            raise ValueError("single-pose generation needs a pretrained terminal head")
        self.initializer = model.terminal_initializer.eval().requires_grad_(False)
        self.cfg, self.kit, self.horizon = model.cfg, kit, horizon
        self.surrogate = TrajectorySurrogate(cost, kit)

    @torch.no_grad()
    def predict(self, q, v, drums, times, speeds, valid, group):
        B = len(q)
        ctx = ContextInputs.zeros(B, num_drums=self.cfg.num_drums,
                                 num_future=self.cfg.num_future, device=q.device)
        n = self.kit.num_drums
        for field, values in (("drum_pos", self.kit.positions), ("drum_axis", self.kit.axes),
                              ("drum_radius", self.kit.radii), ("drum_height", self.kit.heights)):
            getattr(ctx, field)[:, :n] = q.new_tensor(values)
        ctx.drum_valid[:, :n] = True
        active = (drums[:, group] >= 0) & valid[:, group, None]
        now = times.new_zeros(B) if group == 0 else times[:, group-1]
        tth = (times[:, group]-now).clamp_min(.02)
        # The MLP predicts the one deadline pose; all requested hits at that
        # deadline share its timestamp. Inactive arms retain their pose.
        ctx.goal_drum = drums[:, group].clamp_min(0)
        ctx.goal_valid = active
        ctx.goal_tth = tth[:, None]*active
        masks = build_waypoint_masks(ctx.goal_tth[:, 0], ctx.goal_tth[:, 1],
            max_horizon=self.horizon, dt=.02, active_left=active[:, 0], active_right=active[:, 1])
        pose = self.initializer.terminal_pose(ctx, torch.cat((q, v), -1), masks)
        for h, hand in enumerate(("left", "right")):
            cols = [j for j, name in enumerate(G1_UPPER_BODY_JOINTS) if name.startswith(hand+"_")]
            pose[:, cols] = torch.where(active[:, h, None], pose[:, cols], q[:, cols])
        _, J = self.surrogate.contact(pose[:, None], drums[:, group:group+1])
        approach = self.surrogate.approach_velocity(J[:, 0], speeds[:, group], active)
        live = valid[:, group, None]
        return torch.where(live, pose, q), torch.where(live, approach, v)


class TrajectorySurrogate:
    """Perfect tracking of pose connectors, geometric hit credit and random rebound.

    Hermite connectors preserve the generated endpoint velocities. They are
    surrogate paths, not the decoder's intermediate trajectory or Isaac paths.
    """
    def __init__(self, cost, kit, *, collocation_dt=.01):
        self.cost, self.kit = cost, kit
        self.device = cost.device
        self.collocation_dt = collocation_dt
        self.rows = [_strike_row(kit, d, sample_hit_point(kit, d), strike_velocity(kit, d, .5))
                     for d in range(kit.num_drums)]
        self.geometry = {k: torch.as_tensor(np.asarray([row[k] for row in self.rows]),
                                          device=self.device, dtype=torch.float32) for k in self.rows[0]}
        local = []
        for hand in ("left", "right"):
            local.append([np.stack([stick_contact_offset(hand, sigma) for sigma in
                (shaft_hit_sigma_range(name) or (float(self.rows[d]["sigma"]),)*2)])
                for d, name in enumerate(kit.names)])
        self.contact_local = torch.as_tensor(np.asarray(local), device=self.device, dtype=torch.float32)
        ready = torch.tensor(ready_joint_pos(), device=self.device, dtype=torch.float32)
        basis = torch.eye(N_DOF, device=self.device)
        mass = cost.rnea.mass_matrix_torque(ready[None].expand(N_DOF, -1), basis).T
        self.inverse_mass = torch.linalg.inv((mass+mass.T)/2+torch.eye(N_DOF, device=self.device)*1e-5)
        self.torque_limits = ready.new_tensor(isaac_effort_limits())
        self.velocity_limits = ready.new_tensor(isaac_velocity_limits())
        lo, hi = urdf_position_limits(posture_limits=False)
        self.lower, self.upper = ready.new_tensor(lo), ready.new_tensor(hi)

    def contact(self, poses, drums):
        B, G, _ = drums.shape
        q = poses.reshape(-1, N_DOF)
        gaps, normal_rows = [], []
        for hand_index, hand in enumerate(("left", "right")):
            d = drums[..., hand_index].reshape(-1).clamp_min(0)
            a_local = self.contact_local[hand_index, d, 0]
            b_local = self.contact_local[hand_index, d, 1]
            a, _ = contact_jacobian(self.cost.fk, q, hand, a_local)
            b, _ = contact_jacobian(self.cost.fk, q, hand, b_local)
            geom = self.geometry
            dist, fraction, _point = segment_hit_region(a, b, geom["p_star"][d], geom["n_out"][d],
                geom["hit_radius"][d], geom["disk_off"][d], geom["hp_dir"][d], geom["has_hp"][d], *strike_axial_limits())
            selected = a_local+fraction.detach()[..., None]*(b_local-a_local)
            _p, jac = contact_jacobian(self.cost.fk, q, hand, selected)
            normal_rows.append(torch.einsum("bi,bij->bj", geom["n_out"][d], jac).reshape(B, G, N_DOF))
            gaps.append(dist.reshape(B, G))
        return torch.stack(gaps, -1), torch.stack(normal_rows, -2)

    def approach_velocity(self, J, speeds, active):
        """Minimum kinetic-energy joint velocity for the requested normal speed.

        A small regularizer handles simultaneous Jacobians near singularity;
        hard velocity clipping can lower impact speed and is reflected in credit.
        """
        masked = J*active[..., None]
        minv_j = torch.einsum("ij,...hj->...ih", self.inverse_mass, masked)
        gram = masked @ minv_j+torch.eye(2, device=J.device)*.002
        velocity = (minv_j @ torch.linalg.solve(gram, (-speeds*active)[..., None])).squeeze(-1)
        return velocity.clamp(-self.velocity_limits, self.velocity_limits)

    def evaluate(self, q0, v0, poses, velocity_residual, drums, times, speeds, valid, efficiency,
                 *, return_path=False, pre_velocity_override=None, return_segments=False):
        B, G, _ = drums.shape
        active = (drums >= 0) & valid[..., None]
        gap, J = self.contact(poses, drums)
        baseline = self.approach_velocity(J, speeds, active)
        pre_velocity = ((baseline+velocity_residual).clamp(-self.velocity_limits, self.velocity_limits)
                        if pre_velocity_override is None else pre_velocity_override)
        normal_speed = torch.einsum("bghd,bgd->bgh", J, pre_velocity)
        hit = active & (gap <= 1e-5) & (normal_speed < -.1)
        post_velocity = rebound_velocity(pre_velocity.reshape(-1, N_DOF), J.reshape(-1, 2, N_DOF),
            self.inverse_mass, efficiency.reshape(-1, 2), hit.reshape(-1, 2)).reshape(B, G, N_DOF)
        start_q = torch.cat((q0[:, None], poses[:, :-1]), 1)
        start_v = torch.cat((v0[:, None], post_velocity[:, :-1]), 1)
        starts = torch.cat((times.new_zeros(B, 1), times[:, :-1]), 1)
        duration = (times-starts).clamp_min(.001)
        # Packed physical collocations: every interval is sampled at <=10 ms,
        # including its endpoint; no samples are spent on padded hit slots.
        lengths = torch.ceil(duration.detach()/self.collocation_dt).long()*valid
        batch_ids, group_ids, fractions = [], [], []
        for b, row in enumerate(lengths.detach().cpu().tolist()):
            for g, n in enumerate(row):
                if n:
                    batch_ids.extend([b]*(n+1)); group_ids.extend([g]*(n+1))
                    fractions.extend(np.linspace(0, 1, n+1).tolist())
        bi = torch.tensor(batch_ids, device=poses.device)
        gi = torch.tensor(group_ids, device=poses.device)
        u = poses.new_tensor(fractions)
        Q, V, A = hermite(start_q[bi, gi], start_v[bi, gi], poses[bi, gi], pre_velocity[bi, gi],
                          duration[bi, gi], u[:, None])
        Q, V, A = Q[:, 0], V[:, 0], A[:, 0]
        target = drums[bi, gi]
        # Each arm's last contact can precede the previous global event.
        last_drum = drums.new_full((B, 2), -1)
        last_time = times.new_full((B, 2), -1e6)
        previous, previous_times = [], []
        for g in range(G):
            previous.append(last_drum); previous_times.append(last_time)
            last_drum = torch.where(hit[:, g], drums[:, g], last_drum)
            last_time = torch.where(hit[:, g], times[:, g, None], last_time)
        previous = torch.stack(previous, 1)[bi, gi]
        elapsed = duration[bi, gi]*u
        since_contact = starts[bi, gi, None]+elapsed[:, None]-torch.stack(previous_times, 1)[bi, gi]
        previous = torch.where(since_contact <= .08, previous, -1)
        remaining_ticks = torch.ceil(duration[bi, gi]*(1-u)/.02).long()
        ctx = SimpleNamespace(drum_head_centers=None, drum_radii=None, stick_points_local=None,
            target_drum=target, start_drum=previous, hit_index=remaining_ticks[:, None].expand(-1, 2))
        raw, violation, diag = {}, {}, {}
        self.cost._collision(self.cost.fk.forward(Q), len(Q), 1, ctx,
            {h: Q.new_ones(len(Q), 1) for h in ("left", "right")}, raw, violation, {}, diag, {}, None)
        torque = self.cost.rnea(Q, V, A)
        tau_excess = (torque.abs()/self.torque_limits-1).clamp_min(0).amax(-1)
        vel_excess = (V.abs()/self.velocity_limits-1).clamp_min(0).amax(-1)
        box_excess = torch.maximum((self.lower-Q).clamp_min(0), (Q-self.upper).clamp_min(0)).amax(-1)
        collision = torch.maximum(violation["collision"], (-diag["collision_stick_min_sdf"]).clamp_min(0))
        def mean(values):
            total = values.new_zeros(B).scatter_add(0, bi, values)
            counts = values.new_zeros(B).scatter_add(0, bi, values.new_ones(len(values)))
            return total/counts.clamp_min(1)
        def peak(values):
            return values.new_zeros(B).scatter_reduce(0, bi, values, reduce="amax", include_self=True)
        requested = active.sum((1, 2)).clamp_min(1)
        terminal = (gap.square()*active).sum((1, 2))/requested/.05**2
        speed_loss = ((normal_speed+speeds).square()*active).sum((1, 2))/requested
        collision_loss = sum(self.cost.cfg.__dict__["w_collision_"+name]*mean(raw["collision_"+name])
                             for name in ("drum", "under", "self"))
        physics_loss = (100*terminal+speed_loss+collision_loss+20*mean(tau_excess.square())
                        +20*mean(vel_excess.square())+100*mean(box_excess.square()))
        soft_hit = torch.exp(-(gap/.005).square())*torch.sigmoid(-5*(normal_speed+.1))*active
        physics_loss = physics_loss-10*soft_hit.sum((1, 2))/requested
        hits = hit.sum((1, 2))
        score = (10*hits-5*(requested-hits)-10*peak(tau_excess).clamp_max(10)
                 -10*peak(vel_excess).clamp_max(10)-10*(peak(collision)/.01).clamp_max(10)
                 -10*(peak(box_excess)/.05).clamp_max(10)-terminal.clamp_max(100))
        result = dict(loss=physics_loss, score=score, hits=hits, requested=requested,
            hit_mask=hit, gap=gap, normal_speed=normal_speed, pre_velocity=pre_velocity,
            post_velocity=post_velocity, torque_excess=peak(tau_excess),
            velocity_excess=peak(vel_excess), collision_m=peak(collision),
            joint_excess_rad=peak(box_excess), terminal_loss=terminal,
            collocations=len(Q))
        if return_segments:
            # Event-interval rewards for the finite-schedule Bellman equation.
            # Reuse exactly the same physical samples and departure exemptions
            # as the complete rollout, rather than re-scoring isolated segments.
            segment_ids = bi*G+gi
            def segment_peak(values):
                return values.new_zeros(B*G).scatter_reduce(
                    0, segment_ids, values, reduce="amax", include_self=True).reshape(B, G)
            requested_g = active.sum(-1)
            hits_g = hit.sum(-1)
            terminal_g = (gap.square()*active).sum(-1)/requested_g.clamp_min(1)/.05**2
            segment_score = (10*hits_g-5*(requested_g-hits_g)
                -10*segment_peak(tau_excess).clamp_max(10)
                -10*segment_peak(vel_excess).clamp_max(10)
                -10*(segment_peak(collision)/.01).clamp_max(10)
                -10*(segment_peak(box_excess)/.05).clamp_max(10)
                -terminal_g.clamp_max(100))*valid
            result.update(segment_score=segment_score, segment_hits=hits_g,
                          segment_requested=requested_g)
        if return_path:
            result.update(path_q=Q, path_velocity=V, path_acceleration=A, path_batch=bi,
                          path_time=starts[bi, gi]+elapsed)
        return result

    @torch.no_grad()
    def rollout(self, generator, packed, *, return_path=False, return_segments=False):
        q, v = packed["q0"].clone(), packed["v0"].clone()
        poses, velocities = [], []
        for group in range(packed["drums"].shape[1]):
            pose, pre = generator.predict(q, v, packed["drums"], packed["times"],
                                          packed["speeds"], packed["valid"], group)
            gap, J = self.contact(pose[:, None], packed["drums"][:, group:group+1])
            approach = torch.einsum("bhd,bd->bh", J[:, 0], pre)
            hit = ((packed["drums"][:, group] >= 0) & packed["valid"][:, group, None]
                   & (gap[:, 0] <= 1e-5) & (approach < -.1))
            post = rebound_velocity(pre, J[:, 0], self.inverse_mass,
                                    packed["efficiency"][:, group], hit)
            poses.append(pose); velocities.append(pre)
            q, v = pose, post
        poses, velocities = torch.stack(poses, 1), torch.stack(velocities, 1)
        result = self.evaluate(packed["q0"], packed["v0"], poses, torch.zeros_like(velocities),
            packed["drums"], packed["times"], packed["speeds"], packed["valid"], packed["efficiency"],
            return_path=return_path, pre_velocity_override=velocities, return_segments=return_segments)
        result["poses"] = poses
        return result
