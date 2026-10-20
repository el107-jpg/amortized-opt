"""Finite hand-schedule rollouts on actual frozen decoder spline prefixes.

Future waypoints determine spline derivatives, so prefixes are sampled from
the complete original spline rather than rebuilt with an earlier boundary.
Both arms and the waist participate in all physics checks, including idle arms.
"""
from __future__ import annotations

import torch
import numpy as np

from amortized_opt.limits import available_torque_torch
from amortized_opt.trajectory_surrogate import TrajectorySurrogate, rebound_velocity
from isaac_rl.transition_calibration import strike_context


def physical_spline(cost, q, v, plan):
    return cost.spline.build(q, plan['traj'], v, plan['lengths'])


def time_statistics(flags, steps, dt):
    """Per-channel sampled duration and longest run on variable prefix grids."""
    flags = flags.bool()
    B, T, C = flags.shape
    index = torch.arange(T, device=flags.device)[None, :, None]
    weights = torch.full((B, T, 1), dt, device=flags.device)
    weights[:, 0] *= .5
    weights = torch.where(index == steps[:, None, None], weights*.5, weights)
    weights *= index <= steps[:, None, None]
    total = (flags*weights).sum(1)
    run = total.new_zeros(B, C); longest = run.clone()
    for t in range(T):
        run = torch.where(flags[:, t], run+weights[:, t], 0.)
        longest = torch.maximum(longest, run)
    return total, longest


@torch.no_grad()
def prefix_physics(cost, spline, v0, ctx, steps, *, max_steps=None):
    """Score only the executed prefix, retaining the original coefficients."""
    B = len(steps); N = int(steps.max()) if max_steps is None else max_steps
    D = spline.n_dof; ns = cost.cfg.n_sub
    # Sampling the complete spline leaves the derivatives of its prefix intact.
    samples = spline.subsample(ns)
    Q = samples.q[:, :N*ns+1]; S = Q.shape[1]
    sample_index = torch.arange(S, device=Q.device)[None]
    live = sample_index <= steps[:, None]*ns
    capture = {}; diag = {}
    cost._collision(cost.fk.forward(Q.reshape(-1, D)), B, S, ctx,
        {hand: live.to(Q.dtype) for hand in ('left', 'right')},
        {}, {}, {}, diag, {}, None, profile_capture=capture)
    signed = torch.cat(list(capture.values()), -1).amax(-1)
    collision = signed.amax(1).clamp_min(0)
    collision_time, _ = time_statistics((signed>1e-6)[..., None], steps*ns, spline.dt/ns)
    # Use the checkpoint's knot command convention for torque. At a replan
    # boundary, repeat the preceding acceleration instead of using an
    # unexecuted waypoint after the bounce.
    q = spline.x_held[:, :N+1]
    qd = torch.cat((v0[:, None], (q[:, 1:]-q[:, :-1])/spline.dt), 1)
    qdd = torch.cat(((qd[:, 1:]-qd[:, :-1])/spline.dt,
                     (qd[:, -1:]-qd[:, -2:-1])/spline.dt), 1)
    batch = torch.arange(B, device=q.device)
    qdd[batch, steps] = qdd[batch, steps-1]
    tau = cost.rnea.rnea(q.reshape(-1, D), qd.reshape(-1, D), qdd.reshape(-1, D)).reshape(B, N+1, D)
    available = available_torque_torch(qd, cost.tau_hard.view(1, 1, D), cost.v_hard.view(1, 1, D),
        eta=1., saturation_scale=cost.cfg.torque_saturation_scale, derate=cost.cfg.torque_derate)
    valid = torch.arange(N+1, device=q.device)[None, :, None] <= steps[:, None, None]
    ratio = torch.where(valid, tau.abs()/available.clamp_min(1e-9), 0.)
    over = (tau.abs()-available>1e-6)&valid
    total, longest = time_statistics(over, steps, spline.dt)
    duration = steps*spline.dt
    torque_fraction = (total/duration[:, None]).amax(-1)
    velocity = spline.segment_velocity_max_analytic()[:, :N]/cost.v_hard
    seg_live = torch.arange(N, device=q.device)[None, :, None] < steps[:, None, None]
    velocity_ratio = torch.where(seg_live, velocity, 0.).amax((1, 2))
    box = torch.maximum((cost.q_lo_hard-Q).clamp_min(0), (Q-cost.q_hi_hard).clamp_min(0))
    box = torch.where(live[..., None], box, 0.).amax((1, 2))
    peak_ratio = ratio.amax((1, 2)); longest = longest.amax(-1)
    finite = torch.isfinite(Q).all((1, 2))&torch.isfinite(tau).all((1, 2))
    strict = finite&(peak_ratio<=1.+1e-6)&(velocity_ratio<=1.+1e-6)&(box<=1e-6)&(collision<=1e-6)
    relaxed = finite&(peak_ratio<=1.1+1e-6)&(torque_fraction<=.1+1e-6)&(longest<=.04+1e-6)
    relaxed &= (velocity_ratio<=1.+1e-6)&(box<=1e-6)&(collision<=.005+1e-6)
    return dict(torque_peak_ratio=peak_ratio, torque_over_fraction=torque_fraction,
        torque_longest_run_s=longest, collision_m=collision,
        collision_over_fraction=collision_time[:, 0]/duration,
        velocity_peak_ratio=velocity_ratio, joint_excess_rad=box,
        strict_physics=strict, relaxed_physics=relaxed)


class DecoderScheduleEnvironment:
    def __init__(self, generator, *, chunk=64, cuda_graphs=False):
        self.generator = generator; self.cost = generator.cost; self.chunk = chunk
        self.contact = TrajectorySurrogate(self.cost, generator.kit)
        if self.cost.cfg.physics_mode != 'knot_fd':
            raise ValueError('This probe requires the checkpoint knot_fd torque convention')
        self._physics_replays = {}
        self.cuda_graphs = cuda_graphs
        self._rebound = lambda pre, J, eta, hit: rebound_velocity(
            pre, J, self.contact.inverse_mass, eta, hit)
        if cuda_graphs:
            from isaac_rl.cuda_replay import CudaReplay
            if self.cost.device.type != 'cuda':
                raise ValueError('CUDA replay requires a CUDA device')
            if generator.model.training or any(p.requires_grad for p in generator.model.parameters()):
                raise ValueError('CUDA replay requires a frozen evaluation model')
            generator.model.decoder.forward = CudaReplay(generator.model.decoder.forward)
            self.contact.contact = CudaReplay(self.contact.contact)
            # The regularized two-contact Gram matrix and nominal mass are
            # nonsingular by construction. Avoid solver status synchronization
            # inside capture; the normal path retains its solver error checks.
            self._rebound = CudaReplay(lambda pre, J, eta, hit: rebound_velocity(
                pre, J, self.contact.inverse_mass, eta, hit, check_errors=False))

    def _prefix_metrics(self, spline, v, ctx, steps):
        if not self.cuda_graphs:
            return prefix_physics(self.cost, spline, v, ctx, steps)
        import copy
        from amortized_opt.tspline import TrajectorySpline
        from isaac_rl.cuda_replay import CudaReplay
        N = int(steps.max())
        names = tuple(k for k,t in vars(ctx).items() if isinstance(t, torch.Tensor))
        key = (len(steps), N, names, spline.dt)
        replay = self._physics_replays.get(key)
        if replay is None:
            template = copy.copy(ctx)
            dt = spline.dt
            def evaluate(coeffs, x, m, mask, lengths, velocity, ticks, *fields):
                context = copy.copy(template)
                for name, value in zip(names, fields):
                    setattr(context, name, value)
                path = TrajectorySpline(coeffs, x, m, mask, lengths, dt)
                return prefix_physics(self.cost, path, velocity, context, ticks, max_steps=N)
            replay = CudaReplay(evaluate)
            self._physics_replays[key] = replay
        return replay(spline.coeffs, spline.x, spline.m, spline.seg_mask, spline.lengths,
                      v, steps, *(getattr(ctx, k) for k in names))

    @torch.no_grad()
    def rollout(self, packed, *, prune=None):
        q = packed['q0'].clone(); v = packed['v0'].clone()
        B, G, _ = packed['drums'].shape
        last_drum = packed['drums'].new_full((B, 2), -1)
        last_time = packed['times'].new_full((B, 2), -1e6)
        result = {}; poses = []; pre_velocities = []; post_velocities = []
        alive = torch.ones(B, device=q.device, dtype=torch.bool)
        evaluated = torch.zeros(B, device=q.device, dtype=torch.long)
        for g in range(G):
            ids = torch.where(packed['valid'][:, g]&alive)[0]
            if not len(ids):
                poses.append(q.clone()); pre_velocities.append(torch.zeros_like(q))
                post_velocities.append(v.clone())
                continue
            drums = packed['drums'][ids]; times = packed['times'][ids]
            now = times.new_zeros(len(ids)) if g==0 else times[:, g-1]
            steps = ((times[:, g]-now)/.02).round().long()
            if (steps<1).any() or (steps>self.generator.horizon).any():
                raise ValueError('Probe hit intervals must fit the decoder control grid')
            plan = self.generator.predict(q[ids], v[ids], drums, times,
                packed['speeds'][ids], packed['valid'][ids], g, return_plan=True)
            spline = physical_spline(self.cost, q[ids], v[ids], plan)
            pose = spline.x_held[torch.arange(len(ids), device=q.device), steps]
            pre = spline.knot_derivative(steps)
            torch.testing.assert_close(pose, plan['pose'], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(pre, plan['velocity'], atol=2e-4, rtol=2e-4)
            gap, J = self.contact.contact(pose[:, None], drums[:, g:g+1])
            normal = torch.einsum('bhd,bd->bh', J[:, 0], pre)
            active = drums[:, g]>=0
            hit = active&(gap[:, 0]<=1e-5)&(normal<-.1)
            next_target = torch.where(plan['context'].goal_valid, plan['context'].goal_drum, -1)
            ctx = strike_context(self.generator.kit, next_target.cpu().numpy(),
                last_drum[ids].cpu().numpy(), steps.cpu().numpy(), q.device,
                speeds=np.full((len(ids), 2), 2.))
            ctx.hit_index = (plan['context'].goal_tth/.02).round().long()
            ctx.target_top_entry_depth_m = .01
            ctx.start_contact_age_s = now[:, None]-last_time[ids]
            metrics = self._prefix_metrics(spline, v[ids], ctx, steps)
            post = self._rebound(pre, J[:, 0], packed['efficiency'][ids, g], hit)
            last_drum[ids] = torch.where(hit, drums[:, g], last_drum[ids])
            last_time[ids] = torch.where(hit, times[:, g, None], last_time[ids])
            q[ids] = pose; v[ids] = post
            requested = active.sum(-1); hits = hit.sum(-1)
            terminal = (gap[:, 0].square()*active).sum(-1)/requested.clamp_min(1)/.05**2
            # Same score scale as the previous analytic schedule evaluator.
            score = 10*hits-5*(requested-hits)-terminal.clamp_max(100)
            score -= 10*(metrics['torque_peak_ratio']-1).clamp(0, 10)
            score -= 10*(metrics['velocity_peak_ratio']-1).clamp(0, 10)
            score -= 10*(metrics['collision_m']/.01).clamp_max(10)
            score -= 10*(metrics['joint_excess_rad']/.05).clamp_max(10)
            if not torch.isfinite(score).all():
                raise FloatingPointError('Nonfinite schedule score cannot be safely bounded')
            fields = dict(metrics, score=score, hits=hits, requested=requested,
                          hit_mask=hit, gap=gap[:, 0], normal_speed=normal,
                          strict_valid_hits=hits*metrics['strict_physics'],
                          relaxed_valid_hits=hits*metrics['relaxed_physics'])
            for name, values in fields.items():
                if name not in result:
                    result[name] = values.new_zeros((B, G)+values.shape[1:])
                result[name][ids, g] = values
            evaluated[ids] += 1
            poses.append(q.clone()); pre_velocities.append(torch.zeros_like(q).index_copy(0, ids, pre))
            post_velocities.append(v.clone())
            if prune is not None:
                alive &= prune(g, result['score'], alive)
        result.update(poses=torch.stack(poses, 1), pre_velocity=torch.stack(pre_velocities, 1),
                      post_velocity=torch.stack(post_velocities, 1), evaluated_groups=evaluated)
        return result

    @torch.no_grad()
    def score(self, table, q, v, candidates, draws):
        B, C, G = candidates.shape; S = draws.shape[1]
        owners = torch.arange(B, device=q.device).repeat_interleave(C*S)
        actions = candidates[:, :, None].expand(-1, -1, S, -1).reshape(-1, G)
        eta = draws[:, None].expand(-1, C, -1, -1, -1).reshape(-1, draws.shape[-2], 2)
        parts = []
        for start in range(0, len(actions), self.chunk):
            ids = owners[start:start+self.chunk]; swap = actions[start:start+self.chunk, :, None].bool()
            packed = dict(q0=q[ids], v0=v[ids], times=table['times'][ids, :G], valid=table['valid'][ids, :G])
            for field in ('drums', 'speeds'):
                raw = table[field][ids, :G]
                packed[field] = torch.where(swap, raw.flip(-1), raw)
            packed['efficiency'] = torch.where(swap, eta[start:start+self.chunk, :G].flip(-1),
                                                eta[start:start+self.chunk, :G])
            parts.append(self.rollout(packed))
        return {name: torch.cat([p[name] for p in parts]).reshape(B, C, S, *parts[0][name].shape[1:])
                for name in parts[0]}

    @torch.no_grad()
    def search_pruned(self, table, q, v, candidates, draws, seed_indices):
        """Exact best schedule with conservative bounds on discarded rollouts.

        Each score term is at most 10 per requested hit. Both bounce draws
        must stay together: the objective is their mean, not each draw's score.
        Future assignments remain unchanged while a schedule is evaluated.
        """
        B, C, G = candidates.shape; S = draws.shape[1]
        if self.chunk < S:
            raise ValueError('Chunk must hold every bounce sample of one candidate')
        seed_indices = torch.as_tensor(seed_indices, device=q.device, dtype=torch.long)
        if seed_indices.shape != (B,) or ((seed_indices<0)|(seed_indices>=C)).any():
            raise ValueError('One valid incumbent candidate index required per window')
        batch = torch.arange(B, device=q.device)
        seed = self.score(table, q, v, candidates[batch, seed_indices][:, None], draws)
        winner = {k:t[:, 0].clone() for k,t in seed.items()}
        seed_values = seed['score'][:, 0].double().sum(-1).mean(-1)
        incumbent = seed_values.clone()
        values = q.new_full((B, C), -torch.inf, dtype=torch.float64)
        values[batch, seed_indices] = seed_values
        bounds = q.new_full((B, C), torch.inf, dtype=torch.float64)
        bounds[batch, seed_indices] = seed_values
        completed = torch.zeros(B, C, device=q.device, dtype=torch.bool)
        completed[batch, seed_indices] = True
        evaluated = torch.zeros(B, C, S, device=q.device, dtype=torch.long)
        evaluated[batch, seed_indices] = seed['evaluated_groups'][:, 0]
        # Nearby schedules are evaluated first so completed schedules can raise
        # the incumbent before later chunks. No predicted value is used as a bound.
        distance = (candidates!=candidates[batch, seed_indices][:, None]).sum(-1)
        pairs = torch.nonzero(~completed)
        pairs = pairs[torch.argsort(distance[pairs[:, 0], pairs[:, 1]], stable=True)]
        pairs_per_chunk = max(1, self.chunk//S)
        for start in range(0, len(pairs), pairs_per_chunk):
            pair = pairs[start:start+pairs_per_chunk]; owners = pair[:, 0]; indexes = pair[:, 1]
            K = len(pair); rows = owners.repeat_interleave(S)
            actions = candidates[owners, indexes].repeat_interleave(S, dim=0)
            swap = actions[..., None].bool()
            packed = dict(q0=q[rows], v0=v[rows], times=table['times'][rows, :G], valid=table['valid'][rows, :G])
            for field in ('drums', 'speeds'):
                raw = table[field][rows, :G]
                packed[field] = torch.where(swap, raw.flip(-1), raw)
            eta = draws[owners, :, :G].reshape(K*S, G, 2)
            packed['efficiency'] = torch.where(swap, eta.flip(-1), eta)
            requested = ((packed['drums']>=0)&packed['valid'][..., None]).sum(-1)
            def prune(g, scores, alive):
                partial = scores[:, :g+1].double().sum(-1).reshape(K, S).mean(-1)
                remaining = requested[:, g+1:].sum(-1).reshape(K, S).double().mean(-1)
                upper = partial+10*remaining
                active = alive.reshape(K, S).all(-1)
                finished = active&(remaining==0)
                incumbent.scatter_reduce_(0, owners[finished], partial[finished], reduce='amax', include_self=True)
                # Float64 accumulation and an outward numerical margin prevent
                # boundary rounding from pruning equal or nearly equal winners.
                keep = (remaining==0)|(upper+1e-4*max(1,G)>=incumbent[owners])
                discarded = active&~keep
                bounds[owners[discarded], indexes[discarded]] = upper[discarded]
                return keep.repeat_interleave(S)
            rollout = self.rollout(packed, prune=prune)
            count = rollout['evaluated_groups'].reshape(K, S)
            done = (count==packed['valid'].sum(-1).reshape(K, S)).all(-1)
            evaluated[owners, indexes] = count
            completed[owners, indexes] = done
            total = rollout['score'].double().sum(-1).reshape(K, S).mean(-1)
            values[owners[done], indexes[done]] = total[done]
            bounds[owners[done], indexes[done]] = total[done]
            # Keep full diagnostics for the best completed candidate per window.
            # Ties choose the same lowest bank index as exhaustive argmax.
            for owner in owners[done].unique():
                owner = int(owner)
                best = int(values[owner].argmax())
                take = torch.where(done&(owners==owner)&(indexes==best))[0]
                if len(take):
                    j = int(take[0])
                    for name,t in rollout.items():
                        winner[name][owner] = t.reshape(K, S, *t.shape[1:])[j]
        best = values.argmax(-1)
        expected = table['valid'][:, :G].sum(-1)[:, None, None].expand(B, C, S)
        return dict(best_index=best, best_value=values[batch, best], winner=winner,
            completed=completed, values=values, upper_bounds=bounds,
            evaluated_groups=evaluated, full_evaluated_groups=expected,
            saved_group_evaluations=(expected-evaluated).sum())

    @torch.no_grad()
    def search(self, table, q, v, candidates, draws, seed_indices):
        """Use pruning for large banks; retain efficient single-batch search.

        The paired benchmark shows that seeding a separate incumbent doubles
        latency for a 16-candidate window that already fits one GPU batch.
        """
        B,C,G=candidates.shape; S=draws.shape[1]
        if B*C*S>self.chunk:
            result=self.search_pruned(table,q,v,candidates,draws,seed_indices)
            result['strategy']='early_pruning'
            return result
        full=self.score(table,q,v,candidates,draws)
        values=full['score'].double().sum(-1).mean(-1)
        best=values.argmax(-1); batch=torch.arange(B,device=q.device)
        return dict(best_index=best,best_value=values[batch,best],
            winner={k:t[batch,best] for k,t in full.items()},
            completed=torch.ones(B,C,device=q.device,dtype=torch.bool),
            values=values,upper_bounds=values,
            evaluated_groups=full['evaluated_groups'],full_evaluated_groups=full['evaluated_groups'],
            saved_group_evaluations=values.new_zeros((),dtype=torch.long),
            strategy='exhaustive_single_batch')
