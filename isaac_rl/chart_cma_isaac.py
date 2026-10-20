"""Matched batched Isaac episodes for static conditional-chart candidates."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import torch

from amortized_opt.limits import available_torque_torch
from amortized_opt.train_cost import group_lengths
from isaac_rl.assignment_parallel import ParallelCommittedMPC, take_rows
from isaac_rl.chart_cma_reward import NearMissTracker


def prefirst_snapshot_ticks(first_hit_s,dt,*,fractions=(.25,.5,.75),guard_s=.10):
    """Sparse control-grid states with all hits still ahead of the snapshot."""
    if first_hit_s<=guard_s or dt<=0: return ()
    ticks={int(round(float(first_hit_s)*fraction/dt)) for fraction in fractions}
    return tuple(sorted(tick for tick in ticks if tick>=1 and tick*dt<=first_hit_s-guard_s+1e-9))


class ChunkedChartMPC(ParallelCommittedMPC):
    """Keep the existing controller, bounding decoder/cost batch memory."""
    def __init__(self, *args, decode_chunk=128, plan_diagnostics=True, **kwargs):
        super().__init__(*args, **kwargs)
        if decode_chunk < 1:
            raise ValueError("decode_chunk must be positive")
        self.decode_chunk = decode_chunk
        self.plan_diagnostics = plan_diagnostics

    @torch.no_grad()
    def plan(self, q, qd):
        hit_index = self.sched.hit_index()
        key = self._encode_key(hit_index)
        changed = (key != self.goal_keys).any(-1)
        live = changed & (hit_index >= 1).any(-1)
        self.new_plans = live
        self.collision.zero_()
        idle = changed & ~live
        self.reference[idle] = q[idle, None].expand(-1, self.H, -1)
        rows = live.nonzero(as_tuple=True)[0]
        if rows.numel():
            old_yaw = self.yaw_bc.clone()
            encoding, _ = self._encoding(hit_index, q)
            self.yaw_bc[~live] = old_yaw[~live]
            masks = self.masks(hit_index)
            strike = self.sched.strike_context() if self.plan_diagnostics else None
            for offset in range(0, len(rows), self.decode_chunk):
                ids = rows[offset:offset+self.decode_chunk]
                one_masks = take_rows(masks, ids)
                state = torch.cat((q[ids], qd[ids]), -1)
                seed = self.model.initial_trajectory(take_rows(self._ctx, ids), state, one_masks)
                with torch.autocast(device_type=self.device.type, dtype=self.cfg.amp_dtype,
                                    enabled=self.device.type == "cuda" and self.cfg.amp_dtype is not None):
                    output = self.model.refine(seed*one_masks.elem_mask, state,
                        take_rows(encoding, ids), one_masks, iterations=1)
                candidate = self._hold_plan_end(output.traj.to(self.dtype), q[ids], hit_index[ids])
                if self.plan_diagnostics:
                    context = take_rows(strike, ids)
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        score = self.descent_cost(q[ids], candidate, qd[ids], context,
                            lengths=group_lengths(context.hit_index).to(q.dtype), elem_mask=one_masks.elem_mask)
                    self.collision[ids] = score.violations["collision"]
                finite = torch.isfinite(candidate).flatten(1).all(1)
                candidate = torch.where(finite[:, None, None], candidate,
                                        q[ids, None].expand(-1, self.H, -1))
                self.reference.index_copy_(0, ids, candidate)
                self.decoder_passes[ids] += 1
        self.goal_keys.copy_(key)
        self.last_ref0 = self.reference[:, 0].clone()
        self.n_ticks += 1
        return self._command(self.reference, hit_index), hit_index


class IsaacChartEvaluator:
    """Reuse a scene in waves; each wave installs all selected hands explicitly.

    Contacts/hit credit are the original task's. Collision fitness uses measured
    joint-state geometry, rather than the player's uninstrumented counter.
    Torque fitness is positive *demand* excess, accounting for actual velocity.
    """
    def __init__(self, env, mpc, pid, *, seed=0, torque_weight=.01,
                 joint_weight=.1, collision_weight=.05, collision_stride=5,
                 near_miss_weight=0., near_miss_scale_m=.05, near_miss_window_s=.08,
                 collision_chunk=None,rollout_audit=False,clear_reset_contacts=True,match_decoder_speed=True,
                 video_path=None,video_camera='3quarter'):
        if min(torque_weight, joint_weight, collision_weight) < 0 or collision_stride < 1:
            raise ValueError("Invalid fitness weights or collision cadence")
        self.env, self.u, self.mpc, self.pid = env, env.unwrapped, mpc, pid
        self.seed = seed
        self.rollout_audit=bool(rollout_audit)
        self.clear_reset_contacts=bool(clear_reset_contacts)
        self.match_decoder_speed=bool(match_decoder_speed)
        self.video_path=video_path
        self.video_camera=video_camera
        self.torque_weight = torque_weight
        self.joint_weight = joint_weight
        self.collision_weight = collision_weight
        self.collision_stride = collision_stride
        self.collision_chunk = collision_chunk or mpc.decode_chunk
        if self.collision_chunk < 1:
            raise ValueError('Collision batch size must be positive')
        if (not np.isfinite([near_miss_weight, near_miss_scale_m, near_miss_window_s]).all()
                or not 0 <= near_miss_weight < 1 or min(near_miss_scale_m, near_miss_window_s) <= 0):
            raise ValueError('Invalid near-miss reward configuration')
        self.near_miss_weight = near_miss_weight
        self.near_miss_scale_m = near_miss_scale_m
        self.near_miss_window_s = near_miss_window_s

    def _install(self, clips):
        from isaac_rl.arm_assignment import AssignmentQueue, AssignmentConfig
        from amortized_opt.rhythm import strike_speed
        u = self.u
        u.beat_target_speed_m_s=torch.zeros_like(u.beat_velocity) if self.match_decoder_speed else None
        u.beat_valid.zero_(); u.beat_time.fill_(1e6)
        u.beat_hand.zero_(); u.beat_drum_id.zero_()
        for b, clip in enumerate(clips):
            hits = AssignmentQueue(clip, u.assignment_kit, AssignmentConfig()).hits
            n = len(hits)
            if n > u.beat_valid.shape[1]:
                raise ValueError("Rhythm exceeds allocated Isaac beat slots")
            drums = torch.tensor([h.drum for h in hits], device=u.device, dtype=torch.long)
            u.beat_time[b, :n] = torch.tensor([h.time for h in hits], device=u.device)
            u.beat_velocity[b, :n] = torch.tensor([h.velocity for h in hits], device=u.device)
            if self.match_decoder_speed:
                u.beat_target_speed_m_s[b,:n]=u.beat_velocity.new_tensor([strike_speed(h.velocity,
                    speed_min=self.mpc.sched.cfg.speed_min,speed_max=self.mpc.sched.cfg.speed_max) for h in hits])
            u.beat_hand[b, :n] = torch.tensor([h.hand == "right" for h in hits], device=u.device)
            u.beat_drum_id[b, :n] = drums
            u.beat_pos[b, :n] = u.drum_pos_env[b, drums]
            u.beat_drum_height[b, :n] = u.drum_height[drums]
            u.beat_dipole_offset[b, :n] = u.drum_dipole_offset[drums]
            u.beat_valid[b, :n] = True
        u.beat_is_left.copy_((u.beat_hand == 0) & u.beat_valid)
        u.beat_is_right.copy_((u.beat_hand == 1) & u.beat_valid)
        u._field_cache_handoff = False
        u._field_cache_stale = torch.zeros(u.num_envs, device=u.device)

    @torch.no_grad()
    def _collision(self, q):
        cost = self.mpc.descent_cost
        context = self.mpc.sched.strike_context()
        out = q.new_zeros(len(q))
        for offset in range(0, len(q), self.collision_chunk):
            ids = torch.arange(offset, min(offset+self.collision_chunk, len(q)), device=q.device)
            ctx = take_rows(context, ids)
            diag = {}
            cost._collision(cost.fk.forward(q[ids]), len(ids), 1, ctx,
                {h: q.new_ones(len(ids), 1) for h in ("left", "right")},
                {}, {}, {}, diag, {}, None)
            sdf = torch.minimum(diag["collision_min_sdf"], diag["collision_stick_min_sdf"])
            out[ids] = (-sdf).clamp_min(0)
        return out

    def _near_miss_tracker(self, clips):
        from amortized_opt.rhythm import strike_region
        u = self.u
        # Avoid processing the long seed rhythm's padded beat capacity per tick.
        count = max(len(c.hits) for c in clips)
        kit = u.assignment_kit
        geometry = [strike_region(kit, d, frame='env') for d in range(len(kit.names))]
        offsets = np.stack([g[1]-kit.positions[d] for d, g in enumerate(geometry)])
        halves = np.stack([np.zeros(3) if g[3] is None else g[3] for g in geometry])
        drums = u.beat_drum_id[:, :count]
        to_tensor = lambda value: torch.as_tensor(value, device=u.device, dtype=u.beat_pos.dtype)
        centers = u.drum_pos_env.gather(1, drums[..., None].expand(-1, -1, 3))+to_tensor(offsets)[drums]
        return NearMissTracker(u.beat_time[:, :count], u.beat_hand[:, :count],
            u.beat_valid[:, :count], centers, to_tensor(kit.axes)[drums],
            to_tensor([g[2] for g in geometry])[drums], to_tensor(halves)[drums],
            weight=self.near_miss_weight, scale_m=self.near_miss_scale_m,
            window_s=self.near_miss_window_s)

    def _reset_from_states(self, states):
        """Install positions/velocities inside reset before simulator FK refresh."""
        u = self.u
        if not hasattr(u, '_chart_nominal_ready'):
            u._chart_nominal_ready = u.ready_joint_pos.clone()
        u.ready_joint_pos.copy_(u._chart_nominal_ready)
        if states is None:
            self.env.reset()
            if self.clear_reset_contacts:
                from isaac_rl.rollout_reproducibility import clear_reset_contact_buffers
                self._reset_contact_peak=clear_reset_contact_buffers(u)
            return
        q = torch.as_tensor([s['q'] for s in states], device=u.device, dtype=u.ready_joint_pos.dtype)
        qd = torch.as_tensor([s['qd'] for s in states], device=u.device, dtype=q.dtype)
        if q.shape != (u.num_envs, len(self.pid.ids)) or qd.shape != q.shape or not torch.isfinite(q).all() or not torch.isfinite(qd).all():
            raise ValueError('Invalid physical initial q/qd')
        u.ready_joint_pos[:, self.pid.ids] = q
        original_reset = u._reset_idx
        def reset_with_velocity(env_ids):
            original_reset(env_ids)
            ids = torch.arange(u.num_envs, device=u.device) if env_ids is None else env_ids
            positions = u.robot.data.joint_pos[ids].clone()
            velocities = u.robot.data.joint_vel[ids].clone()
            velocities[:, self.pid.ids] = qd[ids]
            u.robot.write_joint_state_to_sim(positions, velocities, None, ids)
        u._reset_idx = reset_with_velocity
        try:
            self.env.reset()  # Original trackers/servo reset, then sim.forward().
        finally:
            u._reset_idx = original_reset
        if self.clear_reset_contacts:
            from isaac_rl.rollout_reproducibility import clear_reset_contact_buffers
            self._reset_contact_peak=clear_reset_contact_buffers(u)

    @torch.inference_mode()
    def capacity_probe(self, clips, ticks=10):
        """Exercise every physical lane without producing training fitness."""
        import time
        if len(clips) != self.u.num_envs or ticks < 1:
            raise ValueError('Capacity probe needs one clip per physical environment')
        torch.manual_seed(self.seed); np.random.seed(self.seed); self.u.seed(self.seed)
        self.env.reset(); self._install(clips)
        ids = torch.arange(self.u.num_envs, device=self.mpc.device)
        self.mpc.sched.reset(ids, clips, [self.u.assignment_kit]*len(clips))
        self.u.env_max_steps.fill_(ticks+2)
        q = self.u.robot.data.joint_pos[:, self.pid.ids].float()
        qd = self.u.robot.data.joint_vel[:, self.pid.ids].float()
        self.mpc.reset(ids, q); self.pid.reset(q)
        zero = q.new_zeros(len(clips), self.u.cfg.action_space)
        torch.cuda.reset_peak_memory_stats(self.mpc.device)
        torch.cuda.synchronize(self.mpc.device)
        started = time.monotonic()
        timings = dict(decoder_planning_s=0., physics_pid_s=0., measured_collision_s=0.)
        for tick in range(ticks):
            stamp = time.monotonic()
            cmd, index = self.mpc.plan(q, qd)
            torch.cuda.synchronize(self.mpc.device)
            timings['decoder_planning_s'] += time.monotonic()-stamp
            stamp = time.monotonic()
            self.pid.prepare(q, cmd, index)
            _, _, terminated, truncated, _ = self.env.step(zero)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError('Unexpected reset during capacity probe')
            q = self.u.robot.data.joint_pos[:, self.pid.ids].float()
            qd = self.u.robot.data.joint_vel[:, self.pid.ids].float()
            self.pid.finish(self.mpc.sched.advance())
            torch.cuda.synchronize(self.mpc.device)
            timings['physics_pid_s'] += time.monotonic()-stamp
            if not torch.isfinite(q).all() or not torch.isfinite(qd).all():
                raise FloatingPointError('Nonfinite capacity-probe state')
            if self.collision_weight and tick % self.collision_stride == 0:
                stamp = time.monotonic()
                self._collision(q)
                torch.cuda.synchronize(self.mpc.device)
                timings['measured_collision_s'] += time.monotonic()-stamp
        torch.cuda.synchronize(self.mpc.device)
        free, total = torch.cuda.mem_get_info(self.mpc.device)
        return dict(physical_envs=len(clips), ticks=ticks, wall_time_s=time.monotonic()-started,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.mpc.device),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.mpc.device),
            device_used_bytes=total-free, device_total_bytes=total,
            decoder_lane_passes=int(self.mpc.decoder_passes.sum()),
            training_fitness=False, **timings)

    @torch.inference_mode()
    def evaluate(self, episodes, progress=None, *, initial_states=None, collect_post_hits=False,
                 collect_prefirst_snapshots=False, command_overrides=None):
        if not episodes:
            return []
        if initial_states is not None and len(initial_states) != len(episodes):
            raise ValueError('One starting state per physical episode required')
        if command_overrides is not None and len(command_overrides) != len(episodes):
            raise ValueError('One command override entry per physical episode required')
        B = self.u.num_envs
        if self.video_path and (B!=1 or len(episodes)!=1):
            raise ValueError('Video capture requires one episode in a single physical slot')
        reports = []
        for offset in range(0, len(episodes), B):
            real = episodes[offset:offset+B]
            if all(clip is None for clip in real):
                reports.extend([None]*len(real));continue
            template=next(clip for clip in real if clip is not None)
            # Unused scene slots stay idle rather than decoding duplicate plans.
            idle=replace(template,hits=[],duration=self.mpc.dt)
            # Explicit None slots let diagnostic curves retain physical slot IDs
            # through invalid perturbations. Their outcomes remain missing labels.
            clips = [idle if clip is None else clip for clip in real]+[idle]*(B-len(real))
            torch.manual_seed(self.seed); np.random.seed(self.seed); self.u.seed(self.seed)
            states = initial_states[offset:offset+B] if initial_states is not None else None
            if states is not None:
                states = states+[states[-1]]*(B-len(states))
            self._reset_from_states(states)
            self._install(clips)
            near = self._near_miss_tracker(clips) if self.near_miss_weight else None
            ids = torch.arange(B, device=self.mpc.device)
            self.mpc.sched.reset(ids, clips, [self.u.assignment_kit]*B)
            if states is not None:
                self.mpc.sched.start_drum.copy_(torch.as_tensor(
                    [s.get('initial_drums', [-1, -1]) for s in states], device=self.mpc.device))
                # Captured states are already40 ms past contact, not contact age0.
                ages = torch.as_tensor([s.get('contact_age_ticks', [100000, 100000]) for s in states],
                                       device=self.mpc.device)
                self.mpc.sched.start_ticks.copy_((int(self.mpc.sched.departure_exempt_wp)-ages).clamp_min(0))
            durations = torch.tensor([int(np.ceil(c.duration/self.mpc.dt)) for c in clips],
                                     device=self.mpc.device)
            ticks = int(durations.max())
            override_values = override_mask = None
            if command_overrides is not None:
                override_values = torch.zeros((B,ticks,len(self.pid.ids)),
                                              device=self.mpc.device,dtype=torch.float32)
                override_mask = torch.zeros((B,ticks),device=self.mpc.device,dtype=torch.bool)
                for row,spec in enumerate(command_overrides[offset:offset+B]):
                    if spec is None:
                        continue
                    start = int(spec['start_tick'])
                    values = torch.as_tensor(spec['commands'],device=self.mpc.device,
                                             dtype=override_values.dtype)
                    end = start + len(values)
                    if start < 0 or end > ticks or values.ndim != 2 or values.shape[1] != len(self.pid.ids):
                        raise ValueError('Command override exceeds episode or joint dimensions')
                    override_values[row,start:end] = values
                    override_mask[row,start:end] = True
            self.u.env_max_steps.fill_(ticks+2)
            q = self.u.robot.data.joint_pos[:, self.pid.ids].float()
            qd = self.u.robot.data.joint_vel[:, self.pid.ids].float()
            initial_q, initial_qd = q.clone(), qd.clone()
            if states is not None:
                wanted_q = q.new_tensor([s['q'] for s in states])
                wanted_qd = qd.new_tensor([s['qd'] for s in states])
                if (q-wanted_q).abs().max() > 1e-5 or (qd-wanted_qd).abs().max() > 1e-5:
                    raise RuntimeError('Isaac initial state differs from neural observation')
                left, _, right, _ = self.u._compute_tip_state()
                initial_tips = torch.stack((left, right), 1)-self.u.scene.env_origins[:, None]
                expected_tips = initial_tips.new_tensor([s['tips'] for s in states])
                if (initial_tips-expected_tips).norm(dim=-1).max() > .001:
                    raise RuntimeError('Cartesian tips installed in Isaac differ from neural observations by over1mm')
            else:
                initial_tips = None
            self.mpc.reset(ids, q); self.pid.reset(q)
            video_frames=[] if self.video_path else None
            if video_frames is not None:
                origin=self.u.scene.env_origins[0].detach().cpu().numpy()
                camera_offsets={'front':[0.,2.2,1.1],
                    '3quarter':[1.6,1.2,1.25],'side':[-2.2,0.,1.1],'top':[.3,.3,2.8]}
                if self.video_camera not in camera_offsets:raise ValueError('Unknown video camera angle')
                self.u.sim.set_camera_view(
                    eye=(origin+np.array(camera_offsets[self.video_camera])).tolist(),
                    target=(origin+np.array([.30,0.,.95])).tolist())
            if self.rollout_audit:
                from isaac_rl.rollout_reproducibility import scoring_state
                audit_before=scoring_state(self.u);audit_frames=[]
            torque_area = q.new_zeros(B); joint_area = q.new_zeros(B)
            collision_area = q.new_zeros(B); collision = q.new_zeros(B)
            done_count = torch.zeros(B, device=q.device, dtype=torch.long)
            credited = torch.zeros_like(self.u.beat_valid)
            zero = q.new_zeros(B, self.u.cfg.action_space)
            cost = self.mpc.descent_cost
            captures = [[None, None] for _ in real]
            prefirst_captures=[[] for _ in real]
            prefirst_due={}
            if collect_prefirst_snapshots:
                for row,clip in enumerate(real):
                    if clip is None or not clip.hits:continue
                    first=min(hit.time for hit in clip.hits)
                    for snapshot_tick in prefirst_snapshot_ticks(first,self.mpc.dt,
                            guard_s=max(.10,self.near_miss_window_s+.02)):
                        prefirst_due.setdefault(snapshot_tick,[]).append(row)
            if collect_post_hits:
                hit_age = torch.full((B, 2), 100000, device=q.device, dtype=torch.long)
                last_drums = torch.full((B, 2), -1, device=q.device, dtype=torch.long)
                if states is not None:
                    last_drums.copy_(self.mpc.sched.start_drum)
                previously_credited = torch.zeros_like(self.u.beat_valid)
            for tick in range(ticks):
                cmd, hit_index = self.mpc.plan(q, qd)
                if override_mask is not None:
                    cmd = torch.where(override_mask[:,tick,None],
                                      override_values[:,tick],cmd)
                self.pid.prepare(q, cmd, hit_index)
                if self.rollout_audit:
                    audit_frame=dict(q=q[:len(real)].cpu().numpy().copy(),qd=qd[:len(real)].cpu().numpy().copy(),
                        command=self.pid.cmd[:len(real)].cpu().numpy().copy(),hit_index=hit_index[:len(real)].cpu().numpy().copy(),
                        contact_age=self.pid.compliance._since_contact[:len(real)].cpu().numpy().copy())
                _, _, terminated, truncated, _ = self.env.step(zero)
                if video_frames is not None:
                    frame=self.u.render()
                    if frame is None:raise RuntimeError('Isaac camera returned no RGB frame')
                    video_frames.append(np.asarray(frame).copy())
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("Unexpected Isaac episode reset during CMA evaluation")
                q = self.u.robot.data.joint_pos[:, self.pid.ids].float()
                qd = self.u.robot.data.joint_vel[:, self.pid.ids].float()
                if not torch.isfinite(q).all() or not torch.isfinite(qd).all():
                    raise FloatingPointError("Nonfinite Isaac joint state")
                fired = self.mpc.sched.advance(); self.pid.finish(fired)
                live = tick < durations
                available = available_torque_torch(qd, cost.tau_hard[None], cost.v_hard[None],
                    eta=1., saturation_scale=cost.cfg.torque_saturation_scale,
                    derate=cost.cfg.torque_derate)
                tau = self.u.robot.data.computed_torque[:, self.pid.ids]
                excess = (tau.abs()-available).clamp_min(0).sum(-1)
                torque_area += excess*self.mpc.dt*live
                box = torch.maximum((q-self.pid.q_hi).clamp_min(0),
                                    (self.pid.q_lo-q).clamp_min(0)).amax(-1)
                joint_area += box*self.mpc.dt*live
                if self.collision_weight and tick % self.collision_stride == 0:
                    collision = self._collision(q)
                collision_area += collision*self.mpc.dt*live
                current = (self.u.beat_done & self.u.beat_valid).sum(-1)
                done_count = torch.where(live, current, done_count)
                credited = torch.where(live[:, None], self.u.beat_done & self.u.beat_valid, credited)
                if tick+1 in prefirst_due:
                    candidate=prefirst_due[tick+1]
                    index=torch.as_tensor(candidate,device=q.device,dtype=torch.long)
                    clear=(~credited[index].any(-1)
                        & ~((self.u.beat_credit[index]>1e-8)
                            & self.u.beat_valid[index]).any(-1)).cpu().numpy()
                    rows=[row for row,okay in zip(candidate,clear) if okay]
                    if rows:
                        ids_now=torch.as_tensor(rows,device=q.device,dtype=torch.long)
                        tips=(self.u.stick_probe_pos_w[ids_now,:,0]
                            -self.u.scene.env_origins[ids_now,None])
                        values=dict(q=q[ids_now].cpu().numpy(),qd=qd[ids_now].cpu().numpy(),
                            tips=tips.cpu().numpy(),
                            contact_age_ticks=self.pid.compliance._since_contact[ids_now].cpu().numpy(),
                            torque_area=torque_area[ids_now].cpu().numpy(),
                            joint_area=joint_area[ids_now].cpu().numpy(),
                            collision_area=collision_area[ids_now].cpu().numpy())
                        for j,row in enumerate(rows):
                            prefirst_captures[row].append(dict(time_s=(tick+1)*self.mpc.dt,
                                q=values['q'][j].tolist(),qd=values['qd'][j].tolist(),
                                tips=values['tips'][j].tolist(),
                                initial_drums=(states[row].get('initial_drums',[-1,-1])
                                    if states is not None else self.mpc.sched.start_drum[row].cpu().tolist()),
                                contact_age_ticks=values['contact_age_ticks'][j].tolist(),
                                prefix_torque_area_Nm_s=float(values['torque_area'][j]),
                                prefix_joint_area_rad_s=float(values['joint_area'][j]),
                                prefix_collision_area_m_s=float(values['collision_area'][j])))
                if self.rollout_audit:
                    audit_frame['hits']=done_count[:len(real)].cpu().numpy().copy()
                    audit_frame['contacts']=self.pid.contacts[:len(real)].cpu().numpy().copy()
                    audit_frame['beat_credit']=self.u.beat_credit[:len(real)].cpu().numpy().copy();audit_frames.append(audit_frame)
                if collect_post_hits:
                    hit_age += 1
                    fresh = credited & ~previously_credited
                    previously_credited.copy_(credited)
                    for hand in (0, 1):
                        hand_events = fresh & (self.u.beat_hand == hand)
                        times = torch.where(hand_events, self.u.beat_time, -1.)
                        newest_time, index = times.max(1)
                        new = newest_time >= 0
                        last_drums[:, hand] = torch.where(new,
                            self.u.beat_drum_id.gather(1, index[:, None])[:, 0], last_drums[:, hand])
                        hit_age[:, hand] = torch.where(new, 0, hit_age[:, hand])
                    due = (hit_age == 2) & live[:, None]  # 40 ms after measured successful contact.
                    events = due.nonzero(as_tuple=False)
                    if len(events):
                        left, _, right, _ = self.u._compute_tip_state()
                        tips = torch.stack((left, right), 1)-self.u.scene.env_origins[:, None]
                        rows = events[:, 0]
                        values = dict(q=q[rows].cpu().numpy(), qd=qd[rows].cpu().numpy(),
                            tips=tips[rows].cpu().numpy(), initial_drums=last_drums[rows].cpu().numpy(),
                            contact_age_ticks=hit_age[rows].clamp_max(100000).cpu().numpy())
                        for j, (row, hand) in enumerate(events.cpu().tolist()):
                            if row < len(real):
                                captures[row][hand] = dict(
                                    **{key: value[j].tolist() for key, value in values.items()},
                                    kind='post_hit', source_hand=hand,
                                    source_drum=int(values['initial_drums'][j, hand]),
                                    capture_time_s=(tick+1)*self.mpc.dt)
                if near is not None:
                    # Read body-derived tip states without invoking the task's
                    # strike tracker a second time or changing contact credit.
                    left, _, right, _ = self.u._compute_tip_state()
                    tips = torch.stack((left, right), dim=1)
                    tips = tips-self.u.scene.env_origins[:, None]
                    near.update((tick+1)*self.mpc.dt, tips, live)
                if progress and (tick+1) % 50 == 0:
                    progress(offset, len(episodes), tick+1, ticks)
            elapsed = durations*self.mpc.dt
            if video_frames is not None:
                import imageio
                from pathlib import Path
                path=Path(self.video_path);path.parent.mkdir(parents=True,exist_ok=True)
                imageio.mimwrite(path,video_frames,fps=round(1/self.mpc.dt),macro_block_size=None)
            credit_cpu = credited.cpu().numpy()
            valid_cpu = self.u.beat_valid.cpu().numpy()
            drums_cpu = self.u.beat_drum_id.cpu().numpy()
            if near is not None:
                partial, quality = near.credits(credited[:, :near.times.shape[1]])
                partial_cpu = partial.cpu().numpy()
                closest_cpu = near.closest.cpu().numpy()
            for b, clip in enumerate(real):
                if clip is None:
                    reports.append(None);continue
                requested = len(clip.hits)
                hits = int(done_count[b])
                torque = float(torque_area[b]/elapsed[b])
                joint = float(joint_area[b]/elapsed[b])
                collision_mean = float(collision_area[b]/elapsed[b])
                near_bonus = float(partial_cpu[b].sum()/requested) if near is not None else 0.
                reward = hits/requested+near_bonus-self.torque_weight*torque/5.
                reward -= self.joint_weight*joint/np.deg2rad(5)
                reward -= self.collision_weight*collision_mean/.005
                if not np.isfinite(reward):
                    raise FloatingPointError("Nonfinite episode fitness")
                by_drum = {}
                for drum, name in enumerate(self.u.assignment_kit.names):
                    mask = valid_cpu[b] & (drums_cpu[b] == drum)
                    n = int(mask.sum())
                    count = int((mask & credit_cpu[b]).sum())
                    by_drum[name] = dict(requested=n, hits=count, misses=n-count)
                if (sum(r["requested"] for r in by_drum.values()) != requested
                        or sum(r["hits"] for r in by_drum.values()) != hits):
                    raise ValueError("Per-drum contact counts differ from episode totals")
                miss_distance = None
                if near is not None:
                    miss = near.valid[b].cpu().numpy() & ~credit_cpu[b, :near.times.shape[1]]
                    distances = closest_cpu[b, miss]
                    distances = distances[np.isfinite(distances)]
                    miss_distance = float(np.median(distances)) if len(distances) else None
                report = dict(hits=hits, requested=requested, hit_rate=hits/requested,
                    near_miss_bonus=near_bonus, hit_quality=hits/requested+near_bonus,
                    missed_hit_distance_median_m=miss_distance,
                    by_drum=by_drum,
                    torque_demand_excess_average_Nm=torque,
                    joint_excess_average_rad=joint,
                    measured_geometry_penetration_average_m=collision_mean, reward=float(reward))
                ordered_hits=sorted(clip.hits,key=lambda h:(h.time,h.drum))
                report['per_hit_outcomes']=[dict(time_s=h.time,drum=h.drum_name,hand=h.hand,
                    hit=bool(credit_cpu[b,j]),closest_distance_m=(float(closest_cpu[b,j])
                        if near is not None and np.isfinite(closest_cpu[b,j]) else None))
                    for j,h in enumerate(ordered_hits)]
                if command_overrides is not None:
                    spec = command_overrides[offset+b]
                    report['command_override'] = (None if spec is None else dict(
                        start_tick=int(spec['start_tick']),
                        knots=len(spec['commands'])))
                if self.match_decoder_speed:
                    graded=self.u._beat_target_speeds()[b,:len(ordered_hits)].cpu().tolist()
                    for outcome,speed in zip(report['per_hit_outcomes'],graded):outcome['target_impact_speed_m_s']=speed
                if states is not None:
                    report['initial_state'] = dict(q=initial_q[b].cpu().tolist(), qd=initial_qd[b].cpu().tolist(),
                                                  tips=initial_tips[b].cpu().tolist(), kind=states[b].get('kind'))
                if collect_post_hits:
                    report['post_hit_states'] = [state for state in captures[b] if state is not None]
                if collect_prefirst_snapshots:
                    for snapshot in prefirst_captures[b]:
                        remaining=float(elapsed[b])-snapshot['time_s']
                        if remaining<=0:raise ValueError('Pre-hit snapshot has no remaining horizon')
                        future_torque=(float(torque_area[b])-snapshot['prefix_torque_area_Nm_s'])/remaining
                        future_joint=(float(joint_area[b])-snapshot['prefix_joint_area_rad_s'])/remaining
                        future_collision=(float(collision_area[b])-snapshot['prefix_collision_area_m_s'])/remaining
                        value=hits/requested+near_bonus-self.torque_weight*future_torque/5.
                        value-=self.joint_weight*future_joint/np.deg2rad(5)
                        value-=self.collision_weight*future_collision/.005
                        snapshot['return_to_go_reward']=float(value)
                        snapshot['future_hit_rate']=hits/requested
                    report['prefirst_snapshots']=prefirst_captures[b]
                if self.rollout_audit:
                    frames=audit_frames[:int(durations[b])]
                    report['rollout_audit']=dict(scoring_before=audit_before,scoring_after=scoring_state(self.u),
                        env_origin=self.u.scene.env_origins[b].cpu().tolist(),
                        trace={key:[frame[key][b].tolist() for frame in frames] for key in ('q','qd','command','hit_index','hits','contact_age','contacts')},
                        beat_credit=frames[-1]['beat_credit'][b,:len(ordered_hits)].tolist(),
                        contact_tracker_stats=(self.u.contact_tracker.stats()
                            if self.u.contact_tracker is not None else None),
                        contact_tracker_debug_events=getattr(self.u.contact_tracker,'debug_events',None),
                        decoder_amp=str(self.mpc.cfg.amp_dtype),reset_discarded_contact_force_max_N=getattr(self,'_reset_contact_peak',None))
                reports.append(report)
        return reports
