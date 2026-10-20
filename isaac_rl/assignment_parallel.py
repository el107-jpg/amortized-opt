"""Parallel committed plans and a lean controller on the original Isaac task.

Physics stepping, contact tracking and hit-credit computation remain the
original environment methods. No task observation or playback trace is built.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import torch

from amortized_opt.impedance import ContactAwareStrikeImpedance, ContactAwareStrikeImpedanceConfig
from amortized_opt.limits import isaac_effort_limits, isaac_runtime_stiffness, urdf_position_limits
from amortized_opt.train_cost import group_lengths
from isaac_rl.mpc import IsaacMPC


def take_rows(value, rows):
    return type(value)(**{f.name: (getattr(value, f.name).index_select(0, rows)
                                  if isinstance(getattr(value, f.name), torch.Tensor)
                                  else getattr(value, f.name)) for f in fields(value)})


class ParallelCommittedMPC(IsaacMPC):
    """Only changing goal rows are sent through the decoder and physics cost.

    Unchanged rows retain their plan bit for bit. Each active new goal receives
    exactly one decoder pass from the measured state and a fresh clean seed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.commit_plan or not self.cfg.fresh_terminal_every_tick:
            raise ValueError("parallel driver needs base commit_plan=False and fresh terminal seeds")
        self.reference = None
        self.goal_keys = None
        self.collision = torch.zeros(self.B, device=self.device)
        self.new_plans = torch.zeros(self.B, device=self.device, dtype=torch.bool)
        self.decoder_passes = torch.zeros(self.B, device=self.device, dtype=torch.long)

    def reset(self, env_ids, q0):
        super().reset(env_ids, q0)
        self.reference = q0[:, None].expand(-1, self.H, -1).clone()
        self.goal_keys = torch.full((self.B, 2), -999, device=self.device, dtype=torch.long)
        self.decoder_passes.zero_()

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
            # The single-env committed path advances the yaw-conditioning
            # ramp only when generating a plan. Preserve that per-row cadence.
            old_yaw = self.yaw_bc.clone()
            enc, _ = self._encoding(hit_index, q)
            self.yaw_bc[~live] = old_yaw[~live]
            masks = take_rows(self.masks(hit_index), rows)
            ctx = take_rows(self._ctx, rows)
            state = torch.cat((q[rows], qd[rows]), -1)
            core = self.model
            init = core.initial_trajectory(ctx, state, masks).to(self.dtype)
            with torch.autocast(device_type=self.device.type, dtype=self.cfg.amp_dtype,
                                enabled=self.device.type == "cuda" and self.cfg.amp_dtype is not None):
                out = core.refine(init*masks.elem_mask, state, take_rows(enc, rows), masks, iterations=1)
            candidate = self._hold_plan_end(out.traj.to(self.dtype), q[rows], hit_index[rows])
            strike = take_rows(self.sched.strike_context(), rows)
            with torch.autocast(device_type=self.device.type, enabled=False):
                score = self.descent_cost(q[rows], candidate, qd[rows], strike,
                                         lengths=group_lengths(strike.hit_index).to(q.dtype),
                                         elem_mask=masks.elem_mask)
            self.collision[rows] = score.violations["collision"]
            finite = torch.isfinite(candidate).flatten(1).all(1)
            candidate = torch.where(finite[:, None, None], candidate,
                                    q[rows, None].expand(-1, self.H, -1))
            self.reference.index_copy_(0, rows, candidate)
            self.decoder_passes[rows] += 1
        self.goal_keys.copy_(key)
        self.last_ref0 = self.reference[:, 0].clone()
        self.n_ticks += 1
        return self._command(self.reference, hit_index), hit_index

    def commit(self, cmd):
        self.prev_cmd = cmd.clone()
        self.reference = torch.cat((self.reference[:, 1:], self.reference[:, -1:]), 1)


class ParallelPID:
    """Batched version of the committed-plan player's PID and runtime hooks."""
    def __init__(self, u, mpc, rnea):
        self.u, self.mpc, self.rnea = u, mpc, rnea
        self.ids = list(u.upper_body_joint_ids)
        self.device, self.dt, self.B = mpc.device, mpc.dt, mpc.B
        self.kp = torch.as_tensor(isaac_runtime_stiffness(mpc.names), device=self.device, dtype=torch.float32)
        self.tau_lim = torch.as_tensor(isaac_effort_limits(mpc.names), device=self.device, dtype=torch.float32)
        lo, hi = urdf_position_limits(mpc.names, posture_limits=False)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=torch.float32)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=torch.float32)
        self.waist = mpc.waist_cols
        self.cmd = torch.zeros(self.B, mpc.n, device=self.device)
        self.integral = torch.zeros_like(self.cmd)
        self.prev_velocity = torch.zeros_like(self.cmd)
        self.waist_tau = None
        self.contacts = torch.zeros(self.B, 2, device=self.device, dtype=torch.bool)
        self.compliance = ContactAwareStrikeImpedance(ContactAwareStrikeImpedanceConfig(),
                                                    batch=self.B, device=self.device, joint_names=mpc.names)
        self.gain_map = []
        mapping = {int(j): c for c, j in enumerate(self.ids)}
        for act in u.robot.actuators.values():
            ji = act.joint_indices
            if isinstance(ji, slice):
                ji = range(*ji.indices(u.robot.num_joints))
            pairs = [(p, mapping[int(j)]) for p, j in enumerate(ji) if int(j) in mapping]
            if pairs:
                if not type(act).__name__.startswith("IdealPD"):
                    raise RuntimeError("parallel compliance requires IdealPD actuators")
                self.gain_map.append((act, torch.tensor([p for p, _ in pairs], device=self.device),
                                      torch.tensor([c for _, c in pairs], device=self.device),
                                      act.stiffness.clone(), act.damping.clone()))
        original_apply = u._apply_action

        def apply():
            original_apply()
            self.sample_contacts()
            if self.waist_tau is not None:
                tau = u._grav_tau.clone()
                tau[:, self.waist] += self.waist_tau
                u.robot.set_joint_effort_target(tau.clamp(-self.tau_lim, self.tau_lim), joint_ids=self.ids)

        u._apply_action = apply
        # Retain the original action preparation for exact physics parity.
        # It is inexpensive relative to diagnostics; overwrite final targets
        # just as the player does. No observations or plan traces are produced.
        original_pre = u._pre_physics_step

        def pre(actions):
            original_pre(actions)
            u.joint_pos_target[:, self.ids] = self.cmd

        u._pre_physics_step = pre
        # Hit credit uses _get_rewards and the physics-rate strike trackers.
        # The task's policy observations (including torque-budget features) are
        # unused by the assignment learner; skip only that observation work.
        u._get_observations = lambda: {"policy": torch.zeros(self.B, 1, device=self.device)}

    def sample_contacts(self):
        force = self.u._read_contact_forces()
        self.contacts |= (torch.linalg.vector_norm(force, dim=-1).amax(-1) > 1.0)

    def reset(self, q):
        self.cmd.copy_(q)
        self.integral.zero_(); self.prev_velocity.zero_(); self.contacts.zero_()
        self.waist_tau = None
        self.compliance.reset()
        self.apply_gains(torch.full((self.B, 2), -1, device=self.device))

    def apply_gains(self, ticks):
        kp, kd = self.compliance.scales(ticks)
        for act, local, cols, kp0, kd0 in self.gain_map:
            act.stiffness[:, local] = kp0[:, local]*kp[:, cols]
            act.damping[:, local] = kd0[:, local]*kd[:, cols]

    def prepare(self, q, cmd, hit_index):
        self.contacts.zero_()
        self.integral.add_((cmd-q)*self.dt).clamp_(-0.25, 0.25)
        corrected = cmd+5.0/self.kp*self.integral
        corrected = corrected.clamp(self.q_lo, self.q_hi)
        corrected = self.cmd+(corrected-self.cmd).clamp(-self.mpc.step_max, self.mpc.step_max)
        velocity = (corrected-self.cmd)/self.dt
        acceleration = (velocity-self.prev_velocity)/self.dt
        self.prev_velocity.copy_(velocity)
        acceleration[:, self.waist] = 0
        self.waist_tau = self.rnea.mass_matrix_torque(q, acceleration)[:, self.waist].contiguous()
        self.cmd.copy_(corrected)
        self.apply_gains(hit_index-1)

    def finish(self, fired):
        self.sample_contacts()
        self.compliance.observe(fired, self.contacts)
        for hand in (0, 1):
            cols = self.mpc.arm_cols[hand]
            self.integral[:, cols] = torch.where(fired[:, hand, None], 0.0, self.integral[:, cols])
        self.mpc.commit(self.cmd)


class ParallelPoseMPC(ParallelCommittedMPC):
    """One next-hit pose MLP call, committed Hermite reference, real contact.

    This adapter never samples a bounce: Isaac supplies all subsequent measured
    velocities. New future assignments do not interrupt the current hit plan.
    """
    def __init__(self, *args, kit, **kwargs):
        super().__init__(*args, **kwargs)
        from amortized_opt.trajectory_surrogate import SingleHitGenerator
        self.pose_generator = SingleHitGenerator(self.model, self.descent_cost, kit, horizon=self.H)
        self.pose_keys = torch.full((self.B, 3), -999, device=self.device, dtype=torch.long)

    def reset(self, env_ids, q0):
        super().reset(env_ids, q0)
        self.pose_keys.fill_(-999)

    @torch.no_grad()
    def plan(self, q, qd):
        from amortized_opt.trajectory_surrogate import hermite
        tth, drum, _pos, vel, _valid, active = self.sched.goal()
        nearest = torch.where(active, tth, 1e6).amin(1)
        due = active & ((tth-nearest[:, None]).abs() < 1e-5)
        live = due.any(1)
        target = torch.where(due, drum, -1)
        absolute = torch.round((nearest.double()+self.sched.t_now)/self.dt).long()
        absolute = torch.where(live, absolute, -1)
        key = torch.cat((absolute[:, None], target), 1)
        changed = (key != self.pose_keys).any(1)
        new = changed & live
        self.new_plans = new; self.collision.zero_()
        self.reference[changed & ~live] = q[changed & ~live, None].expand(-1, self.H, -1)
        rows = new.nonzero(as_tuple=True)[0]
        if len(rows):
            duration = nearest[rows].clamp_min(self.dt)
            drums = target[rows, None]
            speeds = torch.linalg.vector_norm(vel[rows], dim=-1)[:, None]
            valid = torch.ones(len(rows), 1, device=self.device, dtype=torch.bool)
            pose, approach = self.pose_generator.predict(q[rows], qd[rows], drums,
                duration[:, None], speeds, valid, 0)
            u = (torch.arange(1, self.H+1, device=self.device)*self.dt)[None]/duration[:, None]
            path, _, _ = hermite(q[rows], qd[rows], pose, approach, duration, u.clamp_max(1))
            diagnostics = self.pose_generator.surrogate.evaluate(q[rows], qd[rows], pose[:, None],
                torch.zeros_like(pose[:, None]), drums, duration[:, None], speeds, valid,
                torch.ones_like(speeds), pre_velocity_override=approach[:, None])
            self.collision[rows] = diagnostics['collision_m']
            if not torch.isfinite(path).all(): raise FloatingPointError('nonfinite direct-pose plan')
            self.reference.index_copy_(0, rows, path)
            self.decoder_passes[rows] += 1  # Existing report field counts generator calls.
        self.pose_keys.copy_(key)
        ticks = (nearest/self.dt).round().long().clamp(1, self.H)
        hit_index = torch.where(due, ticks[:, None], -1)
        self.last_ref0 = self.reference[:, 0].clone(); self.n_ticks += 1
        return self._command(self.reference, hit_index), hit_index
