"""The model as the MPC, batched over Isaac envs.

One control tick, for all ``num_envs`` at once:

    1. ``BatchedSchedule`` -> ``ContextInputs`` + per-hand ``hit_index``;
    2. ``build_waypoint_masks`` -> the shrinking-horizon element masks;
    3. ONE shared forward pass of the amortized optimizer (the per-env
       parameter-noise hooks of :mod:`isaac_rl.perturb` ride inside it);
    4. the per-joint reference lead, the slew bound and the joint-box clamp,
       in that order -- lead FIRST so the clamp still guarantees an admissible
       command (``amortized_opt.lead``, and ``MPCGenerator.run_episode``);
    5. the caller hands the command to the plant (Isaac), then calls
       :meth:`IsaacMPC.commit` with the measured state and the fired mask,
       which shifts the plan one waypoint and re-seeds the arm columns of any
       hand whose hit just fired.

This is the batched twin of ``MPCGenerator.run_episode`` with the plant
replaced.  Where a semantic had to be re-derived, the docstring names the
``mpc_gen`` / ``MPCRoller`` line it mirrors.  Isaac-free: the class never sees
a simulator, only ``(q, qd)`` tensors in and a command tensor out.

Deliberately carried over from ``MPCGenConfig``
-----------------------------------------------

``cmd_lead_s`` (per joint, ``LEAD_PRESETS["fit"]`` by default -- 7c decision 2),
``cmd_slew_eta``, ``clamp_cmd_to_box``, ``freeze_wp`` / ``freeze_waist``
(default OFF), ``hit_lead_s`` (default 0, lives on
:class:`~isaac_rl.schedule.ScheduleConfig`), ``sticky_goal_point`` (structurally
always on here), ``planned_yaw_bc``.

Deliberately DROPPED
--------------------

``minv_refresh`` / ``randomize_gains`` / ``divergence_box_margin`` -- properties
of the standalone PD stand-in, which Isaac replaces.  ``record_min_sdf``,
``short_horizon_wp`` / ``extra_short_samples``, ``departure_exempt_wp`` --
training-data bookkeeping that belongs to the offline generator; the RL reward
recomputes clearance itself.  ``seed_slew_eta`` -- the numpy quintic seed is
replaced by :func:`~amortized_opt.train_cost.ramp_seed`, which is monotone and
already admissible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.lead import LEAD_PRESETS, lead_fraction, lead_waypoints, resolve_lead
from amortized_opt.limits import isaac_velocity_limits, urdf_position_limits
from amortized_opt.model import ContextEncoding
from amortized_opt.tokenizer import build_waypoint_masks
from amortized_opt.train_cost import ramp_seed
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, waist_slice

from .schedule import BatchedSchedule

__all__ = ["IsaacMPC", "MPCConfig", "TickResult"]


@dataclass
class MPCConfig:
    horizon: int = 50
    dt: float = CONTROL_DT
    #: Refinement iterations per control tick.  The plan's standing default is
    #: one, and section 3.4 measured x1 tying or beating x3 at every seed.
    iterations: int = 1
    #: Per-joint reference lead, seconds.  ``"fit"`` is 7c's recommendation and
    #: the champion's deployment configuration.  Re-fit against ISAAC before a
    #: training run (``scripts/fit_lead_isaac.py``): the shipped vector was
    #: identified against ``amortized_opt.pd_sim``, not against PhysX.
    cmd_lead_s: str | float | tuple[float, ...] | None = "fit"
    #: Slew bound on the executed command as a fraction of the speed rating.
    cmd_slew_eta: float | None = 1.0
    clamp_cmd_to_box: bool = True
    #: Freeze window, per hand, in waypoints.  DEFAULT OFF -- 7c decision 2
    #: ships the per-joint lead alone.
    freeze_wp: int = 0
    freeze_waist: bool = True
    #: Re-encode the context every tick.  ``False`` caches the encoding for the
    #: life of a hit segment, which is cheaper but feeds the decoder a STALE
    #: ``goal_tth``; see the DECISION note in ``isaac_rl/README.md``.
    encode_every_tick: bool = True
    #: Autocast dtype for the model pass (``None`` = fp32).
    amp_dtype: torch.dtype | None = torch.bfloat16
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)
    yaw_bins: int = 129


@dataclass
class TickResult:
    """Everything one control tick produced, for the reward and the logs."""

    cmd: torch.Tensor            # (B, 17) the command handed to the plant
    traj: torch.Tensor           # (B, H, 17) the plan (command row already in)
    elem_mask: torch.Tensor      # (B, H, 17)
    hit_index: torch.Tensor      # (B, 2) long
    active: torch.Tensor         # (B, 2) bool
    aux: dict                    # decoder auxiliary heads (vel/acc/torque/fk)
    frozen: torch.Tensor         # (B,) bool -- the freeze window suppressed a replan
    encoded: bool                # did this tick run the context encoder?


class IsaacMPC:
    """Batched receding-horizon driver around an :class:`AmortizedOptimizer`."""

    def __init__(
        self,
        model,
        schedule: BatchedSchedule,
        cfg: MPCConfig | None = None,
        *,
        device,
        dtype: torch.dtype = torch.float32,
        library: GoalPoseLibrary | None = None,
        kit_names: tuple[str, ...] | None = None,
    ):
        self.model = model
        self.sched = schedule
        self.cfg = cfg or MPCConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.B = schedule.B
        self.H = int(self.cfg.horizon)
        self.dt = float(self.cfg.dt)
        self.names = tuple(self.cfg.joint_names)
        self.n = len(self.names)

        lo, hi = urdf_position_limits(self.names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        eta = self.cfg.cmd_slew_eta
        self.step_max = (
            None if eta is None else
            torch.as_tensor(float(eta) * isaac_velocity_limits(self.names) * self.dt,
                            device=self.device, dtype=dtype)
        )
        self.lead_f = lead_fraction(resolve_lead(self.cfg.cmd_lead_s), self.dt, self.n,
                                    device=self.device, dtype=dtype)
        self.arm_cols = {h: arm_slice(hand, self.names)
                         for h, hand in enumerate(("left", "right"))}
        self.waist_cols = waist_slice(self.names)
        self.yaw_i = self.names.index("waist_yaw_joint")

        # goal-pose library, flattened to a (2, D, bins, n_arm) gather table --
        # the host-side dict lookup of GoalPoseLibrary.select cannot run inside
        # a 50 Hz batched loop (MPCRoller._goal_pose_table, same construction).
        self.library = library if library is not None else GoalPoseLibrary(joint_names=self.names)
        self.D = int(schedule.cfg.num_drums)
        self.kit_names = tuple(kit_names or ()) + ("",) * max(
            0, self.D - len(kit_names or ()))
        self.n_bins = int(self.cfg.yaw_bins)
        self.yaw_lo = float(lo[self.yaw_i])
        self.yaw_step = (float(hi[self.yaw_i]) - self.yaw_lo) / max(self.n_bins - 1, 1)
        self.lib_q, self.lib_ok = self._goal_pose_table()

        # rolling state
        self.init = torch.zeros(self.B, self.H, self.n, device=self.device, dtype=dtype)
        self.prev_cmd = torch.zeros(self.B, self.n, device=self.device, dtype=dtype)
        self._enc: ContextEncoding | None = None
        self.last_ref0 = torch.zeros(self.B, self.n, device=self.device, dtype=dtype)
        self._enc_key: torch.Tensor | None = None
        self.n_encodes = 0
        self.n_ticks = 0

    # ------------------------------------------------------- lookup table --
    def _goal_pose_table(self):
        n_arm = len(self.arm_cols[0])
        q = torch.zeros(2, self.D, self.n_bins, n_arm, device=self.device, dtype=self.dtype)
        ok = torch.zeros(2, self.D, device=self.device, dtype=torch.bool)
        yaws = [self.yaw_lo + i * self.yaw_step for i in range(self.n_bins)]
        for h, hand in enumerate(("left", "right")):
            for d, name in enumerate(self.kit_names[: self.D]):
                if not name or not self.library.has(hand, name):
                    continue
                ok[h, d] = True
                rows = np.stack([self.library.select(hand, name, y).arm_q for y in yaws])
                q[h, d] = torch.as_tensor(rows, device=self.device, dtype=self.dtype)
        return q, ok

    # -------------------------------------------------------------- reset --
    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor, q0: torch.Tensor) -> None:
        """Seed the plan of ``env_ids`` from their measured pose.

        A ramp to the library terminal for every ACTIVE hand, a hold for an
        idle one and for the waist -- the same construction ``MPCRoller._spawn``
        uses for a carried-over slot, and the same one ``_reseed`` uses on a
        fired hand, so a reset and a re-goal put the model on the same kind of
        initialization.
        """
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1).long()
        if env_ids.numel() == 0:
            return
        q0 = q0.to(self.dtype)
        self.init.index_copy_(
            0, env_ids, q0.index_select(0, env_ids).unsqueeze(1).expand(-1, self.H, -1)
            .contiguous()
        )
        self.prev_cmd.index_copy_(0, env_ids, q0.index_select(0, env_ids))
        self._enc = None
        hidx = self.sched.hit_index()
        self._reseed(env_ids, hidx >= 1, q0, hidx)

    # -------------------------------------------------------------- masks --
    def masks(self, hit_index: torch.Tensor):
        active = hit_index >= 0
        t = hit_index.clamp_min(0).to(self.dtype) * self.dt
        return build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=self.H, dt=self.dt,
            active_left=active[:, 0], active_right=active[:, 1],
            joint_names=self.names,
        )

    # --------------------------------------------------------------- tick --
    @torch.no_grad()
    def plan(self, q: torch.Tensor, qd: torch.Tensor) -> TickResult:
        """One refinement + one admissible command.  Does NOT step the clock."""
        cfg = self.cfg
        q = q.to(self.dtype)
        qd = qd.to(self.dtype)
        hit_index = self.sched.hit_index()
        active = hit_index >= 0
        masks = self.masks(hit_index)

        enc, did_encode = self._encoding(hit_index)
        init = self.init * masks.elem_mask
        state = torch.cat([q, qd], dim=-1)

        core = self.model.module if hasattr(self.model, "module") else self.model
        was = core.training
        core.eval()
        amp = (
            torch.autocast(device_type="cuda", dtype=cfg.amp_dtype)
            if (cfg.amp_dtype is not None and self.device.type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with amp:
            out = core.refine(init, state, enc, masks, iterations=int(cfg.iterations))
        if was:
            core.train()
        traj = out.traj.to(self.dtype)

        # ---- the freeze window, per hand (MPCGenerator.run_episode) --------
        frozen = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if cfg.freeze_wp > 0:
            hit_any = torch.zeros_like(frozen)
            for h in (0, 1):
                k = hit_index[:, h]
                sel = (k >= 1) & (k <= int(cfg.freeze_wp))
                if bool(sel.any()):
                    cols = self.arm_cols[h]
                    m = sel.view(-1, 1, 1)
                    traj[:, :, cols] = torch.where(m, self.init[:, :, cols],
                                                   traj[:, :, cols])
                    hit_any |= sel
            if cfg.freeze_waist and bool(hit_any.any()):
                m = hit_any.view(-1, 1, 1)
                traj[:, :, self.waist_cols] = torch.where(
                    m, self.init[:, :, self.waist_cols], traj[:, :, self.waist_cols])
            frozen = hit_any

        # ---- reference lead, then slew, then box ---------------------------
        # ``last_ref0`` is the plan's own first waypoint BEFORE the lead: the
        # pose the planner nominated for the END of this control period.  That,
        # not the (led) command, is what the tracking reward scores the plant
        # against -- scoring against the command would price the lead itself as
        # error and train the model to cancel its own compensation.
        self.last_ref0 = traj[:, 0].clone()
        cmd = self._command(traj, hit_index)
        traj = traj.clone()
        traj[:, 0] = cmd
        self._pending = (traj, hit_index)
        self.n_ticks += 1
        return TickResult(
            cmd=cmd, traj=traj, elem_mask=masks.elem_mask, hit_index=hit_index,
            active=active,
            aux={"vel": out.vel, "acc": out.acc, "torque": out.torque,
                 "fk_pos": out.fk_pos, "fk_vel": out.fk_vel,
                 "point_mask": out.point_mask, "token_mask": out.token_mask},
            frozen=frozen, encoded=did_encode,
        )

    def _encoding(self, hit_index: torch.Tensor):
        """The context encoding, re-run every tick unless caching is asked for."""
        if self.cfg.encode_every_tick or self._enc is None:
            core = self.model.module if hasattr(self.model, "module") else self.model
            amp = (
                torch.autocast(device_type="cuda", dtype=self.cfg.amp_dtype)
                if (self.cfg.amp_dtype is not None and self.device.type == "cuda")
                else torch.autocast(device_type="cuda", enabled=False)
            )
            with amp:
                self._enc = core.encode_context(self.sched.context())
            self.n_encodes += 1
            return self._enc, True
        return self._enc, False

    def _command(self, traj: torch.Tensor, hit_index: torch.Tensor) -> torch.Tensor:
        """``q_ref(t + delta)``, held past each hand's terminal, then clamped.

        The hold matters and is not an implementation detail: past a hand's
        terminal knot the decoder's output is UNPRICED (the element mask zeroes
        it, no cost term touches it), so leading into it would command noise.
        ``amortized_opt.lead`` states the consequence: at the strike tick itself
        the horizon is one waypoint and the lead is a no-op.
        """
        if self.lead_f is None:
            cmd = traj[:, 0]
        else:
            held = traj.clone()
            idx = torch.arange(self.H, device=self.device).view(1, self.H, 1)
            for h in (0, 1):
                k = hit_index[:, h]
                term = (k - 1).clamp(0, self.H - 1)
                keep = (k >= 1).view(-1, 1, 1)
                cols = self.arm_cols[h]
                last = held[:, :, cols].gather(
                    1, term.view(-1, 1, 1).expand(-1, 1, len(cols)))
                beyond = (idx >= term.view(-1, 1, 1)) & keep
                held[:, :, cols] = torch.where(beyond, last.expand(-1, self.H, -1),
                                               held[:, :, cols])
            kw = hit_index.clamp_min(-1).max(dim=1).values
            termw = (kw - 1).clamp(0, self.H - 1)
            keepw = (kw >= 1).view(-1, 1, 1)
            wc = self.waist_cols
            lastw = held[:, :, wc].gather(1, termw.view(-1, 1, 1).expand(-1, 1, len(wc)))
            beyondw = (idx >= termw.view(-1, 1, 1)) & keepw
            held[:, :, wc] = torch.where(beyondw, lastw.expand(-1, self.H, -1),
                                         held[:, :, wc])
            cmd = lead_waypoints(held, self.lead_f)[:, 0]
        if self.step_max is not None:
            cmd = self.prev_cmd + torch.clamp(cmd - self.prev_cmd,
                                              -self.step_max, self.step_max)
        if self.cfg.clamp_cmd_to_box:
            cmd = torch.clamp(cmd, self.q_lo, self.q_hi)
        return cmd

    # ------------------------------------------------------------- commit --
    @torch.no_grad()
    def commit(self, cmd: torch.Tensor, q: torch.Tensor, fired: torch.Tensor) -> None:
        """Shift the plan one waypoint and re-seed the hands that just fired.

        Call AFTER the plant has taken the step and AFTER
        :meth:`BatchedSchedule.advance`, with the command actually executed
        (which may have been modified by the safety filter -- ``prev_cmd`` must
        track what the plant SAW, or the slew bound is computed against a
        command that never happened).
        """
        traj, _hidx = self._pending
        self.prev_cmd = cmd.to(self.dtype)
        self.init = torch.cat([traj[:, 1:], traj[:, -1:]], dim=1)
        new_index = self.sched.hit_index()
        if bool(fired.any()):
            self._reseed(None, fired & (new_index >= 1), q.to(self.dtype), new_index)

    def _reseed(self, env_ids, mask: torch.Tensor, q: torch.Tensor,
                hit_index: torch.Tensor) -> None:
        """Ramp the arm columns of ``mask``'ed hands to their library terminal.

        Mirrors ``MPCRoller._regoal``'s re-seed block: only that hand's arm
        columns move, the other hand and the waist carry straight through, and
        the terminal is the library pose selected at the CURRENT waist-yaw bin
        (the pose library is yaw-indexed, so the entry follows the torso).
        """
        if env_ids is not None:
            keep = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            keep[env_ids] = True
            mask = mask & keep.unsqueeze(-1)
        yaw_bin = torch.round((q[:, self.yaw_i] - self.yaw_lo) / self.yaw_step) \
            .long().clamp(0, self.n_bins - 1)
        drum = self.sched.goal()[1]
        for h in (0, 1):
            rows = mask[:, h].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = self.arm_cols[h]
            d = drum.index_select(0, rows)[:, h].clamp(0, self.D - 1)
            ok = self.lib_ok[h][d]
            term = self.lib_q[h][d, yaw_bin.index_select(0, rows)]
            q0r = q.index_select(0, rows)[:, cols]
            term = torch.where(ok.unsqueeze(-1), term, q0r)
            L = hit_index.index_select(0, rows)[:, h].clamp(1, self.H) \
                .view(-1, 1).expand(-1, len(cols))
            sub = ramp_seed(q0r, term, L, self.H)
            buf = self.init.index_select(0, rows)
            buf[:, :, cols] = sub
            self.init.index_copy_(0, rows, buf)

    # -------------------------------------------------------------- stats --
    @property
    def encode_fraction(self) -> float:
        return self.n_encodes / max(self.n_ticks, 1)
