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
replaced by the shared velocity-matched strike seed; the executed command
still passes through the slew and joint-box bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.fk import ChainFK
from amortized_opt.ik import adjust_arm_terminal, correct_quintic_init_arms
from amortized_opt.lead import LEAD_PRESETS, lead_fraction, lead_waypoints, resolve_lead
from amortized_opt.limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from amortized_opt.model import ContextEncoding
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
from amortized_opt.train_cost import group_lengths, ramp_seed
from amortized_opt.strike_seed import accelerating_path_torch
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice, seed_waist_pr_to_sp, waist_slice

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
    #: Re-encode the context every tick.  DEFAULT OFF as of the decoder-tth
    #: change: the encoding is cached for the life of a goal segment and rebuilt
    #: on a segment boundary or a yaw-BC move (:meth:`IsaacMPC._encode_key`).
    #:
    #: The staleness objection that made every-tick the default is GONE, not
    #: traded away.  ``isaac_rl/README.md`` DECISION 2 defaulted to every-tick
    #: because ``goal_tth`` and ``future_tth`` were both encoder inputs that
    #: ticked down every 20 ms, so a cached encoding fed the decoder a stale
    #: time-to-hit.  ``future_tth`` became the hit-relative ``future_gap`` and
    #: ``goal_tth`` moved to the DECODER (``ModelConfig.dec_tth_feats`` /
    #: ``enc_goal_tth``), which re-runs every tick anyway.  Every remaining
    #: encoder input is now constant inside a goal segment by construction, so
    #: the cache is EXACT, not an approximation -- and the per-tick re-encode it
    #: replaces is the path by which measured state echoed into the encoder and
    #: ran the waist away in closed loop (see ``yaw_cond``).
    #:
    #: ``True`` is kept as the A/B escape hatch.
    encode_every_tick: bool = False
    #: Re-encode when the ``yaw_cond`` conditioning yaw has moved this far, rad,
    #: since the encoding was built.  It is the one encoder input that is not
    #: segment-constant -- a slew-limited ramp toward the analytic aim -- so a
    #: big commanded turn must be allowed to refresh the aim conditioning inside
    #: a long segment.  0.05 rad is ~3 deg, well under the aim law's 0.09 rad
    #: deadband, so the token never drifts far enough to change which side of
    #: that switch the context sits on.
    enc_yaw_tol: float = 0.05
    #: Autocast dtype for the model pass (``None`` = fp32).
    amp_dtype: torch.dtype | None = torch.bfloat16
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)
    yaw_bins: int = 129
    #: Waist pitch/roll ease-to-upright window of the RESET seed, in waypoints
    #: (``warmstart_trajectory``'s ``waist_upright_ease_wp``).  MUST match the
    #: ``ScenarioConfig`` the checkpoint was trained with -- ``play_isaac``
    #: reads it off ``ck["scenario_cfg"]``.  ``None`` = hold the measured lean,
    #: the pre-ease behaviour bit for bit.  (Ported from ``isaac_rl_ppo.mpc``,
    #: which got the field with the 2026-08-13 ease work.)
    waist_upright_ease_wp: int | None = None
    #: What fills the ``yaw_now`` conditioning token (and the aim recompute).
    #: ``"aim_ramp"`` reproduces ``MPCGenConfig.planned_yaw_bc`` -- the rule
    #: every offline loop (mpc_gen, the probe's generator) runs under: the
    #: token is a slew-limited BC state that ramps toward the ANALYTIC aim,
    #: i.e. an anchor exogenous to both the model and the plant, leashed to
    #: the measurement by ``yaw_cond_leash``.  Conditioning on anything the
    #: loop itself produces closes a positive-feedback loop through the
    #: waist-sp head, measured 2026-08-14 on
    #: ``runs/phase2_19m_waist_sp_c/best_probe.pt`` in Isaac: with the
    #: MEASURED yaw the aim token held +37 deg while the head's set-point
    #: chased the yaw from +17 to -17 deg and the torso ratcheted to -95 deg
    #: (the suffix arm mirrored it to +123 deg); with the model's own
    #: ``last_ref0`` as the token the ratchet survived unchanged -- the model
    #: chases its own plan just as happily as it chases the plant.  The same
    #: checkpoint under the analytic-ramp BC (the offline loops) never
    #: ratchets.  ``"measured"`` keeps the old behaviour for the A/B.
    yaw_cond: str = "aim_ramp"
    #: Bound, rad, on the |BC state - measured| offset the token may carry.
    #: Keeps the token honest when the plant is physically held off the ramp
    #: (collision, saturation) without re-admitting the measurement loop --
    #: on the runaway above the plant tracked the command within a degree, so
    #: the leash never engages on that path.
    yaw_cond_leash: float = 0.35
    #: Backtracking acceptance for the learned trajectory proposal.  The
    #: decoder remains the fast proposal generator; a full FusedCost supplied
    #: to IsaacMPC decides whether that proposal is safe to commit.
    descent_accept: bool = False
    descent_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    descent_min_abs: float = 1.0e-4
    descent_min_rel: float = 1.0e-6
    #: Drop the shifted previous plan.  Every tick rebuilds the decoder init
    #: as the constant-acceleration joint-space line from the measured state
    #: to a terminal pose.  That pose is the terminal generator's output from
    #: the first tick of the current goal segment, held fixed until the
    #: segment changes.  A hand whose goal has not changed keeps its arm
    #: columns; a new goal refreshes that hand and the shared waist.
    fresh_accel_init: bool = False
    #: With ``fresh_accel_init``, call the terminal generator again on every
    #: tick from the measured state, instead of holding the goal-start pose.
    fresh_terminal_every_tick: bool = False
    #: Refine once when the scheduled goal changes, then track a fixed plan
    #: until the next goal boundary. Failed feasibility leaves a pose hold.
    commit_plan: bool = False
    plan_max_passes: int = 32
    plan_gate_tol: float = 1.0e-6
    #: "isaac" executes one decoder proposal and lets the plant's existing
    #: hit detector judge it. Analytic constraints remain diagnostic only.
    plan_acceptance: str = "physics"


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
        descent_cost=None,
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
        self.descent_cost = descent_cost
        if self.cfg.descent_accept and self.descent_cost is None:
            raise ValueError("descent_accept=True requires descent_cost")
        if self.cfg.commit_plan:
            if self.cfg.plan_acceptance not in ("physics", "isaac"):
                raise ValueError("plan_acceptance must be physics or isaac")
            if self.B != 1 or self.descent_cost is None:
                raise ValueError("commit_plan requires one environment and a full physics cost")
            if self.cfg.plan_max_passes < 1:
                raise ValueError("plan_max_passes must be positive")
            if self.cfg.iterations != 1:
                raise ValueError("commit_plan checks feasibility after each single decoder pass")
            if self.cfg.descent_accept:
                raise ValueError("commit_plan uses feasibility acceptance, not descent acceptance")
        self._committed_reference = None
        self._committed_goal_key = None
        self.plan_segments = []
        self.plan_attempts = []
        self.plan_feasible = False
        alphas = tuple(float(a) for a in self.cfg.descent_alphas)
        if any(not (0.0 < a <= 1.0) for a in alphas):
            raise ValueError("descent_alphas must lie in (0, 1]")
        if any(a <= b for a, b in zip(alphas, alphas[1:])):
            raise ValueError("descent_alphas must be strictly descending")
        self._descent_alphas = alphas
        self.descent_trials = 0
        self.descent_rejects = 0
        self.last_descent_alpha = torch.ones(
            self.B, device=self.device, dtype=self.dtype)

        lo, hi = urdf_position_limits(self.names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        eta = self.cfg.cmd_slew_eta
        self._vstep = torch.as_tensor(
            isaac_velocity_limits(self.names) * self.dt, device=self.device, dtype=dtype
        )
        self.step_max = None if eta is None else float(eta) * self._vstep
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
        self._ctx: ContextInputs | None = None
        self.last_ref0 = torch.zeros(self.B, self.n, device=self.device, dtype=dtype)
        #: The ``yaw_cond="aim_ramp"`` BC state (fix A's planned yaw), rad.
        self.yaw_bc = torch.zeros(self.B, device=self.device, dtype=dtype)
        #: Segment identity and conditioning yaw AT the cached encoding, plus
        #: the aim the ramp is currently heading for.  ``_aim`` is refreshed
        #: only when the context is rebuilt, which is what keeps the yaw ramp
        #: off the ~0.1 ms/env host-side solve inside ``sched.context``.
        self._enc_key: torch.Tensor | None = None
        self._yaw_enc: torch.Tensor | None = None
        self._aim: torch.Tensor | None = None
        self.n_encodes = 0
        self.n_ticks = 0
        self._fk: ChainFK | None = None
        self._chain_fk()  # build outside inference_mode; reset runs inside it
        self._arm_dls_pending = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._frozen_terminal: torch.Tensor | None = None
        self._frozen_goal_key: torch.Tensor | None = None

    def _chain_fk(self) -> ChainFK:
        if (
            self._fk is None
            or self._fk.device != self.device
            or self._fk.dtype != self.dtype
        ):
            self._fk = ChainFK(device=self.device, dtype=self.dtype)
        return self._fk

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
        if self.cfg.waist_upright_ease_wp is not None:
            # pitch/roll recover to upright instead of holding the lean --
            # same quintic, same 1.875x stretch rule as the training seed
            # (``warmstart_trajectory``'s ``waist_upright_ease_wp`` branch).
            # Isaac resets envs AT REST, so the rest-to-rest ``ramp_seed`` IS
            # the velocity-matched ease with v0 = 0 (``vel_ramp_seed`` reduces
            # to it exactly); a measured-rate BC would need qd at reset, which
            # this signature does not carry.
            pr = [c for c in self.waist_cols if c != self.yaw_i]
            q0w = q0.index_select(0, env_ids)[:, pr]
            step = (ETA_VELOCITY * self._vstep[pr]).clamp_min(1e-9)
            need = torch.ceil(1.875 * q0w.abs() / step)
            L = need.clamp(float(self.cfg.waist_upright_ease_wp), float(self.H))
            sub = ramp_seed(q0w, torch.zeros_like(q0w), L, self.H)
            buf = self.init.index_select(0, env_ids)
            buf[:, :, pr] = sub
            self.init.index_copy_(0, env_ids, buf)
        self.prev_cmd.index_copy_(0, env_ids, q0.index_select(0, env_ids))
        # Anchor the yaw-BC ramp: before the first tick the planned yaw IS the
        # measured one, so ``yaw_cond="aim_ramp"`` starts honest.
        self.yaw_bc.index_copy_(0, env_ids, q0.index_select(0, env_ids)[:, self.yaw_i])
        self._enc = None
        self._ctx = None
        self._enc_key = None
        self._yaw_enc = None
        self._aim = None
        hidx = self.sched.hit_index()
        self._reseed(env_ids, hidx >= 1, q0, hidx)
        self._arm_dls_pending.index_fill_(0, env_ids, True)
        self._committed_reference = None
        self._committed_goal_key = None
        self.plan_segments = []
        self.plan_attempts = []
        self.plan_feasible = False
        if self._frozen_goal_key is not None:
            self._frozen_goal_key = self._frozen_goal_key.clone()
            self._frozen_goal_key[env_ids] = -1

    # -------------------------------------------------------------- masks --
    def masks(self, hit_index: torch.Tensor):
        # ``hit_index`` is WAYPOINT-QUANTIZED, so ``masks.hit_tth`` -- the
        # countdown the decoder reads (``ModelConfig.dec_tth_feats``) -- is
        # quantized too, by at most half a control period (10 ms).  That is
        # below the ~20 ms strike-timing sigma the cost scores against, and it
        # matches training: the phase-1/2 pools are waypoint-aligned, so their
        # ``hit_tth`` is quantized identically.  A continuous path would feed
        # the decoder a resolution the model never saw.
        active = hit_index >= 0
        t = hit_index.clamp_min(0).to(self.dtype) * self.dt
        return build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=self.H, dt=self.dt,
            active_left=active[:, 0], active_right=active[:, 1],
            joint_names=self.names,
        )

    @torch.no_grad()
    def _descent_accept(
        self,
        init: torch.Tensor,
        proposal: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        elem_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Backtrack toward ``init`` until the full cost strictly decreases.

        Selection is per environment.  The first passing alpha is the largest
        admissible step because ``descent_alphas`` is strictly descending.  If
        none pass, alpha zero returns the shifted warm start exactly.
        """
        if not self.cfg.descent_accept:
            one = torch.ones(q.shape[0], device=q.device, dtype=q.dtype)
            nan = torch.full_like(one, float("nan"))
            return proposal, {
                "descent_alpha": one,
                "descent_base_cost": nan,
                "descent_proposal_cost": nan,
                "descent_accepted_cost": nan,
            }
        strike = self.sched.strike_context()
        lengths = group_lengths(strike.hit_index).to(q.dtype)

        def score(traj: torch.Tensor) -> torch.Tensor:
            out = self.descent_cost(
                q, traj, qd, strike, lengths=lengths, elem_mask=elem_mask,
            )
            return out.total.detach().to(dtype=q.dtype)

        base_cost = score(init)
        accepted = init.clone()
        accepted_cost = base_cost.clone()
        alpha_out = torch.zeros_like(base_cost)
        unresolved = torch.ones_like(base_cost, dtype=torch.bool)
        proposal_cost = torch.full_like(base_cost, float("nan"))
        delta = proposal - init
        required = torch.maximum(
            torch.full_like(base_cost, float(self.cfg.descent_min_abs)),
            base_cost.abs() * float(self.cfg.descent_min_rel),
        )
        limit = base_cost - required
        for i, alpha in enumerate(self._descent_alphas):
            candidate = init + float(alpha) * delta
            candidate = candidate * elem_mask.to(candidate.dtype)
            value = score(candidate)
            if i == 0:
                proposal_cost = value
            finite = torch.isfinite(value)
            improves = finite & (
                (~torch.isfinite(base_cost)) | (value <= limit)
            )
            take = unresolved & improves
            if bool(take.any()):
                accepted[take] = candidate[take]
                accepted_cost[take] = value[take]
                alpha_out[take] = float(alpha)
                unresolved[take] = False
        self.descent_trials += int(q.shape[0])
        self.descent_rejects += int(unresolved.sum().item())
        self.last_descent_alpha = alpha_out
        return accepted, {
            "descent_alpha": alpha_out,
            "descent_base_cost": base_cost,
            "descent_proposal_cost": proposal_cost,
            "descent_accepted_cost": accepted_cost,
        }

    # --------------------------------------------------------------- tick --
    def _hold_plan_end(self, traj, q, hit_index):
        """Retain idle joints and hold each group's terminal past its horizon."""
        held = traj.clone()
        for cols, k in [
            (self.arm_cols[0], hit_index[:, 0]),
            (self.arm_cols[1], hit_index[:, 1]),
            (self.waist_cols, hit_index.max(dim=1).values),
        ]:
            end = (k - 1).clamp(0, self.H - 1)
            last = held[:, :, cols].gather(
                1, end[:, None, None].expand(-1, 1, len(cols)))
            last = torch.where((k >= 1)[:, None, None], last, q[:, None, cols])
            beyond = torch.arange(self.H, device=q.device)[None, :, None] >= end[:, None, None]
            held[:, :, cols] = torch.where(beyond, last, held[:, :, cols])
        return held

    def _plan_gate(self, traj, q, qd, masks):
        strike = self.sched.strike_context()
        with torch.autocast(device_type=q.device.type, enabled=False):
            score = self.descent_cost(
                q, traj, qd, strike, lengths=group_lengths(strike.hit_index).to(q.dtype),
                elem_mask=masks.elem_mask,
            )
        values = {name: float(value.detach().max().item())
                  for name, value in score.violations.items()}
        # These are the same hard families used by the frozen-model probes.
        hard = [v for name, v in values.items() if name in ("terminal", "collision")
                or name.startswith(("torque", "joint_box", "velocity"))]
        ok = (bool(torch.isfinite(traj).all()) and bool(torch.isfinite(score.total).all())
              and bool(hard) and all(math.isfinite(v) and v <= self.cfg.plan_gate_tol for v in hard))
        return ok, {"pass": len(self.plan_attempts) + 1, "feasible": ok,
                    "total_cost": float(score.total.item()), "violations": values,
                    "diagnostics": {name: float(value.detach().mean().item())
                                    for name, value in score.diagnostics.items()
                                    if name.startswith("contact_")}}

    def _committed_tick(self, q, hit_index, masks):
        reference = self._committed_reference
        self.last_ref0 = reference[:, 0].clone()
        cmd = self._command(reference, hit_index)
        traj = reference.clone()
        traj[:, 0] = cmd
        self._pending = (traj, hit_index)
        self.n_ticks += 1
        return TickResult(cmd, traj, masks.elem_mask, hit_index, hit_index >= 0,
                          {}, torch.ones(self.B, device=q.device, dtype=torch.bool), False)

    @torch.no_grad()
    def plan(self, q: torch.Tensor, qd: torch.Tensor) -> TickResult:
        """One refinement + one admissible command.  Does NOT step the clock."""
        cfg = self.cfg
        q = q.to(self.dtype)
        qd = qd.to(self.dtype)
        hit_index = self.sched.hit_index()
        active = hit_index >= 0
        masks = self.masks(hit_index)

        if cfg.commit_plan:
            key = self._encode_key(hit_index)
            if (self._committed_reference is not None
                    and torch.equal(key, self._committed_goal_key)):
                return self._committed_tick(q, hit_index, masks)
            if self._committed_reference is not None:
                # A new scheduled request starts a new planning transaction.
                self.init = self._committed_reference.clone()
                self._arm_dls_pending.fill_(True)
            self.plan_attempts = []
            self.plan_feasible = False
            self._committed_goal_key = key.clone()
            if not bool((hit_index >= 1).any()):
                # Stay idle outside the planning horizon, with no model calls.
                self._committed_reference = q[:, None, :].expand(-1, self.H, -1).clone()
                return self._committed_tick(q, hit_index, masks)

        enc, did_encode = self._encoding(hit_index, q)
        core = self.model.module if hasattr(self.model, "module") else self.model
        state = torch.cat([q, qd], dim=-1)
        if cfg.fresh_accel_init:
            self._apply_fresh_accel_init(core, state, masks, hit_index)
        else:
            if getattr(enc, "waist_pr_sp", None) is not None:
                self.init = seed_waist_pr_to_sp(
                    self.init, q, qd, enc.waist_pr_sp,
                    joint_names=self.names, dt=self.dt,
                    ease_wp=self.cfg.waist_upright_ease_wp,
                )
            if bool(self._arm_dls_pending.any()):
                _tth, _d, gp, _w, _v, gactive = self.sched.goal()
                live = (
                    self._arm_dls_pending.unsqueeze(-1)
                    & gactive
                    & (hit_index >= 1)
                )
                if core.terminal_initializer is not None:
                    seed = core.initial_trajectory(self._ctx, state, masks).to(self.dtype)
                    for h in (0, 1):
                        rows = live[:, h]
                        self.init[:, :, self.arm_cols[h]] = torch.where(
                            rows[:, None, None], seed[:, :, self.arm_cols[h]],
                            self.init[:, :, self.arm_cols[h]],
                        )
                    any_live = live.any(dim=-1)
                    self.init[:, :, self.waist_cols] = torch.where(
                        any_live[:, None, None], seed[:, :, self.waist_cols],
                        self.init[:, :, self.waist_cols],
                    )
                else:
                    self.init = correct_quintic_init_arms(
                        self._chain_fk(), self.init, q, hit_index, gp,
                        active=live, steps=1,
                        q_bounds=(self.q_lo, self.q_hi),
                        joint_names=self.names, qd0=qd, dt=self.dt,
                    )
                self._arm_dls_pending.zero_()
        init = self.init * masks.elem_mask

        was = core.training
        core.eval()
        amp = (
            torch.autocast(device_type="cuda", dtype=cfg.amp_dtype)
            if (cfg.amp_dtype is not None and self.device.type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with amp:
            out = core.refine(init, state, enc, masks, iterations=int(cfg.iterations))
            if cfg.commit_plan:
                for pass_index in range(cfg.plan_max_passes):
                    candidate = self._hold_plan_end(out.traj.to(self.dtype), q, hit_index)
                    ok, attempt = self._plan_gate(candidate, q, qd, masks)
                    self.plan_attempts.append(attempt)
                    print(f"[committed-plan] {attempt}", flush=True)
                    if cfg.plan_acceptance == "isaac":
                        self.plan_feasible = ok
                        break
                    if ok:
                        self.plan_feasible = True
                        break
                    if pass_index + 1 < cfg.plan_max_passes:
                        # Same physical state, context, and clock throughout.
                        out = core.refine(candidate * masks.elem_mask, state, enc, masks, iterations=1)
                execute = self.plan_feasible or (
                    cfg.plan_acceptance == "isaac" and bool(torch.isfinite(candidate).all()))
                self._committed_reference = (candidate.clone() if execute
                                             else q[:, None, :].expand(-1, self.H, -1).clone())
                self.plan_segments.append({
                    "tick": self.n_ticks, "time_s": self.n_ticks * self.dt,
                    "goal_key": key.detach().cpu().tolist(),
                    "hit_index": hit_index.detach().cpu().tolist(),
                    "feasible": self.plan_feasible,
                    "executed": execute,
                    "acceptance": cfg.plan_acceptance,
                    "decoder_passes": len(self.plan_attempts),
                    "attempts": list(self.plan_attempts),
                })
        if was:
            core.train()
        if cfg.commit_plan:
            result = self._committed_tick(q, hit_index, masks)
            result.encoded = did_encode
            return result
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

        # The learned pass is a proposal, not an unconditional replacement.
        # Score after freeze because that is the trajectory that would be
        # committed; otherwise the freeze could invalidate the guarantee.
        traj, descent_aux = self._descent_accept(
            init, traj, q, qd, masks.elem_mask,
        )

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
                 "point_mask": out.point_mask, "token_mask": out.token_mask,
                 **descent_aux},
            frozen=frozen, encoded=did_encode,
        )

    def _encode_key(self, hit_index: torch.Tensor) -> torch.Tensor:
        """``(B, 2)`` goal-segment identity -- which scheduled hit each hand owns.

        Derived from the schedule's CURSOR, never from a countdown: ``goal_tth``
        moves every tick and would make every tick a new segment, which is the
        whole thing this change removes.

        The cursor alone is not enough.  A hit further out than ``cap_wp`` is
        demoted to ``future[0]`` and later PROMOTED to the goal without the
        cursor moving (``BatchedSchedule.context``'s ``shift``), and that
        promotion rewrites every goal field.  Pairing the cursor with the goal's
        activity covers the re-goal and the promotion with one integer; a reset
        is covered separately, by ``reset`` clearing the cache outright.
        """
        return self.sched.cursor * 2 + (hit_index >= 0).long()

    def _encoding(self, hit_index: torch.Tensor, q: torch.Tensor | None = None):
        """The context encoding, rebuilt on a SEGMENT boundary, not every tick.

        ``q`` is the measured pose; its waist-yaw column feeds
        ``BatchedSchedule.context(yaw_now=...)`` so a waist-sp-head checkpoint
        is conditioned on the actual torso yaw rather than the field's zero
        default ("hold 0 rad").

        Three things force a rebuild: no cache at all (first tick, or a reset),
        a change of ``_encode_key`` on ANY env, and the conditioning yaw moving
        past ``enc_yaw_tol``.  Nothing else can move an encoder input now -- see
        ``MPCConfig.encode_every_tick``.

        The ``yaw_cond="aim_ramp"`` BC state advances EVERY tick regardless,
        toward the aim of the last encode.  It has to: the ramp is what the
        yaw-move trigger measures, so freezing it between encodes would freeze
        the trigger with it and the aim conditioning would never refresh inside
        a long segment.  Advancing it against the cached aim rather than a fresh
        one keeps ``sched.context``'s host-side aim solve off the cached ticks,
        and re-encoding is exactly what corrects the aim when it has moved.
        """
        cfg = self.cfg
        ramp = cfg.yaw_cond == "aim_ramp" and q is not None
        if q is None:
            yaw_now = None
        elif ramp:
            # fix A's planned-yaw BC: leash the ramp state to reality, then
            # condition on it.
            meas = q[:, self.yaw_i]
            leash = float(cfg.yaw_cond_leash)
            self.yaw_bc = meas + (self.yaw_bc - meas).clamp(-leash, leash)
            yaw_now = self.yaw_bc
        else:
            yaw_now = q[:, self.yaw_i]

        key = self._encode_key(hit_index)
        need = (
            self._enc is None
            or cfg.encode_every_tick
            or self._enc_key is None
            or bool((key != self._enc_key).any())
            or (
                yaw_now is not None and self._yaw_enc is not None
                and bool(((yaw_now - self._yaw_enc).abs() > float(cfg.enc_yaw_tol)).any())
            )
        )
        if need:
            core = self.model.module if hasattr(self.model, "module") else self.model
            amp = (
                torch.autocast(device_type="cuda", dtype=cfg.amp_dtype)
                if (cfg.amp_dtype is not None and self.device.type == "cuda")
                else torch.autocast(device_type="cuda", enabled=False)
            )
            ctx = self.sched.context(yaw_now=yaw_now)
            self._ctx = ctx
            if ramp and ctx.aim_yaw is not None:
                self._aim = ctx.aim_yaw.to(self.yaw_bc.dtype)
            with amp:
                self._enc = core.encode_context(ctx)
            self._enc_key = key.clone()
            # ``.clone()``: ``reset`` writes ``yaw_bc`` IN PLACE, so the
            # reference would follow it and the trigger would never fire.
            self._yaw_enc = None if yaw_now is None else yaw_now.clone()
            self.n_encodes += 1
        if ramp and self._aim is not None:
            step = float(ETA_VELOCITY) * self._vstep[self.yaw_i]
            self.yaw_bc = self.yaw_bc + (self._aim - self.yaw_bc).clamp(-step, step)
        return self._enc, need

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
    def commit(self, cmd: torch.Tensor, q: torch.Tensor, fired: torch.Tensor,
               *, qd: torch.Tensor | None = None) -> None:
        """Shift the plan one waypoint and re-seed the hands that just fired.

        Call AFTER the plant has taken the step and AFTER
        :meth:`BatchedSchedule.advance`, with the command actually executed
        (which may have been modified by the safety filter -- ``prev_cmd`` must
        track what the plant SAW, or the slew bound is computed against a
        command that never happened).
        """
        traj, _hidx = self._pending
        self.prev_cmd = cmd.to(self.dtype)
        if self.cfg.commit_plan:
            # Shift the un-led plan, never the controller's adjusted commands.
            ref = self._committed_reference
            self._committed_reference = torch.cat([ref[:, 1:], ref[:, -1:]], dim=1)
            self.init = self._committed_reference.clone()
            return
        if self.cfg.fresh_accel_init:
            # The next plan() redraws the analytic line from the measured
            # state.  Shifting the refined trajectory here would put the
            # previous plan back in the warm start.
            return
        self.init = torch.cat([traj[:, 1:], traj[:, -1:]], dim=1)
        new_index = self.sched.hit_index()
        if bool(fired.any()):
            self._reseed(None, fired & (new_index >= 1), q.to(self.dtype), new_index, qd=qd)

    def _apply_fresh_accel_init(self, core, state, masks, hit_index) -> None:
        """Redraw the analytic seed onto the goal-start terminal pose."""
        initializer = getattr(core, "terminal_initializer", None)
        if initializer is None:
            raise RuntimeError(
                "fresh_accel_init requires a model with a terminal pose head"
            )
        if self._ctx is None:
            raise RuntimeError("fresh_accel_init ran before the context was built")
        key = self._encode_key(hit_index)
        if self.cfg.fresh_terminal_every_tick:
            self._frozen_terminal = initializer.terminal_pose(
                self._ctx, state, masks
            ).to(self.dtype)
            self._frozen_goal_key = key.clone()
            self.init = initializer.seed_from_terminal(
                self._ctx, state, masks, self._frozen_terminal
            ).to(self.dtype)
            return
        if self._frozen_terminal is None or self._frozen_goal_key is None:
            self._frozen_terminal = initializer.terminal_pose(
                self._ctx, state, masks
            ).to(self.dtype)
            self._frozen_goal_key = key.clone()
        elif bool((key != self._frozen_goal_key).any()):
            pred = initializer.terminal_pose(self._ctx, state, masks).to(self.dtype)
            changed = key != self._frozen_goal_key
            frozen = self._frozen_terminal
            for h in (0, 1):
                take = changed[:, h].view(-1, 1)
                cols = self.arm_cols[h]
                frozen[:, cols] = torch.where(take, pred[:, cols], frozen[:, cols])
            take_w = changed.any(dim=-1).view(-1, 1)
            wc = self.waist_cols
            frozen[:, wc] = torch.where(take_w, pred[:, wc], frozen[:, wc])
            self._frozen_terminal = frozen
            self._frozen_goal_key = key.clone()
        self.init = initializer.seed_from_terminal(
            self._ctx, state, masks, self._frozen_terminal
        ).to(self.dtype)

    def _reseed(self, env_ids, mask: torch.Tensor, q: torch.Tensor,
                hit_index: torch.Tensor, *, qd: torch.Tensor | None = None) -> None:
        """Ramp arm columns of ``mask``'ed hands; rebuild waist from aim.

        Mirrors ``MPCRoller._regoal``: only the fired hand's arm columns move,
        the other hand carries through, and the waist is rewritten from the
        analytic aim whenever ANY hand reseeds (not held leftover).
        """
        if env_ids is not None:
            keep = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            keep[env_ids] = True
            mask = mask & keep.unsqueeze(-1)
        any_rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        self._rebuild_waist_from_aim(any_rows, q)
        yaw_src = q[:, self.yaw_i]
        if self._aim is not None:
            yaw_src = self._aim.to(dtype=q.dtype)
        yaw_bin = torch.round((yaw_src - self.yaw_lo) / self.yaw_step) \
            .long().clamp(0, self.n_bins - 1)
        drum = self.sched.goal()[1]
        _tth, _d, gp, _w, _v, _gactive = self.sched.goal()
        n = self.n
        for h in (0, 1):
            rows = mask[:, h].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = self.arm_cols[h]
            hand = "left" if h == 0 else "right"
            d = drum.index_select(0, rows)[:, h].clamp(0, self.D - 1)
            ok = self.lib_ok[h][d]
            term_lib = self.lib_q[h][d, yaw_bin.index_select(0, rows)]
            q0r = q.index_select(0, rows)
            term_lib = torch.where(ok.unsqueeze(-1), term_lib, q0r[:, cols])
            k = hit_index.index_select(0, rows)[:, h].clamp(1, self.H)
            buf = self.init.index_select(0, rows)
            q_seed = buf.gather(
                1, (k - 1).view(-1, 1, 1).expand(-1, 1, n)
            ).squeeze(1).clone()
            q_seed[:, cols] = term_lib
            q_fix = adjust_arm_terminal(
                self._chain_fk(), q_seed, gp.index_select(0, rows)[:, h], hand,
                steps=1, q_bounds=(self.q_lo, self.q_hi),
                joint_names=self.names,
            )
            term = q_fix[:, cols]
            v0r = (torch.zeros_like(q0r) if qd is None else
                   qd.to(q0r).index_select(0, rows))
            sub = accelerating_path_torch(
                q0r[:, cols], v0r[:, cols], term, k, self.H, self.dt,
            ).clamp(self.q_lo[cols], self.q_hi[cols])
            buf = buf.clone()
            buf[:, :, cols] = sub
            self.init.index_copy_(0, rows, buf)
        self._arm_dls_pending[any_rows] = True

    def _rebuild_waist_from_aim(self, rows: torch.Tensor, q: torch.Tensor) -> None:
        """Waist ``init`` from analytic aim (``waist_reference_block``), sparse.

        Aim comes from ``sched.context`` (same future-aware law as encode), so
        the waist seed and the conditioning token describe the same drums after
        a single-hand promote.
        """
        if rows.numel() == 0:
            return
        from amortized_opt.limits import ETA_VELOCITY, isaac_velocity_limits
        from amortized_opt.waist_aim import waist_reference_block

        wc = self.waist_cols
        H, dt = self.H, self.dt
        waist_names = [self.names[c] for c in wc]
        jy = waist_names.index("waist_yaw_joint")
        v_lim = isaac_velocity_limits(self.names)
        yaw_step = float(ETA_VELOCITY) * float(v_lim[self.yaw_i]) * float(dt)

        # Match ``_encoding``'s yaw_cond: aim_ramp conditions on the BC state.
        yaw_cond = q[:, self.yaw_i]
        if self.cfg.yaw_cond == "aim_ramp" and self.yaw_bc is not None:
            yaw_cond = self.yaw_bc
        ctx = self.sched.context(yaw_now=yaw_cond)
        if ctx.aim_yaw is not None:
            self._aim = ctx.aim_yaw.to(
                self.yaw_bc.dtype if self.yaw_bc is not None else self.dtype
            )

        n = int(rows.numel())
        aim = ctx.aim_yaw.index_select(0, rows).detach().cpu().numpy()
        aval = ctx.aim_valid.index_select(0, rows).detach().cpu().numpy()
        q0_np = q.index_select(0, rows).detach().cpu().numpy()
        block = np.zeros((n, H, len(wc)), dtype=np.float64)
        ease_wp = self.cfg.waist_upright_ease_wp
        for i in range(n):
            yaw_start = float(q0_np[i, self.yaw_i])
            yaw_tgt = float(aim[i]) if bool(aval[i]) else yaw_start
            if ease_wp is not None:
                sp = np.zeros(len(wc), dtype=float)
            else:
                sp = np.array([float(q0_np[i, c]) for c in wc], dtype=float)
            sp[jy] = yaw_start
            ref = waist_reference_block(
                waist_names, yaw_start=yaw_start, yaw_target=yaw_tgt,
                n_waypoints=H + 1, setpoint=sp, dt=dt, settle_wp=3,
                yaw_rate_start=0.0, max_move_s=0.2,
            )
            yaw = ref[:, jy].copy()
            for k in range(1, yaw.shape[0]):
                d = yaw[k] - yaw[k - 1]
                if d > yaw_step:
                    yaw[k] = yaw[k - 1] + yaw_step
                elif d < -yaw_step:
                    yaw[k] = yaw[k - 1] - yaw_step
            ref[:, jy] = yaw
            block[i] = ref[1:]
            if ease_wp is not None:
                pr = [j for j, c in enumerate(wc) if c != self.yaw_i]
                for j in pr:
                    q_from = float(q0_np[i, wc[j]])
                    step_lim = max(ETA_VELOCITY * float(v_lim[wc[j]]) * dt, 1e-9)
                    need = int(np.ceil(1.875 * abs(q_from) / step_lim))
                    k_ease = int(np.clip(max(int(ease_wp), need), 1, H))
                    qq = torch.tensor([[q_from]], dtype=self.dtype, device=self.device)
                    tt = torch.zeros_like(qq)
                    LL = torch.full_like(qq, float(k_ease))
                    block[i, :, j] = ramp_seed(qq, tt, LL, H)[0, :, 0].cpu().numpy()

        sub = torch.as_tensor(block, device=self.device, dtype=self.dtype)
        buf = self.init.index_select(0, rows)
        buf[:, :, wc] = sub
        self.init.index_copy_(0, rows, buf)

    # -------------------------------------------------------------- stats --
    @property
    def encode_fraction(self) -> float:
        return self.n_encodes / max(self.n_ticks, 1)
