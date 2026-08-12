"""The RL reward: scored on the EXECUTED state, not on the plan.

`REPORT.md` §8, the precondition:

    the reward must evaluate the terminal geometry at the pose the plant
    reached, not the pose the plan nominated.  In simulation this is free --
    Isaac *is* the plant.

    keep every existing analytic term (torque, velocity, joint box, collision,
    smoothness, posture, and the auxiliary heads) as a differentiable loss on
    the *plan*, and add the hit reward on the *executed* contact.  The
    differentiable half keeps the un-executed tail of the trajectory valid; the
    RL half carries the signal backprop cannot see.

Four families, all logged separately so the decomposition is recorded rather
than assumed:

``hit.*``
    From the SIMULATED contact.  A batched, executed-state twin of
    ``amortized_opt.probe.hit_latency``: the tip (or the shaft contact point on
    cymbals -- ``contact_sigma``) is tracked across the tick, the crossing of
    the head plane is located to SUB-TICK resolution, and the crossing is
    scored for legal disk, timing against the scheduled tick, and impact speed
    band.  Section 7c is explicit that this, not the at-tick binary, is the
    metric that ranks configurations: "in the disk at tick k" conflates *late*
    with *missed*.

``extra.*``
    The stray-hit punishment: contact with any drum that is not the hand's
    current goal, and contact outside the timing window.  Mirrors the existing
    env's extra-hit mechanism (the plan doc: "refer to the current MLP RL
    system ... just take a look at the hit counter and extra hit punishment
    mechanisms").

``plan.*``
    ``FusedCost`` on the remainder of the plan -- torque, velocity, joint box,
    collision, smoothness, posture, terminal.  Negated into a reward.  The plan
    doc: "we will continue including the previous limit and smoothness terms
    ... making sure it still generates valid paths to the drum at all times in
    our RL reward function."

``track.*``
    PD-tracking error **with the lead accounted for**.  The command executed at
    tick ``k`` was ``q_ref(t_k + delta)``; the *intent* is that the plant lands
    on ``q_ref(t_k)``, i.e. on the UN-led first waypoint of that tick's plan.
    So the error scored here is ``||q_measured(t_{k+1}) - traj_k[0]_pre_lead||``
    -- not against the command, which would score the lead itself as error and
    train the model to cancel its own compensation.

``filter.*``
    QP-safety-filter interventions.  Counted now, penalized in the training
    stage (plan item 3: "filter interventions penalized so the policy does not
    learn to lean on it").

Isaac-free.  The env hands in tensors; nothing here imports a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV
from amortized_opt.cost import CostTerms, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK, stick_axis_local, stick_mount_transform, stick_tip_offset
from amortized_opt.train_cost import group_lengths

__all__ = [
    "ContactScore",
    "ExecutedStrikeScorer",
    "RewardConfig",
    "RewardTerms",
    "reward_from_terms",
]

HANDS = ("left", "right")


# ------------------------------------------------------------------ config --
@dataclass
class RewardConfig:
    """Weights.  Every term is logged raw as well, so a weight of 0 still shows."""

    # ---- executed contact (RL priority 1) ----
    #: Landing inside the legal disk at the crossing.  The headline term.
    w_hit: float = 10.0
    #: Timing.  ``exp(-(latency/tau)^2)``; tau = 20 ms = one control period,
    #: which is the resolution the 7c latency table reports in.
    w_timing: float = 5.0
    timing_tau_s: float = 0.02
    #: Impact speed inside ``[speed_lo, speed_hi]``.  7c reports "speed in band"
    #: as a separate column because a strike can be on time and inaudible.
    w_speed: float = 2.0
    speed_lo: float = 1.0
    speed_hi: float = 3.5
    #: A scheduled hit whose stick never crossed the head plane inside the disk
    #: in the window.  This is the 63 % the RL stage exists to move.
    w_miss: float = 5.0
    #: +/- this many control ticks around the scheduled instant count as the
    #: crossing window (``probe.hit_latency`` uses 5).
    window_ticks: int = 5

    # ---- stray contact ----
    #: Contact with a drum that is not the hand's goal.
    w_extra_hit: float = 5.0
    #: Contact with the goal drum outside the timing window (double bounce,
    #: early arrival that was not the scheduled strike).
    w_early_contact: float = 2.0

    # ---- the analytic half, on the plan ----
    #: Scale on the (negated) ``FusedCost`` total.  Small: the analytic cost is
    #: O(1..100) in its own units and it is a CONSTRAINT-keeping term here, not
    #: the objective -- the objective is the contact.
    w_plan_cost: float = 0.05
    #: Which FusedCost terms enter the reward.  The terminal terms are dropped
    #: by default: in Isaac the executed contact IS the terminal objective, and
    #: keeping the kinematic terminal too re-creates exactly the conflict §7c
    #: identified (a plan that executes on the head is kinematically ~140 mm
    #: through it, which the kinematic terminal scores as a miss AND a
    #: collision).  Set ``plan_terms=None`` to price the whole total.
    plan_terms: tuple[str, ...] | None = (
        "joint_box", "velocity", "accel", "torque", "torque_peak",
        "collision", "smoothness", "posture_mid",
    )

    # ---- tracking ----
    #: Weight on ``exp(-(rms_err / scale)^2)``; scale in radians.
    w_track: float = 1.0
    track_scale_rad: float = 0.05

    # ---- idle arm (RL priority 4) ----
    #: Soft pull of an IDLE hand's arm columns toward the yaw-parameterized,
    #: clearance-maximizing home pose of ``amortized_opt.idle_home``.  README's
    #: "Design note: the idle arm" specifies this as a ``FusedCost`` term
    #: (``w_idle_home``), but that term was never landed in ``cost.py`` and
    #: ``amortized_opt`` is not ours to change, so it lives here as a reward
    #: term on the PLAN instead -- same quantity, same table, applied to the
    #: same waypoints, just priced on the RL side of the objective.
    #: Deliberately small (README: "weight it well below w_terminal_region and
    #: w_collision ... so a mid-trajectory re-activation pulls the arm out of
    #: home smoothly and it is never a hard command").
    w_idle_home: float = 0.0
    #: Scale (rad) of the idle-home error before squaring, so the term is O(1).
    idle_home_scale_rad: float = 0.5
    #: Override the table path (``None`` = ``assets/idle_home_poses.json``).
    idle_home_path: str | None = None

    # ---- safety filter ----
    #: Counted now, penalized later.  Default 0 for the scaffold stage.
    w_filter: float = 0.0

    #: Alive bonus, so an env that has not reached a hit yet has a defined
    #: (small, constant) return rather than only negative regularizers.
    w_alive: float = 0.0

    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)


@dataclass
class RewardTerms:
    """Per-env reward decomposition.  Every tensor is ``(B,)``."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    raw: dict[str, torch.Tensor] = field(default_factory=dict)
    counts: dict[str, torch.Tensor] = field(default_factory=dict)

    def log(self) -> dict[str, float]:
        out = {"reward/total": float(self.total.mean())}
        for grp in ("terms", "raw", "counts"):
            for k, v in getattr(self, grp).items():
                out[f"{grp}/{k}"] = float(v.float().mean())
        return out


def reward_from_terms(t: RewardTerms) -> torch.Tensor:
    return t.total


# ------------------------------------------------------- executed contact ---
@dataclass
class ContactScore:
    """Sub-tick crossing report for one control tick, per hand.  ``(B, 2)``."""

    crossed: torch.Tensor      # bool -- the contact point crossed the head plane
    in_disk: torch.Tensor      # bool -- and did so inside the legal region
    frac: torch.Tensor         # float in [0, 1] -- where in the tick it crossed
    speed: torch.Tensor        # float m/s at the interpolated pose
    axial_m: torch.Tensor      # float -- signed distance to the head, start of tick
    inplane_m: torch.Tensor    # float -- >0 = outside the disk edge, at crossing
    drum: torch.Tensor         # long -- the goal drum it was tested against


class ExecutedStrikeScorer:
    """Batched, executed-state head-frame geometry, on the SIMULATED pose.

    This is the torch/GPU twin of ``amortized_opt.probe.head_frame_error`` plus
    the crossing search of ``probe.hit_latency``.  The offline version scans a
    recorded ``q_hist`` after the episode; here the same quantity is produced
    *online*, one tick at a time, from ``(q_before, q_after)``, because the
    reward has to be emitted on the tick.

    Two geometry details are the reason this cannot be a naive tip-vs-disk test
    and are ported rather than re-derived:

    * the contact point is ``tip + sigma * u`` along the stick, where ``sigma``
      is ``strike.default_contact_sigma`` -- on a cymbal or hi-hat the legal
      contact runs up to ~15 cm from the tip along the SHAFT, and the impact
      speed requirement applies at the contact point, not at the tip;
    * the legal region is a disk about ``disk_center`` (which is offset from
      the head centre on cymbals and toms) intersected with a half-plane
      (``has_half_plane``), which is what keeps a tom strike on the near side.
    """

    def __init__(self, *, fk: ChainFK, dt: float = CONTROL_DT, device=None,
                 dtype: torch.dtype = torch.float32):
        self.fk = fk
        self.dt = float(dt)
        self.device = torch.device(device or fk.device if hasattr(fk, "device") else "cpu")
        self.dtype = dtype
        t = lambda a: torch.as_tensor(np.asarray(a, float),  # noqa: E731
                                      device=self.device, dtype=dtype)
        #: EXACTLY ``FusedCost._terminal``'s construction: the contact point is
        #: ``base + sigma * u`` with ``base`` the stick MOUNT (the stick's
        #: centre) in world and ``sigma`` measured along the stick axis from
        #: there, so ``sigma = STICK_LENGTH/2`` is the tip and smaller values
        #: walk back toward the hand.  Measuring sigma from the TIP instead --
        #: the intuitive reading -- puts a cymbal contact 30 cm off.
        self._mount_p = {h: t(stick_mount_transform(h)[1]) for h in HANDS}
        self._axis = {h: t(stick_axis_local(h)) for h in HANDS}
        self._tip = {h: t(stick_tip_offset(h)) for h in HANDS}

    def contact_points(self, q: torch.Tensor, sigma: torch.Tensor):
        """``(B, 2, 3)`` world contact points and ``(B, 2, 3)`` stick directions.

        ``sigma`` is ``(B, 2)``, the arc coordinate along the stick axis from
        the mount origin.  Frames are the pelvis frame, matching ``ChainFK``.
        """
        pts, dirs = [], []
        for hi, hand in enumerate(HANDS):
            R, p = self.fk.frame(q, f"{hand}_wrist")
            base = p + torch.einsum("bij,j->bi", R, self._mount_p[hand].to(R.dtype))
            u = torch.einsum("bij,j->bi", R, self._axis[hand].to(R.dtype))
            pts.append(base + sigma[:, hi:hi + 1] * u)
            dirs.append(u)
        return torch.stack(pts, dim=1), torch.stack(dirs, dim=1)

    def head_frame(self, pts: torch.Tensor, ctx: StrikeContext):
        """``(axial, inplane)`` of ``(B, 2, 3)`` points in each hand's head frame.

        ``axial > 0`` = short of the head (on the outward side);
        ``inplane > 0`` = outside the legal disk edge.  Identical convention to
        ``probe.head_frame_error``, so the numbers are directly comparable to
        every table in `REPORT.md` §4 and README §7c.
        """
        n = ctx.n_out
        d = pts - ctx.p_star
        axial = (d * n).sum(-1)
        radial = d - axial.unsqueeze(-1) * n
        # the legal disk is centred at p_star + disk_offset, in the head plane
        r = radial - ctx.disk_offset
        r = r - (r * n).sum(-1, keepdim=True) * n
        dist = r.norm(dim=-1)
        inplane = dist - ctx.hit_radius
        # half-plane cut: a point on the far side is illegal regardless of the
        # radius, so the in-plane coordinate takes the worse of the two.
        hp = (r * ctx.half_plane_dir).sum(-1)
        inplane = torch.where(ctx.has_half_plane > 0.5,
                              torch.maximum(inplane, -hp), inplane)
        return axial, inplane

    @torch.no_grad()
    def score(self, q0: torch.Tensor, q1: torch.Tensor, ctx: StrikeContext,
              *, substeps: int = 1) -> ContactScore:
        """Did the contact point cross the head plane between ``q0`` and ``q1``?

        The crossing is located by linear interpolation of the AXIAL coordinate
        (``probe.hit_latency`` does exactly this, and for the same reason: a
        20 ms grid is far too coarse for a +/-10 ms question), and the in-plane
        test and the impact speed are evaluated at the interpolated pose rather
        than at the nearer knot.
        """
        sigma = ctx.contact_sigma
        p0, _u0 = self.contact_points(q0, sigma)
        p1, _u1 = self.contact_points(q1, sigma)
        a0, _i0 = self.head_frame(p0, ctx)
        a1, _i1 = self.head_frame(p1, ctx)
        crossed = (a0 > 0.0) & (a1 <= 0.0) & (ctx.active > 0.5)
        u = (a0 / (a0 - a1).clamp_min(1e-12)).clamp(0.0, 1.0)
        qc = q0 + u.max(dim=1).values.unsqueeze(-1) * (q1 - q0)
        pc, _uc = self.contact_points(qc, sigma)
        ac, ic = self.head_frame(pc, ctx)
        speed = (p1 - p0).norm(dim=-1) / self.dt
        return ContactScore(
            crossed=crossed,
            in_disk=crossed & (ic <= 0.0),
            frac=u,
            speed=speed,
            axial_m=a0,
            inplane_m=ic,
            drum=ctx.target_drum,
        )

    @torch.no_grad()
    def any_drum_contact(self, q0: torch.Tensor, q1: torch.Tensor,
                         heads: torch.Tensor, axes: torch.Tensor,
                         radii: torch.Tensor, valid: torch.Tensor,
                         sigma: torch.Tensor) -> torch.Tensor:
        """``(B, 2, D)`` bool: which drum heads the contact point crossed.

        The stray-hit detector.  Cheap: one FK pass per endpoint (shared with
        :meth:`score` by the caller) and a broadcast against the ``D`` heads.
        """
        p0, _ = self.contact_points(q0, sigma)
        p1, _ = self.contact_points(q1, sigma)
        h = heads.unsqueeze(1)                                    # (B,1,D,3)
        n = axes.unsqueeze(1)
        d0 = ((p0.unsqueeze(2) - h) * n).sum(-1)                  # (B,2,D)
        d1 = ((p1.unsqueeze(2) - h) * n).sum(-1)
        u = (d0 / (d0 - d1).clamp_min(1e-12)).clamp(0.0, 1.0).unsqueeze(-1)
        pc = p0.unsqueeze(2) + u * (p1.unsqueeze(2) - p0.unsqueeze(2))
        rad = pc - h
        rad = rad - ((rad * n).sum(-1, keepdim=True)) * n
        inside = rad.norm(dim=-1) <= radii.unsqueeze(1)
        return (d0 > 0.0) & (d1 <= 0.0) & inside & valid.unsqueeze(1)


# -------------------------------------------------------------- idle arm ----
class IdleHomePenalty:
    """``(q_arm - home(yaw))^2`` on the waypoints of a hand with no goal.

    The home pose is **indexed by the plan's own waist yaw at that waypoint**,
    not by a world-fixed pose -- README's design note is explicit that a
    world-fixed home would fight the waist.  The table
    (``amortized_opt.idle_home``, 2 hands x 17 yaw bins x 7 arm joints) is
    already clearance-maximized offline, so this term needs no SDF evaluation
    at training time; it is a lookup and a masked quadratic.

    Returns ``(B,)`` mean squared error in units of ``scale_rad``, already
    masked to idle hands and to the live element mask.
    """

    def __init__(self, *, joint_names=G1_UPPER_BODY_JOINTS, path=None,
                 device=None, dtype: torch.dtype = torch.float32):
        from amortized_opt.idle_home import load_idle_home_table
        from amortized_opt.warmstart import arm_slice

        tab = load_idle_home_table(path, joint_names=joint_names)
        q_arm, yaw = tab.torch(device=device, dtype=dtype)
        self.q_arm = q_arm                    # (2, n_bins, n_arm)
        self.yaw = yaw                        # (n_bins,) ascending
        self.cols = [list(arm_slice(h, tuple(joint_names))) for h in HANDS]
        self.yaw_i = tuple(joint_names).index("waist_yaw_joint")
        self.device = device
        self.dtype = dtype

    def __call__(self, traj: torch.Tensor, active: torch.Tensor,
                 elem_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``traj`` ``(B, H, n)``; ``active`` ``(B, 2)`` float 0/1 (1 = has a goal)."""
        B, H, _n = traj.shape
        yaw = traj[:, :, self.yaw_i]                                   # (B, H)
        # linear interpolation over the yaw grid, clamped at both ends
        g = self.yaw.to(traj.dtype)
        idx = torch.searchsorted(g, yaw.reshape(-1).contiguous().clamp(g[0], g[-1]))
        hi = idx.clamp(1, g.numel() - 1)
        lo = hi - 1
        w = ((yaw.reshape(-1).clamp(g[0], g[-1]) - g[lo]) /
             (g[hi] - g[lo]).clamp_min(1e-12)).unsqueeze(-1)
        out = traj.new_zeros(B)
        for h in (0, 1):
            tbl = self.q_arm[h].to(traj.dtype)                         # (bins, n_arm)
            home = (1 - w) * tbl[lo] + w * tbl[hi]                     # (B*H, n_arm)
            home = home.reshape(B, H, -1)
            cols = self.cols[h]
            err = (traj[:, :, cols] - home) / float(1.0)
            m = torch.ones_like(err) if elem_mask is None else elem_mask[:, :, cols]
            m = m * (1.0 - active[:, h]).view(B, 1, 1)
            out = out + (err.square() * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)
        return out


# ----------------------------------------------------------------- reward ---
class ExecutedReward:
    """Assembles :class:`RewardTerms` from one control tick.

    ``step`` is called once per control tick, AFTER the plant has taken its
    ``decimation`` physics substeps and AFTER the schedule has advanced.
    """

    def __init__(self, cfg: RewardConfig, *, scorer: ExecutedStrikeScorer,
                 cost: FusedCost | None = None, device=None,
                 dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.scorer = scorer
        self.cost = cost
        self.device = torch.device(device or scorer.device)
        self.dtype = dtype
        self.idle: IdleHomePenalty | None = None
        if float(cfg.w_idle_home) != 0.0:
            self.idle = IdleHomePenalty(joint_names=cfg.joint_names,
                                        path=cfg.idle_home_path,
                                        device=self.device, dtype=dtype)

    @torch.no_grad()
    def step(
        self,
        *,
        q_before: torch.Tensor,          # (B, 17) measured, start of the tick
        q_after: torch.Tensor,           # (B, 17) measured, end of the tick
        qd_after: torch.Tensor,          # (B, 17)
        ctx: StrikeContext,              # THIS tick's strike rows (pre-advance)
        hit_index: torch.Tensor,         # (B, 2) long, pre-advance
        fired: torch.Tensor,             # (B, 2) bool, from BatchedSchedule.advance
        ref0: torch.Tensor,              # (B, 17) the plan's UN-LED first waypoint
        traj: torch.Tensor | None = None,        # (B, H, 17) for the analytic half
        elem_mask: torch.Tensor | None = None,
        q_plan0: torch.Tensor | None = None,     # (B, 17) plan boundary condition
        v_plan0: torch.Tensor | None = None,
        drum_heads: torch.Tensor | None = None,  # (B, D, 3) pelvis frame
        drum_axes: torch.Tensor | None = None,
        drum_radii: torch.Tensor | None = None,
        drum_valid: torch.Tensor | None = None,
        filter_active: torch.Tensor | None = None,   # (B,) bool
    ) -> tuple[RewardTerms, ContactScore]:
        cfg = self.cfg
        B = q_after.shape[0]
        z = torch.zeros(B, device=self.device, dtype=self.dtype)
        terms: dict[str, torch.Tensor] = {}
        raw: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}

        # ---- 1. executed contact -------------------------------------------
        cs = self.scorer.score(q_before, q_after, ctx)
        near = (hit_index >= 1) & (hit_index <= int(cfg.window_ticks))
        # a crossing counts for the scheduled strike when it happens inside the
        # window around the scheduled tick; anything else is a stray.
        sched = cs.in_disk & (near | fired)
        lat_s = torch.where(
            sched,
            ((hit_index.to(self.dtype) - 1.0) + cs.frac) * self.scorer.dt,
            torch.zeros_like(cs.frac),
        )
        timing = torch.exp(-(lat_s / float(cfg.timing_tau_s)) ** 2) * sched
        spd_ok = (cs.speed >= cfg.speed_lo) & (cs.speed <= cfg.speed_hi) & sched
        terms["hit"] = cfg.w_hit * sched.to(self.dtype).sum(1)
        terms["timing"] = cfg.w_timing * timing.sum(1)
        terms["speed"] = cfg.w_speed * spd_ok.to(self.dtype).sum(1)
        raw["hit_latency_s"] = lat_s.sum(1)
        raw["contact_speed"] = (cs.speed * sched).sum(1)
        counts["scheduled_hits"] = sched.to(self.dtype).sum(1)

        # a hand that FIRED without a legal crossing anywhere in the window is
        # the miss the RL stage exists to remove.  The env tracks the window
        # across ticks and passes ``fired``; ``sched`` is this tick's crossing.
        missed = fired & ~cs.in_disk
        terms["miss"] = -cfg.w_miss * missed.to(self.dtype).sum(1)
        counts["fired"] = fired.to(self.dtype).sum(1)
        counts["missed"] = missed.to(self.dtype).sum(1)

        # ---- 2. stray contact ----------------------------------------------
        if drum_heads is not None:
            hits_any = self.scorer.any_drum_contact(
                q_before, q_after, drum_heads, drum_axes, drum_radii, drum_valid,
                ctx.contact_sigma,
            )                                                     # (B, 2, D)
            goal_oh = torch.nn.functional.one_hot(
                ctx.target_drum.clamp_min(0), hits_any.shape[2]
            ).bool() & (ctx.target_drum >= 0).unsqueeze(-1)
            wrong = (hits_any & ~goal_oh).any(dim=2)
            early = (hits_any & goal_oh).any(dim=2) & ~sched
            terms["extra_hit"] = -cfg.w_extra_hit * wrong.to(self.dtype).sum(1)
            terms["early_contact"] = -cfg.w_early_contact * early.to(self.dtype).sum(1)
            counts["extra_hits"] = wrong.to(self.dtype).sum(1)
            counts["early_contacts"] = early.to(self.dtype).sum(1)
        else:
            terms["extra_hit"] = z.clone()
            terms["early_contact"] = z.clone()

        # ---- 3. the analytic half, on the plan ------------------------------
        if self.cost is not None and traj is not None:
            q0 = q_plan0 if q_plan0 is not None else q_before
            v0 = v_plan0 if v_plan0 is not None else torch.zeros_like(q0)
            lengths = group_lengths(ctx.hit_index).to(self.dtype)
            ct: CostTerms = self.cost(q0, traj, v0, ctx, lengths=lengths,
                                      elem_mask=elem_mask)
            if cfg.plan_terms is None:
                pen = ct.total
            else:
                pen = sum((ct.terms[k] for k in cfg.plan_terms if k in ct.terms),
                          torch.zeros_like(ct.total))
            terms["plan_cost"] = -cfg.w_plan_cost * pen
            raw["plan_cost_total"] = ct.total.detach()
            for k, v in ct.terms.items():
                raw[f"plan.{k}"] = v.detach()
            for k, v in ct.violations.items():
                raw[f"viol.{k}"] = v.detach()
        else:
            terms["plan_cost"] = z.clone()

        # ---- 4. tracking, with the lead accounted for -----------------------
        err = (q_after - ref0)
        rms = err.square().mean(dim=1).sqrt()
        terms["track"] = cfg.w_track * torch.exp(-(rms / float(cfg.track_scale_rad)) ** 2)
        raw["track_rms_rad"] = rms

        # ---- 4b. idle-arm home posture (RL priority 4) ------------------------
        if self.idle is not None and traj is not None:
            e = self.idle(traj, ctx.active, elem_mask)
            terms["idle_home"] = -cfg.w_idle_home * (
                e / (float(cfg.idle_home_scale_rad) ** 2))
            raw["idle_home_mse_rad2"] = e
        else:
            terms["idle_home"] = z.clone()

        # ---- 5. safety filter ------------------------------------------------
        if filter_active is not None:
            terms["filter"] = -cfg.w_filter * filter_active.to(self.dtype)
            counts["filter_interventions"] = filter_active.to(self.dtype)
        else:
            terms["filter"] = z.clone()

        terms["alive"] = torch.full_like(z, float(cfg.w_alive))
        total = sum(terms.values(), torch.zeros_like(z))
        return RewardTerms(total=total, terms=terms, raw=raw, counts=counts), cs
