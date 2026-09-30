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

The PPO reward has three continuous signals: distance from the executed contact
point to the probe's legal strike region, impact-speed error, and the existing
smooth collision cost.  Binary hit/miss buckets and the every-tick tracking
bonus are diagnostics only.  The remaining physics constraints stay in the
differentiable auxiliary loss on the complete emitted trajectory.

Isaac-free.  The env hands in tensors; nothing here imports a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV
from amortized_opt.cost import CostTerms, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK, stick_axis_local, stick_mount_transform, stick_tip_offset
from amortized_opt.sdf import segment_hit_region
from amortized_opt.strike import (
    CYMBAL_CONTACT_SIGMA_FALLBACK,
    STICK_TIP_SIGMA,
    strike_axial_limits,
)
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

    # ---- continuous executed contact --------------------------------------
    #: Bounded continuous quality of the executed contact's distance to the
    #: same legal strike region used by ``probe.head_frame_error``.  The score
    #: is ``1 / (1 + (distance / scale)^2)``.
    w_hit_error: float = 10.0
    hit_error_scale_m: float = 0.05
    #: Smooth error between executed contact-point speed and ``|v_star|``.
    w_impact_speed: float = 0.5
    impact_speed_scale_mps: float = 0.5

    reward_mode: str = "continuous"
    w_correct_hit: float = 10.0
    w_wrong_hit: float = 5.0
    hit_window_ticks: int = 5

    # ---- the analytic half, on the plan ----
    #: Continuous collision avoidance from the existing FusedCost collision
    #: field.  Other analytic constraints remain in the differentiable
    #: auxiliary loss and do not enter the PPO reward.
    w_collision: float = 1.0e-3
    w_plan_cost: float = 0.0
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

    # Retained as a diagnostic scale; tracking is not part of the reward.
    track_scale_rad: float = 0.05

    # ---- safety filter ----
    #: Counted now, penalized later.  Default 0 for the scaffold stage.
    w_filter: float = 0.0

    #: Alive bonus, so an env that has not reached a hit yet has a defined
    #: (small, constant) return rather than only negative regularizers.
    w_alive: float = 0.0

    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)

    def __post_init__(self):
        if self.reward_mode not in ("continuous", "binary_hits"):
            raise ValueError("reward_mode must be 'continuous' or 'binary_hits'")
        if self.hit_error_scale_m <= 0.0:
            raise ValueError("hit_error_scale_m must be positive")
        if self.impact_speed_scale_mps <= 0.0:
            raise ValueError("impact_speed_scale_mps must be positive")
        if min(self.w_hit_error, self.w_impact_speed, self.w_collision) < 0.0:
            raise ValueError("continuous reward weights must be non-negative")
        if min(self.w_correct_hit, self.w_wrong_hit) < 0.0:
            raise ValueError("binary hit reward weights must be non-negative")
        if self.hit_window_ticks < 0:
            raise ValueError("hit_window_ticks must be non-negative")


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
    hit_error_m: torch.Tensor  # float -- distance to the probe's legal hit region
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

    * pads are scored at the bead; cymbals and the hi-hat use the playing-shaft
      capsule (``shaft_hit_sigma_range``), the same primitive as
      ``FusedCost._terminal`` / ``probe.head_frame_error``;
    * the legal region is a disk about ``disk_center`` (offset from the head
      centre on cymbals and toms) intersected with a half-plane
      (``has_half_plane``).
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

    def _sigma_span(self, ctx: StrikeContext):
        spec = ctx.contact_sigma
        fb = spec.new_full(spec.shape, float(CYMBAL_CONTACT_SIGMA_FALLBACK))
        if ctx.stick_tip_sigma is None:
            tip = spec.new_full(spec.shape, float(STICK_TIP_SIGMA))
        else:
            tip = ctx.stick_tip_sigma.to(dtype=spec.dtype)
        shaft = ctx.shaft_allowed
        lo = torch.where(shaft > 0.5, fb, spec)
        hi = torch.where(shaft > 0.5, tip, spec)
        return torch.minimum(lo, hi), torch.maximum(lo, hi)

    def best_contact(self, q: torch.Tensor, ctx: StrikeContext):
        """Playing-shaft (or bead) closest point to each hand's legal region."""
        lo, hi = self._sigma_span(ctx)
        a, _u = self.contact_points(q, lo)
        b, _ = self.contact_points(q, hi)
        ax_lo, ax_hi = strike_axial_limits()
        dist, s, p = segment_hit_region(
            a, b, ctx.p_star, ctx.n_out, ctx.hit_radius,
            ctx.disk_offset, ctx.half_plane_dir, ctx.has_half_plane,
            ax_lo, ax_hi,
        )
        return dist, s, p

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
        _dist0, _s0, p0 = self.best_contact(q0, ctx)
        dist1, _s1, p1 = self.best_contact(q1, ctx)
        a0, _i0 = self.head_frame(p0, ctx)
        a1, _i1 = self.head_frame(p1, ctx)
        crossed = (a0 > 0.0) & (a1 <= 0.0) & (ctx.active > 0.5)
        u = (a0 / (a0 - a1).clamp_min(1e-12)).clamp(0.0, 1.0)
        qc = q0 + u.max(dim=1).values.unsqueeze(-1) * (q1 - q0)
        _dc, _sc, pc = self.best_contact(qc, ctx)
        ac, ic = self.head_frame(pc, ctx)
        speed = (p1 - p0).norm(dim=-1) / self.dt
        return ContactScore(
            crossed=crossed,
            in_disk=crossed & (ic <= 0.0),
            frac=u,
            speed=speed,
            axial_m=a0,
            inplane_m=ic,
            hit_error_m=dist1,
            drum=ctx.target_drum,
        )

    @torch.no_grad()
    def any_drum_contact(self, q0: torch.Tensor, q1: torch.Tensor,
                         heads: torch.Tensor, axes: torch.Tensor,
                         radii: torch.Tensor, valid: torch.Tensor,
                         sigma: torch.Tensor,
                         shaft_allowed: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 2, D)`` bool: which drum heads the contact geometry crossed.

        Pads: the bead.  When ``shaft_allowed`` is set, that hand uses the
        playing-shaft capsule against every head (a shaft graze on a neighbouring
        cymbal still counts).
        """
        if shaft_allowed is None:
            lo = hi = sigma
        else:
            fb = sigma.new_full(sigma.shape, float(CYMBAL_CONTACT_SIGMA_FALLBACK))
            tip = sigma.new_full(sigma.shape, float(STICK_TIP_SIGMA))
            lo = torch.where(shaft_allowed > 0.5, fb, sigma)
            hi = torch.where(shaft_allowed > 0.5, tip, sigma)
        a0, _ = self.contact_points(q0, lo)
        b0, _ = self.contact_points(q0, hi)
        a1, _ = self.contact_points(q1, lo)
        b1, _ = self.contact_points(q1, hi)
        h = heads.unsqueeze(1)
        n = axes.unsqueeze(1)
        zoff = h.new_zeros(h.shape)
        hp = h.new_zeros(h.shape)
        dist0, _s0, p0 = segment_hit_region(
            a0.unsqueeze(2), b0.unsqueeze(2), h, n, radii.unsqueeze(1),
            zoff, hp, 0.0, -1.0e3, 1.0e3,
        )
        dist1, _s1, p1 = segment_hit_region(
            a1.unsqueeze(2), b1.unsqueeze(2), h, n, radii.unsqueeze(1),
            zoff, hp, 0.0, -1.0e3, 1.0e3,
        )
        d0 = ((p0 - h) * n).sum(-1)
        d1 = ((p1 - h) * n).sum(-1)
        u = (d0 / (d0 - d1).clamp_min(1e-12)).clamp(0.0, 1.0).unsqueeze(-1)
        pc = p0 + u * (p1 - p0)
        rad = pc - h
        rad = rad - ((rad * n).sum(-1, keepdim=True)) * n
        inside = rad.norm(dim=-1) <= radii.unsqueeze(1)
        return (d0 > 0.0) & (d1 <= 0.0) & inside & valid.unsqueeze(1)


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

        # ---- 1. executed contact ------------------------------------------
        cs = self.scorer.score(q_before, q_after, ctx)
        if cfg.reward_mode == "binary_hits":
            if drum_heads is None:
                raise ValueError("binary_hits reward requires drum geometry")
            hits_any = self.scorer.any_drum_contact(
                q_before, q_after, drum_heads, drum_axes, drum_radii, drum_valid,
                ctx.contact_sigma, ctx.shaft_allowed,
            )
            goal_oh = torch.nn.functional.one_hot(
                ctx.target_drum.clamp_min(0), hits_any.shape[2]
            ).bool() & (ctx.target_drum >= 0).unsqueeze(-1)
            on_goal = (hits_any & goal_oh).any(dim=2)
            near = ((hit_index >= 1) &
                    (hit_index <= int(cfg.hit_window_ticks))) | fired
            correct = on_goal & near & (ctx.active > 0.5)
            wrong_drum = (hits_any & ~goal_oh).any(dim=2)
            wrong = wrong_drum | (on_goal & ~near)
            terms["correct_hit"] = (
                float(cfg.w_correct_hit) * correct.to(self.dtype).sum(1)
            )
            terms["wrong_hit"] = (
                -float(cfg.w_wrong_hit) * wrong.to(self.dtype).sum(1)
            )
            counts["correct_hits"] = correct.to(self.dtype).sum(1)
            counts["wrong_hits"] = wrong.to(self.dtype).sum(1)
            counts["fired"] = fired.to(self.dtype).sum(1)
            raw["contact_speed"] = (cs.speed * correct).sum(1)
        else:
            due = (ctx.active > 0.5) & fired
            due_f = due.to(self.dtype)
            hit_quality = 1.0 / (
                1.0 + (cs.hit_error_m / float(cfg.hit_error_scale_m)).square()
            )
            target_speed = ctx.v_star.norm(dim=-1)
            speed_error = (cs.speed - target_speed).abs()
            speed_quality = 1.0 / (
                1.0 + (speed_error / float(cfg.impact_speed_scale_mps)).square()
            )
            terms["hit_error"] = cfg.w_hit_error * (hit_quality * due_f).sum(1)
            terms["impact_speed"] = (
                cfg.w_impact_speed * hit_quality * speed_quality * due_f
            ).sum(1)
            denom = due_f.sum(1).clamp_min(1.0)
            raw["hit_error_m"] = (
                cs.hit_error_m * due_f
            ).sum(1) / denom
            raw["impact_speed_error_mps"] = (
                speed_error * due_f
            ).sum(1) / denom
            raw["contact_speed"] = (
                cs.speed * due_f
            ).sum(1) / denom
            counts["contact_evaluated"] = due_f.sum(1)
            counts["fired"] = fired.to(self.dtype).sum(1)

        # ---- 2. continuous collision avoidance on the plan -----------------
        if cfg.reward_mode == "binary_hits":
            terms["collision_avoidance"] = z.clone()
        elif self.cost is not None and traj is not None:
            q0 = q_plan0 if q_plan0 is not None else q_before
            v0 = v_plan0 if v_plan0 is not None else torch.zeros_like(q0)
            lengths = group_lengths(ctx.hit_index).to(self.dtype)
            ct: CostTerms = self.cost(q0, traj, v0, ctx, lengths=lengths,
                                      elem_mask=elem_mask)
            collision = ct.terms.get("collision", torch.zeros_like(ct.total))
            terms["collision_avoidance"] = -cfg.w_collision * collision
            raw["plan_cost_total"] = ct.total.detach()
            for k, v in ct.terms.items():
                raw[f"plan.{k}"] = v.detach()
            for k, v in ct.violations.items():
                raw[f"viol.{k}"] = v.detach()
        else:
            terms["collision_avoidance"] = z.clone()

        # Tracking remains diagnostic only; it is deliberately absent from the
        # reward so contact geometry cannot be outvoted by an every-tick bonus.
        err = (q_after - ref0)
        rms = err.square().mean(dim=1).sqrt()
        raw["track_rms_rad"] = rms

        # ---- 3. safety filter (zero by default) -----------------------------
        if filter_active is not None:
            terms["filter"] = -cfg.w_filter * filter_active.to(self.dtype)
            counts["filter_interventions"] = filter_active.to(self.dtype)
        else:
            terms["filter"] = z.clone()

        terms["alive"] = torch.full_like(z, float(cfg.w_alive))
        total = sum(terms.values(), torch.zeros_like(z))
        return RewardTerms(total=total, terms=terms, raw=raw, counts=counts), cs
