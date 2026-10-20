"""Closed-loop strike probe: the metric the MPC is actually judged on.

The 243-case acceptance grid scores a *plan*, kinematically: it asks whether the
waypoint the planner nominates for the hit instant sits on the drum head.  The
deployed system instead executes the first waypoint of a new plan every 20 ms
through a PD servo, so what lands on the drum is the PLANT's pose at the hit
tick.  Section 7b's continuation optimized the first number to 175/243 while the
second stayed at 2/55, which is exactly what happens when a training loop can
only see one of them.

This module is the second number, cheap enough to run inside the training loop:
one rhythm clip through :class:`~amortized_opt.mpc_gen.MPCGenerator` with the
current weights as the planner, scored per executed hit in the head-plane frame
of the drum that was aimed at.

Conventions
-----------

``axial`` is the signed distance from the **selected contact** to the aim plane
along the OUTWARD head normal, so **positive means short of the aim** (toward /
past the geometric surface).  Pads use the bead; cymbals and the hi-hat use the
closest point of the playing-shaft capsule
(:func:`~amortized_opt.strike.shaft_hit_sigma_range`) to the legal region.  The
acceptance band is the interior hit cylinder
(:func:`~amortized_opt.strike.strike_axial_limits`): default 0 to +10 mm, i.e.
from the 1 cm-inset aim out to the head surface.  ``in_plane`` is the distance
beyond the legal disk edge, so <= 0 means inside.  ``in_disk`` is the
conjunction, plus the half-plane cut for the drums that have one -- the same
test :meth:`~amortized_opt.eval_harness.TrajectoryScorer.score` applies to a
plan (capsule on cymbals, bead on pads).

With ``MPCGenConfig.hit_lead_s`` the planner is told every hit is ``lead_s``
early, so ``EpisodeResult.hit_steps`` fires ``round(lead_s / dt)`` ticks before
the rhythm's instant; :func:`score_hits` shifts the measurement back so the lead
is not scored as its own error.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .fk import ChainFK
from .ik import arm_joint_indices, correct_quintic_init_arms, correct_tip, tip_position
from .limits import urdf_position_limits
from .mpc_gen import MPCGenConfig, MPCGenerator, PlanRequest
from .rhythm import build_context_inputs, strike_region
from .sdf import hit_region_dist, segment_hit_region
from .strike import (
    default_contact_sigma,
    shaft_hit_sigma_range,
    strike_axial_limits,
)
from .warmstart import seed_waist_pr_to_sp

__all__ = [
    "closed_loop_probe",
    "hit_latency",
    "head_frame_error",
    "mpc_model_planner",
    "score_hits",
    "summarize_hits",
]


def _strike_context_for_request(req: PlanRequest, *, device, dtype):
    """Build the training cost's exact strike geometry for one MPC request."""
    from .cost import StrikeContext
    from .phase2_data import _empty_strike, _strike_row

    rows = []
    active, target, hit = [], [], []
    for hand in ("left", "right"):
        goal = req.snapshot.goal[hand]
        is_active = bool(goal.active and int(req.hit_index[hand]) >= 1)
        rows.append(
            _strike_row(req.kit, int(goal.drum), goal.position, goal.velocity)
            if is_active else _empty_strike()
        )
        active.append(float(is_active))
        target.append(int(goal.drum) if is_active else -1)
        hit.append(max(int(req.hit_index[hand]), 0))

    def tensor(name, *, long=False):
        arr = np.stack([row[name] for row in rows])[None]
        return torch.as_tensor(
            arr, device=device, dtype=torch.long if long else dtype
        )

    return StrikeContext(
        hit_index=torch.as_tensor([hit], device=device, dtype=torch.long),
        active=torch.as_tensor([active], device=device, dtype=dtype),
        target_drum=torch.as_tensor([target], device=device, dtype=torch.long),
        start_drum=torch.full((1, 2), -1, device=device, dtype=torch.long),
        p_star=tensor("p_star"),
        v_star=tensor("v_star"),
        n_out=tensor("n_out"),
        hit_radius=tensor("hit_radius"),
        disk_offset=tensor("disk_off"),
        half_plane_dir=tensor("hp_dir"),
        has_half_plane=tensor("has_hp"),
        contact_sigma=tensor("sigma"),
        shaft_allowed=tensor("shaft"),
        start_shaft_allowed=tensor("start_shaft"),
        elev_required=tensor("elev"),
        drum_head_centers=torch.as_tensor(
            req.kit.positions_in("pelvis")[None], device=device, dtype=dtype
        ),
        drum_radii=torch.as_tensor(
            req.kit.radii[None], device=device, dtype=dtype
        ),
    )


def _constraint_snapshot(out) -> dict[str, float]:
    """Compact full-cost diagnostics for one locally optimized trajectory."""
    violations = out.violations

    def vmax(*names, prefix=None):
        values = []
        for name, value in violations.items():
            if name in names or (prefix is not None and name.startswith(prefix)):
                values.append(float(value.detach().max().item()))
        return max(values, default=0.0)

    snap = {
        "total": float(out.total.detach().mean().item()),
        "terminal": vmax("terminal"),
        "collision": vmax("collision", "collision_peak"),
        "torque": vmax("torque_peak", prefix="torque_"),
        "velocity": vmax("velocity_peak", prefix="velocity_"),
        "joint_box": vmax("joint_box_peak", prefix="joint_box_"),
    }
    for name in (
        "waist_yaw_monotonic", "waist_yaw_accel_direction",
        "waist_yaw_initial_accel_direction",
    ):
        value = out.terms.get(name)
        if value is not None:
            snap[name] = float(value.detach().mean().item())
    return snap


# ------------------------------------------------------------------ geometry --
def head_frame_error(scorer, q, hand, kit, drum, sigma, *, axial_tol: float | None = None):
    """``(axial, in_plane_beyond_edge, in_disk)`` of the contact at ``q``.

    Pads: the bead (``sigma``).  Cymbals/hi-hat: the closest point of the
    playing-shaft capsule to the legal region; ``sigma`` is ignored.  ``axial``
    and ``in_plane`` are reported at that selected point, so a shaft that spans
    the rim reads as inside even if the shoulder is not.
    """
    name = kit.names[int(drum)]
    rng = shaft_hit_sigma_range(name)
    lo, hi = (float(sigma), float(sigma)) if rng is None else (float(rng[0]), float(rng[1]))
    a, _t, _d = scorer.contact_world(np.asarray(q, float), hand, lo)
    b, _t, _d = scorer.contact_world(np.asarray(q, float), hand, hi)
    _aim, center, disk_r, half = strike_region(kit, int(drum), frame="pelvis")
    axis = np.asarray(kit.axes[int(drum)], float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    if axial_tol is None:
        ax_lo, ax_hi = strike_axial_limits()
    else:
        ax_lo, ax_hi = -float(axial_tol), float(axial_tol)
    hp = np.zeros(3) if half is None else np.asarray(half, float)
    has_hp = 0.0 if half is None else 1.0
    t = lambda x: scorer._t(x)
    dist, _s, p_c = segment_hit_region(
        t(a), t(b), t(center), t(axis), float(disk_r),
        t(np.zeros(3)), t(hp), has_hp, ax_lo, ax_hi,
    )
    _d0, ax_t, d_in_t = hit_region_dist(
        p_c, t(center), t(axis), float(disk_r),
        t(np.zeros(3)), t(hp), has_hp, ax_lo, ax_hi,
    )
    d_in = d_in_t.detach().cpu().numpy()
    r_in = float(np.linalg.norm(d_in))
    ax = float(ax_t.item())
    ok = float(dist.item()) <= 1e-9
    return ax, r_in - float(disk_r), bool(ok)


# ------------------------------------------------------------------- planner --
def mpc_model_planner(model, *, num_drums, num_future, device, amp_dtype=None,
                      iterations: int = 1, record: list | None = None,
                      record_waist_trace: bool = False,
                      waist_cfg=None, clamp_init_to_box: bool = True,
                      terminal_qp_steps: int = 0,
                      terminal_qp_window: int = 16,
                      terminal_qp_damping: float = 0.05,
                      full_cost=None,
                      full_cost_weights: dict | None = None,
                      full_cost_steps: int = 0,
                      full_cost_lr: float = 1.0e-3,
                      full_cost_grad_clip: float = 1.0e4,
                      full_cost_from_init: bool = False,
                      fresh_line_init: bool = False):
    """The decoder wired into ``MPCGenerator``'s planner protocol.

    Returns ``(plan_fn, state)``; ``state["drift"]`` accumulates the
    temporal-consistency drift between successive plans on their shared support
    and ``record``, if given, receives one dict per tick with that tick's plan.

    ``waist_cfg`` is the :class:`~amortized_opt.waist_aim.WaistAimConfig` the
    context's ``aim_yaw`` is computed with -- pass the one the checkpoint was
    trained against, exactly as every pool builder passes ``cfg.waist_cfg``.
    ``None`` keeps the default config (today's behaviour).

    ``clamp_init_to_box`` projects the seed (and the waist P/R set-point it
    eases toward) into the URDF/posture box after ``seed_waist_pr_to_sp`` /
    DLS, so the decoder is not residual-seeded off an illegal torso.  Matches
    ``MPCGenConfig.clamp_init_to_box``; does not clamp the emitted traj.
    """
    core = model.module if hasattr(model, "module") else model
    state = {"prev": None, "drift": [], "last_hit": None, "full_opt": []}
    fk: ChainFK | None = None
    q_lo_np, q_hi_np = urdf_position_limits(G1_UPPER_BODY_JOINTS)
    q_lo_t = torch.as_tensor(q_lo_np, device=device, dtype=torch.float32)
    q_hi_t = torch.as_tensor(q_hi_np, device=device, dtype=torch.float32)
    _pr = (
        G1_UPPER_BODY_JOINTS.index("waist_pitch_joint"),
        G1_UPPER_BODY_JOINTS.index("waist_roll_joint"),
    )

    def _fk(dev, dtype) -> ChainFK:
        nonlocal fk
        if fk is None or fk.device != torch.device(dev) or fk.dtype != dtype:
            fk = ChainFK(device=dev, dtype=dtype)
        return fk

    @torch.no_grad()
    def plan(req: PlanRequest) -> np.ndarray:
        # The idle arm's token: the generator that drives this probe parks the
        # idle arm, so the context has to say so too or the probe scores the
        # model on a context shape it never saw in training.
        ctx = build_context_inputs(
            req.kit, req.snapshot, num_drums=num_drums, num_future=num_future,
            frame="env", device=device, dtype=torch.float32,
            idle_library=getattr(req, "idle_library", None),
            idle_cylinder=getattr(req, "idle_cylinder", None),
            idle_home=getattr(req, "idle_home", None),
            yaw_now=np.asarray(req.q, float)[
                G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
            ],
            waist_cfg=waist_cfg,
        )
        t = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a, float), device=device, dtype=torch.float32
        )
        masks = req.masks.to(device)
        masks = type(masks)(**{
            k: getattr(masks, k).to(
                torch.float32 if getattr(masks, k).dtype != torch.bool else torch.bool
            )
            for k in masks.__dataclass_fields__
        })
        init = t(req.init_traj)[None] * masks.elem_mask
        st = torch.cat([t(req.q), t(req.qd)])[None, :]
        amp = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp_dtype is not None and torch.device(device).type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with amp:
            enc = core.encode_context(ctx)
            n = len(G1_UPPER_BODY_JOINTS)
            if enc.waist_pr_sp is not None:
                sp = enc.waist_pr_sp
                if clamp_init_to_box:
                    sp = sp.detach().clone()
                    sp[..., 0] = sp[..., 0].clamp(q_lo_np[_pr[0]], q_hi_np[_pr[0]])
                    sp[..., 1] = sp[..., 1].clamp(q_lo_np[_pr[1]], q_hi_np[_pr[1]])
                init = seed_waist_pr_to_sp(
                    init, st[:, :n], st[:, n:], sp,
                    joint_names=G1_UPPER_BODY_JOINTS, dt=float(req.dt),
                )
            hit = torch.tensor(
                [[req.hit_index.get("left", -1), req.hit_index.get("right", -1)]],
                device=init.device, dtype=torch.long,
            )
            aims = torch.zeros(1, 2, 3, device=init.device, dtype=init.dtype)
            for hi, hand in enumerate(("left", "right")):
                g = req.snapshot.goal[hand]
                if g.active:
                    aims[0, hi] = t(np.asarray(g.position, float))
            # Fresh quintic only: first tick, or a hand whose horizon jumped
            # (reseed / stale promote).  Every-tick DLS would rebuild arms
            # from q0 and wipe the shifted decoder plan.
            hits = (
                int(req.hit_index.get("left", -1)),
                int(req.hit_index.get("right", -1)),
            )
            last = state["last_hit"]
            pending = last is None or any(
                h >= 1 and (last[i] < 1 or h > last[i]) for i, h in enumerate(hits)
            )
            if pending and not fresh_line_init:
                init = correct_quintic_init_arms(
                    _fk(init.device, init.dtype), init, st[:, :n], hit, aims,
                    steps=1, joint_names=G1_UPPER_BODY_JOINTS,
                    qd0=st[:, n:], dt=float(req.dt),
                )
            if fresh_line_init:
                if getattr(core, "terminal_initializer", None) is None:
                    raise ValueError("fresh-line probes require a terminal pose head")
                init = core.initial_trajectory(ctx, st, masks).to(init.dtype)
            if clamp_init_to_box:
                init = init.clamp(q_lo_t.to(device=init.device, dtype=init.dtype),
                                  q_hi_t.to(device=init.device, dtype=init.dtype))
            state["last_hit"] = hits
            out = core.refine(init, st, enc, masks, iterations=int(iterations))
        traj_t = out.traj[0].float()
        if int(terminal_qp_steps) > 0:
            # Sequential convex terminal corrections.  Each DLS update solves
            # min ||J dq - e||^2 + damping^2 ||dq||^2 for one arm, with the
            # shared waist fixed.  Spread the endpoint edit backward with a
            # quintic minimum-jerk ramp so this tests a usable trajectory, not
            # an isolated last-knot teleport.
            traj_t = traj_t.clone()
            for hi, hand in enumerate(("left", "right")):
                kh = int(hits[hi])
                if kh < 1 or kh > traj_t.shape[0]:
                    continue
                q_hit = traj_t[kh - 1 : kh]
                q_fixed = correct_tip(
                    _fk(traj_t.device, traj_t.dtype),
                    q_hit,
                    aims[:, hi],
                    hand,
                    steps=int(terminal_qp_steps),
                    damping=float(terminal_qp_damping),
                    q_bounds=(q_lo_np, q_hi_np),
                    joint_names=G1_UPPER_BODY_JOINTS,
                )
                cols = torch.as_tensor(
                    arm_joint_indices(hand, G1_UPPER_BODY_JOINTS),
                    device=traj_t.device,
                    dtype=torch.long,
                )
                delta = (q_fixed - q_hit).index_select(1, cols)[0]
                start = max(0, kh - max(int(terminal_qp_window), 1))
                count = kh - start
                u = torch.arange(
                    1, count + 1, device=traj_t.device, dtype=traj_t.dtype
                ) / float(count)
                blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
                block = traj_t[start:kh].index_select(1, cols)
                traj_t[start:kh].index_copy_(
                    1, cols, block + blend[:, None] * delta[None, :]
                )
            traj_t = traj_t.clamp(
                q_lo_t.to(device=traj_t.device, dtype=traj_t.dtype),
                q_hi_t.to(device=traj_t.device, dtype=traj_t.dtype),
            )
        if int(full_cost_steps) > 0:
            if full_cost is None:
                raise ValueError("full_cost_steps requires a FusedCost instance")
            strike = _strike_context_for_request(
                req, device=traj_t.device, dtype=traj_t.dtype
            )
            if (
                float(full_cost.cfg.w_waist_aim) > 0.0
                or float(full_cost.cfg.w_waist_yaw_monotonic) > 0.0
                or float(full_cost.cfg.w_waist_yaw_accel_direction) > 0.0
                or float(full_cost.cfg.w_waist_yaw_initial_accel_direction) > 0.0
            ):
                yaw_ref_i = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
                strike = replace(
                    strike,
                    waist_yaw_ref=init[:, :, yaw_ref_i].detach(),
                )
            q0_cost = t(req.q).view(1, -1)
            qd0_cost = t(req.qd).view(1, -1)
            em_cost = masks.elem_mask.to(device=traj_t.device, dtype=traj_t.dtype)
            fixed = (
                init[0].detach().clone()
                if bool(full_cost_from_init) else traj_t.detach().clone()
            )
            with torch.enable_grad():
                variable = torch.nn.Parameter(fixed.clone())
                optimizer = torch.optim.Adam([variable], lr=float(full_cost_lr))
                before_out = full_cost(
                    q0_cost, variable.unsqueeze(0), qd0_cost, strike,
                    elem_mask=em_cost,
                    weights=full_cost_weights,
                )
                before = _constraint_snapshot(before_out)
                yaw_i = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
                live_hits = [int(value) for value in hits if int(value) >= 1]
                terminal_i = max(live_hits, default=variable.shape[0]) - 1
                terminal_i = int(np.clip(terminal_i, 0, variable.shape[0] - 1))
                total_grad = torch.autograd.grad(
                    before_out.total.mean(), variable, retain_graph=True,
                )[0]
                term_grad: dict[str, dict[str, float]] = {}
                for name in (
                    "terminal", "terminal_aim", "collision", "torque",
                    "torque_peak", "velocity", "joint_box", "smooth",
                    "posture", "waist_yaw_vel", "waist_aim",
                    "waist_yaw_monotonic",
                    "waist_yaw_accel_direction",
                    "waist_yaw_initial_accel_direction",
                ):
                    value = before_out.terms.get(name)
                    if value is None or not value.requires_grad:
                        continue
                    grad = torch.autograd.grad(
                        value.mean(), variable, retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if grad is None:
                        continue
                    gy = grad[:, yaw_i]
                    term_grad[name] = {
                        "first": float(gy[0].detach()),
                        "terminal": float(gy[terminal_i].detach()),
                        "norm": float(gy.float().norm().detach()),
                    }
                yaw_diag = {
                    "terminal_index": terminal_i,
                    "before_first_rad": float(variable[0, yaw_i].detach()),
                    "before_terminal_rad": float(variable[terminal_i, yaw_i].detach()),
                    "total_grad_first": float(total_grad[0, yaw_i].detach()),
                    "total_grad_terminal": float(total_grad[terminal_i, yaw_i].detach()),
                    "total_grad_norm": float(total_grad[:, yaw_i].float().norm().detach()),
                    "term_grad": term_grad,
                }
                for _ in range(int(full_cost_steps)):
                    optimizer.zero_grad(set_to_none=True)
                    cost_out = full_cost(
                        q0_cost, variable.unsqueeze(0), qd0_cost, strike,
                        elem_mask=em_cost,
                        weights=full_cost_weights,
                    )
                    cost_out.total.mean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        [variable], float(full_cost_grad_clip)
                    )
                    optimizer.step()
                    with torch.no_grad():
                        variable.clamp_(
                            q_lo_t.to(variable), q_hi_t.to(variable)
                        )
                        mask = em_cost[0]
                        variable.copy_(variable * mask + fixed * (1.0 - mask))
                after_out = full_cost(
                    q0_cost, variable.unsqueeze(0), qd0_cost, strike,
                    elem_mask=em_cost,
                    weights=full_cost_weights,
                )
                after = _constraint_snapshot(after_out)
                yaw_diag.update({
                    "after_first_rad": float(variable[0, yaw_i].detach()),
                    "after_terminal_rad": float(variable[terminal_i, yaw_i].detach()),
                    "delta_first_rad": float(
                        (variable[0, yaw_i] - fixed[0, yaw_i]).detach()
                    ),
                    "delta_terminal_rad": float(
                        (variable[terminal_i, yaw_i] - fixed[terminal_i, yaw_i]).detach()
                    ),
                })
                traj_t = variable.detach()
            state["full_opt"].append({
                "before": before, "after": after, "yaw": yaw_diag,
            })
        traj = traj_t.cpu().numpy()
        prev = state["prev"]
        if prev is not None and prev.shape == traj.shape:
            m = masks.elem_mask[0].cpu().numpy()
            d = np.abs(traj[:-1] - prev[1:]) * m[:-1]
            state["drift"].append(float(d.sum() / max(m[:-1].sum(), 1.0)))
        state["prev"] = traj.copy()
        if record is not None:
            row = {"step": req.step, "hit": dict(req.hit_index),
                   "traj": traj.copy()}
            if record_waist_trace:
                yaw_i = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
                row["waist_yaw"] = {
                    "measured_rad": float(np.asarray(req.q)[yaw_i]),
                    "aim_yaw_rad": (
                        float(ctx.aim_yaw[0].detach().cpu().item())
                        if ctx.aim_yaw is not None else None
                    ),
                    "init_rad": init[0, :, yaw_i].detach().cpu().float().tolist(),
                    "generated_rad": traj[:, yaw_i].astype(float).tolist(),
                }
            record.append(row)
        return traj

    return plan, state


# -------------------------------------------------------------------- scoring --
def score_hits(scorer, kit, res, record=None, *, lead_ticks: int = 0,
               offsets=(-3, -2, -1, 0, 1, 2, 3), drift_back: int = 13,
               momentum_grace_ticks: int = 0, momentum_substeps: int = 10):
    """Per executed hit: the plant at several tick offsets, and the plan's drift.

    ``planned`` is the per-tick answer to "what did the plan of that tick put at
    its OWN terminal waypoint?" as a function of the remaining horizon ``h``.
    Flat and near zero = the planner is right and the plant is behind; growing
    as ``h`` shrinks = the planner itself degrades in the final approach.
    """
    q = res.q_hist
    by_step = {r["step"]: r for r in (record or [])}
    hits = []
    hit_states = getattr(res, "hit_states", [])
    for hit_i, (k0, hand, drum) in enumerate(res.hit_steps):
        k = int(k0) + int(lead_ticks)
        sigma = default_contact_sigma(kit.names[int(drum)])
        row = {"k": int(k), "k_fire": int(k0), "hand": hand,
               "drum": kit.names[int(drum)], "meas": {}, "planned": [],
               "momentum": []}
        pre_state = hit_states[hit_i] if hit_i < len(hit_states) else None
        cmd_i = int(k0) - 1
        if 0 <= cmd_i < len(res.cmd_hist):
            ax, ip, ok = head_frame_error(
                scorer, res.cmd_hist[cmd_i], hand, kit, drum, sigma,
            )
            row["command"] = {
                "axial_mm": 1000 * ax,
                "inplane_mm": 1000 * ip,
                "in_disk": bool(ok),
            }
        for o in offsets:
            i = k + o
            if 0 <= i < len(q):
                q_score = (
                    pre_state[0]
                    if o == 0 and int(lead_ticks) == 0 and pre_state is not None
                    else q[i]
                )
                ax, ip, ok = head_frame_error(scorer, q_score, hand, kit, drum, sigma)
                row["meas"][int(o)] = {"axial_mm": 1000 * ax, "inplane_mm": 1000 * ip,
                                       "in_disk": bool(ok)}
        if int(momentum_grace_ticks) > 0 and pre_state is not None:
            q_fire, qd_fire = pre_state
            substeps = max(1, int(momentum_substeps))
            for sub in range(1, int(momentum_grace_ticks) * substeps + 1):
                t_s = float(sub) * CONTROL_DT / float(substeps)
                q_coast = q_fire + t_s * qd_fire
                ax, ip, ok = head_frame_error(
                    scorer, q_coast, hand, kit, drum, sigma,
                )
                row["momentum"].append({
                    "t_ms": 1000.0 * t_s,
                    "axial_mm": 1000.0 * ax,
                    "inplane_mm": 1000.0 * ip,
                    "in_disk": bool(ok),
                })
        for back in range(int(drift_back)):
            r = by_step.get(int(k0) - 1 - back)
            if r is None:
                continue
            kh = int(r["hit"].get(hand, -1))
            if kh < 1 or kh > r["traj"].shape[0]:
                continue
            ax, ip, ok = head_frame_error(scorer, r["traj"][kh - 1], hand, kit, drum, sigma)
            row["planned"].append({"h": kh, "axial_mm": 1000 * ax,
                                   "inplane_mm": 1000 * ip, "in_disk": bool(ok)})
        hits.append(row)
    return hits


def hit_latency(scorer, kit, res, *, lead_ticks: int = 0, window_ticks: int = 5,
                dt: float = CONTROL_DT):
    """When does the stick actually ARRIVE, relative to the scheduled tick?

    The binary "in the disk at tick k" test understates the closed loop.  The
    measured offset distribution is asymmetric-late -- almost nothing arrives
    early, most strikes cross the head plane one to three ticks behind the beat
    -- and those ARE hits: real contact happens whenever the stick gets there,
    just behind time.  A configuration that turns 60 ms-late strikes into 20 ms
    late ones is progress the at-tick binary cannot see.

    For each executed strike this scans the PLANT over ``+/- window_ticks``
    around the scheduled instant and returns the first crossing of the head
    plane (``axial`` going from positive to <= 0) that is inside the legal disk,
    with the crossing located to SUB-TICK resolution by linear interpolation of
    the axial coordinate (a 20 ms grid is coarse against a +/-10 ms question).
    The in-plane test and the impact speed are evaluated at the interpolated
    pose, not at the nearer knot.

    ``latency_ms`` is negative for early.  ``None`` means the stick never
    crossed the plane inside the disk in the window -- a genuine MISS, which is
    the statistic the binary metric conflates with "late".
    """
    q = res.q_hist
    out = []
    for (k0, hand, drum) in res.hit_steps:
        k = int(k0) + int(lead_ticks)
        sigma = default_contact_sigma(kit.names[int(drum)])
        rec = {"k": int(k), "hand": hand, "drum": kit.names[int(drum)],
               "latency_ms": None, "speed_mps": float("nan"),
               "in_disk_at_crossing": False, "inplane_mm": float("nan")}
        lo = max(k - int(window_ticks), 1)
        hi = min(k + int(window_ticks), len(q) - 1)
        prev = None
        for i in range(lo - 1, hi + 1):
            ax, _ip, _ok = head_frame_error(scorer, q[i], hand, kit, drum, sigma)
            if prev is not None and prev[1] > 0.0 >= ax:
                j, a0 = prev
                u = a0 / max(a0 - ax, 1e-12)
                qc = q[j] + u * (q[i] - q[j])
                axc, ipc, okc = head_frame_error(scorer, qc, hand, kit, drum, sigma)
                J = scorer.tip_jacobian(qc, hand, sigma)
                v = J @ ((q[i] - q[j]) / dt)
                rec.update(latency_ms=1000.0 * ((j + u) - k) * dt,
                           speed_mps=float(np.linalg.norm(v)),
                           in_disk_at_crossing=bool(ipc <= 0.0),
                           inplane_mm=1000.0 * ipc)
                break
            prev = (i, ax)
        out.append(rec)
    return out


def summarize_latency(rows, *, bands=(10.0, 20.0, 40.0)) -> dict:
    n = max(len(rows), 1)
    hit = [r for r in rows if r["latency_ms"] is not None and r["in_disk_at_crossing"]]
    lat = np.array([r["latency_ms"] for r in hit], float)
    spd = np.array([r["speed_mps"] for r in hit], float)
    out = {
        "n": len(rows),
        "crossed_in_disk": len(hit),
        "miss_rate": 1.0 - len(hit) / n,
        "lat_p50_ms": float(np.percentile(lat, 50)) if lat.size else float("nan"),
        "lat_abs_p50_ms": float(np.percentile(np.abs(lat), 50)) if lat.size else float("nan"),
        "lat_abs_p90_ms": float(np.percentile(np.abs(lat), 90)) if lat.size else float("nan"),
        "early": int((lat < 0).sum()),
        "speed_p50_mps": float(np.percentile(spd, 50)) if spd.size else float("nan"),
        "speed_in_band": int(((spd >= 1.0) & (spd <= 3.5)).sum()),
    }
    for b in bands:
        out[f"within_{int(b)}ms"] = int((np.abs(lat) <= b).sum()) if lat.size else 0
    # the histogram itself, in whole control periods
    edges = [-100, -60, -40, -20, -10, 0, 10, 20, 40, 60, 100]
    out["hist_ms"] = {
        f"[{a},{b})": int(((lat >= a) & (lat < b)).sum()) if lat.size else 0
        for a, b in zip(edges[:-1], edges[1:])
    }
    return out


def _p50(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.percentile(v, 50)) if v else float("nan")


def summarize_hits(hits) -> dict:
    n = len(hits)
    g = lambda o, f: [h["meas"][o][f] for h in hits if o in h["meas"]]  # noqa: E731
    ax_lo_mm, ax_hi_mm = (1000.0 * v for v in strike_axial_limits())
    out = {
        "hits": n,
        "in_disk": sum(1 for h in hits if h["meas"].get(0, {}).get("in_disk")),
        "in_disk_window": sum(
            1 for h in hits
            if any(h["meas"].get(o, {}).get("in_disk") for o in (-1, 0, 1, 2))
        ),
        "axial_p50_mm": _p50(g(0, "axial_mm")),
        "axial_abs_p50_mm": _p50([abs(x) for x in g(0, "axial_mm")]),
        "inplane_p50_mm": _p50(g(0, "inplane_mm")),
        "axial_in_band": sum(
            1 for x in g(0, "axial_mm") if ax_lo_mm - 1e-6 <= x <= ax_hi_mm + 1e-6
        ),
        "inplane_inside": sum(1 for x in g(0, "inplane_mm") if x <= 0.0),
    }
    # A continuous selection score.  Raw ``in_disk`` on a one-clip probe is a
    # small integer that is often 0, which cannot rank two checkpoints; the two
    # half-credit terms are the two independent halves of the hit test, so a
    # model that fixes the aim but not the reach still ranks above one that
    # fixes neither.  Range 0..1.5.
    h = max(n, 1)
    out["score"] = (
        out["in_disk"] / h
        + 0.25 * out["axial_in_band"] / h
        + 0.25 * out["inplane_inside"] / h
    )
    by_h: dict[int, list] = {}
    axial_by_h: dict[int, list] = {}
    in_disk_by_h: dict[int, list] = {}
    for h in hits:
        for p in h["planned"]:
            by_h.setdefault(p["h"], []).append(p["inplane_mm"])
            axial_by_h.setdefault(p["h"], []).append(p["axial_mm"])
            in_disk_by_h.setdefault(p["h"], []).append(float(p["in_disk"]))
    out["planned_inplane_p50_by_h"] = {int(k): _p50(v) for k, v in sorted(by_h.items())}
    out["planned_axial_p50_by_h"] = {
        int(k): _p50(v) for k, v in sorted(axial_by_h.items())
    }
    out["planned_axial_abs_p50_by_h"] = {
        int(k): _p50([abs(x) for x in v]) for k, v in sorted(axial_by_h.items())
    }
    out["planned_in_disk_frac_by_h"] = {
        int(k): float(np.mean(v)) for k, v in sorted(in_disk_by_h.items())
    }
    momentum = [h.get("momentum", []) for h in hits]
    if any(momentum):
        crossed = [next((x for x in rows if x["in_disk"]), None) for rows in momentum]
        final = [rows[-1] for rows in momentum if rows]
        first_ms = [x["t_ms"] for x in crossed if x is not None]
        out.update({
            "momentum_grace_hits": len(momentum),
            "momentum_crossed": sum(x is not None for x in crossed),
            "momentum_first_ms_p50": _p50(first_ms),
            "momentum_final_axial_p50_mm": _p50(
                [x["axial_mm"] for x in final]
            ),
            "momentum_final_inplane_p50_mm": _p50(
                [x["inplane_mm"] for x in final]
            ),
        })
    return out


# ---------------------------------------------------------------- the probe ---
def closed_loop_probe(
    model,
    *,
    kit,
    scorer,
    clips,
    device,
    num_drums: int = 8,
    num_future: int = 3,
    horizon: int = 50,
    steps: int = 250,
    seed: int = 0,
    amp_dtype=None,
    iterations: int = 1,
    lead_s: float = 0.0,
    cmd_lead_s=0.0,
    ki: float = 0.0,
    ki_windup: float = 0.0,
    pid_with_lead: bool = False,
    kp_scale=1.0,
    kd_scale=1.0,
    saturate_effort: bool = False,
    strike_impedance=None,
    freeze_wp: int = 0,
    freeze_waist: bool = True,
    sticky_goal_point: bool = True,
    bounce: object | None = True,
    starts=None,
    record_drift: bool = True,
    record_waist_trace: bool = False,
    waist_upright_ease_wp: int | None = None,
    waist_cfg=None,
    waist_pr_ema: bool = False,
    waist_pr_ema_alpha: float = 0.1,
    waist_pr_ema_leak: float = 0.02,
    perfect_tracking: bool = False,
    momentum_grace_ticks: int = 0,
    terminal_qp_steps: int = 0,
    terminal_qp_window: int = 16,
    terminal_qp_damping: float = 0.05,
    full_cost=None,
    full_cost_weights: dict | None = None,
    full_cost_steps: int = 0,
    full_cost_lr: float = 1.0e-3,
    full_cost_grad_clip: float = 1.0e4,
    full_cost_from_init: bool = False,
    fresh_line_init: bool = False,
    record_rollout: bool = False,
) -> dict:
    """Run ``len(clips)`` closed-loop episodes and score every executed strike.

    Deterministic for a fixed ``seed`` and ``clips``, so it is usable as a
    rolling training metric: the start poses come from a seeded
    :class:`~amortized_opt.starts.StartSampler` draw and every episode's RNG is
    ``1000 + episode``.

    ``bounce`` defaults ON: after each fire the plant gets the BPTT placeholder
    contact impulse (:func:`~amortized_opt.train_cost.apply_bounce`) before the
    next-hit quintic is seeded, matching the training roller's post-hit carry.
    Pass ``False`` to restore the historical through-the-head plant.

    ``record_rollout`` adds ``_rollouts`` with full plant histories, network
    plans, and plans after controller freeze/lead processing for visualization.
    It is opt-in because these arrays are unsuitable for training metric logs.
    """
    was_training = getattr(model, "training", False)
    core = model.module if hasattr(model, "module") else model
    core.eval()
    gen0 = MPCGenerator(kit=kit, waist_cfg=waist_cfg, config=MPCGenConfig(
        horizon=int(horizon), ki=float(ki), ki_windup=float(ki_windup),
        pid_with_lead=bool(pid_with_lead),
        kp_scale=kp_scale, kd_scale=kd_scale,
        saturate_effort=bool(saturate_effort),
        waist_upright_ease_wp=waist_upright_ease_wp,
        waist_pr_ema=bool(waist_pr_ema),
        waist_pr_ema_alpha=float(waist_pr_ema_alpha),
        waist_pr_ema_leak=float(waist_pr_ema_leak),
    ))
    if starts is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                               rnea=gen0.rnea, sdf=gen0.sdf)
        starts = sampler.sample(len(clips), seed=int(seed))
    lead_ticks = int(round(float(lead_s) / CONTROL_DT))

    all_hits, all_lat, episodes, waist_traces = [], [], [], []
    rollouts = []
    t_all = time.perf_counter()
    for ep, clip in enumerate(clips):
        gen = MPCGenerator(
            kit=kit,
            waist_cfg=waist_cfg,
            config=MPCGenConfig(
                horizon=int(horizon), hit_lead_s=float(lead_s),
                cmd_lead_s=cmd_lead_s,
                ki=float(ki), ki_windup=float(ki_windup),
                pid_with_lead=bool(pid_with_lead),
                kp_scale=kp_scale, kd_scale=kd_scale,
                saturate_effort=bool(saturate_effort),
                strike_impedance=strike_impedance,
                freeze_wp=int(freeze_wp), freeze_waist=bool(freeze_waist),
                sticky_goal_point=bool(sticky_goal_point),
                bounce=bounce,
                waist_upright_ease_wp=waist_upright_ease_wp,
                waist_pr_ema=bool(waist_pr_ema),
                waist_pr_ema_alpha=float(waist_pr_ema_alpha),
                waist_pr_ema_leak=float(waist_pr_ema_leak),
                perfect_tracking=bool(perfect_tracking),
            ),
        )
        record: list = [] if record_drift or record_rollout else None
        committed_plans = [] if record_rollout else None
        plan, st = mpc_model_planner(
            model, num_drums=num_drums, num_future=num_future, device=device,
            amp_dtype=amp_dtype, iterations=iterations, record=record,
            record_waist_trace=bool(record_waist_trace),
            waist_cfg=waist_cfg,
            terminal_qp_steps=int(terminal_qp_steps),
            terminal_qp_window=int(terminal_qp_window),
            terminal_qp_damping=float(terminal_qp_damping),
            full_cost=full_cost,
            full_cost_weights=full_cost_weights,
            full_cost_steps=int(full_cost_steps),
            full_cost_lr=float(full_cost_lr),
            full_cost_grad_clip=float(full_cost_grad_clip),
            full_cost_from_init=bool(full_cost_from_init),
            fresh_line_init=bool(fresh_line_init),
        )
        t0 = time.perf_counter()
        res = gen.run_episode(
            clip, starts.q[ep].cpu().numpy(), starts.qd[ep].cpu().numpy(),
            seed=1000 + ep, planner=plan, max_steps=int(steps),
            **({"plan_trace": committed_plans} if record_rollout else {}),
        )
        if record_rollout:
            rollouts.append({"result": res, "plans": record,
                             "committed_plans": committed_plans})
        wall = time.perf_counter() - t0
        hits = score_hits(
            scorer, kit, res, record, lead_ticks=lead_ticks,
            momentum_grace_ticks=int(momentum_grace_ticks),
        )
        all_hits.extend(hits)
        if record_waist_trace:
            waist_traces.append([
                {
                    "step": int(row["step"]),
                    "hit": dict(row["hit"]),
                    "waist_yaw": row["waist_yaw"],
                }
                for row in (record or [])
            ])
        all_lat.extend(hit_latency(scorer, kit, res, lead_ticks=lead_ticks))
        track = (np.abs(res.q_hist[1:] - res.cmd_hist)
                 if len(res.cmd_hist) else np.zeros(1))
        tip_track_mm: list[float] = []
        hit_states = getattr(res, "hit_states", [])
        if len(res.cmd_hist):
            # q_hist stores the deliberate post-contact bounce at fired ticks.
            # Restore the pre-bounce plant pose so the Cartesian statistic is
            # servo tracking error rather than contact-response displacement.
            q_actual_np = np.array(res.q_hist[1:], copy=True)
            for hit_i, (k0, _hand, _drum) in enumerate(res.hit_steps):
                cmd_i = int(k0) - 1
                if hit_i < len(hit_states) and 0 <= cmd_i < len(q_actual_np):
                    q_actual_np[cmd_i] = hit_states[hit_i][0]
            q_actual = scorer._t(q_actual_np)
            q_command = scorer._t(res.cmd_hist)
            with torch.no_grad():
                for hand in ("left", "right"):
                    actual_tip = tip_position(scorer.fk, q_actual, hand)
                    command_tip = tip_position(scorer.fk, q_command, hand)
                    tip_track_mm.extend(
                        (1000.0 * torch.linalg.vector_norm(
                            actual_tip - command_tip, dim=-1
                        )).cpu().tolist()
                    )

        # At a fired strike, q_hist has already been replaced by the bounce
        # response.  Use hit_states so this measures servo lag immediately
        # before contact/bounce against the command issued for that tick.
        hit_tip_track_mm: list[float] = []
        for hit_i, (k0, hand, _drum) in enumerate(res.hit_steps):
            cmd_i = int(k0) - 1
            if not (0 <= cmd_i < len(res.cmd_hist)):
                continue
            q_actual_np = (
                hit_states[hit_i][0]
                if hit_i < len(hit_states)
                else res.q_hist[int(k0)]
            )
            with torch.no_grad():
                actual_tip = tip_position(
                    scorer.fk, scorer._t(q_actual_np).reshape(1, -1), hand
                )
                command_tip = tip_position(
                    scorer.fk, scorer._t(res.cmd_hist[cmd_i]).reshape(1, -1), hand
                )
            hit_tip_track_mm.append(float(
                1000.0 * torch.linalg.vector_norm(actual_tip - command_tip).item()
            ))

        tip_track = np.asarray(tip_track_mm, dtype=float)
        hit_tip_track = np.asarray(hit_tip_track_mm, dtype=float)
        episodes.append({
            "clip": clip.name, "ticks": int(res.cmd_hist.shape[0]),
            "hits": len(hits), "diverged": bool(res.diverged),
            "frozen_ticks": int(getattr(res, "frozen_ticks", 0)),
            "bounced_hands": int(getattr(res, "bounced_hands", 0)),
            "pd_track_p50_mrad": float(np.percentile(track, 50) * 1000),
            "pd_track_p90_mrad": float(np.percentile(track, 90) * 1000),
            "pd_tip_track_mean_mm": (float(tip_track.mean())
                                      if tip_track.size else float("nan")),
            "pd_tip_track_p50_mm": (float(np.percentile(tip_track, 50))
                                     if tip_track.size else float("nan")),
            "pd_tip_track_p90_mm": (float(np.percentile(tip_track, 90))
                                     if tip_track.size else float("nan")),
            "pd_hit_tip_track_mean_mm": (float(hit_tip_track.mean())
                                          if hit_tip_track.size else float("nan")),
            "pd_hit_tip_track_p50_mm": (float(np.percentile(hit_tip_track, 50))
                                         if hit_tip_track.size else float("nan")),
            "pd_hit_tip_track_p90_mm": (float(np.percentile(hit_tip_track, 90))
                                         if hit_tip_track.size else float("nan")),
            "tc_drift_p50_mrad": (float(np.percentile(st["drift"], 50) * 1000)
                                  if st["drift"] else float("nan")),
            "ms_per_tick": 1000 * wall / max(res.cmd_hist.shape[0], 1),
            "full_cost_refine": (
                {
                    phase: {
                        key: float(np.mean([row[phase][key] for row in st["full_opt"]]))
                        for key in st["full_opt"][0][phase]
                    }
                    for phase in ("before", "after")
                }
                if st["full_opt"] else None
            ),
            "full_cost_yaw": (
                st["full_opt"] if st["full_opt"] else None
            ),
        })
    if was_training:
        core.train()
    out = summarize_hits(all_hits)
    out["latency"] = summarize_latency(all_lat)
    # The SELECTION score is now the latency distribution, not the at-tick
    # binary: rank on how many strikes land in the disk at all and how tight
    # their timing is, with a small at-tick credit kept for continuity.
    L = out["latency"]
    nn = max(L["n"], 1)
    out["score"] = (
        (1.0 - L["miss_rate"])
        + 0.5 * L["within_20ms"] / nn
        + 0.25 * L["within_40ms"] / nn
        + 0.25 * out["in_disk"] / nn
    )
    out["episodes"] = episodes
    out["ticks"] = int(sum(e["ticks"] for e in episodes))
    out["diverged"] = int(sum(e["diverged"] for e in episodes))
    out["wall_s"] = time.perf_counter() - t_all
    if record_waist_trace:
        out["waist_trace"] = waist_traces
    out["_hits"] = all_hits
    if record_rollout:
        out["_rollouts"] = rollouts
    return out
