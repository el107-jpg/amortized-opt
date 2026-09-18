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

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .fk import ChainFK
from .ik import correct_quintic_init_arms
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
                      waist_cfg=None, clamp_init_to_box: bool = True):
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
    state = {"prev": None, "drift": [], "last_hit": None}
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
            if pending:
                init = correct_quintic_init_arms(
                    _fk(init.device, init.dtype), init, st[:, :n], hit, aims,
                    steps=1, joint_names=G1_UPPER_BODY_JOINTS,
                    qd0=st[:, n:], dt=float(req.dt),
                )
            if clamp_init_to_box:
                init = init.clamp(q_lo_t.to(device=init.device, dtype=init.dtype),
                                  q_hi_t.to(device=init.device, dtype=init.dtype))
            state["last_hit"] = hits
            out = core.refine(init, st, enc, masks, iterations=int(iterations))
        traj = out.traj[0].float().cpu().numpy()
        prev = state["prev"]
        if prev is not None and prev.shape == traj.shape:
            m = masks.elem_mask[0].cpu().numpy()
            d = np.abs(traj[:-1] - prev[1:]) * m[:-1]
            state["drift"].append(float(d.sum() / max(m[:-1].sum(), 1.0)))
        state["prev"] = traj.copy()
        if record is not None:
            record.append({"step": req.step, "hit": dict(req.hit_index),
                           "traj": traj.copy()})
        return traj

    return plan, state


# -------------------------------------------------------------------- scoring --
def score_hits(scorer, kit, res, record=None, *, lead_ticks: int = 0,
               offsets=(-3, -2, -1, 0, 1, 2, 3), drift_back: int = 13):
    """Per executed hit: the plant at several tick offsets, and the plan's drift.

    ``planned`` is the per-tick answer to "what did the plan of that tick put at
    its OWN terminal waypoint?" as a function of the remaining horizon ``h``.
    Flat and near zero = the planner is right and the plant is behind; growing
    as ``h`` shrinks = the planner itself degrades in the final approach.
    """
    q = res.q_hist
    by_step = {r["step"]: r for r in (record or [])}
    hits = []
    for (k0, hand, drum) in res.hit_steps:
        k = int(k0) + int(lead_ticks)
        sigma = default_contact_sigma(kit.names[int(drum)])
        row = {"k": int(k), "k_fire": int(k0), "hand": hand,
               "drum": kit.names[int(drum)], "meas": {}, "planned": []}
        for o in offsets:
            i = k + o
            if 0 <= i < len(q):
                ax, ip, ok = head_frame_error(scorer, q[i], hand, kit, drum, sigma)
                row["meas"][int(o)] = {"axial_mm": 1000 * ax, "inplane_mm": 1000 * ip,
                                       "in_disk": bool(ok)}
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
    for h in hits:
        for p in h["planned"]:
            by_h.setdefault(p["h"], []).append(p["inplane_mm"])
    out["planned_inplane_p50_by_h"] = {int(k): _p50(v) for k, v in sorted(by_h.items())}
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
    freeze_wp: int = 0,
    freeze_waist: bool = True,
    sticky_goal_point: bool = True,
    bounce: object | None = True,
    starts=None,
    record_drift: bool = True,
    waist_upright_ease_wp: int | None = None,
    waist_cfg=None,
    waist_pr_ema: bool = False,
    waist_pr_ema_alpha: float = 0.1,
    waist_pr_ema_leak: float = 0.02,
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

    all_hits, all_lat, episodes = [], [], []
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
                freeze_wp=int(freeze_wp), freeze_waist=bool(freeze_waist),
                sticky_goal_point=bool(sticky_goal_point),
                bounce=bounce,
                waist_upright_ease_wp=waist_upright_ease_wp,
                waist_pr_ema=bool(waist_pr_ema),
                waist_pr_ema_alpha=float(waist_pr_ema_alpha),
                waist_pr_ema_leak=float(waist_pr_ema_leak),
            ),
        )
        record: list = [] if record_drift else None
        plan, st = mpc_model_planner(
            model, num_drums=num_drums, num_future=num_future, device=device,
            amp_dtype=amp_dtype, iterations=iterations, record=record,
            waist_cfg=waist_cfg,
        )
        t0 = time.perf_counter()
        res = gen.run_episode(
            clip, starts.q[ep].cpu().numpy(), starts.qd[ep].cpu().numpy(),
            seed=1000 + ep, planner=plan, max_steps=int(steps),
        )
        wall = time.perf_counter() - t0
        hits = score_hits(scorer, kit, res, record, lead_ticks=lead_ticks)
        all_hits.extend(hits)
        all_lat.extend(hit_latency(scorer, kit, res, lead_ticks=lead_ticks))
        track = (np.abs(res.q_hist[1:] - res.cmd_hist)
                 if len(res.cmd_hist) else np.zeros(1))
        episodes.append({
            "clip": clip.name, "ticks": int(res.cmd_hist.shape[0]),
            "hits": len(hits), "diverged": bool(res.diverged),
            "frozen_ticks": int(getattr(res, "frozen_ticks", 0)),
            "bounced_hands": int(getattr(res, "bounced_hands", 0)),
            "pd_track_p50_mrad": float(np.percentile(track, 50) * 1000),
            "pd_track_p90_mrad": float(np.percentile(track, 90) * 1000),
            "tc_drift_p50_mrad": (float(np.percentile(st["drift"], 50) * 1000)
                                  if st["drift"] else float("nan")),
            "ms_per_tick": 1000 * wall / max(res.cmd_hist.shape[0], 1),
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
    out["_hits"] = all_hits
    return out
