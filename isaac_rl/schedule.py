"""Batched, GPU-resident rhythm scheduler for the Isaac RL loop.

``amortized_opt.mpc_gen.MPCGenerator`` rebuilds a
:class:`~amortized_opt.rhythm.GoalSnapshot` from the clip on every tick, in
host-side Python, for ONE env.  That costs ~1 ms per tick per env, which is
fine for a 3-clip evaluation and impossible at 256 envs x 50 Hz.

This module is the batched twin.  Every clip is flattened ONCE, at reset, into
per-env padded tables

    hit_t     (B, 2, M)      absolute (quantized) strike time, seconds
    hit_drum  (B, 2, M)      drum index into the env's own kit rows
    hit_pos   (B, 2, M, 3)   the STICKY aim point on the head, env frame
    hit_vel   (B, 2, M, 3)   commanded impact velocity
    hit_valid (B, 2, M)

and every tick is a gather at a per-hand cursor.  Because the aim point is
drawn once, at reset, and read from the table thereafter, this reproduces
``MPCGenConfig.sticky_goal_point = True`` by construction -- the eval/deployment
mismatch section 7c found (a target teleporting by up to 0.2 m every 20 ms)
cannot occur here.

Semantics deliberately mirrored from ``MPCGenerator``, term for term:

* ``_snapshot`` queries the clip strictly FORWARD of the current instant
  (``SNAP_EPS``), so a hit at exactly ``t_now`` has already been executed;
* ``hit_lead_s`` shifts the QUERY (``t_now + hit_lead_s``), not the goals'
  ``time_to_hit`` -- post-shifting makes every hit fire ``lead/dt + 1`` times;
* ``_hit_index`` is ``k = round(tth / dt)`` clamped to ``>= 1``, and ``-1``
  (idle) past the horizon cap;
* ``goal_snapshot(horizon_cap=...)`` keeps an out-of-range hit OUT of the goal
  slot and leaves it as ``future[0]``.

Isaac-free: pure torch + numpy + ``amortized_opt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, PELVIS_IN_ENV, STICK_TIP_SIGMA
from amortized_opt.cost import StrikeContext, drum_heads_from_context
from amortized_opt.phase2_data import drum_name_predicates, strike_geometry_table
from amortized_opt.train_cost import effective_hits, group_lengths
from amortized_opt.rhythm import (
    DrumKit,
    KitRandomization,
    RhythmClip,
    sample_hit_point,
    strike_velocity,
)
from amortized_opt.tokenizer import ContextInputs

__all__ = ["BatchedSchedule", "ScheduleConfig", "clip_hit_table"]

HANDS = ("left", "right")


@dataclass
class ScheduleConfig:
    dt: float = CONTROL_DT
    horizon: int = 50
    num_drums: int = 8
    num_future: int = 3
    #: Beyond this many waypoints a pending hit is not yet the "current goal".
    #: ``None`` = the full horizon (``MPCGenConfig.horizon_cap_wp``).
    horizon_cap_wp: int | None = None
    hit_radius_frac: float = 0.6
    speed_min: float = 1.0
    speed_max: float = 3.0
    #: ``MPCGenConfig.hit_lead_s`` -- schedule every goal this many seconds
    #: early.  0 = off; the champion configuration ships it off and uses the
    #: per-joint reference lead instead (7c decision 2).
    hit_lead_s: float = 0.0
    #: Randomize each env's kit (per-drum xy, global xyz) at reset.  The
    #: phase-2 training distribution used 4 cm / 7.5 cm / 3.5 cm.
    randomize_kit: bool = False
    kit_randomization: KitRandomization | None = None
    #: Maximum hits per hand per env in the padded table.
    max_hits: int = 256

    @property
    def cap_wp(self) -> int:
        return int(self.horizon_cap_wp or self.horizon)


# ------------------------------------------------------------- host tables --
def clip_hit_table(
    clip: RhythmClip,
    kit: DrumKit,
    rng: np.random.Generator | None,
    *,
    cfg: ScheduleConfig,
) -> dict[str, np.ndarray]:
    """One clip x one kit -> the per-hand padded arrays described above.

    The aim jitter is drawn HERE, once per scheduled hit, which is the sticky
    semantics.  ``rng=None`` puts every aim exactly on the drum's canonical
    strike point (deterministic contexts for tests and for the ``sigma = 0``
    smoke run).
    """
    M = int(cfg.max_hits)
    t = np.zeros((2, M))
    drum = np.zeros((2, M), np.int64)
    pos = np.zeros((2, M, 3))
    vel = np.zeros((2, M, 3))
    valid = np.zeros((2, M), bool)
    for hi, hand in enumerate(HANDS):
        hits = [h for h in clip.hits if h.hand == hand]
        if len(hits) > M:
            hits = hits[:M]
        for j, h in enumerate(hits):
            t[hi, j] = float(h.time)
            drum[hi, j] = int(h.drum)
            pos[hi, j] = sample_hit_point(
                kit, int(h.drum), rng, hit_radius_frac=cfg.hit_radius_frac
            )
            vel[hi, j] = strike_velocity(
                kit, int(h.drum), float(h.velocity),
                speed_min=cfg.speed_min, speed_max=cfg.speed_max,
            )
            valid[hi, j] = True
        # Padding times must sort AFTER every real hit or the cursor logic can
        # walk backwards; +inf is the honest filler and is never gathered
        # because ``hit_valid`` gates it, but keep it finite for fp16 safety.
        t[hi, len(hits):] = 1e9
    return {"hit_t": t, "hit_drum": drum, "hit_pos": pos, "hit_vel": vel,
            "hit_valid": valid, "n_hits": np.array([int(valid[0].sum()),
                                                    int(valid[1].sum())])}


# ---------------------------------------------------------------- schedule --
class BatchedSchedule:
    """Per-env rhythm state: the hit tables, the cursors, and the clock.

    The public surface is three calls per control tick::

        sched.context()            # -> ContextInputs, one row per env
        sched.hit_index()          # -> (B, 2) long, waypoints to each hit, -1 idle
        fired = sched.advance()    # -> (B, 2) bool, hands whose hit just passed

    ``advance`` moves the clock by one control period and consumes any hit the
    clock passed; the returned mask is what drives the per-hand plan re-seed and
    (plan item 2) the redraw of the per-env parameter-noise epsilon.
    """

    def __init__(self, n_envs: int, cfg: ScheduleConfig, *, device, dtype=torch.float32):
        self.B = int(n_envs)
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        B, M, D, F = self.B, int(cfg.max_hits), int(cfg.num_drums), int(cfg.num_future)
        z = lambda *s, d=dtype: torch.zeros(*s, device=self.device, dtype=d)  # noqa: E731
        # TIME IS FLOAT64 AND DERIVED FROM AN INTEGER TICK COUNT, always.
        # Accumulating ``t_now += 0.02`` in float32 drifts: measured against
        # ``MPCGenerator._hit_index`` on a 46 s clip it flipped the comparison
        # at a hit landing exactly on a tick boundary (step 85, t = 1.70 s) and
        # left the cursor one hit behind for the rest of the episode.  The clip
        # times are quantized to the control grid, so EVERY hit sits exactly on
        # a boundary and every one of them is that knife edge.
        self.hit_t = torch.full((B, 2, M), 1e9, device=self.device, dtype=torch.float64)
        self.hit_drum = z(B, 2, M, d=torch.long)
        self.hit_pos = z(B, 2, M, 3)
        self.hit_vel = z(B, 2, M, 3)
        self.hit_valid = z(B, 2, M, d=torch.bool)
        self.cursor = z(B, 2, d=torch.long)
        self.tick = z(B, d=torch.long)
        self.t0 = z(B, d=torch.float64)
        self.duration = torch.full((B,), 1e9, device=self.device, dtype=torch.float64)
        self.clip_id = z(B, d=torch.long)
        # per-env kit
        self.drum_pos = z(B, D, 3)
        self.drum_axis = z(B, D, 3)
        self.drum_radius = z(B, D)
        self.drum_height = z(B, D)
        self.drum_valid = z(B, D, d=torch.bool)
        self._M = M
        # ---- strike geometry -------------------------------------------------
        #: ``(B, D, ...)`` per-drum strike table for THIS env's (possibly
        #: randomized) kit, rebuilt at reset.  Same construction as
        #: ``MPCRoller._refresh_geometry``; the numpy path it agrees with to
        #: 1e-6 is ``phase2_data._strike_row``.
        self.kit_names: tuple[str, ...] = ()
        self._pred: dict[str, torch.Tensor] | None = None
        self.geom: dict[str, torch.Tensor] = {}
        #: departure exemption: the drum each hand most recently STRUCK, and
        #: how many control ticks its collision exemption still has to run.
        self.start_drum = torch.full((B, 2), -1, device=self.device, dtype=torch.long)
        self.start_ticks = z(B, 2, d=torch.long)
        self.departure_exempt_wp = 4
        self._pelvis = torch.as_tensor(PELVIS_IN_ENV, device=self.device, dtype=dtype)

    # ------------------------------------------------------------- reset ---
    def reset(
        self,
        env_ids: torch.Tensor,
        clips: list[RhythmClip],
        kits: list[DrumKit],
        *,
        t_start=0.0,
        rng: np.random.Generator | None = None,
        clip_ids: list[int] | None = None,
    ) -> None:
        """Install ``len(env_ids)`` (clip, kit) pairs.  Host-side, at reset only."""
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1).long()
        n = int(env_ids.numel())
        if n == 0:
            return
        if len(clips) != n or len(kits) != n:
            raise ValueError(f"{len(clips)} clips / {len(kits)} kits for {n} envs")
        tabs = [clip_hit_table(c, k, rng, cfg=self.cfg) for c, k in zip(clips, kits)]
        t = lambda a, d=self.dtype: torch.as_tensor(  # noqa: E731
            np.asarray(a), device=self.device, dtype=d)
        self.hit_t.index_copy_(0, env_ids, t([x["hit_t"] for x in tabs], torch.float64))
        self.hit_drum.index_copy_(0, env_ids, t([x["hit_drum"] for x in tabs], torch.long))
        self.hit_pos.index_copy_(0, env_ids, t([x["hit_pos"] for x in tabs]))
        self.hit_vel.index_copy_(0, env_ids, t([x["hit_vel"] for x in tabs]))
        self.hit_valid.index_copy_(0, env_ids, t([x["hit_valid"] for x in tabs], torch.bool))
        self.cursor.index_copy_(0, env_ids, torch.zeros_like(env_ids).unsqueeze(-1)
                                .expand(n, 2).contiguous())
        ts = np.full(n, float(t_start)) if np.isscalar(t_start) else np.asarray(t_start, float)
        self.t0.index_copy_(0, env_ids, t(ts, torch.float64))
        self.tick.index_copy_(0, env_ids, torch.zeros_like(env_ids))
        self.duration.index_copy_(0, env_ids, t([c.duration for c in clips], torch.float64))
        cid = clip_ids if clip_ids is not None else [0] * n
        self.clip_id.index_copy_(0, env_ids, t(cid, torch.long))

        D = int(self.cfg.num_drums)
        dp = np.zeros((n, D, 3)); da = np.zeros((n, D, 3))
        dr = np.zeros((n, D)); dh = np.zeros((n, D)); dv = np.zeros((n, D), bool)
        for i, k in enumerate(kits):
            d = k.num_drums
            if d > D:
                raise ValueError(f"kit has {d} drums, num_drums={D}")
            dp[i, :d] = k.positions_in("env")
            da[i, :d] = k.axes
            dr[i, :d] = k.radii
            dh[i, :d] = k.heights
            dv[i, :d] = True
        self.drum_pos.index_copy_(0, env_ids, t(dp))
        self.drum_axis.index_copy_(0, env_ids, t(da))
        self.drum_radius.index_copy_(0, env_ids, t(dr))
        self.drum_height.index_copy_(0, env_ids, t(dh))
        self.drum_valid.index_copy_(0, env_ids, t(dv, torch.bool))
        if not self.kit_names:
            names = tuple(kits[0].names)
            self.kit_names = names + ("",) * max(0, D - len(names))
            self._pred = drum_name_predicates(self.kit_names, device=self.device,
                                              dtype=self.dtype)
        self._refresh_geometry(env_ids)
        self.start_drum.index_copy_(
            0, env_ids, torch.full((n, 2), -1, device=self.device, dtype=torch.long))
        self.start_ticks.index_copy_(
            0, env_ids, torch.zeros((n, 2), device=self.device, dtype=torch.long))
        # roll the cursor forward to the first hit at/after t_start
        self._sync_cursor(env_ids)

    @torch.no_grad()
    def _refresh_geometry(self, env_ids: torch.Tensor) -> None:
        """Re-price the ``(D,)`` strike table for ``env_ids``' randomized kits."""
        tab = strike_geometry_table(
            self.drum_pos.index_select(0, env_ids) - self._pelvis,
            self.drum_axis.index_select(0, env_ids),
            self.drum_radius.index_select(0, env_ids),
            self.kit_names,
            predicates=self._pred,
        )
        for k, v in tab.items():
            if k not in self.geom:
                self.geom[k] = torch.zeros((self.B,) + tuple(v.shape[1:]),
                                           device=self.device, dtype=v.dtype)
            self.geom[k].index_copy_(0, env_ids, v.contiguous())

    def _sync_cursor(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance every cursor past hits the clock has already gone by."""
        tq = self._query_time().unsqueeze(-1).unsqueeze(-1)          # (B,1,1)
        past = (self.hit_t <= tq) & self.hit_valid                    # (B,2,M)
        self.cursor = past.sum(dim=2).long()

    # ------------------------------------------------------------ queries --
    @property
    def t_now(self) -> torch.Tensor:
        """``(B,)`` float64 clock, exact: ``t0 + tick * dt``, never accumulated."""
        return self.t0 + self.tick.to(torch.float64) * float(self.cfg.dt)

    def _query_time(self) -> torch.Tensor:
        """``t_now + hit_lead_s + SNAP_EPS`` -- the instant the clip is read at.

        ``SNAP_EPS`` (1e-6) is ``MPCGenerator.SNAP_EPS``: a hit at exactly
        ``t_now`` has already been executed by the tick that reached it, so the
        query is strictly forward.
        """
        return self.t_now + float(self.cfg.hit_lead_s) + 1e-6

    def _gather(self, k: int):
        """Row ``cursor + k`` of the table, with an out-of-range validity mask."""
        idx = (self.cursor + int(k)).clamp(0, self._M - 1)            # (B, 2)
        ok = (self.cursor + int(k)) < self._M
        v = self.hit_valid.gather(2, idx.unsqueeze(-1)).squeeze(-1) & ok
        t = self.hit_t.gather(2, idx.unsqueeze(-1)).squeeze(-1)
        d = self.hit_drum.gather(2, idx.unsqueeze(-1)).squeeze(-1)
        i3 = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        p = self.hit_pos.gather(2, i3).squeeze(2)
        w = self.hit_vel.gather(2, i3).squeeze(2)
        return t, d, p, w, v

    def goal(self):
        """``(tth, drum, pos, vel, valid, active)`` for the CURRENT goal.

        ``valid`` is "there is a next hit at all"; ``active`` additionally
        requires it to be inside the horizon cap, which is what
        ``goal_snapshot(horizon_cap=...)`` does.
        """
        t, d, p, w, v = self._gather(0)
        tth = t - self._query_time().unsqueeze(-1)
        cap = self.cfg.cap_wp * self.cfg.dt
        active = v & (tth <= cap + 1e-9)
        tth = torch.where(active, tth, torch.zeros_like(tth))
        # float64 is the CLOCK's dtype; the tokens go out in the model's.
        return tth.to(self.dtype), d, p, w, v, active

    def hit_index(self) -> torch.Tensor:
        """``(B, 2)`` long: waypoints to each hand's hit, ``-1`` when idle.

        ``max(round(tth/dt), 1)`` -- a goal is never "now": the soonest
        actionable terminal is the first waypoint.
        """
        tth, _d, _p, _w, _v, active = self.goal()
        k = torch.round(tth / self.cfg.dt).long().clamp_min(1)
        return torch.where(active, k.clamp_max(self.cfg.cap_wp), torch.full_like(k, -1))

    def context(self) -> ContextInputs:
        """The encoder's ``ContextInputs``, one row per env, on device."""
        cfg = self.cfg
        tth, gd, gp, gv, gvalid, gactive = self.goal()
        # An out-of-range goal is not the goal: it becomes future[0], exactly as
        # ``goal_snapshot`` does when ``horizon_cap`` rejects ``nxt[0]``.
        shift = (gvalid & ~gactive).long()                             # (B,2)
        F = int(cfg.num_future)
        ft = []; fd = []; fp = []; fv = []; fval = []
        for j in range(F):
            t, d, p, w, v = self._gather_shifted(1 + j, shift)
            ft.append((t - self._query_time().unsqueeze(-1)).to(self.dtype))
            fd.append(d); fp.append(p); fv.append(w); fval.append(v)
        stack = lambda xs, dim=2: torch.stack(xs, dim=dim)             # noqa: E731
        f_tth = stack(ft)
        f_val = stack(fval)
        return ContextInputs(
            drum_pos=self.drum_pos, drum_axis=self.drum_axis,
            drum_radius=self.drum_radius, drum_height=self.drum_height,
            drum_valid=self.drum_valid,
            goal_drum=torch.where(gactive, gd, torch.zeros_like(gd)),
            goal_pos=gp * gactive.unsqueeze(-1),
            goal_vel=gv * gactive.unsqueeze(-1),
            goal_tth=tth, goal_valid=gactive,
            future_drum=torch.where(f_val, stack(fd), torch.zeros_like(stack(fd))),
            future_pos=stack(fp) * f_val.unsqueeze(-1),
            future_vel=stack(fv) * f_val.unsqueeze(-1),
            future_tth=torch.where(f_val, f_tth, torch.zeros_like(f_tth)),
            future_valid=f_val,
        )

    def _gather_shifted(self, k: int, shift: torch.Tensor):
        """``_gather(k - shift)`` -- the out-of-range goal reappears as future 0."""
        off = (int(k) - shift).clamp_min(0)
        idx = (self.cursor + off).clamp(0, self._M - 1)
        ok = (self.cursor + off) < self._M
        v = self.hit_valid.gather(2, idx.unsqueeze(-1)).squeeze(-1) & ok
        t = self.hit_t.gather(2, idx.unsqueeze(-1)).squeeze(-1)
        d = self.hit_drum.gather(2, idx.unsqueeze(-1)).squeeze(-1)
        i3 = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        p = self.hit_pos.gather(2, i3).squeeze(2)
        w = self.hit_vel.gather(2, i3).squeeze(2)
        return t, d, p, w, v

    # ------------------------------------------------------ strike context --
    def strike_context(self, hit_index: torch.Tensor | None = None) -> StrikeContext:
        """The :class:`~amortized_opt.cost.StrikeContext` for THIS tick.

        Every row is gathered from the env's own ``geom`` table at its goal
        drum, so a domain-randomized kit is priced against the layout the model
        was actually told about -- the same guarantee ``MPCRoller._regoal``
        provides by re-pricing in place.  Field-for-field this is
        ``phase2_data._strike_row``, batched.
        """
        if hit_index is None:
            hit_index = self.hit_index()
        _tth, gd, gp, gv, _gvalid, active = self.goal()
        # ``effective_hits``: an idle hand borrows the other's horizon, which is
        # exactly what ``build_waypoint_masks`` does.  Getting this wrong
        # silently prices only the first segment of an idle arm.
        hit_eff = effective_hits(hit_index, active, self.cfg.horizon)
        d = gd.clamp(0, int(self.cfg.num_drums) - 1)
        g = self.geom
        gi = d.unsqueeze(-1).expand(-1, -1, 3)
        axis = g["axis"].gather(1, gi)
        disk_c = g["disk_center"].gather(1, gi)
        p_star = gp - self._pelvis
        vn = gv.norm(dim=-1, keepdim=True)
        n_out = torch.where(vn > 1e-9, -gv / vn.clamp_min(1e-30), axis)
        off = disk_c - p_star
        off = off - (off * axis).sum(-1, keepdim=True) * axis
        af = active.to(self.dtype)
        one = lambda k: g[k].gather(1, d)                                # noqa: E731
        e_up = torch.zeros_like(n_out)
        e_up[..., 2] = 1.0
        # the departure exemption: the drum this hand last STRUCK, for as long
        # as its exemption window runs (mpc_gen ``departure_exempt_wp``).
        exempt = self.start_ticks > 0
        start_drum = torch.where(exempt, self.start_drum, torch.full_like(self.start_drum, -1))
        start_shaft = torch.where(
            exempt, self._pred["shaft"][self.start_drum.clamp_min(0)],
            torch.zeros_like(af),
        )
        return StrikeContext(
            hit_index=hit_eff,
            active=af,
            target_drum=torch.where(active, d, torch.full_like(d, -1)),
            start_drum=start_drum,
            p_star=p_star * af.unsqueeze(-1),
            v_star=gv * af.unsqueeze(-1),
            n_out=torch.where(active.unsqueeze(-1), n_out, e_up),
            hit_radius=torch.where(active, one("hit_radius"),
                                   torch.full_like(af, 0.08)),
            disk_offset=off * af.unsqueeze(-1),
            half_plane_dir=g["hp_dir"].gather(1, gi) * af.unsqueeze(-1),
            has_half_plane=one("has_hp") * af,
            contact_sigma=torch.where(active, one("sigma"),
                                      torch.full_like(af, float(STICK_TIP_SIGMA))),
            shaft_allowed=one("shaft") * af,
            start_shaft_allowed=start_shaft,
            elev_required=one("elev") * af,
            drum_head_centers=drum_heads_from_context(self.drum_pos),
        )

    # -------------------------------------------------------------- clock ---
    def advance(self) -> torch.Tensor:
        """Step the clock one control period; return ``(B, 2)`` fired mask.

        A hand "fires" on the tick whose clock crosses its scheduled instant --
        the same instant ``MPCGenerator`` records as ``(step + 1, hand, drum)``
        when its ``hit_index`` reaches 0.

        Also runs the departure-exemption countdown, so the hand that just
        struck stops being scored as colliding with the head it is leaving.
        """
        before = self.cursor.clone()
        struck = self.hit_drum.gather(2, self.cursor.clamp(0, self._M - 1)
                                      .unsqueeze(-1)).squeeze(-1)
        self.tick = self.tick + 1
        self._sync_cursor()
        fired = self.cursor > before
        self.start_drum = torch.where(fired, struck, self.start_drum)
        self.start_ticks = torch.where(
            fired, torch.full_like(self.start_ticks, int(self.departure_exempt_wp)),
            (self.start_ticks - 1).clamp_min(0),
        )
        return fired

    def scheduled_tick(self) -> torch.Tensor:
        """``(B, 2)`` long: the control tick index of each hand's current hit.

        Measured from the clip origin, so it survives the cursor moving.  Used
        by the reward's timing window and by the hit-latency histogram.
        """
        t, _d, _p, _w, v = self._gather(0)
        k = torch.round(t / self.cfg.dt).long()
        return torch.where(v, k, torch.full_like(k, -1))

    def done(self) -> torch.Tensor:
        """``(B,)`` bool: the clip is over (no hit left, or past its duration)."""
        _t, _d, _p, _w, v = self._gather(0)
        return (~v.any(dim=1)) | (self.t_now >= self.duration)
