"""64-chart expanding start clouds for a later generalization stage.

Index by current dual drum mode only. Each collocation is already
``(q, qd, t_hit)``; waist yaw lives in ``q0``, not in the chart. Next drums
and hit-relative ``future_gap`` are copied from the parent. TTH jitter that
would change ``bucket_for`` is rejected.

Seed ingest copies **every** existing pool row (``k=0``) and also every
along-traj slice (stride ``traj_stride``). Those seed rows grow the buffer;
``chart_cap`` only limits later children / harvest. Dual TTH jitter perturbs
the left clock only.

Default-off: the trainer never constructs a cloud unless ``Phase2Config.frontier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .warmstart_data import bucket_for

N_MODE = 8
N_CHART = N_MODE * N_MODE
YAW_I = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")

_REBUILT = frozenset({"q0", "qd0", "init", "hit", "goal_tth", "yaw_now", "traj_k"})


def chart_id(active: torch.Tensor, drum: torch.Tensor) -> torch.Tensor:
    """``(N,)`` chart in ``0..63``. Idle hand → mode 0; else ``drum+1``."""
    act = active.to(torch.bool)
    d = drum.to(torch.long)
    mode = torch.where(act, (d + 1).clamp(1, N_MODE - 1), torch.zeros_like(d))
    return mode[:, 0] * N_MODE + mode[:, 1]


def occupancy(charts: torch.Tensor, n_chart: int = N_CHART) -> torch.Tensor:
    """``(n_chart,)`` counts."""
    out = torch.zeros(int(n_chart), device=charts.device, dtype=torch.long)
    if charts.numel() == 0:
        return out
    ones = torch.ones_like(charts, dtype=torch.long)
    out.scatter_add_(0, charts.clamp(0, int(n_chart) - 1), ones)
    return out


def _bucket_ids(k: torch.Tensor, buckets: tuple[int, ...]) -> torch.Tensor:
    """Batched :func:`bucket_for`: smallest bucket with ``k <= b``."""
    k = k.to(torch.long)
    out = torch.full(k.shape, int(buckets[-1]), device=k.device, dtype=torch.long)
    for b in reversed(buckets):
        out = torch.where(k <= int(b), torch.full_like(out, int(b)), out)
    return out


def _effective_hits(hit: torch.Tensor, active: torch.Tensor, H: int) -> torch.Tensor:
    al, ar = active[:, 0].bool(), active[:, 1].bool()
    kl, kr = hit[:, 0].clamp_min(0), hit[:, 1].clamp_min(0)
    neg = torch.full_like(kl, -1)
    both = torch.maximum(torch.where(al, kl, neg), torch.where(ar, kr, neg))
    both = torch.where(al | ar, both, torch.full_like(both, int(H)))
    return torch.stack([torch.where(al, kl, both), torch.where(ar, kr, both)], dim=1)


def _group_lengths(hit_eff: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.maximum(hit_eff[:, 0], hit_eff[:, 1]), hit_eff[:, 0], hit_eff[:, 1]],
        dim=1,
    )


_ELEM_GROUP: torch.Tensor | None = None


def _element_group(device) -> torch.Tensor:
    global _ELEM_GROUP
    if _ELEM_GROUP is None or _ELEM_GROUP.device != device:
        g = [
            1 if n.startswith("left_") else (2 if n.startswith("right_") else 0)
            for n in G1_UPPER_BODY_JOINTS
        ]
        _ELEM_GROUP = torch.as_tensor(g, device=device, dtype=torch.long)
    return _ELEM_GROUP


def quintic_init(
    q0: torch.Tensor,
    terminal: torch.Tensor,
    hit: torch.Tensor,
    active: torch.Tensor,
    H: int,
) -> torch.Tensor:
    """Rest-to-rest quintic ``q0 → terminal`` over per-joint horizons, then hold.

    Same shape as ``WarmStart.init_traj`` / roller ``ramp_seed``: ``(B, H, 17)``.
    Terminal is gathered from the *parent* strike knot by the caller.
    """
    hit_eff = _effective_hits(hit, active, H)
    lengths = _group_lengths(hit_eff).index_select(1, _element_group(q0.device))
    k = torch.arange(1, H + 1, device=q0.device, dtype=q0.dtype).view(1, H, 1)
    L = lengths.clamp_min(1).to(q0.dtype).unsqueeze(1)
    s = (k / L).clamp(0.0, 1.0)
    ease = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
    return q0.unsqueeze(1) + ease * (terminal - q0).unsqueeze(1)


def parent_terminal(init: torch.Tensor, hit: torch.Tensor, active: torch.Tensor, H: int) -> torch.Tensor:
    """Strike pose at the parent's hit knot, ``(B, 17)``."""
    hit_eff = _effective_hits(hit, active, H).clamp(1, int(H))
    per = _group_lengths(hit_eff).index_select(1, _element_group(init.device))
    idx = (per.clamp(1, int(H)) - 1).unsqueeze(1)
    return init.gather(1, idx).squeeze(1)


def ball_q_scale(device=None, dtype=None) -> torch.Tensor:
    """Per-joint weights for the start ball (× ``sigma_q``).

    Waist and shoulders already move a lot across the cloud (leftover yaw,
    reach). An isotropic 0.05 rad ball, or empirical-std whitening, *amplifies*
    those axes. Weights scale them down; wrists stay at 1.
    """
    w = []
    for n in G1_UPPER_BODY_JOINTS:
        if n == "waist_yaw_joint":
            w.append(0.25)
        elif n.startswith("waist_"):
            w.append(0.35)
        elif "shoulder" in n:
            w.append(0.50)
        elif "elbow" in n:
            w.append(0.75)
        else:
            w.append(1.00)
    t = torch.tensor(w, dtype=dtype or torch.float32)
    return t.to(device) if device is not None else t


def jitter_hit(
    hit: torch.Tensor,
    active: torch.Tensor,
    *,
    dt: float,
    bucket: int,
    sigma_t: float,
    generator: torch.Generator | None = None,
    buckets: tuple[int, ...] = (25, 50, 100),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize TTH. Returns ``(hit', keep)``; ``keep`` is False on bucket change.

    ``sigma_t`` is a **fraction of remaining TTH** (0.05 = 5 %), not seconds —
    a 0.4 s dual and a 1 s idle are different scales. Dual: only the left
    clock is jittered; right ``hit`` stays put. Idle rows (no active hand)
    are kept. A child whose ``bucket_for(max hit')`` leaves the parent
    bucket is rejected -- unlike along-traj slices.
    """
    H = int(bucket)
    act = active.to(torch.bool)
    tth = hit.clamp_min(0).to(dtype=torch.float32) * float(dt)
    eps = torch.zeros_like(tth)
    if float(sigma_t) > 0.0:
        z = torch.randn(tth.shape, device=tth.device, dtype=tth.dtype, generator=generator)
        eps = z * (float(sigma_t) * tth)
    # Dual: left only. Right jitters only when the left hand is idle.
    jitter = act.clone()
    jitter[:, 1] = act[:, 1] & ~act[:, 0]
    tth2 = torch.where(jitter, (tth + eps).clamp(float(dt), float(H) * float(dt)), tth)
    hit2 = torch.where(
        jitter,
        (tth2 / float(dt)).round().to(torch.long).clamp(1, H),
        hit,
    )
    any_act = act.any(dim=1)
    k_old = torch.where(act, hit.clamp_min(0), torch.zeros_like(hit)).max(dim=1).values
    k_new = torch.where(act, hit2.clamp_min(0), torch.zeros_like(hit2)).max(dim=1).values
    new_b = _bucket_ids(k_new, buckets)
    old_b = _bucket_ids(k_old.clamp_min(1), buckets)
    keep = (~any_act) | (
        (k_new >= 1) & (new_b == int(bucket)) & (old_b == int(bucket))
    )
    return hit2, keep


def rebuild_clocks(
    cols: dict[str, torch.Tensor],
    *,
    dt: float,
    yaw_i: int = YAW_I,
) -> dict[str, torch.Tensor]:
    """``goal_tth = hit * dt`` on active hands; ``yaw_now`` from ``q0``. Copy ``future_*``."""
    out = dict(cols)
    hit = out["hit"]
    act = out["active"].to(torch.bool)
    gdtype = out["goal_tth"].dtype if "goal_tth" in out else out["q0"].dtype
    gt = hit.clamp_min(0).to(gdtype) * float(dt)
    if "goal_tth" in out:
        out["goal_tth"] = torch.where(act, gt, out["goal_tth"].to(gdtype))
    else:
        out["goal_tth"] = torch.where(act, gt, torch.zeros_like(gt))
    out["yaw_now"] = out["q0"][:, int(yaw_i)].to(out["q0"].dtype)
    return out


def apply_q_ball(
    q0: torch.Tensor,
    qd0: torch.Tensor,
    *,
    sigma_q: float,
    sigma_qd: float,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    q_scale: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anisotropic ball around the parent. Clip q to the URDF box.

    ``delta_q_i = sigma_q * scale_i * eps``, ``delta_qd_i = sigma_qd * scale_i * eps``.
    ``scale`` is :func:`ball_q_scale` (waist/shoulder < 1). TTH is not in this
    vector — it has its own remaining-time fraction in :func:`jitter_hit`.
    """
    scale = ball_q_scale(device=q0.device, dtype=q0.dtype) if q_scale is None else q_scale
    scale = scale.to(device=q0.device, dtype=q0.dtype).view(1, -1)
    eq = torch.randn(q0.shape, device=q0.device, dtype=q0.dtype, generator=generator)
    ev = torch.randn(qd0.shape, device=qd0.device, dtype=qd0.dtype, generator=generator)
    q2 = (q0 + float(sigma_q) * scale * eq).clamp(q_lo, q_hi)
    qd2 = qd0 + float(sigma_qd) * scale * ev
    return q2, qd2


def _hold_pad(suffix: torch.Tensor, dest_h: int, hold: torch.Tensor) -> torch.Tensor:
    """Suffix of ``q_full[k+1:]``, held out to ``dest_h`` knots. No re-quintic."""
    n, length, dim = suffix.shape
    out = suffix.new_zeros(n, int(dest_h), dim)
    if length <= 0:
        out[:] = hold.unsqueeze(1)
        return out
    take = min(length, int(dest_h))
    out[:, :take] = suffix[:, :take]
    if take < int(dest_h):
        out[:, take:] = suffix[:, take - 1 : take]
    return out


def slice_at_knot(
    cols: dict[str, torch.Tensor],
    k: int,
    *,
    dt: float,
    buckets: tuple[int, ...] = (25, 50, 100),
) -> dict[int, dict[str, torch.Tensor]]:
    """Along-traj collocations at knot ``k``. ``{dest_bucket: cols}``; empty if none.

    ``hit' = hit - k`` on active hands. Skip a row if any active ``hit' < 1``.
    Destination bucket is ``bucket_for(max hit')`` and **may be shorter** than
    the parent -- that is the point. ``future_*`` / drums are copied, not
    re-anchored. ``init'`` is the remaining suffix, held to the new ``H``.
    """
    k = int(k)
    if k <= 0:
        return {}
    init = cols["init"]
    h = int(init.shape[1])
    if k > h:
        return {}
    act = cols["active"].to(torch.bool)
    hit = cols["hit"].to(torch.long)
    hit2 = torch.where(act, hit - k, hit)
    keep = act.any(dim=1) & ~((act & (hit2 < 1)).any(dim=1))
    if not bool(keep.any()):
        return {}
    loc = keep.nonzero(as_tuple=True)[0]
    k_max = torch.where(act, hit2.clamp_min(0), torch.zeros_like(hit2)).max(dim=1).values
    dest = _bucket_ids(k_max, buckets)
    q0 = cols["q0"]
    q_full = torch.cat([q0.unsqueeze(1), init], dim=1)
    q0p = q_full[:, k]
    nxt = min(k + 1, h)
    qd0p = (q_full[:, nxt] - q_full[:, k]) / float(dt)
    suffix = q_full[:, k + 1 :]
    out: dict[int, dict[str, torch.Tensor]] = {}
    dest_kept = dest.index_select(0, loc)
    for b in dest_kept.unique().tolist():
        b = int(b)
        sel = loc[dest_kept == b]
        sub: dict[str, torch.Tensor] = {}
        for key, val in cols.items():
            if key.startswith("_") or key in _REBUILT or not torch.is_tensor(val):
                continue
            sub[key] = val.index_select(0, sel)
        sub["q0"] = q0p.index_select(0, sel)
        sub["qd0"] = qd0p.index_select(0, sel)
        sub["hit"] = hit2.index_select(0, sel)
        sub["init"] = _hold_pad(
            suffix.index_select(0, sel), b, q0p.index_select(0, sel),
        )
        sub["traj_k"] = torch.full(
            (sel.numel(),), k, device=sel.device, dtype=torch.long,
        )
        if "goal_tth" in cols:
            sub["goal_tth"] = cols["goal_tth"].index_select(0, sel)
        if "yaw_now" in cols:
            sub["yaw_now"] = cols["yaw_now"].index_select(0, sel)
        out[b] = rebuild_clocks(sub, dt=float(dt))
    return out


@dataclass
class FrontierConfig:
    enabled: bool = False
    proposals: int = 64
    chart_cap: int = 2048
    traj_stride: int = 4
    fresh_frac: float = 0.5
    spawn_frac: float = 0.25
    sigma_q: float = 0.05
    #: rad/s; same per-joint weights as ``sigma_q``. Velocity is not radians.
    sigma_qd: float = 0.10
    #: Fraction of remaining TTH, not seconds.
    sigma_tth: float = 0.05
    #: 0.25: two hits from the 0.5 prior clear 0.70; three reach ~0.79.
    ema_alpha: float = 0.25
    #: Parent sampling holds the 0.5 prior until this many EMA updates.
    ema_nmin: int = 2
    #: EMA value at which a row is proficient (2 hits at alpha=0.25).
    ema_prof: float = 0.70
    #: Farthest-point landmarks per chart, in proposal-sigma units.
    landmark_k: int = 32
    #: Min sigma-unit distance to add a landmark / keep a proposed child.
    landmark_tau: float = 2.0
    #: Cap on chart parents scored against landmarks each propose.
    propose_candidates: int = 256
    #: Spawn only from hit-leaning parents when any exist (hull, not core).
    hull_ema: float = 0.55
    #: Train-batch fraction drawn inverse-density (min-dist to landmarks).
    train_inv_frac: float = 0.5
    #: Uniform candidate pool size for the inverse-density half.
    train_inv_pool: int = 2048
    include_miss: bool = False
    dt: float = CONTROL_DT
    buckets: tuple[int, ...] = (25, 50, 100)


@dataclass
class FrontierCloud:
    """Per-bucket rings of collocation rows, keyed by :func:`chart_id`.

    Layout: seed block ``[0, n_seed)`` (pool ``k=0`` + along-traj slices,
    ``from_pool=True``) then a child ring of size ``chart_cap * 64`` that wraps
    without touching seed.
    """

    cfg: FrontierConfig
    device: torch.device
    dtype: torch.dtype = torch.float32
    q_lo: torch.Tensor | None = None
    q_hi: torch.Tensor | None = None
    _buf: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    _n: dict[int, int] = field(default_factory=dict)
    _pos: dict[int, int] = field(default_factory=dict)
    _n_seed: dict[int, int] = field(default_factory=dict)
    _child_n: dict[int, int] = field(default_factory=dict)
    _child_pos: dict[int, int] = field(default_factory=dict)
    _child_alloc: dict[int, int] = field(default_factory=dict)
    _next_id: int = 1
    _mean: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _std: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _count: dict[tuple[int, int], int] = field(default_factory=dict)
    #: Greedy FPS of proficient ``q0`` per ``(bucket, chart)``, sigma units.
    _landmarks: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _sigma_den: torch.Tensor | None = field(default=None, repr=False)

    def _child_cap(self) -> int:
        return int(self.cfg.chart_cap) * N_CHART

    def _cap(self) -> int:
        return self._child_cap()

    def occupancy_of(self, bucket: int) -> torch.Tensor:
        b = int(bucket)
        if b not in self._buf or self._n.get(b, 0) == 0:
            return torch.zeros(N_CHART, device=self.device, dtype=torch.long)
        m = self._n[b]
        return occupancy(self._buf[b]["chart"][:m])

    def size(self, bucket: int) -> int:
        return int(self._n.get(int(bucket), 0))

    def n_in_chart(self, bucket: int, chart: int) -> int:
        occ = self.occupancy_of(bucket)
        c = int(chart)
        if c < 0 or c >= N_CHART:
            return 0
        return int(occ[c])

    def _stamp(
        self,
        cols: dict[str, torch.Tensor],
        n: int,
        *,
        from_pool: bool,
        recency_start: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        charts = chart_id(cols["active"], cols["drum"]).to(device=self.device)
        ids = torch.arange(self._next_id, self._next_id + n, device=self.device)
        self._next_id += n
        payload = {k: v for k, v in cols.items() if torch.is_tensor(v) and not str(k).startswith("_")}
        payload["chart"] = charts
        payload["cid"] = ids
        payload["ema_m"] = torch.full((n,), 0.5, device=self.device, dtype=self.dtype)
        payload["ema_n"] = torch.zeros(n, device=self.device, dtype=torch.long)
        payload["n_children"] = torch.zeros(n, device=self.device, dtype=torch.long)
        payload["recency"] = torch.arange(
            recency_start, recency_start + n, device=self.device, dtype=torch.long,
        )
        payload["from_pool"] = torch.full(
            (n,), bool(from_pool), device=self.device, dtype=torch.bool,
        )
        if "traj_k" not in payload:
            payload["traj_k"] = torch.zeros(n, device=self.device, dtype=torch.long)
        return payload, charts

    def _write(self, bucket: int, idx: torch.Tensor, payload: dict[str, torch.Tensor]) -> None:
        d = self._buf[int(bucket)]
        for key, buf in d.items():
            if key in payload:
                buf.index_copy_(0, idx, payload[key].to(device=self.device, dtype=buf.dtype))

    def _grow(self, bucket: int, new_cap: int, like: dict[str, torch.Tensor]) -> None:
        d = self._buf[int(bucket)]
        old_cap = int(d["q0"].shape[0])
        if int(new_cap) > old_cap:
            out: dict[str, torch.Tensor] = {}
            for key, val in d.items():
                bigger = torch.zeros(
                    (int(new_cap),) + tuple(val.shape[1:]),
                    device=val.device, dtype=val.dtype,
                )
                bigger[:old_cap] = val
                out[key] = bigger
            d = out
            self._buf[int(bucket)] = d
        cap = int(d["q0"].shape[0])
        for key, val in like.items():
            if str(key).startswith("_") or key in d or not torch.is_tensor(val):
                continue
            d[key] = torch.zeros(
                (cap,) + tuple(val.shape[1:]), device=self.device, dtype=val.dtype,
            )

    def _alloc(self, bucket: int, like: dict[str, torch.Tensor], cap: int) -> None:
        cap = int(cap)
        d: dict[str, torch.Tensor] = {}
        for key, val in like.items():
            if str(key).startswith("_") or not torch.is_tensor(val):
                continue
            d[key] = torch.zeros(
                (cap,) + tuple(val.shape[1:]), device=self.device, dtype=val.dtype,
            )
        d["chart"] = torch.zeros(cap, device=self.device, dtype=torch.long)
        d["cid"] = torch.zeros(cap, device=self.device, dtype=torch.long)
        d["ema_m"] = torch.zeros(cap, device=self.device, dtype=self.dtype)
        d["ema_n"] = torch.zeros(cap, device=self.device, dtype=torch.long)
        d["n_children"] = torch.zeros(cap, device=self.device, dtype=torch.long)
        d["recency"] = torch.zeros(cap, device=self.device, dtype=torch.long)
        d["from_pool"] = torch.zeros(cap, device=self.device, dtype=torch.bool)
        d["traj_k"] = torch.zeros(cap, device=self.device, dtype=torch.long)
        self._buf[int(bucket)] = d
        self._n[int(bucket)] = 0
        self._pos[int(bucket)] = 0
        self._n_seed[int(bucket)] = 0
        self._child_n[int(bucket)] = 0
        self._child_pos[int(bucket)] = 0
        self._child_alloc[int(bucket)] = 0

    def _ingest_seed(self, bucket: int, cols: dict[str, torch.Tensor]) -> int:
        b = int(bucket)
        n = int(cols["q0"].shape[0])
        n_seed = int(self._n_seed.get(b, 0))
        child_phys = int(self._child_alloc.get(b, 0))
        self._grow(b, n_seed + n + child_phys, cols)
        if child_phys > 0 and n > 0:
            d = self._buf[b]
            src = slice(n_seed, n_seed + child_phys)
            dst = slice(n_seed + n, n_seed + n + child_phys)
            for val in d.values():
                val[dst] = val[src].clone()
        idx = torch.arange(n_seed, n_seed + n, device=self.device)
        payload, charts = self._stamp(
            cols, n, from_pool=True, recency_start=int(self._pos.get(b, 0)),
        )
        self._write(b, idx, payload)
        self._n_seed[b] = n_seed + n
        self._pos[b] = int(self._pos.get(b, 0)) + n
        self._n[b] = self._n_seed[b] + int(self._child_n.get(b, 0))
        self._refresh_whiten(b, charts)
        return n

    def _ensure_child_alloc(self, bucket: int, like: dict[str, torch.Tensor]) -> None:
        """Grow the child ring when ``chart_cap`` rises (resume with a bigger cap).

        Live children already sit in ``[n_seed, n_seed + child_n)``. Reset the
        wrap cursor to ``child_n`` so new rows append instead of ``pos % new_cap``
        landing in empty slots past a previous wrap.
        """
        b = int(bucket)
        want = self._child_cap()
        have = int(self._child_alloc.get(b, 0))
        if have >= want:
            return
        n_seed = int(self._n_seed.get(b, 0))
        self._grow(b, n_seed + want, like)
        self._child_pos[b] = int(self._child_n.get(b, 0))
        self._child_alloc[b] = want

    def _ingest_child(self, bucket: int, cols: dict[str, torch.Tensor]) -> int:
        b = int(bucket)
        n = int(cols["q0"].shape[0])
        n_seed = int(self._n_seed.get(b, 0))
        self._ensure_child_alloc(b, cols)
        child_cap = int(self._child_alloc[b])
        pos = int(self._child_pos.get(b, 0))
        idx = n_seed + (torch.arange(n, device=self.device) + pos) % child_cap
        payload, charts = self._stamp(
            cols, n, from_pool=False, recency_start=int(self._pos.get(b, 0)),
        )
        self._write(b, idx, payload)
        self._child_pos[b] = pos + n
        self._child_n[b] = min(int(self._child_n.get(b, 0)) + n, child_cap)
        self._pos[b] = int(self._pos.get(b, 0)) + n
        self._n[b] = n_seed + self._child_n[b]
        self._refresh_whiten(b, charts)
        return n

    @torch.no_grad()
    def ingest(
        self,
        bucket: int,
        cols: dict[str, torch.Tensor],
        *,
        from_pool: bool = False,
    ) -> int:
        """Append rows. Requires ``q0, qd0, init, hit, active, drum``.

        ``from_pool=True`` (seed / along-traj) grows the buffer so nothing is
        dropped. Children wrap in a region after the seed, never over it.
        """
        b = int(bucket)
        n = int(cols["q0"].shape[0])
        if n == 0:
            return 0
        if b not in self._buf:
            cap = n if from_pool else self._child_cap()
            self._alloc(b, cols, cap)
        if from_pool:
            return self._ingest_seed(b, cols)
        return self._ingest_child(b, cols)

    def _refresh_whiten(self, bucket: int, new_charts: torch.Tensor) -> None:
        b = int(bucket)
        d = self._buf[b]
        m = self._n[b]
        q = torch.cat([d["q0"][:m], d["qd0"][:m]], dim=-1)
        ch = d["chart"][:m]
        for c in torch.unique(new_charts).tolist():
            c = int(c)
            sel = ch == c
            if int(sel.sum()) < 2:
                continue
            x = q[sel]
            key = (b, c)
            self._mean[key] = x.mean(0)
            self._std[key] = x.std(0).clamp_min(1e-3)
            self._count[key] = int(sel.sum())

    def _sigma_denom(self) -> torch.Tensor:
        if self._sigma_den is None or self._sigma_den.device != self.device:
            sc = ball_q_scale(device=self.device, dtype=self.dtype)
            self._sigma_den = (float(self.cfg.sigma_q) * sc).clamp_min(1e-6)
        return self._sigma_den

    def _q_sigma(self, q0: torch.Tensor) -> torch.Tensor:
        den = self._sigma_denom().to(device=q0.device, dtype=q0.dtype).view(1, -1)
        return q0.to(dtype=q0.dtype) / den

    def _min_dist_to_landmarks(
        self, bucket: int, charts: torch.Tensor, q0: torch.Tensor,
    ) -> torch.Tensor:
        """``(N,)`` min sigma-unit distance; inf if the chart has no landmarks."""
        x = self._q_sigma(q0)
        out = torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
        if x.shape[0] == 0:
            return out
        b = int(bucket)
        for c in charts.unique().tolist():
            L = self._landmarks.get((b, int(c)))
            if L is None or L.numel() == 0:
                continue
            sel = charts == int(c)
            d = torch.cdist(x[sel], L.to(device=x.device, dtype=x.dtype))
            out[sel] = d.min(dim=1).values
        return out

    def n_landmarks(self) -> int:
        return int(sum(v.shape[0] for v in self._landmarks.values() if v.numel()))

    @torch.no_grad()
    def _maybe_add_landmark(
        self, bucket: int, chart: int, q0_row: torch.Tensor,
    ) -> None:
        kmax = int(self.cfg.landmark_k)
        tau = float(self.cfg.landmark_tau)
        if kmax <= 0 or tau <= 0.0:
            return
        x = self._q_sigma(q0_row.unsqueeze(0)).squeeze(0)
        key = (int(bucket), int(chart))
        L = self._landmarks.get(key)
        if L is None or L.numel() == 0:
            self._landmarks[key] = x.unsqueeze(0).clone()
            return
        dmin = float(torch.cdist(x.unsqueeze(0), L).min())
        if dmin <= tau:
            return
        if int(L.shape[0]) < kmax:
            self._landmarks[key] = torch.cat([L, x.unsqueeze(0)], 0)
            return
        if int(L.shape[0]) == 1:
            self._landmarks[key] = x.unsqueeze(0).clone()
            return
        dd = torch.cdist(L, L)
        dd.fill_diagonal_(float("inf"))
        drop = int(dd.min(dim=1).values.argmin())
        keep = torch.cat([L[:drop], L[drop + 1 :]], 0)
        self._landmarks[key] = torch.cat([keep, x.unsqueeze(0)], 0)

    @torch.no_grad()
    def rebuild_landmarks(self) -> None:
        """Greedy FPS of current proficient rows. Call after seed / resume."""
        self._landmarks = {}
        nmin = int(self.cfg.ema_nmin)
        prof_m = float(self.cfg.ema_prof)
        for b, d in self._buf.items():
            n = self.size(int(b))
            if n <= 0:
                continue
            ok = (d["ema_n"][:n] >= nmin) & (d["ema_m"][:n] >= prof_m)
            if not bool(ok.any()):
                continue
            loc = ok.nonzero(as_tuple=True)[0]
            order = d["ema_m"][:n].index_select(0, loc).argsort(descending=True)
            loc = loc.index_select(0, order)
            ch = d["chart"].index_select(0, loc)
            q0 = d["q0"].index_select(0, loc)
            for i in range(int(loc.numel())):
                self._maybe_add_landmark(int(b), int(ch[i]), q0[i])

    def _chart_parent_pool(
        self,
        bucket: int,
        chart: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hull-biased parent indices and weights for one chart."""
        b = int(bucket)
        d = self._buf[b]
        m = self._n[b]
        hits = (d["chart"][:m] == int(chart)).nonzero(as_tuple=True)[0]
        empty_w = hits.new_zeros(0, dtype=self.dtype)
        if hits.numel() == 0:
            return hits, empty_w
        hull = float(self.cfg.hull_ema)
        if hull > 0.0:
            lean = (
                (d["ema_m"].index_select(0, hits) >= hull)
                & (d["ema_n"].index_select(0, hits) >= 1)
            )
            if bool(lean.any()):
                hits = hits[lean]
        w = self._parent_weights(b, hits)
        cap = int(self.cfg.propose_candidates)
        if cap > 0 and int(hits.numel()) > cap:
            pick = torch.multinomial(
                w, int(cap), replacement=False, generator=generator,
            )
            hits = hits.index_select(0, pick)
            w = w.index_select(0, pick)
        L = self._landmarks.get((b, int(chart)))
        if L is not None and L.numel() > 0:
            x = self._q_sigma(d["q0"].index_select(0, hits))
            dmin = self._hull_dist(x, L.to(device=x.device, dtype=x.dtype))
            w = (w * dmin.clamp_min(1e-3)).clamp_min(1e-6)
        return hits, w

    def _hull_dist(self, x: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """Min sigma-distance, using 2nd-NN if the row sits on a landmark."""
        dist = torch.cdist(x, landmarks)
        if dist.shape[1] == 1:
            return dist[:, 0]
        tau = float(self.cfg.landmark_tau)
        vals = dist.topk(2, dim=1, largest=False).values
        return torch.where(vals[:, 0] < 0.25 * tau, vals[:, 1], vals[:, 0])

    def _novelty_weights(self, bucket: int, loc: torch.Tensor) -> torch.Tensor:
        """Inverse-density weights: min-dist to landmarks / Voronoi occupancy."""
        d = self._buf[int(bucket)]
        q0 = d["q0"].index_select(0, loc)
        ch = d["chart"].index_select(0, loc)
        x = self._q_sigma(q0)
        dmin = torch.full((loc.numel(),), float("inf"), device=x.device, dtype=x.dtype)
        cell = torch.full((loc.numel(),), -1, device=x.device, dtype=torch.long)
        b = int(bucket)
        for c in ch.unique().tolist():
            L = self._landmarks.get((b, int(c)))
            if L is None or L.numel() == 0:
                continue
            sel = ch == int(c)
            dist = torch.cdist(x[sel], L.to(device=x.device, dtype=x.dtype))
            nn = dist.min(dim=1)
            dmin[sel] = self._hull_dist(x[sel], L.to(device=x.device, dtype=x.dtype))
            cell[sel] = nn.indices
        w = torch.ones(loc.numel(), device=x.device, dtype=x.dtype)
        finite = torch.isfinite(dmin)
        w[finite] = dmin[finite].clamp_min(1e-3)
        if bool(finite.any()):
            packed = ch.clamp_min(0) * 1024 + cell.clamp_min(0)
            # counts among the candidate pool only
            cnt = torch.zeros(
                int(packed[finite].max().item()) + 1,
                device=x.device, dtype=x.dtype,
            )
            ones = torch.ones(int(finite.sum()), device=x.device, dtype=x.dtype)
            cnt.scatter_add_(0, packed[finite], ones)
            w[finite] = w[finite] / (1.0 + cnt[packed[finite]])
        return w.clamp_min(1e-6)

    def _parent_weights(self, bucket: int, loc: torch.Tensor) -> torch.Tensor:
        d = self._buf[int(bucket)]
        nch = d["n_children"].index_select(0, loc).to(self.dtype)
        rec = d["recency"].index_select(0, loc).to(self.dtype)
        pos = float(max(self._pos[int(bucket)] - 1, 0))
        scale = float(max(self.size(int(bucket)), self._child_cap(), 1))
        age = (pos - rec).clamp_min(0.0)
        # Inverse-age, not exp: float32 exp(-2*age/n) underflows to 0 once
        # the ingest counter is large, and multinomial then dies on a
        # seed-only chart.
        recency = 1.0 / (1.0 + 2.0 * age / scale)
        # Untested rows start at 0.5; scored misses shrink, hits grow.
        # Hold that prior until ``ema_nmin`` looks so 2-3 hits, not one noisy
        # sample, are what lift a parent over the default.
        score = d["ema_m"].index_select(0, loc).clamp(1e-3, 1.0)
        nmin = int(self.cfg.ema_nmin)
        if nmin > 0:
            nobs = d["ema_n"].index_select(0, loc).to(self.dtype)
            score = torch.where(
                nobs >= float(nmin), score, score.new_full(score.shape, 0.5),
            )
        return (recency * (1.0 / (1.0 + nch)) * score).clamp_min(1e-6)

    @torch.no_grad()
    def sample_in_chart(
        self,
        bucket: int,
        chart: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For each requested chart, one parent index into the ring (or -1)."""
        b = int(bucket)
        n = int(chart.numel())
        loc = torch.full((n,), -1, device=self.device, dtype=torch.long)
        if b not in self._buf or self._n.get(b, 0) == 0:
            return loc, torch.zeros(n, device=self.device, dtype=torch.bool)
        ok = torch.zeros(n, device=self.device, dtype=torch.bool)
        cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i in range(n):
            c = int(chart[i])
            if c not in cache:
                cache[c] = self._chart_parent_pool(b, c, generator)
            hits, w = cache[c]
            if hits.numel() == 0:
                continue
            pick = torch.multinomial(w, 1, generator=generator)
            loc[i] = hits[pick]
            ok[i] = True
        return loc, ok

    def gather(self, bucket: int, loc: torch.Tensor) -> dict[str, torch.Tensor]:
        d = self._buf[int(bucket)]
        return {k: v.index_select(0, loc) for k, v in d.items()}

    def scored_loc(self, bucket: int) -> torch.Tensor:
        """Live indices that carry strike/context geometry (trainable / EMA)."""
        b = int(bucket)
        n = self.size(b)
        empty = torch.zeros(0, device=self.device, dtype=torch.long)
        if n <= 0 or b not in self._buf:
            return empty
        d = self._buf[b]
        if "p_star" not in d or "drum_pos" not in d:
            return empty
        ok = torch.isfinite(d["p_star"][:n]).reshape(n, -1).all(dim=1)
        ok = ok & torch.isfinite(d["drum_pos"][:n]).reshape(n, -1).all(dim=1)
        return ok.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def sample_idx(
        self,
        bucket: int,
        k: int,
        *,
        generator: torch.Generator | None = None,
        among: torch.Tensor | None = None,
        inv_frac: float = 0.0,
    ) -> torch.Tensor:
        """``k`` indices into the live ring, without replacement.

        ``inv_frac`` of the draw is inverse-density (min-dist to landmarks /
        Voronoi count in a uniform candidate pool). The rest is uniform.
        """
        if among is None:
            n = self.size(int(bucket))
            choices = torch.arange(n, device=self.device)
        else:
            choices = among.to(device=self.device, dtype=torch.long)
            n = int(choices.numel())
        k = min(int(k), n)
        if k <= 0:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        if k == n:
            return choices
        k_inv = int(round(float(k) * float(inv_frac))) if float(inv_frac) > 0.0 else 0
        k_inv = min(max(k_inv, 0), k)
        k_uni = int(k) - k_inv
        perm = torch.randperm(n, device=self.device, generator=generator)
        parts: list[torch.Tensor] = []
        used = torch.zeros(n, device=self.device, dtype=torch.bool)
        if k_uni > 0:
            take = perm[:k_uni]
            parts.append(choices.index_select(0, take))
            used[take] = True
        if k_inv > 0:
            rest = (~used).nonzero(as_tuple=True)[0]
            if rest.numel() <= k_inv:
                parts.append(choices.index_select(0, rest[:k_inv]))
            else:
                pool_n = min(
                    int(rest.numel()),
                    max(k_inv * 4, int(self.cfg.train_inv_pool)),
                )
                rp = rest.index_select(
                    0,
                    torch.randperm(int(rest.numel()), device=self.device, generator=generator)[:pool_n],
                )
                cand = choices.index_select(0, rp)
                w = self._novelty_weights(int(bucket), cand)
                pick = torch.multinomial(
                    w, int(k_inv), replacement=False, generator=generator,
                )
                parts.append(cand.index_select(0, pick))
        if not parts:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        return torch.cat(parts, 0)

    @torch.no_grad()
    def propose(
        self,
        bucket: int,
        k: int,
        *,
        generator: torch.Generator | None = None,
        expand: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        """``k`` children from random occupied-chart parents. None if empty."""
        b = int(bucket)
        if self.size(b) == 0 or k <= 0:
            return None
        d = self._buf[b]
        m = self._n[b]
        occ = occupancy(d["chart"][:m])
        nonempty = (occ > 0).nonzero(as_tuple=True)[0]
        if nonempty.numel() == 0:
            return None
        pick_c = nonempty[torch.randint(
            0, nonempty.numel(), (int(k),), device=self.device, generator=generator,
        )]
        loc, ok = self.sample_in_chart(b, pick_c, generator=generator)
        loc = loc[ok]
        if loc.numel() == 0:
            return None
        parent = self.gather(b, loc)
        H = int(parent["init"].shape[1])
        term = parent_terminal(parent["init"], parent["hit"], parent["active"], H)
        q0, qd0 = parent["q0"], parent["qd0"]
        hit = parent["hit"]
        keep = torch.ones(q0.shape[0], device=self.device, dtype=torch.bool)
        if expand:
            q_lo = self.q_lo if self.q_lo is not None else q0.min(0).values
            q_hi = self.q_hi if self.q_hi is not None else q0.max(0).values
            q0, qd0 = apply_q_ball(
                q0, qd0,
                sigma_q=float(self.cfg.sigma_q),
                sigma_qd=float(self.cfg.sigma_qd),
                q_lo=q_lo, q_hi=q_hi, generator=generator,
            )
            hit, keep = jitter_hit(
                parent["hit"], parent["active"],
                dt=float(self.cfg.dt), bucket=H,
                sigma_t=float(self.cfg.sigma_tth),
                generator=generator, buckets=self.cfg.buckets,
            )
            tau = float(self.cfg.landmark_tau)
            if tau > 0.0:
                dmin = self._min_dist_to_landmarks(b, parent["chart"], q0)
                keep = keep & (~torch.isfinite(dmin) | (dmin >= tau))
        sel = keep.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return None
        child = {key: val.index_select(0, sel) for key, val in parent.items()}
        child["q0"] = q0.index_select(0, sel)
        child["qd0"] = qd0.index_select(0, sel)
        child["hit"] = hit.index_select(0, sel)
        child["init"] = quintic_init(
            child["q0"], term.index_select(0, sel),
            child["hit"], child["active"], H,
        )
        child = rebuild_clocks(child, dt=float(self.cfg.dt))
        d["n_children"].index_add_(
            0, loc.index_select(0, sel),
            torch.ones(sel.numel(), device=self.device, dtype=torch.long),
        )
        return child

    @torch.no_grad()
    def ema_update(self, bucket: int, cids: torch.Tensor, y: torch.Tensor) -> None:
        b = int(bucket)
        if b not in self._buf or cids.numel() == 0:
            return
        d = self._buf[b]
        m = self._n[b]
        table = d["cid"][:m]
        alpha = float(self.cfg.ema_alpha)
        for i in range(int(cids.numel())):
            cid = int(cids[i])
            if cid <= 0:
                continue
            hit = (table == cid).nonzero(as_tuple=True)[0]
            if hit.numel() == 0:
                continue
            j = int(hit[0])
            yi = float(y[i])
            d["ema_m"][j] = (1.0 - alpha) * d["ema_m"][j] + alpha * yi
            d["ema_n"][j] = d["ema_n"][j] + 1
            if (
                int(d["ema_n"][j]) >= int(self.cfg.ema_nmin)
                and float(d["ema_m"][j]) >= float(self.cfg.ema_prof)
            ):
                self._maybe_add_landmark(b, int(d["chart"][j]), d["q0"][j])

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg,
            "buf": {b: {k: v.cpu() for k, v in d.items()} for b, d in self._buf.items()},
            "n": dict(self._n),
            "pos": dict(self._pos),
            "n_seed": dict(self._n_seed),
            "child_n": dict(self._child_n),
            "child_pos": dict(self._child_pos),
            "child_alloc": dict(self._child_alloc),
            "next_id": self._next_id,
        }

    def load_state_dict(self, st: dict) -> None:
        self._n = {int(k): int(v) for k, v in st["n"].items()}
        self._pos = {int(k): int(v) for k, v in st["pos"].items()}
        self._n_seed = {int(k): int(v) for k, v in st.get("n_seed", st["n"]).items()}
        self._child_n = {int(k): int(v) for k, v in st.get("child_n", {}).items()}
        self._child_pos = {int(k): int(v) for k, v in st.get("child_pos", {}).items()}
        self._child_alloc = {int(k): int(v) for k, v in st.get("child_alloc", {}).items()}
        self._next_id = int(st["next_id"])
        self._buf = {}
        for b, d in st["buf"].items():
            self._buf[int(b)] = {
                k: v.to(self.device) for k, v in d.items()
            }


def ingest_pool(cloud: FrontierCloud, pool) -> dict[int, int]:
    """Copy every static-pool row, plus along-traj slices. Returns ``{bucket: n}``.

    ``k=0`` originals always enter their source bucket. Knots ``stride, 2*stride,
    ...`` that keep every active ``hit' >= 1`` are emitted into
    ``bucket_for(max hit')`` (allowed to be shorter). Seed ingest grows so no
    original start is dropped.
    """
    added: dict[int, int] = {}
    stride = max(int(cloud.cfg.traj_stride), 1)
    buckets = tuple(int(b) for b in cloud.cfg.buckets)
    dt = float(cloud.cfg.dt)

    def _add(b: int, n: int) -> None:
        added[int(b)] = added.get(int(b), 0) + int(n)

    for b in pool.buckets:
        d = pool.data[b]
        if "drum" not in d or "q0" not in d:
            continue
        cols = {
            k: v for k, v in d.items()
            if torch.is_tensor(v) and not str(k).startswith("_")
        }
        _add(int(b), cloud.ingest(int(b), cols, from_pool=True))
        h = int(cols["init"].shape[1])
        for k in range(stride, h + 1, stride):
            for b2, sub in slice_at_knot(
                cols, k, dt=dt, buckets=buckets,
            ).items():
                _add(int(b2), cloud.ingest(int(b2), sub, from_pool=True))
    if cloud.q_lo is None:
        cloud.q_lo = pool.q_lo
        cloud.q_hi = pool.q_hi
    return added


def steal_q0(
    cols: dict[str, torch.Tensor],
    cloud: FrontierCloud,
    bucket: int,
    *,
    frac: float,
    kind: torch.Tensor,
    idle_kind: int,
    far_kind: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace idle/far ``q0``/``qd0``/``init`` from same-chart parents. Returns cid."""
    n = int(cols["q0"].shape[0])
    cid = torch.zeros(n, device=cols["q0"].device, dtype=torch.long)
    if frac <= 0.0 or cloud.size(bucket) == 0:
        return cid
    is_if = (kind == int(idle_kind)) | (kind == int(far_kind))
    take = is_if & (
        torch.rand(n, device=cols["q0"].device, generator=generator) < float(frac)
    )
    idx = take.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return cid
    charts = chart_id(cols["active"].index_select(0, idx), cols["drum"].index_select(0, idx))
    loc, ok = cloud.sample_in_chart(bucket, charts, generator=generator)
    if not bool(ok.any()):
        return cid
    src = idx[ok]
    loc = loc[ok]
    parent = cloud.gather(bucket, loc)
    H = int(cols["init"].shape[1])
    term = parent_terminal(
        cols["init"].index_select(0, src),
        cols["hit"].index_select(0, src),
        cols["active"].index_select(0, src),
        H,
    )
    cols["q0"].index_copy_(0, src, parent["q0"].to(cols["q0"].dtype))
    cols["qd0"].index_copy_(0, src, parent["qd0"].to(cols["qd0"].dtype))
    new_init = quintic_init(
        cols["q0"].index_select(0, src), term,
        cols["hit"].index_select(0, src),
        cols["active"].index_select(0, src),
        H,
    )
    cols["init"].index_copy_(0, src, new_init.to(cols["init"].dtype))
    if "yaw_now" in cols:
        cols["yaw_now"].index_copy_(
            0, src, cols["q0"].index_select(0, src)[:, YAW_I].to(cols["yaw_now"].dtype),
        )
    cid[src] = parent["cid"]
    return cid


def patch_spawn_q0(
    cols: dict[str, torch.Tensor],
    cloud: FrontierCloud,
    *,
    frac: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Spawn path: overwrite q0/qd0 from same-chart parents; clocks stay drawn.

    Returns a boolean mask of patched rows. Caller ramps ``init`` like carry.
    """
    n = int(cols["q0"].shape[0])
    take = torch.zeros(n, device=cols["q0"].device, dtype=torch.bool)
    if frac <= 0.0:
        return take
    H = int(cols["init"].shape[1])
    if cloud.size(H) == 0:
        return take
    take = torch.rand(n, device=cols["q0"].device, generator=generator) < float(frac)
    idx = take.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return take
    charts = chart_id(cols["active"].index_select(0, idx), cols["drum"].index_select(0, idx))
    loc, ok = cloud.sample_in_chart(H, charts, generator=generator)
    take[idx] = ok
    src = idx[ok]
    if src.numel() == 0:
        return take
    parent = cloud.gather(H, loc[ok])
    cols["q0"].index_copy_(0, src, parent["q0"].to(cols["q0"].dtype))
    cols["qd0"].index_copy_(0, src, parent["qd0"].to(cols["qd0"].dtype))
    return take
