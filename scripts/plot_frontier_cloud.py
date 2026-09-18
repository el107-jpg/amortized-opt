#!/usr/bin/env python
"""Score and plot frontier collocation start clouds.

Stick-tip positions of both arms at q0, for one drum pair and a TTH window.
Seed (pool k=0) vs expanded (along-traj slices + ball children). Feasible
(ok_hit and ok_collision) vs still-failing.

    python scripts/plot_frontier_cloud.py \
        --ckpt runs/rhythm_bellman_nf1_waist_gain/last.pt \
        --pool runs/p2pool_rhythm_bellman_nf1_prsp_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \
        --left-drum snare --right-drum cymbal_right \
        --tth-lo 0.30 --tth-hi 0.50 \
        --expand-rounds 4 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.constants import CONTROL_DT  # noqa: E402
from amortized_opt.fk import ChainFK, stick_tip_offset  # noqa: E402
from amortized_opt.frontier import (  # noqa: E402
    FrontierCloud,
    FrontierConfig,
    chart_id,
    ingest_pool,
)
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool, Phase2ScenarioConfig  # noqa: E402
from amortized_opt.rhythm import DrumKit  # noqa: E402
from amortized_opt.train_cost import (  # noqa: E402
    build_costs,
    encode,
    phase2_cost_config,
)
from amortized_opt.warmstart_data import CONTEXT_FIELDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Matching eval_harness.AcceptanceThresholds: 2 cm disk slack, 5 mm SDF.
HIT_MARGIN_M = 0.02
COLLISION_MARGIN_M = 0.005


@dataclass
class CloudSlice:
    bucket: int
    q0: np.ndarray
    tip_l: np.ndarray
    tip_r: np.ndarray
    seed: np.ndarray
    ok: np.ndarray
    scored: np.ndarray
    miss_mm: np.ndarray
    sdf_mm: np.ndarray
    tth: np.ndarray
    drum: np.ndarray
    origin_l: np.ndarray | None = None
    origin_r: np.ndarray | None = None


def _load_npz_buckets(cache: Path) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    z = np.load(cache, allow_pickle=True)
    want = json.dumps(list(CONTEXT_FIELDS))
    got = str(z["_schema"]) if "_schema" in z.files else None
    if got != want:
        raise RuntimeError(f"pool schema mismatch\n  cache: {got}\n  code: {want}")
    buckets: dict[int, dict[str, np.ndarray]] = {}
    for k in z.files:
        if k.startswith("_"):
            continue
        b, name = k.split("/", 1)
        buckets.setdefault(int(b), {})[name] = z[k]
    stats = json.loads(str(z["_stats"])) if "_stats" in z.files else {}
    return buckets, stats


def _scenario_from_ckpt(ck: dict, buckets: tuple[int, ...]) -> Phase2ScenarioConfig:
    raw = dict(ck.get("scenario_cfg") or {})
    allow = {f.name for f in Phase2ScenarioConfig.__dataclass_fields__.values()}
    kw = {k: v for k, v in raw.items() if k in allow and k not in (
        "kit_randomization", "waist_cfg", "stick_rand", "joint_names",
        "far_lead_s", "far_horizons", "far_speeds", "buckets",
    )}
    kw["buckets"] = tuple(int(b) for b in (raw.get("buckets") or buckets))
    if "joint_names" in raw and raw["joint_names"] is not None:
        kw["joint_names"] = tuple(raw["joint_names"])
    return Phase2ScenarioConfig(**kw)


def load_pool(path: Path, scen: Phase2ScenarioConfig, device) -> Phase2Pool:
    buckets, stats = _load_npz_buckets(path)
    return Phase2Pool(buckets, cfg=scen, noise=Phase2Noise(), device=device, stats=stats)


def load_model(ckpt: Path, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck, int(ck.get("step") or 0)


def seed_cloud(pool: Phase2Pool, device, dtype=torch.float32) -> FrontierCloud:
    cfg = FrontierConfig(
        enabled=True,
        buckets=tuple(pool.buckets),
        dt=float(pool.cfg.dt),
    )
    cloud = FrontierCloud(
        cfg=cfg, device=torch.device(device), dtype=dtype,
        q_lo=pool.q_lo, q_hi=pool.q_hi,
    )
    added = ingest_pool(cloud, pool)
    print(
        "[cloud] ingest " + " ".join(f"b{b}={n}" for b, n in sorted(added.items())),
        flush=True,
    )
    return cloud


def expand_cloud(cloud: FrontierCloud, *, rounds: int, proposals: int, generator=None) -> int:
    n = 0
    for _ in range(int(rounds)):
        for b in list(cloud._buf):
            child = cloud.propose(int(b), int(proposals), generator=generator)
            if child is None:
                continue
            n += int(cloud.ingest(int(b), child))
    print(f"[cloud] proposed children +{n}", flush=True)
    return n


def _want_chart(left_id: int, right_id: int) -> int:
    return (int(left_id) + 1) * 8 + (int(right_id) + 1)


def filter_loc(
    cloud: FrontierCloud,
    bucket: int,
    *,
    chart: int,
    tth_lo: float,
    tth_hi: float,
    dt: float,
    tth_on: str,
) -> torch.Tensor:
    n = cloud.size(int(bucket))
    if n == 0:
        return torch.zeros(0, device=cloud.device, dtype=torch.long)
    d = cloud._buf[int(bucket)]
    ch = d["chart"][:n]
    act = d["active"][:n].to(torch.bool)
    hit = d["hit"][:n].to(torch.float32)
    tth = hit.clamp_min(0.0) * float(dt)
    keep = (ch == int(chart)) & act.any(dim=1)
    if tth_on == "both":
        for hi in (0, 1):
            live = act[:, hi]
            keep = keep & (~live | ((tth[:, hi] >= tth_lo) & (tth[:, hi] <= tth_hi)))
        keep = keep & act.all(dim=1)
    else:
        tmax = torch.where(act, tth, torch.zeros_like(tth)).amax(dim=1)
        keep = keep & (tmax >= tth_lo) & (tmax <= tth_hi)
    return keep.nonzero(as_tuple=True)[0]


def stick_tips(fk: ChainFK, q0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = []
    for hand in ("left", "right"):
        R, p = fk.frame(q0, f"{hand}_wrist")
        loc = q0.new_tensor(stick_tip_offset(hand)).view(1, 3)
        out.append(ChainFK.points_world(R, p, loc).squeeze(1))
    return out[0], out[1]


def _is_seed(d: dict[str, torch.Tensor], loc: torch.Tensor) -> torch.Tensor:
    fp = d["from_pool"].index_select(0, loc)
    k = d["traj_k"].index_select(0, loc)
    return fp.to(torch.bool) & (k == 0)


def _mini_pool(
    cloud: FrontierCloud, bucket: int, loc: torch.Tensor, cfg, device,
) -> Phase2Pool:
    g = cloud.gather(int(bucket), loc)
    arrays = {}
    for k, v in g.items():
        if not torch.is_tensor(v):
            continue
        arrays[k] = v.detach().cpu().numpy()
    return Phase2Pool(
        {int(bucket): arrays}, cfg=cfg, noise=Phase2Noise(),
        device=device,
    )


@torch.no_grad()
def score_slice(
    model,
    cost,
    mini: Phase2Pool,
    bucket: int,
    *,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ok, miss_m, min_sdf_m for every row of ``mini``."""
    n = mini.size(int(bucket))
    ok = np.zeros(n, dtype=bool)
    miss = np.full(n, np.nan, dtype=np.float64)
    sdf = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return ok, miss, sdf
    if "goal_pos" not in mini.data[int(bucket)]:
        return ok, miss, sdf
    H = int(bucket)
    model.eval()
    for s0 in range(0, n, int(chunk)):
        sl = slice(s0, min(s0 + int(chunk), n))
        idx = torch.arange(sl.start, sl.stop, device=mini.device)
        batch = mini.batch(H, idx, noise=False)
        enc = encode(model, batch["context"])
        out = model(batch["init"], batch["state"], enc, batch["masks"])
        ct = cost.forward(
            batch["q0"], out.traj, batch["v0"], batch["strike"],
        )
        dreg = ct.diagnostics["terminal_region_dist"].detach().cpu().numpy()
        dsdf = ct.diagnostics["collision_min_sdf"].detach().cpu().numpy()
        hit_ok = dreg <= HIT_MARGIN_M
        coll_ok = dsdf >= -COLLISION_MARGIN_M
        ok[sl] = hit_ok & coll_ok
        miss[sl] = dreg
        sdf[sl] = dsdf
    return ok, miss, sdf


def collect_slice(
    cloud: FrontierCloud,
    model,
    costs: dict,
    pool: Phase2Pool,
    bucket: int,
    loc: torch.Tensor,
    *,
    chunk: int,
    score: bool,
) -> CloudSlice | None:
    if loc.numel() == 0:
        return None
    d = cloud._buf[int(bucket)]
    q0 = d["q0"].index_select(0, loc)
    fk = costs[int(bucket)].fk
    tip_l, tip_r = stick_tips(fk, q0.to(device=fk.device, dtype=fk.dtype))
    seed = _is_seed(d, loc).detach().cpu().numpy()
    hit = d["hit"].index_select(0, loc).to(torch.float32)
    act = d["active"].index_select(0, loc).to(torch.bool)
    tth = torch.where(act, hit.clamp_min(0.0), torch.zeros_like(hit)).amax(dim=1)
    tth = (tth * float(pool.cfg.dt)).detach().cpu().numpy()
    n = int(loc.numel())
    scored = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    miss = np.full(n, np.nan)
    sdf = np.full(n, np.nan)
    if score:
        score_dev = next(model.parameters()).device
        mini = _mini_pool(cloud, bucket, loc, pool.cfg, score_dev)
        ok, miss, sdf = score_slice(
            model, costs[int(bucket)], mini, int(bucket), chunk=chunk,
        )
        scored[:] = np.isfinite(miss)
    return CloudSlice(
        bucket=int(bucket),
        q0=q0.detach().cpu().numpy(),
        tip_l=tip_l.detach().cpu().numpy(),
        tip_r=tip_r.detach().cpu().numpy(),
        seed=seed,
        ok=ok,
        scored=scored,
        miss_mm=np.asarray(miss, float) * 1e3,
        sdf_mm=np.asarray(sdf, float) * 1e3,
        tth=tth,
        drum=d["drum"].index_select(0, loc).detach().cpu().numpy(),
    )


def merge_slices(parts: list[CloudSlice]) -> CloudSlice | None:
    if not parts:
        return None
    cat = lambda attr: np.concatenate([getattr(p, attr) for p in parts], 0)  # noqa: E731
    return CloudSlice(
        bucket=-1,
        q0=cat("q0"), tip_l=cat("tip_l"), tip_r=cat("tip_r"),
        seed=cat("seed"), ok=cat("ok"), scored=cat("scored"),
        miss_mm=cat("miss_mm"), sdf_mm=cat("sdf_mm"), tth=cat("tth"),
        drum=cat("drum"),
    )


def summarize(sl: CloudSlice) -> dict:
    def pack(mask: np.ndarray) -> dict:
        sel = mask & sl.scored
        n = int(sel.sum())
        n_ok = int((sl.ok & sel).sum())
        miss = sl.miss_mm[sel]
        sdf = sl.sdf_mm[sel]
        return {
            "n": int(mask.sum()),
            "n_scored": n,
            "n_ok": n_ok,
            "ok_frac": (n_ok / n) if n else None,
            "miss_mm_p50": float(np.median(miss)) if n else None,
            "sdf_mm_p50": float(np.median(sdf)) if n else None,
        }

    seed = sl.seed.astype(bool)
    return {
        "n": int(sl.q0.shape[0]),
        "seed": pack(seed),
        "expanded": pack(~seed),
        "all": pack(np.ones(sl.q0.shape[0], dtype=bool)),
    }


def _disc(center, normal, radius, n=40):
    nrm = np.asarray(normal, float)
    nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
    a = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(nrm, a)
    u = u / max(np.linalg.norm(u), 1e-9)
    v = np.cross(nrm, u)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return center + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)


def _scatter(ax, pts, *, ok, seed, color, is3d):
    """Feasible filled, failing as x; seed larger than expanded."""
    pts = np.asarray(pts)
    if pts.size == 0:
        return
    xyz = (pts[:, 0], pts[:, 1], pts[:, 2]) if is3d else (pts[:, 0], pts[:, 1])
    seed = seed.astype(bool)
    ok = ok.astype(bool)
    for m, marker, s, a, fill in (
        (seed & ok, "o", 28, 0.85, True),
        (seed & ~ok, "x", 22, 0.75, False),
        ((~seed) & ok, "^", 18, 0.55, True),
        ((~seed) & ~ok, "x", 14, 0.40, False),
    ):
        if not bool(m.any()):
            continue
        p = tuple(c[m] for c in xyz)
        if fill:
            ax.scatter(*p, c=color, marker=marker, s=s, alpha=a, linewidths=0.4,
                       edgecolors="k")
        else:
            ax.scatter(*p, c=color, marker=marker, s=s, alpha=a, linewidths=0.9)


def _scatter_origin(ax, pts, color, is3d):
    pts = np.asarray(pts)
    if pts.size == 0:
        return
    xyz = (pts[:, 0], pts[:, 1], pts[:, 2]) if is3d else (pts[:, 0], pts[:, 1])
    ax.scatter(*xyz, c=color, marker=".", s=8, alpha=0.18, linewidths=0)


def draw_figure(sl: CloudSlice, kit: DrumKit, left: str, right: str, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12.5, 6.2), dpi=130)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    pos = kit.positions_in("pelvis")
    for name, lw in ((left, 2.4), (right, 2.4)):
        i = kit.index(name)
        ring = _disc(pos[i], kit.axes[i], kit.radii[i])
        ax3.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="0.35", lw=lw)
        ax2.plot(ring[:, 0], ring[:, 1], color="0.35", lw=lw)
        ax2.scatter(pos[i, 0], pos[i, 1], c="0.35", s=12, zorder=0)
        ax3.text(*pos[i], name, fontsize=7, color="0.25")
        ax2.annotate(name, (pos[i, 0], pos[i, 1]), fontsize=8, color="0.25")

    seed = sl.seed.astype(bool)
    ok = sl.ok.astype(bool)
    if sl.origin_l is not None:
        _scatter_origin(ax3, sl.origin_l, "#1f77b4", True)
        _scatter_origin(ax2, sl.origin_l, "#1f77b4", False)
    if sl.origin_r is not None:
        _scatter_origin(ax3, sl.origin_r, "#d62728", True)
        _scatter_origin(ax2, sl.origin_r, "#d62728", False)
    _scatter(ax3, sl.tip_l, ok=ok, seed=seed, color="#1f77b4", is3d=True)
    _scatter(ax3, sl.tip_r, ok=ok, seed=seed, color="#d62728", is3d=True)
    _scatter(ax2, sl.tip_l, ok=ok, seed=seed, color="#1f77b4", is3d=False)
    _scatter(ax2, sl.tip_r, ok=ok, seed=seed, color="#d62728", is3d=False)

    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.view_init(elev=22, azim=-50)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal")
    ax2.set_title("top-down (pelvis xy)")
    ax3.set_title("stick tips at q0")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=".", color="#1f77b4", markersize=8,
               linestyle="None", alpha=0.4, label="left origin (k=0, any TTH)"),
        Line2D([0], [0], marker=".", color="#d62728", markersize=8,
               linestyle="None", alpha=0.4, label="right origin (k=0, any TTH)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="k", markersize=7, label="left feasible (window)"),
        Line2D([0], [0], marker="x", color="#1f77b4", markersize=8,
               linestyle="None", label="left fail (window)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728",
               markeredgecolor="k", markersize=7, label="right feasible (window)"),
        Line2D([0], [0], marker="x", color="#d62728", markersize=8,
               linestyle="None", label="right fail (window)"),
    ]
    ax2.legend(handles=handles, loc="best", fontsize=7, framealpha=0.9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[frontier-plot] wrote {path}", flush=True)


def report_cloud(
    *,
    cloud: FrontierCloud,
    model,
    costs: dict,
    pool: Phase2Pool,
    out_dir: Path,
    step: int,
    left_drum: str = "snare",
    right_drum: str = "cymbal_right",
    tth_lo: float = 0.30,
    tth_hi: float = 0.50,
    tth_on: str = "both",
    chunk: int = 64,
    score: bool = True,
    max_points: int = 4096,
    tag: str = "",
) -> dict:
    kit = DrumKit.from_layout()
    lid, rid = kit.index(left_drum), kit.index(right_drum)
    chart = _want_chart(lid, rid)
    dt = float(pool.cfg.dt)
    parts: list[CloudSlice] = []
    origins_l: list[np.ndarray] = []
    origins_r: list[np.ndarray] = []
    occ = {}
    rng = np.random.default_rng(0)
    fk0 = costs[int(pool.buckets[0])].fk
    for b in pool.buckets:
        n_chart = cloud.n_in_chart(int(b), chart)
        occ[str(int(b))] = n_chart
        loc = filter_loc(
            cloud, int(b), chart=chart, tth_lo=tth_lo, tth_hi=tth_hi,
            dt=dt, tth_on=tth_on,
        )
        n = cloud.size(int(b))
        d = cloud._buf[int(b)]
        origin = (
            (d["chart"][:n] == int(chart))
            & d["from_pool"][:n].to(torch.bool)
            & (d["traj_k"][:n] == 0)
        ).nonzero(as_tuple=True)[0]
        print(
            f"[frontier-plot] bucket {int(b)} chart={chart} "
            f"in_chart={n_chart} origin_k0={int(origin.numel())} "
            f"tth-window={int(loc.numel())}",
            flush=True,
        )
        if origin.numel() > 0:
            if origin.numel() > 2000:
                pick = rng.choice(int(origin.numel()), 2000, replace=False)
                origin = origin[torch.as_tensor(pick, device=origin.device, dtype=torch.long)]
            q0o = d["q0"].index_select(0, origin).to(device=fk0.device, dtype=fk0.dtype)
            ol, or_ = stick_tips(fk0, q0o)
            origins_l.append(ol.detach().cpu().numpy())
            origins_r.append(or_.detach().cpu().numpy())
        if loc.numel() > int(max_points):
            pick = rng.choice(int(loc.numel()), int(max_points), replace=False)
            loc = loc[torch.as_tensor(pick, device=loc.device, dtype=torch.long)]
        part = collect_slice(
            cloud, model, costs, pool, int(b), loc, chunk=chunk, score=score,
        )
        if part is not None:
            parts.append(part)
    sl = merge_slices(parts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frontier_cloud_step{step}_{left_drum}_{right_drum}"
    meta = {
        "step": int(step),
        "left_drum": left_drum,
        "right_drum": right_drum,
        "chart": int(chart),
        "tth_lo": float(tth_lo),
        "tth_hi": float(tth_hi),
        "tth_on": tth_on,
        "occupancy": occ,
        "n_origin": int(sum(a.shape[0] for a in origins_l)),
        "hit_margin_m": HIT_MARGIN_M,
        "collision_margin_m": COLLISION_MARGIN_M,
        "tag": tag,
        "feasible": "ok_hit (region_dist<=20mm) and ok_collision (min_sdf>=-5mm)",
    }
    if sl is None:
        meta["empty"] = True
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        print("[frontier-plot] no rows in window", flush=True)
        return meta
    if origins_l:
        sl.origin_l = np.concatenate(origins_l, 0)
        sl.origin_r = np.concatenate(origins_r, 0)
    stats = summarize(sl)
    meta["stats"] = stats
    title = (
        f"collocation tips  L={left_drum} R={right_drum}  "
        f"TTH {tth_lo:.2f}-{tth_hi:.2f}s  step {step}"
        + (f"  {tag}" if tag else "")
        + f"  origin k=0 n={meta['n_origin']}"
    )
    png = out_dir / f"{stem}.png"
    draw_figure(sl, kit, left_drum, right_drum, title, png)
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        tip_l=sl.tip_l, tip_r=sl.tip_r, seed=sl.seed, ok=sl.ok,
        scored=sl.scored, miss_mm=sl.miss_mm, sdf_mm=sl.sdf_mm, tth=sl.tth,
        origin_l=sl.origin_l if sl.origin_l is not None else np.zeros((0, 3)),
        origin_r=sl.origin_r if sl.origin_r is not None else np.zeros((0, 3)),
    )
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    s, e = stats["seed"], stats["expanded"]
    def _fmt(p):
        if p["ok_frac"] is None:
            return f"n={p['n']} (unscored)"
        return (
            f"n={p['n']} ok {p['n_ok']}/{p['n_scored']} "
            f"({100.0 * p['ok_frac']:.1f}%) miss_p50={p['miss_mm_p50']:.1f}mm "
            f"sdf_p50={p['sdf_mm_p50']:.1f}mm"
        )
    print(f"[frontier-plot] origin   n={meta['n_origin']} (k=0, any TTH)", flush=True)
    print(f"[frontier-plot] seed     {_fmt(s)}", flush=True)
    print(f"[frontier-plot] expanded {_fmt(e)}", flush=True)
    return meta


def plot_from_live(
    *,
    cloud: FrontierCloud,
    model,
    costs: dict,
    pool: Phase2Pool,
    out_dir,
    step: int,
    left_drum: str = "snare",
    right_drum: str = "cymbal_right",
    tth_lo: float = 0.30,
    tth_hi: float = 0.50,
    chunk: int = 64,
) -> dict:
    """Called from the trainer at ~10k / ~20k extra steps."""
    out = {}
    for left, right in ((left_drum, right_drum), ("hihat", "snare")):
        out[f"{left}+{right}"] = report_cloud(
            cloud=cloud, model=model, costs=costs, pool=pool,
            out_dir=Path(out_dir), step=int(step),
            left_drum=left, right_drum=right,
            tth_lo=tth_lo, tth_hi=tth_hi, chunk=chunk, score=True,
            tag="live-train",
        )
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/rhythm_bellman_nf1_waist_gain/last.pt")
    p.add_argument(
        "--pool",
        default="runs/p2pool_rhythm_bellman_nf1_prsp_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz",
    )
    p.add_argument("--frontier-pt", default="",
                   help="load an existing frontier.pt instead of ingesting the pool")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--left-drum", default="snare")
    p.add_argument("--right-drum", default="cymbal_right")
    p.add_argument("--tth-lo", type=float, default=0.30)
    p.add_argument("--tth-hi", type=float, default=0.50)
    p.add_argument("--tth-on", choices=("both", "max"), default="both")
    p.add_argument("--expand-rounds", type=int, default=4)
    p.add_argument("--proposals", type=int, default=128)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--max-points", type=int, default=4096)
    p.add_argument("--no-score", action="store_true")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    out = Path(args.out) if args.out else ckpt.parent / "frontier_viz"
    if not out.is_absolute():
        out = ROOT / out

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")

    model, ck, step = load_model(ckpt, dev)
    scen = _scenario_from_ckpt(ck, buckets=(25, 50))
    # Cloud ingest is host-side; only the filtered score batch hits the GPU.
    pool = load_pool(pool_path, scen, "cpu")
    costs = build_costs(pool.buckets, phase2_cost_config(), device=dev)

    if args.frontier_pt:
        pt = Path(args.frontier_pt)
        if not pt.is_absolute():
            pt = ROOT / pt
        cloud = FrontierCloud(
            cfg=FrontierConfig(enabled=True, buckets=tuple(pool.buckets),
                               dt=float(pool.cfg.dt)),
            device=torch.device("cpu"), q_lo=pool.q_lo, q_hi=pool.q_hi,
        )
        cloud.load_state_dict(torch.load(pt, map_location="cpu", weights_only=False))
        cloud.q_lo, cloud.q_hi = pool.q_lo, pool.q_hi
        print(f"[cloud] loaded {pt}", flush=True)
    else:
        cloud = seed_cloud(pool, "cpu")
        if int(args.expand_rounds) > 0:
            gen = torch.Generator()
            gen.manual_seed(0)
            expand_cloud(
                cloud, rounds=int(args.expand_rounds),
                proposals=int(args.proposals), generator=gen,
            )

    tag = "baseline" if not args.frontier_pt else "frontier.pt"
    pairs = [(args.left_drum, args.right_drum)]
    extra = ("hihat", "snare")
    if extra not in pairs and (args.left_drum, args.right_drum) != extra:
        pairs.append(extra)
    for left, right in pairs:
        report_cloud(
            cloud=cloud, model=model, costs=costs, pool=pool, out_dir=out,
            step=step, left_drum=left, right_drum=right,
            tth_lo=float(args.tth_lo), tth_hi=float(args.tth_hi),
            tth_on=args.tth_on, chunk=int(args.chunk),
            score=not args.no_score, max_points=int(args.max_points),
            tag=tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
