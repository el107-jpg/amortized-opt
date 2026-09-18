#!/usr/bin/env python
"""Classify rolling-eval collisions and draw drums + stick paths.

Uses the same 81-case standard grid as ``train_phase2 --eval-cases 81``, the
same ``TrajectoryScorer`` that sets ``ok_collision``, and splits every
penetration into drum-body / under-volume / torso-head / stick-stick.

    python scripts/plot_collision_traj.py \
        --ckpt runs/rhythm_bellman_nf1_rope/last.pt \
        --device cuda:0 --out runs/collision_viz_rope

    # same grid, but the cases whose contact knot is not on the head:
    python scripts/plot_collision_traj.py \
        --ckpt runs/rhythm_bellman_nf1_rope/last.pt \
        --select miss --out runs/miss_viz_rope

Needs matplotlib (g1drum env). Does not use Isaac.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.strike import strike_axial_limits  # noqa: E402
from amortized_opt.train_cost import eval_subset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-cases", type=int, default=81)
    p.add_argument("--out", default="runs/collision_viz")
    p.add_argument("--select", choices=("collide", "miss", "all"), default="collide",
                   help="which eval cases to draw: colliding, not-on-head, or every case")
    p.add_argument("--max-plots", type=int, default=16,
                   help="worst selected cases to draw individually")
    p.add_argument("--start-above-drums", nargs="*", default=["snare", "floor_tom"],
                   help="same start filter as train_phase2 --eval-start-above-drums "
                        "(z-floor + idle-at-ready + no stick under any drum head). "
                        "Pass with no names for the historical hanging starts.")
    p.add_argument("--start-clearance", type=float, default=0.02,
                   help="min SDF (m) at q_start; matches train_phase2 "
                        "--eval-start-clearance.  0 restores kissing starts.")
    return p.parse_args(argv)


def fine_path(Q: np.ndarray, substeps: int) -> np.ndarray:
    if substeps <= 1:
        return Q
    frac = np.arange(substeps, dtype=float)[None, :, None] / substeps
    fine = (Q[:-1, None, :] * (1.0 - frac) + Q[1:, None, :] * frac).reshape(-1, Q.shape[1])
    return np.concatenate([fine, Q[-1:]], axis=0)


def classify_case(scorer: A.TrajectoryScorer, case, Q: np.ndarray) -> dict:
    """Per-type min SDF + a list of penetrating (hand, type, drum, point) hits."""
    s = scorer.collision_substeps
    N = int(case.N)
    hand = case.hand
    spec = case.strike
    fine = fine_path(Q, s)
    Qt = scorer._t(fine)
    n_fine = Qt.shape[0]
    first_exempt = max(0, (N - scorer.strike_exempt_waypoints)) * s
    last_start_exempt = min(scorer.start_exempt_waypoints * s + 1, n_fine)
    tgt = scorer.drum_index.get(str(spec.drum), -1)
    src = (
        scorer.drum_index.get(str(case.start_drum), -1)
        if case.start_drum is not None and scorer.start_exempt_waypoints > 0
        else -1
    )
    D = scorer.scene.num_drums
    names = list(scorer.scene.names)
    events = []
    mins = {k: float("inf") for k in (
        "body_L", "body_R", "under_L", "under_R", "self_L", "self_R"
    )}
    tips = {}
    for h in ("left", "right"):
        pts = scorer._points_world(Qt, h)          # (T, P, 3)
        tips[h] = pts[:, 0, :].detach().cpu().numpy()
        tip_w, butt_w, wrist_w = pts[:, 0], pts[:, 1], pts[:, -1]
        cap_b, cap_u = scorer._stick_drum_sdf(butt_w, tip_w)
        wr_b, wr_u = scorer._wrist_drum_sdf(wrist_w)
        sdf_s = scorer._self_sdf(Qt, pts)          # (T, 2) stick, wrist
        keep_cap = torch.ones(n_fine, D, device=scorer.device, dtype=scorer.dtype)
        if h == hand and tgt >= 0:
            keep_cap[first_exempt:, tgt] = 0.0
        if h == hand and src >= 0:
            keep_cap[:last_start_exempt, src] = 0.0
        sb = cap_b.masked_fill(keep_cap <= 0.5, float("inf"))
        su = cap_u.masked_fill(keep_cap <= 0.5, float("inf"))
        tag = "L" if h == "left" else "R"
        mins[f"body_{tag}"] = float(torch.minimum(sb, wr_b).min().item())
        mins[f"under_{tag}"] = float(torch.minimum(su, wr_u).min().item())
        mins[f"self_{tag}"] = float(sdf_s.min().item())
        mid = 0.5 * (butt_w + tip_w)
        for kind, sdf, point, xyz in (
            ("body", sb, "capsule", mid),
            ("under", su, "capsule", mid),
            ("body", wr_b, "wrist", wrist_w),
            ("under", wr_u, "wrist", wrist_w),
        ):
            hit = sdf < 0
            if not bool(hit.any()):
                continue
            idx = hit.nonzero(as_tuple=False)
            worst: dict[int, tuple[int, float]] = {}
            for t, d in idx.tolist():
                depth = -float(sdf[t, d].item())
                if d not in worst or depth > worst[d][1]:
                    worst[d] = (t, depth)
            for d, (t, depth) in worst.items():
                events.append({
                    "hand": h, "type": kind, "drum": names[d],
                    "point": point,
                    "depth_mm": 1000.0 * depth, "t": t,
                    "xyz": xyz[t].detach().cpu().numpy().tolist(),
                })
        hit_s = sdf_s < 0
        if bool(hit_s.any()):
            idx = hit_s.nonzero(as_tuple=False)
            labels = ("capsule", "wrist")
            xyzs = (mid, wrist_w)
            worst_s: dict[int, tuple[int, float]] = {}
            for t, p in idx.tolist():
                depth = -float(sdf_s[t, p].item())
                if p not in worst_s or depth > worst_s[p][1]:
                    worst_s[p] = (t, depth)
            for p, (t, depth) in worst_s.items():
                events.append({
                    "hand": h, "type": "self", "drum": "torso/head",
                    "point": labels[p],
                    "depth_mm": 1000.0 * depth, "t": t,
                    "xyz": xyzs[p][t].detach().cpu().numpy().tolist(),
                })
    stick = float(scorer._stick_stick_sdf(
        scorer._points_world(Qt, "left"), scorer._points_world(Qt, "right")
    ).min().item())
    mins["stick_stick"] = stick
    if stick < 0:
        events.append({
            "hand": "both", "type": "stick_stick", "drum": "cross-hand",
            "point": "capsule", "depth_mm": 1000.0 * (-stick), "t": -1,
            "xyz": [0.0, 0.0, 0.0],
        })
    types = sorted({e["type"] for e in events})
    return {
        "events": events, "mins": mins, "tips": tips, "types": types,
        "penetrates": any(e["type"] in ("body", "under", "self") for e in events),
    }


def draw_case(ax_xy, ax_xz, case, score, tips, events, heads, radii, names,
              p_contact=None):
    import matplotlib.patches as mpatches
    colors = {"left": "#2563eb", "right": "#ea580c"}
    for i, (c, r, n) in enumerate(zip(heads, radii, names)):
        ax_xy.add_patch(mpatches.Circle(
            (c[0], c[1]), r, fill=False, lw=1.2, color="#444",
        ))
        ax_xy.text(c[0], c[1], n.replace("_", "\n"), ha="center", va="center",
                   fontsize=6, color="#444")
        ax_xz.plot([c[0] - r, c[0] + r], [c[2], c[2]], color="#444", lw=1.2)
        ax_xz.text(c[0], c[2] + 0.02, n, ha="center", fontsize=6, color="#444")
    p_star = np.asarray(case.strike.position, float)
    ax_xy.scatter([p_star[0]], [p_star[1]], marker="*", s=80, c="green", zorder=5,
                  label="aim")
    ax_xz.scatter([p_star[0]], [p_star[2]], marker="*", s=80, c="green", zorder=5)
    if p_contact is not None:
        pc = np.asarray(p_contact, float)
        ax_xy.scatter([pc[0]], [pc[1]], marker="D", s=50, c="magenta", zorder=6,
                      label="contact knot")
        ax_xz.scatter([pc[0]], [pc[2]], marker="D", s=50, c="magenta", zorder=6)
        ax_xy.plot([pc[0], p_star[0]], [pc[1], p_star[1]], color="magenta",
                   lw=1.0, ls="--", alpha=0.8)
        ax_xz.plot([pc[0], p_star[0]], [pc[2], p_star[2]], color="magenta",
                   lw=1.0, ls="--", alpha=0.8)
    for h, path in tips.items():
        ax_xy.plot(path[:, 0], path[:, 1], color=colors[h], lw=1.4, label=f"{h} tip")
        ax_xy.scatter(path[0, 0], path[0, 1], color=colors[h], s=18, zorder=4)
        ax_xy.scatter(path[-1, 0], path[-1, 1], color=colors[h], s=28,
                      marker="s", zorder=4)
        ax_xz.plot(path[:, 0], path[:, 2], color=colors[h], lw=1.4)
        ax_xz.scatter(path[0, 0], path[0, 2], color=colors[h], s=18, zorder=4)
        ax_xz.scatter(path[-1, 0], path[-1, 2], color=colors[h], s=28,
                      marker="s", zorder=4)
    pens = [e for e in events if e["type"] in ("body", "under", "self")]
    if pens:
        xyz = np.array([e["xyz"] for e in pens], float)
        ax_xy.scatter(xyz[:, 0], xyz[:, 1], c="red", s=22, marker="x", zorder=6,
                      label="penetration")
        ax_xz.scatter(xyz[:, 0], xyz[:, 2], c="red", s=22, marker="x", zorder=6)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (m, pelvis)")
    ax_xy.set_ylabel("y (m, pelvis)")
    ax_xz.set_xlabel("x (m, pelvis)")
    ax_xz.set_ylabel("z (m, pelvis)")
    if score.ok_hit:
        gate = "HIT"
    else:
        gate = "MISS"
    if not score.ok_collision:
        gate = f"{gate}+COLLIDE"
    ax_xy.set_title(
        f"case {case.case_id}  {case.hand}->{case.drum}  {gate}\n"
        f"hit={int(score.ok_hit)} coll={int(score.ok_collision)} "
        f"tau={int(score.ok_torque)}  "
        f"err {score.contact_pos_err_mm:.0f}mm  "
        f"ax {1000*score.axial_dist_m:+.1f} in {1000*score.inplane_dist_m:.0f} mm",
        fontsize=9,
    )
    ax_xz.set_title("side (x–z)", fontsize=9)
    ax_xy.legend(fontsize=7, loc="upper right")
    ax_xy.grid(True, alpha=0.3)
    ax_xz.grid(True, alpha=0.3)


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import matplotlib.patches as mpatches  # noqa: F401,E402

    # rebind Circle through the module matplotlib.patches already imported in draw
    # (draw_case imports matplotlib.patches lazily). Fine.

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    print(f"[load] {args.ckpt} on {dev}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50)))
    print(f"[ckpt] step={ck.get('step', '?')} rope={mcfg.dec_rope} "
          f"plant={getattr(mcfg, 'dec_plant_feats', False)} buckets={buckets}",
          flush=True)

    scorer = A.TrajectoryScorer(device="cpu")
    above = tuple(args.start_above_drums or ())
    clearance = float(args.start_clearance) if above else 0.0
    cases = eval_subset(
        A.generate_grid(seed=0, grid="standard", scorer=scorer,
                        include_extra_drums=True,
                        start_above_drums=above,
                        start_not_under_drums=bool(above),
                        start_clearance_m=clearance),
        int(args.eval_cases),
    )
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    print(f"[eval] {len(cases)} cases", flush=True)
    plans, wall = A.batched_model_plans(
        model, cases, device=str(dev), kit=kit, library=library,
        buckets=buckets, num_drums=mcfg.num_drums, num_future=mcfg.num_future,
        iterations=1, init="quintic", amp_dtype=torch.bfloat16,
    )
    print(f"[eval] plans in {wall:.1f}s", flush=True)

    heads = np.asarray(scorer.scene.head_centers, float)
    radii = np.asarray(scorer.scene.radii, float)
    names = list(scorer.scene.names)

    ax_lo, ax_hi = strike_axial_limits()
    rows = []
    type_cases = Counter()
    drum_hits = Counter()
    hand_hits = Counter()
    point_hits = Counter()
    selected = []
    miss_gate = Counter()
    miss_by_drum = Counter()

    for c in cases:
        Q = plans[c.case_id]
        score = scorer.score(c, Q)
        detail = classify_case(scorer, c, Q)
        k = int(np.clip(c.strike.waypoint(c.dt), 1, c.N))
        p_c, _, _ = scorer.contact_world(Q[k], c.hand, score.contact_sigma)
        detail["p_contact"] = np.asarray(p_c, float)
        hit_R = float(c.strike.resolved_hit_radius())
        short = score.axial_dist_m > ax_hi + 1e-9
        deep = score.axial_dist_m < ax_lo - 1e-9
        off_disk = score.inplane_dist_m > hit_R + 1e-9
        off_hp = np.isfinite(score.half_plane_margin_m) and score.half_plane_margin_m < -1e-9
        rec = {
            "case_id": c.case_id, "drum": c.drum, "hand": c.hand, "N": c.N,
            "ok_hit": score.ok_hit, "ok_collision": score.ok_collision,
            "ok_torque": score.ok_torque, "accepted": score.accepted,
            "pen_max_mm": 1000.0 * score.coll_pen_max_all_m,
            "min_sdf_mm": 1000.0 * score.coll_min_sdf_all_m,
            "self_min_mm": 1000.0 * score.self_coll_min_sdf_m,
            "stick_stick_mm": 1000.0 * score.stick_stick_min_sdf_m,
            "contact_err_mm": float(score.contact_pos_err_mm),
            "axial_mm": 1000.0 * float(score.axial_dist_m),
            "inplane_mm": 1000.0 * float(score.inplane_dist_m),
            "hit_R_mm": 1000.0 * hit_R,
            "short": bool(short), "deep": bool(deep),
            "off_disk": bool(off_disk), "off_halfplane": bool(off_hp),
            "types": detail["types"],
            "n_events": len(detail["events"]),
        }
        rows.append(rec)
        keep = (
            (args.select == "all")
            or (args.select == "collide" and not score.ok_collision)
            or (args.select == "miss" and not score.ok_hit)
        )
        if keep:
            selected.append((c, score, detail))
        if not score.ok_collision:
            for t in set(detail["types"]) or {"unknown"}:
                type_cases[t] += 1
            for e in detail["events"]:
                if e["type"] in ("body", "under"):
                    drum_hits[e["drum"]] += 1
                    point_hits[e["point"]] += 1
                    hand_hits[e["hand"]] += 1
                elif e["type"] == "self":
                    hand_hits[e["hand"]] += 1
                    point_hits[e["point"]] += 1
        if not score.ok_hit:
            miss_by_drum[c.drum] += 1
            if short:
                miss_gate["short_of_aim"] += 1
            if deep:
                miss_gate["through_head"] += 1
            if off_disk:
                miss_gate["off_disk"] += 1
            if off_hp:
                miss_gate["off_halfplane"] += 1

    n = len(rows)
    n_coll = sum(1 for r in rows if not r["ok_collision"])
    n_miss = sum(1 for r in rows if not r["ok_hit"])
    n_hit = sum(1 for r in rows if r["ok_hit"])
    print("\n======== rolling 81-grid ========", flush=True)
    print(f"  collide {n_coll}/{n}  hit {n_hit}/{n}  miss {n_miss}/{n}  "
          f"tau {sum(r['ok_torque'] for r in rows)}/{n}  "
          f"acc {sum(r['accepted'] for r in rows)}/{n}", flush=True)
    print("  collision types (a case can have several):", flush=True)
    for t in ("body", "under", "self", "stick_stick"):
        print(f"    {t:12s}  {type_cases[t]:3d}/{n_coll} colliding cases", flush=True)
    print("  miss gates (not on-head; a case can fail several):", flush=True)
    for t in ("short_of_aim", "through_head", "off_disk", "off_halfplane"):
        print(f"    {t:14s}  {miss_gate[t]:3d}/{n_miss} miss cases", flush=True)
    print("  miss by drum:", dict(miss_by_drum), flush=True)
    miss_ax = [r["axial_mm"] for r in rows if not r["ok_hit"]]
    miss_ip = [r["inplane_mm"] for r in rows if not r["ok_hit"]]
    if miss_ax:
        print(f"  miss axial mm  p50 {np.median(miss_ax):+.1f}  "
              f"p10 {np.percentile(miss_ax, 10):+.1f}  "
              f"p90 {np.percentile(miss_ax, 90):+.1f}", flush=True)
        print(f"  miss inplane mm p50 {np.median(miss_ip):.1f}  "
              f"p90 {np.percentile(miss_ip, 90):.1f}  "
              f"(legal band axial [{1000*ax_lo:.0f}, {1000*ax_hi:.0f}] mm)",
              flush=True)

    (out / "summary.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "step": ck.get("step"),
        "select": args.select,
        "n": n, "n_collide": n_coll, "n_miss": n_miss, "n_hit": n_hit,
        "type_cases": dict(type_cases),
        "miss_gate": dict(miss_gate),
        "miss_by_drum": dict(miss_by_drum),
        "drum_hits": dict(drum_hits),
        "hand_hits": dict(hand_hits),
        "point_hits": dict(point_hits),
        "rows": rows,
    }, indent=2, default=str))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax_xy, ax_xz = axes
    for c, r, n_ in zip(heads, radii, names):
        ax_xy.add_patch(plt.Circle((c[0], c[1]), r, fill=False, lw=1.4, color="#222"))
        ax_xy.text(c[0], c[1], n_.replace("_", "\n"), ha="center", va="center",
                   fontsize=7, color="#222")
        ax_xz.plot([c[0] - r, c[0] + r], [c[2], c[2]], color="#222", lw=1.4)
    for c, score, detail in selected:
        for h, path in detail["tips"].items():
            col = "#2563eb" if h == "left" else "#ea580c"
            ax_xy.plot(path[:, 0], path[:, 1], color=col, lw=0.7, alpha=0.45)
            ax_xz.plot(path[:, 0], path[:, 2], color=col, lw=0.7, alpha=0.45)
        pc = detail.get("p_contact")
        if pc is not None:
            ax_xy.scatter([pc[0]], [pc[1]], c="magenta", s=12, marker="D",
                          zorder=5, alpha=0.7)
            ax_xz.scatter([pc[0]], [pc[2]], c="magenta", s=12, marker="D",
                          zorder=5, alpha=0.7)
    ax_xy.set_aspect("equal")
    label = {"collide": "colliding", "miss": "miss (not on-head)",
             "all": "all"}[args.select]
    ax_xy.set_title(f"{label} tip paths ({len(selected)}/{n})  "
                    f"blue=L orange=R  magenta=contact knot")
    ax_xy.set_xlabel("x (m, pelvis)"); ax_xy.set_ylabel("y (m, pelvis)")
    ax_xz.set_title("side (x–z)")
    ax_xz.set_xlabel("x (m, pelvis)"); ax_xz.set_ylabel("z (m, pelvis)")
    ax_xy.grid(True, alpha=0.3); ax_xz.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=140)
    plt.close(fig)

    if args.select == "miss":
        selected.sort(key=lambda t: t[1].contact_pos_err_mm, reverse=True)
    else:
        selected.sort(key=lambda t: t[1].coll_pen_max_all_m, reverse=True)
    for i, (c, score, detail) in enumerate(selected[: args.max_plots]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        draw_case(axes[0], axes[1], c, score, detail["tips"], detail["events"],
                  heads, radii, names, p_contact=detail.get("p_contact"))
        fig.tight_layout()
        fig.savefig(out / f"case_{c.case_id:03d}_{c.hand}_{c.drum}.png", dpi=130)
        plt.close(fig)

    print(f"\n[out] {out}", flush=True)
    print(f"      overview.png + {min(len(selected), args.max_plots)} case plots "
          f"+ summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
