#!/usr/bin/env python
"""How far the decoder moves the strike from the quintic-init terminal.

The seed is a library-pose quintic, then one DLS step toward the aim
(``correct_quintic_init_arms``).  Context / aim stay fixed; only the seed
changes.  Contact is the same point the 81-grid scores: bead on pads,
playing-shaft closest point on cymbals / hi-hat.

    python scripts/probe_init_terminal_shift.py \
        --ckpt runs/rhythm_timing_direct_admission_gpu1_20k_20260918/last.pt \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.fk import ChainFK  # noqa: E402
from amortized_opt.idle_home import resolve_idle_home  # noqa: E402
from amortized_opt.ik import correct_quintic_init_arms  # noqa: E402
from amortized_opt.limits import urdf_position_limits  # noqa: E402
from amortized_opt.planner import (  # noqa: E402
    _bucket_groups,
    _warmstart,
    case_goal_snapshot,
)
from amortized_opt.rhythm import build_context_inputs  # noqa: E402
from amortized_opt.sdf import hit_region_dist, segment_hit_region  # noqa: E402
from amortized_opt.strike import shaft_hit_sigma_range, strike_axial_limits  # noqa: E402
from amortized_opt.tokenizer import build_waypoint_masks  # noqa: E402
from amortized_opt.train_cost import eval_subset  # noqa: E402
from amortized_opt.warmstart import arm_slice, seed_waist_pr_to_sp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = (
    "runs/rhythm_timing_direct_admission_gpu1_20k_20260918/last.pt"
)
VARIANTS = ("library", "dls", "dls_plus", "dls_minus", "hold")
OFFSET_M = 0.04


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def inplane_basis(n_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(n_out)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, ref)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(n, e1)
    return e1, e2


def pct(x, qs=(10, 50, 90)) -> dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"mean": float("nan"), "n": 0}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {
        "mean": float(x.mean()),
        "n": int(x.size),
    }


def traj_from_init(case, init_H: np.ndarray) -> np.ndarray:
    n = int(case.N)
    return np.concatenate(
        [np.asarray(case.q_start, float)[None, :], np.asarray(init_H[:n], float)],
        axis=0,
    )


def contact_geom(scorer: A.TrajectoryScorer, case, Q: np.ndarray) -> dict:
    """Strike-knot contact in the pelvis frame, matching TrajectoryScorer.score."""
    Q = np.asarray(Q, float)
    spec = case.strike
    n = int(case.N)
    hand = case.hand
    k = int(np.clip(spec.waypoint(case.dt), 1, n))
    p_star = np.asarray(spec.position, float)
    n_in = _unit(spec.velocity)
    n_out = -n_in
    sigma_spec = spec.resolved_contact_sigma()
    rng = shaft_hit_sigma_range(spec.drum)
    if rng is None:
        sigma_lo = sigma_hi = float(sigma_spec)
    else:
        sigma_lo, sigma_hi = float(rng[0]), float(rng[1])
    a_c, _, _ = scorer.contact_world(Q[k], hand, sigma_lo)
    b_c, p_tip, _ = scorer.contact_world(Q[k], hand, sigma_hi)
    d_off = (
        np.zeros(3) if spec.disk_center_offset is None
        else np.asarray(spec.disk_center_offset, float)
    )
    hp_dir = (
        np.zeros(3) if spec.half_plane_dir is None
        else np.asarray(spec.half_plane_dir, float)
    )
    has_hp = 0.0 if spec.half_plane_dir is None else 1.0
    hit_r = spec.resolved_hit_radius()
    ax_lo, ax_hi = strike_axial_limits()
    tt = scorer._t
    _, s_t, p_c_t = segment_hit_region(
        tt(a_c), tt(b_c), tt(p_star), tt(n_out), hit_r,
        tt(d_off), tt(hp_dir), has_hp, ax_lo, ax_hi, margin=0.0,
    )
    dist_m, _, _ = segment_hit_region(
        tt(a_c), tt(b_c), tt(p_star), tt(n_out), hit_r,
        tt(d_off), tt(hp_dir), has_hp, ax_lo, ax_hi,
        margin=float(scorer.thresholds.hit_margin_m),
    )
    p_c = p_c_t.detach().cpu().numpy()
    _d0, axial_t, d_in_t = hit_region_dist(
        tt(p_c), tt(p_star), tt(n_out), hit_r,
        tt(d_off), tt(hp_dir), has_hp, ax_lo, ax_hi, margin=0.0,
    )
    d_in = d_in_t.detach().cpu().numpy()
    q_hit = Q[k]
    acols = arm_slice(hand)
    return {
        "k_hit": k,
        "p_contact": np.asarray(p_c, float),
        "p_tip": np.asarray(p_tip, float),
        "p_star": p_star,
        "n_out": n_out,
        "axial_m": float(axial_t.item()),
        "inplane_m": float(np.linalg.norm(d_in)),
        "d_in": d_in,
        "on_head": bool(float(dist_m.item()) <= 1e-9),
        "q_hit": q_hit,
        "q_arm": q_hit[acols],
        "sigma": float(sigma_lo + float(s_t.item()) * (sigma_hi - sigma_lo)),
    }


@torch.no_grad()
def run_variants(
    model,
    cases,
    *,
    device: str,
    kit,
    library,
    buckets,
    num_drums: int,
    num_future: int,
    waist_upright_ease_wp,
    amp_dtype,
    offset_m: float,
    joint_names=A.G1_UPPER_BODY_JOINTS,
):
    """One encode per batch; several seeds; context/aim never change."""
    dev = torch.device(device)
    home = resolve_idle_home("default", joint_names=tuple(joint_names))
    q_bounds = urdf_position_limits(joint_names)
    core = model.module if hasattr(model, "module") else model
    core.eval()
    fk = ChainFK(device=dev, dtype=torch.float32)
    n_dof = len(joint_names)

    inits = {v: {} for v in VARIANTS}
    plans = {v: {} for v in VARIANTS}
    t0 = time.perf_counter()
    for H, group in _bucket_groups(cases, buckets).items():
        B = len(group)
        snaps = [case_goal_snapshot(c, kit) for c in group]
        yaw_i = joint_names.index("waist_yaw_joint")
        ctx = build_context_inputs(
            kit, snaps, num_drums=num_drums, num_future=num_future,
            frame="env", device=dev, dtype=torch.float32, idle_home=home,
            yaw_now=np.asarray(
                [np.asarray(c.q_start, float)[yaw_i] for c in group]
            ),
        )
        to_t = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a, float), device=dev, dtype=torch.float32
        )
        hl = to_t([c.N * c.dt if c.hand == "left" else 0.0 for c in group])
        hr = to_t([c.N * c.dt if c.hand == "right" else 0.0 for c in group])
        masks = build_waypoint_masks(
            hl, hr, max_horizon=H, dt=float(group[0].dt),
            active_left=torch.tensor([c.hand == "left" for c in group], device=dev),
            active_right=torch.tensor([c.hand == "right" for c in group], device=dev),
            joint_names=joint_names,
        )
        library_np = np.zeros((B, H, n_dof))
        hold_np = np.zeros((B, H, n_dof))
        aims_true = torch.zeros(B, 2, 3, device=dev, dtype=torch.float32)
        aims_plus = torch.zeros(B, 2, 3, device=dev, dtype=torch.float32)
        aims_minus = torch.zeros(B, 2, 3, device=dev, dtype=torch.float32)
        hit = torch.full((B, 2), -1, device=dev, dtype=torch.long)
        for i, c in enumerate(group):
            ws = _warmstart(
                c, library, H, joint_names=joint_names, idle_home=home,
                waist_upright_ease_wp=waist_upright_ease_wp,
            )
            library_np[i] = np.asarray(ws.init_traj, float)
            hold_np[i] = np.repeat(np.asarray(c.q_start, float)[None, :], H, axis=0)
            hi = 0 if c.hand == "left" else 1
            hit[i, hi] = int(c.N)
            n_out = -_unit(c.strike.velocity)
            e1, _e2 = inplane_basis(n_out)
            p = np.asarray(c.strike.position, float) + A.PELVIS_IN_ENV
            aims_true[i, hi] = to_t(p)
            aims_plus[i, hi] = to_t(p + offset_m * e1)
            aims_minus[i, hi] = to_t(p - offset_m * e1)
        state = torch.cat(
            [to_t([c.q_start for c in group]), to_t([c.qdot_start for c in group])],
            dim=-1,
        )
        amp = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp_dtype is not None and dev.type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with amp:
            enc = core.encode_context(ctx)
            lib_t = to_t(library_np) * masks.elem_mask
            if enc.waist_pr_sp is not None:
                lib_t = seed_waist_pr_to_sp(
                    lib_t, state[:, :n_dof], state[:, n_dof:], enc.waist_pr_sp,
                    joint_names=joint_names, dt=float(group[0].dt),
                    ease_wp=waist_upright_ease_wp,
                )
            seeds = {
                "library": lib_t,
                "hold": to_t(hold_np) * masks.elem_mask,
            }
            for name, aims in (
                ("dls", aims_true),
                ("dls_plus", aims_plus),
                ("dls_minus", aims_minus),
            ):
                seeds[name] = correct_quintic_init_arms(
                    fk, lib_t, state[:, :n_dof], hit, aims, steps=1,
                    q_bounds=q_bounds, joint_names=joint_names,
                    qd0=state[:, n_dof:], dt=float(group[0].dt),
                )
            for name, seed in seeds.items():
                res = core.refine(seed, state, enc, masks, iterations=1)
                seed_np = seed.float().cpu().numpy()
                traj_np = res.traj.float().cpu().numpy()
                for i, c in enumerate(group):
                    cid = int(c.case_id)
                    inits[name][cid] = traj_from_init(c, seed_np[i])
                    plans[name][cid] = traj_from_init(c, traj_np[i])
    return inits, plans, time.perf_counter() - t0


def mm(x: float) -> float:
    return 1000.0 * float(x)


def row_for(case, init_g: dict, model_g: dict) -> dict:
    dp = model_g["p_contact"] - init_g["p_contact"]
    to_aim = init_g["p_star"] - init_g["p_contact"]
    d_norm = float(np.linalg.norm(dp))
    a_norm = float(np.linalg.norm(to_aim))
    cos_aim = float(dp @ to_aim) / max(d_norm * a_norm, 1e-12)
    n_out = init_g["n_out"]
    d_ax = float(dp @ n_out)
    d_ip = dp - d_ax * n_out
    dq_arm = model_g["q_arm"] - init_g["q_arm"]
    return {
        "case_id": int(case.case_id),
        "drum": str(case.drum),
        "hand": str(case.hand),
        "N": int(case.N),
        "horizon_s": float(case.horizon_s),
        "init_err_mm": mm(np.linalg.norm(init_g["p_contact"] - init_g["p_star"])),
        "model_err_mm": mm(np.linalg.norm(model_g["p_contact"] - model_g["p_star"])),
        "init_tip_err_mm": mm(np.linalg.norm(init_g["p_tip"] - init_g["p_star"])),
        "model_tip_err_mm": mm(np.linalg.norm(model_g["p_tip"] - model_g["p_star"])),
        "delta_contact_mm": mm(d_norm),
        "delta_tip_mm": mm(np.linalg.norm(model_g["p_tip"] - init_g["p_tip"])),
        "delta_axial_mm": mm(d_ax),
        "delta_inplane_mm": mm(np.linalg.norm(d_ip)),
        "delta_arm_l2_rad": float(np.linalg.norm(dq_arm)),
        "delta_arm_linf_rad": float(np.max(np.abs(dq_arm))),
        "cos_toward_aim": cos_aim,
        "init_axial_mm": mm(init_g["axial_m"]),
        "model_axial_mm": mm(model_g["axial_m"]),
        "init_inplane_mm": mm(init_g["inplane_m"]),
        "model_inplane_mm": mm(model_g["inplane_m"]),
        "init_on_head": bool(init_g["on_head"]),
        "model_on_head": bool(model_g["on_head"]),
        "p_init": init_g["p_contact"].tolist(),
        "p_model": model_g["p_contact"].tolist(),
        "p_star": init_g["p_star"].tolist(),
    }


def pair_gain(base_init, base_model, other_init, other_model) -> dict:
    di = other_init["p_contact"] - base_init["p_contact"]
    dm = other_model["p_contact"] - base_model["p_contact"]
    ni = float(np.linalg.norm(di))
    nm = float(np.linalg.norm(dm))
    cos = float(di @ dm) / max(ni * nm, 1e-12)
    return {
        "init_shift_mm": mm(ni),
        "model_shift_mm": mm(nm),
        "gain": float(nm / max(ni, 1e-6)),
        "cos": cos,
    }


def summarize_rows(rows: list[dict]) -> dict:
    d = np.array([r["delta_contact_mm"] for r in rows], float)
    bins = {
        "lt_10mm": int(np.sum(d < 10.0)),
        "10_30mm": int(np.sum((d >= 10.0) & (d < 30.0))),
        "30_80mm": int(np.sum((d >= 30.0) & (d < 80.0))),
        "ge_80mm": int(np.sum(d >= 80.0)),
    }
    by_drum = {}
    for r in rows:
        by_drum.setdefault(r["drum"], []).append(r["delta_contact_mm"])
    return {
        "n": len(rows),
        "delta_contact_mm": pct(d),
        "delta_tip_mm": pct([r["delta_tip_mm"] for r in rows]),
        "delta_axial_mm": pct([r["delta_axial_mm"] for r in rows]),
        "delta_inplane_mm": pct([r["delta_inplane_mm"] for r in rows]),
        "delta_arm_l2_rad": pct([r["delta_arm_l2_rad"] for r in rows]),
        "delta_arm_linf_mrad": pct([1000.0 * r["delta_arm_linf_rad"] for r in rows]),
        "init_err_mm": pct([r["init_err_mm"] for r in rows]),
        "model_err_mm": pct([r["model_err_mm"] for r in rows]),
        "cos_toward_aim": pct([r["cos_toward_aim"] for r in rows]),
        "init_on_head": sum(1 for r in rows if r["init_on_head"]),
        "model_on_head": sum(1 for r in rows if r["model_on_head"]),
        "moved_on_head": sum(
            1 for r in rows if r["model_on_head"] and not r["init_on_head"]
        ),
        "left_head": sum(
            1 for r in rows if r["init_on_head"] and not r["model_on_head"]
        ),
        "delta_bins": bins,
        "by_drum_delta_p50_mm": {
            k: float(np.median(v)) for k, v in sorted(by_drum.items())
        },
        "by_drum_delta_p90_mm": {
            k: float(np.percentile(v, 90)) for k, v in sorted(by_drum.items())
        },
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-cases", type=int, default=81)
    p.add_argument("--offset-m", type=float, default=OFFSET_M)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default="analysis/init_terminal_shift_20260920.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16

    print(f"[load] {ckpt} on {dev}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    scen = ck.get("scenario_cfg") or {}
    buckets = tuple(scen.get("buckets", (25, 50, 100)))
    ease_wp = scen.get("waist_upright_ease_wp")
    print(
        f"[ckpt] step={ck.get('step', '?')} rope={mcfg.dec_rope} "
        f"plant={getattr(mcfg, 'dec_plant_feats', False)} buckets={buckets} "
        f"ease_wp={ease_wp}",
        flush=True,
    )

    scorer = A.TrajectoryScorer(device="cpu")
    cases = eval_subset(
        A.generate_grid(
            seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
            start_above_drums=("snare", "floor_tom"),
            start_not_under_drums=True, start_clearance_m=0.02,
        ),
        int(args.eval_cases),
    )
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    print(f"[eval] {len(cases)} cases  offset={1000*args.offset_m:.0f} mm", flush=True)

    inits, plans, wall = run_variants(
        model, cases, device=str(dev), kit=kit, library=library, buckets=buckets,
        num_drums=mcfg.num_drums, num_future=mcfg.num_future,
        waist_upright_ease_wp=ease_wp, amp_dtype=amp, offset_m=float(args.offset_m),
    )
    print(f"[eval] {len(VARIANTS)} variants in {wall:.1f}s", flush=True)

    geoms = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for c in cases:
            cid = int(c.case_id)
            geoms[v][cid] = {
                "init": contact_geom(scorer, c, inits[v][cid]),
                "model": contact_geom(scorer, c, plans[v][cid]),
            }

    variant_rows = {}
    for v in VARIANTS:
        variant_rows[v] = [
            row_for(c, geoms[v][int(c.case_id)]["init"], geoms[v][int(c.case_id)]["model"])
            for c in cases
        ]

    gains_plus, gains_minus, gains_lib = [], [], []
    for c in cases:
        cid = int(c.case_id)
        dls_i, dls_m = geoms["dls"][cid]["init"], geoms["dls"][cid]["model"]
        gains_plus.append(pair_gain(
            dls_i, dls_m, geoms["dls_plus"][cid]["init"], geoms["dls_plus"][cid]["model"],
        ))
        gains_minus.append(pair_gain(
            dls_i, dls_m, geoms["dls_minus"][cid]["init"], geoms["dls_minus"][cid]["model"],
        ))
        gains_lib.append(pair_gain(
            dls_i, dls_m, geoms["library"][cid]["init"], geoms["library"][cid]["model"],
        ))

    def gain_summary(gains):
        return {
            "init_shift_mm": pct([g["init_shift_mm"] for g in gains]),
            "model_shift_mm": pct([g["model_shift_mm"] for g in gains]),
            "gain": pct([g["gain"] for g in gains]),
            "cos": pct([g["cos"] for g in gains]),
            "gain_lt_0.25": int(sum(g["gain"] < 0.25 for g in gains)),
            "gain_0.25_0.75": int(sum(0.25 <= g["gain"] < 0.75 for g in gains)),
            "gain_ge_0.75": int(sum(g["gain"] >= 0.75 for g in gains)),
        }

    report = {
        "ckpt": str(ckpt.relative_to(ROOT) if ckpt.is_relative_to(ROOT) else ckpt),
        "step": ck.get("step"),
        "n": len(cases),
        "offset_mm": 1000.0 * float(args.offset_m),
        "grid": "standard 81-subsample, extra drums, z-floor, 2cm start clearance",
        "variants": {v: summarize_rows(variant_rows[v]) for v in VARIANTS},
        "sensitivity": {
            "dls_vs_plus": gain_summary(gains_plus),
            "dls_vs_minus": gain_summary(gains_minus),
            "dls_vs_library": gain_summary(gains_lib),
        },
        "rows": {
            "dls": variant_rows["dls"],
            "library": variant_rows["library"],
            "hold": variant_rows["hold"],
        },
        "gains": {
            "plus": [
                {"case_id": int(c.case_id), "drum": c.drum, **g}
                for c, g in zip(cases, gains_plus)
            ],
            "minus": [
                {"case_id": int(c.case_id), "drum": c.drum, **g}
                for c, g in zip(cases, gains_minus)
            ],
        },
    }

    dls = report["variants"]["dls"]
    sens = report["sensitivity"]["dls_vs_plus"]
    print("\n======== init → model (DLS seed, the live path) ========", flush=True)
    print(
        f"  contact Δ  p10 {dls['delta_contact_mm']['p10']:.1f}  "
        f"p50 {dls['delta_contact_mm']['p50']:.1f}  "
        f"p90 {dls['delta_contact_mm']['p90']:.1f} mm",
        flush=True,
    )
    print(
        f"  in-plane Δ p50 {dls['delta_inplane_mm']['p50']:.1f}  "
        f"axial Δ p50 {dls['delta_axial_mm']['p50']:+.1f} mm",
        flush=True,
    )
    print(
        f"  arm |Δq|   p50 {dls['delta_arm_l2_rad']['p50']:.3f} rad  "
        f"linf p50 {dls['delta_arm_linf_mrad']['p50']:.1f} mrad",
        flush=True,
    )
    print(
        f"  aim err    init p50 {dls['init_err_mm']['p50']:.1f} → "
        f"model p50 {dls['model_err_mm']['p50']:.1f} mm",
        flush=True,
    )
    print(
        f"  on-head    init {dls['init_on_head']}/{dls['n']}  "
        f"model {dls['model_on_head']}/{dls['n']}  "
        f"rescued {dls['moved_on_head']}  left {dls['left_head']}",
        flush=True,
    )
    print(f"  |Δ| bins   {dls['delta_bins']}", flush=True)
    print(f"  by drum p50 {dls['by_drum_delta_p50_mm']}", flush=True)
    print(
        f"  cos(Δ, aim-init) p50 {dls['cos_toward_aim']['p50']:+.2f}",
        flush=True,
    )
    print("\n======== seed terminal moved, context/aim held ========", flush=True)
    print(
        f"  ±{report['offset_mm']:.0f} mm in-plane DLS offset: "
        f"init shift p50 {sens['init_shift_mm']['p50']:.1f} mm → "
        f"model shift p50 {sens['model_shift_mm']['p50']:.1f} mm  "
        f"gain p50 {sens['gain']['p50']:.2f}  cos p50 {sens['cos']['p50']:+.2f}",
        flush=True,
    )
    lib = report["sensitivity"]["dls_vs_library"]
    print(
        f"  skip DLS (library vs DLS seed): "
        f"init shift p50 {lib['init_shift_mm']['p50']:.1f} mm → "
        f"model shift p50 {lib['model_shift_mm']['p50']:.1f} mm  "
        f"gain p50 {lib['gain']['p50']:.2f}",
        flush=True,
    )
    hold = report["variants"]["hold"]
    print(
        f"  hold init: model vs hold-terminal Δ p50 "
        f"{hold['delta_contact_mm']['p50']:.1f} mm  "
        f"on-head {hold['model_on_head']}/{hold['n']}",
        flush=True,
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
