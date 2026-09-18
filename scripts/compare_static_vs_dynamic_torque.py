#!/usr/bin/env python
"""Compare gravity-only static torque vs full knot RNEA on drumming plans.

    python scripts/compare_static_vs_dynamic_torque.py \\
        --pool runs/p2pool_....npz --ckpt runs/phase1_.../best.pt

For each valid (waypoint, joint) on knot finite-difference states, reports
how much looser the static hinge ``|RNEA(q,0,0)|`` is than the dynamic
``|RNEA(q,qd,qdd)|`` that training would charge under full torque physics.

``--source decoder`` scores the model's ``dec.traj`` (default).
``--source init`` scores the noised warm-start spline the decoder conditions on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.constants import G1_UPPER_BODY_JOINTS  # noqa: E402
from amortized_opt.limits import (  # noqa: E402
    available_torque_torch,
    isaac_effort_limits,
    scale_soft_wrist_effort,
    isaac_velocity_limits,
)
from amortized_opt.phase2_data import (  # noqa: E402
    KIND_FAR_FETCHED,
    KIND_POST_HIT,
    KIND_RHYTHM_IDLE,
    Phase2Noise,
    Phase2Pool,
    Phase2ScenarioConfig,
)
from amortized_opt.train_cost import encode, phase2_cost_config  # noqa: E402
from amortized_opt.tspline import TrajectorySplineBuilder  # noqa: E402
from amortized_opt.warmstart_data import CONTEXT_FIELDS  # noqa: E402


KIND_NAMES = {
    int(KIND_RHYTHM_IDLE): "idle",
    int(KIND_POST_HIT): "post_hit",
    int(KIND_FAR_FETCHED): "far",
}


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


def _softplus_hinge(x: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.nn.functional.softplus(x / beta) * beta


def _joint_group(name: str) -> str:
    if "shoulder" in name:
        return "shoulder"
    if "elbow" in name:
        return "elbow"
    if "wrist" in name:
        return "wrist"
    if "waist" in name:
        return "waist"
    return "other"


@torch.no_grad()
def compare(
    *,
    pool_npz: Path,
    ckpt: Path | None,
    device: str,
    n_batches: int,
    batch_size: int,
    bucket: int,
    seed: int,
    source: str,
    eta_torque: float,
    soft_wrist_effort_scale: float,
    near_hit_wp: int,
) -> dict:
    names = tuple(G1_UPPER_BODY_JOINTS)
    dev = torch.device(device)
    dtype = torch.float32

    buckets, stats = _load_npz_buckets(pool_npz)
    bucket_keys = tuple(sorted(buckets))
    if bucket not in buckets:
        bucket = max(buckets)
    scen_cfg = Phase2ScenarioConfig(
        horizon=max(bucket_keys), buckets=bucket_keys,
    )
    ck = None
    if ckpt is not None:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        scen_cfg = _scenario_from_ckpt(ck, bucket_keys)
    pool = Phase2Pool(
        buckets, cfg=scen_cfg, noise=Phase2Noise(), device=dev, stats=stats,
    )
    rnea = A.BatchedRNEA.from_urdf(
        joint_names=names, with_armature=True, device=dev, dtype=dtype,
    )
    builder = TrajectorySplineBuilder(
        max_segments=int(bucket), dt=float(scen_cfg.dt), joint_names=names,
        device=dev, dtype=dtype,
    )
    tau_hard = torch.as_tensor(
        scale_soft_wrist_effort(
            isaac_effort_limits(names), names, scale=soft_wrist_effort_scale,
        ),
        device=dev, dtype=dtype,
    )
    v_hard = torch.as_tensor(isaac_velocity_limits(names), device=dev, dtype=dtype)
    beta = float(phase2_cost_config().beta_torque)

    model = None
    if source == "decoder":
        if ckpt is None or ck is None:
            raise ValueError("--source decoder requires --ckpt")
        mcfg = A.model_config_from_checkpoint(ck, num_future=1)
        model = A.build_model(mcfg, dev)
        sd = dict(ck["model"])
        key = "encoder.future_rank_emb.weight"
        if key in sd and int(mcfg.num_future) < sd[key].shape[0]:
            sd[key] = sd[key][: int(mcfg.num_future)]
        model.load_state_dict(sd, strict=False)
        model.eval()

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    # Per-joint accumulators (masked elements only).
    acc = {j: defaultdict(float) for j in names}
    acc_n = {j: 0 for j in names}
    kind_acc = defaultdict(lambda: defaultdict(float))
    kind_n = defaultdict(int)
    near_acc = defaultdict(float)
    near_n = 0

    for _ in range(n_batches):
        idx = pool.sample_idx(bucket, batch_size, generator=gen)
        batch = pool.batch(bucket, idx, generator=gen)
        masks = batch["masks"]
        H = int(bucket)
        if source == "decoder":
            enc = encode(model, batch["context"])
            wp = model(batch["init"], batch["state"], enc, masks).traj
        elif source == "init":
            wp = batch["init"]
        else:
            raise ValueError(f"unknown source {source!r}")

        sp = builder.from_masks(
            batch["q0"], wp[:, :H], batch["v0"], masks.elem_mask[:, :H],
        )
        q, qd, qdd = sp.knot_finite_differences(batch["v0"])
        B, n, D = q.shape
        idx_t = torch.arange(sp.max_segments + 1, device=dev).view(1, -1, 1)
        knot_mask = (idx_t <= sp.lengths.unsqueeze(1)).to(dtype).expand(B, n, D)
        # Match ``FusedCost._torque*``: knot-fd states, valid where the spline
        # has length, not the strike elem mask (that lives on waypoints).
        m = knot_mask.reshape(B * n, D)

        qf = q.reshape(B * n, D)
        z = torch.zeros_like(qf)
        tau_g = rnea.rnea(qf, z, z).abs()
        tau_dyn = rnea.rnea(qf, qd.reshape(B * n, D), qdd.reshape(B * n, D)).abs()

        lim_hard = available_torque_torch(
            qd, tau_hard.view(1, 1, D), v_hard.view(1, 1, D), eta=1.0,
        )
        lim = float(eta_torque) * lim_hard
        lim_flat = lim.reshape(B * n, D)

        hinge_s = _softplus_hinge(tau_g - lim_flat, beta)
        hinge_d = _softplus_hinge(tau_dyn - lim_flat, beta)
        over_s = (tau_g - lim_flat).clamp_min(0.0)
        over_d = (tau_dyn - lim_flat).clamp_min(0.0)
        dyn_extra = (tau_dyn - tau_g).clamp_min(0.0)

        qd_abs = qd.reshape(B * n, D).abs()
        qdd_abs = qdd.reshape(B * n, D).abs()

        kinds = batch["kind"].detach().cpu().numpy()
        hit = batch["hit"]
        act = batch["active"]
        near_hit = torch.zeros(B, n, dtype=torch.bool, device=dev)
        for b in range(B):
            hits = []
            for h in (0, 1):
                if bool(act[b, h].item()):
                    hits.append(int(hit[b, h].clamp_min(0).item()))
            if hits:
                h0 = min(hits)
                lo = max(0, h0 - int(near_hit_wp))
                hi = min(n - 1, h0 + int(near_hit_wp))
                near_hit[b, lo:hi + 1] = True

        for j, nm in enumerate(names):
            k = m[:, j] > 0.5
            if int(k.sum()) < 1:
                continue
            acc[nm]["tau_g_rms"] += float(torch.sqrt((tau_g[k, j] ** 2).mean()))
            acc[nm]["tau_dyn_rms"] += float(torch.sqrt((tau_dyn[k, j] ** 2).mean()))
            acc[nm]["qd_rms"] += float(torch.sqrt((qd_abs[k, j] ** 2).mean()))
            acc[nm]["qdd_rms"] += float(torch.sqrt((qdd_abs[k, j] ** 2).mean()))
            acc[nm]["hinge_static"] += float(hinge_s[k, j].mean())
            acc[nm]["hinge_dynamic"] += float(hinge_d[k, j].mean())
            acc[nm]["frac_over_static"] += float((over_s[k, j] > 0).float().mean())
            acc[nm]["frac_over_dynamic"] += float((over_d[k, j] > 0).float().mean())
            acc[nm]["frac_dyn_over_static"] += float((dyn_extra[k, j] > 0).float().mean())
            acc[nm]["dyn_minus_static_rms"] += float(
                torch.sqrt((dyn_extra[k, j] ** 2).mean())
            )
            acc[nm]["p99_ratio"] += float(
                torch.quantile(tau_dyn[k, j] / tau_g[k, j].clamp_min(1e-3), 0.99)
            )
            acc_n[nm] += 1

            near_k = (near_hit.reshape(B * n) & k)
            if int(near_k.sum()) > 0:
                near_acc["hinge_static"] += float(hinge_s[near_k, j].mean())
                near_acc["hinge_dynamic"] += float(hinge_d[near_k, j].mean())
                near_acc["tau_dyn_rms"] += float(
                    torch.sqrt((tau_dyn[near_k, j] ** 2).mean())
                )
                near_acc["tau_g_rms"] += float(
                    torch.sqrt((tau_g[near_k, j] ** 2).mean())
                )
                near_n += 1

        for b in range(B):
            kind = int(kinds[b])
            km = KIND_NAMES.get(kind, f"kind{kind}")
            kb = knot_mask[b]
            if float(kb.sum()) < 1:
                continue
            kind_acc[km]["hinge_static"] += float(
                (hinge_s.reshape(B, n, D)[b] * kb).sum() / kb.sum()
            )
            kind_acc[km]["hinge_dynamic"] += float(
                (hinge_d.reshape(B, n, D)[b] * kb).sum() / kb.sum()
            )
            kind_acc[km]["tau_dyn_rms"] += float(
                torch.sqrt(((tau_dyn.reshape(B, n, D)[b] * kb) ** 2).sum() / kb.sum())
            )
            kind_acc[km]["tau_g_rms"] += float(
                torch.sqrt(((tau_g.reshape(B, n, D)[b] * kb) ** 2).sum() / kb.sum())
            )
            kind_n[km] += 1

    def _avg(d: dict, n: int) -> dict:
        return {k: v / max(n, 1) for k, v in d.items()}

    per_joint = []
    for nm in names:
        n = acc_n[nm]
        if n < 1:
            continue
        a = _avg(acc[nm], n)
        rating = float(isaac_effort_limits(names)[names.index(nm)])
        rec = {
            "joint": nm,
            "group": _joint_group(nm),
            "rating_nm": rating,
            "tau_g_rms": a["tau_g_rms"],
            "tau_dyn_rms": a["tau_dyn_rms"],
            "dyn_over_static_rms": a["tau_dyn_rms"] / max(a["tau_g_rms"], 1e-6),
            "qd_rms": a["qd_rms"],
            "qdd_rms": a["qdd_rms"],
            "hinge_static": a["hinge_static"],
            "hinge_dynamic": a["hinge_dynamic"],
            "hinge_dyn_over_static": a["hinge_dynamic"] / max(a["hinge_static"], 1e-9),
            "pct_over_static": 100.0 * a["frac_over_static"],
            "pct_over_dynamic": 100.0 * a["frac_over_dynamic"],
            "pct_dyn_exceeds_static": 100.0 * a["frac_dyn_over_static"],
            "dyn_minus_static_rms": a["dyn_minus_static_rms"],
            "p99_dyn_over_static": a["p99_ratio"],
        }
        per_joint.append(rec)

    groups = defaultdict(list)
    for r in per_joint:
        groups[r["group"]].append(r)

    def _mean_group(key: str, field: str) -> float:
        rows = groups.get(key, [])
        return float(np.mean([r[field] for r in rows])) if rows else float("nan")

    summary = {
        "pool": str(pool_npz),
        "ckpt": str(ckpt) if ckpt else None,
        "source": source,
        "bucket": int(bucket),
        "eta_torque": float(eta_torque),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "n_elements_per_joint": {nm: acc_n[nm] for nm in names if acc_n[nm]},
        "near_hit_wp": int(near_hit_wp),
        "near_hit": _avg(near_acc, near_n) if near_n else {},
        "by_kind": {
            k: _avg(dict(v), kind_n[k]) for k, v in sorted(kind_acc.items())
        },
        "group_means": {
            g: {
                "tau_g_rms": _mean_group(g, "tau_g_rms"),
                "tau_dyn_rms": _mean_group(g, "tau_dyn_rms"),
                "dyn_over_static_rms": _mean_group(g, "dyn_over_static_rms"),
                "hinge_static": _mean_group(g, "hinge_static"),
                "hinge_dynamic": _mean_group(g, "hinge_dynamic"),
                "hinge_dyn_over_static": _mean_group(g, "hinge_dyn_over_static"),
                "pct_over_static": _mean_group(g, "pct_over_static"),
                "pct_over_dynamic": _mean_group(g, "pct_over_dynamic"),
            }
            for g in ("shoulder", "elbow", "wrist", "waist")
        },
        "per_joint": per_joint,
    }
    return summary


def _print_table(summary: dict) -> None:
    print(
        f"\n[{summary['source']}] bucket={summary['bucket']} "
        f"eta_torque={summary['eta_torque']}"
    )
    print(
        f"{'group':<10}{'tau_g':>8}{'tau_dyn':>8}{'dyn/g':>7}"
        f"{'hinge_s':>9}{'hinge_d':>9}{'h_d/h_s':>8}"
        f"{'>lim_s%':>8}{'>lim_d%':>8}"
    )
    for g, r in summary["group_means"].items():
        if not np.isfinite(r["tau_g_rms"]):
            continue
        print(
            f"{g:<10}"
            f"{r['tau_g_rms']:8.2f}"
            f"{r['tau_dyn_rms']:8.2f}"
            f"{r['dyn_over_static_rms']:7.2f}"
            f"{r['hinge_static']:9.4f}"
            f"{r['hinge_dynamic']:9.4f}"
            f"{r['hinge_dyn_over_static']:8.2f}"
            f"{r['pct_over_static']:8.2f}"
            f"{r['pct_over_dynamic']:8.2f}"
        )
    if summary.get("near_hit"):
        nh = summary["near_hit"]
        print(
            f"\nnear-hit ±{summary['near_hit_wp']} wp: "
            f"tau_g={nh.get('tau_g_rms', float('nan')):.2f} "
            f"tau_dyn={nh.get('tau_dyn_rms', float('nan')):.2f} "
            f"hinge_s={nh.get('hinge_static', float('nan')):.4f} "
            f"hinge_d={nh.get('hinge_dynamic', float('nan')):.4f}"
        )
    if summary.get("by_kind"):
        print("\nby scenario kind (mean hinge):")
        for k, r in summary["by_kind"].items():
            hs = r.get("hinge_static", float("nan"))
            hd = r.get("hinge_dynamic", float("nan"))
            print(f"  {k:<10} hinge_s={hs:.4f} hinge_d={hd:.4f}  dyn/g={hd/max(hs,1e-9):.2f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pool",
        default="runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_"
        "jit0.05x0.2_rb0.25_tau1.0.npz",
    )
    p.add_argument("--ckpt", default="runs/phase1_rhythm_bellman_nf1_prsp/best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bucket", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", choices=("decoder", "init"), default="decoder")
    p.add_argument("--eta-torque", type=float, default=0.8)
    p.add_argument("--soft-wrist-effort-scale", type=float, default=1.0)
    p.add_argument("--near-hit-wp", type=int, default=5)
    p.add_argument("--json", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = compare(
        pool_npz=Path(args.pool),
        ckpt=Path(args.ckpt) if args.ckpt else None,
        device=args.device,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        bucket=args.bucket,
        seed=args.seed,
        source=args.source,
        eta_torque=args.eta_torque,
        soft_wrist_effort_scale=args.soft_wrist_effort_scale,
        near_hit_wp=args.near_hit_wp,
    )
    _print_table(summary)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
