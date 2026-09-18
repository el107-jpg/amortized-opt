#!/usr/bin/env python
"""Open-loop waist P/R band probe on decoder waypoint trajectories.

Scores the model's *generated* plans (not MPC plant state): every valid
waypoint knot in ``dec.traj``.  Reports what :meth:`FusedCost._waist_pr_band`
sees during training (often zero because ``waist_pr_ref`` is unset) versus
hypothetical band excess around ref=0 or ref=q0 P/R at encode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool, Phase2ScenarioConfig  # noqa: E402
from amortized_opt.warmstart_data import CONTEXT_FIELDS  # noqa: E402
from amortized_opt.train_cost import encode, phase2_cost_config  # noqa: E402


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


def _band_excess(q_pr, ref, band: float, mask) -> torch.Tensor:
    """Per-batch peak ReLU(|q-ref|-band) over waypoints × (pitch, roll)."""
    excess = (q_pr - ref).abs() - float(band)
    over = excess.clamp_min(0.0) * mask
    return over.amax(dim=(1, 2))


@torch.no_grad()
def probe(
    ckpt: Path,
    pool_npz: Path,
    *,
    device: str,
    n_batches: int,
    batch_size: int,
    band_rad: float,
    seed: int,
) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck, num_future=1)
    dev = torch.device(device)
    model = A.build_model(mcfg, dev)
    sd = dict(ck["model"])
    key = "encoder.future_rank_emb.weight"
    if key in sd and int(mcfg.num_future) < sd[key].shape[0]:
        sd[key] = sd[key][: int(mcfg.num_future)]
    model.load_state_dict(sd, strict=False)
    model.eval()

    buckets, stats = _load_npz_buckets(pool_npz)
    pool = Phase2Pool(
        buckets,
        cfg=_scenario_from_ckpt(ck, tuple(sorted(buckets))),
        noise=Phase2Noise(),
        device=dev,
        stats=stats,
    )
    cost = A.FusedCost(
        phase2_cost_config(w_waist_pr_band=1.0, waist_pr_band_rad=band_rad),
        max_horizon=max(buckets),
        device=dev,
        dtype=torch.float32,
        dt=float(getattr(pool.cfg, "dt", 0.02)),
    )
    pitch_i = cost.joint_names.index("waist_pitch_joint")
    roll_i = cost.joint_names.index("waist_roll_joint")
    pr_idx = torch.tensor([pitch_i, roll_i], device=dev)

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))

    buckets = [b for b in pool.buckets if b >= 50]
    if not buckets:
        buckets = [max(pool.buckets)]
    rec = {
        "ckpt": str(ckpt),
        "step": int(ck.get("step", -1)),
        "band_rad": float(band_rad),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "waist_pr_ref_active_in_training": False,
        "cost_v_waist_pr_band": [],
        "peak_vs_zero": [],
        "peak_vs_q0": [],
        "any_vs_zero": [],
        "any_vs_q0": [],
        "yaw_vel_rms": [],
    }

    for bi in range(n_batches):
        bucket = int(buckets[bi % len(buckets)])
        idx = pool.sample_idx(bucket, batch_size, generator=gen)
        batch = pool.batch(bucket, idx, generator=gen)
        enc = encode(model, batch["context"])
        masks = batch["masks"]
        dec = model(batch["init"], batch["state"], enc, masks)
        wp = dec.traj
        B, H, _ = wp.shape
        m2 = masks.elem_mask[:, :H].index_select(2, pr_idx)
        q_pr = wp[:, :H].index_select(2, pr_idx)
        q0_pr = batch["q0"][:, pr_idx].unsqueeze(1).expand(-1, H, -1)
        ct = cost.forward(
            batch["q0"], wp, batch["v0"], batch["strike"],
            elem_mask=masks.elem_mask[:, :H],
            weights={},
        )
        v_band = ct.violations["waist_pr_band"].detach().cpu().numpy()
        peak0 = _band_excess(q_pr, torch.zeros_like(q_pr), band_rad, m2)
        peak_q0 = _band_excess(q_pr, q0_pr, band_rad, m2)
        yaw_i = cost.joint_names.index("waist_yaw_joint")
        qd = dec.vel[:, :H, yaw_i]
        vm = masks.elem_mask[:, :H, yaw_i]
        yaw_rms = torch.sqrt((qd.pow(2) * vm).sum() / vm.sum().clamp_min(1.0))

        rec["cost_v_waist_pr_band"].extend(v_band.tolist())
        rec["peak_vs_zero"].extend(peak0.detach().cpu().tolist())
        rec["peak_vs_q0"].extend(peak_q0.detach().cpu().tolist())
        rec["any_vs_zero"].extend((peak0 > 0).detach().cpu().tolist())
        rec["any_vs_q0"].extend((peak_q0 > 0).detach().cpu().tolist())
        rec["yaw_vel_rms"].append(float(yaw_rms.item()))

    def _summ(arr, name: str, out: dict) -> None:
        a = np.asarray(arr, float)
        if a.size == 0:
            return
        out[f"{name}_mean"] = float(a.mean())
        out[f"{name}_p50"] = float(np.percentile(a, 50))
        out[f"{name}_p90"] = float(np.percentile(a, 90))
        out[f"{name}_max"] = float(a.max())
        if name.startswith("frac"):
            out[name] = float(a.mean())

    summary: dict = {}
    _summ(rec["cost_v_waist_pr_band"], "cost_v_waist_pr_band", summary)
    _summ(rec["peak_vs_zero"], "peak_vs_zero_rad", summary)
    _summ(rec["peak_vs_q0"], "peak_vs_q0_rad", summary)
    _summ(rec["any_vs_zero"], "frac_batches_vs_zero", summary)
    _summ(rec["any_vs_q0"], "frac_batches_vs_q0", summary)
    _summ(rec["yaw_vel_rms"], "yaw_vel_rms", summary)
    summary["n_samples"] = len(rec["peak_vs_zero"])
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/phase1_rhythm_bellman_nf1_prsp/best.pt")
    p.add_argument(
        "--pool",
        default="runs/p2pool_rhythm_bellman_nf1_frontier_fromstart_mix306010_"
        "evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--band-rad", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    summary = probe(
        Path(args.ckpt), Path(args.pool),
        device=args.device,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        band_rad=args.band_rad,
        seed=args.seed,
    )
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
