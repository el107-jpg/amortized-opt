#!/usr/bin/env python
"""Causal probe: does the decoder read the waist set-point token?

    python scripts/probe_waist_follow.py --ckpt runs/smoke_waist_follow_on/best.pt \\
        --device cuda:0

Loads a phase-1 checkpoint and a small val pool (or ``--pool``), substitutes
``sp+Δ`` into the yaw token, decodes with the ORIGINAL init, and reports
``corr(Δ, hold−aim)``.  A model trained with ``--waist-follow-frac 0`` should
score near 0; follow-on should track (corr ≥ 0.6 after a short train).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A
from amortized_opt.train_warmstart import waist_follow_probe
from amortized_opt.warmstart_data import NoiseConfig, ScenarioConfig, generate_pool


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--scenarios", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--horizon", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    ckpt = A.load_checkpoint(args.ckpt, map_location=dev)
    model_cfg = A.model_config_from_checkpoint(ckpt)
    model = A.build_model(model_cfg, dev)
    A.load_checkpoint(args.ckpt, model=model, map_location=dev)
    model.eval()

    H = int(args.horizon)
    buckets = (25, 50, 100) if H >= 100 else tuple(b for b in (25, 50, 100) if b <= H)
    if not buckets or buckets[-1] != H:
        buckets = tuple(b for b in buckets if b < H) + (H,)
    scen = ScenarioConfig(horizon=H, buckets=buckets)
    fk = A.ChainFK(device=str(dev), dtype=torch.float32)
    rnea = A.BatchedRNEA.from_urdf(with_armature=True, device=str(dev), dtype=torch.float32)
    pool = generate_pool(
        args.scenarios, seed=int(args.seed) + 7, cfg=scen, noise=NoiseConfig(),
        split="eval", workers=int(args.workers), device=str(dev), fk=fk, rnea=rnea,
    )
    rep = waist_follow_probe(
        model, pool, batch_size=int(args.batch), n_batches=int(args.batches),
        seed=int(args.seed),
    )
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
