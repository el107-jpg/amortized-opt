#!/usr/bin/env python
"""Phase-1 warm start: generate the scenario pools and run the BC training loop.

    python scripts/train_phase1.py --device cuda:1 --steps 20000 --batch 1024 \
        --train-scenarios 500000 --val-scenarios 40000

Everything lands in ``amortized/runs/phase1_<timestamp>/``:
``metrics.jsonl`` (one record per log / val / event), ``best.pt`` and
``last.pt`` (model + optimizer + configs, so section 7 can resume), and
``summary.json``.

Resolved decision 5: 20 000 steps is a HARD cap and the accuracy numbers are
diagnostics, not gates.  Resolved decision 6: bf16 autocast, batch >= 512,
``torch.compile`` over forward+losses, horizon bucketing.
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
from amortized_opt.train_warmstart import (  # noqa: E402
    LossWeights,
    TrainConfig,
    imitation_report,
    train,
)
from amortized_opt.warmstart_data import (  # noqa: E402
    NoiseConfig,
    ScenarioConfig,
    generate_pool,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=20_000, help="HARD cap (plan decision 5)")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--train-scenarios", type=int, default=500_000)
    p.add_argument("--val-scenarios", type=int, default=40_000)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--buckets", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=None, help="decoder width (default 384)")
    p.add_argument("--dec-layers", type=int, default=None, help="decoder depth (default 3)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-kit-randomization", action="store_true")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out", default=None, help="run directory (default runs/phase1_<ts>)")
    p.add_argument("--pool-device", default=None,
                   help="where the scenario pool lives (default: --device)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.out) if args.out else (
        root / "runs" / f"phase1_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}", flush=True)

    scen = ScenarioConfig(
        horizon=int(args.horizon),
        buckets=tuple(args.buckets),
        randomize_kit=not args.no_kit_randomization,
    )
    noise = NoiseConfig()
    pool_dev = args.pool_device or args.device

    # Shared FK / RNEA for the auxiliary-label pass (building the RNEA needs
    # Pinocchio and takes a moment; both pools reuse the same instance).
    fk = A.ChainFK(device=pool_dev, dtype=torch.float32)
    rnea = A.BatchedRNEA.from_urdf(with_armature=True, device=pool_dev, dtype=torch.float32)

    t0 = time.perf_counter()
    print(f"[data] generating {args.train_scenarios} train scenarios "
          f"({args.workers} workers) ...", flush=True)
    train_pool = generate_pool(
        args.train_scenarios, seed=args.seed, cfg=scen, noise=noise, split="train",
        workers=args.workers, device=pool_dev, fk=fk, rnea=rnea,
    )
    # Val is held out TWICE over: the four eval RCC clips, and a disjoint start
    # seed stream (the start sampler is seeded off this seed alone).
    print(f"[data] generating {args.val_scenarios} val scenarios ...", flush=True)
    val_pool = generate_pool(
        args.val_scenarios, seed=args.seed + 991_1993, cfg=scen, noise=noise, split="eval",
        workers=max(args.workers // 4, 1), device=pool_dev, fk=fk, rnea=rnea,
    )
    t_data = time.perf_counter() - t0
    for name, pool in (("train", train_pool), ("val", val_pool)):
        s = pool.stats
        pr = np.asarray(s.get("peak_rate_ratio", []), dtype=float)
        print(
            f"[data] {name}: {len(pool)} samples, buckets {pool.distribution()}; "
            f"accepted {s['accepted']}/{s['requested']} "
            f"(drop {100 * (1 - s['accepted'] / max(s['requested'], 1)):.1f}%: "
            f"|qd| {s['reject_velocity']}, idle {s['reject_idle']}); "
            f"dual-arm {s['dual_active']} (asym {s['asymmetric']}); "
            f"peak|qd|/limit p50={np.percentile(pr, 50):.2f} p99={np.percentile(pr, 99):.2f}",
            flush=True,
        )
    print(f"[data] {t_data:.1f} s", flush=True)

    model_cfg = A.ModelConfig(
        max_horizon=int(args.horizon), num_drums=scen.num_drums, num_future=scen.num_future
    )
    for attr in ("d_model", "dec_layers"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(model_cfg, attr, val)
    model = A.build_model(model_cfg, dev)
    print(f"[model] {A.param_counts(model)}", flush=True)

    cfg = TrainConfig(
        max_steps=int(args.steps),
        batch_size=int(args.batch),
        lr=float(args.lr),
        warmup_steps=int(args.warmup),
        amp_dtype="float32" if args.no_amp else "bfloat16",
        compile_mode=None if args.no_compile else "default",
        eval_every=int(args.eval_every),
        patience_evals=int(args.patience),
        seed=int(args.seed),
        device=str(args.device),
    )
    weights = LossWeights(aux=float(args.aux_weight))

    summary = train(
        model=model, train_pool=train_pool, val_pool=val_pool, cfg=cfg,
        model_cfg=model_cfg, weights=weights, run_dir=run_dir,
        scenario_cfg=scen, noise_cfg=noise,
    )
    summary["data_gen_s"] = t_data
    summary["train_pool"] = train_pool.distribution()
    summary["val_pool"] = val_pool.distribution()
    summary["train_stats"] = {
        k: (float(np.percentile(v, 99)) if isinstance(v, np.ndarray) else v)
        for k, v in train_pool.stats.items()
    }

    # ---- diagnostics on the held-out pool, from the BEST checkpoint ----------
    best = run_dir / "best.pt"
    if best.exists():
        A.load_checkpoint(best, model=model, map_location=dev)
    amp = None if args.no_amp else torch.bfloat16
    for tag, hold in (("quintic_init", False), ("hold_init", True)):
        rep = imitation_report(
            model, val_pool, batch_size=min(512, cfg.batch_size), n_batches=16,
            seed=1234, amp_dtype=amp, hold_init=hold,
        )
        summary[f"imitation_{tag}"] = rep
        print(f"[val:{tag}] " + json.dumps(
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rep.items()}
        ), flush=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] {summary['steps']} steps in {summary['wall_s']:.0f} s "
          f"({summary['steps_per_s']:.2f} it/s), best val {summary['best_val']:.4f} "
          f"@ step {summary['best_step']} ({summary['stopped']})", flush=True)
    print(f"[done] buckets used: {summary['bucket_counts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
