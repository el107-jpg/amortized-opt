#!/usr/bin/env python
"""Collect exact fixed-actor labels for the minimal timing-critic features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from amortized_opt.train_cost import phase2_config_from_checkpoint, train_phase2
from amortized_opt.violation_critic import ViolationCriticConfig
from scripts.run_timing_critic_trial import (
    configure_run,
    load_model,
    load_pool,
    scenario_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--pool-cache", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=16_384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    scenario_cfg = scenario_from_checkpoint(source)
    pool = load_pool(Path(args.pool_cache), scenario_cfg, args.device)
    model, model_cfg = load_model(source, args.device)
    start_step = int(source["step"])
    cfg = phase2_config_from_checkpoint(source, device=args.device)
    # GPU checkpoints commonly carry ``compile_model=True``.  Recompiling the
    # actor for a short CPU-only label collection run costs far more than the
    # evaluation itself and previously made even a one-step smoke test appear
    # to hang.  Label collection does not benefit from compiled training
    # graphs, so keep all components eager on CPU.
    if device.type == "cpu":
        cfg.compile_model = False
        cfg.compile_cost = False
        cfg.compile_aux = False
    # Freeze the actor while retaining the exact ordinary + suffix evaluation
    # path.  The scalar critic is only a vehicle for its bounded labeled ring.
    cfg.lr = 0.0
    cfg.grad_term_every = 0
    cfg.timing_critic = ViolationCriticConfig(
        mode="shadow", num_heads=1, hidden_dim=32, depth=1,
        dropout_probability=0.1, mc_dropout_samples=2,
        batch_size=128, updates_per_step=1,
        recent_capacity=int(args.capacity), audit_capacity=0,
    )
    configure_run(
        cfg, max_steps=start_step + int(args.steps),
        batch=int(args.batch), log_every=max(10, int(args.steps) // 5),
    )
    run_dir = Path(args.run_dir)
    summary = train_phase2(
        model=model, pool=pool, cfg=cfg, model_cfg=model_cfg,
        run_dir=run_dir, scenario_cfg=scenario_cfg,
        al_state=source["al"], opt_state=None,
        timing_critic_state=None, start_step=start_step,
    )
    final = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    state = final["extra"]["timing_critic"]
    report = {
        "source": args.source,
        "source_step": start_step,
        "final_step": int(summary["steps"]),
        "rows": int(state["recent"]["size"]),
        "input_dim": int(state["input_dim"]),
        "critic_format_version": int(state["format_version"]),
    }
    (run_dir / "minimal_critic_dataset.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("MINIMAL_CRITIC_DATASET=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
