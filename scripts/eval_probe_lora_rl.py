#!/usr/bin/env python3
"""Evaluate a saved analytic-probe LoRA PPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.lead import resolve_lead  # noqa: E402
from amortized_opt.probe import closed_loop_probe  # noqa: E402
from isaac_rl_ppo.actor import AmortizedPpoActor  # noqa: E402
from isaac_rl_ppo.cfg import PpoRunConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument("--perfect-tracking", action="store_true")
    p.add_argument("--lead-scale", type=float, default=1.0)
    p.add_argument("--ki", type=float, default=0.0)
    p.add_argument("--ki-windup", type=float, default=0.0)
    p.add_argument("--pid-with-lead", action="store_true")
    p.add_argument("--record-waist-trace", action="store_true")
    args = p.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    # Match training and deployed PPO numerics.  Closed-loop replanning is
    # sensitive enough that TF32-vs-highest matmul drift changes which head
    # crossings occur even with identical weights.
    torch.set_float32_matmul_precision("high")
    ck = torch.load(args.backbone, map_location="cpu", weights_only=False)
    blob = torch.load(args.lora, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    scenario_cfg = ck.get("scenario_cfg") or {}
    train_cfg = ck.get("train_cfg") or {}
    cfg = PpoRunConfig()
    saved = blob.get("cfg") or {}
    actor_saved = saved.get("actor") or {}
    cfg.actor.horizon = int(scenario_cfg.get("horizon", 50))
    cfg.actor.lora_rank = int(actor_saved.get("lora_rank", 8))
    cfg.actor.lora_alpha = float(actor_saved.get("lora_alpha", 16.0))
    cfg.actor.lora_dropout = float(actor_saved.get("lora_dropout", 0.0))
    cfg.actor.lora_targets = tuple(
        actor_saved.get("lora_targets", cfg.actor.lora_targets)
    )
    rows = actor_saved.get("lora_output_rows")
    cfg.actor.lora_output_rows = (
        None if rows is None else tuple(int(value) for value in rows)
    )
    cfg.actor.freeze_wp = int(actor_saved.get("freeze_wp", 5))
    actor = AmortizedPpoActor(
        model, cfg.actor, cfg.critic, num_future=mcfg.num_future, device=dev,
    )
    missing, unexpected = actor.load_state_dict(blob["actor"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected LoRA checkpoint keys: {unexpected}")

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    scorer = A.TrajectoryScorer(
        device="cpu", track_torque=False,
        thresholds=A.AcceptanceThresholds(hit_margin_m=0.0),
    )
    resolved_lead = resolve_lead(train_cfg.get("cmd_lead_s", 0.0))
    if not isinstance(resolved_lead, (int, float)):
        resolved_lead = tuple(float(v) * float(args.lead_scale) for v in resolved_lead)
    else:
        resolved_lead = float(resolved_lead) * float(args.lead_scale)
    result = closed_loop_probe(
        actor.model, kit=kit, scorer=scorer, clips=clips, device=dev,
        num_drums=mcfg.num_drums, num_future=mcfg.num_future,
        horizon=cfg.actor.horizon, steps=250, seed=0,
        amp_dtype=torch.bfloat16, lead_s=0.0,
        cmd_lead_s=(
            0.0 if args.perfect_tracking
            else resolved_lead
        ),
        freeze_wp=cfg.actor.freeze_wp, bounce=True, record_drift=True,
        waist_upright_ease_wp=scenario_cfg.get("waist_upright_ease_wp"),
        perfect_tracking=bool(args.perfect_tracking),
        ki=float(args.ki), ki_windup=float(args.ki_windup),
        pid_with_lead=bool(args.pid_with_lead),
        record_waist_trace=bool(args.record_waist_trace),
    )
    result.pop("_hits", None)
    out = {
        "backbone": str(Path(args.backbone).resolve()),
        "lora": str(Path(args.lora).resolve()),
        "lora_iteration": int(blob.get("iteration", -1)),
        "perfect_tracking": bool(args.perfect_tracking),
        "lead_scale": float(args.lead_scale),
        "ki": float(args.ki),
        "ki_windup": float(args.ki_windup),
        "pid_with_lead": bool(args.pid_with_lead),
        "missing_keys": list(missing),
        "probe": result,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps({
        "iteration": out["lora_iteration"],
        "score": result["score"],
        "in_disk": result["in_disk"],
        "crossed": result["latency"]["crossed_in_disk"],
        "axial_abs_p50_mm": result["axial_abs_p50_mm"],
        "pd_hit_tip_track_mean_mm": result["episodes"][0]["pd_hit_tip_track_mean_mm"],
    }, indent=2))


if __name__ == "__main__":
    main()
