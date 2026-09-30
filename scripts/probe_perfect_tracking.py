#!/usr/bin/env python
"""Compare the deployed PD probe with an otherwise identical ideal tracker."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.probe import closed_loop_probe  # noqa: E402
from amortized_opt.constants import ready_joint_pos  # noqa: E402


def _planned_summary(hits: list[dict]) -> dict:
    rows = [p for hit in hits for p in hit["planned"]]
    by_h: dict[int, list[dict]] = {}
    for row in rows:
        by_h.setdefault(int(row["h"]), []).append(row)

    def summarize(group: list[dict]) -> dict:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "in_disk": sum(bool(x["in_disk"]) for x in group),
            "axial_p50_mm": float(np.percentile([x["axial_mm"] for x in group], 50)),
            "inplane_p50_mm": float(np.percentile([x["inplane_mm"] for x in group], 50)),
        }

    return {
        "all": summarize(rows),
        "by_h": {str(h): summarize(group) for h, group in sorted(by_h.items())},
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", default=None,
                    help="optional isaac_rl_ppo actor checkpoint applied to --ckpt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--clips", type=int, default=1)
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--first-hit-s", type=float, default=None)
    ap.add_argument("--rest-start", action="store_true",
                    help="use ready_joint_pos with exactly zero qd")
    ap.add_argument("--momentum-grace-ticks", type=int, default=0)
    ap.add_argument("--freeze-wp", type=int, default=5)
    ap.add_argument("--strike-impedance", action="store_true")
    ap.add_argument("--imp-kp", type=float, default=0.5)
    ap.add_argument("--imp-kd", type=float, default=0.7)
    ap.add_argument("--imp-before-wp", type=int, default=3)
    ap.add_argument("--imp-after-wp", type=int, default=3)
    ap.add_argument("--terminal-qp-steps", type=int, default=0)
    ap.add_argument("--terminal-qp-window", type=int, default=16)
    ap.add_argument("--terminal-qp-damping", type=float, default=0.05)
    ap.add_argument("--full-cost-steps", type=int, default=0)
    ap.add_argument("--full-cost-lr", type=float, default=1.0e-3)
    ap.add_argument("--full-cost-grad-clip", type=float, default=1.0e4)
    ap.add_argument("--w-waist-yaw-monotonic", type=float, default=None)
    ap.add_argument("--w-waist-yaw-accel-direction", type=float, default=None)
    ap.add_argument("--w-waist-yaw-initial-accel-direction", type=float, default=None)
    ap.add_argument("--full-cost-from-init", action="store_true",
                    help="start direct trajectory descent from the analytic initializer")
    ap.add_argument("--only-pd", action="store_true")
    ap.add_argument("--record-waist-trace", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(model_cfg, device)
    model.load_state_dict(ck["model"])
    model.eval()

    if args.lora:
        from isaac_rl_ppo.actor import AmortizedPpoActor
        from isaac_rl_ppo.cfg import PpoRunConfig

        blob = torch.load(args.lora, map_location="cpu", weights_only=False)
        cfg = PpoRunConfig()
        saved_actor = (blob.get("cfg") or {}).get("actor") or {}
        cfg.actor.horizon = int((ck.get("scenario_cfg") or {}).get("horizon", 50))
        cfg.actor.lora_rank = int(saved_actor.get("lora_rank", 8))
        cfg.actor.lora_alpha = float(saved_actor.get("lora_alpha", 16.0))
        cfg.actor.lora_dropout = float(saved_actor.get("lora_dropout", 0.0))
        cfg.actor.lora_targets = tuple(
            saved_actor.get("lora_targets", cfg.actor.lora_targets)
        )
        rows = saved_actor.get("lora_output_rows")
        cfg.actor.lora_output_rows = (
            None if rows is None else tuple(int(value) for value in rows)
        )
        cfg.actor.freeze_wp = int(saved_actor.get("freeze_wp", args.freeze_wp))
        actor = AmortizedPpoActor(
            model, cfg.actor, cfg.critic,
            num_future=model_cfg.num_future, device=device,
        )
        actor.load_state_dict(blob["actor"], strict=False)
        model = actor.model

    scenario_cfg = ck.get("scenario_cfg") or {}
    train_cfg = ck.get("train_cfg") or {}
    kit = A.DrumKit.from_layout()
    pool = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    start_i = int(args.clip_index)
    clips = pool[start_i : start_i + int(args.clips)]
    if args.first_hit_s is not None:
        clips = [A.pad_first_hit_to(clip, float(args.first_hit_s))[0] for clip in clips]
    starts = None
    if args.rest_start:
        q = torch.as_tensor(
            ready_joint_pos(), device=device, dtype=torch.float64,
        ).view(1, -1).expand(len(clips), -1).clone()
        starts = SimpleNamespace(q=q, qd=torch.zeros_like(q))
    scorer = A.TrajectoryScorer(
        device="cpu",
        track_torque=False,
        soft_wrist_effort_scale=1.0,
        thresholds=A.AcceptanceThresholds(hit_margin_m=0.0),
    )
    full_cost = None
    full_cost_weights = None
    if int(args.full_cost_steps) > 0:
        cost_dict = dict(train_cfg["cost"])
        if args.w_waist_yaw_monotonic is not None:
            cost_dict["w_waist_yaw_monotonic"] = float(
                args.w_waist_yaw_monotonic
            )
        if args.w_waist_yaw_accel_direction is not None:
            cost_dict["w_waist_yaw_accel_direction"] = float(
                args.w_waist_yaw_accel_direction
            )
        if args.w_waist_yaw_initial_accel_direction is not None:
            cost_dict["w_waist_yaw_initial_accel_direction"] = float(
                args.w_waist_yaw_initial_accel_direction
            )
        cost_cfg = A.CostConfig(**cost_dict)
        full_cost = A.FusedCost(
            cost_cfg,
            max_horizon=int((scenario_cfg.get("buckets") or (25, 50))[-1]),
            device=device,
            dtype=torch.float32,
        )
        al = A.AugmentedLagrangian(A.ALConfig(**train_cfg["al"]))
        al.load_state_dict(ck["al"])
        full_cost_weights = al.cost_weights()
    common = dict(
        kit=kit,
        scorer=scorer,
        clips=clips,
        device=device,
        num_drums=model_cfg.num_drums,
        num_future=model_cfg.num_future,
        horizon=int((scenario_cfg.get("buckets") or (25, 50))[-1]),
        steps=args.steps,
        seed=0,
        amp_dtype=torch.bfloat16,
        lead_s=0.0,
        cmd_lead_s=train_cfg.get("cmd_lead_s", 0.0),
        freeze_wp=int(args.freeze_wp),
        bounce=True,
        waist_upright_ease_wp=scenario_cfg.get("waist_upright_ease_wp"),
        saturate_effort=bool(train_cfg.get("plant_saturate_effort", False)),
        record_drift=True,
        record_waist_trace=bool(args.record_waist_trace),
        waist_pr_ema=bool(train_cfg.get("waist_pr_ema", False)),
        waist_pr_ema_alpha=float(train_cfg.get("waist_pr_ema_alpha", 0.1)),
        waist_pr_ema_leak=float(train_cfg.get("waist_pr_ema_leak", 0.02)),
        momentum_grace_ticks=int(args.momentum_grace_ticks),
        strike_impedance=(
            A.StrikeImpedanceConfig(
                kp_scale=float(args.imp_kp),
                kd_scale=float(args.imp_kd),
                before_wp=int(args.imp_before_wp),
                after_wp=int(args.imp_after_wp),
            )
            if args.strike_impedance else None
        ),
        terminal_qp_steps=int(args.terminal_qp_steps),
        terminal_qp_window=int(args.terminal_qp_window),
        terminal_qp_damping=float(args.terminal_qp_damping),
        full_cost=full_cost,
        full_cost_weights=full_cost_weights,
        full_cost_steps=int(args.full_cost_steps),
        full_cost_lr=float(args.full_cost_lr),
        full_cost_grad_clip=float(args.full_cost_grad_clip),
        full_cost_from_init=bool(args.full_cost_from_init),
        starts=starts,
    )

    out = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_step": int(ck.get("step", -1)),
        "clips": [clip.name for clip in clips],
        "settings": {
            "steps": args.steps,
            "freeze_wp": common["freeze_wp"],
            "cmd_lead_s": common["cmd_lead_s"],
            "bounce": common["bounce"],
            "momentum_grace_ticks": int(args.momentum_grace_ticks),
            "strike_impedance": bool(args.strike_impedance),
            "imp_kp": float(args.imp_kp),
            "imp_kd": float(args.imp_kd),
            "imp_before_wp": int(args.imp_before_wp),
            "imp_after_wp": int(args.imp_after_wp),
            "terminal_qp_steps": int(args.terminal_qp_steps),
            "terminal_qp_window": int(args.terminal_qp_window),
            "terminal_qp_damping": float(args.terminal_qp_damping),
            "full_cost_steps": int(args.full_cost_steps),
            "full_cost_lr": float(args.full_cost_lr),
            "full_cost_grad_clip": float(args.full_cost_grad_clip),
            "full_cost_from_init": bool(args.full_cost_from_init),
            "w_waist_yaw_monotonic": args.w_waist_yaw_monotonic,
            "w_waist_yaw_accel_direction": args.w_waist_yaw_accel_direction,
            "w_waist_yaw_initial_accel_direction": (
                args.w_waist_yaw_initial_accel_direction
            ),
            "rest_start": bool(args.rest_start),
            "clip_index": int(args.clip_index),
            "first_hit_s": args.first_hit_s,
        },
        "runs": {},
    }
    modes = (("pd", False),) if args.only_pd else (
        ("pd", False), ("perfect_tracking", True)
    )
    for label, perfect in modes:
        result = closed_loop_probe(model, perfect_tracking=perfect, **common)
        hits = result.pop("_hits")
        result["planned"] = _planned_summary(hits)
        result["hits_detail"] = hits
        out["runs"][label] = result
        print(
            f"[{label}] hits={result['hits']} in_disk={result['in_disk']} "
            f"window={result['in_disk_window']} axial_p50={result['axial_p50_mm']:+.1f}mm "
            f"inplane_p50={result['inplane_p50_mm']:+.1f}mm "
            f"grace_cross={result.get('momentum_crossed', 0)} "
            f"diverged={result['diverged']} score={result['score']:.3f}",
            flush=True,
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(out), indent=2) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
