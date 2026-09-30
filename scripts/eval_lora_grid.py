#!/usr/bin/env python3
"""Merge decoder LoRA into a phase-2 backbone and run the 81-case grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.train_cost import eval_subset, rolling_eval  # noqa: E402


def merge_lora(state, actor, scaling, output_rows=None):
    count = 0
    for key, a in actor.items():
        if not key.endswith(".lora_a"):
            continue
        prefix = key[: -len("lora_a")]
        target = prefix[len("model."):] if prefix.startswith("model.") else prefix
        weight_key = target + "weight"
        b = actor[prefix + "lora_b"].float()
        if output_rows is not None and target.endswith("head_traj."):
            masked = torch.zeros_like(b)
            masked[list(output_rows)] = b[list(output_rows)]
            b = masked
        state[weight_key] = (
            state[weight_key] + float(scaling) * (b @ a.float()).to(state[weight_key])
        )
        count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    ck = torch.load(args.backbone, map_location="cpu", weights_only=False)
    blob = torch.load(args.lora, map_location="cpu", weights_only=False)
    saved_actor = (blob.get("cfg") or {}).get("actor") or {}
    scaling = float(saved_actor.get("lora_alpha", 16.0)) / float(
        saved_actor.get("lora_rank", 8)
    )
    state = dict(ck["model"])
    rows = saved_actor.get("lora_output_rows")
    output_rows = None if rows is None else tuple(int(value) for value in rows)
    merged = merge_lora(state, blob["actor"], scaling, output_rows)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(state)
    model.eval()

    scorer = A.TrajectoryScorer(device="cpu", track_torque=False)
    cases = eval_subset(
        A.generate_grid(
            seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
            start_above_drums=("snare", "floor_tom"),
            start_not_under_drums=True, start_clearance_m=0.02,
        ),
        81,
    )
    kit, library = A.DrumKit.from_layout(), A.GoalPoseLibrary()
    scenario = ck.get("scenario_cfg") or {}
    result = rolling_eval(
        model, cases, scorer, device=dev, kit=kit, library=library,
        buckets=tuple(scenario.get("buckets") or (25, 50)),
        iterations=1, amp_dtype=torch.bfloat16,
        num_drums=mcfg.num_drums, num_future=mcfg.num_future,
        init="quintic",
    )
    result.update({
        "merged_lora_pairs": merged,
        "lora_iteration": int(blob.get("iteration", -1)),
    })
    Path(args.out).write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
