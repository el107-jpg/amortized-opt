#!/usr/bin/env python
"""Build one compact, reusable cache of exact actor-labeled critic rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.ablate_monotonic_confidence_critic import chronological_rows


CACHE_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=CHECKPOINT",
        help="labeled collector checkpoint; repeat to add datasets",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--dataset must have the form NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("dataset name and checkpoint path must be nonempty")
    return name, Path(path)


def main() -> None:
    args = parse_args()
    datasets: dict[str, dict] = {}
    for value in args.dataset:
        name, path = parse_dataset(value)
        if name in datasets:
            raise ValueError(f"duplicate dataset name {name!r}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        features, targets, steps, config = chronological_rows(checkpoint)
        datasets[name] = {
            "features": features.to(torch.float16).contiguous(),
            "targets": targets.to(torch.float16).contiguous(),
            "actor_step": steps.to(torch.int64).contiguous(),
            "source_config": config,
            "source_checkpoint": str(path),
            "source_actor_step": int(checkpoint.get("step", -1)),
        }

    output = {
        "format": "monotonic_critic_actor_labels",
        "format_version": CACHE_FORMAT_VERSION,
        "datasets": datasets,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    summary = {
        "output": str(destination),
        "datasets": {name: int(row["features"].shape[0]) for name, row in datasets.items()},
        "rows": sum(int(row["features"].shape[0]) for row in datasets.values()),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("MONOTONIC_CRITIC_DATA_CACHE=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
