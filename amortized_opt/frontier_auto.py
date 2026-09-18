"""Shared logic for auto-enabling frontier collocation from probe + train metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

DEFAULT_VIOL_KEYS: tuple[str, ...] = ("torque_peak", "collision", "joint_box_peak")


def probe_miss_mm(probe: dict[str, Any] | None) -> float | None:
    """Closed-loop axial miss in mm (absolute p50 preferred)."""
    if not probe or probe.get("error"):
        return None
    if "axial_abs_p50_mm" in probe:
        v = probe["axial_abs_p50_mm"]
        if v is not None and math.isfinite(float(v)):
            return float(v)
    if "axial_p50_mm" in probe:
        v = probe["axial_p50_mm"]
        if v is not None and math.isfinite(float(v)):
            return abs(float(v))
    return None


def violations_spiking(
    train_rows: list[dict[str, Any]],
    *,
    keys: Iterable[str] = DEFAULT_VIOL_KEYS,
    cap: float = 8.0,
    ratio: float = 3.0,
    min_rows: int = 3,
) -> bool:
    """True when any monitored ``v_*`` is above cap or ratio × prior median."""
    if len(train_rows) < min_rows:
        return False
    latest = train_rows[-1]
    prior = train_rows[:-1]
    for key in keys:
        vk = f"v_{key}"
        cur = latest.get(vk)
        if cur is None or not math.isfinite(float(cur)):
            continue
        cur = float(cur)
        hist = [
            float(r[vk]) for r in prior
            if vk in r and r[vk] is not None and math.isfinite(float(r[vk]))
        ]
        if not hist:
            if cur > cap:
                return True
            continue
        med = float(sorted(hist)[len(hist) // 2])
        thresh = max(cap, med * ratio)
        if cur > thresh:
            return True
    return False


def frontier_ready(
    probe: dict[str, Any] | None,
    train_rows: list[dict[str, Any]],
    *,
    miss_mm: float = 30.0,
    viol_keys: Iterable[str] = DEFAULT_VIOL_KEYS,
    viol_cap: float = 8.0,
    viol_ratio: float = 3.0,
    viol_window: int = 10,
) -> tuple[bool, str]:
    """Whether probe miss and train violations satisfy auto-frontier gates."""
    miss = probe_miss_mm(probe)
    if miss is None:
        return False, "no_probe"
    if miss > miss_mm:
        return False, f"miss={miss:.1f}mm>{miss_mm:.0f}"
    window = train_rows[-viol_window:] if viol_window > 0 else train_rows
    if violations_spiking(
        window,
        keys=viol_keys,
        cap=viol_cap,
        ratio=viol_ratio,
    ):
        return False, "violation_spike"
    return True, f"miss={miss:.1f}mm"


def load_metrics_tail(
    path: Path | str,
    *,
    train_window: int = 10,
) -> dict[str, Any]:
    """Scan metrics.jsonl once; return latest probe and recent train rows."""
    path = Path(path)
    latest_probe: dict[str, Any] | None = None
    train_rows: list[dict[str, Any]] = []
    frontier_on = False
    if not path.is_file():
        return {
            "path": str(path),
            "latest_probe": None,
            "train_rows": [],
            "frontier_on": False,
            "last_step": 0,
        }
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("event")
            if ev == "probe":
                latest_probe = rec
            elif ev == "train":
                train_rows.append(rec)
                if rec.get("frontier") or rec.get("frontier_landmarks"):
                    frontier_on = True
            elif ev == "frontier_auto":
                frontier_on = True
            elif ev == "start" and rec.get("cfg", {}).get("frontier"):
                frontier_on = True
    if train_window > 0 and len(train_rows) > train_window:
        train_rows = train_rows[-train_window:]
    last_step = 0
    if train_rows:
        last_step = int(train_rows[-1].get("step", 0))
    elif latest_probe:
        last_step = int(latest_probe.get("step", 0))
    return {
        "path": str(path),
        "latest_probe": latest_probe,
        "train_rows": train_rows,
        "frontier_on": frontier_on,
        "last_step": last_step,
    }


def evaluate_run(
    run_dir: Path | str,
    *,
    miss_mm: float = 30.0,
    viol_cap: float = 8.0,
    viol_ratio: float = 3.0,
    viol_window: int = 10,
) -> dict[str, Any]:
    """Status dict for one run directory."""
    run_dir = Path(run_dir)
    snap = load_metrics_tail(run_dir / "metrics.jsonl", train_window=viol_window)
    done = (run_dir / ".frontier_monitor.done").is_file()
    flagged = (run_dir / ".frontier_activate").is_file()
    probe = snap["latest_probe"]
    miss = probe_miss_mm(probe) if probe else None
    ready, reason = frontier_ready(
        probe,
        snap["train_rows"],
        miss_mm=miss_mm,
        viol_cap=viol_cap,
        viol_ratio=viol_ratio,
        viol_window=viol_window,
    )
    return {
        "run_dir": str(run_dir),
        "last_step": snap["last_step"],
        "frontier_on": bool(snap["frontier_on"]),
        "done": done,
        "flagged": flagged,
        "miss_mm": miss,
        "ready": ready and not snap["frontier_on"] and not done,
        "reason": reason,
        "probe_step": int(probe.get("step", 0)) if probe else None,
    }
