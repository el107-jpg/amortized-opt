#!/usr/bin/env python3
"""Read-only forensic profiler for phase-2 metrics.jsonl files.

The script writes only beneath its own analysis directory. It intentionally
uses no project imports so it can inspect logs even when the training
environment is unavailable.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_RUNS = (
    "runs/rhythm_bellman_nf1_ls_full_rnea_fresh",
    "runs/rhythm_bellman_nf1_ls_full_rnea_replay_dw",
    "runs/rhythm_bellman_nf1_ls_full_rnea_replay_dw_frontier",
)


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric(child, name))
    return out


def _json_lines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line {line_no}: non-object JSON")
                continue
            obj["_line"] = line_no
            records.append(obj)
    return records, errors


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = _finite(values)
    if not arr.size:
        return {"n": 0, "min": None, "p10": None, "p50": None,
                "p90": None, "p99": None, "max": None}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _safe_ratio(num: float, den: float) -> float:
    return num / den if math.isfinite(num) and math.isfinite(den) and abs(den) > 1e-30 else math.nan


def _rolling_context(rows: list[dict[str, Any]], i: int, width: int = 10) -> dict[str, float]:
    loss = np.asarray([float(row.get("loss", math.nan)) for row in rows], dtype=np.float64)
    before = _finite(loss[max(0, i - width):i])
    after = _finite(loss[i + 1:min(len(rows), i + 1 + width)])
    current = float(loss[i])
    before_med = float(np.median(before)) if before.size else math.nan
    after_med = float(np.median(after)) if after.size else math.nan
    return {
        "loss": current,
        "before_median": before_med,
        "after_median": after_med,
        "ratio_to_before": _safe_ratio(current, before_med),
        "after_to_before": _safe_ratio(after_med, before_med),
    }


def _rank_spikes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        ctx = _rolling_context(rows, i, width=10)
        if i < 3 or not math.isfinite(ctx["ratio_to_before"]):
            continue
        ranked.append({
            "index": i,
            "step": row.get("step"),
            "line": row.get("_line"),
            **ctx,
        })
    ranked.sort(
        key=lambda item: (
            max(item["ratio_to_before"], item["after_to_before"]),
            item["loss"],
        ),
        reverse=True,
    )
    return ranked[:40]


def _series(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    flattened = [_flatten_numeric(row) for row in rows]
    keys = sorted(set().union(*(row.keys() for row in flattened)))
    return {
        key: np.asarray([row.get(key, math.nan) for row in flattened], dtype=np.float64)
        for key in keys
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return math.nan, n
    xx, yy = x[mask], y[mask]
    if float(np.std(xx)) <= 0.0 or float(np.std(yy)) <= 0.0:
        return math.nan, n
    return float(np.corrcoef(xx, yy)[0, 1]), n


def _lead_lag_correlations(
    series: dict[str, np.ndarray],
    target: str = "loss",
    max_lag: int = 10,
) -> list[dict[str, Any]]:
    y = series.get(target)
    if y is None:
        return []
    out: list[dict[str, Any]] = []
    excluded = {
        "_line", "step", "elapsed_s", "steps_per_s", "loss",
        "loss_cost", "cost", "ls_u",
    }
    for key, x in series.items():
        if key in excluded:
            continue
        best: tuple[float, int, float, int] | None = None
        for lag in range(-max_lag, max_lag + 1):
            # lag < 0 means metric leads loss; lag > 0 means metric follows loss.
            if lag < 0:
                corr, n = _pearson(x[:lag], y[-lag:])
            elif lag > 0:
                corr, n = _pearson(x[lag:], y[:-lag])
            else:
                corr, n = _pearson(x, y)
            if not math.isfinite(corr):
                continue
            candidate = (abs(corr), lag, corr, n)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is not None:
            out.append({
                "metric": key,
                "best_abs_corr": best[0],
                "lag_rows": best[1],
                "corr": best[2],
                "n": best[3],
            })
    out.sort(key=lambda item: item["best_abs_corr"], reverse=True)
    return out


def _nonfinite_inventory(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for rec in records:
        for key, value in _flatten_numeric(rec).items():
            if not math.isfinite(value):
                counts[key] += 1
    return dict(counts.most_common())


def _row_extract(row: dict[str, Any]) -> dict[str, Any]:
    flat = _flatten_numeric(row)
    preferred = (
        "step", "loss", "cost", "loss_cost", "loss_tc", "loss_bptt",
        "grad_norm", "lr", "torque", "torque_peak", "terminal", "collision",
        "joint_box", "joint_box_peak", "miss_m", "min_sdf_m", "mpc_init_tie",
        "sfx_cost", "sfx_tie", "replay_frac", "train_B",
        "roller.replay_viol_score", "roller.replay_viol_n",
        "ls_epsilon", "ls_searched", "ls_accepted", "ls_trial_count",
        "ls_F0", "ls_Fprobe", "ls_decrease", "ls_a", "ls_P2",
        "ls_R", "ls_kappa", "ls_K", "ls_floor_commit",
        "ls_u_total_norm", "ls_u_fresh_norm", "ls_u_mom_norm",
        "ls_u_fresh_frac", "ls_u_mom_frac", "ls_u_mom_over_fresh",
        "ls_u_fresh_frac_mean", "ls_u_mom_frac_mean",
        "ls_u_mom_over_fresh_mean",
    )
    result = {"line": row.get("_line")}
    for key in preferred:
        if key in flat:
            result[key] = flat[key]
    for key in sorted(flat):
        if (
            key.startswith("viol_")
            or key.startswith("grad_")
            or key.startswith("lam_i.")
            or key.startswith("pen.")
            or key.startswith("batch_short.")
            or key.startswith("train_bucket_frac.")
        ):
            result[key] = flat[key]
    if "ls_failure" in row:
        result["ls_failure"] = row["ls_failure"]
    if "ls_trigger_reasons" in row:
        result["ls_trigger_reasons"] = row["ls_trigger_reasons"]
    return result


def _step_windows(rows: list[dict[str, Any]], ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    used: set[int] = set()
    for spike in ranked:
        i = int(spike["index"])
        if any(abs(i - prior) <= 4 for prior in used):
            continue
        used.add(i)
        lo, hi = max(0, i - 3), min(len(rows), i + 7)
        windows.append({
            "anchor": spike,
            "rows": [_row_extract(rows[j]) for j in range(lo, hi)],
        })
        if len(windows) >= 12:
            break
    return windows


def _write_train_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [_flatten_numeric(row) for row in rows]
    keys = sorted(set().union(*(row.keys() for row in flat_rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def analyze_run(repo: Path, run_rel: str, out_dir: Path) -> dict[str, Any]:
    run_dir = repo / run_rel
    records, errors = _json_lines(run_dir / "metrics.jsonl")
    events = collections.Counter(str(record.get("event", "<missing>")) for record in records)
    trains = [record for record in records if record.get("event") == "train"]
    ranked = _rank_spikes(trains)
    series = _series(trains)
    name = run_dir.name
    _write_train_csv(out_dir / f"{name}_train.csv", trains)

    start_records = [record for record in records if record.get("event") == "start"]
    train_steps = [int(row["step"]) for row in trains if row.get("step") is not None]
    reversals = [
        {"row": i, "from": train_steps[i - 1], "to": train_steps[i]}
        for i in range(1, len(train_steps))
        if train_steps[i] <= train_steps[i - 1]
    ]
    losses = [float(row.get("loss", math.nan)) for row in trains]
    report = {
        "run": run_rel,
        "record_count": len(records),
        "parse_errors": errors,
        "event_counts": dict(events),
        "start_records": len(start_records),
        "start_steps": [
            record.get("start_step", record.get("step")) for record in start_records
        ],
        "train_rows": len(trains),
        "train_step_first": train_steps[0] if train_steps else None,
        "train_step_last": train_steps[-1] if train_steps else None,
        "step_reversals_or_duplicates": reversals,
        "loss_stats": _percentiles(losses),
        "nonfinite_fields": _nonfinite_inventory(records),
        "top_spikes": ranked,
        "spike_windows": _step_windows(trains, ranked),
        "lead_lag_loss_correlations": _lead_lag_correlations(series)[:100],
        "train_numeric_key_count": len(series),
        "train_numeric_keys": sorted(series),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    runs = tuple(args.runs or DEFAULT_RUNS)
    out_dir = (args.out or Path(__file__).resolve().parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "repo": str(repo),
        "runs": [analyze_run(repo, run, out_dir) for run in runs],
    }
    output = out_dir / "metrics_forensics.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    for run in result["runs"]:
        print(
            f"{Path(run['run']).name}: records={run['record_count']} "
            f"events={run['event_counts']} train={run['train_rows']} "
            f"steps={run['train_step_first']}..{run['train_step_last']} "
            f"loss={run['loss_stats']}"
        )
        print("  top spikes:")
        for spike in run["top_spikes"][:10]:
            print(
                f"    step={spike['step']} loss={spike['loss']:.6g} "
                f"xprev={spike['ratio_to_before']:.3g} "
                f"post/prev={spike['after_to_before']:.3g}"
            )
        print(f"  nonfinite={run['nonfinite_fields']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
