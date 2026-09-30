"""Summarize and plot the terminal-pose learning-rate ablation."""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("runs/terminal_pose/lr_ablation_feasible_qpth_20260926")
RUNS = {
    "Constant 1e-3": {
        "logs": [Path("runs/terminal_pose/ablate_feasible_qpth_20260926/mix32_10k.log")],
        "color": "#64748b",
        "marker": "o",
        "style": "--",
    },
    "Constant 5e-4": {
        "logs": [OUT_DIR / "constant_5e-4_10k.log"],
        "color": "#2563eb",
        "marker": "s",
        "style": "-",
    },
    "Cosine 1e-3 → 1e-4": {
        "logs": [OUT_DIR / "cosine_1e-3_to_1e-4_10k.log"],
        "color": "#059669",
        "marker": "^",
        "style": "-",
    },
    "Line search / 25": {
        "logs": [
            OUT_DIR / "line_search_every25_seed_1e-3_10k.log",
            OUT_DIR / "line_search_every25_seed_1e-3_10k_to_15k.log",
        ],
        "color": "#dc2626",
        "marker": "D",
        "style": "-",
    },
}


def fields(line: str) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in re.findall(r"(\w+)=([+.\deE-]+)", line)
    }


def load_run(paths: list[Path]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    evaluations = {}
    step_logs = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("[eval]"):
                row = fields(line)
                evaluations[int(row["step"])] = row
            elif line.startswith("step="):
                row = fields(line)
                step_logs[int(row["step"])] = row
    return (
        [evaluations[key] for key in sorted(evaluations)],
        [step_logs[key] for key in sorted(step_logs)],
    )


def cosine_lr(step: np.ndarray) -> np.ndarray:
    t = (step - 1.0) / 9999.0
    return 1e-4 + (1e-3 - 1e-4) * 0.5 * (1.0 + np.cos(np.pi * t))


def summary(rows: list[dict[str, float]], stop: int = 10_000) -> dict:
    base = [row for row in rows if row["step"] <= stop]
    late = [row for row in base if row["step"] >= 7_000]
    x = np.asarray([row["step"] / 1000.0 for row in late])
    return {
        "checkpoint_count": len(base),
        "mean_hit_med_mm": float(np.mean([row["hit_med_mm"] for row in base])),
        "mean_hit_1cm": float(np.mean([row["hit_1cm"] for row in base])),
        "mean_viol_med_mm": float(np.mean([row["viol_med_mm"] for row in base])),
        "mean_col_ok": float(np.mean([row["col_ok"] for row in base])),
        "late_hit_slope_mm_per_1k": float(
            np.polyfit(x, [row["hit_med_mm"] for row in late], 1)[0]
        ),
        "late_col_ok_slope_per_1k": float(
            np.polyfit(x, [row["col_ok"] for row in late], 1)[0]
        ),
        "final": base[-1],
    }


def main() -> None:
    loaded = {
        label: (*load_run(spec["logs"]), spec)
        for label, spec in RUNS.items()
    }
    metrics = {
        label: summary(evaluations)
        for label, (evaluations, _steps, _spec) in loaded.items()
    }
    line_evals = loaded["Line search / 25"][0]
    metrics["Line search / 25"]["extension_10k_to_15k"] = [
        row for row in line_evals if row["step"] >= 10_000
    ]
    (OUT_DIR / "lr_ablation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    fig.suptitle("QP + 32 learning-rate ablation", fontsize=15, fontweight="bold")
    for label, (evaluations, _steps, spec) in loaded.items():
        x = [row["step"] / 1000 for row in evaluations]
        common = dict(
            color=spec["color"], marker=spec["marker"], linestyle=spec["style"],
            linewidth=2, markersize=5, label=label,
        )
        axes[0].plot(x, [row["hit_med_mm"] for row in evaluations], **common)
        axes[1].plot(x, [100 * row["col_ok"] for row in evaluations], **common)

    x = np.linspace(1, 10_000, 400)
    axes[2].plot(x / 1000, np.full_like(x, 1e-3), "--", color="#64748b", label="Constant 1e-3")
    axes[2].plot(x / 1000, np.full_like(x, 5e-4), color="#2563eb", label="Constant 5e-4")
    axes[2].plot(x / 1000, cosine_lr(x), color="#059669", label="Cosine")
    line_steps = loaded["Line search / 25"][1]
    ls_rows = [row for row in line_steps if "ls_eps" in row]
    axes[2].plot(
        [row["step"] / 1000 for row in ls_rows],
        [row["ls_eps"] for row in ls_rows],
        color="#dc2626", marker="D", linewidth=2, markersize=5, label="Line search / 25",
    )

    axes[0].set_title("Held-out hit accuracy")
    axes[0].set_xlabel("Training step (thousands)")
    axes[0].set_ylabel("Median hit error (mm) ↓")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_title("Held-out collision feasibility")
    axes[1].set_xlabel("Training step (thousands)")
    axes[1].set_ylabel("Collision-clean samples (%) ↑")
    axes[2].set_title("Applied / selected learning rate")
    axes[2].set_xlabel("Training step (thousands)")
    axes[2].set_ylabel("Learning rate (log scale)")
    axes[2].set_yscale("log")
    axes[2].axvline(10, color="#94a3b8", linewidth=1, alpha=0.7)
    axes[2].text(
        10.12, 2e-4, "10k continuation", color="#64748b", fontsize=8, rotation=90,
        va="center",
    )
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    output = OUT_DIR / "lr_ablation_chart.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
