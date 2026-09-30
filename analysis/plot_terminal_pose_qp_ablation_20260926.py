"""Plot the feasible-QP terminal-pose ablation from its training logs."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


RUN_DIR = Path("runs/terminal_pose/ablate_feasible_qpth_20260926")
RUNS = {
    "Naked QP": ("qpout_10k.log", "#2563eb", "o"),
    "QP + 32": ("mix32_10k.log", "#059669", "s"),
    "QP + 128": ("mix128_10k.log", "#dc2626", "^"),
}
LABEL_OFFSETS = {
    "Naked QP": (6, 8),
    "QP + 32": (6, -13),
    "QP + 128": (-58, -13),
}


def read_evals(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("[eval]"):
            continue
        rows.append(
            {
                key: float(value)
                for key, value in re.findall(r"(\w+)=([+.\deE-]+)", line)
            }
        )
    return rows


def main() -> None:
    data = {
        label: (read_evals(RUN_DIR / filename), color, marker)
        for label, (filename, color, marker) in RUNS.items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fig.suptitle("Feasible-QP terminal-pose ablation", fontsize=15, fontweight="bold")

    for label, (rows, color, marker) in data.items():
        steps = [row["step"] / 1000 for row in rows]
        axes[0].plot(
            steps,
            [row["hit_med_mm"] for row in rows],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            steps,
            [100 * row["col_ok"] for row in rows],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )
        x = [row["hit_med_mm"] for row in rows]
        y = [100 * row["col_ok"] for row in rows]
        axes[2].plot(x, y, color=color, linewidth=1.2, alpha=0.55)
        axes[2].scatter(x, y, color=color, marker=marker, s=32, alpha=0.8)
        axes[2].scatter(
            x[-1], y[-1], color=color, marker="*", s=180, edgecolor="white", linewidth=0.8
        )
        axes[2].annotate(
            label,
            (x[-1], y[-1]),
            xytext=LABEL_OFFSETS[label],
            textcoords="offset points",
            color=color,
            fontsize=9,
        )

    axes[0].set_title("Hit accuracy (lower is better)")
    axes[0].set_xlabel("Training step (thousands)")
    axes[0].set_ylabel("Median hit error (mm)")
    axes[0].set_xticks(range(1, 11))
    axes[0].legend(frameon=False)

    axes[1].set_title("Collision feasibility (higher is better)")
    axes[1].set_xlabel("Training step (thousands)")
    axes[1].set_ylabel("Collision-clean samples (%)")
    axes[1].set_xticks(range(1, 11))

    axes[2].set_title("Accuracy–feasibility frontier")
    axes[2].set_xlabel("Median hit error (mm)  ← better")
    axes[2].set_ylabel("Collision-clean samples (%)  → better")
    axes[2].text(
        0.03,
        0.97,
        "★ = 10k checkpoint",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
        color="#475569",
    )

    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    output = RUN_DIR / "ablation_chart.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
