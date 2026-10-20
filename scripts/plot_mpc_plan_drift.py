#!/usr/bin/env python
"""Plot the paired plan-drift diagnostic JSON as a standalone figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.report.read_text())
    sources = [("fresh_training_parent", "Fresh training parents"),
               ("saved_training_mpc_parent", "Saved training MPC parents")]
    modes = [("ou", "OU", "#2878b5"),
             ("anchored_ou", "Anchored OU", "#d95f02"),
             ("smooth_bias", "Smooth bias", "#269461")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for row, (source, label) in enumerate(sources):
        for mode, name, color in modes:
            records = sorted([r for r in data["reports"] if r["source"] == source
                              and r["mode"] == mode and r["iterations"] == 1],
                             key=lambda r: r["amplitude"])
            x = [1000*r["groups"]["all"]["input_drift_rad"]["p50"] for r in records]
            axes[row, 0].plot(x, [100*r["groups"]["all"]["output_within_5mm_rate"]
                                for r in records], "o-", color=color, label=name)
            axes[row, 1].plot(x, [100*r["groups"]["all"][
                "preserve_clean_output_on_head_rate"] for r in records],
                "o-", color=color)
            axes[row, 2].plot(x[1:], [1000*r["groups"]["all"]["output_drift_rad"]["p50"]
                                    for r in records[1:]], "o-", color=color)
        axes[row, 0].set_title(f"{label}: output within 5 mm")
        axes[row, 1].set_title("Retain clean output's exact on-head result")
        axes[row, 2].set_title("Paired decoder output displacement")
        axes[row, 0].set_ylabel("Collocations (%)")
        axes[row, 1].set_ylabel("Clean on-head outputs retained (%)")
        axes[row, 2].set_ylabel("Output drift, median (mrad RMS)")
        for ax in axes[row]:
            ax.set_xlabel("Realized input drift, median (mrad RMS)")
            ax.set_xscale("symlog", linthresh=5)
            ax.grid(alpha=.2)
        axes[row, 0].set_ylim(0, 103)
        axes[row, 1].set_ylim(0, 103)
        axes[row, 2].set_xscale("log")
        axes[row, 2].set_yscale("log")
        axes[row, 2].plot([5, 300], [5, 300], "--", color=".65", label="Equal drift")
    axes[0, 0].legend(frameon=False)
    axes[0, 2].legend(frameon=False)
    fig.suptitle(f"Plan-only drift at step {data['step']} — one decoder pass\n"
                 "Fixed state, goals and hit timing; balanced remaining-horizon bands",
                 fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for source, name in sources:
        for iteration, style in [(1, "-"), (4, "--")]:
            rs = sorted([r for r in data["reports"] if r["source"] == source
                         and r["mode"] == "phase_lag" and r["iterations"] == iteration],
                        key=lambda r: r["amplitude"])
            if not rs:
                continue
            x = [r["amplitude"] for r in rs]
            label = f"{name}, {iteration} pass(es)"
            axes[0].plot(x, [100*r["groups"]["all"]["output_within_5mm_rate"] for r in rs],
                         style, marker="o", label=label)
            axes[1].plot(x, [100*r["groups"]["all"][
                "preserve_clean_output_on_head_rate"] for r in rs], style, marker="o")
    axes[0].set_title("Output within 5 mm")
    axes[1].set_title("Clean exact on-head outputs retained")
    for ax in axes:
        ax.set_xlabel("Input plan delay (20 ms ticks); goal clock unchanged")
        ax.set_ylabel("Collocations (%)")
        ax.set_ylim(0, 103)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Timing drift of the input plan")
    fig.savefig(args.out.with_name(args.out.stem + "_phase.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
