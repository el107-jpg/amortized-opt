#!/usr/bin/env python3
"""Before/after planner-time heatmaps for a conditional chart snapshot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out/"after_snapshot.npz"
    snapshot.write_bytes(args.after.read_bytes())  # Freeze one atomic exported version.
    base = MinimumTimeChart.load(args.table)
    before = ConditionalMinimumTimeChart.load(args.before, base)
    after = ConditionalMinimumTimeChart.load(snapshot, base)
    generation = after.metadata.get("parameter_search", {}).get("generation", "?")
    selection = after.metadata.get("parameter_search", {}).get("selection", "")
    chart_label = "CMA training chart" if selection in ("current_matched_training_panel", "cma_population_mean") else "Conditional planning chart"
    active = after.trainable_mask
    initialized = after.initialized_unknown_mask
    baseline = before.anchors[..., 1]*1000
    anchors = after.anchors*1000
    maximum = np.ceil(max(float(baseline[before.trainable_mask].max()), float(anchors[active].max()))/50)*50
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    labels = [dict(snare="Snare", hihat="Hi-hat", tom1="Tom 1", tom2="Tom 2",
                   cymbal_left="Cym L", cymbal_right="Cym R", floor_tom="Floor").get(n, n)
              for n in base.names]
    def panel(ax, values, hand, title, delta=False, limit=None, guessed=None):
        cmap = plt.get_cmap("RdBu_r" if delta else "viridis").copy()
        cmap.set_bad("#ededed")
        shown = np.ma.masked_array(values, mask=~(base.allowed[hand] & np.isfinite(values)))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit) if delta else None
        im = ax.imshow(shown, cmap=cmap, norm=norm,
                       **({} if delta else dict(vmin=0, vmax=maximum)))
        for s in range(len(labels)):
            for d in range(len(labels)):
                if not base.allowed[hand, s, d]:
                    ax.add_patch(Rectangle((d-.5, s-.5), 1, 1, facecolor="#c6c6c6", edgecolor="white"))
                    text = "—"; color = "#555555"
                elif not np.isfinite(values[s, d]):
                    text = "new*" if delta and initialized[hand, s, d] else "?"
                    color = "#555555"
                else:
                    value = values[s, d]
                    text = f"{value:+.0f}" if delta else f"{value:.0f}"
                    if guessed is not None and guessed[hand, s, d]:
                        text += "*"
                    rgba = cmap(norm(value) if delta else value/maximum)
                    color = "white" if np.dot(rgba[:3], [.299, .587, .114]) < .48 else "black"
                ax.text(d, s, text, ha="center", va="center", fontsize=8.5, color=color)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Destination drum"); ax.set_ylabel("Source drum")
        ax.set_title(title, fontsize=11)
        return im
    footer = "Planning time in ms • ? = unknown • — = ineligible • * = originally initialized from a guess; displayed values are learned times"
    figure, axes = plt.subplots(2, 3, figsize=(13, 10), constrained_layout=True)
    delta = np.full_like(baseline, np.nan)
    common = active & before.trainable_mask
    delta[common] = anchors[..., 1][common]-baseline[common]
    bound = max(10., np.ceil(np.abs(delta[common]).max()/10)*10)
    for h, name in enumerate(("Left hand", "Right hand")):
        old = panel(axes[h, 0], baseline[h], h, name+" — BEFORE", guessed=before.initialized_unknown_mask)
        new = panel(axes[h, 1], anchors[h, ..., 1], h, name+" — AFTER, proxy = 0", guessed=initialized)
        change = panel(axes[h, 2], delta[h], h, "Change: after − before", True, bound)
    figure.colorbar(new, ax=axes[:, :2], shrink=.65, label="Planning time (ms)")
    figure.colorbar(change, ax=axes[:, 2], shrink=.65, label="Change (ms): blue shorter, red longer")
    title = (f"{chart_label} • generation {generation} snapshot • initialized cells marked *" if initialized.any() else
        f"Initial vs validation-selected conditional chart • generation {generation} snapshot")
    figure.suptitle(title, fontsize=15)
    figure.supxlabel(footer, fontsize=9)
    figure.savefig(args.out/"before_after_center.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(17, 10), constrained_layout=True)
    for h, name in enumerate(("Left hand", "Right hand")):
        panel(axes[h, 0], baseline[h], h, name+" — BEFORE", guessed=before.initialized_unknown_mask)
        for j, proxy in enumerate((-1, 0, 1)):
            im = panel(axes[h, j+1], anchors[h, ..., j], h, f"AFTER: other-hand proxy {proxy:+d}", guessed=initialized)
    figure.colorbar(im, ax=axes, shrink=.65, label="Planning time (ms), shared scale")
    figure.suptitle("All three anchors • "+title, fontsize=15)
    figure.supxlabel(footer, fontsize=9)
    figure.savefig(args.out/"before_after_all_anchors.png", dpi=160)
    plt.close(figure)
    (args.out/"snapshot.json").write_text(json.dumps(dict(generation=generation,
        after_source=str(args.after.resolve()), before_source=str(args.before.resolve()),
        parameters=len(after.get_parameters()), initialized_unknown_cells=int(initialized.sum()),
        values="learned_planning_times_with_guess_origin_marked" if initialized.any()
            else "learned_planning_times", marker_meaning="original_initialization_provenance_not_frozen_values"), indent=2)+"\n")
    print(args.out/"before_after_center.png")
    print(args.out/"before_after_all_anchors.png")


if __name__ == "__main__": main()
