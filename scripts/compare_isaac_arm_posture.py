#!/usr/bin/env python3
"""Compare measured arm COM proxies at matched Isaac hit-approach timestamps."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import amortized_opt as A
from amortized_opt.fk import stick_mount_transform


def measured_rows(path):
    saved = json.loads(Path(path).read_text())
    states = {}
    for event in saved["events"]:
        for row in event["approach"]:
            tick = round(row["time_s"]/.02)
            q = np.asarray(row["q"], dtype=np.float64)
            if tick in states:
                np.testing.assert_allclose(states[tick], q, atol=1e-6, rtol=0)
            states[tick] = q
    return states


def arm_proxies(q):
    fk = A.ChainFK(frames=dict(A.FusedCost.FRAMES), device="cpu", dtype=torch.float64)
    with torch.no_grad():
        frames = fk.forward(torch.as_tensor(q, dtype=torch.float64))
        Rt, pt = frames["torso"]
        heights, reaches, elbow = [], [], []
        masses = (2.095, .600, .824, .100)
        for hand in ("left", "right"):
            shoulder = frames[hand+"_shoulder"][1]
            e = frames[hand+"_elbow"][1]
            Rw, w = frames[hand+"_wrist"]
            mount = torch.as_tensor(stick_mount_transform(hand)[1], dtype=q.dtype if torch.is_tensor(q) else torch.float64)
            offset = (Rw @ mount[:, None]).squeeze(-1)
            com = (masses[0]*(.65*shoulder+.35*e)+masses[1]*(.5*(e+w))
                   +masses[2]*(w+.25*offset)+masses[3]*(w+offset))/sum(masses)
            torso = (Rt.transpose(-1, -2) @ (com-pt)[..., None]).squeeze(-1)
            heights.append(torso[:, 2])
            reaches.append(torso[:, :2].norm(dim=-1))
            elbow.append((e[:, 2]-.5*(shoulder[:, 2]+w[:, 2])-.05).clamp_min(0))
        return {name: torch.stack(values, -1).numpy() for name, values in
                (("height_m", heights), ("reach_m", reaches), ("elbow_excess_m", elbow))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--rhythm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    before, after = measured_rows(args.before), measured_rows(args.after)
    ticks = sorted(set(before)&set(after))
    if len(ticks) < 100:
        raise ValueError("Not enough matched measured-state samples")
    times = np.asarray(ticks)*.02
    clip = A.load_rcc(args.rhythm, A.DrumKit.from_layout())
    active = np.zeros((len(ticks), 2), bool)
    for h, hand in enumerate(("left", "right")):
        hit_times = np.array([x.time for x in clip.hits if x.hand == hand])
        upcoming = np.where(hit_times[None] > times[:, None]+1e-8,
                            hit_times[None]-times[:, None], np.inf).min(-1)
        active[:, h] = upcoming <= 1.+1e-8
    a = arm_proxies(np.stack([before[t] for t in ticks]))
    b = arm_proxies(np.stack([after[t] for t in ticks]))
    summary = dict(matched_timestamps=len(ticks), first_time_s=float(times[0]),
                   last_time_s=float(times[-1]), active_arm_samples=int(active.sum()),
                   definition="Same four-point mass-weighted arm COM proxy as the loss, in torso coordinates",
                   sample_scope="Measured pre-control states in saved hit approaches, matched timestamps and scheduled arm activity",
                   before={}, after={}, differences={})
    for name in a:
        va, vb = float(a[name][active].mean()), float(b[name][active].mean())
        summary["before"][name] = va
        summary["after"][name] = vb
        summary["differences"][name] = dict(after_minus_before_m=vb-va,
            reduction_percent=100*(va-vb)/va if abs(va) > 1e-12 else None)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"measured_posture.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(args.out/"measured_posture.npz", times=times, active=active,
        before_q=np.stack([before[t] for t in ticks]), after_q=np.stack([after[t] for t in ticks]),
        **{"before_"+k:v for k,v in a.items()}, **{"after_"+k:v for k,v in b.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, key, label in zip(axes, ("height_m", "reach_m"), ("Arm COM height (cm)", "Arm COM reach (cm)")):
        for values, tag, color in ((a, "Before: step9600", "#888888"), (b, "After: joint fine-tune12000", "#3288bd")):
            count = active.sum(-1)
            mean = (values[key]*active).sum(-1)/np.maximum(count, 1)
            mean[count==0] = np.nan
            ax.plot(times, mean*100, label=tag, color=color, linewidth=1.)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Isaac simulated time (s)")
    fig.tight_layout()
    fig.savefig(args.out/"measured_posture.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
