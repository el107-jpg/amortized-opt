#!/usr/bin/env python3
"""Select complete recorded MIDI windows for balanced seven-drum evaluation.

Selection uses drum counts only, never a policy's scores or hit outcomes.
Sources already converted into the training/evaluation pools are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from amortized_opt.rhythm import DrumKit, load_rcc, quantize_time
from amortized_opt.hand_assign import assign_rcc_groups
from scripts.convert_groove_midi import parse_midi_hits, PITCH_TO_DRUM


@dataclass
class Window:
    source: str
    start: int
    counts: np.ndarray


def source_inventory(directories):
    used = set()
    for directory in directories:
        for path in Path(directory).glob("*.json"):
            source = json.loads(path.read_text()).get("metadata", {}).get("source_midi")
            if source:
                used.add(Path(source).resolve())
    return used


def select_windows(windows, count, groups, max_per_source):
    """Greedy balance followed by improving swaps; no overlapping excerpts."""
    matrix = np.stack([w.counts for w in windows])
    chosen = []
    totals = np.zeros(matrix.shape[1], dtype=int)

    def allowed(ids):
        mask = np.ones(len(windows), dtype=bool)
        by_source = defaultdict(list)
        for index in ids:
            by_source[windows[index].source].append(windows[index].start)
        for i, window in enumerate(windows):
            starts = by_source[window.source]
            if len(starts) >= max_per_source or any(abs(window.start-s) < groups for s in starts):
                mask[i] = False
        return mask

    def errors(base):
        next_counts = matrix + base
        proportions = next_counts / next_counts.sum(axis=1, keepdims=True)
        return ((proportions - 1./matrix.shape[1])**2).sum(axis=1)

    for _ in range(count):
        scores = errors(totals)
        scores[~allowed(chosen)] = np.inf
        index = int(np.argmin(scores))
        if not np.isfinite(scores[index]):
            raise ValueError("Not enough nonoverlapping windows under the source cap")
        chosen.append(index)
        totals += matrix[index]
    # Remove each window in turn and find a strictly better replacement.
    for _ in range(5):
        changed = False
        for position, old in enumerate(chosen):
            scores = errors(totals-matrix[old])
            scores[~allowed(chosen[:position]+chosen[position+1:])] = np.inf
            new = int(np.argmin(scores))
            if scores[new] < scores[old] - 1e-12:
                totals += matrix[new]-matrix[old]
                chosen[position] = new
                changed = True
        if not changed:
            break
    return [windows[i] for i in chosen], totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groove-root", type=Path, default=ROOT/"assets/rcc/groove")
    parser.add_argument("--exclude-dir", type=Path, action="append")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--max-per-source", type=int, default=4)
    args = parser.parse_args()
    if min(args.scenarios, args.groups, args.max_per_source) < 1:
        parser.error("Counts must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    excluded_dirs = args.exclude_dir or [ROOT/"assets/rcc/train", ROOT/"assets/rcc/eval"]
    used = source_inventory(excluded_dirs)
    kit = DrumKit.from_layout()
    names = kit.names
    windows, sources = [], {}
    rejection_counts = Counter()
    with (args.groove_root/"info.csv").open() as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: r["midi_filename"])
    for row in rows:
        source = (args.groove_root/row["midi_filename"]).resolve()
        if row["split"] not in ("test", "validation") or not source.exists() or source in used:
            continue
        hits, _, ignored, bpm = parse_midi_hits(source, PITCH_TO_DRUM)
        # Keep every mapped arm event. Reject windows containing >2 simultaneous
        # arm hits or repeated mapped drums rather than silently dropping notes.
        buckets = defaultdict(list)
        for hit in hits:
            if hit.drum != "kick":
                buckets[quantize_time(hit.time_sec)].append(hit)
        groups = sorted(buckets.items())
        sources[str(source)] = dict(row=row, bpm=bpm, groups=groups, ignored=dict(ignored))
        for start in range(len(groups)-args.groups+1):
            selected = groups[start:start+args.groups]
            if any(len(hs) > 2 or len({h.drum for h in hs}) != len(hs) for _, hs in selected):
                rejection_counts["simultaneous_or_duplicate_arm_events"] += 1
                continue
            if np.any(np.diff([t for t, _ in selected]) > 1.+1e-8):
                rejection_counts["gap_above_one_second"] += 1
                continue
            counts = Counter(h.drum for _, hs in selected for h in hs)
            windows.append(Window(str(source), start, np.array([counts[n] for n in names])))
    chosen, totals = select_windows(windows, args.scenarios, args.groups, args.max_per_source)
    args.out.mkdir(parents=True)
    clip_dir = args.out/"clips"
    clip_dir.mkdir()
    entries = []
    for i, window in enumerate(chosen):
        source = sources[window.source]
        selected = source["groups"][window.start:window.start+args.groups]
        offset = 1.-selected[0][0]
        groups = [dict(time=round(t+offset, 8), targets=[dict(
            drum=h.drum, drum_id=h.drum_id, limb="left_hand", velocity=h.velocity/127.
        ) for h in hs]) for t, hs in selected]
        assigned = assign_rcc_groups(groups)
        if sum(len(g["targets"]) for g in assigned) != int(window.counts.sum()):
            raise ValueError("Assignment unexpectedly dropped an arm event")
        stem = source["row"]["midi_filename"].replace("/", "_").removesuffix(".mid")
        relative_path = f"clips/{i:02d}_{stem}_group{window.start}.json"
        metadata = dict(source_dataset="groove_midi_dataset", source_midi=window.source,
            source_split=source["row"]["split"], original_bpm=source["bpm"],
            source_start_s=selected[0][0], source_end_s=selected[-1][0],
            source_group_start=window.start, hand_hit_groups=args.groups,
            drum_counts=dict(zip(names, window.counts.tolist())),
            preprocessing="20 ms timing grid; kicks omitted; no mapped arm hits dropped; no tempo scaling",
            selection="drum coverage only; no controller results used")
        payload = dict(metadata=metadata, groups=assigned)
        (args.out/relative_path).write_text(json.dumps(payload, indent=2)+"\n")
        parsed = load_rcc(args.out/relative_path, kit)
        actual = Counter(h.drum_name for h in parsed.hits)
        assert [actual[n] for n in names] == window.counts.tolist()
        entries.append(dict(file=relative_path, **metadata))
    summary = dict(scenarios=len(chosen), groups_per_window=args.groups,
        requested=int(totals.sum()), drum_counts=dict(zip(names, totals.tolist())),
        drum_percentages=dict(zip(names, (100*totals/totals.sum()).tolist())),
        source_recordings=len({w.source for w in chosen}),
        drummers=sorted({sources[w.source]["row"]["drummer"] for w in chosen}),
        styles=dict(Counter(sources[w.source]["row"]["style"] for w in chosen)),
        beat_types=dict(Counter(sources[w.source]["row"]["beat_type"] for w in chosen)),
        eligible_windows=len(windows), unused_official_heldout_sources=len(sources),
        excluded_source_recordings=len(used), rejected_windows=dict(rejection_counts),
        max_windows_per_source=args.max_per_source,
        exclusion_directories=[str(p.resolve()) for p in excluded_dirs],
        original_source_disjoint_from_existing_train_and_eval=True,
        source_windows_do_not_overlap=True, selection_uses_hit_outcomes=False,
        note="Balanced drum-coverage stress test, not the natural frequency distribution of music")
    (args.out/"manifest.json").write_text(json.dumps(dict(summary=summary, clips=entries), indent=2)+"\n")
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    export_rhythm_index(args.out, entries, names)
    print(json.dumps(summary, indent=2))


def export_rhythm_index(out, entries, names):
    """Small source-level index for inspecting the chosen recordings."""
    by_source = defaultdict(list)
    for entry in entries:
        by_source[entry["source_midi"]].append(entry)
    with (out/"rhythms.csv").open("w") as handle:
        fields = ["recording", "bpm", "windows", "requested", *names, "source_midi"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source, rows in sorted(by_source.items()):
            counts = Counter()
            for row in rows:
                counts.update(row["drum_counts"])
            writer.writerow(dict(recording=Path(source).stem, bpm=round(rows[0]["original_bpm"], 2),
                windows=len(rows), requested=sum(counts.values()),
                **{n: counts[n] for n in names}, source_midi=source))


if __name__ == "__main__":
    main()
