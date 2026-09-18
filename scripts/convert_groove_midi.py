#!/usr/bin/env python3
"""Convert the GMD-style MIDI grooves under assets/rcc/groove/ to RCC JSON.

Standalone reimplementation of the reference converter
(/data2/chris/g1-drummer-lab/scripts/convert_gmd_midi_to_rcc.py): the
pitch->drum mapping, per-segment dedupe/grouping/validation and the RCC
JSON schema (metadata + groups[{time, targets[...]}]) are reproduced.
Stick assignment is this repo's layout-graph min-cost matching (not the
reference preference lists).  Other differences are operational only:

* input is this repo's ``assets/rcc/groove/`` tree (drummer*/session*/*.mid),
  output is ``assets/rcc/train/`` -- except inputs whose (drummer, session,
  basename) belong to the eval split (``EVAL_SOURCE_KEYS``, plus anything a
  clip already in ``assets/rcc/eval/`` cites as its source), which are routed
  to eval instead so eval sources never leak into the training pool;
* an input whose source is already cited by a clip in its destination dir is
  skipped, so nothing is ever double-weighted;
* output names are ``groove[_eval]_<drummer>_<session>_<midi-stem>_seg<NN>.json``;
* the tom and cymbal pitch groups are finer than the reference's (see
  ``PITCH_TO_DRUM`` below) and emit kit-layout drum names directly;
* stick limbs are assigned by walking the clip on the fixed drum-layout
  graph (``amortized_opt.hand_assign``): each beat goes to the closer
  stick given where that stick last played (or idle on its side of the
  kit).  Kicks stay ``right_foot``;
* one segment length (default 30 s), no file caps.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import mido
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mido is required. Install with: pip install mido") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- mapping tables: identical to the reference converter -------------------

# Ids follow assets/drum_kit_layout.yaml; kick (not in the layout) sits last.
DRUM_IDS: dict[str, int] = {
    "snare": 0,
    "hihat": 1,
    "tom1": 2,
    "tom2": 3,
    "cymbal_left": 4,
    "cymbal_right": 5,
    "floor_tom": 6,
    "kick": 7,
}

# Stick limbs are NOT a fixed drum→hand map.  See amortized_opt.hand_assign:
# idle rests on each side of the kit, then greedy closer-stick matching
# walking the clip in time.  Kicks are always right_foot.


def _hand_assign():
    """Load ``amortized_opt.hand_assign`` without importing the torch package."""
    import importlib.util
    import sys

    name = "_groove_hand_assign"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = PROJECT_ROOT / "amortized_opt" / "hand_assign.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Deviations from the reference converter's map, both emitting kit-layout
# names directly (NOT "crash", which load_rcc's DEFAULT_DRUM_ALIASES would
# fold into cymbal_right and defeat the split):
#   * toms follow GMD's published grouping -- high (tom1) / low-mid (tom2) /
#     high floor a.k.a. tom3 (floor_tom) -- instead of a two-way collapse;
#   * crashes (49, 52, 55, 57) -> cymbal_left, rides (51, 53, 59) ->
#     cymbal_right, instead of crash-only with rides ignored.
# Every other pitch is mapped exactly as the reference maps it.
PITCH_TO_DRUM: dict[int, str] = {}
for pitch in (35, 36):
    PITCH_TO_DRUM[pitch] = "kick"
for pitch in (37, 38, 40):
    PITCH_TO_DRUM[pitch] = "snare"
for pitch in (22, 26, 42, 44, 46):
    PITCH_TO_DRUM[pitch] = "hihat"
for pitch in (48, 50):
    PITCH_TO_DRUM[pitch] = "tom1"
for pitch in (45, 47):
    PITCH_TO_DRUM[pitch] = "tom2"
for pitch in (43, 58):
    PITCH_TO_DRUM[pitch] = "floor_tom"
for pitch in (49, 52, 55, 57):
    PITCH_TO_DRUM[pitch] = "cymbal_left"
for pitch in (51, 53, 59):
    PITCH_TO_DRUM[pitch] = "cymbal_right"


@dataclass
class MidiHit:
    time_sec: float
    pitch: int
    velocity: int
    drum: str
    drum_id: int


@dataclass
class SegmentCandidate:
    source_midi: Path
    source_split: str
    segment_start_sec: float
    segment_duration_sec: float
    original_bpm: float | None
    hits: list[MidiHit] = field(default_factory=list)
    pitch_counts: Counter = field(default_factory=Counter)
    ignored_pitch_counts: Counter = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groove_root", type=Path, default=Path("assets/rcc/groove"))
    parser.add_argument("--output_dir", type=Path, default=Path("assets/rcc/train"))
    parser.add_argument("--output_eval_dir", type=Path, default=Path("assets/rcc/eval"))
    parser.add_argument("--segment_seconds", type=float, default=30.0)
    parser.add_argument("--min_events", type=int, default=8)
    parser.add_argument("--max_events_per_second", type=float, default=8.0)
    parser.add_argument("--max_simultaneous_targets", type=int, default=3)
    parser.add_argument("--min_gap_sec", type=float, default=0.06)
    parser.add_argument("--simultaneity_window_sec", type=float, default=0.010)
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow replacing an existing output JSON (MIDI convert).",
    )
    parser.add_argument(
        "--reassign-json",
        action="store_true",
        default=False,
        help="Rewrite stick limbs on existing groove_*.json clips and exit. "
        "Curriculum clips (same_hand_*, two_drum_*) are left alone.",
    )
    parser.add_argument(
        "--reassign-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory to scan with --reassign-json (repeatable). "
        "Default: assets/rcc/train and assets/rcc/eval.",
    )
    return parser.parse_args()


# ---- MIDI -> hits -> groups -------------------------------------------------

def parse_midi_hits(
    midi_path: Path, pitch_map: dict[int, str]
) -> tuple[list[MidiHit], Counter, Counter, float | None]:
    mid = mido.MidiFile(str(midi_path))
    hits: list[MidiHit] = []
    pitch_counts: Counter = Counter()
    ignored_pitch_counts: Counter = Counter()
    original_bpm: float | None = None

    abs_time = 0.0
    for msg in mid:
        abs_time += float(msg.time)
        if msg.type == "set_tempo":
            original_bpm = mido.tempo2bpm(msg.tempo)
        if msg.type != "note_on" or msg.velocity <= 0:
            continue
        pitch_counts[msg.note] += 1
        drum = pitch_map.get(msg.note)
        if drum is None:
            ignored_pitch_counts[msg.note] += 1
            continue
        hits.append(
            MidiHit(
                time_sec=abs_time,
                pitch=msg.note,
                velocity=int(msg.velocity),
                drum=drum,
                drum_id=DRUM_IDS[drum],
            )
        )

    hits.sort(key=lambda h: (h.time_sec, h.drum_id, -h.velocity))
    return hits, pitch_counts, ignored_pitch_counts, original_bpm


def dedupe_hits(hits: list[MidiHit], min_gap_sec: float) -> list[MidiHit]:
    if not hits:
        return []
    out: list[MidiHit] = []
    last_time_by_drum: dict[str, float] = {}
    for hit in hits:
        last = last_time_by_drum.get(hit.drum)
        if last is not None and (hit.time_sec - last) < min_gap_sec:
            if out and out[-1].drum == hit.drum and abs(out[-1].time_sec - hit.time_sec) < min_gap_sec:
                if hit.velocity > out[-1].velocity:
                    out[-1] = hit
            continue
        out.append(hit)
        last_time_by_drum[hit.drum] = hit.time_sec
    return out


def hits_to_groups(
    hits: list[MidiHit],
    max_simultaneous_targets: int,
    simultaneity_window_sec: float,
) -> list[dict[str, Any]]:
    """Bucket near-simultaneous hits, then assign sticks on the layout graph."""
    if not hits:
        return []
    groups: list[dict[str, Any]] = []
    bucket: list[MidiHit] = [hits[0]]
    bucket_time = hits[0].time_sec

    def flush_bucket() -> None:
        nonlocal bucket, bucket_time
        if not bucket:
            return
        drum_to_hit: dict[str, MidiHit] = {}
        for hit in bucket:
            prev = drum_to_hit.get(hit.drum)
            if prev is None or hit.velocity > prev.velocity:
                drum_to_hit[hit.drum] = hit
        targets = []
        for drum, hit in sorted(drum_to_hit.items(), key=lambda kv: DRUM_IDS[kv[0]]):
            targets.append(
                {
                    "drum": drum,
                    "drum_id": hit.drum_id,
                    "limb": "right_foot" if drum == "kick" else "left_hand",
                    "velocity": round(min(1.0, hit.velocity / 127.0), 4),
                }
            )
        if targets:
            groups.append({"time": round(bucket_time, 6), "targets": targets})
        bucket = []

    for hit in hits[1:]:
        if abs(hit.time_sec - bucket_time) <= simultaneity_window_sec:
            bucket.append(hit)
        else:
            flush_bucket()
            bucket = [hit]
            bucket_time = hit.time_sec
    flush_bucket()
    # Two arms; the foot does not occupy a stick slot.  max_simultaneous_targets
    # still caps the JSON group size via validate_segment after assignment.
    max_stick = max(1, int(max_simultaneous_targets) - 1)
    return _hand_assign().assign_rcc_groups(groups, max_stick=max_stick)


def count_targets(groups: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for group in groups:
        for target in group["targets"]:
            counts[str(target["drum"])] += 1
    return counts


def max_simultaneous_in_groups(groups: list[dict[str, Any]]) -> int:
    if not groups:
        return 0
    return max(len(group["targets"]) for group in groups)


def validate_segment(
    groups: list[dict[str, Any]],
    segment_duration_sec: float,
    min_events: int,
    max_events_per_second: float,
    max_simultaneous_targets: int,
) -> str | None:
    event_count = sum(len(g["targets"]) for g in groups)
    if event_count < min_events:
        return f"too_few_events:{event_count}"
    eps = max(segment_duration_sec, 1e-6)
    eps_rate = event_count / eps
    if eps_rate > max_events_per_second:
        return f"too_dense:{eps_rate:.2f}_eps"
    drum_set = {t["drum"] for g in groups for t in g["targets"]}
    if len(drum_set) == 1 and event_count < (min_events * 2):
        return "single_drum_sparse"
    max_sim = max_simultaneous_in_groups(groups)
    if max_sim > max_simultaneous_targets:
        return f"too_many_simultaneous:{max_sim}"
    for group in groups:
        limbs = [t["limb"] for t in group["targets"]]
        if len(limbs) != len(set(limbs)):
            return "duplicate_limb_in_group"
        drums = [t["drum"] for t in group["targets"]]
        if len(drums) != len(set(drums)):
            return "duplicate_drum_in_group"
        if float(group["time"]) < -1e-6 or float(group["time"]) > segment_duration_sec + 1e-3:
            return "time_out_of_range"
    return None


def build_rcc_payload(
    groups: list[dict[str, Any]],
    candidate: SegmentCandidate,
    duration_tag: str,
    tolerance: float,
) -> dict[str, Any]:
    drum_counts = count_targets(groups)
    return {
        "metadata": {
            "name": f"groove_{duration_tag}",
            "source_dataset": "groove_midi_dataset",
            "source_midi": str(candidate.source_midi),
            "source_split": candidate.source_split,
            "original_bpm": candidate.original_bpm,
            "segment_start_sec": candidate.segment_start_sec,
            "segment_duration_sec": candidate.segment_duration_sec,
            "event_count": sum(len(g["targets"]) for g in groups),
            "drum_counts": dict(drum_counts),
            "pitch_counts": {str(k): v for k, v in candidate.pitch_counts.items()},
            "ignored_pitch_counts": {str(k): v for k, v in candidate.ignored_pitch_counts.items()},
            "generated_by": "scripts/convert_groove_midi.py",
            "limb_assignment": "layout_graph_mincost",
            "tolerance": tolerance,
        },
        "groups": groups,
    }


def iter_segments(
    hits: list[MidiHit], segment_seconds: float, midi_duration: float
) -> list[tuple[float, float, list[MidiHit]]]:
    segments: list[tuple[float, float, list[MidiHit]]] = []
    start = 0.0
    while start + segment_seconds <= midi_duration + 1e-6:
        end = start + segment_seconds
        seg_hits = [h for h in hits if start <= h.time_sec < end]
        rel_hits = [
            MidiHit(
                time_sec=h.time_sec - start,
                pitch=h.pitch,
                velocity=h.velocity,
                drum=h.drum,
                drum_id=h.drum_id,
            )
            for h in seg_hits
        ]
        segments.append((start, segment_seconds, rel_hits))
        start += segment_seconds
    return segments


# ---- train/eval routing by source MIDI ---------------------------------------

_DRUMMER_SESSION_RE = re.compile(r"(drummer\d+)[/\\]+([^/\\]+)[/\\]+([^/\\]+\.midi?)$", re.IGNORECASE)

#: The 14 source MIDIs behind the vendored 40-clip ``gmd_eval_*`` split
#: (extracted from their ``metadata.source_midi`` on 2026-08-14).  Any input
#: matching one of these is routed to assets/rcc/eval/, never train, so the
#: eval split's sources cannot leak into the training pool.  Baked in rather
#: than read from the eval dir so the routing survives the old gmd_eval_*
#: clips being retired.
EVAL_SOURCE_KEYS: set[tuple[str, str, str]] = {
    ("drummer1", "eval_session", "10_soul-groove10_102_beat_4-4.mid"),
    ("drummer1", "eval_session", "4_soul-groove4_80_beat_4-4.mid"),
    ("drummer1", "eval_session", "5_funk-groove5_84_beat_4-4.mid"),
    ("drummer1", "eval_session", "8_rock-groove8_65_beat_4-4.mid"),
    ("drummer1", "eval_session", "9_soul-groove9_105_beat_4-4.mid"),
    ("drummer1", "session1", "239_funk-purdieshuffle_130_beat_4-4.mid"),
    ("drummer1", "session1", "2_funk_80_beat_4-4.mid"),
    ("drummer1", "session2", "110_rock-prog_190_beat_4-4.mid"),
    ("drummer1", "session2", "142_gospel_120_beat_4-4.mid"),
    ("drummer1", "session2", "1_latin-chacarera_157_beat_3-4.mid"),
    ("drummer1", "session2", "35_punk_128_beat_4-4.mid"),
    ("drummer1", "session2", "62_punk_144_beat_4-4.mid"),
    ("drummer1", "session3", "12_latin-bomba_96_beat_4-4.mid"),
    ("drummer1", "session3", "13_rock_115_beat_4-4.mid"),
}


def source_key(midi_path: str | Path) -> tuple[str, str, str] | None:
    """(drummer, session, basename) extracted from a source-MIDI path."""
    m = _DRUMMER_SESSION_RE.search(str(midi_path))
    if m is None:
        return None
    return (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())


def existing_source_keys(clip_dir: Path) -> set[tuple[str, str, str]]:
    """Source keys of every clip already in ``clip_dir`` (via metadata.source_midi).

    Inputs whose key is already represented in the destination dir are
    SKIPPED, never written twice: while the old collapsed-mapping gmd_train_*
    / gmd_eval_* clips are still on disk their sources would otherwise be
    double-weighted, and after those are retired a re-run of this script
    converts the freed sources like any other input.
    """
    keys: set[tuple[str, str, str]] = set()
    for f in sorted(clip_dir.glob("*.json")):
        try:
            meta = json.loads(f.read_text()).get("metadata", {})
        except Exception:
            continue
        src = meta.get("source_midi")
        if not src:
            continue
        key = source_key(src)
        if key is not None:
            keys.add(key)
    return keys


# ---- main -------------------------------------------------------------------

_GROOVE_JSON_RE = re.compile(r"^groove(?:_eval)?_", re.IGNORECASE)


def reassign_existing_json(dirs: list[Path], *, dry_run: bool) -> int:
    """Rewrite stick limbs on converted groove clips; skip synthetic curriculum."""
    HA = _hand_assign()
    graph = HA.kit_graph_from_layout()
    n_files = 0
    n_changed = 0
    for d in dirs:
        if not d.is_dir():
            print(f"[WARN] skip missing dir {d}", flush=True)
            continue
        for path in sorted(d.glob("*.json")):
            if not _GROOVE_JSON_RE.match(path.name):
                continue
            n_files += 1
            if HA.reassign_rcc_file(path, graph, dry_run=dry_run):
                n_changed += 1
    print(
        f"[INFO] reassign_json dry_run={dry_run} scanned={n_files} changed={n_changed}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()

    if args.reassign_json:
        raw = args.reassign_dir or [Path("assets/rcc/train"), Path("assets/rcc/eval")]
        dirs = [p if p.is_absolute() else PROJECT_ROOT / p for p in raw]
        return reassign_existing_json(dirs, dry_run=args.dry_run)

    pitch_map = dict(PITCH_TO_DRUM)

    groove_root = args.groove_root if args.groove_root.is_absolute() else PROJECT_ROOT / args.groove_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_eval_dir = (
        args.output_eval_dir if args.output_eval_dir.is_absolute() else PROJECT_ROOT / args.output_eval_dir
    )

    midi_files = sorted(groove_root.rglob("*.mid")) + sorted(groove_root.rglob("*.midi"))
    midi_files = [p for p in midi_files if p.is_file()]
    if not midi_files:
        print(f"[ERROR] No MIDI files under {groove_root}", flush=True)
        return 2

    eval_keys = EVAL_SOURCE_KEYS | existing_source_keys(output_eval_dir)
    known_train = existing_source_keys(output_dir)
    known_eval = existing_source_keys(output_eval_dir)
    print(f"[INFO] {len(midi_files)} MIDI files; {len(eval_keys)} eval-routed sources; "
          f"already present: {len(known_train)} train / {len(known_eval)} eval", flush=True)

    duration_tag = f"{int(args.segment_seconds)}s"
    written = 0
    written_eval = 0
    skipped_dup = 0
    rejected = 0
    parse_failed = 0
    rejection_reasons: Counter = Counter()
    written_paths: list[Path] = []

    for midi_path in midi_files:
        key = source_key(midi_path)
        to_eval = key is not None and key in eval_keys
        if key is not None and key in (known_eval if to_eval else known_train):
            skipped_dup += 1
            continue
        drummer = key[0] if key else "unknown"
        session = key[1] if key else "unknown"

        try:
            hits, pitch_counts, ignored_counts, bpm = parse_midi_hits(midi_path, pitch_map)
        except Exception as exc:
            parse_failed += 1
            print(f"[WARN] parse failed {midi_path}: {exc}", flush=True)
            continue
        if not hits:
            rejected += 1
            rejection_reasons["no_mapped_hits"] += 1
            continue

        midi_duration = max(h.time_sec for h in hits)
        for seg_idx, (seg_start, seg_duration, seg_hits) in enumerate(
            iter_segments(hits, args.segment_seconds, midi_duration)
        ):
            if not seg_hits:
                rejected += 1
                rejection_reasons["empty_segment"] += 1
                continue
            cleaned = dedupe_hits(seg_hits, args.min_gap_sec)
            groups = hits_to_groups(cleaned, args.max_simultaneous_targets, args.simultaneity_window_sec)
            reason = validate_segment(
                groups,
                seg_duration,
                args.min_events,
                args.max_events_per_second,
                args.max_simultaneous_targets,
            )
            if reason:
                rejected += 1
                rejection_reasons[reason.split(":")[0]] += 1
                continue

            candidate = SegmentCandidate(
                source_midi=midi_path,
                source_split="eval" if to_eval else "train",
                segment_start_sec=seg_start,
                segment_duration_sec=seg_duration,
                original_bpm=bpm,
                pitch_counts=pitch_counts,
                ignored_pitch_counts=ignored_counts,
            )
            payload = build_rcc_payload(groups, candidate, duration_tag, args.tolerance)
            prefix = "groove_eval" if to_eval else "groove"
            out_name = f"{prefix}_{drummer}_{session}_{midi_path.stem}_seg{seg_idx:02d}.json"
            out_path = (output_eval_dir if to_eval else output_dir) / out_name
            if out_path.exists() and not args.overwrite:
                raise SystemExit(f"[ERROR] refusing to overwrite existing {out_path}")
            if not args.dry_run:
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if to_eval:
                written_eval += 1
            else:
                written += 1
            written_paths.append(out_path)

    print(f"[INFO] dry_run={args.dry_run}", flush=True)
    print(f"[INFO] written_train={written} written_eval={written_eval} "
          f"skipped_dup_midis={skipped_dup} rejected_segments={rejected} "
          f"parse_failed={parse_failed}", flush=True)
    print(f"[INFO] rejection_reasons={dict(rejection_reasons)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
