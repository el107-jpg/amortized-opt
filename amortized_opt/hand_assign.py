"""Layout-graph stick assignment for RCC clips.

Each stick starts idle on its side of the kit.  Walking the clip in time,
every beat is given the min-cost matching of free sticks to the drums in
that group, measured on the Euclidean graph of drum centres plus a
laterality penalty for far-side reaches.  A stick that just played stays
parked on that drum until it is assigned again.

Numpy + yaml only — the MIDI converter imports this without torch.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_LAYOUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "drum_kit_layout.yaml"

__all__ = [
    "LIMB_FOOT",
    "LIMB_LEFT",
    "LIMB_RIGHT",
    "STICK_LIMBS",
    "KitGraph",
    "assign_rcc_groups",
    "kit_graph_from_layout",
    "reassign_rcc_file",
    "reassign_rcc_payload",
]

LIMB_LEFT = "left_hand"
LIMB_RIGHT = "right_hand"
LIMB_FOOT = "right_foot"
STICK_LIMBS: tuple[str, str] = (LIMB_LEFT, LIMB_RIGHT)

_INF = 1e9
#: Kit +y is the robot's LEFT (hihat / crash).  Added to travel so a stick
#: already sitting on the snare (true travel 0) will not keep the snare and
#: force the other stick onto a far-side cymbal.
_LATERAL_M = 0.75
_LEFT_SIDE_Y = 0.40   # hihat 0.52, cymbal_left 0.47
_RIGHT_SIDE_Y = -0.10  # tom1 -0.15, ride -0.37, floor_tom -0.50


@dataclass(frozen=True)
class KitGraph:
    """Drum centres, pairwise distances, no-crossover bans, idle rests."""

    names: tuple[str, ...]
    positions: np.ndarray  # (D, 3)
    dist: np.ndarray  # (D, D) Euclidean
    idle: dict[str, np.ndarray]  # limb -> (3,)
    banned: dict[str, int]  # limb -> drum index that stick must not play

    @property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def eligible(self, limb: str, drum: str) -> bool:
        i = self.index.get(drum)
        if i is None:
            return False
        return i != self.banned.get(limb, -1)

    def laterality(self, limb: str, drum: str) -> float:
        """Extra cost for a stick taking a drum on the other side of the kit."""
        i = self.index.get(drum)
        if i is None:
            return 0.0
        y = float(self.positions[i, 1])
        if limb == LIMB_LEFT and y <= _RIGHT_SIDE_Y:
            return _LATERAL_M
        if limb == LIMB_RIGHT and y >= _LEFT_SIDE_Y:
            return _LATERAL_M
        return 0.0

    def travel(self, limb: str, parked: str | None, drum: str) -> float:
        """Distance from this stick's current rest (idle or last drum) to ``drum``."""
        if not self.eligible(limb, drum):
            return _INF
        j = self.index[drum]
        if parked is None:
            geom = float(np.linalg.norm(self.idle[limb] - self.positions[j]))
        else:
            i = self.index.get(parked)
            if i is None:
                geom = float(np.linalg.norm(self.idle[limb] - self.positions[j]))
            else:
                geom = float(self.dist[i, j])
        return geom + self.laterality(limb, drum)


def kit_graph_from_layout(layout_path: str | Path | None = None) -> KitGraph:
    """Build the assignment graph from ``assets/drum_kit_layout.yaml``."""
    path = Path(layout_path) if layout_path is not None else _LAYOUT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"drum layout missing at {path}")
    with open(path) as f:
        layout = yaml.safe_load(f)
    items = list(layout["drums"].items())
    if all("id" in spec for _n, spec in items):
        items.sort(key=lambda kv: int(kv[1]["id"]))
    names = tuple(str(n) for n, _s in items)
    pos = np.stack([np.asarray(spec["position"], dtype=float) for _n, spec in items])
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    ys = pos[:, 1]
    # Left loses the far-right drum (min y), right loses the far-left (max y) —
    # same no-crossover rule as DrumKit.eligible.
    banned = {LIMB_LEFT: int(np.argmin(ys)), LIMB_RIGHT: int(np.argmax(ys))}
    left_mask = ys >= 0.0
    right_mask = ys <= 0.0
    if not np.any(left_mask):
        left_mask[:] = True
    if not np.any(right_mask):
        right_mask[:] = True
    idle = {
        LIMB_LEFT: pos[left_mask].mean(axis=0),
        LIMB_RIGHT: pos[right_mask].mean(axis=0),
    }
    return KitGraph(names=names, positions=pos, dist=d, idle=idle, banned=banned)


def _tie_key(cost: float, limb: str, drum: str, graph: KitGraph) -> tuple:
    """Closer first; on a tie, the stick on the same side of the kit as the drum."""
    y = float(graph.positions[graph.index[drum], 1])
    same_side = 0 if (limb == LIMB_LEFT and y >= 0.0) or (limb == LIMB_RIGHT and y < 0.0) else 1
    limb_rank = 0 if limb == LIMB_LEFT else 1
    return (cost, same_side, limb_rank)


def _best_matching(
    drums: list[str],
    parked: dict[str, str | None],
    graph: KitGraph,
) -> tuple[float, list[tuple[str, str]]]:
    """Min-cost assignment of ``drums`` (1 or 2) to the two sticks.

    Both sticks leave from ``parked`` (the start-of-beat rests).  A greedy
    "closest pair first" walk is wrong here: a stick already on the snare
    has travel 0, so it keeps the snare and the other stick is forced onto
    the crash even when that crash is on the far side of the kit.
    """
    drums = [d for d in drums if d in graph.index]
    if not drums:
        return _INF, []
    if len(drums) == 1:
        d = drums[0]
        best_key: tuple | None = None
        pick: tuple[str, str] | None = None
        for limb in STICK_LIMBS:
            cost = graph.travel(limb, parked[limb], d)
            key = _tie_key(cost, limb, d, graph)
            if best_key is None or key < best_key:
                best_key = key
                pick = (d, limb)
        if pick is None or best_key is None or best_key[0] >= _INF / 2:
            return _INF, []
        return float(best_key[0]), [pick]
    d0, d1 = drums[0], drums[1]
    options: list[tuple[float, list[tuple[str, str]]]] = []
    for assign in (
        ((d0, LIMB_LEFT), (d1, LIMB_RIGHT)),
        ((d0, LIMB_RIGHT), (d1, LIMB_LEFT)),
    ):
        cost = 0.0
        ok = True
        for drum, limb in assign:
            c = graph.travel(limb, parked[limb], drum)
            if c >= _INF / 2:
                ok = False
                break
            cost += c
        if ok:
            options.append((cost, list(assign)))
    if not options:
        return _INF, []
    options.sort(key=lambda item: item[0])
    return options[0]


def assign_simultaneous(
    drums: list[str],
    parked: dict[str, str | None],
    graph: KitGraph,
    *,
    max_stick: int = 2,
) -> list[tuple[str, str]]:
    """Min-cost stick matching for one simultaneous group.

    Stick parks are not updated until the whole group is done.  Extra
    hand-drums beyond ``max_stick`` are dropped by keeping the pair (or
    single) with the lowest matching cost.
    """
    remaining = [d for d in drums if d in graph.index]
    k = min(int(max_stick), 2, len(remaining))
    if k <= 0:
        return []
    if k == 1:
        best_cost = _INF
        best_out: list[tuple[str, str]] = []
        for drum in remaining:
            cost, out = _best_matching([drum], parked, graph)
            if cost < best_cost:
                best_cost, best_out = cost, out
        return best_out
    best_cost = _INF
    best_out: list[tuple[str, str]] = []
    pairs = combinations(remaining, 2) if len(remaining) > 2 else (tuple(remaining),)
    for pair in pairs:
        cost, out = _best_matching(list(pair), parked, graph)
        if cost < best_cost:
            best_cost, best_out = cost, out
    return best_out


def assign_rcc_groups(
    groups: list[dict[str, Any]],
    graph: KitGraph | None = None,
    *,
    max_stick: int = 2,
) -> list[dict[str, Any]]:
    """Rewrite ``limb`` on every stick target, walking the clip in time.

    Kicks stay ``right_foot`` and do not occupy a stick.  Groups are copied;
    ``max_stick`` extra hand-drums in one group are dropped (two arms).
    """
    graph = graph or kit_graph_from_layout()
    parked: dict[str, str | None] = {LIMB_LEFT: None, LIMB_RIGHT: None}
    out: list[dict[str, Any]] = []
    for group in groups:
        targets = list(group.get("targets") or [])
        kicks = [t for t in targets if str(t.get("drum")) == "kick"]
        stick = [t for t in targets if str(t.get("drum")) != "kick"]
        # Highest-velocity hit wins if a drum is duplicated in the group.
        by_drum: dict[str, dict[str, Any]] = {}
        for t in stick:
            name = str(t["drum"])
            prev = by_drum.get(name)
            if prev is None or float(t.get("velocity", 0.0)) > float(prev.get("velocity", 0.0)):
                by_drum[name] = t
        assigned = assign_simultaneous(list(by_drum.keys()), parked, graph, max_stick=max_stick)
        new_targets: list[dict[str, Any]] = []
        for t in kicks:
            nt = dict(t)
            nt["limb"] = LIMB_FOOT
            new_targets.append(nt)
        for drum, limb in assigned:
            nt = dict(by_drum[drum])
            nt["limb"] = limb
            new_targets.append(nt)
            parked[limb] = drum
        new_targets.sort(key=lambda t: (0 if t["drum"] == "kick" else 1, str(t["drum"])))
        if new_targets:
            ng = {k: v for k, v in group.items() if k != "targets"}
            ng["targets"] = new_targets
            out.append(ng)
    return out


def reassign_rcc_payload(
    payload: dict[str, Any],
    graph: KitGraph | None = None,
) -> dict[str, Any]:
    """Return a copy of an RCC JSON object with stick limbs rewritten."""
    graph = graph or kit_graph_from_layout()
    data = copy.deepcopy(payload)
    groups = assign_rcc_groups(list(data.get("groups") or []), graph)
    data["groups"] = groups
    meta = dict(data.get("metadata") or {})
    meta["limb_assignment"] = "layout_graph_mincost"
    counts: dict[str, int] = {}
    n = 0
    for g in groups:
        for t in g.get("targets") or []:
            n += 1
            name = str(t["drum"])
            counts[name] = counts.get(name, 0) + 1
    if "event_count" in meta:
        meta["event_count"] = n
    if "drum_counts" in meta:
        meta["drum_counts"] = counts
    data["metadata"] = meta
    return data


def reassign_rcc_file(
    path: str | Path,
    graph: KitGraph | None = None,
    *,
    dry_run: bool = False,
) -> bool:
    """Rewrite stick limbs in place. Returns True if any limb changed."""
    path = Path(path)
    original = json.loads(path.read_text(encoding="utf-8"))
    updated = reassign_rcc_payload(original, graph)
    old = [
        (g["time"], t["drum"], t["limb"])
        for g in original.get("groups", [])
        for t in g.get("targets", [])
    ]
    new = [
        (g["time"], t["drum"], t["limb"])
        for g in updated.get("groups", [])
        for t in g.get("targets", [])
    ]
    changed = old != new
    if changed and not dry_run:
        path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return changed
