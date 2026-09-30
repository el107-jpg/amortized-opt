"""Compact, model-independent feasibility witnesses indexed by state voxels.

The pool records positive existential evidence only: a key is feasible after at
least one trajectory from that cell, for one stored *paired* hit schedule, has
passed the hard feasibility gate.  A failed proposal is therefore unknown, not
infeasible, and never removes an existing witness.

``VoxelKey`` is deliberately trajectory-free.  Exact hit pairs live on the
cell because two individually tight hand times must never be combined unless a
single trajectory certified that complete pair.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF, PELVIS_IN_ENV
from .cost import StrikeContext
from .frontier import N_CHART, N_MODE
from .strike_seed import strike_seed
from .tokenizer import ContextInputs, build_waypoint_masks


HitPair = tuple[int, int]
CACHE_FORMAT_VERSION = 1
PROBE_CACHE_FORMAT_VERSION = 2
DEFAULT_QD_EDGES = (-0.5, 0.0, 0.5)
MAX_CERTIFIED_HIT_PAIRS = 32
MAX_CERTIFIED_LONGER_LAYERS = 4

__all__ = [
    "BootstrapCandidate",
    "CACHE_FORMAT_VERSION",
    "ChartTemplate",
    "ExpansionProposal",
    "HitPair",
    "ProbeCandidateKey",
    "ProbeBuffer",
    "ProbeRecord",
    "VoxelCell",
    "VoxelKey",
    "VoxelPool",
    "chart_modes",
    "gate_definition_fingerprint",
    "ingest_certified_candidates",
    "ordinary_slice_batches",
    "ordinary_bootstrap_candidate",
    "post_hit_bootstrap_candidate",
    "rest_bootstrap_candidate",
    "validate_hit_pair",
]


def _ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def chart_modes(chart_id: int) -> tuple[int, int]:
    """Return ``(left_mode, right_mode)`` for a chart in ``0..63``."""
    chart = int(chart_id)
    if not 0 <= chart < N_CHART:
        raise ValueError(f"chart_id must be in [0, {N_CHART}), got {chart}")
    return divmod(chart, N_MODE)


def validate_hit_pair(chart_id: int, hit_pair: Sequence[int]) -> HitPair:
    """Validate active positive hits and the ``-1`` idle sentinel."""
    if len(hit_pair) != 2:
        raise ValueError(f"hit_pair must have two entries, got {len(hit_pair)}")
    pair = (int(hit_pair[0]), int(hit_pair[1]))
    modes = chart_modes(chart_id)
    for hand, (mode, hit) in enumerate(zip(modes, pair)):
        if mode == 0 and hit != -1:
            raise ValueError(f"idle hand {hand} must use hit=-1, got {hit}")
        if mode != 0 and hit < 1:
            raise ValueError(f"active hand {hand} must use hit>=1, got {hit}")
    return pair


def gate_definition_fingerprint(definition: Mapping[str, Any] | str) -> str:
    """Stable identifier for the exact hard-feasibility definition.

    A voxel cache is positive evidence only while the robot, collision scene,
    limits, and numerical tolerances used by its gate are unchanged.  Callers
    should therefore pass a mapping containing those values rather than a
    human label alone.  JSON canonicalisation makes ordering irrelevant.
    """
    if isinstance(definition, str):
        payload = definition
    else:
        payload = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_state(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    return {"dtype": arr.dtype.str, "shape": list(arr.shape), "data": arr.tolist()}


def _array_from_state(state: Mapping[str, Any]) -> np.ndarray:
    arr = np.asarray(state["data"], dtype=np.dtype(str(state["dtype"])))
    shape = tuple(int(x) for x in state["shape"])
    return arr.reshape(shape)


def _row_numpy(value: torch.Tensor, row: int) -> np.ndarray:
    return value[int(row)].detach().cpu().numpy().copy()


@dataclass
class ChartTemplate:
    """Compact task/geometry payload shared by cells of one chart and bucket.

    Arrays are unbatched.  Timing is intentionally absent: complete hit pairs
    remain attached to positive witnesses and are selected atomically during
    materialisation.  Future-token arrays may be copied from a legacy row, but
    :meth:`VoxelPool.materialize_batch` always clears them.
    """

    chart_id: int
    bucket: int
    terminal: np.ndarray
    context: dict[str, np.ndarray]
    strike: dict[str, np.ndarray | None]
    # Deprecated Phase2Pool schema filler.  Voxel storage has no rest/post-hit
    # kind taxonomy; chart_id is the only task category, so this is canonical 0.
    kind: int = 0

    def __post_init__(self) -> None:
        chart_modes(self.chart_id)
        self.chart_id = int(self.chart_id)
        self.bucket = int(self.bucket)
        if self.bucket <= 0:
            raise ValueError("template bucket must be positive")
        self.terminal = np.asarray(self.terminal).reshape(-1).copy()
        if self.terminal.size != N_DOF or not np.isfinite(self.terminal).all():
            raise ValueError(f"template terminal must be finite ({N_DOF},)")
        self.context = {str(k): np.asarray(v).copy() for k, v in self.context.items()}
        self.strike = {
            str(k): None if v is None else np.asarray(v).copy()
            for k, v in self.strike.items()
        }
        self.kind = 0

        required_ctx = {f.name for f in fields(ContextInputs)}
        missing_ctx = required_ctx.difference(self.context)
        if missing_ctx:
            raise ValueError(f"template missing context fields: {sorted(missing_ctx)}")
        required_strike = {f.name for f in fields(StrikeContext)}
        missing_strike = required_strike.difference(self.strike)
        if missing_strike:
            raise ValueError(f"template missing strike fields: {sorted(missing_strike)}")

        # Timing belongs to the certified VoxelCell witness, never the shared
        # chart template.  Likewise, future hits and torso-reference paths are
        # outside the current-goal-only voxel contract.  Canonicalising here
        # prevents a cache from preserving stale values that a future caller
        # might accidentally consume before materialisation overwrites them.
        self.context["goal_tth"] = np.zeros_like(self.context["goal_tth"])
        for name in ("future_drum", "future_pos", "future_vel", "future_gap"):
            self.context[name] = np.zeros_like(self.context[name])
        self.context["future_valid"] = np.zeros_like(
            self.context["future_valid"], dtype=bool,
        )
        self.strike["hit_index"] = np.zeros_like(self.strike["hit_index"])
        self.strike["waist_yaw_ref"] = None
        self.strike["waist_pr_ref"] = None

        for owner, arrays in (("context", self.context), ("strike", self.strike)):
            for name, value in arrays.items():
                if value is not None and np.issubdtype(value.dtype, np.floating):
                    if not np.isfinite(value).all():
                        raise ValueError(f"template {owner}.{name} must be finite")

        modes = chart_modes(self.chart_id)
        expected = np.asarray([mode != 0 for mode in modes], dtype=bool)
        actual = np.asarray(self.context["goal_valid"], dtype=bool).reshape(2)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"template goal_valid {actual.tolist()} disagrees with chart {self.chart_id}"
            )
        idle_valid = np.asarray(self.context["idle_valid"], dtype=bool).reshape(2)
        if not np.array_equal(idle_valid, ~expected):
            raise ValueError("template idle_valid must equal ~goal_valid")
        if bool((~expected).any()):
            idle_cyl = np.asarray(self.context["idle_cyl"]).reshape(2, -1)
            if idle_cyl.shape[1] < 8 or np.any(idle_cyl[~expected, -2:] <= 0.0):
                raise ValueError("idle chart hands require a non-empty idle cylinder")
        goal_drum = np.asarray(self.context["goal_drum"], dtype=np.int64).reshape(2)
        expected_drum = np.asarray([max(mode - 1, 0) for mode in modes], dtype=np.int64)
        if np.any(expected & (goal_drum != expected_drum)):
            raise ValueError("template goal_drum disagrees with chart modes")
        strike_active = np.asarray(self.strike["active"], dtype=bool).reshape(2)
        if not np.array_equal(strike_active, expected):
            raise ValueError("template strike.active disagrees with chart modes")
        strike_drum = np.asarray(self.strike["target_drum"], dtype=np.int64).reshape(2)
        if np.any(expected & (strike_drum != expected_drum)):
            raise ValueError("template strike.target_drum disagrees with chart modes")
        goal_pos = np.asarray(self.context["goal_pos"]).reshape(2, 3)
        p_star = np.asarray(self.strike["p_star"]).reshape(2, 3)
        if np.any(expected) and not np.allclose(
            goal_pos[expected] - np.asarray(PELVIS_IN_ENV), p_star[expected],
            rtol=0.0, atol=1.0e-6,
        ):
            raise ValueError("template context goal_pos and strike.p_star disagree")
        n_out = np.asarray(self.strike["n_out"]).reshape(2, 3)
        if np.any(expected):
            norm = np.linalg.norm(n_out[expected], axis=-1)
            if not np.allclose(norm, 1.0, rtol=0.0, atol=1.0e-5):
                raise ValueError("active strike normals must be unit vectors")

    @classmethod
    def from_batch(
        cls,
        batch: Mapping[str, Any],
        row: int,
        *,
        chart_id: int | None = None,
        terminal: np.ndarray | torch.Tensor | None = None,
    ) -> "ChartTemplate":
        """Extract one unbatched template from a Phase2Pool-shaped batch."""
        ctx = batch["context"]
        strike = batch["strike"]
        active = batch["active"][int(row)].detach().cpu().to(torch.bool)
        if chart_id is None:
            drums = ctx.goal_drum[int(row)].detach().cpu().to(torch.long)
            modes = torch.where(active, drums + 1, torch.zeros_like(drums))
            chart_id = int(modes[0]) * N_MODE + int(modes[1])
        if terminal is None:
            terminal = batch["seed"][int(row), -1]
        term = (
            terminal.detach().cpu().numpy() if torch.is_tensor(terminal)
            else np.asarray(terminal)
        )
        context = {
            f.name: _row_numpy(getattr(ctx, f.name), row)
            for f in fields(ContextInputs)
        }
        strike_arrays: dict[str, np.ndarray | None] = {}
        for f in fields(StrikeContext):
            value = getattr(strike, f.name)
            strike_arrays[f.name] = None if value is None else _row_numpy(value, row)
        kind_value = batch.get("kind", 0)
        kind = int(kind_value[int(row)]) if torch.is_tensor(kind_value) else int(kind_value)
        return cls(
            chart_id=int(chart_id), bucket=int(batch["bucket"]), terminal=term,
            context=context, strike=strike_arrays, kind=kind,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "bucket": self.bucket,
            "terminal": _array_state(self.terminal),
            "context": {k: _array_state(v) for k, v in self.context.items()},
            "strike": {
                k: None if v is None else _array_state(v) for k, v in self.strike.items()
            },
            "kind": self.kind,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ChartTemplate":
        return cls(
            chart_id=int(state["chart_id"]), bucket=int(state["bucket"]),
            terminal=_array_from_state(state["terminal"]),
            context={k: _array_from_state(v) for k, v in state["context"].items()},
            strike={
                k: None if v is None else _array_from_state(v)
                for k, v in state["strike"].items()
            },
            kind=int(state.get("kind", 0)),
        )


@dataclass(frozen=True, order=True)
class VoxelKey:
    """Hashable state cell and current dual-goal chart."""

    chart_id: int
    bucket: int
    q0_cell: tuple[int, ...]
    qd_bin: tuple[int, ...]

    def __post_init__(self) -> None:
        chart_modes(self.chart_id)
        if int(self.bucket) <= 0:
            raise ValueError(f"bucket must be positive, got {self.bucket}")
        if len(self.q0_cell) != N_DOF:
            raise ValueError(f"q0_cell must have {N_DOF} entries")
        if len(self.qd_bin) != N_DOF:
            raise ValueError(f"qd_bin must have {N_DOF} entries")
        object.__setattr__(self, "chart_id", int(self.chart_id))
        object.__setattr__(self, "bucket", int(self.bucket))
        object.__setattr__(self, "q0_cell", _ints(self.q0_cell))
        object.__setattr__(self, "qd_bin", _ints(self.qd_bin))

    @classmethod
    def from_state(
        cls,
        *,
        chart_id: int,
        bucket: int,
        q0: Sequence[float],
        qd0: Sequence[float],
        q_cell_rad: float = 0.02,
        qd_edges: Sequence[float] = (-0.5, 0.0, 0.5),
    ) -> "VoxelKey":
        """Quantize one state; three ordered edges produce four velocity bins."""
        q = np.asarray(q0, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd0, dtype=np.float64).reshape(-1)
        if q.size != N_DOF or qd.size != N_DOF:
            raise ValueError(f"q0 and qd0 must each have {N_DOF} entries")
        width = float(q_cell_rad)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"q_cell_rad must be finite and positive, got {width}")
        edges = np.asarray(qd_edges, dtype=np.float64).reshape(-1)
        if (
            edges.size != 3
            or not np.isfinite(edges).all()
            or not np.all(edges[1:] > edges[:-1])
        ):
            raise ValueError("qd_edges must contain three finite, strictly increasing values")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("cannot voxelize a non-finite state")
        return cls(
            chart_id=int(chart_id),
            bucket=int(bucket),
            q0_cell=tuple(np.floor(q / width).astype(np.int64).tolist()),
            qd_bin=tuple(
                np.searchsorted(edges, qd, side="right").astype(np.int64).tolist()
            ),
        )


@dataclass
class VoxelCell:
    """Minimal positive evidence attached to one :class:`VoxelKey`."""

    feas_margin: float = 0.0
    sources: set[str] = field(default_factory=set)
    hit_pairs: list[HitPair] = field(default_factory=list)
    ingress_step: int = -1
    finishing_hand: int = -1
    pair_finishing_hands: dict[HitPair, int] = field(default_factory=dict)

    def add_hit_pair(
        self, chart_id: int, hit_pair: Sequence[int], *, finishing_hand: int = -1,
    ) -> bool:
        """Add one complete certified pair without synthesizing per-hand minima.

        Dominated schedules are redundant and may be dropped, but a new pair is
        always compared as a whole.  The implementation never constructs a
        componentwise minimum from two different witnesses.
        """
        pair = validate_hit_pair(chart_id, hit_pair)
        if int(finishing_hand) not in (-1, 0, 1):
            raise ValueError("finishing_hand must be -1, 0, or 1")
        if pair in self.hit_pairs:
            if finishing_hand >= 0:
                self.pair_finishing_hands[pair] = int(finishing_hand)
            return False
        active = tuple(mode != 0 for mode in chart_modes(chart_id))

        def dominates(a: HitPair, b: HitPair) -> bool:
            return all((not active[h]) or a[h] <= b[h] for h in range(2))

        self.hit_pairs.append(pair)
        self.pair_finishing_hands[pair] = int(finishing_hand)

        # Keep the fast Pareto frontier plus a four-knot certified interior
        # behind it.  Those slower witnesses are deliberately retained: they
        # are the thick, easier side of timing exploration and useful training
        # examples, even though they do not improve the minimum-time frontier.
        pairs = list(self.hit_pairs)
        frontier = [
            candidate for candidate in pairs
            if not any(
                other != candidate and dominates(other, candidate)
                for other in pairs
            )
        ]

        def within_longer_shell(candidate: HitPair) -> bool:
            for fast in frontier:
                deltas = [
                    candidate[h] - fast[h]
                    for h, mode in enumerate(active) if mode
                ]
                if (
                    deltas and min(deltas) >= 0
                    and max(deltas) <= MAX_CERTIFIED_LONGER_LAYERS
                ):
                    return True
            return False

        shell = [
            candidate for candidate in pairs
            if candidate not in frontier and within_longer_shell(candidate)
        ]
        order_key = lambda candidate: (
            sum(x for x in candidate if x >= 0), candidate,
        )
        ordered = (
            sorted(frontier, key=order_key) + sorted(shell, key=order_key)
        )[:MAX_CERTIFIED_HIT_PAIRS]
        self.hit_pairs = ordered
        self.pair_finishing_hands = {
            candidate: self.pair_finishing_hands.get(candidate, -1)
            for candidate in ordered
        }
        return pair in self.hit_pairs

    def finishing_hand_for(self, hit_pair: Sequence[int]) -> int:
        pair = tuple(int(x) for x in hit_pair)
        return int(self.pair_finishing_hands.get(pair, self.finishing_hand))

    def timing_frontier(self, chart_id: int) -> tuple[HitPair, ...]:
        """Return complete timing pairs not dominated by another witness."""
        active = tuple(mode != 0 for mode in chart_modes(chart_id))

        def dominates(a: HitPair, b: HitPair) -> bool:
            return all((not active[h]) or a[h] <= b[h] for h in range(2))

        return tuple(
            candidate for candidate in self.hit_pairs
            if not any(
                other != candidate and dominates(other, candidate)
                for other in self.hit_pairs
            )
        )


@dataclass(frozen=True)
class ExpansionProposal:
    """Uncertified spatial or timing-face proposal from a positive witness."""

    parent: VoxelKey
    key: VoxelKey
    parent_hit_pair: HitPair
    hit_pair: HitPair
    finishing_hand: int = -1


class VoxelPool:
    """Positive feasibility witnesses plus their unexplored coverage frontier."""

    def __init__(
        self, *, q_cell_rad: float = 0.02,
        qd_edges: Sequence[float] = DEFAULT_QD_EDGES,
    ):
        if not np.isfinite(q_cell_rad) or float(q_cell_rad) <= 0.0:
            raise ValueError("q_cell_rad must be finite and positive")
        edges = np.asarray(qd_edges, dtype=np.float64).reshape(-1)
        if (
            edges.size != 3 or not np.isfinite(edges).all()
            or not np.all(edges[1:] > edges[:-1])
        ):
            raise ValueError("qd_edges must contain three finite, increasing values")
        self.q_cell_rad = float(q_cell_rad)
        self.qd_edges = tuple(float(x) for x in edges)
        self._feasible: dict[VoxelKey, VoxelCell] = {}
        self._boundary: set[VoxelKey] = set()
        self._by_chart: dict[int, set[VoxelKey]] = {}
        self._by_source: dict[str, set[VoxelKey]] = {}
        self._chart_templates: dict[tuple[int, int], ChartTemplate] = {}

    def __len__(self) -> int:
        return len(self._feasible)

    def __contains__(self, key: VoxelKey) -> bool:
        return key in self._feasible

    def cell(self, key: VoxelKey) -> VoxelCell:
        return self._feasible[key]

    def keys(
        self, *, bucket: int | None = None, boundary: bool = False,
    ) -> tuple[VoxelKey, ...]:
        """Return a deterministic snapshot of occupied cells.

        ``boundary=True`` restricts the result to occupied cells with at least
        one unknown positional face neighbour.  Expansion from those parents
        includes pairwise diagonals too; keeping parent bookkeeping face-based
        avoids a 578-neighbour update on every bootstrap ingress.
        """
        source = self._boundary if boundary else self._feasible.keys()
        keys = source if bucket is None else (k for k in source if k.bucket == int(bucket))
        return tuple(sorted(keys))

    def unknown_face_neighbors(self, key: VoxelKey) -> tuple[VoxelKey, ...]:
        """Unknown axis-aligned positional neighbours of one positive witness."""
        return tuple(n for n in self.face_neighbors(key) if n not in self._feasible)

    def positional_neighbors(
        self, key: VoxelKey, *, max_changed_joints: int = 2,
    ) -> Iterator[VoxelKey]:
        """Yield adjacent positional cells through faces and pairwise diagonals.

        The full Moore neighbourhood has ``3**N_DOF - 1`` entries and is not a
        usable stencil for a 17-DoF state.  Changing at most two joints includes
        every face and edge neighbour (34 + 544 cells) while keeping expansion
        and boundary maintenance bounded.
        """
        order = int(max_changed_joints)
        if order < 1 or order > 2:
            raise ValueError("max_changed_joints must be 1 or 2")
        yield from self.face_neighbors(key)
        if order == 1:
            return
        for joint_a, joint_b in itertools.combinations(range(N_DOF), 2):
            for delta_a in (-1, 1):
                for delta_b in (-1, 1):
                    q = list(key.q0_cell)
                    q[joint_a] += delta_a
                    q[joint_b] += delta_b
                    yield VoxelKey(key.chart_id, key.bucket, tuple(q), key.qd_bin)

    def unknown_positional_neighbors(self, key: VoxelKey) -> tuple[VoxelKey, ...]:
        """Unknown face and pairwise-diagonal positional neighbours."""
        return tuple(
            neighbor for neighbor in self.positional_neighbors(key)
            if neighbor not in self._feasible
        )

    def expand_at(
        self,
        key: VoxelKey,
        *,
        delta_hit: int = 1,
        shorter_layers: int = 1,
        longer_layers: int = 4,
        max_proposals: int | None = None,
    ) -> tuple[ExpansionProposal, ...]:
        """Return uncertified spatial and asymmetric timing-face proposals.

        This method never mutates feasibility.  Every returned proposal must be
        materialised and hard-gated before :meth:`ingest` is called.  Timing
        exploration is asymmetric: by default it proposes one layer
        toward a shorter time-to-hit and four layers toward a longer time.  A
        post-hit witness moves only its recorded finishing hand; an ordinary
        dual-hit witness shifts both active hands together so their relative
        separation and complete-pair provenance remain intact.  Spatial
        neighbours retain the certified parent timing; timing faces retain the
        parent state voxel, avoiding diagonal jumps in state-by-timing space.
        """
        if key not in self._feasible:
            raise KeyError("cannot expand an unknown voxel")
        if max_proposals is not None and int(max_proposals) < 0:
            raise ValueError("max_proposals must be nonnegative or None")
        if int(delta_hit) <= 0:
            raise ValueError("delta_hit must be positive")
        if int(shorter_layers) < 0 or int(longer_layers) < 0:
            raise ValueError("timing layer counts must be nonnegative")
        neighbors = self.unknown_positional_neighbors(key)
        proposals: list[ExpansionProposal] = []
        cell = self.cell(key)
        timing_frontier = set(cell.timing_frontier(key.chart_id))
        for parent_pair in cell.hit_pairs:
            hand = cell.finishing_hand_for(parent_pair)
            for neighbor in neighbors:
                proposals.append(ExpansionProposal(
                    parent=key, key=neighbor, parent_hit_pair=parent_pair,
                    hit_pair=parent_pair, finishing_hand=hand,
                ))
            variants = (
                self.timing_probe_pairs(
                    key, parent_pair, delta_hit=delta_hit,
                    shorter_layers=shorter_layers, longer_layers=longer_layers,
                )
                if parent_pair in timing_frontier else ()
            )
            for pair in variants:
                if pair not in cell.hit_pairs:
                    proposals.append(ExpansionProposal(
                        parent=key, key=key, parent_hit_pair=parent_pair,
                        hit_pair=pair, finishing_hand=hand,
                    ))
        if max_proposals is not None:
            proposals = proposals[:int(max_proposals)]
        return tuple(proposals)

    def timing_probe_pairs(
        self,
        key: VoxelKey,
        hit_pair: Sequence[int],
        *,
        delta_hit: int = 1,
        shorter_layers: int = 1,
        longer_layers: int = 4,
    ) -> tuple[HitPair, ...]:
        """Return bounded, atomic shorter/longer timing candidates."""
        pair = validate_hit_pair(key.chart_id, hit_pair)
        if key not in self._feasible:
            raise KeyError("cannot expand timing from an unknown voxel")
        step = int(delta_hit)
        if step <= 0:
            raise ValueError("delta_hit must be positive")
        if int(shorter_layers) < 0 or int(longer_layers) < 0:
            raise ValueError("timing layer counts must be nonnegative")
        modes = chart_modes(key.chart_id)
        active_hands = [h for h, mode in enumerate(modes) if mode != 0]
        if not active_hands:
            return (pair,)
        finishing = self.cell(key).finishing_hand_for(pair)
        moved = [finishing] if finishing in (0, 1) else active_hands
        variants: list[HitPair] = []
        offsets = [
            -layer * step for layer in range(1, int(shorter_layers) + 1)
        ] + [
            layer * step for layer in range(1, int(longer_layers) + 1)
        ]
        for offset in offsets:
            candidate = list(pair)
            for hand in moved:
                candidate[hand] += offset
            if any(candidate[hand] < 1 or candidate[hand] > key.bucket for hand in moved):
                continue
            complete = validate_hit_pair(key.chart_id, candidate)
            if complete != pair and complete not in variants:
                variants.append(complete)
        return tuple(variants)

    def ingest(
        self,
        key: VoxelKey,
        *,
        hit_pair: Sequence[int],
        feas_margin: float = 0.0,
        source: str = "bootstrap",
        ingress_step: int = -1,
        finishing_hand: int = -1,
    ) -> bool:
        """Record a hard-valid witness; return True only for a newly occupied cell."""
        pair = validate_hit_pair(key.chart_id, hit_pair)
        margin = float(feas_margin)
        if not np.isfinite(margin):
            raise ValueError("feas_margin must be finite")
        src = str(source)
        if not src:
            raise ValueError("source must be nonempty")
        if finishing_hand not in (-1, 0, 1):
            raise ValueError("finishing_hand must be -1, 0, or 1")

        new = key not in self._feasible
        if new:
            cell = VoxelCell(
                feas_margin=margin,
                sources={src},
                hit_pairs=[pair],
                ingress_step=int(ingress_step),
                finishing_hand=int(finishing_hand),
                pair_finishing_hands={pair: int(finishing_hand)},
            )
            self._feasible[key] = cell
            self._by_chart.setdefault(key.chart_id, set()).add(key)
        else:
            cell = self._feasible[key]
            cell.feas_margin = max(cell.feas_margin, margin)
            cell.sources.add(src)
            cell.add_hit_pair(
                key.chart_id, pair, finishing_hand=int(finishing_hand),
            )
            if ingress_step >= 0:
                cell.ingress_step = max(cell.ingress_step, int(ingress_step))
            if finishing_hand >= 0:
                cell.finishing_hand = int(finishing_hand)
        self._by_source.setdefault(src, set()).add(key)
        self._refresh_boundary_near(key)
        return new

    def face_neighbors(self, key: VoxelKey) -> Iterator[VoxelKey]:
        """Yield the 34 positional face-neighbor keys in deterministic order."""
        for joint in range(N_DOF):
            for delta in (-1, 1):
                q = list(key.q0_cell)
                q[joint] += delta
                yield VoxelKey(key.chart_id, key.bucket, tuple(q), key.qd_bin)

    def is_boundary(self, key: VoxelKey) -> bool:
        return key in self._boundary

    def _refresh_one_boundary(self, key: VoxelKey) -> None:
        if key not in self._feasible:
            self._boundary.discard(key)
            return
        if any(neighbor not in self._feasible for neighbor in self.face_neighbors(key)):
            self._boundary.add(key)
        else:
            self._boundary.discard(key)

    def _refresh_boundary_near(self, key: VoxelKey) -> None:
        self._refresh_one_boundary(key)
        for neighbor in self.face_neighbors(key):
            if neighbor in self._feasible:
                self._refresh_one_boundary(neighbor)

    def sample_q0(
        self,
        keys: Sequence[VoxelKey],
        *,
        rng: np.random.Generator,
        q_lo: float | Sequence[float] | None = None,
        q_hi: float | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Uniformly materialize positions inside each cell/bounds intersection."""
        if not keys:
            return np.empty((0, N_DOF), dtype=np.float32)
        cells = np.asarray([key.q0_cell for key in keys], dtype=np.float64)
        cell_lower = cells * self.q_cell_rad
        cell_upper = (cells + 1.0) * self.q_cell_rad
        lower = cell_lower
        upper = cell_upper
        if (q_lo is None) != (q_hi is None):
            raise ValueError("q_lo and q_hi must be supplied together")
        if q_lo is not None:
            limit_lo = np.broadcast_to(np.asarray(q_lo, dtype=np.float64), (N_DOF,))
            limit_hi = np.broadcast_to(np.asarray(q_hi, dtype=np.float64), (N_DOF,))
            if (
                not np.isfinite(limit_lo).all() or not np.isfinite(limit_hi).all()
                or np.any(limit_lo >= limit_hi)
            ):
                raise ValueError("q_lo/q_hi must be finite ordered joint bounds")
            # Joint boxes are closed while voxel cells are [lower, upper).
            # A cell whose lower face equals q_hi therefore has the one valid
            # boundary point q_hi; a cell whose upper face equals q_lo has no
            # point in common because that face belongs to the next cell.
            intersects = (
                (cell_lower <= limit_hi[None, :])
                & (cell_upper > limit_lo[None, :])
            )
            if not bool(intersects.all()):
                raise ValueError("voxel cell does not intersect the supplied joint bounds")
            lower = np.maximum(cell_lower, limit_lo)
            upper = np.minimum(cell_upper, limit_hi)
        q = lower + (upper - lower) * rng.random(cells.shape)
        return q.astype(np.float32)

    def q0_cell_intersects_bounds(
        self,
        key: VoxelKey,
        *,
        q_lo: float | Sequence[float],
        q_hi: float | Sequence[float],
    ) -> bool:
        """Whether a half-open position cell intersects a closed joint box."""
        limit_lo = np.broadcast_to(np.asarray(q_lo, dtype=np.float64), (N_DOF,))
        limit_hi = np.broadcast_to(np.asarray(q_hi, dtype=np.float64), (N_DOF,))
        if (
            not np.isfinite(limit_lo).all() or not np.isfinite(limit_hi).all()
            or np.any(limit_lo >= limit_hi)
        ):
            raise ValueError("q_lo/q_hi must be finite ordered joint bounds")
        cells = np.asarray(key.q0_cell, dtype=np.float64)
        cell_lower = cells * self.q_cell_rad
        cell_upper = (cells + 1.0) * self.q_cell_rad
        return bool(np.all((cell_lower <= limit_hi) & (cell_upper > limit_lo)))

    def sample_qd0(
        self,
        keys: Sequence[VoxelKey],
        *,
        rng: np.random.Generator,
        qd_lo: float | Sequence[float] = -1.0,
        qd_hi: float | Sequence[float] = 1.0,
    ) -> np.ndarray:
        """Uniformly sample each coarse velocity bin, including finite tails."""
        if not keys:
            return np.empty((0, N_DOF), dtype=np.float32)
        lo = np.broadcast_to(np.asarray(qd_lo, dtype=np.float64), (N_DOF,))
        hi = np.broadcast_to(np.asarray(qd_hi, dtype=np.float64), (N_DOF,))
        edges = np.asarray(self.qd_edges, dtype=np.float64)
        if (
            not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo >= hi) or np.any(lo > edges[0]) or np.any(hi < edges[-1])
        ):
            raise ValueError("qd_lo/qd_hi must be finite and enclose qd_edges")
        bins = np.asarray([key.qd_bin for key in keys], dtype=np.int64)
        if np.any((bins < 0) | (bins > len(edges))):
            raise ValueError("VoxelKey contains an invalid velocity bin")
        boundaries = np.concatenate([lo[:, None], np.broadcast_to(edges, (N_DOF, 3)), hi[:, None]], axis=1)
        lower = np.take_along_axis(boundaries[None], bins[:, :, None], axis=2).squeeze(2)
        upper = np.take_along_axis(boundaries[None], (bins + 1)[:, :, None], axis=2).squeeze(2)
        return (lower + (upper - lower) * rng.random(bins.shape)).astype(np.float32)

    def register_template(self, template: ChartTemplate, *, replace: bool = False) -> None:
        """Install the compact template for ``(chart_id, bucket)``."""
        slot = (int(template.chart_id), int(template.bucket))
        if slot in self._chart_templates and not replace:
            return
        self._chart_templates[slot] = template

    def has_template(self, chart_id: int, bucket: int) -> bool:
        return (int(chart_id), int(bucket)) in self._chart_templates

    def template(self, chart_id: int, bucket: int) -> ChartTemplate:
        return self._chart_templates[(int(chart_id), int(bucket))]

    def tightened_hit_pair(
        self, key: VoxelKey, hit_pair: Sequence[int], *, delta_hit: int = 1,
    ) -> HitPair:
        """Propose a tighter post-hit schedule without certifying it.

        Only the finishing hand moves.  The paired schedule is returned as one
        value and may be inserted only after a new trajectory passes the hard
        gate.  Ordinary cells (no finishing-hand metadata) are rejected.
        """
        pair = validate_hit_pair(key.chart_id, hit_pair)
        hand = self.cell(key).finishing_hand_for(pair)
        if hand not in (0, 1):
            raise ValueError("timing tightening requires post-hit finishing-hand metadata")
        delta = int(delta_hit)
        if delta <= 0:
            raise ValueError("delta_hit must be positive")
        out = list(pair)
        out[hand] = max(1, out[hand] - delta)
        return validate_hit_pair(key.chart_id, out)

    @staticmethod
    def _effective_hits(
        hit: torch.Tensor, active: torch.Tensor, horizon: int,
    ) -> torch.Tensor:
        active = active.to(torch.bool)
        neg = torch.full_like(hit[:, 0], -1)
        last = torch.maximum(
            torch.where(active[:, 0], hit[:, 0], neg),
            torch.where(active[:, 1], hit[:, 1], neg),
        )
        last = torch.where(active.any(dim=1), last, torch.full_like(last, int(horizon)))
        return torch.stack([
            torch.where(active[:, 0], hit[:, 0], last),
            torch.where(active[:, 1], hit[:, 1], last),
        ], dim=1)

    @staticmethod
    def _element_lengths(
        hit: torch.Tensor, active: torch.Tensor, horizon: int,
        joint_names: Sequence[str],
    ) -> torch.Tensor:
        eff = VoxelPool._effective_hits(hit, active, horizon)
        shared = torch.maximum(eff[:, 0], eff[:, 1])
        lengths = []
        for name in joint_names:
            lengths.append(
                eff[:, 0] if str(name).startswith("left_") else
                eff[:, 1] if str(name).startswith("right_") else shared
            )
        return torch.stack(lengths, dim=1)

    @staticmethod
    def _stack_template_field(
        templates: Sequence[ChartTemplate], owner: str, name: str,
        *, device: torch.device | str, dtype: torch.dtype,
    ) -> torch.Tensor | None:
        values = [getattr(t, owner)[name] for t in templates]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ValueError(f"mixed optional template field {owner}.{name}")
        tensor = torch.as_tensor(np.stack(values), device=device)
        return tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor

    def materialize_batch(
        self,
        keys: Sequence[VoxelKey],
        *,
        rng: np.random.Generator,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        joint_names: Sequence[str] = G1_UPPER_BODY_JOINTS,
        dt: float = CONTROL_DT,
        q_lo: float | Sequence[float] | None = None,
        q_hi: float | Sequence[float] | None = None,
        qd_lo: float | Sequence[float] = -1.0,
        qd_hi: float | Sequence[float] = 1.0,
        strike_speed_min: float = 1.0,
        strike_speed_max: float = 3.0,
        hit_pairs: Sequence[Sequence[int]] | None = None,
        q0_values: np.ndarray | torch.Tensor | None = None,
        qd0_values: np.ndarray | torch.Tensor | None = None,
        probe: bool = False,
    ) -> tuple[dict[str, Any], list[HitPair]]:
        """Build a Phase2Pool-shaped batch from compact voxel evidence.

        Positions and velocities are sampled independently within their cells;
        warm starts are regenerated from those exact states.  Hit pairs are
        selected atomically, futures are invalidated, and randomized strike
        speed is written to both encoder and cost context.
        """
        if not keys:
            raise ValueError("cannot materialize an empty voxel batch")
        horizon = int(keys[0].bucket)
        if any(int(key.bucket) != horizon for key in keys):
            raise ValueError("materialize_batch requires one horizon bucket")
        if len(tuple(joint_names)) != N_DOF:
            raise ValueError(f"joint_names must contain {N_DOF} entries")
        templates = [self.template(key.chart_id, key.bucket) for key in keys]
        for key in keys:
            if not probe and key not in self._feasible:
                raise KeyError("only probes may materialize an unknown voxel")

        if hit_pairs is None:
            if probe:
                raise ValueError("unknown probes require an explicit parent hit pair")
            selected = []
            for key in keys:
                pairs = self.cell(key).hit_pairs
                if not pairs:
                    raise RuntimeError(f"voxel {key} has no certified hit pair")
                selected.append(pairs[int(rng.integers(0, len(pairs)))])
        else:
            if len(hit_pairs) != len(keys):
                raise ValueError("one hit pair is required per key")
            selected = [validate_hit_pair(k.chart_id, p) for k, p in zip(keys, hit_pairs)]
            if not probe:
                for key, pair in zip(keys, selected):
                    if pair not in self.cell(key).hit_pairs:
                        raise ValueError("requested hit pair is not a certified witness")

        q0 = torch.as_tensor(
            self.sample_q0(keys, rng=rng, q_lo=q_lo, q_hi=q_hi)
            if q0_values is None else q0_values,
            device=device, dtype=dtype,
        )
        qd0 = torch.as_tensor(
            self.sample_qd0(keys, rng=rng, qd_lo=qd_lo, qd_hi=qd_hi)
            if qd0_values is None else qd0_values,
            device=device, dtype=dtype,
        )
        expected_shape = (len(keys), N_DOF)
        if tuple(q0.shape) != expected_shape or tuple(qd0.shape) != expected_shape:
            raise ValueError(
                f"q0_values/qd0_values must have shape {expected_shape}"
            )
        if not bool(torch.isfinite(q0).all()) or not bool(torch.isfinite(qd0).all()):
            raise ValueError("materialized q0/qd0 values must be finite")
        if q0_values is not None or qd0_values is not None:
            # Exact-state overrides are used by bootstrap collocations.  They
            # are safe only when they still describe the requested keys; an
            # unchecked override would attach a feasibility witness to the
            # wrong voxel.
            for row, key in enumerate(keys):
                actual = VoxelKey.from_state(
                    chart_id=key.chart_id, bucket=key.bucket,
                    q0=q0[row].detach().cpu().numpy(),
                    qd0=qd0[row].detach().cpu().numpy(),
                    q_cell_rad=self.q_cell_rad, qd_edges=self.qd_edges,
                )
                if actual != key:
                    raise ValueError("exact state override does not match VoxelKey")
        hit = torch.as_tensor(selected, device=device, dtype=torch.long)
        modes = torch.as_tensor(
            [chart_modes(key.chart_id) for key in keys], device=device, dtype=torch.long,
        )
        active = modes != 0
        target_drum = torch.where(active, modes - 1, torch.full_like(modes, -1))

        ctx_kwargs = {
            f.name: self._stack_template_field(
                templates, "context", f.name, device=device, dtype=dtype,
            )
            for f in fields(ContextInputs)
        }
        ctx = ContextInputs(**ctx_kwargs)
        ctx.goal_valid = active
        ctx.goal_drum = target_drum.clamp_min(0)
        ctx.goal_tth = torch.where(
            active, hit.clamp_min(0).to(dtype) * float(dt),
            torch.zeros_like(hit, dtype=dtype),
        )
        ctx.goal_pos = torch.where(
            active.unsqueeze(-1), ctx.goal_pos, torch.zeros_like(ctx.goal_pos),
        )
        for name in ("future_drum", "future_pos", "future_vel", "future_gap"):
            value = getattr(ctx, name)
            setattr(ctx, name, torch.zeros_like(value))
        ctx.future_valid = torch.zeros_like(ctx.future_valid, dtype=torch.bool)
        ctx.idle_valid = ~active
        ctx.idle_cyl = ctx.idle_cyl * ctx.idle_valid.unsqueeze(-1).to(dtype)
        ctx.yaw_now = q0[:, tuple(joint_names).index("waist_yaw_joint")]

        strike_kwargs = {
            f.name: self._stack_template_field(
                templates, "strike", f.name, device=device, dtype=dtype,
            )
            for f in fields(StrikeContext)
        }
        # Stale trajectory references are not part of a compact current-goal
        # task and would make a freshly regenerated warm start self-referential.
        strike_kwargs["waist_yaw_ref"] = None
        strike_kwargs["waist_pr_ref"] = None
        strike = StrikeContext(**strike_kwargs)
        strike.hit_index = hit.clamp_min(0)
        strike.active = active.to(dtype)
        strike.target_drum = target_drum

        speed_min, speed_max = float(strike_speed_min), float(strike_speed_max)
        if (
            not np.isfinite(speed_min) or not np.isfinite(speed_max)
            or speed_min < 0.0 or speed_min > speed_max
        ):
            raise ValueError("strike speed bounds must be finite, nonnegative, and ordered")
        speed = torch.as_tensor(
            rng.uniform(speed_min, speed_max, size=(len(keys), 2)),
            device=device, dtype=dtype,
        )
        randomized_velocity = -speed.unsqueeze(-1) * strike.n_out
        randomized_velocity = randomized_velocity * active.unsqueeze(-1).to(dtype)
        strike.v_star = randomized_velocity
        ctx.goal_vel = randomized_velocity

        # Bake analytic aim from current goals only.  The future list is empty
        # by construction, so no forward-looking context leaks into the token.
        from .waist_aim import aim_yaw

        aim = []
        goal_pos_pelvis = ctx.goal_pos.detach().cpu().numpy() - np.asarray(PELVIS_IN_ENV)
        gtth = ctx.goal_tth.detach().cpu().numpy()
        yaw_now = ctx.yaw_now.detach().cpu().numpy()
        for row in range(len(keys)):
            targets = {
                hand: goal_pos_pelvis[row, h]
                for h, hand in enumerate(("left", "right")) if bool(active[row, h])
            }
            times = {
                hand: float(gtth[row, h])
                for h, hand in enumerate(("left", "right")) if bool(active[row, h])
            }
            aim.append(aim_yaw(targets, yaw_current=float(yaw_now[row]), goal_tth=times, futures={}))
        ctx.aim_yaw = torch.as_tensor(aim, device=device, dtype=dtype)
        ctx.aim_valid = torch.ones(len(keys), device=device, dtype=torch.bool)

        terminal = torch.as_tensor(
            np.stack([template.terminal for template in templates]),
            device=device, dtype=dtype,
        )
        lengths = self._element_lengths(hit, active, horizon, joint_names)
        seed = strike_seed(
            q0, qd0, terminal, lengths, horizon, float(dt),
            active=active, joint_names=joint_names,
        )
        tth = hit.clamp_min(0).to(dtype) * float(dt)
        masks = build_waypoint_masks(
            tth[:, 0], tth[:, 1], max_horizon=horizon, dt=float(dt),
            active_left=active[:, 0], active_right=active[:, 1],
            joint_names=tuple(joint_names),
        )
        init = seed * masks.elem_mask
        kind = torch.as_tensor([t.kind for t in templates], device=device, dtype=torch.long)
        return {
            "bucket": horizon,
            "q0": q0,
            "qd0": qd0,
            "v0": qd0,
            "state": torch.cat([q0, qd0], dim=-1),
            "init": init,
            "seed": seed,
            "masks": masks,
            "context": ctx,
            "strike": strike,
            "hit": hit,
            "active": active,
            "kind": kind,
            "mpc_init": torch.zeros(len(keys), device=device, dtype=torch.bool),
            "al_ok": torch.full((len(keys),), not probe, device=device, dtype=torch.bool),
            "voxel_kind": torch.full(
                (len(keys),), 3 if probe else 1, device=device, dtype=torch.long,
            ),
        }, selected

    def state_dict(self, *, gate_definition: Mapping[str, Any] | str) -> dict[str, Any]:
        """JSON-safe persistent state, excluding the derived boundary index."""
        cells = []
        for key in self.keys():
            cell = self.cell(key)
            cells.append({
                "key": {
                    "chart_id": key.chart_id, "bucket": key.bucket,
                    "q0_cell": list(key.q0_cell), "qd_bin": list(key.qd_bin),
                },
                "feas_margin": cell.feas_margin,
                "sources": sorted(cell.sources),
                "hit_pairs": [list(pair) for pair in cell.hit_pairs],
                "ingress_step": cell.ingress_step,
                "finishing_hand": cell.finishing_hand,
                "pair_finishing_hands": [
                    [list(pair), int(hand)]
                    for pair, hand in sorted(cell.pair_finishing_hands.items())
                ],
            })
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "gate_fingerprint": gate_definition_fingerprint(gate_definition),
            "q_cell_rad": self.q_cell_rad,
            "qd_edges": list(self.qd_edges),
            "cells": cells,
            "templates": [
                self._chart_templates[slot].to_state()
                for slot in sorted(self._chart_templates)
            ],
        }

    @classmethod
    def load_state_dict(
        cls, state: Mapping[str, Any], *, expected_gate_definition: Mapping[str, Any] | str,
    ) -> "VoxelPool":
        if int(state.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError("unsupported voxel cache format version")
        expected = gate_definition_fingerprint(expected_gate_definition)
        if str(state.get("gate_fingerprint", "")) != expected:
            raise ValueError("voxel cache hard-gate fingerprint mismatch")
        pool = cls(q_cell_rad=float(state["q_cell_rad"]), qd_edges=state["qd_edges"])
        for template_state in state.get("templates", []):
            pool.register_template(ChartTemplate.from_state(template_state), replace=True)
        for item in state.get("cells", []):
            ks = item["key"]
            key = VoxelKey(
                ks["chart_id"], ks["bucket"], tuple(ks["q0_cell"]), tuple(ks["qd_bin"]),
            )
            pairs = item.get("hit_pairs", [])
            if not pairs:
                raise ValueError("cached feasible cell has no positive hit-pair witness")
            pair_hands = {
                tuple(int(x) for x in pair): int(hand)
                for pair, hand in item.get("pair_finishing_hands", [])
            }
            for i, pair in enumerate(pairs):
                sources = item.get("sources", ["bootstrap"])
                # Re-ingesting all sources rebuilds source indices; the first
                # call creates the cell and subsequent calls remain monotone.
                for source in sources:
                    pool.ingest(
                        key, hit_pair=pair, feas_margin=float(item.get("feas_margin", 0.0)),
                        source=source, ingress_step=int(item.get("ingress_step", -1)),
                        finishing_hand=pair_hands.get(
                            tuple(int(x) for x in pair), int(item.get("finishing_hand", -1)),
                        ),
                    )
        return pool

    def save_npz(
        self, path: str | Path, *, gate_definition: Mapping[str, Any] | str,
    ) -> None:
        """Write a pickle-free compressed cache."""
        payload = json.dumps(
            self.state_dict(gate_definition=gate_definition),
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        np.savez_compressed(Path(path), payload=np.asarray(payload))

    @classmethod
    def load_npz(
        cls, path: str | Path, *, expected_gate_definition: Mapping[str, Any] | str,
    ) -> "VoxelPool":
        with np.load(Path(path), allow_pickle=False) as data:
            payload = str(data["payload"].item())
        return cls.load_state_dict(
            json.loads(payload), expected_gate_definition=expected_gate_definition,
        )


@dataclass
class BootstrapCandidate:
    """Uncertified bootstrap proposal plus the trajectory to hard-check.

    Constructing this object does *not* change a :class:`VoxelPool`.  The
    trajectory must be evaluated by the dedicated hard gate and the resulting
    boolean passed to :func:`ingest_certified_candidates`.
    """

    key: VoxelKey
    hit_pair: HitPair
    template: ChartTemplate
    witness_trajectory: np.ndarray | torch.Tensor
    source: str = "bootstrap"
    finishing_hand: int = -1
    ingress_step: int = -1

    def __post_init__(self) -> None:
        self.hit_pair = validate_hit_pair(self.key.chart_id, self.hit_pair)
        if (self.template.chart_id, self.template.bucket) != (
            self.key.chart_id, self.key.bucket,
        ):
            raise ValueError("candidate template and voxel key disagree")
        if int(self.finishing_hand) not in (-1, 0, 1):
            raise ValueError("finishing_hand must be -1, 0, or 1")
        if not str(self.source):
            raise ValueError("candidate source must be nonempty")
        shape = tuple(self.witness_trajectory.shape)
        if shape != (int(self.key.bucket), N_DOF):
            raise ValueError(
                f"witness trajectory must have shape ({self.key.bucket}, {N_DOF})"
            )
        finite = (
            bool(torch.isfinite(self.witness_trajectory).all())
            if torch.is_tensor(self.witness_trajectory)
            else bool(np.isfinite(np.asarray(self.witness_trajectory)).all())
        )
        if not finite:
            raise ValueError("witness trajectory must be finite")


def ingest_certified_candidates(
    pool: VoxelPool,
    candidates: Sequence[BootstrapCandidate],
    hard_valid: Sequence[bool] | np.ndarray | torch.Tensor,
    *,
    feas_margins: Sequence[float] | None = None,
) -> int:
    """Ingest only explicitly hard-valid positive witnesses.

    False entries are ignored and remain unknown.  In particular there is no
    negative set and no path that removes an existing feasible cell.
    """
    if torch.is_tensor(hard_valid):
        if hard_valid.dtype != torch.bool:
            raise TypeError("hard_valid must be a boolean hard-gate result")
        valid = hard_valid.detach().cpu().numpy()
    else:
        raw_valid = np.asarray(hard_valid)
        if raw_valid.dtype != np.bool_:
            raise TypeError("hard_valid must be a boolean hard-gate result")
        valid = raw_valid
    valid = valid.reshape(-1)
    if valid.size != len(candidates):
        raise ValueError("hard_valid must have one entry per candidate")
    margins = (
        np.zeros(len(candidates), dtype=np.float64) if feas_margins is None
        else np.asarray(feas_margins, dtype=np.float64).reshape(-1)
    )
    if margins.size != len(candidates) or not np.isfinite(margins).all():
        raise ValueError("feas_margins must be finite and aligned")
    added = 0
    for candidate, ok, margin in zip(candidates, valid, margins):
        if not bool(ok):
            continue
        pool.register_template(candidate.template)
        added += int(pool.ingest(
            candidate.key, hit_pair=candidate.hit_pair,
            feas_margin=float(margin), source=candidate.source,
            ingress_step=int(candidate.ingress_step),
            finishing_hand=int(candidate.finishing_hand),
        ))
    return added


def ordinary_slice_batches(
    cols: dict[str, torch.Tensor],
    *,
    dt: float = CONTROL_DT,
    stride: int = 4,
    buckets: tuple[int, ...] = (25, 50, 100),
    include_k0: bool = True,
) -> tuple[tuple[int, dict[str, torch.Tensor]], ...]:
    """Build ordinary along-trajectory bootstrap proposals lazily.

    This returns candidate column batches only; it deliberately cannot ingest
    them.  The caller must price each suffix trajectory with the hard gate and
    then construct :class:`BootstrapCandidate` objects for positive rows.
    """
    from .frontier import slice_at_knot

    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    horizon = int(cols["init"].shape[1])
    out: list[tuple[int, dict[str, torch.Tensor]]] = []
    if include_k0:
        out.append((horizon, cols))
    for knot in range(int(stride), horizon + 1, int(stride)):
        for bucket, sliced in sorted(
            slice_at_knot(cols, knot, dt=float(dt), buckets=buckets).items()
        ):
            out.append((int(bucket), sliced))
    return tuple(out)


def ordinary_bootstrap_candidate(
    *,
    q0: Sequence[float],
    qd0: Sequence[float],
    template: ChartTemplate,
    hit_pair: Sequence[int],
    witness_trajectory: np.ndarray | torch.Tensor,
    q_cell_rad: float = 0.02,
    qd_edges: Sequence[float] = DEFAULT_QD_EDGES,
    source: str = "bootstrap",
) -> BootstrapCandidate:
    """Create an uncertified ordinary collocation proposal.

    As with the post-hit and rest helpers, creation is deliberately side-effect
    free; only :func:`ingest_certified_candidates` can turn the proposal into
    positive evidence after the dedicated hard gate returns a boolean result.
    """
    pair = validate_hit_pair(template.chart_id, hit_pair)
    key = VoxelKey.from_state(
        chart_id=template.chart_id, bucket=template.bucket, q0=q0, qd0=qd0,
        q_cell_rad=float(q_cell_rad), qd_edges=qd_edges,
    )
    return BootstrapCandidate(
        key=key, hit_pair=pair, template=template,
        witness_trajectory=witness_trajectory, source=source,
    )


def post_hit_bootstrap_candidate(
    *,
    q0: Sequence[float],
    qd0: Sequence[float],
    template: ChartTemplate,
    hit_pair: Sequence[int],
    finishing_hand: int,
    witness_trajectory: np.ndarray | torch.Tensor,
    anchor_tth_s: float = 1.0,
    dt: float = CONTROL_DT,
    q_cell_rad: float = 0.02,
    qd_edges: Sequence[float] = DEFAULT_QD_EDGES,
    source: str = "bootstrap",
) -> BootstrapCandidate:
    """Create an uncertified bounce/post-hit proposal.

    The finishing hand receives the conservative new-goal anchor.  The other
    hand's hit remains byte-for-byte unchanged; expansion may later tighten
    only the finishing hand via :meth:`VoxelPool.tightened_hit_pair`.
    """
    hand = int(finishing_hand)
    if hand not in (0, 1):
        raise ValueError("finishing_hand must be 0 or 1")
    if chart_modes(template.chart_id)[hand] == 0:
        raise ValueError("post-hit finishing hand must have a new active goal")
    anchor = int(round(float(anchor_tth_s) / float(dt)))
    if anchor < 1 or anchor > int(template.bucket):
        raise ValueError("post-hit anchor must lie within the template horizon")
    pair = list(validate_hit_pair(template.chart_id, hit_pair))
    pair[hand] = anchor
    pair_t = validate_hit_pair(template.chart_id, pair)
    key = VoxelKey.from_state(
        chart_id=template.chart_id, bucket=template.bucket, q0=q0, qd0=qd0,
        q_cell_rad=float(q_cell_rad), qd_edges=qd_edges,
    )
    return BootstrapCandidate(
        key=key, hit_pair=pair_t, template=template,
        witness_trajectory=witness_trajectory, source=source,
        finishing_hand=hand,
    )


def rest_bootstrap_candidate(
    *,
    q0: Sequence[float],
    qd0: Sequence[float],
    template: ChartTemplate,
    hit_pair: Sequence[int],
    witness_trajectory: np.ndarray | torch.Tensor,
    q_cell_rad: float = 0.02,
    qd_edges: Sequence[float] = DEFAULT_QD_EDGES,
    source: str = "bootstrap",
    gap_s: float = 1.0,
    min_gap_s: float = 1.0,
) -> BootstrapCandidate:
    """Create an uncertified long-gap/rest proposal in the unified taxonomy."""
    if all(mode != 0 for mode in chart_modes(template.chart_id)):
        raise ValueError("rest bootstrap requires at least one idle chart hand")
    if (
        not np.isfinite(gap_s) or not np.isfinite(min_gap_s)
        or float(min_gap_s) < 0.0 or float(gap_s) < float(min_gap_s)
    ):
        raise ValueError("rest bootstrap requires gap_s >= min_gap_s")
    pair = validate_hit_pair(template.chart_id, hit_pair)
    key = VoxelKey.from_state(
        chart_id=template.chart_id, bucket=template.bucket, q0=q0, qd0=qd0,
        q_cell_rad=float(q_cell_rad), qd_edges=qd_edges,
    )
    return BootstrapCandidate(
        key=key, hit_pair=pair, template=template,
        witness_trajectory=witness_trajectory, source=source,
    )


@dataclass(frozen=True, order=True)
class ProbeCandidateKey:
    """Identity of one unknown state-and-complete-timing candidate."""

    key: VoxelKey
    hit_pair: HitPair

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hit_pair", validate_hit_pair(self.key.chart_id, self.hit_pair),
        )


@dataclass
class ProbeRecord:
    candidate: ProbeCandidateKey
    created_step: int
    attempts: int = 0

    @property
    def key(self) -> VoxelKey:
        return self.candidate.key

    @property
    def hit_pair(self) -> HitPair:
        return self.candidate.hit_pair


class ProbeBuffer:
    """Bounded insertion-ordered set of unknown state/timing candidates.

    Failure increments ``attempts`` but deliberately retains the complete
    candidate as unknown.  Multiple atomic timing pairs may coexist for one
    spatial voxel.  Only a hard-valid witness promotes and removes one pair.
    """

    def __init__(self, capacity: int = 1000):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._records: OrderedDict[ProbeCandidateKey, ProbeRecord] = OrderedDict()

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, candidate: ProbeCandidateKey) -> bool:
        return candidate in self._records

    def add(self, candidate: ProbeCandidateKey, *, step: int) -> bool:
        if not isinstance(candidate, ProbeCandidateKey):
            raise TypeError("probe identity must include VoxelKey and complete hit_pair")
        if candidate in self._records:
            return False
        while len(self._records) >= self.capacity:
            self._records.popitem(last=False)
        self._records[candidate] = ProbeRecord(
            candidate=candidate, created_step=int(step),
        )
        return True

    def record_failure(self, candidate: ProbeCandidateKey) -> None:
        """Record an unsuccessful attempt without assigning a negative label."""
        self._records[candidate].attempts += 1

    def promote(
        self,
        candidate: ProbeCandidateKey,
        pool: VoxelPool,
        *,
        feas_margin: float = 0.0,
        source: str = "bootstrap",
        ingress_step: int = -1,
        finishing_hand: int = -1,
        hard_valid: bool,
    ) -> bool:
        if candidate not in self._records:
            raise KeyError("cannot promote a candidate that is not in the probe buffer")
        if not isinstance(hard_valid, (bool, np.bool_)):
            raise TypeError("hard_valid must be the boolean hard-gate result")
        if not bool(hard_valid):
            self.record_failure(candidate)
            return False
        new = pool.ingest(
            candidate.key,
            hit_pair=candidate.hit_pair,
            feas_margin=feas_margin,
            source=source,
            ingress_step=ingress_step,
            finishing_hand=finishing_hand,
        )
        del self._records[candidate]
        return new

    def records(self) -> tuple[ProbeRecord, ...]:
        return tuple(self._records.values())

    def state_dict(
        self, *, gate_definition: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = {
            "format_version": PROBE_CACHE_FORMAT_VERSION,
            "capacity": self.capacity,
            "records": [{
                "key": {
                    "chart_id": rec.key.chart_id, "bucket": rec.key.bucket,
                    "q0_cell": list(rec.key.q0_cell), "qd_bin": list(rec.key.qd_bin),
                },
                "hit_pair": list(rec.hit_pair),
                "created_step": rec.created_step,
                "attempts": rec.attempts,
            } for rec in self.records()],
        }
        if gate_definition is not None:
            state["gate_fingerprint"] = gate_definition_fingerprint(gate_definition)
        return state

    @classmethod
    def load_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_gate_definition: Mapping[str, Any] | str | None = None,
    ) -> "ProbeBuffer":
        version = int(state.get("format_version", CACHE_FORMAT_VERSION))
        if version != PROBE_CACHE_FORMAT_VERSION:
            raise ValueError("unsupported probe-buffer cache format version")
        if expected_gate_definition is not None:
            expected = gate_definition_fingerprint(expected_gate_definition)
            if str(state.get("gate_fingerprint", "")) != expected:
                raise ValueError("probe-buffer hard-gate fingerprint mismatch")
        out = cls(capacity=int(state["capacity"]))
        for item in state.get("records", []):
            ks = item["key"]
            key = VoxelKey(
                ks["chart_id"], ks["bucket"], tuple(ks["q0_cell"]), tuple(ks["qd_bin"]),
            )
            candidate = ProbeCandidateKey(key, tuple(item["hit_pair"]))
            out._records[candidate] = ProbeRecord(
                candidate=candidate, created_step=int(item["created_step"]),
                attempts=int(item.get("attempts", 0)),
            )
        return out

    def save_npz(
        self, path: str | Path, *, gate_definition: Mapping[str, Any] | str,
    ) -> None:
        payload = json.dumps(
            self.state_dict(gate_definition=gate_definition),
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        np.savez_compressed(Path(path), payload=np.asarray(payload))

    @classmethod
    def load_npz(
        cls, path: str | Path, *, expected_gate_definition: Mapping[str, Any] | str,
    ) -> "ProbeBuffer":
        with np.load(Path(path), allow_pickle=False) as data:
            payload = str(data["payload"].item())
        return cls.load_state_dict(
            json.loads(payload), expected_gate_definition=expected_gate_definition,
        )
