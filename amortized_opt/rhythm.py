"""Rhythm loading, drum-kit domain randomization and goal construction (section 5).

Three jobs, in the order the data generator uses them.

1. **Load a rhythm.** RCC JSON clips (``{groups: [{time, targets: [{drum, limb,
   velocity}]}]}``) are flattened into per-hand hit lists, with the same
   vocabulary mapping the repo's ``tasks/direct/g1_drum/rhythm.py`` uses:
   ``left_hand``/``right_hand`` -> stick, ``right_foot`` (the kick) dropped,
   ``crash`` -> ``cymbal_right``, drums absent from the layout dropped.
   Velocity defaults to 1.0 when the clip omits it.
2. **Quantize** every hit time onto the 20 ms waypoint grid (plan: "quantize our
   drum hit times to the closest 20ms"), so both hands live on one grid and a
   hit always lands exactly on a waypoint. Max induced error is 10 ms.
3. **Build goals.** For a given "now", each hand gets a *current goal* (the next
   hit it owns) and ``num_future`` upcoming hits, converted into the tensors
   :mod:`amortized_opt.tokenizer` expects (:class:`~amortized_opt.tokenizer.ContextInputs`).

Domain randomization lives here too, because the goal positions have to be
built against the *randomized* kit, not the nominal one:

* per-drum horizontal jitter (default <= 4 cm, never vertical) -- the kit is
  never assembled twice the same way;
* one global offset applied to every drum (default <= 7.5 cm horizontal,
  <= 3.5 cm vertical) -- the robot is never rolled into the same spot twice.

Both are drawn from a caller-supplied ``numpy.random.Generator``, so a seed
reproduces a kit exactly.

Strike points on the head come from :func:`amortized_opt.strike.strike_aim_on_head`
-- the SAME geometry the section-3 terminal cost prices -- jittered inside the
legal hit region: a disk of ``hit_radius_frac`` of the legal radius about the
aim, folded back onto the playable semicircle and clamped into the hit disk.
So a cymbal goal sits out near the near edge, a tom goal on the near side of
the rim-buffered head, and a snare goal around the centre, exactly where the
cost wants the contact point to land. (Earlier revisions used a plain
``center + uniform-in-disk(0.6 r)``, which put cymbal goals on the bell.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from .constants import HIT_QUANTIZATION_S, PELVIS_IN_ENV, default_layout_path
from .strike import strike_aim_on_head, strike_disk_center_offset, strike_half_plane_dir
from .tokenizer import ContextInputs

__all__ = [
    "DEFAULT_DRUM_ALIASES",
    "LIMB_TO_HAND",
    "HANDS",
    "Hit",
    "RhythmClip",
    "DrumKit",
    "KitRandomization",
    "HandGoal",
    "GoalSnapshot",
    "default_rcc_dir",
    "load_rcc",
    "load_rcc_pool",
    "shift_clip",
    "pad_first_hit_to",
    "shift_rcc_json",
    "quantize_time",
    "sample_hit_point",
    "strike_velocity",
    "goal_snapshot",
    "build_context_inputs",
    "head_plane_basis",
    "strike_region",
]

#: RCC limb name -> stick hand. Other limbs (the ``right_foot`` kick) are
#: dropped: the pelvis-pinned G1 drums with arms only.
LIMB_TO_HAND: dict[str, str] = {"left_hand": "left", "right_hand": "right"}
#: RCC drum name -> layout drum name for names that differ.
DEFAULT_DRUM_ALIASES: dict[str, str] = {"crash": "cymbal_right"}
HANDS: tuple[str, str] = ("left", "right")
#: Hand index used everywhere downstream (tokenizer rows, mask order).
HAND_INDEX: dict[str, int] = {"left": 0, "right": 1}

_ASSET_RCC = Path(__file__).resolve().parent.parent / "assets" / "rcc"


def default_rcc_dir(split: str = "train") -> Path:
    """Vendored RCC clips: real Groove-MIDI-Dataset grooves, nothing else.

    2026-08-14: the corpus is now GROOVES ONLY — 543 train / 16 eval 30-s
    segments converted from the 1 150 GMD performances in
    ``assets/rcc/groove/`` by ``scripts/convert_groove_midi.py``.  The
    synthetic curriculum (stages 0-6, two_drum, long_twodrum) and the old
    ``gmd_train_*``/``gmd_eval_*`` conversions were retired the same day:
    the old converter collapsed every floor-tom pitch into tom2 and every
    cymbal into cymbal_right, so those drums existed only synthetically.
    The new map splits toms (48,50 / 45,47 / 43,58 -> tom1 / tom2 /
    floor_tom) and cymbals (crash -> cymbal_left, ride -> cymbal_right).
    Stick limbs are assigned by walking the clip on the kit layout graph
    (``amortized_opt.hand_assign``): each beat goes to the closer stick
    given where that stick last played, or idle on its side of the kit.
    Eval sources (the 14
    drummer1 MIDIs behind the retired eval set) can never enter train --
    the converter routes them to ``eval/`` by source identity.  Residual
    rarity (tom2/floor_tom/cymbal_left are 1-3 %% of hits) is a sampling
    problem, not a data problem: ``Phase2ScenarioConfig.rare_drum_boost``.
    """
    p = _ASSET_RCC / split
    if not p.is_dir():
        raise FileNotFoundError(f"vendored RCC clips missing at {p}")
    return p


# ----------------------------------------------------------------- drum kit --
@dataclass(frozen=True)
class KitRandomization:
    """Bounds for the plan's two independent kit perturbations (metres).

    ``per_drum_xy`` moves each drum on its own, horizontally only. ``global_xy``
    / ``global_z`` move the WHOLE kit, which is how a change in where the robot
    is standing shows up in the robot's own frame.
    """

    per_drum_xy: float = 0.04
    global_xy: float = 0.075
    global_z: float = 0.035
    #: Per-drum TILT of the head normal, radians of half-angle.  The direction
    #: is uniform on the cone about the nominal normal, so no lean is
    #: privileged, and the result is renormalised.
    #:
    #: This is already in the model's input space -- ``drum_features`` carries
    #: the normal as ``ctx.drum_axis`` (3) AND as the scalar ``_tilt_angle``
    #: (1), 4 of ``DRUM_FEAT_DIM = 9`` -- but nothing has ever varied it, so
    #: those four numbers were constant across the whole training set and the
    #: encoder had no reason to read them.  0.0 reproduces the pre-2026-08-13
    #: kit exactly.
    per_drum_tilt: float = 0.0
    #: Per-drum RADIUS scale, a FRACTION: drum ``i``'s radius is multiplied by
    #: an independent uniform draw in ``[1 - f, 1 + f]``.
    #:
    #: Radius was the LAST drum feature that never varied.  ``drum_features``
    #: carries it (1 of ``DRUM_FEAT_DIM = 9``), but every sample gave a drum id
    #: its nominal radius, so radius was perfectly confounded with identity and
    #: the weights reading it were UNIDENTIFIED -- the ``yaw_now`` failure class
    #: (an input that never varies independently is read in an arbitrary,
    #: untrained way the moment deployment presents a novel value).  Same
    #: argument, and the same remedy, as ``per_drum_tilt`` above.  0.0
    #: reproduces the pre-2026-08-14 kit exactly, RNG stream included.
    per_drum_radius: float = 0.0

    def __post_init__(self) -> None:
        for k in ("per_drum_xy", "global_xy", "global_z", "per_drum_tilt",
                  "per_drum_radius"):
            if float(getattr(self, k)) < 0.0:
                raise ValueError(f"{k} must be >= 0")
        # A fraction of 1 or more can draw a zero/negative radius, which is not
        # a drum -- the strike disk, the hit-point sampler and the collision
        # cylinder all degenerate.  Rejected here rather than clamped, so a
        # typo'd knob is loud.
        if float(self.per_drum_radius) >= 1.0:
            raise ValueError("per_drum_radius must be < 1 (it is a fraction)")


@dataclass
class DrumKit:
    """Drum geometry in the ENV frame, ordered by the layout's ``id``.

    Row ``i`` is drum id ``i``, matching
    :class:`~amortized_opt.sdf.DrumSceneSpec` and the SDF's ``target_drum_*``
    exemption indices, so a goal index can be handed straight to either.
    """

    names: list[str]
    positions: np.ndarray  # (D, 3) head centres, env frame
    axes: np.ndarray  # (D, 3) unit head normals
    radii: np.ndarray  # (D,)
    heights: np.ndarray  # (D,)
    shaft_hit: np.ndarray  # (D,) bool
    edge_hit: np.ndarray  # (D,) bool

    @property
    def num_drums(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def positions_in(self, frame: str) -> np.ndarray:
        if frame == "env":
            return self.positions
        if frame == "pelvis":
            return self.positions - PELVIS_IN_ENV[None, :]
        raise ValueError(f"frame must be 'env' or 'pelvis', got {frame!r}")

    @classmethod
    def from_layout(cls, layout_path: str | Path | None = None) -> "DrumKit":
        import yaml

        path = Path(layout_path) if layout_path is not None else default_layout_path()
        with open(path) as f:
            layout = yaml.safe_load(f)
        items = list(layout["drums"].items())
        if all("id" in spec for _n, spec in items):
            items.sort(key=lambda kv: int(kv[1]["id"]))
        names, pos, axes, radii, heights, shaft, edge = [], [], [], [], [], [], []
        for name, spec in items:
            names.append(str(name))
            pos.append(np.asarray(spec["position"], dtype=float))
            v = np.asarray(spec.get("normal", [0.0, 0.0, 1.0]), dtype=float)
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-8:
                raise ValueError(f"drum {name!r} has a zero-length normal")
            axes.append(v / nrm)
            radii.append(float(spec["radius"]))
            heights.append(float(spec.get("height", 0.0)))
            shaft.append(bool(spec.get("shaft_hit", False)))
            edge.append(bool(spec.get("edge_hit", False)))
        return cls(
            names=names,
            positions=np.stack(pos),
            axes=np.stack(axes),
            radii=np.asarray(radii, float),
            heights=np.asarray(heights, float),
            shaft_hit=np.asarray(shaft, bool),
            edge_hit=np.asarray(edge, bool),
        )

    # ------------------------------------------------------ randomization --
    def randomized(
        self,
        rng: np.random.Generator,
        cfg: KitRandomization | None = None,
    ) -> "DrumKit":
        """A perturbed copy. Positions, and optionally NORMALS and RADII, move;
        heights do not.

        The per-drum jitter is uniform in a horizontal DISK (not a square) so no
        direction is privileged, and the vertical component is identically zero
        --- the plan is explicit that individual drums never move up or down.
        The global offset is uniform in a horizontal disk plus an independent
        vertical interval.

        The tilt and radius draws happen only when their knob is non-zero and
        AFTER the position draws, so leaving either at 0.0 leaves the RNG
        stream -- and therefore every pool generated before it existed --
        bit-identical.
        """
        cfg = cfg or KitRandomization()
        D = self.num_drums
        pos = self.positions.copy()
        if cfg.per_drum_xy > 0.0:
            pos[:, :2] += _uniform_disk(rng, D, cfg.per_drum_xy)
        g = np.zeros(3)
        if cfg.global_xy > 0.0:
            g[:2] = _uniform_disk(rng, 1, cfg.global_xy)[0]
        if cfg.global_z > 0.0:
            g[2] = rng.uniform(-cfg.global_z, cfg.global_z)
        pos += g[None, :]
        axes = self.axes
        if cfg.per_drum_tilt > 0.0:
            axes = _tilted(rng, self.axes, cfg.per_drum_tilt, pos)
        radii = self.radii
        if cfg.per_drum_radius > 0.0:
            f = float(cfg.per_drum_radius)
            radii = self.radii * rng.uniform(1.0 - f, 1.0 + f, D)
        return replace(self, positions=pos, axes=axes, radii=radii)

    # -------------------------------------------------------- eligibility --
    def eligible(self, hand: str) -> list[int]:
        """Drum indices a hand may strike (the no-crossover kit rule).

        Ported from ``control/teacher/sampler.py::drums_for_hand`` with
        ``NO_CROSSOVER_ELIGIBILITY`` on: a hand may play every drum except the
        one at the far end of the OTHER side, found geometrically from the drum
        centres (left loses the min-y drum, right loses the max-y drum) so a
        layout edit cannot silently desync this from the curriculum.
        """
        if hand not in HAND_INDEX:
            raise ValueError(f"hand must be left|right, got {hand!r}")
        D = self.num_drums
        if D < 2:
            return list(range(D))
        ys = self.positions[:, 1]
        banned = int(np.argmin(ys)) if hand == "left" else int(np.argmax(ys))
        out = [i for i in range(D) if i != banned]
        return out if out else list(range(D))

    def is_eligible(self, hand: str, drum: int) -> bool:
        return int(drum) in self.eligible(hand)


def _tilted(
    rng: np.random.Generator,
    axes: np.ndarray,
    half_angle: float,
    positions: np.ndarray,
) -> np.ndarray:
    """``(D, 3)`` unit normals tilted by <= ``half_angle`` rad, never leaning AWAY.

    The polar angle is drawn as ``arccos(1 - u (1 - cos a))`` so the tip of the
    normal is uniform on the spherical CAP rather than uniform in the angle --
    the latter would over-sample small tilts.  The azimuth is uniform, so no
    lean direction is privileged, matching how ``_uniform_disk`` treats
    translation.

    THE HALF-CONE CONSTRAINT.  A head that leans away from the robot is not a
    kit you could play: the stick would have to come over the far rim.  Every
    shipped drum already leans toward the robot (0.11-0.37 of horizontal lean,
    ``floor_tom`` exactly vertical at 0.0), so the sampled normal is required to
    keep a NON-NEGATIVE component along the horizontal drum->robot direction.
    A tilt may therefore lean the head further toward the robot, or stand it
    up toward vertical, but never past vertical the other way.

    Enforced by REJECTION -- resample the offending rows -- rather than by
    folding the azimuth, which would pile up twice the density on the allowed
    side instead of leaving the cap uniform over it.  Rows that never pass keep
    their nominal normal, which satisfies the constraint by construction.
    """
    axes = np.asarray(axes, float)
    n0 = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    D = n0.shape[0]
    cos_a = np.cos(float(half_angle))

    # horizontal unit vector from each drum toward the robot (pelvis xy = origin)
    xy = np.asarray(positions, float)[:, :2]
    r = np.linalg.norm(xy, axis=1, keepdims=True)
    toward = np.divide(-xy, np.maximum(r, 1e-9))
    toward[r[:, 0] < 1e-9] = 0.0        # a drum under the robot has no direction

    # an orthonormal pair spanning the plane perpendicular to each normal
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (D, 1))
    flip = np.abs((n0 * ref).sum(1)) > 0.9
    ref[flip] = np.array([1.0, 0.0, 0.0])
    u = np.cross(n0, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n0, u)

    out = n0.copy()
    todo = np.arange(D)
    for _ in range(24):
        if todo.size == 0:
            break
        m = todo.size
        theta = np.arccos(1.0 - rng.uniform(0.0, 1.0, m) * (1.0 - cos_a))
        phi = rng.uniform(0.0, 2.0 * np.pi, m)
        st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
        cand = (ct * n0[todo]
                + st * (np.cos(phi)[:, None] * u[todo] + np.sin(phi)[:, None] * v[todo]))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        ok = (cand[:, :2] * toward[todo]).sum(1) >= 0.0
        out[todo[ok]] = cand[ok]
        todo = todo[~ok]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _uniform_disk(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """(n, 2) samples uniform on a disk of the given radius."""
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rad = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


# ------------------------------------------------------------------ rhythm --
def quantize_time(t: float, grid: float = HIT_QUANTIZATION_S) -> float:
    """Snap to the nearest multiple of ``grid`` (20 ms). Max error grid/2."""
    return float(round(float(t) / float(grid)) * float(grid))


@dataclass(frozen=True)
class Hit:
    """One stick strike."""

    time: float  # quantized, seconds from clip start
    raw_time: float  # as written in the clip
    drum: int  # index into DrumKit rows
    drum_name: str
    hand: str  # "left" / "right"
    velocity: float  # 0..1 normalized loudness from the clip


@dataclass
class RhythmClip:
    name: str
    hits: list[Hit]
    duration: float
    #: Hits whose clip-assigned hand could not legally play that drum.
    ineligible: int = 0

    @property
    def num_hits(self) -> int:
        return len(self.hits)

    def hand_hits(self, hand: str) -> list[Hit]:
        return [h for h in self.hits if h.hand == hand]

    def next_hits(self, hand: str, t_now: float, count: int, *, eps: float = 1e-9) -> list[Hit]:
        """The next ``count`` hits for ``hand`` at or after ``t_now``."""
        out = [h for h in self.hits if h.hand == hand and h.time >= t_now - eps]
        return out[: int(count)]


def shift_clip(
    clip: RhythmClip, dt: float, *, grid: float = HIT_QUANTIZATION_S,
) -> RhythmClip:
    """Add ``dt`` seconds to every hand hit. Inter-hit gaps are unchanged."""
    dt = float(dt)
    if abs(dt) < 1e-12:
        return clip
    hits = [
        replace(
            h,
            time=quantize_time(float(h.time) + dt, grid),
            raw_time=float(h.raw_time) + dt,
        )
        for h in clip.hits
    ]
    hits.sort(key=lambda h: (h.time, h.hand))
    return replace(clip, hits=hits, duration=float(clip.duration) + dt)


def pad_first_hit_to(
    clip: RhythmClip, t_first: float, *, grid: float = HIT_QUANTIZATION_S,
) -> tuple[RhythmClip, float]:
    """Delay the clip so the first hand hit is at ``t_first``.

    Idle-family training adds ``--lead-s 1.0`` on top of a ready start, so the
    first dual goal from rest is ~1 s away. Playback of a clip whose first
    unison is at 0.38 s is the same pose and pair with a third of that lead.
    This pad restores the trained clock without changing later spacing.
    """
    if not clip.hits:
        return clip, 0.0
    t0 = min(float(h.time) for h in clip.hits)
    dt = quantize_time(float(t_first) - t0, grid)
    if dt <= 1e-12:
        return clip, 0.0
    return shift_clip(clip, dt, grid=grid), dt


def shift_rcc_json(
    src: str | Path, dst: str | Path, dt: float,
) -> Path:
    """Copy an RCC file with every group time (hands and kick) shifted by ``dt``.

    The Isaac overlay reads this JSON, not the in-memory :class:`RhythmClip`.
    """
    src, dst = Path(src), Path(dst)
    data = json.loads(src.read_text())
    dt = float(dt)
    for group in data.get("groups", []):
        group["time"] = float(group["time"]) + dt
    meta = data.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["first_hit_pad_s"] = dt
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2))
    return dst


def load_rcc(
    path: str | Path,
    kit: DrumKit,
    *,
    quantize: bool = True,
    grid: float = HIT_QUANTIZATION_S,
    drum_aliases: dict[str, str] | None = None,
    on_ineligible: str = "reassign",
    post_buffer_s: float = 1.0,
) -> RhythmClip:
    """Parse one RCC JSON clip into a :class:`RhythmClip`.

    ``on_ineligible`` decides what happens to a hit the clip charges to a hand
    the no-crossover rule forbids (only the two extreme drums can produce one):
    ``"reassign"`` (default) hands it to the other stick, ``"drop"`` removes it,
    ``"keep"`` leaves it alone. Reassignment is the default because the rhythm
    curriculum's own split-point hand assignment would never have emitted the
    crossover in the first place; keeping it would train the model on a reach
    the sampler never generates.
    """
    path = Path(path)
    aliases = DEFAULT_DRUM_ALIASES if drum_aliases is None else drum_aliases
    if on_ineligible not in ("reassign", "drop", "keep"):
        raise ValueError(f"on_ineligible must be reassign|drop|keep, got {on_ineligible!r}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"{path} is not an RCC clip object (no 'groups')")

    elig = {h: set(kit.eligible(h)) for h in HANDS}
    hits: list[Hit] = []
    bad = 0
    for group in data["groups"]:
        t_raw = float(group["time"])
        t = quantize_time(t_raw, grid) if quantize else t_raw
        for target in group.get("targets", []):
            hand = LIMB_TO_HAND.get(str(target.get("limb", "")))
            if hand is None:
                continue  # kick / foot: no effector
            name = str(target.get("drum", ""))
            name = aliases.get(name, name)
            if name not in kit.names:
                continue  # e.g. the kick drum, absent from this layout
            idx = kit.index(name)
            if idx not in elig[hand]:
                bad += 1
                if on_ineligible == "drop":
                    continue
                if on_ineligible == "reassign":
                    other = "right" if hand == "left" else "left"
                    if idx not in elig[other]:
                        continue
                    hand = other
            hits.append(
                Hit(
                    time=t,
                    raw_time=t_raw,
                    drum=idx,
                    drum_name=name,
                    hand=hand,
                    velocity=float(target.get("velocity", 1.0)),
                )
            )
    hits.sort(key=lambda h: (h.time, h.hand))
    duration = (hits[-1].time if hits else 0.0) + float(post_buffer_s)
    return RhythmClip(name=path.stem, hits=hits, duration=duration, ineligible=bad)


def load_rcc_pool(
    source: str | Path | list, kit: DrumKit, **kw
) -> list[RhythmClip]:
    """Load a file, a directory of clips, or an explicit list. Empty clips are
    skipped (a clip that is all kick has nothing for the arms to play)."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    else:
        files = [Path(x) for x in source]
    out: list[RhythmClip] = []
    for f in files:
        try:
            clip = load_rcc(f, kit, **kw)
        except ValueError:
            continue  # dataset summaries / stats files living next to clips
        if clip.num_hits:
            out.append(clip)
    return out


# ------------------------------------------------------------------- goals --
def head_plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane directions for a head whose normal is ``axis``."""
    a = np.asarray(axis, float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, a)
    t1 /= max(float(np.linalg.norm(t1)), 1e-12)
    return t1, np.cross(a, t1)


def strike_region(
    kit: DrumKit, drum: int, *, frame: str = "env"
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """``(aim, disk_center, disk_radius, half_plane_dir)`` for one drum.

    A thin adapter over :mod:`amortized_opt.strike` so the goal context and the
    terminal cost read the same numbers. The legal region is
    ``|p - disk_center| <= disk_radius`` in the head plane, intersected with
    ``(p - disk_center) . half_plane_dir >= 0`` when a half-plane applies.
    """
    axis = kit.axes[drum]
    c = kit.positions_in(frame)[drum]
    name = kit.names[drum]
    aim, disk_r = strike_aim_on_head(c, axis, float(kit.radii[drum]), name=name)
    off = strike_disk_center_offset(c, axis, float(kit.radii[drum]), name=name)
    center = aim + (np.zeros(3) if off is None else off)
    return aim, center, float(disk_r), strike_half_plane_dir(c, axis, name=name)


def sample_hit_point(
    kit: DrumKit,
    drum: int,
    rng: np.random.Generator | None = None,
    *,
    hit_radius_frac: float = 0.6,
    frame: str = "env",
) -> np.ndarray:
    """A point in the drum's LEGAL strike region, jittered about the strike aim.

    The aim, the hit-disk radius, the disk centre and the playable-semicircle
    cut all come from :mod:`amortized_opt.strike`, so a goal token describes the
    same geometry the section-3 terminal cost prices. The jitter is uniform in a
    disk of ``hit_radius_frac`` of the legal radius about the aim; the result is
    folded onto the legal half-plane and clamped into the hit disk, so it is a
    point the strike cost would accept.

    ``rng=None`` returns the aim exactly (deterministic contexts for tests).
    """
    axis = np.asarray(kit.axes[drum], float)
    aim, center, disk_r, half = strike_region(kit, drum, frame=frame)
    if rng is None or hit_radius_frac <= 0.0:
        return aim.copy()
    t1, t2 = head_plane_basis(axis)
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    rad = float(max(hit_radius_frac, 0.0) * disk_r * math.sqrt(rng.uniform(0.0, 1.0)))
    d = (aim - center) + rad * (np.cos(ang) * t1 + np.sin(ang) * t2)
    d = d - float(d @ axis) * axis  # numerical hygiene: stay in the head plane
    if half is not None:
        d = d - min(float(d @ half), 0.0) * half  # fold onto the near semicircle
    n = float(np.linalg.norm(d))
    if n > disk_r:
        d = d * (disk_r / n)
    return center + d


def strike_velocity(
    kit: DrumKit,
    drum: int,
    velocity: float,
    *,
    speed_min: float = 1.0,
    speed_max: float = 3.0,
) -> np.ndarray:
    """Commanded impact velocity (3,): into the head, along ``-axis``.

    The RCC ``velocity`` field is a normalized 0..1 loudness; it is mapped
    linearly onto ``[speed_min, speed_max]`` m/s, the same 1/2/3 m/s band the
    evaluation grid uses.
    """
    v = float(np.clip(velocity, 0.0, 1.0))
    speed = float(speed_min) + v * (float(speed_max) - float(speed_min))
    return -speed * kit.axes[drum]


@dataclass
class HandGoal:
    """One hand's current goal, or one of its future hits."""

    active: bool
    drum: int
    position: np.ndarray  # (3,) on the head, env frame
    velocity: np.ndarray  # (3,) commanded impact velocity
    time_to_hit: float  # seconds from "now"; <= 0 means it is due
    drum_name: str = ""
    loudness: float = 1.0

    @classmethod
    def inactive(cls) -> "HandGoal":
        return cls(False, 0, np.zeros(3), np.zeros(3), 0.0, "", 0.0)


@dataclass
class GoalSnapshot:
    """Everything the context encoder needs at one instant, per hand."""

    t_now: float
    goal: dict[str, HandGoal]
    future: dict[str, list[HandGoal]]

    def time_to_hit(self, hand: str, *, inactive: float = float("inf")) -> float:
        g = self.goal[hand]
        return g.time_to_hit if g.active else inactive

    def active(self, hand: str) -> bool:
        return self.goal[hand].active


def _hit_to_goal(
    kit: DrumKit,
    hit: Hit,
    t_now: float,
    rng: np.random.Generator | None,
    *,
    hit_radius_frac: float,
    speed_min: float,
    speed_max: float,
) -> HandGoal:
    return HandGoal(
        active=True,
        drum=hit.drum,
        position=sample_hit_point(kit, hit.drum, rng, hit_radius_frac=hit_radius_frac),
        velocity=strike_velocity(
            kit, hit.drum, hit.velocity, speed_min=speed_min, speed_max=speed_max
        ),
        time_to_hit=float(hit.time) - float(t_now),
        drum_name=hit.drum_name,
        loudness=float(hit.velocity),
    )


def goal_snapshot(
    clip: RhythmClip,
    kit: DrumKit,
    t_now: float,
    *,
    num_future: int = 3,
    rng: np.random.Generator | None = None,
    hit_radius_frac: float = 0.6,
    speed_min: float = 1.0,
    speed_max: float = 3.0,
    horizon_cap: float | None = None,
    time_jitter_s: float = 0.0,
    loudness_jitter: float = 0.0,
) -> GoalSnapshot:
    """Per-hand current goal + ``num_future`` upcoming hits at time ``t_now``.

    A hand with no remaining hit gets an inactive goal (the tokenizer masks it
    and :func:`~amortized_opt.tokenizer.build_waypoint_masks` puts it under the
    other hand's horizon rather than dropping it from the loss).
    ``horizon_cap`` marks a current goal inactive when it is further away than
    the model's horizon can represent.

    ``time_jitter_s`` / ``loudness_jitter`` perturb each hit's time (uniform
    ``+-s``) and its 0-1 loudness (uniform ``+-``, clipped) at SNAPSHOT time,
    so two draws of the same clip instant are two slightly different rhythms
    -- the clip corpus is finite, the augmented one is not.  The time offset
    is shared by every hit at one clip time (a unison stays a unison, a flam
    stays a flam) and each hand's hit ORDER is preserved by clamping; a hit
    jittered before ``t_now`` becomes due-now rather than vanishing.  The
    defaults are bit-exact the old behaviour, RNG stream included.
    """
    jit = rng is not None and (time_jitter_s > 0.0 or loudness_jitter > 0.0)
    t_off: dict[float, float] = {}

    def _jit(h: Hit) -> Hit:
        key = round(float(h.time), 6)
        if key not in t_off:
            t_off[key] = (
                float(rng.uniform(-time_jitter_s, time_jitter_s))
                if time_jitter_s > 0.0 else 0.0
            )
        v = float(h.velocity)
        if loudness_jitter > 0.0:
            v = float(np.clip(v + rng.uniform(-loudness_jitter, loudness_jitter), 0.0, 1.0))
        return replace(h, time=max(float(h.time) + t_off[key], float(t_now) + 1e-6),
                       velocity=v)

    goal: dict[str, HandGoal] = {}
    future: dict[str, list[HandGoal]] = {}
    for hand in HANDS:
        nxt = clip.next_hits(hand, t_now, int(num_future) + 1)
        if jit:
            nxt = [_jit(h) for h in nxt]
            for i in range(1, len(nxt)):
                if nxt[i].time < nxt[i - 1].time:
                    nxt[i] = replace(nxt[i], time=nxt[i - 1].time)
        if nxt and (horizon_cap is None or (nxt[0].time - t_now) <= horizon_cap + 1e-9):
            goal[hand] = _hit_to_goal(
                kit, nxt[0], t_now, rng,
                hit_radius_frac=hit_radius_frac, speed_min=speed_min, speed_max=speed_max,
            )
            rest = nxt[1:]
        else:
            goal[hand] = HandGoal.inactive()
            rest = nxt[:int(num_future)]
        future[hand] = [
            _hit_to_goal(
                kit, h, t_now, rng,
                hit_radius_frac=hit_radius_frac, speed_min=speed_min, speed_max=speed_max,
            )
            for h in rest[: int(num_future)]
        ]
    return GoalSnapshot(t_now=float(t_now), goal=goal, future=future)


def aim_inputs_from_snapshot(
    snap: GoalSnapshot, *, frame: str = "pelvis"
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, list]]:
    """``(targets, goal_tth, futures)`` for :func:`~amortized_opt.waist_aim.aim_yaw`.

    ONE source for the aim law's inputs, so the context token, the seed's yaw
    ramp and the deployment schedule cannot each assemble the schedule slightly
    differently.  ``targets`` holds only the ACTIVE current goals (the law's
    pre-2026-08-14 argument); ``futures`` holds every hand's upcoming hits as
    ``(position, time_to_hit)``, including for a hand whose current goal the
    horizon cap demoted -- that hand votes through its futures alone, which is
    what stops the aim square-waving as a goal crosses the cap.

    Times are FROM NOW on both sides; ``ContextInputs.future_gap``'s
    hit-relative anchoring is a token convention and does not apply here.
    ``frame`` picks the position convention -- only XY reaches the aim law and
    ``PELVIS_IN_ENV`` has none, so the two agree to the bit; the argument exists
    so a caller cannot get the Z convention wrong by accident.
    """
    if frame not in ("pelvis", "env"):
        raise ValueError(f"frame must be 'pelvis' or 'env', got {frame!r}")
    off = PELVIS_IN_ENV if frame == "pelvis" else 0.0
    targets = {
        hand: snap.goal[hand].position - off
        for hand in HANDS
        if snap.goal[hand].active
    }
    tth = {
        hand: float(snap.goal[hand].time_to_hit)
        for hand in HANDS
        if snap.goal[hand].active
    }
    futures = {
        hand: [
            (fut.position - off, float(fut.time_to_hit))
            for fut in snap.future[hand]
            if fut.active
        ]
        for hand in HANDS
    }
    return targets, tth, futures


# ------------------------------------------------------------- tokenization --
def build_context_inputs(
    kits: DrumKit | list[DrumKit],
    snapshots: GoalSnapshot | list[GoalSnapshot],
    *,
    num_drums: int | None = None,
    num_future: int = 3,
    frame: str = "env",
    device=None,
    dtype: torch.dtype = torch.float32,
    idle_home=None,
    idle_library=None,
    idle_cylinder=None,
    fk=None,
    sticks=None,
    yaw_now=None,
    waist_cfg=None,
) -> ContextInputs:
    """Pack (kit, goals) into the encoder's :class:`ContextInputs`, batched.

    One kit and one snapshot per batch row (pass a single object to broadcast).
    Drum rows are padded to ``num_drums`` with ``drum_valid = False``; row ``i``
    is drum id ``i``, so ``goal_drum`` doubles as the SDF's ``target_drum_*``
    exemption index.

    THE FUTURE TOKENS ARE HIT-RELATIVE.  ``GoalSnapshot`` carries every time as
    a from-now ``time_to_hit``; the conversion to
    :attr:`~amortized_opt.tokenizer.ContextInputs.future_gap` happens HERE, so
    every producer that tokenizes through this function (the phase-1 and
    phase-2 pools, ``MPCGenerator``, the closed-loop probe) gets the same
    anchoring for free and none of them can invent its own.

    ``idle_library`` / ``idle_cylinder`` fill the IDLE token's packed cylinder
    ``(B, 2, 8)``.  Prefer ``derive_cylinder(kit, fk)`` per row when ``fk`` is
    given; otherwise use the explicit cylinder or ``idle_library.cylinder``.
    ``idle_valid = ~goal_valid``.  Legacy ``idle_home`` is ignored for the
    cylinder payload (kept only so old call sites still type-check).

    ``yaw_now`` (scalar or ``(B,)``, rad) is the waist yaw the robot is at when
    this context is built.  Supplying it fills the waist-yaw set-point inputs:
    ``ctx.aim_yaw`` becomes the analytic
    :func:`~amortized_opt.waist_aim.aim_yaw` command over each hand's ACTIVE
    current goal AND its upcoming future hits, weighted by time to hit (pelvis
    frame, gate and deadband included), evaluated at that yaw —
    the same law, with the same default config, that the warm-start yaw ramp
    terminates at, so the head's residual parameterization starts exactly on
    the ramp.  Omit it and the block stays zeroed with ``aim_valid`` False.
    ``waist_cfg`` overrides the :class:`~amortized_opt.waist_aim.WaistAimConfig`
    used; leave it None everywhere the warm-start builder does the same.
    """
    snaps = [snapshots] if isinstance(snapshots, GoalSnapshot) else list(snapshots)
    B = len(snaps)
    kit_list = [kits] * B if isinstance(kits, DrumKit) else list(kits)
    if len(kit_list) != B:
        raise ValueError(f"{len(kit_list)} kits for {B} snapshots")
    D = max(k.num_drums for k in kit_list) if num_drums is None else int(num_drums)
    if any(k.num_drums > D for k in kit_list):
        raise ValueError(f"num_drums={D} cannot hold every kit")
    F = int(num_future)

    ctx = ContextInputs.zeros(B, num_drums=D, num_future=F, device=device, dtype=dtype)
    np_f = lambda t: t  # noqa: E731

    dp = np.zeros((B, D, 3)); da = np.zeros((B, D, 3))
    dr = np.zeros((B, D)); dh = np.zeros((B, D)); dv = np.zeros((B, D), bool)
    gd = np.zeros((B, 2), np.int64); gp = np.zeros((B, 2, 3)); gv = np.zeros((B, 2, 3))
    gt = np.zeros((B, 2)); gval = np.zeros((B, 2), bool)
    fd = np.zeros((B, 2, F), np.int64); fp = np.zeros((B, 2, F, 3))
    fv = np.zeros((B, 2, F, 3)); fg = np.zeros((B, 2, F)); fval = np.zeros((B, 2, F), bool)

    for b, (kit, snap) in enumerate(zip(kit_list, snaps)):
        d = kit.num_drums
        dp[b, :d] = kit.positions_in(frame)
        da[b, :d] = kit.axes
        dr[b, :d] = kit.radii
        dh[b, :d] = kit.heights
        dv[b, :d] = True
        off = np.zeros(3) if frame == "env" else PELVIS_IN_ENV
        for hand, hi in HAND_INDEX.items():
            g = snap.goal[hand]
            gd[b, hi] = g.drum
            gp[b, hi] = g.position - (off if frame == "pelvis" else 0.0)
            gv[b, hi] = g.velocity
            gt[b, hi] = g.time_to_hit
            gval[b, hi] = g.active
            # THE ANCHOR for this hand's future timing (see
            # ``ContextInputs.future_gap``): its current goal's strike when it
            # has one, else now.  ``GoalSnapshot`` keeps every time from-now, so
            # subtracting the goal's own ``time_to_hit`` is the conversion --
            # and it is what makes the future tokens constant across the ticks
            # of one goal segment, since both terms slide by the same dt.
            anchor = g.time_to_hit if g.active else 0.0
            for j, fut in enumerate(snap.future[hand][:F]):
                fd[b, hi, j] = fut.drum
                fp[b, hi, j] = fut.position - (off if frame == "pelvis" else 0.0)
                fv[b, hi, j] = fut.velocity
                fg[b, hi, j] = fut.time_to_hit - anchor
                fval[b, hi, j] = fut.active

    t = lambda a: torch.as_tensor(a, device=device, dtype=dtype)  # noqa: E731
    tb = lambda a: torch.as_tensor(a, device=device, dtype=torch.bool)  # noqa: E731
    tl = lambda a: torch.as_tensor(a, device=device, dtype=torch.long)  # noqa: E731
    ctx.drum_pos, ctx.drum_axis = t(dp), t(da)
    ctx.drum_radius, ctx.drum_height, ctx.drum_valid = t(dr), t(dh), tb(dv)
    ctx.goal_drum, ctx.goal_pos, ctx.goal_vel = tl(gd), t(gp), t(gv)
    ctx.goal_tth, ctx.goal_valid = t(gt), tb(gval)
    ctx.future_drum, ctx.future_pos, ctx.future_vel = tl(fd), t(fp), t(fv)
    ctx.future_gap, ctx.future_valid = t(fg), tb(fval)

    # ---- the IDLE arm's own goal token -------------------------------------
    # Valid exactly where the goal token is NOT, so a hand is never handed both
    # a strike target and a park target.  The table is indexed by the waist yaw
    # the arm will be at, which the caller supplies -- this function only sees
    # the kit and the rhythm, not the robot's state.
    # ---- the fitted STICK, per hand -----------------------------------------
    # ``sticks`` is one ``(left_spec, right_spec)`` pair per batch row (or a
    # single pair to broadcast).  Omit it and the block stays zeroed and
    # invalid, which is what a caller with a nominal stick means.
    if sticks is not None:
        from .fk import StickSpec, stick_axis_local, stick_tip_offset

        pairs = [sticks] * B if isinstance(sticks, tuple) and len(sticks) == 2 \
            and isinstance(sticks[0], StickSpec) else list(sticks)
        if len(pairs) != B:
            raise ValueError(f"{len(pairs)} stick pairs for {B} snapshots")
        sa = np.zeros((B, 2, 3)); st = np.zeros((B, 2, 3)); sl = np.zeros((B, 2))
        for b, pair in enumerate(pairs):
            for hand, hi in HAND_INDEX.items():
                sp = pair[hi]
                sa[b, hi] = stick_axis_local(hand, sp)
                st[b, hi] = stick_tip_offset(hand, sp)
                sl[b, hi] = sp.length
        ctx.stick_axis, ctx.stick_tip = t(sa), t(st)
        ctx.stick_length = t(sl)
        ctx.stick_valid = tb(np.ones((B, 2), bool))

    if idle_library is not None or idle_cylinder is not None:
        from .idle_cylinder import IdleCylinder, derive_cylinder

        ic = np.zeros((B, 2, 8))
        iv = ~gval                                            # (B, 2)
        # ``idle_cylinder`` may be one cylinder or a length-B sequence.
        cyls = None
        if idle_cylinder is not None:
            if isinstance(idle_cylinder, IdleCylinder):
                cyls = [idle_cylinder] * B
            else:
                cyls = list(idle_cylinder)
                if len(cyls) != B:
                    raise ValueError(f"{len(cyls)} idle cylinders for {B} snapshots")
        for b, kit in enumerate(kit_list):
            if cyls is not None:
                cyl = cyls[b]
            elif fk is not None:
                cyl = derive_cylinder(kit, fk)
            else:
                cyl = idle_library.cylinder
            pack = np.asarray(cyl.pack(), float)
            for hi in range(2):
                ic[b, hi] = pack * float(iv[b, hi])
        ctx.idle_cyl, ctx.idle_valid = t(ic), tb(iv)
    elif idle_home is not None:
        # Legacy path: no cylinder payload; leave idle_cyl zeroed / invalid.
        pass

    # ---- the analytic waist-yaw aim, for the set-point head ----------------
    # Targets go to the PELVIS frame regardless of ``frame``: the aim law
    # measures azimuth from the shoulders, which ride the waist-yaw axis.
    if yaw_now is not None:
        from .waist_aim import WaistAimConfig, aim_yaw

        yn = np.broadcast_to(
            np.asarray(yaw_now, dtype=float).reshape(-1), (B,)
        ).copy()
        wcfg = waist_cfg if waist_cfg is not None else WaistAimConfig()
        av = np.zeros(B)
        for b, snap in enumerate(snaps):
            # The FUTURE hits vote too, weighted by time to hit alone -- the
            # same list the future tokens above are built from, assembled by the
            # one shared helper.
            targets, tth, futs = aim_inputs_from_snapshot(snap, frame="pelvis")
            av[b] = aim_yaw(targets, cfg=wcfg, yaw_current=float(yn[b]),
                            goal_tth=tth, futures=futs)
        ctx.yaw_now, ctx.aim_yaw = t(yn), t(av)
        ctx.aim_valid = tb(np.ones(B, bool))
    return ctx
