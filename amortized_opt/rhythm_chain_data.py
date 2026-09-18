"""Rhythm-derived Phase A / schedule schema / synthetic dual transitions.

See the rhythm-derived data plan: idle mid-clip entries, hidden schedule with
``num_future=1`` projection, and a filtered dual-hand transition clip corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import torch

from .constants import (
    CONTROL_DT,
    G1_UPPER_BODY_JOINTS,
    HIT_QUANTIZATION_S,
    STICK_MIN_SEP,
)
from .fk import ChainFK
from .idle_cylinder import derive_cylinder, resolve_idle_cylinder_library, tip_in_torso
from .ik import tip_position
from .limits import isaac_velocity_limits, urdf_position_limits
from .rhythm import (
    HANDS,
    DrumKit,
    GoalSnapshot,
    HandGoal,
    Hit,
    RhythmClip,
    aim_inputs_from_snapshot,
    build_context_inputs,
    sample_hit_point,
    strike_velocity,
)
from .waist_aim import _DEFAULT_YAW_LIMIT, WaistAimConfig
from .warmstart import GoalPoseLibrary, arm_slice, warmstart_trajectory
from .warmstart_data import CONTEXT_FIELDS, bucket_for

__all__ = [
    "KIND_RHYTHM_IDLE",
    "KIND_POST_HIT",
    "KIND_FAR_FETCHED",
    "SCHEDULE_SCHEMA",
    "SCHEDULE_FIELDS",
    "PROVENANCE_FIELDS",
    "DualTransitionConfig",
    "build_dual_transition_clips",
    "dual_transition_kept_count",
    "IdleEntry",
    "build_idle_entry_index",
    "rare_drum_weights",
    "pack_schedule_from_hits",
    "project_schedule_slot0_to_future",
    "hits_after_clip_time",
    "refill_lookahead_from_clip",
    "RhythmChainScenarioConfig",
    "generate_rhythm_idle_scenarios",
    "generate_far_fetched_scenarios",
    "tip_straight_line_speed",
    "tip_crossover_depth_m",
    "effective_kind_fracs",
    "roller_clip_list",
    "sample_post_hit_tip",
    "TIP_MODE_HIT",
    "TIP_MODE_AROUND",
    "TIP_MODE_CYMBAL_BELOW",
]

#: Phase-2 scenario kinds (extend ``phase2_data`` KIND_*).
KIND_RHYTHM_IDLE = 4
KIND_POST_HIT = 5
KIND_FAR_FETCHED = 6

#: Bump when schedule/provenance column set changes; pool caches refuse mismatch.
SCHEDULE_SCHEMA = "rhythm_chain_v1_nf1_sched2"

SCHEDULE_FIELDS: tuple[str, ...] = (
    "sched_drum",
    "sched_pos",
    "sched_vel",
    "sched_dt",
    "sched_valid",
)

PROVENANCE_FIELDS: tuple[str, ...] = (
    "clip_i",
    "hand_hit_time",
)

#: Post-hit tip displace modes (Phase B distiller).
TIP_MODE_HIT = "hit"
TIP_MODE_AROUND = "around"
TIP_MODE_CYMBAL_BELOW = "cymbal_below"


def effective_kind_fracs(
    fracs: dict[int, float],
    counts: dict[int, int],
    *,
    log_fn=None,
    log_state: dict | None = None,
) -> dict[int, float]:
    """Renormalize fresh-mix fracs over nonempty families.

    When ``KIND_POST_HIT`` is advertised but ``N_B=0``, the surviving mass is
    reweighted (e.g. 0.30/0.10 idle/far → 0.75/0.25) and a one-shot loud log
    records the effective mix so training does not silently steal from idle
    while still advertising 0.60 post-hit.  Live roller harvests overlay
    ``counts[KIND_POST_HIT]`` once the on-policy buffer has rows; a second
    one-shot log records the mix flipping on.
    """
    present = {
        int(k): float(f)
        for k, f in fracs.items()
        if float(f) > 0.0 and int(counts.get(int(k), 0)) > 0
    }
    if not present:
        raise RuntimeError(
            f"no nonempty kind families for fresh mix "
            f"(fracs={fracs}, counts={counts})"
        )
    total = sum(present.values())
    out = {k: v / total for k, v in present.items()}
    want_b = float(fracs.get(KIND_POST_HIT, 0.0)) > 0.0
    have_b = int(counts.get(KIND_POST_HIT, 0)) > 0
    state = log_state if log_state is not None else {}
    if want_b and not have_b and log_fn is not None:
        if not state.get("empty_b_logged"):
            pretty = {k: round(v, 4) for k, v in sorted(out.items())}
            adv = {k: round(float(v), 4) for k, v in sorted(fracs.items())}
            log_fn(
                f"[data] N_B=0: renormalizing fresh mix to {pretty} "
                f"(CLI still advertises {adv})"
            )
            state["empty_b_logged"] = True
    if want_b and have_b and log_fn is not None:
        if not state.get("live_b_logged"):
            pretty = {k: round(v, 4) for k, v in sorted(out.items())}
            log_fn(
                f"[data] KIND_POST_HIT online "
                f"(n={int(counts[KIND_POST_HIT])}): fresh mix {pretty}"
            )
            state["live_b_logged"] = True
    return out


def roller_clip_list(
    kit: DrumKit,
    *,
    library: GoalPoseLibrary | None = None,
    cfg: Any = None,
    rcc_clips: list[RhythmClip] | None = None,
) -> list[RhythmClip]:
    """RCC train + synth dual clips; ``clip_i`` matches rhythm-idle generation.

    Pool rows store ``clip_i`` into ``rcc + synth`` (see
    :func:`build_idle_entry_index`).  The roller must refill lookahead from
    that same concatenated list or provenance indexes the wrong clip.
    """
    from .rhythm import default_rcc_dir, load_rcc_pool

    library = library if library is not None else GoalPoseLibrary()
    rcc = (
        list(rcc_clips) if rcc_clips is not None
        else list(load_rcc_pool(default_rcc_dir("train"), kit))
    )
    dual_cfg = _cfg_get(cfg, "dual", None) if cfg is not None else None
    if dual_cfg is None:
        dual_cfg = DualTransitionConfig()
    synth = build_dual_transition_clips(kit, library, cfg=dual_cfg)
    return rcc + list(synth)


def sample_post_hit_tip(
    kit: DrumKit,
    drum: int,
    aim_env: np.ndarray,
    n_out: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator,
    hit_radius: float = 0.08,
    hit_height: tuple[float, float] = (0.02, 0.12),
    around_scale: float = 1.35,
    below_drop: float = 0.03,
) -> np.ndarray:
    """Sample a near-drum tip pose for Phase B (ENV frame).

    Modes
    -----
    ``hit``: soft cylinder above the strike aim (rebound).
    ``around``: in-plane toward/past the rim (whiff / miss).
    ``cymbal_below``: past rim then slight underside (cymbals only).
    """
    aim = np.asarray(aim_env, float).reshape(3)
    n = np.asarray(n_out, float).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    # Orthonormal tangent frame in the head plane.
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, n)
    t1 /= max(float(np.linalg.norm(t1)), 1e-12)
    t2 = np.cross(n, t1)
    r = float(hit_radius)
    name = kit.names[int(drum)] if 0 <= int(drum) < len(kit.names) else ""
    is_cym = "cymbal" in name

    if mode == TIP_MODE_CYMBAL_BELOW and is_cym:
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = r * float(rng.uniform(1.05, around_scale))
        lateral = rad * (np.cos(ang) * t1 + np.sin(ang) * t2)
        return aim + lateral - float(below_drop) * n

    if mode == TIP_MODE_AROUND:
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = r * float(rng.uniform(0.85, around_scale))
        lateral = rad * (np.cos(ang) * t1 + np.sin(ang) * t2)
        h = float(rng.uniform(hit_height[0], hit_height[1]))
        return aim + lateral + h * n

    # hit / rebound (default)
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    rad = r * float(np.sqrt(rng.uniform(0.0, 1.0)))  # area-uniform disk
    lateral = rad * (np.cos(ang) * t1 + np.sin(ang) * t2)
    h = float(rng.uniform(hit_height[0], hit_height[1]))
    return aim + lateral + h * n


# ============================================================ synth dual =====
@dataclass
class DualTransitionConfig:
    """Feasibility knobs for simultaneous L a→b × R c→d clips."""

    sep_max: float = 0.45
    gap_s: float = 0.60
    max_gap_s: float = 1.0
    v_travel: float = 4.0
    allow_same_drum_layer: bool = False
    stick_min_sep: float = STICK_MIN_SEP
    cross_eps: float = 1e-3
    loudness: float = 0.5
    post_buffer_s: float = 1.0
    quantize: float = HIT_QUANTIZATION_S


def _tip(kit: DrumKit, drum: int) -> np.ndarray:
    return sample_hit_point(kit, int(drum), rng=None)


def _pair_ok(
    kit: DrumKit,
    d_l: int,
    d_r: int,
    *,
    sep_max: float,
    stick_min_sep: float,
    cross_eps: float,
) -> bool:
    p_l, p_r = _tip(kit, d_l), _tip(kit, d_r)
    sep = float(np.linalg.norm(p_l - p_r))
    if sep < stick_min_sep or sep > sep_max:
        return False
    if float(p_l[1]) < float(p_r[1]) - cross_eps:
        return False
    return True


def _travel_gap(
    kit: DrumKit,
    a: int,
    b: int,
    *,
    gap_s: float,
    max_gap_s: float,
    v_travel: float,
) -> float | None:
    dist = float(np.linalg.norm(_tip(kit, b) - _tip(kit, a)))
    need = dist / max(float(v_travel), 1e-9)
    g = max(float(gap_s), need)
    if g > float(max_gap_s) + 1e-12:
        return None
    return g


def build_dual_transition_clips(
    kit: DrumKit,
    library: GoalPoseLibrary | None = None,
    *,
    cfg: DualTransitionConfig | None = None,
) -> list[RhythmClip]:
    """Enumerate feasible simultaneous dual-hand transitions as short clips.

    Each clip: both hands hit ``(La, Ra)`` at t=0, then ``(Lb, Rb)`` at ``gap``.
    """
    cfg = cfg or DualTransitionConfig()
    library = library or GoalPoseLibrary()
    left = list(kit.eligible("left"))
    right = list(kit.eligible("right"))
    q = float(cfg.quantize)

    def _has(hand: str, di: int) -> bool:
        return library.has(hand, kit.names[int(di)])

    clips: list[RhythmClip] = []
    for la in left:
        for lb in left:
            if la == lb:
                continue
            for ra in right:
                for rb in right:
                    if ra == rb:
                        continue
                    if not cfg.allow_same_drum_layer and (la == ra or lb == rb):
                        continue
                    if not all(
                        (
                            _has("left", la),
                            _has("left", lb),
                            _has("right", ra),
                            _has("right", rb),
                        )
                    ):
                        continue
                    if not _pair_ok(
                        kit, la, ra,
                        sep_max=cfg.sep_max,
                        stick_min_sep=cfg.stick_min_sep,
                        cross_eps=cfg.cross_eps,
                    ):
                        continue
                    if not _pair_ok(
                        kit, lb, rb,
                        sep_max=cfg.sep_max,
                        stick_min_sep=cfg.stick_min_sep,
                        cross_eps=cfg.cross_eps,
                    ):
                        continue
                    g_l = _travel_gap(
                        kit, la, lb,
                        gap_s=cfg.gap_s, max_gap_s=cfg.max_gap_s, v_travel=cfg.v_travel,
                    )
                    g_r = _travel_gap(
                        kit, ra, rb,
                        gap_s=cfg.gap_s, max_gap_s=cfg.max_gap_s, v_travel=cfg.v_travel,
                    )
                    if g_l is None or g_r is None:
                        continue
                    gap = max(g_l, g_r)
                    gap_q = max(q, round(gap / q) * q) if q > 0 else gap
                    name = (
                        f"synth_dual_{kit.names[la]}_{kit.names[lb]}"
                        f"__{kit.names[ra]}_{kit.names[rb]}"
                    )
                    hits = [
                        Hit(0.0, 0.0, la, kit.names[la], "left", cfg.loudness),
                        Hit(0.0, 0.0, ra, kit.names[ra], "right", cfg.loudness),
                        Hit(gap_q, gap_q, lb, kit.names[lb], "left", cfg.loudness),
                        Hit(gap_q, gap_q, rb, kit.names[rb], "right", cfg.loudness),
                    ]
                    clips.append(
                        RhythmClip(
                            name=name,
                            hits=hits,
                            duration=float(gap_q + cfg.post_buffer_s),
                        )
                    )
    return clips


def dual_transition_kept_count(
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    *,
    cfg: DualTransitionConfig | None = None,
) -> int:
    kit = kit or DrumKit.from_layout()
    return len(build_dual_transition_clips(kit, library, cfg=cfg))


# ============================================================ entry index ====
@dataclass(frozen=True)
class IdleEntry:
    """One Phase A idle-entry draw key."""

    clip_i: int
    hand: str
    hit_time: float
    drum: int
    synth: bool = False


def build_idle_entry_index(
    rcc_clips: list[RhythmClip],
    synth_clips: list[RhythmClip] | None = None,
) -> tuple[list[IdleEntry], list[RhythmClip]]:
    """RCC mid-hit anchors + synth first-layer anchors; unified clip list.

    Returns ``(entries, clips)`` where ``clips = rcc + synth`` and ``clip_i``
    indexes that concatenated list (synth ids start at ``len(rcc)``).
    """
    synth_clips = list(synth_clips or [])
    clips = list(rcc_clips) + synth_clips
    entries: list[IdleEntry] = []
    for ci, clip in enumerate(rcc_clips):
        for h in clip.hits:
            entries.append(
                IdleEntry(
                    clip_i=ci,
                    hand=h.hand,
                    hit_time=float(h.time),
                    drum=int(h.drum),
                    synth=False,
                )
            )
    n_rcc = len(rcc_clips)
    for si, clip in enumerate(synth_clips):
        # First layer only (t≈0 simultaneous pair) — one entry per hand.
        layer0 = [h for h in clip.hits if abs(float(h.time)) < 1e-9]
        seen: set[str] = set()
        for h in layer0:
            if h.hand in seen:
                continue
            seen.add(h.hand)
            entries.append(
                IdleEntry(
                    clip_i=n_rcc + si,
                    hand=h.hand,
                    hit_time=float(h.time),
                    drum=int(h.drum),
                    synth=True,
                )
            )
    return entries, clips


def rare_drum_weights(entries: Iterable[IdleEntry]) -> np.ndarray:
    """Inverse-frequency weights over ``entries`` by drum (RCC-style)."""
    ents = list(entries)
    if not ents:
        return np.zeros(0)
    counts: dict[int, int] = {}
    for e in ents:
        counts[e.drum] = counts.get(e.drum, 0) + 1
    w = np.array([1.0 / counts[e.drum] for e in ents], dtype=float)
    s = float(w.sum())
    return w / s if s > 0 else np.full(len(ents), 1.0 / len(ents))


# ============================================================== schedule =====
def pack_schedule_from_hits(
    kit: DrumKit,
    post_goal_hits: dict[str, list[Hit]],
    *,
    goal_hit_time: dict[str, float],
    n_slots: int = 2,
    speed_min: float = 1.0,
    speed_max: float = 3.0,
) -> dict[str, np.ndarray]:
    """Pack up to ``n_slots`` post-goal hits per hand into schedule tensors.

    ``sched_dt`` is seconds from that hand's **active goal** strike time
    (``goal_hit_time[hand]``). Invalid slots: ``sched_valid=False``, drum=-1.
    """
    from .rhythm import strike_velocity

    sched_drum = np.full((2, n_slots), -1, dtype=np.int64)
    sched_pos = np.zeros((2, n_slots, 3), dtype=np.float64)
    sched_vel = np.zeros((2, n_slots, 3), dtype=np.float64)
    sched_dt = np.zeros((2, n_slots), dtype=np.float64)
    sched_valid = np.zeros((2, n_slots), dtype=bool)
    for hi, hand in enumerate(HANDS):
        t_goal = float(goal_hit_time.get(hand, 0.0))
        hits = list(post_goal_hits.get(hand, []))[:n_slots]
        for s, h in enumerate(hits):
            sched_drum[hi, s] = int(h.drum)
            sched_pos[hi, s] = sample_hit_point(kit, int(h.drum), rng=None)
            sched_vel[hi, s] = strike_velocity(
                kit, int(h.drum), float(h.velocity),
                speed_min=speed_min, speed_max=speed_max,
            )
            sched_dt[hi, s] = float(h.time) - t_goal
            sched_valid[hi, s] = True
    return {
        "sched_drum": sched_drum,
        "sched_pos": sched_pos,
        "sched_vel": sched_vel,
        "sched_dt": sched_dt,
        "sched_valid": sched_valid,
    }


def project_schedule_slot0_to_future(
    schedule: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map ``schedule[..., 0]`` → context future tensors with ``F=1``."""
    valid = np.asarray(schedule["sched_valid"][:, 0], bool)
    return {
        "future_drum": np.asarray(schedule["sched_drum"][:, :1], np.int64),
        "future_pos": np.asarray(schedule["sched_pos"][:, :1, :], float),
        "future_vel": np.asarray(schedule["sched_vel"][:, :1, :], float),
        # Caller converts to hit-relative ``future_gap`` in build_context_inputs;
        # here we expose from-goal dt as a convenience for snapshot builders.
        "future_dt": np.asarray(schedule["sched_dt"][:, :1], float),
        "future_valid": valid.reshape(2, 1),
    }


def hits_after_clip_time(
    clip: RhythmClip,
    hand: str,
    t_exclusive: float,
    count: int,
    *,
    eps: float = 1e-9,
) -> list[Hit]:
    """Next ``count`` hits for ``hand`` strictly after ``t_exclusive`` (clip time)."""
    out = [
        h for h in clip.hand_hits(hand)
        if float(h.time) > float(t_exclusive) + eps
    ]
    return out[: int(count)]


def refill_lookahead_from_clip(
    kit: DrumKit,
    clip: RhythmClip,
    hand: str,
    *,
    new_goal_hit_time: float,
    n_future: int = 1,
    n_schedule_extra: int = 1,
    speed_min: float = 1.0,
    speed_max: float = 3.0,
) -> tuple[dict[str, np.ndarray], bool]:
    """Refill ``future[0..]`` (+ optional schedule extras) after a promote.

    Pulls hits from ``clip`` strictly after the new active goal's clip time.
    Returns ``(fields, ok)`` where ``ok`` is False when fewer than ``n_future``
    hits remain (clip tail -- caller should drop / respawn).

    ``fields`` keys (ENV frame, goal-relative ``future_gap`` / ``sched_dt``):
    ``future_drum/pos/vel/gap/valid`` shaped ``(n_future,)`` / ``(n_future, 3)``,
    and when ``n_schedule_extra > 0`` also ``sched_*`` for slots beyond future
    (typically schedule[1] with ``n_future=1``, ``n_schedule_extra=1``).
    """
    n_f = max(int(n_future), 0)
    n_x = max(int(n_schedule_extra), 0)
    need = n_f + n_x
    hits = hits_after_clip_time(clip, hand, new_goal_hit_time, need)
    ok = len(hits) >= n_f
    t_goal = float(new_goal_hit_time)

    fut_drum = np.full(n_f, -1, dtype=np.int64)
    fut_pos = np.zeros((n_f, 3), dtype=np.float64)
    fut_vel = np.zeros((n_f, 3), dtype=np.float64)
    fut_gap = np.zeros(n_f, dtype=np.float64)
    fut_valid = np.zeros(n_f, dtype=bool)
    for j in range(min(n_f, len(hits))):
        h = hits[j]
        fut_drum[j] = int(h.drum)
        fut_pos[j] = sample_hit_point(kit, int(h.drum), rng=None)
        fut_vel[j] = strike_velocity(
            kit, int(h.drum), float(h.velocity),
            speed_min=speed_min, speed_max=speed_max,
        )
        fut_gap[j] = float(h.time) - t_goal
        fut_valid[j] = True

    out: dict[str, np.ndarray] = {
        "future_drum": fut_drum,
        "future_pos": fut_pos,
        "future_vel": fut_vel,
        "future_gap": fut_gap,
        "future_valid": fut_valid,
    }
    if n_x > 0:
        rest = hits[n_f: n_f + n_x]
        sched_drum = np.full(n_x, -1, dtype=np.int64)
        sched_pos = np.zeros((n_x, 3), dtype=np.float64)
        sched_vel = np.zeros((n_x, 3), dtype=np.float64)
        sched_dt = np.zeros(n_x, dtype=np.float64)
        sched_valid = np.zeros(n_x, dtype=bool)
        for j, h in enumerate(rest):
            sched_drum[j] = int(h.drum)
            sched_pos[j] = sample_hit_point(kit, int(h.drum), rng=None)
            sched_vel[j] = strike_velocity(
                kit, int(h.drum), float(h.velocity),
                speed_min=speed_min, speed_max=speed_max,
            )
            sched_dt[j] = float(h.time) - t_goal
            sched_valid[j] = True
        out.update({
            "sched_drum": sched_drum,
            "sched_pos": sched_pos,
            "sched_vel": sched_vel,
            "sched_dt": sched_dt,
            "sched_valid": sched_valid,
        })
    return out, ok


@dataclass
class RhythmChainScenarioConfig:
    """Knobs for Phase A / far-fetched generation (subset of the master plan)."""

    lead_s: float = 1.0
    hit_time_jitter_s: float = 0.05
    tip_speed_max: float = 4.0
    synth_transition_frac: float = 0.25
    num_future: int = 1
    schedule_slots: int = 2
    dual: DualTransitionConfig = field(default_factory=DualTransitionConfig)
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)
    dt: float = CONTROL_DT
    #: Modest Gaussian arm noise on ready arms for Phase A (rad).
    idle_arm_noise: float = 0.05
    #: Near-zero joint velocity noise scale (fraction of Isaac rating).
    idle_qd_scale: float = 0.02
    #: Unused: far starts are eval-style box draws, not ready+noise.
    far_arm_noise: float = 0.18
    #: Discrete eval-grid horizons / speeds (seconds, m/s). Far samples these
    #: cells the rolling 81-grid uses, not a continuous lead window.
    far_lead_s: tuple[float, float] = (0.40, 1.20)
    far_horizons: tuple[float, ...] = (0.45, 0.60, 0.80)
    far_speeds: tuple[float, ...] = (1.0, 2.0, 3.0)
    far_start_q_scale: float = 0.35
    far_exclude_eval_q_tol: float = 1e-3
    far_future_gap_s: float = 0.55
    start_sdf_tries: int = 8
    #: Max torso-frame tip crossover depth ``max(0, tip_R.y - tip_L.y)`` (m)
    #: accepted on rhythm-idle starts. ``None`` disables the gate.
    idle_max_tip_crossover_m: float | None = 0.05
    #: Same gate for far-fetched starts (looser: some crossover allowed).
    far_max_tip_crossover_m: float | None = 0.15
    waist_cfg: WaistAimConfig | None = None
    idle_cylinder_path: str | None = "default"
    idle_home_path: str | None = None  # deprecated alias
    idle_settle_wp: int = 10
    vel_reject_scale: float = 1.5
    seed_slew_eta: float | None = None
    waist_upright_ease_wp: int | None = None
    randomize_kit: bool = True
    hit_radius_frac: float = 0.6
    speed_min: float = 1.0
    speed_max: float = 3.0
    num_drums: int = 8
    horizon: int = 100
    buckets: tuple[int, ...] = (25, 50, 100)
    stick_rand: Any = None
    kit_randomization: Any = None


def tip_straight_line_speed(
    fk: ChainFK,
    q0: np.ndarray,
    hand: str,
    p_star_env: np.ndarray,
    t_hit: float,
) -> float:
    """``‖p_tip(q0) - p_star‖ / t_hit`` in the env frame (m/s)."""
    qt = torch.as_tensor(np.asarray(q0, float)[None, :], dtype=torch.float64)
    tip = tip_position(fk, qt, hand, frame="env")[0].detach().cpu().numpy()
    dist = float(np.linalg.norm(np.asarray(p_star_env, float) - tip))
    return dist / max(float(t_hit), 1e-9)


def _cfg_get(cfg: Any, name: str, default):
    return getattr(cfg, name, default)


def _yaw_limit(cfg: Any) -> float:
    wcfg = _cfg_get(cfg, "waist_cfg", None)
    if wcfg is not None:
        return float(wcfg.yaw_limit)
    return float(_DEFAULT_YAW_LIMIT)


def _jitter_hit(h: Hit, rng: np.random.Generator, jitter_s: float) -> Hit:
    if jitter_s <= 0.0:
        return h
    return replace(h, time=float(h.time) + float(rng.uniform(-jitter_s, jitter_s)))


def _sdf_clear(sampler, q: np.ndarray, kit: DrumKit) -> bool:
    t = lambda a: torch.as_tensor(  # noqa: E731
        np.asarray(a, float), device=sampler.device, dtype=sampler.dtype
    )
    with torch.no_grad():
        _p, det = sampler.sdf.forward(
            t(q[None]),
            drum_head_centers=t(kit.positions_in("env")).unsqueeze(0),
            drum_radii=t(kit.radii).unsqueeze(0),
            return_details=True,
        )
    return float(det.min_sdf[0].item()) > 0.0


def _tip_under_kit(fk: ChainFK, q: np.ndarray, kit: DrumKit) -> bool:
    """True if either bead is in a drum's XY disk and below that head."""
    from .fk import stick_tip_offset

    qt = torch.as_tensor(np.asarray(q, float)[None, :], dtype=torch.float64,
                        device=fk.device)
    heads = np.asarray(kit.positions_in("pelvis"), float)
    radii = np.asarray(kit.radii, float)
    with torch.no_grad():
        for hand in HANDS:
            R, p = fk.frame(qt, f"{hand}_wrist")
            loc = torch.as_tensor(
                stick_tip_offset(hand), dtype=qt.dtype, device=qt.device
            ).view(1, 3, 1)
            tip = (p + (R @ loc).squeeze(-1))[0].detach().cpu().numpy()
            dxy = np.hypot(tip[0] - heads[:, 0], tip[1] - heads[:, 1])
            if np.any((dxy <= radii) & (tip[2] < heads[:, 2])):
                return True
    return False


#: Rolling-eval ``generate_grid(seed=0)``. Far uses any other stream and
#: additionally drops starts that land on these poses.
EVAL_GRID_SEED = 0


@lru_cache(maxsize=2)
def _eval_grid_q0s():
    """``(M, 17)`` start poses from the rolling eval grid (seed 0, extra drums)."""
    from .eval_harness import TrajectoryScorer, generate_grid

    sc = TrajectoryScorer(device="cpu")
    cases = generate_grid(
        seed=EVAL_GRID_SEED, grid="standard", scorer=sc,
        include_extra_drums=True,
        start_above_drums=("snare", "floor_tom"),
        start_clearance_m=0.02,
    )
    return np.stack([np.asarray(c.q_start, float) for c in cases])


def _overlaps_eval_start(q0: np.ndarray, eval_q0: np.ndarray, tol: float) -> bool:
    if eval_q0.size == 0:
        return False
    return bool(np.min(np.linalg.norm(eval_q0 - np.asarray(q0, float)[None, :], axis=1)) < tol)


def _eval_style_drum_hand() -> dict[str, str]:
    from .eval_harness import DRUM_HAND

    out = dict(DRUM_HAND)
    out.setdefault("floor_tom", "right")
    return out


def tip_crossover_depth_m(fk: ChainFK, q0: np.ndarray) -> float:
    """Torso-frame tip crossover depth ``max(0, tip_R.y - tip_L.y)`` (metres).

    Correct lateral order has left tip at greater ``+y`` (robot-left) than the
    right tip, so depth is zero. Positive depth is how far the tips have swapped.
    """
    qt = torch.as_tensor(np.asarray(q0, float)[None, :], dtype=torch.float64,
                        device=fk.device)
    with torch.no_grad():
        yl = float(tip_in_torso(fk, qt, "left")[0, 1].item())
        yr = float(tip_in_torso(fk, qt, "right")[0, 1].item())
    return float(max(0.0, yr - yl))


def _idle_q0(
    *,
    idle_library,
    names: tuple[str, ...],
    rng: np.random.Generator,
    cfg: Any,
    q_bounds,
    sampler,
    kit_s: DrumKit,
    arm_noise: float,
    fk: ChainFK | None = None,
    prefer_ready: bool = True,
    max_tip_crossover_m: float | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Ready- or library-biased start + yaw + arm noise; SDF / crossover gates.

    Returns ``(q0, None)`` on success. On exhaustion returns
    ``(None, "sdf"|"crossover")`` for the last failed gate (callers bump the
    matching reject counter).
    """
    from .constants import ready_joint_pos

    lo, hi = q_bounds
    yaw_i = names.index("waist_yaw_joint")
    lim = min(float(_yaw_limit(cfg)), float(min(-lo[yaw_i], hi[yaw_i])))
    tries = max(int(_cfg_get(cfg, "start_sdf_tries", 8)), 1)
    last_reject = "sdf"
    for _ in range(tries):
        yaw = float(rng.uniform(-lim, lim))
        q0 = ready_joint_pos(names).copy()
        q0[yaw_i] = yaw
        if not prefer_ready:
            for hi_h, hand in enumerate(HANDS):
                k = int(rng.integers(idle_library.K))
                q0[arm_slice(hand, names)] = idle_library.q_arm[hi_h, k]
        if arm_noise > 0.0:
            for hand in HANDS:
                cols = arm_slice(hand, names)
                q0[cols] = q0[cols] + rng.normal(0.0, arm_noise, size=len(cols))
        q0 = np.clip(q0, lo, hi)
        q0[yaw_i] = float(np.clip(q0[yaw_i], -lim, lim))
        if not _sdf_clear(sampler, q0, kit_s):
            last_reject = "sdf"
            continue
        if max_tip_crossover_m is not None:
            if fk is None:
                raise ValueError("fk is required when max_tip_crossover_m is set")
            if tip_crossover_depth_m(fk, q0) > float(max_tip_crossover_m):
                last_reject = "crossover"
                continue
        return q0, None
    return None, last_reject


def _resolve_idle_library(cfg, names):
    path = _cfg_get(cfg, "idle_cylinder_path", None)
    if path is None:
        path = _cfg_get(cfg, "idle_home_path", "default")
    return resolve_idle_cylinder_library(path, joint_names=names)


def _hand_goal_from_hit(
    kit: DrumKit,
    hit: Hit,
    tth: float,
    rng: np.random.Generator,
    *,
    hit_radius_frac: float,
    speed_min: float,
    speed_max: float,
) -> HandGoal:
    return HandGoal(
        active=True,
        drum=int(hit.drum),
        position=sample_hit_point(
            kit, int(hit.drum), rng, hit_radius_frac=hit_radius_frac
        ),
        velocity=strike_velocity(
            kit, int(hit.drum), float(hit.velocity),
            speed_min=speed_min, speed_max=speed_max,
        ),
        time_to_hit=float(tth),
        drum_name=hit.drum_name,
        loudness=float(hit.velocity),
    )


def _dual_active_goals(
    clip: RhythmClip,
    kit: DrumKit,
    t_anchor: float,
    rng: np.random.Generator,
    *,
    lead_s: float,
    jitter_s: float,
    schedule_slots: int,
    hit_radius_frac: float,
    speed_min: float,
    speed_max: float,
    dt: float,
) -> tuple[dict[str, HandGoal], dict[str, list[Hit]], dict[str, float], np.ndarray] | None:
    """Remap clip hits at ``t_anchor`` with ``lead_s`` + per-hit independent jitter.

    Returns ``(goals, post_goal_hits, goal_hit_time, hand_hit_time)`` or ``None``
    when no hand is active or an active hand lacks a post-goal beat.
    """
    goals: dict[str, HandGoal] = {}
    post: dict[str, list[Hit]] = {}
    goal_hit_time: dict[str, float] = {}
    hand_hit_time = np.full(2, -1.0, dtype=np.float64)
    any_active = False
    for hi, hand in enumerate(HANDS):
        nxt = clip.next_hits(hand, float(t_anchor), 1 + int(schedule_slots))
        if not nxt:
            goals[hand] = HandGoal.inactive()
            post[hand] = []
            continue
        g_hit = nxt[0]
        post_raw = list(nxt[1 : 1 + int(schedule_slots)])
        if not post_raw:
            return None
        jit_g = float(rng.uniform(-jitter_s, jitter_s)) if jitter_s > 0 else 0.0
        tth = (float(g_hit.time) - float(t_anchor)) + float(lead_s) + jit_g
        tth = max(tth, float(dt))
        goals[hand] = _hand_goal_from_hit(
            kit, g_hit, tth, rng,
            hit_radius_frac=hit_radius_frac,
            speed_min=speed_min, speed_max=speed_max,
        )
        # Per-hit independent jitter on schedule beats (order preserved).
        post_j: list[Hit] = []
        t_prev = float(g_hit.time)
        for ph in post_raw:
            pj = _jitter_hit(ph, rng, jitter_s)
            if float(pj.time) < t_prev + 1e-6:
                pj = replace(pj, time=t_prev + 1e-6)
            post_j.append(pj)
            t_prev = float(pj.time)
        post[hand] = post_j
        goal_hit_time[hand] = float(g_hit.time)
        hand_hit_time[hi] = float(g_hit.time)
        any_active = True
    if not any_active:
        return None
    return goals, post, goal_hit_time, hand_hit_time


def _snapshot_from_goals_and_schedule(
    goals: dict[str, HandGoal],
    post: dict[str, list[Hit]],
    goal_hit_time: dict[str, float],
    kit: DrumKit,
    *,
    schedule_slots: int,
    speed_min: float,
    speed_max: float,
) -> tuple[GoalSnapshot, dict[str, np.ndarray]]:
    sched = pack_schedule_from_hits(
        kit, post, goal_hit_time=goal_hit_time, n_slots=int(schedule_slots),
        speed_min=speed_min, speed_max=speed_max,
    )
    future: dict[str, list[HandGoal]] = {h: [] for h in HANDS}
    for hi, hand in enumerate(HANDS):
        g = goals[hand]
        if not g.active or not post.get(hand):
            continue
        h0 = post[hand][0]
        dt_from_goal = float(h0.time) - float(goal_hit_time[hand])
        future[hand] = [
            HandGoal(
                active=True,
                drum=int(h0.drum),
                position=np.asarray(sched["sched_pos"][hi, 0], float),
                velocity=np.asarray(sched["sched_vel"][hi, 0], float),
                time_to_hit=float(g.time_to_hit) + dt_from_goal,
                drum_name=h0.drum_name,
                loudness=float(h0.velocity),
            )
        ]
    snap = GoalSnapshot(t_now=0.0, goal=goals, future=future)
    return snap, sched


_LONG_FIELDS = frozenset({"goal_drum", "future_drum"})
_BOOL_FIELDS = frozenset(
    {"drum_valid", "goal_valid", "future_valid", "idle_valid", "stick_valid", "aim_valid"}
)


def _empty_bucket_keys(cfg) -> tuple[str, ...]:
    from .phase2_data import STRIKE_FIELDS

    return (
        "q0", "qd0", "init", "hit", "active", "drum", "start_drum", "kind",
        *STRIKE_FIELDS, *CONTEXT_FIELDS, *SCHEDULE_FIELDS, *PROVENANCE_FIELDS,
    )


def _append_scenario(
    rec: dict[str, list],
    *,
    q0, qd0, init, hit, active, drum, start_drum, kind,
    kit_s, snap, ctx, schedule, clip_i, hand_hit_time, sticks=None,
):
    from .phase2_data import STRIKE_FIELDS, _empty_strike, _strike_row
    from .strike import shaft_hit_allowed

    rec["q0"].append(np.asarray(q0, np.float32))
    rec["qd0"].append(np.asarray(qd0, np.float32))
    rec["init"].append(np.asarray(init, np.float32))
    rec["hit"].append(np.array([hit["left"], hit["right"]], dtype=np.int64))
    rec["active"].append(np.array(active, dtype=bool))
    rec["drum"].append(np.asarray(drum, np.int64))
    rec["start_drum"].append(np.asarray(start_drum, np.int64))
    rec["kind"].append(np.int64(kind))
    rows = []
    for h in HANDS:
        if hit[h] < 0:
            rows.append(_empty_strike())
        else:
            g = snap.goal[h]
            rows.append(_strike_row(kit_s, g.drum, g.position, g.velocity))
    for hi_, s in enumerate(np.asarray(start_drum, np.int64)):
        rows[hi_]["start_shaft"] = (
            float(shaft_hit_allowed(kit_s.names[int(s)])) if int(s) >= 0 else 0.0
        )
    for f in STRIKE_FIELDS:
        rec[f].append(
            np.stack([np.asarray(r[f], float) for r in rows]).astype(np.float32)
        )
    for f in CONTEXT_FIELDS:
        v = getattr(ctx, f)[0].numpy()
        rec[f].append(
            v.astype(np.int64) if f in _LONG_FIELDS
            else (v.astype(bool) if f in _BOOL_FIELDS else v.astype(np.float32))
        )
    for f in SCHEDULE_FIELDS:
        rec[f].append(np.asarray(schedule[f]))
    rec["clip_i"].append(np.int64(clip_i))
    rec["hand_hit_time"].append(np.asarray(hand_hit_time, np.float32))


def _pack_buckets(out: dict[int, dict[str, list]], stats: dict) -> dict:
    packed = {
        b: {k: np.stack(v) for k, v in rec.items()}
        for b, rec in out.items() if rec["q0"]
    }
    stats["peak_rate_ratio"] = np.asarray(stats["peak_rate_ratio"], dtype=np.float32)
    return {"buckets": packed, "stats": stats}


def _sample_entry(
    rng: np.random.Generator,
    rcc_ents: list[IdleEntry],
    synth_ents: list[IdleEntry],
    rcc_w: np.ndarray | None,
    synth_frac: float,
) -> IdleEntry | None:
    use_synth = bool(synth_ents) and (
        (not rcc_ents) or float(rng.uniform()) < float(synth_frac)
    )
    if use_synth:
        return synth_ents[int(rng.integers(len(synth_ents)))]
    if not rcc_ents:
        return None
    if rcc_w is not None and len(rcc_w) == len(rcc_ents):
        return rcc_ents[int(rng.choice(len(rcc_ents), p=rcc_w))]
    return rcc_ents[int(rng.integers(len(rcc_ents)))]


def generate_rhythm_idle_scenarios(
    n: int,
    clips: list[RhythmClip],
    *,
    seed: int,
    cfg: Any = None,
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    sampler=None,
) -> dict:
    """Phase A idle mid-rhythm scenarios (``KIND_RHYTHM_IDLE``).

    Same ``{buckets, stats}`` shape as
    :func:`~amortized_opt.phase2_data.generate_cost_scenarios`.
    """
    from .phase2_data import Phase2ScenarioConfig
    from .rhythm import KitRandomization

    if cfg is None:
        cfg = Phase2ScenarioConfig(
            hit_time_jitter_s=0.05,
            num_future=1,
            eval_like_frac=0.0,
            cold_start_frac=0.0,
            pair_frac=0.0,
            lead_s=1.0,
            tip_speed_max=4.0,
            synth_transition_frac=0.25,
        )
    rng = np.random.default_rng(seed)
    names = tuple(_cfg_get(cfg, "joint_names", G1_UPPER_BODY_JOINTS))
    kit = kit or DrumKit.from_layout()
    library = library or GoalPoseLibrary(joint_names=names)
    if sampler is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=library)
    q_bounds = urdf_position_limits(names)
    v_limit = isaac_velocity_limits(names)
    H = int(_cfg_get(cfg, "horizon", 100))
    buckets = tuple(_cfg_get(cfg, "buckets", (25, 50, 100)))
    dt = float(_cfg_get(cfg, "dt", CONTROL_DT))
    lead_s = float(_cfg_get(cfg, "lead_s", 1.0))
    jitter_s = float(_cfg_get(cfg, "hit_time_jitter_s", 0.05))
    tip_max = float(_cfg_get(cfg, "tip_speed_max", 4.0))
    synth_frac = float(_cfg_get(cfg, "synth_transition_frac", 0.25))
    n_fut = 1  # locked for this family
    n_slots = int(_cfg_get(cfg, "schedule_slots", 2))
    idle_library = _resolve_idle_library(cfg, names)
    if idle_library is None:
        raise RuntimeError(
            "generate_rhythm_idle_scenarios requires an IdleCylinderLibrary "
            "(idle_cylinder_path='default' or a path)"
        )
    fk = ChainFK(device="cpu", dtype=torch.float64)
    idle_cylinder = None
    try:
        idle_cylinder = derive_cylinder(kit, fk)
    except Exception:
        idle_cylinder = idle_library.cylinder
    dual_cfg = _cfg_get(cfg, "dual", None)
    if dual_cfg is None:
        dual_cfg = DualTransitionConfig()
    kit_rand = _cfg_get(cfg, "kit_randomization", None)
    if kit_rand is None and _cfg_get(cfg, "randomize_kit", True):
        kit_rand = KitRandomization()

    synth_clips = build_dual_transition_clips(kit, library, cfg=dual_cfg)
    entries, all_clips = build_idle_entry_index(list(clips), synth_clips)
    rcc_ents = [e for e in entries if not e.synth]
    synth_ents = [e for e in entries if e.synth]
    rcc_w = rare_drum_weights(rcc_ents) if rcc_ents else None

    keys = _empty_bucket_keys(cfg)
    out: dict[int, dict[str, list]] = {b: {k: [] for k in keys} for b in buckets}
    stats = {
        "requested": int(n), "accepted": 0,
        "reject_idle": 0, "reject_velocity": 0, "reject_library": 0,
        "reject_sdf": 0, "reject_crossover": 0, "reject_tip_speed": 0,
        "reject_future": 0, "reject_entry": 0,
        "dual_active": 0, "asymmetric": 0, "single_idle": 0,
        "synth_entry": 0, "rcc_entry": 0,
        "peak_rate_ratio": [],
    }

    arm_noise = float(_cfg_get(cfg, "idle_arm_noise", 0.05))
    cross_cap = _cfg_get(cfg, "idle_max_tip_crossover_m", 0.05)
    if cross_cap is not None:
        cross_cap = float(cross_cap)
    qd_scale = float(_cfg_get(cfg, "idle_qd_scale", 0.02))
    hit_radius_frac = float(_cfg_get(cfg, "hit_radius_frac", 0.6))
    speed_min = float(_cfg_get(cfg, "speed_min", 1.0))
    speed_max = float(_cfg_get(cfg, "speed_max", 3.0))
    vel_reject = float(_cfg_get(cfg, "vel_reject_scale", 1.5))
    waist_cfg = _cfg_get(cfg, "waist_cfg", None)
    idle_settle = int(_cfg_get(cfg, "idle_settle_wp", 10))
    slew = _cfg_get(cfg, "seed_slew_eta", None)
    upright = _cfg_get(cfg, "waist_upright_ease_wp", None)
    num_drums = int(_cfg_get(cfg, "num_drums", 8))

    for _ in range(int(n)):
        entry = _sample_entry(rng, rcc_ents, synth_ents, rcc_w, synth_frac)
        if entry is None:
            stats["reject_entry"] += 1
            continue
        kit_s = (
            kit.randomized(rng, kit_rand)
            if _cfg_get(cfg, "randomize_kit", True) and kit_rand is not None
            else kit
        )
        clip = all_clips[int(entry.clip_i)]
        remapped = _dual_active_goals(
            clip, kit_s, float(entry.hit_time), rng,
            lead_s=lead_s, jitter_s=jitter_s, schedule_slots=n_slots,
            hit_radius_frac=hit_radius_frac,
            speed_min=speed_min, speed_max=speed_max, dt=dt,
        )
        if remapped is None:
            stats["reject_future"] += 1
            continue
        goals_h, post, goal_hit_time, hand_hit_time = remapped
        # Lookahead gate: every active hand must have future[0].
        if any(
            goals_h[h].active and not post.get(h)
            for h in HANDS
        ):
            stats["reject_future"] += 1
            continue

        q0, reject = _idle_q0(
            idle_library=idle_library, names=names, rng=rng, cfg=cfg,
            q_bounds=q_bounds, sampler=sampler, kit_s=kit_s, arm_noise=arm_noise,
            fk=fk, prefer_ready=True, max_tip_crossover_m=cross_cap,
        )
        if q0 is None:
            key = "reject_crossover" if reject == "crossover" else "reject_sdf"
            stats[key] += 1
            continue
        qd0 = qd_scale * v_limit * rng.uniform(-1.0, 1.0, size=len(names))

        snap, schedule = _snapshot_from_goals_and_schedule(
            goals_h, post, goal_hit_time, kit_s,
            schedule_slots=n_slots, speed_min=speed_min, speed_max=speed_max,
        )
        # Projected future must be valid for every active hand.
        proj = project_schedule_slot0_to_future(schedule)
        bad_fut = False
        for hi, hand in enumerate(HANDS):
            if goals_h[hand].active and not bool(proj["future_valid"][hi, 0]):
                bad_fut = True
                break
        if bad_fut:
            stats["reject_future"] += 1
            continue

        hit = {
            h: (
                int(np.clip(round(snap.goal[h].time_to_hit / dt), 1, H))
                if snap.goal[h].active else -1
            )
            for h in HANDS
        }
        active = [hit[h] >= 0 for h in HANDS]
        if not any(active):
            stats["reject_idle"] += 1
            continue

        # Tip-speed filter on active hands.
        tip_bad = False
        for h in HANDS:
            if hit[h] < 0:
                continue
            g = snap.goal[h]
            spd = tip_straight_line_speed(fk, q0, h, g.position, g.time_to_hit)
            if spd > tip_max:
                tip_bad = True
                break
        if tip_bad:
            stats["reject_tip_speed"] += 1
            continue

        goals_ws: dict[str, tuple[str, float] | None] = {}
        aim: dict[str, np.ndarray] = {}
        _t, aim_tth, aim_futures = aim_inputs_from_snapshot(snap, frame="env")
        for h in HANDS:
            g = snap.goal[h]
            if hit[h] < 0:
                goals_ws[h] = None
                aim_tth.pop(h, None)
                continue
            goals_ws[h] = (g.drum_name, hit[h] * dt)
            aim[h] = np.asarray(g.position, float)
        if not all(library.has(h, goals_ws[h][0]) for h in HANDS if goals_ws[h] is not None):
            stats["reject_library"] += 1
            continue

        k_max = max(hit[h] for h in HANDS if hit[h] >= 0)
        Hb = bucket_for(k_max, buckets)
        sticks = None
        stick_rand = _cfg_get(cfg, "stick_rand", None)
        if stick_rand is not None:
            sticks = (stick_rand.sample(rng, "left"), stick_rand.sample(rng, "right"))

        ws = warmstart_trajectory(
            q0, qd0, horizon=Hb, goals=goals_ws, library=library,
            aim_targets=aim or None, aim_tth=aim_tth, aim_futures=aim_futures,
            dt=dt, joint_names=names, waist_cfg=waist_cfg, q_bounds=q_bounds,
            waist_upright_ease_wp=upright, slew_limit_eta=slew,
            idle_library=idle_library, idle_cylinder=idle_cylinder, fk=fk,
            idle_settle_wp=idle_settle,
        )
        init = np.asarray(ws.init_traj, float)
        ratio = float((np.asarray(ws.peak_rate_raw, float) / v_limit).max())
        stats["peak_rate_ratio"].append(ratio)
        if ratio > vel_reject:
            stats["reject_velocity"] += 1
            continue

        ctx = build_context_inputs(
            kit_s, snap, num_drums=num_drums, num_future=n_fut,
            frame="env", device="cpu", dtype=torch.float64,
            idle_library=idle_library, idle_cylinder=idle_cylinder, fk=fk,
            sticks=sticks,
            yaw_now=q0[names.index("waist_yaw_joint")], waist_cfg=waist_cfg,
        )
        drum = np.array(
            [snap.goal[h].drum if hit[h] >= 0 else -1 for h in HANDS], np.int64
        )
        src = np.full(2, -1, dtype=np.int64)
        _append_scenario(
            out[Hb],
            q0=q0, qd0=qd0, init=init, hit=hit, active=active, drum=drum,
            start_drum=src, kind=KIND_RHYTHM_IDLE, kit_s=kit_s, snap=snap,
            ctx=ctx, schedule=schedule, clip_i=int(entry.clip_i),
            hand_hit_time=hand_hit_time, sticks=sticks,
        )
        stats["accepted"] += 1
        if entry.synth:
            stats["synth_entry"] += 1
        else:
            stats["rcc_entry"] += 1
        if all(active):
            stats["dual_active"] += 1
            if hit["left"] != hit["right"]:
                stats["asymmetric"] += 1
        elif any(active):
            stats["single_idle"] += 1

    return _pack_buckets(out, stats)


def generate_far_fetched_scenarios(
    n: int,
    clips: list[RhythmClip] | None = None,
    *,
    seed: int,
    cfg: Any = None,
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    sampler=None,
) -> dict:
    """Eval-style box-grid family (``KIND_FAR_FETCHED``).

    Same cells as rolling eval: waist + striking arm in a posture box, idle at
    ready, rest, one strike, empty future, discrete horizons 0.45/0.60/0.80 s
    and speeds 1/2/3 m/s.  Independent RNG -- never ``EVAL_GRID_SEED``'s poses,
    which are dropped even if a draw lands on them.  Same ``{buckets, stats}``
    shape as the other rhythm families.
    """
    from .constants import ready_joint_pos
    from .eval_harness import _sample_box_start
    from .phase2_data import Phase2ScenarioConfig
    from .rhythm import KitRandomization

    del clips  # far does not index RCC
    if cfg is None:
        cfg = Phase2ScenarioConfig(
            num_future=1,
            eval_like_frac=0.0,
            cold_start_frac=0.0,
            pair_frac=0.0,
        )
    rng = np.random.default_rng(int(seed))
    names = tuple(_cfg_get(cfg, "joint_names", G1_UPPER_BODY_JOINTS))
    kit = kit or DrumKit.from_layout()
    library = library or GoalPoseLibrary(joint_names=names)
    if sampler is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=library)
    q_bounds = urdf_position_limits(names)
    q_lo, q_hi = urdf_position_limits(names, posture_limits=True)
    v_limit = isaac_velocity_limits(names)
    H = int(_cfg_get(cfg, "horizon", 100))
    buckets = tuple(_cfg_get(cfg, "buckets", (25, 50, 100)))
    dt = float(_cfg_get(cfg, "dt", CONTROL_DT))
    horizons = tuple(_cfg_get(cfg, "far_horizons", (0.45, 0.60, 0.80)))
    speeds = tuple(_cfg_get(cfg, "far_speeds", (1.0, 2.0, 3.0)))
    q_scale = float(_cfg_get(cfg, "far_start_q_scale", 0.35))
    n_fut = int(_cfg_get(cfg, "num_future", 1))
    n_slots = int(_cfg_get(cfg, "schedule_slots", 2))
    idle_library = _resolve_idle_library(cfg, names)
    if idle_library is None:
        raise RuntimeError(
            "generate_far_fetched_scenarios requires an IdleCylinderLibrary"
        )
    fk = ChainFK(device="cpu", dtype=torch.float64)
    idle_cylinder = None
    try:
        idle_cylinder = derive_cylinder(kit, fk)
    except Exception:
        idle_cylinder = idle_library.cylinder
    kit_rand = _cfg_get(cfg, "kit_randomization", None)
    if kit_rand is None and _cfg_get(cfg, "randomize_kit", True):
        kit_rand = KitRandomization()

    keys = _empty_bucket_keys(cfg)
    out: dict[int, dict[str, list]] = {b: {k: [] for k in keys} for b in buckets}
    stats = {
        "requested": int(n), "accepted": 0,
        "reject_sdf": 0, "reject_crossover": 0, "reject_tip_speed": 0,
        "reject_future": 0, "reject_box_start": 0, "reject_eval_overlap": 0,
        "reject_library": 0, "reject_velocity": 0, "reject_idle": 0,
        "dual_active": 0, "single_idle": 0,
        "peak_rate_ratio": [],
    }
    hit_radius_frac = float(_cfg_get(cfg, "hit_radius_frac", 0.6))
    vel_reject = float(_cfg_get(cfg, "vel_reject_scale", 1.5))
    waist_cfg = _cfg_get(cfg, "waist_cfg", None)
    idle_settle = int(_cfg_get(cfg, "idle_settle_wp", 10))
    slew = _cfg_get(cfg, "seed_slew_eta", None)
    upright = _cfg_get(cfg, "waist_upright_ease_wp", None)
    num_drums = int(_cfg_get(cfg, "num_drums", 8))
    exclude_tol = float(_cfg_get(cfg, "far_exclude_eval_q_tol", 1e-3))
    eval_q0 = _eval_grid_q0s()
    hold = np.clip(ready_joint_pos(names), q_lo, q_hi)
    drum_hand = _eval_style_drum_hand()
    eval_names = [nm for nm in drum_hand if nm in kit.names and library.has(drum_hand[nm], nm)]
    if not eval_names:
        return _pack_buckets(out, stats)

    n_box_try = 256
    for _ in range(int(n)):
        kit_s = (
            kit.randomized(rng, kit_rand)
            if _cfg_get(cfg, "randomize_kit", True) and kit_rand is not None
            else kit
        )
        name = eval_names[int(rng.integers(len(eval_names)))]
        hand = drum_hand[name]
        other = "right" if hand == "left" else "left"
        if name not in kit_s.names or not library.has(hand, name):
            stats["reject_library"] += 1
            continue
        di = int(kit_s.index(name))
        horizon = float(horizons[int(rng.integers(len(horizons)))])
        speed = float(speeds[int(rng.integers(len(speeds)))])
        N = max(int(round(horizon / dt)), 4)
        N = int(np.clip(N, 1, H))
        t_lead = N * dt

        q0 = None
        for _try in range(n_box_try):
            cand = _sample_box_start(rng, hand, q_lo, q_hi, q_scale, hold=hold)
            if not _sdf_clear(sampler, cand, kit_s):
                continue
            if _tip_under_kit(fk, cand, kit_s):
                continue
            q0 = cand
            break
        if q0 is None:
            stats["reject_box_start"] += 1
            continue
        if _overlaps_eval_start(q0, eval_q0, exclude_tol):
            stats["reject_eval_overlap"] += 1
            continue
        qd0 = np.zeros(len(names), dtype=float)

        g_hit = Hit(0.0, 0.0, di, name, hand, 1.0)
        goal_h = _hand_goal_from_hit(
            kit_s, g_hit, t_lead, rng,
            hit_radius_frac=hit_radius_frac,
            speed_min=speed, speed_max=speed,
        )
        goals_h = {hand: goal_h, other: HandGoal.inactive()}
        post = {h: [] for h in HANDS}
        goal_hit_time = {hand: 0.0}
        hand_hit_time = np.full(2, -1.0, dtype=np.float64)
        hand_hit_time[HANDS.index(hand)] = 0.0
        snap, schedule = _snapshot_from_goals_and_schedule(
            goals_h, post, goal_hit_time, kit_s,
            schedule_slots=n_slots, speed_min=speed, speed_max=speed,
        )

        hit = {
            h: (
                int(np.clip(round(snap.goal[h].time_to_hit / dt), 1, H))
                if snap.goal[h].active else -1
            )
            for h in HANDS
        }
        active = [hit[h] >= 0 for h in HANDS]
        goals_ws = {
            h: ((snap.goal[h].drum_name, hit[h] * dt) if hit[h] >= 0 else None)
            for h in HANDS
        }
        aim = {
            h: np.asarray(snap.goal[h].position, float)
            for h in HANDS if hit[h] >= 0
        }
        if not all(library.has(h, goals_ws[h][0]) for h in HANDS if goals_ws[h] is not None):
            stats["reject_library"] += 1
            continue
        _t, aim_tth, aim_futures = aim_inputs_from_snapshot(snap, frame="env")
        k_max = max(hit[h] for h in HANDS if hit[h] >= 0)
        Hb = bucket_for(k_max, buckets)
        ws = warmstart_trajectory(
            q0, qd0, horizon=Hb, goals=goals_ws, library=library,
            aim_targets=aim or None, aim_tth=aim_tth, aim_futures=aim_futures,
            dt=dt, joint_names=names, waist_cfg=waist_cfg, q_bounds=q_bounds,
            waist_upright_ease_wp=upright, slew_limit_eta=slew,
            idle_library=idle_library, idle_cylinder=idle_cylinder, fk=fk,
            idle_settle_wp=idle_settle,
        )
        init = np.asarray(ws.init_traj, float)
        ratio = float((np.asarray(ws.peak_rate_raw, float) / v_limit).max())
        stats["peak_rate_ratio"].append(ratio)
        if ratio > vel_reject:
            stats["reject_velocity"] += 1
            continue

        ctx = build_context_inputs(
            kit_s, snap, num_drums=num_drums, num_future=n_fut,
            frame="env", device="cpu", dtype=torch.float64,
            idle_library=idle_library, idle_cylinder=idle_cylinder, fk=fk,
            yaw_now=q0[names.index("waist_yaw_joint")], waist_cfg=waist_cfg,
        )
        drum = np.array(
            [snap.goal[h].drum if hit[h] >= 0 else -1 for h in HANDS], np.int64
        )
        _append_scenario(
            out[Hb],
            q0=q0, qd0=qd0, init=init, hit=hit, active=active, drum=drum,
            start_drum=np.full(2, -1, dtype=np.int64), kind=KIND_FAR_FETCHED,
            kit_s=kit_s, snap=snap, ctx=ctx, schedule=schedule,
            clip_i=-1, hand_hit_time=hand_hit_time,
        )
        stats["accepted"] += 1
        stats["single_idle"] += 1

    return _pack_buckets(out, stats)
