"""Phase-1 (warm start) training data: scenarios, targets and auxiliary labels.

Section 6 of ``docs/amortized_optimizer_plan.md``.  Phase 1 is plain imitation
of a *cheap* trajectory:

    warm up the transformer to follow cheaply produced non optimized
    trajectories (literally just a quintic straight line to the closest joint
    position out of a library per drum ...) with noised input initializations
    ... we will also train it to follow a pre built waist aim system ... This
    training step will also include training the auxiliary terms to be accurate.

So one sample is

* a **start state** from :class:`~amortized_opt.starts.StartSampler` (90 %
  drum-departure, both hands placed on a drum they may play);
* a **goal context** from :mod:`amortized_opt.rhythm` -- a random instant in a
  random RCC clip against a *randomized* kit (domain randomization ON), giving
  each hand its own current goal and ``num_future`` upcoming hits;
* the **target trajectory** from
  :func:`~amortized_opt.warmstart.warmstart_trajectory` (per-arm quintic
  straight line to the library pose + the waist-aim yaw ramp), and
* the **input initialization**, which is that target plus OU + Gaussian noise.
  The noise is drawn fresh on the GPU at batch time (see :class:`WarmstartPool`),
  so a sample is never seen with the same perturbation twice.

Why i.i.d. scenarios and not :class:`~amortized_opt.mpc_gen.MPCGenerator`
-------------------------------------------------------------------------
The MPC generator is the phase-2 DAgger engine: its value is that the *init* is
the previous solution shifted, which only means something once the model is the
planner.  In phase 1 the init is a noised quintic by construction, so rolling a
simulated MPC buys nothing and costs 19 ms/tick.  It also currently winds the
waist up over long episodes (see the section-6 report), which would poison BC.
Independent scenarios are ~40x cheaper and cover the horizon range evenly.

Horizon bucketing
-----------------
Samples are sorted into buckets by their longest active hand horizon
(``{25, 50, 100}`` waypoints by default).  A bucket is a fixed tensor shape, so
``torch.compile`` sees exactly three static shapes instead of one padded-to-100
shape that wastes 4x the tokens on a 25-waypoint strike.

Auxiliary labels
----------------
Computed ONCE per pool entry on the GPU, from the *target* trajectory:

* ``vel`` / ``acc`` -- the C2 spline's own derivatives at each waypoint
  (:mod:`amortized_opt.tspline`, clamped to ``v0`` at the start, natural at the
  strike -- the same spline the cost builds);
* ``torque`` -- by default (``torque_mode="rnea_segmax"``)
  ``max(|tau_{i-1}|, |tau_i|)``, i.e. the peak |tau| over the segment ending at
  waypoint ``i``, from :class:`~amortized_opt.rnea.BatchedRNEA` driven by the
  ``knot_fd`` finite differences :mod:`amortized_opt.cost` uses in its default
  ``physics_mode``.  With ``torque_mode="tracking"`` it is instead the DEMANDED
  PD tracking torque ``|tau_rnea + tau_fb|`` at the waypoint, the very quantity
  ``w_track_torque`` prices -- see :func:`aux_targets`;
* ``fk_pos`` / ``fk_vel`` -- tip / wrist / elbow of both arms
  (:data:`~amortized_opt.tokenizer.FK_POINT_NAMES` order), positions from
  :class:`~amortized_opt.fk.ChainFK` at the waypoint and velocities by the
  scorer's finite difference ``(p_i - p_{i-1}) / dt``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .fk import ChainFK, StickRandomization, stick_tip_offset
from .idle_home import resolve_idle_home
from .ik import corrected_arm_terminals
from .limits import isaac_velocity_limits, urdf_position_limits
from .rhythm import (
    HANDS,
    DrumKit,
    KitRandomization,
    RhythmClip,
    aim_inputs_from_snapshot,
    build_context_inputs,
    default_rcc_dir,
    goal_snapshot,
    load_rcc_pool,
)
from .rnea import BatchedRNEA
from .tokenizer import FK_POINT_NAMES, ContextInputs, build_waypoint_masks, joint_group_masks
from .tspline import TrajectorySplineBuilder
from .waist_aim import WaistAimConfig
from .warmstart import GoalPoseLibrary, warmstart_trajectory

__all__ = [
    "CONTEXT_FIELDS",
    "ScenarioConfig",
    "NoiseConfig",
    "WarmstartPool",
    "bucket_for",
    "generate_scenarios",
    "generate_pool",
    "ou_kernel",
]

#: The :class:`~amortized_opt.tokenizer.ContextInputs` fields, in a fixed order.
CONTEXT_FIELDS: tuple[str, ...] = tuple(ContextInputs.__dataclass_fields__)

_LONG_FIELDS = frozenset({"goal_drum", "future_drum"})
_BOOL_FIELDS = frozenset(
    {"drum_valid", "goal_valid", "future_valid", "idle_valid", "stick_valid", "aim_valid"}
)


# =============================================================== config ======
@dataclass
class ScenarioConfig:
    """What a phase-1 scenario looks like. Every default is the plan's."""

    horizon: int = 100
    buckets: tuple[int, ...] = (25, 50, 100)
    dt: float = CONTROL_DT
    num_drums: int = 8
    num_future: int = 3
    #: Include single-hand target clips for every eligible hand/drum goal.
    all_hand_drum_goals: bool = True
    synthetic_goal_frac: float = 0.5

    #: Domain randomization of the kit (plan: per-drum <= 4 cm horizontal, whole
    #: kit <= 7.5 cm horizontal / 3.5 cm vertical).
    randomize_kit: bool = True
    kit_randomization: KitRandomization = field(default_factory=KitRandomization)

    hit_radius_frac: float = 0.6
    speed_min: float = 1.0
    speed_max: float = 3.0

    #: Rhythm augmentation at snapshot time (see
    #: :func:`~amortized_opt.rhythm.goal_snapshot`); 0.0 = old stream.
    hit_time_jitter_s: float = 0.0
    hit_loudness_jitter: float = 0.0

    #: Fraction of scenarios whose "now" is snapped to within
    #: ``short_horizon_wp`` waypoints of a hit -- the plan's "include a few more
    #: training examples with shorter horizon lengths".
    short_horizon_frac: float = 0.25
    short_horizon_wp: int = 5

    #: Reject a scenario whose quintic target commands a per-joint peak |qd|
    #: beyond ``vel_reject_scale`` x the joint's rating.  The seed is
    #: velocity-unlimited by construction and an absurd hit spacing makes it
    #: command hundreds of rad/s; imitating that poisons BC.
    vel_reject_scale: float = 1.5

    #: Drop a scenario whose two hands are both idle (nothing to plan).
    require_active_hand: bool = True

    # ---------------------------------------------------------- idle arm --
    #: Park an idle hand at the home posture instead of holding it wherever it
    #: started.  ``"default"`` uses ``assets/idle_home_poses.json`` when it
    #: exists, a path names another table, ``None`` keeps the hold-in-place
    #: target.  Mirrors :attr:`Phase2ScenarioConfig.idle_home_path` field for
    #: field, and for the same reason: the seed IS the statement of where home
    #: is, because the decoder reads it as its waypoint tokens.
    #:
    #: This also decides whether the encoder's IDLE context token carries a
    #: payload.  Without a table ``build_context_inputs`` leaves the idle block
    #: zeroed and ``idle_valid`` False, the encoder masks those rows out, and
    #: ``encoder.idle_proj`` receives exactly zero gradient for the whole run --
    #: so a phase-1 model trained without one hands phase 2 a randomly
    #: initialised token, having learnt nothing about the idle arm and having
    #: been taught the OPPOSITE behaviour by the hold-in-place target.
    #:
    #: ``None`` reproduces the pre-2026-08-12 pool exactly.
    idle_home_path: str | None = None
    #: Waypoints the retraction eases over before holding home.  Stretched
    #: automatically when the arm is far enough away that this would exceed its
    #: speed rating (``warmstart_trajectory``'s idle branch does the stretch).
    idle_settle_wp: int = 10
    #: REST SCENARIOS.  Fraction of draws kept even when NEITHER hand has a
    #: goal -- the bar rest ``require_active_hand`` otherwise throws away.  Only
    #: meaningful with a home table: without one the row is a pose held in place
    #: with nothing to imitate but itself.  Single-hand-idle scenarios are NOT
    #: affected -- those were always kept, and are ~32 % of the pool.
    #: 0.0 reproduces the pre-2026-08-12 distribution exactly.
    rest_frac: float = 0.0

    #: THE FITTED STICK.  ``None`` = every sample uses the nominal stick from
    #: ``constants.py``, which is the shipped behaviour.  Set it and each
    #: scenario draws its own per-hand stick, which is carried in the context
    #: (``TYPE_STICK``) and priced by the cost -- both are required, or the
    #: variation is unlearnable noise the way the drum jitter was.
    stick_rand: "StickRandomization | None" = None
    #: One-shot full-arm DLS on quintic terminals (kit jitter / yaw-bin).
    #: 0 disables; default 1 matches phase-2 so training inits match deployment.
    #: Batched like apex pass 2.  Apex teacher (if used) keeps its own 3-step.
    correct_terminal_steps: int = 1
    correct_terminal_damping: float = 0.05

    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)
    waist_cfg: WaistAimConfig | None = None
    #: Pitch/roll ease-to-upright window for the seed AND the imitation target
    #: (``warmstart_trajectory``'s ``waist_upright_ease_wp``).  None keeps the
    #: hold-the-measured-lean behaviour bit for bit; ~5 teaches phase 1 to
    #: RECOVER upright instead of blessing the sampled lean, which is the
    #: prior the phase-2 upright well otherwise has to fight.
    waist_upright_ease_wp: int | None = None

    def __post_init__(self) -> None:
        self.buckets = tuple(sorted(int(b) for b in self.buckets))
        if self.buckets[-1] != int(self.horizon):
            raise ValueError(
                f"the largest bucket {self.buckets[-1]} must equal horizon {self.horizon}"
            )


@dataclass
class NoiseConfig:
    """OU + Gaussian perturbation of the INPUT initialization (targets stay clean).

    ``sigma_corr`` is drawn log-uniformly in ``[sigma_corr_min, sigma_corr_max]``
    per sample, so the model sees inits from nearly-clean to badly wrong instead
    of one fixed displacement -- an MPC warm start is sometimes a good guess and
    sometimes not.  ``tau_steps`` is the OU correlation length in waypoints.
    """

    sigma_corr_min: float = 0.02
    sigma_corr_max: float = 0.30
    sigma_white: float = 0.01
    tau_steps: float = 5.0
    clip_to_box: bool = True


def bucket_for(k: int, buckets: tuple[int, ...]) -> int:
    """Smallest bucket that can hold a ``k``-waypoint horizon."""
    for b in buckets:
        if k <= b:
            return int(b)
    return int(buckets[-1])


# ============================================================ scenarios ======
def _peak_rate(q0: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    """Per-joint peak |qd| of the commanded trajectory (rad/s)."""
    X = np.concatenate([q0[None, :], target], axis=0)
    return np.abs(np.diff(X, axis=0) / float(dt)).max(axis=0)


def _draw_time(clip: RhythmClip, rng: np.random.Generator, cfg: ScenarioConfig) -> float:
    """A "now" in the clip, biased toward the run-up to a hit."""
    if clip.hits and rng.uniform() < cfg.short_horizon_frac:
        hit = clip.hits[int(rng.integers(len(clip.hits)))]
        k = int(rng.integers(1, max(int(cfg.short_horizon_wp), 1) + 1))
        return max(float(hit.time) - k * cfg.dt, 0.0)
    return float(rng.uniform(0.0, max(clip.duration, cfg.dt)))


def phase1_synthetic_goal_clips(kit, library, cfg):
    """Supply target supervision absent from the historical RCC corpus."""
    if not getattr(cfg, "all_hand_drum_goals", True):
        return []
    from .rhythm_chain_data import build_focused_single_transition_clips
    return build_focused_single_transition_clips(kit, library, cfg)


def draw_phase1_clip(clips, synthetic, rng, cfg):
    fraction = float(getattr(cfg, "synthetic_goal_frac", 0.5))
    if not 0 <= fraction <= 1:
        raise ValueError("synthetic_goal_frac must lie in [0,1]")
    use_synthetic = bool(synthetic) and (not clips or rng.uniform() < fraction)
    choices = synthetic if use_synthetic else clips
    if not choices:
        raise ValueError("Phase-1 generation has no goal clips")
    return choices[int(rng.integers(len(choices)))]


def record_phase1_goal_counts(stats, kit, snap, active):
    for hand, live in zip(HANDS, active):
        if live:
            key = f"active_goal/{hand}/{kit.names[int(snap.goal[hand].drum)]}"
            stats[key] = stats.get(key, 0) + 1


def phase1_goal_coverage(pool, cfg):
    """Count accepted active targets and fail closed on missing supervision."""
    kit = DrumKit.from_layout()
    counts = np.zeros((2, kit.num_drums), dtype=int)
    for data in pool.data.values():
        goal = data["drum"].detach().cpu().numpy()
        active = data["active"].detach().cpu().numpy().astype(bool)
        for h in (0, 1):
            np.add.at(counts[h], goal[:, h][active[:, h]], 1)
    expected = [(h, d) for h, hand in enumerate(HANDS) for d in kit.eligible(hand)]
    missing = [f"{HANDS[h]}:{kit.names[d]}" for h, d in expected if counts[h, d] == 0]
    if cfg.all_hand_drum_goals and missing:
        raise RuntimeError(f"Accepted phase-1 pool is missing targets {missing}; "
                           "increase proposals or fix rejection sampling")
    return dict(rows=len(pool), expected_targets=len(expected), missing_targets=missing,
                active_goal_counts={hand: dict(zip(kit.names, counts[h].tolist()))
                                    for h, hand in enumerate(HANDS)})


def generate_scenarios(
    n: int,
    clips: list[RhythmClip],
    *,
    seed: int,
    cfg: ScenarioConfig | None = None,
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    sampler=None,
    start_chunk: int = 256,
    progress: bool = False,
) -> dict:
    """``n`` accepted scenarios as flat numpy arrays, keyed by bucket.

    Returns ``{"buckets": {H: {...arrays...}}, "stats": {...}}``.  Rejected
    scenarios (idle hands, impossible velocity demand) are counted in ``stats``
    and *not* replaced, so the caller can see the true drop rate; ask for more
    than you need if you want an exact count.
    """
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(seed)
    names = tuple(cfg.joint_names)
    kit = kit or DrumKit.from_layout()
    library = library or GoalPoseLibrary(joint_names=names)
    synthetic_clips = phase1_synthetic_goal_clips(kit, library, cfg)
    if sampler is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=library)
    q_bounds = urdf_position_limits(names)
    v_limit = isaac_velocity_limits(names)
    H = int(cfg.horizon)
    # Resolved per call, which under ``generate_pool_parallel`` means once per
    # worker process -- the config travels as a path string, so the table never
    # has to be pickled across the fork.
    idle_home = resolve_idle_home(cfg.idle_home_path, joint_names=names)
    fk64 = ChainFK(device="cpu", dtype=torch.float64)

    out: dict[int, dict[str, list]] = {
        b: {k: [] for k in ("q0", "qd0", "target", "hit", "active", "drum", *CONTEXT_FIELDS)}
        for b in cfg.buckets
    }
    stats = {
        "requested": int(n),
        "accepted": 0,
        "reject_idle": 0,
        "reject_velocity": 0,
        "reject_library": 0,
        "dual_active": 0,
        "asymmetric": 0,
        "short_horizon": 0,
        "single_idle": 0,
        "rest": 0,
        "terminal_corrected": 0,
        "peak_rate_ratio": [],
    }

    starts = None
    si = 0
    pending: list[dict] = []
    _CHUNK = 512

    def _emit_ws(*, q0, qd0, target, hit, active, kit_s, snap, sticks, Hb, k_max):
        ctx = build_context_inputs(
            kit_s, snap, num_drums=cfg.num_drums, num_future=cfg.num_future,
            frame="env", device="cpu", dtype=torch.float64,
            idle_home=idle_home, sticks=sticks,
            yaw_now=q0[names.index("waist_yaw_joint")], waist_cfg=cfg.waist_cfg,
        )
        rec = out[Hb]
        rec["q0"].append(q0.astype(np.float32))
        rec["qd0"].append(qd0.astype(np.float32))
        rec["target"].append(np.asarray(target, float).astype(np.float32))
        rec["hit"].append(np.array([hit["left"], hit["right"]], dtype=np.int64))
        rec["active"].append(np.array(active, dtype=bool))
        rec["drum"].append(
            np.array(
                [snap.goal[h].drum if hit[h] >= 0 else -1 for h in HANDS], dtype=np.int64
            )
        )
        for f in CONTEXT_FIELDS:
            v = getattr(ctx, f)[0].numpy()
            rec[f].append(v.astype(np.int64) if f in _LONG_FIELDS
                          else (v.astype(bool) if f in _BOOL_FIELDS else v.astype(np.float32)))
        stats["accepted"] += 1
        record_phase1_goal_counts(stats, kit_s, snap, active)
        if all(active):
            stats["dual_active"] += 1
            if hit["left"] != hit["right"]:
                stats["asymmetric"] += 1
        elif any(active):
            stats["single_idle"] += 1
        if any(active) and k_max <= cfg.short_horizon_wp:
            stats["short_horizon"] += 1

    def _flush_pending():
        if not pending:
            return
        if cfg.correct_terminal_steps > 0:
            terms = corrected_arm_terminals(
                fk64, pending,
                steps=int(cfg.correct_terminal_steps),
                damping=float(cfg.correct_terminal_damping),
                q_bounds=q_bounds, joint_names=names, library=library,
            )
            for p, term in zip(pending, terms):
                if term:
                    p["ws"] = warmstart_trajectory(
                        p["q0"], p["qd0"], terminal_arm=term, **p["ws_kw"]
                    )
                    stats["terminal_corrected"] += len(term)
        for p in pending:
            target = np.asarray(p["ws"].init_traj, float)
            ratio = float((_peak_rate(p["q0"], target, cfg.dt) / v_limit).max())
            stats["peak_rate_ratio"].append(ratio)
            if ratio > cfg.vel_reject_scale:
                stats["reject_velocity"] += 1
                continue
            _emit_ws(
                q0=p["q0"], qd0=p["qd0"], target=target, hit=p["hit"],
                active=p["active"], kit_s=p["kit_s"], snap=p["snap"],
                sticks=p["sticks"], Hb=p["Hb"], k_max=p["k_max"],
            )
        pending.clear()
    for i in range(int(n)):
        if starts is None or si >= len(starts):
            starts = sampler.sample(int(start_chunk), rng=rng)
            si = 0
        q0 = starts.q[si].detach().cpu().numpy().astype(float)
        qd0 = starts.qd[si].detach().cpu().numpy().astype(float)
        si += 1
        if progress and (i % 500 == 0):
            print(f"  ... {i}/{n}", flush=True)

        kit_s = kit.randomized(rng, cfg.kit_randomization) if cfg.randomize_kit else kit
        clip = draw_phase1_clip(clips, synthetic_clips, rng, cfg)
        t_now = _draw_time(clip, rng, cfg)
        snap = goal_snapshot(
            clip, kit_s, t_now,
            num_future=cfg.num_future, rng=rng,
            hit_radius_frac=cfg.hit_radius_frac,
            speed_min=cfg.speed_min, speed_max=cfg.speed_max,
            horizon_cap=H * cfg.dt,
            time_jitter_s=cfg.hit_time_jitter_s,
            loudness_jitter=cfg.hit_loudness_jitter,
        )
        hit = {}
        goals: dict[str, tuple[str, float] | None] = {}
        aim: dict[str, np.ndarray] = {}
        # Future-aware aim inputs, the same ones ``build_context_inputs`` below
        # tokenizes: the imitation TARGET's yaw ramp and the ``aim_yaw`` token
        # the model conditions on have to name one yaw.
        _t, aim_tth, aim_futures = aim_inputs_from_snapshot(snap, frame="env")
        for hand in HANDS:
            g = snap.goal[hand]
            if not g.active:
                hit[hand] = -1
                goals[hand] = None
                aim_tth.pop(hand, None)
                continue
            k = int(np.clip(round(g.time_to_hit / cfg.dt), 1, H))
            hit[hand] = k
            goals[hand] = (g.drum_name, k * cfg.dt)
            aim[hand] = np.asarray(g.position, float)
        active = [hit[h] >= 0 for h in HANDS]
        if not any(active):
            # A both-hands-idle draw is a REST.  Keep it at ``rest_frac`` when a
            # home table is available to park the arms at; without one the
            # target is the start pose held for the whole horizon, which teaches
            # only that nothing ever moves.
            if cfg.require_active_hand and (
                idle_home is None or rng.uniform() >= cfg.rest_frac
            ):
                stats["reject_idle"] += 1
                continue
            stats["rest"] += 1
        if not all(
            library.has(h, goals[h][0]) for h in HANDS if goals[h] is not None
        ):
            stats["reject_library"] += 1
            continue

        if any(active):
            k_max = max(hit[h] for h in HANDS if hit[h] >= 0)
            Hb = bucket_for(k_max, cfg.buckets)
        else:
            # A rest has no hit to size the horizon from, and ``bucket_for(1)``
            # would file every one of them in the shortest bucket.  Both arms are
            # masked over the WHOLE grid when neither is active (see
            # ``build_waypoint_masks``), so spread rests across the buckets
            # instead -- otherwise the retraction is only ever imitated at the
            # one horizon too short to show it settle.
            k_max = 1
            Hb = int(cfg.buckets[int(rng.integers(len(cfg.buckets)))])
        sticks = None
        if cfg.stick_rand is not None:
            sticks = (cfg.stick_rand.sample(rng, "left"),
                      cfg.stick_rand.sample(rng, "right"))
        ws_kw = dict(
            horizon=Hb, goals=goals, library=library,
            aim_targets=aim or None, aim_tth=aim_tth, aim_futures=aim_futures,
            dt=cfg.dt, joint_names=names,
            waist_cfg=cfg.waist_cfg, q_bounds=q_bounds,
            waist_upright_ease_wp=cfg.waist_upright_ease_wp,
            idle_home=idle_home, idle_settle_wp=cfg.idle_settle_wp,
        )
        pending.append({
            "ws": warmstart_trajectory(q0, qd0, **ws_kw),
            "ws_kw": ws_kw, "q0": q0, "qd0": qd0, "goals": goals,
            "aim": aim, "hit": hit, "active": active, "sticks": sticks,
            "kit_s": kit_s, "snap": snap, "Hb": Hb, "k_max": k_max,
        })
        if len(pending) >= _CHUNK:
            _flush_pending()

    _flush_pending()

    packed: dict[int, dict[str, np.ndarray]] = {}
    for b, rec in out.items():
        if not rec["q0"]:
            continue
        packed[b] = {k: np.stack(v) for k, v in rec.items()}
    stats["peak_rate_ratio"] = np.asarray(stats["peak_rate_ratio"], dtype=np.float32)
    return {"buckets": packed, "stats": stats}


def _worker(args) -> dict:
    n, seed, cfg, split, clip_dir = args
    torch.set_num_threads(1)
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(clip_dir, kit)
    return generate_scenarios(n, clips, seed=seed, cfg=cfg, kit=kit)


def _merge(parts: list[dict]) -> dict:
    buckets: dict[int, dict[str, list[np.ndarray]]] = {}
    stats: dict = {}
    for p in parts:
        for b, arrays in p["buckets"].items():
            tgt = buckets.setdefault(b, {k: [] for k in arrays})
            for k, v in arrays.items():
                tgt[k].append(v)
        for k, v in p["stats"].items():
            if isinstance(v, np.ndarray):
                stats.setdefault(k, []).append(v)
            else:
                stats[k] = stats.get(k, 0) + v
    out_b = {b: {k: np.concatenate(v) for k, v in arrays.items()}
             for b, arrays in buckets.items()}
    if "peak_rate_ratio" in stats:
        stats["peak_rate_ratio"] = np.concatenate(stats["peak_rate_ratio"])
    return {"buckets": out_b, "stats": stats}


def generate_pool_parallel(
    n: int,
    *,
    seed: int,
    cfg: ScenarioConfig,
    split: str = "train",
    clip_dir: str | Path | None = None,
    workers: int | None = None,
) -> dict:
    """:func:`generate_scenarios` fanned out over processes (CPU only)."""
    import multiprocessing as mp

    clip_dir = str(clip_dir if clip_dir is not None else default_rcc_dir(split))
    workers = int(workers or min(os.cpu_count() or 8, 64))
    workers = max(1, min(workers, int(n)))
    per = [n // workers] * workers
    for i in range(n - sum(per)):
        per[i] += 1
    args = [(per[i], seed * 100003 + i, cfg, split, clip_dir) for i in range(workers)]
    if workers == 1:
        return _merge([_worker(args[0])])
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_worker, args)
    return _merge(parts)


# ======================================================= auxiliary labels ====
def _fk_points(fk: ChainFK, q: torch.Tensor) -> torch.Tensor:
    """(B, 6, 3) tip/wrist/elbow of both arms, in :data:`FK_POINT_NAMES` order."""
    frames = fk.forward(q)
    cols: dict[str, torch.Tensor] = {}
    for hand in ("left", "right"):
        R, p = frames[f"{hand}_wrist"]
        tip_local = torch.as_tensor(
            stick_tip_offset(hand)[None, :], device=q.device, dtype=q.dtype
        )
        cols[f"{hand}_tip"] = ChainFK.points_world(R, p, tip_local)[:, 0]
        cols[f"{hand}_wrist"] = p
        cols[f"{hand}_elbow"] = frames[f"{hand}_elbow"][1]
    return torch.stack([cols[n] for n in FK_POINT_NAMES], dim=1)


@torch.no_grad()
def aux_targets(
    q0: torch.Tensor,        # (B, 17)
    target: torch.Tensor,    # (B, H, 17)
    v0: torch.Tensor,        # (B, 17)
    elem_mask: torch.Tensor,  # (B, H, 17)
    *,
    builder: TrajectorySplineBuilder,
    fk: ChainFK,
    rnea: BatchedRNEA,
    dt: float = CONTROL_DT,
    torque_mode: str = "rnea_segmax",
    tau_kernel: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Spline / torque / FK labels for one bucket-homogeneous block.

    All five heads' targets, per waypoint:
    ``vel``/``acc``/``torque`` ``(B, H, 17)``, ``fk_pos``/``fk_vel``
    ``(B, H, 6, 3)``.

    ``torque_mode`` picks WHICH physical torque the ``torque`` label -- and so
    the model's ``head_torque`` -- is asked to predict:

    * ``"rnea_segmax"`` (**the default, and what every checkpoint before
      2026-08-16 was trained on**): ``max(|tau_rnea[i-1]|, |tau_rnea[i]|)``,
      the peak inverse-dynamics torque over the segment ENDING at waypoint
      ``i``, from the knot finite differences.  It is the plan's own OPEN-LOOP
      dynamics: it says nothing about what the servo has to spend to make the
      plant follow the plan.
    * ``"tracking"``: ``|tau_rnea[i] + tau_fb[i]|`` at waypoint ``i``, the
      torque the PD servo actually DEMANDS while tracking this plan, from
      :func:`~amortized_opt.pd_sim.tracking_torque_knots` -- literally the same
      function, kernel and DC removal that
      :meth:`~amortized_opt.cost.FusedCost._track_torque` prices.  This is the
      mode consistent with ``w_track_torque`` / ``w_track_torque_peak``: with
      it on, the head predicts the quantity the cost charges for, so the
      planner's own torque estimate and the objective agree.  It is a
      per-waypoint value, not a segment max, because the FIR bank lives on the
      control-tick grid.

    ``tau_kernel`` is the ``(17, L)`` FIR bank; ``"tracking"`` identifies one
    with :func:`~amortized_opt.pd_sim.nominal_torque_kernels` when it is not
    supplied.  Pass it when calling in a loop -- identification runs a small PD
    rollout.  Ignored under ``"rnea_segmax"``.
    """
    if torque_mode not in ("rnea_segmax", "tracking"):
        raise ValueError(
            f"torque_mode must be 'rnea_segmax' or 'tracking', got {torque_mode!r}"
        )
    B, H, D = target.shape
    sp = builder.from_masks(q0, target, v0, elem_mask)

    end = torch.full((1,), float(dt), device=target.device, dtype=target.dtype)
    _q, qd, qdd, _j = sp.eval_offsets(end)  # (B, H, 1, D) at each segment's right end
    vel = qd[:, :, 0, :]
    acc = qdd[:, :, 0, :]

    if torque_mode == "tracking":
        from .pd_sim import nominal_torque_kernels, tracking_torque_knots

        if tau_kernel is None:
            tau_kernel = nominal_torque_kernels(
                rnea, joint_names=tuple(builder.joint_names), dt=float(dt),
                device=target.device, dtype=target.dtype,
            )
        tau_hat, xk, _qd_fd = tracking_torque_knots(sp, v0, rnea, tau_kernel)
        torque = tau_hat[:, 1:].abs()          # (B, H, D), waypoints 1..H
        n = xk.shape[1]
    else:
        xk, qd_fd, qdd_fd = sp.knot_finite_differences(v0)  # (B, H+1, D)
        n = xk.shape[1]
        tau = rnea.rnea(
            xk.reshape(B * n, D), qd_fd.reshape(B * n, D), qdd_fd.reshape(B * n, D)
        ).reshape(B, n, D).abs()
        torque = torch.maximum(tau[:, :-1], tau[:, 1:])  # segment ending at waypoint i

    pts = _fk_points(fk, xk.reshape(B * n, D)).reshape(B, n, len(FK_POINT_NAMES), 3)
    fk_pos = pts[:, 1:]
    fk_vel = (pts[:, 1:] - pts[:, :-1]) / float(dt)
    return {"vel": vel, "acc": acc, "torque": torque, "fk_pos": fk_pos, "fk_vel": fk_vel}


# ================================================================= pool ======
def ou_kernel(H: int, tau_steps: float, device, dtype) -> torch.Tensor:
    """Lower-triangular ``L`` with ``L @ eps`` an OU path of unit marginal sd.

    Reproduces :func:`amortized_opt.warmstart.correlated_noise` exactly:
    ``x_0 = eps_0`` and ``x_k = a x_{k-1} + sqrt(1 - a^2) eps_k``, so
    ``L[i, 0] = a^i`` and ``L[i, j] = sqrt(1 - a^2) a^{i-j}`` for ``j >= 1``.
    """
    a = float(np.exp(-1.0 / max(float(tau_steps), 1e-6)))
    i = torch.arange(H, device=device, dtype=dtype).view(-1, 1)
    j = torch.arange(H, device=device, dtype=dtype).view(1, -1)
    p = (i - j).clamp_min(0.0)
    L = torch.where(i >= j, a ** p, torch.zeros((), device=device, dtype=dtype))
    L[:, 1:] *= float(np.sqrt(max(1.0 - a * a, 0.0)))
    return L


class WarmstartPool:
    """Bucketed phase-1 dataset resident on one device.

    ``pool.batch(bucket, idx, generator=...)`` returns everything a training
    step needs, with the input initialization noised freshly on the device.
    """

    def __init__(
        self,
        buckets: dict[int, dict[str, np.ndarray]],
        *,
        cfg: ScenarioConfig,
        noise: NoiseConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        fk: ChainFK | None = None,
        rnea: BatchedRNEA | None = None,
        aux_chunk: int = 131_072,
        stats: dict | None = None,
        aux_torque_tracking: bool = False,
        trajectory_transform=None,
    ):
        #: ``True`` -> the stored ``aux_torque`` label is the DEMANDED PD
        #: tracking torque instead of the RNEA segment max.  Opt-in: ``False``
        #: reproduces every pool built before 2026-08-16 bit-for-bit.  See
        #: :func:`aux_targets`.
        self.aux_torque_tracking = bool(aux_torque_tracking)
        self.cfg = cfg
        self.noise = noise or NoiseConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.stats = dict(stats or {})
        names = tuple(cfg.joint_names)
        lo, hi = urdf_position_limits(names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)

        fk = fk or ChainFK(device=self.device, dtype=dtype)
        rnea = rnea or BatchedRNEA.from_urdf(
            joint_names=names, with_armature=True, device=self.device, dtype=dtype
        )
        # Identify the command -> demand FIR bank ONCE, not per bucket-chunk.
        tau_kernel = None
        if self.aux_torque_tracking:
            from .pd_sim import nominal_torque_kernels

            tau_kernel = nominal_torque_kernels(
                rnea, joint_names=names, dt=float(cfg.dt),
                device=self.device, dtype=dtype,
            )
        self.aux_tau_kernel = tau_kernel
        self.data: dict[int, dict[str, torch.Tensor]] = {}
        self._ou: dict[int, torch.Tensor] = {}
        for H, arrays in sorted(buckets.items()):
            if not len(arrays["q0"]):
                continue
            d: dict[str, torch.Tensor] = {}
            for k, v in arrays.items():
                if v.dtype == np.bool_:
                    d[k] = torch.as_tensor(v, device=self.device)
                elif np.issubdtype(v.dtype, np.integer):
                    d[k] = torch.as_tensor(v, device=self.device, dtype=torch.long)
                else:
                    d[k] = torch.as_tensor(v, device=self.device, dtype=dtype)
            masks = self.masks_for(d["hit"], d["active"], H)
            d["elem_mask"] = masks.elem_mask
            d["point_mask"] = masks.point_mask
            if trajectory_transform is not None:
                ctx = ContextInputs(
                    **{f: d[f] for f in CONTEXT_FIELDS if f in d}
                )
                replacements = trajectory_transform(d, ctx, masks, int(H))
                for key, value in replacements.items():
                    if key not in ("target", "init_base"):
                        raise ValueError(f"trajectory transform cannot replace {key!r}")
                    if value.shape != d[key].shape:
                        raise ValueError(
                            f"trajectory transform {key} shape {tuple(value.shape)} "
                            f"!= {tuple(d[key].shape)}"
                        )
                    d[key] = value.to(device=self.device, dtype=dtype)
            builder = TrajectorySplineBuilder(
                max_segments=H, dt=cfg.dt, joint_names=names,
                device=self.device, dtype=dtype,
            )
            aux: dict[str, list[torch.Tensor]] = {}
            B = int(d["q0"].shape[0])
            # ``aux_chunk`` is a budget in FK/RNEA *states*, not samples, so a
            # 100-waypoint bucket does not allocate 4x what a 25 one does.
            step = max(int(aux_chunk) // (H + 1), 1)
            for b0 in range(0, B, step):
                sl = slice(b0, min(b0 + step, B))
                a = aux_targets(
                    d["q0"][sl], d["target"][sl], d["qd0"][sl], d["elem_mask"][sl],
                    builder=builder, fk=fk, rnea=rnea, dt=cfg.dt,
                    torque_mode=(
                        "tracking" if self.aux_torque_tracking else "rnea_segmax"
                    ),
                    tau_kernel=tau_kernel,
                )
                for k, v in a.items():
                    aux.setdefault(k, []).append(v)
            for k, chunks in aux.items():
                d[f"aux_{k}"] = torch.cat(chunks, dim=0)
            self.data[H] = d
            self._ou[H] = ou_kernel(H, self.noise.tau_steps, self.device, dtype)

    # ---------------------------------------------------------- accessors --
    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.data))

    def __len__(self) -> int:
        return sum(int(d["q0"].shape[0]) for d in self.data.values())

    def size(self, bucket: int) -> int:
        return int(self.data[int(bucket)]["q0"].shape[0])

    def distribution(self) -> dict[int, int]:
        return {b: self.size(b) for b in self.buckets}

    def masks_for(self, hit: torch.Tensor, active: torch.Tensor, H: int):
        """:func:`~amortized_opt.tokenizer.build_waypoint_masks` for stored hits."""
        t = hit.clamp_min(0).to(self.dtype) * float(self.cfg.dt)
        return build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=int(H), dt=self.cfg.dt,
            active_left=active[:, 0], active_right=active[:, 1],
            joint_names=self.cfg.joint_names,
        )

    def context(self, bucket: int, idx: torch.Tensor) -> ContextInputs:
        d = self.data[int(bucket)]
        # Missing fields fall back to ``ContextInputs``'s zero + invalid
        # defaults, so a phase-1 pool built before the idle block existed loads.
        return ContextInputs(
            **{f: d[f].index_select(0, idx) for f in CONTEXT_FIELDS if f in d}
        )

    # -------------------------------------------------------------- noise --
    def noised_init(
        self,
        target: torch.Tensor,
        bucket: int,
        *,
        generator: torch.Generator | None = None,
        sigma_corr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OU + white perturbation of ``target`` (B, H, 17), clipped to the box."""
        nc = self.noise
        B, H, D = target.shape
        L = self._ou[int(bucket)]
        eps = torch.randn(B, H, D, device=target.device, dtype=target.dtype,
                          generator=generator)
        if sigma_corr is None:
            u = torch.rand(B, 1, 1, device=target.device, dtype=target.dtype,
                           generator=generator)
            lo, hi = math.log(nc.sigma_corr_min), math.log(nc.sigma_corr_max)
            sigma_corr = torch.exp(lo + (hi - lo) * u)
        out = target + sigma_corr * torch.einsum("ij,bjd->bid", L, eps)
        if nc.sigma_white > 0.0:
            out = out + nc.sigma_white * torch.randn(
                B, H, D, device=target.device, dtype=target.dtype, generator=generator
            )
        if nc.clip_to_box:
            out = torch.clamp(out, self.q_lo, self.q_hi)
        return out

    # -------------------------------------------------------------- batch --
    def batch(
        self,
        bucket: int,
        idx: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        hold_init: bool = False,
    ) -> dict:
        """One bucket-homogeneous minibatch (dict of tensors + a ContextInputs)."""
        H = int(bucket)
        d = self.data[H]
        pick = lambda k: d[k].index_select(0, idx)  # noqa: E731
        q0, qd0, target = pick("q0"), pick("qd0"), pick("target")
        elem = pick("elem_mask")
        if hold_init:
            init = q0.unsqueeze(1).expand_as(target).contiguous()
        else:
            init = self.noised_init(target, H, generator=generator)
        masks = self.masks_for(pick("hit"), pick("active"), H)
        return {
            "bucket": H,
            "state": torch.cat([q0, qd0], dim=-1),
            "q0": q0,
            "v0": qd0,
            "init": init * elem,
            "target": target,
            "masks": masks,
            "context": self.context(H, idx),
            "hit": pick("hit"),
            "active": pick("active"),
            "drum": pick("drum"),
            "aux": {k: pick(f"aux_{k}") for k in ("vel", "acc", "torque", "fk_pos", "fk_vel")},
        }


def generate_pool(
    n: int,
    *,
    seed: int,
    cfg: ScenarioConfig | None = None,
    noise: NoiseConfig | None = None,
    split: str = "train",
    clip_dir: str | Path | None = None,
    workers: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    fk: ChainFK | None = None,
    rnea: BatchedRNEA | None = None,
    aux_torque_tracking: bool = False,
) -> WarmstartPool:
    """Generate ``n`` scenarios and wrap them in a :class:`WarmstartPool`."""
    cfg = cfg or ScenarioConfig()
    raw = generate_pool_parallel(
        n, seed=seed, cfg=cfg, split=split, clip_dir=clip_dir, workers=workers
    )
    return WarmstartPool(
        raw["buckets"], cfg=cfg, noise=noise, device=device, dtype=dtype,
        fk=fk, rnea=rnea, stats=raw["stats"],
        aux_torque_tracking=aux_torque_tracking,
    )


def group_indices(joint_names=G1_UPPER_BODY_JOINTS) -> dict[str, np.ndarray]:
    """Column indices of the waist / left / right joint groups."""
    g = joint_group_masks(joint_names)
    return {k: np.nonzero(v > 0.5)[0] for k, v in g.items()}
