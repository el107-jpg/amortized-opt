"""Phase-2 (direct cost descent) scenario generation and on-device pool.

Section 7 of ``docs/amortized_optimizer_plan.md``.  Phase 2 has **no labels** --
the gradient comes from :class:`~amortized_opt.cost.FusedCost` -- so this module
only has to put the model in front of the right distribution of *situations*,
and to hand the cost everything it needs to price the answer.

Two scenario families (prerequisite fix C)
------------------------------------------

``plan-native`` (70 % by default)
    What :mod:`amortized_opt.warmstart_data` generates: a
    :class:`~amortized_opt.starts.StartSampler` state (90 % drum-departure,
    *both* hands placed on a drum with an outward tip velocity) plus a goal
    snapshot taken at a random instant of a random RCC clip.  Mostly dual-arm,
    mostly short horizons, always moving.

``eval-like`` (30 %)
    The deployment-relevant regime the phase-1 report found the model had never
    seen: a **rest** start (zero velocity, waist + one arm sampled in a scaled
    joint box, the idle arm at URDF neutral), a **single** active hand, a
    horizon drawn from 0.44-0.80 s, and no future-hit tokens.  This is the
    *shape* of the acceptance grid, drawn from a different random stream --
    post-rest first hits and single-hand passages, not the graded cases.

Both families are built against a **domain-randomized kit** (fix B) and both
carry the strike geometry the terminal cost prices, derived from that same
randomized kit, so the context tokens, the collision scene and the terminal
target can never disagree.

What a scenario carries
-----------------------

============ ==============================================================
``q0/qd0``   start state
``init``     the quintic + waist-aim seed (slew-limited), the tick-0
             initialization; phase 2 noises it at batch time and never
             regresses against it
``hit``      per-hand strike waypoint index, -1 for an idle hand
strike block ``p_star``/``v_star``/``n_out``/``hit_radius``/``disk_off``/
             ``hp_dir``/``has_hp``/``sigma``/``shaft``/``start_shaft``/
             ``elev`` -- the :class:`~amortized_opt.cost.StrikeContext`
             columns, in the cost's PELVIS frame
context      the :class:`~amortized_opt.tokenizer.ContextInputs` fields, in
             the ENV frame (the tokenizer convention)
============ ==============================================================

The impossible-transition filter from phase 1 is kept: a scenario whose quintic
seed demands more than ``vel_reject_scale`` x a joint's speed rating is dropped
rather than trained on.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV, ready_joint_pos
from .cost import StrikeContext
from .limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from .rhythm import (
    HAND_INDEX,
    HANDS,
    DrumKit,
    GoalSnapshot,
    HandGoal,
    KitRandomization,
    RhythmClip,
    build_context_inputs,
    default_rcc_dir,
    goal_snapshot,
    load_rcc_pool,
    sample_hit_point,
    strike_region,
    strike_velocity,
)
from .strike import (
    CYMBAL_EDGE_FRACTION,
    HIHAT_RIM_FRACTION,
    HIHAT_SHOULDER_SIGMA,
    STICK_TIP_SIGMA,
    TOM_NEAR_FRACTION,
    default_contact_sigma,
    is_cymbal_name,
    is_hihat_name,
    is_tom_name,
    shaft_hit_allowed,
    strike_elevation_required,
)
from .tokenizer import ContextInputs, build_waypoint_masks
from .idle_home import resolve_idle_home
from .warmstart import GoalPoseLibrary, _slew_limited, warmstart_trajectory
from .warmstart_apex import ApexConfig, ApexSolver, apex_warmstart_trajectory
from .warmstart_data import CONTEXT_FIELDS, bucket_for, ou_kernel
from .waist_aim import WaistAimConfig

__all__ = [
    "KIND_PLAN",
    "KIND_EVAL_LIKE",
    "Phase2Noise",
    "Phase2Pool",
    "Phase2ScenarioConfig",
    "STRIKE_FIELDS",
    "drum_name_predicates",
    "generate_cost_pool",
    "generate_cost_scenarios",
    "strike_context_from_arrays",
    "strike_geometry_table",
]

KIND_PLAN = 0
KIND_EVAL_LIKE = 1

_LONG_FIELDS = frozenset({"goal_drum", "future_drum"})
_BOOL_FIELDS = frozenset({"drum_valid", "goal_valid", "future_valid", "idle_valid"})

#: The per-scenario strike-geometry arrays (shapes are per sample).
STRIKE_FIELDS: tuple[str, ...] = (
    "p_star", "v_star", "n_out", "hit_radius", "disk_off", "hp_dir",
    "has_hp", "sigma", "shaft", "start_shaft", "elev",
)

#: Drum -> hand for the eval-like family. Identical to the acceptance grid's
#: map (``eval_harness.DRUM_HAND``), which is itself the repo bench's, so the
#: hand a drum is played with is one convention across the whole package.
EVAL_DRUM_HAND: dict[str, str] = {
    "snare": "right", "hihat": "left", "tom1": "right",
    "tom2": "left", "cymbal_left": "left", "cymbal_right": "right",
}


# ================================================================ config =====
@dataclass
class Phase2ScenarioConfig:
    """The phase-2 input distribution."""

    horizon: int = 100
    buckets: tuple[int, ...] = (25, 50, 100)
    dt: float = CONTROL_DT
    num_drums: int = 8
    num_future: int = 3

    #: Domain randomization of the kit -- ON, and now consistent end to end.
    randomize_kit: bool = True
    kit_randomization: KitRandomization = field(default_factory=KitRandomization)

    hit_radius_frac: float = 0.6
    speed_min: float = 1.0
    speed_max: float = 3.0

    #: Fraction of scenarios drawn from the eval-like family (fix C).
    eval_like_frac: float = 0.30
    #: Horizons the eval-like family draws from (seconds). The acceptance grid
    #: uses 0.45 / 0.60 / 0.80; this is the same band, continuous.
    eval_horizon_s: tuple[float, float] = (0.44, 0.80)
    eval_speeds: tuple[float, ...] = (1.0, 2.0, 3.0)
    eval_start_q_scale: float = 0.35
    eval_start_jitter: float = 0.05
    #: Fraction of eval-like starts given a small non-zero velocity anyway, so
    #: the family is not degenerate in ``v_start``.
    eval_moving_frac: float = 0.25
    eval_v_scale: float = 0.10

    #: Plan-native "now" biased toward the run-up to a hit.
    short_horizon_frac: float = 0.25
    short_horizon_wp: int = 5

    #: Impossible-transition filter, inherited from phase 1.
    vel_reject_scale: float = 1.5
    require_active_hand: bool = True
    #: REST SCENARIOS.  Fraction of plan-native draws kept even when NEITHER hand
    #: has a goal -- i.e. the bar rest that ``require_active_hand`` otherwise
    #: throws away.  The idle-arm terms (``CostConfig.w_idle_home`` /
    #: ``w_idle_vel``) are only meaningful on states where an arm is parked, so a
    #: pool with no rests trains them entirely off-distribution.  Single-hand-idle
    #: scenarios are NOT affected -- those were always kept, and are already ~45 %
    #: of the pool.  0.0 reproduces the pre-2026-08-12 distribution exactly.
    rest_frac: float = 0.0
    #: Park the idle arm at the home posture instead of holding it where
    #: it started.  ``"default"`` uses ``assets/idle_home_poses.json`` when it
    #: exists, a path names another table, ``None`` keeps the hold-in-place seed.
    #: This is also how the *model* is told where home is: the decoder reads the
    #: seed as its waypoint tokens.
    idle_home_path: str | None = None
    idle_settle_wp: int = 10
    #: The seed is EXECUTED inside the MPC roller, so bound it (fix A).
    seed_slew_eta: float | None = ETA_VELOCITY
    #: Fraction of fresh inits built as an APEX three-point spline instead of a
    #: straight-line quintic.  The apex A/B found it a better *geometry*
    #: (teacher collision-free 14 -> 24/162, impact direction cosine
    #: -0.77 -> +0.08) but a worse *imitation target* (2.7x the torque excess).
    #: Phase 2 does not imitate anything, so it can take the good half: the
    #: apex gives cost descent a collision-friendlier starting basin and the
    #: cost prices away the torque and jerk.  Apex constructions that fail the
    #: impossible-transition filter fall back to the straight line for that
    #: scenario rather than dropping it -- the apex builder's own drop rate is
    #: 45.6 % vs 30.7 %, and discarding would bias the scenario distribution.
    apex_init_frac: float = 0.5

    apex_cfg: ApexConfig = field(default_factory=ApexConfig)

    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)
    waist_cfg: WaistAimConfig | None = None

    def __post_init__(self) -> None:
        self.buckets = tuple(sorted(int(b) for b in self.buckets))
        if self.buckets[-1] != int(self.horizon):
            raise ValueError(
                f"the largest bucket {self.buckets[-1]} must equal horizon {self.horizon}"
            )
        if not 0.0 <= self.eval_like_frac <= 1.0:
            raise ValueError("eval_like_frac must be in [0, 1]")


@dataclass
class Phase2Noise:
    """OU + white perturbation of the initialization (same law as phase 1)."""

    sigma_corr_min: float = 0.02
    sigma_corr_max: float = 0.30
    sigma_white: float = 0.01
    tau_steps: float = 5.0
    clip_to_box: bool = True


# ====================================================== strike geometry ======
def _strike_row(kit: DrumKit, drum: int, position_env: np.ndarray, velocity: np.ndarray):
    """Per-hand strike-cost columns for one goal, in the PELVIS frame."""
    name = kit.names[int(drum)]
    aim, center, disk_r, half = strike_region(kit, int(drum), frame="pelvis")
    axis = np.asarray(kit.axes[int(drum)], float)
    p = np.asarray(position_env, float) - PELVIS_IN_ENV
    # ``disk_center_offset`` is the IN-PLANE vector p_star -> disk centre, which
    # is what ``FusedCost._terminal`` subtracts before measuring the disk radius.
    off = center - p
    off = off - float(off @ axis) * axis
    v = np.asarray(velocity, float)
    nv = float(np.linalg.norm(v))
    n_out = -v / nv if nv > 1e-9 else axis
    return dict(
        p_star=p,
        v_star=v,
        n_out=n_out,
        hit_radius=float(disk_r),
        disk_off=off,
        hp_dir=np.zeros(3) if half is None else np.asarray(half, float),
        has_hp=0.0 if half is None else 1.0,
        sigma=float(default_contact_sigma(name)),
        shaft=float(shaft_hit_allowed(name)),
        start_shaft=0.0,     # filled from the DEPARTURE drum, not this one
        elev=float(strike_elevation_required(name)),
    )


def _empty_strike() -> dict:
    return dict(
        p_star=np.zeros(3), v_star=np.zeros(3), n_out=np.array([0.0, 0.0, 1.0]),
        hit_radius=0.08, disk_off=np.zeros(3), hp_dir=np.zeros(3), has_hp=0.0,
        sigma=float(STICK_TIP_SIGMA), shaft=0.0, start_shaft=0.0, elev=0.0,
    )


def drum_name_predicates(
    drum_names, *, device=None, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Per-drum-INDEX constant tables for the name predicates of :mod:`.strike`.

    The kit's drum ordering is fixed (row ``i`` is drum id ``i``) and kit
    randomization only moves positions, so every name-driven quantity --
    cymbal / hi-hat / tom class, the aim fraction, the contact sigma, shaft
    eligibility, the elevation requirement -- is a constant vector over the
    drum axis.  Names may be padded with ``""``/``None`` to the tokenizer's
    ``num_drums``; padded rows read as "plain pad", which is inert because
    nothing ever selects them.
    """
    names = list(drum_names)
    cym = torch.tensor([bool(is_cymbal_name(n)) for n in names], device=device)
    hh = torch.tensor([bool(is_hihat_name(n)) for n in names], device=device)
    tom = torch.tensor([bool(is_tom_name(n)) for n in names], device=device)
    frac = torch.where(
        hh,
        torch.full_like(cym, HIHAT_RIM_FRACTION, dtype=dtype),
        torch.where(
            cym,
            torch.full_like(cym, CYMBAL_EDGE_FRACTION, dtype=dtype),
            torch.where(
                tom,
                torch.full_like(cym, TOM_NEAR_FRACTION, dtype=dtype),
                torch.zeros(len(names), device=device, dtype=dtype),
            ),
        ),
    )
    return {
        "cymbal": cym,
        "hihat": hh,
        "tom": tom,
        #: an aim offset / half-plane exists exactly for cymbals and toms
        "has_hp": cym | tom,
        "aim_frac": frac,
        "sigma": torch.where(
            cym,
            torch.full_like(frac, HIHAT_SHOULDER_SIGMA),
            torch.full_like(frac, STICK_TIP_SIGMA),
        ),
        "shaft": cym.to(dtype),
        "elev": torch.tensor(
            [float(strike_elevation_required(n)) for n in names], device=device, dtype=dtype
        ),
    }


@torch.no_grad()
def strike_geometry_table(
    drum_pos_pelvis: torch.Tensor,     # (B, D, 3) head centres, PELVIS frame
    drum_axis: torch.Tensor,           # (B, D, 3) head normals (need not be unit)
    drum_radius: torch.Tensor,         # (B, D)
    drum_names,                        # length-D names, "" for padding
    *,
    surface_offset: float = 0.005,
    rim_buffer: float = 0.01,
    predicates: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Batched port of :func:`~amortized_opt.rhythm.strike_region` and friends.

    Returns ``(B, D, ...)`` tensors ``aim`` / ``disk_center`` / ``hit_radius`` /
    ``hp_dir`` / ``has_hp`` / ``sigma`` / ``shaft`` / ``elev``, all in the frame
    the centres were handed in (the caller subtracts ``PELVIS_IN_ENV`` first, as
    :func:`_strike_row` does -- the half-plane is frame-independent because the
    robot stands at ``x = y = 0`` in both).

    ``tests/test_roller_regoal.py`` asserts this agrees with the numpy path to
    1e-6 on randomized kits, which is the contract the MPC roller relies on when
    it re-prices a hand's strike row mid-episode against the slot's OWN kit.
    """
    dev, dt_ = drum_pos_pelvis.device, drum_pos_pelvis.dtype
    p = drum_name_predicates(drum_names, device=dev, dtype=dt_) if predicates is None \
        else predicates
    c = drum_pos_pelvis
    r = drum_radius

    # ---- _unit3(axis) ----
    a_norm = drum_axis.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(drum_axis)
    up[..., 2] = 1.0
    n = torch.where(a_norm > 1e-12, drum_axis / a_norm.clamp_min(1e-30), up)

    # ---- strike_half_plane_dir ----
    toward = torch.stack([-c[..., 0], -c[..., 1], torch.zeros_like(c[..., 0])], dim=-1)
    t1 = toward - (toward * n).sum(-1, keepdim=True) * n
    t1n = t1.norm(dim=-1, keepdim=True)
    ok1 = (toward.norm(dim=-1, keepdim=True) > 1e-6) & (t1n > 1e-9)

    down = torch.zeros_like(n)
    down[..., 2] = -1.0
    t2 = down - (down * n).sum(-1, keepdim=True) * n
    t2n = t2.norm(dim=-1, keepdim=True)
    pref = torch.zeros_like(n)
    pref[..., 0] = -1.0
    t3 = pref - (pref * n).sum(-1, keepdim=True) * n
    t3n = t3.norm(dim=-1, keepdim=True)
    fallback = torch.where(
        t2n < 1e-9, t3 / t3n.clamp_min(1e-12), t2 / t2n.clamp_min(1e-12)
    )
    hp = torch.where(ok1, t1 / t1n.clamp_min(1e-30), fallback)
    has_hp = p["has_hp"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(c.shape[:-1])
    hp_dir = torch.where(has_hp.unsqueeze(-1), hp, torch.zeros_like(hp))

    # ---- _strike_aim_offset / strike_aim_on_head ----
    cym = p["cymbal"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(c.shape[:-1])
    frac = p["aim_frac"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(c.shape[:-1])
    buf = max(float(rim_buffer), 0.0)
    r_use = torch.where(cym, r, (r - buf).clamp_min(1e-3))
    off = (frac * r_use).unsqueeze(-1) * hp_dir          # zero where has_hp is False
    aim = c - float(surface_offset) * n + off
    # ``strike_disk_center_offset`` is exactly ``-off``, so the disk centre is
    # the surface-inset head centre for EVERY drum class.
    disk_center = c - float(surface_offset) * n
    hit_radius = torch.where(cym, r.clamp_min(1e-3), (r - buf).clamp_min(1e-3))

    shape = c.shape[:-1]
    return {
        "aim": aim,
        "disk_center": disk_center,
        "hit_radius": hit_radius,
        "hp_dir": hp_dir,
        "has_hp": has_hp.to(dt_),
        "sigma": p["sigma"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(shape).to(dt_),
        "shaft": p["shaft"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(shape).to(dt_),
        "elev": p["elev"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(shape).to(dt_),
        "axis": n,
    }


def strike_context_from_arrays(
    d: dict[str, torch.Tensor], idx: torch.Tensor, *, drum_head_centers: torch.Tensor,
    waist_ref: torch.Tensor | None = None,
) -> StrikeContext:
    """Assemble a :class:`~amortized_opt.cost.StrikeContext` from pool columns.

    ``waist_ref`` is the ``(B, H)`` waist-yaw reference the IL term prices
    against.  It is passed in rather than picked out of ``d`` because its two
    sources differ: the fresh pool reads it off its own seed, the replay pool
    carries it as a harvested column -- see :meth:`Phase2Pool.waist_ref`.
    ``None`` leaves ``CostConfig.w_waist_aim`` with nothing to price, which is
    the pre-2026-08-12 behaviour and was silently the ONLY behaviour.
    """
    pick = lambda k: d[k].index_select(0, idx)  # noqa: E731
    return StrikeContext(
        waist_yaw_ref=waist_ref,
        hit_index=pick("hit").clamp_min(0),
        active=pick("active").to(drum_head_centers.dtype),
        target_drum=pick("drum"),
        start_drum=pick("start_drum"),
        p_star=pick("p_star"),
        v_star=pick("v_star"),
        n_out=pick("n_out"),
        hit_radius=pick("hit_radius"),
        disk_offset=pick("disk_off"),
        half_plane_dir=pick("hp_dir"),
        has_half_plane=pick("has_hp"),
        contact_sigma=pick("sigma"),
        shaft_allowed=pick("shaft"),
        start_shaft_allowed=pick("start_shaft"),
        elev_required=pick("elev"),
        drum_head_centers=drum_head_centers,
    )


# ============================================================ scenarios ======
def _peak_rate(q0: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    X = np.concatenate([q0[None, :], target], axis=0)
    return np.abs(np.diff(X, axis=0) / float(dt)).max(axis=0)


def _draw_time(clip: RhythmClip, rng: np.random.Generator, cfg: Phase2ScenarioConfig) -> float:
    if clip.hits and rng.uniform() < cfg.short_horizon_frac:
        hit = clip.hits[int(rng.integers(len(clip.hits)))]
        k = int(rng.integers(1, max(int(cfg.short_horizon_wp), 1) + 1))
        return max(float(hit.time) - k * cfg.dt, 0.0)
    return float(rng.uniform(0.0, max(clip.duration, cfg.dt)))


class _BoxStartPool:
    """Chunked, collision-screened rest starts for the eval-like family.

    Mirrors ``eval_harness._sample_box_start``: the waist and the *striking*
    arm are drawn in a scaled posture box, the idle arm sits at URDF neutral,
    and the pose is rejected if it collides.  Screening is batched over a chunk
    so it costs one SDF call per chunk rather than one per sample.
    """

    def __init__(self, sampler, cfg: Phase2ScenarioConfig, chunk: int = 256):
        self.s = sampler
        self.cfg = cfg
        self.chunk = int(chunk)
        names = tuple(cfg.joint_names)
        self.names = names
        lo, hi = urdf_position_limits(names, posture_limits=True)
        self.lo, self.hi = lo, hi
        self.mid, self.half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.idx = {
            h: [i for i, n in enumerate(names) if n.startswith("waist_") or n.startswith(f"{h}_")]
            for h in HANDS
        }
        self.v_limit = isaac_velocity_limits(names)
        self._buf: dict[str, list[np.ndarray]] = {h: [] for h in HANDS}

    def _refill(self, hand: str, rng: np.random.Generator, kit: DrumKit) -> None:
        cfg = self.cfg
        n = self.chunk
        idx = self.idx[hand]
        q = np.zeros((n, len(self.names)))
        q[:, idx] = (
            self.mid[idx][None, :]
            + cfg.eval_start_q_scale * self.half[idx][None, :]
            * rng.uniform(-1.0, 1.0, size=(n, len(idx)))
        )
        q = np.clip(q, self.lo[None, :], self.hi[None, :])
        heads = torch.as_tensor(
            kit.positions_in("env"), device=self.s.device, dtype=self.s.dtype
        ).unsqueeze(0).expand(n, -1, -1)
        with torch.no_grad():
            _p, det = self.s.sdf.forward(
                torch.as_tensor(q, device=self.s.device, dtype=self.s.dtype),
                drum_head_centers=heads, return_details=True,
            )
        ok = det.min_sdf.detach().cpu().numpy() > 0.0
        self._buf[hand] = list(q[ok])

    def draw(self, hand: str, rng: np.random.Generator, kit: DrumKit) -> np.ndarray | None:
        if not self._buf[hand]:
            self._refill(hand, rng, kit)
        if not self._buf[hand]:
            return None
        return self._buf[hand].pop()


def _apex_init(
    q0, qd0, Hb, goals, aim, kit_s, library, solver, cfg, names,
    q_bounds, v_limit, rng, idle_home=None,
):
    """The apex three-point spline as an INIT, or ``None`` to fall back.

    Returns ``None`` whenever the apex construction is unavailable (no active
    hand with room for an interior waypoint) or fails the same
    impossible-transition filter the straight line is held to.  The caller then
    builds the straight line for that scenario, so the apex path can never
    reshape the scenario distribution -- only the initialization.
    """
    active = [h for h in HANDS if goals.get(h) is not None]
    if not active:
        return None
    base = warmstart_trajectory(
        q0, qd0, horizon=Hb, goals={h: None for h in HANDS}, library=library,
        aim_targets=aim or None, dt=cfg.dt, joint_names=names,
        waist_cfg=cfg.waist_cfg, q_bounds=q_bounds,
        slew_limit_eta=cfg.seed_slew_eta,
    )
    waist = np.asarray(base.q, float)
    heads = torch.as_tensor(kit_s.positions_in("env"), device=solver.device,
                            dtype=solver.dtype).unsqueeze(0)
    apex_arm: dict[str, np.ndarray | None] = {h: None for h in HANDS}
    solved = False
    for hand in active:
        k_hit = int(round(goals[hand][1] / cfg.dt))
        if k_hit < 2:
            continue
        q_ref = q0.copy()
        k_ref = int(np.clip(round(0.5 * k_hit), 1, max(k_hit - 1, 1)))
        q_ref[:3] = waist[k_ref, :3]
        centre = np.asarray(aim[hand], float) + np.array([0.0, 0.0, cfg.apex_cfg.height])
        t = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a, float)[None, :], device=solver.device, dtype=solver.dtype
        )
        with torch.no_grad():
            res = solver.solve(hand, t(q_ref), t(centre), head_centers=heads, rng=rng)
        arm = res["q"][0].detach().cpu().numpy()
        apex_arm[hand] = arm[[i for i, nm in enumerate(names) if nm.startswith(f"{hand}_")]]
        solved = True
    if not solved:
        return None
    # ``apex_warmstart_trajectory`` builds its base with BOTH hands idle and then
    # overwrites only the active arms, so forwarding the home table here parks
    # the idle arm exactly as the straight-line path does.
    ws = apex_warmstart_trajectory(
        q0, qd0, horizon=Hb, goals=goals, library=library, apex_arm=apex_arm,
        aim_targets=aim or None, dt=cfg.dt, joint_names=names, cfg=cfg.apex_cfg,
        q_bounds=q_bounds, waist_cfg=cfg.waist_cfg,
        slew_limit_eta=cfg.seed_slew_eta,
        idle_home=idle_home, idle_settle_wp=cfg.idle_settle_wp,
    )
    target = np.asarray(ws.target, float)
    if float((_peak_rate(q0, target, cfg.dt) / v_limit).max()) > cfg.vel_reject_scale:
        return None
    if cfg.seed_slew_eta:
        step = float(cfg.seed_slew_eta) * v_limit * float(cfg.dt)
        full = np.concatenate([q0[None, :], target], axis=0)
        for j in range(full.shape[1]):
            full[:, j] = _slew_limited(full[:, j], float(step[j]))
        target = full[1:]
    return target


def generate_cost_scenarios(
    n: int,
    clips: list[RhythmClip],
    *,
    seed: int,
    cfg: Phase2ScenarioConfig | None = None,
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    sampler=None,
    start_chunk: int = 256,
) -> dict:
    """``n`` accepted phase-2 scenarios, keyed by horizon bucket."""
    cfg = cfg or Phase2ScenarioConfig()
    rng = np.random.default_rng(seed)
    names = tuple(cfg.joint_names)
    kit = kit or DrumKit.from_layout()
    library = library or GoalPoseLibrary(joint_names=names)
    if sampler is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=library)
    q_bounds = urdf_position_limits(names)
    v_limit = isaac_velocity_limits(names)
    H = int(cfg.horizon)
    idle_home = resolve_idle_home(cfg.idle_home_path, joint_names=names)
    box = _BoxStartPool(sampler, cfg)
    eval_names = [n_ for n_ in EVAL_DRUM_HAND if n_ in kit.names]
    solver = (
        ApexSolver(cfg=cfg.apex_cfg, fk=sampler.fk, sdf=sampler.sdf,
                   joint_names=names, dtype=sampler.dtype)
        if cfg.apex_init_frac > 0.0 else None
    )

    keys = ("q0", "qd0", "init", "hit", "active", "drum", "start_drum", "kind",
            *STRIKE_FIELDS, *CONTEXT_FIELDS)
    out: dict[int, dict[str, list]] = {b: {k: [] for k in keys} for b in cfg.buckets}
    stats = {
        "requested": int(n), "accepted": 0, "reject_idle": 0, "reject_velocity": 0,
        "reject_library": 0, "reject_box_start": 0, "dual_active": 0,
        "asymmetric": 0, "short_horizon": 0, "eval_like": 0, "rest": 0,
        "single_idle": 0,
        "apex_requested": 0, "apex_used": 0, "apex_fallback": 0,
        "peak_rate_ratio": [],
    }

    starts = None
    si = 0
    for _ in range(int(n)):
        kit_s = kit.randomized(rng, cfg.kit_randomization) if cfg.randomize_kit else kit
        eval_like = bool(rng.uniform() < cfg.eval_like_frac)

        # ---------------------------------------------------- the scenario --
        if eval_like:
            name = eval_names[int(rng.integers(len(eval_names)))]
            hand = EVAL_DRUM_HAND[name]
            di = kit_s.index(name)
            q0 = box.draw(hand, rng, kit_s)
            if q0 is None:
                stats["reject_box_start"] += 1
                continue
            sel = np.array(
                [1.0 if (nm.startswith("waist_") or nm.startswith(f"{hand}_")) else 0.0
                 for nm in names]
            )
            # The grid jitters its three repeats around one base pose; here every
            # sample is its own draw, so the jitter is just extra spread.
            q0 = np.clip(
                q0 + sel * rng.uniform(
                    -cfg.eval_start_jitter, cfg.eval_start_jitter, size=len(names)
                ),
                *q_bounds,
            )
            qd0 = np.zeros(len(names))
            if rng.uniform() < cfg.eval_moving_frac:
                qd0 = cfg.eval_v_scale * v_limit * rng.uniform(-1, 1, size=len(names)) * sel
            t_h = float(rng.uniform(*cfg.eval_horizon_s))
            k_hit = int(np.clip(round(t_h / cfg.dt), 4, H))
            speed = float(cfg.eval_speeds[int(rng.integers(len(cfg.eval_speeds)))])
            pos = sample_hit_point(kit_s, di, rng, hit_radius_frac=cfg.hit_radius_frac)
            vel = -speed * np.asarray(kit_s.axes[di], float)
            goal = {h: HandGoal.inactive() for h in HANDS}
            goal[hand] = HandGoal(
                active=True, drum=di, position=pos, velocity=vel,
                time_to_hit=k_hit * cfg.dt, drum_name=name, loudness=1.0,
            )
            snap = GoalSnapshot(t_now=0.0, goal=goal, future={h: [] for h in HANDS})
            hit = {hand: k_hit, ("right" if hand == "left" else "left"): -1}
            src = np.full(2, -1, dtype=np.int64)
            kind = KIND_EVAL_LIKE
        else:
            if starts is None or si >= len(starts):
                starts = sampler.sample(int(start_chunk), rng=rng, kit=kit_s)
                si = 0
            q0 = starts.q[si].detach().cpu().numpy().astype(float)
            qd0 = starts.qd[si].detach().cpu().numpy().astype(float)
            src = starts.drum[si].detach().cpu().numpy().astype(np.int64)
            si += 1
            clip = clips[int(rng.integers(len(clips)))]
            snap = goal_snapshot(
                clip, kit_s, _draw_time(clip, rng, cfg),
                num_future=cfg.num_future, rng=rng,
                hit_radius_frac=cfg.hit_radius_frac,
                speed_min=cfg.speed_min, speed_max=cfg.speed_max,
                horizon_cap=H * cfg.dt,
            )
            hit = {}
            for h in HANDS:
                g = snap.goal[h]
                hit[h] = (
                    int(np.clip(round(g.time_to_hit / cfg.dt), 1, H)) if g.active else -1
                )
            kind = KIND_PLAN

        active = [hit[h] >= 0 for h in HANDS]
        if not any(active):
            # A both-hands-idle draw is a REST.  Keep it at ``rest_frac`` when a
            # home table is available to park the arms at (without one the row is
            # a pose held in place with nothing pricing it, which is the state
            # the idle terms exist to replace).
            if cfg.require_active_hand and (
                idle_home is None or rng.uniform() >= cfg.rest_frac
            ):
                stats["reject_idle"] += 1
                continue
            stats["rest"] += 1
        goals: dict[str, tuple[str, float] | None] = {}
        aim: dict[str, np.ndarray] = {}
        for h in HANDS:
            g = snap.goal[h]
            if hit[h] < 0:
                goals[h] = None
                continue
            goals[h] = (g.drum_name, hit[h] * cfg.dt)
            aim[h] = np.asarray(g.position, float)
        if not all(library.has(h, goals[h][0]) for h in HANDS if goals[h] is not None):
            stats["reject_library"] += 1
            continue

        k_max = max([hit[h] for h in HANDS if hit[h] >= 0], default=1)
        if any(active):
            Hb = bucket_for(k_max, cfg.buckets)
        else:
            # A rest has no hit to size the horizon from, and ``bucket_for(1)``
            # would file every one of them in the shortest bucket.  Both arms are
            # priced over the WHOLE grid when neither is active (see
            # ``effective_hits``), so spread rests across the buckets instead.
            Hb = int(cfg.buckets[int(rng.integers(len(cfg.buckets)))])
        if any(active) and not all(active):
            stats["single_idle"] += 1

        init = None
        want_apex = solver is not None and rng.uniform() < cfg.apex_init_frac
        if want_apex:
            stats["apex_requested"] += 1
            init = _apex_init(
                q0, qd0, Hb, goals, aim, kit_s, library, solver, cfg, names,
                q_bounds, v_limit, rng, idle_home=idle_home,
            )
            if init is None:
                stats["apex_fallback"] += 1
            else:
                stats["apex_used"] += 1
                stats["peak_rate_ratio"].append(
                    float((_peak_rate(q0, init, cfg.dt) / v_limit).max())
                )

        if init is None:
            ws = warmstart_trajectory(
                q0, qd0, horizon=Hb, goals=goals, library=library,
                aim_targets=aim or None, dt=cfg.dt, joint_names=names,
                waist_cfg=cfg.waist_cfg, q_bounds=q_bounds,
                slew_limit_eta=cfg.seed_slew_eta,
                idle_home=idle_home, idle_settle_wp=cfg.idle_settle_wp,
            )
            init = np.asarray(ws.init_traj, float)
            # The filter reads the PRE-slew demand: the slew limiter would make
            # an impossible hit spacing look feasible instead of dropping it.
            ratio = float((np.asarray(ws.peak_rate_raw, float) / v_limit).max())
            stats["peak_rate_ratio"].append(ratio)
            if ratio > cfg.vel_reject_scale:
                stats["reject_velocity"] += 1
                continue

        ctx = build_context_inputs(
            kit_s, snap, num_drums=cfg.num_drums, num_future=cfg.num_future,
            frame="env", device="cpu", dtype=torch.float64,
            idle_home=idle_home,
        )

        rec = out[Hb]
        rec["q0"].append(q0.astype(np.float32))
        rec["qd0"].append(qd0.astype(np.float32))
        rec["init"].append(init.astype(np.float32))
        rec["hit"].append(np.array([hit["left"], hit["right"]], dtype=np.int64))
        rec["active"].append(np.array(active, dtype=bool))
        rec["drum"].append(
            np.array([snap.goal[h].drum if hit[h] >= 0 else -1 for h in HANDS], np.int64)
        )
        rec["start_drum"].append(np.asarray(src, np.int64))
        rec["kind"].append(np.int64(kind))
        rows = []
        for hi_, h in enumerate(HANDS):
            if hit[h] < 0:
                rows.append(_empty_strike())
            else:
                g = snap.goal[h]
                rows.append(_strike_row(kit_s, g.drum, g.position, g.velocity))
        # start-drum shaft eligibility comes from the DEPARTURE drum, per hand.
        for hi_, s in enumerate(np.asarray(src, np.int64)):
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

        stats["accepted"] += 1
        stats["eval_like"] += int(eval_like)
        if all(active):
            stats["dual_active"] += 1
            if hit["left"] != hit["right"]:
                stats["asymmetric"] += 1
        if k_max <= cfg.short_horizon_wp:
            stats["short_horizon"] += 1

    packed = {
        b: {k: np.stack(v) for k, v in rec.items()}
        for b, rec in out.items() if rec["q0"]
    }
    stats["peak_rate_ratio"] = np.asarray(stats["peak_rate_ratio"], dtype=np.float32)
    return {"buckets": packed, "stats": stats}


def _worker(args) -> dict:
    n, seed, cfg, clip_dir = args
    torch.set_num_threads(1)
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(clip_dir, kit)
    return generate_cost_scenarios(n, clips, seed=seed, cfg=cfg, kit=kit)


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
    out_b = {b: {k: np.concatenate(v) for k, v in a.items()} for b, a in buckets.items()}
    if "peak_rate_ratio" in stats:
        stats["peak_rate_ratio"] = np.concatenate(stats["peak_rate_ratio"])
    return {"buckets": out_b, "stats": stats}


# ================================================================= pool ======
class Phase2Pool:
    """Bucketed phase-2 scenario pool, resident on one device."""

    def __init__(
        self,
        buckets: dict[int, dict[str, np.ndarray]],
        *,
        cfg: Phase2ScenarioConfig,
        noise: Phase2Noise | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        stats: dict | None = None,
    ):
        self.cfg = cfg
        self.noise = noise or Phase2Noise()
        self.device = torch.device(device)
        self.dtype = dtype
        self.stats = dict(stats or {})
        names = tuple(cfg.joint_names)
        lo, hi = urdf_position_limits(names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        self._pelvis = torch.as_tensor(PELVIS_IN_ENV, device=self.device, dtype=dtype)

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

    # ------------------------------------------------- horizon bookkeeping --
    def hit_min(self, bucket: int) -> torch.Tensor:
        """``(N,)`` nearest ACTIVE hit index per stored scenario.

        A scenario with no active hand reads ``bucket + 1``, i.e. "longer than
        anything in this bucket", so it never counts as short.  This is the
        statistic the MPC actually lives in: the closed loop spends every strike
        passing through ``k = 5, 4, 3, 2, 1`` and those are the plans whose
        terminal is executed.
        """
        d = self.data[int(bucket)]
        hit, act = d["hit"], d["active"]
        big = torch.full_like(hit, int(bucket) + 1)
        return torch.where(act, hit.clamp_min(0), big).min(dim=1).values

    def horizon_histogram(self, bucket: int, edges=(1, 2, 3, 5, 8, 12, 16, 25, 50)) -> dict:
        """``{edge: count}`` of :meth:`hit_min` at or below each edge, cumulative."""
        k = self.hit_min(bucket).float()
        n = int(k.numel())
        out = {"n": n}
        prev = 0
        for e in edges:
            c = int((k <= e).sum())
            out[f"<= {e}"] = c - prev
            prev = c
        out["> %d" % edges[-1]] = n - prev
        return out

    def short_weight(self, bucket: int, *, short_wp: int, boost: float) -> torch.Tensor:
        """Per-row sampling weight that oversamples the final-approach regime.

        The roller harvests every slot on every tick, so its horizon
        distribution is roughly uniform over each goal's initial horizon --
        which puts only a few percent of the mix inside the ``k <= 5`` window
        the MPC's terminal waypoint is drawn from.  Measured on the closed loop,
        the plan's own terminal aim is inside the hit disk down to ``k ~ 5`` and
        then drifts ~60 mm OUT of it by ``k = 1``, which is a competence gap in
        exactly the band that is under-represented.  ``boost`` multiplies those
        rows' sampling weight; 1.0 restores uniform.
        """
        k = self.hit_min(bucket)
        w = torch.ones(k.shape, device=k.device, dtype=self.dtype)
        if short_wp > 0 and boost != 1.0:
            w = torch.where(k <= int(short_wp), w * float(boost), w)
        return w

    def sample_idx(
        self,
        bucket: int,
        n: int,
        generator: torch.Generator | None = None,
        *,
        short_wp: int = 0,
        short_boost: float = 1.0,
    ) -> torch.Tensor:
        m = self.size(bucket)
        if short_wp <= 0 or short_boost == 1.0:
            return torch.randint(0, m, (n,), device=self.device, generator=generator)
        w = self.short_weight(bucket, short_wp=short_wp, boost=short_boost)
        return torch.multinomial(w, n, replacement=True, generator=generator)

    def masks_for(self, hit: torch.Tensor, active: torch.Tensor, H: int):
        t = hit.clamp_min(0).to(self.dtype) * float(self.cfg.dt)
        return build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=int(H), dt=self.cfg.dt,
            active_left=active[:, 0], active_right=active[:, 1],
            joint_names=self.cfg.joint_names,
        )

    def context(self, bucket: int, idx: torch.Tensor) -> ContextInputs:
        d = self.data[int(bucket)]
        # Fields absent from the stored arrays are left to ``ContextInputs``'s
        # own defaults, which materialize zeros + invalid.  That is what lets a
        # pool cached BEFORE the idle-home block existed still load: those rows
        # simply carry no idle token, which is the truth about them.
        return ContextInputs(
            **{f: d[f].index_select(0, idx) for f in CONTEXT_FIELDS if f in d}
        )

    #: Where :meth:`waist_ref` gets the waist-yaw reference when there is no
    #: explicit ``waist_ref`` column.  True on the FRESH pool, whose ``init`` is
    #: the warm-start seed and whose waist-yaw column is therefore
    #: ``yaw_reference(yaw_start, aim_yaw(targets))`` evaluated by
    #: :func:`~amortized_opt.warmstart.warmstart_trajectory` with this pool's own
    #: ``waist_cfg`` -- i.e. exactly the reference, already computed, per sample.
    #:
    #: :class:`~amortized_opt.train_cost.ReplayPool` sets it FALSE, because its
    #: ``init`` is the roller's shifted PREVIOUS PLAN -- the model's own output.
    #: Pricing the model against that would be self-referential: any yaw
    #: trajectory it produced would be its own reference and the term would read
    #: zero no matter what the torso did.  The replay pool must carry a real
    #: harvested column or contribute no waist-aim gradient at all.
    waist_ref_from_seed: bool = True

    def waist_ref(self, bucket: int) -> torch.Tensor | None:
        """``(N, H)`` waist-yaw reference for this bucket, or ``None``."""
        d = self.data[int(bucket)]
        if "waist_ref" in d:
            return d["waist_ref"]
        if not self.waist_ref_from_seed:
            return None
        yaw_i = tuple(self.cfg.joint_names).index("waist_yaw_joint")
        return d["init"][:, :, yaw_i]

    def noised_init(
        self, init: torch.Tensor, bucket: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        nc = self.noise
        B, H, D = init.shape
        L = self._ou[int(bucket)]
        eps = torch.randn(B, H, D, device=init.device, dtype=init.dtype, generator=generator)
        u = torch.rand(B, 1, 1, device=init.device, dtype=init.dtype, generator=generator)
        lo, hi = math.log(nc.sigma_corr_min), math.log(nc.sigma_corr_max)
        sigma = torch.exp(lo + (hi - lo) * u)
        out = init + sigma * torch.einsum("ij,bjd->bid", L, eps)
        if nc.sigma_white > 0.0:
            out = out + nc.sigma_white * torch.randn(
                B, H, D, device=init.device, dtype=init.dtype, generator=generator
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
        noise: bool = True,
    ) -> dict:
        H = int(bucket)
        d = self.data[H]
        pick = lambda k: d[k].index_select(0, idx)  # noqa: E731
        q0, v0, seed = pick("q0"), pick("qd0"), pick("init")
        masks = self.masks_for(pick("hit"), pick("active"), H)
        init = self.noised_init(seed, H, generator=generator) if noise else seed.clone()
        ctx = self.context(H, idx)
        heads = ctx.drum_pos - self._pelvis
        wref = self.waist_ref(H)
        sctx = strike_context_from_arrays(
            d, idx, drum_head_centers=heads,
            waist_ref=None if wref is None else wref.index_select(0, idx),
        )
        return {
            "bucket": H,
            "q0": q0,
            "v0": v0,
            "state": torch.cat([q0, v0], dim=-1),
            "init": init * masks.elem_mask,
            "seed": seed,
            "masks": masks,
            "context": ctx,
            "strike": sctx,
            "hit": pick("hit"),
            "active": pick("active"),
            "kind": pick("kind"),
        }


def generate_cost_pool(
    n: int,
    *,
    seed: int,
    cfg: Phase2ScenarioConfig | None = None,
    noise: Phase2Noise | None = None,
    split: str = "train",
    clip_dir: str | Path | None = None,
    workers: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Phase2Pool:
    """Fan :func:`generate_cost_scenarios` out over processes and wrap it."""
    import multiprocessing as mp

    cfg = cfg or Phase2ScenarioConfig()
    clip_dir = str(clip_dir if clip_dir is not None else default_rcc_dir(split))
    workers = int(workers or min(os.cpu_count() or 8, 64))
    workers = max(1, min(workers, int(n)))
    per = [n // workers] * workers
    for i in range(n - sum(per)):
        per[i] += 1
    args = [(per[i], seed * 100003 + i, cfg, clip_dir) for i in range(workers)]
    if workers == 1:
        raw = _merge([_worker(args[0])])
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            raw = _merge(pool.map(_worker, args))
    return Phase2Pool(
        raw["buckets"], cfg=cfg, noise=noise, device=device, dtype=dtype,
        stats=raw["stats"],
    )
