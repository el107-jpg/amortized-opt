"""Phase-1 warm-start trajectories: accelerating strike seeds + input noise.

The phase-1 curriculum imitates cheap trajectories to library terminal poses,
with the waist following the pre-built aim system. Active arms use a short
acceleration ramp followed by constant acceleration, matching measured initial
velocity and leaving arrival velocity free. Waist and idle motion still use
rest-at-target quintics.

Three pieces:

* :class:`GoalPoseLibrary` -- the per-(hand, drum, waist-yaw) terminal pose
  library, read from the vendored ``assets/drum_goal_poses.json``.
* :func:`warmstart_trajectory` -- a dual-arm, waist-coupled (H+1, 17) command
  trajectory on the 50 Hz grid. Each active arm reaches its library pose at
  its own hit time and continues moving through contact; the waist
  runs :func:`~amortized_opt.waist_aim.waist_reference_block`, which is a
  quintic yaw ramp with pitch and roll frozen. The two arms are therefore
  *decoupled in the seed* and coupled only through the shared waist -- which is
  the coupling the model is supposed to learn.
* :class:`InitNoise` -- Gaussian + temporally-correlated (Ornstein-Uhlenbeck)
  perturbation of that trajectory, so the model never sees a clean seed and
  cannot learn to copy one.

KNOWN STALENESS (not fixed here, flagged deliberately): the vendored
``drum_goal_poses.json`` predates the ``waist_aim`` damping change 0.7 -> 0.3
(2026-08-04). Two of its 33 waist-yaw *pair* poses move under the current law.
The per-drum ``yaw_grid`` this module actually reads is indexed by waist yaw
directly, so it is insensitive to the damping value -- the staleness bites only
a caller that uses the ``pairs`` block. Regenerate with the repo's
``scripts/gen_drum_goal_poses.py`` before trusting ``pairs``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import (
    CONTROL_DT,
    DRUM_GOAL_POSES_PATH,
    G1_UPPER_BODY_JOINTS,
    ready_joint_pos,
)
from .limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from .spline import retime_joint_path
from .strike_seed import accelerating_path_numpy
from .waist_aim import WaistAimConfig, aim_yaw, waist_reference_block

__all__ = [
    "GoalPoseLibrary",
    "LibraryEntry",
    "WarmStart",
    "InitNoise",
    "correlated_noise",
    "arm_slice",
    "waist_slice",
    "waist_pr_slice",
    "warmstart_trajectory",
    "seed_waist_pr_to_sp",
    "update_waist_pr_ema",
    "rewrite_waist_target_to_sp",
]

_WAIST_NAMES = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")


def arm_slice(hand: str, joint_names=G1_UPPER_BODY_JOINTS) -> list[int]:
    return [i for i, n in enumerate(joint_names) if n.startswith(f"{hand}_")]


def waist_slice(joint_names=G1_UPPER_BODY_JOINTS) -> list[int]:
    return [i for i, n in enumerate(joint_names) if n.startswith("waist_")]


def waist_pr_slice(joint_names=G1_UPPER_BODY_JOINTS) -> list[int]:
    """Pitch then roll, matching ``FusedCost.c_waist_pr`` order."""
    names = tuple(joint_names)
    return [names.index("waist_pitch_joint"), names.index("waist_roll_joint")]


def update_waist_pr_ema(ema, cmd_pr, *, alpha: float = 0.1, leak: float = 0.02):
    """Leaky EMA of the committed (clipped) waist pitch/roll command.

    ``ema ← (1 − leak) · ((1 − α) · ema + α · cmd_pr)``.  ``α = 1`` is
    lag-1; ``leak = 1`` snaps to upright.  The two jobs are separate:
    memory stops closed-loop hunting, leak is the attractor toward 0.
    Works on numpy arrays and torch tensors.  Caller ``stopgrad``s.
    """
    a = float(alpha)
    b = float(leak)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {a}")
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"leak must be in [0, 1], got {b}")
    return (1.0 - b) * ((1.0 - a) * ema + a * cmd_pr)


def seed_waist_pr_to_sp(
    init,
    q0,
    qd0,
    sp_pr,
    *,
    joint_names=None,
    dt: float = CONTROL_DT,
    ease_wp: int | None = None,
):
    """Ease ``init`` pitch/roll to the encoder set-point -- the yaw-ramp analogue.

    Yaw's quintic already ramps to :func:`~amortized_opt.waist_aim.aim_yaw`
    (the encoder yaw head's untrained set-point).  Pitch/roll have no analytic
    aim; ``sp_pr`` *is* that set-point.  Untrained (``sp_pr == 0``) is the
    existing ease-to-upright.  ``sp_pr`` is detached: the seed is a prior, not
    a gradient path.  No-op when ``sp_pr`` is None.

    ``ease_wp`` is the minimum window (waypoints); the same 1.875x stretch as
    ``waist_upright_ease_wp`` applies.  ``None`` uses 5 -- yaw always ramps,
    so a head-on seed always eases rather than holding the last command.
    """
    if sp_pr is None:
        return init
    import torch

    names = tuple(joint_names or G1_UPPER_BODY_JOINTS)
    pitch_i = names.index("waist_pitch_joint")
    roll_i = names.index("waist_roll_joint")
    pr = [pitch_i, roll_i]
    _B, H, _n = init.shape
    k_min = 5 if ease_wp is None else max(int(ease_wp), 1)
    dt = float(dt)

    out = init.clone()
    q0t = q0.to(device=init.device, dtype=init.dtype)
    q0w = q0t[:, pr]
    term = sp_pr.detach().to(device=init.device, dtype=init.dtype).reshape(-1, 2)
    v_lim = isaac_velocity_limits(names)
    step = torch.as_tensor(
        [float(ETA_VELOCITY) * float(v_lim[i]) * dt for i in pr],
        device=init.device, dtype=init.dtype,
    ).clamp_min(1e-9)
    v0w = torch.zeros_like(q0w)
    if qd0 is not None:
        cap = step / dt
        v0w = qd0.to(device=init.device, dtype=init.dtype)[:, pr].clamp(-cap, cap)

    def _ramp(lengths):
        k = torch.arange(1, H + 1, device=init.device, dtype=init.dtype).view(1, H, 1)
        L = lengths.clamp_min(1).to(init.dtype)
        u = (k / L.unsqueeze(1)).clamp(0.0, 1.0)
        d = (term - q0w).unsqueeze(1)
        V = (v0w * L * dt).unsqueeze(1)
        u3 = u * u * u
        return (
            q0w.unsqueeze(1) + V * u
            + (10.0 * d - 6.0 * V) * u3
            + (8.0 * V - 15.0 * d) * u3 * u
            + (6.0 * d - 3.0 * V) * u3 * u * u
        )

    need = torch.ceil(1.875 * (term - q0w).abs() / step)
    L = need.clamp(float(k_min), float(H))
    pr_sub = _ramp(L)
    d0 = (pr_sub[:, 0, :] - q0w).abs().unsqueeze(1)
    peak = torch.cat([d0, (pr_sub[:, 1:] - pr_sub[:, :-1]).abs()], 1).amax(1)
    ratio = peak / step
    over = ratio > 1.0
    if bool(over.any()):
        pr_sub = _ramp(torch.where(over, (L * ratio).ceil().clamp(1.0, float(H)), L))
    out[:, :, pitch_i] = pr_sub[:, :, 0]
    out[:, :, roll_i] = pr_sub[:, :, 1]
    return out


def rewrite_waist_target_to_sp(
    target,
    q0,
    sp,
    cols,
    *,
    k_min: int,
    elem_mask,
    dt: float = CONTROL_DT,
    row_mask=None,
):
    """Rewrite ``target`` waist columns to a rest-to-rest quintic ``q0 → sp``.

    Arms are untouched.  ``k*`` is ``clamp(ceil(1.875 |Δq| / qdot_max), k_min,
    H_waist)`` per waist DoF, with ``H_waist`` from ``elem_mask`` (not bucket
    H).  Rows whose waist horizon is ``< 2`` are left alone.  ``row_mask``, if
    given, selects which batch rows to rewrite.

    The seed ``init`` is deliberately NOT rewritten -- the decoder has to read
    the set-point token rather than copy init.
    """
    import torch

    if not cols:
        return target
    out = target.clone()
    B, H, _n = target.shape
    device, dtype = target.device, target.dtype
    cols_t = torch.as_tensor(list(cols), device=device, dtype=torch.long)
    n_c = cols_t.numel()
    q0w = q0.to(device=device, dtype=dtype).index_select(-1, cols_t)
    term = sp.to(device=device, dtype=dtype).reshape(B, n_c)
    names = G1_UPPER_BODY_JOINTS
    v_lim = isaac_velocity_limits(names)
    step = torch.as_tensor(
        [float(ETA_VELOCITY) * float(v_lim[int(i)]) * float(dt) for i in cols],
        device=device, dtype=dtype,
    ).clamp_min(1e-9)

    waist_m = elem_mask.index_select(-1, cols_t).amax(-1)  # (B, H)
    idx = torch.arange(H, device=device, dtype=dtype).view(1, H)
    H_w = (waist_m * (idx + 1.0)).amax(-1).clamp_min(0.0)  # (B,)
    skip = H_w < 2.0
    if row_mask is not None:
        skip = skip | ~row_mask.to(dtype=torch.bool, device=device)

    need = torch.ceil(1.875 * (term - q0w).abs() / step)
    hi = H_w.unsqueeze(-1).clamp_min(1.0).expand_as(need)
    lo = torch.minimum(torch.full_like(hi, float(max(int(k_min), 1))), hi)
    L = need.clamp(min=lo, max=hi)
    k = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H, 1)
    u = (k / L.unsqueeze(1).clamp_min(1.0)).clamp(0.0, 1.0)
    d = (term - q0w).unsqueeze(1)
    u3 = u * u * u
    ramp = q0w.unsqueeze(1) + (10.0 * d) * u3 + (-15.0 * d) * u3 * u + (6.0 * d) * u3 * u * u

    keep = (~skip).to(dtype).view(B, 1, 1)
    valid = (elem_mask.index_select(-1, cols_t) > 0.5).to(dtype)
    w = keep * valid
    cur = out.index_select(-1, cols_t)
    mixed = ramp * w + cur * (1.0 - w)
    out.index_copy_(-1, cols_t, mixed)
    return out


def _boundary_quintic(
    q_from: float, v_from: float, q_to: float, k: int, H: int, dt: float
) -> np.ndarray:
    """Rows ``0..H`` of the quintic with BCs ``p(0)=q_from, p'(0)=v_from,
    p''(0)=0`` and ``p=q_to`` at rest from waypoint ``k`` on (held after).

    ``v_from = 0`` reduces exactly to the quintic smoothstep
    ``q_from + (q_to-q_from)*u^3*(10-15u+6u^2)``.  The GPU twin is
    :func:`amortized_opt.train_cost.vel_ramp_seed`; the two must agree
    element-for-element (pinned by ``tests/test_waist_upright_ease.py``).
    """
    u = np.clip(np.arange(H + 1, dtype=float) / float(max(int(k), 1)), 0.0, 1.0)
    d = float(q_to) - float(q_from)
    V = float(v_from) * float(max(int(k), 1)) * float(dt)
    u3 = u * u * u
    return (
        q_from + V * u
        + (10.0 * d - 6.0 * V) * u3
        + (8.0 * V - 15.0 * d) * u3 * u
        + (6.0 * d - 3.0 * V) * u3 * u * u
    )


def _slew_limited(x: np.ndarray, max_step: float) -> np.ndarray:
    """Forward pass limiting ``|x[k] - x[k-1]|`` to ``max_step``, anchored at ``x[0]``.

    A plain clip of the *differences* would drift the whole tail; this walks the
    sequence and tracks the target as closely as the step bound allows, so a
    reference that was already inside the bound is returned unchanged (bit for
    bit) and one that was not is the fastest admissible approach to it.
    """
    x = np.asarray(x, float)
    if x.size < 2 or not np.isfinite(max_step) or max_step <= 0.0:
        return x
    d = np.diff(x)
    if float(np.max(np.abs(d))) <= max_step:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    for k in range(1, x.size):
        out[k] = out[k - 1] + float(np.clip(x[k] - out[k - 1], -max_step, max_step))
    return out


# ------------------------------------------------------------------ library --
@dataclass(frozen=True)
class LibraryEntry:
    hand: str
    drum: str
    waist_yaw: float
    q: np.ndarray  # (17,) with only this hand's arm + the waist filled
    arm_q: np.ndarray  # (7,) this hand's arm joints, optimizer order
    tip_err: float
    wrist_headroom: float


class GoalPoseLibrary:
    """Per-drum terminal joint poses on a waist-yaw grid.

    ``assets/drum_goal_poses.json`` stores, for each (hand, drum), up to nine
    IK solutions -- one per 22.5 deg step of waist yaw over +/-100 deg -- each
    annotated with the tip error it achieved and the wrist headroom it left.
    :meth:`select` returns the entry whose waist yaw is closest to a requested
    one *among those that satisfy the terminal condition*, which is the plan's
    "closest joint position out of a library per drum that satisfies terminal
    position conditions".

    Note the library is already hand-eligible: ``left/floor_tom`` and
    ``right/hihat`` simply do not exist in it, matching the no-crossover rule.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        joint_names=G1_UPPER_BODY_JOINTS,
        max_tip_err: float = 0.12,
        min_wrist_headroom: float = 0.0,
    ):
        self.path = Path(path) if path is not None else DRUM_GOAL_POSES_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"goal-pose library missing at {self.path}")
        self.joint_names = tuple(joint_names)
        self.max_tip_err = float(max_tip_err)
        self.min_wrist_headroom = float(min_wrist_headroom)
        raw = json.loads(self.path.read_text())
        self.dt = float(raw.get("dt", CONTROL_DT))
        self.horizon_s = float(raw.get("horizon_s", 0.48))
        self.strike_speed_mps = float(raw.get("strike_speed_mps", 0.8))
        self.yaw_grid_rad = [float(v) for v in raw.get("yaw_grid_rad", [])]
        self._raw = raw

        self._entries: dict[tuple[str, str], list[LibraryEntry]] = {}
        for hand, per_drum in raw.get("yaw_grid", {}).items():
            for drum, rows in per_drum.items():
                out = []
                for row in rows:
                    q = self._dict_to_vec(row["q"])
                    out.append(
                        LibraryEntry(
                            hand=hand,
                            drum=drum,
                            waist_yaw=float(row["waist_yaw"]),
                            q=q,
                            arm_q=q[arm_slice(hand, self.joint_names)],
                            tip_err=float(row.get("tip_err", np.nan)),
                            wrist_headroom=float(row.get("wrist_headroom", np.nan)),
                        )
                    )
                out.sort(key=lambda e: e.waist_yaw)
                self._entries[(hand, drum)] = out
        # Fall back to the yaw-free base poses for any (hand, drum) with no grid.
        for hand, per_drum in raw.get("poses", {}).items():
            for drum, qd in per_drum.items():
                key = (hand, drum)
                if self._entries.get(key):
                    continue
                q = self._dict_to_vec(qd)
                self._entries[key] = [
                    LibraryEntry(hand, drum, float(q[0]), q, q[arm_slice(hand, self.joint_names)],
                                 float("nan"), float("nan"))
                ]

    def _dict_to_vec(self, d: dict) -> np.ndarray:
        v = np.zeros(len(self.joint_names))
        for i, name in enumerate(self.joint_names):
            if name in d:
                v[i] = float(d[name])
        return v

    # ------------------------------------------------------------ queries --
    @property
    def pairs(self) -> list[tuple[str, str]]:
        return sorted(self._entries)

    def drums(self, hand: str) -> list[str]:
        return sorted(d for (h, d) in self._entries if h == hand)

    def has(self, hand: str, drum: str) -> bool:
        return bool(self._entries.get((hand, drum)))

    def entries(self, hand: str, drum: str) -> list[LibraryEntry]:
        rows = self._entries.get((hand, drum))
        if not rows:
            raise KeyError(f"no library pose for hand={hand!r} drum={drum!r}")
        return rows

    def select(self, hand: str, drum: str, waist_yaw: float = 0.0) -> LibraryEntry:
        """Closest-in-yaw entry that satisfies the terminal condition.

        The condition is ``tip_err <= max_tip_err`` and
        ``wrist_headroom >= min_wrist_headroom``. If nothing passes (the two
        hard cross-body reaches can fail it at some yaws), the entry with the
        smallest tip error is returned instead -- a degraded seed is far better
        than no seed, and the cost function prices the shortfall anyway.
        """
        rows = self.entries(hand, drum)
        ok = [
            e
            for e in rows
            if not (e.tip_err == e.tip_err)  # NaN => unannotated base pose, accept
            or (e.tip_err <= self.max_tip_err and e.wrist_headroom >= self.min_wrist_headroom)
        ]
        pool = ok if ok else rows
        if not ok:
            return min(pool, key=lambda e: (e.tip_err, abs(e.waist_yaw - float(waist_yaw))))
        return min(pool, key=lambda e: (abs(e.waist_yaw - float(waist_yaw)), e.tip_err))

    def terminal_q(
        self,
        hand: str,
        drum: str,
        waist_yaw: float = 0.0,
        *,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        """(17,) pose: ``base`` with this hand's arm replaced by the library's."""
        e = self.select(hand, drum, waist_yaw)
        q = (ready_joint_pos(self.joint_names) if base is None else np.asarray(base, float)).copy()
        q[arm_slice(hand, self.joint_names)] = e.arm_q
        return q


# ----------------------------------------------------------------- the seed --
@dataclass
class WarmStart:
    """A dual-arm command trajectory on the 50 Hz grid.

    ``q`` is ``(H+1, 17)``: row 0 is the (pinned) start pose and row ``k`` is
    the command at ``k * dt``. ``hit_index`` is each hand's terminal waypoint
    index, i.e. its horizon in waypoints; ``-1`` for an inactive hand.
    """

    q: np.ndarray
    dt: float
    hit_index: dict[str, int]
    yaw_target: float
    peak_rate: dict[str, float] = field(default_factory=dict)
    library: dict[str, str] = field(default_factory=dict)
    #: Per-joint peak commanded ``|qd|`` (rad/s) of the seed BEFORE any slew
    #: limiting.  The impossible-transition filter has to read this, not the
    #: emitted trajectory: the slew limit would otherwise make every hit
    #: spacing look feasible by construction.
    peak_rate_raw: np.ndarray | None = None
    #: The (clamped, slew-limited) waist-yaw reference column, ``(H+1,)``.  A
    #: receding-horizon caller carries ``yaw_state_after(k)`` forward instead of
    #: re-reading the measured yaw rate -- see :func:`warmstart_trajectory`.
    yaw_ref: np.ndarray | None = None

    @property
    def horizon(self) -> int:
        return int(self.q.shape[0]) - 1

    @property
    def init_traj(self) -> np.ndarray:
        """Rows 1..H -- the decoder's waypoint tokens (t = dt .. H*dt)."""
        return self.q[1:]

    def yaw_state_after(self, k: int = 1) -> tuple[float, float]:
        """Planned ``(yaw, yaw_rate)`` at waypoint ``k`` of this reference.

        This is the boundary condition the NEXT tick should be seeded with. Do
        not use the measured yaw rate there: the loop
        measured-rate -> ``yaw_rate_start`` -> FOH command -> measured-rate is
        positive feedback and it diverges (see the module docstring).
        """
        if self.yaw_ref is None:
            raise ValueError("this WarmStart carries no yaw reference")
        ref = np.asarray(self.yaw_ref, float)
        k = int(np.clip(k, 0, ref.shape[0] - 1))
        nxt = min(k + 1, ref.shape[0] - 1)
        rate = (ref[nxt] - ref[k]) / self.dt if nxt > k else 0.0
        return float(ref[k]), float(rate)


def warmstart_trajectory(
    q_start: np.ndarray,
    qd_start: np.ndarray,
    *,
    horizon: int,
    goals: dict[str, tuple[str, float] | None],
    library: GoalPoseLibrary,
    aim_targets: dict[str, np.ndarray] | None = None,
    aim_tth: dict[str, float] | None = None,
    aim_futures: dict[str, object] | None = None,
    dt: float = CONTROL_DT,
    joint_names=G1_UPPER_BODY_JOINTS,
    waist_cfg: WaistAimConfig | None = None,
    waist_settle_wp: int = 3,
    waist_move_s: float = 0.2,
    clamp: bool = True,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    yaw_state: tuple[float, float] | None = None,
    yaw_rate_limit: float | None = None,
    yaw_slew_eta: float = ETA_VELOCITY,
    waist_setpoint: np.ndarray | None = None,
    waist_upright_ease_wp: int | None = None,
    slew_limit_eta: float | None = None,
    idle_home=None,
    idle_library=None,
    idle_cylinder=None,
    fk=None,
    idle_settle_wp: int = 10,
    terminal_arm: dict[str, np.ndarray | None] | None = None,
) -> WarmStart:
    """Velocity-matched accelerating strike seeds + settling waist/idle ramps.

    ``goals`` maps a hand to ``(drum_name, time_to_hit_seconds)`` or ``None``
    (idle). ``aim_targets`` are the per-hand tip targets the waist aims at
    (env/pelvis XY is all that matters); omit to hold the current yaw.

    ``aim_tth`` / ``aim_futures`` are :func:`~amortized_opt.waist_aim.aim_yaw`'s
    future-aware inputs -- this hand's time to its current goal, and its
    upcoming ``(position, time_to_hit)`` hits beyond it.  The SEED must ramp
    toward the same aim the context token carries or the model's residual
    parameterization starts off the ramp, so every caller that has a snapshot
    threads both (``mpc_gen``, ``phase2_data``, ``warmstart_data``,
    ``warmstart_apex``).  A caller with no schedule -- the eval harness, whose
    case IS a single hit -- passes neither and gets the current-goal law.
    ``aim_futures`` alone is enough to aim a hand whose goal the horizon cap
    demoted, which is why the ``aim_targets``-empty case still runs the law.

    Active arms use :func:`~amortized_opt.strike_seed.accelerating_path_numpy`:
    match measured initial velocity, ramp acceleration over the first 10%
    of the window, then hold acceleration through the strike. From rest the
    path is straight; off-path initial velocities necessarily bend it. The
    masked post-hit tail tapers acceleration to zero and coasts, subject to
    the existing joint-box clamp. Idle arms still settle at rest.

    Waist-yaw safety (2026-08-10, phase-2 prerequisite fix A)
    --------------------------------------------------------
    The yaw ramp used to take its ``yaw_rate_start`` boundary condition from the
    *measured* ``qd_start``.  In a receding-horizon loop that closes a positive
    feedback path -- measured rate -> quintic initial slope -> a first waypoint
    ``yaw_start + rate * dt`` far away -> a huge velocity feed-forward -> a
    larger measured rate.  Measured with the phase-1 stand-in planner it took
    ``waist_yaw`` from 1.5 to 2.8 rad in ~90 ticks, out of the joint box, and
    then produced a 5.2 rad command step (262 rad/s of feed-forward).

    Three independent guards now apply, all cheap and all defensive:

    * ``yaw_state=(yaw, rate)`` overrides the measured boundary condition with
      the **planned** one.  ``MPCGenerator`` carries
      :meth:`WarmStart.yaw_state_after` forward, which breaks the feedback path
      at the source.
    * ``yaw_rate_limit`` (default ``yaw_slew_eta`` x the joint's speed rating)
      clamps whatever rate is used, so even a measured-rate caller cannot ask
      for a slope the actuator could not produce.
    * the emitted yaw column is clipped into the joint box and slew-limited to
      ``yaw_slew_eta * rating * dt`` per waypoint, so **no** command step can
      exceed the velocity limit regardless of the ramp's shape.

    ``waist_setpoint`` freezes pitch and roll at a *planned* value instead of at
    the measured one.  Freezing at the measured value is the same zero-stiffness
    loop in disguise: the command is always "wherever you are now", so any
    velocity the arms couple into the waist integrates freely.  Measured over
    300 ticks it walked ``waist_pitch`` 0.79 rad outside a +/-0.52 rad box.

    ``waist_upright_ease_wp`` (None by default, so every existing target and
    seed is unchanged bit for bit) replaces the pitch/roll HOLD with a
    recovery: a quintic smoothstep from the measured pose to UPRIGHT (0 rad,
    or ``waist_setpoint`` when given) over that many waypoints, arriving at
    rest and holding.  Holding the measured lean is what taught phase 1 the
    "stay leaned" prior the upright well then has to fight -- the starts
    sample pitch/roll in a posture box, so every imitation target said "keep
    it".  A raw zero setpoint is NOT the fix: it steps up to 0.52 rad inside
    one control period and the acceleration lands on ``torque_peak`` (the same
    reason ``ModelConfig.frozen_ramp_wp`` ramps).  The ease is stretched like
    the idle-arm park when the requested window would exceed the joint's speed
    rating (quintic peak rate is 1.875x its mean).

    The idle arm (idle safety cylinder)
    -----------------------------------
    ``idle_library`` + ``idle_cylinder`` (+ ``fk``) park a hand with
    ``goals[hand] is None`` inside the torso-frame idle cylinder (see
    :mod:`~amortized_opt.idle_cylinder`):

    * tip already inside -> hold the measured 7 arm DoFs for the horizon;
    * tip outside -> nearest library pose (shoulder_weight=10) ->
      :func:`~amortized_opt.idle_cylinder.nudge_into_cylinder` -> arm-only
      quintic over ``L = clamp(speed_need, idle_settle_wp, H)``, then hold.

    ``idle_home`` (an object with ``arm_pose(hand)``) remains as a legacy
    fallback when no cylinder library is supplied.  ``None`` for both keeps
    hold-in-place.

    ``slew_limit_eta`` (off by default) applies the same per-waypoint speed
    bound to **every** joint. The seed is velocity-unlimited, so an
    aggressive hit spacing makes it command several times the rating.  Turn it
    on wherever the seed is *executed* rather than merely used as an init.
    """
    names = tuple(joint_names)
    n = len(names)
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon must be >= 1, got {H}")
    q0 = np.asarray(q_start, float).reshape(n)
    v0 = np.asarray(qd_start, float).reshape(n)
    lo, hi = q_bounds if q_bounds is not None else urdf_position_limits(names)
    v_lim = isaac_velocity_limits(names)

    out = np.repeat(q0[None, :], H + 1, axis=0)
    hit_index: dict[str, int] = {}
    peak: dict[str, float] = {}
    used: dict[str, str] = {}

    # ---- waist: quintic yaw ramp, pitch/roll frozen at the current value ----
    yaw_i = names.index("waist_yaw_joint")
    yaw_lo, yaw_hi = float(lo[yaw_i]), float(hi[yaw_i])
    if yaw_state is None:
        yaw_now = float(q0[yaw_i])
        yaw_rate_now = float(v0[yaw_i])
    else:
        yaw_now, yaw_rate_now = float(yaw_state[0]), float(yaw_state[1])
    # (1) the BC itself must be representable.
    yaw_now = float(np.clip(yaw_now, yaw_lo, yaw_hi))
    rate_cap = (
        float(yaw_slew_eta) * float(v_lim[yaw_i])
        if yaw_rate_limit is None
        else float(yaw_rate_limit)
    )
    yaw_rate_now = float(np.clip(yaw_rate_now, -rate_cap, rate_cap))
    if aim_targets or aim_futures:
        yaw_target = float(
            aim_yaw(aim_targets or {}, cfg=waist_cfg, yaw_current=yaw_now,
                    goal_tth=aim_tth, futures=aim_futures)
        )
    else:
        yaw_target = yaw_now
    yaw_target = float(np.clip(yaw_target, yaw_lo, yaw_hi))
    waist_names = [nm for nm in names if nm.startswith("waist_")]
    if waist_setpoint is not None:
        setpoint = np.asarray(waist_setpoint, float).reshape(len(waist_names)).copy()
    elif waist_upright_ease_wp is not None:
        # ease-to-upright: the hold target is 0 rad, not the measured lean
        setpoint = np.zeros(len(waist_names), dtype=float)
    else:
        setpoint = np.array([q0[names.index(nm)] for nm in waist_names], dtype=float)
    setpoint = np.clip(
        setpoint,
        np.array([lo[names.index(nm)] for nm in waist_names]),
        np.array([hi[names.index(nm)] for nm in waist_names]),
    )
    waist_ref = waist_reference_block(
        waist_names,
        yaw_start=yaw_now,
        yaw_target=yaw_target,
        n_waypoints=H + 1,
        setpoint=setpoint,
        dt=dt,
        settle_wp=waist_settle_wp,
        yaw_rate_start=yaw_rate_now,
        max_move_s=waist_move_s,
    )
    # (2) clip into the box and (3) slew-limit, so the FOH command derived from
    # consecutive rows can never exceed the joint's speed rating.
    jy = waist_names.index("waist_yaw_joint")
    waist_ref[:, jy] = _slew_limited(
        np.clip(waist_ref[:, jy], yaw_lo, yaw_hi),
        max_step=float(yaw_slew_eta) * float(v_lim[yaw_i]) * float(dt),
    )
    for j, nm in enumerate(waist_names):
        out[:, names.index(nm)] = waist_ref[:, j]
    yaw_ref = waist_ref[:, jy].copy()

    if waist_upright_ease_wp is not None:
        # ---- pitch/roll: ease from the measured STATE to the hold target ----
        # A VELOCITY-MATCHED boundary quintic: p(0) = measured pose, p'(0) =
        # measured rate (clipped to the speed rating), p''(0) = 0, arriving at
        # the target at rest.  ``v_from = 0`` reduces exactly to the quintic
        # smoothstep (the ``frozen_ramp_wp`` shape), so a rest start is
        # bit-identical to the pre-velocity-matched ease.  The match is what
        # makes a per-tick rebuild a CONTINUATION rather than a re-anchor: a
        # seed rebuilt mid-recovery keeps the recovery's current rate instead
        # of braking to zero slope and crawling at ~s(1/k) per tick.  The
        # window stretches under the rest-to-rest 1.875x rule, then one
        # correction round re-stretches if the velocity term pushed a step
        # over the rating.  Row 0 evaluates to the measured value exactly.
        for j, nm in enumerate(waist_names):
            if nm == "waist_yaw_joint":
                continue
            ci = names.index(nm)
            step_lim = max(ETA_VELOCITY * float(v_lim[ci]) * float(dt), 1e-9)
            q_from, q_to = float(q0[ci]), float(setpoint[j])
            v_from = float(np.clip(v0[ci], -step_lim / float(dt), step_lim / float(dt)))
            need = int(np.ceil(1.875 * abs(q_to - q_from) / step_lim))
            k_ease = int(np.clip(max(int(waist_upright_ease_wp), need), 1, H))
            col = _boundary_quintic(q_from, v_from, q_to, k_ease, H, float(dt))
            peak_step = float(np.abs(np.diff(col)).max())
            if peak_step > step_lim:
                k_ease = int(np.clip(
                    int(np.ceil(k_ease * peak_step / step_lim)), 1, H))
                col = _boundary_quintic(q_from, v_from, q_to, k_ease, H, float(dt))
            out[:, ci] = col

    # ---- arms: independent velocity-matched strikes to the library pose ----
    for hand in ("left", "right"):
        idx = arm_slice(hand, names)
        goal = goals.get(hand)
        if goal is None:
            hit_index[hand] = -1
            lib = idle_library
            cyl = idle_cylinder
            if cyl is None and lib is not None and hasattr(lib, "cylinder"):
                cyl = lib.cylinder
            if lib is not None and fk is not None and cyl is not None:
                # Lazy import: idle_cylinder imports arm_slice from this module.
                import torch
                from .idle_cylinder import nudge_into_cylinder, tip_in_torso, tip_inside

                q_t = torch.as_tensor(
                    q0[None], device=fk.device, dtype=getattr(fk, "dtype", None)
                )
                tip = tip_in_torso(fk, q_t, hand).squeeze(0).detach().cpu().numpy()
                if tip_inside(cyl, tip):
                    out[:, idx] = q0[idx][None, :]
                    used[hand] = "idle_hold"
                    continue
                _k, q_near = lib.nearest_arm_pose(hand, q0[idx], shoulder_weight=10.0)
                q_full = q0.copy()
                q_full[idx] = np.asarray(q_near, float)
                q_nudge = nudge_into_cylinder(
                    fk, q_full, hand, cyl,
                    shoulder_weight=10.0,
                    joint_names=names,
                    q_bounds=(lo, hi) if clamp else None,
                )
                Q = np.stack([q0[idx], np.asarray(q_nudge, float)[idx]])
                span = np.abs(Q[1] - Q[0])
                need = np.ceil(
                    1.875 * span
                    / np.maximum(ETA_VELOCITY * v_lim[idx] * float(dt), 1e-9)
                ).max()
                k_set = int(np.clip(max(int(idle_settle_wp), int(need)), 1, H))
                X, rate = retime_joint_path(
                    Q, k_set + 1, float(dt), v_start=v0[idx], v_limit=v_lim[idx],
                    q_lower=lo[idx] if clamp else None,
                    q_upper=hi[idx] if clamp else None,
                )
                peak[hand] = float(rate)
                used[hand] = "idle_cyl"
                out[: k_set + 1, idx] = X
                out[k_set + 1 :, idx] = X[-1][None, :]
                continue
            if idle_home is None:
                out[:, idx] = q0[idx][None, :]
                continue
            # Legacy single-home park.
            Q = np.stack([q0[idx], np.asarray(idle_home.arm_pose(hand), float)])
            span = np.abs(Q[1] - Q[0])
            need = np.ceil(
                1.875 * span / np.maximum(ETA_VELOCITY * v_lim[idx] * float(dt), 1e-9)
            ).max()
            k_set = int(np.clip(max(int(idle_settle_wp), int(need)), 1, H))
            X, rate = retime_joint_path(
                Q, k_set + 1, float(dt), v_start=v0[idx], v_limit=v_lim[idx],
                q_lower=lo[idx] if clamp else None,
                q_upper=hi[idx] if clamp else None,
            )
            peak[hand] = float(rate)
            used[hand] = "home"
            out[: k_set + 1, idx] = X
            out[k_set + 1 :, idx] = X[-1][None, :]
            continue
        drum, tth = goal
        k_hit = int(np.clip(round(float(tth) / float(dt)), 1, H))
        hit_index[hand] = k_hit
        # ``terminal_arm`` overrides the library pick -- the caller has already
        # re-aimed it at the drum that is actually there (``ik.correct_tip``),
        # which the library, keyed on (hand, drum, waist_yaw), cannot know about.
        q_term = None if terminal_arm is None else terminal_arm.get(hand)
        if q_term is None:
            entry = library.select(hand, drum, yaw_target)
            used[hand] = f"{drum}@{entry.waist_yaw:+.3f}"
            q_term = entry.arm_q
        else:
            q_term = np.asarray(q_term, float).reshape(len(idx))
            used[hand] = f"{drum}@corrected"
        X = accelerating_path_numpy(q0[idx], v0[idx], q_term, k_hit, H, float(dt))
        peak[hand] = float(np.linalg.norm(np.diff(X, axis=0) / float(dt), axis=1).max())
        out[:, idx] = X

    if clamp:
        out = np.clip(out, lo[None, :], hi[None, :])
    out[0] = q0  # the start is pinned, never clamped away
    peak_raw = np.abs(np.diff(out, axis=0) / float(dt)).max(axis=0)
    if slew_limit_eta is not None and slew_limit_eta > 0.0:
        max_step = float(slew_limit_eta) * v_lim * float(dt)
        for j in range(n):
            out[:, j] = _slew_limited(out[:, j], float(max_step[j]))
        yaw_ref = out[:, yaw_i].copy()
    return WarmStart(q=out, dt=float(dt), hit_index=hit_index, yaw_target=yaw_target,
                     peak_rate=peak, library=used, yaw_ref=yaw_ref,
                     peak_rate_raw=peak_raw)


# ------------------------------------------------------------------- noise --
def correlated_noise(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    *,
    sigma: float = 1.0,
    tau_steps: float = 5.0,
    axis: int = -2,
) -> np.ndarray:
    """Stationary Ornstein-Uhlenbeck noise along ``axis``.

    ``x_k = a x_{k-1} + sqrt(1 - a^2) * eps_k`` with ``a = exp(-1/tau_steps)``,
    started from the stationary distribution, so the marginal is exactly
    ``N(0, sigma^2)`` at every index and the lag-1 autocorrelation is ``a``.
    ``tau_steps = 5`` gives ``a = 0.819`` -- adjacent waypoints move together,
    which is what makes the perturbation look like a *different plan* rather
    than like measurement noise.
    """
    a = float(np.exp(-1.0 / max(float(tau_steps), 1e-6)))
    shape = tuple(int(s) for s in shape)
    ax = axis if axis >= 0 else len(shape) + axis
    K = shape[ax]
    eps = rng.standard_normal(shape)
    x = np.empty(shape, dtype=float)
    sl = [slice(None)] * len(shape)
    sl[ax] = 0
    x[tuple(sl)] = eps[tuple(sl)]
    scale = float(np.sqrt(max(1.0 - a * a, 0.0)))
    for k in range(1, K):
        cur, prev = list(sl), list(sl)
        cur[ax], prev[ax] = k, k - 1
        x[tuple(cur)] = a * x[tuple(prev)] + scale * eps[tuple(cur)]
    return float(sigma) * x


@dataclass
class InitNoise:
    """Perturbation of the initialization trajectory fed to the model.

    Two components, per the plan ("likely Gaussian and temporally correlated
    noise on the Quintic straight line"):

    * ``sigma_corr`` -- OU noise with correlation length ``tau_steps``
      waypoints. This is the one that matters: it displaces the *plan*, so the
      model has to re-solve rather than denoise.
    * ``sigma_white`` -- i.i.d. per waypoint, a small roughness floor.

    Both are in radians and are scaled per joint by ``joint_scale`` (default
    uniform). ``pin_start`` leaves row 0 of the block untouched; it defaults to
    False because the block this is normally applied to is
    ``WarmStart.init_traj`` -- rows 1..H, which do NOT include the measured
    start pose (that reaches the decoder as the state token, unnoised). Set it
    True when noising a full ``(H+1, 17)`` trajectory whose row 0 IS the start.
    ``taper`` linearly fades the noise to zero at each hand's terminal waypoint
    so the seed's terminal pose stays a plausible strike.
    """

    sigma_corr: float = 0.05
    sigma_white: float = 0.01
    tau_steps: float = 5.0
    pin_start: bool = False
    taper: bool = False
    joint_scale: np.ndarray | None = None

    def sample(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        *,
        hit_index: dict[str, int] | None = None,
        joint_names=G1_UPPER_BODY_JOINTS,
    ) -> np.ndarray:
        """Noise shaped like a ``(..., K, n)`` trajectory block."""
        noise = np.zeros(shape, dtype=float)
        if self.sigma_corr > 0.0:
            noise += correlated_noise(shape, rng, sigma=self.sigma_corr,
                                      tau_steps=self.tau_steps, axis=-2)
        if self.sigma_white > 0.0:
            noise += self.sigma_white * rng.standard_normal(shape)
        if self.joint_scale is not None:
            noise = noise * np.asarray(self.joint_scale, float).reshape(1, -1)
        if self.pin_start:
            noise[..., 0, :] = 0.0
        if self.taper and hit_index:
            K = shape[-2]
            k = np.arange(K, dtype=float)
            names = tuple(joint_names)
            for hand, k_hit in hit_index.items():
                if k_hit is None or k_hit < 0:
                    continue
                w = np.clip(1.0 - k / max(float(k_hit), 1.0), 0.0, 1.0)
                cols = [i for i, nm in enumerate(names) if nm.startswith(f"{hand}_")]
                noise[..., :, cols] *= w[:, None]
        return noise

    def apply(
        self,
        traj: np.ndarray,
        rng: np.random.Generator,
        *,
        hit_index: dict[str, int] | None = None,
        q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        joint_names=G1_UPPER_BODY_JOINTS,
    ) -> np.ndarray:
        """Noised copy of ``traj``, clipped back into the joint box."""
        out = np.asarray(traj, float) + self.sample(
            traj.shape, rng, hit_index=hit_index, joint_names=joint_names
        )
        if q_bounds is not None:
            lo, hi = q_bounds
            out = np.clip(out, lo, hi)
        return out
