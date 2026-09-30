"""Phase-1 warm start, APEX variant: start -> apex-above-the-drum -> terminal.

This module is a **strictly additive** experiment alongside
:mod:`amortized_opt.warmstart` / :mod:`amortized_opt.warmstart_data`.  Nothing
here edits those; everything is either a copy of the minimal driver or a call
into their public functions.

The idea
--------
Phase 1's original imitation target was a *straight line* in joint space from
the start pose to a library terminal pose, walked by a quintic timing law. It is
obstacle-unaware by construction: it drives the stick through whatever sits
between the two poses, and it arrives at the head from an arbitrary direction.

The repo's trajopt SQP does not seed that way.  Its seed
(``trajopt_hpipm/sqp.py``, ``seed_mode="apex"`` + ``seed_apex_first``) is a
three-knot path::

    q_start  ->  q_apex  ->  q_terminal

where ``q_apex`` is a pose whose stick tip lies inside a **vertical cylinder**
centred ``seed_apex_height = 0.30 m`` above the strike position
(``seed_apex_radius = 0.15 m``, ``seed_apex_half_height = 0.10 m``), chosen as
close as possible **in joint space to the start pose alone** (apex-first
ordering), and the terminal strike pose is then chosen **nearest that apex**.
The two low-acceleration legs through those knots are already descending when
the path reaches the head, which is the shape a strike has.

This module mirrors that convention standalone, so phase 1 can be A/B'd on
"direct low-acceleration" vs "two-segment apex" imitation targets.

What one sample looks like
--------------------------
The input initialization uses the shared velocity-matched accelerating seed;
the imitation target applies the same law to both apex legs:

* target  = ``accelerating(q_start -> q_apex)`` followed by
            ``accelerating(q_apex -> q_terminal)``
* init    = ``accelerating_path_numpy(q_start, v_start, q_terminal, ...)``
            + OU/white noise, to the *same* terminal pose

The model learns to refine the standard strike initializer into the apex
trajectory. Both share the terminal pose. Without an apex, the target also
uses the standard strike initializer.

Deviations from the repo's apex machinery (deliberate, measured)
----------------------------------------------------------------
1. **The apex IK is a batched Levenberg-Marquardt in torch**, not the repo's
   ``collision_free_ik`` pre-seed + ``apex_pose`` Gauss-Newton with a
   collision-checked line search.  The objective is identical --
   ``min ||q - q_ref||^2_W + w_cyl (s_r^2 + s_z^2)`` with ``s_r``/``s_z`` the
   radial/axial overhang out of the cylinder -- and the waist is frozen exactly
   as the repo freezes it (``free[waist_idx] = 0``).  Collision is handled by
   *multi-seed selection* (``n_seeds`` random restarts, prefer the
   collision-free ones) rather than by rejecting every intermediate iterate.
   Rationale: 450 k scenarios generated in a fork pool cannot afford a
   per-iterate SDF call, and the seed only has to be a *good* apex, not a
   certified one -- the whole point of phase 2 is to price what is left.
2. **The terminal comes from the pose library, not from a fresh strike IK.**
   Phase 1 has no strike IK; ``GoalPoseLibrary`` is the terminal source in the
   straight-line variant too.  "Nearest the apex" is therefore implemented as
   ``argmin ||arm_q - apex_arm||`` over the library entries that satisfy the
   terminal condition, optionally restricted to a waist-yaw window around the
   aim yaw (``ApexConfig.terminal_yaw_window``) -- without that window a
   far-yaw entry can win on joint distance while being an IK solution for a
   torso orientation the waist ramp will never reach, which moves the tip.
3. **Uniform vs lever-weighted joint metric** is a config flag
   (``lever_weighting``, default on, matching ``seed_apex_lever_weighting``).
   The weights are ``r_j^2 = ||J_j(q_ready)||^2`` evaluated once at the ready
   posture and cached, exactly as ``G1UpperBody.tip_lever_weight`` does.
4. **No axial-twist term** (``seed_w_axial_twist``).  It is a posture
   preference the phase-2 cost already carries.

Everything else -- the waist-aim ramp with its three safety guards, the start
sampler, the kit randomization, the goal snapshots, the impossible-transition
filter, the auxiliary-label pass, the horizon buckets -- is the phase-1
straight-line pipeline, called unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, ready_joint_pos
from .fk import ChainFK
from .idle_home import resolve_idle_home
from .ik import adjust_arm_terminal, tip_jacobian, tip_position
from .limits import isaac_velocity_limits, urdf_position_limits
from .rhythm import (
    HANDS,
    DrumKit,
    RhythmClip,
    aim_inputs_from_snapshot,
    build_context_inputs,
    default_rcc_dir,
    goal_snapshot,
    load_rcc_pool,
)
from .sdf import DrumSceneSpec, SdfPenalty
from .spline import joint_path_spline
from .strike_seed import (
    accelerating_apex_path_numpy,
    accelerating_apex_path_torch,
    accelerating_path_numpy,
)
from .warmstart import GoalPoseLibrary, LibraryEntry, arm_slice, warmstart_trajectory
from .warmstart_data import (
    CONTEXT_FIELDS,
    NoiseConfig,
    ScenarioConfig,
    WarmstartPool,
    _BOOL_FIELDS,
    _LONG_FIELDS,
    bucket_for,
)

__all__ = [
    "ApexConfig",
    "ApexSolver",
    "ApexWarmStart",
    "ApexWarmstartPool",
    "apex_warmstart_trajectory",
    "generate_apex_pool",
    "generate_apex_scenarios",
    "select_terminal",
]


# ================================================================= config ====
@dataclass
class ApexConfig:
    """The apex cylinder and how it is solved.

    Defaults mirror ``TrajOptConfig``'s ``seed_apex_*`` fields
    (``problem.py`` lines 2101-2159).
    """

    #: metres above the strike position, to the cylinder axis centre.
    height: float = 0.30
    #: cylinder radius about the vertical axis (m).
    radius: float = 0.15
    #: half-height of the cylinder band (m).
    half_height: float = 0.10
    #: overhang penalty vs the joint-proximity term.
    w_cyl: float = 1000.0
    #: charge each joint for the tip motion it causes (``tip_lever_weight``).
    lever_weighting: bool = True

    #: Levenberg-Marquardt iterations.  60 (not 30) because the cylinder-miss
    #: p90 falls 18.5 mm -> 6.0 mm between the two for 1.8x the cost, and the
    #: whole hypothesis under test is "does the path visit the cylinder".
    iters: int = 60
    lm_lambda0: float = 1.0e-2
    lm_lambda_min: float = 1.0e-8
    lm_lambda_max: float = 1.0e6

    #: random restarts (seed 0 is always the unperturbed reference pose).
    n_seeds: int = 3
    #: sd of the joint-space perturbation of the extra seeds (rad).
    seed_noise: float = 0.20
    #: a solved apex counts as collision-free when ``min_sdf >= this``.
    min_clearance: float = 0.0
    #: penalty added to the seed-selection score for a colliding pose.
    collision_score_penalty: float = 1.0e3

    #: ``"nearest_apex"`` (the repo convention) or ``"nearest_yaw"`` (the
    #: straight-line variant's ``GoalPoseLibrary.select``).
    terminal_mode: str = "nearest_apex"
    #: Only library entries within this many radians of the aim yaw may be
    #: chosen by ``nearest_apex``.  ``None`` removes the restriction.
    terminal_yaw_window: float | None = 0.40

    #: Fall back to the straight line when the cylinder was missed by more than
    #: this (m).  ``None`` keeps every apex, however poor.
    max_outside: float | None = 0.25

    # ---- re-aiming the library terminal (2026-08-13) ----------------------
    #: Damped-least-squares steps that move the chosen terminal's ARM joints so
    #: its stick tip lands on the aim point that is ACTUALLY there.
    #:
    #: ``GoalPoseLibrary`` is keyed on ``(hand, drum, waist_yaw)`` and stores a
    #: joint pose, so it is blind to both ``KitRandomization`` and the fitted
    #: stick.  Measured over 60 randomized kits at the shipped jitter, the
    #: library terminal misses the head centre by a median 12.9 cm; 3 steps
    #: bring that to 0.06 cm, and 10 cm of per-drum jitter then costs almost
    #: nothing over 4 cm (0.11 vs 0.06 cm).  Without it the jitter is
    #: UNLEARNABLE in phase 1 -- the context carries the randomized geometry
    #: while the target ignores it.
    #:
    #: 0 disables, reproducing the pre-2026-08-13 target exactly.
    correct_terminal_steps: int = 3
    #: DLS damping, metres of tip error per radian.  0.05 converges in 3 steps
    #: without overshooting the two cross-body reaches, whose targets are
    #: genuinely outside the workspace at some jitters.
    correct_terminal_damping: float = 0.05

    def __post_init__(self) -> None:
        if self.terminal_mode not in ("nearest_apex", "nearest_yaw"):
            raise ValueError(f"terminal_mode must be nearest_apex|nearest_yaw, "
                             f"got {self.terminal_mode!r}")


# ============================================================= apex solver ===
class ApexSolver:
    """Batched apex-pose IK: tip inside a vertical cylinder, near a reference.

    ::

        solver = ApexSolver(device="cpu")
        out = solver.solve("left", q_ref, centers_env, head_centers=heads)
        out["q"]        # (B, 17) apex poses (waist columns == q_ref's)
        out["outside"]  # (B,) metres the tip still sits outside the cylinder
        out["min_sdf"]  # (B,) clearance of the returned pose
    """

    def __init__(
        self,
        *,
        cfg: ApexConfig | None = None,
        fk: ChainFK | None = None,
        sdf: SdfPenalty | None = None,
        joint_names=G1_UPPER_BODY_JOINTS,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        check_collision: bool = True,
    ):
        self.cfg = cfg or ApexConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.names = tuple(joint_names)
        self.n = len(self.names)
        self.fk = fk or ChainFK(device=self.device, dtype=dtype)
        self.check_collision = bool(check_collision)
        self.sdf = sdf
        if self.sdf is None and self.check_collision:
            self.sdf = SdfPenalty(
                DrumSceneSpec.from_layout(), fk=self.fk, device=self.device, dtype=dtype
            )
        lo, hi = urdf_position_limits(self.names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        self._arm_idx = {h: arm_slice(h, self.names) for h in HANDS}
        self._lever: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------- metric --
    def lever_weight(self, hand: str) -> torch.Tensor:
        """``r_j^2`` at the ready posture, cached (see ``tip_lever_weight``)."""
        if hand not in self._lever:
            q = torch.as_tensor(
                ready_joint_pos(self.names)[None, :], device=self.device, dtype=self.dtype
            )
            with torch.enable_grad():
                J = tip_jacobian(self.fk, q, hand)[0]  # (3, n)
            self._lever[hand] = (J * J).sum(dim=0).detach()  # (n,)
        return self._lever[hand]

    # -------------------------------------------------------------- solve --
    def solve(
        self,
        hand: str,
        q_ref: torch.Tensor,          # (B, 17) reference pose (waist already set)
        center_env: torch.Tensor,     # (B, 3) cylinder centre, ENV frame
        *,
        head_centers: torch.Tensor | None = None,  # (B, D, 3) env, randomized kit
        head_radii: torch.Tensor | None = None,    # (B, D) radii of the same kit
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        B = int(q_ref.shape[0])
        S = max(int(cfg.n_seeds), 1)
        idx = torch.as_tensor(self._arm_idx[hand], device=self.device, dtype=torch.long)
        m = int(idx.numel())

        # seed 0 is q_ref itself; the rest are jittered on the ARM joints only.
        q = q_ref.detach().to(self.dtype).unsqueeze(0).repeat(S, 1, 1)  # (S, B, n)
        if S > 1 and cfg.seed_noise > 0.0:
            rng = rng or np.random.default_rng(0)
            noise = torch.as_tensor(
                rng.normal(0.0, cfg.seed_noise, size=(S - 1, B, m)),
                device=self.device, dtype=self.dtype,
            )
            q[1:, :, idx] = q[1:, :, idx] + noise
        q = q.reshape(S * B, self.n)
        q = torch.clamp(q, self.q_lo, self.q_hi)
        ref = q_ref.detach().to(self.dtype).unsqueeze(0).repeat(S, 1, 1).reshape(S * B, self.n)
        ctr = center_env.detach().to(self.dtype).unsqueeze(0).repeat(S, 1, 1).reshape(S * B, 3)

        w = (
            self.lever_weight(hand).index_select(0, idx).clamp_min(1e-3)
            if cfg.lever_weighting
            else torch.ones(m, device=self.device, dtype=self.dtype)
        )
        sw = torch.sqrt(w).view(1, m)                 # residual scale of the prox term
        sc = float(np.sqrt(max(cfg.w_cyl, 0.0)))
        eye = torch.eye(m, device=self.device, dtype=self.dtype).unsqueeze(0)
        lam = torch.full((S * B, 1, 1), float(cfg.lm_lambda0),
                         device=self.device, dtype=self.dtype)

        def residual(qq: torch.Tensor):
            """(r_prox (N,m), r_cyl (N,2), tip (N,3)) at ``qq``."""
            tip = tip_position(self.fk, qq, hand, frame="env")
            d = tip - ctr
            rho = torch.sqrt((d[:, :2] * d[:, :2]).sum(-1) + 1e-18)
            s_r = (rho - cfg.radius).clamp_min(0.0)
            s_z = (d[:, 2].abs() - cfg.half_height).clamp_min(0.0)
            r_prox = sw * (qq.index_select(1, idx) - ref.index_select(1, idx))
            r_cyl = sc * torch.stack([s_r, s_z], dim=-1)
            return r_prox, r_cyl, d, rho

        def objective(qq: torch.Tensor) -> torch.Tensor:
            rp, rc, _d, _rho = residual(qq)
            return (rp * rp).sum(-1) + (rc * rc).sum(-1)

        f = objective(q)
        for _ in range(int(cfg.iters)):
            with torch.enable_grad():
                J = tip_jacobian(self.fk, q, hand).index_select(2, idx)  # (N, 3, m)
            J = J.detach()
            rp, rc, d, rho = residual(q)
            act_r = (rc[:, 0] > 0.0).to(self.dtype).unsqueeze(-1)
            act_z = (rc[:, 1] > 0.0).to(self.dtype).unsqueeze(-1)
            u_xy = d[:, :2] / rho.unsqueeze(-1).clamp_min(1e-12)
            g_r = sc * act_r * (u_xy.unsqueeze(1) @ J[:, :2, :]).squeeze(1)   # (N, m)
            g_z = sc * act_z * torch.sign(d[:, 2]).unsqueeze(-1) * J[:, 2, :]
            Jr = torch.cat([torch.diag_embed(sw.expand(J.shape[0], m)),
                            g_r.unsqueeze(1), g_z.unsqueeze(1)], dim=1)       # (N, m+2, m)
            r = torch.cat([rp, rc], dim=-1)                                   # (N, m+2)
            JtJ = Jr.transpose(1, 2) @ Jr
            g = (Jr.transpose(1, 2) @ r.unsqueeze(-1))                        # (N, m, 1)
            delta = torch.linalg.solve(JtJ + lam * eye, -g).squeeze(-1)
            q_new = torch.clamp(q.index_copy(1, idx, q.index_select(1, idx) + delta),
                                self.q_lo, self.q_hi)
            f_new = objective(q_new)
            better = (f_new < f).unsqueeze(-1)
            q = torch.where(better, q_new, q)
            f = torch.where(better.squeeze(-1), f_new, f)
            lam = torch.where(
                better.unsqueeze(-1),
                (lam * 0.5).clamp_min(cfg.lm_lambda_min),
                (lam * 4.0).clamp_max(cfg.lm_lambda_max),
            )

        # --- cylinder overhang of the converged poses -------------------------
        tip = tip_position(self.fk, q, hand, frame="env").detach()
        rp, rc, d, rho = residual(q)
        s_r = rc[:, 0] / max(sc, 1e-12)
        s_z = rc[:, 1] / max(sc, 1e-12)
        outside = torch.sqrt(s_r * s_r + s_z * s_z)

        # --- collision screen -------------------------------------------------
        if self.check_collision and self.sdf is not None:
            hc = None
            if head_centers is not None:
                hc = head_centers.detach().to(self.dtype).unsqueeze(0)
                hc = hc.repeat(S, 1, 1, 1).reshape(S * B, *hc.shape[2:])
            # Radii follow the centres seed-for-seed: a screen that knows where
            # the drums moved but not how big they are is still the fix-B
            # disagreement.
            hr = None
            if head_radii is not None:
                hr = head_radii.detach().to(self.dtype).unsqueeze(0)
                hr = hr.repeat(S, 1, 1).reshape(S * B, *hr.shape[2:])
            with torch.no_grad():
                _pen, det = self.sdf.forward(
                    q, drum_head_centers=hc, drum_radii=hr, return_details=True
                )
            min_sdf = det.min_sdf.detach()
        else:
            min_sdf = torch.zeros(S * B, device=self.device, dtype=self.dtype)

        # --- pick the best seed ----------------------------------------------
        colliding = (min_sdf < cfg.min_clearance).to(self.dtype)
        score = (f + cfg.collision_score_penalty * colliding).reshape(S, B)
        best = score.argmin(dim=0).to(torch.long)                      # (B,)
        ar = torch.arange(B, device=self.device)
        pick = lambda t: t.reshape(S, B, *t.shape[1:])[best, ar]  # noqa: E731
        return {
            "q": pick(q),
            "tip": pick(tip),
            "outside": pick(outside),
            "min_sdf": pick(min_sdf),
            "objective": pick(f),
            "inside": pick(outside) <= 1e-6,
            "collision_free": pick(min_sdf) >= cfg.min_clearance,
        }


# ========================================================= terminal choice ===
def select_terminal(
    library: GoalPoseLibrary,
    hand: str,
    drum: str,
    *,
    yaw_target: float,
    apex_arm: np.ndarray | None,
    cfg: ApexConfig,
) -> LibraryEntry:
    """The library terminal pose, chosen NEAREST THE APEX (repo convention).

    Mirrors ``sqp._strike_goal_q(..., q_near=q_apex)``: the apex decides which
    IK branch the terminal lands in.  Falls back to
    :meth:`GoalPoseLibrary.select` (nearest in waist yaw) when there is no apex
    or ``terminal_mode == "nearest_yaw"``.
    """
    if apex_arm is None or cfg.terminal_mode == "nearest_yaw":
        return library.select(hand, drum, yaw_target)
    rows = library.entries(hand, drum)
    ok = [
        e
        for e in rows
        if not (e.tip_err == e.tip_err)
        or (e.tip_err <= library.max_tip_err and e.wrist_headroom >= library.min_wrist_headroom)
    ]
    pool = ok or rows
    if cfg.terminal_yaw_window is not None:
        near = [e for e in pool if abs(e.waist_yaw - float(yaw_target)) <= cfg.terminal_yaw_window]
        pool = near or pool
    a = np.asarray(apex_arm, float)
    return min(pool, key=lambda e: float(np.linalg.norm(e.arm_q - a)))


# ============================================================== the target ===
@dataclass
class ApexWarmStart:
    """The apex target and its straight-line companion init.

    ``target`` and ``init`` are both ``(H, 17)`` -- rows 1..H of the command
    trajectory, i.e. the decoder's waypoint tokens, exactly like
    :attr:`~amortized_opt.warmstart.WarmStart.init_traj`.
    """

    target: np.ndarray
    init: np.ndarray
    q0: np.ndarray
    dt: float
    hit_index: dict[str, int]
    yaw_target: float
    apex_wp: dict[str, int] = field(default_factory=dict)
    apex_arm: dict[str, np.ndarray] = field(default_factory=dict)
    terminal_arm: dict[str, np.ndarray] = field(default_factory=dict)
    peak_rate_raw: np.ndarray | None = None
    peak_rate_init: np.ndarray | None = None
    library: dict[str, str] = field(default_factory=dict)

    @property
    def q(self) -> np.ndarray:
        """``(H+1, 17)`` apex target with the start pose prepended."""
        return np.concatenate([self.q0[None, :], self.target], axis=0)

    @property
    def q_init(self) -> np.ndarray:
        return np.concatenate([self.q0[None, :], self.init], axis=0)


def apex_warmstart_trajectory(
    q_start: np.ndarray,
    qd_start: np.ndarray,
    *,
    horizon: int,
    goals: dict[str, tuple[str, float] | None],
    library: GoalPoseLibrary,
    apex_arm: dict[str, np.ndarray | None] | None = None,
    terminal_arm: dict[str, np.ndarray] | None = None,
    aim_targets: dict[str, np.ndarray] | None = None,
    dt: float = CONTROL_DT,
    joint_names=G1_UPPER_BODY_JOINTS,
    cfg: ApexConfig | None = None,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    **warmstart_kwargs,
) -> ApexWarmStart:
    """Two-segment low-acceleration apex target and matching direct init.

    The **waist** column and ``yaw_target`` come from
    :func:`~amortized_opt.warmstart.warmstart_trajectory` called with both hands
    idle, so the yaw ramp (and its three safety guards) is bit-for-bit the one
    the straight-line phase-1 pool uses.  Only the arm columns are rebuilt here.

    ``apex_arm[hand]`` is the pre-solved apex arm pose (7,) or ``None`` for the
    straight line; ``terminal_arm[hand]`` overrides the library pick.
    """
    cfg = cfg or ApexConfig()
    names = tuple(joint_names)
    n = len(names)
    H = int(horizon)
    q0 = np.asarray(q_start, float).reshape(n)
    v0 = np.asarray(qd_start, float).reshape(n)
    lo, hi = q_bounds if q_bounds is not None else urdf_position_limits(names)
    v_lim = isaac_velocity_limits(names)
    apex_arm = dict(apex_arm or {})
    terminal_arm = dict(terminal_arm or {})

    # --- waist ramp: the straight-line builder, both hands idle ---------------
    base = warmstart_trajectory(
        q0, v0, horizon=H, goals={h: None for h in HANDS}, library=library,
        aim_targets=aim_targets, dt=dt, joint_names=names, q_bounds=(lo, hi),
        **warmstart_kwargs,
    )
    tgt = np.array(base.q, dtype=float)      # (H+1, 17), arms already holding q0
    ini = np.array(base.q, dtype=float)
    yaw_target = float(base.yaw_target)

    hit_index: dict[str, int] = {}
    apex_wp: dict[str, int] = {}
    used: dict[str, str] = {}
    term_used: dict[str, np.ndarray] = {}
    apex_used: dict[str, np.ndarray] = {}

    for hand in HANDS:
        idx = arm_slice(hand, names)
        goal = goals.get(hand)
        if goal is None:
            hit_index[hand] = -1
            continue
        drum, tth = goal
        k_hit = int(np.clip(round(float(tth) / float(dt)), 1, H))
        hit_index[hand] = k_hit
        qa = apex_arm.get(hand)
        qa = None if qa is None else np.asarray(qa, float).reshape(len(idx))
        if hand in terminal_arm and terminal_arm[hand] is not None:
            q_term = np.asarray(terminal_arm[hand], float).reshape(len(idx))
            used[hand] = f"{drum}@given"
        else:
            entry = select_terminal(library, hand, drum, yaw_target=yaw_target,
                                    apex_arm=qa, cfg=cfg)
            q_term = np.asarray(entry.arm_q, float)
            used[hand] = f"{drum}@{entry.waist_yaw:+.3f}"
        term_used[hand] = q_term

        # An apex knot needs room for an interior waypoint (repo: hit_wp >= 2).
        knots = [q0[idx]]
        if qa is not None and k_hit >= 2:
            knots.append(qa)
            apex_used[hand] = qa
        knots.append(q_term)
        Q = np.stack(knots)
        if len(knots) == 3:
            # ApexSolver was conditioned on the waist at the midpoint of this
            # strike window.  Put the arm apex at that same waypoint so the
            # full-body FK pose actually visits the solved cylinder.
            ka = int(np.clip(round(0.5 * k_hit), 1, k_hit - 1))
            X = accelerating_apex_path_numpy(
                q0[idx], v0[idx], qa, q_term, ka, k_hit, k_hit, float(dt),
            )
        else:
            X = accelerating_path_numpy(
                q0[idx], v0[idx], q_term, k_hit, k_hit, float(dt),
            )
        tgt[: k_hit + 1, idx] = X
        tgt[k_hit + 1 :, idx] = X[-1][None, :]
        if len(knots) == 3:
            apex_wp[hand] = ka

        # The shared velocity-matched strike initializer to the SAME terminal.
        Y = accelerating_path_numpy(q0[idx], v0[idx], q_term, k_hit, H, float(dt))
        ini[:, idx] = Y
        if len(knots) == 2:
            # No apex: use the standard phase-1 target as well.
            tgt[:, idx] = Y

    tgt = np.clip(tgt, lo[None, :], hi[None, :])
    ini = np.clip(ini, lo[None, :], hi[None, :])
    tgt[0] = q0
    ini[0] = q0
    return ApexWarmStart(
        target=tgt[1:], init=ini[1:], q0=q0, dt=float(dt), hit_index=hit_index,
        yaw_target=yaw_target, apex_wp=apex_wp, apex_arm=apex_used,
        terminal_arm=term_used, library=used,
        peak_rate_raw=np.abs(np.diff(tgt, axis=0) / float(dt)).max(axis=0),
        peak_rate_init=np.abs(np.diff(ini, axis=0) / float(dt)).max(axis=0),
    )


def _apex_waypoint(Q: np.ndarray, k_hit: int) -> int:
    """Waypoint index the apex knot lands on, by joint-space chord length.

    The repo's ``sqp._apex_waypoint``: ``frac = d0 / (d0 + d1)``.  Here the path
    is arc-length parameterized and *then* retimed, so this is a diagnostic (the
    spline places the knot itself); it is what the tests probe.
    """
    d0 = float(np.linalg.norm(Q[1] - Q[0]))
    d1 = float(np.linalg.norm(Q[2] - Q[1]))
    frac = d0 / (d0 + d1) if (d0 + d1) > 1e-9 else 0.5
    return int(np.clip(round(frac * k_hit), 1, max(k_hit - 1, 1)))


def path_derivative_jump(Q: np.ndarray, order: int = 1, eps: float = 1e-5) -> float:
    """|d^order q/du^order| discontinuity of the path spline AT the apex knot.

    :func:`~amortized_opt.spline.joint_path_spline` fits
    ``make_interp_spline(u, Q, k=min(len(u)-1, 3))``, so a three-knot apex path
    is ONE quadratic polynomial piece and every derivative is continuous by
    construction.  This measures it rather than asserting it, so a future change
    to the knot count (which would introduce real interior knots) shows up.
    """
    path, _length = joint_path_spline(Q)
    if int(order) > int(path.k):
        return 0.0  # a knot was coincident and got dropped: one linear piece
    u_apex = float(np.linalg.norm(Q[1] - Q[0]))
    d = path.derivative(int(order))
    lhs = np.asarray(d(max(u_apex - eps, 0.0)), float)
    rhs = np.asarray(d(u_apex + eps), float)
    return float(np.abs(rhs - lhs).max())


# ============================================================== generation ===
def _peak_rate(q0: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    X = np.concatenate([q0[None, :], target], axis=0)
    return np.abs(np.diff(X, axis=0) / float(dt)).max(axis=0)


def _draw_time(clip: RhythmClip, rng: np.random.Generator, cfg: ScenarioConfig) -> float:
    """Copy of ``warmstart_data._draw_time`` (kept local: additive-only rule)."""
    if clip.hits and rng.uniform() < cfg.short_horizon_frac:
        hit = clip.hits[int(rng.integers(len(clip.hits)))]
        k = int(rng.integers(1, max(int(cfg.short_horizon_wp), 1) + 1))
        return max(float(hit.time) - k * cfg.dt, 0.0)
    return float(rng.uniform(0.0, max(clip.duration, cfg.dt)))


_APEX_KEYS = (
    "q0", "qd0", "target", "init_base", "hit", "active", "drum",
    "apex_arm", "apex_wp",
)


def generate_apex_scenarios(
    n: int,
    clips: list[RhythmClip],
    *,
    seed: int,
    cfg: ScenarioConfig | None = None,
    apex_cfg: ApexConfig | None = None,
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    sampler=None,
    solver: ApexSolver | None = None,
    start_chunk: int = 256,
    apex_chunk: int = 512,
    progress: bool = False,
    debug_out: list | None = None,
) -> dict:
    """``n`` accepted APEX scenarios, in ``generate_scenarios``'s output format.

    This is the minimal generation driver copied from
    ``warmstart_data.generate_scenarios`` (the additive-only rule forbids
    editing it) and split into three passes per chunk so the apex IK can be
    solved **batched**:

    1. sample start + goal snapshot + kit + waist ramp, per scenario;
    2. one batched :meth:`ApexSolver.solve` per hand over the whole chunk;
    3. pick the terminal nearest the apex, build the apex target and the
       straight-line init, apply the impossible-transition filter.
    """
    cfg = cfg or ScenarioConfig()
    acfg = apex_cfg or ApexConfig()
    rng = np.random.default_rng(seed)
    names = tuple(cfg.joint_names)
    kit = kit or DrumKit.from_layout()
    library = library or GoalPoseLibrary(joint_names=names)
    if sampler is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=library)
    solver = solver or ApexSolver(cfg=acfg, fk=sampler.fk, sdf=sampler.sdf,
                                  joint_names=names, dtype=sampler.dtype)
    q_bounds = urdf_position_limits(names)
    v_limit = isaac_velocity_limits(names)
    H = int(cfg.horizon)
    yaw_i = names.index("waist_yaw_joint")
    # Same ``ScenarioConfig`` field the straight-line pool reads, so the two
    # phase-1 generators cannot disagree about where an idle arm goes.
    idle_home = resolve_idle_home(cfg.idle_home_path, joint_names=names)

    out: dict[int, dict[str, list]] = {
        b: {k: [] for k in (*_APEX_KEYS, *CONTEXT_FIELDS)} for b in cfg.buckets
    }
    stats: dict = {
        "requested": int(n), "accepted": 0, "reject_idle": 0, "reject_velocity": 0,
        "reject_library": 0, "dual_active": 0, "asymmetric": 0, "short_horizon": 0,
        "single_idle": 0, "rest": 0, "terminal_corrected": 0,
        "peak_rate_ratio": [], "peak_rate_ratio_init": [],
        "apex_solved": 0, "apex_inside": 0, "apex_collision_free": 0,
        "apex_no_room": 0, "apex_rejected_outside": 0,
        "apex_outside_m": [], "apex_height_m": [], "apex_tip_dz_m": [],
        "terminal_moved": 0, "terminal_seen": 0,
    }

    starts = None
    si = 0
    done = 0
    while done < int(n):
        m = min(int(apex_chunk), int(n) - done)
        done += m
        if progress:
            print(f"  ... {done}/{n}", flush=True)

        # ---------------- pass 1: scenario ingredients ------------------------
        recs: list[dict] = []
        for _ in range(m):
            if starts is None or si >= len(starts):
                starts = sampler.sample(int(start_chunk), rng=rng)
                si = 0
            q0 = starts.q[si].detach().cpu().numpy().astype(float)
            qd0 = starts.qd[si].detach().cpu().numpy().astype(float)
            si += 1

            kit_s = kit.randomized(rng, cfg.kit_randomization) if cfg.randomize_kit else kit
            clip = clips[int(rng.integers(len(clips)))]
            t_now = _draw_time(clip, rng, cfg)
            snap = goal_snapshot(
                clip, kit_s, t_now, num_future=cfg.num_future, rng=rng,
                hit_radius_frac=cfg.hit_radius_frac,
                speed_min=cfg.speed_min, speed_max=cfg.speed_max,
                horizon_cap=H * cfg.dt,
            )
            hit: dict[str, int] = {}
            goals: dict[str, tuple[str, float] | None] = {}
            aim: dict[str, np.ndarray] = {}
            # Future-aware aim inputs (one helper, same as the context token's).
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
                # A REST.  Kept at ``rest_frac`` when there is a home table to
                # park at; the apex solve skips it by construction (pass 2 only
                # gathers rows with a goal), so it falls through to the parked
                # base trajectory, which is exactly the target a rest wants.
                if cfg.require_active_hand and (
                    idle_home is None or rng.uniform() >= cfg.rest_frac
                ):
                    stats["reject_idle"] += 1
                    continue
                stats["rest"] += 1
            if not all(library.has(h, goals[h][0]) for h in HANDS if goals[h] is not None):
                stats["reject_library"] += 1
                continue

            if any(active):
                k_max = max(hit[h] for h in HANDS if hit[h] >= 0)
                Hb = bucket_for(k_max, cfg.buckets)
            else:
                # No hit to size the horizon from; spread rests over the buckets
                # rather than filing every one of them in the shortest.
                k_max = 1
                Hb = int(cfg.buckets[int(rng.integers(len(cfg.buckets)))])
            # The waist ramp (both hands idle) -- gives yaw_target and the yaw
            # the torso holds mid-reach, which is where the apex is solved.
            base = warmstart_trajectory(
                q0, qd0, horizon=Hb, goals={h: None for h in HANDS}, library=library,
                aim_targets=aim or None, aim_tth=aim_tth, aim_futures=aim_futures,
                dt=cfg.dt, joint_names=names,
                waist_cfg=cfg.waist_cfg, q_bounds=q_bounds,
                waist_upright_ease_wp=cfg.waist_upright_ease_wp,
            )
            sticks = None
            if cfg.stick_rand is not None:
                sticks = (cfg.stick_rand.sample(rng, "left"),
                          cfg.stick_rand.sample(rng, "right"))
            recs.append({
                "sticks": sticks,
                "q0": q0, "qd0": qd0, "hit": hit, "goals": goals, "aim": aim,
                "aim_tth": aim_tth, "aim_futures": aim_futures,
                "active": active, "Hb": Hb, "k_max": k_max, "snap": snap,
                "kit": kit_s, "waist": np.asarray(base.q, float),
                "yaw_target": float(base.yaw_target),
            })

        if not recs:
            continue

        # ---------------- pass 2: batched apex IK, per hand -------------------
        apex: dict[str, dict[int, dict]] = {h: {} for h in HANDS}
        for hand in HANDS:
            rows = [
                i for i, r in enumerate(recs)
                if r["hit"][hand] >= 2 and r["goals"][hand] is not None
            ]
            for i, r in enumerate(recs):
                if r["goals"][hand] is not None and r["hit"][hand] < 2:
                    stats["apex_no_room"] += 1
            if not rows:
                continue
            q_ref = np.stack([recs[i]["q0"].copy() for i in rows])
            for j, i in enumerate(rows):
                r = recs[i]
                k_ref = int(np.clip(round(0.5 * r["hit"][hand]), 1, max(r["hit"][hand] - 1, 1)))
                q_ref[j, :3] = r["waist"][k_ref, :3]     # waist columns are 0..2
            centers = np.stack([
                np.asarray(recs[i]["aim"][hand], float) + np.array([0.0, 0.0, acfg.height])
                for i in rows
            ])
            heads = np.stack([recs[i]["kit"].positions_in("env") for i in rows])
            radii = np.stack([recs[i]["kit"].radii for i in rows])
            t = lambda a: torch.as_tensor(  # noqa: E731
                np.asarray(a, float), device=solver.device, dtype=solver.dtype
            )
            res = solver.solve(hand, t(q_ref), t(centers), head_centers=t(heads),
                               head_radii=t(radii), rng=rng)
            q_apex = res["q"].detach().cpu().numpy()
            outside = res["outside"].detach().cpu().numpy()
            min_sdf = res["min_sdf"].detach().cpu().numpy()
            tips = res["tip"].detach().cpu().numpy()
            for j, i in enumerate(rows):
                apex[hand][i] = {
                    "q": q_apex[j], "outside": float(outside[j]),
                    "min_sdf": float(min_sdf[j]), "tip": tips[j],
                    "dz": float(tips[j][2] - recs[i]["aim"][hand][2]),
                }

        # ---- pass 2.5: apex_arm per rec, then RE-AIM the library terminal ---
        # ``apex_arm`` is hoisted out of pass 3 so the terminal correction can
        # use the SAME branch choice pass 3 does (``select_terminal`` keys on
        # the apex) instead of duplicating the rejection logic.
        apex_arm_of: list[dict[str, np.ndarray | None]] = []
        for i, _r in enumerate(recs):
            per: dict[str, np.ndarray | None] = {}
            for hand in HANDS:
                a = apex[hand].get(i)
                if a is None:
                    per[hand] = None
                    continue
                stats["apex_solved"] += 1
                stats["apex_outside_m"].append(a["outside"])
                stats["apex_tip_dz_m"].append(a["dz"])
                if a["outside"] <= 1e-6:
                    stats["apex_inside"] += 1
                if a["min_sdf"] >= acfg.min_clearance:
                    stats["apex_collision_free"] += 1
                if acfg.max_outside is not None and a["outside"] > acfg.max_outside:
                    stats["apex_rejected_outside"] += 1
                    per[hand] = None
                    continue
                per[hand] = a["q"][arm_slice(hand, names)]
            apex_arm_of.append(per)

        # The library is keyed on (hand, drum, waist_yaw) and stores a JOINT
        # pose, so it cannot know where KitRandomization put the head or what
        # stick is fitted.  Correct it toward the aim point that is actually
        # there -- batched, because that is 200x cheaper than per scenario.
        term_fix: dict[str, dict[int, np.ndarray]] = {h: {} for h in HANDS}
        #: the PRE-correction ``select_terminal`` pick, kept so the
        #: terminal_moved stat can compare apex pick vs yaw pick (the corrected
        #: pose always differs from every library entry, which measures nothing)
        term_pick: dict[str, dict[int, np.ndarray]] = {h: {} for h in HANDS}
        if acfg.correct_terminal_steps > 0:
            for hand in HANDS:
                acols = arm_slice(hand, names)
                rows = [i for i, r in enumerate(recs) if r["goals"][hand] is not None]
                if not rows:
                    continue
                qs, aims = [], []
                for i in rows:
                    r = recs[i]
                    e = select_terminal(library, hand, r["goals"][hand][0],
                                        yaw_target=r["yaw_target"],
                                        apex_arm=apex_arm_of[i][hand], cfg=acfg)
                    term_pick[hand][i] = np.asarray(e.arm_q, float)
                    # The WAIST the correction solves against must be the
                    # waist the trajectory will actually be at when it strikes:
                    # the yaw ramp does not necessarily arrive at
                    # ``yaw_target`` by ``k_hit``, and correcting against the
                    # target yaw instead of the reached one leaves a residual
                    # that scales with whatever the ramp still had to travel.
                    k_hit = int(np.clip(r["hit"][hand], 1, r["waist"].shape[0] - 1))
                    q = np.zeros(len(names))
                    q[:3] = r["waist"][k_hit, :3]
                    q[acols] = e.arm_q
                    qs.append(q)
                    aims.append(np.asarray(r["aim"][hand], float))
                tt = lambda a: torch.as_tensor(  # noqa: E731
                    np.asarray(a), device=solver.device, dtype=solver.dtype)
                # A randomized stick moves the tip, so the correction has to
                # aim the stick this scenario actually carries.  ``correct_tip``
                # takes PER-ROW sticks (the stick is just a per-row constant
                # local offset), so this stays ONE batched call per hand even
                # when every scenario drew its own stick -- grouping by spec
                # would degenerate to batch-1 under continuous randomization.
                hi_ = 0 if hand == "left" else 1
                sticks_rows = [
                    None if recs[i]["sticks"] is None else recs[i]["sticks"][hi_]
                    for i in rows
                ]
                stick_arg = (None if all(sp is None for sp in sticks_rows)
                             else sticks_rows)
                qc = adjust_arm_terminal(
                    solver.fk, tt(np.stack(qs)), tt(np.stack(aims)), hand,
                    steps=int(acfg.correct_terminal_steps),
                    damping=float(acfg.correct_terminal_damping),
                    q_bounds=q_bounds, joint_names=names, stick=stick_arg,
                ).detach().cpu().numpy()
                for j, i in enumerate(rows):
                    term_fix[hand][i] = qc[j, acols]
                stats["terminal_corrected"] += len(rows)

        # ---------------- pass 3: targets + filter ----------------------------
        for i, r in enumerate(recs):
            Hb, hit, goals = r["Hb"], r["hit"], r["goals"]
            apex_arm = apex_arm_of[i]

            # ``idle_home`` reaches the base (both-hands-idle) call through
            # ``**warmstart_kwargs``, which parks BOTH arms; the apex spline then
            # overwrites only the columns of a hand that has a goal.  So an idle
            # hand keeps the retraction and an active one keeps its apex.
            ws = apex_warmstart_trajectory(
                r["q0"], r["qd0"], horizon=Hb, goals=goals, library=library,
                apex_arm=apex_arm, aim_targets=r["aim"] or None,
                aim_tth=r["aim_tth"], aim_futures=r["aim_futures"], dt=cfg.dt,
                joint_names=names, cfg=acfg, q_bounds=q_bounds,
                waist_cfg=cfg.waist_cfg,
                waist_upright_ease_wp=cfg.waist_upright_ease_wp,
                terminal_arm={h: term_fix[h].get(i) for h in HANDS},
                idle_home=idle_home, idle_settle_wp=cfg.idle_settle_wp,
            )
            # how often "nearest apex" moved the terminal off the yaw pick --
            # compared on the PRE-correction pick: the DLS-corrected pose always
            # differs from ``library.select`` and would pin the ratio at ~1.0
            for hand in HANDS:
                if goals[hand] is None:
                    continue
                stats["terminal_seen"] += 1
                pick = term_pick[hand].get(i)
                if pick is None:  # correction off: ws carries the raw pick
                    pick = ws.terminal_arm[hand]
                ref = library.select(hand, goals[hand][0], ws.yaw_target)
                if float(np.linalg.norm(ref.arm_q - pick)) > 1e-9:
                    stats["terminal_moved"] += 1

            if debug_out is not None:
                debug_out.append({
                    "ws": ws, "q0": r["q0"], "qd0": r["qd0"], "hit": dict(hit),
                    "goals": dict(goals), "aim": dict(r["aim"]), "Hb": Hb,
                    "yaw_target": ws.yaw_target, "kit": r["kit"],
                    "apex": {h: apex[h].get(i) for h in HANDS},
                })

            target = np.asarray(ws.target, float)
            peak = _peak_rate(r["q0"], target, cfg.dt)
            ratio = float((peak / v_limit).max())
            stats["peak_rate_ratio"].append(ratio)
            stats["peak_rate_ratio_init"].append(
                float((_peak_rate(r["q0"], np.asarray(ws.init, float), cfg.dt) / v_limit).max())
            )
            if ratio > cfg.vel_reject_scale:
                stats["reject_velocity"] += 1
                continue

            ctx = build_context_inputs(
                r["kit"], r["snap"], num_drums=cfg.num_drums, num_future=cfg.num_future,
                frame="env", device="cpu", dtype=torch.float64,
                idle_home=idle_home, sticks=r["sticks"],
                yaw_now=r["q0"][names.index("waist_yaw_joint")],
                waist_cfg=cfg.waist_cfg,
            )
            rec = out[Hb]
            rec["q0"].append(r["q0"].astype(np.float32))
            rec["qd0"].append(r["qd0"].astype(np.float32))
            rec["target"].append(target.astype(np.float32))
            rec["init_base"].append(np.asarray(ws.init, float).astype(np.float32))
            rec["hit"].append(np.array([hit["left"], hit["right"]], dtype=np.int64))
            rec["active"].append(np.array(r["active"], dtype=bool))
            rec["drum"].append(np.array(
                [r["snap"].goal[h].drum if hit[h] >= 0 else -1 for h in HANDS], dtype=np.int64
            ))
            rec["apex_arm"].append(np.stack([
                np.asarray(ws.apex_arm.get(h, np.full(7, np.nan)), dtype=np.float32)
                for h in HANDS
            ]))
            rec["apex_wp"].append(np.asarray(
                [ws.apex_wp.get(h, -1) for h in HANDS], dtype=np.int64
            ))
            for f in CONTEXT_FIELDS:
                v = getattr(ctx, f)[0].numpy()
                rec[f].append(v.astype(np.int64) if f in _LONG_FIELDS
                              else (v.astype(bool) if f in _BOOL_FIELDS else v.astype(np.float32)))
            stats["accepted"] += 1
            if all(r["active"]):
                stats["dual_active"] += 1
                if hit["left"] != hit["right"]:
                    stats["asymmetric"] += 1
            elif any(r["active"]):
                stats["single_idle"] += 1
            # A rest carries the placeholder ``k_max = 1`` and has no hit, so it
            # is not run-up-to-a-strike data.
            if any(r["active"]) and r["k_max"] <= cfg.short_horizon_wp:
                stats["short_horizon"] += 1

    packed: dict[int, dict[str, np.ndarray]] = {}
    for b, rec in out.items():
        if not rec["q0"]:
            continue
        packed[b] = {k: np.stack(v) for k, v in rec.items()}
    for k in ("peak_rate_ratio", "peak_rate_ratio_init", "apex_outside_m",
              "apex_height_m", "apex_tip_dz_m"):
        stats[k] = np.asarray(stats[k], dtype=np.float32)
    return {"buckets": packed, "stats": stats}


def _apex_worker(args) -> dict:
    n, seed, cfg, acfg, clip_dir, threads = args
    torch.set_num_threads(int(threads))
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(clip_dir, kit)
    return generate_apex_scenarios(n, clips, seed=seed, cfg=cfg, apex_cfg=acfg, kit=kit)


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
    for k, v in list(stats.items()):
        if isinstance(v, list):
            stats[k] = np.concatenate(v) if v else np.asarray([], dtype=np.float32)
    return {"buckets": out_b, "stats": stats}


def generate_apex_parallel(
    n: int,
    *,
    seed: int,
    cfg: ScenarioConfig,
    apex_cfg: ApexConfig,
    split: str = "train",
    clip_dir: str | Path | None = None,
    workers: int | None = None,
    threads_per_worker: int = 1,
) -> dict:
    import multiprocessing as mp

    clip_dir = str(clip_dir if clip_dir is not None else default_rcc_dir(split))
    workers = int(workers or min(os.cpu_count() or 8, 64))
    workers = max(1, min(workers, int(n)))
    per = [n // workers] * workers
    for i in range(n - sum(per)):
        per[i] += 1
    args = [
        (per[i], seed * 100003 + i, cfg, apex_cfg, clip_dir, threads_per_worker)
        for i in range(workers)
    ]
    if workers == 1:
        return _merge([_apex_worker(args[0])])
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_apex_worker, args)
    return _merge(parts)


# ==================================================================== pool ===
class ApexWarmstartPool(WarmstartPool):
    """:class:`~amortized_opt.warmstart_data.WarmstartPool` with a separate init.

    The stored ``target`` is the **two-segment apex path** (and every auxiliary
    label is computed from it by the parent constructor).  The stored
    ``init_base`` is the **direct accelerating path to the same terminal pose**,
    and that -- not the target -- is what gets the OU + white perturbation at
    batch time.  So the model always sees a straight-line warm start and is
    asked to bend it into the apex shape, which is what deployment looks like.

    ``init_mode`` on :meth:`batch` selects the input channel:

    * ``"mix"`` -- **the default and the trained regime** (Eric, 2026-08-10,
      update 1): a per-sample coin flip, ``init_mix`` of the batch gets the
      noised APEX path (the refine/denoise mapping) and the rest the noised
      direct path (the bend mapping).  Pure direct->apex would make
      the target a deterministic function of the context alone, and the model
      is free to learn to ignore its init -- which destroys exactly the
      refinement-operator property an MPC warm start depends on.  The mix keeps
      the init informative.
    * ``"straight"`` / ``"apex"`` -- one channel only (probes and ablations);
    * ``"straight_clean"`` -- the un-noised direct accelerating path;
    * ``"hold"`` -- zero-motion hold at the start pose.

    ``white_noise`` (Eric, update 2) replaces the OU kernel with the identity,
    i.e. pure i.i.d. Gaussian per waypoint at the SAME total per-element
    variance, so the ablation isolates what temporal correlation buys.
    """

    #: default channel for callers that do not pass ``init_mode`` (the trainer)
    default_init_mode: str = "mix"
    #: fraction of a batch initialized from the apex path under ``"mix"``
    init_mix: float = 0.5
    #: drop the OU correlation, keep the total variance (the section-2 ablation)
    white_noise: bool = False

    def noised_init(
        self,
        target: torch.Tensor,
        bucket: int,
        *,
        generator: torch.Generator | None = None,
        sigma_corr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OU (or, with ``white_noise``, i.i.d.) + white perturbation of ``target``.

        The variance match is exact: the OU path from :func:`ou_kernel` has unit
        marginal sd at every index, so replacing ``L`` with the identity leaves
        ``Var = sigma_corr^2 + sigma_white^2`` per element and changes only the
        lag-1 autocorrelation (0.819 -> 0).
        """
        if not self.white_noise:
            return super().noised_init(target, bucket, generator=generator,
                                       sigma_corr=sigma_corr)
        nc = self.noise
        B, H, D = target.shape
        eps = torch.randn(B, H, D, device=target.device, dtype=target.dtype,
                          generator=generator)
        if sigma_corr is None:
            u = torch.rand(B, 1, 1, device=target.device, dtype=target.dtype,
                           generator=generator)
            lo, hi = math.log(nc.sigma_corr_min), math.log(nc.sigma_corr_max)
            sigma_corr = torch.exp(lo + (hi - lo) * u)
        out = target + sigma_corr * eps  # L = I: same variance, zero correlation
        if nc.sigma_white > 0.0:
            out = out + nc.sigma_white * torch.randn(
                B, H, D, device=target.device, dtype=target.dtype, generator=generator
            )
        if nc.clip_to_box:
            out = torch.clamp(out, self.q_lo, self.q_hi)
        return out

    def batch(
        self,
        bucket: int,
        idx: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        hold_init: bool = False,
        init_mode: str | None = None,
    ) -> dict:
        H = int(bucket)
        d = self.data[H]
        pick = lambda k: d[k].index_select(0, idx)  # noqa: E731
        q0, qd0, target = pick("q0"), pick("qd0"), pick("target")
        elem = pick("elem_mask")
        base = pick("init_base")
        mode = init_mode or ("hold" if hold_init else self.default_init_mode)
        if mode == "hold":
            init = q0.unsqueeze(1).expand_as(target).contiguous()
        elif mode == "apex":
            init = self.noised_init(target, H, generator=generator)
        elif mode == "straight":
            init = self.noised_init(base, H, generator=generator)
        elif mode == "straight_clean":
            init = base
        elif mode == "mix":
            # ONE noise draw, applied to a per-sample choice of source, so the
            # two halves see the same perturbation statistics.
            take_apex = torch.rand(
                target.shape[0], 1, 1, device=target.device, dtype=target.dtype,
                generator=generator,
            ) < float(self.init_mix)
            src = torch.where(take_apex, target, base)
            init = self.noised_init(src, H, generator=generator)
        else:
            raise ValueError(f"unknown init_mode {mode!r}")
        masks = self.masks_for(pick("hit"), pick("active"), H)
        return {
            "bucket": H,
            "state": torch.cat([q0, qd0], dim=-1),
            "q0": q0,
            "v0": qd0,
            "init": init * elem,
            "target": target,
            "init_base": pick("init_base"),
            "masks": masks,
            "context": self.context(H, idx),
            "hit": pick("hit"),
            "active": pick("active"),
            "drum": pick("drum"),
            "aux": {k: pick(f"aux_{k}") for k in ("vel", "acc", "torque", "fk_pos", "fk_vel")},
        }


def generate_apex_pool(
    n: int,
    *,
    seed: int,
    cfg: ScenarioConfig | None = None,
    apex_cfg: ApexConfig | None = None,
    noise: NoiseConfig | None = None,
    split: str = "train",
    clip_dir: str | Path | None = None,
    workers: int | None = None,
    threads_per_worker: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    fk: ChainFK | None = None,
    rnea=None,
    init_mode: str = "mix",
    init_mix: float = 0.5,
    white_noise: bool = False,
    aux_torque_tracking: bool = False,
    terminal_initializer=None,
) -> ApexWarmstartPool:
    cfg = cfg or ScenarioConfig()
    acfg = apex_cfg or ApexConfig()
    raw = generate_apex_parallel(
        n, seed=seed, cfg=cfg, apex_cfg=acfg, split=split, clip_dir=clip_dir,
        workers=workers, threads_per_worker=threads_per_worker,
    )
    stats = dict(raw["stats"])
    stats["apex_cfg"] = asdict(acfg)
    pool = ApexWarmstartPool(
        raw["buckets"], cfg=cfg, noise=noise, device=device, dtype=dtype,
        fk=fk, rnea=rnea, stats=stats,
        aux_torque_tracking=bool(aux_torque_tracking),
        trajectory_transform=(
            None if terminal_initializer is None
            else _terminal_head_transform(terminal_initializer, cfg)
        ),
    )
    pool.default_init_mode = str(init_mode)
    pool.init_mix = float(init_mix)
    pool.white_noise = bool(white_noise)
    return pool


def _terminal_head_transform(initializer, cfg: ScenarioConfig):
    """Retarget an apex pool before its FK/RNEA auxiliary labels are built."""
    names = tuple(cfg.joint_names)
    q_lo_np, q_hi_np = urdf_position_limits(names)
    q_lo = torch.as_tensor(q_lo_np, device=initializer.nominal_axis.device)
    q_hi = torch.as_tensor(q_hi_np, device=initializer.nominal_axis.device)
    arm_cols = {
        hand: torch.as_tensor(
            np.arange(len(names))[arm_slice(hand, names)],
            device=initializer.nominal_axis.device, dtype=torch.long,
        )
        for hand in HANDS
    }

    @torch.no_grad()
    def transform(d, ctx, masks, H):
        state = torch.cat([d["q0"], d["qd0"]], dim=-1)
        terminal = initializer.terminal_pose(ctx, state, masks)
        seed = initializer(ctx, state, masks)
        target = seed.clone()
        # Active trajectories visit the pre-solved apex, but both legs now use
        # exactly the same low-acceleration law as the direct initializer.
        for hi, hand in enumerate(HANDS):
            cols = arm_cols[hand].to(d["q0"].device)
            active = d["active"][:, hi] & (d["apex_wp"][:, hi] >= 1)
            if not bool(active.any()):
                continue
            rows = active.nonzero(as_tuple=True)[0]
            path = accelerating_apex_path_torch(
                d["q0"].index_select(0, rows).index_select(1, cols),
                d["qd0"].index_select(0, rows).index_select(1, cols),
                d["apex_arm"].index_select(0, rows)[:, hi],
                terminal.index_select(0, rows).index_select(1, cols),
                d["apex_wp"].index_select(0, rows)[:, hi],
                d["hit"].index_select(0, rows)[:, hi],
                int(H), float(cfg.dt),
            )
            target[rows[:, None, None], torch.arange(H, device=target.device)[None, :, None],
                   cols[None, None, :]] = path
        target = target.clamp(q_lo.to(target), q_hi.to(target))
        return {"target": target * masks.elem_mask, "init_base": seed}

    return transform


# ================================================================ planners ===
def apex_plan_for_case(
    case,
    *,
    library: GoalPoseLibrary,
    solver: ApexSolver,
    cfg: ApexConfig,
    horizon: int,
    joint_names=G1_UPPER_BODY_JOINTS,
    waist_cfg=None,
    waist_upright_ease_wp: int | None = None,
) -> ApexWarmStart:
    """The apex warm start for one :class:`~amortized_opt.eval_harness.EvalCase`.

    Frames: the harness is pelvis-rooted, the kit / aim tokens are env-rooted,
    so the aim point is lifted by ``PELVIS_IN_ENV`` exactly as
    :mod:`amortized_opt.planner` does.
    """
    from .constants import PELVIS_IN_ENV

    names = tuple(joint_names)
    H = int(horizon)
    goals: dict[str, tuple[str, float] | None] = {h: None for h in HANDS}
    goals[case.hand] = (str(case.drum), float(case.N) * float(case.dt))
    aim = {case.hand: np.asarray(case.strike.position, float) + PELVIS_IN_ENV}
    q0 = np.asarray(case.q_start, float)
    v0 = np.asarray(case.qdot_start, float)
    q_bounds = urdf_position_limits(names)

    k_hit = int(np.clip(round(float(case.N)), 1, H))
    apex_arm: dict[str, np.ndarray | None] = {h: None for h in HANDS}
    if k_hit >= 2:
        base = warmstart_trajectory(
            q0, v0, horizon=H, goals={h: None for h in HANDS}, library=library,
            aim_targets=aim, dt=float(case.dt), joint_names=names,
            waist_cfg=waist_cfg, q_bounds=q_bounds,
            waist_upright_ease_wp=waist_upright_ease_wp,
        )
        k_ref = int(np.clip(round(0.5 * k_hit), 1, max(k_hit - 1, 1)))
        q_ref = q0.copy()
        q_ref[:3] = np.asarray(base.q, float)[k_ref, :3]
        center = aim[case.hand] + np.array([0.0, 0.0, cfg.height])
        t = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a, float)[None, ...], device=solver.device, dtype=solver.dtype
        )
        res = solver.solve(case.hand, t(q_ref), t(center))
        if cfg.max_outside is None or float(res["outside"][0]) <= cfg.max_outside:
            apex_arm[case.hand] = (
                res["q"][0].detach().cpu().numpy()[arm_slice(case.hand, names)]
            )
    return apex_warmstart_trajectory(
        q0, v0, horizon=H, goals=goals, library=library, apex_arm=apex_arm,
        aim_targets=aim, dt=float(case.dt), joint_names=names, cfg=cfg,
        q_bounds=q_bounds, waist_cfg=waist_cfg,
        waist_upright_ease_wp=waist_upright_ease_wp,
    )


def apex_case_planner(
    *,
    library: GoalPoseLibrary | None = None,
    apex_cfg: ApexConfig | None = None,
    solver: ApexSolver | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    waist_cfg=None,
    waist_upright_ease_wp: int | None = None,
):
    """The apex teacher, as an eval-harness planner (``plan(case) -> PlanResult``)."""
    from .eval_harness import PlanResult

    lib = library or GoalPoseLibrary(joint_names=tuple(joint_names))
    cfg = apex_cfg or ApexConfig()
    sol = solver or ApexSolver(cfg=cfg, joint_names=joint_names, dtype=torch.float64)

    def plan(case):
        ws = apex_plan_for_case(
            case, library=lib, solver=sol, cfg=cfg, horizon=int(case.N),
            joint_names=joint_names, waist_cfg=waist_cfg,
            waist_upright_ease_wp=waist_upright_ease_wp,
        )
        return PlanResult(Q=np.asarray(ws.q, float), meta={"converged": True})

    return plan


@dataclass
class ApexModelPlanner:
    """The trained decoder as a grid planner, with an APEX initialization option.

    A thin composition over :class:`~amortized_opt.planner.ModelPlanner` -- the
    only thing it changes is which trajectory goes into the ``init`` slot:
    ``"quintic"`` / ``"hold"`` delegate, ``"apex"`` feeds the apex teacher's
    trajectory instead.
    """

    model: object
    device: str = "cuda:1"
    iterations: int = 1
    init: str = "quintic"          # quintic | hold | apex
    buckets: tuple[int, ...] = (25, 50, 100)
    num_drums: int = 8
    num_future: int = 3
    library: GoalPoseLibrary | None = None
    kit: DrumKit | None = None
    apex_cfg: ApexConfig | None = None
    solver: ApexSolver | None = None
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS
    amp_dtype: torch.dtype | None = None
    waist_cfg: object = None
    #: Waist pitch/roll ease-to-upright window of the seed; must match the
    #: checkpoint's ``ScenarioConfig.waist_upright_ease_wp``.  ``None`` = hold.
    waist_upright_ease_wp: int | None = None

    def __post_init__(self) -> None:
        from .planner import ModelPlanner

        self.library = self.library or GoalPoseLibrary(joint_names=tuple(self.joint_names))
        self.kit = self.kit or DrumKit.from_layout()
        self.apex_cfg = self.apex_cfg or ApexConfig()
        if self.init == "apex" and self.solver is None:
            self.solver = ApexSolver(cfg=self.apex_cfg, joint_names=self.joint_names,
                                     dtype=torch.float64)
        self._inner = ModelPlanner(
            model=self.model, device=self.device, iterations=self.iterations,
            init="hold" if self.init == "hold" else "quintic",
            buckets=self.buckets, num_drums=self.num_drums, num_future=self.num_future,
            library=self.library, kit=self.kit, joint_names=self.joint_names,
            amp_dtype=self.amp_dtype, waist_cfg=self.waist_cfg,
            waist_upright_ease_wp=self.waist_upright_ease_wp,
        )
        if self.init == "apex":
            inner = self._inner

            def _apex_init(case, H, _inner=inner):
                ws = apex_plan_for_case(
                    case, library=self.library, solver=self.solver, cfg=self.apex_cfg,
                    horizon=int(H), joint_names=self.joint_names, waist_cfg=self.waist_cfg,
                    waist_upright_ease_wp=self.waist_upright_ease_wp,
                )
                return np.asarray(ws.target, float)

            inner._init_traj = _apex_init

    def __call__(self, case):
        return self._inner(case)


# ============================================================ diagnostics ====
@torch.no_grad()
def cylinder_report(
    trajectories: np.ndarray,
    hands: list[str],
    centers: np.ndarray,
    *,
    fk: ChainFK,
    cfg: ApexConfig,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Does the emitted path actually visit the cylinder?

    ``trajectories`` is ``(B, K, 17)``, ``centers`` ``(B, 3)`` the cylinder
    centres in the env frame.  Returns the fraction of paths with at least one
    waypoint inside, and the height above the strike of the closest approach.
    """
    B, K, _ = trajectories.shape
    q = torch.as_tensor(trajectories.reshape(B * K, -1), dtype=dtype, device=fk.device)
    tips = torch.empty(B * K, 3, dtype=dtype, device=fk.device)
    for hand in set(hands):
        sel = torch.as_tensor(
            np.repeat([h == hand for h in hands], K), device=fk.device
        ).nonzero().squeeze(-1)
        if sel.numel():
            tips[sel] = tip_position(fk, q.index_select(0, sel), hand, frame="env")
    tips = tips.reshape(B, K, 3).cpu().numpy()
    c = np.asarray(centers, float).reshape(B, 1, 3)
    d = tips - c
    rho = np.linalg.norm(d[..., :2], axis=-1)
    s_r = np.maximum(rho - cfg.radius, 0.0)
    s_z = np.maximum(np.abs(d[..., 2]) - cfg.half_height, 0.0)
    outside = np.sqrt(s_r ** 2 + s_z ** 2)
    k_best = outside.argmin(axis=1)
    best = outside[np.arange(B), k_best]
    dz = d[np.arange(B), k_best, 2] + cfg.height  # height above the strike point
    return {
        "inside_frac": float((best <= 1e-9).mean()),
        "min_outside_m": best,
        "closest_wp": k_best,
        "apex_height_m": dz,
    }
