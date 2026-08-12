"""Evaluation harness: score ANY trajectory-producing callable on one grid.

Section 2 of ``docs/amortized_optimizer_plan.md``.  A fork, in spirit, of
``scripts/bench_trajopt_offline.py`` -- but where that script welds the scenario
generator, the solver call and the scorer into one file, this splits them so the
*same* randomized single-strike grid and the *same* scorer can be pointed at

* the repo's HPIPM trajopt (via ``scripts/repo_adapters/``), to record the
  reference baseline; and later
* the amortized transformer, so its numbers are directly comparable.

Everything here scores with the **vendored** physics -- ``amortized_opt``'s FK,
RNEA, SDF and limit tables.  Nothing in this module imports the repo.

Interface
---------

A *planner* is any callable::

    plan(case: EvalCase) -> PlanResult | np.ndarray

returning ``case.n_waypoints`` rows of 17 joint angles on the 50 Hz grid
(``CONTROL_DT`` = 0.02, resolved decision 1), row 0 being the start pose and row
``case.N`` the strike waypoint.  Returning a bare array is shorthand for
``PlanResult(Q=array)``.

Conventions
-----------

* **Frame.** Everything is pelvis-rooted, matching the vendored FK and
  ``drums_from_layout``'s default (which subtracts ``PELVIS_IN_ENV`` from the
  layout's ground-referenced positions).  ``DrumSceneSpec.from_layout`` is
  therefore built with ``frame="pelvis"``.
* **Rates.**  ``qd[0] = case.qdot_start`` (the boundary condition the planner was
  given), ``qd[i] = (Q[i] - Q[i-1]) / dt`` for ``i >= 1``;
  ``qdd[i] = (qd[i+1] - qd[i]) / dt``, with ``qdd[N] = qdd[N-1]``.  This differs
  by one index from ``bench_trajopt_offline``'s ``ops.d1``/``ops.d2`` pairing,
  which evaluates the torque at knot ``i`` using the *forward* difference there;
  the convention here uses the real start velocity instead of discarding it.
* **Contact point.**  Position, speed and direction are scored at the strike's
  ``contact_sigma`` point on the stick -- the bead on pads, the shoulder on
  cymbals and the hi-hat -- not at the tip.  The tip is reported alongside so the
  numbers stay comparable to the repo benches, which are tip-only.
* **Collision.**  Evaluated on the path interpolated at ``collision_substeps``
  samples per segment (waypoints included), so between-waypoint tunnelling is
  caught the way ``opt.verify()`` catches it.  The striking hand's contact geoms
  are exempted from the target drum (and its under-volume) over the last
  ``strike_exempt_waypoints`` waypoints -- you cannot both avoid a drum and hit
  it.  Shaft points are exempted only for a shaft contact, mirroring
  ``robot.strike_ignore_pairs``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .constants import (
    CONTROL_DT,
    G1_UPPER_BODY_JOINTS,
    N_DOF,
    default_layout_path,
)
from .fk import (
    ChainFK,
    stick_axis_local,
    stick_contact_offset,
    stick_shaft_offsets,
    stick_tip_offset,
)
from .limits import (
    isaac_effort_limits,
    isaac_runtime_stiffness,
    isaac_velocity_limits,
    urdf_position_limits,
)
from .rnea import BatchedRNEA
from .sdf import (
    DrumSceneSpec,
    point_capsule_sdf,
    point_cylinder_sdf,
    point_sphere_sdf,
)
from .strike import (
    StrikeSpec,
    shaft_hit_allowed,
    stick_elevation_from_surface,
    stick_elevation_ok,
    strike_elevation_required,
    strike_from_drum,
)

__all__ = [
    "AcceptanceThresholds",
    "CaseScore",
    "DRUM_HAND",
    "EvalCase",
    "PlanResult",
    "TrajectoryScorer",
    "aggregate",
    "generate_grid",
    "load_run",
    "run_eval",
    "save_run",
    "write_csv",
]


#: Hand each drum is struck with in the acceptance grid.  Copied from
#: ``scripts/bench_trajopt_norrt.py::_DRUM_HAND`` so the 162-case grid this
#: harness reproduces is cell-for-cell the repo's.  ``floor_tom`` is absent
#: there and is therefore opt-in here (see :func:`generate_grid`).
DRUM_HAND: dict[str, str] = {
    "snare": "right",
    "hihat": "left",
    "tom1": "right",
    "tom2": "left",
    "cymbal_left": "left",
    "cymbal_right": "right",
}

#: Hand for drums outside :data:`DRUM_HAND`, by kit side (y > 0 is the robot's
#: left).  Only reached with ``include_extra_drums=True``.
_EXTRA_DRUM_HAND: dict[str, str] = {"floor_tom": "right"}

_WAIST_IDX = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("waist_")]
_ARM_IDX = {
    "left": [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("left_")],
    "right": [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("right_")],
}
_WRIST_IDX = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if "wrist" in n]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


# ================================================================== cases ====


@dataclass
class EvalCase:
    """One scored scenario: where the robot starts and what it must hit.

    ``q_start`` / ``qdot_start`` are full 17-DoF states even when the planner
    only moves a subset (the trajopt baseline freezes the idle arm; see
    ``scripts/repo_adapters/trajopt_hpipm_planner.py``).  The scorer always
    scores all 17.
    """

    case_id: int
    grid: str
    drum: str
    drum_index: int
    hand: str
    N: int
    dt: float
    strike_speed: float
    q_start: np.ndarray
    qdot_start: np.ndarray
    strike: StrikeSpec
    start_mode: str = "box"
    start_drum: str | None = None
    horizon_s: float = 0.0
    layout: str = ""

    def __post_init__(self) -> None:
        self.q_start = np.asarray(self.q_start, float).reshape(N_DOF)
        self.qdot_start = np.asarray(self.qdot_start, float).reshape(N_DOF)
        if not self.horizon_s:
            self.horizon_s = float(self.N) * float(self.dt)

    @property
    def n_waypoints(self) -> int:
        return int(self.N) + 1

    # -------------------------------------------------------- serialization --
    def to_dict(self) -> dict:
        s = self.strike
        return dict(
            case_id=int(self.case_id),
            grid=str(self.grid),
            drum=str(self.drum),
            drum_index=int(self.drum_index),
            hand=str(self.hand),
            N=int(self.N),
            dt=float(self.dt),
            strike_speed=float(self.strike_speed),
            q_start=np.asarray(self.q_start, float).tolist(),
            qdot_start=np.asarray(self.qdot_start, float).tolist(),
            start_mode=str(self.start_mode),
            start_drum=None if self.start_drum is None else str(self.start_drum),
            horizon_s=float(self.horizon_s),
            layout=str(self.layout),
            strike=dict(
                time=float(s.time),
                hand=str(s.hand),
                position=np.asarray(s.position, float).tolist(),
                velocity=np.asarray(s.velocity, float).tolist(),
                drum=s.drum,
                hit_radius=None if s.hit_radius is None else float(s.hit_radius),
                half_plane_dir=(
                    None if s.half_plane_dir is None
                    else np.asarray(s.half_plane_dir, float).tolist()
                ),
                disk_center_offset=(
                    None if s.disk_center_offset is None
                    else np.asarray(s.disk_center_offset, float).tolist()
                ),
                contact_sigma=None if s.contact_sigma is None else float(s.contact_sigma),
            ),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        s = d["strike"]
        strike = StrikeSpec(
            time=float(s["time"]),
            hand=str(s["hand"]),
            position=np.asarray(s["position"], float),
            velocity=np.asarray(s["velocity"], float),
            drum=s.get("drum"),
            hit_radius=s.get("hit_radius"),
            half_plane_dir=(
                None if s.get("half_plane_dir") is None
                else np.asarray(s["half_plane_dir"], float)
            ),
            disk_center_offset=(
                None if s.get("disk_center_offset") is None
                else np.asarray(s["disk_center_offset"], float)
            ),
            contact_sigma=s.get("contact_sigma"),
        )
        return cls(
            case_id=int(d["case_id"]),
            grid=str(d["grid"]),
            drum=str(d["drum"]),
            drum_index=int(d["drum_index"]),
            hand=str(d["hand"]),
            N=int(d["N"]),
            dt=float(d["dt"]),
            strike_speed=float(d["strike_speed"]),
            q_start=np.asarray(d["q_start"], float),
            qdot_start=np.asarray(d["qdot_start"], float),
            strike=strike,
            start_mode=str(d.get("start_mode", "box")),
            start_drum=d.get("start_drum"),
            horizon_s=float(d.get("horizon_s", 0.0)),
            layout=str(d.get("layout", "")),
        )


@dataclass
class PlanResult:
    """What a planner returns: the trajectory plus whatever it wants recorded.

    ``meta`` is free-form; the keys the CSV promotes to columns are
    ``converged``, ``iterations``, ``message`` and ``failed``.
    """

    Q: np.ndarray
    solve_s: float = float("nan")
    meta: dict = field(default_factory=dict)


Planner = Callable[[EvalCase], "PlanResult | np.ndarray"]


# ============================================================= thresholds ====


@dataclass
class AcceptanceThresholds:
    """Pass/fail bands for the acceptance-style counts in the summary.

    Deliberately loose-but-physical rather than tuned: the point of the harness
    is the *paired comparison* between planners, and a threshold sweep would
    make the headline counts a function of the threshold.  ``contact_pos_tol_m``
    is 2 cm because that is roughly a stick-bead diameter against a 14-19 cm
    head; the speed band is the looser of an absolute and a relative bound so a
    3 m/s strike is not held to the same absolute error as a 1 m/s one.
    """

    #: ``ok_hit_tight`` only.  The primary hit test is membership of the legal
    #: hit disk (``on_head``), because that is what the solver is *constrained*
    #: to and what the RL hit detector scores.  Measured on the trajopt
    #: baseline, distance to the *aim* runs ~11 cm at the p50 while every plan
    #: is still on the head: the aim is where the objective pulls
    #: (``w_strike_position = 10``, vs ``mu_hit = 5e7`` on the disk row), not
    #: where a hit has to land.  Using the aim distance as the pass/fail would
    #: score a perfectly good strike as a miss.
    contact_pos_tol_m: float = 0.02
    speed_err_tol_mps: float = 0.25
    speed_err_tol_frac: float = 0.25
    collision_margin_m: float = 0.005
    torque_tol_nm: float = 0.0
    velocity_tol_rads: float = 0.0
    position_tol_rad: float = 0.0


# ================================================================ scoring ====

@dataclass
class CaseScore:
    """Every metric the harness records for one (case, trajectory) pair."""

    # --- identification -----------------------------------------------------
    case_id: int
    grid: str
    drum: str
    hand: str
    start_mode: str
    N: int
    horizon_s: float
    strike_speed: float
    contact_sigma: float
    # --- planner-reported ---------------------------------------------------
    solve_s: float
    planner_failed: bool
    converged: bool
    iterations: int
    message: str
    # --- strike quality at the contact waypoint -----------------------------
    contact_pos_err_mm: float
    tip_pos_err_mm: float
    on_head: bool
    inplane_dist_m: float          # |in-plane offset from the head centre|
    axial_dist_m: float            # signed distance along the head normal
    half_plane_margin_m: float     # >= 0 means on the playable semicircle
    contact_speed_mps: float
    contact_speed_target_mps: float
    contact_speed_err_mps: float
    contact_dir_cos: float
    into_head_mps: float
    tangential_mps: float
    approach_deg: float
    tip_speed_mps: float
    stick_elev_deg: float
    elev_ok: bool
    # --- boundary conditions the planner was handed -------------------------
    start_pos_err_rad: float
    start_vel_err_rads: float
    # --- collision (striking hand) ------------------------------------------
    coll_min_sdf_m: float
    coll_pen_max_m: float
    coll_pen_count: int
    coll_margin_count: int
    # --- collision (both hands, incl. any idle arm the planner froze) -------
    coll_min_sdf_all_m: float
    coll_pen_max_all_m: float
    coll_pen_count_all: int
    self_coll_min_sdf_m: float
    # --- torque (RNEA + armature, vs the Isaac effort limits) ---------------
    tau_max_nm: float
    tau_excess_max_nm: float
    tau_excess_max_frac: float
    tau_frac_over: float
    tau_knots_over: int
    pd_track_err_needed_max_rad: float
    # --- velocity / joint box ------------------------------------------------
    vel_excess_max_rads: float
    vel_frac_over: float
    qpos_excess_max_rad: float
    qpos_posture_excess_max_rad: float
    # --- shape ---------------------------------------------------------------
    travel_l1_rad: float
    travel_l2_rad: float
    travel_net_rad: float
    travel_waste: float
    travel_wrist_rad: float
    reversals: int
    jerk_integral: float
    accel_integral: float
    tip_path_len_m: float
    # --- acceptance ----------------------------------------------------------
    ok_hit: bool        # contact point inside the legal hit disk
    ok_hit_tight: bool  # ... and within contact_pos_tol_m of the AIM point
    ok_speed: bool
    ok_collision: bool
    ok_collision_margin: bool
    ok_torque: bool
    ok_velocity: bool
    ok_joint_limits: bool
    accepted: bool


class TrajectoryScorer:
    """Scores a trajectory against an :class:`EvalCase` with vendored physics.

    One instance owns the FK / RNEA / drum-scene tensors and is reusable across
    cases (and cheap to reuse -- building the RNEA needs Pinocchio, the scoring
    itself is pure torch).

    ``device`` follows the package rule: call ``torch.cuda.set_device`` before
    constructing one on a non-zero CUDA device.  Scoring runs in float64 by
    default; the batches are tiny (a few hundred poses per case) and mm-level
    position numbers should not carry fp32 noise.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        layout_path: str | Path | None = None,
        urdf_path: str | Path | None = None,
        collision_substeps: int = 4,
        strike_exempt_waypoints: int = 2,
        start_exempt_waypoints: int | None = None,
        collision_margin: float = 0.005,
        shaft_samples: int = 4,
        under_depth: float = 0.08,
        under_clearance: float = 0.01,
        thresholds: AcceptanceThresholds | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.collision_substeps = max(int(collision_substeps), 1)
        self.strike_exempt_waypoints = max(int(strike_exempt_waypoints), 0)
        # Mirror image of the strike exemption: a case that STARTS on a drum
        # (``EvalCase.start_drum``, the near_drum grid) is in contact with that
        # head at waypoint 0 by construction -- the start pose is that drum's
        # canonical terminal strike pose.  Without this, every departure case
        # scores as a collision at its own boundary condition.  Same reasoning as
        # the repo's departure exemption (docs: "collapse from contact start").
        self.start_exempt_waypoints = (
            self.strike_exempt_waypoints
            if start_exempt_waypoints is None
            else max(int(start_exempt_waypoints), 0)
        )
        self.collision_margin = float(collision_margin)
        self.thresholds = thresholds or AcceptanceThresholds()
        self.layout_path = Path(layout_path) if layout_path else default_layout_path()

        self.fk = ChainFK(urdf_path, device=self.device, dtype=dtype)
        self.rnea = BatchedRNEA.from_urdf(
            urdf_path, device=self.device, dtype=dtype, with_armature=True
        )
        # Pelvis frame: matches the vendored FK and the repo's drums_from_layout
        # default (layout position - PELVIS_IN_ENV).
        self.scene = DrumSceneSpec.from_layout(
            self.layout_path,
            frame="pelvis",
            under_depth=under_depth,
            under_clearance=under_clearance,
        )
        self.drum_index = {n: i for i, n in enumerate(self.scene.names)}

        t = lambda a: torch.as_tensor(np.asarray(a, float), device=self.device, dtype=dtype)
        self._heads = t(self.scene.head_centers)                    # (D, 3)
        self._axes = t(self.scene.axes)                             # (D, 3)
        self._radii = t(self.scene.radii)                           # (D,)
        self._half_h = t(self.scene.heights / 2.0)                  # (D,)
        # Same construction as SdfPenalty._cylinder_centers: the solid drum hangs
        # height/2 below the head, and the under-drum no-go volume hangs
        # under_clearance below that.
        self._body_c = self._heads - self._half_h.unsqueeze(-1) * self._axes
        under_top = (
            self._heads
            - t(self.scene.heights + under_clearance).unsqueeze(-1) * self._axes
        )
        self._under_c = under_top - (under_depth / 2.0) * self._axes
        self._under_half = t(np.full(self.scene.num_drums, under_depth / 2.0))

        # Robot sample points per hand: [tip, shaft*S, wrist], wrist-local.
        from .constants import STICK_RADIUS, TIP_RADIUS, TORSO_CAPSULE, HEAD_SPHERE, WRIST_RADIUS

        self._pts_local = {}
        for hand in ("left", "right"):
            local = np.concatenate(
                [
                    stick_tip_offset(hand)[None, :],
                    stick_shaft_offsets(hand, shaft_samples),
                    np.zeros((1, 3)),
                ],
                axis=0,
            )
            self._pts_local[hand] = t(local)
        self.n_points = int(self._pts_local["left"].shape[0])
        self.tip_index = 0
        self.shaft_indices = list(range(1, 1 + shaft_samples))
        self._pt_radii = t(
            np.concatenate([[TIP_RADIUS], np.full(shaft_samples, STICK_RADIUS), [WRIST_RADIUS]])
        )
        a, b, r = TORSO_CAPSULE
        self._torso_a, self._torso_b, self._torso_r = t(a), t(b), float(r)
        hc, hr = HEAD_SPHERE
        self._head_c, self._head_r = t(hc), float(hr)
        self._stick_axis_local = {h: t(stick_axis_local(h)) for h in ("left", "right")}

        self.tau_limit = isaac_effort_limits()
        self.v_limit = isaac_velocity_limits()
        self.kp = isaac_runtime_stiffness()
        self.q_lo_hard, self.q_hi_hard = urdf_position_limits(posture_limits=False)
        self.q_lo_post, self.q_hi_post = urdf_position_limits(posture_limits=True)

    # ------------------------------------------------------------- helpers --
    def _t(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, float), device=self.device, dtype=self.dtype)

    def contact_world(
        self, q: np.ndarray, hand: str, sigma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(contact point, tip, stick axis) in the pelvis frame for one pose."""
        Q = self._t(np.asarray(q, float).reshape(1, N_DOF))
        R, p = self.fk.frame(Q, f"{hand}_wrist")
        loc = self._t(stick_contact_offset(hand, sigma)).view(1, 3, 1)
        tip = self._t(stick_tip_offset(hand)).view(1, 3, 1)
        ax = self._stick_axis_local[hand].view(1, 3, 1)
        c = (p + (R @ loc).squeeze(-1))[0].cpu().numpy()
        tp = (p + (R @ tip).squeeze(-1))[0].cpu().numpy()
        d = (R @ ax).squeeze(-1)[0].cpu().numpy()
        return c, tp, d

    def tip_jacobian(self, q: np.ndarray, hand: str, sigma: float | None = None) -> np.ndarray:
        """(3, 17) world Jacobian of the stick point at ``sigma`` via autograd.

        Used by the near-drum start sampler to turn a commanded departure tip
        velocity into joint velocities; exact, since the point is rigidly
        attached to the wrist-yaw link.
        """
        loc = self._t(
            stick_tip_offset(hand) if sigma is None else stick_contact_offset(hand, sigma)
        ).view(1, 3, 1)
        Q = self._t(np.asarray(q, float).reshape(1, N_DOF)).requires_grad_(True)

        def f(x: torch.Tensor) -> torch.Tensor:
            R, p = self.fk.frame(x, f"{hand}_wrist")
            return (p + (R @ loc).squeeze(-1)).reshape(3)

        J = torch.autograd.functional.jacobian(f, Q, vectorize=True)
        return J.reshape(3, N_DOF).detach().cpu().numpy()

    # ---------------------------------------------------------- collision ---
    def _points_world(self, Qt: torch.Tensor, hand: str) -> torch.Tensor:
        R, p = self.fk.frame(Qt, f"{hand}_wrist")
        return ChainFK.points_world(R, p, self._pts_local[hand])  # (B, P, 3)

    def _drum_sdf(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, P, D) signed distances to the drum bodies and under-volumes."""
        x = pts.unsqueeze(2)
        ax = self._axes.view(1, 1, -1, 3)
        rad = self._radii.view(1, 1, -1)
        body = point_cylinder_sdf(
            x, self._body_c.view(1, 1, -1, 3), ax, rad, self._half_h.view(1, 1, -1)
        )
        under = point_cylinder_sdf(
            x, self._under_c.view(1, 1, -1, 3), ax, rad, self._under_half.view(1, 1, -1)
        )
        r = self._pt_radii.view(1, -1, 1)
        return body - r, under - r

    def _self_sdf(self, Qt: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        Rt, pt = self.fk.frame(Qt, "torso")
        a = pt + (Rt @ self._torso_a.view(1, 3, 1)).squeeze(-1)
        b = pt + (Rt @ self._torso_b.view(1, 3, 1)).squeeze(-1)
        hc = pt + (Rt @ self._head_c.view(1, 3, 1)).squeeze(-1)
        s_t = point_capsule_sdf(pts, a.unsqueeze(1), b.unsqueeze(1), self._torso_r)
        s_h = point_sphere_sdf(pts, hc.unsqueeze(1), self._head_r)
        return torch.minimum(s_t, s_h) - self._pt_radii.view(1, -1)

    # -------------------------------------------------------------- score ---
    def score(
        self,
        case: EvalCase,
        Q: np.ndarray,
        *,
        solve_s: float = float("nan"),
        meta: dict | None = None,
    ) -> CaseScore:
        meta = dict(meta or {})
        Q = np.asarray(Q, float)
        if Q.ndim != 2 or Q.shape[1] != N_DOF:
            raise ValueError(f"trajectory must be (T, {N_DOF}), got {Q.shape}")
        if Q.shape[0] != case.n_waypoints:
            raise ValueError(
                f"case {case.case_id} wants {case.n_waypoints} waypoints, got {Q.shape[0]}"
            )
        if not np.all(np.isfinite(Q)):
            raise ValueError(f"case {case.case_id}: trajectory has non-finite entries")

        dt = float(case.dt)
        N = int(case.N)
        hand = case.hand
        spec = case.strike
        sigma = spec.resolved_contact_sigma()
        thr = self.thresholds

        # ---- rates ---------------------------------------------------------
        qd = np.empty_like(Q)
        qd[0] = case.qdot_start
        qd[1:] = np.diff(Q, axis=0) / dt
        qdd = np.empty_like(Q)
        qdd[:-1] = np.diff(qd, axis=0) / dt
        qdd[-1] = qdd[-2] if N >= 1 else 0.0
        qddd = np.zeros_like(Q)
        if N >= 2:
            qddd[:-1] = np.diff(qdd, axis=0) / dt
            qddd[-1] = qddd[-2]

        # ---- terminal strike geometry --------------------------------------
        k = int(np.clip(spec.waypoint(dt), 1, N))
        p_c, p_tip, d_stick = self.contact_world(Q[k], hand, sigma)
        Jc = self.tip_jacobian(Q[k], hand, sigma)
        Jt = self.tip_jacobian(Q[k], hand, None)
        v_c = Jc @ qd[k]
        v_tip = Jt @ qd[k]

        p_star = np.asarray(spec.position, float)
        v_star = np.asarray(spec.velocity, float)
        n_in = _unit(v_star)          # into the head
        n_out = -n_in                 # head normal, outward
        speed = float(np.linalg.norm(v_c))
        speed_tgt = float(np.linalg.norm(v_star))
        into = float(v_c @ n_in)
        tang = float(np.linalg.norm(v_c - into * n_in))
        approach_deg = float(np.degrees(np.arccos(np.clip(into / max(speed, 1e-9), -1.0, 1.0))))

        # on-head test, in the head plane, with the aim's disk offset applied
        e = p_c - p_star
        axial = float(e @ n_out)
        e_in = e - axial * n_out
        d_off = (
            np.zeros(3) if spec.disk_center_offset is None
            else np.asarray(spec.disk_center_offset, float)
        )
        inplane = float(np.linalg.norm(e_in - d_off))
        hit_R = spec.resolved_hit_radius()
        hp_margin = (
            float("inf") if spec.half_plane_dir is None
            else float((e_in - d_off) @ np.asarray(spec.half_plane_dir, float))
        )
        on_head = bool(inplane <= hit_R and abs(axial) <= 0.02 and hp_margin >= -1e-9)

        elev = stick_elevation_from_surface(d_stick, n_out)
        elev_deg = float(np.degrees(elev))
        elev_ok = bool(
            (not strike_elevation_required(spec.drum))
            or stick_elevation_ok(d_stick, n_out)
        )

        # ---- collision on the interpolated path ------------------------------
        s = self.collision_substeps
        if s > 1:
            frac = np.arange(s, dtype=float)[None, :, None] / s
            fine = (Q[:-1, None, :] * (1.0 - frac) + Q[1:, None, :] * frac).reshape(-1, N_DOF)
            fine = np.concatenate([fine, Q[-1:]], axis=0)
        else:
            fine = Q
        Qt = self._t(fine)
        n_fine = Qt.shape[0]
        # first fine sample of the exempt tail / last of the exempt head
        first_exempt = max(0, (N - self.strike_exempt_waypoints)) * s
        last_start_exempt = min(self.start_exempt_waypoints * s + 1, n_fine)
        tgt = self.drum_index.get(str(spec.drum), -1)
        src = (
            self.drum_index.get(str(case.start_drum), -1)
            if case.start_drum is not None and self.start_exempt_waypoints > 0
            else -1
        )

        stats = {}
        for h in ("left", "right"):
            pts = self._points_world(Qt, h)
            sdf_b, sdf_u = self._drum_sdf(pts)
            sdf_self = self._self_sdf(Qt, pts)
            keep = torch.ones(n_fine, self.n_points, self.scene.num_drums,
                              device=self.device, dtype=self.dtype)
            if h == hand and tgt >= 0:
                rows = [self.tip_index]
                if spec.is_shaft_contact():
                    rows += self.shaft_indices
                keep[first_exempt:, rows, tgt] = 0.0
            if h == hand and src >= 0:
                rows = [self.tip_index]
                if shaft_hit_allowed(case.start_drum):
                    rows += self.shaft_indices
                keep[:last_start_exempt, rows, src] = 0.0
            m = keep <= 0.5
            sb = sdf_b.masked_fill(m, float("inf"))
            su = sdf_u.masked_fill(m, float("inf"))
            allsdf = torch.cat(
                [sb.reshape(n_fine, -1), su.reshape(n_fine, -1), sdf_self], dim=1
            )
            finite = allsdf[torch.isfinite(allsdf)]
            stats[h] = dict(
                min_sdf=float(finite.min().item()) if finite.numel() else float("inf"),
                pen_max=float((-allsdf).clamp_min(0.0).max().item()),
                pen_count=int((allsdf < 0.0).sum().item()),
                margin_count=int((allsdf < self.collision_margin).sum().item()),
                self_min=float(sdf_self.min().item()),
            )

        act, oth = stats[hand], stats["left" if hand == "right" else "right"]
        coll_min_all = min(act["min_sdf"], oth["min_sdf"])
        coll_pen_all = max(act["pen_max"], oth["pen_max"])
        coll_cnt_all = act["pen_count"] + oth["pen_count"]
        self_min = min(act["self_min"], oth["self_min"])

        # ---- torque ----------------------------------------------------------
        with torch.no_grad():
            tau = self.rnea.rnea(self._t(Q), self._t(qd), self._t(qdd)).cpu().numpy()
            tau_ff = self.rnea.rnea(
                self._t(Q), self._t(qd), torch.zeros_like(self._t(qdd))
            ).cpu().numpy()
        tau_abs = np.abs(tau)
        tau_ex = np.maximum(0.0, tau_abs - self.tau_limit[None, :])
        tau_frac_matrix = tau_abs / np.maximum(self.tau_limit[None, :], 1e-9)
        pd_err = np.abs(tau - tau_ff) / np.maximum(self.kp[None, :], 1e-9)

        # ---- velocity / joint box --------------------------------------------
        v_ex = np.maximum(0.0, np.abs(qd) - self.v_limit[None, :])
        q_ex_hard = np.maximum(
            np.maximum(0.0, self.q_lo_hard[None, :] - Q),
            np.maximum(0.0, Q - self.q_hi_hard[None, :]),
        )
        q_ex_post = np.maximum(
            np.maximum(0.0, self.q_lo_post[None, :] - Q),
            np.maximum(0.0, Q - self.q_hi_post[None, :]),
        )

        # ---- shape ------------------------------------------------------------
        dQ = np.diff(Q, axis=0)
        per_joint = np.abs(dQ).sum(axis=0)
        net = np.abs(Q[-1] - Q[0])
        sig = np.sign(np.where(np.abs(dQ) > 1e-4, dQ, 0.0))
        rev = 0
        for j in range(sig.shape[1]):
            nz = sig[:, j][sig[:, j] != 0.0]
            rev += int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
        l1 = float(per_joint.sum())
        net_sum = float(net.sum())

        with torch.no_grad():
            Rw, pw = self.fk.frame(self._t(Q), f"{hand}_wrist")
            tips = (pw + (Rw @ self._t(stick_tip_offset(hand)).view(1, 3, 1)).squeeze(-1))
            tips = tips.cpu().numpy()
        tip_len = float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum())

        # ---- acceptance --------------------------------------------------------
        contact_err = float(np.linalg.norm(p_c - p_star))
        speed_err = abs(speed - speed_tgt)
        speed_band = max(thr.speed_err_tol_mps, thr.speed_err_tol_frac * speed_tgt)
        failed = bool(meta.get("failed", False))

        ok_hit = bool((not failed) and on_head)
        ok_hit_tight = bool(ok_hit and contact_err <= thr.contact_pos_tol_m)
        ok_speed = bool((not failed) and speed_err <= speed_band)
        ok_coll = bool(act["pen_count"] == 0)
        ok_coll_margin = bool(act["margin_count"] == 0)
        ok_tau = bool(float(tau_ex.max(initial=0.0)) <= thr.torque_tol_nm)
        ok_vel = bool(float(v_ex.max(initial=0.0)) <= thr.velocity_tol_rads)
        ok_pos = bool(float(q_ex_post.max(initial=0.0)) <= thr.position_tol_rad)

        return CaseScore(
            case_id=int(case.case_id),
            grid=str(case.grid),
            drum=str(case.drum),
            hand=str(hand),
            start_mode=str(case.start_mode),
            N=N,
            horizon_s=float(case.horizon_s),
            strike_speed=float(case.strike_speed),
            contact_sigma=float(sigma),
            solve_s=float(solve_s),
            planner_failed=failed,
            converged=bool(meta.get("converged", not failed)),
            iterations=int(meta.get("iterations", -1)),
            message=str(meta.get("message", "")),
            contact_pos_err_mm=1000.0 * contact_err,
            tip_pos_err_mm=1000.0 * float(np.linalg.norm(p_tip - p_star)),
            on_head=on_head,
            inplane_dist_m=inplane,
            axial_dist_m=axial,
            half_plane_margin_m=hp_margin,
            contact_speed_mps=speed,
            contact_speed_target_mps=speed_tgt,
            contact_speed_err_mps=speed_err,
            contact_dir_cos=float(_unit(v_c) @ n_in),
            into_head_mps=into,
            tangential_mps=tang,
            approach_deg=approach_deg,
            tip_speed_mps=float(np.linalg.norm(v_tip)),
            stick_elev_deg=elev_deg,
            elev_ok=elev_ok,
            start_pos_err_rad=float(np.max(np.abs(Q[0] - case.q_start))),
            start_vel_err_rads=float(np.max(np.abs(qd[1] - case.qdot_start))) if N >= 1 else 0.0,
            coll_min_sdf_m=act["min_sdf"],
            coll_pen_max_m=act["pen_max"],
            coll_pen_count=act["pen_count"],
            coll_margin_count=act["margin_count"],
            coll_min_sdf_all_m=coll_min_all,
            coll_pen_max_all_m=coll_pen_all,
            coll_pen_count_all=coll_cnt_all,
            self_coll_min_sdf_m=self_min,
            tau_max_nm=float(tau_abs.max()),
            tau_excess_max_nm=float(tau_ex.max(initial=0.0)),
            tau_excess_max_frac=float(tau_frac_matrix.max()),
            tau_frac_over=float(np.mean(tau_ex > 0.0)),
            tau_knots_over=int(np.sum(np.any(tau_ex > 0.0, axis=1))),
            pd_track_err_needed_max_rad=float(pd_err.max()),
            vel_excess_max_rads=float(v_ex.max(initial=0.0)),
            vel_frac_over=float(np.mean(v_ex > 0.0)),
            qpos_excess_max_rad=float(q_ex_hard.max(initial=0.0)),
            qpos_posture_excess_max_rad=float(q_ex_post.max(initial=0.0)),
            travel_l1_rad=l1,
            travel_l2_rad=float(np.linalg.norm(dQ, axis=1).sum()),
            travel_net_rad=net_sum,
            travel_waste=(l1 / net_sum if net_sum > 1e-9 else float("nan")),
            travel_wrist_rad=float(per_joint[_WRIST_IDX].sum()),
            reversals=rev,
            jerk_integral=float(np.sum(qddd**2) * dt),
            accel_integral=float(np.sum(qdd**2) * dt),
            tip_path_len_m=tip_len,
            ok_hit=ok_hit,
            ok_hit_tight=ok_hit_tight,
            ok_speed=ok_speed,
            ok_collision=ok_coll,
            ok_collision_margin=ok_coll_margin,
            ok_torque=ok_tau,
            ok_velocity=ok_vel,
            ok_joint_limits=ok_pos,
            accepted=bool(
                ok_hit and ok_speed and ok_coll and ok_tau and ok_vel and ok_pos
            ),
        )


# ======================================================== case generation ====


def _sample_box_start(
    rng: np.random.Generator,
    hand: str,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Mid-range +- ``scale`` * half-range on the waist and the striking arm.

    Mirrors ``bench_trajopt_norrt._sample_q_start``, which samples the reduced
    (waist + one arm) model.  The idle arm is left at URDF neutral -- the pose
    the repo's reduced model locks it at -- so the two grids describe the same
    physical scenario even though this one carries all 17 DoF.
    """
    mid = 0.5 * (q_lo + q_hi)
    half = 0.5 * (q_hi - q_lo)
    q = np.zeros(N_DOF)
    idx = _WAIST_IDX + _ARM_IDX[hand]
    q[idx] = mid[idx] + scale * half[idx] * rng.uniform(-1.0, 1.0, size=len(idx))
    return np.clip(q, q_lo, q_hi)


def _sample_box_velocity(
    rng: np.random.Generator, hand: str, v_limit: np.ndarray, scale: float
) -> np.ndarray:
    v = np.zeros(N_DOF)
    idx = _WAIST_IDX + _ARM_IDX[hand]
    v[idx] = scale * v_limit[idx] * rng.uniform(-1.0, 1.0, size=len(idx))
    return v


def _goal_pose_vector(poses: dict, hand: str, drum: str) -> np.ndarray | None:
    """A 17-DoF pose from ``drum_goal_poses.json`` (idle arm at neutral)."""
    d = poses.get("poses", {}).get(hand, {}).get(drum)
    if not d:
        return None
    q = np.zeros(N_DOF)
    for i, name in enumerate(G1_UPPER_BODY_JOINTS):
        if name in d:
            q[i] = float(d[name])
    return q


def generate_grid(
    *,
    seed: int = 0,
    grid: str = "standard",
    scorer: TrajectoryScorer | None = None,
    layout_path: str | Path | None = None,
    dt: float = CONTROL_DT,
    horizons: Sequence[float] = (0.45, 0.60, 0.80),
    speeds: Sequence[float] = (1.0, 2.0, 3.0),
    repeats: int = 3,
    jitter_rad: float = 0.05,
    start_q_scale: float = 0.35,
    start_v_scale: float = 0.0,
    include_extra_drums: bool = False,
    near_drum_repeats: int = 3,
    near_drum_departure_mps: float = 0.6,
    near_drum_speed_noise: float = 0.15,
    near_drum_jitter_rad: float = 0.06,
    start_id: int = 0,
) -> list[EvalCase]:
    """Build a reproducible list of :class:`EvalCase`.

    ``grid="standard"`` reproduces the repo's 162-case acceptance grid cell for
    cell: 6 drums x 3 horizons (0.45 / 0.60 / 0.80 s) x 3 commanded impact
    speeds (1 / 2 / 3 m/s) x 3 jittered start poses, with the drum -> hand map
    of :data:`DRUM_HAND` (three drums per hand, so both hands are covered
    equally).  ``floor_tom`` is excluded, as it is in the repo grid; pass
    ``include_extra_drums=True`` to add it (189 cases).

    ``grid="near_drum"`` is the training distribution the plan cares about --
    starts *on* another drum, departing from it.  The start pose is that drum's
    canonical terminal pose from the vendored ``drum_goal_poses.json`` plus
    joint jitter, and the start velocity is an inertia-free damped-pseudoinverse
    solution for a tip velocity pointing off that head (noised in direction and
    magnitude).  One case per (target drum, source drum != target, repeat), at
    the middle horizon and speed.

    ``grid="standard+near_drum"`` concatenates the two, renumbering ids.

    Determinism: everything is drawn from ``np.random.default_rng(seed)`` in a
    fixed order, so the same seed gives byte-identical cases.  Rejection
    sampling (collision-free starts) consumes a variable number of draws, which
    is why the two sub-grids are generated with independent child generators
    rather than one shared stream.
    """
    if "+" in grid:
        out: list[EvalCase] = []
        nid = start_id
        for part in grid.split("+"):
            sub = generate_grid(
                seed=seed, grid=part, scorer=scorer, layout_path=layout_path, dt=dt,
                horizons=horizons, speeds=speeds, repeats=repeats, jitter_rad=jitter_rad,
                start_q_scale=start_q_scale, start_v_scale=start_v_scale,
                include_extra_drums=include_extra_drums,
                near_drum_repeats=near_drum_repeats,
                near_drum_departure_mps=near_drum_departure_mps,
                near_drum_speed_noise=near_drum_speed_noise,
                near_drum_jitter_rad=near_drum_jitter_rad,
                start_id=nid,
            )
            out.extend(sub)
            nid += len(sub)
        return out

    sc = scorer or TrajectoryScorer(layout_path=layout_path)
    scene = sc.scene
    layout = str(sc.layout_path)
    q_lo, q_hi = sc.q_lo_post, sc.q_hi_post
    v_limit = sc.v_limit

    hand_of: dict[str, str] = dict(DRUM_HAND)
    if include_extra_drums:
        for i, n in enumerate(scene.names):
            if n in hand_of:
                continue
            hand_of[n] = _EXTRA_DRUM_HAND.get(
                n, "left" if float(scene.head_centers[i][1]) > 0 else "right"
            )
    names = [n for n in scene.names if n in hand_of]

    def collision_free(q: np.ndarray) -> bool:
        Qt = sc._t(q.reshape(1, N_DOF))
        worst = float("inf")
        for h in ("left", "right"):
            pts = sc._points_world(Qt, h)
            b, u = sc._drum_sdf(pts)
            s = sc._self_sdf(Qt, pts)
            worst = min(worst, float(b.min()), float(u.min()), float(s.min()))
        return worst > 0.0

    cases: list[EvalCase] = []
    cid = int(start_id)

    if grid == "standard":
        rng = np.random.default_rng(seed)
        for name in names:
            hand = hand_of[name]
            i = scene.names.index(name)
            for horizon in horizons:
                for speed in speeds:
                    base = None
                    for _ in range(48):
                        cand = _sample_box_start(rng, hand, q_lo, q_hi, start_q_scale)
                        if collision_free(cand):
                            base = cand
                            break
                    if base is None:
                        base = np.clip(
                            _sample_box_start(rng, hand, q_lo, q_hi, start_q_scale), q_lo, q_hi
                        )
                    N = max(int(round(float(horizon) / dt)), 4)
                    for rep in range(int(repeats)):
                        q0 = base
                        if rep:
                            j = np.zeros(N_DOF)
                            idx = _WAIST_IDX + _ARM_IDX[hand]
                            j[idx] = rng.uniform(-jitter_rad, jitter_rad, size=len(idx))
                            q0 = np.clip(base + j, q_lo, q_hi)
                        v0 = (
                            np.zeros(N_DOF) if start_v_scale <= 0.0
                            else _sample_box_velocity(rng, hand, v_limit, start_v_scale)
                        )
                        cases.append(
                            EvalCase(
                                case_id=cid, grid="standard", drum=name, drum_index=i,
                                hand=hand, N=N, dt=dt, strike_speed=float(speed),
                                q_start=q0, qdot_start=v0,
                                strike=strike_from_drum(
                                    time=N * dt, hand=hand, name=name,
                                    center=scene.head_centers[i], axis=scene.axes[i],
                                    radius=float(scene.radii[i]), speed=float(speed),
                                ),
                                start_mode="box", layout=layout,
                            )
                        )
                        cid += 1
        return cases

    if grid == "near_drum":
        from .constants import DRUM_GOAL_POSES_PATH

        rng = np.random.default_rng(seed + 10_000)
        local = Path(layout).parent / "drum_goal_poses.json"
        src_path = local if local.exists() else DRUM_GOAL_POSES_PATH
        if not src_path.exists():
            raise FileNotFoundError(
                f"the near_drum grid needs drum_goal_poses.json (looked at {src_path})"
            )
        poses = json.loads(src_path.read_text())
        horizon = float(horizons[len(horizons) // 2])
        speed = float(speeds[len(speeds) // 2])
        N = max(int(round(horizon / dt)), 4)
        for name in names:
            hand = hand_of[name]
            i = scene.names.index(name)
            for src in names:
                if src == name:
                    continue
                q_src = _goal_pose_vector(poses, hand, src)
                if q_src is None:
                    continue
                j_src = scene.names.index(src)
                out_n = _unit(np.asarray(scene.axes[j_src], float))
                for rep in range(int(near_drum_repeats)):
                    jit = np.zeros(N_DOF)
                    idx = _WAIST_IDX + _ARM_IDX[hand]
                    jit[idx] = rng.uniform(-near_drum_jitter_rad, near_drum_jitter_rad,
                                           size=len(idx))
                    q0 = np.clip(q_src + jit, q_lo, q_hi)
                    # departure tip velocity: off the source head, noised
                    d = out_n + near_drum_speed_noise * rng.normal(size=3)
                    v_tip = float(
                        near_drum_departure_mps * (1.0 + 0.25 * rng.normal())
                    ) * _unit(d)
                    J = sc.tip_jacobian(q0, hand, None)
                    v0 = J.T @ np.linalg.solve(J @ J.T + (0.05**2) * np.eye(3), v_tip)
                    v0 = np.clip(v0, -v_limit, v_limit)
                    cases.append(
                        EvalCase(
                            case_id=cid, grid="near_drum", drum=name, drum_index=i,
                            hand=hand, N=N, dt=dt, strike_speed=speed,
                            q_start=q0, qdot_start=v0,
                            strike=strike_from_drum(
                                time=N * dt, hand=hand, name=name,
                                center=scene.head_centers[i], axis=scene.axes[i],
                                radius=float(scene.radii[i]), speed=speed,
                            ),
                            start_mode="near_drum", start_drum=src, layout=layout,
                        )
                    )
                    cid += 1
        return cases

    raise ValueError(f"unknown grid {grid!r}")


# ================================================================ running ====


def run_eval(
    planner: Planner,
    cases: Iterable[EvalCase],
    *,
    scorer: TrajectoryScorer | None = None,
    verbose: bool = False,
    keep_trajectories: bool = True,
) -> tuple[list[CaseScore], dict[int, np.ndarray]]:
    """Run ``planner`` over ``cases`` and score every result.

    A planner that raises is recorded as a failure with a straight-line
    fallback trajectory, so one bad case cannot lose the rest of the batch.
    """
    sc = scorer or TrajectoryScorer()
    rows: list[CaseScore] = []
    trajs: dict[int, np.ndarray] = {}
    for case in cases:
        t0 = time.perf_counter()
        try:
            res = planner(case)
            wall = time.perf_counter() - t0
            if isinstance(res, PlanResult):
                Q, solve_s, meta = res.Q, res.solve_s, dict(res.meta)
            else:
                Q, solve_s, meta = np.asarray(res, float), wall, {}
            if not np.isfinite(solve_s):
                solve_s = wall
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
            wall = time.perf_counter() - t0
            Q = np.repeat(case.q_start[None, :], case.n_waypoints, axis=0)
            solve_s, meta = wall, {"failed": True, "message": f"{type(exc).__name__}: {exc}"}
        row = sc.score(case, Q, solve_s=solve_s, meta=meta)
        rows.append(row)
        if keep_trajectories:
            trajs[int(case.case_id)] = np.asarray(Q, float)
        if verbose:
            print(
                f"[{row.case_id:04d}] {row.drum:12s} {row.hand[0]} T={row.horizon_s:.2f}s "
                f"v={row.strike_speed:.1f}  pos={row.contact_pos_err_mm:7.1f}mm  "
                f"spd_err={row.contact_speed_err_mps:5.2f}  "
                f"coll={row.coll_pen_max_m*1000:6.1f}mm  "
                f"tau_ex={row.tau_excess_max_nm:7.2f}  "
                f"{'ACC' if row.accepted else '---'}  {row.solve_s*1e3:6.0f}ms",
                flush=True,
            )
    return rows, trajs


# =============================================================== reporting ===


def _p(xs: Sequence[float], q: float) -> float:
    v = [float(x) for x in xs if np.isfinite(x)]
    return float(np.percentile(v, q)) if v else float("nan")


_SUMMARY_METRICS: tuple[tuple[str, str, float], ...] = (
    ("contact_pos_err_mm", "contact pos err (mm)", 1.0),
    ("tip_pos_err_mm", "tip pos err (mm)", 1.0),
    ("contact_speed_err_mps", "impact speed err (m/s)", 1.0),
    ("contact_dir_cos", "impact dir cosine", 1.0),
    ("approach_deg", "approach angle (deg)", 1.0),
    ("coll_min_sdf_m", "min clearance, active (mm)", 1000.0),
    ("coll_pen_max_m", "max penetration, active (mm)", 1000.0),
    ("coll_pen_max_all_m", "max penetration, both (mm)", 1000.0),
    ("tau_excess_max_nm", "torque excess (N.m)", 1.0),
    ("tau_excess_max_frac", "peak torque / limit", 1.0),
    ("vel_excess_max_rads", "velocity excess (rad/s)", 1.0),
    ("qpos_posture_excess_max_rad", "joint-box excess (rad)", 1.0),
    ("travel_l1_rad", "joint travel L1 (rad)", 1.0),
    ("travel_waste", "travel waste (x)", 1.0),
    ("travel_wrist_rad", "wrist travel (rad)", 1.0),
    ("jerk_integral", "jerk integral", 1.0),
    ("accel_integral", "accel integral", 1.0),
    ("pd_track_err_needed_max_rad", "PD track err needed (rad)", 1.0),
    ("solve_s", "solve time (ms)", 1000.0),
)

_ACCEPT_FLAGS: tuple[tuple[str, str], ...] = (
    ("ok_hit", "contact inside the legal hit disk"),
    ("ok_hit_tight", "... and <= 2 cm from the aim point"),
    ("ok_speed", "impact speed in band"),
    ("ok_collision", "no penetration (active hand)"),
    ("ok_collision_margin", "keeps 5 mm margin"),
    ("ok_torque", "no torque-limit excess"),
    ("ok_velocity", "no velocity-limit excess"),
    ("ok_joint_limits", "inside the joint box"),
    ("elev_ok", "stick elevation in band"),
    ("converged", "solver converged"),
    ("planner_failed", "planner reported failure"),
    ("accepted", "ACCEPTED (hit+speed+coll+tau+vel+box)"),
)


def aggregate(rows: Sequence[CaseScore]) -> dict:
    """Mean / median / p90 / max per metric plus the acceptance counts."""
    n = len(rows)
    out: dict = {"n": n, "metrics": {}, "counts": {}, "by_drum": {}}
    if not n:
        return out
    for attr, label, scale in _SUMMARY_METRICS:
        v = [float(getattr(r, attr)) * scale for r in rows]
        fin = [x for x in v if np.isfinite(x)]
        out["metrics"][attr] = dict(
            label=label,
            mean=float(np.mean(fin)) if fin else float("nan"),
            p50=_p(v, 50), p90=_p(v, 90),
            max=float(np.max(fin)) if fin else float("nan"),
            min=float(np.min(fin)) if fin else float("nan"),
        )
    for attr, label in _ACCEPT_FLAGS:
        c = sum(1 for r in rows if bool(getattr(r, attr)))
        out["counts"][attr] = dict(label=label, count=c, n=n)
    for drum in sorted({r.drum for r in rows}):
        sub = [r for r in rows if r.drum == drum]
        out["by_drum"][drum] = dict(
            n=len(sub),
            hand=sub[0].hand,
            accepted=sum(1 for r in sub if r.accepted),
            contact_pos_err_mm_p50=_p([r.contact_pos_err_mm for r in sub], 50),
            speed_err_p50=_p([r.contact_speed_err_mps for r in sub], 50),
            pen_max_mm_p90=_p([r.coll_pen_max_m * 1000 for r in sub], 90),
            tau_excess_p90=_p([r.tau_excess_max_nm for r in sub], 90),
            solve_ms_p50=_p([r.solve_s * 1000 for r in sub], 50),
        )
    return out


def format_summary(rows: Sequence[CaseScore], title: str = "eval summary") -> str:
    agg = aggregate(rows)
    if not agg["n"]:
        return f"{title}: no rows\n"
    L = [f"===== {title} ({agg['n']} cases) =====", "",
         f"{'metric':32s} {'mean':>11s} {'p50':>11s} {'p90':>11s} {'max':>11s}", "-" * 82]
    for attr, m in agg["metrics"].items():
        L.append(
            f"{m['label']:32s} {m['mean']:11.4f} {m['p50']:11.4f} "
            f"{m['p90']:11.4f} {m['max']:11.4f}"
        )
    L += ["", f"{'acceptance check':36s} {'count':>12s}", "-" * 52]
    for attr, c in agg["counts"].items():
        L.append(f"{c['label']:36s} {c['count']:6d}/{c['n']:<6d}")
    L += ["", f"{'drum':14s} {'hand':>5s} {'n':>4s} {'acc':>6s} {'pos p50 mm':>11s} "
              f"{'spd p50':>9s} {'pen p90 mm':>11s} {'tau p90':>9s} {'ms p50':>8s}", "-" * 86]
    for drum, d in agg["by_drum"].items():
        L.append(
            f"{drum:14s} {d['hand']:>5s} {d['n']:4d} {d['accepted']:3d}/{d['n']:<2d} "
            f"{d['contact_pos_err_mm_p50']:11.1f} {d['speed_err_p50']:9.3f} "
            f"{d['pen_max_mm_p90']:11.2f} {d['tau_excess_p90']:9.2f} {d['solve_ms_p50']:8.0f}"
        )
    return "\n".join(L) + "\n"


# ============================================================== persistence ==


def write_csv(path: str | Path, rows: Sequence[CaseScore]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [f.name for f in fields(CaseScore)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def save_run(
    out_dir: str | Path,
    cases: Sequence[EvalCase],
    rows: Sequence[CaseScore],
    trajs: dict[int, np.ndarray],
    *,
    prefix: str = "run",
    meta: dict | None = None,
) -> dict[str, Path]:
    """Persist cases (JSON), scores (CSV) and trajectories (NPZ).

    The NPZ is the important one: it lets a later cost/metric change be
    evaluated against *stored* baseline trajectories instead of re-running the
    (nondeterministic, ~1 s/case) solver.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_cases = out_dir / f"{prefix}_cases.json"
    p_csv = out_dir / f"{prefix}_per_case.csv"
    p_npz = out_dir / f"{prefix}_trajectories.npz"
    p_cases.write_text(
        json.dumps(
            {"meta": meta or {}, "cases": [c.to_dict() for c in cases]}, indent=1
        )
    )
    write_csv(p_csv, rows)
    np.savez_compressed(
        p_npz,
        _meta=np.array(json.dumps(meta or {})),
        _case_ids=np.array(sorted(trajs)),
        **{f"case_{int(k):05d}": np.asarray(v, float) for k, v in trajs.items()},
    )
    return {"cases": p_cases, "csv": p_csv, "npz": p_npz}


def load_run(
    out_dir: str | Path, *, prefix: str = "run"
) -> tuple[list[EvalCase], dict[int, np.ndarray], dict]:
    """Inverse of :func:`save_run` -- cases + trajectories + meta, for re-scoring."""
    out_dir = Path(out_dir)
    blob = json.loads((out_dir / f"{prefix}_cases.json").read_text())
    cases = [EvalCase.from_dict(d) for d in blob["cases"]]
    z = np.load(out_dir / f"{prefix}_trajectories.npz")
    trajs = {
        int(k.split("_")[1]): np.asarray(z[k], float)
        for k in z.files
        if k.startswith("case_")
    }
    return cases, trajs, blob.get("meta", {})
