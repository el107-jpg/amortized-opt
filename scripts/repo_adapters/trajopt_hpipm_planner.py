#!/usr/bin/env python3
"""REPO-ONLY adapter: the repo's HPIPM trajopt SQP as a harness planner.

.. warning::

   **This is the one file under ``amortized/`` that imports the main repo.**
   It is the analogue of the conformance-test exception documented in
   ``amortized/README.md``: everything in ``amortized_opt/`` is standalone, and
   this adapter exists solely to produce the reference baseline the amortized
   optimizer must beat.  If you hand the folder to someone without the repo,
   this file (and only this file) stops working -- ``import_error()`` says so
   politely and the rest of the harness is unaffected.

What it does
------------

The repo's offline benches (``scripts/bench_trajopt_offline.py``,
``bench_trajopt_hpipm.py``, ``test_trajopt_hpipm_acceptance.py``) all solve a
**single-arm** problem: the Pinocchio model contains the 3 waist joints plus the
7 joints of the striking arm, and the idle arm is not in the model at all --
Pinocchio locks it at the URDF neutral pose.  That is feasibility-review Gap 7,
and it is the expert's *native* problem; nothing in the repo plans both arms and
the waist jointly.

The harness scores all 17 DoF, so this adapter embeds the 10-DoF solution back
into a 17-DoF trajectory with the idle arm held at the value it has in
``case.q_start`` (which the grid generator sets to URDF neutral, exactly the
pose the repo model locks it at).  The idle arm is therefore *constant* over the
trajectory: it contributes no travel, no velocity and no torque, but its
collision geoms are still scored.  Read the baseline with that asymmetry in
mind: **trajopt gets its easier, native problem.**

Waist policy (``waist_aim=``)
----------------------------

* ``waist_aim=False`` -- **v1 baseline behaviour.** The 3 waist joints are
  ordinary optimization variables inside the full URDF box, starting wherever
  the case's start pose puts them, with no reference.  This is what
  ``bench_trajopt_hpipm.py`` / ``test_trajopt_hpipm_acceptance.py`` do
  (``_make_arm_robot(..., freeze_waist=False)``).  Note it is *not* the
  "frozen at mid-range" convention of ``bench_trajopt_offline.py`` -- that
  script defines ``_make_arm_robot`` twice and the second, non-freezing
  definition shadows the first, so its docstring is stale.
* ``waist_aim=True`` -- **v2 baseline, and what the repo's own pipelines do.**
  Waist yaw follows the aim law; pitch and roll are held. Mirrors
  ``teacher/adapters_trajopt.py::_waist_reference`` / ``_with_waist_bc`` and
  ``play_control_theory.py::_update_waist_aim``, both of which:

  1. pick a yaw target with ``waist_aim.aim_yaw({hand: tip_target}, cfg,
     yaw_current=<current yaw>)`` -- a damped, gated partial turn toward the
     drum, not a square-up;
  2. build ``waist_aim.waist_reference_block(...)``, a leading quintic yaw ramp
     that commits within ``max_move_s`` and then HOLDS, settled
     ``settle_wp`` waypoints before contact;
  3. hand it to the solver as ``TrajOptConfig.waist_reference``, which Block 2
     pins hard (``trajopt_hpipm/ocp.py``, ``sqp.py::_apply_waist_reference``);
  4. source ``v_start``'s waist rows from
     ``waist_aim.waist_start_velocity(ref, dt)`` -- Block 1 pins
     ``(q1-q0)/dt == v_start`` as a hard equality, so waist rows that disagree
     with the reference make the QP primal-infeasible on iteration 1.

  One deliberate departure: the teacher passes ``setpoint=zeros`` for
  pitch/roll because its rollout states always have zero pitch/roll.  This grid
  samples them, so the adapter passes ``setpoint=q_start``'s own pitch/roll
  instead.  "Pitch and roll stay frozen" is then frozen *at the start pose*
  rather than at zero, which keeps ``ref[0] == q_start[waist]`` exactly (the
  Block 1/Block 2 consistency the point above is about) and -- more importantly
  -- leaves the evaluation cases bit-identical between v1 and v2, so the
  comparison is paired.

``refine_yaw=True`` adds the teacher's extra step (``pipeline.refine_yaw``):
score the proposed yaw and 0.0/0.5x/1.5x/2x of it with the strike-IK oracle and
keep whichever minimises worst-arm-joint travel.  0 is always a candidate, so
the aimed plan is never worse than the unaimed one *by that measure*.  The live
MPC path does not do this; the teacher/expert-data path does.  Off by default
because it costs several extra strike-IK solves per case.

Seed policy
-----------

Left at the ``TrajOptConfig`` default, which is already the apex-in-cylinder
seed the plan asks for: ``seed_mode="apex"`` + ``seed_apex_first=True`` means
``sqp._resolve_apex_first_goal`` solves the apex against the START pose alone
(tip inside a vertical cylinder ``seed_apex_height`` = 0.30 m above the strike
position, radius 0.15 m, half-height 0.10 m, closest in joint space), then takes
the terminal strike pose NEAREST that apex and writes it into
``StrikeSpec.cached_goal_q``.  ``validate()`` touches no ``seed_*`` field.

Config policy
-------------

Only the *problem* is overridden -- horizon, joint set, obstacles, the strike,
the boundary conditions and (in v2) the waist reference.  Every cost weight,
slack penalty, seed setting and solver tolerance is left at the current
``TrajOptConfig`` dataclass default, and ``validate()`` is allowed to do its
usual silent zeroing under ``simplified_cost=True`` (feasibility-review Gap 8).
``describe()`` reports the **post-validate** values, because the declared
dataclass defaults are not what the solver sees: ``validate()`` zeroes
``w_strike_position``, ``w_posture``, ``w_strike_posture``, ``w_stick_elev``,
``strike_min_velocity_fraction`` and several more.

The strike handed to the solver is built from ``case.strike``, not re-derived,
so the solver optimizes for precisely the target the harness scores against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
#: <repo>/amortized/scripts/repo_adapters/this.py  ->  <repo>
REPO_ROOT = _THIS.parents[3]
_CONTROL = REPO_ROOT / "source" / "g1_drummer" / "g1_drummer" / "control"

_AMORTIZED = _THIS.parents[2]
if str(_AMORTIZED) not in sys.path:
    sys.path.insert(0, str(_AMORTIZED))

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF  # noqa: E402
from amortized_opt.eval_harness import EvalCase, PlanResult  # noqa: E402

_WAIST = tuple(j for j in G1_UPPER_BODY_JOINTS if j.startswith("waist_"))
_LEFT = tuple(j for j in G1_UPPER_BODY_JOINTS if j.startswith("left_"))
_RIGHT = tuple(j for j in G1_UPPER_BODY_JOINTS if j.startswith("right_"))
#: The reduced joint set each single-arm solve uses, in the repo's own order.
ARM_JOINTS: dict[str, tuple[str, ...]] = {
    "left": _WAIST + _LEFT,
    "right": _WAIST + _RIGHT,
}
#: index in the 17-DoF vector of each reduced joint, per hand
ARM_INDEX: dict[str, list[int]] = {
    h: [G1_UPPER_BODY_JOINTS.index(n) for n in names] for h, names in ARM_JOINTS.items()
}


def repo_available() -> bool:
    return _CONTROL.is_dir()


def _import_waist_aim():
    """``control/waist_aim.py``, imported the repo's own offline way.

    Backend-agnostic (it only builds reference blocks), so it is shared by the
    OSQP teacher and the HPIPM live path alike.
    """
    if not repo_available():
        raise ImportError(f"repo control package not found at {_CONTROL}")
    if str(_CONTROL) not in sys.path:
        sys.path.insert(0, str(_CONTROL))
    import waist_aim

    return waist_aim


#: ``teacher/adapters_trajopt.py::_WAIST_SETTLE_S`` -- stationary torso through
#: the strike, so impact speed is the arm's alone.
WAIST_SETTLE_S = 0.06
#: ``teacher/adapters_trajopt.py::_WAIST_MAX_MOVE_S`` -- the ramp leads and
#: commits inside this window, then holds.
WAIST_MAX_MOVE_S = 0.2
#: ``pipeline.refine_yaw`` candidate multipliers, 0 first so a yaw that buys
#: nothing is never taken.
YAW_REFINE_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)


def _import_trajopt(backend: str = "hpipm"):
    """Import the repo backend without dragging in Isaac Lab via ``g1_drummer``."""
    if not repo_available():
        raise ImportError(
            f"repo control package not found at {_CONTROL}; this adapter is the "
            "one repo-coupled file in amortized/ and cannot run standalone"
        )
    if str(_CONTROL) not in sys.path:
        sys.path.insert(0, str(_CONTROL))
    if backend == "hpipm":
        import trajopt_hpipm as mod
    elif backend == "osqp":
        import trajopt as mod
    else:
        raise ValueError(f"backend must be 'hpipm' or 'osqp', got {backend!r}")
    return mod


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _with_under_drums(mod, drums, depth: float, clearance: float = 0.01):
    """Add the under-drum no-go volumes, exactly as every repo bench does."""
    if depth <= 0:
        return list(drums)
    out = list(drums)
    for d in drums:
        axis = _unit(np.asarray(d.axis, float))
        center = np.asarray(d.center, float)
        out.append(
            mod.CylinderObstacle(
                center=(center - clearance * axis) - 0.5 * depth * axis,
                axis=axis,
                radius=float(d.radius),
                height=depth,
                name=f"{d.name}__under",
                head_frame=False,
            )
        )
    return out


class TrajoptPlanner:
    """Callable planner wrapping one repo SQP backend.

    Holds one ``G1UpperBody`` per hand (building it costs an ACM sample sweep,
    so it must not be rebuilt per case) and mutates ``rob.cfg`` per solve, which
    is what the repo benches do to keep ``strike_ignore_pairs()`` in sync
    without rebuilding geometry.
    """

    def __init__(
        self,
        *,
        backend: str = "hpipm",
        layout: str | Path | None = None,
        under_drum_depth: float = 0.08,
        allow_rrt: bool = False,
        max_iterations: int | None = None,
        dt: float = 0.02,
        overrides: dict | None = None,
        waist_aim: bool = False,
        refine_yaw: bool = False,
        waist_settle_s: float = WAIST_SETTLE_S,
        waist_max_move_s: float | None = WAIST_MAX_MOVE_S,
        waist_aim_damping: float | None = None,
        waist_aim_yaw_limit: float | None = None,
        waist_aim_engage_in_deg: float | None = None,
        waist_aim_engage_out_deg: float | None = None,
    ):
        self.mod = _import_trajopt(backend)
        self.backend = backend
        self.allow_rrt = bool(allow_rrt)
        self.max_iterations = max_iterations
        self.overrides = dict(overrides or {})
        self.waist_aim = bool(waist_aim)
        self.refine_yaw = bool(refine_yaw) and self.waist_aim
        self.waist_settle_s = float(waist_settle_s)
        self.waist_max_move_s = (
            None if waist_max_move_s is None else float(waist_max_move_s)
        )
        self._wa = None
        self.aim_cfg = None
        if self.waist_aim:
            self._wa = _import_waist_aim()
            kw = {}
            if waist_aim_damping is not None:
                kw["damping"] = float(waist_aim_damping)
            if waist_aim_yaw_limit is not None:
                kw["yaw_limit"] = float(waist_aim_yaw_limit)
            if waist_aim_engage_in_deg is not None:
                kw["engage_in"] = self._wa.engage_from_deg(waist_aim_engage_in_deg)
            if waist_aim_engage_out_deg is not None:
                kw["engage_out"] = self._wa.engage_from_deg(waist_aim_engage_out_deg)
            # Defaults are the shipped WaistAimConfig: damping 0.3 (lowered from
            # 0.7 on 2026-08-04), yaw limit 100 deg, 40 deg in / 40 deg out gate.
            self.aim_cfg = self._wa.WaistAimConfig(**kw)

        mod = self.mod
        layout_path = Path(layout) if layout else (
            REPO_ROOT / "configs" / "assets" / "drum_kit_layout.yaml"
        )
        # drums_from_layout subtracts PELVIS_IN_ENV by default -> pelvis frame,
        # the same frame the harness scores in.
        self.drums = {d.name: d for d in mod.drums_from_layout(layout_path)}
        self.obstacles = _with_under_drums(
            mod, list(self.drums.values()), float(under_drum_depth)
        )
        self.robots = {}
        for hand in ("left", "right"):
            cfg = mod.TrajOptConfig(
                N=8, dt=float(dt), joint_names=ARM_JOINTS[hand], obstacles=self.obstacles
            )
            self.robots[hand] = mod.G1UpperBody(cfg)
        #: column of waist_yaw within the reduced joint vector (0 -- waist first)
        self.waist_idx = {
            h: [i for i, n in enumerate(names) if n.startswith("waist_")]
            for h, names in ARM_JOINTS.items()
        }
        self.yaw_col = {
            h: [n for n in ARM_JOINTS[h] if n.startswith("waist_")].index(
                "waist_yaw_joint"
            )
            for h in ARM_JOINTS
        }

    # -------------------------------------------------------------- waist ---
    def _yaw_target(self, case: EvalCase, q0: np.ndarray) -> float:
        """Commanded waist yaw for this strike (``waist_aim.aim_yaw``).

        Single-handed: exactly the ``aim_yaw_for`` call the teacher makes with
        one active goal.  ``yaw_current`` is the start pose's own yaw, which is
        what both pipelines pass -- it is the gate's and the deadband's variable,
        and passing 0 instead would ask "how far off centre is the drum", a
        different and, for a gate, wrong question.
        """
        w = self.waist_idx[case.hand]
        yaw_now = float(np.asarray(q0, float)[w][self.yaw_col[case.hand]])
        targets = {case.hand: np.asarray(case.strike.position, float)}
        return float(self._wa.aim_yaw(targets, cfg=self.aim_cfg, yaw_current=yaw_now))

    def _waist_reference(
        self, case: EvalCase, q0: np.ndarray, yaw_target: float, N: int
    ) -> np.ndarray:
        """``(N+1, 3)`` waist reference: quintic yaw ramp, pitch/roll held.

        Anchored at ``q0``'s own waist so ``ref[0] == q0[waist]`` by
        construction -- see the module docstring on the pitch/roll setpoint.
        """
        names = ARM_JOINTS[case.hand]
        w = self.waist_idx[case.hand]
        waist0 = np.asarray(q0, float)[w]
        setpoint = waist0.copy()  # pitch/roll frozen at the start pose
        return self._wa.waist_reference_block(
            names,
            yaw_start=float(waist0[self.yaw_col[case.hand]]),
            yaw_target=float(yaw_target),
            n_waypoints=int(N) + 1,
            setpoint=setpoint,
            dt=float(case.dt),
            settle_wp=max(int(round(self.waist_settle_s / float(case.dt))), 0),
            max_move_s=self.waist_max_move_s,
        )

    def _with_waist_bc(self, case: EvalCase, v0: np.ndarray, ref) -> np.ndarray:
        """``v_start`` with waist rows sourced from the reference's own slope."""
        if ref is None:
            return v0
        rob = self.robots[case.hand]
        v = np.asarray(v0, float).copy()
        v[self.waist_idx[case.hand]] = self._wa.waist_start_velocity(
            ref, float(case.dt)
        )
        return np.clip(v, -np.asarray(rob.v_limit, float), np.asarray(rob.v_limit, float))

    def _strike_travel(self, case: EvalCase, q0, v0, yaw: float) -> float:
        """Worst arm-joint travel to the strike IK pose at waist yaw ``yaw``.

        ``pipeline.refine_yaw``'s scoring oracle, reimplemented against the
        HPIPM ``DrumTrajOpt`` (the teacher's own ``strike_travel`` lives on
        ``TrajOptArmTeacher``, which imports the OSQP backend unconditionally --
        feasibility-review Gap 9 -- and would silently make this an OSQP number).
        """
        rob = self.robots[case.hand]
        ref = self._waist_reference(case, q0, yaw, int(case.N))
        cfg = self._build_cfg(case, q0, self._with_waist_bc(case, v0, ref), ref)
        prev = rob.cfg
        try:
            rob.cfg = cfg
            q_goal = self.mod.DrumTrajOpt(cfg, robot=rob)._strike_goal_q(q0)
        except Exception:  # noqa: BLE001 - an unreachable candidate just loses
            return float("inf")
        finally:
            rob.cfg = prev
        if q_goal is None:
            return float("inf")
        arm = [i for i in range(len(ARM_JOINTS[case.hand]))
               if i not in self.waist_idx[case.hand]]
        return float(np.max(np.abs((np.asarray(q_goal, float) - q0)[arm])))

    def _refine_yaw(self, case: EvalCase, q0, v0, psi: float) -> float:
        """Back the proposed yaw off until it demonstrably beats not turning."""
        if abs(float(psi)) < 1e-9:
            return 0.0
        lim = float(self.aim_cfg.yaw_limit)
        cands: list[float] = []
        for m in YAW_REFINE_MULTIPLIERS:
            c = float(np.clip(m * float(psi), -lim, lim))
            if not any(abs(c - x) < 1e-6 for x in cands):
                cands.append(c)
        best, best_score = 0.0, float("inf")
        for c in cands:
            s = self._strike_travel(case, q0, v0, c)
            if np.isfinite(s) and s < best_score - 1e-6:
                best, best_score = c, s
        return float(best)

    # ------------------------------------------------------------------------
    def _build_cfg(self, case: EvalCase, q0: np.ndarray, v0: np.ndarray, waist_ref=None):
        mod = self.mod
        s = case.strike
        hand = case.hand
        n = len(ARM_JOINTS[hand])
        kw = dict(
            N=int(case.N),
            dt=float(case.dt),
            joint_names=ARM_JOINTS[hand],
            obstacles=self.obstacles,
            strikes=[
                mod.StrikeSpec(
                    float(s.time),
                    hand,
                    np.asarray(s.position, float),
                    np.asarray(s.velocity, float),
                    drum=s.drum,
                    hit_radius=s.resolved_hit_radius(),
                    half_plane_dir=(
                        None if s.half_plane_dir is None
                        else np.asarray(s.half_plane_dir, float)
                    ),
                    disk_center_offset=(
                        None if s.disk_center_offset is None
                        else np.asarray(s.disk_center_offset, float)
                    ),
                    contact_sigma=s.contact_sigma,
                )
            ],
            hit_radius=s.resolved_hit_radius(),
            q_start=q0,
            q_goal=q0.copy(),  # soft terminal; the seed uses strike IK
            v_start=v0,
            v_goal=None,
            a_start=np.zeros(n),
            a_goal=np.zeros(n),
        )
        if waist_ref is not None:
            kw["waist_reference"] = np.asarray(waist_ref, float)
        if self.max_iterations is not None:
            kw["max_iterations"] = int(self.max_iterations)
        kw.update(self.overrides)
        return mod.TrajOptConfig(**kw)

    def __call__(self, case: EvalCase) -> PlanResult:
        import time

        hand = case.hand
        idx = ARM_INDEX[hand]
        rob = self.robots[hand]

        q0 = rob.clamp(np.asarray(case.q_start, float)[idx])
        v0 = np.clip(
            np.asarray(case.qdot_start, float)[idx],
            -np.asarray(rob.v_limit, float),
            np.asarray(rob.v_limit, float),
        )
        t0 = time.perf_counter()
        waist_ref = None
        yaw_target = float("nan")
        yaw_proposed = float("nan")
        if self.waist_aim:
            yaw_proposed = self._yaw_target(case, q0)
            yaw_target = (
                self._refine_yaw(case, q0, v0, yaw_proposed)
                if self.refine_yaw
                else yaw_proposed
            )
            waist_ref = self._waist_reference(case, q0, yaw_target, int(case.N))
            v0 = self._with_waist_bc(case, v0, waist_ref)

        cfg = self._build_cfg(case, q0, v0, waist_ref)
        rob.cfg = cfg  # keeps strike_ignore_pairs() in sync (repo bench idiom)
        opt = self.mod.DrumTrajOpt(cfg, robot=rob)
        result = opt.solve(None, allow_rrt=self.allow_rrt)
        solve_s = time.perf_counter() - t0

        Q10 = np.asarray(result.Q, float)
        Q17 = np.repeat(np.asarray(case.q_start, float)[None, :], Q10.shape[0], axis=0)
        Q17[:, idx] = Q10
        return PlanResult(
            Q=Q17,
            solve_s=solve_s,
            meta=dict(
                failed=bool(result.solver_failed),
                converged=bool(result.converged),
                iterations=int(result.iterations),
                message=str(result.message),
                used_rrt_seed=bool(getattr(result, "used_rrt_seed", False)),
                used_bridged_seed=bool(getattr(result, "used_bridged_seed", False)),
                max_slack=float(getattr(result, "max_slack", float("nan"))),
                backend=self.backend,
                waist_aim=self.waist_aim,
                yaw_proposed_rad=yaw_proposed,
                yaw_target_rad=yaw_target,
                yaw_start_rad=float(q0[self.waist_idx[hand]][self.yaw_col[hand]]),
            ),
        )


def make_planner(**kwargs) -> TrajoptPlanner:
    """Entry point for ``run_eval.py --planner`` (``module:make_planner``)."""
    return TrajoptPlanner(**kwargs)


_DESCRIBE_KEYS = (
    "seed_mode", "seed_apex_first", "seed_apex_height", "seed_apex_radius",
    "seed_apex_half_height", "seed_apex_weight", "start_velocity_mode",
    "simplified_cost", "simplified_w_v", "w_v", "w_a", "w_j", "w_ke", "w_posture",
    "w_strike_posture", "w_strike_goal_posture", "w_strike_position",
    "w_strike_velocity", "w_strike_velocity_normal", "w_strike_velocity_tangential",
    "w_stick_elev", "w_hit", "w_drum_proximity", "strike_min_velocity_fraction",
    "hit_tight_radius", "max_iterations", "enable_torque_limits", "torque_scale",
    "enable_accel_limits", "accel_limit", "collision_margin", "d_on", "d_off",
    "acm_samples", "stick_elev_min_deg", "stick_elev_max_deg", "limit_margin",
    "waist_pin_tol",
)


def describe(planner: TrajoptPlanner) -> dict:
    """Provenance recorded into the run's metadata.

    Reports both the DECLARED dataclass defaults and the values that survive
    ``validate()``.  They differ: ``simplified_cost`` is on by default and
    silently zeroes ~12 weights, including ``w_strike_position`` (10 -> 0),
    ``w_posture``, ``w_strike_posture``, ``w_stick_elev`` and
    ``strike_min_velocity_fraction``.  Recording only the declared values --
    which is what v1 of this baseline did -- misreports what the solver
    actually optimized (feasibility-review Gap 8).
    """
    mod = planner.mod
    declared = mod.TrajOptConfig(joint_names=ARM_JOINTS["right"])
    # A minimal but *complete* problem, so validate() runs the same branches it
    # runs in a real solve.
    n = len(ARM_JOINTS["right"])
    effective = mod.TrajOptConfig(
        N=8, dt=0.02, joint_names=ARM_JOINTS["right"], obstacles=planner.obstacles,
        strikes=[mod.StrikeSpec(0.16, "right", np.zeros(3), np.array([0.0, 0.0, -1.0]),
                                drum="snare", hit_radius=0.18)],
        q_start=np.zeros(n), q_goal=np.zeros(n), v_start=np.zeros(n),
        a_start=np.zeros(n), a_goal=np.zeros(n),
    )
    try:
        effective.validate()
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        effective_dump = {"validate_failed": repr(exc)}
    else:
        effective_dump = {k: _jsonable(getattr(effective, k, None)) for k in _DESCRIBE_KEYS}

    out = {
        "backend": planner.backend,
        "allow_rrt": planner.allow_rrt,
        "max_iterations_override": planner.max_iterations,
        "overrides": planner.overrides,
        "under_drum_obstacles": len(planner.obstacles) - len(planner.drums),
        "arm_joints": {h: list(v) for h, v in ARM_JOINTS.items()},
        "waist_policy": "aim" if planner.waist_aim else "free (no reference)",
        "refine_yaw": planner.refine_yaw,
        "trajopt_declared_defaults": {
            k: _jsonable(getattr(declared, k, None)) for k in _DESCRIBE_KEYS
        },
        "trajopt_effective_after_validate": effective_dump,
    }
    if planner.waist_aim:
        c = planner.aim_cfg
        out["waist_aim"] = {
            "damping": float(c.damping),
            "yaw_limit_rad": float(c.yaw_limit),
            "engage_in_rad": None if c.engage_in is None else float(c.engage_in),
            "engage_out_rad": None if c.engage_out is None else float(c.engage_out),
            "az_pref_rad": float(c.az_pref),
            "deadband_rad": float(c.deadband),
            "settle_s": planner.waist_settle_s,
            "max_move_s": planner.waist_max_move_s,
            "pitch_roll_setpoint": "held at q_start (teacher uses zeros)",
        }
    return out


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
