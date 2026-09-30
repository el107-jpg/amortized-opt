"""Planners for the section-2 evaluation grid.

Two callables with the harness signature ``plan(case) -> PlanResult``:

* :func:`quintic_case_planner` -- the phase-1 *target* generator turned into a
  planner: the same per-arm accelerating strike seed to the goal-pose library plus
  the waist-aim yaw ramp that :mod:`amortized_opt.warmstart` builds.  This is
  the reference line the warm-started transformer should reach (and no more --
  it is exactly what phase 1 imitates).
* :class:`ModelPlanner` -- the transformer as an MPC planner.  One context
  encode plus ``iterations`` weight-shared refinement passes, started either
  from the noised quintic seed (``init="quintic"``) or from a zero-motion hold
  (``init="hold"``).

Frames.  The harness is **pelvis**-rooted; the model's context tokens are built
by :func:`~amortized_opt.rhythm.build_context_inputs` in the **env** frame (the
:mod:`amortized_opt.mpc_gen` default, so phase-2 data collates identically).
Everything crossing that boundary here goes through ``PELVIS_IN_ENV``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .constants import G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV
from .eval_harness import EvalCase, PlanResult
from .fk import ChainFK
from .ik import correct_quintic_init_arms
from .limits import urdf_position_limits
from .idle_home import resolve_idle_home
from .rhythm import HANDS, DrumKit, HandGoal, GoalSnapshot, build_context_inputs
from .tokenizer import build_waypoint_masks
from .waist_aim import WaistAimConfig
from .warmstart import GoalPoseLibrary, InitNoise, seed_waist_pr_to_sp, warmstart_trajectory
from .warmstart_data import bucket_for

__all__ = [
    "ModelPlanner",
    "batched_model_plans",
    "case_goal_snapshot",
    "precomputed_planner",
    "quintic_case_planner",
]


def case_goal_snapshot(case: EvalCase, kit: DrumKit) -> GoalSnapshot:
    """The one-strike :class:`~amortized_opt.rhythm.GoalSnapshot` for a grid case.

    The idle hand is inactive; there are no future hits.  ``case.strike`` is in
    the pelvis frame, so the position is lifted into the env frame the context
    builder expects.
    """
    goal = {h: HandGoal.inactive() for h in HANDS}
    goal[case.hand] = HandGoal(
        active=True,
        drum=int(case.drum_index),
        position=np.asarray(case.strike.position, float) + PELVIS_IN_ENV,
        velocity=np.asarray(case.strike.velocity, float),
        time_to_hit=float(case.N) * float(case.dt),
        drum_name=str(case.drum),
        loudness=1.0,
    )
    return GoalSnapshot(t_now=0.0, goal=goal, future={h: [] for h in HANDS})


def _warmstart(
    case: EvalCase,
    library: GoalPoseLibrary,
    horizon: int,
    *,
    waist_cfg: WaistAimConfig | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    idle_home=None,
    waist_upright_ease_wp: int | None = None,
):
    goals = {h: None for h in HANDS}
    goals[case.hand] = (str(case.drum), float(case.N) * float(case.dt))
    aim = {case.hand: np.asarray(case.strike.position, float) + PELVIS_IN_ENV}
    return warmstart_trajectory(
        case.q_start, case.qdot_start,
        horizon=int(horizon), goals=goals, library=library, aim_targets=aim,
        dt=float(case.dt), joint_names=joint_names, waist_cfg=waist_cfg,
        q_bounds=urdf_position_limits(joint_names),
        idle_home=idle_home,
        waist_upright_ease_wp=waist_upright_ease_wp,
    )


def quintic_case_planner(
    *,
    library: GoalPoseLibrary | None = None,
    waist_cfg: WaistAimConfig | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    waist_upright_ease_wp: int | None = None,
):
    """The raw strike + waist-aim seed, as a harness planner (legacy name)."""
    lib = library or GoalPoseLibrary(joint_names=tuple(joint_names))

    def plan(case: EvalCase) -> PlanResult:
        ws = _warmstart(case, lib, int(case.N), waist_cfg=waist_cfg, joint_names=joint_names,
                        waist_upright_ease_wp=waist_upright_ease_wp)
        return PlanResult(Q=np.asarray(ws.q, float), meta={"converged": True})

    return plan


@dataclass
class ModelPlanner:
    """The trained decoder as a grid planner.

    ``init="quintic"`` feeds the (optionally noised) warm-start seed;
    ``init="hold"`` feeds a zero-motion hold at the start pose, which is the
    hardest possible initialization and is reported separately.
    """

    model: torch.nn.Module
    device: str = "cuda:1"
    iterations: int = 1
    init: str = "quintic"
    noise: InitNoise | None = None
    seed: int = 0
    buckets: tuple[int, ...] = (25, 50, 100)
    num_drums: int = 8
    num_future: int = 3
    library: GoalPoseLibrary | None = None
    kit: DrumKit | None = None
    waist_cfg: WaistAimConfig | None = None
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS
    amp_dtype: torch.dtype | None = None
    #: Idle-arm park posture; ``"default"`` matches the training roller.  Every
    #: grid case has one idle hand, so this decides whether the model is scored
    #: on the distribution it was trained on.  ``None`` = the old behaviour.
    idle_home: object = "default"
    #: Waist pitch/roll ease-to-upright window of the quintic seed.  MUST match
    #: the ``ScenarioConfig.waist_upright_ease_wp`` the checkpoint was trained
    #: with (eval scripts read it off ``ck["scenario_cfg"]``), or the model is
    #: scored on an init distribution it never saw.  ``None`` = hold the lean.
    waist_upright_ease_wp: int | None = None

    def __post_init__(self) -> None:
        if self.init not in ("quintic", "hold"):
            raise ValueError(f"init must be quintic|hold, got {self.init!r}")
        self.library = self.library or GoalPoseLibrary(joint_names=tuple(self.joint_names))
        self.kit = self.kit or DrumKit.from_layout()
        if not hasattr(self.idle_home, "arm_pose"):
            self.idle_home = resolve_idle_home(
                self.idle_home, joint_names=tuple(self.joint_names)
            )
        self._rng = np.random.default_rng(self.seed)
        self.model.eval()
        self._fk: ChainFK | None = None

    def _chain_fk(self, device, dtype) -> ChainFK:
        if (
            self._fk is None
            or self._fk.device != torch.device(device)
            or self._fk.dtype != dtype
        ):
            self._fk = ChainFK(device=device, dtype=dtype)
        return self._fk

    def _init_traj(self, case: EvalCase, H: int) -> np.ndarray:
        if self.init == "hold":
            return np.repeat(np.asarray(case.q_start, float)[None, :], H, axis=0)
        ws = _warmstart(case, self.library, H, waist_cfg=self.waist_cfg,
                        joint_names=self.joint_names, idle_home=self.idle_home,
                        waist_upright_ease_wp=self.waist_upright_ease_wp)
        out = ws.init_traj
        if self.noise is not None:
            out = self.noise.apply(
                out, self._rng, hit_index=ws.hit_index,
                q_bounds=urdf_position_limits(self.joint_names),
                joint_names=self.joint_names,
            )
        return np.asarray(out, float)

    @torch.no_grad()
    def __call__(self, case: EvalCase) -> PlanResult:
        dev = torch.device(self.device)
        H = bucket_for(int(case.N), tuple(self.buckets))
        snap = case_goal_snapshot(case, self.kit)
        ctx = build_context_inputs(
            self.kit, snap, num_drums=self.num_drums, num_future=self.num_future,
            frame="env", device=dev, dtype=torch.float32,
            idle_home=self.idle_home,
            yaw_now=np.asarray(case.q_start, float)[
                self.joint_names.index("waist_yaw_joint")
            ],
            waist_cfg=self.waist_cfg,
        )
        t = lambda a: torch.as_tensor(np.asarray(a, float), device=dev, dtype=torch.float32)
        hl = float(case.N) * float(case.dt) if case.hand == "left" else 0.0
        hr = float(case.N) * float(case.dt) if case.hand == "right" else 0.0
        masks = build_waypoint_masks(
            t([hl]), t([hr]), max_horizon=H, dt=float(case.dt),
            active_left=torch.tensor([case.hand == "left"], device=dev),
            active_right=torch.tensor([case.hand == "right"], device=dev),
            joint_names=self.joint_names,
        )
        init = t(self._init_traj(case, H))[None, :, :] * masks.elem_mask
        state = torch.cat([t(case.q_start), t(case.qdot_start)])[None, :]

        amp = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if (self.amp_dtype is not None and dev.type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        core = self.model.module if hasattr(self.model, "module") else self.model
        with amp:
            enc = core.encode_context(ctx)
            terminal_seed = core.terminal_initializer is not None
            if terminal_seed and self.init != "hold":
                init = core.initial_trajectory(ctx, state, masks).to(init.dtype)
            if self.init != "hold":
                if enc.waist_pr_sp is not None:
                    init = seed_waist_pr_to_sp(
                        init, state[:, : len(self.joint_names)],
                        state[:, len(self.joint_names):], enc.waist_pr_sp,
                        joint_names=self.joint_names, dt=float(case.dt),
                        ease_wp=self.waist_upright_ease_wp,
                    )
                if not terminal_seed:
                    hit = torch.full((1, 2), -1, device=dev, dtype=torch.long)
                    hit[0, 0 if case.hand == "left" else 1] = int(case.N)
                    aims = torch.zeros(1, 2, 3, device=dev, dtype=init.dtype)
                    aims[0, 0 if case.hand == "left" else 1] = t(
                        np.asarray(case.strike.position, float) + PELVIS_IN_ENV
                    )
                    init = correct_quintic_init_arms(
                        self._chain_fk(dev, init.dtype), init,
                        state[:, : len(self.joint_names)], hit, aims, steps=1,
                        q_bounds=urdf_position_limits(self.joint_names),
                        joint_names=self.joint_names,
                        qd0=state[:, len(self.joint_names):], dt=float(case.dt),
                    )
            out = core.refine(init, state, enc, masks, iterations=int(self.iterations))
        traj = out.traj[0, : int(case.N)].float().cpu().numpy()
        Q = np.concatenate([np.asarray(case.q_start, float)[None, :], traj], axis=0)
        return PlanResult(
            Q=Q,
            meta={"converged": True, "iterations": int(self.iterations),
                  "message": f"init={self.init} H={H}"},
        )


# ------------------------------------------------------------ batched eval --
def _bucket_groups(cases, buckets):
    groups: dict[int, list] = {}
    for c in cases:
        groups.setdefault(bucket_for(int(c.N), tuple(buckets)), []).append(c)
    return groups


@torch.no_grad()
def batched_model_plans(
    model: torch.nn.Module,
    cases,
    *,
    device: str = "cuda:1",
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    buckets: tuple[int, ...] = (25, 50, 100),
    num_drums: int = 8,
    num_future: int = 3,
    iterations: int = 1,
    init: str = "quintic",
    noise: InitNoise | None = None,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
    waist_cfg: WaistAimConfig | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    chunk: int = 512,
    idle_home="default",
    waist_upright_ease_wp: int | None = None,
) -> tuple[dict[int, np.ndarray], float]:
    """Solve a whole grid in a handful of batched forward passes.

    :class:`ModelPlanner` is one eager batch-1 call per case (~15 ms); a rolling
    eval every 2 k steps cannot afford 243 of those plus the compile guards they
    trip.  This groups the cases by horizon bucket, builds one batch per group
    and runs the same encode + ``refine`` path, so the numbers are identical up
    to batching non-determinism.

    THE IDLE ARM.  Every grid case is single-hand -- one hand strikes, the other
    is idle -- so the grid is *entirely* made of the states the idle-arm terms
    shape.  ``idle_home`` therefore defaults to ``"default"``, the same
    resolution the training roller uses: the idle arm is seeded parked and its
    context token is filled.  Passing ``None`` reproduces the pre-2026-08-12
    evaluator, which held the idle arm in place and emitted no idle token --
    i.e. it scored the model on a distribution it was not trained on, and
    ``best.pt`` is gated on this number.

    Returns ``({case_id: (N+1, 17) trajectory}, seconds of solve wall clock)``.
    """
    dev = torch.device(device)
    kit = kit or DrumKit.from_layout()
    library = library or GoalPoseLibrary(joint_names=tuple(joint_names))
    home = (
        resolve_idle_home(idle_home, joint_names=tuple(joint_names))
        if not hasattr(idle_home, "arm_pose") else idle_home
    )
    rng = np.random.default_rng(seed)
    q_bounds = urdf_position_limits(joint_names)
    core = model.module if hasattr(model, "module") else model
    core.eval()
    fk = ChainFK(device=dev, dtype=torch.float32)

    out: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for H, group in _bucket_groups(cases, buckets).items():
        for c0 in range(0, len(group), int(chunk)):
            sub = group[c0 : c0 + int(chunk)]
            B = len(sub)
            snaps = [case_goal_snapshot(c, kit) for c in sub]
            yaw_i = joint_names.index("waist_yaw_joint")
            ctx = build_context_inputs(
                kit, snaps, num_drums=num_drums, num_future=num_future,
                frame="env", device=dev, dtype=torch.float32,
                idle_home=home,
                yaw_now=np.asarray(
                    [np.asarray(c.q_start, float)[yaw_i] for c in sub]
                ),
                waist_cfg=waist_cfg,
            )
            t = lambda a: torch.as_tensor(  # noqa: E731
                np.asarray(a, float), device=dev, dtype=torch.float32
            )
            hl = t([c.N * c.dt if c.hand == "left" else 0.0 for c in sub])
            hr = t([c.N * c.dt if c.hand == "right" else 0.0 for c in sub])
            masks = build_waypoint_masks(
                hl, hr, max_horizon=H, dt=float(sub[0].dt),
                active_left=torch.tensor([c.hand == "left" for c in sub], device=dev),
                active_right=torch.tensor([c.hand == "right" for c in sub], device=dev),
                joint_names=joint_names,
            )
            inits = np.zeros((B, H, len(joint_names)))
            for i, c in enumerate(sub):
                if init == "hold":
                    inits[i] = np.repeat(np.asarray(c.q_start, float)[None, :], H, axis=0)
                else:
                    ws = _warmstart(c, library, H, waist_cfg=waist_cfg,
                                    joint_names=joint_names, idle_home=home,
                                    waist_upright_ease_wp=waist_upright_ease_wp)
                    x = ws.init_traj
                    if noise is not None:
                        x = noise.apply(x, rng, hit_index=ws.hit_index,
                                        q_bounds=q_bounds, joint_names=joint_names)
                    inits[i] = x
            init_t = t(inits) * masks.elem_mask
            state = torch.cat(
                [t([c.q_start for c in sub]), t([c.qdot_start for c in sub])], dim=-1
            )
            amp = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if (amp_dtype is not None and dev.type == "cuda")
                else torch.autocast(device_type="cuda", enabled=False)
            )
            with amp:
                enc = core.encode_context(ctx)
                terminal_seed = core.terminal_initializer is not None
                if terminal_seed and init != "hold":
                    init_t = core.initial_trajectory(ctx, state, masks).to(init_t.dtype)
                if init != "hold":
                    if enc.waist_pr_sp is not None:
                        n = len(joint_names)
                        init_t = seed_waist_pr_to_sp(
                            init_t, state[:, :n], state[:, n:], enc.waist_pr_sp,
                            joint_names=joint_names, dt=float(sub[0].dt),
                            ease_wp=waist_upright_ease_wp,
                        )
                    if not terminal_seed:
                        n = len(joint_names)
                        hit = torch.full((B, 2), -1, device=dev, dtype=torch.long)
                        aims = torch.zeros(B, 2, 3, device=dev, dtype=init_t.dtype)
                        for i, c in enumerate(sub):
                            hi = 0 if c.hand == "left" else 1
                            hit[i, hi] = int(c.N)
                            aims[i, hi] = t(
                                np.asarray(c.strike.position, float) + PELVIS_IN_ENV
                            )
                        init_t = correct_quintic_init_arms(
                            fk,
                            init_t, state[:, :n], hit, aims, steps=1,
                            q_bounds=q_bounds, joint_names=joint_names,
                            qd0=state[:, n:], dt=float(sub[0].dt),
                        )
                res = core.refine(init_t, state, enc, masks, iterations=int(iterations))
            traj = res.traj.float().cpu().numpy()
            for i, c in enumerate(sub):
                out[int(c.case_id)] = np.concatenate(
                    [np.asarray(c.q_start, float)[None, :], traj[i, : int(c.N)]], axis=0
                )
    return out, time.perf_counter() - t0


def precomputed_planner(plans: dict[int, np.ndarray], solve_s: float = float("nan")):
    """Wrap a ``{case_id: trajectory}`` dict as a harness planner."""

    def plan(case: EvalCase) -> PlanResult:
        return PlanResult(
            Q=np.asarray(plans[int(case.case_id)], float),
            solve_s=float(solve_s),
            meta={"converged": True, "message": "batched"},
        )

    return plan
