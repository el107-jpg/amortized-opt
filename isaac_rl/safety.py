"""Per-step command projection: a batched safety backstop for the Isaac RL loop.

Adapted from ``source/g1_drummer/g1_drummer/control/safety/qp_filter.py``
(:class:`SafetyFilter`, driven by ``scripts/rsl_rl/play.py::attach_safety_filter``).

Why this file exists
--------------------

The repo already has a correct per-step safety projection.  It is not usable as
written inside a ``num_envs=256`` training loop, for three measured reasons:

1. **It is single-config and CPU-bound.**  ``SafetyFilter.project`` takes
   ``(q, qd, q_cmd)`` as three ``(17,)`` float64 numpy vectors and returns one
   :class:`FilterResult`.  ``attach_safety_filter`` drives it with a literal
   ``for e in range(cmd_np.shape[0])`` Python loop, round-tripping the whole
   batch through ``.cpu().double().numpy()`` first.  Measured on this box
   2.7 ms/env on near-ready poses and ~7 ms/env with the pool full (16 active
   pairs), i.e. **0.7-1.8 s of wall clock per control tick at B=256** against a
   20 ms tick (``amortized_opt.constants.CONTROL_DT``).  That is a 35-90x
   real-time overrun and it is not a tuning problem -- it is one OSQP solve,
   one coal query set and one Python loop iteration per env.

2. **Its collision rows come from Pinocchio + coal.**  Every row is a
   ``trajopt.robot.G1UpperBody.closest_pairs`` query (coal signed distances and
   gradients on the real meshes), reached via
   ``safety.scene.build_drum_scene``, which imports ``trajopt`` and therefore
   Pinocchio.  Building that scene costs ~0.35 s and the import prints the
   ``osqp_guard`` banner; neither belongs in an Isaac training process, and
   Isaac's ``pip_prebundle`` shadows ``osqp`` in a way the guard itself calls
   unusable (see DECISIONS in the accompanying report).

3. **The main repo does not use the QP as its training-time signal either.**
   In-loop, the repo scores clearance with the torch-only
   ``control/safety/sdf_torch.py``, vendored here as
   :class:`amortized_opt.sdf.SdfPenalty` -- batched, GPU-resident,
   autograd-clean, no host sync.  The QP appears only in ``play.py`` behind
   ``--safety-filter``.

And the plan agrees.  ``docs/amortized_optimizer_plan.md``, "Features for
effective final usage as an mpc / Safety":

    If the final solve does not produce a valid trajectory, we will take that
    trajectory and use a standard HPIPM QP solver to correct the trajectory
    [...] **This will not be included in the IsaacSim RL training as it's
    supposed to be an option of last resort in real world situations.**

So the split here is deliberate:

* ``mode="box"`` -- the **training default**.  A pure-torch projection onto the
  admissible set ``amortized_opt`` already defines: the URDF/posture joint box
  (:func:`amortized_opt.limits.urdf_position_limits`), a per-joint slew bound
  ``eta_vel * v_rating * dt`` against the previous *command*
  (:func:`amortized_opt.limits.isaac_velocity_limits`), and a projected-gradient
  clearance step on :class:`~amortized_opt.sdf.SdfPenalty`.  No host sync, no
  Python loop over envs, no Pinocchio.  This is a *backstop*, not a solver: it
  keeps the executed command inside the box and pushes shallow penetrations out;
  it makes no viability (braking) guarantee, which is exactly the guarantee the
  QP's braking rows buy and the reason ``qp`` still exists below.
* ``mode="qp"`` -- the **eval / last-resort** path.  The real thing: the repo's
  OSQP filter, per env, on the CPU, run only on the envs ``box`` mode flagged
  (or all of them with ``qp_all=True``).  Use it for a scripted evaluation, a
  hardware dry-run, or to measure how much the cheap backstop misses.  It is
  imported lazily inside a ``try``/``except`` and falls back to ``box`` with one
  log line when Pinocchio / the repo is absent, so this module imports and runs
  with no Isaac, no Pinocchio and no repo checkout present.
* ``mode="off"`` -- identity, for A/B.

The clearance step
------------------

:class:`~amortized_opt.sdf.SdfPenalty` is built here with ``power=1``, so the
penalty is **metres of penetration summed over violated pairs** rather than the
squared hinge the reward uses.  That matters for the step rule: with a linear
penalty the Polyak step

    dq = -relax * (pen / ||g||^2) * g,      g = d(pen)/dq

is the exact Newton step to ``pen = 0`` for the local linearization, so it is
self-scaling in rad -- no step-size constant to tune per joint.  A squared hinge
would collapse the gradient as the penetration closes and need a hand-tuned
gain.  Each step is re-clamped into the box-and-slew interval, which makes the
iteration a projected gradient method on that intersection.

The step is gated per row by ``pen > sdf_threshold``, evaluated **branch-free**
(``torch.where``, no ``.any()``); the whole batch pays for the FK, but the
alternative -- compacting the violating rows -- costs a device->host sync, which
is the exact thing ``amortized_opt.sdf`` was vendored to remove (see its
docstring on the ``if bool(valid.any())`` sync it deleted).

Cost, measured at B=256 on an RTX 5090 (cuda:1), one clearance step:

    eager                       ~55 ms/call   -- and *flat* from B=64 to B=2048,
                                                 i.e. entirely kernel-launch /
                                                 autograd-node bound, not FLOP
                                                 bound (FK fwd alone is 4 ms,
                                                 penalty fwd 8 ms, the other
                                                 ~38 ms is the FK backward's
                                                 node-by-node launch overhead)
    torch.compile default        ~12 ms/call
    torch.compile reduce-overhead ~3.5 ms/call  (CUDA graphs; the plan's own
                                                 "Computational Optimization
                                                 techniques" entry)

End to end at the defaults (box + slew + 2 compiled clearance steps), B=256:
**8.9 ms/call**, of which the box and slew stages are 0.6 ms; the clearance
step is essentially the whole bill.  Two steps removed 72% of the summed
penetration depth on a stress batch shoved 15 cm into the kit (13.82 m of
summed depth over 186/256 rows -> 3.92 m over 109 rows; 4 steps -> 1.67 m).
It is a backstop that walks the command out, not a solver that lands on the
boundary -- ``sdf_max_step`` deliberately caps how far one tick may move.

55 ms does not fit in a 20 ms tick, so ``sdf_compile="reduce-overhead"`` is the
default, with an automatic one-time fallback to eager if compilation or the
first call raises.  Tensors read out of the compiled region are cloned before
use because CUDA-graph outputs live in a pool that the next replay overwrites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF, default_layout_path
from amortized_opt.limits import isaac_velocity_limits, urdf_position_limits
from amortized_opt.sdf import DrumSceneSpec, SdfPenalty

__all__ = ["BatchedSafetyFilter", "SafetyConfig"]

_LOG = logging.getLogger(__name__)

#: ``<repo>/amortized/isaac_rl/safety.py`` -> ``<repo>``.  Only ever used inside
#: the lazy QP import; the module is fine when this path does not exist.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_CONTROL = _REPO_ROOT / "source" / "g1_drummer" / "g1_drummer" / "control"


# ------------------------------------------------------------------ config --
@dataclass
class SafetyConfig:
    """Knobs for :class:`BatchedSafetyFilter`."""

    #: ``"box"`` (training default, pure torch), ``"qp"`` (per-env CPU OSQP, the
    #: plan's last resort) or ``"off"`` (identity).  ``"qp"`` degrades to
    #: ``"box"`` at construction time if the repo / Pinocchio is unavailable.
    mode: str = "box"
    #: Extra clearance demanded on top of the geometry's own margins (m).  Same
    #: name and same 5 mm default as ``qp_filter.SafetyFilter.margin``; in box
    #: mode it is the :class:`~amortized_opt.sdf.SdfPenalty` hinge margin, in QP
    #: mode it is handed straight to the repo filter.
    margin: float = 0.005
    #: Distance (m) inside which a collision pair earns a QP row.  **QP mode
    #: only** -- box mode has no row culling, its hinge activates at ``margin``.
    d_activate: float = 0.10
    #: Slew bound as a fraction of the per-joint speed rating: the command may
    #: move at most ``eta_vel * isaac_velocity_limits() * dt`` rad per tick.
    #: 1.0 = the full rating (a hard-limit backstop, not a comfort limit); the
    #: cost hinges use ``limits.ETA_VELOCITY = 0.70`` for the *comfortable*
    #: band, which is a different question and deliberately not the default
    #: here -- a backstop that clips inside the machine's real envelope would
    #: silently cap the policy's strike speed.
    eta_vel: float = 1.0
    #: Control period the slew bound is built for (s).  The amortized stack runs
    #: a 50 Hz waypoint grid (``amortized_opt.constants.CONTROL_DT``); pass the
    #: live ``env.step_dt`` if the Isaac task disagrees.
    dt: float = CONTROL_DT
    #: Projected-gradient clearance steps per call.  0 disables the SDF stage
    #: (box + slew only).  Each step costs one SdfPenalty fwd+bwd; see the
    #: module docstring for the measured per-step cost.
    sdf_steps: int = 2
    #: Penalty (m of summed penetration) above which a row gets a clearance
    #: step.  Small but nonzero so numerically-zero rows do not divide by an
    #: all-but-zero gradient.
    sdf_threshold: float = 1.0e-4
    #: Per-joint cap on one clearance step (rad).  Guards the Polyak step when
    #: ``||g||`` is tiny because the contact normal is nearly orthogonal to
    #: every joint axis (a near-singular direction), where the unclipped Newton
    #: step is arbitrarily large.
    sdf_max_step: float = 0.05
    #: Over-relaxation on the Polyak step.  1.0 lands on the linearized
    #: boundary; >1 overshoots outward, which converges faster on curved
    #: geometry at the cost of a bigger correction.
    sdf_relax: float = 1.0
    #: Include the torso capsule / head sphere self-collision terms in the
    #: clearance step (``SdfPenalty(include_self=...)``).
    sdf_include_self: bool = True
    #: ``"reduce-overhead"`` | ``"default"`` | ``"off"``.  See the module
    #: docstring: eager is ~55 ms/call at B=256 and launch-bound, CUDA graphs
    #: take it to ~3.5 ms.  Any failure falls back to eager with one log line.
    sdf_compile: str = "reduce-overhead"
    #: Shrink the URDF position box by this much on both sides (rad), mirroring
    #: ``trajopt.robot``'s ``limit_margin``.
    limit_margin: float = 0.0
    #: Apply ``G1_DRUM_POSTURE_LIMITS_RAD`` (elbow-down / palm-out) on top of the
    #: URDF box, as ``trajopt.robot`` does by default.
    posture_limits: bool = True
    #: A command component must move by more than this (rad) to count as
    #: "changed" for ``engaged`` and the per-mechanism counters.
    tol: float = 1.0e-6
    #: QP mode: solve every env instead of only the ones box mode flagged.
    #: Costs ~7 ms x B; only for scripted evaluation.
    qp_all: bool = False
    #: QP mode: pooled collision slots (``SafetyFilter.k_pool``).
    qp_k_pool: int = 16
    #: QP mode: include the braking / viability rows.  These are the guarantee
    #: box mode does *not* reproduce.
    qp_use_braking: bool = True
    #: QP mode: fraction of the derived accel budget the braking rows may spend.
    qp_eta_brake: float = 0.5
    #: QP mode: include self-collision rows.
    qp_include_self: bool = True
    #: QP mode: extra OSQP settings passed through to ``SafetyFilter``.
    #: ``None`` keeps the repo's own defaults.  Measured (B=8, near-ready poses,
    #: 0.45 rad typical correction): running every env through ONE shared OSQP
    #: instance instead of one instance per env moves the answer by 8.4e-3 rad
    #: with the repo default ``adaptive_rho=True``, and by only 4.1e-4 rad with
    #: ``{"adaptive_rho": False}`` -- rho is solver state that survives
    #: ``update()``, so it is a second cross-env channel beyond ``_z_prev``.
    #: Pass ``{"adaptive_rho": False}`` when reproducibility across batch
    #: composition matters more than convergence speed.
    qp_osqp_settings: dict | None = None


# ----------------------------------------------------------------- filter --
class BatchedSafetyFilter:
    """Project a batch of commanded joint targets onto the admissible set.

    ``project(cmd, q, qd)`` returns ``(safe_cmd, engaged)`` and is meant to be
    called once per control tick, on the GPU, with ``cmd`` the policy's (or the
    MPC's) 17-DoF position target.  The filter keeps the previous *command* --
    not the previous measured ``q`` -- as the slew reference, because that is
    what the plant is actually chasing; call :meth:`reset` with the envs that
    Isaac just reset so their slew reference restarts at the measured pose.
    """

    def __init__(
        self,
        cfg: SafetyConfig,
        *,
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        layout_path: str | Path | None = None,
    ):
        self.cfg = cfg
        self.joint_names = tuple(joint_names)
        self.n = len(self.joint_names)
        self.device = torch.device(device)
        self.dtype = dtype
        self.layout_path = Path(layout_path) if layout_path is not None else default_layout_path()

        mode = str(cfg.mode).lower()
        if mode not in ("box", "qp", "off"):
            raise ValueError(f"mode must be 'box', 'qp' or 'off', got {cfg.mode!r}")
        #: The mode actually running -- ``"qp"`` demotes itself to ``"box"`` if
        #: the repo filter cannot be built.
        self.mode = mode

        lo, hi = urdf_position_limits(
            self.joint_names,
            limit_margin=float(cfg.limit_margin),
            posture_limits=bool(cfg.posture_limits),
        )
        self._q_lower = self._t(lo)
        self._q_upper = self._t(hi)
        #: Per-tick slew allowance (rad), broadcast over the batch.
        self._dq_max = self._t(
            float(cfg.eta_vel) * isaac_velocity_limits(self.joint_names) * float(cfg.dt)
        )

        self._scene: DrumSceneSpec | None = None
        self._sdf: SdfPenalty | None = None
        self._pen_fn = None
        self._compile_failed = False
        if self.mode != "off" and int(cfg.sdf_steps) > 0:
            self._scene = DrumSceneSpec.from_layout(self.layout_path)
            self._sdf = SdfPenalty(
                self._scene,
                margin=float(cfg.margin),
                power=1,  # metres of penetration -> self-scaling Polyak step
                include_self=bool(cfg.sdf_include_self),
                device=self.device,
                dtype=self.dtype,
            )
            self._pen_fn = self._build_pen_fn()

        # Lazily-sized batch state (B is not known until the first call).
        self._prev_cmd: torch.Tensor | None = None
        #: [engaged, box, slew, sdf] env-hits, accumulated on device so the hot
        #: path never syncs; :attr:`stats` is what pays for the ``.item()``.
        self._counts = torch.zeros(4, dtype=torch.int64, device=self.device)
        self._calls = 0
        self._envs_seen = 0
        self._qp_solves = 0
        self._qp_active = 0
        self._qp_fallback = 0

        # QP backend, built eagerly so the fallback is known before training
        # starts rather than at the first near-collision.
        self._qp = None
        self._qp_z_prev: list[np.ndarray | None] = []
        self._qp_ignore_fn = None
        self._qp_drum_names: tuple[str, ...] = ()
        if self.mode == "qp":
            if not self._build_qp():
                self.mode = "box"

    # ---------------------------------------------------------- construction --
    def _t(self, arr) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr, dtype=np.float64).tolist(), device=self.device, dtype=self.dtype)

    def _build_pen_fn(self):
        """Return ``(x, tgt_l, tgt_r) -> (B,)`` penalty, compiled if asked."""
        sdf = self._sdf

        def eager(x, tl, tr):
            return sdf(x, target_drum_left=tl, target_drum_right=tr)

        mode = str(self.cfg.sdf_compile).lower()
        if mode in ("off", "none", "eager") or not hasattr(torch, "compile"):
            return eager
        try:
            return torch.compile(eager, mode=None if mode == "default" else mode)
        except Exception as exc:  # pragma: no cover - depends on the torch build
            _LOG.warning("[safety] torch.compile unavailable (%s); running eager SDF", exc)
            self._compile_failed = True
            return eager

    def _build_qp(self) -> bool:
        """Import and construct the repo's OSQP filter.  False = fall back.

        Everything repo-facing lives in this method: the module must import with
        no Isaac, no Pinocchio and no repo checkout, and ``mode="box"`` must
        never touch this path.
        """
        try:
            import sys

            if not _REPO_CONTROL.is_dir():
                raise ImportError(f"repo control package not found at {_REPO_CONTROL}")
            if str(_REPO_CONTROL) not in sys.path:
                # Flat layout, the same trick amortized/scripts/repo_adapters
                # uses: ``control/`` on sys.path so ``safety.qp_filter``'s
                # ``from control_rate import CONTROL_DT`` resolves without
                # importing ``g1_drummer`` (which would drag in Isaac Lab).
                sys.path.insert(0, str(_REPO_CONTROL))
            self._ensure_good_osqp()
            from safety.qp_filter import SafetyFilter  # type: ignore[import-not-found]
            from safety.scene import build_drum_scene, strike_ignore_pairs  # type: ignore[import-not-found]

            rob = build_drum_scene(str(self.layout_path))
            if int(rob.n) != self.n:
                raise ValueError(f"repo robot has {rob.n} DoF, filter has {self.n}")
            self._qp = SafetyFilter(
                rob,
                dt=float(self.cfg.dt),
                margin=float(self.cfg.margin),
                d_activate=float(self.cfg.d_activate),
                k_pool=int(self.cfg.qp_k_pool),
                use_braking=bool(self.cfg.qp_use_braking),
                eta_brake=float(self.cfg.qp_eta_brake),
                include_self=bool(self.cfg.qp_include_self),
                osqp_settings=dict(self.cfg.qp_osqp_settings or {}) or None,
            )
            self._qp_ignore_fn = strike_ignore_pairs
            # Drum id -> name.  DrumSceneSpec.from_layout sorts by the layout's
            # ``id`` so row i is drum id i, which is the same indexing the
            # target_drum_* exemptions use; the names are the layout keys, which
            # is exactly what the repo's obstacles are called.
            scene = self._scene if self._scene is not None else DrumSceneSpec.from_layout(self.layout_path)
            self._qp_drum_names = tuple(scene.names)
            return True
        except Exception as exc:
            _LOG.warning(
                "[safety] mode='qp' unavailable (%s: %s); falling back to mode='box'. "
                "The QP needs pinocchio+coal via trajopt.robot.G1UpperBody and the repo "
                "checkout; box mode needs neither.",
                type(exc).__name__,
                exc,
            )
            self._qp = None
            return False

    @staticmethod
    def _ensure_good_osqp() -> None:
        """Pin a non-prebundle ``osqp`` before ``qp_filter`` imports one.

        Measured with this repo's own conda + ``setup_conda_env.sh`` preamble:
        ``import osqp`` inside ``qp_filter`` resolves to **0.6.5** out of
        ``.../omni.isaac.core_archive/pip_prebundle/``, because Isaac's setup
        script puts the prebundle on ``PYTHONPATH`` ahead of everything else,
        while the good copy on this box is 1.1.3 in user site-packages.  With
        the guard in place the same import lands on 1.1.3 and the banner goes
        away (verified).  ``trajopt`` prints that banner and refuses to
        solve on 0.6.x ("silently contaminated every box-generated dataset").
        The safety QP is a far smaller problem than the SQP that banner is
        about, but running the last-resort backstop on the copy the repo calls
        unusable is not a trade worth making silently, so we do what
        ``trajopt/osqp_guard.py`` tells Isaac-side entry points to do: load the
        guard standalone by path (importing ``trajopt`` would import osqp first
        and lose the race) and let it mask the prebundle for one import.

        Best-effort: any failure leaves whatever osqp the environment resolves.
        """
        import importlib.util
        import sys

        if "osqp" in sys.modules:
            return  # already bound; masking cannot undo an import
        guard_path = _REPO_CONTROL / "trajopt" / "osqp_guard.py"
        if not guard_path.is_file():
            return
        try:
            spec = importlib.util.spec_from_file_location("_amortized_osqp_guard", guard_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            osqp = mod.ensure_osqp()
            err = mod.version_error(osqp)
            if err:
                _LOG.warning("[safety] %s", err)
        except Exception as exc:  # pragma: no cover - guard is advisory
            _LOG.debug("[safety] osqp guard unavailable (%s)", exc)

    # ---------------------------------------------------------- batch state --
    def _ensure_batch(self, B: int, q: torch.Tensor) -> None:
        if self._prev_cmd is None or self._prev_cmd.shape[0] != B:
            # First tick (or a resize): seed the slew reference at the measured
            # pose so step 0 is not spuriously slew-limited.
            self._prev_cmd = q.detach().to(device=self.device, dtype=self.dtype).clone()
            self._qp_z_prev = [None] * B

    def reset(self, env_ids: torch.Tensor | None, q: torch.Tensor) -> None:
        """Restart the slew reference (and the QP warm start) for ``env_ids``.

        ``q`` is either the full ``(B, 17)`` measured pose or the ``(len(ids), 17)``
        rows for ``env_ids``.  ``env_ids=None`` resets everything.
        """
        q = q.detach().to(device=self.device, dtype=self.dtype)
        if self._prev_cmd is None:
            self._ensure_batch(q.shape[0], q)
            if env_ids is None:
                return
        if env_ids is None:
            self._prev_cmd = q.clone()
            self._qp_z_prev = [None] * q.shape[0]
            return
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1).long()
        rows = q[ids] if q.shape[0] == self._prev_cmd.shape[0] else q
        self._prev_cmd[ids] = rows
        if self._qp_z_prev:
            for e in ids.detach().cpu().tolist():  # eval-only path; sync is fine
                self._qp_z_prev[int(e)] = None

    # ---------------------------------------------------------------- stats --
    @property
    def stats(self) -> dict:
        """Cumulative counters.  Syncs once (``.tolist()``) -- call off the hot path."""
        engaged, box, slew, sdf = (int(v) for v in self._counts.detach().cpu().tolist())
        envs = max(self._envs_seen, 1)
        return {
            "mode": self.mode,
            "requested_mode": str(self.cfg.mode).lower(),
            "calls": self._calls,
            "env_steps": self._envs_seen,
            "engaged": engaged,
            "engaged_frac": engaged / envs,
            "box": box,
            "slew": slew,
            "sdf": sdf,
            "qp_solves": self._qp_solves,
            "qp_active": self._qp_active,
            "qp_fallback": self._qp_fallback,
            "sdf_compiled": self._sdf is not None
            and str(self.cfg.sdf_compile).lower() not in ("off", "none", "eager")
            and not self._compile_failed,
        }

    def reset_stats(self) -> None:
        self._counts.zero_()
        self._calls = 0
        self._envs_seen = 0
        self._qp_solves = 0
        self._qp_active = 0
        self._qp_fallback = 0

    # ------------------------------------------------------------ SDF stage --
    def _split_targets(self, target_drum, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, 2)`` -> per-hand ``(B,)`` long tensors, ``-1`` = no exemption.

        ``(B,)`` is accepted and broadcast to both hands (the single-target
        convenience form); anything else is an error rather than a guess.
        """
        none = torch.full((B,), -1, dtype=torch.long, device=self.device)
        if target_drum is None:
            return none, none.clone()
        t = torch.as_tensor(target_drum, device=self.device).long()
        if t.dim() == 1:
            return t.reshape(B).clone(), t.reshape(B).clone()
        if t.dim() == 2 and t.shape[1] == 2:
            return t[:, 0].contiguous(), t[:, 1].contiguous()
        raise ValueError(f"target_drum must be (B,) or (B, 2), got {tuple(t.shape)}")

    def _pen_and_grad(
        self, x: torch.Tensor, tl: torch.Tensor, tr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(pen, dpen/dx)``, both detached and cloned out of any graph pool."""
        with torch.enable_grad():
            leaf = x.detach().requires_grad_(True)
            try:
                pen = self._pen_fn(leaf, tl, tr)
            except Exception as exc:  # compile blew up on the first real call
                if self._compile_failed:
                    raise
                _LOG.warning("[safety] compiled SDF failed (%s); reverting to eager", exc)
                self._compile_failed = True
                self._pen_fn = lambda a, b, c: self._sdf(a, target_drum_left=b, target_drum_right=c)
                pen = self._pen_fn(leaf, tl, tr)
            (grad,) = torch.autograd.grad(pen.sum(), leaf)
        # CUDA-graph outputs live in a pool the next replay overwrites; clone so
        # nothing downstream can read a stale buffer.
        return pen.detach().clone(), grad.detach().clone()

    def _clearance_steps(
        self,
        x: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        tl: torch.Tensor,
        tr: torch.Tensor,
    ) -> torch.Tensor:
        """Projected-gradient descent on the penetration depth, inside ``[lo, hi]``."""
        cfg = self.cfg
        max_step = float(cfg.sdf_max_step)
        thresh = float(cfg.sdf_threshold)
        relax = float(cfg.sdf_relax)
        for _ in range(int(cfg.sdf_steps)):
            pen, grad = self._pen_and_grad(x, tl, tr)
            # Polyak / Newton step for the linearized penalty: pen + g.dq = 0.
            gn2 = (grad * grad).sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            step = -(relax * pen.unsqueeze(-1) / gn2) * grad
            step = step.clamp(-max_step, max_step)
            active = (pen > thresh).unsqueeze(-1)  # branch-free gate, no sync
            x = torch.where(active, torch.clamp(x + step, lo, hi), x)
        return x

    # -------------------------------------------------------------- project --
    def project(
        self,
        cmd: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        *,
        target_drum: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,17) safe command, (B,) bool ``engaged``.

        ``cmd`` / ``q`` / ``qd`` are ``(B, 17)`` in ``joint_names`` order.
        ``target_drum`` is ``(B, 2)`` drum ids ``[left, right]`` (``-1`` = the
        hand has no upcoming strike); the hand's *tip* is then exempted from its
        own target drum, mirroring ``trajopt``'s ``strike_ignore_pairs`` so a
        legitimate strike is not filtered away.

        The returned command is **detached**: this is a runtime backstop, not a
        differentiable layer.  The differentiable clearance signal the policy is
        supposed to learn from is the ``SdfPenalty`` term in the reward.
        """
        if cmd.dim() != 2 or cmd.shape[1] != self.n:
            raise ValueError(f"cmd must be (B, {self.n}), got {tuple(cmd.shape)}")
        B = cmd.shape[0]
        cmd_d = cmd.detach().to(device=self.device, dtype=self.dtype)

        if self.mode == "off":
            self._calls += 1
            self._envs_seen += B
            return cmd, torch.zeros(B, dtype=torch.bool, device=cmd.device)

        q_d = q.detach().to(device=self.device, dtype=self.dtype)
        qd_d = qd.detach().to(device=self.device, dtype=self.dtype)
        self._ensure_batch(B, q_d)
        tol = float(self.cfg.tol)

        # --- 1. joint box ---------------------------------------------------
        lo_q = self._q_lower.unsqueeze(0)
        hi_q = self._q_upper.unsqueeze(0)
        x_box = torch.clamp(cmd_d, lo_q, hi_q)
        hit_box = (x_box - cmd_d).abs().amax(dim=-1) > tol

        # --- 2. slew vs the previous COMMAND --------------------------------
        prev = self._prev_cmd
        dq = self._dq_max.unsqueeze(0)
        lo = torch.maximum(lo_q, prev - dq)
        hi = torch.minimum(hi_q, prev + dq)
        # The intersection is empty only when ``prev`` sits outside the joint box
        # (possible right after ``reset`` seeds it from a measured pose that the
        # posture limits do not admit).  Collapsing to ``hi`` then means "move
        # toward the box at the full slew rate", which is the right recovery and
        # keeps the interval non-empty branch-free.
        lo = torch.minimum(lo, hi)
        x_slew = torch.clamp(x_box, lo, hi)
        hit_slew = (x_slew - x_box).abs().amax(dim=-1) > tol

        # --- 3. differentiable-SDF clearance --------------------------------
        x = x_slew
        if self._sdf is not None and int(self.cfg.sdf_steps) > 0:
            tl, tr = self._split_targets(target_drum, B)
            x = self._clearance_steps(x_slew, lo, hi, tl, tr)
        hit_sdf = (x - x_slew).abs().amax(dim=-1) > tol

        engaged = (x - cmd_d).abs().amax(dim=-1) > tol

        # --- 4. QP last resort ----------------------------------------------
        if self.mode == "qp" and self._qp is not None:
            x, engaged = self._project_qp(x, cmd_d, q_d, qd_d, engaged, target_drum)

        self._prev_cmd = x
        self._calls += 1
        self._envs_seen += B
        self._counts += torch.stack(
            [engaged.sum(), hit_box.sum(), hit_slew.sum(), hit_sdf.sum()]
        ).to(torch.int64)
        return x.to(device=cmd.device, dtype=cmd.dtype), engaged.to(cmd.device)

    # ------------------------------------------------------------- QP stage --
    def _project_qp(
        self,
        x: torch.Tensor,
        cmd: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        engaged: torch.Tensor,
        target_drum,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env OSQP projection on the CPU.  Syncs, by construction.

        The repo filter is a single-config solver and ``play.py`` drives it with
        a Python loop; there is no batched form to borrow, so this is that loop,
        restricted to the rows box mode already flagged (``qp_all=True`` for all
        of them).  The commanded target handed to the QP is the RAW ``cmd``, not
        the box-projected one -- the QP's own objective is
        ``1/2||x - q_cmd||^2`` subject to its own (strictly tighter, because it
        also carries the accel and braking rows) box, so pre-clipping would
        silently change the objective and bias the projection.

        Warm starts: ``SafetyFilter`` keeps ONE ``_z_prev`` primal vector, so a
        single shared instance hands env e+1 the *primal solution of env e* -- a
        different QP with different active rows.  (``play.py`` has this bug
        today; it is invisible because OSQP still converges, just from a worse
        point.)  **Chosen fix: one shared filter plus a per-env ``_z_prev``
        pool, swapped in around each call**, rather than a pool of
        ``SafetyFilter`` instances -- each instance re-derives
        ``derive_accel_limits`` over 200 Pinocchio CRBA samples and allocates
        its own OSQP workspace (~0.35 s and tens of MB), which at B=256 is
        minutes of construction and gigabytes for a path that is already the
        slow one.  ``_z_prev`` is the only *explicit* per-solve state ``project``
        carries: the duals are reset to zero on every solve, and ``vals`` /
        ``_l`` / ``_u`` / ``q_vec`` are fully rewritten each time.

        It is not bit-equivalent to a pool of instances, though, and the gap is
        measured: OSQP's adaptive rho survives ``update()``, so the shared
        workspace still leaks *convergence* state between envs -- 8.4e-3 rad on
        a 0.45 rad correction, dropping to 4.1e-4 rad with
        ``qp_osqp_settings={"adaptive_rho": False}``.  See that field.
        """
        B = x.shape[0]
        sel = torch.arange(B, device=x.device) if self.cfg.qp_all else engaged.nonzero(as_tuple=False).reshape(-1)
        idx = sel.detach().cpu().tolist()
        if not idx:
            return x, engaged

        q_np = q.detach().double().cpu().numpy()
        qd_np = qd.detach().double().cpu().numpy()
        cmd_np = cmd.detach().double().cpu().numpy()
        if target_drum is None:
            tgt_np = np.full((B, 2), -1, dtype=np.int64)
        else:
            t = torch.as_tensor(target_drum).long().detach().cpu().numpy()
            tgt_np = np.stack([t, t], axis=1) if t.ndim == 1 else t

        out = x.detach().cpu().double().numpy()
        eng = engaged.detach().cpu().numpy().copy()
        for e in idx:
            ignore: set[tuple[str, str]] = set()
            for hand, col in (("left", 0), ("right", 1)):
                d = int(tgt_np[e, col])
                if 0 <= d < len(self._qp_drum_names):
                    ignore |= self._qp_ignore_fn(hand, self._qp_drum_names[d])
            self._qp._z_prev = self._qp_z_prev[e]  # per-env warm start, see docstring
            res = self._qp.project(q_np[e], qd_np[e], cmd_np[e], ignore_pairs=ignore or None)
            self._qp_z_prev[e] = self._qp._z_prev
            out[e] = res.q_safe
            eng[e] = bool(res.active) or bool(eng[e])
            self._qp_solves += 1
            self._qp_active += int(bool(res.active))
            self._qp_fallback += int(bool(res.fallback))
        x_new = torch.as_tensor(out.tolist(), device=x.device, dtype=x.dtype)
        return x_new, torch.as_tensor(eng.tolist(), device=x.device, dtype=torch.bool)


def _self_test() -> None:  # pragma: no cover - manual smoke run
    """``python -m isaac_rl.safety`` from ``amortized/``: shapes, box, timing."""
    import time

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    filt = BatchedSafetyFilter(SafetyConfig(mode="box"), device=dev)
    B = 256
    q = torch.zeros(B, N_DOF, device=dev)
    cmd = q + 0.5 * torch.randn(B, N_DOF, device=dev)
    safe, engaged = filt.project(cmd, q, torch.zeros_like(q))
    assert torch.isfinite(safe).all()
    t0 = time.perf_counter()
    for _ in range(100):
        safe, engaged = filt.project(cmd, q, torch.zeros_like(q))
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"{(time.perf_counter() - t0) * 10:.2f} ms/call, stats={filt.stats}")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
