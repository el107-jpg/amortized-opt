"""Batched, differentiable, noised PD-tracking simulator (section 5).

This is the plan's "Noised PD Controller Simulation": the model of what the
robot's servo *actually* reaches when driven by a commanded 50 Hz waypoint
trajectory. It is the inner loop of the simulated-MPC data generator
(:mod:`amortized_opt.mpc_gen`), so it has to be batched over thousands of
rollouts and differentiable end to end.

Ported from ``control/trajopt/closed_loop.py`` (numpy, one rollout at a time,
non-differentiable) following the validated prototype
``scratch/amortized_perf/bench_pd.py``.

The plant
---------
A linear state-space model of the PD servo plus the RNEA gravity/Coriolis
feedforward, with **all joints coupled** through the real effective mass
matrix, frozen per control knot::

    Minv_i = inv(M(q_bar_i) + diag(armature))

Per PD substep of control period ``i`` (substep ``s = 1..S``, ``h = dt / S``)::

    q_tgt   = x_{i-1} + (s / S) * (x_i - x_{i-1})      # the FOH position ramp
    qd_cmd  = (x_i - x_{i-1}) / dt                     # held across the period
    acc     = Minv_i @ (kp * (q_tgt - qa) + kd * (ff_scale * qd_cmd - va))
    qa     += h * va
    va     += h * acc

Two things to note.

* **The FOH ramp is the real env's.** ``g1_drummer_env._init_foh`` writes
  ``q_tgt(s) = q_cmd_prev + (s / decimation) * (q_cmd - q_cmd_prev)`` with the
  constant slope ``(q_cmd - q_cmd_prev) / step_dt`` as the velocity target. The
  repo's ``closed_loop`` collapses that to a single step at ``x_i``, which is
  exactly what the recursion above degenerates to at ``S = 1`` --- so
  ``substeps=1`` reproduces ``simulate_actual_state`` bit for bit and no
  separate "conformance mode" is needed.
* **Resolved decision 1**: this folder runs the PD/physics loop at 250 Hz
  against a 50 Hz waypoint grid, i.e. ``PD_SUBSTEPS = 5`` substeps of 4 ms per
  control tick. That is the default here.

Gain randomization
------------------
The plan generates MPC-like tracking error by randomizing the servo rather than
by modelling friction or mass error: :meth:`BatchedPDSim.sample_gains` draws
per-joint, per-rollout log-normal multipliers around the *runtime* Kp/Kd from
:mod:`amortized_opt.limits` (note runtime Kd is 2x the CLAUDE.md table). The
multiplier is log-normal rather than normal so a large sigma can never flip a
gain negative, which would make the plant unstable rather than merely wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    CONTROL_DT,
    G1_UPPER_BODY_JOINTS,
    PD_SUBSTEPS,
    ready_joint_pos,
)
from .limits import (
    available_torque_torch,
    isaac_armature,
    isaac_effort_limits,
    isaac_runtime_damping,
    isaac_runtime_stiffness,
    isaac_velocity_limits,
)

__all__ = [
    "PDSimConfig",
    "BatchedPDSim",
    "simulate_pd",
    "mass_matrix",
    "frozen_minv",
    "extract_torque_kernels",
    "nominal_torque_kernels",
    "feedback_torque",
    "tracking_torque_knots",
]


# --------------------------------------------------------------- mass matrix --
def mass_matrix(
    rnea,
    q: torch.Tensor,
    *,
    include_armature: bool = True,
    chunk: int | None = None,
) -> torch.Tensor:
    """``M(q) + diag(armature)`` for a batch of poses, via unit-acceleration RNEA.

    ``rnea`` is a :class:`~amortized_opt.rnea.BatchedRNEA`. Column ``j`` is
    ``rnea(q, 0, e_j) - rnea(q, 0, 0)``, which already carries the armature term
    (``BatchedRNEA.rnea`` adds ``diag(armature) @ qdd``). Differentiable in
    ``q``.

    The ``n + 1`` right-hand sides are stacked into ONE RNEA call of batch
    ``B * (n + 1)`` rather than looped: at ``B = 1`` the Featherstone recursion
    is entirely Python/launch overhead, so looping made this 18x slower than it
    needed to be and dominated the MPC generator's tick (measured 8.2 s of an
    8.9 s 40-tick episode). ``chunk`` caps the effective batch when ``B`` is
    large enough for ``B * 18`` rows to be a memory problem.

    Returns ``(B, n, n)``, symmetric to RNEA round-off.
    """
    if q.dim() != 2:
        raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
    B, n = q.shape
    step = int(chunk) if chunk else B
    outs = []
    rhs = torch.cat([q.new_zeros(1, n), torch.eye(n, device=q.device, dtype=q.dtype)], dim=0)
    for b0 in range(0, B, max(step, 1)):
        qc = q[b0 : b0 + max(step, 1)]
        Bc = qc.shape[0]
        qb = qc.unsqueeze(1).expand(Bc, n + 1, n).reshape(Bc * (n + 1), n)
        qdd = rhs.unsqueeze(0).expand(Bc, n + 1, n).reshape(Bc * (n + 1), n)
        tau = rnea.rnea(qb, torch.zeros_like(qb), qdd).reshape(Bc, n + 1, n)
        # tau[:, 0] is gravity alone; tau[:, 1 + j] - bias is column j of M.
        outs.append((tau[:, 1:] - tau[:, :1]).transpose(1, 2))
    M = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
    if not include_armature and rnea.armature is not None:
        M = M - torch.diag_embed(rnea.armature.to(M.dtype).expand(B, n))
    return M


def frozen_minv(
    rnea,
    q_bar: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> torch.Tensor:
    """``inv(M(q) + diag(armature))`` at every knot of a nominal trajectory.

    ``q_bar`` is ``(B, K, n)``; returns ``(B, K, n, n)``. "Frozen" is the
    trajopt convention: one inverse per knot, held across that knot's control
    period (and, here, across its PD substeps).
    """
    if q_bar.dim() != 3:
        raise ValueError(f"q_bar must be (B, K, n), got {tuple(q_bar.shape)}")
    B, K, n = q_bar.shape
    M = mass_matrix(rnea, q_bar.reshape(B * K, n))
    if ridge:
        M = M + float(ridge) * torch.eye(n, device=M.device, dtype=M.dtype)
    return torch.linalg.inv(M).reshape(B, K, n, n)


# -------------------------------------------------------------------- config --
@dataclass
class PDSimConfig:
    """Plant + randomization knobs.

    ``dt`` is the CONTROL period (the waypoint spacing); ``substeps`` splits it
    into PD steps. ``substeps=1`` reproduces the repo's ``closed_loop`` exactly.
    """

    dt: float = CONTROL_DT
    substeps: int = PD_SUBSTEPS
    #: How much velocity-feedforward authority the model assumes the servo has.
    #: 1.0 models the real env exactly (see closed_loop.py's module docstring).
    ff_scale: float = 1.0
    #: Log-normal sigma of the per-joint Kp / Kd / Ki multipliers (0 = deterministic).
    kp_sigma: float = 0.15
    kd_sigma: float = 0.15
    ki_sigma: float = 0.0
    #: Integral gain (N·m/rad·s).  0 = pure PD (the pre-PID behaviour).
    ki: float = 0.0
    #: Anti-windup clamp on the integral error, in rad·s.  0 = no clamp.
    #: Clamping to ``ki_windup`` means the integral contribution ``ki * integral``
    #: is bounded by ``ki * ki_windup`` in torque space.
    ki_windup: float = 0.0
    #: Correlated component: one extra multiplier shared by all joints of a
    #: rollout, modelling a global gain-scale error on top of the per-joint one.
    gain_common_sigma: float = 0.0
    #: DETERMINISTIC multiplier on the runtime Kp / Kd tables -- scalar or
    #: per-joint.  This is a *plant* change (what the robot's actuators would be
    #: configured to), not a randomization: it multiplies the nominal before
    #: ``sample_gains``' log-normal draw, so an ablation over stiffness composes
    #: with the usual gain noise instead of replacing it.  1.0 = the shipped
    #: config (arms Kp 40 / Kd 4.0, waist Kp 200 / Kd 20.0).
    kp_scale: float | tuple[float, ...] = 1.0
    kd_scale: float | tuple[float, ...] = 1.0
    #: CLAMP the demanded generalized force to the joint's usable torque
    #: (:func:`~amortized_opt.limits.available_torque_torch`, at the current
    #: substep velocity) before it reaches the effective mass.  This is the
    #: offline stand-in for the effort limit Isaac enforces at deployment: with
    #: it OFF -- the default, and bit-for-bit the historical plant -- training
    #: feels nothing where the real robot saturates.
    saturate_effort: bool = False
    #: Fraction of the RATING the clamp is placed at.  1.0 (the default) is the
    #: hardware limit, which is what Isaac clamps at; ``ETA_TORQUE`` would model
    #: a controller that keeps the cost's own conservatism margin.
    saturate_eta: float = 1.0
    saturate_scale: float = 1.0
    saturate_derate: str = "linear"
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if int(self.substeps) < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        self.substeps = int(self.substeps)
        if self.saturate_derate not in ("none", "linear"):
            raise ValueError(f"saturate_derate={self.saturate_derate!r}")

    @property
    def pd_dt(self) -> float:
        return float(self.dt) / float(self.substeps)


def _scale_vec(scale, n: int, what: str) -> np.ndarray:
    """Scalar / ``(n,)`` gain multiplier -> ``(n,)``, validated positive."""
    arr = np.asarray(scale, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{what} must be scalar or ({n},), got {arr.shape}")
    if np.any(arr <= 0.0):
        raise ValueError(f"{what} must be positive, got min {arr.min():.4g}")
    return arr


# ------------------------------------------------------------------ the core --
def simulate_pd(
    x_cmd: torch.Tensor,
    Minv: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    dt: float,
    qa0: torch.Tensor,
    va0: torch.Tensor,
    *,
    substeps: int = 1,
    ff_scale: float = 1.0,
    ki: torch.Tensor | None = None,
    ki_windup: float = 0.0,
    integral0: torch.Tensor | None = None,
    saturate_effort: bool = False,
    tau_limit: torch.Tensor | None = None,
    vel_limit: torch.Tensor | None = None,
    saturate_eta: float = 1.0,
    saturate_scale: float = 1.0,
    saturate_derate: str = "linear",
    return_torque: bool = False,
    torque_reduce: str = "max",
) -> tuple[torch.Tensor, ...]:
    """Batched differentiable PD (or PID) rollout. Pure tensors, no dataclasses.

    ``x_cmd``  (B, K+1, n) commanded waypoints; row 0 is the pinned start.
    ``Minv``   (B, K, n, n) or (B, n, n) or (n, n) -- frozen inverse effective
               mass per control period (period ``i`` uses knot ``i-1``).
    ``kp``/``kd`` (B, n) or (n,) diagonal gains.
    ``ki``     (B, n) or (n,) or None -- integral gain; None/zero = pure PD.
    ``ki_windup`` anti-windup clamp in rad·s (0 = no clamp).
    ``integral0`` (B, n) or None -- initial integral error in rad·s.
    ``qa0``/``va0`` (B, n) initial actual state.

    Returns ``(qa, va, integral)`` where ``qa``, ``va`` are ``(B, K+1, n)``
    sampled at the CONTROL knots and ``integral`` is ``(B, n)`` (the final
    accumulated integral error).  ``integral`` is ``None`` when ``ki`` is None or
    identically zero.

    Demanded torque
    ---------------
    ``return_torque=True`` appends a FOURTH element, ``(B, K, n)``: the
    generalized force the servo asks for, i.e. the ``err`` vector *before*
    ``Minv`` -- the quantity this plant has always thrown away and the real
    robot's actuators have to produce.  Element ``i-1`` is control period ``i``,
    so it is aligned with knot ``i`` of ``qa`` / ``va`` (there is no demand at
    the pinned start).  The 3-tuple return is unchanged when the flag is off,
    so every existing ``qa, va, _ = ...`` call site is untouched.

    ``torque_reduce`` picks the ONE number reported per control tick out of the
    ``substeps`` values computed inside it:

    * ``"max"`` (default) -- the substep with the largest magnitude, sign
      preserved.  This is the number an effort limit is a limit ON.
    * ``"last"`` -- the final substep's demand.  Exactly linear in the command,
      but for the joints that actually saturate it reports approximately
      nothing: a wrist settles well within one 20 ms tick at ``PD_SUBSTEPS``
      = 10, so its unit-impulse demand is 204 N*m at the first substep and 0.26
      N*m at the tenth.  Kept because the LTI property is occasionally what a
      test wants; it is not a sane default.

    The default is therefore the nonlinear one, and
    :func:`extract_torque_kernels` identifies its FIR against it.  That is exact
    for any joint whose worst substep is the first (every arm and wrist joint --
    they settle inside the tick, so the demand is monotonically decaying across
    it) and approximate for the slow waist joints, which is measured rather than
    assumed by ``scripts/validate_track_torque.py``.

    Effort saturation
    -----------------
    ``saturate_effort=True`` clamps that demand to
    ``+/- available_torque(va)`` (:func:`~amortized_opt.limits.available_torque_torch`,
    evaluated at the CURRENT substep velocity) before it is divided by the
    effective mass -- the offline stand-in for the effort limit Isaac applies at
    deployment.  ``tau_limit`` / ``vel_limit`` are ``(n,)`` or broadcastable
    ratings; they default to the Isaac tables.  Default OFF, and the branch is
    on a python bool, so the unsaturated plant is bit-for-bit unchanged.
    """
    if torque_reduce not in ("last", "max"):
        raise ValueError(f"torque_reduce must be 'last' or 'max', got {torque_reduce!r}")
    if x_cmd.dim() != 3:
        raise ValueError(f"x_cmd must be (B, K+1, n), got {tuple(x_cmd.shape)}")
    B, Kp1, n = x_cmd.shape
    K = Kp1 - 1
    S = int(substeps)
    h = float(dt) / S
    inv_dt = 1.0 / float(dt)

    if Minv.dim() == 2:
        Minv = Minv.view(1, 1, n, n).expand(B, K, n, n)
    elif Minv.dim() == 3:
        Minv = Minv.unsqueeze(1).expand(B, K, n, n)
    elif Minv.dim() != 4:
        raise ValueError(f"Minv must be 2/3/4-dim, got {tuple(Minv.shape)}")

    kp = kp.view(1, n) if kp.dim() == 1 else kp
    kd = kd.view(1, n) if kd.dim() == 1 else kd

    has_ki = ki is not None and (ki.abs().sum() > 0)
    ki_t = (
        ki.view(1, n) if (ki is not None and ki.dim() == 1) else ki
        if has_ki else None
    )
    integral = (
        torch.zeros(B, n, device=x_cmd.device, dtype=x_cmd.dtype)
        if integral0 is None and has_ki
        else integral0.clone() if integral0 is not None and has_ki
        else None if has_ki
        else None
    )
    windup = float(ki_windup) if has_ki and ki_windup > 0 else None

    if saturate_effort and (tau_limit is None or vel_limit is None):
        # Defaulting is only meaningful on the full upper body: the tables are
        # keyed by joint NAME and this function only ever sees a width, so a
        # narrower plant has to say what its joints are rated at rather than be
        # handed the first ``n`` rows of somebody else's kinematics.
        if n != len(G1_UPPER_BODY_JOINTS):
            raise ValueError(
                f"saturate_effort on a {n}-joint plant needs explicit "
                "tau_limit / vel_limit"
            )
        kw = {"device": x_cmd.device, "dtype": x_cmd.dtype}
        if tau_limit is None:
            tau_limit = torch.as_tensor(isaac_effort_limits(), **kw)
        if vel_limit is None:
            vel_limit = torch.as_tensor(isaac_velocity_limits(), **kw)

    qa, va = qa0, va0
    qs: list[torch.Tensor] = [qa]
    vs: list[torch.Tensor] = [va]
    taus: list[torch.Tensor] = []
    for i in range(1, K + 1):
        x_prev = x_cmd[:, i - 1]
        x_new = x_cmd[:, i]
        dx = x_new - x_prev
        qd_cmd = dx * inv_dt  # held across the whole control period
        Mi = Minv[:, i - 1]
        kd_ff = kd * (ff_scale * qd_cmd)
        sub: list[torch.Tensor] = []
        for s in range(1, S + 1):
            # First-order hold: the position target ramps prev -> new.
            q_tgt = x_prev + (float(s) / S) * dx
            err_pos = q_tgt - qa
            i_term = ki_t * integral if has_ki else 0.0
            err = kp * err_pos + i_term + kd_ff - kd * va
            if saturate_effort:
                cap = available_torque_torch(
                    va, tau_limit, vel_limit, eta=saturate_eta,
                    saturation_scale=saturate_scale, derate=saturate_derate,
                )
                err = torch.minimum(torch.maximum(err, -cap), cap)
            if return_torque:
                sub.append(err)
            acc = torch.matmul(Mi, err.unsqueeze(-1)).squeeze(-1)
            qa = qa + h * va
            va = va + h * acc
            if has_ki:
                integral = integral + err_pos * h
                if windup is not None:
                    integral = torch.clamp(integral, -windup, windup)
        qs.append(qa)
        vs.append(va)
        if return_torque:
            if torque_reduce == "last":
                taus.append(sub[-1])
            else:
                st = torch.stack(sub, dim=1)                       # (B, S, n)
                j = st.abs().argmax(dim=1, keepdim=True)
                taus.append(st.gather(1, j).squeeze(1))
    if return_torque:
        tau = (
            torch.stack(taus, dim=1) if taus
            else x_cmd.new_zeros(B, 0, n)
        )
        return torch.stack(qs, dim=1), torch.stack(vs, dim=1), integral, tau
    return torch.stack(qs, dim=1), torch.stack(vs, dim=1), integral


# ------------------------------------------------------------------- wrapper --
class BatchedPDSim:
    """Stateful convenience wrapper: gain tables, randomization, one-tick steps.

    ::

        sim = BatchedPDSim(device="cuda:1")
        kp, kd = sim.sample_gains(batch=1024, generator=g)
        qa, va = sim.rollout(x_cmd, kp=kp, kd=kd, qa0=q0, va0=v0, Minv=Minv)

    ``Minv`` may be omitted, in which case a constant diagonal
    ``inv(diag(M_ref) + armature)`` fallback is used --- adequate for smoke
    tests, NOT for data generation (the coupling is ~34% of the answer). Pass
    a real one from :func:`frozen_minv`.
    """

    def __init__(
        self,
        config: PDSimConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        rnea=None,
    ):
        self.cfg = config if config is not None else PDSimConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.rnea = rnea
        names = tuple(self.cfg.joint_names)
        self.joint_names = names
        self.n = len(names)
        t = lambda a: torch.as_tensor(np.asarray(a, float), device=self.device, dtype=dtype)
        self.kp_nominal = t(isaac_runtime_stiffness(names)) * t(
            _scale_vec(self.cfg.kp_scale, self.n, "kp_scale")
        )
        self.kd_nominal = t(isaac_runtime_damping(names)) * t(
            _scale_vec(self.cfg.kd_scale, self.n, "kd_scale")
        )
        self.ki_nominal = torch.full((self.n,), float(self.cfg.ki),
                                     device=self.device, dtype=self.dtype)
        self.armature = t(isaac_armature(names))
        #: Ratings the optional effort clamp is placed against.
        self.tau_limit = t(isaac_effort_limits(names))
        self.vel_limit = t(isaac_velocity_limits(names))

    # ------------------------------------------------------------- gains --
    def sample_gains(
        self,
        batch: int,
        *,
        generator: torch.Generator | None = None,
        kp_sigma: float | None = None,
        kd_sigma: float | None = None,
        ki_sigma: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Per-rollout, per-joint randomized ``(kp, kd, ki)``, each ``(B, n)``.

        Log-normal about the runtime nominal, so the multiplier is positive by
        construction and ``exp(N(0, s))`` has median 1. ``sigma = 0`` returns
        the nominal broadcast, exactly.  ``ki`` is ``None`` when ``cfg.ki == 0``.
        """
        B, n = int(batch), self.n
        sk = self.cfg.kp_sigma if kp_sigma is None else float(kp_sigma)
        sd = self.cfg.kd_sigma if kd_sigma is None else float(kd_sigma)
        si = self.cfg.ki_sigma if ki_sigma is None else float(ki_sigma)
        sc = float(self.cfg.gain_common_sigma)

        def draw(sigma: float) -> torch.Tensor:
            if sigma <= 0.0 and sc <= 0.0:
                return torch.ones(B, n, device=self.device, dtype=self.dtype)
            z = torch.randn(B, n, device=self.device, dtype=self.dtype, generator=generator)
            log = sigma * z
            if sc > 0.0:
                zc = torch.randn(B, 1, device=self.device, dtype=self.dtype, generator=generator)
                log = log + sc * zc
            return torch.exp(log)

        kp = self.kp_nominal * draw(sk)
        kd = self.kd_nominal * draw(sd)
        ki = None
        if self.cfg.ki > 0.0:
            ki = self.ki_nominal * draw(si)
        return kp, kd, ki

    def nominal_gains(self, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        one = torch.ones(int(batch), self.n, device=self.device, dtype=self.dtype)
        ki = None
        if self.cfg.ki > 0.0:
            ki = self.ki_nominal * one
        return self.kp_nominal * one, self.kd_nominal * one, ki

    # ------------------------------------------------------------ inertia --
    def default_minv(self, batch: int = 1) -> torch.Tensor:
        """Constant diagonal fallback ``(B, n, n)`` -- smoke tests only."""
        # A crude but positive-definite stand-in: unit link inertia + armature.
        d = 1.0 / (1.0 + self.armature)
        return torch.diag_embed(d).expand(int(batch), self.n, self.n)

    def frozen_minv(self, q_bar: torch.Tensor, **kw) -> torch.Tensor:
        if self.rnea is None:
            raise RuntimeError("BatchedPDSim was built without an rnea instance")
        return frozen_minv(self.rnea, q_bar, **kw)

    # ------------------------------------------------------------ rollout --
    def rollout(
        self,
        x_cmd: torch.Tensor,
        *,
        qa0: torch.Tensor,
        va0: torch.Tensor,
        kp: torch.Tensor | None = None,
        kd: torch.Tensor | None = None,
        ki: torch.Tensor | None = None,
        Minv: torch.Tensor | None = None,
        substeps: int | None = None,
        integral0: torch.Tensor | None = None,
        return_torque: bool = False,
        torque_reduce: str = "max",
    ) -> tuple[torch.Tensor, ...]:
        """Full rollout over the commanded trajectory; see :func:`simulate_pd`.

        ``return_torque=True`` adds the ``(B, K, n)`` demanded-torque series as
        a fourth return value; the effort clamp follows
        ``PDSimConfig.saturate_effort``.
        """
        B = x_cmd.shape[0]
        if kp is None or kd is None:
            kp_d, kd_d, ki_d = self.nominal_gains(B)
            kp = kp_d if kp is None else kp
            kd = kd_d if kd is None else kd
            if ki is None:
                ki = ki_d
        if Minv is None:
            Minv = self.default_minv(B)
        return simulate_pd(
            x_cmd,
            Minv,
            kp,
            kd,
            self.cfg.dt,
            qa0,
            va0,
            substeps=self.cfg.substeps if substeps is None else int(substeps),
            ff_scale=self.cfg.ff_scale,
            ki=ki,
            ki_windup=self.cfg.ki_windup,
            integral0=integral0,
            saturate_effort=self.cfg.saturate_effort,
            tau_limit=self.tau_limit,
            vel_limit=self.vel_limit,
            saturate_eta=self.cfg.saturate_eta,
            saturate_scale=self.cfg.saturate_scale,
            saturate_derate=self.cfg.saturate_derate,
            return_torque=return_torque,
            torque_reduce=torque_reduce,
        )

    def step(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        qa: torch.Tensor,
        va: torch.Tensor,
        *,
        kp: torch.Tensor | None = None,
        kd: torch.Tensor | None = None,
        ki: torch.Tensor | None = None,
        Minv: torch.Tensor | None = None,
        integral: torch.Tensor | None = None,
        gain_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Advance exactly ONE 20 ms control tick (``substeps`` PD substeps).

        This is what the simulated-MPC loop calls: plan, execute one tick,
        re-plan from where the servo actually got to.  Returns ``(qa, va, integral)``
        where ``integral`` is the updated accumulated error (``None`` when ki=0).

        ``gain_scale`` is an optional ``(kp_mul, kd_mul)`` pair of ``(B, n)``
        multipliers applied ON TOP of the resolved gains for THIS tick only --
        the strike-window impedance schedule (:mod:`amortized_opt.impedance`).
        The Isaac plant accepts the same kwarg, so the schedule does not fork
        the two plants.
        """
        if gain_scale is not None:
            B = x_prev.shape[0]
            if kp is None or kd is None:
                kp_n, kd_n, ki_n = self.nominal_gains(B)
                kp = kp_n if kp is None else kp
                kd = kd_n if kd is None else kd
                if ki is None:
                    ki = ki_n
            kp = kp * gain_scale[0]
            kd = kd * gain_scale[1]
        x_cmd = torch.stack([x_prev, x_next], dim=1)
        qa_out, va_out, int_out = self.rollout(
            x_cmd, qa0=qa, va0=va, kp=kp, kd=kd, ki=ki, Minv=Minv, integral0=integral,
        )
        return qa_out[:, 1], va_out[:, 1], int_out


# ======================================================= tracking torque =====
# The plan is priced by its RNEA (M qdd + C qd + g at the planned states), but
# the thing an actuator has to deliver at deployment is what the PD controller
# ASKS FOR while tracking that plan, and the two are not the same quantity: the
# servo lags the first-order-hold command and then over-corrects, and the
# resulting kp*e + kd*(ff*qd_cmd - va) transient is invisible to any function of
# the plan's own states.  (Measured on the 19m-grooves checkpoint: zero RNEA
# excess on every joint, 5 N*m wrist ratings exceeded on 44-74 % of ticks with
# 27-38 N*m peaks.)
#
# With the gains, the substep count and a FROZEN DIAGONAL effective mass held
# fixed, that map is (per substep) exactly linear time-invariant:
#
#     tau_dem[i] = sum_l  k_l * x_cmd[i - l]                       (per joint)
#
# because ``simulate_pd``'s recursion is affine in ``(x_cmd, qa0, va0)`` with no
# constant term -- this plant carries NO bias forces, ``acc = Minv @ err``.  So
# the whole feedback demand is one short FIR filter per joint, identified once
# and convolved in the cost for the price of a grouped conv1d.
#
# The per-tick reduction is the WORST SUBSTEP (``torque_reduce="max"``), because
# that is what an effort limit limits, and the reduction is where the exactness
# is spent: the max over substeps is linear only while the worst substep is the
# same one for every input.  It is, for every joint light enough to settle
# inside a 20 ms tick -- the arms and wrists, i.e. exactly the joints that
# saturate.  Measured at the ready pose, a unit-impulse command on
# ``wrist_pitch`` demands 204 N*m at the first substep and 0.26 N*m at the tenth;
# the resulting 3-tap kernel [204, -408, 204] is 204 x the second difference of
# the command, i.e. ~kd*dt*qdd_cmd -- 0.082 N*m per rad/s^2 against a physical
# inertia of 0.012, so the servo's REFLECTED DAMPING demand is 6.6x the joint's
# own inertial torque and the RNEA sees none of it.  That is the whole finding
# this estimator exists to price.
#
# ``sum_l l * k_l = 0`` (see below) makes the first moment vanish, so the
# convolution reduces to exactly that second-difference form for the fast joints
# and to a genuine transient for the slow ones.
#
# DC REMOVAL.  The filter's steady-state gain is ``K_dc = sum_l k_l``.  The true
# steady-state gain is exactly ZERO -- a settled servo holding a constant target
# demands no torque when there is no gravity in the plant -- and the quasi-static
# torque that IS real (gravity, Coriolis) is supplied exactly by the RNEA term
# the estimate is added to.  Any nonzero ``K_dc`` in a TRUNCATED kernel is
# therefore pure double counting against an absolute joint angle of a couple of
# radians.  :func:`feedback_torque` removes it by convolving the command's
# DEVIATION from its current value,
#
#     tau_fb[i] = sum_l k_l * (x[i - l] - x[i])  =  (k * x)[i] - K_dc * x[i],
#
# which is identically zero on any constant command whatever the truncation, and
# identical to the plain convolution when ``K_dc`` is 0.
#
# MEASURED, at the ready pose and PD_SUBSTEPS = 10, so the claims above are not
# claims.  Untruncated (80 taps) the kernel's moments are
#
#     |sum_l k_l|       <= 0.094 N*m/rad        (DC, against peaks of ~1000)
#     |sum_l l * k_l|   <= 1.4                  (first moment, ditto)
#     sum_l l^2 k_l/2 * dt^2 = 0.081 N*m/(rad/s^2)  on wrist_pitch
#
# i.e. both the constant and the ramp response vanish, as they must (a settled
# servo tracking a constant or a constant-velocity command demands nothing), and
# the leading term is the ACCELERATION response.  That coefficient is 0.081,
# which is ``kd * dt`` = 0.080, NOT the joint's effective inertia 0.0123: the
# demand is the servo's reflected DAMPING, and RNEA's diagonal ``M qdd`` accounts
# for 15 % of it.  So the overlap with the RNEA term is real, second order, and
# small -- and the other 85 %, plus the whole transient, is invisible to any
# function of the plan's own states.
#
# Truncated to the shipped 13 taps those moments become 2.2 and 27, which is
# what the DC removal is actually cleaning up: it is a no-op on the exact kernel
# and a 2 N*m/rad correction on the one in use.
def extract_torque_kernels(
    kp: torch.Tensor,
    kd: torch.Tensor,
    minv_diag: torch.Tensor,
    *,
    dt: float = CONTROL_DT,
    substeps: int = PD_SUBSTEPS,
    ff_scale: float = 1.0,
    taps: int = 24,
    tol: float = 1e-3,
    torque_reduce: str = "max",
) -> torch.Tensor:
    """Per-joint impulse response of ``command -> demanded torque``, ``(n, L)``.

    Identified by RUNNING THE REAL PLANT, not by re-deriving it: one
    :func:`simulate_pd` rollout of ``n`` samples, sample ``j`` commanding a unit
    impulse on joint ``j`` at knot 1 from a zero state, with ``Minv`` the
    diagonal ``minv_diag``.  Linearity (see the module note above) then makes
    the recorded demand the impulse response directly::

        x[m] = delta(m, 1)   =>   tau_dem[i] = k_{i-1}

    so kernel tap ``l`` is the demand at control tick ``l + 1``.  The result is
    truncated at the last tap exceeding ``tol`` x the joint's peak (or ``taps``,
    whichever is shorter); the response is that of an over-damped second-order
    servo, so it decays geometrically and 10-20 taps at 50 Hz cover it.

    ``minv_diag`` is the DECOUPLED effective mobility, ``(n,)``.  The natural
    choice is ``diag(inv(M(q_ref) + armature))`` -- the diagonal of the INVERSE,
    which is what a row of the coupled ``acc = Minv @ tau`` actually applies to
    that joint's own demand.
    """
    kp = kp.reshape(-1)
    kd = kd.reshape(-1)
    n = kp.numel()
    dev, dt_t = kp.device, kp.dtype
    T = max(int(taps), 1)
    x = torch.zeros(n, T + 1, n, device=dev, dtype=dt_t)
    x[torch.arange(n, device=dev), 1, torch.arange(n, device=dev)] = 1.0
    zero = torch.zeros(n, n, device=dev, dtype=dt_t)
    Minv = torch.diag_embed(minv_diag.reshape(-1).to(dev, dt_t))  # (n, n)
    _qa, _va, _ig, tau = simulate_pd(
        x, Minv, kp.expand(n, n), kd.expand(n, n), float(dt), zero, zero,
        substeps=int(substeps), ff_scale=float(ff_scale),
        return_torque=True, torque_reduce=torque_reduce,
    )                                                             # (n, T, n)
    k = tau[torch.arange(n, device=dev), :, torch.arange(n, device=dev)]  # (n, T)
    peak = k.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    live = (k.abs() > float(tol) * peak).any(dim=0)
    idx = torch.nonzero(live).flatten()
    L = int(idx[-1].item()) + 1 if idx.numel() else 1
    return k[:, :L].contiguous()


def feedback_torque(x_cmd: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """DC-removed feedback torque ``(B, K, n)`` for a command series ``(B, K, n)``.

    ``tau_fb[i] = sum_l kernel[l] * (x[i - l] - x[i])``, with the pre-horizon
    command held at ``x[0]`` (the plan's pinned start, which is what the servo
    was actually being commanded before the plan begins).  See the module note
    for why the ``- x[i]`` is there.  Differentiable in ``x_cmd``; the kernel is
    a constant.
    """
    if x_cmd.dim() != 3:
        raise ValueError(f"x_cmd must be (B, K, n), got {tuple(x_cmd.shape)}")
    B, K, n = x_cmd.shape
    if kernel.shape[0] != n:
        raise ValueError(f"kernel is for {kernel.shape[0]} joints, x_cmd has {n}")
    L = int(kernel.shape[1])
    w = kernel.to(device=x_cmd.device, dtype=x_cmd.dtype)
    xt = x_cmd.transpose(1, 2)                                    # (B, n, K)
    xp = torch.cat([xt[:, :, :1].expand(B, n, L - 1), xt], dim=2) if L > 1 else xt
    # conv1d is a CROSS-correlation, so the kernel is flipped to make it a
    # convolution: out[i] = sum_l w_flip[l] xp[i + l] = sum_m k_m x[i - m].
    y = F.conv1d(xp, w.flip(-1).unsqueeze(1), groups=n)            # (B, n, K)
    return (y - w.sum(dim=1).view(1, n, 1) * xt).transpose(1, 2)


def nominal_torque_kernels(
    rnea,
    q_ref: torch.Tensor | None = None,
    *,
    joint_names=G1_UPPER_BODY_JOINTS,
    dt: float = CONTROL_DT,
    substeps: int = PD_SUBSTEPS,
    taps: int = 24,
    tol: float = 1.0e-3,
    torque_reduce: str = "max",
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The per-joint ``command -> demanded torque`` FIR bank at NOMINAL gains.

    THE one place the bank is identified.  :meth:`amortized_opt.cost.FusedCost.
    build_torque_kernels` and the phase-1/2 auxiliary torque label
    (:func:`amortized_opt.warmstart_data.aux_targets` in ``"tracking"`` mode)
    both call this, so the label and the price cannot describe different
    plants.

    Runs :func:`extract_torque_kernels` at the plant's ``kp_nominal`` /
    ``kd_nominal`` (what :class:`BatchedPDSim` uses before gain randomization)
    and a frozen diagonal mobility ``diag(inv(M(q_ref) + armature))``, with
    ``q_ref`` defaulting to the drummer's ready pose.  Identified in float64
    whatever the caller's dtype -- it is a couple dozen taps of a stiff
    recursion, done once -- and returned in ``dtype`` on ``device``.
    """
    f64 = torch.float64
    dev = torch.device(device) if device is not None else torch.device("cpu")
    kp = torch.as_tensor(isaac_runtime_stiffness(joint_names), device=dev, dtype=f64)
    kd = torch.as_tensor(isaac_runtime_damping(joint_names), device=dev, dtype=f64)
    if q_ref is None:
        q_ref = torch.as_tensor(ready_joint_pos(joint_names), device=dev, dtype=f64)
    rnea64 = rnea.to(dev, f64)
    with torch.no_grad():
        M = mass_matrix(rnea64, q_ref.to(device=dev, dtype=f64).view(1, -1))
        minv_diag = torch.diagonal(torch.linalg.inv(M)[0])
    k = extract_torque_kernels(
        kp, kd, minv_diag, dt=float(dt), substeps=int(substeps),
        taps=int(taps), tol=float(tol), torque_reduce=torque_reduce,
    )
    return k.to(device=dev, dtype=dtype)


def tracking_torque_knots(
    sp,
    v0: torch.Tensor,
    rnea,
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(tau_hat, q, qd)`` at the plan's KNOTS -- the demanded tracking torque.

    ``tau_hat[t] = tau_rnea[t] + tau_fb[t]`` per (knot, joint), where

    * ``tau_rnea`` is the plan's own inverse dynamics at the knot finite
      differences (:meth:`~amortized_opt.tspline.TrajectorySpline.
      knot_finite_differences`) -- the states a first-order hold actually
      commands, and the grid the FIR bank lives on;
    * ``tau_fb`` is :func:`feedback_torque` on the commanded knot positions,
      DC-removed so the quasi-static part is counted once (by the RNEA term)
      and not twice.

    ``sp`` is a :class:`~amortized_opt.tspline.TrajectorySpline`.  All three
    outputs are ``(B, H + 1, D)``: index 0 is the pinned start knot.  This is
    the single definition shared by :meth:`amortized_opt.cost.FusedCost.
    _track_torque` (the price) and the ``"tracking"`` auxiliary torque label
    (the target), so the two cannot drift.
    """
    q, qd, qdd = sp.knot_finite_differences(v0)
    B, n, D = q.shape
    tau_r = rnea.rnea(
        q.reshape(B * n, D), qd.reshape(B * n, D), qdd.reshape(B * n, D)
    ).reshape(B, n, D)
    return tau_r + feedback_torque(q, kernel), q, qd
