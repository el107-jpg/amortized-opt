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

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, PD_SUBSTEPS
from .limits import isaac_armature, isaac_runtime_damping, isaac_runtime_stiffness

__all__ = [
    "PDSimConfig",
    "BatchedPDSim",
    "simulate_pd",
    "mass_matrix",
    "frozen_minv",
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
    #: config (arms Kp 40 / Kd 2.0, waist Kp 200 / Kd 10.0).
    kp_scale: float | tuple[float, ...] = 1.0
    kd_scale: float | tuple[float, ...] = 1.0
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if int(self.substeps) < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        self.substeps = int(self.substeps)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    """
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

    qa, va = qa0, va0
    qs: list[torch.Tensor] = [qa]
    vs: list[torch.Tensor] = [va]
    for i in range(1, K + 1):
        x_prev = x_cmd[:, i - 1]
        x_new = x_cmd[:, i]
        dx = x_new - x_prev
        qd_cmd = dx * inv_dt  # held across the whole control period
        Mi = Minv[:, i - 1]
        kd_ff = kd * (ff_scale * qd_cmd)
        for s in range(1, S + 1):
            # First-order hold: the position target ramps prev -> new.
            q_tgt = x_prev + (float(s) / S) * dx
            err_pos = q_tgt - qa
            i_term = ki_t * integral if has_ki else 0.0
            err = kp * err_pos + i_term + kd_ff - kd * va
            acc = torch.matmul(Mi, err.unsqueeze(-1)).squeeze(-1)
            qa = qa + h * va
            va = va + h * acc
            if has_ki:
                integral = integral + err_pos * h
                if windup is not None:
                    integral = torch.clamp(integral, -windup, windup)
        qs.append(qa)
        vs.append(va)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Full rollout over the commanded trajectory; see :func:`simulate_pd`."""
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
