"""Differentiable batched cubic spline over the 50 Hz waypoint grid.

Section 3 of ``docs/amortized_optimizer_plan.md``.  The transformer emits joint
positions on a fixed ``CONTROL_DT`` grid; the cost needs continuous position,
velocity, acceleration and jerk between those knots (per-segment velocity
maxima, sub-sampled torque, closed-form smoothness integrals).  This module is
that layer.

Semantics
---------

The interpolation rule is the same one the vendored :mod:`amortized_opt.spline`
uses: a **C2 cubic interpolating spline** through the knots -- i.e. exactly what
``scipy.interpolate.make_interp_spline(t, Q, k=3, bc_type=...)`` builds, which is
the engine ``spline.joint_path_spline`` calls.  Two differences, both deliberate:

* the parameter is **time on a uniform grid** (``CONTROL_DT``), not joint-space
  arc length -- the cost needs d/dt, and the waypoints are already time-stamped;
* the start boundary condition is **clamped to the measured start velocity**
  (``bc_start="clamped"``), because ``v_start`` is a hard boundary condition of
  the MPC problem, not something the spline may choose.  The end is **natural**
  (zero acceleration at the strike), which leaves the terminal *velocity* free --
  the impact-speed term prices that, so pinning it here would double-count.

``bc_start="not-a-knot"`` / ``bc_end="not-a-knot"`` reproduce scipy's default
(and therefore ``joint_path_spline``'s) end conditions exactly;
``tests/test_tspline.py`` asserts conformance to scipy at <= 1e-6 for every
supported combination.  Degenerate short horizons fall back to ``natural``
(not-a-knot needs 4 knots at an end, 4 total when used at both).

Why a gain table
----------------

Solving for the knot derivatives ``m`` is a linear system ``A m = B_x x + B_v
v0``.  On a **uniform** grid ``A`` depends only on the number of segments ``K``,
so both gain matrices ``G_x = A^-1 B_x`` and ``G_v = A^-1 B_v`` are precomputed
once per ``K`` at construction and the hot path is a single gather plus one
batched matmul -- no linear solve, no Python loop over segments, no
device->host sync.  Rows and columns beyond ``K`` are exactly zero, which is
what makes the per-hand horizon masking *exact*: a waypoint past a hand's hit
index cannot influence any coefficient that hand's cost reads.

Per-element horizons
--------------------

The 17 DoF are partitioned into groups (waist / left arm / right arm; see
:func:`~amortized_opt.tokenizer.joint_group_masks`) and each group carries its
own segment count ``K``.  That is precisely resolved decision 2 of the plan:
left-arm elements live on the left horizon, right-arm on the right, the waist on
the longer of the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .tokenizer import joint_group_masks

__all__ = [
    "SplineGains",
    "SplineSamples",
    "TrajectorySpline",
    "TrajectorySplineBuilder",
    "hermite_coefficients",
    "spline_gain_matrices",
]

_BC_START = ("clamped", "natural", "not-a-knot")
_BC_END = ("natural", "not-a-knot")


# ------------------------------------------------------------------ build ----
def spline_gain_matrices(
    K: int, H: int, h: float, bc_start: str = "clamped", bc_end: str = "natural"
) -> tuple[np.ndarray, np.ndarray]:
    """``(G_x, G_v)`` for a ``K``-segment spline padded to ``H`` segments.

    The knot derivatives are ``m = G_x @ x + G_v * v0`` with ``x`` the ``H + 1``
    knot values (rows beyond ``K`` ignored) and ``v0`` the start velocity.
    ``G_x`` has shape ``(H + 1, H + 1)``, ``G_v`` shape ``(H + 1,)``; rows and
    columns strictly beyond ``K`` are zero.

    Equations (uniform spacing ``h``, Hermite form on each segment):

    * interior ``i``: ``m[i-1] + 4 m[i] + m[i+1] = 3 (x[i+1] - x[i-1]) / h``
    * clamped start: ``m[0] = v0``
    * natural start: ``2 m[0] + m[1] = 3 (x[1] - x[0]) / h``
    * natural end:   ``m[K-1] + 2 m[K] = 3 (x[K] - x[K-1]) / h``
    * not-a-knot start: ``m[0] - m[2] = 2 (2 x[1] - x[0] - x[2]) / h``
    * not-a-knot end:   ``m[K-2] - m[K] = 2 (2 x[K-1] - x[K-2] - x[K]) / h``
    """
    if bc_start not in _BC_START:
        raise ValueError(f"bc_start must be one of {_BC_START}, got {bc_start!r}")
    if bc_end not in _BC_END:
        raise ValueError(f"bc_end must be one of {_BC_END}, got {bc_end!r}")
    M = int(H) + 1
    Gx = np.zeros((M, M), dtype=np.float64)
    Gv = np.zeros(M, dtype=np.float64)
    K = int(K)
    if K < 1:  # no segments: every derivative is zero
        return Gx, Gv

    A = np.zeros((M, M), dtype=np.float64)
    Bx = np.zeros((M, M), dtype=np.float64)
    Bv = np.zeros(M, dtype=np.float64)
    inv_h = 1.0 / float(h)

    start = bc_start
    end = bc_end
    # not-a-knot needs three distinct segments at that end; with both ends
    # not-a-knot and K < 4 the two rows collapse onto the same equation.
    if start == "not-a-knot" and K < 3:
        start = "natural"
    if end == "not-a-knot" and (K < 3 or (start == "not-a-knot" and K < 4)):
        end = "natural"

    if start == "clamped":
        A[0, 0] = 1.0
        Bv[0] = 1.0
    elif start == "natural":
        A[0, 0], A[0, 1] = 2.0, 1.0
        Bx[0, 0], Bx[0, 1] = -3.0 * inv_h, 3.0 * inv_h
    else:  # not-a-knot
        A[0, 0], A[0, 2] = 1.0, -1.0
        Bx[0, 0], Bx[0, 1], Bx[0, 2] = -2.0 * inv_h, 4.0 * inv_h, -2.0 * inv_h

    for i in range(1, K):
        A[i, i - 1], A[i, i], A[i, i + 1] = 1.0, 4.0, 1.0
        Bx[i, i - 1], Bx[i, i + 1] = -3.0 * inv_h, 3.0 * inv_h

    if end == "natural":
        A[K, K - 1], A[K, K] = 1.0, 2.0
        Bx[K, K - 1], Bx[K, K] = -3.0 * inv_h, 3.0 * inv_h
    else:  # not-a-knot
        A[K, K - 2], A[K, K] = 1.0, -1.0
        Bx[K, K - 2], Bx[K, K - 1], Bx[K, K] = -2.0 * inv_h, 4.0 * inv_h, -2.0 * inv_h

    for i in range(K + 1, M):
        A[i, i] = 1.0

    if abs(float(np.linalg.det(A[: K + 1, : K + 1]))) < 1e-12:
        raise RuntimeError(
            f"singular spline system for K={K} ({bc_start!r}/{bc_end!r}); "
            "this is a bug in the boundary-condition fallbacks"
        )
    Ainv = np.linalg.inv(A)
    Gx = Ainv @ Bx
    Gv = Ainv @ Bv
    # Numerical hygiene: the padded block is exactly zero by construction.
    Gx[K + 1 :, :] = 0.0
    Gx[:, K + 1 :] = 0.0
    Gv[K + 1 :] = 0.0
    return Gx, Gv


@dataclass
class SplineGains:
    """Precomputed gain tables, one row per segment count ``K = 0 .. H``."""

    Gx: torch.Tensor  # (H+1, H+1, H+1)
    Gv: torch.Tensor  # (H+1, H+1)
    max_segments: int
    dt: float
    bc_start: str
    bc_end: str


def hermite_coefficients(
    x: torch.Tensor, m: torch.Tensor, h: float
) -> torch.Tensor:
    """Per-segment cubic coefficients from knot values and derivatives.

    ``x`` / ``m`` are ``(B, H+1, n)``; the result is ``(B, H, n, 4)`` with the
    local polynomial ``p(s) = a0 + a1 s + a2 s^2 + a3 s^3`` on ``s in [0, h]``.
    """
    x0, x1 = x[:, :-1], x[:, 1:]
    m0, m1 = m[:, :-1], m[:, 1:]
    d = (x1 - x0) / h
    a0 = x0
    a1 = m0
    a2 = (3.0 * d - 2.0 * m0 - m1) / h
    a3 = (-2.0 * d + m0 + m1) / (h * h)
    return torch.stack([a0, a1, a2, a3], dim=-1)


# ------------------------------------------------------------------ eval -----
@dataclass
class SplineSamples:
    """Spline evaluated on a uniform sub-sample grid.

    ``q`` / ``qd`` / ``qdd`` / ``qddd`` are ``(B, S, n)``; ``mask`` is 1.0 where
    the sample belongs to a segment inside that element's horizon; ``t`` is the
    ``(S,)`` sample time in seconds and ``knot_stride`` the index step between
    consecutive knots (so knot ``i`` is at sample ``i * knot_stride``).
    """

    q: torch.Tensor
    qd: torch.Tensor
    qdd: torch.Tensor
    qddd: torch.Tensor
    mask: torch.Tensor
    t: torch.Tensor
    knot_stride: int

    @property
    def n_samples(self) -> int:
        return int(self.q.shape[1])


class TrajectorySpline:
    """One built spline for a batch: coefficients + horizon bookkeeping.

    Construct via :class:`TrajectorySplineBuilder`; this object is cheap and
    holds only the coefficients and masks.
    """

    __slots__ = ("coeffs", "x", "m", "seg_mask", "lengths", "dt", "n_dof", "max_segments")

    def __init__(
        self,
        coeffs: torch.Tensor,  # (B, H, n, 4)
        x: torch.Tensor,  # (B, H+1, n)
        m: torch.Tensor,  # (B, H+1, n)
        seg_mask: torch.Tensor,  # (B, H, n)
        lengths: torch.Tensor,  # (B, n) long -- segments per element
        dt: float,
    ):
        self.coeffs = coeffs
        self.x = x
        self.m = m
        self.seg_mask = seg_mask
        self.lengths = lengths
        self.dt = float(dt)
        self.n_dof = int(x.shape[-1])
        self.max_segments = int(coeffs.shape[1])

    # ---------------------------------------------------------------- eval --
    def eval_offsets(self, s: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate at per-segment offsets ``s`` (shape ``(P,)``, in seconds).

        Returns ``(q, qd, qdd, qddd)``, each ``(B, H, P, n)``.
        """
        a0, a1, a2, a3 = (self.coeffs[..., i] for i in range(4))  # (B, H, n)
        s = s.to(a0.dtype).view(1, 1, -1, 1)  # (1, 1, P, 1)
        a0, a1, a2, a3 = (a.unsqueeze(2) for a in (a0, a1, a2, a3))
        s2 = s * s
        s3 = s2 * s
        q = a0 + a1 * s + a2 * s2 + a3 * s3
        qd = a1 + 2.0 * a2 * s + 3.0 * a3 * s2
        qdd = 2.0 * a2 + 6.0 * a3 * s
        qddd = (6.0 * a3).expand_as(qdd)
        return q, qd, qdd, qddd

    def subsample_foh(self, n_per_segment: int = 5) -> SplineSamples:
        """Actual first-order-hold command, holding shorter-arm terminal poses.

        This describes the servo command, not the servo's reached state.
        """
        n = max(int(n_per_segment), 1)
        B, _, D = self.x.shape
        x = self.x_held
        fraction = torch.arange(n, device=x.device, dtype=x.dtype)/n
        delta = x[:, 1:]-x[:, :-1]
        q = x[:, :-1, None]+fraction[None, None, :, None]*delta[:, :, None]
        qd = (delta/self.dt)[:, :, None].expand_as(q)
        q = torch.cat((q.reshape(B, -1, D), x[:, -1:]), dim=1)
        qd = torch.cat((qd.reshape(B, -1, D), qd[:, -1, -1:, :]), dim=1)
        index = torch.arange(q.shape[1], device=x.device)[None, :, None]
        mask = (index <= self.lengths[:, None]*n).to(x.dtype)
        zero = torch.zeros_like(q)
        return SplineSamples(q, qd, zero, zero, mask,
            torch.arange(q.shape[1], device=x.device, dtype=x.dtype)*(self.dt/n), n)

    def segment_position_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact cubic position extrema, including all interior velocity roots."""
        a0, a1, a2, a3 = (self.coeffs[..., i] for i in range(4))
        A, B, C = 3*a3, 2*a2, a1
        quadratic = A.abs() > 1e-12
        safe_A = torch.where(quadratic, A, torch.ones_like(A))
        discriminant = B.square()-4*A*C
        root = torch.sqrt(discriminant.clamp_min(1e-24))
        candidates = []
        for sign in (-1, 1):
            s = (-B+sign*root)/(2*safe_A)
            valid = quadratic & (discriminant >= 0) & (s >= 0) & (s <= self.dt)
            candidates.append(torch.where(valid, s, torch.zeros_like(s)))
        linear = (~quadratic) & (B.abs() > 1e-12)
        s = -C/torch.where(linear, B, torch.ones_like(B))
        candidates.append(torch.where(linear & (s >= 0) & (s <= self.dt), s, torch.zeros_like(s)))
        candidates.extend((torch.zeros_like(a0), torch.full_like(a0, self.dt)))
        s = torch.stack(candidates, dim=-1)
        values = a0[..., None]+s*(a1[..., None]+s*(a2[..., None]+s*a3[..., None]))
        return values.amin(-1), values.amax(-1)

    def subsample(self, n_per_segment: int = 1, *, include_end: bool = True) -> SplineSamples:
        """Uniform grid of ``n_per_segment`` samples per segment.

        With ``include_end`` (the default) the terminal knot is appended, so
        ``S = H * n_per_segment + 1`` and knot ``i`` sits at index
        ``i * n_per_segment`` -- the indexing the collision/strike windows use.
        """
        n = max(int(n_per_segment), 1)
        h = self.dt
        dev, dtype = self.coeffs.device, self.coeffs.dtype
        s = torch.arange(n, device=dev, dtype=dtype) * (h / n)  # (n,)
        q, qd, qdd, qddd = self.eval_offsets(s)  # (B, H, n, dof)
        B, H, _, D = q.shape
        flat = lambda a: a.reshape(B, H * n, D)
        q, qd, qdd, qddd = (flat(a) for a in (q, qd, qdd, qddd))
        t = (torch.arange(H * n, device=dev, dtype=dtype)) * (h / n)
        if include_end:
            end = torch.full((1,), h, device=dev, dtype=dtype)
            qe, qde, qdde, qddde = self.eval_offsets(end)  # (B, H, 1, dof)
            take = lambda a: a[:, -1]  # last segment, the only endpoint we need
            q = torch.cat([q, take(qe)], dim=1)
            qd = torch.cat([qd, take(qde)], dim=1)
            qdd = torch.cat([qdd, take(qdde)], dim=1)
            qddd = torch.cat([qddd, take(qddde)], dim=1)
            t = torch.cat([t, torch.full((1,), H * h, device=dev, dtype=dtype)])
        # A sample is valid up to and INCLUDING that element's terminal knot
        # (sample ``K * n``), which is where the strike is scored -- deriving the
        # mask from ``seg_mask`` would drop exactly that sample, since segment K
        # is already outside the horizon.  ``hold_terminal`` in the builder is
        # what makes the sample AT ``K * n`` equal the terminal knot.
        s_idx = torch.arange(q.shape[1], device=dev, dtype=torch.long).view(1, -1, 1)
        mask = (s_idx <= (self.lengths * n).unsqueeze(1)).to(dtype)
        return SplineSamples(
            q=q, qd=qd, qdd=qdd, qddd=qddd, mask=mask, t=t, knot_stride=n
        )

    # ------------------------------------------------------- per-segment ----
    def segment_velocity_max(self, n_probe: int = 5) -> torch.Tensor:
        """``(B, H, n)`` max ``|qd|`` per segment, by sub-sampling.

        ``n_probe`` points spread over the *closed* segment (both endpoints
        included), per the GPU review's recommendation to sample rather than
        locate the extremum analytically.  Velocity is quadratic in ``s``, so
        five probes bound the true maximum to well under a percent in practice;
        :meth:`segment_velocity_max_analytic` is the exact version and
        ``tests/test_tspline.py`` reports the gap.
        """
        n = max(int(n_probe), 2)
        s = torch.arange(n, device=self.coeffs.device, dtype=self.coeffs.dtype)
        s = s * (self.dt / (n - 1))
        _q, qd, _a, _j = self.eval_offsets(s)
        return qd.abs().amax(dim=2) * self.seg_mask

    def segment_velocity_max_analytic(self) -> torch.Tensor:
        """``(B, H, n)`` exact max ``|qd|`` per segment.

        ``qd(s) = a1 + 2 a2 s + 3 a3 s^2`` is a parabola: its extremum sits at
        ``s* = -a2 / (3 a3)``, so the maximum of ``|qd|`` over ``[0, h]`` is
        attained at ``0``, ``h`` or the clamped ``s*``.
        """
        a1, a2, a3 = self.coeffs[..., 1], self.coeffs[..., 2], self.coeffs[..., 3]
        h = self.dt
        star = (-a2 / torch.where(a3.abs() > 1e-12, 3.0 * a3, torch.full_like(a3, 1e-12)))
        star = star.clamp(0.0, h)
        vel = lambda s: a1 + 2.0 * a2 * s + 3.0 * a3 * s * s
        v0 = vel(torch.zeros_like(star))
        v1 = vel(torch.full_like(star, h))
        vs = vel(star)
        out = torch.maximum(torch.maximum(v0.abs(), v1.abs()), vs.abs())
        return out * self.seg_mask

    def segment_integrals(self) -> dict[str, torch.Tensor]:
        """Closed-form ``int v^2``, ``int a^2``, ``int j^2`` per segment.

        Each is ``(B, H, n)``, already zeroed outside the element's horizon.
        For ``p(s) = a0 + a1 s + a2 s^2 + a3 s^3`` on ``[0, h]``::

            int v^2 = a1^2 h + 2 a1 a2 h^2 + (4 a2^2 + 6 a1 a3) h^3 / 3
                      + 3 a2 a3 h^4 + 9 a3^2 h^5 / 5
            int a^2 = 4 a2^2 h + 12 a2 a3 h^2 + 12 a3^2 h^3
            int j^2 = 36 a3^2 h
        """
        a1, a2, a3 = self.coeffs[..., 1], self.coeffs[..., 2], self.coeffs[..., 3]
        h = self.dt
        h2, h3, h4, h5 = h * h, h**3, h**4, h**5
        iv = (
            a1 * a1 * h
            + 2.0 * a1 * a2 * h2
            + (4.0 * a2 * a2 + 6.0 * a1 * a3) * (h3 / 3.0)
            + 3.0 * a2 * a3 * h4
            + 9.0 * a3 * a3 * (h5 / 5.0)
        )
        ia = 4.0 * a2 * a2 * h + 12.0 * a2 * a3 * h2 + 12.0 * a3 * a3 * h3
        ij = 36.0 * a3 * a3 * h
        msk = self.seg_mask
        return {"vel2": iv * msk, "acc2": ia * msk, "jerk2": ij * msk}

    # -------------------------------------------------- knot finite diffs ----
    @property
    def x_held(self) -> torch.Tensor:
        """Knot values with each element HELD at its terminal beyond its horizon.

        ``self.x`` zeroes those entries (that is what makes the gain-table
        masking exact); this view replaces them with the terminal value so a
        finite difference across the boundary is zero rather than a cliff.
        """
        idx = torch.arange(
            self.max_segments + 1, device=self.x.device, dtype=torch.long
        ).view(1, -1, 1)
        valid = (idx <= self.lengths.unsqueeze(1)).to(self.x.dtype)
        term = self.x.gather(1, self.lengths.unsqueeze(1))
        return self.x * valid + term * (1.0 - valid)

    def knot_finite_differences(
        self, v0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q, qd, qdd)`` at the knots under the SCORER's convention.

        ``qd[0] = v0``, ``qd[i] = (x[i] - x[i-1]) / dt``,
        ``qdd[i] = (qd[i+1] - qd[i]) / dt`` with ``qdd[K] = qdd[K-1]`` --
        exactly ``eval_harness.TrajectoryScorer.score``, which is also what a
        first-order-hold PD plant is actually commanded to do between knots.
        Each is ``(B, H+1, n)``; entries beyond an element's horizon are the
        held terminal (position) or zero (rates).
        """
        h = self.dt
        x = self.x_held
        qd = torch.cat([v0.unsqueeze(1), (x[:, 1:] - x[:, :-1]) / h], dim=1)
        qdd_f = torch.cat(
            [(qd[:, 1:] - qd[:, :-1]) / h, (qd[:, -1:] - qd[:, -2:-1]) / h], dim=1
        )
        qdd_prev = torch.cat([qdd_f[:, :1], qdd_f[:, :-1]], dim=1)
        idx = torch.arange(
            self.max_segments + 1, device=x.device, dtype=torch.long
        ).view(1, -1, 1)
        at_end = (idx == self.lengths.unsqueeze(1))
        qdd = torch.where(at_end, qdd_prev, qdd_f)
        inside = (idx <= self.lengths.unsqueeze(1)).to(x.dtype)
        return x, qd * inside, qdd * inside

    # ---------------------------------------------------------- terminals ----
    def knot(self, index: torch.Tensor) -> torch.Tensor:
        """Gather knot positions at per-sample indices ``(B,)`` -> ``(B, n)``."""
        idx = index.clamp(0, self.max_segments).view(-1, 1, 1).expand(-1, 1, self.n_dof)
        return self.x.gather(1, idx).squeeze(1)

    def knot_derivative(self, index: torch.Tensor) -> torch.Tensor:
        """Gather knot spline velocities at per-sample indices ``(B,)``."""
        idx = index.clamp(0, self.max_segments).view(-1, 1, 1).expand(-1, 1, self.n_dof)
        return self.m.gather(1, idx).squeeze(1)


class TrajectorySplineBuilder:
    """Holds the gain tables; turns waypoints into a :class:`TrajectorySpline`.

    ``group_of_joint`` assigns every DoF to a horizon group.  The default is the
    plan's waist / left / right split, taken from the joint names rather than
    assumed positionally.
    """

    def __init__(
        self,
        *,
        max_segments: int = 100,
        dt: float = CONTROL_DT,
        bc_start: str = "clamped",
        bc_end: str = "natural",
        hold_terminal: bool = True,
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.max_segments = int(max_segments)
        self.dt = float(dt)
        self.bc_start = bc_start
        self.bc_end = bc_end
        self.hold_terminal = bool(hold_terminal)
        self.joint_names = tuple(joint_names)
        self.n_dof = len(self.joint_names)
        self.device = torch.device(device)
        self.dtype = dtype

        groups = joint_group_masks(self.joint_names)
        order = ("waist", "left", "right")
        g = np.zeros(self.n_dof, dtype=np.int64)
        for gi, name in enumerate(order):
            g[groups[name] > 0.5] = gi
        self.group_names = order
        self.n_groups = len(order)
        self._group_np = g
        self._group_cols = [np.nonzero(g == gi)[0] for gi in range(self.n_groups)]

        H = self.max_segments
        Gx = np.zeros((H + 1, H + 1, H + 1), dtype=np.float64)
        Gv = np.zeros((H + 1, H + 1), dtype=np.float64)
        for K in range(H + 1):
            gx, gv = spline_gain_matrices(K, H, self.dt, bc_start, bc_end)
            Gx[K], Gv[K] = gx, gv
        self._Gx_np, self._Gv_np = Gx, Gv
        self._tables: dict[torch.dtype, SplineGains] = {}
        self._cols: dict[tuple, list[torch.Tensor]] = {}
        self._gains(self.dtype)

    # --------------------------------------------------------------- build --
    def _gains(self, dtype: torch.dtype) -> SplineGains:
        hit = self._tables.get(dtype)
        if hit is not None:
            return hit
        t = lambda a: torch.tensor(a.tolist(), device=self.device, dtype=dtype)
        out = SplineGains(
            Gx=t(self._Gx_np),
            Gv=t(self._Gv_np),
            max_segments=self.max_segments,
            dt=self.dt,
            bc_start=self.bc_start,
            bc_end=self.bc_end,
        )
        self._tables[dtype] = out
        return out

    def _col_index(self, device, dtype=None) -> list[torch.Tensor]:
        key = (str(device),)
        hit = self._cols.get(key)
        if hit is None:
            hit = [
                torch.tensor(c.tolist(), device=device, dtype=torch.long)
                for c in self._group_cols
            ]
            self._cols[key] = hit
        return hit

    def _group_row(self, device) -> torch.Tensor:
        """``(n,)`` group id per joint, cached on ``device`` (no per-call H2D)."""
        key = ("grp", str(device))
        hit = self._cols.get(key)
        if hit is None:
            hit = torch.tensor(
                self._group_np.tolist(), device=device, dtype=torch.long
            )
            self._cols[key] = hit
        return hit

    def to(self, device, dtype: torch.dtype | None = None) -> "TrajectorySplineBuilder":
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self._tables.clear()
        self._cols.clear()
        self._gains(self.dtype)
        return self

    def lengths_from_mask(self, elem_mask: torch.Tensor) -> torch.Tensor:
        """``(B, H, n)`` prefix mask -> ``(B, G)`` segment counts per group.

        The mask is a per-element prefix (``build_waypoint_masks`` guarantees
        it), so the count of ones is the hit index for that element.
        """
        counts = elem_mask.to(torch.float32).sum(dim=1)  # (B, n)
        cols = self._col_index(elem_mask.device, elem_mask.dtype)
        return torch.stack(
            [counts.index_select(1, c).amax(dim=1) for c in cols], dim=1
        ).round().to(torch.long)

    def build(
        self,
        q0: torch.Tensor,  # (B, n)
        waypoints: torch.Tensor,  # (B, H, n)
        v0: torch.Tensor,  # (B, n)
        lengths: torch.Tensor,  # (B, G) long, segments per group
    ) -> TrajectorySpline:
        if waypoints.shape[1] != self.max_segments:
            raise ValueError(
                f"builder was made for H={self.max_segments} segments, "
                f"got {waypoints.shape[1]}"
            )
        dtype = waypoints.dtype
        gains = self._gains(dtype)
        H = self.max_segments
        B, _, D = waypoints.shape
        cols = self._col_index(waypoints.device, dtype)

        lengths = lengths.clamp(1, H).to(torch.long)
        x = torch.cat([q0.unsqueeze(1), waypoints], dim=1)  # (B, H+1, n)

        # Knot index grid, for zeroing beyond each element's horizon.
        knot_i = torch.arange(H + 1, device=x.device, dtype=torch.long).view(1, -1, 1)
        per_elem_len = lengths.index_select(1, self._group_row(x.device))  # (B, n)
        knot_valid = (knot_i <= per_elem_len.unsqueeze(1)).to(dtype)  # (B, H+1, n)
        x = x * knot_valid

        m_parts = []
        for gi, c in enumerate(cols):
            k = lengths[:, gi]  # (B,)
            Gx = gains.Gx.index_select(0, k)  # (B, H+1, H+1)
            Gv = gains.Gv.index_select(0, k)  # (B, H+1)
            xg = x.index_select(2, c)  # (B, H+1, ng)
            vg = v0.index_select(1, c)  # (B, ng)
            mg = torch.bmm(Gx, xg) + Gv.unsqueeze(-1) * vg.unsqueeze(1)
            m_parts.append(mg)
        m = x.new_zeros(B, H + 1, D)
        for c, mg in zip(cols, m_parts):
            m = m.index_copy(2, c, mg)

        coeffs = hermite_coefficients(x, m, self.dt)
        seg_i = torch.arange(H, device=x.device, dtype=torch.long).view(1, -1, 1)
        seg_mask = (seg_i < per_elem_len.unsqueeze(1)).to(dtype)  # (B, H, n)
        if self.hold_terminal:
            # Past its horizon an element HOLDS its terminal value instead of
            # dropping to zero.  Nothing masked reads it, but the *other* group's
            # terms do -- the waist torque and the idle arm's collision geometry
            # are then computed against an arm parked where it finished, which is
            # the physical picture, rather than against URDF neutral.
            x_term = x.gather(1, per_elem_len.unsqueeze(1))  # (B, 1, n)
            a0 = coeffs[..., 0] * seg_mask + (1.0 - seg_mask) * x_term
            coeffs = torch.stack(
                [a0] + [coeffs[..., i] * seg_mask for i in (1, 2, 3)], dim=-1
            )
        else:
            coeffs = coeffs * seg_mask.unsqueeze(-1)
        return TrajectorySpline(coeffs, x, m, seg_mask, per_elem_len, self.dt)

    def from_masks(
        self,
        q0: torch.Tensor,
        waypoints: torch.Tensor,
        v0: torch.Tensor,
        elem_mask: torch.Tensor,
    ) -> TrajectorySpline:
        """Convenience wrapper: derive per-group horizons from an element mask."""
        return self.build(q0, waypoints, v0, self.lengths_from_mask(elem_mask))
