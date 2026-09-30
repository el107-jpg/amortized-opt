"""Learned convex quadratic program for the terminal-pose network.

The hidden state writes every block of

    min_z  1/2 z^T Q z + p^T z
    s.t.   A z = b
           l <= G z <= u

``Q``, ``p``, ``A``, ``b``, ``G``, ``l`` and ``u`` are generated from the
hidden state around fixed, symmetry-breaking initial values.  Nothing in the
program is a joint limit or a ready pose.  The parameterization constructs a
strictly feasible witness for every program, allowing the solve and backward
pass to use batched, device-native qpth operations rather than serial CPU
ProxQP calls.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import N_DOF


class JointQuadraticLayer(nn.Module):
    """Hidden features write a convex QP in the 17 joints, then we solve it.

    .. math::

        min_z  1/2 z^T Q z + p^T z
        s.t.   l <= G z <= u
               A z = b

    ``Q``, ``p``, ``A``, ``b``, ``G``, ``l`` and ``u`` all depend on the
    hidden state.  Nothing in the program is a joint limit, a ready pose,
    or a collision row supplied by us.  Fixed identity rows only provide a
    well-conditioned, symmetry-breaking initialization; every row can move.
    The bounds are two-sided, ``l <= G z <= u``, so a learned collision plane
    is just another row.

    Every generated program is feasible by construction.  ``A`` is kept full
    row rank and the minimum-norm equality witness

    ``z0 = A.T @ solve(A @ A.T, b)``

    satisfies ``A z0 = b``.  Positive learned margins place ``G z0`` strictly
    between ``l`` and ``u``.  This is not a restriction on feasible QPs: every
    feasible program has at least one such witness.  It does remove the
    meaningless parameterizations where the network writes contradictory
    constraints.

    ``Q = ε I + L L^T`` keeps the quadratic bounded.  Importantly, ``L``
    starts at the identity rather than zero: at ``L = 0`` the derivative of
    ``L L^T`` with respect to ``L`` is zero and the Hessian head can never
    start learning.  Inequality normals likewise start nonzero and their
    half-widths are positive, so the constraint heads do not begin at the
    degenerate program ``0 <= 0 <= 0``.
    """

    def __init__(
        self,
        hidden: int,
        n: int = N_DOF,
        n_eq: int = 4,
        n_ineq: int = 4,
    ):
        super().__init__()
        if n_eq < 1 or n_eq > n:
            raise ValueError(f"n_eq must be in [1, {n}], got {n_eq}")
        if n_ineq < 1:
            raise ValueError(f"n_ineq must be >= 1, got {n_ineq}")
        self.n = int(n)
        self.n_eq = int(n_eq)
        self.n_ineq = int(n_ineq)
        # The ridge guarantees strict convexity.  The identity factor is
        # also necessary for gradient flow through L @ L.T at initialization.
        self.q_eps = 1e-2
        self.q_factor_init = 1.0
        self.bound_half_min = 1e-4
        self.bound_half_init = 1e-2
        # tanh-bounded changes to the leading k-by-k identity block leave it
        # strictly diagonally dominant, hence A remains full row rank.
        self.a_delta_scale = 0.5 / self.n_eq
        self.bound_width_offset = math.log(
            math.expm1(self.bound_half_init - self.bound_half_min)
        )
        tril = torch.tril_indices(self.n, self.n)
        self.register_buffer("tril_r", tril[0])
        self.register_buffer("tril_c", tril[1])
        self.to_chol = nn.Linear(hidden, self.n * (self.n + 1) // 2)
        self.to_p = nn.Linear(hidden, n)
        self.to_G = nn.Linear(hidden, self.n_ineq * n)
        self.to_l = nn.Linear(hidden, self.n_ineq)
        self.to_u = nn.Linear(hidden, self.n_ineq)
        self.to_A = nn.Linear(hidden, self.n_eq * n)
        self.to_b = nn.Linear(hidden, self.n_eq)

        # Deterministic bases make the initial A full-row-rank and G nonzero.
        # They are non-persistent so old checkpoints keep the same state-dict
        # schema.  G starts on different coordinates from A where possible.
        A0 = torch.zeros(self.n_eq, self.n)
        A0[torch.arange(self.n_eq), torch.arange(self.n_eq)] = 1.0
        G0 = torch.zeros(self.n_ineq, self.n)
        grow = torch.arange(self.n_ineq)
        G0[grow, (self.n_eq + grow) % self.n] = 1.0
        self.register_buffer("A0", A0, persistent=False)
        self.register_buffer("G0", G0, persistent=False)

        for layer in (
            self.to_chol, self.to_p, self.to_G, self.to_l, self.to_u, self.to_A, self.to_b,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # A small instance-dependent linear objective moves the initial solve
        # onto some inequality faces.  Exact zero would leave every inequality
        # inactive (or weakly active with zero multiplier), hence gradientless.
        nn.init.normal_(self.to_p.weight, std=0.02)

    def qp_data(self, hidden: torch.Tensor):
        """Return the QP tensors and their guaranteed feasible witness.

        Exposing this separately from :meth:`forward` makes activation and
        constraint-health checks possible without duplicating the
        parameterization in tests or diagnostics.
        """
        B, n, k, m = hidden.shape[0], self.n, self.n_eq, self.n_ineq
        raw = self.to_chol(hidden)
        L = hidden.new_zeros(B, n, n)
        L[:, self.tril_r, self.tril_c] = raw
        diag = torch.arange(n, device=hidden.device)
        L[:, diag, diag] += self.q_factor_init
        eye = torch.eye(n, device=hidden.device, dtype=hidden.dtype)
        Q = self.q_eps * eye + L @ L.transpose(1, 2)
        p = self.to_p(hidden)

        A0 = self.A0.to(device=hidden.device, dtype=hidden.dtype)
        A = A0.view(1, k, n) + self.a_delta_scale * torch.tanh(
            self.to_A(hidden).reshape(B, k, n)
        )
        b = self.to_b(hidden)
        gram = A @ A.transpose(1, 2)
        dual_witness = torch.linalg.solve(gram, b.unsqueeze(-1))
        z0 = (A.transpose(1, 2) @ dual_witness).squeeze(-1)

        G0 = self.G0.to(device=hidden.device, dtype=hidden.dtype)
        G = G0.view(1, m, n) + self.to_G(hidden).reshape(B, m, n)
        center = (G @ z0.unsqueeze(-1)).squeeze(-1)
        # The two heads independently control the distance from the witness to
        # the lower and upper faces.  Softplus plus a floor gives strict Slater
        # feasibility and keeps the bounds ordered for every hidden state.
        lower_margin = self.bound_half_min + F.softplus(
            self.to_l(hidden) + self.bound_width_offset
        )
        upper_margin = self.bound_half_min + F.softplus(
            self.to_u(hidden) + self.bound_width_offset
        )
        lower = center - lower_margin
        upper = center + upper_margin
        return Q, p, A, b, G, lower, upper, z0

    def forward(
        self,
        hidden: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        ready: torch.Tensor,
    ) -> torch.Tensor:
        del lo, hi, ready  # the program does not use the joint box or ready pose
        Q, p, A, b, G, lower, upper, _z0 = self.qp_data(hidden)
        z = solve_feasible_qp(
            Q.double(), p.double(), A.double(), b.double(),
            G.double(), lower.double(), upper.double(),
        )
        return z.to(dtype=hidden.dtype)


class CenteredQuadraticLayer(nn.Module):
    """Earlier quadratic layer: preferred pose, fixed joint box, four equalities.

    The equalities start as the first four joints equal to the ready pose.
    The solve is ``qpth``'s refined KKT step, which stays on the tensor device.
    """

    def __init__(self, hidden: int, n: int = N_DOF, n_eq: int = 4, rank: int = 4):
        super().__init__()
        self.n = int(n)
        self.n_eq = int(n_eq)
        self.rank = int(rank)
        self.to_pref = nn.Linear(hidden, n)
        self.to_diag = nn.Linear(hidden, n)
        self.to_factor = nn.Linear(hidden, n * self.rank)
        self.to_dA = nn.Linear(hidden, self.n_eq * n)
        self.to_db = nn.Linear(hidden, self.n_eq)
        for layer in (self.to_pref, self.to_diag, self.to_factor, self.to_dA, self.to_db):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        seed = torch.zeros(self.n_eq, n)
        seed[:, : self.n_eq] = torch.eye(self.n_eq)
        self.register_buffer("A0", seed)
        factor_seed = torch.zeros(n, self.rank)
        d = min(n, self.rank)
        factor_seed[:d, :d] = torch.eye(d)
        # Non-persistent preserves compatibility with checkpoints made before
        # the zero-factor gradient bug was fixed.
        self.register_buffer("factor0", factor_seed, persistent=False)

    def forward(self, hidden, lo, hi, ready):
        B, n, k, r = hidden.shape[0], self.n, self.n_eq, self.rank
        half = (0.5 * (hi - lo)).clamp_min(1e-3)
        t = ready + half * torch.tanh(self.to_pref(hidden))
        diag = F.softplus(self.to_diag(hidden)) + 0.1
        factor0 = self.factor0.to(device=hidden.device, dtype=hidden.dtype)
        factor = 0.1 * (
            factor0.view(1, n, r) + self.to_factor(hidden).reshape(B, n, r)
        )
        Q = torch.diag_embed(diag) + factor @ factor.transpose(1, 2)
        p = -torch.bmm(Q, t.unsqueeze(-1)).squeeze(-1)
        A = self.A0.view(1, k, n) + 0.1 * self.to_dA(hidden).reshape(B, k, n)
        ready_b = ready.view(1, n).expand(B, n)
        b = torch.bmm(A, ready_b.unsqueeze(-1)).squeeze(-1) + 0.1 * self.to_db(hidden)
        eye = torch.eye(n, device=hidden.device, dtype=torch.float64)
        G = torch.cat([eye, -eye], dim=0)
        bound = torch.cat([hi.double(), -lo.double()], dim=0)
        z = _BatchedFeasibleQP.apply(
            Q.double(), p.double(), G, bound, A.double(), b.double(),
        )
        return z.to(dtype=hidden.dtype)


class _BatchedFeasibleQP(torch.autograd.Function):
    """Device-native batched qpth solve with a refined KKT backward pass."""

    @staticmethod
    def forward(ctx, Q, p, G, h, A, b):
        from qpth.solvers.pdipm import batch as pdipm
        from qpth.util import expandParam, extract_nBatch

        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Qe, _ = expandParam(Q, nBatch, 3)
        pe, _ = expandParam(p, nBatch, 2)
        Ge, _ = expandParam(G, nBatch, 3)
        he, _ = expandParam(h, nBatch, 2)
        Ae, _ = expandParam(A, nBatch, 3)
        be, _ = expandParam(b, nBatch, 2)
        Q_LU, S_LU, R = pdipm.pre_factor_kkt(Qe, Ge, Ae)
        z, nu, lam, slack = pdipm.forward(
            Qe, pe, Ge, he, Ae, be, Q_LU, S_LU, R,
            eps=1e-10, verbose=-1, notImprovedLim=8, maxIter=40,
            solver=pdipm.KKTSolvers.LU_PARTIAL,
        )
        ctx.save_for_backward(z, Q, p, G, h, A, b, nu, lam, slack)
        return z

    @staticmethod
    def backward(ctx, dl_dz):
        from qpth.solvers.pdipm import batch as pdipm
        from qpth.util import bger, expandParam, extract_nBatch

        z, Q, p, G, h, A, b, nu, lam, slack = ctx.saved_tensors
        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Qe, Q_e = expandParam(Q, nBatch, 3)
        pe, p_e = expandParam(p, nBatch, 2)
        Ge, G_e = expandParam(G, nBatch, 3)
        he, h_e = expandParam(h, nBatch, 2)
        Ae, A_e = expandParam(A, nBatch, 3)
        be, b_e = expandParam(b, nBatch, 2)
        neq, nineq = Ae.shape[1], Ge.shape[1]
        Q_LU, S_LU, R = pdipm.pre_factor_kkt(Qe, Ge, Ae)
        d = lam.clamp(min=1e-8) / slack.clamp(min=1e-8)
        pdipm.factor_kkt(S_LU, R, d)
        zeros_i = torch.zeros(nBatch, nineq, dtype=z.dtype, device=z.device)
        zeros_e = torch.zeros(nBatch, neq, dtype=z.dtype, device=z.device)
        dx, _, dlam, dnu = pdipm.solve_kkt(
            Q_LU, d, Ge, Ae, S_LU, dl_dz, zeros_i, zeros_i, zeros_e,
        )
        dQ = 0.5 * (bger(dx, z) + bger(z, dx))
        dp = dx
        dG = bger(dlam, z) + bger(lam, dx)
        dh = -dlam
        dA = bger(dnu, z) + bger(nu, dx)
        db = -dnu
        if Q_e:
            dQ = dQ.mean(0)
        if p_e:
            dp = dp.mean(0)
        if G_e:
            dG = dG.mean(0)
        if h_e:
            dh = dh.mean(0)
        if A_e:
            dA = dA.mean(0)
        if b_e:
            db = db.mean(0)
        return dQ, dp, dG, dh, dA, db


def solve_feasible_qp(Q, p, A, b, G, lower, upper):
    """Solve a feasible two-sided QP using batched operations on its device."""
    # qpth uses one-sided rows C z <= h.  The witness constructed by
    # JointQuadraticLayer satisfies both halves strictly.
    C = torch.cat([G, -G], dim=1)
    h = torch.cat([upper, -lower], dim=1)
    return _BatchedFeasibleQP.apply(Q, p, C, h, A, b)
