"""L-BFGS staging adapter for SaLSa-style line search (probe = exec = -H g)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW

from .adamw_ls import DirectionStats, group_ratios_from_optimizer, math_isfinite

__all__ = [
    "LbfgsLineSearchAdapter",
    "StagedLbfgs",
    "step_vector_breakdown_lbfgs",
]


def _flat_dot(xs: Sequence[Tensor], ys: Sequence[Tensor]) -> float:
    acc = torch.zeros((), dtype=torch.float64, device=xs[0].device)
    for x, y in zip(xs, ys):
        acc = acc + torch.dot(
            x.detach().reshape(-1).to(dtype=torch.float64),
            y.detach().reshape(-1).to(dtype=torch.float64),
        )
    return float(acc.item())


def _flat_add(
    acc: list[Tensor], xs: Sequence[Tensor], alpha: float,
) -> list[Tensor]:
    for a, x in zip(acc, xs):
        a.add_(x, alpha=float(alpha))
    return acc


def _zeros_like_params(params: Sequence[Tensor]) -> list[Tensor]:
    return [torch.zeros_like(p) for p in params]


@dataclass
class StagedLbfgs:
    p_probe: list[Tensor]
    p_momentum: list[Tensor]
    p_decay: list[Tensor]
    p_exec: list[Tensor]
    theta0: list[Tensor]
    stats: DirectionStats
    has_grad: list[bool]


class LbfgsLineSearchAdapter:
    """Limited-memory quasi-Newton directions with the same LS surface as AdamW."""

    def __init__(
        self,
        optimizer: AdamW,
        *,
        eta_ref: float | None = None,
        group_ratios: Sequence[float] | None = None,
        history_size: int = 10,
    ) -> None:
        if not isinstance(optimizer, AdamW):
            raise TypeError(
                f"LbfgsLineSearchAdapter requires torch.optim.AdamW shell, "
                f"got {type(optimizer)}"
            )
        self.opt = optimizer
        if eta_ref is None:
            lrs = [float(g["lr"]) for g in optimizer.param_groups if float(g["lr"]) > 0]
            if not lrs:
                raise ValueError("eta_ref required when all group LRs are zero")
            eta_ref = max(lrs)
        self.eta_ref = float(eta_ref)
        if group_ratios is None:
            group_ratios = group_ratios_from_optimizer(optimizer, self.eta_ref)
        if len(group_ratios) != len(optimizer.param_groups):
            raise ValueError("group_ratios length must match optimizer.param_groups")
        self.group_ratios = [float(r) for r in group_ratios]
        self.history_size = max(int(history_size), 1)

        self._params: list[Tensor] = []
        self._param_group_idx: list[int] = []
        for gi, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                self._params.append(p)
                self._param_group_idx.append(gi)

        self._theta0: list[Tensor] | None = None
        self._buf_candidate: list[Tensor] | None = None
        self._p_probe: list[Tensor] | None = None
        self._p_mom: list[Tensor] | None = None
        self._p_decay: list[Tensor] | None = None
        self._p_exec: list[Tensor] | None = None
        self._theta0_ready: bool = False
        self._staged: StagedLbfgs | None = None
        self._s_hist: list[list[Tensor]] = []
        self._y_hist: list[list[Tensor]] = []
        self.n_pairs_skipped: int = 0
        self.n_pairs_accepted: int = 0

    def reset_history(self) -> None:
        self._s_hist.clear()
        self._y_hist.clear()

    # ------------------------------------------------------------------ storage
    def _ensure_buffers(self) -> None:
        if self._theta0 is not None and len(self._theta0) == len(self._params):
            ok = True
            for b, p in zip(self._theta0, self._params):
                if b.shape != p.shape or b.dtype != p.dtype or b.device != p.device:
                    ok = False
                    break
            if ok:
                return
        self._theta0 = [torch.empty_like(p) for p in self._params]
        self._buf_candidate = [torch.empty_like(p) for p in self._params]
        self._p_probe = [torch.empty_like(p) for p in self._params]
        self._p_mom = [torch.empty_like(p) for p in self._params]
        self._p_decay = [torch.empty_like(p) for p in self._params]
        self._p_exec = [torch.empty_like(p) for p in self._params]
        self._theta0_ready = False

    def snapshot_theta0(self) -> list[Tensor]:
        self._ensure_buffers()
        assert self._theta0 is not None
        for dst, p in zip(self._theta0, self._params):
            dst.copy_(p.detach())
        self._theta0_ready = True
        return self._theta0

    def mark_theta0_snapshot(self) -> list[Tensor]:
        return self.snapshot_theta0()

    def restore_theta0(self) -> None:
        if self._theta0 is None:
            return
        with torch.no_grad():
            for p, t0 in zip(self._params, self._theta0):
                p.copy_(t0)

    def clear_transaction(self) -> None:
        self._staged = None
        self._theta0_ready = False

    # --------------------------------------------------------------- two-loop
    def _lbfgs_direction(self, g: list[Tensor | None], has_grad: list[bool]) -> list[Tensor]:
        assert self._p_probe is not None
        work = [
            torch.zeros_like(p) if (not has_grad[i] or g[i] is None)
            else g[i].detach().clone()
            for i, p in enumerate(self._params)
        ]
        alphas: list[float] = []
        rhos: list[float] = []
        m = len(self._s_hist)
        for i in range(m - 1, -1, -1):
            s_i, y_i = self._s_hist[i], self._y_hist[i]
            denom = _flat_dot(s_i, y_i)
            if not math_isfinite(denom) or denom <= 1.0e-20:
                continue
            rho = 1.0 / denom
            alpha = rho * _flat_dot(s_i, work)
            alphas.append(alpha)
            rhos.append(rho)
            _flat_add(work, y_i, -alpha)

        if m > 0:
            s_last, y_last = self._s_hist[-1], self._y_hist[-1]
            yty = _flat_dot(y_last, y_last)
            if yty > 1.0e-20:
                gamma = _flat_dot(y_last, s_last) / yty
                if math_isfinite(gamma) and gamma > 0.0:
                    for w in work:
                        w.mul_(gamma)

        for i in range(len(rhos)):
            s_i, y_i = self._s_hist[m - len(rhos) + i], self._y_hist[m - len(rhos) + i]
            rho = rhos[i]
            alpha = alphas[len(rhos) - 1 - i]
            beta = rho * _flat_dot(y_i, work)
            _flat_add(work, s_i, alpha - beta)

        out = self._p_probe
        for i, w in enumerate(work):
            out[i].copy_(-w)
        return out

    # ------------------------------------------------------------------- stage
    def stage_from_q(
        self,
        *,
        g_objective: Sequence[Tensor | None] | None = None,
        g_preclip: Sequence[Tensor | None] | None = None,
        use_param_grad_as_q: bool = True,
        q_list: Sequence[Tensor | None] | None = None,
    ) -> StagedLbfgs:
        self._ensure_buffers()
        assert (
            self._theta0 is not None
            and self._p_probe is not None
            and self._p_mom is not None
            and self._p_decay is not None
            and self._p_exec is not None
        )
        if not self._theta0_ready:
            self.snapshot_theta0()

        has_grad: list[bool] = []
        g_work: list[Tensor | None] = []
        s_probe_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        P2_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        g2_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        g2_objective_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)

        for i, p in enumerate(self._params):
            if q_list is not None:
                q = q_list[i]
            elif use_param_grad_as_q:
                q = p.grad
            else:
                q = None
            pm = self._p_mom[i]
            pd = self._p_decay[i]
            pe = self._p_exec[i]
            t0 = self._theta0[i]
            gi = self._param_group_idx[i]
            wd = float(self.opt.param_groups[gi]["weight_decay"])
            r = float(self.group_ratios[gi])

            if q is None:
                has_grad.append(False)
                g_work.append(None)
                pm.zero_()
                pd.zero_()
                pe.zero_()
                self._p_probe[i].zero_()
                continue

            has_grad.append(True)
            grad = q.detach()
            g_work.append(grad)
            if wd != 0.0:
                pd.copy_((-r * wd) * t0)
            else:
                pd.zero_()
            pm.zero_()

        pp = self._lbfgs_direction(g_work, has_grad)
        for i, ok in enumerate(has_grad):
            pe = self._p_exec[i]
            if not ok:
                pe.zero_()
                continue
            pe.copy_(pp[i]).add_(self._p_decay[i])

        for i, ok in enumerate(has_grad):
            if not ok:
                continue
            q64 = g_work[i].reshape(-1).to(dtype=torch.float64)
            g_obj = (
                g_objective[i]
                if g_objective is not None
                else (
                    g_preclip[i]
                    if g_preclip is not None
                    else g_work[i]
                )
            )
            if g_obj is None:
                raise ValueError(
                    "g_objective is missing for a parameter with optimizer gradient"
                )
            go64 = g_obj.detach().reshape(-1).to(dtype=torch.float64)
            pp64 = pp[i].reshape(-1).to(dtype=torch.float64)
            pe64 = self._p_exec[i].reshape(-1).to(dtype=torch.float64)
            s_probe_acc = s_probe_acc + torch.dot(go64, pp64)
            P2_acc = P2_acc + torch.dot(pp64, pp64)
            g2_acc = g2_acc + torch.dot(q64, q64)
            g2_objective_acc = g2_objective_acc + torch.dot(go64, go64)

        s_probe = float(s_probe_acc.item())
        a = -s_probe
        P2 = float(P2_acc.item())
        probe_valid = math_isfinite(a) and math_isfinite(P2) and a > 0.0 and P2 > 0.0
        s_exec_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_dec_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_probe_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_exec_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_dec_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        for i, ok in enumerate(has_grad):
            if not ok:
                continue
            q64 = g_work[i].reshape(-1).to(dtype=torch.float64)
            g_obj = (
                g_objective[i]
                if g_objective is not None
                else (
                    g_preclip[i]
                    if g_preclip is not None
                    else g_work[i]
                )
            )
            assert g_obj is not None
            go64 = g_obj.detach().reshape(-1).to(dtype=torch.float64)
            pp64 = pp[i].reshape(-1).to(dtype=torch.float64)
            pe64 = self._p_exec[i].reshape(-1).to(dtype=torch.float64)
            pd64 = self._p_decay[i].reshape(-1).to(dtype=torch.float64)
            s_exec_acc = s_exec_acc + torch.dot(go64, pe64)
            s_dec_acc = s_dec_acc + torch.dot(go64, pd64)
            s_probe_q_acc = s_probe_q_acc + torch.dot(q64, pp64)
            s_exec_q_acc = s_exec_q_acc + torch.dot(q64, pe64)
            s_dec_q_acc = s_dec_q_acc + torch.dot(q64, pd64)
        stats = DirectionStats(
            a=a, P2=P2, s_probe=-a,
            s_exec=float(s_exec_acc.item()),
            s_momentum=0.0,
            s_decay=float(s_dec_acc.item()),
            s_probe_q=float(s_probe_q_acc.item()),
            s_exec_q=float(s_exec_q_acc.item()),
            s_momentum_q=0.0,
            s_decay_q=float(s_dec_q_acc.item()),
            gnorm_q=float(g2_acc.sqrt().item()),
            gnorm_objective=float(g2_objective_acc.sqrt().item()),
            gnorm_preclip=float(g2_objective_acc.sqrt().item()),
            probe_valid=probe_valid,
        )

        staged = StagedLbfgs(
            p_probe=pp,
            p_momentum=self._p_mom,
            p_decay=self._p_decay,
            p_exec=self._p_exec,
            theta0=self._theta0,
            stats=stats,
            has_grad=has_grad,
        )
        self._staged = staged
        return staged

    @property
    def moment_on(self) -> str:
        return "gradient"

    # ---------------------------------------------------------- history update
    @torch.no_grad()
    def push_pair(
        self,
        s_list: Sequence[Tensor],
        y_list: Sequence[Tensor],
    ) -> bool:
        """Append ``(s, y)`` if ``y^T s > 0``; return whether accepted."""
        ys = _flat_dot(s_list, y_list)
        if not (math_isfinite(ys) and ys > 0.0):
            self.n_pairs_skipped += 1
            return False
        s_store = [t.detach().clone() for t in s_list]
        y_store = [t.detach().clone() for t in y_list]
        self._s_hist.append(s_store)
        self._y_hist.append(y_store)
        if len(self._s_hist) > self.history_size:
            self._s_hist.pop(0)
            self._y_hist.pop(0)
        self.n_pairs_accepted += 1
        return True

    @torch.no_grad()
    def displacement_from_theta0(self) -> list[Tensor]:
        assert self._theta0 is not None
        return [p.detach().clone().sub_(t0) for p, t0 in zip(self._params, self._theta0)]

    # --------------------------------------------------------------- apply/commit
    @torch.no_grad()
    def write_candidate(
        self,
        epsilon: float,
        direction: Sequence[Tensor],
        *,
        out: list[Tensor] | None = None,
        into_params: bool = False,
    ) -> list[Tensor]:
        self._ensure_buffers()
        assert self._theta0 is not None and self._buf_candidate is not None
        dest = self._params if into_params else (out if out is not None else self._buf_candidate)
        eps = float(epsilon)
        for i, p_dir in enumerate(direction):
            dest[i].copy_(self._theta0[i]).add_(p_dir, alpha=eps)
        return dest  # type: ignore[return-value]

    def execute_ordinary_step(
        self,
        epsilon: float,
        *,
        g_preclip: Sequence[Tensor | None] | None = None,
    ) -> StagedLbfgs:
        self.snapshot_theta0()
        st = self.stage_from_q(g_preclip=g_preclip, use_param_grad_as_q=True)
        self.write_candidate(epsilon, st.p_exec, into_params=True)
        self.clear_transaction()
        return st

    def commit(
        self,
        staged: StagedLbfgs | None = None,
        *,
        params_already_written: bool = True,
        epsilon: float | None = None,
        use_exec: bool = True,
    ) -> None:
        st = staged if staged is not None else self._staged
        if st is None:
            raise RuntimeError("no staged L-BFGS state to commit")
        with torch.no_grad():
            if not params_already_written:
                if epsilon is None:
                    raise ValueError("epsilon required when params_already_written=False")
                direction = st.p_exec if use_exec else st.p_probe
                self.write_candidate(float(epsilon), direction, into_params=True)
        self.clear_transaction()

    def discard(self, *, restore_params: bool = True) -> None:
        if restore_params:
            self.restore_theta0()
        self.clear_transaction()

    def set_group_lrs(self, epsilon: float) -> None:
        eps = float(epsilon)
        for gi, group in enumerate(self.opt.param_groups):
            group["lr"] = eps * float(self.group_ratios[gi])

    def reset_step_moments(self) -> None:
        self.reset_history()


@torch.no_grad()
def step_vector_breakdown_lbfgs(staged: StagedLbfgs, epsilon: float) -> dict[str, float]:
    eps = float(epsilon)
    u2 = fresh2 = dec2 = 0.0
    fresh_dot = dec_dot = 0.0
    for i, ok in enumerate(staged.has_grad):
        if not ok:
            continue
        pp = staged.p_probe[i].detach().to(dtype=torch.float64).reshape(-1)
        pd = staged.p_decay[i].detach().to(dtype=torch.float64).reshape(-1)
        fv = pp * eps
        dv = pd * eps
        fresh2 += float(torch.dot(fv, fv).item())
        dec2 += float(torch.dot(dv, dv).item())
        uv = fv + dv
        u2 += float(torch.dot(uv, uv).item())
        fresh_dot += float(torch.dot(fv, uv).item())
        dec_dot += float(torch.dot(dv, uv).item())
    u_tot = u2 ** 0.5
    fresh_n = fresh2 ** 0.5
    dec_n = dec2 ** 0.5
    part_sum = fresh_n + dec_n
    inv_parts = 1.0 / part_sum if part_sum > 0.0 else 0.0
    inv_u2 = 1.0 / u2 if u2 > 0.0 else 0.0
    inv_f = 1.0 / fresh_n if fresh_n > 0.0 else 0.0
    return {
        "ls_u_fresh_norm": fresh_n,
        "ls_u_mom_norm": 0.0,
        "ls_u_decay_norm": dec_n,
        "ls_u_total_norm": u_tot,
        "ls_u_fresh_frac": fresh_dot * inv_u2,
        "ls_u_mom_frac": 0.0,
        "ls_u_decay_frac": dec_dot * inv_u2,
        "ls_u_fresh_norm_frac": fresh_n * inv_parts,
        "ls_u_mom_norm_frac": 0.0,
        "ls_u_decay_norm_frac": dec_n * inv_parts,
        "ls_u_mom_over_fresh": 0.0,
    }
