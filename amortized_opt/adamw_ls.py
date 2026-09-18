"""AdamW staging adapter for SaLSa-style line search.

**gradient** mode (default): EMA ``m,v`` on clipped/transformed ``q``;
``θ ← θ0 + ε p_exec``.  Armijo slopes are always measured against the raw
scalar-objective gradient supplied separately from ``q``.

**step** mode: EMA ``m,v`` on executed steps
``u = ε p_probe + p_momentum + p_decay`` (carry not scaled by ε);
``θ ← θ0 + u``.  ``D`` uses ``v`` from past ``u``; ``p_momentum`` from ``m(u)``.

Probe / Armijo still use ``p_probe = -D q``, ``a = dot(q, D q)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW

__all__ = [
    "AdamWLineSearchAdapter",
    "StagedAdamW",
    "DirectionStats",
    "build_adamw_like_train_cost",
    "group_ratios_from_optimizer",
    "displacement_matches_adamw_step",
    "step_vector_breakdown",
    "MOMENT_ON_GRADIENT",
    "MOMENT_ON_STEP",
]


def group_ratios_from_optimizer(opt: AdamW, eta_ref: float) -> list[float]:
    if not (eta_ref > 0 and math_isfinite(eta_ref)):
        raise ValueError(f"eta_ref must be positive finite, got {eta_ref}")
    return [float(g["lr"]) / float(eta_ref) for g in opt.param_groups]


def math_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


MOMENT_ON_GRADIENT = "gradient"
MOMENT_ON_STEP = "step"


def _normalize_moment_on(moment_on: str) -> str:
    mode = str(moment_on).strip().lower()
    if mode not in (MOMENT_ON_GRADIENT, MOMENT_ON_STEP):
        raise ValueError(
            f"moment_on must be {MOMENT_ON_GRADIENT!r} or {MOMENT_ON_STEP!r}, got {moment_on!r}"
        )
    return mode


def build_adamw_like_train_cost(
    named_params: Iterable[tuple[str, Tensor]],
    *,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Mirror ``train_cost.build_optimizer`` grouping: decay vs emb/1-D no_decay."""
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if (p.dim() < 2 or "emb" in name) else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("no requires_grad parameters")
    return AdamW(groups, lr=lr, betas=betas, eps=eps)


@torch.no_grad()
def step_vector_breakdown(
    staged: StagedAdamW,
    epsilon: float,
    moment_on: str,
) -> dict[str, float]:
    """Norm ratios of the executed displacement ``u`` (fresh vs momentum vs decay).

    **gradient** mode: ``u = ε (p_current + p_history + p_decay)`` where
    ``p_current + p_history`` exactly decomposes the staged Adam ``m_next``.
    **step** mode: ``u = ε p_probe + p_momentum + p_decay`` (carry not scaled
    by ε). Fractions are projections onto ``u``; norm fractions are component
    magnitudes normalized by their sum.
    """
    eps = float(epsilon)
    mode = _normalize_moment_on(moment_on)
    u2 = fresh2 = mom2 = dec2 = 0.0
    fresh_dot = mom_dot = dec_dot = 0.0
    for i, ok in enumerate(staged.has_grad):
        if not ok:
            continue
        pm = staged.p_momentum[i].detach().to(dtype=torch.float64).reshape(-1)
        pd = staged.p_decay[i].detach().to(dtype=torch.float64).reshape(-1)
        if mode == MOMENT_ON_STEP:
            pp = staged.p_probe[i].detach().to(dtype=torch.float64).reshape(-1)
            fv = pp * eps
            mv = pm
            dv = pd
        else:
            pc = staged.p_current[i].detach().to(dtype=torch.float64).reshape(-1)
            ph = staged.p_history[i].detach().to(dtype=torch.float64).reshape(-1)
            fv = pc * eps
            mv = ph * eps
            dv = pd * eps
        fresh2 += float(torch.dot(fv, fv).item())
        mom2 += float(torch.dot(mv, mv).item())
        dec2 += float(torch.dot(dv, dv).item())
        uv = fv + mv + dv
        u2 += float(torch.dot(uv, uv).item())
        fresh_dot += float(torch.dot(fv, uv).item())
        mom_dot += float(torch.dot(mv, uv).item())
        dec_dot += float(torch.dot(dv, uv).item())
    u_tot = u2 ** 0.5
    fresh_n = fresh2 ** 0.5
    mom_n = mom2 ** 0.5
    dec_n = dec2 ** 0.5
    part_sum = fresh_n + mom_n + dec_n
    inv_parts = 1.0 / part_sum if part_sum > 0.0 else 0.0
    inv_u2 = 1.0 / u2 if u2 > 0.0 else 0.0
    inv_f = 1.0 / fresh_n if fresh_n > 0.0 else 0.0
    return {
        "ls_u_fresh_norm": fresh_n,
        "ls_u_mom_norm": mom_n,
        "ls_u_decay_norm": dec_n,
        "ls_u_total_norm": u_tot,
        # Projection onto executed ``u`` (sum to 1 when ``u = fresh + mom + decay``).
        "ls_u_fresh_frac": fresh_dot * inv_u2,
        "ls_u_mom_frac": mom_dot * inv_u2,
        "ls_u_decay_frac": dec_dot * inv_u2,
        # L1 magnitude shares (sum to 1; diagnostic when directions differ).
        "ls_u_fresh_norm_frac": fresh_n * inv_parts,
        "ls_u_mom_norm_frac": mom_n * inv_parts,
        "ls_u_decay_norm_frac": dec_n * inv_parts,
        "ls_u_mom_over_fresh": mom_n * inv_f,
    }


@dataclass
class DirectionStats:
    a: float
    P2: float
    s_probe: float
    s_exec: float
    s_momentum: float
    s_decay: float
    s_probe_q: float
    s_exec_q: float
    s_momentum_q: float
    s_decay_q: float
    gnorm_q: float
    gnorm_objective: float
    gnorm_preclip: float
    probe_valid: bool


@dataclass
class StagedAdamW:
    """Tentative AdamW moments + directions for one search transaction."""

    # Parallel to flattened param list in adapter._params
    m_next: list[Tensor]
    v_next: list[Tensor]
    step_next: list[Tensor]  # 0-dim long tensors, matching opt state
    D: list[Tensor]
    p_momentum: list[Tensor]
    p_current: list[Tensor]
    p_history: list[Tensor]
    p_decay: list[Tensor]
    p_exec: list[Tensor]
    p_probe: list[Tensor]
    theta0: list[Tensor]
    stats: DirectionStats
    group_ratios: list[float]
    eta_ref: float
    # Per-param metadata
    param_group_idx: list[int]
    has_grad: list[bool]


class AdamWLineSearchAdapter:
    """Stage / probe / exec / commit without mutating the live optimizer until commit."""

    def __init__(
        self,
        optimizer: AdamW,
        *,
        eta_ref: float | None = None,
        group_ratios: Sequence[float] | None = None,
        moment_on: str = MOMENT_ON_GRADIENT,
    ) -> None:
        if not isinstance(optimizer, AdamW):
            raise TypeError(
                f"AdamWLineSearchAdapter requires torch.optim.AdamW, got {type(optimizer)}"
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
        self.moment_on = _normalize_moment_on(moment_on)

        self._params: list[Tensor] = []
        self._param_group_idx: list[int] = []
        for gi, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                self._params.append(p)
                self._param_group_idx.append(gi)

        n = len(self._params)
        self._theta0: list[Tensor] | None = None
        self._buf_candidate: list[Tensor] | None = None
        self._staged: StagedAdamW | None = None
        # Lazily sized reusable direction buffers
        self._D: list[Tensor] | None = None
        self._p_probe: list[Tensor] | None = None
        self._p_mom: list[Tensor] | None = None
        self._p_current: list[Tensor] | None = None
        self._p_history: list[Tensor] | None = None
        self._p_decay: list[Tensor] | None = None
        self._p_exec: list[Tensor] | None = None
        self._theta0_ready: bool = False

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
        self._D = [torch.empty_like(p) for p in self._params]
        self._p_probe = [torch.empty_like(p) for p in self._params]
        self._p_mom = [torch.empty_like(p) for p in self._params]
        self._p_current = [torch.empty_like(p) for p in self._params]
        self._p_history = [torch.empty_like(p) for p in self._params]
        self._p_decay = [torch.empty_like(p) for p in self._params]
        self._p_exec = [torch.empty_like(p) for p in self._params]
        self._theta0_ready = False

    def snapshot_theta0(self) -> list[Tensor]:
        """Copy live parameters into reusable ``theta0`` buffers; return them."""
        self._ensure_buffers()
        assert self._theta0 is not None
        for dst, p in zip(self._theta0, self._params):
            dst.copy_(p.detach())
        self._theta0_ready = True
        return self._theta0

    def mark_theta0_snapshot(self) -> list[Tensor]:
        """Alias for ``snapshot_theta0`` (explicit transaction open)."""
        return self.snapshot_theta0()

    def restore_theta0(self) -> None:
        """Write ``theta0`` back into live parameters (discard probe mutations)."""
        if self._theta0 is None:
            return
        with torch.no_grad():
            for p, t0 in zip(self._params, self._theta0):
                p.copy_(t0)

    # -------------------------------------------------------------------- stage
    def _init_state_if_needed(self, p: Tensor) -> dict:
        state = self.opt.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    def _state_for_staging(self, p: Tensor) -> dict:
        """Read Adam state without creating live optimizer entries.

        A searched update is transactional: a later actual-update veto must
        leave a fresh optimizer byte-for-byte equivalent to its pre-search
        state.  Materialize zero state only in this ephemeral staging view;
        :meth:`commit` creates the live entry after acceptance.
        """
        state = self.opt.state.get(p)
        if state:
            return state
        return {
            "step": torch.tensor(0.0, dtype=torch.float32),
            "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.zeros_like(
                p, memory_format=torch.preserve_format
            ),
        }

    def stage_from_q(
        self,
        *,
        g_objective: Sequence[Tensor | None] | None = None,
        g_preclip: Sequence[Tensor | None] | None = None,
        use_param_grad_as_q: bool = True,
        q_list: Sequence[Tensor | None] | None = None,
    ) -> StagedAdamW:
        """Stage next AdamW moments from clipped ``q``; build ``D``, directions.

        ``q`` defaults to each parameter's live ``.grad`` and owns the staged
        optimizer direction. ``g_objective`` is the raw gradient of the replayed
        scalar objective and owns all Armijo/directional derivatives. The legacy
        ``g_preclip`` argument aliases ``g_objective`` when the latter is absent.
        Does **not** write optimizer state or parameters.
        """
        self._ensure_buffers()
        assert (
            self._theta0 is not None
            and self._D is not None
            and self._p_probe is not None
            and self._p_mom is not None
            and self._p_current is not None
            and self._p_history is not None
            and self._p_decay is not None
            and self._p_exec is not None
        )
        # Contract: snapshot_theta0() before staging. Auto-snapshot if omitted.
        if not self._theta0_ready:
            for dst, p in zip(self._theta0, self._params):
                dst.copy_(p.detach())
            self._theta0_ready = True

        m_next: list[Tensor] = []
        v_next: list[Tensor] = []
        step_next: list[Tensor] = []
        has_grad: list[bool] = []

        s_probe_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        P2_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_exec_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_mom_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_dec_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_probe_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_exec_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_mom_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        s_dec_q_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        q2_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)
        g2_objective_acc = torch.zeros((), dtype=torch.float64, device=self._params[0].device)

        for i, p in enumerate(self._params):
            gi = self._param_group_idx[i]
            group = self.opt.param_groups[gi]
            beta1, beta2 = group["betas"]
            adam_eps = float(group["eps"])
            wd = float(group["weight_decay"])
            r = float(self.group_ratios[gi])
            maximize = bool(group.get("maximize", False))

            if q_list is not None:
                q = q_list[i]
            elif use_param_grad_as_q:
                q = p.grad
            else:
                q = None

            D = self._D[i]
            pm = self._p_mom[i]
            pc = self._p_current[i]
            ph = self._p_history[i]
            pd = self._p_decay[i]
            pe = self._p_exec[i]
            pp = self._p_probe[i]
            t0 = self._theta0[i]

            if q is None:
                has_grad.append(False)
                D.zero_()
                pm.zero_()
                pc.zero_()
                ph.zero_()
                pd.zero_()
                pe.zero_()
                pp.zero_()
                # Still keep placeholders matching state shapes for commit hygiene.
                state = self._state_for_staging(p)
                m_next.append(state["exp_avg"].clone())
                v_next.append(state["exp_avg_sq"].clone())
                step_next.append(state["step"].clone())
                continue

            has_grad.append(True)
            q_raw = q.detach()
            grad = q_raw
            if maximize:
                grad = -grad
            g_obj = (
                g_objective[i]
                if g_objective is not None
                else (
                    g_preclip[i]
                    if g_preclip is not None
                    else q_raw
                )
            )
            if g_obj is None:
                raise ValueError(
                    "g_objective is missing for a parameter with optimizer gradient"
                )
            g_obj = g_obj.detach()

            state = self._state_for_staging(p)
            m_old = state["exp_avg"]
            v_old = state["exp_avg_sq"]
            step_t = state["step"]

            # Match PyTorch: increment step, then bias-correct with new step.
            step_new = step_t.clone()
            step_new += 1

            step_val = int(step_new.item()) if step_new.numel() == 1 else int(step_new)
            bc1 = 1.0 - (beta1 ** step_val)
            bc2 = 1.0 - (beta2 ** step_val)
            bc2_sqrt = bc2 ** 0.5

            if self.moment_on == MOMENT_ON_STEP:
                m_n = m_old.clone()
                v_n = v_old.clone()
                denom = v_old.sqrt() / bc2_sqrt + adam_eps
                D.copy_(r / denom)
                pm.copy_((-r / bc1) * m_old / denom)
                if wd != 0.0:
                    pd.copy_((-r * wd) * t0)
                else:
                    pd.zero_()
                pp.copy_(-D * grad)
                pc.copy_(pp)
                ph.copy_(pm)
                pe.copy_(pp + pm + pd)
            else:
                m_n = m_old.clone()
                v_n = v_old.clone()
                m_n.lerp_(grad, 1 - beta1)
                v_n.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = v_n.sqrt() / bc2_sqrt + adam_eps
                D.copy_(r / denom)
                pc.copy_((-r * (1.0 - beta1) / bc1) * grad / denom)
                ph.copy_((-r * beta1 / bc1) * m_old / denom)
                pm.copy_(pc).add_(ph)
                if wd != 0.0:
                    pd.copy_((-r * wd) * t0)
                else:
                    pd.zero_()
                pe.copy_(pm + pd)
                pp.copy_(-D * grad)

            m_next.append(m_n)
            v_next.append(v_n)
            step_next.append(step_new)

            # Reductions in float64 for stability.
            q64 = grad.detach().to(dtype=torch.float64).reshape(-1)
            go64 = g_obj.to(dtype=torch.float64).reshape(-1)
            pp64 = pp.detach().to(dtype=torch.float64).reshape(-1)
            pe64 = pe.detach().to(dtype=torch.float64).reshape(-1)
            pm64 = pm.detach().to(dtype=torch.float64).reshape(-1)
            pd64 = pd.detach().to(dtype=torch.float64).reshape(-1)
            s_probe_acc = s_probe_acc + torch.dot(go64, pp64)
            P2_acc = P2_acc + torch.dot(pp64, pp64)
            s_exec_acc = s_exec_acc + torch.dot(go64, pe64)
            s_mom_acc = s_mom_acc + torch.dot(go64, pm64)
            s_dec_acc = s_dec_acc + torch.dot(go64, pd64)
            s_probe_q_acc = s_probe_q_acc + torch.dot(q64, pp64)
            s_exec_q_acc = s_exec_q_acc + torch.dot(q64, pe64)
            s_mom_q_acc = s_mom_q_acc + torch.dot(q64, pm64)
            s_dec_q_acc = s_dec_q_acc + torch.dot(q64, pd64)
            q2_acc = q2_acc + torch.dot(q64, q64)
            g2_objective_acc = g2_objective_acc + torch.dot(go64, go64)

        s_probe = float(s_probe_acc.item())
        a = -s_probe
        P2 = float(P2_acc.item())
        probe_valid = math_isfinite(a) and math_isfinite(P2) and a > 0.0 and P2 > 0.0
        stats = DirectionStats(
            a=a,
            P2=P2,
            s_probe=s_probe,
            s_exec=float(s_exec_acc.item()),
            s_momentum=float(s_mom_acc.item()),
            s_decay=float(s_dec_acc.item()),
            s_probe_q=float(s_probe_q_acc.item()),
            s_exec_q=float(s_exec_q_acc.item()),
            s_momentum_q=float(s_mom_q_acc.item()),
            s_decay_q=float(s_dec_q_acc.item()),
            gnorm_q=float(q2_acc.sqrt().item()),
            gnorm_objective=float(g2_objective_acc.sqrt().item()),
            gnorm_preclip=float(g2_objective_acc.sqrt().item()),
            probe_valid=probe_valid,
        )
        staged = StagedAdamW(
            m_next=m_next,
            v_next=v_next,
            step_next=step_next,
            D=self._D,
            p_momentum=self._p_mom,
            p_current=self._p_current,
            p_history=self._p_history,
            p_decay=self._p_decay,
            p_exec=self._p_exec,
            p_probe=self._p_probe,
            theta0=self._theta0,
            stats=stats,
            group_ratios=list(self.group_ratios),
            eta_ref=self.eta_ref,
            param_group_idx=list(self._param_group_idx),
            has_grad=has_grad,
        )
        self._staged = staged
        return staged

    def clear_transaction(self) -> None:
        self._staged = None
        self._theta0_ready = False

    # --------------------------------------------------------------- apply / commit
    @torch.no_grad()
    def write_candidate(
        self,
        epsilon: float,
        direction: Sequence[Tensor],
        *,
        out: list[Tensor] | None = None,
        into_params: bool = False,
    ) -> list[Tensor]:
        """``out = theta0 + epsilon * direction`` into reusable buffers or live params."""
        self._ensure_buffers()
        assert self._theta0 is not None and self._buf_candidate is not None
        dest = self._params if into_params else (out if out is not None else self._buf_candidate)
        eps = float(epsilon)
        for i, p_dir in enumerate(direction):
            dest[i].copy_(self._theta0[i]).add_(p_dir, alpha=eps)
        return dest  # type: ignore[return-value]

    @torch.no_grad()
    def apply_epsilon_to_params(self, epsilon: float, direction: Sequence[Tensor]) -> None:
        self.write_candidate(epsilon, direction, into_params=True)

    @torch.no_grad()
    def write_executed_step(
        self,
        epsilon: float,
        staged: StagedAdamW,
        *,
        into_params: bool = False,
        out: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """``θ0 + ε p_probe + p_mom + p_decay`` (step mode executed vector ``u``)."""
        if self.moment_on != MOMENT_ON_STEP:
            raise RuntimeError("write_executed_step requires moment_on='step'")
        self._ensure_buffers()
        assert staged.theta0 is not None
        dest = (
            self._params if into_params
            else (out if out is not None else self._buf_candidate)
        )
        eps = float(epsilon)
        for i in range(len(self._params)):
            if not staged.has_grad[i]:
                continue
            dest[i].copy_(staged.theta0[i])
            dest[i].add_(staged.p_probe[i], alpha=eps)
            dest[i].add_(staged.p_momentum[i])
            dest[i].add_(staged.p_decay[i])
        return dest  # type: ignore[return-value]

    @torch.no_grad()
    def _commit_step_moments(self, st: StagedAdamW, epsilon: float) -> None:
        eps = float(epsilon)
        for i, p in enumerate(self._params):
            if not st.has_grad[i]:
                continue
            gi = self._param_group_idx[i]
            group = self.opt.param_groups[gi]
            beta1, beta2 = group["betas"]
            state = self._init_state_if_needed(p)
            u_i = st.p_probe[i].mul(eps).add(st.p_momentum[i]).add(st.p_decay[i])
            m_old = state["exp_avg"]
            v_old = state["exp_avg_sq"]
            m_old.lerp_(u_i, 1.0 - float(beta1))
            v_old.mul_(float(beta2)).addcmul_(u_i, u_i, value=1.0 - float(beta2))
            if not torch.is_tensor(state["step"]):
                state["step"] = st.step_next[i].clone()
            else:
                state["step"].copy_(st.step_next[i])

    def execute_ordinary_step(
        self,
        epsilon: float,
        *,
        g_preclip: Sequence[Tensor | None] | None = None,
    ) -> StagedAdamW | None:
        """Non-search update: stage, apply ``u`` or native ``opt.step()``, commit."""
        if self.moment_on == MOMENT_ON_STEP:
            self.snapshot_theta0()
            st = self.stage_from_q(g_preclip=g_preclip, use_param_grad_as_q=True)
            self.write_executed_step(epsilon, st, into_params=True)
            self._commit_step_moments(st, epsilon)
            self.clear_transaction()
            return st
        self.set_group_lrs(epsilon)
        self.opt.step()
        return None

    def reset_step_moments(self) -> None:
        """Zero ``exp_avg`` / ``exp_avg_sq`` (step-mode objective discontinuity)."""
        with torch.no_grad():
            for p in self._params:
                st = self.opt.state.get(p)
                if not st:
                    continue
                if "exp_avg" in st:
                    st["exp_avg"].zero_()
                if "exp_avg_sq" in st:
                    st["exp_avg_sq"].zero_()

    def commit(self, staged: StagedAdamW | None = None, *, params_already_written: bool = False,
               epsilon: float | None = None, use_exec: bool = True) -> None:
        """Commit staged moments and (optionally) write the executed step."""
        st = staged if staged is not None else self._staged
        if st is None:
            raise RuntimeError("no staged AdamW state to commit")
        with torch.no_grad():
            if self.moment_on == MOMENT_ON_STEP:
                if epsilon is None:
                    raise ValueError("epsilon required for step-mode commit")
                if not params_already_written:
                    self.write_executed_step(float(epsilon), st, into_params=True)
                self._commit_step_moments(st, float(epsilon))
                self.clear_transaction()
                return
            if not params_already_written:
                if epsilon is None:
                    raise ValueError("epsilon required when params_already_written=False")
                direction = st.p_exec if use_exec else st.p_probe
                self.apply_epsilon_to_params(float(epsilon), direction)
            for i, p in enumerate(self._params):
                if not st.has_grad[i]:
                    continue
                state = self._init_state_if_needed(p)
                state["exp_avg"].copy_(st.m_next[i])
                state["exp_avg_sq"].copy_(st.v_next[i])
                if not torch.is_tensor(state["step"]):
                    state["step"] = st.step_next[i].clone()
                else:
                    state["step"].copy_(st.step_next[i])
        self.clear_transaction()

    def discard(self, *, restore_params: bool = True) -> None:
        """Drop staged moments; optionally restore ``theta0`` into live params."""
        if restore_params:
            self.restore_theta0()
        self.clear_transaction()

    def set_group_lrs(self, epsilon: float) -> None:
        """Ordinary reuse path: ``lr_group = epsilon * fixed_group_ratio``."""
        eps = float(epsilon)
        for gi, group in enumerate(self.opt.param_groups):
            group["lr"] = eps * float(self.group_ratios[gi])


def displacement_matches_adamw_step(
    adapter: AdamWLineSearchAdapter,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> tuple[bool, dict]:
    """Debug helper: ``ε=η_ref`` exec displacement vs a fresh ``opt.step()``.

    Mutates clones only; restores the live optimizer/params afterward.
    """
    opt = adapter.opt
    params = adapter._params
    # Snapshot live
    theta_live = [p.detach().clone() for p in params]
    state_live = {
        id(p): {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
        for p in params
        if p in opt.state and len(opt.state[p]) > 0
    }
    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

    eta = adapter.eta_ref
    adapter.mark_theta0_snapshot()
    # Restore grads after any prior ops
    for p, g in zip(params, grads):
        p.grad = None if g is None else g.clone()
    staged = adapter.stage_from_q()
    adapter.write_candidate(eta, staged.p_exec, into_params=True)
    theta_ls = [p.detach().clone() for p in params]
    # Restore for native step
    with torch.no_grad():
        for p, t in zip(params, theta_live):
            p.copy_(t)
    for p, g in zip(params, grads):
        p.grad = None if g is None else g.clone()
    # Restore optimizer state
    for p in params:
        if id(p) in state_live:
            opt.state[p] = {
                k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_live[id(p)].items()
            }
        elif p in opt.state:
            opt.state[p].clear()
    # Ensure group LRs = eta_ref * ratio for fair comparison
    adapter.set_group_lrs(eta)
    opt.step()
    theta_opt = [p.detach().clone() for p in params]

    ok = True
    max_abs = 0.0
    for a, b in zip(theta_ls, theta_opt):
        diff = (a - b).abs().max().item()
        max_abs = max(max_abs, diff)
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            ok = False

    # Restore live completely
    with torch.no_grad():
        for p, t in zip(params, theta_live):
            p.copy_(t)
    for p, g in zip(params, grads):
        p.grad = None if g is None else g.clone()
    for p in params:
        if id(p) in state_live:
            opt.state[p] = {
                k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_live[id(p)].items()
            }
    adapter.clear_transaction()
    return ok, {"max_abs_diff": max_abs, "eta_ref": eta}
