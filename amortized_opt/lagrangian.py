"""Augmented-Lagrangian multiplier state for the constraint groups.

The plan: *"For our cost gradients, we can use an augmented lagrangian to
balance our limit/boundary gradients with the optimization gradients."*

Why this rather than fixed weights
----------------------------------

The five constraint groups (torque, velocity, joint box, collision, terminal)
live in incomparable units -- N*m, rad/s, rad, metres of penetration, metres of
miss -- and their *typical* magnitudes move by orders of magnitude across the
curriculum.  A fixed weight either lets a group be ignored early or lets it
dominate the objective once it is satisfied.  A multiplier that integrates the
observed violation does neither: it grows while the group is violated on average
and decays back once it is not, so at convergence the constraint terms cost
almost nothing and the smoothness / posture objective is what shapes the
solution.

Update rule
-----------

Per group ``g``, with ``V_g`` the BATCH MEAN of that group's per-sample raw
violation (``CostTerms.violations[g]``, which is zero when feasible -- unlike
the softplus penalty, which never is; how each per-sample violation is reduced
over its own elements is the group's business, and ``torque`` and
``torque_peak`` deliberately differ)::

    lambda_g <- clip(lambda_g + rho_g * (V_g - target_g), lambda_min, lambda_max)

and the cost is evaluated with ``lambda_g`` multiplying that group's penalty.
``target_g`` is the tolerated violation (default 0).  ``ema`` optionally smooths
``V_g`` before the update, which matters when the minibatch is small enough that
a single pathological sample would spike a multiplier.

Why ``torque_peak`` is its own group
------------------------------------

``FusedCost`` prices the torque hinge **twice** -- a MEAN over the ~850
``(knot, joint)`` elements and a smooth MAX over the same elements (see
:meth:`~amortized_opt.cost.FusedCost._torque`) -- because the acceptance gate is
a zero-tolerance max while the loss-shaping half has to be an average.  Those
two halves are driven by violations of completely different SHAPE: a case that
is over on one element of 850 by 3.7 N*m has a mean raw violation of 4e-3 N*m
and a max raw violation of 3.7 N*m, a factor of ~850 apart.  A single
multiplier calibrated on the mean therefore has no idea the peak exists, and
raising the peak's price also amplifies it -- which is exactly what
destabilized the two continuations recorded in the README's "What is left of
the torque gap".  ``torque_peak`` is consequently a group in its own right,
driven by ``violations["torque_peak"]`` (the per-sample MAX excess over the
HARD effort limits) with its own, ~100x smaller ``rho``, so the peak price can
rise while the mean price relaxes.

This is the practical "multiplier method" form the plan describes, not the exact
Powell-Hestenes-Rockafellar augmented Lagrangian: the penalty being scaled is the
softplus hinge (a smooth, conservatively-shifted surrogate) rather than
``max(0, g)^2``, because the softplus is what supplies gradient on the *safe*
side of a limit, which is the whole reason the plan asks for softplus hinges.
An optional quadratic ``rho/2 * V^2`` term can be added on top via
:meth:`AugmentedLagrangian.quadratic_penalty`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = ["ALConfig", "AugmentedLagrangian", "CONSTRAINT_GROUPS"]

#: The groups :class:`~amortized_opt.cost.FusedCost` reports violations for.
#:
#: ORDER IS STABLE and APPEND-ONLY: ``torque_peak`` is last so that any state
#: (checkpointed ``lam`` dicts, saved ``groups`` lists, per-group ``rho``
#: dicts) written before it existed still lines up.  It is a separate group
#: from ``torque`` on purpose -- see the module docstring.
CONSTRAINT_GROUPS: tuple[str, ...] = (
    "torque",
    "velocity",
    "joint_box",
    "collision",
    "terminal",
    "torque_peak",
)


@dataclass
class ALConfig:
    """Per-group multiplier dynamics.

    Scalars apply to every group; a dict overrides individual groups.  The
    defaults are deliberately slow (``rho`` small relative to the violation
    scale) because a multiplier that outruns the optimizer oscillates.
    """

    groups: tuple[str, ...] = CONSTRAINT_GROUPS
    lam_init: float | dict[str, float] = 1.0
    #: Step size on the multiplier, per unit of mean violation.  Violations are
    #: in the group's own units (N*m, rad/s, rad, m, m), so these are per-group
    #: by default -- a 1 mm collision and a 1 N*m torque overshoot should not
    #: move their multipliers by the same amount.
    rho: float | dict[str, float] = field(
        default_factory=lambda: {
            "torque": 0.5,      # N*m, per-ELEMENT mean excess
            "velocity": 0.5,    # rad/s
            "joint_box": 20.0,  # rad
            "collision": 200.0,  # m
            "terminal": 100.0,  # m
            # N*m of per-sample MAX excess.  Same quantity as ``torque`` but a
            # max instead of a mean over ~850 elements, so the same physical
            # infeasibility reads ~100x (sparse case: ~n_valid) larger here.
            # 1/100 of ``torque``'s rho keeps the two multipliers on the same
            # timescale.
            "torque_peak": 0.005,
        }
    )
    #: Tolerated violation before the multiplier grows, in the same units the
    #: group reports (a mean excess for ``torque``, a MAX excess for
    #: ``torque_peak`` -- 0 there is the acceptance gate itself).
    target: float | dict[str, float] = 0.0
    lam_min: float = 0.0
    lam_max: float = 1.0e4
    #: EMA coefficient on the observed violation (0 = use the batch value).
    ema: float = 0.0
    #: Multiplicative decay applied every update; < 1 lets a satisfied group's
    #: multiplier relax even when the violation is exactly zero.
    decay: float = 1.0

    def _per_group(self, value, name: str) -> dict[str, float]:
        if isinstance(value, dict):
            missing = [g for g in self.groups if g not in value]
            if missing:
                raise ValueError(f"{name} missing groups {missing}")
            return {g: float(value[g]) for g in self.groups}
        return {g: float(value) for g in self.groups}


class AugmentedLagrangian:
    """Stateful multipliers, one per constraint group.

    ::

        al = AugmentedLagrangian()
        out = cost(q0, wp, v0, ctx, weights=al.weights())
        loss = out.total
        loss.backward()
        al.update(out.violations)          # after the optimizer step
    """

    def __init__(self, cfg: ALConfig | None = None):
        self.cfg = cfg or ALConfig()
        c = self.cfg
        self.groups = tuple(c.groups)
        self.rho = c._per_group(c.rho, "rho")
        self.target = c._per_group(c.target, "target")
        init = c._per_group(c.lam_init, "lam_init")
        self.lam: dict[str, float] = {g: float(init[g]) for g in self.groups}
        self.avg_violation: dict[str, float] = {g: 0.0 for g in self.groups}
        self.steps: int = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------- weights --
    def weights(self) -> dict[str, float]:
        """Per-group multipliers, ready for ``FusedCost.forward(weights=...)``.

        Groups the cost knows about but this object does not (``smooth``,
        ``posture``) are simply absent, and default to 1.0 there.
        """
        return dict(self.lam)

    # -------------------------------------------------------------- update --
    def update(
        self, violations: dict[str, torch.Tensor | float], *, record: bool = True
    ) -> dict[str, float]:
        """One multiplier step from a batch's violations.  Returns the new lambdas.

        ``violations`` maps group -> per-sample violation ``(B,)`` (or a scalar);
        the batch mean is what drives the update.  Groups absent from the dict
        are left untouched, so a curriculum stage can simply not report a group.
        """
        beta = float(self.cfg.ema)
        for g in self.groups:
            v = violations.get(g)
            if v is None:
                continue
            val = float(v.detach().mean().item()) if torch.is_tensor(v) else float(v)
            if not (val == val):  # NaN guard: never let a bad batch poison lambda
                continue
            prev = self.avg_violation[g]
            smoothed = beta * prev + (1.0 - beta) * val if beta > 0.0 else val
            self.avg_violation[g] = smoothed
            lam = self.lam[g] * float(self.cfg.decay)
            lam = lam + self.rho[g] * (smoothed - self.target[g])
            self.lam[g] = float(min(max(lam, self.cfg.lam_min), self.cfg.lam_max))
        self.steps += 1
        if record:
            self.history.append({"step": float(self.steps), **self.lam})
        return dict(self.lam)

    def quadratic_penalty(
        self, violations: dict[str, torch.Tensor]
    ) -> torch.Tensor | float:
        """Optional ``sum_g rho_g / 2 * V_g^2`` term (per sample).

        Not added automatically -- ``violations`` is detached in
        :class:`~amortized_opt.cost.CostTerms`, so this is only useful when a
        caller recomputes a differentiable violation itself.
        """
        out = 0.0
        for g in self.groups:
            v = violations.get(g)
            if v is None:
                continue
            out = out + 0.5 * self.rho[g] * v * v
        return out

    # --------------------------------------------------------------- state --
    def state_dict(self) -> dict:
        return {
            "lam": dict(self.lam),
            "avg_violation": dict(self.avg_violation),
            "steps": int(self.steps),
            "groups": list(self.groups),
            "rho": dict(self.rho),
            "target": dict(self.target),
        }

    def load_state_dict(self, state: dict) -> "AugmentedLagrangian":
        self.lam.update({k: float(v) for k, v in state.get("lam", {}).items()})
        self.avg_violation.update(
            {k: float(v) for k, v in state.get("avg_violation", {}).items()}
        )
        self.steps = int(state.get("steps", 0))
        if "rho" in state:
            self.rho.update({k: float(v) for k, v in state["rho"].items()})
        if "target" in state:
            self.target.update({k: float(v) for k, v in state["target"].items()})
        return self

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), indent=1))
        return p

    @classmethod
    def load(cls, path: str | Path, cfg: ALConfig | None = None) -> "AugmentedLagrangian":
        state = json.loads(Path(path).read_text())
        obj = cls(cfg or ALConfig(groups=tuple(state.get("groups", CONSTRAINT_GROUPS))))
        return obj.load_state_dict(state)

    def __repr__(self) -> str:
        body = ", ".join(f"{g}={self.lam[g]:.4g}" for g in self.groups)
        return f"AugmentedLagrangian(step={self.steps}, {body})"
