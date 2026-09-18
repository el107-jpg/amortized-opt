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

What is actually implemented
----------------------------

Powell-Hestenes-Rockafellar AL has three pieces:

1. an integral dual ``λ`` (I-control) on the raw ReLU residual,
2. a quadratic penalty ``(pen / 2) [g]₊²`` on the SAME residual, with ``pen``
   adapted à la Bertsekas: grow when the residual does not improve for
   ``pen_patience`` consecutive updates *and* ``λ`` is below ``lam_max``,
   shrink when it is inside ``target``.  An exactly-zero ``pen`` is disabled
   and is not raised to ``pen_min`` by shrinking.  Growing while the dual is
   clipped is what exploded the 32k continuation (``1.5^{20}`` in 2k steps).
3. a PID *used* multiplier ``λ_used = clip(λ + Kp e + Kd Δe)`` so a feasible
   minibatch cheapens the hard ReLU wall immediately even if the integrator is
   still high.

``rho`` in this module is the dual step (Ki), as it always was -- it is a
well-tuned per-unit table and is NOT the quadratic coefficient.  Reusing it as
both Ki and the PHR ``ρ`` would make ``λ += ρ e`` explode the moment the
quadratic wall had grown.  The quadratic coefficient is ``pen``.

The hinge the network sees as a *warning* is still the softplus (gradient
on the safe side of a limit), priced at a fixed ``CostConfig.w_*`` and
never by λ.  The hard wall is standard PHR on the raw ReLU excess, which
*is* zero when feasible: ``λ [g]₊ + (pen / 2) [g]₊²``.

Default :class:`ALConfig` keeps the historical I-only update (``kp = kd = 0``,
``pen = 0``, ``pen_grow = 1``) so existing tests stay bit-identical.
:func:`~amortized_opt.train_cost.phase2_al_config` turns the rest on.

Update rule
-----------

Per group ``g``, with ``V_g`` the BATCH MEAN of that group's per-sample raw
violation (``CostTerms.violations[g]``, which is zero when feasible)::

    e = ema(V_g) - target_g
    leak = feasible_decay  if e < 0  else decay
    λ ← clip(leak · λ + rho_g · e, λmin, λmax)          # I, always
    pen ← grow if e > 0 and V did not drop by improve_tol
             for pen_patience consecutive updates AND λ is below lam_max
        ← shrink if e ≤ 0 and pen > 0
        ← freeze if λ is already at lam_max
             (the dual is the price; stacking an unbounded quadratic on a
             clipped multiplier is what 1.5^20'd the 32k continuation)
    λ_used = clip(λ + Kp · e + Kd · (e - e_prev), λmin, λmax)

``Kd Δe`` is captured at this update and *held* on ``λ_used`` until the next
one.  The training loop freezes the EMA for ``al_every`` cost steps, so a D
term recomputed against a just-written ``e_prev`` would be identically zero
on every step that actually prices the hard wall.

The cost is evaluated with ``λ_used`` multiplying the ReLU residual
(``live`` / ``violations``) and ``pen / 2 · V²`` added on the same live
residual.  The softplus warning stays at ``CostConfig.w_*``.  The primal is a
neural net, so the λ-update is NEVER skipped when ``pen`` grows -- Bertsekas
skip-λ assumes an inner minimizer that this training loop does not have.

Why ``waist_pr_band`` is its own group
--------------------------------------

Decoder waist pitch/roll must stay inside a deadband of the encoder set-point
(``StrikeContext.waist_pr_ref``).  The residual is a **peak** ReLU excess in
rad, ``relu(|q_pr - stopgrad(sp)| - band)``, same shape reason as
``torque_peak`` / ``collision_peak``: a mean over the sub-sample × 2 lean rows
lets a 2 rad dump hide.  ``stopgrad(sp)`` is load-bearing -- a two-way
gradient is how yaw SP co-drifted.  This is **not**
``w_waist_pr_sp_tether`` (encoder residual to upright); tether/hold keep
``sp`` small, this group only moves the decoder.

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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "ALConfig",
    "AugmentedLagrangian",
    "CONSTRAINT_GROUPS",
    "TERM_FOR_GROUP",
    "ACTUATOR_SEGMENTS",
    "ACTUATOR_SEGMENT_NAMES",
    "LIMIT_FAMILIES",
    "LIMIT_GROUP_KEYS",
    "limit_group_name",
]

#: Actuator classes the three limit families (torque / velocity / joint_box)
#: are split into.  Arms are split by hand so a hot left wrist cannot raise the
#: price on the right shoulder.  Waist is one class (not a hand).  Every name
#: is a G1 upper-body joint; the 17 joints partition these seven classes.
ACTUATOR_SEGMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("waist", (
        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    )),
    ("left_shoulder", (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
    )),
    ("right_shoulder", (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
    )),
    ("left_elbow", ("left_elbow_joint",)),
    ("right_elbow", ("right_elbow_joint",)),
    ("left_wrist", (
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    )),
    ("right_wrist", (
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    )),
)
ACTUATOR_SEGMENT_NAMES: tuple[str, ...] = tuple(n for n, _ in ACTUATOR_SEGMENTS)
LIMIT_FAMILIES: tuple[str, ...] = ("torque", "velocity", "joint_box")


def limit_group_name(family: str, segment: str) -> str:
    return f"{family}_{segment}"


LIMIT_GROUP_KEYS: tuple[str, ...] = tuple(
    limit_group_name(fam, seg)
    for fam in LIMIT_FAMILIES
    for seg in ACTUATOR_SEGMENT_NAMES
)


def _spread_limit_defaults(
    *,
    torque: float,
    velocity: float,
    joint_box: float,
    collision: float,
    terminal: float,
    torque_peak: float,
    velocity_peak: float,
    collision_peak: float,
    joint_box_peak: float,
    waist_pr_band: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for seg in ACTUATOR_SEGMENT_NAMES:
        out[limit_group_name("torque", seg)] = float(torque)
        out[limit_group_name("velocity", seg)] = float(velocity)
        out[limit_group_name("joint_box", seg)] = float(joint_box)
    out["collision"] = float(collision)
    out["terminal"] = float(terminal)
    out["torque_peak"] = float(torque_peak)
    out["velocity_peak"] = float(velocity_peak)
    out["collision_peak"] = float(collision_peak)
    out["joint_box_peak"] = float(joint_box_peak)
    out["waist_pr_band"] = float(waist_pr_band)
    return out


#: The groups :class:`~amortized_opt.cost.FusedCost` reports AL violations for.
#:
#: Limit families are split by :data:`ACTUATOR_SEGMENTS`.  Peak groups
#: (``torque_peak``, ``velocity_peak``, ``collision_peak``, ``joint_box_peak``)
#: stay GLOBAL (the acceptance gate is a max).  ``waist_pr_band`` /
#: ``velocity_peak`` append last so older checkpoint group lists still migrate.
CONSTRAINT_GROUPS: tuple[str, ...] = LIMIT_GROUP_KEYS + (
    "collision",
    "terminal",
    "torque_peak",
    "collision_peak",
    "joint_box_peak",
    "waist_pr_band",
    "velocity_peak",
)

#: Cost term that receives the PHR quadratic for each AL group.
TERM_FOR_GROUP: dict[str, str] = {
    **{limit_group_name(fam, seg): fam
       for fam in LIMIT_FAMILIES for seg in ACTUATOR_SEGMENT_NAMES},
    "torque_peak": "torque",
    "velocity_peak": "velocity",
    "collision": "collision",
    "collision_peak": "collision",
    "joint_box_peak": "joint_box",
    "terminal": "terminal",
    "waist_pr_band": "waist_pr_band",
}


assert set(TERM_FOR_GROUP) == set(CONSTRAINT_GROUPS)


@dataclass
class ALConfig:
    """Per-group multiplier dynamics.

    Scalars apply to every group; a dict overrides individual groups.  The
    defaults are deliberately the historical I-only update so
    :class:`AugmentedLagrangian` with no arguments is bit-identical to the
    pre-PHR / pre-PID object.
    """

    groups: tuple[str, ...] = CONSTRAINT_GROUPS
    lam_init: float | dict[str, float] = 1.0
    #: Dual step (Ki) per unit of mean raw violation, in the group's own units.
    #: This is NOT the PHR quadratic coefficient -- see ``pen``.
    rho: float | dict[str, float] = field(
        default_factory=lambda: _spread_limit_defaults(
            torque=0.5, velocity=0.5, joint_box=20.0,
            collision=200.0, terminal=100.0,
            torque_peak=0.005, velocity_peak=0.005,
            collision_peak=20.0,
            joint_box_peak=0.3,
            waist_pr_band=1.0,
        )
    )
    #: Tolerated violation before the integrator grows, in the same units the
    #: group reports.  Peak gates that are zero-tolerance keep 0; mean groups
    #: whose residual never quite hits zero need slack or ``λ`` integrates to
    #: ``lam_max`` (measured: 0.012 rad of waist-box owned the objective).
    target: float | dict[str, float] = 0.0
    lam_min: float | dict[str, float] = 0.0
    lam_max: float | dict[str, float] = 1.0e4
    #: EMA coefficient on the observed violation (0 = use the batch value).
    ema: float = 0.0
    #: Multiplicative leak on ``λ`` while the group is still above target.
    decay: float = 1.0
    #: Faster leak applied when ``e < 0`` so a dual pinned at ``lam_max`` can
    #: actually come down.  1.0 = same as ``decay`` (historical).
    feasible_decay: float = 1.0
    #: Proportional / derivative gains on the *used* multiplier.  0 = I-only.
    kp: float | dict[str, float] = 0.0
    kd: float | dict[str, float] = 0.0
    #: PHR quadratic coefficient ``(pen / 2) V²``.  Exactly 0 disables the
    #: quadratic (and :class:`~amortized_opt.cost.FusedCost` will not even add
    #: it); ``pen_min`` is only a floor for an already-positive penalty.
    #: Adapted in :meth:`AugmentedLagrangian.update` when either factor is active.
    pen: float | dict[str, float] = 0.0
    pen_min: float | dict[str, float] = 0.0
    pen_max: float | dict[str, float] = 1.0e12
    #: Bertsekas growth factor.  1.0 disables growth; with ``pen_shrink > 1``
    #: this deliberately gives decay-only adaptation rather than a full freeze.
    pen_grow: float = 1.0
    #: Feasible-penalty divisor.  1.0 disables shrinking.  Shrinking never
    #: enables an exactly-zero penalty, even when ``pen_min`` is positive.
    pen_shrink: float = 1.0
    #: Consecutive non-improving updates required before ``pen`` grows.
    #: 1 = grow on the first stale look (historical); the phase-2 config uses
    #: 10 so a noisy 100-step EMA cannot compound every ``al_every``.
    pen_patience: int = 1
    #: ``V`` "improved" if ``V <= improve_tol * V_prev``.  Only consulted when
    #: ``pen_grow != 1``.
    improve_tol: float = 0.9

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
        out = cost(q0, wp, v0, ctx, weights=al.cost_weights())
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
        self.lam_min = c._per_group(c.lam_min, "lam_min")
        self.lam_max = c._per_group(c.lam_max, "lam_max")
        self.kp = c._per_group(c.kp, "kp")
        self.kd = c._per_group(c.kd, "kd")
        self.pen = c._per_group(c.pen, "pen")
        self.pen_min = c._per_group(c.pen_min, "pen_min")
        self.pen_max = c._per_group(c.pen_max, "pen_max")
        init = c._per_group(c.lam_init, "lam_init")
        self.lam: dict[str, float] = {g: float(init[g]) for g in self.groups}
        self.avg_violation: dict[str, float] = {g: 0.0 for g in self.groups}
        self.err_prev: dict[str, float | None] = {g: None for g in self.groups}
        self._d: dict[str, float] = {g: 0.0 for g in self.groups}
        self._seen: dict[str, bool] = {g: False for g in self.groups}
        self._stale: dict[str, int] = {g: 0 for g in self.groups}
        self.steps: int = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------- weights --
    def weights(self) -> dict[str, float]:
        """PID-adjusted multipliers for ``FusedCost.forward(weights=...)``.

        ``self.lam`` is the I-state.  The cost is priced at
        ``λ + Kp e + Kd Δe`` so a feasible batch drops the hard-wall price
        without waiting for the integrator.  ``Kd Δe`` is the kick from the last
        :meth:`update` (held until the next one): the EMA is frozen between
        dual steps, so recomputing ``Δe`` against a just-written ``err_prev``
        would make D identically zero for the 100 cost steps that actually
        see ``weights()``.

        Groups the cost knows about but this object does not (``smooth``,
        ``posture``) are simply absent, and default to 1.0 there.

        ``pen_*`` is NOT in this dict -- :meth:`pen_weights` carries those so a
        caller that only wanted λ (every test of the two torque halves) cannot
        silently pick up a quadratic.
        """
        out: dict[str, float] = {}
        for g in self.groups:
            e = self.avg_violation[g] - self.target[g]
            used = self.lam[g] + self.kp[g] * e + self._d[g]
            out[g] = float(min(max(used, self.lam_min[g]), self.lam_max[g]))
        return out

    def pen_weights(self) -> dict[str, float]:
        """Per-group PHR ``pen``, 0 entries omitted so the cost stays bit-identical."""
        return {g: float(self.pen[g]) for g in self.groups if self.pen[g] != 0.0}

    def cost_weights(self) -> dict[str, float]:
        """``weights()`` plus ``pen_<group>`` for the quadratic."""
        out = self.weights()
        for g, v in self.pen_weights().items():
            out[f"pen_{g}"] = v
        return out

    # -------------------------------------------------------------- update --
    def update(
        self, violations: dict[str, torch.Tensor | float], *, record: bool = True
    ) -> dict[str, float]:
        """One dual step from a batch's violations.  Returns the new *used* lambdas.

        ``violations`` maps group -> per-sample violation ``(B,)`` (or a scalar);
        the batch mean is what drives the update.  Groups absent from the dict
        are left untouched, so a curriculum stage can simply not report a group.
        """
        beta = float(self.cfg.ema)
        grow = float(self.cfg.pen_grow)
        shrink = float(self.cfg.pen_shrink)
        tol = float(self.cfg.improve_tol)
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
            e = smoothed - self.target[g]
            leak = (
                float(self.cfg.feasible_decay) if e < 0.0 else float(self.cfg.decay)
            )
            lam = self.lam[g] * leak + self.rho[g] * e
            self.lam[g] = float(min(max(lam, self.lam_min[g]), self.lam_max[g]))
            prev_e = self.err_prev[g]
            self._d[g] = (
                0.0 if prev_e is None else self.kd[g] * (e - prev_e)
            )
            if grow != 1.0 or shrink != 1.0:
                seen = self._seen[g]
                improved = smoothed <= tol * prev if seen else True
                at_cap = self.lam[g] >= self.lam_max[g]
                patience = max(int(self.cfg.pen_patience), 1)
                if e <= 0.0:
                    self._stale[g] = 0
                    # Zero is the explicit "quadratic disabled" state.
                    # ``pen_min`` floors only penalties that were enabled.
                    if shrink != 1.0 and self.pen[g] > 0.0:
                        self.pen[g] = float(max(self.pen[g] / shrink, self.pen_min[g]))
                elif at_cap:
                    # Dual already at the clip: growing pen is a second,
                    # unbounded price on the same residual.  Measured: 20
                    # grows of 1.5x in 2k steps took cost 4k -> 270k.
                    self._stale[g] = 0
                elif seen and not improved:
                    self._stale[g] += 1
                    if self._stale[g] >= patience:
                        self.pen[g] = float(min(self.pen[g] * grow, self.pen_max[g]))
                        self._stale[g] = 0
                else:
                    self._stale[g] = 0
            self._seen[g] = True
            self.err_prev[g] = e
        self.steps += 1
        if record:
            rec = {"step": float(self.steps), **self.lam}
            rec.update({f"pen_{g}": self.pen[g] for g in self.groups})
            self.history.append(rec)
        return self.weights()

    def quadratic_penalty(
        self, violations: dict[str, torch.Tensor]
    ) -> torch.Tensor | float:
        """``sum_g pen_g / 2 * V_g^2`` (per sample).  Prefer the in-cost path.

        :class:`~amortized_opt.cost.FusedCost` adds this on the live residual
        when ``pen_*`` keys are in ``weights``.  This helper is for callers that
        already have an undetached ``V``.
        """
        out = 0.0
        for g in self.groups:
            v = violations.get(g)
            if v is None:
                continue
            out = out + 0.5 * self.pen[g] * v * v
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
            "pen": dict(self.pen),
            "err_prev": {
                g: (None if self.err_prev[g] is None else float(self.err_prev[g]))
                for g in self.groups
            },
            "d_term": dict(self._d),
            "seen": {g: bool(self._seen[g]) for g in self.groups},
            "stale": {g: int(self._stale[g]) for g in self.groups},
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
        if "pen" in state:
            self.pen.update({k: float(v) for k, v in state["pen"].items()})
        if "err_prev" in state:
            for k, v in state["err_prev"].items():
                if k in self.err_prev:
                    self.err_prev[k] = None if v is None else float(v)
        if "d_term" in state:
            self._d.update({k: float(v) for k, v in state["d_term"].items()
                            if k in self._d})
        if "seen" in state:
            for k, v in state["seen"].items():
                if k in self._seen:
                    self._seen[k] = bool(v)
        if "stale" in state:
            for k, v in state["stale"].items():
                if k in self._stale:
                    self._stale[k] = int(v)
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
