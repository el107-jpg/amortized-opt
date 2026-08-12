"""Strike-window impedance schedule: soften the distal arm around impact.

Why this exists (2026-08-12): a strike-speed arm swing puts its reaction
torque through the waist -- ``tests/test_rnea.py`` measures 24.7 N*m on the
waist rows for a 300 rad/s^2 shoulder acceleration, i.e. essentially the
whole 25 N*m waist pitch/roll effort budget -- and the impact impulse itself
kicks back through the wrist and elbow into the shoulder and waist.  Lowering
the striking hand's wrist/elbow stiffness and damping in a short window
around its hit shrinks both: the servo stops fighting the impact (less
torque reflected up the chain) and the rebound is absorbed distally instead
of rippling into the posture joints.

Deliberately LOCAL in time and in joints.  README section 9.3 found that a
GLOBAL ``kd x0.5`` is the one gain setting that diverges under the reference
lead ("under-damping plus a reference lead is an instability"), so this
schedule never touches the shoulder or waist rows, ramps linearly in and out
of the window instead of stepping, and returns exactly to nominal outside it.

Both plants accept the resulting multipliers through the same kwarg --
``BatchedPDSim.step(..., gain_scale=...)`` and the Isaac
``DrumRLEnv.plant(..., gain_scale=...)`` -- so turning the schedule on does
not make the analytic model and Isaac different plants.

Tick conventions differ per caller and that is fine at these window widths:
``ticks_to_hit = 0`` means "the hit lands during / at the end of the tick
about to be executed"; negative means the hand has no scheduled hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import G1_UPPER_BODY_JOINTS

__all__ = ["StrikeImpedanceConfig", "StrikeImpedance", "distal_joint_masks"]

#: Joint-name substrings that mark the distal (softened) part of each arm.
_DISTAL = ("elbow", "wrist")


def distal_joint_masks(
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
) -> torch.Tensor:
    """``(2, n)`` 0/1 mask of each hand's elbow + wrist joints, left = row 0.

    Shoulder and waist columns are zero by construction -- the schedule must
    never soften the joints that hold the posture.
    """
    m = torch.zeros(2, len(joint_names))
    for j, name in enumerate(joint_names):
        for h, hand in enumerate(("left", "right")):
            if name.startswith(hand + "_") and any(p in name for p in _DISTAL):
                m[h, j] = 1.0
    return m


@dataclass
class StrikeImpedanceConfig:
    """Depth and width of the per-hand gain well around a strike.

    ``kp_scale`` / ``kd_scale`` are the multipliers AT the hit tick; outside
    the window the gains are exactly nominal.  The damping default is milder
    than the stiffness one on purpose (section 9.3: under-damping is the
    dangerous direction).
    """

    kp_scale: float = 0.5
    kd_scale: float = 0.7
    #: Ticks before the hit over which the well ramps in (0 = only the hit tick).
    before_wp: int = 3
    #: Ticks after the hit over which the well ramps back out.
    after_wp: int = 3

    def __post_init__(self) -> None:
        for what, v in (("kp_scale", self.kp_scale), ("kd_scale", self.kd_scale)):
            if not 0.0 < float(v) <= 1.0:
                raise ValueError(f"{what} must be in (0, 1], got {v}")
        for what, v in (("before_wp", self.before_wp), ("after_wp", self.after_wp)):
            if int(v) < 0:
                raise ValueError(f"{what} must be >= 0, got {v}")
        self.before_wp = int(self.before_wp)
        self.after_wp = int(self.after_wp)


class StrikeImpedance:
    """Stateful schedule: ``scales()`` BEFORE the plant tick, ``observe_fired()``
    AFTER the clock advances, ``reset()`` on episode / env reset.

    The well depth of a hand is ``max(approach, recovery)`` with two linear
    ramps: approach rises from 0 at ``before_wp + 1`` ticks out to 1 at the
    hit tick; recovery falls from 1 on the tick after the hit to 0 once
    ``after_wp + 1`` ticks have passed.  ``max`` (not sum) so a fast double
    hit on one hand saturates at the configured depth instead of stacking.
    """

    _FAR = 1_000_000  # "no hit seen yet" sentinel for the since-hit counter

    def __init__(
        self,
        cfg: StrikeImpedanceConfig,
        batch: int,
        device: torch.device | str = "cpu",
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self._mask = distal_joint_masks(joint_names).to(self.device)  # (2, n)
        self._since = torch.full((int(batch), 2), self._FAR,
                                 device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._since.fill_(self._FAR)
        else:
            self._since[env_ids] = self._FAR

    def scales(self, ticks_to_hit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, 2)`` ticks-to-hit (negative = idle) -> ``(kp_mul, kd_mul)`` (B, n)."""
        tth = ticks_to_hit.to(device=self.device, dtype=self._mask.dtype)
        b = float(self.cfg.before_wp) + 1.0
        a = float(self.cfg.after_wp) + 1.0
        approach = ((b - tth) / b).clamp(0.0, 1.0) * (tth >= 0)
        recovery = ((a - self._since.to(self._mask.dtype)) / a).clamp(0.0, 1.0)
        depth = torch.maximum(approach, recovery)          # (B, 2)
        w = depth @ self._mask                              # (B, n); hands disjoint
        kp_mul = 1.0 + w * (float(self.cfg.kp_scale) - 1.0)
        kd_mul = 1.0 + w * (float(self.cfg.kd_scale) - 1.0)
        return kp_mul, kd_mul

    def observe_fired(self, fired: torch.Tensor) -> None:
        """``(B, 2)`` bool -- which hands' hits fired during the tick just run.

        Call EVERY tick (all-false included): the counter that decays the
        recovery ramp advances here.
        """
        self._since = torch.where(
            fired.to(self.device), torch.zeros_like(self._since),
            (self._since + 1).clamp(max=self._FAR),
        )
