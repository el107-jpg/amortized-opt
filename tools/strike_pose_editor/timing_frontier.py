"""Frozen production timing-critic inference for editor boundary tasks.

The editor must show the same proposal distribution as the actor's boundary
rows without running an actor/physics admission pass. This adapter restores
only the checkpoint's frozen EMA distributional ensemble and uses the same
random positive ray, timing grid, calibrated risk threshold, and failed-anchor
exploration rule as the production training loop.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import torch

from amortized_opt.distributional_critic import (
    DistributionalEnsemble,
    distributional_risk,
)
from amortized_opt.timing_curriculum import (
    directional_pairs,
    first_exit_indices,
    sample_directions,
)
from amortized_opt.train_cost import effective_hits, retime_batch_timing
from amortized_opt.violation_critic import (
    ViolationCriticConfig,
    build_critic_features,
    critic_feature_dim,
)

__all__ = ["CheckpointTimingFrontier"]


# The teaching app needs enough pre-strike trajectory for a human to inspect
# and correct.  Hit indices are one-based, so 20 means the earliest contact is
# waypoint 20 (0.40 s on the current 50 Hz grid).
MIN_HIT_WAYPOINT = 20


class CheckpointTimingFrontier:
    """Read-only calibrated timing selector restored from an actor checkpoint."""

    def __init__(
        self,
        cfg: ViolationCriticConfig,
        state: Mapping[str, Any],
        *,
        checkpoint_step: int,
        n_dof: int,
    ) -> None:
        if cfg.representation != "distributional":
            raise ValueError("editor boundary sampling requires a distributional critic")
        if state.get("representation") != "distributional":
            raise ValueError("checkpoint does not contain a distributional timing critic")
        self.cfg = cfg
        self.checkpoint_step = int(checkpoint_step)
        self.input_dim = critic_feature_dim(n_dof)
        if int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError("checkpoint timing-critic feature width is incompatible")
        self.model = DistributionalEnsemble(
            self.input_dim,
            num_heads=cfg.num_heads,
            kind="softmax",
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
        ).cpu().eval()
        self.model.load_state_dict(state["ema"], strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.risk_threshold = float(state.get("risk_threshold", float("-inf")))
        self.curriculum_open = bool(state.get("curriculum_open", False))
        if not self.curriculum_open or not math.isfinite(self.risk_threshold):
            raise ValueError("checkpoint timing curriculum is not calibrated/open")
        self.calibration_precision = float(state.get("calibration_precision", 0.0))
        self.calibration_recall = float(state.get("calibration_recall", 0.0))
        self.anchor_samples = int(state.get("anchor_samples", 0))
        self.generator = torch.Generator(device="cpu")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("checkpoint timing critic has no generator state")
        self.generator.set_state(generator_state.detach().cpu())

    @classmethod
    def from_checkpoint(
        cls, raw: Mapping[str, Any], *, n_dof: int,
    ) -> "CheckpointTimingFrontier":
        train_cfg = raw.get("train_cfg")
        extra = raw.get("extra")
        if not isinstance(train_cfg, Mapping) or not isinstance(extra, Mapping):
            raise ValueError("checkpoint lacks training or extra state")
        cfg_raw = train_cfg.get("timing_critic")
        state = extra.get("timing_critic")
        if not isinstance(cfg_raw, Mapping) or not isinstance(state, Mapping):
            raise ValueError("checkpoint lacks a saved timing critic")
        return cls(
            ViolationCriticConfig(**dict(cfg_raw)),
            state,
            checkpoint_step=int(raw.get("step", 0)),
            n_dof=n_dof,
        )

    def _risk(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            logits, _distance = self.model(features.detach().cpu().float())
            return distributional_risk(
                logits.softmax(dim=-1),
                acceptable_limit=self.cfg.acceptable_normalized_violation,
                severity_linear=self.cfg.risk_severity_linear,
                severity_quadratic=self.cfg.risk_severity_quadratic,
                tail_alpha=self.cfg.risk_tail_alpha,
                tail_weight=self.cfg.risk_tail_weight,
                epistemic_beta=self.cfg.risk_epistemic_beta,
            )

    @torch.no_grad()
    def apply(self, batch: dict, pool: Any) -> dict[str, Any]:
        """Select the production boundary pair and retime ``batch`` in place."""
        active = batch["active"].detach().cpu().bool()
        if active.ndim != 2 or active.shape[1] != 2 or not bool(active.any(dim=1).all()):
            raise ValueError("timing frontier requires at least one active hand per task")
        horizon = int(batch["bucket"])
        dt = float(pool.cfg.dt)
        directions = sample_directions(
            active, generator=self.generator, axis_probability=0.10,
        )
        pairs, _alpha = directional_pairs(
            active, directions, horizon=horizon,
            step=int(self.cfg.coarse_step_knots),
        )
        if horizon < MIN_HIT_WAYPOINT:
            raise ValueError(
                f"timing horizon {horizon} is shorter than the editor minimum "
                f"hit waypoint {MIN_HIT_WAYPOINT}"
            )
        # The editor currently samples one physical task at a time.  Truncate
        # its timing ray before critic evaluation so risks, selected indices,
        # and the displayed Pareto ray all describe the same admissible grid.
        if pairs.shape[0] != 1:
            raise ValueError("editor timing selector currently expects one task")
        admissible = torch.logical_or(
            ~active[:, None, :], pairs >= MIN_HIT_WAYPOINT,
        ).all(dim=-1)[0]
        pairs = pairs[:, admissible]
        if pairs.shape[1] == 0:
            raise ValueError("timing ray has no pair at or beyond the minimum hit waypoint")
        b, k, _ = pairs.shape
        repeated = lambda value: value.detach().cpu().repeat_interleave(k, dim=0)  # noqa: E731
        pair_rows = pairs.reshape(-1, 2)
        goal_tth = torch.where(
            active.repeat_interleave(k, dim=0),
            pair_rows.float() * dt,
            torch.zeros_like(pair_rows, dtype=torch.float32),
        )
        features = build_critic_features(
            q0=repeated(batch["q0"]),
            qd0=repeated(batch["v0"]),
            goal_pos=repeated(batch["context"].goal_pos),
            goal_tth=goal_tth,
        )
        risk, detail = self._risk(features)
        risk = risk.reshape(b, k)
        manageable = risk <= self.risk_threshold
        _unused, last_pass, first_fail = first_exit_indices(manageable)
        anchor_ok = manageable[:, 0]
        # Production's no-fallback rule: if the anchor is uncertain, still
        # advance one cell and obtain the informative exact label in-batch.
        explore = torch.full_like(last_pass, min(1, k - 1))
        selected = torch.where(anchor_ok, last_pass, explore)
        selected = selected.clamp_min(min(1, k - 1))
        row = torch.arange(b)
        selected_pair = pairs[row, selected]
        selected_risk = risk[row, selected]
        selected_ok = manageable[row, selected]

        retime_batch_timing(batch, selected_pair.to(batch["q0"].device), pool)
        batch["strike"] = replace(
            batch["strike"],
            hit_index=effective_hits(batch["hit"], batch["active"], horizon),
        )

        selected_i = int(selected[0])
        component = {
            name: float(value.reshape(b, k)[0, selected_i])
            for name, value in detail.items()
        }
        return {
            "timing_source": "production_critic_boundary",
            "timing_checkpoint_step": self.checkpoint_step,
            "timing_pair": [int(v) for v in selected_pair[0].tolist()],
            "timing_tth_s": [float(v * dt) if v >= 0 else None for v in selected_pair[0].tolist()],
            "timing_ray_direction": [float(v) for v in directions[0].tolist()],
            "timing_ray_index": selected_i,
            "timing_ray_pairs": [[int(v) for v in pair] for pair in pairs[0].tolist()],
            "timing_min_hit_waypoint": MIN_HIT_WAYPOINT,
            "timing_risk": float(selected_risk[0]),
            "timing_risk_threshold": self.risk_threshold,
            "timing_risk_components": component,
            "timing_anchor_manageable": bool(anchor_ok[0]),
            "timing_selected_manageable": bool(selected_ok[0]),
            "timing_ray_all_manageable": bool(first_fail[0] == k),
            "timing_calibration_precision": self.calibration_precision,
            "timing_calibration_recall": self.calibration_recall,
            "timing_anchor_samples": self.anchor_samples,
            "timing_actor_validated": False,
        }
