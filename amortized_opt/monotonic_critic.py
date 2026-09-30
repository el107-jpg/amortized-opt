"""Experimental monotonic violation critic with observable confidence.

This module is deliberately independent of the production timing critic.  It
is used to test three ideas on a stationary actor-label stream:

* constraint residuals are coordinatewise non-increasing in active-hand TTH;
* confidence predicts whether the *pre-update* residual prediction was close;
* confident old predictions resist unrelated updates through function-space
  inertia.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .critic_task_sampler import GlobalCriticTaskSampler
from .violation_critic import (
    VIOLATION_NAMES,
    critic_loss_per_head_row,
    residuals_from_gate,
    robust_inverse,
    robust_transform,
    ViolationCriticConfig,
)


MONOTONIC_CRITIC_FORMAT_VERSION = 2


@dataclass
class MonotonicConfidenceConfig:
    num_heads: int = 3
    # The capacity/anchor ablation selected width 256 as the safety/coverage
    # knee. Width 512 improved average error but exceeded the shifted
    # false-feasible budget.
    hidden_dim: int = 256
    depth: int = 2
    knots: int = 8
    max_tth_s: float = 1.0
    trunk_architecture: str = "mlp"
    expert_count: int = 8
    expert_top_k: int = 2
    expert_hidden_dim: int = 80
    expert_router_temperature: float = 1.0
    expert_balance_weight: float = 0.01
    expert_router_z_weight: float = 1.0e-4
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    ema_decay: float = 0.95
    bootstrap_probability: float = 0.8
    batch_size: int = 256
    updates_per_observation: int = 1
    # Retain the complete default pre-anchor warmup so the initial anchor bank
    # can be seeded from the same base-training window used in the ablation.
    replay_capacity: int = 32_768

    # Signed asinh-space initialization.  Three is deliberately pessimistic:
    # it corresponds to roughly 10x the physical scale for each constraint.
    initial_violation: float = 3.0
    initial_derivative_logit: float = -5.0
    initial_confidence: float = 0.01
    # Confidence predicts correct feasibility classification for each signed
    # residual.  Magnitude accuracy remains visible through the regression
    # metrics and is not conflated with decision reliability.
    confidence_tolerance: float = 0.50
    confidence_loss_weight: float = 1.0

    # Auxiliary exposure of valid but unlabeled physical-domain tasks.  These
    # are evidence for low support, not fabricated violation targets.
    background_fraction: float = 1.0
    background_confidence_weight: float = 0.05

    # A frozen prior perturbs positive derivative logits only, so it can create
    # ensemble diversity without breaking the monotonicity guarantee.
    randomized_prior_scale: float = 0.10

    # Nearest-memory support factor used to temper learned correctness.
    support_points: int = 512
    support_bandwidth: float = 0.35

    # Function-space consolidation.  Only previously confident residuals are
    # anchored; initial pessimism therefore does not freeze the network.
    # A replay-turnover sweep found 0.5 to be the useful knee: it reduced
    # old-function drift without the adaptation loss visible at 2.0.
    inertia_weight: float = 0.50
    inertia_warmup_updates: int = 100
    reference_refresh_updates: int = 100

    # Optional long-lived functional anchors.  Ordinary replay remains the
    # fast memory; anchors decay over multiple replay turnovers and may be
    # locally refreshed by newer exact labels.
    anchor_capacity: int = 2_048
    # Anchoring the zero-confidence pessimistic initialization would oppose
    # the first useful learning updates.  The selected experiment trained on
    # 18,432 labels before seeding its anchor bank, so preserve that lifecycle
    # in the default online trainer.
    anchor_warmup_rows: int = 18_432
    anchor_batch_size: int = 128
    anchor_exact_weight: float = 0.25
    anchor_inertia_weight: float = 1.0
    anchor_drift_tolerance: float = 0.05
    anchor_half_life_rows: int = 55_296
    anchor_slow_fraction: float = 0.20
    anchor_slow_half_life_multiplier: float = 5.0
    anchor_match_radius: float = 0.20
    anchor_replacement: bool = True

    # Match production target and asymmetric regression semantics.
    terminal_limit: float = 0.020
    collision_sdf_limit: float = -0.005
    joint_limit: float = 0.020
    torque_limit: float = 8.0
    terminal_scale: float = 0.020
    collision_scale: float = 0.010
    joint_scale: float = 0.020
    torque_scale: float = 8.0
    huber_delta: float = 1.0
    boundary_loss_weight: float = 0.25
    boundary_temperature: float = 0.50
    boundary_target_band: float = 0.50
    underprediction_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.num_heads < 1 or self.hidden_dim < 1 or self.depth < 1:
            raise ValueError("critic head counts and widths must be positive")
        if self.trunk_architecture not in {"mlp", "local_experts"}:
            raise ValueError("trunk_architecture must be 'mlp' or 'local_experts'")
        if (
            self.expert_count < 1
            or not 1 <= self.expert_top_k <= self.expert_count
            or self.expert_hidden_dim < 1
            or self.expert_router_temperature <= 0.0
            or self.expert_balance_weight < 0.0
            or self.expert_router_z_weight < 0.0
        ):
            raise ValueError("invalid local-expert settings")
        if self.knots < 1 or self.max_tth_s <= 0.0:
            raise ValueError("monotonic basis settings must be positive")
        if not 0.0 < self.initial_confidence < 1.0:
            raise ValueError("initial_confidence must be in (0, 1)")
        if not 0.0 < self.bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must be in (0, 1]")
        if self.batch_size < 1 or self.replay_capacity < 1:
            raise ValueError("batch and replay sizes must be positive")
        if self.updates_per_observation < 1:
            raise ValueError("updates_per_observation must be positive")
        if self.support_points < 1 or self.support_bandwidth <= 0.0:
            raise ValueError("support settings must be positive")
        if self.reference_refresh_updates < 1 or self.inertia_warmup_updates < 0:
            raise ValueError("inertia update intervals are invalid")
        if (
            self.anchor_capacity < 0
            or self.anchor_batch_size < 1
            or self.anchor_warmup_rows < 0
        ):
            raise ValueError("anchor capacity must be nonnegative and batch size positive")
        if self.anchor_half_life_rows < 1 or self.anchor_match_radius <= 0.0:
            raise ValueError("anchor lifetime and match radius must be positive")
        if not 0.0 <= self.anchor_slow_fraction <= 1.0:
            raise ValueError("anchor_slow_fraction must be in [0, 1]")
        if self.anchor_slow_half_life_multiplier < 1.0:
            raise ValueError("slow anchor half-life multiplier must be >= 1")
        if min(
            self.background_fraction,
            self.background_confidence_weight,
            self.randomized_prior_scale,
            self.inertia_weight,
            self.anchor_exact_weight,
            self.anchor_inertia_weight,
            self.anchor_drift_tolerance,
        ) < 0.0:
            raise ValueError("critic regularization weights must be nonnegative")

    @property
    def scales(self) -> tuple[float, float, float, float]:
        return (
            float(self.terminal_scale), float(self.collision_scale),
            float(self.joint_scale), float(self.torque_scale),
        )

    def regression_config(self) -> ViolationCriticConfig:
        """Minimal production config used by shared target/loss helpers."""
        return ViolationCriticConfig(
            terminal_limit=self.terminal_limit,
            collision_sdf_limit=self.collision_sdf_limit,
            joint_limit=self.joint_limit,
            torque_limit=self.torque_limit,
            terminal_scale=self.terminal_scale,
            collision_scale=self.collision_scale,
            joint_scale=self.joint_scale,
            torque_scale=self.torque_scale,
            huber_delta=self.huber_delta,
            boundary_loss_weight=self.boundary_loss_weight,
            boundary_temperature=self.boundary_temperature,
            boundary_target_band=self.boundary_target_band,
            underprediction_multiplier=self.underprediction_multiplier,
        )


class _FrozenDerivativePrior(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LocalExpertTrunk(nn.Module):
    """Top-k local experts with an input-dependent, balanced router."""

    def __init__(self, context_dim: int, cfg: MonotonicConfidenceConfig):
        super().__init__()
        shared_dim = max(32, int(cfg.hidden_dim) // 2)
        self.expert_count = int(cfg.expert_count)
        self.top_k = int(cfg.expert_top_k)
        self.temperature = float(cfg.expert_router_temperature)
        self.balance_weight = float(cfg.expert_balance_weight)
        self.z_weight = float(cfg.expert_router_z_weight)
        self.output_dim = int(cfg.expert_hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(context_dim, shared_dim), nn.SiLU(),
        )
        self.router = nn.Linear(context_dim, self.expert_count)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, self.output_dim), nn.SiLU(),
            )
            for _ in range(self.expert_count)
        ])

    def _routing(
        self, context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(context)
        top_logits, top_index = torch.topk(logits, self.top_k, dim=-1)
        top_weight = F.softmax(top_logits / self.temperature, dim=-1)
        return logits, top_index, top_weight

    def forward(
        self, context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(context)
        logits, top_index, top_weight = self._routing(context)
        expert_output = torch.stack(
            [expert(shared) for expert in self.experts], dim=1,
        )
        selected = expert_output.gather(
            1,
            top_index[:, :, None].expand(-1, -1, self.output_dim),
        )
        hidden = (selected * top_weight[:, :, None]).sum(dim=1)

        probability = F.softmax(logits / self.temperature, dim=-1)
        assignment = torch.zeros_like(probability).scatter_add(
            1, top_index,
            torch.full_like(top_weight, 1.0 / float(self.top_k)),
        )
        balance = self.expert_count * (
            probability.mean(dim=0) * assignment.mean(dim=0)
        ).sum() - 1.0
        router_z = torch.logsumexp(logits, dim=-1).square().mean()
        auxiliary = self.balance_weight * balance + self.z_weight * router_z
        return hidden, auxiliary

    @torch.no_grad()
    def routing_statistics(self, context: torch.Tensor) -> dict[str, float]:
        logits, top_index, _top_weight = self._routing(context)
        counts = torch.bincount(
            top_index.flatten(), minlength=self.expert_count,
        ).float()
        fraction = counts / counts.sum().clamp_min(1.0)
        probability = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
        return {
            "active_experts": float((counts > 0.0).sum()),
            "max_selection_fraction": float(fraction.max()),
            "min_selection_fraction": float(fraction.min()),
            "router_entropy": float(entropy.mean()),
        }


class MonotonicConfidenceHead(nn.Module):
    """One additive, coordinatewise-monotonic TTH residual model."""

    def __init__(self, input_dim: int, cfg: MonotonicConfidenceConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = self.input_dim  # replace two TTHs with two active bits
        self.knots = int(cfg.knots)
        self.max_tth_s = float(cfg.max_tth_s)
        if cfg.trunk_architecture == "local_experts":
            self.trunk = _LocalExpertTrunk(self.context_dim, cfg)
            width = int(cfg.expert_hidden_dim)
        else:
            layers: list[nn.Module] = []
            width = self.context_dim
            for _ in range(int(cfg.depth)):
                layer = nn.Linear(width, int(cfg.hidden_dim))
                nn.init.zeros_(layer.weight)
                nn.init.uniform_(layer.bias, -0.10, 0.10)
                layers.extend((layer, nn.SiLU()))
                width = int(cfg.hidden_dim)
            self.trunk = nn.Sequential(*layers)

        # baseline + left slopes + right slopes + positive interaction slopes
        self.residual_output = nn.Linear(width, 4 * (1 + 3 * self.knots))
        nn.init.zeros_(self.residual_output.weight)
        with torch.no_grad():
            shaped = self.residual_output.bias.view(4, 1 + 3 * self.knots)
            shaped[:, 0].fill_(float(cfg.initial_violation))
            shaped[:, 1:].fill_(float(cfg.initial_derivative_logit))

        self.confidence_output = nn.Linear(width, 4)
        nn.init.zeros_(self.confidence_output.weight)
        initial_logit = math.log(
            float(cfg.initial_confidence) / (1.0 - float(cfg.initial_confidence))
        )
        nn.init.constant_(self.confidence_output.bias, initial_logit)

        self.prior_scale = float(cfg.randomized_prior_scale)
        self.prior = _FrozenDerivativePrior(
            self.context_dim, int(cfg.hidden_dim), 4 * 3 * self.knots,
        )
        self.prior.requires_grad_(False)
        self.register_buffer(
            "basis_knots", torch.linspace(0.0, 1.0, self.knots + 1)[:-1],
        )

    def _context_and_basis(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tth = features[:, -2:].clamp_min(0.0)
        active = (tth > 0.0).to(features.dtype)
        context = torch.cat((features[:, :-2], active), dim=1)
        normalized = (tth / self.max_tth_s).clamp(0.0, 1.0)
        basis = (
            normalized[:, :, None] - self.basis_knots[None, None, :]
        ).clamp_min(0.0)
        basis = basis * active[:, :, None]
        interaction = basis[:, 0] * basis[:, 1]
        return context, basis[:, 0], basis[:, 1], interaction

    def forward(
        self, features: torch.Tensor, *, return_aux: bool = False,
    ):
        context, left_basis, right_basis, interaction = self._context_and_basis(features)
        if isinstance(self.trunk, _LocalExpertTrunk):
            hidden, expert_aux = self.trunk(context)
        else:
            hidden = self.trunk(context)
            expert_aux = hidden.new_zeros(())
        raw = self.residual_output(hidden).view(
            -1, 4, 1 + 3 * self.knots,
        )
        baseline = raw[:, :, 0]
        derivative_logits = raw[:, :, 1:]
        if self.prior_scale > 0.0:
            derivative_logits = derivative_logits + self.prior_scale * self.prior(
                context
            ).view(-1, 4, 3 * self.knots)
        derivative = F.softplus(derivative_logits)
        basis = torch.cat((left_basis, right_basis, interaction), dim=1)
        residual = baseline - (derivative * basis[:, None, :]).sum(dim=-1)
        confidence = self.confidence_output(hidden)
        if return_aux:
            return residual, confidence, expert_aux
        return residual, confidence

    @torch.no_grad()
    def routing_statistics(self, features: torch.Tensor) -> dict[str, float] | None:
        if not isinstance(self.trunk, _LocalExpertTrunk):
            return None
        context, _left, _right, _interaction = self._context_and_basis(features)
        return self.trunk.routing_statistics(context)


class MonotonicConfidenceEnsemble(nn.Module):
    def __init__(self, input_dim: int, cfg: MonotonicConfidenceConfig):
        super().__init__()
        self.heads = nn.ModuleList([
            MonotonicConfidenceHead(input_dim, cfg)
            for _ in range(int(cfg.num_heads))
        ])

    def forward(self, features: torch.Tensor, *, return_aux: bool = False):
        values = [head(features, return_aux=return_aux) for head in self.heads]
        prediction = torch.stack([value[0] for value in values])
        confidence = torch.stack([value[1] for value in values])
        if return_aux:
            auxiliary = torch.stack([value[2] for value in values]).mean()
            return prediction, confidence, auxiliary
        return prediction, confidence


class ConfidenceReplay:
    """Bounded labels with one residual-calibration holdout head per row."""

    def __init__(self, capacity: int, input_dim: int, num_heads: int):
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.features = torch.empty(self.capacity, self.input_dim, dtype=torch.float16)
        self.targets = torch.empty(self.capacity, 4, dtype=torch.float16)
        self.calibration_head = torch.empty(self.capacity, dtype=torch.int16)
        self.actor_step = torch.empty(self.capacity, dtype=torch.int64)
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        calibration_head: torch.Tensor,
        *,
        actor_step: int,
    ) -> None:
        f = features.detach().cpu().to(torch.float16)
        y = targets.detach().cpu().to(torch.float16)
        held_out = calibration_head.detach().cpu().to(torch.int16).flatten()
        if held_out.shape != (f.shape[0],):
            raise ValueError("calibration_head must have one entry per row")
        for row in range(int(f.shape[0])):
            index = self.cursor
            self.features[index].copy_(f[row])
            self.targets[index].copy_(y[row])
            self.calibration_head[index] = held_out[row]
            self.actor_step[index] = int(actor_step)
            self.cursor = (self.cursor + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self)
        if n == 0:
            raise RuntimeError("cannot sample empty confidence replay")
        index = torch.randint(0, n, (min(int(count), n),), generator=generator)
        return (
            self.features.index_select(0, index).to(device, torch.float32),
            self.targets.index_select(0, index).to(device, torch.float32),
            self.calibration_head.index_select(0, index).to(device).long(),
        )

    def state_dict(self) -> dict[str, Any]:
        n = len(self)
        return {
            "capacity": self.capacity, "input_dim": self.input_dim,
            "num_heads": self.num_heads, "size": n, "cursor": self.cursor,
            "features": self.features[:n].clone(),
            "targets": self.targets[:n].clone(),
            "calibration_head": self.calibration_head[:n].clone(),
            "actor_step": self.actor_step[:n].clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = min(int(state["size"]), self.capacity)
        self.features[:n].copy_(state["features"][:n].to(torch.float16))
        self.targets[:n].copy_(state["targets"][:n].to(torch.float16))
        self.calibration_head[:n].copy_(state["calibration_head"][:n].short())
        self.actor_step[:n].copy_(state["actor_step"][:n].long())
        self.size = n
        self.cursor = int(state.get("cursor", n % self.capacity)) % self.capacity


class ElasticAnchorMemory:
    """Slow exact-label memory with age decay and stochastic local refresh."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        num_heads: int,
        *,
        slow_fraction: float,
        half_life_rows: int,
        slow_half_life_multiplier: float,
        match_radius: float,
        replacement: bool,
    ):
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.slow_capacity = int(round(self.capacity * float(slow_fraction)))
        self.half_life_rows = int(half_life_rows)
        self.slow_half_life_multiplier = float(slow_half_life_multiplier)
        self.match_radius = float(match_radius)
        self.replacement = bool(replacement)
        self.features = torch.empty(self.capacity, self.input_dim, dtype=torch.float16)
        self.targets = torch.empty(self.capacity, 4, dtype=torch.float16)
        self.reference = torch.empty(
            self.capacity, self.num_heads, 4, dtype=torch.float16,
        )
        self.birth_row = torch.zeros(self.capacity, dtype=torch.int64)
        self.slow = torch.zeros(self.capacity, dtype=torch.bool)
        self.size = 0
        self.rows_seen = 0
        self.offers = 0
        self.local_replacements = 0
        self.novel_replacements = 0

    def __len__(self) -> int:
        return int(self.size)

    def _write(
        self,
        index: int,
        feature: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor,
        *,
        slow: bool,
    ) -> None:
        self.features[index].copy_(feature.detach().cpu().to(torch.float16))
        self.targets[index].copy_(target.detach().cpu().to(torch.float16))
        self.reference[index].copy_(reference.detach().cpu().to(torch.float16))
        self.birth_row[index] = int(self.rows_seen)
        self.slow[index] = bool(slow)

    def seed(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        """Seed a boundary-aware but distribution-spanning initial bank."""
        if self.capacity == 0 or len(features) == 0:
            return
        n = min(self.capacity, int(features.shape[0]))
        uniform = torch.linspace(0, len(features) - 1, n).long()
        chosen = uniform
        if n >= 4:
            boundary_n = n // 2
            boundary = targets.amax(dim=-1).abs()
            near = torch.topk(boundary, boundary_n, largest=False).indices
            rest = uniform[: n - boundary_n]
            chosen = torch.cat((near, rest)).unique()
            if len(chosen) < n:
                missing = torch.ones(len(features), dtype=torch.bool)
                missing[chosen] = False
                chosen = torch.cat((chosen, missing.nonzero(as_tuple=True)[0][: n-len(chosen)]))
            chosen = chosen[:n]
        boundary = targets.index_select(0, chosen).amax(dim=-1).abs()
        slow_local = torch.topk(
            boundary, min(self.slow_capacity, n), largest=False,
        ).indices if self.slow_capacity else torch.empty(0, dtype=torch.long)
        slow_mask = torch.zeros(n, dtype=torch.bool)
        slow_mask[slow_local] = True
        self.rows_seen = int(features.shape[0])
        for local, source in enumerate(chosen.tolist()):
            self._write(
                local, features[source], targets[source], reference[:, source],
                slow=bool(slow_mask[local]),
            )
        self.size = n

    def offer(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        reference: torch.Tensor,
        *,
        scale: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        """Stochastically refresh nearby old anchors and admit novel rows."""
        count = int(features.shape[0])
        self.offers += count
        start_row = self.rows_seen
        self.rows_seen += count
        if self.capacity == 0 or count == 0:
            return
        x = features.detach().cpu().float()
        y = targets.detach().cpu().float()
        ref = reference.detach().cpu().float()
        fill = min(self.capacity - self.size, count)
        for source in range(fill):
            self._write(
                self.size, x[source], y[source], ref[:, source],
                slow=self.size < self.slow_capacity,
            )
            self.size += 1
        if not self.replacement or self.size == 0:
            return

        anchor = self.features[:self.size].float()
        norm = scale.detach().cpu().float().clamp_min(1.0e-6)
        distance = torch.cdist(x / norm, anchor / norm) / math.sqrt(self.input_dim)
        active_new = x[:, -2:] > 0.0
        active_old = anchor[:, -2:] > 0.0
        compatible = (active_new[:, None, :] == active_old[None, :, :]).all(dim=-1)
        distance = distance.masked_fill(~compatible, float("inf"))
        nearest_distance, nearest = distance.min(dim=1)
        elastic = (~self.slow[:self.size]).nonzero(as_tuple=True)[0]
        for row in range(fill, count):
            index = int(nearest[row])
            dist = float(nearest_distance[row])
            age = max(0, start_row + row - int(self.birth_row[index]))
            half_life = float(self.half_life_rows) * (
                self.slow_half_life_multiplier if bool(self.slow[index]) else 1.0
            )
            age_probability = 1.0 - 2.0 ** (-age / max(half_life, 1.0))
            if math.isfinite(dist) and dist <= self.match_radius:
                similarity = max(0.0, 1.0 - dist / self.match_radius)
                probability = similarity * age_probability
                if float(torch.rand((), generator=generator)) < probability:
                    self._write(
                        index, x[row], y[row], ref[:, row],
                        slow=bool(self.slow[index]),
                    )
                    self.local_replacements += 1
                continue
            if not int(elastic.numel()):
                continue
            reservoir_probability = min(
                1.0, len(elastic) / max(float(start_row + row + 1), 1.0),
            )
            if float(torch.rand((), generator=generator)) >= reservoir_probability:
                continue
            candidates = elastic.index_select(
                0, torch.randint(0, len(elastic), (min(16, len(elastic)),), generator=generator),
            )
            candidate_age = (start_row + row - self.birth_row[candidates]).float()
            replace = int(candidates[candidate_age.argmax()])
            self._write(replace, x[row], y[row], ref[:, row], slow=False)
            self.novel_replacements += 1

    def diagnostics(self) -> dict[str, float]:
        n = max(self.size, 1)
        age = (self.rows_seen - self.birth_row[:self.size]).float()
        return {
            "size": float(self.size),
            "rows_seen": float(self.rows_seen),
            "slow_fraction": float(self.slow[:self.size].float().mean()) if self.size else 0.0,
            "mean_age_rows": float(age.mean()) if self.size else 0.0,
            "offers": float(self.offers),
            "local_replacements": float(self.local_replacements),
            "novel_replacements": float(self.novel_replacements),
            "replacement_fraction": (
                (self.local_replacements + self.novel_replacements) / max(self.offers, 1)
            ),
        }

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample empty anchor memory")
        index = torch.randint(
            0, self.size, (min(int(count), self.size),), generator=generator,
        )
        age = (self.rows_seen - self.birth_row.index_select(0, index)).float()
        slow = self.slow.index_select(0, index)
        half_life = torch.full_like(age, float(self.half_life_rows))
        half_life[slow] *= self.slow_half_life_multiplier
        age_weight = torch.pow(2.0, -age / half_life.clamp_min(1.0))
        target_cpu = self.targets.index_select(0, index).to(torch.float32)
        boundary_weight = torch.exp(-target_cpu.amax(dim=-1).abs() / 0.5)
        importance = age_weight * (0.25 + 0.75 * boundary_weight)
        return (
            self.features.index_select(0, index).to(device, torch.float32),
            target_cpu.to(device),
            self.reference.index_select(0, index).to(device, torch.float32)
                .permute(1, 0, 2),
            importance.to(device),
        )

    def state_dict(self) -> dict[str, Any]:
        n = self.size
        return {
            "capacity": self.capacity, "input_dim": self.input_dim,
            "num_heads": self.num_heads, "size": n,
            "slow_capacity": self.slow_capacity,
            "half_life_rows": self.half_life_rows,
            "slow_half_life_multiplier": self.slow_half_life_multiplier,
            "match_radius": self.match_radius, "replacement": self.replacement,
            "rows_seen": self.rows_seen,
            "offers": self.offers,
            "local_replacements": self.local_replacements,
            "novel_replacements": self.novel_replacements,
            "features": self.features[:n].clone(),
            "targets": self.targets[:n].clone(),
            "reference": self.reference[:n].clone(),
            "birth_row": self.birth_row[:n].clone(),
            "slow": self.slow[:n].clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = min(int(state["size"]), self.capacity)
        self.features[:n].copy_(state["features"][:n].to(torch.float16))
        self.targets[:n].copy_(state["targets"][:n].to(torch.float16))
        self.reference[:n].copy_(state["reference"][:n].to(torch.float16))
        self.birth_row[:n].copy_(state["birth_row"][:n].long())
        self.slow[:n].copy_(state["slow"][:n].bool())
        self.size = n
        self.rows_seen = int(state.get("rows_seen", n))
        self.offers = int(state.get("offers", 0))
        self.local_replacements = int(state.get("local_replacements", 0))
        self.novel_replacements = int(state.get("novel_replacements", 0))


def expected_calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    *,
    bins: int = 10,
) -> torch.Tensor:
    confidence = confidence.float().flatten().clamp(0.0, 1.0)
    correct = correct.float().flatten()
    bucket = torch.clamp((confidence * bins).long(), max=bins - 1)
    total = confidence.new_zeros(())
    for index in range(int(bins)):
        mask = bucket == index
        if bool(mask.any()):
            total = total + mask.float().mean() * (
                confidence[mask].mean() - correct[mask].mean()
            ).abs()
    return total


class MonotonicConfidenceTrainer:
    """Online trainer with prequential confidence and functional inertia."""

    def __init__(
        self,
        input_dim: int,
        cfg: MonotonicConfidenceConfig,
        *,
        device: torch.device | str,
        seed: int,
    ):
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.live = MonotonicConfidenceEnsemble(input_dim, cfg).to(self.device)
        self.ema = copy.deepcopy(self.live).eval()
        self.reference = copy.deepcopy(self.live).eval()
        for model in (self.ema, self.reference):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.live.parameters() if p.requires_grad),
            lr=float(cfg.lr), weight_decay=float(cfg.weight_decay),
        )
        self.recent = ConfidenceReplay(
            cfg.replay_capacity, input_dim, cfg.num_heads,
        )
        self.anchors = (
            ElasticAnchorMemory(
                cfg.anchor_capacity, input_dim, cfg.num_heads,
                slow_fraction=cfg.anchor_slow_fraction,
                half_life_rows=cfg.anchor_half_life_rows,
                slow_half_life_multiplier=cfg.anchor_slow_half_life_multiplier,
                match_radius=cfg.anchor_match_radius,
                replacement=cfg.anchor_replacement,
            )
            if int(cfg.anchor_capacity) > 0 else None
        )
        self.generator = torch.Generator().manual_seed(int(seed) + 1)
        self.anchor_generator = torch.Generator().manual_seed(int(seed) + 10_001)
        self.global_sampler = GlobalCriticTaskSampler(max_tth_s=cfg.max_tth_s)
        self.regression_cfg = cfg.regression_config()
        self.updates = 0
        self.rows_observed = 0
        self._metric_sum: dict[str, float] = {}
        self._metric_n = 0

    def _support_scale(self, features: torch.Tensor) -> torch.Tensor:
        scale = features.new_ones(self.input_dim)
        n_dof = (self.input_dim - 8) // 2
        if 2 * n_dof + 8 == self.input_dim:
            scale[2 * n_dof:2 * n_dof + 6] = 0.50
            scale[-2:] = float(self.cfg.max_tth_s)
        return scale

    @torch.no_grad()
    def support_probability(self, features: torch.Tensor) -> torch.Tensor:
        query = features.detach().to(self.device, torch.float32)
        n = len(self.recent)
        if n == 0:
            return query.new_zeros(query.shape[0])
        count = min(n, int(self.cfg.support_points))
        if count == n:
            index = torch.arange(n)
        else:
            index = torch.linspace(0, n - 1, count).long()
        support = self.recent.features.index_select(0, index).to(
            self.device, torch.float32,
        )
        scale = self._support_scale(query)
        distance = torch.cdist(query / scale, support / scale) / math.sqrt(
            float(self.input_dim)
        )
        nearest = distance.amin(dim=1)
        return torch.exp(
            -0.5 * (nearest / float(self.cfg.support_bandwidth)).square()
        )

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        *,
        ema: bool = True,
        physical: bool = False,
    ) -> torch.Tensor:
        model = self.ema if ema else self.live
        prediction, _ = model(features.detach().to(self.device, torch.float32))
        if physical:
            return robust_inverse(prediction, self.cfg.scales)
        return prediction

    @torch.no_grad()
    def confidence(
        self,
        features: torch.Tensor,
        *,
        ema: bool = True,
        temper_by_support: bool = True,
    ) -> torch.Tensor:
        model = self.ema if ema else self.live
        x = features.detach().to(self.device, torch.float32)
        _, logits = model(x)
        probability = logits.sigmoid()
        if temper_by_support:
            probability = probability * self.support_probability(x)[None, :, None]
        return probability

    @torch.no_grad()
    def routing_statistics(self, features: torch.Tensor) -> dict[str, float] | None:
        """Aggregate local-expert utilization across ensemble members."""
        if self.cfg.trunk_architecture != "local_experts":
            return None
        x = features.detach().to(self.device, torch.float32)
        values = [head.routing_statistics(x) for head in self.ema.heads]
        rows = [value for value in values if value is not None]
        if not rows:
            return None
        return {
            key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]
        }

    @torch.no_grad()
    def prequential_metrics(
        self, features: torch.Tensor, targets: torch.Tensor,
    ) -> dict[str, float]:
        x = features.detach().to(self.device, torch.float32)
        target = targets.detach().to(self.device, torch.float32)
        prediction, logits = self.ema(x)
        confidence = logits.sigmoid() * self.support_probability(x)[None, :, None]
        correct = (prediction <= 0.0) == (target[None] <= 0.0)
        mean = prediction.mean(dim=0)
        mean_confidence = confidence.mean(dim=0)
        mean_correct = (mean <= 0.0) == (target <= 0.0)
        pred_safe = mean.amax(dim=-1) <= 0.0
        true_safe = target.amax(dim=-1) <= 0.0
        accepted = pred_safe & (mean_confidence.amin(dim=-1) >= 0.5)
        stats = {
            "mae_normalized": float((mean - target).abs().mean()),
            "correct_rate": float(correct.float().mean()),
            "confidence_mean": float(confidence.mean()),
            "confidence_brier": float(
                (confidence - correct.float()).square().mean()
            ),
            "confidence_ece": float(expected_calibration_error(
                mean_confidence, mean_correct,
            )),
            "true_feasible": float(true_safe.float().mean()),
            "pred_feasible": float(pred_safe.float().mean()),
            "false_feasible": float((pred_safe & ~true_safe).float().mean()),
            "confident_coverage": float(accepted.float().mean()),
            "confident_false_feasible": float(
                (accepted & ~true_safe).float().mean()
            ),
            "support_mean": float(self.support_probability(x).mean()),
        }
        physical_error = (
            robust_inverse(mean, self.cfg.scales)
            - robust_inverse(target, self.cfg.scales)
        ).abs().mean(dim=0)
        for name, value in zip(VIOLATION_NAMES, physical_error):
            stats[f"mae_{name}"] = float(value)
        return stats

    def observe_targets(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        actor_step: int,
    ) -> dict[str, float]:
        x = features.detach().to(self.device, torch.float32)
        target = targets.detach().to(self.device, torch.float32)
        with torch.no_grad():
            prequential = self.prequential_metrics(x, target)
        calibration_head = torch.randint(
            0, int(self.cfg.num_heads), (x.shape[0],), generator=self.generator,
        )
        self.recent.add(
            features, targets, calibration_head, actor_step=actor_step,
        )
        self.rows_observed += int(x.shape[0])
        seed_anchors_after_update = False
        with torch.no_grad():
            if self.anchors is not None:
                if (
                    len(self.anchors) == 0
                    and self.rows_observed >= int(self.cfg.anchor_warmup_rows)
                ):
                    # Finish the warmup-crossing batch as ordinary base
                    # training before snapshotting its updated EMA.  This is
                    # the same train-then-seed order used by the selected arm.
                    seed_anchors_after_update = True
                elif len(self.anchors) > 0:
                    anchor_reference, _ = self.ema(x)
                    self.anchors.offer(
                        features, targets, anchor_reference,
                        scale=self._support_scale(x).cpu(),
                        generator=self.anchor_generator,
                    )
        stats = prequential
        for _ in range(int(self.cfg.updates_per_observation)):
            stats = {**prequential, **self._update_once()}
        if seed_anchors_after_update:
            count = len(self.recent)
            self.seed_anchors(
                self.recent.features[:count].float(),
                self.recent.targets[:count].float(),
            )
        return stats

    @torch.no_grad()
    def seed_anchors(
        self, features: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        """Initialize protected anchors from an existing labeled stream."""
        if self.anchors is None:
            raise RuntimeError("anchor memory is disabled")
        x = features.detach().to(self.device, torch.float32)
        reference, _ = self.ema(x)
        self.anchors.seed(features, targets, reference)

    @torch.no_grad()
    def enable_anchors(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        capacity: int = 2048,
        slow_fraction: float = 0.20,
        half_life_rows: int = 55_296,
        slow_half_life_multiplier: float = 5.0,
        match_radius: float = 0.20,
        replacement: bool = True,
        batch_size: int = 128,
        exact_weight: float = 0.25,
        inertia_weight: float = 1.0,
        drift_tolerance: float = 0.05,
    ) -> None:
        """Enable and seed functional anchors after base critic training."""
        self.cfg.anchor_capacity = int(capacity)
        self.cfg.anchor_slow_fraction = float(slow_fraction)
        self.cfg.anchor_half_life_rows = int(half_life_rows)
        self.cfg.anchor_slow_half_life_multiplier = float(
            slow_half_life_multiplier
        )
        self.cfg.anchor_match_radius = float(match_radius)
        self.cfg.anchor_replacement = bool(replacement)
        self.cfg.anchor_batch_size = int(batch_size)
        self.cfg.anchor_exact_weight = float(exact_weight)
        self.cfg.anchor_inertia_weight = float(inertia_weight)
        self.cfg.anchor_drift_tolerance = float(drift_tolerance)
        self.anchors = ElasticAnchorMemory(
            capacity, self.input_dim, self.cfg.num_heads,
            slow_fraction=slow_fraction,
            half_life_rows=half_life_rows,
            slow_half_life_multiplier=slow_half_life_multiplier,
            match_radius=match_radius,
            replacement=replacement,
        )
        self.seed_anchors(features, targets)

    def observe_gate(
        self,
        features: torch.Tensor,
        gate: torch.Tensor,
        *,
        actor_step: int,
    ) -> dict[str, float]:
        residual = residuals_from_gate(gate, self.regression_cfg)
        target = robust_transform(residual, self.cfg.scales)
        return self.observe_targets(features, target, actor_step=actor_step)

    def _update_once(self) -> dict[str, float]:
        x, target, calibration_head = self.recent.sample(
            self.cfg.batch_size, generator=self.generator, device=self.device,
        )
        prediction, confidence_logits, expert_routing_loss = self.live(
            x, return_aux=True,
        )
        per_row, regression, boundary = critic_loss_per_head_row(
            prediction, target, self.regression_cfg,
        )
        confidence_target = (
            prediction.detach() <= 0.0
        ) == (target[None] <= 0.0)
        confidence_element = F.binary_cross_entropy_with_logits(
            confidence_logits, confidence_target.float(), reduction="none",
        )
        confidence_row = confidence_element.mean(dim=-1)
        bootstrap = torch.rand(
            per_row.shape, generator=self.generator,
        ).to(self.device) < float(self.cfg.bootstrap_probability)
        head_index = torch.arange(
            int(self.cfg.num_heads), device=self.device,
        )[:, None]
        calibration_mask = calibration_head[None, :] == head_index
        mask = bootstrap & ~calibration_mask
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            fallback = (~calibration_mask[empty]).float().argmax(dim=1)
            mask[empty, fallback] = True
        denominator = mask.sum().clamp_min(1)
        regression_loss = (per_row * mask).sum() / denominator
        confidence_denominator = calibration_mask.sum().clamp_min(1)
        confidence_loss = (
            confidence_row * calibration_mask
        ).sum() / confidence_denominator
        loss = (
            regression_loss
            + float(self.cfg.confidence_loss_weight) * confidence_loss
            + expert_routing_loss
        )

        background_loss = x.new_zeros(())
        background_n = int(round(
            float(x.shape[0]) * float(self.cfg.background_fraction)
        ))
        if background_n > 0 and self.input_dim == self.global_sampler.sample(
            1, generator=self.generator,
        ).shape[1]:
            background = self.global_sampler.sample(
                background_n, generator=self.generator,
            ).to(self.device)
            _, background_logits = self.live(background)
            background_loss = F.binary_cross_entropy_with_logits(
                background_logits, torch.zeros_like(background_logits),
            )
            loss = loss + float(
                self.cfg.background_confidence_weight
            ) * background_loss

        inertia_loss = x.new_zeros(())
        if (
            self.updates >= int(self.cfg.inertia_warmup_updates)
            and float(self.cfg.inertia_weight) > 0.0
        ):
            with torch.no_grad():
                reference_prediction, reference_logits = self.reference(x)
                importance = reference_logits.sigmoid().mean(dim=-1)
            inertia_per_row = (
                prediction - reference_prediction
            ).square().mean(dim=-1)
            inertia_loss = (
                inertia_per_row * importance
            ).sum() / importance.sum().clamp_min(1.0)
            loss = loss + float(self.cfg.inertia_weight) * inertia_loss

        anchor_exact_loss = x.new_zeros(())
        anchor_confidence_loss = x.new_zeros(())
        anchor_drift_loss = x.new_zeros(())
        if self.anchors is not None and len(self.anchors):
            anchor_x, anchor_target, anchor_reference, anchor_importance = (
                self.anchors.sample(
                    self.cfg.anchor_batch_size,
                    generator=self.anchor_generator, device=self.device,
                )
            )
            anchor_prediction, anchor_logits = self.live(anchor_x)
            anchor_per_row, _anchor_regression, _anchor_boundary = (
                critic_loss_per_head_row(
                    anchor_prediction, anchor_target, self.regression_cfg,
                )
            )
            # Keep the normalization fixed so age decay reduces an anchor's
            # absolute training force.  Dividing by importance.sum() would
            # only reweight anchors relative to one another and would leave a
            # uniformly old bank just as strong as a fresh one.
            anchor_exact_loss = (
                anchor_per_row * anchor_importance[None, :]
            ).mean()
            anchor_correct = (
                anchor_prediction.detach() <= 0.0
            ) == (anchor_target[None] <= 0.0)
            anchor_confidence_row = F.binary_cross_entropy_with_logits(
                anchor_logits, anchor_correct.float(), reduction="none",
            ).mean(dim=-1)
            anchor_confidence_loss = (
                anchor_confidence_row * anchor_importance[None, :]
            ).mean()
            anchor_drift = F.relu(
                (anchor_prediction - anchor_reference).abs()
                - float(self.cfg.anchor_drift_tolerance)
            ).square().mean(dim=-1).mean(dim=0)
            anchor_drift_loss = (anchor_drift * anchor_importance).mean()
            loss = loss + float(self.cfg.anchor_exact_weight) * (
                anchor_exact_loss
                + float(self.cfg.confidence_loss_weight) * anchor_confidence_loss
            )
            loss = loss + float(
                self.cfg.anchor_inertia_weight
            ) * anchor_drift_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        self._update_ema()
        if self.updates % int(self.cfg.reference_refresh_updates) == 0:
            self.reference.load_state_dict(self.ema.state_dict())

        stats = {
            "loss": float(loss.detach()),
            "loss_regression": float(regression_loss.detach()),
            "loss_confidence": float(confidence_loss.detach()),
            "loss_background": float(background_loss.detach()),
            "loss_inertia": float(inertia_loss.detach()),
            "loss_expert_routing": float(expert_routing_loss.detach()),
            "loss_anchor_exact": float(anchor_exact_loss.detach()),
            "loss_anchor_confidence": float(anchor_confidence_loss.detach()),
            "loss_anchor_drift": float(anchor_drift_loss.detach()),
            "train_correct_target": float(
                confidence_target[calibration_mask].float().mean()
            ),
            "recent_size": float(len(self.recent)),
            "updates": float(self.updates),
            "loss_regression_raw": float(
                (regression * mask).sum() / denominator
            ),
            "loss_boundary": float((boundary * mask).sum() / denominator),
        }
        for key, value in stats.items():
            self._metric_sum[key] = self._metric_sum.get(key, 0.0) + value
        self._metric_n += 1
        return stats

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = float(self.cfg.ema_decay)
        for ema, live in zip(self.ema.parameters(), self.live.parameters()):
            ema.mul_(decay).add_(live, alpha=1.0 - decay)

    def metrics(self, *, reset: bool = False) -> dict[str, float]:
        n = max(self._metric_n, 1)
        result = {key: value / n for key, value in self._metric_sum.items()}
        result.update(updates=float(self.updates), recent_size=float(len(self.recent)))
        if reset:
            self._metric_sum.clear()
            self._metric_n = 0
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": MONOTONIC_CRITIC_FORMAT_VERSION,
            "input_dim": self.input_dim,
            "config": asdict(self.cfg),
            "live": self.live.state_dict(), "ema": self.ema.state_dict(),
            "reference": self.reference.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "recent": self.recent.state_dict(),
            "anchors": None if self.anchors is None else self.anchors.state_dict(),
            "generator_state": self.generator.get_state(),
            "anchor_generator_state": self.anchor_generator.get_state(),
            "updates": self.updates,
            "rows_observed": self.rows_observed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != MONOTONIC_CRITIC_FORMAT_VERSION:
            raise ValueError("unsupported monotonic critic checkpoint")
        if int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError("monotonic critic input width changed")
        self.live.load_state_dict(state["live"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.reference.load_state_dict(state["reference"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        self.recent.load_state_dict(state["recent"])
        if self.anchors is not None and state.get("anchors") is not None:
            self.anchors.load_state_dict(state["anchors"])
        self.generator.set_state(state["generator_state"].cpu())
        if "anchor_generator_state" in state:
            self.anchor_generator.set_state(state["anchor_generator_state"].cpu())
        self.updates = int(state.get("updates", 0))
        if "rows_observed" in state:
            self.rows_observed = int(state["rows_observed"])
        elif self.anchors is not None and len(self.anchors):
            self.rows_observed = int(self.anchors.rows_seen)
        else:
            self.rows_observed = len(self.recent)
