"""Distributional violation heads used by OOD/frontier ablations.

The classes here intentionally do not participate in actor training yet.  They
provide a compact, testable comparison between ordinary categorical logits and
IsoMax-style prototype distances before either representation is promoted into
the production timing critic.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, vmap

from .violation_critic import (
    PriorityAuditCoreset,
    RecentCriticReplay,
    ViolationCriticConfig,
    residuals_from_gate,
    robust_transform,
)
from .critic_task_sampler import GlobalCriticTaskSampler


N_PHYSICAL_CONSTRAINTS = 4
N_DISTRIBUTIONS = 5
N_BINS = 10
# Ten intervals with zero as an exact feasibility boundary.  These operate in
# the critic's signed asinh-normalized residual space.
BIN_EDGES = (
    -math.inf, -2.0, -1.0, -0.5, -0.2, 0.0,
    0.2, 0.5, 1.0, 2.0, math.inf,
)
MANAGEABLE_BINS = 5


def calibrated_risk_threshold(
    risk: torch.Tensor,
    truth: torch.Tensor,
    *,
    min_precision: float,
) -> tuple[float, float, float] | None:
    """Select the widest risk threshold satisfying a precision constraint.

    The sweep remains on the input device and only transfers the winning
    threshold/metrics.  Tied risks are atomic because a threshold necessarily
    includes every row with that value.
    """
    if risk.ndim != 1 or truth.ndim != 1 or risk.numel() != truth.numel():
        raise ValueError("risk and truth must be equally-sized vectors")
    if risk.numel() == 0:
        return None
    order = torch.argsort(risk, stable=True)
    risk_sorted = risk.index_select(0, order)
    truth_sorted = truth.bool().index_select(0, order)
    count_all = torch.arange(
        1, risk_sorted.numel() + 1,
        device=risk.device, dtype=torch.long,
    )
    true_all = truth_sorted.long().cumsum(dim=0)
    distinct_end = torch.ones_like(truth_sorted, dtype=torch.bool)
    if risk_sorted.numel() > 1:
        distinct_end[:-1] = risk_sorted[:-1] != risk_sorted[1:]
    candidate_count = count_all[distinct_end]
    candidate_true = true_all[distinct_end]
    candidate_risk = risk_sorted[distinct_end]
    candidate_precision = (
        candidate_true.float() / candidate_count.float().clamp_min(1.0)
    )
    truth_total = truth_sorted.long().sum().clamp_min(1)
    candidate_recall = candidate_true.float() / truth_total.float()
    precision_ok = (
        candidate_precision + 1.0e-12 >= float(min_precision)
    )
    if not bool(precision_ok.any()):
        return None
    # Original lexicographic tie-break: recall, count, threshold.  Candidate
    # count and threshold are monotone, so the final maximum-TP entry wins.
    best_true = candidate_true[precision_ok].max()
    winner = precision_ok & (candidate_true == best_true)
    best_i = winner.nonzero(as_tuple=True)[0][-1]
    return (
        float(candidate_risk[best_i]),
        float(candidate_precision[best_i]),
        float(candidate_recall[best_i]),
    )


def bin_representatives(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Finite ordered representatives for the categorical residual bins."""
    # The open tails extend the width of their nearest finite interval.  These
    # values are decision utilities, not reconstructions of physical residuals.
    values = (-3.0, -1.5, -0.75, -0.35, -0.1, 0.1, 0.35, 0.75, 1.5, 3.0)
    return torch.tensor(values, device=device, dtype=dtype)


def distributional_risk(
    member_probability: torch.Tensor,
    *,
    acceptable_limit: float = 0.2,
    severity_linear: float = 0.25,
    severity_quadratic: float = 0.10,
    tail_alpha: float = 0.90,
    tail_weight: float = 0.25,
    epistemic_beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decision risk from the full worst-violation distribution.

    Risk combines expected threshold exceedance/severity, categorical upper-
    tail CVaR, and between-head variation in expected consequence.  Unlike
    entropy, it distinguishes central, bimodal, and uniform predictions by the
    consequences relevant to collocation admission.
    """
    if member_probability.ndim != 4 or member_probability.shape[-2:] != (
        N_DISTRIBUTIONS, N_BINS,
    ):
        raise ValueError("member_probability must have shape (heads, batch, 5, 10)")
    if not 0.0 <= float(tail_alpha) < 1.0:
        raise ValueError("tail_alpha must be in [0, 1)")
    if min(
        float(severity_linear), float(severity_quadratic),
        float(tail_weight), float(epistemic_beta),
    ) < 0.0:
        raise ValueError("risk weights must be nonnegative")

    worst = member_probability[..., -1, :]
    value = bin_representatives(device=worst.device, dtype=worst.dtype)
    excess = (value - float(acceptable_limit)).clamp_min(0.0)
    consequence = (
        (value > float(acceptable_limit)).to(worst.dtype)
        + float(severity_linear) * excess
        + float(severity_quadratic) * excess.square()
    )
    member_expected = (worst * consequence).sum(dim=-1)
    expected = member_expected.mean(dim=0)
    epistemic = member_expected.std(dim=0, unbiased=False)

    # Exact discrete upper-tail CVaR of the ensemble mixture.  Consequence is
    # ordered by bin, so accumulate from the final bin and fractionally consume
    # the boundary bin.
    mixture = worst.mean(dim=0)
    remaining = mixture.new_full((mixture.shape[0],), 1.0 - float(tail_alpha))
    tail_sum = mixture.new_zeros(mixture.shape[0])
    for k in range(N_BINS - 1, -1, -1):
        take = torch.minimum(mixture[:, k], remaining)
        tail_sum = tail_sum + take * consequence[k]
        remaining = (remaining - take).clamp_min(0.0)
    tail_mass = max(1.0 - float(tail_alpha), 1.0e-6)
    cvar = tail_sum / tail_mass
    total = expected + float(tail_weight) * cvar + float(epistemic_beta) * epistemic
    return total, {
        "expected": expected,
        "cvar": cvar,
        "epistemic": epistemic,
    }


def distributional_targets(target: torch.Tensor) -> torch.Tensor:
    """Append worst residual and discretize to ``(B, 5)`` ordered bins."""
    if target.ndim != 2 or target.shape[1] != N_PHYSICAL_CONSTRAINTS:
        raise ValueError("target must have shape (B, 4)")
    values = torch.cat((target.float(), target.float().amax(dim=1, keepdim=True)), dim=1)
    boundaries = values.new_tensor(BIN_EDGES[1:-1])
    return torch.bucketize(values, boundaries, right=False)


def probabilities(logits: torch.Tensor, temperature: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Temperature-scale ``(..., 5, 10)`` logits and normalize them."""
    if logits.shape[-2:] != (N_DISTRIBUTIONS, N_BINS):
        raise ValueError("logits must end in (5, 10)")
    temp = torch.as_tensor(temperature, device=logits.device, dtype=logits.dtype)
    if temp.ndim == 1:
        if temp.shape != (N_DISTRIBUTIONS,):
            raise ValueError("temperature vector must have shape (5,)")
        temp = temp.view(*([1] * (logits.ndim - 2)), N_DISTRIBUTIONS, 1)
    return (logits / temp.clamp_min(1.0e-4)).softmax(dim=-1)


def normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    """Categorical entropy in ``[0, 1]``, retaining leading dimensions."""
    return -(probability.clamp_min(1.0e-8).log() * probability).sum(dim=-1) / math.log(N_BINS)


def categorical_ordinal_loss(
    logits: torch.Tensor,
    target_bin: torch.Tensor,
    *,
    ordinal_weight: float = 0.25,
) -> torch.Tensor:
    """Proper categorical loss plus an order-aware cumulative penalty."""
    if logits.ndim != 4 or logits.shape[-2:] != (N_DISTRIBUTIONS, N_BINS):
        raise ValueError("logits must have shape (M, B, 5, 10)")
    if target_bin.shape != logits.shape[1:3]:
        raise ValueError("target_bin must have shape (B, 5)")
    truth = target_bin.unsqueeze(0).expand(logits.shape[0], -1, -1)
    ce = F.cross_entropy(
        logits.reshape(-1, N_BINS), truth.reshape(-1), reduction="mean",
    )
    prob = logits.softmax(dim=-1)
    pred_cdf = prob.cumsum(dim=-1)[..., :-1]
    thresholds = torch.arange(N_BINS - 1, device=logits.device)
    true_cdf = (
        thresholds.view(1, 1, 1, -1)
        >= truth.unsqueeze(-1)
    ).to(pred_cdf.dtype)
    ordinal = (pred_cdf - true_cdf).square().mean()
    return ce + float(ordinal_weight) * ordinal


def high_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Cross entropy from a uniform target, minimized by uniform predictions."""
    if logits.ndim != 4 or logits.shape[-2:] != (N_DISTRIBUTIONS, N_BINS):
        raise ValueError("logits must have shape (M, B, 5, 10)")
    return -logits.log_softmax(dim=-1).mean()


class DistributionalHead(nn.Module):
    """One categorical or IsoMax-style violation-distribution head."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        depth: int = 2,
        kind: str = "softmax",
        embedding_dim: int = 64,
    ):
        super().__init__()
        if kind not in {"softmax", "isomax"}:
            raise ValueError(f"unknown head kind {kind!r}")
        self.kind = kind
        layers: list[nn.Module] = []
        width = int(input_dim)
        for _ in range(int(depth)):
            layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        if kind == "softmax":
            self.output = nn.Linear(width, N_DISTRIBUTIONS * N_BINS)
            self.embedding = None
            self.prototypes = None
            self.log_distance_scale = None
        else:
            self.output = None
            self.embedding = nn.Linear(width, embedding_dim)
            self.prototypes = nn.Parameter(torch.empty(
                N_DISTRIBUTIONS, N_BINS, embedding_dim,
            ))
            nn.init.normal_(self.prototypes, std=1.0 / math.sqrt(embedding_dim))
            self.log_distance_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.trunk(x)
        if self.kind == "softmax":
            return self.output(hidden).reshape(-1, N_DISTRIBUTIONS, N_BINS), None
        embedding = F.normalize(self.embedding(hidden), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        distance = (
            embedding[:, None, None, :] - prototypes[None, :, :, :]
        ).square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
        scale = F.softplus(self.log_distance_scale) + 1.0e-4
        return -scale * distance, distance.amin(dim=-1)


class DistributionalEnsemble(nn.Module):
    """Independent distributional heads with optional IsoMax distances."""

    def __init__(
        self,
        input_dim: int,
        *,
        num_heads: int,
        kind: str,
        hidden_dim: int = 128,
        depth: int = 2,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.kind = kind
        self.heads = nn.ModuleList([
            DistributionalHead(
                input_dim, hidden_dim=hidden_dim, depth=depth, kind=kind,
                embedding_dim=embedding_dim,
            )
            for _ in range(self.num_heads)
        ])

    def _softmax_per_head(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate independent heads as one batched program.

        Parameters remain owned by the checkpoint-compatible ``ModuleList``;
        the differentiable stacks are transient and preserve per-head grads.
        """
        if x.ndim != 3 or int(x.shape[0]) != self.num_heads:
            raise ValueError("per-head input must have shape (heads, batch, dim)")
        names = tuple(dict(self.heads[0].named_parameters()))
        parameters = {
            name: torch.stack([
                dict(head.named_parameters())[name] for head in self.heads
            ])
            for name in names
        }

        def call_head(params, features):
            return functional_call(self.heads[0], params, (features,))[0]

        return vmap(call_head)(parameters, x)

    def forward_per_head(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "softmax":
            return self._softmax_per_head(x), None
        rows = [head(x[i]) for i, head in enumerate(self.heads)]
        return (
            torch.stack([row[0] for row in rows], dim=0),
            torch.stack([row[1] for row in rows], dim=0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        expanded = x.unsqueeze(0).expand(self.num_heads, *x.shape)
        return self.forward_per_head(expanded)


class OnlineDistributionalCriticTrainer:
    """Online categorical critic plus anchor-first timing-curriculum state."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        cfg: ViolationCriticConfig,
        *,
        device: torch.device | str,
        seed: int,
    ):
        if cfg.representation != "distributional":
            raise ValueError("online distributional trainer requires distributional config")
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.live = DistributionalEnsemble(
                input_dim, num_heads=cfg.num_heads, kind="softmax",
                hidden_dim=cfg.hidden_dim, depth=cfg.depth,
            ).to(self.device)
        self.ema = copy.deepcopy(self.live).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.live.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.recent = RecentCriticReplay(cfg.recent_capacity, input_dim)
        self.audit = PriorityAuditCoreset(
            cfg.audit_capacity, input_dim,
            boundary_band=cfg.replay_boundary_band,
        )
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        self.global_sampler = GlobalCriticTaskSampler()
        if (
            float(cfg.global_ood_fraction) > 0.0
            and self.global_sampler.sample(
                1, generator=torch.Generator().manual_seed(int(seed) + 2),
            ).shape[1] != self.input_dim
        ):
            raise ValueError("global OOD sampler width does not match critic input")
        self.updates = 0
        self.risk_threshold = float("-inf")
        self.calibration_precision = 0.0
        self.calibration_recall = 0.0
        self.anchor_samples = 0
        self.anchor_close_ema = 0.0
        self.curriculum_open = not bool(cfg.anchor_curriculum)
        self._metric_sum: dict[str, float] = {}
        self._metric_n = 0
        self._calibration_diagnostics: dict[str, float] = {}

    def _risk(self, member_probability: torch.Tensor):
        return distributional_risk(
            member_probability,
            acceptable_limit=self.cfg.acceptable_normalized_violation,
            severity_linear=self.cfg.risk_severity_linear,
            severity_quadratic=self.cfg.risk_severity_quadratic,
            tail_alpha=self.cfg.risk_tail_alpha,
            tail_weight=self.cfg.risk_tail_weight,
            epistemic_beta=self.cfg.risk_epistemic_beta,
        )

    @torch.no_grad()
    def risk_score(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self.ema(features.detach().to(self.device, torch.float32))
        probability = logits.softmax(dim=-1)
        return self._risk(probability)[0]

    @torch.no_grad()
    def manageable(self, features: torch.Tensor) -> torch.Tensor:
        if not self.curriculum_open or not math.isfinite(self.risk_threshold):
            return torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
        return self.risk_score(features) <= float(self.risk_threshold)

    def observe_gate(
        self,
        features: torch.Tensor,
        gate: torch.Tensor,
        *,
        actor_step: int,
        anchor_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        residual = residuals_from_gate(gate, self.cfg)
        target = robust_transform(residual, self.cfg.scales)
        with torch.no_grad():
            logits, _ = self.ema(features.detach().to(self.device, torch.float32))
            probability = logits.softmax(dim=-1)
            risk, _ = self._risk(probability)
            truth = target.to(self.device).amax(dim=1)
            priority = (risk - (truth > self.cfg.acceptable_normalized_violation).float()).abs()
        self.recent.add(features, target, actor_step=actor_step)
        self.audit.add(features, target, priority, actor_step=actor_step)

        if anchor_mask is not None and bool(anchor_mask.any()):
            anchor_truth = target.detach().amax(dim=1)[
                anchor_mask.detach().to(target.device, torch.bool)
            ]
            rate = float(
                (anchor_truth <= self.cfg.acceptable_normalized_violation).float().mean()
            )
            count = int(anchor_truth.numel())
            decay = float(self.cfg.anchor_rate_ema)
            if self.anchor_samples == 0:
                self.anchor_close_ema = rate
            else:
                self.anchor_close_ema = decay * self.anchor_close_ema + (1.0 - decay) * rate
            self.anchor_samples += count

        stats: dict[str, float] = {}
        for _ in range(int(self.cfg.updates_per_step)):
            stats = self._update_once()
        if self.updates % int(self.cfg.risk_recalibrate_every) == 0:
            self._recalibrate()
        if (
            self.cfg.anchor_curriculum
            and not self.curriculum_open
            and self.anchor_samples >= int(self.cfg.anchor_min_samples)
            and self.anchor_close_ema >= float(self.cfg.anchor_min_close_rate)
            and math.isfinite(self.risk_threshold)
        ):
            self.curriculum_open = True
        return stats

    def _update_once(self) -> dict[str, float]:
        features = []
        targets = []
        unknowns = []
        unknown_n = int(round(
            float(self.cfg.batch_size) * float(self.cfg.global_ood_fraction)
        ))
        use_ood = unknown_n > 0 and float(self.cfg.global_ood_weight) > 0.0
        # Keep the legacy per-head sample/OOD draw ordering so fusion changes
        # execution only, not the critic's stochastic training stream.
        for _head in self.live.heads:
            x, target = self.recent.sample(
                self.cfg.batch_size, generator=self.generator, device=self.device,
                manageable_fraction=self.cfg.replay_manageable_fraction,
                near_failure_fraction=self.cfg.replay_near_failure_fraction,
                boundary_band=self.cfg.replay_boundary_band,
            )
            features.append(x)
            targets.append(target)
            if use_ood:
                unknowns.append(self.global_sampler.sample(
                    unknown_n, generator=self.generator,
                ))
        feature_batch = torch.stack(features)
        target_batch = torch.stack(targets)
        logits, _ = self.live.forward_per_head(feature_batch)
        supervised_loss = categorical_ordinal_loss(
            logits.reshape(1, -1, N_DISTRIBUTIONS, N_BINS),
            distributional_targets(
                target_batch.reshape(-1, N_PHYSICAL_CONSTRAINTS)
            ),
        )
        loss = supervised_loss
        train_entropy = normalized_entropy(
            logits.detach().softmax(dim=-1)
        ).mean()
        ood_loss = None
        ood_entropy = None
        if use_ood:
            unknown = torch.stack(unknowns).to(self.device)
            unknown_logits, _ = self.live.forward_per_head(unknown)
            ood_loss = high_entropy_loss(unknown_logits)
            loss = loss + float(self.cfg.global_ood_weight) * ood_loss
            ood_entropy = normalized_entropy(
                unknown_logits.detach().softmax(dim=-1)
            ).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        decay = float(self.cfg.ema_decay)
        with torch.no_grad():
            for ema, live in zip(self.ema.parameters(), self.live.parameters()):
                ema.mul_(decay).add_(live, alpha=1.0 - decay)
        self.updates += 1
        stats = {
            "loss": float(loss.detach()),
            "loss_supervised": float(supervised_loss.detach()),
            "entropy_train": float(train_entropy),
            "recent_size": float(len(self.recent)),
            "anchor_close_ema": float(self.anchor_close_ema),
            "curriculum_open": float(self.curriculum_open),
        }
        if ood_loss is not None and ood_entropy is not None:
            stats["loss_ood"] = float(ood_loss.detach())
            stats["entropy_ood"] = float(ood_entropy)
        for key, value in stats.items():
            self._metric_sum[key] = self._metric_sum.get(key, 0.0) + float(value)
        self._metric_n += 1
        return stats

    @torch.no_grad()
    def _recalibrate(self) -> None:
        n = len(self.recent)
        if n < max(32, int(self.cfg.num_heads) * 4):
            return
        steps = self.recent.actor_step[:n]
        unique = steps.unique(sorted=True)
        keep_steps = max(1, int(math.ceil(0.2 * len(unique))))
        keep = steps >= unique[-keep_steps]
        x = self.recent.features[:n][keep].to(self.device, torch.float32)
        target = self.recent.targets[:n][keep].to(self.device, torch.float32)
        logits, _ = self.ema(x)
        probability = logits.softmax(dim=-1)
        risk, risk_parts = self._risk(probability)
        truth = target.amax(dim=1) <= float(self.cfg.acceptable_normalized_violation)
        best = calibrated_risk_threshold(
            risk,
            truth,
            min_precision=float(self.cfg.risk_min_precision),
        )
        if best is not None:
            (
                self.risk_threshold,
                self.calibration_precision,
                self.calibration_recall,
            ) = best

        mixture_entropy = normalized_entropy(probability.mean(dim=0))
        member_entropy = normalized_entropy(probability).mean(dim=0)
        diagnostic_generator = torch.Generator().manual_seed(31_337 + self.updates)
        ood = self.global_sampler.sample(
            min(256, max(32, int(self.cfg.batch_size))),
            generator=diagnostic_generator,
        ).to(self.device)
        self._calibration_diagnostics = {
            "risk_mean": float(risk.mean()),
            "risk_expected_mean": float(risk_parts["expected"].mean()),
            "risk_cvar_mean": float(risk_parts["cvar"].mean()),
            "risk_epistemic_mean": float(risk_parts["epistemic"].mean()),
            "entropy_ema_train": float(member_entropy.mean()),
            "ensemble_mi_train": float(
                (mixture_entropy - member_entropy).clamp_min(0.0).mean()
            ),
            "true_manageable_rate": float(truth.float().mean()),
            "predicted_manageable_rate": float(
                (risk <= float(self.risk_threshold)).float().mean()
            ),
        }
        if ood.shape[1] == self.input_dim:
            ood_logits, _ = self.ema(ood)
            ood_probability = ood_logits.softmax(dim=-1)
            ood_mixture_entropy = normalized_entropy(ood_probability.mean(dim=0))
            ood_member_entropy = normalized_entropy(ood_probability).mean(dim=0)
            self._calibration_diagnostics.update({
                "entropy_ema_ood": float(ood_member_entropy.mean()),
                "ensemble_mi_ood": float(
                    (ood_mixture_entropy - ood_member_entropy).clamp_min(0.0).mean()
                ),
            })

    def metrics(self, *, reset: bool = False) -> dict[str, float]:
        n = max(self._metric_n, 1)
        out = {key: value / n for key, value in self._metric_sum.items()}
        out.update({
            "updates": float(self.updates),
            "recent_size": float(len(self.recent)),
            "audit_size": float(len(self.audit)),
            "risk_threshold": float(self.risk_threshold),
            "calibration_precision": float(self.calibration_precision),
            "calibration_recall": float(self.calibration_recall),
            "anchor_samples": float(self.anchor_samples),
            "anchor_close_ema": float(self.anchor_close_ema),
            "curriculum_open": float(self.curriculum_open),
        })
        out.update(self._calibration_diagnostics)
        if reset:
            self._metric_sum.clear()
            self._metric_n = 0
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "representation": "distributional",
            "input_dim": self.input_dim,
            "config": asdict(self.cfg),
            "live": self.live.state_dict(), "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "recent": self.recent.state_dict(), "audit": self.audit.state_dict(),
            "generator_state": self.generator.get_state(),
            "updates": self.updates, "risk_threshold": self.risk_threshold,
            "calibration_precision": self.calibration_precision,
            "calibration_recall": self.calibration_recall,
            "anchor_samples": self.anchor_samples,
            "anchor_close_ema": self.anchor_close_ema,
            "curriculum_open": self.curriculum_open,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("representation") != "distributional":
            raise ValueError("not a distributional timing-critic checkpoint")
        if int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError("distributional critic input width changed")
        self.live.load_state_dict(state["live"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        self.recent.load_state_dict(state["recent"])
        self.audit.load_state_dict(state["audit"])
        self.generator.set_state(state["generator_state"].cpu())
        self.updates = int(state.get("updates", 0))
        self.risk_threshold = float(state.get("risk_threshold", float("-inf")))
        self.calibration_precision = float(state.get("calibration_precision", 0.0))
        self.calibration_recall = float(state.get("calibration_recall", 0.0))
        self.anchor_samples = int(state.get("anchor_samples", 0))
        self.anchor_close_ema = float(state.get("anchor_close_ema", 0.0))
        self.curriculum_open = bool(state.get("curriculum_open", False))
