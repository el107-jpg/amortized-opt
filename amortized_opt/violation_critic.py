"""Actor-conditioned violation critic for task-time frontier exploration.

The critic is deliberately separate from the actor optimizer.  It observes the
exact per-row diagnostics already produced by the Phase-2 cost pass and learns
four continuous, positive-means-bad residuals.  Its predictions are proposals,
never feasibility certificates.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


VIOLATION_NAMES: tuple[str, ...] = (
    "terminal",
    "collision",
    "joint",
    "torque",
)
CRITIC_FORMAT_VERSION = 2


@dataclass
class ViolationCriticConfig:
    """Configuration for the task-time violation ensemble.

    Only ``off`` and ``shadow`` affect training in the first rollout.  The
    later modes are reserved names so checkpoints/config files do not need an
    incompatible rename when guarded frontier sampling is enabled.
    """

    mode: str = "off"
    num_heads: int = 5
    hidden_dim: int = 128
    depth: int = 2
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    # The actor changes quickly during the anchor curriculum, so the decision
    # copy must track the gradient-updated critic on that same timescale.
    ema_decay: float = 0.95
    bootstrap_probability: float = 0.8
    dropout_probability: float = 0.0
    mc_dropout_samples: int = 1
    # ``heterogeneous`` cycles through a plain SiLU MLP, a deeper GELU MLP,
    # and a residual SiLU MLP.  All still predict the same four normalized
    # residuals, so their outputs remain directly comparable.
    head_architecture_mix: str = "homogeneous"
    # With exactly three heads, train them on the newest 25%, newest 50%, and
    # full replay respectively.  This is data diversity, not extra storage.
    replay_timescale_diversity: bool = False
    # A separate frozen random MLP is added to each learned head when this is
    # positive.  Learned predictions cancel the prior around observed data;
    # away from support, the distinct priors preserve epistemic disagreement.
    randomized_prior_scale: float = 0.0
    batch_size: int = 256
    # Four is the first online increase justified by the frozen-replay sweep.
    # Effective reuse depends on actor-label influx: with critic batch 256 it
    # is about 12.8 presentations/label for 80 new labels per outer step, 3.2
    # for 320, and 1.6 for 640.  No actor or physics forwards are repeated.
    updates_per_step: int = 4
    recent_capacity: int = 8192
    audit_capacity: int = 1024
    huber_delta: float = 1.0
    # Regression preserves violation magnitude.  The auxiliary sign term makes
    # the zero crossing useful at finite data/capacity, while the asymmetric
    # multiplier discourages unsafe underestimation without forcing every
    # prediction upward through a hard output bias.
    boundary_loss_weight: float = 0.25
    boundary_temperature: float = 0.50
    boundary_target_band: float = 0.50
    underprediction_multiplier: float = 2.0
    conservative_quantile: float = 0.8
    # Calibrated in robust normalized residual space.  Zero preserves the
    # nominal physical violation boundary; negative values move proposals
    # backward into the predicted feasible region.  This is checkpoint/data
    # specific and therefore deliberately defaults to the uncalibrated rule.
    decision_threshold: float = 0.0
    # Critic minibatches reserve mass for manageable and near-outside rows so
    # the much larger far-failure population cannot teach an all-failure prior.
    replay_manageable_fraction: float = 0.40
    replay_near_failure_fraction: float = 0.30
    replay_boundary_band: float = 0.50

    # Guarded assist sampler.  It remains dormant until the replay has enough
    # exact shadow labels, and exact verification can still reject every row.
    assist_fraction: float = 0.05
    min_assist_samples: int = 2048
    coarse_step_knots: int = 5
    collar_inside_probability: float = 0.45
    collar_boundary_probability: float = 0.45
    collar_outside_probability: float = 0.10

    # Anchor-first primary actor mixture.  At batch 512, 40 independent base
    # tasks are each expanded over the two-dimensional timing table into two
    # conservative frontier points and four component-wise-looser interior
    # points (240 rows).  The remaining 272 rows use the exact-confirmed MPC
    # generation produced by the preceding outer step.
    primary_base_tasks_per_512: int = 40
    primary_boundary_per_task: int = 2
    primary_interior_per_task: int = 4
    # Exact physical ingress limits for the previous-generation MPC ring.
    # Joint and torque retain the ordinary close-feasibility rule below.
    primary_mpc_terminal_limit: float = 0.040
    primary_mpc_collision_sdf_limit: float = -0.010

    # Manageability thresholds in the physical units carried by ``gate``.
    terminal_limit: float = 0.020
    collision_sdf_limit: float = -0.005
    joint_limit: float = 0.020
    torque_limit: float = 8.0

    # Robust-transform scales, in the same physical units.
    terminal_scale: float = 0.020
    collision_scale: float = 0.010
    joint_scale: float = 0.020
    torque_scale: float = 8.0

    # Distributional-risk timing curriculum.  The regression representation
    # remains the checkpoint-compatible default.  Distributional mode starts
    # every fresh task at the horizon anchor and opens shorter-time proposals
    # only after exact anchor labels demonstrate actor proficiency.
    representation: str = "regression"
    acceptable_normalized_violation: float = 0.2
    risk_severity_linear: float = 0.25
    risk_severity_quadratic: float = 0.10
    risk_tail_alpha: float = 0.90
    risk_tail_weight: float = 0.25
    risk_epistemic_beta: float = 1.0
    global_ood_fraction: float = 4.0
    global_ood_weight: float = 0.0025
    risk_min_precision: float = 0.50
    risk_recalibrate_every: int = 25
    anchor_curriculum: bool = False
    anchor_min_samples: int = 2048
    anchor_min_close_rate: float = 0.50
    anchor_rate_ema: float = 0.95

    def __post_init__(self) -> None:
        if self.representation not in {"regression", "distributional"}:
            raise ValueError("invalid timing critic representation")
        if self.mode not in {"off", "shadow", "assist", "control"}:
            raise ValueError(f"invalid timing critic mode {self.mode!r}")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim < 1 or self.depth < 1:
            raise ValueError("hidden_dim and depth must be positive")
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid critic optimizer settings")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 < self.bootstrap_probability <= 1.0:
            raise ValueError("bootstrap_probability must be in (0, 1]")
        if not 0.0 <= float(self.dropout_probability) < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        if self.mc_dropout_samples < 1:
            raise ValueError("mc_dropout_samples must be positive")
        if self.head_architecture_mix not in {"homogeneous", "heterogeneous"}:
            raise ValueError("invalid critic head_architecture_mix")
        if self.replay_timescale_diversity and self.num_heads != 3:
            raise ValueError("replay_timescale_diversity requires exactly three heads")
        if (
            self.num_heads == 1
            and (
                self.dropout_probability <= 0.0
                or self.mc_dropout_samples < 2
            )
        ):
            raise ValueError(
                "one critic head requires dropout and at least two MC samples"
            )
        if (
            not math.isfinite(float(self.randomized_prior_scale))
            or self.randomized_prior_scale < 0.0
        ):
            raise ValueError("randomized_prior_scale must be finite and nonnegative")
        if self.batch_size < 1 or self.updates_per_step < 1:
            raise ValueError("critic batch/update counts must be positive")
        if self.recent_capacity < 1 or self.audit_capacity < 0:
            raise ValueError("invalid critic replay capacities")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if (
            not math.isfinite(float(self.boundary_loss_weight))
            or self.boundary_loss_weight < 0.0
            or not math.isfinite(float(self.boundary_temperature))
            or self.boundary_temperature <= 0.0
            or not math.isfinite(float(self.boundary_target_band))
            or self.boundary_target_band <= 0.0
            or not math.isfinite(float(self.underprediction_multiplier))
            or self.underprediction_multiplier < 1.0
        ):
            raise ValueError("invalid critic boundary-loss configuration")
        replay_fracs = (
            float(self.replay_manageable_fraction),
            float(self.replay_near_failure_fraction),
        )
        if (
            any(x < 0.0 for x in replay_fracs)
            or sum(replay_fracs) > 1.0 + 1.0e-9
            or not math.isfinite(float(self.replay_boundary_band))
            or float(self.replay_boundary_band) <= 0.0
        ):
            raise ValueError("invalid critic replay strata configuration")
        if not 0.5 <= self.conservative_quantile <= 1.0:
            raise ValueError("conservative_quantile must be in [0.5, 1]")
        if self.representation == "regression" and (
            not math.isfinite(float(self.decision_threshold))
            or float(self.decision_threshold) > 0.0
        ):
            raise ValueError("decision_threshold must be finite and nonpositive")
        if not 0.0 <= self.assist_fraction <= 1.0:
            raise ValueError("assist_fraction must be in [0, 1]")
        if self.primary_base_tasks_per_512 < 1:
            raise ValueError("primary_base_tasks_per_512 must be positive")
        if self.primary_boundary_per_task < 1:
            raise ValueError("primary_boundary_per_task must be positive")
        if self.primary_interior_per_task < 1:
            raise ValueError("primary_interior_per_task must be positive")
        if self.primary_mpc_terminal_limit < 0.0:
            raise ValueError("primary_mpc_terminal_limit must be nonnegative")
        if not math.isfinite(float(self.primary_mpc_collision_sdf_limit)):
            raise ValueError("primary_mpc_collision_sdf_limit must be finite")
        if self.min_assist_samples < 1 or self.coarse_step_knots < 1:
            raise ValueError("assist sample threshold and coarse step must be positive")
        collar = (
            self.collar_inside_probability,
            self.collar_boundary_probability,
            self.collar_outside_probability,
        )
        if any(x < 0.0 for x in collar) or not math.isclose(
            sum(collar), 1.0, rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError(f"collar probabilities must be nonnegative and sum to 1: {collar}")
        if any(not math.isfinite(x) or x <= 0.0 for x in self.scales):
            raise ValueError(f"critic target scales must be finite and positive: {self.scales}")
        if not math.isfinite(float(self.acceptable_normalized_violation)):
            raise ValueError("acceptable_normalized_violation must be finite")
        if not 0.0 < float(self.risk_min_precision) <= 1.0:
            raise ValueError("risk_min_precision must be in (0, 1]")
        if self.global_ood_fraction < 0.0 or self.global_ood_weight < 0.0:
            raise ValueError("global OOD fraction/weight must be nonnegative")
        if self.risk_recalibrate_every < 1 or self.anchor_min_samples < 1:
            raise ValueError("risk/anchor sample intervals must be positive")
        if not 0.0 <= float(self.anchor_min_close_rate) <= 1.0:
            raise ValueError("anchor_min_close_rate must be in [0, 1]")
        if not 0.0 <= float(self.anchor_rate_ema) < 1.0:
            raise ValueError("anchor_rate_ema must be in [0, 1)")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def scales(self) -> tuple[float, float, float, float]:
        return (
            float(self.terminal_scale),
            float(self.collision_scale),
            float(self.joint_scale),
            float(self.torque_scale),
        )


def residuals_from_gate(
    gate: torch.Tensor,
    cfg: ViolationCriticConfig,
) -> torch.Tensor:
    """Convert the four exact gate diagnostics to positive-means-bad values.

    ``gate[:, 4:8]`` is terminal distance, minimum collision SDF, joint-box
    excess, and available-torque excess.  The collision sign is reversed
    because larger SDF is safer.
    """
    if gate.ndim != 2 or gate.shape[1] < 8:
        raise ValueError(f"critic gate requires shape (B, >=8), got {tuple(gate.shape)}")
    exact = gate[:, 4:8].detach().float()
    limits = exact.new_tensor(
        [cfg.terminal_limit, cfg.collision_sdf_limit, cfg.joint_limit, cfg.torque_limit]
    )
    signs = exact.new_tensor([1.0, -1.0, 1.0, 1.0])
    # Collision is limit - value; all other residuals are value - limit.
    return (exact - limits) * signs


def robust_transform(values: torch.Tensor, scales) -> torch.Tensor:
    """Reversible signed compression used for critic regression targets."""
    scale = torch.as_tensor(scales, device=values.device, dtype=values.dtype)
    return torch.asinh(values / scale)


def robust_inverse(values: torch.Tensor, scales) -> torch.Tensor:
    """Inverse of :func:`robust_transform`, returning physical residuals."""
    scale = torch.as_tensor(scales, device=values.device, dtype=values.dtype)
    return torch.sinh(values) * scale


def conservative_score_from_prediction(
    prediction: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    """Reduce per-head constraint residuals to one conservative row score.

    Inputs and outputs remain in robust normalized residual space.  A row is
    predicted manageable when its score is at or below a calibrated threshold.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != len(VIOLATION_NAMES):
        raise ValueError("critic prediction must have shape (heads, batch, 4)")
    if not 0.5 <= float(quantile) <= 1.0:
        raise ValueError("critic score quantile must be in [0.5, 1]")
    return torch.quantile(
        prediction.amax(dim=-1).float(), float(quantile), dim=0,
    )


def manageability_metrics(
    score: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    """Measure a scalar manageability rule against exact residual targets."""
    if score.ndim != 1 or target.ndim != 2 or target.shape != (score.numel(), 4):
        raise ValueError("score/target shapes must be (batch,) and (batch, 4)")
    truth = target.float().amax(dim=-1) <= 0.0
    guess = score.float() <= float(threshold)
    tp = int((truth & guess).sum())
    fp = int(((~truth) & guess).sum())
    fn = int((truth & (~guess)).sum())
    tn = int(((~truth) & (~guess)).sum())
    n = max(int(score.numel()), 1)
    predicted = tp + fp
    positives = tp + fn
    negatives = fp + tn
    return {
        "n": float(score.numel()),
        "threshold": float(threshold),
        "safety_margin": float(max(0.0, -float(threshold))),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "true_manageable_rate": float(positives / n),
        "pred_manageable_rate": float(predicted / n),
        "manageable_recall": float(tp / max(positives, 1)),
        "manageable_precision": float(tp / max(predicted, 1)),
        # Overall rate is the metric historically logged by this critic.
        "false_manageable_rate": float(fp / n),
        # These condition on the true and predicted populations respectively.
        "false_positive_rate": float(fp / max(negatives, 1)),
        "false_discovery_rate": float(fp / max(predicted, 1)),
    }


def calibrate_manageability_rule(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    max_false_manageable_rate: float = 0.02,
) -> dict[str, Any]:
    """Choose the best ensemble quantile and conservative score threshold.

    The threshold is constrained to be no greater than zero, so calibration
    can only add pessimism to the nominal residual boundary.  Among rules that
    fit the empirical false-manageable budget, choose maximum feasible recall,
    then precision, then the more conservative quantile.
    """
    if target.ndim != 2 or target.shape[-1] != len(VIOLATION_NAMES):
        raise ValueError("critic target must have shape (batch, 4)")
    if prediction.shape[1:] != target.shape:
        raise ValueError("critic prediction and target batches must match")
    rate = float(max_false_manageable_rate)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("max_false_manageable_rate must be in [0, 1]")
    if not quantiles:
        raise ValueError("at least one calibration quantile is required")

    candidates: list[dict[str, Any]] = []
    for quantile in quantiles:
        score = conservative_score_from_prediction(
            prediction, quantile=float(quantile),
        )
        finite = score[torch.isfinite(score) & (score <= 0.0)].unique().sort().values
        # A threshold just below the smallest score supplies the valid
        # zero-acceptance fallback when even that row exceeds the error budget.
        floor = min(
            float(score[torch.isfinite(score)].min()) - 1.0e-6,
            -1.0e-6,
        )
        thresholds = torch.cat((score.new_tensor([floor]), finite))
        best_threshold = float(thresholds[0])
        for value in thresholds:
            metrics = manageability_metrics(
                score, target, threshold=float(value),
            )
            if metrics["false_manageable_rate"] <= rate + 1.0e-12:
                best_threshold = float(value)
        metrics = manageability_metrics(score, target, threshold=best_threshold)
        candidates.append({"quantile": float(quantile), **metrics})

    selected = max(
        candidates,
        key=lambda item: (
            item["manageable_recall"],
            item["manageable_precision"],
            item["quantile"],
        ),
    )
    return {
        "max_false_manageable_rate": rate,
        "selected": selected,
        "candidates": candidates,
    }


def critic_loss_per_head_row(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: ViolationCriticConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, regression, and near-boundary sign loss per head/row.

    ``prediction`` is ``(M, B, 4)`` and ``target`` is ``(B, 4)`` in signed
    robust-transform space.  Positive means violated.  The same scalar output
    therefore serves as both a continuous estimate and a sign logit; there is
    no second head whose feasibility answer can contradict its regression.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != len(VIOLATION_NAMES):
        raise ValueError("critic prediction must have shape (heads, batch, 4)")
    if target.shape != prediction.shape[1:]:
        raise ValueError("critic target must have shape (batch, 4)")
    truth = target.unsqueeze(0).expand_as(prediction)
    regression_element = F.smooth_l1_loss(
        prediction,
        truth,
        beta=float(cfg.huber_delta),
        reduction="none",
    )
    # A too-small signed residual is the unsafe direction.  Detach the branch
    # selection: this is a cost multiplier, not another learned target.
    under = (prediction.detach() < truth).to(regression_element.dtype)
    regression_element = regression_element * (
        1.0 + under * (float(cfg.underprediction_multiplier) - 1.0)
    )
    regression = regression_element.mean(dim=-1)

    sign_target = (truth > 0.0).to(prediction.dtype)
    boundary_weight = torch.exp(
        -truth.abs() / float(cfg.boundary_target_band)
    )
    boundary = (
        F.binary_cross_entropy_with_logits(
            prediction / float(cfg.boundary_temperature),
            sign_target,
            reduction="none",
        )
        * boundary_weight
    ).mean(dim=-1)
    total = regression + float(cfg.boundary_loss_weight) * boundary
    return total, regression, boundary


def critic_feature_dim(n_dof: int) -> int:
    """Width of the minimal physical task description.

    Features are only ``q0``, ``qd0``, two ordered Cartesian drum goals, and
    the atomic two-hand time-to-hit pair.
    """
    return 2 * int(n_dof) + 8


@torch.no_grad()
def build_critic_features(
    *,
    q0: torch.Tensor,
    qd0: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_tth: torch.Tensor,
) -> torch.Tensor:
    """Build the minimal, detached critic feature vector.

    Goal positions are raw Cartesian metres in ordered left/right slots.  The
    critic MLP performs their encoding; using actor embeddings would make the
    critic input drift as the actor changes.  A zero TTH marks an inactive
    hand, whose goal is explicitly zeroed to prevent stale context leakage.
    """
    if q0.ndim != 2 or qd0.shape != q0.shape:
        raise ValueError("q0 and qd0 must have matching shape (B, N)")
    b = int(q0.shape[0])
    if goal_pos.shape != (b, 2, 3):
        raise ValueError("goal_pos must have shape (B, 2, 3)")
    if goal_tth.shape != (b, 2):
        raise ValueError("goal_tth must have shape (B, 2)")
    tth = goal_tth.detach().float().clamp_min(0.0)
    scheduled = tth > 0.0
    goals = goal_pos.detach().float() * scheduled.unsqueeze(-1)
    out = torch.cat(
        [
            q0.detach().float() / math.pi,
            qd0.detach().float() / 10.0,
            goals.reshape(b, 6),
            tth,
        ],
        dim=1,
    )
    expected = critic_feature_dim(q0.shape[1])
    if out.shape[1] != expected:
        raise RuntimeError(f"critic feature width {out.shape[1]} != expected {expected}")
    return out.detach()


def _activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"invalid critic activation {name!r}")


class _ResidualCriticNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, blocks: int,
        dropout_probability: float, activation: str,
    ):
        super().__init__()
        stem: list[nn.Module] = [
            nn.Linear(int(input_dim), int(hidden_dim)), _activation(activation),
        ]
        if float(dropout_probability) > 0.0:
            stem.append(nn.Dropout(float(dropout_probability)))
        self.stem = nn.Sequential(*stem)
        residual = []
        for _ in range(int(blocks)):
            layers: list[nn.Module] = [
                nn.Linear(int(hidden_dim), int(hidden_dim)), _activation(activation),
            ]
            if float(dropout_probability) > 0.0:
                layers.append(nn.Dropout(float(dropout_probability)))
            residual.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(residual)
        self.output = nn.Linear(int(hidden_dim), len(VIOLATION_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            # Preserve a comparable activation scale as residual depth changes.
            x = (x + block(x)) / math.sqrt(2.0)
        return self.output(x)


class _CriticHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, depth: int,
        dropout_probability: float = 0.0,
        *, activation: str = "silu", residual: bool = False,
    ):
        super().__init__()
        self._spec = (
            int(input_dim), int(hidden_dim), int(depth),
            float(dropout_probability), str(activation), bool(residual),
        )
        self.net = self._new_net()
        self.prior: nn.Module | None = None
        self.prior_scale = 0.0

    def _new_net(self) -> nn.Module:
        input_dim, hidden_dim, depth, dropout, activation, residual = self._spec
        if residual:
            return _ResidualCriticNet(
                input_dim, hidden_dim, depth, dropout, activation,
            )
        layers: list[nn.Module] = []
        width = int(input_dim)
        for _ in range(int(depth)):
            layers.extend((nn.Linear(width, int(hidden_dim)), _activation(activation)))
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            width = int(hidden_dim)
        layers.append(nn.Linear(width, len(VIOLATION_NAMES)))
        return nn.Sequential(*layers)

    def attach_randomized_prior(self, scale: float) -> None:
        self.prior = self._new_net()
        self.prior.requires_grad_(False)
        self.prior_scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.net(x)
        if self.prior is None:
            return learned
        return learned + self.prior_scale * self.prior(x)


class ViolationCriticEnsemble(nn.Module):
    """Independent small heads; output shape is ``(M, B, 4)``."""

    def __init__(self, input_dim: int, cfg: ViolationCriticConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        def spec(index: int) -> tuple[int, int, str, bool]:
            if cfg.head_architecture_mix == "homogeneous":
                return int(cfg.hidden_dim), int(cfg.depth), "silu", False
            narrow = max(32, int(round(float(cfg.hidden_dim) * 0.75)))
            variants = (
                (int(cfg.hidden_dim), int(cfg.depth), "silu", False),
                (narrow, int(cfg.depth) + 1, "gelu", False),
                (narrow, max(2, int(cfg.depth)), "silu", True),
            )
            return variants[index % len(variants)]

        self.heads = nn.ModuleList()
        for index in range(cfg.num_heads):
            width, depth, activation, residual = spec(index)
            self.heads.append(_CriticHead(
                input_dim, width, depth, cfg.dropout_probability,
                activation=activation, residual=residual,
            ))
        # Attach priors only after every learned head has been initialized.
        # Thus a fixed seed gives identical learned initializations in the
        # prior/no-prior ablation; prior initialization cannot perturb the RNG
        # stream used by later learned heads.
        if float(cfg.randomized_prior_scale) > 0.0:
            for head in self.heads:
                head.attach_randomized_prior(float(cfg.randomized_prior_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"critic input requires (B, {self.input_dim}), got {tuple(x.shape)}"
            )
        return torch.stack([head(x) for head in self.heads], dim=0)

    @torch.no_grad()
    def conservative_score(self, x: torch.Tensor, quantile: float = 0.8) -> torch.Tensor:
        # CUDA quantile has no bfloat16 kernel.  This method is normally called
        # from inside the actor's autocast region, so keep the cheap head
        # inference mixed precision but make the tiny MxB reduction FP32.
        per_head = self(x).amax(dim=-1).float()
        return torch.quantile(per_head, float(quantile), dim=0)


def _target_strata(targets: torch.Tensor, boundary_band: float) -> torch.Tensor:
    """Return exclusive manageable, near-outside, and far-outside classes."""
    score = targets.float().amax(dim=-1)
    return torch.where(
        score <= 0.0,
        torch.zeros_like(score, dtype=torch.long),
        torch.where(
            score <= float(boundary_band),
            torch.ones_like(score, dtype=torch.long),
            torch.full_like(score, 2, dtype=torch.long),
        ),
    )


class RecentCriticReplay:
    """Bounded CPU ring using float16 storage for predictable memory use."""

    def __init__(self, capacity: int, input_dim: int):
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.features = torch.empty(self.capacity, self.input_dim, dtype=torch.float16)
        self.targets = torch.empty(self.capacity, len(VIOLATION_NAMES), dtype=torch.float16)
        self.actor_step = torch.empty(self.capacity, dtype=torch.int64)
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(self, features: torch.Tensor, targets: torch.Tensor, *, actor_step: int) -> None:
        f = features.detach().to(device="cpu", dtype=torch.float16)
        t = targets.detach().to(device="cpu", dtype=torch.float16)
        if f.ndim != 2 or f.shape[1] != self.input_dim or t.shape != (f.shape[0], 4):
            raise ValueError("invalid critic replay batch shape")
        for i in range(f.shape[0]):
            j = self.cursor
            self.features[j].copy_(f[i])
            self.targets[j].copy_(t[i])
            self.actor_step[j] = int(actor_step)
            self.cursor = (self.cursor + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        manageable_fraction: float = 0.0,
        near_failure_fraction: float = 0.0,
        boundary_band: float = 0.5,
        newest_fraction: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty critic replay")
        fraction = float(newest_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("newest_fraction must be in (0, 1]")
        eligible = torch.ones(self.size, dtype=torch.bool)
        if fraction < 1.0:
            unique_steps = self.actor_step[: self.size].unique(sorted=True)
            n_steps = max(1, int(round(float(unique_steps.numel()) * fraction)))
            eligible = self.actor_step[: self.size] >= unique_steps[-n_steps]
        # Draw with replacement inside the window, preserving the same batch
        # width for all heads even when the newest window is still small.
        count = min(int(n), self.size)
        fractions = torch.tensor(
            [manageable_fraction, near_failure_fraction,
             1.0 - manageable_fraction - near_failure_fraction],
            dtype=torch.float32,
        )
        strata = _target_strata(self.targets[: self.size], boundary_band)
        members = [
            (eligible & (strata == c)).nonzero(as_tuple=True)[0]
            for c in range(3)
        ]
        available = torch.tensor([m.numel() > 0 for m in members])
        probs = fractions * available.float()
        if float(probs.sum()) <= 0.0:
            probs = available.float()
        probs = probs / probs.sum()
        chosen = torch.multinomial(
            probs, count, replacement=True, generator=generator,
        )
        pieces = []
        for c, pool in enumerate(members):
            nc = int((chosen == c).sum())
            if nc:
                draw = torch.randint(
                    0, int(pool.numel()), (nc,), generator=generator,
                )
                pieces.append(pool.index_select(0, draw))
        idx = torch.cat(pieces)
        idx = idx.index_select(
            0, torch.randperm(int(idx.numel()), generator=generator),
        )
        return (
            self.features.index_select(0, idx).to(device=device, dtype=torch.float32),
            self.targets.index_select(0, idx).to(device=device, dtype=torch.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        n = int(self.size)
        return {
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "size": n,
            "cursor": int(self.cursor),
            "features": self.features[:n].clone(),
            "targets": self.targets[:n].clone(),
            "actor_step": self.actor_step[:n].clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError("critic replay input width changed")
        n = min(int(state.get("size", 0)), self.capacity)
        if n:
            self.features[:n].copy_(state["features"][:n].to(torch.float16))
            self.targets[:n].copy_(state["targets"][:n].to(torch.float16))
            self.actor_step[:n].copy_(state["actor_step"][:n].to(torch.int64))
        self.size = n
        self.cursor = int(state.get("cursor", n)) % self.capacity if self.capacity else 0


class PriorityAuditCoreset:
    """Bounded, stratified high-error/disagreement observations."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        *,
        boundary_band: float = 0.5,
        fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.boundary_band = float(boundary_band)
        self.fractions = tuple(float(x) for x in fractions)
        self.features = torch.empty(self.capacity, self.input_dim, dtype=torch.float16)
        self.targets = torch.empty(self.capacity, 4, dtype=torch.float16)
        self.priority = torch.empty(self.capacity, dtype=torch.float32)
        self.actor_step = torch.empty(self.capacity, dtype=torch.int64)
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        priority: torch.Tensor,
        *,
        actor_step: int,
    ) -> None:
        if self.capacity == 0:
            return
        f = features.detach().cpu().to(torch.float16)
        t = targets.detach().cpu().to(torch.float16)
        p = priority.detach().cpu().float().reshape(-1)
        incoming_class = _target_strata(t, self.boundary_band)
        for i in range(f.shape[0]):
            if self.size < self.capacity:
                j = self.size
                self.size += 1
            else:
                existing_class = _target_strata(
                    self.targets[: self.size], self.boundary_band,
                )
                counts = torch.bincount(existing_class, minlength=3)
                quota = torch.tensor(self.fractions) * float(self.capacity)
                cls = int(incoming_class[i])
                under_quota = float(counts[cls]) + 1.0e-6 < float(quota[cls])
                if under_quota:
                    over = counts.float() - quota
                    victim_class = int(over.argmax())
                    candidates = (
                        existing_class == victim_class
                    ).nonzero(as_tuple=True)[0]
                    if not int(candidates.numel()):
                        candidates = torch.arange(self.size)
                else:
                    candidates = (
                        existing_class == cls
                    ).nonzero(as_tuple=True)[0]
                    if not int(candidates.numel()):
                        continue
                j = int(candidates[self.priority[candidates].argmin()])
                if not under_quota and float(p[i]) <= float(self.priority[j]):
                    continue
            self.features[j].copy_(f[i])
            self.targets[j].copy_(t[i])
            self.priority[j] = p[i]
            self.actor_step[j] = int(actor_step)

    def state_dict(self) -> dict[str, Any]:
        n = int(self.size)
        return {
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "size": n,
            "features": self.features[:n].clone(),
            "targets": self.targets[:n].clone(),
            "priority": self.priority[:n].clone(),
            "actor_step": self.actor_step[:n].clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError("critic audit input width changed")
        n = min(int(state.get("size", 0)), self.capacity)
        if n:
            self.features[:n].copy_(state["features"][:n].to(torch.float16))
            self.targets[:n].copy_(state["targets"][:n].to(torch.float16))
            self.priority[:n].copy_(state["priority"][:n].float())
            self.actor_step[:n].copy_(state["actor_step"][:n].long())
        self.size = n


class ViolationCriticTrainer:
    """Own the ensemble, EMA, replay, optimizer, RNG, and shadow metrics."""

    def __init__(
        self,
        input_dim: int,
        cfg: ViolationCriticConfig,
        *,
        device: torch.device | str,
        seed: int,
    ):
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.device = torch.device(device)
        # Keep initialization independent of the actor/global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.live = ViolationCriticEnsemble(input_dim, cfg).to(self.device)
        self.ema = copy.deepcopy(self.live).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.live.parameters() if p.requires_grad),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.recent = RecentCriticReplay(cfg.recent_capacity, input_dim)
        self.audit = PriorityAuditCoreset(
            cfg.audit_capacity,
            input_dim,
            boundary_band=cfg.replay_boundary_band,
            fractions=(
                cfg.replay_manageable_fraction,
                cfg.replay_near_failure_fraction,
                1.0 - cfg.replay_manageable_fraction
                - cfg.replay_near_failure_fraction,
            ),
        )
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + 1)
        # MC-dropout uses a separate stream so changing the number of stochastic
        # critic passes cannot change replay sampling or frontier directions.
        self.mc_generator = torch.Generator(device="cpu")
        self.mc_generator.manual_seed(int(seed) + 2)
        self.updates = 0
        self._metric_sum: dict[str, float] = {}
        self._metric_n = 0

    @torch.no_grad()
    def predict(self, features: torch.Tensor, *, ema: bool = True, physical: bool = False) -> torch.Tensor:
        net = self.ema if ema else self.live
        x = features.detach().to(self.device, torch.float32)
        samples = int(self.cfg.mc_dropout_samples)
        if float(self.cfg.dropout_probability) <= 0.0 or samples == 1:
            pred = net(x)
        else:
            dropout = [module for module in net.modules() if isinstance(module, nn.Dropout)]
            states = [module.training for module in dropout]
            seed = int(torch.randint(
                0, 2**31 - 1, (1,), generator=self.mc_generator,
            ))
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            try:
                for module in dropout:
                    module.train(True)
                # Dropout uses the device-global RNG.  Fork it so critic MC
                # sampling cannot perturb actor/data randomness.
                with torch.random.fork_rng(devices=devices):
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed(seed)
                    else:
                        torch.manual_seed(seed)
                    pred = torch.cat([net(x) for _ in range(samples)], dim=0)
            finally:
                for module, training in zip(dropout, states):
                    module.train(training)
        return robust_inverse(pred, self.cfg.scales) if physical else pred

    @torch.no_grad()
    def conservative_score(self, features: torch.Tensor) -> torch.Tensor:
        prediction = self.predict(features, ema=True, physical=False)
        return conservative_score_from_prediction(
            prediction, quantile=float(self.cfg.conservative_quantile),
        )

    def observe_gate(
        self,
        features: torch.Tensor,
        gate: torch.Tensor,
        *,
        actor_step: int,
    ) -> dict[str, float]:
        residual = residuals_from_gate(gate, self.cfg)
        target = robust_transform(residual, self.cfg.scales)
        with torch.no_grad():
            pred = self.ema(features.detach().to(self.device, torch.float32))
            mean = pred.mean(dim=0)
            err = (mean - target.to(self.device)).abs().mean(dim=1)
            disagreement = pred.std(dim=0, unbiased=False).mean(dim=1)
            priority = err + disagreement
        self.recent.add(features, target, actor_step=actor_step)
        self.audit.add(
            features, target, priority,
            actor_step=actor_step,
        )
        stats: dict[str, float] = {}
        for _ in range(int(self.cfg.updates_per_step)):
            stats = self._update_once()
        return stats

    def _update_once(self) -> dict[str, float]:
        sample_kw = dict(
            generator=self.generator, device=self.device,
            manageable_fraction=self.cfg.replay_manageable_fraction,
            near_failure_fraction=self.cfg.replay_near_failure_fraction,
            boundary_band=self.cfg.replay_boundary_band,
        )
        timescale = bool(self.cfg.replay_timescale_diversity)
        if timescale:
            batches = [
                self.recent.sample(
                    self.cfg.batch_size, newest_fraction=fraction, **sample_kw,
                )
                for fraction in (0.25, 0.50, 1.0)
            ]
            # Metrics use the full-window batch, evaluated by every EMA head.
            x, target = batches[-1]
        else:
            x, target = self.recent.sample(self.cfg.batch_size, **sample_kw)

        def forward_and_loss():
            if not timescale:
                prediction = self.live(x)
                total, regression, boundary = critic_loss_per_head_row(
                    prediction, target, self.cfg,
                )
                targets = [target for _ in self.live.heads]
            else:
                predictions = [
                    head(features)
                    for head, (features, _target) in zip(self.live.heads, batches)
                ]
                parts = [
                    critic_loss_per_head_row(
                        prediction.unsqueeze(0), batch_target, self.cfg,
                    )
                    for prediction, (_features, batch_target)
                    in zip(predictions, batches)
                ]
                prediction = torch.stack(predictions, dim=0)
                total = torch.cat([part[0] for part in parts], dim=0)
                regression = torch.cat([part[1] for part in parts], dim=0)
                boundary = torch.cat([part[2] for part in parts], dim=0)
                targets = [batch_target for _features, batch_target in batches]
            return prediction, total, regression, boundary, targets

        if float(self.cfg.dropout_probability) > 0.0:
            seed = int(torch.randint(
                0, 2**31 - 1, (1,), generator=self.mc_generator,
            ))
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                else:
                    torch.manual_seed(seed)
                pred, per_row, regression_per_row, boundary_per_row, train_targets = (
                    forward_and_loss()
                )
        else:
            pred, per_row, regression_per_row, boundary_per_row, train_targets = (
                forward_and_loss()
            )
        mask = torch.rand(
            per_row.shape, generator=self.generator, device="cpu",
        ).to(self.device) < float(self.cfg.bootstrap_probability)
        # A tiny early replay can otherwise leave a head with no gradient.
        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty, 0] = True
        loss = (per_row * mask).sum() / mask.sum().clamp_min(1)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._update_ema()
        self.updates += 1

        with torch.no_grad():
            ema_pred = self.ema(x)
            pred_phys = robust_inverse(ema_pred.mean(dim=0), self.cfg.scales)
            target_phys = robust_inverse(target, self.cfg.scales)
            disagreement = ema_pred.std(dim=0, unbiased=False).mean()
            conservative = conservative_score_from_prediction(
                ema_pred, quantile=float(self.cfg.conservative_quantile),
            )
            true_score = target.amax(dim=-1)
            threshold = float(self.cfg.decision_threshold)
            false_manageable = (
                (conservative <= threshold) & (true_score > 0.0)
            ).float().mean()
            true_manageable = true_score <= 0.0
            pred_manageable = conservative <= threshold
            feasible_recall = (
                (true_manageable & pred_manageable).float().sum()
                / true_manageable.float().sum().clamp_min(1.0)
            )
            stats = {
                "loss": float(loss.detach()),
                "loss_regression": float(
                    (regression_per_row * mask).sum() / mask.sum().clamp_min(1)
                ),
                "loss_boundary": float(
                    (boundary_per_row * mask).sum() / mask.sum().clamp_min(1)
                ),
                "underprediction_rate": float(
                    torch.stack([
                        (head_pred.detach() < head_target).float().mean()
                        for head_pred, head_target in zip(pred, train_targets)
                    ]).mean()
                ),
                "disagreement": float(disagreement),
                "false_manageable": float(false_manageable),
                "sample_manageable": float(true_manageable.float().mean()),
                "pred_manageable": float(pred_manageable.float().mean()),
                "feasible_recall": float(feasible_recall),
                "recent_size": float(len(self.recent)),
                "audit_size": float(len(self.audit)),
            }
            mae = (pred_phys - target_phys).abs().mean(dim=0)
            for name, value in zip(VIOLATION_NAMES, mae):
                stats[f"mae_{name}"] = float(value)
        self._accumulate_metrics(stats)
        return stats

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = float(self.cfg.ema_decay)
        for ema, live in zip(self.ema.parameters(), self.live.parameters()):
            ema.mul_(decay).add_(live, alpha=1.0 - decay)

    def _accumulate_metrics(self, stats: dict[str, float]) -> None:
        for key, value in stats.items():
            self._metric_sum[key] = self._metric_sum.get(key, 0.0) + float(value)
        self._metric_n += 1

    def metrics(self, *, reset: bool = False) -> dict[str, float]:
        n = max(self._metric_n, 1)
        out = {key: value / n for key, value in self._metric_sum.items()}
        out["updates"] = float(self.updates)
        out["recent_size"] = float(len(self.recent))
        out["audit_size"] = float(len(self.audit))
        if reset:
            self._metric_sum.clear()
            self._metric_n = 0
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": CRITIC_FORMAT_VERSION,
            "input_dim": self.input_dim,
            "config": asdict(self.cfg),
            "live": self.live.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "recent": self.recent.state_dict(),
            "audit": self.audit.state_dict(),
            "generator_state": self.generator.get_state(),
            "mc_generator_state": self.mc_generator.get_state(),
            "updates": int(self.updates),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("format_version", -1))
        if version != CRITIC_FORMAT_VERSION:
            raise ValueError(f"unsupported timing critic checkpoint version {version}")
        if int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError(
                f"timing critic input width {state.get('input_dim')} != {self.input_dim}"
            )
        stored_cfg = state.get("config") or {}
        semantic_fields = (
            "num_heads", "hidden_dim", "depth",
            "dropout_probability", "head_architecture_mix",
            "replay_timescale_diversity",
            "randomized_prior_scale",
            "terminal_limit", "collision_sdf_limit", "joint_limit", "torque_limit",
            "terminal_scale", "collision_scale", "joint_scale", "torque_scale",
        )
        changed = [
            name for name in semantic_fields
            if name in stored_cfg and stored_cfg[name] != getattr(self.cfg, name)
        ]
        if changed:
            raise ValueError(
                "timing critic checkpoint has incompatible model/target config: "
                + ", ".join(changed)
            )
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
        if "mc_generator_state" in state:
            self.mc_generator.set_state(state["mc_generator_state"].cpu())
        self.updates = int(state.get("updates", 0))
