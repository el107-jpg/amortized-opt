"""Collection and update for executed-prefix PPO.

Route taken, and why
--------------------

**A compact self-contained PPO**, whose update mirrors rsl_rl 3.1.2's
``PPO.update`` term for term (same clip, the same adaptive-KL controller with
the same thresholds and bounds, the same loss composition), rather than a
subclass of ``rsl_rl.algorithms.PPO``.  Three concrete blockers, in order of
weight:

1. **The auxiliary analytic cost has to share the forward pass.**  rsl_rl's
   ``update`` is a monolithic loop with no per-loss hook -- which is exactly why
   ``bc_aux.DrumBcAuxPPO`` does not fuse its BC term at all, but instead takes
   *separate* gradient steps after ``super().update()`` returns, sharing only
   the Adam optimizer.  That pattern is right for BC (an MLP forward costs
   nothing) and wrong here: the aux cost is evaluated on the SAME emitted
   trajectory the policy loss uses, so fusing it is free, while bc_aux's
   post-hoc pattern would mean a second 19 M-parameter refinement pass per
   minibatch -- doubling the single most expensive part of the update to avoid
   writing ~150 lines.
2. **The policy's input is not an observation vector.**  It is a stateful warm
   start (the previous plan advanced one tick and re-goaled on hit completion)
   plus a structured ``ContextInputs``.  rsl_rl's ``RolloutStorage`` stores a
   flat ``TensorDict`` per transition and its ``mini_batch_generator``
   reshuffles it; reconstructing a warm start from that is either impossible or
   wrong in a way nothing would catch.  The storage here keeps the exact MPC
   inputs each tick consumed, which is also what makes the ratio-at-epoch-0
   health check meaningful.
3. **The sibling env is deliberately not a gym surface.**  ``isaac_rl.env``
   states it outright ("the env's own ``step`` is never used"); it exposes a
   batched ``plant(prev_cmd, cmd, q, qd)`` callable driven by
   ``DrumMPCLoop``.  Faking ``DirectRLEnv.step`` so ``OnPolicyRunner`` could
   drive it would put the MPC's warm-start state machine on the wrong side of
   the interface.

A secondary hazard settles it: rsl_rl calls ``policy.act`` *inside* the
minibatch loop, and the ported AR(1) sampler mutates its per-env noise state on
every ``act``.  Under rsl_rl that state would be advanced 12 extra times per
iteration, at minibatch rather than env shape.

What is kept identical to rsl_rl 3.1.2 so the tuning transfers: the Gaussian KL
estimator and the ``kl > 2*desired`` / ``kl < desired/2`` thresholds with the
1.5x step, ``surrogate + c_v * value - c_e * entropy``, clipped value loss,
global grad-norm clipping, and advantage normalisation.  Deliberate deviations
are marked ``DEVIATION`` at the point they occur.
"""

from __future__ import annotations

import math
import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.cost import FusedCost, StrikeContext
from amortized_opt.tokenizer import ContextInputs
from amortized_opt.train_cost import group_lengths, phase2_cost_config

from .actor import (
    AmortizedPpoActor,
    MpcInputs,
    executed_command,
    waypoint_masks,
)
from .cfg import PpoRunConfig

__all__ = [
    "PpoStorage",
    "PpoUpdater",
    "RolloutDriver",
    "combine_wp0_orth",
    "clone_encoding",
    "compute_gae",
    "terminal_waist_reward",
    "index_dataclass",
    "stack_dataclass",
]


def terminal_waist_reward(
    q_after: torch.Tensor,
    target: torch.Tensor,
    due: torch.Tensor,
    *,
    yaw_index: int,
    weight: float,
    scale_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse endpoint reward against the first plan's terminal waist yaw."""
    error = (q_after[:, int(yaw_index)] - target).abs()
    quality = 1.0 / (1.0 + (error / float(scale_rad)).square())
    return float(weight) * quality * due.to(q_after.dtype), error


# ============================================================== dataclass ops =
def stack_dataclass(items: list):
    """``[dc(B, ...)] * T`` -> ``dc(T * B, ...)``, preserving ``None`` fields."""
    cls = type(items[0])
    out = {}
    for f in items[0].__dataclass_fields__:
        vals = [getattr(x, f) for x in items]
        out[f] = None if vals[0] is None else torch.stack(vals, 0).flatten(0, 1)
    return cls(**out)


def index_dataclass(dc, idx: torch.Tensor):
    """Row-select every tensor field of a flat dataclass."""
    cls = type(dc)
    return cls(**{
        f: (None if getattr(dc, f) is None else getattr(dc, f).index_select(0, idx))
        for f in dc.__dataclass_fields__
    })


# ================================================================== storage ===
@dataclass
class PpoStorage:
    """Flat rollout buffer.  Everything needed to REPLAY the policy exactly.

    Sizes at 64 envs x 48 steps: the warm start dominates at
    ``64 * 48 * 50 * 17 * 4 B = 20.9 MB``; the contexts add ~1 MB.  Scaling to
    512 envs x 48 steps takes the warm start to 167 MB, which is still small
    next to the model's own activations.
    """

    device: torch.device
    horizon: int
    n_dof: int

    init: list = field(default_factory=list)          # (B, H, 17) per step
    state: list = field(default_factory=list)         # (B, 34)
    hit_index: list = field(default_factory=list)     # (B, 2) long
    ctx: list = field(default_factory=list)           # ContextInputs
    encoding: list = field(default_factory=list)      # ContextEncoding used by refine
    strike: list = field(default_factory=list)        # StrikeContext
    critic_obs: list = field(default_factory=list)    # (B, obs_dim)
    actions: list = field(default_factory=list)       # (B, 17)
    log_probs: list = field(default_factory=list)     # (B,)
    mu: list = field(default_factory=list)            # (B, 17)
    sigma: list = field(default_factory=list)         # (B, 17)
    values: list = field(default_factory=list)        # (B,)
    rewards: list = field(default_factory=list)       # (B,)
    dones: list = field(default_factory=list)         # (B,) bool
    waist_anchor: list = field(default_factory=list)  # (B, H), time-aligned
    waist_anchor_mask: list = field(default_factory=list)  # (B, H) bool

    # filled by compute_gae
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    flat: dict = field(default_factory=dict)

    def add(self, **kw) -> None:
        for k, v in kw.items():
            getattr(self, k).append(v)

    @property
    def num_steps(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        for k in ("init", "state", "hit_index", "ctx", "encoding", "strike",
                  "critic_obs",
                  "actions", "log_probs", "mu", "sigma", "values", "rewards",
                  "dones", "waist_anchor", "waist_anchor_mask"):
            getattr(self, k).clear()
        self.returns = None
        self.advantages = None
        self.flat = {}

    # ------------------------------------------------------------ flatten --
    def finalize(self) -> None:
        """Stack ``(T, B, ...)`` and flatten to ``(T * B, ...)``."""
        cat = lambda xs: torch.stack(xs, 0).flatten(0, 1)   # noqa: E731
        init = cat(self.init)
        self.flat = {
            "init": init,
            "state": cat(self.state),
            "hit_index": cat(self.hit_index),
            "critic_obs": cat(self.critic_obs),
            "actions": cat(self.actions),
            "log_probs": cat(self.log_probs),
            "mu": cat(self.mu),
            "sigma": cat(self.sigma),
            "values": cat(self.values),
            "returns": self.returns.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "waist_anchor": (
                cat(self.waist_anchor)
                if self.waist_anchor else init.new_zeros(init.shape[:2])
            ),
            "waist_anchor_mask": (
                cat(self.waist_anchor_mask)
                if self.waist_anchor_mask
                else torch.zeros(init.shape[:2], device=init.device, dtype=torch.bool)
            ),
        }
        self.flat["ctx"] = stack_dataclass(self.ctx)
        self.flat["encoding"] = stack_dataclass(self.encoding)
        self.flat["strike"] = stack_dataclass(self.strike)

    def minibatches(self, num_mini_batches: int, generator: torch.Generator | None = None):
        """Yield index tensors.  DEVIATION: reshuffled every call.

        rsl_rl 3.1.2 draws ONE ``randperm`` and reuses that same partition for
        every epoch.  Reshuffling per epoch is the standard formulation and
        strictly decorrelates the minibatch composition across epochs; it
        changes no term of the loss.
        """
        n = self.flat["actions"].shape[0]
        perm = torch.randperm(n, device=self.device, generator=generator)
        size = n // int(num_mini_batches)
        for i in range(int(num_mini_batches)):
            yield perm[i * size: (i + 1) * size]


def compute_gae(
    rewards: torch.Tensor,     # (T, B)
    values: torch.Tensor,      # (T, B)
    dones: torch.Tensor,       # (T, B) bool
    last_values: torch.Tensor,  # (B,)
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard truncated GAE(lambda).  Returns ``(returns, advantages)``.

    ``dones[t]`` means the episode ended ON step ``t``, so the bootstrap through
    ``t + 1`` is cut.  Episodes here end on clip exhaustion, which is a
    time-limit rather than a failure -- see the DECISION note in README about
    bootstrapping time-limited terminations.
    """
    T, B = rewards.shape
    adv = torch.zeros(T, B, device=rewards.device, dtype=rewards.dtype)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        not_done = (~dones[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae
    return adv + values, adv


def clone_encoding(enc):
    """Detach-clone a ``ContextEncoding`` so the next tick cannot overwrite it."""
    if enc is None:
        return None
    def c(v):
        return None if v is None else v.detach().clone()
    return type(enc)(
        memory=enc.memory.detach().clone(),
        mask=enc.mask.detach().clone(),
        waist_yaw_sp=c(getattr(enc, "waist_yaw_sp", None)),
        waist_yaw_res=c(getattr(enc, "waist_yaw_res", None)),
        waist_pr_sp=c(getattr(enc, "waist_pr_sp", None)),
        waist_pr_res=c(getattr(enc, "waist_pr_res", None)),
    )


def combine_wp0_orth(
    g_rl: torch.Tensor,
    g_cost: torch.Tensor,
    *,
    cost_to_rl_norm: float = 1.0,
    rl_to_cost_norm: float = 1.0,
    traj_grad_scale: float = 1.0,
    wp0_leak: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Always-project RL onto the cost nullspace at the executed waypoint.

    ``g_rl`` / ``g_cost`` are ``(B, H, n)`` trajectory gradients.  PPO only
    scores waypoint 0, so the tail of ``g_rl`` is ignored.  Cost is **rescaled
    per sample** so ``||g_cost||_F = cost_to_rl_norm * ||g_rl0||``; without
    that, FusedCost's raw gradient (~aux 4000) swallows the 17-dim KL budget
    and the plan NaNs.  The projected PPO leftover is then added at waypoint 0
    at PPO's native scale (``rl_to_cost_norm``), not inflated to ``||g_cost0||``.
    ``traj_grad_scale`` then shrinks the whole combined tensor (relative mix
    unchanged) so one Adam step stays inside the 17-dim KL budget.

    ``wp0_leak`` in ``[0, 1]`` re-injects that fraction of the *parallel* PPO
    component at waypoint 0 (the knot that actually runs).  0 is a hard
    nullspace; 1 is full PPO at wp0 on top of scaled cost.  The tail stays
    cost-only either way.
    """
    if g_rl.shape != g_cost.shape:
        raise ValueError(
            f"g_rl {tuple(g_rl.shape)} vs g_cost {tuple(g_cost.shape)}")
    B = g_rl.shape[0]
    leak = float(wp0_leak)
    g_rl0 = g_rl[:, 0]
    g_c0 = g_cost[:, 0]
    dots = (g_rl0 * g_c0).sum(dim=-1, keepdim=True)
    c2 = (g_c0 * g_c0).sum(dim=-1, keepdim=True)
    g_para = torch.where(
        c2 > eps, (dots / c2.clamp_min(eps)) * g_c0, torch.zeros_like(g_rl0))
    g_orth = torch.where(c2 > eps, g_rl0 - g_para, g_rl0)
    orth_norm = g_orth.norm(dim=-1, keepdim=True)
    cost0_norm = g_c0.norm(dim=-1, keepdim=True)
    rl_norm = g_rl0.norm(dim=-1, keepdim=True)
    rel_orth = orth_norm / rl_norm.clamp_min(eps)
    g_orth = torch.where(rel_orth > 1e-4, g_orth, torch.zeros_like(g_orth))
    g_orth = torch.where(cost0_norm > eps, g_orth, g_rl0)
    g_used = g_orth + leak * g_para

    cost_ref = g_cost.flatten(1).norm(dim=-1).clamp_min(eps)          # (B,)
    rl_ref = rl_norm.squeeze(-1)
    scale = torch.where(
        rl_ref > eps,
        float(cost_to_rl_norm) * rl_ref / cost_ref,
        torch.where(cost_ref > eps, float(cost_to_rl_norm) / cost_ref,
                    torch.zeros_like(cost_ref)),
    )
    g = g_cost * scale.view(B, 1, 1)
    g[:, 0] = g[:, 0] + float(rl_to_cost_norm) * g_used
    g = g * float(traj_grad_scale)

    valid = (rl_ref > eps) & (cost0_norm.squeeze(-1) > eps)
    cos = (g_rl0 * g_c0).sum(dim=-1) / (
        rl_norm.squeeze(-1) * cost0_norm.squeeze(-1)).clamp_min(eps)
    remain = orth_norm.squeeze(-1) / rl_norm.squeeze(-1).clamp_min(eps)
    para_frac = g_para.norm(dim=-1) / rl_norm.squeeze(-1).clamp_min(eps)
    stats = {
        "orth_cos": float(cos[valid].mean()) if bool(valid.any()) else 0.0,
        "orth_remain": float(remain[rl_ref > eps].mean())
        if bool((rl_ref > eps).any()) else 0.0,
        "orth_cost_norm": float(cost0_norm.mean()),
        "orth_rl_norm": float(rl_norm.mean()),
        "orth_cost_scale": float(scale.mean()),
        "orth_traj_scale": float(traj_grad_scale),
        "orth_leak": leak,
        "orth_para_frac": float(para_frac[rl_ref > eps].mean())
        if bool((rl_ref > eps).any()) else 0.0,
    }
    return g, stats


# ================================================================== updater ===
class PpoUpdater:
    """The PPO update, with the analytic cost fused into the same forward.

    Default combine (``OrthGradConfig``): always-project the waypoint-0 PPO
    gradient onto the analytic cost's nullspace, then add it back as a small
    scaled direction.  The tail of the plan is cost-only -- that tail is the
    next tick's warm start.  ``--no-orth`` restores the previous scalar sum.
    """

    def __init__(
        self,
        actor: AmortizedPpoActor,
        cfg: PpoRunConfig,
        *,
        device: torch.device | str,
        cost: FusedCost | None = None,
        cost_weights: dict[str, float] | None = None,
        critic_only: bool = False,
        waist_consistency_weight: float = 0.0,
        waist_consistency_beta: float = 0.02,
        actor_loss_weight: float = 1.0,
    ):
        self.actor = actor
        self.cfg = cfg
        self.p = cfg.ppo
        self.device = torch.device(device)
        self.learning_rate = float(self.p.learning_rate)
        self.critic_only = bool(critic_only)
        owned = (
            list(actor.critic.parameters())
            if self.critic_only else actor.trainable_parameters()
        )
        self.optimizer = torch.optim.Adam(owned, lr=self.learning_rate)
        self.cost = cost
        self.cost_weights = cost_weights
        self.waist_consistency_weight = float(waist_consistency_weight)
        self.waist_consistency_beta = float(waist_consistency_beta)
        self.actor_loss_weight = float(actor_loss_weight)
        self.yaw_i = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
        self._checked_frozen = False
        self.lr_floor_hits = 0

    # ---------------------------------------------------------- aux losses --
    def _aux_cost(self, traj, elem_mask, state, strike) -> torch.Tensor:
        """``FusedCost`` on the FULL emitted trajectory (the tail's anchor).

        Only waypoint 0 is executed, so nothing in the PPO return prices
        waypoints 1..H-1 -- and those are precisely what the next tick's warm
        start is made of.  Without this term the plan is free to degenerate
        exactly where the environment cannot see it.
        """
        if self.cost is None or not self.cfg.aux.enabled:
            return torch.zeros((), device=self.device)
        n = self.actor.n
        q0, v0 = state[:, :n], state[:, n:]
        lengths = group_lengths(strike.hit_index).to(traj.dtype)
        ct = self.cost(
            q0, traj, v0, strike, lengths=lengths, elem_mask=elem_mask,
            weights=self.cost_weights,
        )
        terms = self.cfg.aux.terms
        if terms is None:
            return ct.total.mean()
        picked = [ct.terms[k] for k in terms if k in ct.terms]
        if not picked:
            raise KeyError(
                f"aux.terms {terms} matched no FusedCost group; available: "
                f"{sorted(ct.terms)}")
        return torch.stack(picked, 0).sum(0).mean()

    def _waist_consistency(
        self,
        traj: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Huber drift and MAE against the time-aligned first MPC plan."""
        err = traj[:, :, self.yaw_i] - target.detach().to(traj.dtype)
        valid = mask.to(device=traj.device, dtype=traj.dtype)
        n = valid.sum().clamp_min(1.0)
        ae = err.abs()
        beta = max(float(self.waist_consistency_beta), 1.0e-9)
        huber = torch.where(
            ae < beta,
            0.5 * err.square() / beta,
            ae - 0.5 * beta,
        )
        return (huber * valid).sum() / n, (ae * valid).sum() / n

    def _combined_aux(self, flat: dict, sel: torch.Tensor, plan, inputs):
        # The executed-prefix mean includes the trainable correction head.  If
        # physics is evaluated on ``plan.traj`` unchanged, that head bypasses
        # every torque/collision/terminal constraint and can move the actual
        # command arbitrarily.  Price the corrected first waypoint; the tail
        # remains the decoder trajectory that becomes the next warm start.
        aux_traj = plan.traj.clone()
        aux_traj[:, 0] = plan.mean
        physics = self._aux_cost(
            aux_traj, plan.elem_mask, inputs.state,
            index_dataclass(flat["strike"], sel),
        )
        waist, waist_mae = self._waist_consistency(
            plan.traj,
            flat["waist_anchor"].index_select(0, sel),
            flat["waist_anchor_mask"].index_select(0, sel),
        )
        weighted = (
            float(self.cfg.aux.weight) * physics
            + self.waist_consistency_weight * waist
        )
        return physics, waist, waist_mae, weighted

    # -------------------------------------------------------------- update --
    def update(self, storage: PpoStorage) -> dict[str, float]:
        if self.critic_only:
            return self._update_critic_only(storage)
        p = self.p
        flat = storage.flat
        n = flat["actions"].shape[0]
        micro = int(p.micro_batch_size) or n

        adv_all = flat["advantages"]
        if not torch.isfinite(adv_all).all() or not torch.isfinite(flat["returns"]).all():
            return {
                "surrogate": float("nan"), "value": float("nan"),
                "entropy": float("nan"), "aux": float("nan"), "kl": float("nan"),
                "clip_frac": 0.0, "ratio_dev": 0.0, "orth_cos": 0.0,
                "orth_remain": 0.0, "orth_cost_norm": 0.0, "orth_rl_norm": 0.0,
                "param_cos": 0.0, "param_cos_distance": 1.0,
                "param_cost_norm": 0.0, "param_rl_norm": 0.0,
                "separate_ppo_grad_norm": 0.0,
                "separate_aux_grad_norm": 0.0,
                "separate_value_grad_norm": 0.0,
                "separate_ppo_clip": 1.0,
                "separate_aux_clip": 1.0,
                "separate_value_clip": 1.0,
                "trust_before": 0.0, "trust_after": 0.0,
                "trust_alpha": 0.0, "trust_rejected": 0.0,
                "orth_cost_scale": 0.0, "orth_traj_scale": 0.0,
                "orth_leak": 0.0, "orth_para_frac": 0.0,
                "orth_skipped": 1.0,
                "learning_rate": self.learning_rate,
                "lr_floor_hits": float(self.lr_floor_hits),
                "noise_std_mean": float(self.actor.current_std().mean()),
                "noise_std_max_dim": float(self.actor.current_std().max()),
                "kl_early_stops": 0.0,
            }
        if p.normalize_advantage:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        acc = {k: 0.0 for k in ("surrogate", "value", "entropy", "aux",
                                "aux_objective",
                                "waist_consistency", "waist_consistency_mae_rad", "kl",
                                "clip_frac", "ratio_dev", "orth_cos",
                                "orth_remain", "orth_cost_norm", "orth_rl_norm",
                                "param_cos", "param_cos_distance",
                                "param_cost_norm", "param_rl_norm",
                                "separate_ppo_grad_norm",
                                "separate_aux_grad_norm",
                                "separate_value_grad_norm",
                                "separate_ppo_clip", "separate_aux_clip",
                                "separate_value_clip",
                                "trust_before", "trust_after",
                                "trust_alpha", "trust_rejected",
                                "orth_cost_scale", "orth_traj_scale",
                                "orth_leak", "orth_para_frac",
                                "orth_skipped")}
        n_updates = 0
        kl_early_stops = 0

        stop = False
        for epoch in range(int(p.num_learning_epochs)):
            for idx in storage.minibatches(p.num_mini_batches):
                mb = idx.shape[0]
                self.optimizer.zero_grad(set_to_none=True)
                stats = {k: 0.0 for k in acc}
                lora_params = self._lora_params()
                trust_base = (
                    [parameter.detach().clone() for parameter in lora_params]
                    if self.cfg.orth.trust_region else None
                )
                trust_state = (
                    {
                        parameter: copy.deepcopy(self.optimizer.state.get(parameter, {}))
                        for parameter in lora_params
                    }
                    if self.cfg.orth.trust_region else None
                )
                if not self.cfg.orth.enabled:
                    self._begin_separate_grads()
                else:
                    self._begin_balanced_head_grads()

                for s in range(0, mb, micro):
                    sel = idx[s: s + micro]
                    w = sel.shape[0] / mb          # exact chunk reweighting, so
                    # gradient accumulation is numerically identical to one pass
                    part = self._chunk_loss(flat, sel, adv_all, weight=w)
                    for k in stats:
                        if k in part:
                            stats[k] += w * float(part[k])
                if not self.cfg.orth.enabled:
                    stats.update(self._finish_separate_grads())
                else:
                    stats.update(self._finish_balanced_head_grads())

                if not self._checked_frozen:
                    self.actor.assert_frozen()
                    self._checked_frozen = True

                if not math.isfinite(stats["kl"]):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # ---- adaptive KL, in the minibatch loop as rsl_rl does it ----
                # Fresh-rollout minibatch 0 has KL≈0 after a correct replay, so
                # a raise there ratchets LR every iteration (v3: 1e-5 → 6.7e-4
                # in 6 iters) and then early-stop cuts only once.  Only cut on
                # minibatch 0; raises start after at least one step this iter.
                if p.desired_kl is not None and p.schedule == "adaptive":
                    self._adapt_lr(stats["kl"], allow_increase=(n_updates > 0))

                if self.cfg.orth.enabled:
                    nn.utils.clip_grad_norm_(self.actor.actor_parameters(),
                                             p.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.actor.critic.parameters(),
                                             p.max_grad_norm)
                self.optimizer.step()
                if self.cfg.orth.trust_region:
                    stats.update(self._apply_lora_trust_region(
                        flat, idx, micro,
                        # Trust protects the original full-trajectory physics
                        # objective.  The task loss is allowed to fall only
                        # along steps that do not worsen this anchor.
                        baseline=float(stats["aux"]),
                        params=lora_params,
                        base=trust_base,
                        optimizer_state=trust_state,
                    ))
                for k in acc:
                    acc[k] += stats[k]
                n_updates += 1
                if (p.kl_early_stop and p.desired_kl is not None
                        and stats["kl"] > p.desired_kl * 2.0):
                    stop = True
                    kl_early_stops += 1
                    break
            if stop:
                break

        out = {k: v / max(n_updates, 1) for k, v in acc.items()}
        out["learning_rate"] = self.learning_rate
        out["lr_floor_hits"] = float(self.lr_floor_hits)
        out["noise_std_mean"] = float(self.actor.current_std().detach().mean())
        out["noise_std_max_dim"] = float(self.actor.current_std().detach().max())
        out["kl_early_stops"] = float(kl_early_stops)
        with torch.no_grad():
            returns = flat["returns"]
            predicted = self.actor.critic(flat["critic_obs"])
            residual_var = (returns - predicted).var(unbiased=False)
            return_var = returns.var(unbiased=False)
            explained = 1.0 - residual_var / return_var.clamp_min(1.0e-12)
            out["critic_explained_variance"] = float(explained)
            out["critic_return_mean"] = float(returns.mean())
            out["critic_return_std"] = float(returns.std(unbiased=False))
            out["critic_value_mean"] = float(predicted.mean())
        return out

    def _begin_separate_grads(self) -> None:
        """Allocate minibatch accumulators for separately clipped objectives."""
        self._sep_policy_params = list(self.actor.actor_parameters())
        self._sep_aux_params = self._lora_params()
        self._sep_value_params = list(self.actor.critic.parameters())
        self._sep_policy_grads = [torch.zeros_like(p) for p in self._sep_policy_params]
        self._sep_aux_grads = [torch.zeros_like(p) for p in self._sep_aux_params]
        self._sep_value_grads = [torch.zeros_like(p) for p in self._sep_value_params]

    def _begin_balanced_head_grads(self) -> None:
        """Accumulate correction-head objectives over the whole minibatch.

        Norm matching inside each memory microbatch changes the algorithm when
        the split changes.  These buffers postpone that nonlinear operation
        until every microbatch has contributed its exactly weighted gradient.
        """
        head = self.actor.mean_correction
        self._bal_head_params = [] if head is None else list(head.parameters())
        self._bal_head_ppo = [torch.zeros_like(p) for p in self._bal_head_params]
        self._bal_head_aux = [torch.zeros_like(p) for p in self._bal_head_params]

    def _finish_balanced_head_grads(self) -> dict[str, float]:
        ppo = self._bal_head_ppo
        aux = self._bal_head_aux
        ppo_norm = self._grad_list_norm(ppo)
        aux_norm = self._grad_list_norm(aux)
        eps = float(self.cfg.orth.eps)
        if ppo and float(aux_norm) > eps:
            dot = sum((gr.double() * gc.double()).sum() for gr, gc in zip(ppo, aux))
            c2 = aux_norm.square().clamp_min(eps)
            proj = dot / c2
            scale = (float(self.cfg.orth.cost_to_rl_norm) * ppo_norm /
                     aux_norm.clamp_min(eps))
            mix = float(self.cfg.orth.rl_to_cost_norm)
            shrink = float(self.cfg.orth.traj_grad_scale)
            for param, gr, gc in zip(self._bal_head_params, ppo, aux):
                parallel = proj.to(gr) * gc
                used = gr - parallel + float(self.cfg.orth.wp0_leak) * parallel
                grad = shrink * (scale.to(gr) * gc + mix * used)
                param.grad = grad if param.grad is None else param.grad + grad
        else:
            scale = ppo_norm.new_zeros(())
            for param, grad in zip(self._bal_head_params, ppo):
                param.grad = grad if param.grad is None else param.grad + grad
        stats = {
            "separate_ppo_grad_norm": float(ppo_norm),
            "separate_aux_grad_norm": float(aux_norm),
            "separate_ppo_clip": 1.0,
            "separate_aux_clip": float(scale) if ppo else 1.0,
        }
        for name in ("_bal_head_params", "_bal_head_ppo", "_bal_head_aux"):
            delattr(self, name)
        return stats

    @staticmethod
    def _accumulate_grad_list(buffers, grads, weight: float) -> None:
        for buffer, grad in zip(buffers, grads):
            if grad is not None and torch.isfinite(grad).all():
                buffer.add_(grad.detach(), alpha=float(weight))

    def _accumulate_separate_grads(
        self,
        ppo_head: torch.Tensor,
        aux_w: torch.Tensor,
        value_loss: torch.Tensor,
        *,
        weight: float,
    ) -> None:
        """Accumulate PPO, auxiliary, and critic gradients independently."""
        ppo_grads = torch.autograd.grad(
            ppo_head, self._sep_policy_params, retain_graph=True,
            allow_unused=True,
        )
        aux_grads = (
            torch.autograd.grad(
                aux_w, self._sep_aux_params, retain_graph=True,
                allow_unused=True,
            )
            if aux_w.requires_grad else tuple(None for _ in self._sep_aux_params)
        )
        value_grads = torch.autograd.grad(
            self.p.value_loss_coef * value_loss,
            self._sep_value_params, retain_graph=False,
            allow_unused=True,
        )
        self._accumulate_grad_list(self._sep_policy_grads, ppo_grads, weight)
        self._accumulate_grad_list(self._sep_aux_grads, aux_grads, weight)
        self._accumulate_grad_list(self._sep_value_grads, value_grads, weight)

    @staticmethod
    def _grad_list_norm(grads) -> torch.Tensor:
        if not grads:
            return torch.tensor(0.0)
        total = grads[0].new_zeros((), dtype=torch.float64)
        for grad in grads:
            total = total + grad.double().square().sum()
        return total.sqrt()

    def _finish_separate_grads(self) -> dict[str, float]:
        """Clip each objective globally, then add shared LoRA contributions."""
        limit = float(self.p.max_grad_norm)
        ppo_norm = self._grad_list_norm(self._sep_policy_grads)
        aux_norm = self._grad_list_norm(self._sep_aux_grads)
        value_norm = self._grad_list_norm(self._sep_value_grads)

        def coefficient(norm: torch.Tensor) -> float:
            value = float(norm)
            return min(1.0, limit / max(value, 1.0e-12))

        ppo_clip = coefficient(ppo_norm)
        aux_clip = coefficient(aux_norm)
        value_clip = coefficient(value_norm)
        for parameter, grad in zip(self._sep_policy_params, self._sep_policy_grads):
            parameter.grad = grad.mul(ppo_clip)
        for parameter, grad in zip(self._sep_aux_params, self._sep_aux_grads):
            scaled = grad.mul(aux_clip)
            parameter.grad = (
                scaled if parameter.grad is None else parameter.grad + scaled
            )
        for parameter, grad in zip(self._sep_value_params, self._sep_value_grads):
            parameter.grad = grad.mul(value_clip)

        stats = {
            "separate_ppo_grad_norm": float(ppo_norm),
            "separate_aux_grad_norm": float(aux_norm),
            "separate_value_grad_norm": float(value_norm),
            "separate_ppo_clip": ppo_clip,
            "separate_aux_clip": aux_clip,
            "separate_value_clip": value_clip,
        }
        for name in (
            "_sep_policy_params", "_sep_aux_params", "_sep_value_params",
            "_sep_policy_grads", "_sep_aux_grads", "_sep_value_grads",
        ):
            delattr(self, name)
        return stats

    def _update_critic_only(self, storage: PpoStorage) -> dict[str, float]:
        """Fit Bellman/GAE returns while leaving every policy parameter frozen."""
        p = self.p
        flat = storage.flat
        returns = flat["returns"]
        old_values = flat["values"]
        if not torch.isfinite(returns).all():
            return {
                "surrogate": 0.0, "value": float("nan"), "entropy": 0.0,
                "aux": 0.0, "kl": 0.0, "clip_frac": 0.0, "ratio_dev": 0.0,
                "orth_cos": 0.0, "orth_remain": 0.0,
                "orth_cost_norm": 0.0, "orth_rl_norm": 0.0,
                "param_cos": 0.0, "param_cos_distance": 1.0,
                "param_cost_norm": 0.0, "param_rl_norm": 0.0,
                "orth_cost_scale": 0.0, "orth_traj_scale": 0.0,
                "orth_leak": 0.0, "orth_para_frac": 0.0,
                "orth_skipped": 1.0, "trust_before": 0.0,
                "trust_after": 0.0, "trust_alpha": 0.0,
                "trust_rejected": 0.0, "learning_rate": self.learning_rate,
                "lr_floor_hits": 0.0,
                "noise_std_mean": float(self.actor.current_std().detach().mean()),
                "noise_std_max_dim": float(self.actor.current_std().detach().max()),
                "kl_early_stops": 0.0, "critic_explained_variance": float("nan"),
            }
        losses = []
        for _ in range(int(p.num_learning_epochs)):
            for idx in storage.minibatches(p.num_mini_batches):
                self.optimizer.zero_grad(set_to_none=True)
                value = self.actor.critic(flat["critic_obs"].index_select(0, idx))
                target = returns.index_select(0, idx)
                old = old_values.index_select(0, idx)
                if p.use_clipped_value_loss:
                    clipped = old + (value - old).clamp(
                        -p.clip_param, p.clip_param,
                    )
                    loss = torch.max(
                        (value - target).square(),
                        (clipped - target).square(),
                    ).mean()
                else:
                    loss = (value - target).square().mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.critic.parameters(), p.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        with torch.no_grad():
            predicted = self.actor.critic(flat["critic_obs"])
            residual_var = (returns - predicted).var(unbiased=False)
            return_var = returns.var(unbiased=False)
            explained = 1.0 - residual_var / return_var.clamp_min(1.0e-12)
        return {
            "surrogate": 0.0,
            "value": float(sum(losses) / max(len(losses), 1)),
            "entropy": 0.0, "aux": 0.0, "kl": 0.0,
            "clip_frac": 0.0, "ratio_dev": 0.0,
            "orth_cos": 0.0, "orth_remain": 0.0,
            "orth_cost_norm": 0.0, "orth_rl_norm": 0.0,
            "param_cos": 0.0, "param_cos_distance": 1.0,
            "param_cost_norm": 0.0, "param_rl_norm": 0.0,
            "orth_cost_scale": 0.0, "orth_traj_scale": 0.0,
            "orth_leak": 0.0, "orth_para_frac": 0.0,
            "orth_skipped": 0.0, "trust_before": 0.0,
            "trust_after": 0.0, "trust_alpha": 0.0,
            "trust_rejected": 0.0, "learning_rate": self.learning_rate,
            "lr_floor_hits": 0.0,
            "noise_std_mean": float(self.actor.current_std().detach().mean()),
            "noise_std_max_dim": float(self.actor.current_std().detach().max()),
            "kl_early_stops": 0.0,
            "critic_explained_variance": float(explained),
            "critic_return_mean": float(returns.mean()),
            "critic_return_std": float(returns.std(unbiased=False)),
            "critic_value_mean": float(predicted.mean()),
        }

    def _rl_only_params(self) -> list[nn.Parameter]:
        """Parameters PPO owns that the traj-space combine must not touch.

        ``std`` is not a function of the plan; the mean-correction linear reads
        a decoder hidden state but is not in ``traj``'s graph.  Both get the
        raw PPO gradient.  LoRA is updated only through ``traj.backward``.
        """
        params: list[nn.Parameter] = [self.actor.std]
        if self.actor.mean_correction is not None:
            params.extend(self.actor.mean_correction.parameters())
        return params

    def _lora_params(self) -> list[nn.Parameter]:
        return [
            parameter for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad
            and (".lora_a" in name or ".lora_b" in name)
        ]

    @torch.no_grad()
    def _trust_aux(self, flat: dict, idx: torch.Tensor, micro: int) -> float:
        total = 0.0
        n = int(idx.numel())
        for start in range(0, n, micro):
            sel = idx[start : start + micro]
            weight = float(sel.numel()) / float(max(n, 1))
            inputs = MpcInputs(
                init=flat["init"].index_select(0, sel),
                state=flat["state"].index_select(0, sel),
                ctx=index_dataclass(flat["ctx"], sel),
                hit_index=flat["hit_index"].index_select(0, sel),
            )
            plan = self.actor.plan(
                inputs,
                encoding=index_dataclass(flat["encoding"], sel)
                if "encoding" in flat else None,
            )
            value = self._aux_cost(
                plan.traj, plan.elem_mask, inputs.state,
                index_dataclass(flat["strike"], sel),
            )
            total += weight * float(value)
        return total

    @torch.no_grad()
    def _apply_lora_trust_region(
        self,
        flat: dict,
        idx: torch.Tensor,
        micro: int,
        *,
        baseline: float,
        params: list[nn.Parameter],
        base: list[torch.Tensor],
        optimizer_state: dict,
    ) -> dict[str, float]:
        cfg = self.cfg.orth
        proposed = [parameter.detach().clone() for parameter in params]
        limit = baseline * (1.0 + float(cfg.trust_rel_tol)) + float(cfg.trust_abs_tol)
        accepted_alpha = 0.0
        after = float("inf")
        alpha = 1.0
        for _ in range(int(cfg.trust_backtracks)):
            for parameter, origin, candidate in zip(params, base, proposed):
                parameter.copy_(origin + alpha * (candidate - origin))
            after = self._trust_aux(flat, idx, micro)
            if math.isfinite(after) and after <= limit:
                accepted_alpha = alpha
                break
            alpha *= float(cfg.trust_shrink)
        rejected = accepted_alpha == 0.0
        if rejected:
            for parameter, origin in zip(params, base):
                parameter.copy_(origin)
                if parameter in optimizer_state:
                    self.optimizer.state[parameter] = copy.deepcopy(
                        optimizer_state[parameter]
                    )
                else:
                    self.optimizer.state.pop(parameter, None)
            after = baseline
        return {
            "trust_before": float(baseline),
            "trust_after": float(after),
            "trust_alpha": float(accepted_alpha),
            "trust_rejected": float(rejected),
        }

    def _chunk_loss(self, flat: dict, sel: torch.Tensor, adv_all: torch.Tensor,
                    *, weight: float = 1.0) -> dict:
        p = self.p
        actor = self.actor
        w = float(weight)

        inputs = MpcInputs(
            init=flat["init"].index_select(0, sel),
            state=flat["state"].index_select(0, sel),
            ctx=index_dataclass(flat["ctx"], sel),
            hit_index=flat["hit_index"].index_select(0, sel),
        )
        plan = actor.plan(
            inputs, encoding=index_dataclass(flat["encoding"], sel)
            if "encoding" in flat else None)
        dist = actor.update_distribution(plan.mean)

        actions = flat["actions"].index_select(0, sel)
        old_logp = flat["log_probs"].index_select(0, sel)
        old_mu = flat["mu"].index_select(0, sel)
        old_sigma = flat["sigma"].index_select(0, sel)
        adv = adv_all.index_select(0, sel)
        returns = flat["returns"].index_select(0, sel)
        old_values = flat["values"].index_select(0, sel)

        logp = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        mu, sigma = dist.mean, dist.stddev

        # ---- surrogate ------------------------------------------------------
        ratio = torch.exp(logp - old_logp)
        surrogate = -adv * ratio
        clipped = -adv * ratio.clamp(1.0 - p.clip_param, 1.0 + p.clip_param)
        surrogate_loss = torch.max(surrogate, clipped).mean()

        # ---- value ----------------------------------------------------------
        value = actor.critic(flat["critic_obs"].index_select(0, sel))
        if p.use_clipped_value_loss:
            v_clipped = old_values + (value - old_values).clamp(
                -p.clip_param, p.clip_param)
            value_loss = torch.max((value - returns).square(),
                                   (v_clipped - returns).square()).mean()
        else:
            value_loss = (value - returns).square().mean()

        # ---- the analytic anchor -------------------------------------------
        aux, waist_consistency, waist_mae, aux_w = self._combined_aux(
            flat, sel, plan, inputs,
        )
        actor_objective = self.actor_loss_weight * surrogate_loss
        ppo_head = self.actor_loss_weight * (
            surrogate_loss - p.entropy_coef * entropy.mean()
        )
        loss = ppo_head + p.value_loss_coef * value_loss + aux_w

        orth_stats = {"orth_cos": 0.0, "orth_remain": 0.0,
                      "orth_cost_norm": 0.0, "orth_rl_norm": 0.0,
                      "param_cos": 0.0, "param_cos_distance": 1.0,
                      "param_cost_norm": 0.0, "param_rl_norm": 0.0,
                      "separate_ppo_grad_norm": 0.0,
                      "separate_aux_grad_norm": 0.0,
                      "separate_value_grad_norm": 0.0,
                      "separate_ppo_clip": 1.0,
                      "separate_aux_clip": 1.0,
                      "separate_value_clip": 1.0,
                      "trust_before": 0.0, "trust_after": 0.0,
                      "trust_alpha": 0.0, "trust_rejected": 0.0,
                      "orth_cost_scale": 0.0, "orth_traj_scale": 0.0,
                      "orth_leak": 0.0, "orth_para_frac": 0.0,
                      "orth_skipped": 0.0}
        if self.cfg.orth.enabled:
            self._backward_orth(plan.traj, actor_objective, ppo_head, aux_w,
                                value_loss, weight=w, stats=orth_stats)
        else:
            self._accumulate_separate_grads(
                ppo_head, aux_w, value_loss, weight=w,
            )

        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.square() + (old_mu - mu).square())
                / (2.0 * sigma.square()) - 0.5, dim=-1).mean()
            clip_frac = ((ratio - 1.0).abs() > p.clip_param).float().mean()
            ratio_dev = (ratio - 1.0).abs().mean()

        return {"loss": loss.detach(), "surrogate": surrogate_loss.detach(),
                "value": value_loss.detach(), "entropy": entropy.mean().detach(),
                "aux": aux.detach(),
                "aux_objective": aux_w.detach(),
                "waist_consistency": waist_consistency.detach(),
                "waist_consistency_mae_rad": waist_mae.detach(),
                "kl": kl, "clip_frac": clip_frac,
                "ratio_dev": ratio_dev, **orth_stats}

    def _backward_orth(
        self,
        traj: torch.Tensor,
        surrogate_loss: torch.Tensor,
        ppo_head: torch.Tensor,
        aux_w: torch.Tensor,
        value_loss: torch.Tensor,
        *,
        weight: float,
        stats: dict[str, float],
    ) -> None:
        """Two-backward combine: cost + always-orth RL at waypoint 0.

        Decoder LoRA sees the combined trajectory gradient.  ``std`` sees raw
        PPO; the mean-correction head sees norm-balanced physics plus the PPO
        component orthogonal to physics.  The critic sees only value loss.
        """
        p = self.p
        zeros = torch.zeros_like(traj)
        g_rl = torch.autograd.grad(
            surrogate_loss, traj, retain_graph=True, allow_unused=True)[0]
        if g_rl is None:
            g_rl = zeros
        if aux_w.requires_grad:
            g_cost = torch.autograd.grad(
                aux_w, traj, retain_graph=True, allow_unused=True)[0]
            if g_cost is None:
                g_cost = zeros
        else:
            g_cost = zeros
        if not torch.isfinite(g_rl).all() or not torch.isfinite(g_cost).all():
            stats["orth_skipped"] = 1.0
            return

        # The trajectory-space cosine above is local to waypoint 0.  Report
        # the actual conflict after both objectives pass through the shared
        # decoder Jacobian into the trainable LoRA subspace as well.  This is a
        # diagnostic only: the executed combine remains in trajectory space.
        lora_params = [
            param for name, param in self.actor.named_parameters()
            if param.requires_grad and (".lora_a" in name or ".lora_b" in name)
        ]
        rl_param = torch.autograd.grad(
            surrogate_loss, lora_params, retain_graph=True, allow_unused=True,
        )
        cost_param = torch.autograd.grad(
            aux_w, lora_params, retain_graph=True, allow_unused=True,
        ) if aux_w.requires_grad else tuple(None for _ in lora_params)
        dot = traj.new_zeros((), dtype=torch.float64)
        rl_sq = traj.new_zeros((), dtype=torch.float64)
        cost_sq = traj.new_zeros((), dtype=torch.float64)
        for gr, gc in zip(rl_param, cost_param):
            if gr is not None:
                rl_sq = rl_sq + gr.detach().double().square().sum()
            if gc is not None:
                cost_sq = cost_sq + gc.detach().double().square().sum()
            if gr is not None and gc is not None:
                dot = dot + (gr.detach().double() * gc.detach().double()).sum()
        rl_norm_p = rl_sq.sqrt()
        cost_norm_p = cost_sq.sqrt()
        denom_p = (rl_norm_p * cost_norm_p).clamp_min(float(self.cfg.orth.eps))
        param_cos = dot / denom_p
        stats.update({
            "param_cos": float(param_cos),
            "param_cos_distance": float(1.0 - param_cos),
            "param_rl_norm": float(rl_norm_p),
            "param_cost_norm": float(cost_norm_p),
        })

        g, orth_stats = combine_wp0_orth(
            g_rl, g_cost,
            cost_to_rl_norm=self.cfg.orth.cost_to_rl_norm,
            rl_to_cost_norm=self.cfg.orth.rl_to_cost_norm,
            traj_grad_scale=self.cfg.orth.traj_grad_scale,
            wp0_leak=self.cfg.orth.wp0_leak,
            eps=self.cfg.orth.eps)
        stats.update(orth_stats)
        if not torch.isfinite(g).all():
            stats["orth_skipped"] = 1.0
            return
        traj.backward(weight * g, retain_graph=True)

        rl_params = self._rl_only_params()
        # ``std`` has no physics path.  The mean-correction head does, now that
        # aux_traj[0] is plan.mean.  Give those parameters the same
        # cost-plus-orthogonal-PPO treatment used for trajectory/LoRA instead
        # of letting the raw PPO gradient bypass the auxiliary loss.
        head_params = self._bal_head_params
        cost_head = (
            torch.autograd.grad(aux_w, head_params, retain_graph=True,
                                allow_unused=True)
            if aux_w.requires_grad and head_params
            else tuple(None for _ in head_params)
        )
        ppo_head_grads = torch.autograd.grad(
            ppo_head, rl_params, retain_graph=False, allow_unused=True)
        # std is the first RL-only parameter and has no physics path.
        grad_std = ppo_head_grads[0]
        if grad_std is not None and torch.isfinite(grad_std).all():
            scaled = weight * grad_std
            self.actor.std.grad = (
                scaled if self.actor.std.grad is None
                else self.actor.std.grad + scaled
            )
        self._accumulate_grad_list(
            self._bal_head_ppo, ppo_head_grads[1:], weight)
        self._accumulate_grad_list(self._bal_head_aux, cost_head, weight)

        if torch.isfinite(value_loss):
            (weight * p.value_loss_coef * value_loss).backward()

    def _adapt_lr(self, kl_mean: float, *, allow_increase: bool = True) -> None:
        """rsl_rl's controller, with the floor made visible.

        The documented in-repo failure is not that this controller is wrong but
        that it rails: it runs ``epochs * minibatches`` times per iteration, each
        hit divides by 1.5, and once the learning rate reaches the floor the run
        silently stops learning while every other logged quantity looks normal.
        The floor hit is counted and surfaced so the trace is readable without
        patching anything.

        ``allow_increase=False`` blocks the *1.5 branch.  Used on the first
        minibatch of an iteration, whose KL is ~0 whenever replay matches.
        """
        p = self.p
        if kl_mean > p.desired_kl * 2.0:
            self.learning_rate = max(p.lr_min, self.learning_rate / 1.5)
            if self.learning_rate <= p.lr_min * (1.0 + 1e-9):
                self.lr_floor_hits += 1
        elif allow_increase and 0.0 < kl_mean < p.desired_kl / 2.0:
            self.learning_rate = min(p.lr_max, self.learning_rate * 1.5)
        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate


# =============================================================== collection ===
class RolloutDriver:
    """One control tick, assembled -- the PPO twin of ``isaac_rl.DrumMPCLoop``.

    The tick ORDER is theirs and is load bearing (their step 6 before step 7:
    after ``commit`` the goal has already switched, so a reward computed then
    would price the NEXT strike's geometry against the pose the plant reached
    for the previous one).  What is inserted is the Gaussian head between the
    plan and the command, and the storage write.
    """

    def __init__(
        self,
        *,
        actor: AmortizedPpoActor,
        mpc,
        schedule,
        reward,
        plant,
        cfg: PpoRunConfig,
        fk=None,
        on_done=None,
        safety=None,
        waist_terminal_reward_weight: float = 0.0,
        waist_terminal_reward_scale_rad: float = 0.10,
    ):
        self.actor = actor
        self.mpc = mpc
        self.sched = schedule
        self.reward = reward
        self.plant = plant
        self.cfg = cfg
        self.fk = fk
        self.on_done = on_done
        self.safety = safety
        self.waist_terminal_reward_weight = float(waist_terminal_reward_weight)
        self.waist_terminal_reward_scale_rad = float(waist_terminal_reward_scale_rad)
        if self.waist_terminal_reward_weight < 0.0:
            raise ValueError("waist terminal reward weight must be non-negative")
        if self.waist_terminal_reward_scale_rad <= 0.0:
            raise ValueError("waist terminal reward scale must be positive")
        self.device = actor.device_
        self.n = actor.n
        self.t_plan = 0.0
        self.t_plant = 0.0
        self.t_reward = 0.0
        # pd_sim grew a PID integral; Isaac's plant has no such parameter.
        try:
            import inspect
            self._plant_takes_integral = "integral" in inspect.signature(
                plant).parameters
        except (TypeError, ValueError):
            self._plant_takes_integral = False
        self._integral = None
        self._waist_anchor = torch.zeros(
            self.mpc.B, self.mpc.H, device=self.device, dtype=self.mpc.dtype,
        )
        self._waist_anchor_age = torch.full(
            (self.mpc.B,), self.mpc.H, device=self.device, dtype=torch.long,
        )
        self._waist_anchor_len = torch.full(
            (self.mpc.B,), self.mpc.H, device=self.device, dtype=torch.long,
        )
        self._waist_terminal_due = torch.zeros(
            self.mpc.B, device=self.device, dtype=torch.bool,
        )
        self._waist_terminal_target = torch.zeros(
            self.mpc.B, device=self.device, dtype=self.mpc.dtype,
        )
        self._yaw_i = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")

    @torch.no_grad()
    def _aligned_waist_anchor(
        self,
        plan_ref: torch.Tensor,
        elem_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shift the first plan to the current absolute MPC tick.

        A fresh anchor block begins on reset and whenever the previous block's
        horizon is exhausted.  At age ``a``, current waypoint ``j`` is compared
        with anchor waypoint ``a + j``.  This preserves the first plan's valid
        initial-velocity transient instead of forcing an immediate turn.
        """
        refresh = self._waist_anchor_age >= self._waist_anchor_len
        if bool(refresh.any()):
            self._waist_anchor[refresh] = plan_ref[refresh, :, self._yaw_i]
            self._waist_anchor_age[refresh] = 0
            lengths = (elem_mask[refresh, :, self._yaw_i] > 0).sum(1).long()
            self._waist_anchor_len[refresh] = lengths.clamp(1, self.mpc.H)
        terminal_index = (self._waist_anchor_len - 1).clamp(0, self.mpc.H - 1)
        self._waist_terminal_target = self._waist_anchor.gather(
            1, terminal_index.view(-1, 1),
        )[:, 0]
        self._waist_terminal_due = self._waist_anchor_age == terminal_index
        offsets = torch.arange(self.mpc.H, device=self.device).view(1, -1)
        absolute = self._waist_anchor_age.view(-1, 1) + offsets
        inside = absolute < self._waist_anchor_len.view(-1, 1)
        target = self._waist_anchor.gather(
            1, absolute.clamp_max(self.mpc.H - 1),
        )
        valid = inside & (elem_mask[:, :, self._yaw_i] > 0)
        return target, valid

    # ------------------------------------------------------ privileged obs --
    def privileged_obs(self, q, qd, mean, ctx: ContextInputs,
                       hit_index: torch.Tensor) -> torch.Tensor:
        """Assemble the critic's flat input.  Order matches ``PrivilegedObsSpec``."""
        spec = self.actor.obs_spec
        prev_cmd = self.mpc.prev_cmd
        active = (hit_index >= 0).to(q.dtype)
        tth = hit_index.clamp_min(0).to(q.dtype) * float(self.mpc.dt)
        blocks = [
            q, qd, prev_cmd, mean - q, prev_cmd - q,
            tth, active,
            ctx.goal_pos.flatten(1), ctx.goal_vel.flatten(1),
            ctx.goal_valid.to(q.dtype),
            ctx.future_gap.flatten(1),
        ]
        if spec.include_tips:
            blocks.append(self._tip_positions(q))
        obs = torch.cat([b.to(q.dtype) for b in blocks], dim=-1)
        if obs.shape[-1] != spec.dim:
            raise RuntimeError(
                f"privileged obs width {obs.shape[-1]} != spec {spec.dim}; the "
                f"block table in PrivilegedObsSpec is out of date")
        return obs

    # ---------------------------------------------------------- the plant --
    def _step_plant(self, prev_cmd, cmd, q, qd):
        """One plant tick, tolerant of both plant signatures.

        The two plants have drifted apart: Isaac's ``env.plant`` returns
        ``(q, qd)``, while ``amortized_opt.pd_sim.BatchedPDSim.step`` gained a
        PID integral term and now returns ``(q, qd, integral)``.  The sibling's
        ``DrumMPCLoop`` still unpacks two and therefore no longer runs against
        pd_sim -- reported, not patched, since ``isaac_rl`` is read-only here.

        The integral is threaded back only when the plant accepts it, so a
        ``ki > 0`` run keeps its accumulator instead of silently resetting it
        every control tick (which would make the I term a no-op).
        """
        if self._plant_takes_integral:
            out = self.plant(prev_cmd, cmd, q, qd, integral=self._integral)
        else:
            out = self.plant(prev_cmd, cmd, q, qd)
        if isinstance(out, tuple) and len(out) >= 3:
            self._integral = out[2]
        return out[0], out[1]

    def _tip_positions(self, q: torch.Tensor) -> torch.Tensor:
        if self.fk is None:
            return torch.zeros(q.shape[0], 6, device=q.device, dtype=q.dtype)
        from amortized_opt.fk import stick_tip_offset
        frames = self.fk.forward(q)              # {name: (R, p)}
        out = []
        for hand in ("left", "right"):
            off = torch.as_tensor(stick_tip_offset(hand), device=q.device,
                                  dtype=q.dtype).reshape(1, 3)
            R, p = frames[f"{hand}_wrist"]
            out.append(self.fk.points_world(R, p, off).reshape(q.shape[0], 3))
        return torch.cat(out, dim=-1)

    # ---------------------------------------------------------------- tick --
    @torch.no_grad()
    def tick(self, q, qd, storage: PpoStorage | None = None):
        cfg = self.cfg
        mpc, sched, actor = self.mpc, self.sched, self.actor

        t0 = time.perf_counter()
        ctx_strike = sched.strike_context()
        ctx_inputs = sched.context()
        hit_index = sched.hit_index()
        state = torch.cat([q, qd], dim=-1)

        res = mpc.plan(q, qd)             # deterministic feature extractor
        plan_ref = getattr(res, "plan_ref", None)
        if plan_ref is None:
            # Compatibility with the pinned/sibling MPC surface.  Its full
            # trajectory differs only at row zero after lead/slew; restore the
            # planner's un-led first waypoint from ``last_ref0``.
            plan_ref = res.traj.clone()
            plan_ref[:, 0] = mpc.last_ref0
        waist_anchor, waist_anchor_mask = self._aligned_waist_anchor(
            plan_ref, res.elem_mask,
        )
        # Replay must see the init refine actually consumed (waist-PR seed /
        # arm DLS mutate ``mpc.init`` in place) and the encoding it used.
        # Storing the pre-plan warm start and re-encoding from ``ctx`` is what
        # made first-minibatch KL ~0.07 before any Adam step.
        init_replay = mpc.init.clone()
        encoding = clone_encoding(getattr(mpc, "_enc", None))
        mean = mpc.last_ref0              # traj[:, 0] BEFORE the lead
        if actor.mean_correction is not None and actor._hidden is not None:
            mean = mean + cfg.actor.mean_correction_scale * actor.mean_correction(
                actor._hidden[:, 1])

        dist = actor.update_distribution(mean)
        if actor._ar1_active:
            eps = actor._sample_ar1(mean)
            action = mean + dist.stddev * eps
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        delta = action - mean

        cmd = executed_command(
            res.cmd, delta, prev_cmd=mpc.prev_cmd, step_max=mpc.step_max,
            q_lo=mpc.q_lo, q_hi=mpc.q_hi)
        critic_obs = self.privileged_obs(q, qd, mean, ctx_inputs, hit_index)
        value = actor.critic(critic_obs)
        t1 = time.perf_counter()

        engaged = None
        if self.safety is not None:
            cmd, engaged = self.safety.project(cmd, q, qd)

        q_before = q
        prev_cmd = mpc.prev_cmd
        q, qd = self._step_plant(prev_cmd, cmd, q, qd)
        t2 = time.perf_counter()

        fired = sched.advance()

        # The reference the tracking term scores the plant against.  Either way
        # it is the UN-LED reference: scoring against the executed (led) command
        # would price the lead itself as error and train the model to cancel its
        # own compensation (isaac_rl/mpc.py, ``last_ref0``).
        ref0 = action if cfg.track_reference == "sampled" else mean
        terms, _cs = self.reward.step(
            q_before=q_before, q_after=q, qd_after=qd, ctx=ctx_strike,
            hit_index=res.hit_index, fired=fired, ref0=ref0,
            traj=res.traj, elem_mask=res.elem_mask,
            q_plan0=q_before, v_plan0=None,
            drum_heads=ctx_strike.drum_head_centers,
            drum_axes=sched.geom.get("axis"), drum_radii=sched.drum_radius,
            drum_valid=sched.drum_valid, filter_active=engaged,
        )
        if self.waist_terminal_reward_weight > 0.0:
            waist_bonus, waist_error = terminal_waist_reward(
                q, self._waist_terminal_target, self._waist_terminal_due,
                yaw_index=self._yaw_i,
                weight=self.waist_terminal_reward_weight,
                scale_rad=self.waist_terminal_reward_scale_rad,
            )
            terms.terms["waist_terminal"] = waist_bonus
            terms.raw["waist_terminal_error_rad"] = waist_error
            terms.counts["waist_terminal_due"] = self._waist_terminal_due.to(q.dtype)
            terms.total = terms.total + waist_bonus
        t3 = time.perf_counter()

        mpc.commit(cmd, q, fired, qd=qd)

        done = sched.done()
        if storage is not None:
            if encoding is None:
                raise RuntimeError(
                    "IsaacMPC.plan left no context encoding to replay; "
                    "without it the update re-encodes and first-minibatch KL "
                    "is not zero")
            storage.add(
                init=init_replay, state=state, hit_index=hit_index,
                ctx=ctx_inputs, encoding=encoding, strike=ctx_strike,
                critic_obs=critic_obs,
                actions=action, log_probs=log_prob, mu=mean.clone(),
                sigma=dist.stddev.clone(), values=value,
                rewards=terms.total, dones=done,
                waist_anchor=waist_anchor,
                waist_anchor_mask=waist_anchor_mask,
            )

        if bool(done.any()) and self.on_done is not None:
            q, qd = self.on_done(done.nonzero(as_tuple=True)[0], q, qd)
        self._waist_anchor_age.add_(1)
        self._waist_anchor_age.masked_fill_(done, self.mpc.H)
        self._waist_anchor_len.masked_fill_(done, self.mpc.H)
        actor.reset(done)

        self.t_plan += t1 - t0
        self.t_plant += t2 - t1
        self.t_reward += t3 - t2
        return q, qd, terms

    @torch.no_grad()
    def collect(self, q, qd, steps: int, storage: PpoStorage):
        logs: dict[str, list[float]] = {}
        for _ in range(int(steps)):
            q, qd, terms = self.tick(q, qd, storage)
            for k, v in terms.log().items():
                logs.setdefault(k, []).append(v)
        return q, qd, {k: float(np.mean(v)) for k, v in logs.items()}
