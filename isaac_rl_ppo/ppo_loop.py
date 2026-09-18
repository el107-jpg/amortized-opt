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
    "index_dataclass",
    "stack_dataclass",
]


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
                  "dones"):
            getattr(self, k).clear()
        self.returns = None
        self.advantages = None
        self.flat = {}

    # ------------------------------------------------------------ flatten --
    def finalize(self) -> None:
        """Stack ``(T, B, ...)`` and flatten to ``(T * B, ...)``."""
        cat = lambda xs: torch.stack(xs, 0).flatten(0, 1)   # noqa: E731
        self.flat = {
            "init": cat(self.init),
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
    ):
        self.actor = actor
        self.cfg = cfg
        self.p = cfg.ppo
        self.device = torch.device(device)
        self.learning_rate = float(self.p.learning_rate)
        self.optimizer = torch.optim.Adam(actor.trainable_parameters(),
                                          lr=self.learning_rate)
        self.cost = cost
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
        ct = self.cost(q0, traj, v0, strike, lengths=lengths, elem_mask=elem_mask)
        terms = self.cfg.aux.terms
        if terms is None:
            return ct.total.mean()
        picked = [ct.terms[k] for k in terms if k in ct.terms]
        if not picked:
            raise KeyError(
                f"aux.terms {terms} matched no FusedCost group; available: "
                f"{sorted(ct.terms)}")
        return torch.stack(picked, 0).sum(0).mean()

    # -------------------------------------------------------------- update --
    def update(self, storage: PpoStorage) -> dict[str, float]:
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

        acc = {k: 0.0 for k in ("surrogate", "value", "entropy", "aux", "kl",
                                "clip_frac", "ratio_dev", "orth_cos",
                                "orth_remain", "orth_cost_norm", "orth_rl_norm",
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

                for s in range(0, mb, micro):
                    sel = idx[s: s + micro]
                    w = sel.shape[0] / mb          # exact chunk reweighting, so
                    # gradient accumulation is numerically identical to one pass
                    part = self._chunk_loss(flat, sel, adv_all, weight=w)
                    for k in stats:
                        if k in part:
                            stats[k] += w * float(part[k])

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

                nn.utils.clip_grad_norm_(self.actor.trainable_parameters(),
                                         p.max_grad_norm)
                self.optimizer.step()
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
        out["noise_std_mean"] = float(self.actor.current_std().mean())
        out["noise_std_max_dim"] = float(self.actor.current_std().max())
        out["kl_early_stops"] = float(kl_early_stops)
        return out

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
        aux = self._aux_cost(plan.traj, plan.elem_mask, inputs.state,
                             index_dataclass(flat["strike"], sel))
        aux_w = self.cfg.aux.weight * aux
        ppo_head = surrogate_loss - p.entropy_coef * entropy.mean()
        loss = ppo_head + p.value_loss_coef * value_loss + aux_w

        orth_stats = {"orth_cos": 0.0, "orth_remain": 0.0,
                      "orth_cost_norm": 0.0, "orth_rl_norm": 0.0,
                      "orth_cost_scale": 0.0, "orth_traj_scale": 0.0,
                      "orth_leak": 0.0, "orth_para_frac": 0.0,
                      "orth_skipped": 0.0}
        if self.cfg.orth.enabled:
            self._backward_orth(plan.traj, surrogate_loss, ppo_head, aux_w,
                                value_loss, weight=w, stats=orth_stats)
        else:
            (w * loss).backward()

        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.square() + (old_mu - mu).square())
                / (2.0 * sigma.square()) - 0.5, dim=-1).mean()
            clip_frac = ((ratio - 1.0).abs() > p.clip_param).float().mean()
            ratio_dev = (ratio - 1.0).abs().mean()

        return {"loss": loss.detach(), "surrogate": surrogate_loss.detach(),
                "value": value_loss.detach(), "entropy": entropy.mean().detach(),
                "aux": aux.detach(), "kl": kl, "clip_frac": clip_frac,
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

        Decoder LoRA sees the combined traj gradient.  ``std`` / mean-correction
        see raw PPO.  The critic sees only the value loss.
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
        head_grads = torch.autograd.grad(
            ppo_head, rl_params, retain_graph=False, allow_unused=True)
        for param, grad in zip(rl_params, head_grads):
            if grad is None or not torch.isfinite(grad).all():
                continue
            scaled = weight * grad
            param.grad = scaled if param.grad is None else param.grad + scaled

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
        t3 = time.perf_counter()

        mpc.commit(cmd, q, fired)

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
            )

        if bool(done.any()) and self.on_done is not None:
            q, qd = self.on_done(done.nonzero(as_tuple=True)[0], q, qd)
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
