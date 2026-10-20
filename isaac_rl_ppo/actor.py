"""The amortized optimizer as a PPO actor-critic (executed-prefix formulation).

The action is the **executed first waypoint**, 17 dims -- not the 50x17
trajectory.  That single choice is what makes standard PPO apply at all; the
rejected alternative and its five failure mechanisms are summarised in
``README.md`` and argued in the approved analysis.

The pipeline for one control tick
---------------------------------

1. The pretrained transformer runs its MPC pass as a **deterministic feature
   extractor**: warm start (previous plan advanced one tick, arm columns
   re-seeded when a hit completes) + measured state + context -> one refinement
   -> a 50x17 plan.  Deterministic given ``(obs, warm start)`` is a PPO
   VALIDITY requirement, not a nicety: any stochasticity inside refinement
   would make the Gaussian head's log-prob something other than the true
   sampling density.  :func:`AmortizedPpoActor.plan` is asserted deterministic
   by ``tests/test_determinism.py``.
2. A Gaussian head over the plan's first waypoint: ``mean = traj[:, 0]``
   (optionally plus a small zero-initialised linear correction read off the
   decoder's waypoint-0 hidden state), ``std`` a learned state-independent
   per-dim parameter.  Log-prob and entropy are stock ``Normal`` -- the PPO
   math is exact and untouched.
3. The sample is expressed in un-led first-waypoint coordinates. Its offset
   from the raw first waypoint is applied to the deployed command AFTER the
   per-joint reference lead, slew bound and joint-box clamp (see
   :func:`executed_command`). This sends the mean correction and exploration
   to the plant while keeping the fitted lead outside the policy distribution.

What is trainable
-----------------

LoRA deltas on the decoder's attention out-projections and FFN matrices, the
Gaussian head's ``std``, the optional mean correction, and the critic.  The
19.3 M-parameter backbone is frozen; :func:`AmortizedPpoActor.assert_frozen`
turns "frozen" into a checked property rather than an intention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.model import AmortizedOptimizer, ContextEncoding, ModelConfig
from amortized_opt.tokenizer import ContextInputs, WaypointMasks, build_waypoint_masks
from amortized_opt.warmstart import arm_slice, waist_slice

from .cfg import ActorConfig, CriticConfig

__all__ = [
    "AR1_GROUPS",
    "AmortizedPpoActor",
    "Critic",
    "LoRALinear",
    "MpcInputs",
    "PlanOutput",
    "PrivilegedObsSpec",
    "apply_lora",
    "build_ar1_rho",
    "executed_command",
    "register_with_rsl_rl",
    "waypoint_masks",
]

Normal.set_default_validate_args(False)

#: Joint-group membership for the AR(1) time constants, derived from the joint
#: NAMES rather than from positional group sizes.  The MLP stack's
#: ``build_ar1_rho`` slices positionally with ``arm_group_sizes = (3, 1, 3)``;
#: ``G1_UPPER_BODY_JOINTS`` happens to lay out waist(3) | left(3,1,3) |
#: right(3,1,3) identically, and :func:`build_ar1_rho` asserts the two agree.
AR1_GROUPS: tuple[str, ...] = ("waist", "shoulder", "elbow", "wrist")


def _joint_group(name: str) -> str:
    for g in ("waist", "shoulder", "elbow", "wrist"):
        if g in name:
            return g
    raise ValueError(f"joint {name!r} belongs to no AR(1) group")


def build_ar1_rho(
    tau_s: dict[str, float],
    control_dt: float,
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
) -> tuple[torch.Tensor, list[tuple[str, float, float]]]:
    """Per-action AR(1) pole vector from per-group correlation times.

    ``rho_j = exp(-control_dt / tau_group(j))`` -- a group decorrelates with
    time constant tau REGARDLESS of the control rate, which a raw rho cannot
    promise.  This matters here specifically: the MLP stack's shipped rho values
    were computed at 60 Hz and this project runs at 50 Hz (resolved decision 1),
    so carrying rho across would silently lengthen every correlation time by
    20 %.  Carrying **tau** across is exactly right.

    ``tau <= 0`` yields ``rho = 0``, i.e. plain white noise for that group.
    Ported from ``factored_actor_critic.build_ar1_rho`` (:673-727), reindexed by
    joint name.
    """
    if control_dt <= 0.0:
        raise ValueError(f"control_dt must be > 0, got {control_dt}")
    unknown = set(tau_s) - set(AR1_GROUPS)
    missing = set(AR1_GROUPS) - set(tau_s)
    if unknown or missing:
        raise ValueError(
            f"AR(1) tau keys must be exactly {AR1_GROUPS}; "
            f"unknown={sorted(unknown)} missing={sorted(missing)}")

    rho = torch.zeros(len(joint_names))
    table: list[tuple[str, float, float]] = []
    for i, name in enumerate(joint_names):
        g = _joint_group(name)
        tau = float(tau_s[g])
        r = 0.0 if tau <= 0.0 else math.exp(-control_dt / tau)
        rho[i] = r
        table.append((name, tau, r))

    # The positional layout the MLP stack assumes: waist(3) | arm(3,1,3) x 2.
    expect = ["waist"] * 3 + (["shoulder"] * 3 + ["elbow"] + ["wrist"] * 3) * 2
    got = [_joint_group(n) for n in joint_names]
    if got != expect:
        raise RuntimeError(
            "G1_UPPER_BODY_JOINTS no longer lays out waist(3)|arm(3,1,3)x2; the "
            f"ported AR(1) group construction assumes it does.\n  got    {got}\n"
            f"  expect {expect}")
    return rho, table


# ==================================================================== LoRA ====
class LoRALinear(nn.Module):
    """``nn.Linear`` + a frozen-base rank-``r`` delta.

    ``y = base(x) + (alpha / r) * B(A(x))``, ``A`` Kaiming-init, ``B`` zero-init,
    so the wrapped layer is EXACTLY the base layer at step 0.  That is what lets
    the fine-tune start from the pretrained policy's behaviour bit for bit --
    and it is what makes the determinism test meaningful before any training.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(
            rank, base.in_features, device=device, dtype=dtype,
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, rank, device=device, dtype=dtype,
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep dropout on the ordinary adapter path.  ``weight`` below exists
        # for projection heads that the backbone evaluates with ``F.linear``
        # directly (TrajectoryDecoder's fp32 head path).
        if isinstance(self.dropout, nn.Identity):
            return F.linear(x, self.weight, self.bias)
        out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        mask = getattr(self, "output_mask", None)
        if mask is not None:
            delta = delta * mask.to(device=delta.device, dtype=delta.dtype)
        return out + self.scaling * delta

    @property
    def weight(self) -> torch.Tensor:
        delta = self.lora_b @ self.lora_a
        mask = getattr(self, "output_mask", None)
        if mask is not None:
            delta = delta * mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return self.base.weight + self.scaling * delta

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def restrict_output_rows(self, rows: tuple[int, ...]) -> None:
        """Make the LoRA delta identically zero outside selected output rows."""
        clean = tuple(sorted({int(row) for row in rows}))
        if not clean or clean[0] < 0 or clean[-1] >= self.base.out_features:
            raise ValueError(
                f"LoRA output rows {clean} invalid for width {self.base.out_features}"
            )
        mask = torch.zeros(self.base.out_features)
        mask[list(clean)] = 1.0
        self.register_buffer("output_mask", mask, persistent=False)

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}")


def apply_lora(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    targets: tuple[str, ...],
    dropout: float = 0.0,
) -> list[str]:
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets``.

    Returns the dotted paths that were wrapped, so the caller can log them and
    a test can assert the expected set.  Matching is on the ATTRIBUTE name
    (``o_proj``, ``w12``, ``w3``), not on a substring of the full path, so
    ``q_proj`` is never caught by a ``proj`` target by accident.
    """
    wrapped: list[str] = []
    for mod_name, module in list(root.named_modules()):
        for attr, child in list(module.named_children()):
            if attr in targets and isinstance(child, nn.Linear):
                setattr(module, attr, LoRALinear(child, rank, alpha, dropout))
                wrapped.append(f"{mod_name}.{attr}" if mod_name else attr)
    return wrapped


# ============================================================ MPC plumbing ====
@dataclass
class MpcInputs:
    """Everything the deterministic feature extractor consumes for one tick.

    Stored per rollout step so the PPO update can recompute the mean WITH
    gradient from exactly the inputs collection saw.  This is the whole reason
    the actor does not take a flat observation vector: the policy's input is a
    stateful warm start plus a structured context, and flattening it into
    rsl_rl's ``RolloutStorage`` would either lose the warm start or force a
    reconstruction that is wrong in a way nothing would catch.
    """

    init: torch.Tensor            # (B, H, 17) warm start, BEFORE masking
    state: torch.Tensor           # (B, 34) = [q, qd]
    ctx: ContextInputs
    hit_index: torch.Tensor       # (B, 2) long

    @property
    def batch(self) -> int:
        return int(self.init.shape[0])


@dataclass
class PlanOutput:
    """One deterministic MPC pass."""

    mean: torch.Tensor            # (B, 17) the Gaussian mean
    traj: torch.Tensor            # (B, H, 17) the full emitted plan
    elem_mask: torch.Tensor       # (B, H, 17)
    raw_first: torch.Tensor       # (B, 17) traj[:, 0] before the mean correction
    hidden: torch.Tensor | None   # (B, d_model) waypoint-0 decoder hidden state


def waypoint_masks(
    hit_index: torch.Tensor,
    *,
    horizon: int,
    dt: float = CONTROL_DT,
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
    dtype: torch.dtype = torch.float32,
) -> WaypointMasks:
    """The shrinking-horizon masks from a hit index.  Mirrors ``IsaacMPC.masks``.

    Reimplemented here (four lines) rather than called through the sibling so
    the actor and its unit tests need no ``IsaacMPC`` instance -- and so a
    change to the sibling's method shows up as a numerical disagreement in
    ``tests/test_warm_start.py`` instead of silently re-shaping the masks the
    stored rollout is replayed under.
    """
    active = hit_index >= 0
    t = hit_index.clamp_min(0).to(dtype) * float(dt)
    return build_waypoint_masks(
        t[:, 0], t[:, 1], max_horizon=int(horizon), dt=float(dt),
        active_left=active[:, 0], active_right=active[:, 1],
        joint_names=joint_names,
    )


def executed_command(
    cmd_det: torch.Tensor,
    delta: torch.Tensor,
    *,
    prev_cmd: torch.Tensor,
    step_max: torch.Tensor | None,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
) -> torch.Tensor:
    """Turn the deterministic command + the exploration offset into a command.

    ``cmd_det`` is what the deployed path produced: ``lead -> slew -> box`` on
    the untouched plan. ``delta`` is the sampled un-led waypoint minus the
    planner's raw un-led waypoint, so it includes both the mean correction and
    exploration. The offset is added AFTERWARDS and the slew and box bounds are
    re-imposed on the result.

    Why after the lead and not on ``traj[:, 0]``:  the lead is an interpolation
    between waypoints 0 and 1 with a PER-JOINT fraction, so perturbing waypoint
    0 by ``d`` moves the command by ``(1 - f_j) * d`` -- a different effective
    sigma on every joint, and one that changes whenever the fit is re-run.  The
    log-prob would still be exact, but the sigma the KL controller reasons about
    would no longer be the sigma the plant sees.  Adding the offset to the
    finished command keeps "one unit of action" equal to "one radian at the
    joint" for all 17 dims, which is what the shared 17-dim KL budget assumes.

    Re-clamping means the EXECUTED command can differ from the sampled action
    near a limit.  That is ordinary action clipping: the log-prob stays the
    density of the unclipped sample, which is what PPO's ratio requires.
    """
    cmd = cmd_det + delta
    if step_max is not None:
        cmd = prev_cmd + torch.clamp(cmd - prev_cmd, -step_max, step_max)
    return torch.clamp(cmd, q_lo, q_hi)


# ================================================================== critic ====
@dataclass
class PrivilegedObsSpec:
    """The critic's flat input, block by block.

    Asymmetric and deliberately cheap: the actor is the expensive half, and the
    critic only has to predict a scalar.  Every block is something the loop
    already holds, so building it costs no extra model pass (the tip block costs
    one ``ChainFK`` forward, which is ~0.1 ms at these batch sizes).
    """

    n_dof: int = N_DOF
    num_future: int = 3
    include_tips: bool = True

    @property
    def blocks(self) -> tuple[tuple[str, int], ...]:
        b: list[tuple[str, int]] = [
            ("q", self.n_dof),                    # measured joint position
            ("qd", self.n_dof),                   # measured joint velocity
            ("prev_cmd", self.n_dof),             # last executed command
            ("plan0_minus_q", self.n_dof),        # where the plan wants to go
            ("cmd_minus_q", self.n_dof),          # standing servo error
            ("hit_tth", 2),                       # time-to-hit per hand, s
            ("hit_active", 2),                    # per-hand active flag
            ("goal_pos", 6),                      # aim point per hand
            ("goal_vel", 6),                      # commanded impact velocity
            ("goal_valid", 2),
            # the next few beats, as HIT-RELATIVE gaps from the current goal's
            # strike (``ContextInputs.future_gap``) -- same width, but it no
            # longer slides with the tick.
            ("future_gap", 2 * self.num_future),
        ]
        if self.include_tips:
            b.append(("tip_pos", 6))              # both stick tips, pelvis frame
        return tuple(b)

    @property
    def dim(self) -> int:
        return sum(n for _, n in self.blocks)


class RunningNorm(nn.Module):
    """Welford running mean/var normaliser, updated only during collection."""

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps = float(eps)
        self.clip = float(clip)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n = torch.tensor(float(x.shape[0]), device=x.device)
        if n <= 0:
            return
        m, v = x.mean(0), x.var(0, unbiased=False)
        delta = m - self.mean
        tot = self.count + n
        self.mean += delta * (n / tot)
        m_a = self.var * self.count
        m_b = v * n
        self.var = (m_a + m_b + delta.square() * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return out.clamp(-self.clip, self.clip)


class Critic(nn.Module):
    """Plain MLP value function over the privileged observation."""

    _ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}

    def __init__(self, spec: PrivilegedObsSpec, cfg: CriticConfig):
        super().__init__()
        self.spec = spec
        act = self._ACT[cfg.activation]
        dims = [spec.dim, *cfg.hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.normalizer = RunningNorm(spec.dim) if cfg.normalize else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(obs)).squeeze(-1)


# =================================================================== actor ====
class AmortizedPpoActor(nn.Module):
    """rsl_rl-shaped ActorCritic around the amortized optimizer.

    The rsl_rl surface (``act`` / ``act_inference`` / ``get_actions_log_prob`` /
    ``evaluate`` / ``entropy`` / ``action_mean`` / ``action_std`` /
    ``distribution`` / ``reset``) is implemented faithfully so this class can be
    dropped into rsl_rl the day someone builds a gym surface over the sibling's
    env.  It is NOT what drives training today -- see ``ppo_loop.py`` for the
    route actually taken and why.
    """

    is_recurrent = False

    def __init__(
        self,
        model: AmortizedOptimizer,
        cfg: ActorConfig,
        critic_cfg: CriticConfig,
        *,
        control_dt: float = CONTROL_DT,
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
        num_future: int = 3,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device_ = torch.device(device)
        self.joint_names = tuple(joint_names)
        self.n = len(self.joint_names)
        self.dt = float(control_dt)
        self.H = int(cfg.horizon)
        self.arm_cols = {h: arm_slice(hand, self.joint_names)
                         for h, hand in enumerate(("left", "right"))}
        self.waist_cols = waist_slice(self.joint_names)

        # ---- freeze the backbone -------------------------------------------
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ---- LoRA on the decoder body --------------------------------------
        root = self.model.decoder if cfg.lora_decoder_only else self.model
        self.lora_paths = apply_lora(
            root, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
            targets=tuple(cfg.lora_targets), dropout=cfg.lora_dropout)
        if not self.lora_paths:
            raise RuntimeError(
                f"LoRA targets {cfg.lora_targets} matched nothing under "
                f"{'decoder' if cfg.lora_decoder_only else 'model'}; the "
                f"fine-tune would have no trainable backbone parameters.")
        if cfg.lora_output_rows is not None:
            for module in root.modules():
                if isinstance(module, LoRALinear):
                    module.restrict_output_rows(tuple(cfg.lora_output_rows))

        # ---- the Gaussian head ---------------------------------------------
        # ``std`` is the parameter (rsl_rl's "scalar" noise_std_type), not
        # log-std: the MLP stack ships the scalar form and the documented
        # sigma-divergence fix (clamp on the FORWARD path, so the clamp's zero
        # gradient stops the parameter being driven upward) is written against
        # it.  Keeping the same parameterisation keeps that fix valid verbatim.
        self.std = nn.Parameter(float(cfg.noise_std_init) * torch.ones(self.n))
        self.mean_correction = None
        if cfg.mean_correction:
            d_model = self.model.cfg.d_model
            lin = nn.Linear(d_model, self.n)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.mean_correction = lin

        # ---- AR(1) ----------------------------------------------------------
        taus = {"waist": cfg.tau_waist, "shoulder": cfg.tau_shoulder,
                "elbow": cfg.tau_elbow, "wrist": cfg.tau_wrist}
        rho, self.ar1_table = build_ar1_rho(taus, self.dt, self.joint_names)
        self._ar1_rho = rho
        self._ar1_mix = (1.0 - rho * rho).clamp_min(0.0).sqrt()
        self._ar1_active = bool(cfg.ar1_enabled and float(rho.max()) > 0.0)
        self._ar1_eps: torch.Tensor | None = None

        # ---- critic ----------------------------------------------------------
        self.obs_spec = PrivilegedObsSpec(
            n_dof=self.n, num_future=int(num_future),
            include_tips=bool(critic_cfg.include_tips))
        self.critic = Critic(self.obs_spec, critic_cfg)

        # ---- decoder hidden-state capture -----------------------------------
        # A forward hook on the decoder's final norm, rather than an edit to
        # amortized_opt (which this package may not modify) or a reimplemented
        # decoder (which would drift).  ``TrajectoryDecoder.forward`` computes
        # ``h = self.norm(x)[:, 1:]``, so the hook output is (B, 1 + H, d) and
        # index 1 is waypoint 0.
        self._hidden: torch.Tensor | None = None
        self.model.decoder.norm.register_forward_hook(self._capture_hidden)

        self.distribution: Normal | None = None
        self.to(self.device_)

    def _capture_hidden(self, _module, _inp, out):
        self._hidden = out

    # ------------------------------------------------------------- freezing --
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def actor_parameters(self):
        """Everything the policy loss may touch (i.e. not the critic)."""
        critic_ids = {id(p) for p in self.critic.parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in critic_ids]

    def param_counts(self) -> dict[str, int]:
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and (".lora_a" in n or ".lora_b" in n))
        head = self.std.numel() + (
            sum(p.numel() for p in self.mean_correction.parameters())
            if self.mean_correction is not None else 0)
        critic = sum(p.numel() for p in self.critic.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"lora": lora, "gaussian_head": head, "critic": critic,
                "trainable": lora + head + critic, "frozen": frozen,
                "backbone_total": frozen + lora}

    def assert_frozen(self) -> None:
        """No frozen backbone parameter may carry a gradient.

        Called after the first backward of a run.  A LoRA fine-tune that
        silently trains the whole backbone is not detectable from the loss
        curve -- only from this check and from the parameter count.
        """
        bad = [n for n, p in self.model.named_parameters()
               if not p.requires_grad and p.grad is not None
               and bool(p.grad.abs().sum() > 0)]
        if bad:
            raise RuntimeError(
                f"{len(bad)} frozen backbone parameters received a gradient, "
                f"e.g. {bad[:5]}. The LoRA subspace is not isolated.")

    # ------------------------------------------------------------- the plan --
    def plan(
        self,
        inputs: MpcInputs,
        *,
        encoding: ContextEncoding | None = None,
        masks: WaypointMasks | None = None,
        iterations: int | None = None,
    ) -> PlanOutput:
        """One deterministic refinement.  Differentiable through the LoRA path.

        The context ENCODER is run under ``no_grad``: LoRA lives on the decoder
        only (the plan's "low-rank deltas on the decoder only"), so the encoder
        has no trainable parameter and its activations need no graph.  That is
        most of the update's memory saving.
        """
        cfg = self.cfg
        if masks is None:
            masks = waypoint_masks(inputs.hit_index, horizon=self.H, dt=self.dt,
                                   joint_names=self.joint_names)
        if encoding is None:
            with torch.no_grad():
                encoding = self.model.encode_context(inputs.ctx)

        init = inputs.init * masks.elem_mask
        self._hidden = None
        # Collection's IsaacMPC.plan runs refine under eval(); dropout in
        # train() was a ~0.07-nat mean mismatch before any PPO step.
        was = self.model.training
        self.model.eval()
        try:
            out = self.model.refine(
                init, inputs.state, encoding, masks,
                iterations=int(cfg.iterations if iterations is None else iterations))
        finally:
            if was:
                self.model.train()

        traj = out.traj
        if cfg.freeze_wp > 0:
            traj = traj.clone()
            self._apply_freeze(traj, inputs.init, inputs.hit_index)

        raw_first = traj[:, 0]
        hidden = None
        mean = raw_first
        if self.mean_correction is not None and self._hidden is not None:
            hidden = self._hidden[:, 1]                     # waypoint 0
            mean = mean + cfg.mean_correction_scale * self.mean_correction(hidden)
        return PlanOutput(mean=mean, traj=traj, elem_mask=out.elem_mask,
                          raw_first=raw_first, hidden=hidden)

    def _apply_freeze(
        self,
        traj: torch.Tensor,
        init: torch.Tensor,
        hit_index: torch.Tensor,
    ) -> None:
        """Copy the previous plan through contact for the striking arm.

        Mirrors ``IsaacMPC.plan`` so the PPO replay mean matches collection
        on frozen ticks.  In-place on ``traj``.
        """
        n = int(self.cfg.freeze_wp)
        hit_any = torch.zeros(traj.shape[0], dtype=torch.bool, device=traj.device)
        for h in (0, 1):
            k = hit_index[:, h]
            sel = (k >= 1) & (k <= n)
            if bool(sel.any()):
                cols = self.arm_cols[h]
                m = sel.view(-1, 1, 1)
                traj[:, :, cols] = torch.where(m, init[:, :, cols], traj[:, :, cols])
                hit_any |= sel
        if self.cfg.freeze_waist and bool(hit_any.any()):
            m = hit_any.view(-1, 1, 1)
            wc = self.waist_cols
            traj[:, :, wc] = torch.where(m, init[:, :, wc], traj[:, :, wc])

    # ---------------------------------------------------- the Gaussian head --
    def current_std(self) -> torch.Tensor:
        """sigma, clamped on the FORWARD path.

        Clamping here (not by mutating the parameter after the optimizer step)
        keeps the fix inside this class and, because ``clamp`` has zero gradient
        above the bound, the parameter stops being driven upward once it
        saturates instead of drifting arbitrarily far while the effective sigma
        is pinned.  The in-repo failure this prevents: at iteration 7200 the
        MLP run's wrists had grown to sigma 1.46, so ~half of all wrist actions
        were being clamped.
        """
        return self.std.clamp(min=self.cfg.noise_std_min, max=self.cfg.noise_std_max)

    def update_distribution(self, mean: torch.Tensor) -> Normal:
        std = self.current_std()
        self.distribution = Normal(mean, std.expand_as(mean))
        return self.distribution

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_obs: torch.Tensor, **_kw) -> torch.Tensor:
        return self.critic(critic_obs)

    # ---------------------------------------------------------- AR(1) noise --
    def _sample_ar1(self, mean: torch.Tensor) -> torch.Tensor:
        """Variance-preserving correlated sampling.

        ``eps_t = rho * eps_{t-1} + sqrt(1 - rho^2) * z_t`` is variance
        preserving for ANY rho, so ``eps`` keeps its exact ``N(0, 1)`` marginal
        and the action stays a draw from ``Normal(mean, std)`` -- which is why
        the stock log-prob remains the true sampling density and PPO is
        undisturbed.  Ported from ``factored_actor_critic._sample_ar1``
        (:1209-1233).

        The beat-scheduled rho of the MLP stack is deliberately NOT ported: it
        reads one upcoming beat time out of FIXED MLP observation indices which
        do not exist here.  Recorded as a future knob in README.
        """
        rho = self._ar1_rho.to(mean.device)
        mix = self._ar1_mix.to(mean.device)
        eps = self._ar1_eps
        if eps is None or eps.shape != mean.shape or eps.device != mean.device:
            eps = torch.randn_like(mean)          # fresh, so the marginal is
        else:                                     # N(0,1) from the first step
            eps = rho * eps + mix * torch.randn_like(mean)
        self._ar1_eps = eps.detach()
        return eps

    def reset(self, dones=None) -> None:
        """Re-draw the AR(1) state of envs that just reset.

        Fresh ``randn``, NOT zeros: zeroing would make the first post-reset
        action exactly the mean and leave eps non-stationary for several ticks,
        which biases the very transitions an episode boundary already makes
        hardest to credit.
        """
        if self._ar1_eps is None or dones is None:
            return
        mask = dones.reshape(-1).to(torch.bool)
        if mask.shape[0] != self._ar1_eps.shape[0]:
            self._ar1_eps = None                  # env count changed; re-init lazily
            return
        if bool(mask.any()):
            self._ar1_eps[mask] = torch.randn(
                int(mask.sum()), self._ar1_eps.shape[-1],
                device=self._ar1_eps.device, dtype=self._ar1_eps.dtype)

    # ------------------------------------------------------- rsl_rl surface --
    def forward(self):
        raise NotImplementedError

    def act(self, inputs: MpcInputs, **kw) -> torch.Tensor:
        """Sample an executed waypoint.  ``inputs`` is structured, not a vector.

        Deviation from rsl_rl's signature (which takes an observation
        TensorDict) and the reason this class does not drive rsl_rl's runner:
        the policy's input is a stateful warm start plus a structured context.
        """
        with torch.no_grad():
            plan = self.plan(inputs, **kw)
        self.update_distribution(plan.mean)
        if self._ar1_active:
            eps = self._sample_ar1(plan.mean)
            return plan.mean + self.action_std * eps
        return self.distribution.sample()

    def act_inference(self, inputs: MpcInputs, **kw) -> torch.Tensor:
        """The deterministic policy: the plan's own first waypoint."""
        with torch.no_grad():
            return self.plan(inputs, **kw).mean

    # --------------------------------------------------------------- saving --
    def policy_state_dict(self) -> dict:
        """Only what is trainable, plus the buffers needed to rebuild it."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in self.state_dict().items()
              if k in keep or k.startswith("critic.normalizer.")}
        return sd


def register_with_rsl_rl() -> None:
    """Expose :class:`AmortizedPpoActor` to rsl_rl's ``eval(class_name)`` lookup.

    ``OnPolicyRunner`` resolves ``policy.class_name`` with ``eval`` against the
    ``rsl_rl.runners.on_policy_runner`` namespace, so registration is a
    monkey-patch of the class into that module (and, for newer rsl_rl, into
    ``rsl_rl.modules``).  Mirrors ``factored_actor_critic.register_with_rsl_rl``
    (:1260-1278), but registration lives in THIS module -- the sibling repo file
    is import-only for this package.

    Registering does not make the actor usable from rsl_rl's runner on its own;
    ``act``/``evaluate`` take structured MPC inputs.  It is here so that a
    future gym surface over the sibling env has one less thing to build.
    """
    try:
        from rsl_rl.runners import on_policy_runner
    except ImportError:
        return
    on_policy_runner.AmortizedPpoActor = AmortizedPpoActor
    try:
        import rsl_rl.modules as modules
        modules.AmortizedPpoActor = AmortizedPpoActor
    except Exception:
        pass
