"""Configuration for executed-prefix PPO, in one place.

One dataclass tree (:class:`PpoRunConfig`) covers the actor, the critic, the
PPO update, the analytic auxiliary loss, the environment selection and the run
bookkeeping.  Torch-free and Isaac-free so the whole configuration surface can
be imported, diffed and unit-tested without a simulator.

The rate constants are NOT redefined here -- they come from the sibling's
``env_cfg`` through :mod:`bridge` (250 Hz physics, decimation 5, 50 Hz control;
resolved decision 1), because a second copy of "the control period" is exactly
the drift this project has already been bitten by.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

__all__ = [
    "ActorConfig",
    "AuxCostConfig",
    "CriticConfig",
    "EnvConfig",
    "PpoConfig",
    "PpoRunConfig",
    "CheckpointChoice",
    "resolve_checkpoint",
]

#: Repository-relative root of the amortized project (``amortized/``).
AMORTIZED_ROOT = Path(__file__).resolve().parent.parent


# ===================================================== checkpoint resolution ==
#: Checkpoints that must NEVER be auto-selected, with the reason.
#:
#: ``phase2c_main`` is a KNOWN-REGRESSED continuation: it overfit the seed-0
#: validation grid (+7 seed-0, -8 on the fresh seed-7 test) and its closed loop
#: collapsed to 8/55 strikes against phase2b's 25/55.  README chapter 7c
#: documents the rejection and states "runs/phase2b_main/best.pt therefore
#: remains the champion".  ``phase2_main`` is the superseded first phase-2 run.
BLOCKLIST: dict[str, str] = {
    "phase2c_main": "known-regressed (README 7c: overfit seed-0, closed loop 8/55 vs 25/55)",
    "phase2_main": "superseded by phase2b_main",
}

#: Resolution order.  ``phase2d_waist`` is a CANDIDATE only -- it is selected
#: solely when its own report chapter marks the waist/effort trade ACCEPTED,
#: because two legs write into that directory and only the later leg's best is
#: the candidate.  File existence alone is explicitly not enough.
CANDIDATES: tuple[tuple[str, bool], ...] = (
    ("phase2d_waist", True),    # (run name, requires an explicit acceptance mark)
    ("phase2b_main", False),    # the standing champion and the safe default
)

_ACCEPT_KEYS = ("accepted", "trade_accepted", "is_accepted", "verdict", "decision")
_ACCEPT_RE = re.compile(r"\bACCEPTED\b", re.IGNORECASE)


@dataclass
class CheckpointChoice:
    """What :func:`resolve_checkpoint` picked and, in words, why."""

    path: Path | None
    run: str | None
    reason: str
    considered: list[tuple[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"[ckpt] resolved -> {self.path}  ({self.reason})"]
        for run, why in self.considered:
            lines.append(f"[ckpt]   {run}: {why}")
        return "\n".join(lines)


def _acceptance_mark(run_dir: Path, run: str) -> tuple[bool, str]:
    """Is this run explicitly marked ACCEPTED?  Conservative: default no.

    Two places are consulted, in order: any ``summary*.json`` in the run
    directory (an explicit boolean/verdict key), then the report chapter for the
    run in ``amortized/README.md`` / ``amortized/REPORT.md``.  Anything
    ambiguous reads as NOT accepted -- selecting a regressed checkpoint by
    accident costs a training run, declining to select a good one costs a config
    flag.
    """
    for summary in sorted(run_dir.glob("summary*.json")):
        try:
            blob = json.loads(summary.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(blob, dict):
            continue
        for key in _ACCEPT_KEYS:
            if key not in blob:
                continue
            val = blob[key]
            if isinstance(val, bool):
                if val:
                    return True, f"{summary.name} has {key}=true"
                return False, f"{summary.name} has {key}=false"
            if isinstance(val, str) and _ACCEPT_RE.search(val):
                return True, f"{summary.name} {key}={val!r}"
    # the report chapters
    for doc in ("README.md", "REPORT.md"):
        path = AMORTIZED_ROOT / doc
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        # the chapter heading for a run is written as "## Section 7d ..." /
        # "### Section 7d ..."; take the chapter body up to the next heading of
        # the same or higher level.
        m = re.search(rf"^(#{{2,4}})\s.*\b{re.escape(run)}\b.*$", text, re.MULTILINE)
        if m is None:
            m = re.search(r"^(#{2,4})\s.*\b7d\b.*$", text, re.MULTILINE)
        if m is None:
            continue
        level = len(m.group(1))
        rest = text[m.end():]
        nxt = re.search(rf"^#{{1,{level}}}\s", rest, re.MULTILINE)
        chapter = rest[: nxt.start()] if nxt else rest
        if _ACCEPT_RE.search(chapter):
            return True, f"{doc} chapter marks it ACCEPTED"
        return False, f"{doc} chapter exists but has no ACCEPTED verdict"
    return False, "no summary key and no report chapter marking it accepted"


def resolve_checkpoint(
    explicit: str | os.PathLike | None = None,
    *,
    runs_dir: str | os.PathLike | None = None,
    file_name: str = "best.pt",
) -> CheckpointChoice:
    """Pick the pretrained amortized-optimizer checkpoint to fine-tune.

    An explicit path always wins and is never second-guessed (but is still
    checked against :data:`BLOCKLIST`, loudly).  Otherwise :data:`CANDIDATES` is
    walked in order.
    """
    runs = Path(runs_dir) if runs_dir is not None else AMORTIZED_ROOT / "runs"
    considered: list[tuple[str, str]] = []

    if explicit is not None:
        path = Path(explicit)
        run = path.parent.name
        if run in BLOCKLIST:
            raise ValueError(
                f"refusing to load {path}: {run} is {BLOCKLIST[run]}. "
                f"Pass a different --ckpt if you really mean it."
            )
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        return CheckpointChoice(path=path, run=run, reason="explicit --ckpt",
                                considered=considered)

    for run, needs_mark in CANDIDATES:
        path = runs / run / file_name
        if not path.exists():
            considered.append((run, f"absent ({path} not found)"))
            continue
        if needs_mark:
            ok, why = _acceptance_mark(runs / run, run)
            if not ok:
                considered.append((run, f"present but NOT marked accepted -- {why}"))
                continue
            considered.append((run, f"present and accepted -- {why}"))
            return CheckpointChoice(path=path, run=run,
                                    reason=f"accepted candidate ({why})",
                                    considered=considered)
        considered.append((run, "present (standing champion)"))
        return CheckpointChoice(path=path, run=run,
                                reason="standing champion / safe default",
                                considered=considered)

    return CheckpointChoice(
        path=None, run=None,
        reason="no candidate checkpoint found -- the actor will start from a "
               "randomly initialised transformer (smoke tests only)",
        considered=considered,
    )


# ================================================================= the actor ==
@dataclass
class ActorConfig:
    """The amortized transformer as a PPO actor."""

    # ---- pretrained backbone ----
    #: ``None`` -> :func:`resolve_checkpoint`.  Never resolves to phase2c_main.
    ckpt_path: str | None = None
    ckpt_file: str = "best.pt"
    #: Allow a randomly-initialised backbone when no checkpoint resolves.  Only
    #: sane for tests; a fine-tune from noise is not a fine-tune.
    allow_untrained_backbone: bool = False

    # ---- the MPC feature extractor ----
    horizon: int = 50
    #: Refinement iterations per tick.  x1 ties or beats x3 at every seed
    #: (REPORT 3.4); it is also what the deployed loop runs.
    iterations: int = 1
    #: fp32 by default -- NOT the sibling's bf16.  PPO re-evaluates the mean at
    #: update time and the ratio is only 1.0 at the first epoch if collection
    #: and update agree numerically; bf16 carries ~3 decimal digits and would
    #: put a visible bias in log(ratio) before a single gradient step.  Costs
    #: throughput; see README "Throughput".
    amp_dtype: str | None = None            # None | "bf16" | "fp16"

    # ---- LoRA ----
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    #: Which decoder sub-matrices carry a delta.  The plan's "low-rank deltas on
    #: the decoder only": attention OUT-projections and the two FFN matrices.
    lora_targets: tuple[str, ...] = ("o_proj", "w12", "w3")
    #: Restrict LoRA to the decoder body.  The encoder runs frozen and under
    #: ``no_grad`` at update time, which is most of the update's memory saving.
    lora_decoder_only: bool = True

    # ---- the Gaussian head ----
    #: Per-dim init std, rad.  0.45-0.50 per the delta_vel lesson
    #: (rsl_rl_ppo_cfg.py:251).
    noise_std_init: float = 0.45
    #: Hard clamp.  sigma divergence under flat-past-clamp rewards is a
    #: documented in-repo failure mode; the clamp is the fix that shipped.
    noise_std_max: float = 1.0
    noise_std_min: float = 0.02
    #: Optional trainable linear correction on the mean, read off the decoder's
    #: last hidden state at waypoint 0.  Zero-initialised, so the actor starts
    #: EXACTLY at the pretrained plan.
    mean_correction: bool = True
    mean_correction_scale: float = 0.1

    # ---- AR(1) correlated exploration ----
    ar1_enabled: bool = True
    #: Per-joint-group correlation times, seconds.  Ported from the MLP stack's
    #: shipped values; adapted to this project's 50 Hz control dt (the sibling
    #: MLP env runs 60 Hz, so rho differs even though tau does not).
    tau_waist: float = 0.30
    tau_shoulder: float = 0.25
    tau_elbow: float = 0.15
    tau_wrist: float = 0.10
    #: Beat-scheduled rho is NOT implemented in v1: the MLP version reads one
    #: upcoming beat time from FIXED MLP observation indices that do not exist
    #: in this stack.  Recorded as a future knob in README.
    ar1_beat_schedule: bool = False


@dataclass
class CriticConfig:
    """Asymmetric MLP value function over a flat privileged observation."""

    hidden: tuple[int, ...] = (512, 256, 256)
    activation: str = "elu"
    #: Include stick-tip world positions (one extra ChainFK pass per tick).
    include_tips: bool = True
    #: Normalise the privileged observation with a running mean/var.
    normalize: bool = True


# =================================================================== the loss =
@dataclass
class AuxCostConfig:
    """The analytic cost as an auxiliary loss on the FULL emitted trajectory.

    The tail's constraint anchor: only waypoint 0 is executed, so nothing in the
    PPO return prices waypoints 1..H-1, and without this term the plan is free
    to become garbage everywhere the environment cannot see -- which is exactly
    where the next tick's warm start comes from.  Applied through the SAME
    optimizer as the PPO losses (the ``bc_aux.DrumBcAuxPPO`` pattern).
    """

    enabled: bool = True
    #: In-Isaac gate.  ``FusedCost`` needs ``BatchedRNEA``, whose construction
    #: reads ``model.parents`` through Boost.Python -- and that conversion
    #: raises inside a running Isaac process (``No Python class registered for
    #: C++ class std::vector<unsigned long>``), because Isaac's runtime shadows
    #: the converter registration.
    #:
    #: ``"auto"`` (default) enables the aux loss when the RNEA was successfully
    #: built and disables it otherwise, reporting which happened.  This package
    #: builds the RNEA **before** ``AppLauncher`` (see ``train_ppo.py``), which
    #: sidesteps the collision entirely and is measured working in-Isaac -- so
    #: in practice ``auto`` resolves to ON.  ``"off"`` forces it off; ``"on"``
    #: forces it on and lets the construction error surface.
    #:
    #: When RL-1b's cached-topology fix lands the gate needs no code change:
    #: ``auto`` simply keeps resolving to ON.
    in_isaac: str = "auto"                  # "auto" | "on" | "off"
    weight: float = 0.1
    #: Which ``FusedCost`` groups enter the loss.  The TERMINAL groups are out by
    #: default: in Isaac the executed contact is the terminal objective, and
    #: keeping the kinematic terminal too re-creates the conflict README 7c
    #: identified (a plan that EXECUTES on the head is kinematically ~140 mm
    #: short of it).  ``None`` -> the full ``CostTerms.total``.
    terms: tuple[str, ...] | None = (
        "collision", "torque", "velocity", "joint_box", "smooth", "posture",
    )
    #: Use ``train_cost.phase2_cost_config()`` (the TRAINING-calibrated weights)
    #: rather than ``CostConfig()``'s validation weights, whose group means
    #: differ by three orders of magnitude.
    phase2_weights: bool = True
    #: Evaluate the aux cost on this fraction of each minibatch (it is the most
    #: expensive term in the update).  1.0 = all of it.
    minibatch_fraction: float = 1.0


@dataclass
class PpoConfig:
    """Executed-prefix PPO.  17-dim action, so the MLP stack's budgets transfer.

    ``desired_kl = 0.015`` is a SUMMED-over-17-dims budget (~8.8e-4/dim), which
    is why the action formulation matters: at the rejected 850-dim
    full-trajectory action the same number is 1.8e-5/dim and the adaptive
    controller rails the learning rate at rsl_rl's 1e-5 floor.
    """

    clip_param: float = 0.2
    gamma: float = 0.995
    lam: float = 0.95
    desired_kl: float = 0.015
    num_learning_epochs: int = 3
    num_mini_batches: int = 4
    entropy_coef: float = 0.00175
    #: 0.5, from the MLP stack's tuned algorithm cfg (rsl_rl_ppo_cfg.py:275),
    #: not rsl_rl's 1.0 default.
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    max_grad_norm: float = 1.0
    #: 3e-5, NOT rsl_rl's 1e-3 or the 1e-4 this package started at.  Measured:
    #: at 1e-4 the 20-iteration Isaac smoke railed the adaptive controller at
    #: the 1e-5 floor 28 times, with KL median 0.021 against a 0.015 target and
    #: clip_frac median 0.41 (max 0.65) where a healthy PPO run sits at
    #: 0.1-0.2.  That is the update moving the policy far too far per step --
    #: the documented in-repo railing failure, reproduced.
    #:
    #: The documented alternative, if throughput matters more than caution:
    #: keep 1e-4 and drop ``num_learning_epochs`` 3 -> 2, which cuts the number
    #: of LR adjustments per iteration from 12 to 8 and roughly halves the
    #: per-iteration policy movement.  A pilot at both is the right call; this
    #: default is the conservative one.
    learning_rate: float = 3.0e-5
    schedule: str = "adaptive"              # "adaptive" | "fixed"
    #: rsl_rl's own floor.  Documented failure mode: the adaptive controller
    #: rails HERE and the run silently stops learning, so the floor is logged
    #: every iteration and a warning fires when it is hit.
    lr_min: float = 1.0e-5
    lr_max: float = 1.0e-3
    normalize_advantage: bool = True
    #: Steps of rollout per env per iteration.
    num_steps_per_env: int = 48
    #: Gradient-accumulation chunk INSIDE a minibatch.  A minibatch of
    #: ``num_envs * num_steps_per_env / num_mini_batches`` transitions would put
    #: a decoder forward+backward at batch ~768 x H=50 on the GPU, which is
    #: ~10 GB on a box whose GPUs are shared.  Chunks are reweighted exactly by
    #: their size, so the accumulated gradient is identical to one pass -- this
    #: is a memory knob, not an algorithmic one.  0 = no chunking.
    micro_batch_size: int = 128


# ============================================================== the env & run =
@dataclass
class EnvConfig:
    """Selection passed through to the sibling's ``DrumRLEnvCfg``.

    Field names mirror theirs exactly so this is a pass-through, not a
    translation layer -- a translation layer is where rate mismatches hide.
    """

    num_envs: int = 64
    device: str = "cuda:1"
    seed: int = 0
    episode_length_s: float = 12.0
    rhythm_split: str = "train"
    random_clip_offset: bool = True
    contact_sensors: bool = True
    gravity_feedforward: bool = True
    foh_ramp: bool = True
    pin_pelvis: bool = True
    freeze_legs: bool = True
    use_ideal_pd: bool = True
    pd_damping_scale: float = 2.0
    hit_mode: str = "geometric"
    randomize_kit: bool = False
    lag_substeps: tuple[int, int] = (0, 0)

    # ---- which plant this config drives ----
    #: ``"isaac"`` (PhysX) or ``"analytic"`` (``amortized_opt.pd_sim``).  The two
    #: regimes take DIFFERENT lead vectors and must not be conflated -- see
    #: :meth:`PpoRunConfig.assert_lead_convention`.
    plant_regime: str = "isaac"

    # ---- the reference lead ----
    #: ``None`` -> resolved by :meth:`PpoRunConfig.resolve_lead` from the regime:
    #: the ISAAC-FITTED vector for ``plant_regime="isaac"``, the analytic
    #: ``"fit"`` preset for ``"analytic"``.  An explicit preset name, float or
    #: 17-tuple overrides the resolution.
    #:
    #: The two plants need DIFFERENT vectors and it is not a small correction.
    #: ``LEAD_PRESETS["fit"]`` was identified against ``amortized_opt.pd_sim``,
    #: an analytic FOH servo model; RL-1b re-fitted against PhysX and the
    #: **wrists inverted** -- from 0.0 ms ("flat, a lead cannot touch them" on
    #: the analytic plant) to a 38.8 ms median with 20-32 % tracking gains.  Any
    #: reasoning carried over from the analytic per-group reading is therefore
    #: wrong on this plant; use the published file, not intuition.
    cmd_lead_s: str | float | tuple | None = None
    #: The published Isaac fit (RL-1b), relative to ``amortized/``.  One source
    #: of truth: the file, not a copy of the numbers.
    lead_json: str = "runs/rl1b/lead_isaac.json"
    #: ``best_strike_s`` is fitted on the strike window (6 knots), which is the
    #: interval the reward actually scores; ``best_all_s`` optimises whole-path
    #: tracking instead.
    lead_key: str = "best_strike_s"
    hit_lead_s: float = 0.0
    freeze_wp: int = 0
    #: The QP safety filter rewrites the executed command, which changes the
    #: action the log-prob was computed for.  OFF in v1; DECISION NEEDING
    #: REVIEW in README.
    safety_filter: bool = False


@dataclass
class PpoRunConfig:
    """The whole run."""

    name: str = "ppo_v1"
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    aux: AuxCostConfig = field(default_factory=AuxCostConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    max_iterations: int = 2000
    save_interval: int = 50
    log_interval: int = 1
    resume: str | None = None
    #: ``amortized/runs/isaac_rl_ppo_<name>/``
    runs_root: str | None = None

    # ---- reward wiring ----
    #: The sibling's ``RewardConfig.w_plan_cost`` prices the analytic cost as a
    #: (non-differentiable) REWARD term.  Zeroed here by default because
    #: :class:`AuxCostConfig` prices the same quantity as a differentiable loss,
    #: which is strictly better signal for the same thing; leaving both on
    #: double-counts it.  Set >0 to restore the sibling's behaviour.
    reward_plan_cost_weight: float = 0.0
    #: Which reference the tracking term scores the plant against.
    #: ``"sampled"`` = the pose the STOCHASTIC policy nominated (mean + noise);
    #: ``"mean"`` = the sibling's ``mpc.last_ref0`` (the un-led plan waypoint).
    #: Either way it is the UN-LED reference: scoring against the led command
    #: would price the lead itself as error.
    track_reference: str = "sampled"

    # ---------------------------------------------------------------- checks --
    def resolve_lead(self) -> tuple[object, str]:
        """``(value_for_MPCConfig.cmd_lead_s, human-readable source)``.

        Resolution order: an explicit ``cmd_lead_s`` wins; otherwise the regime
        decides -- Isaac loads the published RL-1b fit from
        :attr:`EnvConfig.lead_json`, analytic uses the ``"fit"`` preset.
        Reading the JSON rather than embedding the 17 floats keeps one source of
        truth, so a re-fit propagates by replacing a file.
        """
        env = self.env
        if env.cmd_lead_s is not None:
            return env.cmd_lead_s, f"explicit ({env.cmd_lead_s!r})"
        if env.plant_regime == "analytic":
            return "fit", "analytic preset LEAD_PRESETS['fit']"
        path = Path(env.lead_json)
        if not path.is_absolute():
            path = AMORTIZED_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"the Isaac-fitted lead {path} is missing.  It is published by "
                f"RL-1b (isaac_rl/scripts/fit_lead_isaac.py); see the "
                f"'ISAAC-fitted reference lead' section of "
                f"isaac_rl/INTERFACES.md.  Set env.cmd_lead_s=0 to run without "
                f"a lead, but do NOT substitute LEAD_PRESETS['fit'] -- it is "
                f"the analytic vector and its wrist entries are inverted on "
                f"this plant.")
        blob = json.loads(path.read_text())
        if env.lead_key not in blob:
            raise KeyError(
                f"{path} has no key {env.lead_key!r}; available: {sorted(blob)}")
        vec = tuple(float(x) for x in blob[env.lead_key])
        if len(vec) != 17:
            raise ValueError(f"{path}[{env.lead_key}] has {len(vec)} entries, expected 17")
        return vec, f"{env.lead_json}[{env.lead_key}]"

    def assert_lead_convention(self) -> None:
        """The lead is a CONFIG constant, never something the head learns.

        The Gaussian head emits an additive joint-space offset on the executed
        command.  Nothing stops it, in principle, from learning a constant
        offset that acts like extra lead -- which would silently double-apply
        the compensation and make the fitted ``cmd_lead_s`` vector wrong.  Two
        guards hold regardless of regime: the mean correction is
        zero-initialised and small-scaled, and the command path applies the lead
        exactly once (``test_lead_is_not_applied_twice``).

        What this method pins is the REGIME, because the two plants need
        different vectors and using the wrong one is silent:

        * ``analytic`` -- ``amortized_opt.pd_sim``.  ``"fit"`` is right here; it
          is the vector identified against this exact servo.
        * ``isaac`` -- PhysX.  Use the RL-1b published fit.  The analytic preset
          is refused outright: it is not a small correction but a sign change on
          the wrists (0.0 ms -> 38.8 ms median).
        """
        regime = self.env.plant_regime
        if regime not in ("isaac", "analytic"):
            raise ValueError(f"plant_regime={regime!r}; expected isaac|analytic")

        if regime == "isaac" and self.env.cmd_lead_s == "fit":
            raise ValueError(
                "env.cmd_lead_s='fit' with plant_regime='isaac'.  That is the "
                "ANALYTIC preset, fitted against amortized_opt.pd_sim, and it "
                "is not a small correction on PhysX -- RL-1b measured the "
                "wrists inverting from 0.0 ms to a 38.8 ms median.  Leave "
                "cmd_lead_s=None to load the published Isaac fit "
                "(runs/rl1b/lead_isaac.json), or set 0 to disable the lead.")
        if regime == "analytic" and self.env.cmd_lead_s not in (None, "fit"):
            raise ValueError(
                f"plant_regime='analytic' with cmd_lead_s="
                f"{self.env.cmd_lead_s!r}; the analytic plant is the one the "
                f"'fit' preset is correct for.")
        if self.env.hit_lead_s or self.env.freeze_wp:
            raise ValueError(
                "hit_lead_s / freeze_wp are OFF in the deployment config "
                "(README 7c decision 2: the 3-mechanism stack is WORSE than the "
                "lead alone on the latency metric). Layer them only against a "
                "measured residual.")

    def validate(self) -> None:
        self.assert_lead_convention()
        if self.track_reference not in ("sampled", "mean"):
            raise ValueError(f"track_reference={self.track_reference!r}")
        if self.ppo.num_steps_per_env % self.ppo.num_mini_batches:
            raise ValueError(
                "num_steps_per_env * num_envs must divide num_mini_batches "
                "evenly for the flat minibatch generator")
        if not (0.0 < self.actor.noise_std_init <= self.actor.noise_std_max):
            raise ValueError("noise_std_init must be in (0, noise_std_max]")

    # --------------------------------------------------------------- paths ---
    def run_dir(self) -> Path:
        root = Path(self.runs_root) if self.runs_root else AMORTIZED_ROOT / "runs"
        return root / f"isaac_rl_ppo_{self.name}"

    def to_dict(self) -> dict:
        return asdict(self)
