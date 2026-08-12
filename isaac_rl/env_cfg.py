"""Env configuration for the amortized-optimizer RL fine-tune.

Resolved decision 1 (2026-08-10):

    Waypoint grid: 50 Hz, physics/PD at 250 Hz.  ... for the amortized
    optimizer's stack, sim dt is brought up to **250 Hz** (decimation 5 -> a
    true 50 Hz control period, and it matches the plan's "PD controller at
    250hz").  This applies **in the new amortized-optimizer folder only** --
    the main repo's SIM_DT/DECIMATION are untouched.

So this file defines its OWN rate constants rather than importing
``g1_drummer.control.control_rate``.  That module is imported by the offline
AMP/teacher/BC generators AND by ``qp_filter.SafetyFilter``'s default ``dt``,
so mutating it would silently reshard every dataset in the main repo.  The cost
of owning the constants is that they can drift; the guard against that is
:func:`assert_rate_consistency`, which pins ``SIM_DT * DECIMATION`` to
``amortized_opt.CONTROL_DT`` -- the waypoint grid the model was trained on.

What is carried over from ``G1DrummerEnvCfg`` and what is dropped is tabulated
in ``isaac_rl/README.md``; the short version is that everything about the *plant*
is carried (robot cfg, drum USD, contact sensors, pelvis pin, gravity/Coriolis
feedforward, FOH ramp) and everything about the *MLP policy* is dropped
(observation assembly, lookahead/field/credit blocks, AMP, curriculum,
action decoding and its 20-odd knobs) because in this stack the policy is the
amortized optimizer and its "action" is a whole trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amortized_opt.constants import CONTROL_DT as _WAYPOINT_DT

__all__ = [
    "CONTROL_DT",
    "CONTROL_HZ",
    "DECIMATION",
    "SIM_DT",
    "DrumRLEnvCfg",
    "assert_rate_consistency",
    "build_env_cfg",
]

# ---------------------------------------------------------------- the rate --
#: Physics step.  250 Hz, per resolved decision 1.  The main repo runs 1/120.
SIM_DT: float = 1.0 / 250.0
#: Physics substeps per control tick.  5 -> a true 50 Hz control period, which
#: is exactly the model's waypoint grid, so "one waypoint = one control tick"
#: holds without resampling.  The main repo runs 2 (at 1/120 -> 60 Hz).
DECIMATION: int = 5
CONTROL_DT: float = SIM_DT * DECIMATION          # 0.02 s
CONTROL_HZ: float = 1.0 / CONTROL_DT             # 50.0


def assert_rate_consistency() -> None:
    """The control period MUST equal the model's waypoint spacing.

    Every horizon in the stack -- ``hit_index``, the element masks, the Fourier
    time codes, the reference lead expressed in waypoint fractions -- assumes
    one waypoint per control tick.  A mismatch does not raise anywhere; it just
    makes every horizon wrong by a constant factor, which is exactly the class
    of bug that is invisible in the logs and fatal in the numbers.
    """
    if abs(CONTROL_DT - _WAYPOINT_DT) > 1e-12:
        raise RuntimeError(
            f"control period {CONTROL_DT} != waypoint grid {_WAYPOINT_DT}; "
            "SIM_DT * DECIMATION must equal amortized_opt.CONTROL_DT"
        )


assert_rate_consistency()


# ------------------------------------------------------------------ config --
@dataclass
class DrumRLEnvCfg:
    """Plain dataclass; converted to an Isaac ``@configclass`` in :mod:`env`.

    Keeping the knobs in a torch/Isaac-free dataclass means the whole
    configuration surface (and the knob table in the README) can be imported,
    diffed and tested without a simulator.
    """

    # ---- scene ----
    num_envs: int = 256
    env_spacing: float = 4.0
    #: Reassert the pelvis pose every physics substep.  Carried over from
    #: ``G1DrummerEnvCfg.pin_pelvis``: some Isaac Lab builds silently ignore
    #: the spawn-time ``fix_root_link``, leaving the G1 to topple.
    pin_pelvis: bool = True
    #: Freeze the legs in the composed USD (``get_g1_cfg(freeze_legs=...)``).
    freeze_legs: bool = True
    #: ``IdealPDActuatorCfg`` with Kd doubled -- the runtime gains the whole
    #: amortized stack was identified against (see the "Action space vs torque
    #: budget" note: the real Kd is 2x the CLAUDE.md table).
    use_ideal_pd: bool = True
    pd_stiffness_scale: float = 1.0
    pd_damping_scale: float = 2.0

    # ---- episode ----
    episode_length_s: float = 12.0
    #: Rhythm clips: ``"eval"`` / ``"train"`` selects the vendored RCC split in
    #: ``amortized/assets/rcc/``; a path uses that directory instead.
    rhythm_split: str = "eval"
    rhythm_dir: str | None = None
    #: Start every env at a random offset into its clip.  Off for the smoke run
    #: so the histogram is reproducible.
    random_clip_offset: bool = False

    # ---- MPC ----
    horizon: int = 50
    iterations: int = 1
    cmd_lead_s: str | float | tuple = "fit"
    hit_lead_s: float = 0.0
    freeze_wp: int = 0

    # ---- contact / hit detection ----
    #: ``"geometric"`` (default) scores the crossing with the SAME analytic
    #: strike geometry the cost prices; ``"contact"`` gates it on the PhysX
    #: contact sensors as well.  See the DECISION note in the README -- the two
    #: legal regions genuinely differ (the repo's detector has no half-plane
    #: rule and uses a rim margin instead of the analytic disk centre offset).
    hit_mode: str = "geometric"
    #: Newtons.  ``G1DrummerEnvCfg.contact_force_threshold``'s role.
    contact_force_threshold: float = 1.0
    #: Build the two wrist ContactSensors.  Needed for ``hit_mode="contact"``
    #: and for the contact diagnostics; costs a little PhysX time.
    contact_sensors: bool = True

    # ---- feedforward ----
    #: Gravity + Coriolis (+ optional inertial) feedforward torque, applied
    #: every physics substep.  Carried over: the amortized stack's PD model
    #: (``amortized_opt.pd_sim``) assumes the servo does NOT have to hold
    #: gravity with position error, so dropping this would change the plant the
    #: per-joint lead was fitted against.
    gravity_feedforward: bool = True
    #: First-order hold on the position target across the ``DECIMATION``
    #: substeps, with the matching velocity target -- the FOH semantics
    #: ``amortized_opt.pd_sim`` reproduces analytically.
    foh_ramp: bool = True

    # ---- domain randomization (scaffold: declared, mostly off) ----
    randomize_kit: bool = False
    #: Residual command lag, in PHYSICS substeps, drawn per episode.  RL
    #: priority 3: "the residual lag around [the lead] must be randomized so
    #: the planner learns lag-robust trajectories rather than leaning on an
    #: exact delta."  ``(0, 0)`` = off.
    lag_substeps: tuple[int, int] = (0, 0)
    #: Contact restitution / friction randomization range.  RL priority 2:
    #: "put contact parameters in the domain randomization, or distillation
    #: overfits to simulated bounce quirks."  ``None`` = leave the USD's values.
    restitution_range: tuple[float, float] | None = None
    friction_range: tuple[float, float] | None = None

    # ---- misc ----
    device: str = "cuda:1"
    seed: int = 0
    #: Record ``(T, B, 17)`` measured poses for the post-hoc latency histogram.
    record: bool = False

    @property
    def sim_dt(self) -> float:
        return SIM_DT

    @property
    def decimation(self) -> int:
        return DECIMATION

    @property
    def control_dt(self) -> float:
        return CONTROL_DT

    def knob_table(self) -> list[tuple[str, str, str]]:
        """``(knob, value, provenance)`` -- the table the README publishes."""
        return [
            ("sim.dt", f"{SIM_DT:.6f} (250 Hz)", "NEW -- resolved decision 1 (repo: 1/120)"),
            ("decimation", str(DECIMATION), "NEW -- resolved decision 1 (repo: 2)"),
            ("step_dt", f"{CONTROL_DT} (50 Hz)", "NEW (repo: 1/60)"),
            ("action_space", "trajectory (50 x 17)", "REPLACED (repo: 17 scalars)"),
            ("observation_space", "n/a -- ContextInputs", "DROPPED"),
            ("robot_cfg", "get_g1_cfg(ideal_pd, Kd x2, frozen legs)", "CARRIED"),
            ("drum_kit_cfg", "simple_drum_kit_6drums.usd", "CARRIED"),
            ("pin_pelvis", str(self.pin_pelvis), "CARRIED"),
            ("contact sensors", str(self.contact_sensors), "CARRIED"),
            ("FOH ramp", str(self.foh_ramp), "CARRIED"),
            ("gravity/Coriolis ff", str(self.gravity_feedforward), "CARRIED"),
            ("hit detection", self.hit_mode, "REPLACED -- see DECISION 1"),
            ("extra-hit punishment", "reward.extra_hit / early_contact", "CARRIED (reshaped)"),
            ("QP safety filter", "isaac_rl.safety, always on", "PROMOTED (repo: eval only)"),
            ("curriculum", "none", "DROPPED"),
            ("AMP", "none", "DROPPED"),
            ("action_scale / delta_vel / soft clip", "none", "DROPPED"),
            ("lookahead / field-grad / credit obs", "none", "DROPPED"),
            ("PPO / rsl_rl", "none", "DROPPED -- episodic param-space search"),
        ]


def build_env_cfg(**kw) -> DrumRLEnvCfg:
    return DrumRLEnvCfg(**kw)
