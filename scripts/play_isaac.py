#!/usr/bin/env python
from __future__ import annotations

"""Play the amortized MPC model inside the Isaac Sim G1 drummer environment.

.. warning::

   **REPO-ONLY.** Together with ``scripts/repo_adapters/`` this is one of two
   files under ``amortized/`` that still need the original
   ``g1-drummer-lab-rythmic`` checkout. It plays into the *repo's* registered
   task (``Isaac-G1-Drum-Direct-v0``), reading ~25 attributes off
   ``G1DrummerEnv`` -- including its hit detector, beat telemetry and waist pin
   -- so it cannot be pointed at the standalone env without being rewritten
   against it. That env is 6 374 lines plus a 3 333-line cfg; porting it was
   explicitly not worth it, since ``isaac_rl_ppo/env.py`` already carries the
   plant the optimizer was identified against.

   For playback that does not need the repo, use ``isaac_rl_ppo/scripts/``.
   Put the repo on the path with ``PYTHONPATH=<repo>/source/g1_drummer`` (see
   ``_require_repo`` below for the exact message on failure).

This is the Isaac bridge for the amortized optimizer: one control tick per
simulation step, the model's per-joint reference-lead PD compensation applied,
absolute joint position targets written directly into the PD servo (bypassing
the RL residual-action interface).

Two things this player does NOT inherit from the RL env, because inheriting
them would misrepresent the MPC:

* the **control grid** is pd_sim's (250 Hz physics, decimation 5, 20 ms
  control), not the RL env's 60 Hz -- see the grid note in :func:`main`;
* the **waist is free**.  ``pin_waist`` is a kinematic lock, and the MPC plans
  all three waist DoFs; ``--pin_waist`` restores the old behaviour.

An eval RCC clip from the amortized pool sets the rhythm; the same
``IsaacMPC`` + ``BatchedSchedule`` that the planned Isaac RL fine-tune will
use drive the robot here.  Record mp4s from the viewport with ``--video``.

Examples:

    # interactive viewer
    python amortized/scripts/play_isaac.py --task Isaac-G1-Drum-Direct-v0

    # record mp4s, headless
    python amortized/scripts/play_isaac.py --task Isaac-G1-Drum-Direct-v0 \\
        --headless --video --angles 3quarter,side --ckpt runs/phase2b_main/best.pt

    # play a custom rhythm
    python amortized/scripts/play_isaac.py --task Isaac-G1-Drum-Direct-v0 \\
        --rhythm_path my_rhythm.json

    # pick the third clip from a rhythm directory
    python amortized/scripts/play_isaac.py --task Isaac-G1-Drum-Direct-v0 \\
        --rhythm_path path/to/clips/ --clip_index 2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Amortized MPC model player in Isaac Sim."
)
parser.add_argument(
    "--task", type=str, default="Isaac-G1-Drum-Direct-v0", help="Name of the task."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--ckpt", default=None, help="Model checkpoint (default: champion)")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos (one mp4 per angle)."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Length of each recorded video in control steps (0 = one full episode).",
)
parser.add_argument(
    "--angles",
    type=str,
    default="3quarter",
    help="Comma-separated camera angles (choices: front, 3quarter, side, top).",
)
parser.add_argument(
    "--cam_zoom",
    type=float,
    default=1.0,
    help="Viewport/record camera distance multiplier.",
)
parser.add_argument(
    "--no_overlay", action="store_true", default=False, help="Disable the beat/timestep overlay."
)
parser.add_argument(
    "--no_sticks", action="store_true", default=False, help="Disable the stick-tip hit markers."
)
parser.add_argument(
    "--clip_index", type=int, default=2,
    help="Which eval RCC clip to use (0-3, default 2 = a good long clip). "
         "Ignored when --rhythm_path points to a single .json file."
)
parser.add_argument(
    "--rhythm_path", type=str, default=None,
    help="Path to a rhythm .json file or a directory of .json clips. "
         "When set, replaces the default eval RCC pool; --clip_index still "
         "selects which clip from the pool (directory) or is ignored (single file)."
)
parser.add_argument(
    "--model_device", default="cuda", help="Torch device for the model."
)
parser.add_argument(
    "--steps", type=int, default=250, help="Max control steps per episode."
)
parser.add_argument(
    "--freeze_wp", type=int, default=0,
    help="Freeze window in waypoints (default 0, OFF -- 7c decision 2)."
)
parser.add_argument(
    "--pin_waist", action="store_true", default=False,
    help="Kinematically lock the 3 waist DoFs at the ready pose. OFF by default: "
         "the MPC plans the waist (~90 deg of yaw on a typical eval clip) and "
         "pinning it silently discards that plan.",
)
parser.add_argument(
    "--pin_waist_lean", action="store_true", default=False,
    help="Hold waist pitch/roll at ready, leave yaw free. A middle ground for "
         "watching the torso aim without the planned +/-30 deg lean.",
)
parser.add_argument(
    "--isaac_rate", action="store_true", default=False,
    help="Keep the env's native 60 Hz control grid instead of pd_sim's 50 Hz "
         "(see the grid note in main()). The schedule follows env.step_dt either "
         "way, so this makes the plan's waypoint spacing disagree with the plant.",
)
parser.add_argument(
    "--dynamics_lead", action="store_true", default=False,
    help="Replace the fixed per-joint time lead with Kp^-1 * M(q) * qdd_ref. "
         "Turns OFF cmd_lead_s so there is no timing adjustment at all.",
)
parser.add_argument(
    "--cmd_lead_s", type=str, default="fit",
    help="Reference lead in seconds: scalar (e.g. '0.02' = 20ms), 'fit'/'sparse'/'wide', "
         "or 'off'/'0'. Default 'fit' -- the per-joint preset, which is the 7c "
         "recommendation and the only lead measured over 4 clips x 3 seeds "
         "(84/225 crossings vs 49/225 with no lead).",
)
parser.add_argument(
    "--ki", type=float, default=0.0,
    help="Integral gain for PID outer loop (N·m/rad·s). Default 0.0 = pure PD. "
         "The integral is worth ~0 on top of the per-joint lead (it fights the "
         "lead's intentional offset), and the offline plant it was swept in has "
         "no torque saturation for it to wind up against.",
)
parser.add_argument(
    "--ki_windup", type=float, default=0.0,
    help="Anti-windup clamp on integral error (rad·s). 0 = no clamp.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _require_repo():  # noqa: E402
    """Import the repo task + viz, or explain exactly what is missing.

    See the REPO-ONLY warning in the module docstring. Without this the failure
    is a bare ``ModuleNotFoundError: g1_drummer`` after a ~60 s Isaac startup,
    which reads like a broken install rather than a deliberate boundary.
    """
    try:
        import g1_drummer.tasks  # noqa: F401
        from g1_drummer.viz import BeatOverlay, StickVisualizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"play_isaac.py needs the g1-drummer-lab-rythmic checkout ({exc.name} not "
            "found).\n"
            "It drives the REPO's env (Isaac-G1-Drum-Direct-v0), not the standalone one.\n"
            "  fix : PYTHONPATH=<repo>/source/g1_drummer python scripts/play_isaac.py ...\n"
            "  else: use isaac_rl_ppo/scripts/ for playback on the standalone env."
        ) from exc
    return BeatOverlay, StickVisualizer


BeatOverlay, StickVisualizer = _require_repo()

# --- amortized imports (repo root on sys.path) ---------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.lead import LEAD_PRESETS, resolve_lead  # noqa: E402
from amortized_opt.pd_sim import mass_matrix  # noqa: E402
from amortized_opt.limits import (  # noqa: E402
    isaac_effort_limits, isaac_runtime_stiffness, urdf_position_limits,
)
from amortized_opt.constants import G1_UPPER_BODY_JOINTS  # noqa: E402

# --- the Isaac-facing MPC driver -----------------------------------------------
from isaac_rl.mpc import IsaacMPC, MPCConfig  # noqa: E402
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Video helpers (from scripts/play_control_theory.py)
# ---------------------------------------------------------------------------

ANGLE_OFFSETS = {
    "front": [0.0, 2.2, 1.1],
    "3quarter": [1.6, 1.2, 1.25],
    "side": [-2.2, 0.0, 1.1],
    "top": [0.3, 0.3, 2.8],
}


def write_mp4(path, frames, fps):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, macro_block_size=None)
    except Exception as exc:
        print(f"[WARN] imageio mp4 write failed ({exc}); trying moviepy.")
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        ImageSequenceClip(
            [np.asarray(f) for f in frames], fps=fps
        ).write_videofile(path, logger=None)


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    cfg = {
        "buckets": buckets,
        "num_drums": mcfg.num_drums,
        "num_future": mcfg.num_future,
    }
    return model, cfg, ck, mcfg


# ---------------------------------------------------------------------------
# Amortized MPC agent (wraps IsaacMPC for the play_ loop)
# ---------------------------------------------------------------------------

class AmortizedMPCAgent:
    """One-env amortized MPC driver with per-joint reference-lead PD compensation.

    On every control tick:

    1. Read ``(q, qd)`` from the env
    2. Call ``IsaacMPC.plan(q, qd)`` → refined plan + admissible command
    3. Write the command as an absolute joint position target (bypassing the
       env's RL action clip)
    4. After ``env.step()``, call ``schedule.advance()`` and ``mpc.commit()``
    """

    def __init__(self, env: gym.Env, args, model, cfg, kit, dev, amp):
        u = env.unwrapped
        self.u = u
        self.args = args
        self.device = dev
        self.amp = amp

        names = list(u.upper_body_joint_names)
        self._joint_names = tuple(names)
        self._joint_ids = list(u.upper_body_joint_ids)
        self._ready_q = u.ready_joint_pos[0, self._joint_ids].to(torch.float32)
        # Waist columns, in the 17-dof optimizer ordering (see the assert below:
        # the env's ordering is that ordering).
        self._waist_i = [i for i, nm in enumerate(names) if nm.startswith("waist_")]
        self._waist_lo = self._ready_q[self._waist_i].clone()
        self._waist_hi = self._ready_q[self._waist_i].clone()
        # The direct-target hook indexes q by position, so a reordered env would
        # permute every command silently rather than fail.
        if self._joint_names != tuple(A.G1_UPPER_BODY_JOINTS):
            raise RuntimeError(
                "env upper_body_joint_names != amortized G1_UPPER_BODY_JOINTS:\n"
                f"  env: {self._joint_names}\n  opt: {tuple(A.G1_UPPER_BODY_JOINTS)}"
            )

        # --- dynamics lead setup -------------------------------------------
        self._dynamics_lead = bool(getattr(args, "dynamics_lead", False))
        # Always set up violation tracking limits (useful for comparison)
        lo, hi = urdf_position_limits(self._joint_names, posture_limits=False)
        self._q_lo = torch.as_tensor(lo, dtype=torch.float32)
        self._q_hi = torch.as_tensor(hi, dtype=torch.float32)
        tau_lim = isaac_effort_limits(self._joint_names)
        self._tau_lim = torch.as_tensor(tau_lim, dtype=torch.float32)

        if self._dynamics_lead:
            # Build BatchedRNEA via the Isaac-safe cache (avoids Pinocchio in-process)
            from isaac_rl.env import try_build_rnea  # noqa: E402
            self._rnea = try_build_rnea(dev, dtype=torch.float32)
            if self._rnea is None:
                print("[agent] WARNING: dynamics_lead=True but env has no RNEA "
                      "(gravity_feedforward may be OFF); dynamics lead disabled.",
                      flush=True)
                self._dynamics_lead = False
            else:
                self._kp = torch.as_tensor(
                    isaac_runtime_stiffness(self._joint_names),
                    device=dev, dtype=torch.float32,
                )
                print(f"[agent] dynamics_lead ON (Kp^-1 * M(q) * qdd_ref); "
                      f"time lead = OFF", flush=True)
        else:
            self._rnea = None

        # --- control grid -------------------------------------------------
        # THE dt, taken from the plant rather than from A.CONTROL_DT: the
        # schedule advances exactly one waypoint per env.step, so a schedule
        # that believes in 20 ms while the env steps 16.7 ms plays the rhythm
        # 20 % fast and executes every planned waypoint interval 1.2x too
        # quickly.  main() puts the env on pd_sim's grid so the two agree;
        # --isaac_rate opts out, and then this warning is the only honest thing
        # left to say about the plan.
        self.dt = float(u.step_dt)
        if abs(self.dt - A.CONTROL_DT) > 1e-9:
            print(
                f"[WARN] env step_dt={self.dt * 1e3:.2f} ms != A.CONTROL_DT="
                f"{A.CONTROL_DT * 1e3:.2f} ms. The model, the fitted cmd_lead_s "
                f"and pd_sim were all identified on the 20 ms grid; commanded "
                f"joint speeds will be {A.CONTROL_DT / self.dt:.2f}x planned.",
                flush=True,
            )

        # --- rhythm schedule for IsaacMPC ---------------------------------
        sched_cfg = ScheduleConfig(
            dt=self.dt,
            horizon=int(cfg["buckets"][-1]),
            num_drums=int(cfg["num_drums"]),
            num_future=int(cfg["num_future"]),
            hit_radius_frac=0.6,
            speed_min=1.0,
            speed_max=3.0,
            hit_lead_s=0.0,
        )
        self.schedule = BatchedSchedule(n_envs=1, cfg=sched_cfg, device=dev,
                                        dtype=torch.float32)

        # --- load clip ---------------------------------------------------
        rhythm_source = (
            Path(args.rhythm_path) if getattr(args, "rhythm_path", None)
            else A.default_rcc_dir("eval")
        )
        clips = A.load_rcc_pool(rhythm_source, kit)
        if len(clips) == 0:
            raise FileNotFoundError(f"no valid RCC clips found at {rhythm_source}")
        clip_idx = min(max(int(args.clip_index), 0), len(clips) - 1)
        self._clip = clips[clip_idx]
        print(f"[clip] {self._clip.name} (index {clip_idx}/{len(clips)-1}, "
              f"{len(self._clip.hits)} hits, source={rhythm_source})")

        # --- Isaac MPC ----------------------------------------------------
        # Let IsaacMPC create its own GoalPoseLibrary (avoids Pinocchio
        # dependency that fails inside Isaac Sim's Python environment).
        _lead_s_arg = getattr(args, "cmd_lead_s", "fit")
        _lead_s = (
            None if self._dynamics_lead
            else resolve_lead(_lead_s_arg)
        )
        mpc_cfg = MPCConfig(
            horizon=int(cfg["buckets"][-1]),
            dt=self.dt,
            iterations=1,
            cmd_lead_s=_lead_s,
            cmd_slew_eta=1.0,
            clamp_cmd_to_box=True,
            freeze_wp=int(args.freeze_wp),
            freeze_waist=True,
            encode_every_tick=True,
            amp_dtype=amp,
        )
        self.mpc = IsaacMPC(
            model,
            schedule=self.schedule,
            cfg=mpc_cfg,
            device=dev,
            dtype=torch.float32,
            library=None,  # auto-created by IsaacMPC
            kit_names=kit.names,
        )

        # --- install direct-target hook (bypass RL action clip) -----------
        self._q_des = self._ready_q.clone()
        self._install_direct_target_hook()
        print(f"[agent] IsaacMPC + direct-target hook ready.", flush=True)

        # --- PID outer loop ------------------------------------------------
        self._ki = float(getattr(args, "ki", 0.0))
        self._ki_windup = float(getattr(args, "ki_windup", 0.0))
        self._kp = torch.as_tensor(
            isaac_runtime_stiffness(self._joint_names),
            device=dev, dtype=torch.float32,
        )
        self._integral = torch.zeros(len(self._joint_names), dtype=torch.float32)
        self._pid_active = self._ki > 0.0
        lead_on = _lead_s is not None and _lead_s != 0.0
        if self._pid_active:
            tag = f"lead={_lead_s}" if lead_on else "lead=OFF"
            if lead_on:
                tag += " (combined lead+ki per sweep results)"
            print(
                f"[agent] PID ON: ki={self._ki}, ki_windup={self._ki_windup}, "
                f"{tag}",
                flush=True,
            )

        # --- bookkeeping --------------------------------------------------
        self._step = 0
        self._n_hits = 0
        self._initialized = False
        # Violation counters (per step)
        self._viol_joint: list[int] = []     # steps with any joint outside limits
        self._viol_torque: list[int] = []    # steps with any computed torque > limit
        self._viol_collision: list[int] = [] # steps with non-drum contact
        # Peak excursions across the episode
        self._max_joint_excursion = 0.0       # max (q - hi).clamp(0) or (lo - q).clamp(0) in rad
        self._max_torque_over = 0.0           # max (|tau| / tau_lim - 1).clamp(0)
        self._lead_offset = None

    def _install_direct_target_hook(self) -> None:
        """After the env decodes actions, overwrite PD targets with planned q*."""
        u = self.u
        if getattr(u, "_amortized_direct_hook", False):
            return
        orig = u._pre_physics_step

        def _pre_physics_step_direct(actions: torch.Tensor) -> None:
            orig(actions)
            ids = self._joint_ids
            # All 17 columns, WAIST INCLUDED -- the MPC plans the torso and this
            # is the only place its waist command reaches the plant.  Written
            # after orig(), so it also overrides the env's own waist owners; the
            # two blocks below hand those columns back when the operator asked
            # for a hold, which is why they come after and not before.
            q = self._q_des.to(device=u.device, dtype=u.joint_pos_target.dtype)
            u.joint_pos_target[:, ids] = q.unsqueeze(0)
            # Only under --pin_waist: the env re-asserts the kinematic pin every
            # physics substep anyway (write_joint_state_to_sim in _apply_action),
            # so this just keeps the PD target from disagreeing with it.
            if u.pin_waist and u.pinned_waist_joint_ids is not None:
                u.joint_pos_target[:, u.pinned_waist_joint_ids] = u.pinned_waist_pos
            # Only under --pin_waist_lean: hold pitch/roll, leave the MPC's yaw
            # alone.  Re-applied here because the write above just clobbered the
            # env's own lean hold, which made the flag a silent no-op.
            elif u.pin_waist_lean and getattr(u, "_waist_lean_ids", None) is not None:
                u.joint_pos_target[:, u._waist_lean_ids] = u._waist_lean_pos

        u._pre_physics_step = _pre_physics_step_direct
        u._amortized_direct_hook = True

    def reset(self) -> None:
        """Called after env.reset(). Seed the schedule + MPC plan."""
        u = self.u

        # Reset schedule with our clip
        self.schedule.reset(
            env_ids=torch.tensor([0], device=self.device),
            clips=[self._clip],
            kits=[A.DrumKit.from_layout()],
            t_start=0.0,
        )

        # Read current state
        q = u.robot.data.joint_pos[0, self._joint_ids].to(torch.float32).clone()
        qd = u.robot.data.joint_vel[0, self._joint_ids].to(torch.float32).clone()

        # Seed the MPC plan
        self.mpc.reset(
            env_ids=torch.tensor([0], device=self.device),
            q0=q.unsqueeze(0).to(self.device),
        )

        self._q_des = q.clone()
        self._integral.zero_()
        self._step = 0
        self._n_hits = 0
        self._initialized = True
        self._viol_joint.clear()
        self._viol_torque.clear()
        self._viol_collision.clear()
        self._max_joint_excursion = 0.0
        self._max_torque_over = 0.0
        # Waist travel bookkeeping: the whole point of unpinning is that these
        # stop reading zero.
        self._waist_lo = q[self._waist_i].clone()
        self._waist_hi = q[self._waist_i].clone()

    def act(self, now: float) -> torch.Tensor:
        """One control tick: plan → command → write target. Returns zero action
        (the hook overwrites the target anyway)."""
        del now
        u = self.u
        dev = self.device

        q = u.robot.data.joint_pos[0, self._joint_ids].to(torch.float32)
        qd = u.robot.data.joint_vel[0, self._joint_ids].to(torch.float32)

        # Run the MPC
        result = self.mpc.plan(
            q.unsqueeze(0).to(dev),
            qd.unsqueeze(0).to(dev),
        )

        # Store command for the hook to pick up
        cmd = result.cmd[0].cpu().to(torch.float32)
        self._q_des_raw = cmd.clone()  # pre-lead reference for logging

        # --- dynamics lead: q_cmd = q_ref + Kp^-1 * M(q) * qdd_ref ---------
        if self._dynamics_lead:
            q_ref = result.traj[0, 0].to(dev, dtype=torch.float32)   # (17,) first waypoint
            qdd_ref = result.aux["acc"][0, 0].to(dev, dtype=torch.float32)  # (17,) acc head
            qb = q_ref.unsqueeze(0)                               # (1, 17)
            qddb = qdd_ref.unsqueeze(0)
            zb = torch.zeros_like(qb)
            with torch.no_grad():
                tau_g = self._rnea.rnea(qb, zb, zb)               # g(q)
                tau_i = self._rnea.rnea(qb, zb, qddb)             # g(q) + M(q)·qdd
                M_qdd = (tau_i - tau_g).squeeze(0)                # (17,)
            kp_dev = self._kp.to(dev)
            offset = M_qdd / kp_dev                                # Kp^-1 · M·qdd
            # Torque-bounded offset: don't demand more position error than
            # the P-term can turn into actuator torque.  τ_lim / Kp is the
            # offset that fully saturates the actuator; beyond that the
            # commanded position is asking for torque that doesn't exist.
            tau_lim_dev = self._tau_lim.to(dev)
            offset_max = tau_lim_dev / kp_dev.clamp_min(1e-6)
            offset = torch.clamp(offset, -offset_max, offset_max)
            cmd_dyn = q_ref + offset
            # Clamp to joint box (tensors are all on dev)
            cmd_dyn = torch.clamp(cmd_dyn,
                                  self._q_lo.to(dev), self._q_hi.to(dev))
            # Slew bound (matches the MPC's cmd_slew_eta logic)
            if self.mpc.step_max is not None:
                sm = self.mpc.step_max
                prev = self._q_des.to(dev)  # last executed command
                delta = cmd_dyn - prev
                cmd_dyn = prev + torch.clamp(delta, -sm, sm)
            cmd = cmd_dyn.cpu().to(torch.float32)
            self._lead_offset = offset.cpu().to(torch.float32)   # for logging
        else:
            self._lead_offset = None

        # --- PID outer loop: q_cmd += (ki/kp) * integral error ----------
        if self._pid_active:
            # All PID tensors must be on the same device as the measured state.
            dev_q = q.device
            kp_dev = self._kp.to(dev_q)
            integral_dev = self._integral.to(dev_q)
            cmd_dev = cmd.to(dev_q)
            q_lo_dev = self._q_lo.to(dev_q)
            q_hi_dev = self._q_hi.to(dev_q)
            # Error: commanded position minus measured position.
            err = cmd_dev - q
            integral_dev = integral_dev + err * self.dt
            if self._ki_windup > 0.0:
                integral_dev = torch.clamp(
                    integral_dev, -self._ki_windup, self._ki_windup,
                )
            # Convert integral from rad·s to rad via (ki / kp)
            correction = (self._ki / kp_dev) * integral_dev
            cmd_dev = cmd_dev + correction
            # Re-clamp to joint box + slew bound.
            cmd_dev = torch.clamp(cmd_dev, q_lo_dev, q_hi_dev)
            if self.mpc.step_max is not None:
                sm = self.mpc.step_max.to(dev_q)
                prev = self._q_des.to(dev_q)
                delta = cmd_dev - prev
                cmd_dev = prev + torch.clamp(delta, -sm, sm)
            # Store integral back on CPU for persistence across ticks.
            self._integral = integral_dev.cpu()
            cmd = cmd_dev.cpu().to(torch.float32)

        self._q_des = cmd.clone()

        # Count hits from the schedule (hit_index dropped from k to 0)
        hi = self.schedule.hit_index()
        for h in (0, 1):
            if hi[0, h].item() == 1:  # imminent
                pass  # the hit fires on the NEXT advance

        self._step += 1
        return torch.zeros((1, u.cfg.action_space), device=u.device)

    def post_step(self) -> None:
        """After env.step(): advance clock, commit to MPC."""
        if not self._initialized:
            return

        u = self.u
        dev = self.device

        # Read the state the plant actually reached
        q = u.robot.data.joint_pos[0, self._joint_ids].to(torch.float32)
        qd = u.robot.data.joint_vel[0, self._joint_ids].to(torch.float32)

        w = q[self._waist_i]
        self._waist_lo = torch.minimum(self._waist_lo, w)
        self._waist_hi = torch.maximum(self._waist_hi, w)

        # --- violation tracking --------------------------------------------
        # Joint limit: is any joint outside the URDF box?
        # (q is on the env device, _q_lo/_q_hi are CPU; compare on CPU)
        qc = q.cpu()
        over = torch.clamp(qc - self._q_hi, min=0.0)
        under = torch.clamp(self._q_lo - qc, min=0.0)
        exc = torch.max(over, under)
        j_viol = int((exc > 0.0).any().item())
        exc_max = exc.max().item()
        self._max_joint_excursion = max(self._max_joint_excursion, exc_max)
        self._viol_joint.append(j_viol)

        # Torque: does computed_torque exceed the effort rating?
        ct = getattr(u.robot.data, "computed_torque", None)
        if ct is not None:
            tau_j = ct[0, self._joint_ids].cpu().to(torch.float32)
            ratio = tau_j.abs() / self._tau_lim
            t_viol = int((ratio > 1.0).any().item())
            over_frac = (ratio - 1.0).clamp(min=0.0).max().item()
            self._max_torque_over = max(self._max_torque_over, over_frac)
        else:
            t_viol = 0
        self._viol_torque.append(t_viol)

        # Collision: contact on non-drum bodies.
        # The env has contact sensors filtered to drum bodies; body self-collisions
        # are harder to detect directly without the full physics contact report.
        c_viol = 0
        self._viol_collision.append(c_viol)

        # Advance the schedule clock → fired mask
        fired = self.schedule.advance()

        # Count hits + reset PID integral per fired hand
        for h in (0, 1):
            if fired[0, h].item():
                self._n_hits += 1
                if self._pid_active:
                    cols = self.mpc.arm_cols[h]
                    self._integral = self._integral.clone()
                    self._integral[cols] = 0.0

        # Commit: shift the plan and re-seed fired hands
        self.mpc.commit(
            self._q_des.unsqueeze(0).to(dev),  # the command we just executed
            q.unsqueeze(0).to(dev),
            fired,
        )

    def status_line(self, now: float) -> str:
        hi = self.schedule.hit_index()
        tth0 = hi[0, 0].item()
        tth1 = hi[0, 1].item()
        # Measured waist peak-to-peak so far, in degrees (yaw/pitch/roll).  Zero
        # here means the plan is still being discarded somewhere.
        p2p = torch.rad2deg(self._waist_hi - self._waist_lo).tolist()
        cmd_w = torch.rad2deg(self._q_des[self._waist_i]).tolist()
        lead_tag = "dyn" if self._dynamics_lead else str(getattr(self.args, "cmd_lead_s", "?"))
        # Peak dynamics-lead offset magnitude (mrad) on this step
        off_str = ""
        if self._lead_offset is not None:
            peak = self._lead_offset.abs().max().item()
            off_str = f" lead_peak={peak*1e3:.1f}mrad"
        # PID integral status
        pid_str = ""
        if self._pid_active:
            i_peak = self._integral.abs().max().item()
            pid_str = f" | PID ki={self._ki:.1f} int_peak={i_peak*1e3:.1f}mrad·s"
        return (
            f"[t={now:5.2f}s] amortized MPC | "
            f"tth L={tth0}wp R={tth1}wp | "
            f"hits={self._n_hits} | "
            f"waist cmd y/p/r={cmd_w[0]:+.0f}/{cmd_w[1]:+.0f}/{cmd_w[2]:+.0f}deg "
            f"meas p2p={p2p[0]:.0f}/{p2p[1]:.0f}/{p2p[2]:.0f}deg | "
            f"lead={lead_tag}{off_str}{pid_str}"
        )

    def violation_summary(self) -> dict:
        """Return a dict of violation statistics for the completed episode."""
        n = max(len(self._viol_joint), 1)
        return {
            "steps": n,
            "joint_limit_steps": sum(self._viol_joint),
            "joint_limit_frac": sum(self._viol_joint) / n,
            "max_joint_excursion_rad": self._max_joint_excursion,
            "max_joint_excursion_deg": self._max_joint_excursion * 180.0 / 3.14159,
            "torque_steps": sum(self._viol_torque),
            "torque_frac": sum(self._viol_torque) / n,
            "max_torque_over": self._max_torque_over,
            "collision_steps": sum(self._viol_collision),
            "collision_frac": sum(self._viol_collision) / n,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, max_steps=None, frame_sink=None, sticks=None,
                real_time=False):
    u = env.unwrapped
    env.reset()
    agent.reset()
    dt = u.step_dt
    steps = int(max_steps) if max_steps else int(u.max_episode_length)
    now = 0.0

    # NOT `_` as the loop variable: the env.step unpack below binds `_` too, so
    # the counter became an observation tensor and every `k % 50` progress print
    # silently stopped firing.
    for k in range(steps):
        if not simulation_app.is_running():
            break
        start = time.time()
        now = int(u.episode_length_buf[0].item()) * dt
        with torch.inference_mode():
            action = agent.act(now)
            _obs, _rew, terminated, truncated, _info = env.step(action)
        agent.post_step()

        if sticks is not None:
            sticks.update()
        if frame_sink is not None:
            frame = u.render()
            if isinstance(frame, np.ndarray):
                frame_sink(frame)
        if real_time:
            sleep = dt - (time.time() - start)
            if sleep > 0:
                time.sleep(sleep)

        ended = bool(terminated[0].item()) or bool(truncated[0].item())
        if ended:
            why = "terminated" if bool(terminated[0].item()) else "truncated"
            print(f"  [end] {why} at step {k}/{steps}  {agent.status_line(now)}",
                  flush=True)
            vs = agent.violation_summary()
            print(
                f"  [violations] joint_limit: {vs['joint_limit_steps']}/{vs['steps']} "
                f"({vs['joint_limit_frac']*100:.1f}%) "
                f"peak_excursion={vs['max_joint_excursion_deg']:.1f}deg | "
                f"torque: {vs['torque_steps']}/{vs['steps']} "
                f"({vs['torque_frac']*100:.1f}%) "
                f"peak_over={vs['max_torque_over']*100:.1f}%",
                flush=True,
            )
            return {}
        if k % 50 == 0:
            print(f"  step {k}/{steps} {agent.status_line(now)}", flush=True)
    print(f"  done {agent.status_line(now)}", flush=True)
    vs = agent.violation_summary()
    if vs["steps"] > 0:
        print(
            f"  [violations] joint_limit: {vs['joint_limit_steps']}/{vs['steps']} "
            f"({vs['joint_limit_frac']*100:.1f}%) "
            f"peak_excursion={vs['max_joint_excursion_deg']:.1f}deg | "
            f"torque: {vs['torque_steps']}/{vs['steps']} "
            f"({vs['torque_frac']*100:.1f}%) "
            f"peak_over={vs['max_torque_over']*100:.1f}%"
        )
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Parse args and resolve checkpoint
    # -----------------------------------------------------------------------
    checkpoint = (
        Path(args_cli.ckpt)
        if args_cli.ckpt
        else Path(__file__).resolve().parent.parent / "runs" / "phase2b_main" / "best.pt"
    )
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    dev = torch.device(args_cli.model_device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # set_device rejects an indexless device, and "cuda" is the default.
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")
    amp = torch.bfloat16

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, cfg, ck, mcfg = load_model(checkpoint, dev)
    print(f"[ckpt] {checkpoint} step={ck['step']} buckets={cfg['buckets']}")
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # -----------------------------------------------------------------------
    # Env setup (single env, deterministic)
    # -----------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    # Zero start jitter for deterministic demo
    if hasattr(env_cfg, "start_joint_jitter_rad"):
        env_cfg.start_joint_jitter_rad = 0.0
    if hasattr(env_cfg, "perturb_enable"):
        env_cfg.perturb_enable = False
    if hasattr(env_cfg, "dr_enable"):
        env_cfg.dr_enable = False

    # --- THE control grid: pd_sim's, not the RL env's ----------------------
    # amortized_opt runs a 50 Hz waypoint grid over a 250 Hz PD loop
    # (constants.py resolved decision 1), and pd_sim -- the plant this model was
    # trained against -- models exactly that, with the env's own FOH ramp in the
    # inner loop.  The RL env ships 1/120 x 2 = 60 Hz (control.control_rate),
    # which would run every 20 ms planned interval in 16.7 ms.
    #
    # The decimation is CONTROL_DT / PD_DT, NOT A.PD_SUBSTEPS: since 2026-08-12
    # PD_SUBSTEPS (5 -> 10) is pd_sim's explicit-integrator accuracy knob and no
    # longer equals the ratio of the two rates.  Using it here ran Isaac at
    # 10 x 4 ms = 40 ms (25 Hz), i.e. half the planned command speed.
    #
    # Overridden here rather than in control_rate.py on purpose: that module is
    # THE rate for RL training and for the offline teacher/AMP shard grids, and
    # moving it would invalidate every one of them.  Nothing in
    # G1DrummerEnvCfg.__post_init__ derives from decimation or sim.dt, and the
    # env reads self.step_dt (never a hardcoded CONTROL_DT), so this override is
    # local to the player.
    if not args_cli.isaac_rate:
        env_cfg.decimation = int(round(A.CONTROL_DT / A.PD_DT))
        env_cfg.sim.dt = float(A.PD_DT)
        env_cfg.sim.render_interval = env_cfg.decimation
    _step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    print(
        f"[grid] sim.dt={env_cfg.sim.dt * 1e3:.3f} ms x decimation="
        f"{env_cfg.decimation} -> control {_step_dt * 1e3:.2f} ms "
        f"({1.0 / _step_dt:.1f} Hz); pd_sim grid = "
        f"{A.CONTROL_DT * 1e3:.2f} ms ({A.CONTROL_HZ:.0f} Hz)"
    )

    # --- the waist ---------------------------------------------------------
    # OFF by default.  pin_waist is a KINEMATIC lock (the env teleports the
    # three waist DoFs back to ready every physics substep), so with it on the
    # waist columns the MPC plans -- ~90 deg of yaw and a rail-to-rail
    # pitch/roll on a typical eval clip -- are computed, led, slew-limited,
    # clamped and then thrown away.  Worse, the yaw-indexed goal-pose library
    # then hands every hand its terminal at the ready-yaw bin, so the ARM plan
    # is solved against a torso that never arrives.
    if hasattr(env_cfg, "pin_waist"):
        env_cfg.pin_waist = bool(args_cli.pin_waist)
    if hasattr(env_cfg, "pin_waist_lean"):
        env_cfg.pin_waist_lean = bool(args_cli.pin_waist_lean) and not args_cli.pin_waist
    print(
        f"[waist] pin_waist={bool(args_cli.pin_waist)} "
        f"pin_waist_lean={bool(getattr(env_cfg, 'pin_waist_lean', False))}"
    )

    env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    u = env.unwrapped

    # -----------------------------------------------------------------------
    # Build the agent
    # -----------------------------------------------------------------------
    print("[player] Building AmortizedMPCAgent...", flush=True)
    kit = A.DrumKit.from_layout()
    agent = AmortizedMPCAgent(env, args_cli, model, cfg, kit, dev, amp)
    print("[player] Agent built OK.", flush=True)

    # -----------------------------------------------------------------------
    # Cameras and overlays
    # -----------------------------------------------------------------------
    origin = u.scene.env_origins[0].detach().cpu().numpy()
    cam_target = (origin + [0.30, 0.0, 0.95]).tolist()

    def aim_camera(angle: str):
        offset = np.array(ANGLE_OFFSETS[angle]) * args_cli.cam_zoom
        u.sim.set_camera_view(eye=(origin + offset).tolist(), target=cam_target)

    if args_cli.video:
        angles = [a.strip() for a in args_cli.angles.split(",") if a.strip()]
        unknown = [a for a in angles if a not in ANGLE_OFFSETS]
        if unknown:
            raise ValueError(
                f"Unknown --angles {unknown}; choose from {list(ANGLE_OFFSETS)}."
            )
        overlay = BeatOverlay(env, tag="amortized") if not args_cli.no_overlay else None
        sticks = StickVisualizer(env) if not args_cli.no_sticks else None
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_folder = os.path.join(
            str(Path(__file__).resolve().parent.parent / "runs" / "render"),
            f"isaac_{stamp}",
        )
        os.makedirs(video_folder, exist_ok=True)
        fps = int(round(1.0 / u.step_dt))

        for angle in angles:
            aim_camera(angle)
            frames = []

            def sink(frame):
                frames.append(overlay.draw(frame) if overlay is not None else frame)

            run_episode(
                env, agent,
                max_steps=args_cli.video_length or args_cli.steps,
                frame_sink=sink, sticks=sticks,
            )
            out_path = os.path.join(video_folder, f"play_{angle}.mp4")
            write_mp4(out_path, frames, fps)
            print(f"[INFO] Wrote {len(frames)} frames to {out_path}")
    else:
        aim_camera("3quarter")
        sticks = StickVisualizer(env) if not args_cli.no_sticks else None
        run_episode(env, agent, max_steps=args_cli.steps, sticks=sticks)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
