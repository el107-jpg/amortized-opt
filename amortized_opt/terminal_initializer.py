"""Terminal-pose head and analytic trajectory initialization.

The transformer encoder and this small MLP run side by side.  The MLP sees
the same current goals/kit geometry plus the measured start state, predicts a
terminal joint pose, and :func:`strike_seed` expands that pose into the
decoder's initial trajectory.  By default the terminal branch is frozen and
its result is detached: trajectory losses train the decoder, never the IK
network. Joint fine-tuning can explicitly enable this entire gradient path.
"""

from __future__ import annotations

from pathlib import Path
from functools import wraps

import numpy as np
import torch
import torch.nn as nn

from .constants import G1_UPPER_BODY_JOINTS, N_DOF, STICK_LENGTH
from .fk import stick_axis_local, stick_mount_transform
from .rhythm import DrumKit
from .strike import CYMBAL_CONTACT_SIGMA_FALLBACK, is_cymbal_name
from .strike_seed import strike_seed
from .terminal_pose import TerminalPoseNet, feature_dim, pack_features
from .tokenizer import ContextInputs, WaypointMasks

__all__ = ["TerminalInitializer"]


def _terminal_grad_mode(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            return fn(self, *args, **kwargs)
    return wrapped


class TerminalInitializer(nn.Module):
    """Terminal MLP plus analytic strike seed, frozen unless explicitly trainable."""

    def __init__(
        self,
        num_drums: int,
        *,
        width: int = 128,
        depth: int = 2,
        dt: float = 0.02,
        hihat_library_fallback: bool = True,
        trainable: bool = False,
        joint_names=G1_UPPER_BODY_JOINTS,
    ):
        super().__init__()
        self.num_drums = int(num_drums)
        self.dt = float(dt)
        self.hihat_library_fallback = bool(hihat_library_fallback)
        self.trainable = bool(trainable)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != N_DOF:
            raise ValueError(f"terminal initializer requires {N_DOF} joints")
        self.head = TerminalPoseNet(
            in_dim=feature_dim(self.num_drums), width=int(width), depth=int(depth)
        )

        axes, origins = [], []
        for hand in ("left", "right"):
            axes.append(stick_axis_local(hand))
            origins.append(stick_mount_transform(hand)[1])
        self.register_buffer("nominal_axis", torch.as_tensor(np.stack(axes), dtype=torch.float32))
        self.register_buffer(
            "nominal_origin", torch.as_tensor(np.stack(origins), dtype=torch.float32)
        )
        kit = DrumKit.from_layout()
        if self.num_drums < kit.num_drums:
            raise ValueError(
                f"terminal num_drums={self.num_drums} cannot hold the {kit.num_drums}-drum kit"
            )
        cymbal = [is_cymbal_name(n) for n in kit.names]
        cymbal.extend([False] * (self.num_drums - len(cymbal)))
        self.register_buffer("cymbal_by_drum", torch.as_tensor(cymbal, dtype=torch.bool))
        # Old frozen heads never saw right-hand hi-hat. A verified library
        # seed supplies that missing branch without an iterative runtime IK.
        # Checkpoint keys remain unchanged.
        self.hihat_index = kit.names.index('hihat') if 'hihat' in kit.names else -1
        from .warmstart import GoalPoseLibrary
        library = GoalPoseLibrary()
        rows = library.entries('right', 'hihat') if library.has('right', 'hihat') else []
        self.register_buffer('_hihat_right_poses', torch.tensor(
            np.asarray([r.q for r in rows]).reshape(-1, N_DOF), dtype=torch.float32), persistent=False)
        self.set_trainable(self.trainable)

    def set_trainable(self, enabled: bool) -> "TerminalInitializer":
        self.trainable = bool(enabled)
        self.head.requires_grad_(self.trainable)
        self.head.train(self.training if self.trainable else False)
        return self

    def freeze(self) -> "TerminalInitializer":
        """Freeze the terminal solver; the analytic seed has no learned state."""
        return self.set_trainable(False)

    def train(self, mode: bool = True):
        # Frozen heads stay in eval mode; explicitly trainable heads follow
        # the parent model's train/eval mode.
        super().train(mode)
        self.head.train(mode if self.trainable else False)
        return self

    def load_pretrained(self, path: str | Path, *, map_location="cpu") -> dict:
        """Load a checkpoint produced by ``scripts/train_terminal_pose.py``."""
        ck = torch.load(Path(path), map_location=map_location, weights_only=False)
        if bool(ck.get("optnet", False)):
            raise ValueError("the integrated terminal head expects the plain MLP checkpoint")
        expected = feature_dim(self.num_drums)
        if int(ck.get("in_dim", expected)) != expected:
            raise ValueError(
                f"terminal checkpoint in_dim={ck.get('in_dim')} but num_drums="
                f"{self.num_drums} requires {expected}"
            )
        if int(ck.get("width", self.head.width)) != self.head.width:
            raise ValueError("terminal checkpoint width does not match ModelConfig")
        if int(ck.get("depth", self.head.depth)) != self.head.depth:
            raise ValueError("terminal checkpoint depth does not match ModelConfig")
        self.head.load_state_dict(ck["model"], strict=True)
        if ck.get("all_hand_drum_goals", False):
            self.hihat_library_fallback = False
        self.set_trainable(self.trainable)
        return ck

    def _features(
        self, ctx: ContextInputs, state: torch.Tensor, masks: WaypointMasks
    ) -> torch.Tensor:
        if ctx.num_drums < self.num_drums:
            raise ValueError(
                f"context has {ctx.num_drums} drums, terminal head needs {self.num_drums}"
            )
        q0 = state[:, :N_DOF].float()
        v0 = state[:, N_DOF:].float()
        active = ctx.goal_valid.bool()
        drum_index = torch.where(
            active, ctx.goal_drum, torch.full_like(ctx.goal_drum, -1)
        )
        valid_stick = ctx.stick_valid.bool()
        axis0 = self.nominal_axis.to(device=q0.device).unsqueeze(0).expand(q0.shape[0], -1, -1)
        axis = torch.where(valid_stick.unsqueeze(-1), ctx.stick_axis.float(), axis0)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        nominal_sigma = 0.5 * float(STICK_LENGTH)
        tip_sigma = torch.where(
            valid_stick,
            ctx.stick_length.float() - 0.5 * float(STICK_LENGTH),
            q0.new_full((q0.shape[0], 2), nominal_sigma),
        )
        origin0 = self.nominal_origin.to(device=q0.device).unsqueeze(0).expand_as(axis)
        # Context carries the tip offset rather than mount origin.  They are
        # exactly related by tip = origin + axis * tip_sigma.
        origin = torch.where(
            valid_stick.unsqueeze(-1),
            ctx.stick_tip.float() - axis * tip_sigma.unsqueeze(-1),
            origin0,
        )
        shaft_lo = torch.minimum(
            tip_sigma - 1e-3,
            q0.new_full(tip_sigma.shape, float(CYMBAL_CONTACT_SIGMA_FALLBACK)),
        )
        cym = self.cymbal_by_drum[drum_index.clamp_min(0)] & active
        return pack_features(
            ctx.drum_pos[:, :self.num_drums].float(),
            ctx.drum_axis[:, :self.num_drums].float(),
            ctx.drum_radius[:, :self.num_drums].float(),
            drum_index, active, axis, origin, tip_sigma, shaft_lo, cym,
            q0, v0, masks.hit_tth.float(),
        )

    @_terminal_grad_mode
    def terminal_pose(
        self, ctx: ContextInputs, state: torch.Tensor, masks: WaypointMasks
    ) -> torch.Tensor:
        """Predict in float32; gradients are opt-in through the whole seed path."""
        device_type = "cuda" if state.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            pose = self.head(self._features(ctx, state, masks))
            if not self.hihat_library_fallback or self.hihat_index < 0 or not len(self._hihat_right_poses):
                return pose
            live = (ctx.goal_valid[:, 1].bool() & (ctx.goal_drum[:, 1] == self.hihat_index) &
                    (ctx.goal_pos[:, 1].norm(dim=-1) > 1e-6))
            pose = pose.clone()
            from .ik import arm_joint_indices
            yaw = torch.where(ctx.aim_valid, ctx.aim_yaw, pose[:, 0])
            rows = self._hihat_right_poses.to(pose)
            nearest = (yaw[:, None]-rows[None, :, 0]).abs().argmin(-1)
            chosen = rows[nearest]
            columns = arm_joint_indices('right')
            # Dense selection avoids CPU/GPU synchronization or per-row loops.
            pose[:, columns] = torch.where(live[:, None], chosen[:, columns], pose[:, columns])
            single = live & ~ctx.goal_valid[:, 0].bool()
            pose[:, :3] = torch.where(single[:, None], chosen[:, :3], pose[:, :3])
            return pose

    @_terminal_grad_mode
    def seed_from_terminal(
        self,
        ctx: ContextInputs,
        state: torch.Tensor,
        masks: WaypointMasks,
        terminal: torch.Tensor,
    ) -> torch.Tensor:
        """Expand a supplied terminal pose into the analytic decoder seed.

        Active arms follow the ramped-constant-acceleration joint-space line.
        Waist and idle arms settle with the same seed's rest quintic.  The
        terminal is not recomputed here, so a caller can freeze one pose and
        redraw the line from a later measured state.
        """
        q0 = state[:, :N_DOF].float()
        v0 = state[:, N_DOF:].float()
        terminal = terminal.to(device=q0.device, dtype=torch.float32)
        if terminal.shape != q0.shape:
            raise ValueError(
                f"terminal shape {tuple(terminal.shape)} != state joints {tuple(q0.shape)}"
            )
        B, H = state.shape[0], masks.elem_mask.shape[1]
        left = (masks.tau0_left.float() / self.dt).round().clamp_min(1)
        right = (masks.tau0_right.float() / self.dt).round().clamp_min(1)
        waist = torch.maximum(left, right)
        lengths = q0.new_empty(B, N_DOF)
        for j, name in enumerate(self.joint_names):
            lengths[:, j] = left if name.startswith("left_") else (
                right if name.startswith("right_") else waist
            )
        seed = strike_seed(
            q0, v0, terminal, lengths, H, self.dt,
            active=ctx.goal_valid.bool(), joint_names=self.joint_names,
        )
        # Match the existing warm-start builders' final admissibility step.
        # The head endpoints are already in the box; this only catches an
        # intermediate overshoot caused by a large measured starting velocity.
        seed = seed.clamp(
            self.head.q_lo.to(seed).view(1, 1, -1),
            self.head.q_hi.to(seed).view(1, 1, -1),
        )
        return seed * masks.elem_mask.float()

    @_terminal_grad_mode
    def forward(
        self, ctx: ContextInputs, state: torch.Tensor, masks: WaypointMasks
    ) -> torch.Tensor:
        """Return the detached decoder seed, shape ``(B, H, 17)``."""
        return self.seed_from_terminal(
            ctx, state, masks, self.terminal_pose(ctx, state, masks)
        )
