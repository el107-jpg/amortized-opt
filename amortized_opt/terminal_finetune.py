"""Original terminal-generator objective and bounded joint-training gradients."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from .starts import StartSampler
from .terminal_pose import TerminalPoseAL, TerminalPoseConfig, TerminalPoseCost, sample_terminal_batch, collision_violation


class TerminalRoleLoss:
    """Retain the pretrained objective/AL prices on its original task law.

    Prices can be fixed or updated after committed optimizer steps. Probe
    evaluation accepts explicit price snapshots and never updates AL state.
    """

    def __init__(self, checkpoint, device, *, w_com=None, dynamic_al=False,
                 lam_max=None, contact_scoring_mode=None, physics_loss_version=None):
        checkpoint = Path(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = TerminalPoseConfig(**saved["config"])
        saved_contact_mode=config.contact_scoring_mode
        saved_physics_version=config.physics_loss_version
        if physics_loss_version is not None:
            config.physics_loss_version=physics_loss_version
        if contact_scoring_mode is not None:
            if contact_scoring_mode not in ('legacy','isaac_v4','isaac_v5'):
                raise ValueError('Unknown terminal-role contact mode')
            config.contact_scoring_mode=contact_scoring_mode
        contact_al_reset=(config.contact_scoring_mode!=saved_contact_mode
                          or config.physics_loss_version!=saved_physics_version)
        if w_com is not None:
            if not np.isfinite(w_com) or w_com < 0:
                raise ValueError("terminal COM weight must be finite and nonnegative")
            config.w_com = float(w_com)
        self.cost = TerminalPoseCost(config, device=device)
        if lam_max is not None and (not np.isfinite(lam_max) or lam_max <= 0):
            raise ValueError("terminal AL multiplier ceiling must be finite and positive")
        self.al = TerminalPoseAL(**({} if lam_max is None else {"lam_max": float(lam_max)}))
        if not contact_al_reset:self.al.load_state_dict(saved["al"])
        self.dynamic_al = bool(dynamic_al)
        self.committed_al_updates = 0
        self.device = device
        train_cfg = saved["train_config"]
        self.start_bank = None
        size = int(train_cfg.get("start_bank_size", 4096))
        if size:
            rng = np.random.default_rng(int(train_cfg.get("seed", 0))+7919)
            sampler = StartSampler(device=device, dtype=torch.float32)
            bank = [sampler.sample(min(512, size-i), rng=rng) for i in range(0, size, 512)]
            self.start_bank = (torch.cat([b.q.detach() for b in bank]),
                               torch.cat([b.qd.detach() for b in bank]))
        self.metadata = dict(checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            config=asdict(self.cost.cfg), weights=dict(self.weights), start_bank_size=size,
            original_config=saved["config"],
            overrides=dict(**({} if w_com is None else {"w_com": float(w_com)}),
                **({} if contact_scoring_mode is None else {"contact_scoring_mode":contact_scoring_mode}),
                **({} if physics_loss_version is None else {"physics_loss_version":physics_loss_version})),
            contact_al_reset_on_upgrade=contact_al_reset,
            start_bank_seed=int(train_cfg.get("seed", 0))+7919,
            objective=(f"{config.contact_scoring_mode} pose contact geometry; contact AL reset on upgrade"
                       if contact_al_reset else
                       "Original full terminal loss; AL updates after commits"
                       if self.dynamic_al else
                       "Original full terminal loss; AL prices held fixed"))

    @property
    def weights(self):
        return self.al.cost_weights()

    def state_dict(self):
        return dict(self.metadata, version=1, weights=dict(self.weights),
                    dynamic_al=self.dynamic_al, al=self.al.state_dict(),
                    al_config=asdict(self.al.cfg), committed_al_updates=self.committed_al_updates)

    def load_state_dict(self, state):
        if state and "al" in state:
            previous_mode=(state.get('config') or {}).get('contact_scoring_mode','legacy')
            previous_physics=(state.get('config') or {}).get('physics_loss_version','legacy')
            if (previous_mode!=self.cost.cfg.contact_scoring_mode
                    or previous_physics!=self.cost.cfg.physics_loss_version):
                # A changed hit-region law invalidates legacy contact duals.
                self.committed_al_updates=0
                return self
            self.al.load_state_dict(state["al"])
            self.committed_al_updates = int(state.get("committed_al_updates", 0))
        return self

    def commit(self, violations, *, executed):
        if executed and self.dynamic_al:
            self.al.update(violations)
            self.committed_al_updates += 1
            return True
        return False

    def draw(self, n, rng):
        kwargs = {}
        if self.start_bank is not None:
            q, v = self.start_bank
            ids = torch.as_tensor(rng.integers(0, len(q), n), device=q.device)
            kwargs = dict(q_start=q[ids], qd_start=v[ids])
        return sample_terminal_batch(n, rng=rng, device=self.device, dtype=torch.float32, **kwargs)

    def evaluate(self, model, batch, *, weights=None):
        with torch.autocast(device_type="cuda" if batch.features.is_cuda else "cpu", enabled=False):
            pose = model.terminal_initializer.head(batch.features.float())
            return self.cost(pose, batch, self.weights if weights is None else weights)

    @torch.no_grad()
    def metrics(self, model, batch):
        out = self.evaluate(model, batch)
        hits = out.aux["hit_dist"][batch.active]
        collision = collision_violation(out.aux["min_sdf"], self.cost.cfg.collision_allow_m)
        return dict(terminal_role_eval_loss=float(out.total.mean()),
                    terminal_role_hit_median_mm=float(hits.median()*1000),
                    terminal_role_hit_within_1cm=float((hits < .01).float().mean()),
                    terminal_role_collision_allowed=float((collision <= 0).float().mean()))


def cap_terminal_gradient(model, fraction, *, mode="combined_l2"):
    """Cap either total L2 share or head/rest RMS gradients per parameter."""
    if not 0 < fraction < 1:
        raise ValueError("terminal gradient fraction cap must lie strictly between zero and one")
    if mode not in ("combined_l2", "parameter_rms"):
        raise ValueError("Unknown terminal gradient cap mode")
    head = [p for p in model.terminal_initializer.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    rest_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    rest = [p for p in rest_parameters if p.grad is not None]
    head_count = sum(p.numel() for p in head)
    rest_count = sum(p.numel() for p in rest_parameters)
    live = [p for p in head if p.grad is not None]
    if not live:
        return dict(terminal_head_grad_norm_before=0., terminal_head_grad_norm_after=0.,
                    terminal_head_grad_cap_scale=1., terminal_head_grad_fraction_after=0.)
    norm = lambda params: torch.stack([p.grad.detach().float().square().sum() for p in params]).sum().sqrt()
    before = norm(live)
    other = norm(rest) if rest else before.new_zeros(())
    if mode == "parameter_rms":
        # ||g_head||/sqrt(N_head) <= fraction * ||g_rest||/sqrt(N_rest).
        allowed = other * fraction * np.sqrt(head_count/max(rest_count, 1))
    else:
        allowed = other * (fraction / np.sqrt(1-fraction**2))
    scale = (allowed / before.clamp_min(1e-30)).clamp(max=1.)
    # Leave nonfinite gradients to the train loop's normal skip checks.
    if not torch.isfinite(before) or not torch.isfinite(other):
        scale = before.new_ones(())
    for p in live:
        p.grad.mul_(scale.to(p.grad))
    after = before * scale
    total = torch.sqrt(after.square()+other.square()).clamp_min(1e-30)
    head_rms_before = before / np.sqrt(max(head_count, 1))
    head_rms_after = after / np.sqrt(max(head_count, 1))
    rest_rms = other / np.sqrt(max(rest_count, 1))
    return dict(terminal_head_grad_norm_before=float(before),
                terminal_head_grad_norm_after=float(after),
                terminal_head_grad_cap_scale=float(scale),
                terminal_head_grad_fraction_after=float(after/total),
                terminal_head_parameter_count=head_count,
                terminal_rest_parameter_count=rest_count,
                terminal_head_grad_rms_before=float(head_rms_before),
                terminal_head_grad_rms_after=float(head_rms_after),
                terminal_rest_grad_rms=float(rest_rms),
                terminal_head_grad_rms_ratio_after=float(head_rms_after/rest_rms.clamp_min(1e-30)))
