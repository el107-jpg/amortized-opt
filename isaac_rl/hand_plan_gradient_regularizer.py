"""Input-gradient L1 on relative legal-plan scores in physical coordinates."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch

from amortized_opt.constants import ready_joint_pos
from amortized_opt.fk import ChainFK, stick_tip_offset


@dataclass(frozen=True)
class InputGradientConfig:
    timing_scale_s: float = .02
    position_scale_rad: float = math.pi / 180
    velocity_scale_rad_s: float = .1
    loudness_scale: float = .02

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Positive finite physical derivative scales required')


def tip_jacobians(observations, geometry=None, *, device='cpu', chunk=512):
    """Compute fixed [case,6,17] FK Jacobians; labels and weights are unused."""
    device = torch.device(device)
    fk = ChainFK(frames={f'{h}_wrist': f'{h}_wrist_yaw_joint' for h in ('left', 'right')}, device=device)
    offsets = torch.as_tensor(np.stack([stick_tip_offset(h) for h in ('left', 'right')]) if geometry is None
        else geometry['tip_offsets_local_m'], device=device, dtype=torch.float32)
    ready = torch.as_tensor(ready_joint_pos(), device=device, dtype=torch.float32)
    blocks = []
    with torch.enable_grad():
        for start in range(0, len(observations), chunk):
            obs = torch.as_tensor(observations[start:start+chunk], device=device)
            q = (obs[:,80:97] * math.pi + ready).detach().requires_grad_(True)
            tips = []
            for i, hand in enumerate(('left', 'right')):
                rotation, position = fk.frame(q, f'{hand}_wrist')
                tips.append(position + torch.einsum('bij,j->bi', rotation, offsets[i]))
            values = torch.stack(tips, 1).flatten(1)
            jac = [torch.autograd.grad(values[:,i].sum(), q, retain_graph=i < 5)[0] for i in range(6)]
            blocks.append(torch.stack(jac, 1).detach().cpu())
    return torch.cat(blocks)


def physical_score_gradient(observation_gradient, observations, tip_jacobian, config=InputGradientConfig()):
    """Chain rule through interval/deadline and joint/tip redundant features."""
    groups = observation_gradient[:,:80].reshape(-1,4,20)
    # Altering interval j moves deadlines j..3; deadlines are encoded /4.
    timing = groups[:,:,19] + groups[:,:,18].flip(1).cumsum(1).flip(1) / 4
    present = observations[:,:80].reshape(-1,4,20)[:,:,[8,17]].flatten(1)
    loudness = groups[:,:,[7,16]].flatten(1) * present
    position = observation_gradient[:,80:97] / math.pi
    position = position + torch.einsum('bti,bt->bi', tip_jacobian, observation_gradient[:,130:136])
    velocity = observation_gradient[:,97:114] / 10
    return torch.cat((timing * config.timing_scale_s, loudness * config.loudness_scale,
        position * config.position_scale_rad, velocity * config.velocity_scale_rad_s), 1)


def input_gradient_l1(model, observations, legal, tip_jacobian, *, generator,
                      config=InputGradientConfig(), create_graph=True):
    """Sample one legal score row; count weighting estimates the full Jacobian L1.

    Independent samples average to sum_a ||d(s_a-mean_legal(s))/dz||_1.
    z consists of four intervals, eight present loudness slots,17 q and17 qd,
    scaled in physical units. Drum IDs, presence and previous targets stay fixed.
    Evaluation mode removes dropout while keeping autograd enabled.
    """
    legal = legal.bool()
    if not legal.any(1).all():
        raise ValueError('At least one legal plan required per case')
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.enable_grad():
            obs = observations.detach().requires_grad_(True)
            scores = model(obs)
            count = legal.sum(1)
            centered = scores - (scores * legal).sum(1,keepdim=True) / count[:,None]
            chosen = torch.multinomial(legal.to(scores.dtype), 1, generator=generator)
            selected = centered.gather(1, chosen).sum()
            gradient = torch.autograd.grad(selected, obs, create_graph=create_graph)[0]
            physical = physical_score_gradient(gradient, obs, tip_jacobian, config)
            magnitude = physical.abs()
            row_norm = magnitude.sum(1)
            penalty = (row_norm * count).mean()
            present=observations[:,:80].reshape(-1,4,20)[:,:,[8,17]].flatten(1).bool()
            active=torch.cat((torch.ones_like(present[:,:4]),present,
                torch.ones(len(obs),34,dtype=torch.bool,device=obs.device)),1) & (count>1)[:,None]
            stats = dict(gradient_l1=penalty.detach(), sampled_plan_gradient_l1=row_norm.mean().detach(),
                timing_gradient_l1=magnitude[:,:4].sum(1).mean().detach(),
                loudness_gradient_l1=magnitude[:,4:12].sum(1).mean().detach(),
                position_gradient_l1=magnitude[:,12:29].sum(1).mean().detach(),
                velocity_gradient_l1=magnitude[:,29:].sum(1).mean().detach(),
                fraction_scaled_derivatives_below_1e_3=((magnitude < 1e-3)&active).sum().float().div(active.sum().clamp_min(1)).detach(),
                gradient_probe_choice_cases=(count>1).sum().detach())
            return penalty, stats
    finally:
        for module, was_training in flags:
            module.training = was_training
