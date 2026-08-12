"""Shared fixtures.

Everything here runs WITHOUT Isaac.  The sibling's ``schedule.py`` and
``mpc.py`` are Isaac-free by construction (they see only ``(q, qd)`` tensors),
so these tests exercise the REAL warm-start state machine and the REAL context
builder rather than a mock of them -- a mock of a state machine tests the mock.
Only ``env.py`` needs a simulator, and nothing here touches it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A

from isaac_rl_ppo.actor import AmortizedPpoActor
from isaac_rl_ppo.cfg import ActorConfig, CriticConfig, PpoRunConfig

#: A deliberately tiny transformer.  The machinery under test (LoRA wrapping,
#: the Gaussian head, AR(1), the storage replay) is width-independent, and a
#: 19 M-parameter backbone would make the suite unusable on CPU.
TINY = dict(max_horizon=50, d_model=64, dec_layers=2, dec_heads=4,
            d_ctx=64, ctx_layers=1, ctx_heads=4, num_drums=8, num_future=3)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def model():
    torch.manual_seed(0)
    return A.AmortizedOptimizer(A.ModelConfig(**TINY))


@pytest.fixture
def run_cfg() -> PpoRunConfig:
    """Config for the unit suite, which drives ``amortized_opt.pd_sim``.

    ``plant_regime="analytic"`` is the load-bearing part: these tests emulate
    the ANALYTIC plant, which is the one the ``"fit"`` per-joint lead was
    identified against, so the lead is correct here.  The Isaac path defaults to
    the lead OFF (it over-compensates against PhysX -- 158/158 strikes early)
    until RL-1b publishes an Isaac-fitted vector.  Keeping the two regimes in
    separate configs is what stops one plant's lead leaking into the other.
    """
    cfg = PpoRunConfig()
    cfg.env.plant_regime = "analytic"
    cfg.env.cmd_lead_s = "fit"
    cfg.actor.lora_rank = 4
    cfg.actor.horizon = 50
    cfg.ppo.num_steps_per_env = 8
    cfg.ppo.num_mini_batches = 2
    cfg.ppo.micro_batch_size = 8
    cfg.env.num_envs = 4
    cfg.validate()
    return cfg


@pytest.fixture
def actor(model, run_cfg) -> AmortizedPpoActor:
    torch.manual_seed(0)
    return AmortizedPpoActor(model, run_cfg.actor, run_cfg.critic, device="cpu")


@pytest.fixture(scope="session")
def kit_and_clips():
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    return kit, clips


@pytest.fixture(scope="session")
def surface():
    """The isaac_rl surface, resolved exactly as the training run resolves it.

    Going through :mod:`bridge` rather than importing the snapshot directly is
    the point: it means this suite is a conformance test of whichever source is
    configured, so ``AMORTIZED_PPO_ISAAC_RL_SOURCE=pinned pytest ...`` and the
    default sibling run both prove something useful.
    """
    from isaac_rl_ppo import bridge

    return bridge.load_isaac_rl()


@pytest.fixture
def scheduled(kit_and_clips, surface):
    """A real :class:`BatchedSchedule` with real RCC clips, 4 envs."""
    kit, clips = kit_and_clips
    B = 4
    sched = surface.BatchedSchedule(B, surface.ScheduleConfig(), device="cpu")
    sched.reset(torch.arange(B), [clips[i % len(clips)] for i in range(B)],
                [kit] * B, rng=np.random.default_rng(0))
    return sched, kit, B


@pytest.fixture
def mpc(actor, scheduled, surface):
    """A real :class:`IsaacMPC` driving the tiny model."""
    sched, _kit, _B = scheduled
    return surface.IsaacMPC(
        actor.model, sched,
        surface.MPCConfig(horizon=50, amp_dtype=None, iterations=1),
        device="cpu", kit_names=sched.kit_names)


def make_inputs(actor, sched, B: int, seed: int = 0):
    """A batch of :class:`MpcInputs` from the real schedule."""
    from isaac_rl_ppo.actor import MpcInputs

    g = torch.Generator().manual_seed(seed)
    q = 0.1 * torch.randn(B, 17, generator=g)
    qd = 0.1 * torch.randn(B, 17, generator=g)
    init = q.unsqueeze(1).expand(-1, actor.H, -1).contiguous()
    return MpcInputs(init=init, state=torch.cat([q, qd], dim=-1),
                     ctx=sched.context(), hit_index=sched.hit_index())
