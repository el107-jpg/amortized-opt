"""``DrumMPCLoop`` against the ANALYTIC plant -- the Isaac-free regression.

Why this file exists: ``loop.tick`` used to unpack the plant's return with
``q, qd = self.plant(...)``.  ``amortized_opt.pd_sim.BatchedPDSim.step`` grew a
third return value (the PD integral state) in a concurrent fine-tune, so any
caller that handed ``sim.step`` straight to the loop -- which the loop's own
docstring invites, since ``plant``'s signature is documented as *identical* to
``BatchedPDSim.step`` -- died with a bare ``too many values to unpack`` one tick
in.  The loop now takes ``out[0], out[1]`` and ignores the tail.

The rest of the file is a smoke of the assembled tick order (plan -> plant ->
advance -> reward -> commit) on CPU with a tiny model, which nothing else
covered: the schedule, reward and perturbation suites each test their own piece.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, mass_matrix  # noqa: E402
from isaac_rl.loop import DrumMPCLoop  # noqa: E402
from isaac_rl.mpc import IsaacMPC, MPCConfig  # noqa: E402
from isaac_rl.reward import ExecutedReward, ExecutedStrikeScorer, RewardConfig  # noqa: E402
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig  # noqa: E402

H = 50
B = 2


@pytest.fixture(scope="module")
def stack():
    """A tiny (untrained) model + schedule + reward, on CPU."""
    torch.manual_seed(0)
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    mcfg = A.ModelConfig(d_model=32, dec_heads=2, dec_layers=1,
                         d_ctx=32, ctx_heads=2, ctx_layers=1, max_horizon=H)
    model = A.build_model(mcfg, "cpu").eval()
    scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future)
    sched = BatchedSchedule(B, scfg, device="cpu")
    sched.reset(torch.arange(B), [clips[0]] * B, [kit] * B,
                rng=np.random.default_rng(0))
    mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, amp_dtype=None),
                   device="cpu", kit_names=tuple(kit.names))
    fk = A.ChainFK(device="cpu")
    scorer = ExecutedStrikeScorer(fk=fk, device="cpu")
    rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=None, device="cpu")
    return model, kit, sched, mpc, rew


def _pd_plant(device="cpu", dtype=torch.float32):
    """``BatchedPDSim.step`` UNWRAPPED -- the exact 3-tuple callable."""
    rnea = A.BatchedRNEA.from_urdf(device=device).to(device, dtype)
    sim = BatchedPDSim(PDSimConfig(dt=A.CONTROL_DT), device=device, dtype=dtype,
                       rnea=rnea)

    def plant(prev_cmd, cmd, q, qd):
        Minv = torch.linalg.inv(mass_matrix(rnea, q))
        return sim.step(prev_cmd, cmd, q, qd, Minv=Minv)

    return plant, sim


def test_pd_sim_step_returns_three(stack):
    """Pin the shape the loop must tolerate, so a future change is visible here."""
    plant, _ = _pd_plant()
    q = torch.zeros(B, A.N_DOF)
    out = plant(q, q, q, torch.zeros_like(q))
    assert len(out) == 3, "BatchedPDSim.step is expected to return (q, qd, integral)"


def test_loop_runs_against_analytic_plant(stack):
    """The reported bug: an un-adapted ``sim.step`` must drive the loop."""
    _model, _kit, _sched, mpc, rew = stack
    plant, _sim = _pd_plant()
    q = torch.zeros(B, A.N_DOF)
    qd = torch.zeros_like(q)
    mpc.reset(torch.arange(B), q)
    loop = DrumMPCLoop(schedule=_sched, mpc=mpc, reward=rew, plant=plant,
                       record=True)
    q, qd, log = loop.rollout(q, qd, 5)
    assert torch.isfinite(q).all() and torch.isfinite(qd).all()
    assert loop.stats.ticks == 5
    assert "reward/total" in log
    assert loop.q_hist().shape == (5, B, A.N_DOF)


def test_loop_tolerates_two_tuple_plant(stack):
    """...and the Isaac plant's 2-tuple keeps working (env.plant's shape)."""
    _model, _kit, _sched, mpc, rew = stack
    plant3, _sim = _pd_plant()

    def plant2(prev_cmd, cmd, q, qd):
        a, b, _c = plant3(prev_cmd, cmd, q, qd)
        return a, b

    q = torch.zeros(B, A.N_DOF)
    qd = torch.zeros_like(q)
    mpc.reset(torch.arange(B), q)
    loop = DrumMPCLoop(schedule=_sched, mpc=mpc, reward=rew, plant=plant2)
    q, qd, _log = loop.rollout(q, qd, 3)
    assert torch.isfinite(q).all()
