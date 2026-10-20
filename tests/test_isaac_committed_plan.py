from types import SimpleNamespace

import numpy as np
import torch

import amortized_opt as A
from isaac_rl.mpc import IsaacMPC, MPCConfig
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig


class _Decoder(torch.nn.Module):
    terminal_initializer = None

    def __init__(self):
        super().__init__()
        self.inputs = []

    def refine(self, init, state, enc, masks, iterations):
        self.inputs.append((init.clone(), state.clone()))
        return SimpleNamespace(traj=(init + 0.01) * masks.elem_mask)


class _Gate:
    def __init__(self, pass_at):
        self.pass_at = pass_at
        self.inputs = []

    def __call__(self, q, traj, qd, strike, **kwargs):
        self.inputs.append((q.clone(), qd.clone(), traj.clone()))
        value = 0.0 if len(self.inputs) >= self.pass_at else 0.1
        return SimpleNamespace(total=torch.tensor([value]), diagnostics={},
                               violations={"terminal": torch.tensor([value])})


def _stack(pass_at, multiple_hits=False):
    kit = A.DrumKit.from_layout()
    clip = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[0]
    h = clip.hits[0]
    from dataclasses import replace
    clip = replace(clip, hits=[replace(h, time=0.1, raw_time=0.1)])
    if multiple_hits:
        clip.hits.append(replace(h, time=0.2, raw_time=0.2))
    schedule = BatchedSchedule(1, ScheduleConfig(horizon=5), device="cpu")
    schedule.reset(torch.tensor([0]), [clip], [kit], rng=np.random.default_rng(0))
    decoder, gate = _Decoder(), _Gate(pass_at)
    mpc = IsaacMPC(decoder, schedule, MPCConfig(
        horizon=5, commit_plan=True, plan_max_passes=3, cmd_lead_s=None,
        cmd_slew_eta=None, amp_dtype=None), device="cpu", descent_cost=gate)
    q = torch.zeros(1, A.N_DOF)
    mpc.reset(torch.tensor([0]), q)
    mpc.init.zero_()
    mpc._arm_dls_pending.zero_()
    mpc._encoding = lambda hit, q: (None, False)
    return mpc, decoder, gate, q


def test_refines_prior_output_then_tracks_without_network_or_controller_feedback():
    mpc, decoder, gate, q = _stack(pass_at=3)
    result = mpc.plan(q, q)
    assert mpc.plan_feasible and len(decoder.inputs) == 3
    assert all(torch.equal(x[1], decoder.inputs[0][1]) for x in decoder.inputs)
    mask = result.elem_mask
    assert torch.equal(decoder.inputs[1][0], gate.inputs[0][2] * mask)
    assert torch.equal(decoder.inputs[2][0], gate.inputs[1][2] * mask)
    cols = mpc.arm_cols[0]
    assert torch.allclose(decoder.inputs[2][0][:, :, cols], torch.full_like(q[:, None, cols].expand(-1, 5, -1), .02))
    assert all(torch.equal(x[0], q) and torch.equal(x[1], q) for x in gate.inputs)
    reference = mpc._committed_reference.clone()
    mpc.sched.advance()
    # A controller offset and measurement error must not rewrite the plan.
    mpc.commit(result.cmd + .2, q + .1, torch.zeros(1, 2, dtype=torch.bool), qd=q + 1)
    next_tick = mpc.plan(q + .1, q + 1)
    assert len(decoder.inputs) == 3 and len(gate.inputs) == 3
    assert torch.equal(mpc.last_ref0, reference[:, 1])
    assert torch.equal(next_tick.cmd, reference[:, 1])


def test_no_feasible_candidate_holds_initial_pose_without_repeated_attempts():
    mpc, decoder, gate, q = _stack(pass_at=100)
    result = mpc.plan(q, q)
    assert not mpc.plan_feasible and len(decoder.inputs) == 3
    assert torch.equal(result.cmd, q)
    mpc.sched.advance()
    mpc.commit(result.cmd, q, torch.zeros(1, 2, dtype=torch.bool), qd=q)
    assert torch.equal(mpc.plan(q, q).cmd, q)
    assert len(decoder.inputs) == 3


def test_new_hit_starts_new_plan_and_network_stays_idle_between_hits():
    mpc, decoder, gate, q = _stack(pass_at=1, multiple_hits=True)
    for tick in range(10):
        result = mpc.plan(q, torch.zeros_like(q))
        assert len(decoder.inputs) == (1 if tick < 5 else 2)
        q = result.cmd.clone()
        fired = mpc.sched.advance()
        mpc.commit(result.cmd, q, fired, qd=torch.zeros_like(q))
    assert len(mpc.plan_segments) == 2
    assert [s["tick"] for s in mpc.plan_segments] == [0, 5]
    # Finished rhythm: hold without synthesizing an idle plan.
    mpc.plan(q, torch.zeros_like(q))
    assert len(decoder.inputs) == 2


def test_isaac_acceptance_executes_proposal_even_when_analytic_gate_fails():
    mpc, decoder, gate, q = _stack(pass_at=100)
    mpc.cfg.plan_acceptance = "isaac"
    result = mpc.plan(q, q)
    assert not mpc.plan_feasible
    assert len(decoder.inputs) == 1
    assert mpc.plan_segments[0]["executed"]
    assert mpc.plan_segments[0]["acceptance"] == "isaac"
    assert torch.equal(result.cmd, gate.inputs[0][2][:, 0])
    assert not torch.equal(result.cmd, q)
    mpc.sched.advance()
    mpc.commit(result.cmd, q, torch.zeros(1, 2, dtype=torch.bool), qd=q)
    mpc.plan(q, q)
    assert len(decoder.inputs) == 1
