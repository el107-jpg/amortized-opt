"""Isaac-dependent checks.  ``pytest isaac_rl/tests -m isaac --isaac-device cuda:1``

These are the assertions that can only be made against the real plant:

* the rate really is 250 Hz / decimation 5 -> a true 50 Hz control tick (the
  brief says: if Isaac cannot run at 250 Hz stably, STOP and report);
* the articulation's joint order really is ``G1_UPPER_BODY_JOINTS`` -- a silent
  permutation here would scramble the per-joint lead vector and every limit
  table, and nothing downstream would raise;
* the plant is a contraction toward a held command, i.e. PhysX is not blowing
  up at the finer substep;
* the contact sensors' filter index really is the drum id.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.isaac


def test_rates_are_250hz_decimation_5(isaac_env):
    from isaac_rl.env_cfg import CONTROL_DT, DECIMATION, SIM_DT

    assert isaac_env.cfg.sim.dt == pytest.approx(SIM_DT)
    assert isaac_env.cfg.decimation == DECIMATION
    assert isaac_env.step_dt == pytest.approx(CONTROL_DT)
    assert isaac_env.step_dt == pytest.approx(0.02)


def test_joint_order_is_the_optimizer_order(isaac_env):
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS

    _ids, names = isaac_env.robot.find_joints(list(G1_UPPER_BODY_JOINTS),
                                              preserve_order=True)
    assert tuple(names) == G1_UPPER_BODY_JOINTS


def test_physx_is_stable_at_250hz(isaac_env):
    """Hold the command; the plant must settle, not diverge.

    This is the "if Isaac cannot run at 250 Hz stably, STOP" gate.  A finer
    substep with the same 8/4 solver iterations is *more* accurate, not less,
    so a failure here means something else (contact, armature, the pelvis pin)
    reacted to the rate change.
    """
    q, qd = isaac_env.read_state()
    hold = q.clone()
    err0 = None
    for i in range(60):
        q, qd = isaac_env.plant(hold, hold, q, qd)
        assert torch.isfinite(q).all() and torch.isfinite(qd).all(), f"NaN at tick {i}"
        e = (q - hold).abs().max()
        if err0 is None:
            err0 = float(e)
    assert float((q - hold).abs().max()) < 0.5, "the plant walked away from a held command"
    assert float(qd.abs().max()) < 20.0


def test_plant_tracks_a_step_command(isaac_env):
    """A small step must move the plant TOWARD it, monotonically enough."""
    q, qd = isaac_env.read_state()
    tgt = q + 0.05
    prev = q.clone()
    e_start = float((q - tgt).abs().mean())
    for _ in range(40):
        q, qd = isaac_env.plant(prev, tgt, q, qd)
        prev = tgt
    e_end = float((q - tgt).abs().mean())
    assert e_end < e_start, f"tracking error grew: {e_start:.4f} -> {e_end:.4f}"


def test_contact_sensor_filter_index_is_the_drum_id(isaac_env):
    f = isaac_env.contact_forces()
    if f is None:
        pytest.skip("contact sensors disabled")
    from amortized_opt.rhythm import DrumKit

    kit = DrumKit.from_layout()
    assert f.shape[0] == isaac_env.num_envs
    assert f.shape[1] == 2                     # hand axis, 0 = left
    assert f.shape[2] >= kit.num_drums
    # the env's own ordering must be the layout's id ordering
    assert tuple(isaac_env.contact_drum_names[: kit.num_drums]) == tuple(kit.names)


def test_reset_to_teleports_exactly(isaac_env):
    torch.manual_seed(0)
    q, _qd = isaac_env.read_state()
    tgt = q + 0.1 * torch.randn_like(q)
    ids = torch.arange(isaac_env.num_envs, device=q.device)
    isaac_env.reset_to(ids, tgt)
    q2, qd2 = isaac_env.read_state()
    assert torch.allclose(q2, tgt, atol=1e-5)
    assert float(qd2.abs().max()) < 1e-4


def test_full_loop_runs_and_scores(isaac_env, request):
    """One short closed-loop rollout end to end, with the champion checkpoint."""
    from pathlib import Path

    import amortized_opt as A
    from isaac_rl.env_cfg import CONTROL_DT
    from isaac_rl.loop import DrumMPCLoop
    from isaac_rl.measure import latency_report
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    from isaac_rl.reward import ExecutedReward, ExecutedStrikeScorer, RewardConfig
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    ckpt = Path(__file__).resolve().parents[2] / "runs/phase2b_main/best.pt"
    if not ckpt.is_file():
        pytest.skip(f"champion checkpoint missing at {ckpt}")
    dev = torch.device(request.config.getoption("--isaac-device"))
    B = isaac_env.num_envs
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    H = int((ck.get("scenario_cfg") or {}).get("horizon", 50))

    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:2]
    sched = BatchedSchedule(B, ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                                              num_future=mcfg.num_future),
                            device=dev)
    sched.reset(torch.arange(B, device=dev), [clips[i % len(clips)] for i in range(B)],
                [kit] * B, rng=np.random.default_rng(0))
    mpc = IsaacMPC(model, sched, MPCConfig(horizon=H, cmd_lead_s="fit"),
                   device=dev, kit_names=tuple(kit.names))
    fk = A.ChainFK(device=dev, dtype=torch.float32)
    scorer = ExecutedStrikeScorer(fk=fk, dt=CONTROL_DT, device=dev)
    rew = ExecutedReward(RewardConfig(), scorer=scorer, cost=None, device=dev)

    q, qd = isaac_env.read_state()
    mpc.reset(torch.arange(B, device=dev), q)
    loop = DrumMPCLoop(schedule=sched, mpc=mpc, reward=rew, plant=isaac_env.plant,
                       record=True)
    q, qd, log = loop.rollout(q, qd, 60)
    assert torch.isfinite(q).all()
    assert "terms/hit" in log and "terms/track" in log
    rep = latency_report(loop, kit, device="cpu")
    assert rep["n_strikes"] > 0, "no scheduled strike came due in 60 ticks"
    assert 0.0 <= rep["overall"]["miss_rate"] <= 1.0
