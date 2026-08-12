"""Unit tests for the PID (integral) extension to the PD simulator.

Covers:
- ki=0 reproduces existing PD behavior exactly
- ki>0 produces non-zero integral contribution
- Anti-windup clamp works
- BatchedPDSim passes ki through correctly
- MPC generator handles integral state
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import CONTROL_DT
from amortized_opt.pd_sim import (
    BatchedPDSim,
    PDSimConfig,
    frozen_minv,
    simulate_pd,
)


# ----------------------------------------------------------------- helpers --
def _make_sim(device="cpu", dtype=torch.float64, **kw):
    return BatchedPDSim(PDSimConfig(**kw), device=device, dtype=dtype)


# ---------------------------------------------------------- ki=0 = pure PD --
def test_ki_none_is_pure_pd():
    """ki=None produces byte-identical results to ki not passed."""
    B, K, N = 2, 5, 3
    torch.manual_seed(42)
    x = torch.randn(B, K + 1, N, dtype=torch.float64)
    x[:, 0] = 0.0  # pinned start at origin
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp = torch.full((B, N), 40.0, dtype=torch.float64)
    kd = torch.full((B, N), 2.0, dtype=torch.float64)
    q0 = torch.zeros(B, N, dtype=torch.float64)
    v0 = torch.zeros(B, N, dtype=torch.float64)

    # No ki argument at all
    qa_pd, va_pd, int_pd = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                         substeps=5)

    # Explicit ki=None
    qa_ki, va_ki, int_ki = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                         substeps=5, ki=None)

    assert torch.allclose(qa_pd, qa_ki)
    assert torch.allclose(va_pd, va_ki)
    assert int_pd is None
    assert int_ki is None  # both return None (no integral accumulator)


def test_ki_zero_tensor_behaves_like_none():
    """A tensor of all-zero ki is treated the same as None (short-circuit)."""
    B, K, N = 1, 5, 3
    x = torch.randn(B, K + 1, N, dtype=torch.float64)
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp = torch.full((B, N), 40.0, dtype=torch.float64)
    kd = torch.full((B, N), 2.0, dtype=torch.float64)
    ki_zero = torch.zeros(B, N, dtype=torch.float64)
    q0 = torch.zeros(B, N, dtype=torch.float64)
    v0 = torch.zeros(B, N, dtype=torch.float64)

    qa_z, va_z, int_z = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                       substeps=5, ki=ki_zero)
    qa_n, va_n, int_n = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                       substeps=5, ki=None)
    assert torch.allclose(qa_z, qa_n)
    assert torch.allclose(va_z, va_n)
    assert int_z is None  # zero tensor → treated as "no ki"
    assert int_n is None


# ---------------------------------------------- ki>0 produces integral growth --
def test_ki_produces_nonzero_integral():
    """With ki>0 and non-zero tracking error, integral should accumulate."""
    B, K, N = 1, 10, 1  # single joint
    ki, kp, kd = 10.0, 40.0, 2.0
    # Command at q=1.0, start at q=0.0 → positive tracking error
    x = torch.ones(B, K + 1, N, dtype=torch.float64)
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp_t = torch.full((B, N), kp, dtype=torch.float64)
    kd_t = torch.full((B, N), kd, dtype=torch.float64)
    ki_t = torch.full((B, N), ki, dtype=torch.float64)
    q0 = torch.zeros(B, N, dtype=torch.float64)
    v0 = torch.zeros(B, N, dtype=torch.float64)

    _, _, integral = simulate_pd(x, Minv, kp_t, kd_t, CONTROL_DT, q0, v0,
                                   substeps=1, ki=ki_t)

    assert integral is not None
    # The integral grows as error * dt accumulates over the rollout.
    # With kp=40, the PD response quickly reduces error, but the integral
    # still accumulates some non-zero value.
    assert abs(integral[0, 0].item()) > 1e-6, \
        f"integral should be non-zero with ki>0, got {integral[0, 0].item():.6f}"


def test_pid_different_from_pd():
    """PID trajectory should differ from PD when same gains and non-zero ki."""
    B, K, N = 1, 20, 1
    ki_val = 10.0
    x = torch.full((B, K + 1, N), 0.5, dtype=torch.float64)
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp = torch.full((B, N), 40.0, dtype=torch.float64)
    kd = torch.full((B, N), 2.0, dtype=torch.float64)
    ki = torch.full((B, N), ki_val, dtype=torch.float64)
    q0 = torch.zeros(B, N, dtype=torch.float64)
    v0 = torch.zeros(B, N, dtype=torch.float64)

    qa_pd, _, _ = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                 substeps=5, ki=None)
    qa_pid, _, _ = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                  substeps=5, ki=ki)

    # The trajectories should differ because the integral term adds torque
    assert not torch.allclose(qa_pd, qa_pid), \
        "PID should produce a different trajectory than PD with ki>0"


# -------------------------------------------------------- anti-windup clamp --
def test_ki_windup_limits_integral():
    """With windup set, the absolute integral should never exceed the clamp."""
    B, K, N = 1, 50, 1
    # A step command — the controller will overshoot, making the integral
    # swing both positive and negative.  We check that |integral| ≤ windup.
    x = torch.full((B, K + 1, N), 1.0, dtype=torch.float64)
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp = torch.full((B, N), 40.0, dtype=torch.float64)
    kd = torch.full((B, N), 2.0, dtype=torch.float64)
    ki = torch.full((B, N), 10.0, dtype=torch.float64)
    q0 = torch.zeros(B, N, dtype=torch.float64)
    v0 = torch.zeros(B, N, dtype=torch.float64)

    windup = 0.1  # rad·s
    _, _, int_clamped = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                      substeps=1, ki=ki, ki_windup=windup)
    assert int_clamped is not None
    assert abs(int_clamped[0, 0].item()) <= windup + 1e-10, \
        f"integral {int_clamped[0, 0].item():.6f} should be clamped to ±{windup}"


def test_no_windup_allows_larger_integral():
    """Without windup, the integral can grow larger than with a tight clamp."""
    B, K, N = 1, 50, 1
    x = torch.full((B, K + 1, N), 1.0, dtype=torch.float64)
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp = torch.full((B, N), 40.0, dtype=torch.float64)
    kd = torch.full((B, N), 2.0, dtype=torch.float64)
    ki = torch.full((B, N), 10.0, dtype=torch.float64)
    q0 = torch.zeros(B, N, dtype=torch.float64)
    v0 = torch.zeros(B, N, dtype=torch.float64)

    _, _, int_free = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                    substeps=1, ki=ki, ki_windup=0.0)
    _, _, int_clamped = simulate_pd(x, Minv, kp, kd, CONTROL_DT, q0, v0,
                                      substeps=1, ki=ki, ki_windup=0.01)

    assert int_free is not None and int_clamped is not None
    assert abs(int_clamped[0, 0].item()) <= 0.01001
    # The free integral should differ from the clamped one
    assert not torch.allclose(int_free, int_clamped), \
        "clamped integral should differ from free when clamp is tight"


# --------------------------------------------------- BatchedPDSim integration --
def test_batched_pd_sim_nominal_gains_returns_ki():
    """When ki>0 in config, nominal_gains returns a non-None ki tensor."""
    sim = _make_sim(ki=5.0)
    kp, kd, ki = sim.nominal_gains(1)
    assert ki is not None
    assert ki.shape == (1, 17)
    assert ki[0, 0].item() == 5.0


def test_batched_pd_sim_ki_zero_returns_none():
    """When ki=0 in config, nominal_gains returns ki=None."""
    sim = _make_sim(ki=0.0)
    kp, kd, ki = sim.nominal_gains(1)
    assert ki is None


def test_batched_pd_sim_step_with_ki():
    """step() with ki>0 should return updated integral."""
    sim = _make_sim(ki=5.0)
    kp, kd, ki = sim.nominal_gains(1)

    prev = torch.zeros(1, 17, dtype=torch.float64)
    cmd = torch.full((1, 17), 0.1, dtype=torch.float64)
    q = torch.zeros(1, 17, dtype=torch.float64)
    qd = torch.zeros(1, 17, dtype=torch.float64)
    integral = torch.zeros(1, 17, dtype=torch.float64)

    q_next, v_next, int_next = sim.step(prev, cmd, q, qd, kp=kp, kd=kd, ki=ki,
                                          integral=integral)
    assert q_next.shape == (1, 17)
    assert v_next.shape == (1, 17)
    assert int_next is not None
    # Integral should have grown (positive error → positive accumulation)
    assert int_next.abs().sum() > 0


def test_batched_pd_sim_step_with_ki_none():
    """step() with ki=None returns int_next=None (backward compat)."""
    sim = _make_sim(ki=0.0)
    kp, kd, ki = sim.nominal_gains(1)
    assert ki is None

    prev = torch.zeros(1, 17, dtype=torch.float64)
    cmd = torch.full((1, 17), 0.1, dtype=torch.float64)
    q = torch.zeros(1, 17, dtype=torch.float64)
    qd = torch.zeros(1, 17, dtype=torch.float64)

    q_next, v_next, int_next = sim.step(prev, cmd, q, qd, kp=kp, kd=kd, ki=ki)
    assert q_next.shape == (1, 17)
    assert v_next.shape == (1, 17)
    assert int_next is None


# -------------------------------------------------------- integrator basics --
def test_integral_starts_from_zero():
    """integral0=None + ki>0 → integral initialized to zeros."""
    B, K, N = 1, 5, 3
    ki = torch.full((1, 3), 5.0, dtype=torch.float64)
    x = torch.randn(B, K + 1, N, dtype=torch.float64)
    Minv = torch.eye(N).unsqueeze(0).expand(B, K, N, N).to(torch.float64)
    kp = torch.full((1, 3), 40.0, dtype=torch.float64)
    kd = torch.full((1, 3), 2.0, dtype=torch.float64)

    _, _, integral = simulate_pd(x, Minv, kp, kd, CONTROL_DT,
                                   torch.zeros(B, N, dtype=torch.float64),
                                   torch.zeros(B, N, dtype=torch.float64),
                                   substeps=1, ki=ki, integral0=None)
    assert integral is not None
    assert integral.shape == (B, N)


# --------------------------------------------------- MPC generator regression --
def test_mpc_generator_runs_with_ki():
    """An episode with ki>0 and lead=off must complete without divergence."""
    from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator
    from amortized_opt.rhythm import DrumKit, RhythmClip, Hit
    from amortized_opt.warmstart import GoalPoseLibrary

    kit = DrumKit.from_layout()
    lib = GoalPoseLibrary()
    # Single-hit clip: left hand at 0.2s
    clip = RhythmClip(
        name="test_pid_mpc",
        hits=[Hit(time=0.2, raw_time=0.2, drum=0, drum_name="snare",
                   hand="left", velocity=2.0)],
        duration=1.0,
    )

    cfg = MPCGenConfig(
        horizon=50, ki=5.0, ki_windup=0.0,
        cmd_lead_s=0.0,   # lead OFF so PID is active
        randomize_gains=False,
    )
    gen = MPCGenerator(kit=kit, library=lib, config=cfg)

    q0 = np.zeros(17)
    arm = gen._arm["left"]
    q0[arm] = lib.select("left", "snare", 0.0).arm_q

    res = gen.run_episode(clip, q0, np.zeros(17), seed=42, max_steps=25)
    # The episode should not diverge
    assert not res.diverged, f"episode diverged at step {res.diverged_step}"
    # The left hand should have fired
    left_hits = [hs for hs in res.hit_steps if hs[1] == "left"]
    assert len(left_hits) > 0, f"expected left hand to fire, got {res.hit_steps}"


def test_mpc_generator_lead_disables_integral():
    """When cmd_lead_s is non-zero, the episode runs OK (integral guard)."""
    from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator
    from amortized_opt.rhythm import DrumKit, RhythmClip, Hit
    from amortized_opt.warmstart import GoalPoseLibrary
    from amortized_opt.lead import LEAD_PRESETS

    kit = DrumKit.from_layout()
    lib = GoalPoseLibrary()
    clip = RhythmClip(
        name="test_lead_pid",
        hits=[Hit(time=0.2, raw_time=0.2, drum=0, drum_name="snare",
                   hand="left", velocity=2.0)],
        duration=1.0,
    )
    cfg = MPCGenConfig(
        horizon=50,
        ki=10.0,
        cmd_lead_s=LEAD_PRESETS["fit"],  # lead ON → should disable PID
        randomize_gains=False,
    )
    gen = MPCGenerator(kit=kit, library=lib, config=cfg)
    q0 = np.zeros(17)
    q0[gen._arm["left"]] = lib.select("left", "snare", 0.0).arm_q
    res = gen.run_episode(clip, q0, np.zeros(17), seed=42, max_steps=25)
    assert not res.diverged, \
        f"episode diverged at step {res.diverged_step} — integral likely fought the lead"
