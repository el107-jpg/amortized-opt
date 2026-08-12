"""Strike-window impedance schedule: masks, ramps, and plant equivalence.

The schedule exists to keep impact torque out of the shoulder and waist
(section 9b follow-up), so the load-bearing assertions are the NEGATIVE
ones: the waist and shoulder columns must never move, and outside the
window the plant must be bit-identical to nominal.
"""

from __future__ import annotations

import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS
from amortized_opt.impedance import (
    StrikeImpedance,
    StrikeImpedanceConfig,
    distal_joint_masks,
)
from amortized_opt.pd_sim import BatchedPDSim

N = len(G1_UPPER_BODY_JOINTS)
LEFT_DISTAL = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS)
               if n.startswith("left_") and ("elbow" in n or "wrist" in n)]
RIGHT_DISTAL = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS)
                if n.startswith("right_") and ("elbow" in n or "wrist" in n)]
POSTURE = [i for i in range(N) if i not in LEFT_DISTAL + RIGHT_DISTAL]


def _idle(b: int = 1) -> torch.Tensor:
    return torch.full((b, 2), -1.0)


def test_masks_are_the_distal_arm_and_nothing_else():
    m = distal_joint_masks()
    assert m.shape == (2, N)
    assert m[0].nonzero().flatten().tolist() == LEFT_DISTAL
    assert m[1].nonzero().flatten().tolist() == RIGHT_DISTAL
    assert len(LEFT_DISTAL) == len(RIGHT_DISTAL) == 4  # elbow + wrist r/p/y
    # posture columns (waist + shoulders) untouched by construction
    assert m[:, POSTURE].abs().sum() == 0.0


def test_idle_hands_leave_the_gains_alone():
    imp = StrikeImpedance(StrikeImpedanceConfig(), batch=3)
    kp_mul, kd_mul = imp.scales(_idle(3))
    assert torch.equal(kp_mul, torch.ones(3, N))
    assert torch.equal(kd_mul, torch.ones(3, N))


def test_well_bottom_is_exactly_the_configured_scale_on_that_hand_only():
    cfg = StrikeImpedanceConfig(kp_scale=0.5, kd_scale=0.7, before_wp=3, after_wp=3)
    imp = StrikeImpedance(cfg, batch=1)
    kp_mul, kd_mul = imp.scales(torch.tensor([[0.0, -1.0]]))  # left hits this tick
    assert torch.allclose(kp_mul[0, LEFT_DISTAL], torch.full((4,), 0.5))
    assert torch.allclose(kd_mul[0, LEFT_DISTAL], torch.full((4,), 0.7))
    assert torch.equal(kp_mul[0, RIGHT_DISTAL], torch.ones(4))
    assert torch.equal(kp_mul[0, POSTURE], torch.ones(len(POSTURE)))
    assert torch.equal(kd_mul[0, POSTURE], torch.ones(len(POSTURE)))


def test_approach_ramps_monotonically_into_the_hit():
    imp = StrikeImpedance(StrikeImpedanceConfig(before_wp=3), batch=1)
    j = LEFT_DISTAL[0]
    prev = 1.0
    for tth in (4.0, 3.0, 2.0, 1.0, 0.0):
        kp_mul, _ = imp.scales(torch.tensor([[tth, -1.0]]))
        cur = float(kp_mul[0, j])
        assert cur <= prev + 1e-6
        prev = cur
    assert prev == 0.5  # bottom at the hit tick
    kp_far, _ = imp.scales(torch.tensor([[4.0, -1.0]]))
    assert float(kp_far[0, j]) == 1.0  # outside the window: exactly nominal


def test_recovery_decays_back_to_nominal_after_the_hit():
    cfg = StrikeImpedanceConfig(kp_scale=0.5, before_wp=0, after_wp=3)
    imp = StrikeImpedance(cfg, batch=1)
    j = RIGHT_DISTAL[0]
    imp.observe_fired(torch.tensor([[False, True]]))  # right hand just struck
    seen = []
    for _ in range(5):
        kp_mul, _ = imp.scales(_idle())
        seen.append(float(kp_mul[0, j]))
        imp.observe_fired(torch.tensor([[False, False]]))
    # full depth right after the hit, monotone decay, exactly 1 once past the window
    assert seen[0] == 0.5
    assert all(a <= b + 1e-6 for a, b in zip(seen, seen[1:]))
    assert seen[-1] == 1.0
    # the LEFT arm never moved
    kp_mul, _ = imp.scales(_idle())
    assert torch.equal(kp_mul[0, LEFT_DISTAL], torch.ones(4))


def test_reset_clears_the_recovery_state():
    imp = StrikeImpedance(StrikeImpedanceConfig(), batch=2)
    imp.observe_fired(torch.tensor([[True, False], [False, True]]))
    imp.reset(torch.tensor([0]))
    kp_mul, _ = imp.scales(_idle(2))
    assert torch.equal(kp_mul[0], torch.ones(N))          # reset env is nominal
    assert float(kp_mul[1, RIGHT_DISTAL[0]]) < 1.0        # other env still soft


def test_pd_sim_step_gain_scale_equals_prescaled_gains():
    """The kwarg must be EXACTLY multiplicative on the resolved gains."""
    sim = BatchedPDSim(device="cpu")
    kp, kd, _ = sim.nominal_gains(2)
    g = torch.Generator().manual_seed(0)
    q0 = torch.randn(2, N, generator=g) * 0.1
    v0 = torch.randn(2, N, generator=g) * 0.1
    prev = q0.clone()
    cmd = q0 + torch.randn(2, N, generator=g) * 0.05
    mul_p = torch.rand(2, N, generator=g) * 0.5 + 0.5
    mul_d = torch.rand(2, N, generator=g) * 0.5 + 0.5
    qa1, va1, _ = sim.step(prev, cmd, q0, v0, gain_scale=(mul_p, mul_d))
    qa2, va2, _ = sim.step(prev, cmd, q0, v0, kp=kp * mul_p, kd=kd * mul_d)
    assert torch.equal(qa1, qa2)
    assert torch.equal(va1, va2)
    # and no gain_scale means the untouched nominal path
    qa3, _, _ = sim.step(prev, cmd, q0, v0)
    qa4, _, _ = sim.step(prev, cmd, q0, v0, kp=kp, kd=kd)
    assert torch.equal(qa3, qa4)
