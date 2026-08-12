"""``amortized_opt.lead`` -- the reference-lead interpolation.

The invariants that matter are small but each one is a bug that would be silent
in a two-hour closed-loop eval: the lead must be an exact interpolation (so a
whole-tick lead is a pure shift), it must HOLD past the end of the plan (the
region past a hand's terminal is unpriced), it must be per-joint, and it must be
a bit-exact no-op at zero.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.lead import (
    LEAD_PRESETS,
    is_schedule,
    lead_fraction,
    lead_waypoints,
    lead_waypoints_np,
    resolve_lead,
    schedule_fraction,
)

N = len(G1_UPPER_BODY_JOINTS)


def test_zero_lead_is_a_bit_exact_no_op():
    traj = torch.randn(3, 12, N, dtype=torch.float64)
    assert lead_fraction(0.0, CONTROL_DT, N) is None
    assert lead_fraction(None, CONTROL_DT, N) is None
    assert lead_fraction([0.0] * N, CONTROL_DT, N) is None
    assert torch.equal(lead_waypoints(traj, None), traj)


def test_whole_tick_lead_is_an_exact_shift():
    traj = torch.arange(10, dtype=torch.float64).view(1, 10, 1).repeat(1, 1, N)
    f = lead_fraction(CONTROL_DT, CONTROL_DT, N)
    out = lead_waypoints(traj, torch.as_tensor(f))
    assert np.allclose(out[0, :9, 0].numpy(), np.arange(1, 10))
    # ...and the tail HOLDS rather than extrapolating
    assert out[0, 9, 0].item() == pytest.approx(9.0)


def test_sub_tick_lead_interpolates_linearly():
    traj = torch.arange(6, dtype=torch.float64).view(1, 6, 1).repeat(1, 1, N)
    out = lead_waypoints(traj, torch.full((N,), 0.5, dtype=torch.float64))
    assert np.allclose(out[0, :5, 0].numpy(), np.arange(5) + 0.5)
    assert out[0, 5, 0].item() == pytest.approx(5.0)


def test_lead_is_per_joint():
    traj = torch.arange(6, dtype=torch.float64).view(1, 6, 1).repeat(1, 1, 3)
    out = lead_waypoints(traj, torch.tensor([0.0, 0.5, 1.5], dtype=torch.float64))
    assert np.allclose(out[0, 0].numpy(), [0.0, 0.5, 1.5])
    assert np.allclose(out[0, 3].numpy(), [3.0, 3.5, 4.5])


def test_lead_holds_past_the_end_for_any_magnitude():
    """A lead longer than the plan may not read garbage or wrap around."""
    traj = torch.arange(5, dtype=torch.float64).view(1, 5, 1).repeat(1, 1, N)
    for f in (2.5, 4.0, 9.0, 40.0):
        out = lead_waypoints(traj, torch.full((N,), f, dtype=torch.float64))
        assert torch.isfinite(out).all()
        assert out.max().item() <= 4.0 + 1e-12
        assert out.min().item() >= 0.0


def test_negative_lead_is_rejected():
    with pytest.raises(ValueError):
        lead_fraction(-0.01, CONTROL_DT, N)


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        lead_fraction([0.01, 0.02], CONTROL_DT, N)


def test_gradient_flows_to_both_bracketing_waypoints():
    traj = torch.zeros(1, 4, 1, dtype=torch.float64, requires_grad=True)
    out = lead_waypoints(traj, torch.tensor([0.25], dtype=torch.float64))
    out[0, 0, 0].backward()
    g = traj.grad[0, :, 0].numpy()
    assert np.allclose(g, [0.75, 0.25, 0.0, 0.0])


def test_numpy_twin_matches_torch():
    traj = np.random.default_rng(0).normal(size=(9, N))
    f = lead_fraction(0.007, CONTROL_DT, N)
    a = lead_waypoints_np(traj, f)
    b = lead_waypoints(torch.as_tensor(traj)[None], torch.as_tensor(f))[0].numpy()
    assert np.allclose(a, b)


def test_presets_are_well_formed_and_resolvable():
    for name, v in LEAD_PRESETS.items():
        assert len(v) == N, name
        assert all(x >= 0.0 for x in v), name
        assert resolve_lead(name) == v
    assert resolve_lead("0.01") == pytest.approx(0.01)
    assert resolve_lead("off") == 0.0
    assert resolve_lead(0.02) == 0.02
    # sparse must be a subset of fit's support, and never larger per joint
    for a, b in zip(LEAD_PRESETS["sparse"], LEAD_PRESETS["fit"]):
        assert a == 0.0 or a <= b + 1e-12


# ------------------------------------------------- the horizon-dependent lead --
def test_is_schedule_only_fires_on_the_dict_form():
    assert is_schedule({"schedule": np.zeros((4, N)).tolist()})
    assert not is_schedule(0.0)
    assert not is_schedule(LEAD_PRESETS["fit"])
    assert not is_schedule(None)


def test_schedule_fraction_converts_rows_to_tick_fractions():
    tab = np.zeros((3, N))
    tab[0] = CONTROL_DT          # exactly one waypoint
    tab[1] = 0.5 * CONTROL_DT    # half a waypoint
    f = schedule_fraction({"schedule": tab.tolist()}, CONTROL_DT, N)
    assert f.shape == (3, N)
    assert np.allclose(f[0], 1.0)
    assert np.allclose(f[1], 0.5)
    assert np.allclose(f[2], 0.0)


def test_schedule_rejects_bad_shape_and_negative_leads():
    with pytest.raises(ValueError):
        schedule_fraction({"schedule": np.zeros((3, N - 1)).tolist()}, CONTROL_DT, N)
    with pytest.raises(ValueError):
        schedule_fraction({"schedule": np.zeros(N).tolist()}, CONTROL_DT, N)
    bad = np.zeros((2, N))
    bad[1, 0] = -0.01
    with pytest.raises(ValueError):
        schedule_fraction({"schedule": bad.tolist()}, CONTROL_DT, N)


def test_a_constant_schedule_row_reproduces_the_fixed_lead_exactly():
    """The schedule must be a strict generalization: a constant table is the
    fixed per-joint lead, waypoint for waypoint."""
    fixed = np.asarray(LEAD_PRESETS["fit"], dtype=float)
    tab = np.tile(fixed, (5, 1))
    f_sched = schedule_fraction({"schedule": tab.tolist()}, CONTROL_DT, N)
    f_fixed = lead_fraction(fixed, CONTROL_DT, N)
    traj = np.random.default_rng(0).normal(size=(9, N))
    for h in range(5):
        assert np.allclose(lead_waypoints_np(traj, f_sched[h]),
                           lead_waypoints_np(traj, f_fixed))
