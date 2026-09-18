import pytest
import torch

from amortized_opt.replay_boundary_diag import (
    boundary_kinematics,
    c1_bridge_waypoints,
    finite_stats,
    gradient_cosine,
    localize_torque_excess,
    make_c1_roller_plan_transform,
    pearsonr_finite,
)


def test_boundary_kinematics_matches_knot_fd_definition():
    q0 = torch.tensor([[1.0, -1.0]])
    v0 = torch.tensor([[0.5, -0.5]])
    waypoints = torch.tensor([[[1.02, -1.04], [1.1, -1.1]]])

    got = boundary_kinematics(q0, v0, waypoints, dt=0.02)

    torch.testing.assert_close(got["delta"], torch.tensor([[0.02, -0.04]]))
    torch.testing.assert_close(got["implied_velocity"], torch.tensor([[1.0, -2.0]]))
    torch.testing.assert_close(
        got["implied_acceleration"], torch.tensor([[25.0, -75.0]]),
        atol=1e-4, rtol=1e-6,
    )
    assert float(got["q0_first_abs_max_rad"]) == pytest.approx(0.04)


def test_c1_bridge_preserves_join_suffix_and_short_horizon():
    # Joint 0 has five live waypoints and is bridged to original knot 3.
    # Joint 1 has one live waypoint, which cannot be repaired without moving
    # its terminal and must therefore remain unchanged.
    q0 = torch.tensor([[0.0, 2.0]])
    v0 = torch.tensor([[0.0, 0.0]])
    waypoints = torch.tensor([[
        [0.9, 2.4],
        [1.2, 0.0],
        [1.5, 0.0],
        [1.8, 0.0],
        [2.1, 0.0],
    ]])
    mask = torch.tensor([[
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]])

    got = c1_bridge_waypoints(
        waypoints, q0, v0, dt=0.02, bridge_knots=3, elem_mask=mask,
    )

    assert not torch.equal(got[:, :2, 0], waypoints[:, :2, 0])
    torch.testing.assert_close(got[:, 2:, 0], waypoints[:, 2:, 0])
    torch.testing.assert_close(got[:, :, 1], waypoints[:, :, 1])


def test_roller_bridge_helper_zero_is_identity_and_positive_matches_projection():
    assert make_c1_roller_plan_transform(0) is None
    with pytest.raises(ValueError, match="must be >= 0"):
        make_c1_roller_plan_transform(-1)

    q0 = torch.tensor([[0.1]])
    v0 = torch.tensor([[0.2]])
    traj = torch.tensor([[[0.4], [0.6], [0.8], [1.0]]])
    mask = torch.ones_like(traj)
    transform = make_c1_roller_plan_transform(3)
    got = transform(traj=traj, q0=q0, v0=v0, elem_mask=mask, dt=0.02)
    expected = c1_bridge_waypoints(
        traj, q0, v0, dt=0.02, bridge_knots=3, elem_mask=mask,
    )
    torch.testing.assert_close(got, expected)


def test_torque_localization_finds_boundary_joint_and_segment():
    tau = torch.zeros(2, 6, 3)
    limit = torch.ones_like(tau) * 5.0
    mask = torch.ones_like(tau)
    tau[0, 0, 1] = 9.0  # boundary excess 4
    tau[0, 4, 2] = 7.0  # later excess 2
    tau[1, 3, 0] = 6.0  # later excess 1

    got = localize_torque_excess(
        tau, limit, mask,
        segment_indices={"proximal": [0, 1], "distal": [2]},
    )

    torch.testing.assert_close(got["row_peak_excess_nm"], torch.tensor([4.0, 1.0]))
    assert got["row_peak_knot"].tolist() == [0, 3]
    assert got["row_peak_joint"].tolist() == [1, 0]
    torch.testing.assert_close(
        got["windows"]["first_1"]["peak_excess_nm"], torch.tensor([4.0, 0.0]),
    )
    torch.testing.assert_close(
        got["windows"]["rest_after_3"]["peak_excess_nm"], torch.tensor([2.0, 1.0]),
    )
    torch.testing.assert_close(
        got["by_segment_peak_excess_nm"]["proximal"], torch.tensor([4.0, 1.0]),
    )
    torch.testing.assert_close(
        got["by_segment_peak_excess_nm"]["distal"], torch.tensor([2.0, 0.0]),
    )


def test_small_summary_helpers_handle_nonfinite_and_conflict():
    stats = finite_stats(torch.tensor([1.0, 2.0, float("nan"), float("inf")]))
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(1.5)

    corr = pearsonr_finite([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
    assert corr == {"n": 3, "r": pytest.approx(-1.0)}
    assert pearsonr_finite([1.0, 1.0], [2.0, 3.0])["r"] is None

    cos = gradient_cosine(
        [torch.tensor([1.0, 0.0]), None],
        [torch.tensor([-2.0, 0.0]), torch.tensor([3.0])],
    )
    assert cos["cosine"] == pytest.approx(-2.0 / (13.0 ** 0.5))
    assert cos["conflict"] is True
