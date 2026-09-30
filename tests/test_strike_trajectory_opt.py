"""Focused contract tests for the actor-free strike trajectory editor."""

from __future__ import annotations

import json

import pytest
import torch

from tools.strike_pose_editor.trajectory_opt import (
    DirectTrajectoryOptimizer,
    TaskSpaceConstraint,
    WaypointKeyframe,
    quintic_terminal_initialization,
    sample_trajectory_scenario,
    warp_terminal_approach,
)


@pytest.fixture(scope="module")
def scenario():
    # This seed has asymmetric hand timing, which catches the historical bug
    # where both terminal edits were placed at max(left_hit, right_hit).
    return sample_trajectory_scenario(12, start_chunk=8)


def test_scenario_is_complete_replayable_json(scenario):
    doc = scenario.to_json()
    assert doc["horizon"] == 50
    assert len(doc["q0"]) == 17
    assert len(doc["init"]) == 50 and len(doc["init"][0]) == 17
    assert doc["active"] == [True, True]
    assert [g["hit_index"] for g in doc["goals"]] == doc["hit_index"]
    assert [g["tth_s"] for g in doc["goals"]] == pytest.approx(
        [k * doc["dt"] for k in doc["hit_index"]]
    )
    assert len(doc["kit"]["drums"]) == 7
    assert len(doc["masks"]["elem_mask"]) == 50
    assert "goal_pos" in doc["context"]
    assert "p_star" in doc["strike_context"]
    json.dumps(doc, allow_nan=False)


def test_hit_keyframes_resolve_each_hands_own_index_and_hard_project(scenario):
    left_joint = "left_elbow_joint"
    right_joint = "right_elbow_joint"
    li = scenario.joint_names.index(left_joint)
    ri = scenario.joint_names.index(right_joint)
    lh, rh = (int(v) for v in scenario.hit_index[0])
    assert lh != rh
    lv = float(scenario.init[0, lh - 1, li])
    rv = float(scenario.init[0, rh - 1, ri])
    opt = DirectTrajectoryOptimizer(
        scenario,
        initial_trajectory=scenario.init,
        keyframes=[
            WaypointKeyframe.at_hit("left", {left_joint: lv}, mode="hard"),
            WaypointKeyframe.at_hit("right", {right_joint: rv}, mode="hard"),
        ],
    )
    assert [k["waypoint"] for k in opt.keyframes_json()] == [lh, rh]
    progress = opt.step(2)
    assert len(progress) == 2 and progress[-1]["iteration"] == 2
    assert torch.isfinite(opt.trajectory).all()
    assert float(opt.trajectory[0, lh - 1, li]) == pytest.approx(lv)
    assert float(opt.trajectory[0, rh - 1, ri]) == pytest.approx(rv)


def test_result_contains_teacher_geometry_and_cost_diagnostics(scenario):
    joint = "left_wrist_pitch_joint"
    hit = int(scenario.hit_index[0, 0])
    col = scenario.joint_names.index(joint)
    target = float(scenario.init[0, hit - 1, col]) + 0.01
    opt = DirectTrajectoryOptimizer(
        scenario,
        initial_trajectory=scenario.init,
        keyframes=[WaypointKeyframe.at_hit("left", {joint: target}, weight=2.0)],
    )
    initial = opt.result_json(include_geometry=False)["diagnostics"]["cost"]["total"]
    opt.step(3)
    doc = opt.result_json()
    assert len(doc["trajectory"]) == 50
    assert doc["teacher"] == doc["trajectory"]
    assert len(doc["teacher_mask"]) == 50
    assert len(doc["geometry"]["frames"]) == 50
    assert len(doc["geometry"]["paths"]["left_tip"]) == 50
    assert doc["geometry"]["frames"][0]["waypoint"] == 1
    assert doc["geometry"]["frames"][-1]["waypoint"] == 50
    assert doc["initialization_source"] == "actor_proposal"
    assert torch.allclose(
        torch.tensor(doc["actor_initial_trajectory"]), scenario.init[0]
    )
    assert doc["diagnostics"]["feasibility"]["accepted"] == doc[
        "diagnostics"
    ]["feasibility"]["feasible"]
    assert "approach_corridor" in doc["diagnostics"]
    for row in doc["diagnostics"]["approach_corridor"].values():
        assert 1 <= len(row["waypoints"]) <= 10
        assert row["required_clearance_m"][0] > row["required_clearance_m"][-1]
        assert all(value >= 0.0 for value in row["violation_m"])
    assert doc["diagnostics"]["cost"]["total"] < initial
    json.dumps(doc, allow_nan=False)

    bundle = scenario.training_bundle(opt.trajectory)
    assert tuple(bundle["state"].shape) == (1, 34)
    assert tuple(bundle["teacher_q"].shape) == (1, 50, 17)
    assert tuple(bundle["teacher_mask"].shape) == (1, 50, 17)


def test_task_space_constraint_is_persistent_and_reduces_terminal_error(scenario):
    reference = DirectTrajectoryOptimizer(scenario, initial_trajectory=scenario.init)
    geometry = reference.geometry_json()
    hand = "left"
    hit = int(scenario.hit_index[0, 0])
    frame = geometry["frames"][hit - 1]["sticks"][hand]
    target = TaskSpaceConstraint(
        hand=hand,
        contact_position=frame["contact"],
        stick_direction=frame["axis"],
        waist=scenario.init[0, hit - 1, :3].tolist(),
    )
    perturbed = scenario.init.clone()
    elbow = scenario.joint_names.index("left_elbow_joint")
    perturbed[0, hit - 1, elbow] += 0.12
    optimizer = DirectTrajectoryOptimizer(
        scenario, initial_trajectory=perturbed, task_constraints=[target],
    )
    before = optimizer.result_json(include_geometry=False)
    optimizer.step(12)
    after = optimizer.result_json(include_geometry=False)
    assert after["diagnostics"]["task_space_loss"] < before["diagnostics"]["task_space_loss"]
    assert after["task_constraints"][0]["hand"] == hand
    assert "left_task_direction" in after["diagnostics"]["feasibility"]["checks"]


def test_terminal_warp_spreads_endpoint_edit_with_minimum_jerk(scenario):
    hit = int(scenario.hit_index[0, 0])
    cols = [
        index for index, name in enumerate(scenario.joint_names)
        if name.startswith("waist_") or name.startswith("left_")
    ]
    terminal = scenario.init[0, hit - 1].clone()
    terminal[cols] += 0.08
    warped = warp_terminal_approach(
        scenario.init,
        terminal,
        hit_waypoint=hit,
        joint_indices=cols,
        window_waypoints=16,
    )
    delta = warped - scenario.init
    first = max(0, hit - 16)
    assert torch.allclose(warped[0, hit - 1, cols], terminal[cols])
    assert bool((delta[0, first : hit - 1, cols].abs() > 0.0).any())
    assert torch.allclose(delta[0, :first], torch.zeros_like(delta[0, :first]))
    other = [
        index for index, name in enumerate(scenario.joint_names)
        if name.startswith("right_")
    ]
    assert torch.allclose(delta[0, :, other], torch.zeros_like(delta[0, :, other]))
    # The quintic displacement increases monotonically without an endpoint
    # jump, unlike replacing only the hit row.
    edit = delta[0, first:hit, cols[0]]
    assert bool((edit[1:] >= edit[:-1]).all())


@pytest.mark.parametrize("hand", ("left", "right"))
def test_fresh_terminal_quintic_replaces_only_selected_side(scenario, hand):
    hand_index = 0 if hand == "left" else 1
    hit = int(scenario.hit_index[0, hand_index])
    selected = [
        index for index, name in enumerate(scenario.joint_names)
        if name.startswith("waist_") or name.startswith(f"{hand}_")
    ]
    other = [
        index for index, name in enumerate(scenario.joint_names)
        if name.startswith(f"{'right' if hand == 'left' else 'left'}_")
    ]
    terminal = scenario.init[0, hit - 1].clone()
    terminal[selected] += 0.03
    fresh = quintic_terminal_initialization(scenario, terminal, hand=hand)

    assert tuple(fresh.shape) == (1, 50, 17)
    torch.testing.assert_close(fresh[0, hit - 1, selected], terminal[selected])
    torch.testing.assert_close(
        fresh[0, hit:, selected],
        terminal[selected].expand(50 - hit, -1),
    )
    torch.testing.assert_close(fresh[0, :, other], scenario.init[0, :, other])
    assert bool(torch.isfinite(fresh).all())


@pytest.mark.parametrize("hand", ("left", "right"))
def test_single_hand_teacher_bundle_masks_other_arm_and_late_rows(scenario, hand):
    bundle = scenario.training_bundle(scenario.init, teaching_hand=hand)
    mask = bundle["teacher_mask"]
    hit = int(scenario.hit_index[0, 0 if hand == "left" else 1])
    selected = [
        index for index, name in enumerate(scenario.joint_names)
        if name.startswith("waist_") or name.startswith(f"{hand}_")
    ]
    other = [
        index for index, name in enumerate(scenario.joint_names)
        if name.startswith(f"{'right' if hand == 'left' else 'left'}_")
    ]
    assert bundle["teaching_hand"] == hand
    assert bool((mask[:, :hit, selected] > 0).any())
    assert not bool((mask[:, :, other] > 0).any())
    assert not bool((mask[:, hit:, :] > 0).any())

    with pytest.raises(ValueError, match="teaching_hand"):
        scenario.training_bundle(scenario.init, teaching_hand="both")


def test_solver_improves_terminal_and_collision_and_reports_infeasibility(scenario):
    # Stand in for an imperfect actor proposal with a localized terminal error.
    # The quintic remains scenario.init (the actor INPUT); the explicit tensor
    # below is the only trajectory the optimizer is allowed to refine.
    actor_q = scenario.init.clone()
    hit = int(scenario.hit_index[0, 0])
    elbow = scenario.joint_names.index("left_elbow_joint")
    actor_q[0, hit - 1, elbow] += 0.2
    opt = DirectTrajectoryOptimizer(scenario, initial_trajectory=actor_q)
    before = opt.status_json()["feasibility"]
    doc = opt.solve(max_iterations=20, min_iterations=20)
    after = doc["diagnostics"]["feasibility"]

    assert before["terminal_region_max_m"] - after["terminal_region_max_m"] > 1e-3
    assert (
        before["collision_penetration_max_m"]
        - after["collision_penetration_max_m"]
        > 1e-3
    )
    assert after["normalized_violation_score"] < before["normalized_violation_score"]
    # Twenty iterations improve this deliberately poor proposal but do not meet
    # the zero-tolerance torque/collision gates.  The result must say so rather
    # than bless a merely lower scalar objective as a teacher.
    assert after["accepted"] is False and after["feasible"] is False
    assert doc["diagnostics"]["convergence"]["converged"] is False
    assert doc["diagnostics"]["convergence"]["reason"] == "max_iterations_infeasible"
