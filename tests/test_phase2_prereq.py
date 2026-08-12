"""Prerequisite fixes for phase 2 (section 7).

Fix A -- the MPC generator's waist wind-up.  The loop
measured-yaw-rate -> ``waist_reference_block(yaw_rate_start=...)`` -> FOH
command -> measured-yaw-rate is positive feedback; it took ``waist_yaw`` from
1.5 to 2.8 rad in ~90 ticks and then commanded a 5.2 rad step (262 rad/s of
feed-forward), blowing the PD state up to 1e11.  Three guards now apply: the
yaw BC comes from the PLANNED state, everything is clipped into the joint box,
and both the seed and the executed command are slew-limited.

Fix B -- kit domain randomization has to reach the *cost*, not only the goal
tokens.  ``FusedCost`` now accepts per-sample drum head centres, and the
canonical way to build them is from the context tokens themselves, so the two
cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext, drum_heads_from_context
from amortized_opt.limits import isaac_velocity_limits, urdf_position_limits
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator

NAMES = A.G1_UPPER_BODY_JOINTS
DT = A.CONTROL_DT


# ============================================================== fix A ========
@pytest.fixture(scope="module")
def long_clip_and_gen():
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    clip = max(clips, key=lambda c: c.duration)
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(divergence_box_margin=1e9))
    sampler = A.StartSampler(
        kit=kit, library=gen.library, fk=gen.fk, rnea=gen.rnea, sdf=gen.sdf
    )
    return kit, clip, gen, sampler


def test_mpc_generator_waist_does_not_wind_up(long_clip_and_gen):
    """300 ticks on a real clip with the quintic stand-in stay admissible."""
    _kit, clip, gen, sampler = long_clip_and_gen
    starts = sampler.sample(3, seed=0)
    lo, hi = urdf_position_limits(NAMES)
    v_lim = isaac_velocity_limits(NAMES)
    yaw_i = NAMES.index("waist_yaw_joint")

    for ep in range(3):
        res = gen.run_episode(
            clip,
            starts.q[ep].cpu().numpy(),
            starts.qd[ep].cpu().numpy(),
            seed=100 + ep,
            max_steps=300,
        )
        q, qd, cmd = res.q_hist, res.qd_hist, res.cmd_hist
        assert cmd.shape[0] == 300, f"episode {ep} stopped early ({cmd.shape[0]} ticks)"
        assert not res.diverged

        # PD state finite.
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(qd))

        # waist_yaw inside the joint box, measured AND commanded.
        assert lo[yaw_i] <= q[:, yaw_i].min() and q[:, yaw_i].max() <= hi[yaw_i]
        assert lo[yaw_i] <= cmd[:, yaw_i].min() and cmd[:, yaw_i].max() <= hi[yaw_i]

        # No command step beyond velocity limit x dt, on ANY joint.
        step = np.abs(np.diff(cmd, axis=0))
        assert np.all(step <= v_lim[None, :] * DT + 1e-9), (
            f"episode {ep}: worst command rate "
            f"{(step / (v_lim[None, :] * DT)).max():.3f} x the rating"
        )

        # The waist as a whole stays sane: pitch/roll are frozen at a PLANNED
        # setpoint, so they cannot random-walk out of their +-0.52 rad box.
        w = [i for i, n in enumerate(NAMES) if n.startswith("waist_")]
        assert np.all(cmd[:, w] >= lo[None, w] - 1e-9)
        assert np.all(cmd[:, w] <= hi[None, w] + 1e-9)


def test_planned_yaw_bc_is_what_fixes_it(long_clip_and_gen):
    """The old measured-rate BC is still reachable, and it is still broken."""
    kit, clip, _gen, sampler = long_clip_and_gen
    old = MPCGenerator(
        kit=kit,
        config=MPCGenConfig(
            planned_yaw_bc=False, seed_slew_eta=None, cmd_slew_eta=None,
            divergence_box_margin=1e9,
        ),
    )
    starts = sampler.sample(1, seed=0)
    res = old.run_episode(
        clip, starts.q[0].cpu().numpy(), starts.qd[0].cpu().numpy(),
        seed=100, max_steps=300,
    )
    yaw_i = NAMES.index("waist_yaw_joint")
    lo, hi = urdf_position_limits(NAMES)
    span = float(res.q_hist[:, yaw_i].max() - res.q_hist[:, yaw_i].min())
    out_of_box = bool(
        res.q_hist[:, yaw_i].min() < lo[yaw_i] or res.q_hist[:, yaw_i].max() > hi[yaw_i]
    )
    # Either it leaves the box or it swings absurdly; the fixed path does neither.
    assert out_of_box or span > 3.0


def test_warmstart_slew_limit_is_opt_in_and_exact():
    """``slew_limit_eta=None`` reproduces the phase-1 seed bit for bit."""
    lib = A.GoalPoseLibrary()
    rng = np.random.default_rng(0)
    q0 = ready_joint_pos(NAMES) + 0.05 * rng.normal(size=len(NAMES))
    v0 = np.zeros(len(NAMES))
    kw = dict(
        horizon=25, goals={"left": ("hihat", 0.30), "right": ("snare", 0.24)},
        library=lib, aim_targets=None, dt=DT, joint_names=NAMES,
        q_bounds=urdf_position_limits(NAMES),
    )
    base = A.warmstart_trajectory(q0, v0, **kw)
    same = A.warmstart_trajectory(q0, v0, slew_limit_eta=None, **kw)
    assert np.array_equal(base.q, same.q)

    limited = A.warmstart_trajectory(q0, v0, slew_limit_eta=0.7, **kw)
    v_lim = isaac_velocity_limits(NAMES)
    step = np.abs(np.diff(limited.q, axis=0))
    assert np.all(step <= 0.7 * v_lim[None, :] * DT + 1e-9)


def test_yaw_state_override_beats_a_hostile_measured_rate():
    """A 200 rad/s measured yaw rate must not produce a 4 rad first waypoint."""
    lib = A.GoalPoseLibrary()
    q0 = ready_joint_pos(NAMES)
    v0 = np.zeros(len(NAMES))
    v0[NAMES.index("waist_yaw_joint")] = 200.0
    kw = dict(
        horizon=25, goals={"left": ("hihat", 0.30), "right": None},
        library=lib, aim_targets={"left": np.array([0.35, 0.30, 1.05])},
        dt=DT, joint_names=NAMES, q_bounds=urdf_position_limits(NAMES),
    )
    yaw_i = NAMES.index("waist_yaw_joint")
    v_lim = isaac_velocity_limits(NAMES)[yaw_i]

    measured = A.warmstart_trajectory(q0, v0, **kw)
    planned = A.warmstart_trajectory(q0, v0, yaw_state=(0.0, 0.0), **kw)
    lo, hi = urdf_position_limits(NAMES)

    for ws in (measured, planned):
        yaw = ws.q[:, yaw_i]
        assert lo[yaw_i] <= yaw.min() and yaw.max() <= hi[yaw_i]
        # the yaw column is always slew-limited, even without the override
        assert np.all(np.abs(np.diff(yaw)) <= 0.7 * v_lim * DT + 1e-9)
    # ...and the override additionally removes the 200 rad/s slope entirely.
    assert abs(planned.q[1, yaw_i] - planned.q[0, yaw_i]) <= abs(
        measured.q[1, yaw_i] - measured.q[0, yaw_i]
    ) + 1e-12

    # the carry-forward state is the planned one, not the measured one
    y, r = planned.yaw_state_after(1)
    assert abs(r) <= 0.7 * v_lim + 1e-6
    assert lo[yaw_i] <= y <= hi[yaw_i]


# ============================================================== fix B ========
def test_context_tokens_and_cost_scene_are_the_same_array():
    """The goal context matches the scene exactly, by construction."""
    kit = A.DrumKit.from_layout()
    rng = np.random.default_rng(7)
    kits = [kit.randomized(rng) for _ in range(4)]
    snaps = [
        A.GoalSnapshot(t_now=0.0, goal={h: A.HandGoal.inactive() for h in ("left", "right")},
                       future={h: [] for h in ("left", "right")})
        for _ in kits
    ]
    ctx = A.build_context_inputs(kits, snaps, num_drums=8, num_future=3, frame="env",
                                 dtype=torch.float64)
    heads = drum_heads_from_context(ctx.drum_pos)
    for b, k in enumerate(kits):
        d = k.num_drums
        np.testing.assert_allclose(
            heads[b, :d].numpy(), k.positions_in("pelvis"), atol=1e-12
        )


@pytest.fixture(scope="module")
def cost_and_case():
    dev = "cpu"
    cost = FusedCost(CostConfig(), max_horizon=30, device=dev, dtype=torch.float64)
    scorer = A.TrajectoryScorer(device=dev)
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)
    return cost, scorer, cases


def _batch_from_case(cost, case, planner):
    res = planner(case)
    Q = np.asarray(res.Q if hasattr(res, "Q") else res, float)
    H = cost.max_horizon
    wp = np.repeat(Q[-1][None, :], H, axis=0)
    wp[: Q.shape[0] - 1] = Q[1:]
    ctx = A.strike_context_from_cases([case], drum_index=cost.drum_index,
                                      device=cost.device, dtype=cost.dtype)
    t = lambda a: torch.as_tensor(np.asarray(a, float), device=cost.device, dtype=cost.dtype)
    return t(Q[0])[None, :], t(wp)[None, :, :], t(case.qdot_start)[None, :], ctx


def test_shifting_a_drum_moves_the_cost_optimum(cost_and_case):
    """Moving a drum 5 cm moves the terminal AND the collision optimum."""
    cost, _scorer, cases = cost_and_case
    case = next(c for c in cases if c.drum == "snare")
    plan = A.quintic_case_planner()
    q0, wp, v0, ctx = _batch_from_case(cost, case, plan)

    D = cost.n_drums
    nominal = torch.as_tensor(
        cost.scene.head_centers, device=cost.device, dtype=cost.dtype
    ).unsqueeze(0)                              # (1, D, 3) pelvis frame
    shift = torch.zeros_like(nominal)
    shift[0, :, 0] = 0.05                       # whole kit 5 cm forward in x

    ctx_nom = StrikeContext(**{**ctx.__dict__, "drum_head_centers": nominal})
    ctx_shift_scene = StrikeContext(**{**ctx.__dict__, "drum_head_centers": nominal + shift})
    # ...and the same shift applied consistently to the strike target too.
    d = ctx.__dict__.copy()
    d["p_star"] = ctx.p_star + torch.tensor([0.05, 0.0, 0.0], device=cost.device,
                                            dtype=cost.dtype)
    d["drum_head_centers"] = nominal + shift
    ctx_shift_all = StrikeContext(**d)

    out_nom = cost.forward(q0, wp, v0, ctx_nom)
    out_scene = cost.forward(q0, wp, v0, ctx_shift_scene)
    out_all = cost.forward(q0, wp, v0, ctx_shift_all)

    # Passing the nominal centres explicitly must reproduce the baked-in scene.
    out_default = cost.forward(q0, wp, v0, ctx)
    assert torch.allclose(out_nom.raw["collision_drum"], out_default.raw["collision_drum"])
    assert torch.allclose(out_nom.total, out_default.total)

    # The collision term sees the shifted bodies.
    assert not torch.allclose(
        out_nom.raw["collision_drum"], out_scene.raw["collision_drum"], atol=1e-9
    )
    assert float((out_nom.diagnostics["collision_min_sdf"]
                  - out_scene.diagnostics["collision_min_sdf"]).abs().max()) > 1e-3

    # The terminal term sees the shifted target: a trajectory aimed at the
    # nominal head is further from the legal region of the shifted one.
    assert float(out_all.diagnostics["terminal_region_dist"][0]) > float(
        out_nom.diagnostics["terminal_region_dist"][0]
    ) - 1e-12
    assert float(out_all.diagnostics["contact_pos_err"][0]) > float(
        out_nom.diagnostics["contact_pos_err"][0]
    )


def test_start_sampler_screens_against_the_randomized_kit():
    """The collision screen uses the kit the sample will be scored against."""
    kit = A.DrumKit.from_layout()
    rng = np.random.default_rng(3)
    rk = kit.randomized(rng)
    sampler = A.StartSampler(kit=kit)
    batch = sampler.sample(48, seed=1, kit=rk)
    assert float(batch.min_sdf.min()) >= 0.0

    # Re-screen the accepted poses against the randomized geometry explicitly.
    heads = torch.as_tensor(
        rk.positions_in("env"), device=sampler.device, dtype=sampler.dtype
    ).unsqueeze(0).expand(len(batch), -1, -1)
    with torch.no_grad():
        _pen, det = sampler.sdf.forward(
            batch.q,
            target_drum_left=batch.drum[:, 0],
            target_drum_right=batch.drum[:, 1],
            drum_head_centers=heads,
            return_details=True,
        )
    assert float(det.min_sdf.min()) >= 0.0
