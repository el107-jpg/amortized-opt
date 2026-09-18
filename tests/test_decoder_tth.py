"""The current-goal countdown moves from the ENCODER to the DECODER.

Completing the program ``ContextInputs.future_gap`` started
(``tests/test_future_gap.py``): ``goal_tth`` was the last from-now input the
context encoder had, and a from-now input ticks down every 20 ms, so the
deployment MPC had to re-encode the whole context every control tick
(``isaac_rl/README.md`` DECISION 2) -- which is the path by which the robot's
own measured state echoed back into the encoder and ran the waist away in
closed loop.

Four claims, each load-bearing:

1. **the seam** -- ``build_waypoint_masks`` returns the per-hand strike time on
   ``WaypointMasks.hit_tth``, so the countdown reaches the decoder without one
   producer in the tree changing its call signature; the quantized pool /
   deployment callers carry it too;
2. **the decoder reads it** under ``ModelConfig.dec_tth_feats``, and is exactly
   insensitive to it when the flag is off;
3. **the encoder does not** under ``enc_goal_tth=False``: two contexts that
   differ ONLY in ``goal_tth`` encode BIT-IDENTICALLY.  This is the structural
   property the whole change exists for -- it is what makes a per-segment
   encoder cache exact rather than approximate -- so it is asserted as exact
   equality, not a tolerance;
4. **a checkpoint reconstructs to the flags its weights and its stored config
   actually have**, in both directions.  A clean break is planned, so there is
   no weight migration; getting the reconstruction wrong is how an old
   checkpoint became unloadable by 21 entry points at once on 2026-08-12
   (``tests/test_model.py``, bottom).

Plus the consequence at deployment: ``IsaacMPC`` encodes on a reset, on a goal
segment boundary and on a yaw-BC move, and NOT on a plain tick inside a segment.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from amortized_opt.model import (  # noqa: E402
    AmortizedOptimizer,
    ModelConfig,
    model_config_from_checkpoint,
)
from amortized_opt.tokenizer import (  # noqa: E402
    ContextInputs,
    build_waypoint_masks,
    pack_state,
)
from amortized_opt.warmstart_data import WarmstartPool  # noqa: E402

DT = A.CONTROL_DT
H = 25


def _small(**kw) -> ModelConfig:
    return ModelConfig(d_model=32, dec_heads=2, dec_layers=1, d_ctx=32,
                       ctx_heads=2, ctx_layers=1, max_horizon=H, **kw)


def _model(**kw) -> AmortizedOptimizer:
    torch.manual_seed(0)
    return AmortizedOptimizer(_small(**kw)).eval()


def _ctx(B: int = 3, tth=(0.3, 0.5)) -> ContextInputs:
    ctx = ContextInputs.zeros(B, num_drums=8, num_future=3)
    ctx.drum_valid[:, :4] = True
    ctx.drum_pos[:, :4] = torch.tensor([0.4, 0.1, 0.9])
    ctx.goal_valid[:] = True
    ctx.goal_pos[:] = torch.tensor([0.35, 0.2, 0.85])
    ctx.goal_vel[:, :, 2] = -3.0
    ctx.goal_tth[:] = torch.tensor(tth)
    ctx.future_valid[:, :, :2] = True
    ctx.future_gap[:] = torch.tensor([0.25, 0.5, 0.0])
    return ctx


def _batch(m: AmortizedOptimizer, B: int = 3, tth=(0.3, 0.5)):
    ctx = _ctx(B, tth)
    tl = torch.full((B,), float(tth[0]))
    tr = torch.full((B,), float(tth[1]))
    masks = build_waypoint_masks(tl, tr, max_horizon=H, dt=DT)
    init = torch.zeros(B, H, A.N_DOF)
    state = pack_state(torch.zeros(B, A.N_DOF), torch.zeros(B, A.N_DOF))
    with torch.no_grad():
        enc = m.encode_context(ctx)
    return ctx, enc, init, state, masks


# =========================================================== 1. the seam ====
def test_masks_carry_the_hit_times():
    tl = torch.tensor([0.30, 0.44, 0.10])
    tr = torch.tensor([0.22, 0.60, 0.80])
    m = build_waypoint_masks(tl, tr, max_horizon=H, dt=DT)
    assert m.hit_tth.shape == (3, 2)
    assert torch.equal(m.hit_tth, torch.stack([tl, tr], dim=-1))


def test_an_idle_hand_reads_exactly_zero():
    """``hit_tth`` is NOT ``tau0``: an idle hand borrows a horizon, it has no hit.

    The gate lives in ``build_waypoint_masks`` because the producers disagree
    about what they pass for an idle hand -- the deployment MPCs pass 0, the
    phase-2 roller passes ``effective_hits`` (the borrowed horizon) -- and the
    decoder has no activity mask of its own to gate with.
    """
    tl = torch.tensor([0.30, 0.30])
    tr = torch.tensor([0.50, 0.50])
    act_l = torch.tensor([True, True])
    act_r = torch.tensor([True, False])
    m = build_waypoint_masks(tl, tr, max_horizon=H, dt=DT,
                             active_left=act_l, active_right=act_r)
    assert float(m.hit_tth[1, 1]) == 0.0
    # ...while the RESOLVED horizon it borrows is the left hand's, unchanged.
    assert float(m.tau0_right[1]) == pytest.approx(0.30)
    # the roller's convention (pass the borrowed horizon) gates to the same 0
    roller = build_waypoint_masks(tl, torch.tensor([0.50, 0.30]), max_horizon=H,
                                  dt=DT, active_left=act_l, active_right=act_r)
    assert torch.equal(roller.hit_tth, m.hit_tth)


def test_the_quantized_pool_path_carries_it_too():
    """``WarmstartPool.masks_for`` -- the phase-1/2 training path.

    Called unbound on a stand-in so the assertion is about THAT method rather
    than about a re-derivation of it; building a real pool costs minutes and
    tests nothing extra here.
    """
    pool = SimpleNamespace(
        dtype=torch.float32,
        cfg=SimpleNamespace(dt=DT, joint_names=A.G1_UPPER_BODY_JOINTS),
    )
    hit = torch.tensor([[7, 3], [4, -1]])
    active = hit >= 0
    m = WarmstartPool.masks_for(pool, hit, active, H)
    assert torch.allclose(
        m.hit_tth, torch.tensor([[7 * DT, 3 * DT], [4 * DT, 0.0]]), atol=1e-7
    )


# ====================================================== 2. the decoder ======
def test_the_decoder_reads_the_countdown():
    m = _model(dec_tth_feats=True)
    _ctx_, enc, init, state, masks = _batch(m)
    other = replace(masks, hit_tth=masks.hit_tth + 0.4)
    with torch.no_grad():
        a = m(init, state, enc, masks).traj
        b = m(init, state, enc, other).traj
    live = masks.elem_mask > 0.5
    assert (a - b)[live].abs().max().item() > 0.0


def test_the_flag_off_is_exactly_insensitive():
    m = _model(dec_tth_feats=False)
    _ctx_, enc, init, state, masks = _batch(m)
    other = replace(masks, hit_tth=masks.hit_tth + 0.4)
    with torch.no_grad():
        a = m(init, state, enc, masks).traj
        b = m(init, state, enc, other).traj
    assert torch.equal(a, b)


def test_the_flag_moves_the_state_projection_width():
    on, off = _model(dec_tth_feats=True), _model(dec_tth_feats=False)
    fw = on.cfg.fourier_width
    assert (on.decoder.state_proj.in_features
            - off.decoder.state_proj.in_features) == 2 * fw
    # no extra TOKEN: the decoder sequence is unchanged
    _c, enc, init, state, masks = _batch(on)
    with torch.no_grad():
        assert on(init, state, enc, masks).traj.shape == (3, H, A.N_DOF)


# ====================================================== 3. the encoder ======
def test_the_encoder_is_blind_to_goal_tth():
    """THE property: inside a goal segment the encoding cannot move.

    Exact equality, not a tolerance.  ``IsaacMPC``'s per-segment cache is only
    correct if the encoding it reuses is the one a re-encode would have
    produced, bit for bit.
    """
    m = _model(enc_goal_tth=False)
    with torch.no_grad():
        e0 = m.encode_context(_ctx(tth=(0.30, 0.50)))
        e1 = m.encode_context(_ctx(tth=(0.02, 4.00)))
    assert torch.equal(e0.memory, e1.memory)
    assert torch.equal(e0.mask, e1.mask)


def test_the_old_encoder_did_see_it():
    """``enc_goal_tth=True`` reproduces the pre-change encoder, for the A/B."""
    m = _model(enc_goal_tth=True)
    with torch.no_grad():
        e0 = m.encode_context(_ctx(tth=(0.30, 0.50)))
        e1 = m.encode_context(_ctx(tth=(0.02, 4.00)))
    assert not torch.equal(e0.memory, e1.memory)


def test_the_future_tokens_keep_their_gap():
    """Dropping the goal token's time must not disturb ``future_gap``.

    The two share ``goal_proj``, so an implementation that narrowed the shared
    projection would silently take the future tokens' timing with it.
    """
    m = _model(enc_goal_tth=False)
    a, b = _ctx(), _ctx()
    b.future_gap[:] = torch.tensor([1.5, 2.0, 0.0])
    with torch.no_grad():
        assert not torch.equal(m.encode_context(a).memory,
                               m.encode_context(b).memory)


# ================================================= 4. checkpoint round-trip ==
def _ckpt(cfg: ModelConfig, *, drop: tuple[str, ...] = ()) -> dict:
    """A checkpoint dict as ``save_checkpoint`` writes one, minus ``drop``."""
    sd = AmortizedOptimizer(cfg).state_dict()
    stored = {k: v for k, v in vars(cfg).items() if k not in drop}
    return {"model_cfg": stored, "model": sd}


def test_a_new_checkpoint_round_trips():
    cfg = _small()
    assert (cfg.dec_tth_feats, cfg.enc_goal_tth) == (True, False)
    got = model_config_from_checkpoint(_ckpt(cfg))
    assert got.dec_tth_feats is True and got.enc_goal_tth is False


def test_an_old_checkpoint_reconstructs_to_the_old_behaviour():
    """A config written before either field existed: OFF / ON, from the weights.

    ``dec_tth_feats`` is proved by ``decoder.state_proj``'s INPUT WIDTH (the key
    exists in every checkpoint; only its width moved).  ``enc_goal_tth`` moved no
    shape at all, so its proof is the field's ABSENCE from the stored config.
    """
    old = _small(dec_tth_feats=False, enc_goal_tth=True)
    ck = _ckpt(old, drop=("dec_tth_feats", "enc_goal_tth"))
    got = model_config_from_checkpoint(ck)
    assert got.dec_tth_feats is False and got.enc_goal_tth is True
    AmortizedOptimizer(got).load_state_dict(ck["model"])  # strict


def test_a_new_checkpoint_with_a_stripped_config_still_reads_its_weights():
    """The width proof has to work in the ON direction too, not just OFF."""
    new = _small(dec_tth_feats=True)
    ck = _ckpt(new, drop=("dec_tth_feats",))
    got = model_config_from_checkpoint(ck)
    assert got.dec_tth_feats is True
    AmortizedOptimizer(got).load_state_dict(ck["model"])


def test_a_stored_flag_is_believed_over_the_default():
    for flag in (True, False):
        ck = _ckpt(_small(enc_goal_tth=flag))
        assert model_config_from_checkpoint(ck).enc_goal_tth is flag


# ============================================= 5. deployment encode cadence ==
B = 2
MPC_H = 50


@pytest.fixture(scope="module")
def kit_and_clip():
    kit = A.DrumKit.from_layout()
    clip = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[0]
    return kit, clip


def _mpc(kit_and_clip, **cfg_kw):
    """A fresh CPU ``IsaacMPC`` + its schedule (Isaac-free, as in test_isaac_rl_loop)."""
    from isaac_rl.mpc import IsaacMPC, MPCConfig
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    kit, clip = kit_and_clip
    torch.manual_seed(0)
    mcfg = A.ModelConfig(d_model=32, dec_heads=2, dec_layers=1, d_ctx=32,
                         ctx_heads=2, ctx_layers=1, max_horizon=MPC_H)
    model = A.build_model(mcfg, "cpu").eval()
    scfg = ScheduleConfig(horizon=MPC_H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future)
    sched = BatchedSchedule(B, scfg, device="cpu")
    sched.reset(torch.arange(B), [clip] * B, [kit] * B,
                rng=np.random.default_rng(0))
    mpc = IsaacMPC(model, sched, MPCConfig(horizon=MPC_H, amp_dtype=None, **cfg_kw),
                   device="cpu", kit_names=tuple(kit.names))
    return mpc, sched


def _q(yaw: float = 0.0):
    q = torch.zeros(B, A.N_DOF)
    q[:, A.G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")] = yaw
    return q, torch.zeros(B, A.N_DOF)


def test_the_deployment_default_is_segment_cadence(kit_and_clip):
    from isaac_rl.mpc import MPCConfig

    assert MPCConfig().encode_every_tick is False


def test_reset_encodes_and_a_plain_tick_does_not(kit_and_clip):
    """The cache settles: the yaw ramp converges and then nothing moves."""
    mpc, _sched = _mpc(kit_and_clip)
    q, qd = _q()
    mpc.reset(torch.arange(B), q)
    seen = [mpc.plan(q, qd).encoded for _ in range(6)]
    assert seen[0] is True, "the first tick after a reset has no cache"
    assert seen[-1] is False, "a plain tick inside a goal segment must not encode"
    assert mpc.encode_fraction < 1.0


def test_a_regoal_forces_a_re_encode(kit_and_clip):
    """The segment key is the schedule's cursor + goal activity, not a countdown."""
    mpc, sched = _mpc(kit_and_clip)
    q, qd = _q()
    mpc.reset(torch.arange(B), q)
    for _ in range(6):
        mpc.plan(q, qd)
    assert mpc.plan(q, qd).encoded is False
    before = mpc._encode_key(sched.hit_index()).clone()
    for _ in range(200):
        sched.advance()
        if bool((mpc._encode_key(sched.hit_index()) != before).any()):
            break
    else:  # pragma: no cover - the eval clips all contain hits
        pytest.fail("the schedule never crossed a goal segment boundary")
    assert mpc.plan(q, qd).encoded is True


def test_a_yaw_bc_move_forces_a_re_encode(kit_and_clip):
    """The aim conditioning is the one input that is not segment-constant."""
    mpc, _sched = _mpc(kit_and_clip)
    q, qd = _q()
    mpc.reset(torch.arange(B), q)
    for _ in range(6):
        mpc.plan(q, qd)
    assert mpc.plan(q, qd).encoded is False
    # A big commanded turn: the leash drags the BC state after the measurement,
    # well past ``enc_yaw_tol``, and the aim token has to be recomputed there.
    q_turned, qd = _q(yaw=1.2)
    assert mpc.plan(q_turned, qd).encoded is True
    # ...and it re-encoded at the LEASHED BC state, not at the measurement:
    # the token follows the plant only as far as ``yaw_cond_leash`` allows.
    assert (mpc._yaw_enc - (1.2 - mpc.cfg.yaw_cond_leash)).abs().max().item() < 1e-5


def test_encode_every_tick_is_still_available(kit_and_clip):
    mpc, _sched = _mpc(kit_and_clip, encode_every_tick=True)
    q, qd = _q()
    mpc.reset(torch.arange(B), q)
    assert all(mpc.plan(q, qd).encoded for _ in range(4))
    assert mpc.encode_fraction == 1.0


def test_the_two_mpc_trees_agree(kit_and_clip):
    """``isaac_rl`` and ``isaac_rl_ppo`` carry the same cadence, line for line."""
    import isaac_rl.mpc as a
    import isaac_rl_ppo.mpc as b

    assert a.MPCConfig().encode_every_tick == b.MPCConfig().encode_every_tick
    assert a.MPCConfig().enc_yaw_tol == b.MPCConfig().enc_yaw_tol
    import inspect

    assert (inspect.getsource(a.IsaacMPC._encode_key)
            == inspect.getsource(b.IsaacMPC._encode_key))
    assert (inspect.getsource(a.IsaacMPC._encoding)
            == inspect.getsource(b.IsaacMPC._encoding))
