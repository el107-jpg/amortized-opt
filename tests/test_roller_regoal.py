"""The MPC roller's per-hand, in-place goal switch (README decisions 3 and 8).

The phase-2 model is strong single-shot and diverges in closed loop, and the
diagnosed cause is that the training roller **respawned** a finished slot from
the scenario pool while the deployment loop
(:meth:`~amortized_opt.mpc_gen.MPCGenerator.run_episode`) does an in-place,
per-hand re-goal over a shared waist.  These tests pin the roller to the
deployment semantics:

1. the tensorized strike table the re-goal re-prices with agrees with the numpy
   path (:func:`~amortized_opt.phase2_data._strike_row` /
   :func:`~amortized_opt.rhythm.strike_region`) to 1e-6 on randomized kits;
2. a hand that fires mid-flight switches goal **without** touching the plant,
   the gains, the other hand's plan columns or the other hand's strike row;
3. the context's time-to-hit tokens tick down with the mask's tau encodings
   (they did not before -- every replay row carried a stale ``goal_tth``);
4. a hand with no promotable future goes idle, and a slot with no active hand
   respawns;
5. divergence still hard-respawns;
6. harvested contexts still collate through :class:`ReplayPool`;
7. ``regoal=False`` still reproduces the old respawn-on-hit behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import PELVIS_IN_ENV
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig
from amortized_opt.phase2_data import (
    Phase2Pool,
    Phase2ScenarioConfig,
    _strike_row,
    generate_cost_scenarios,
    strike_geometry_table,
)
from amortized_opt.rhythm import DrumKit, KitRandomization, strike_region
from amortized_opt.train_cost import MPCRoller, Phase2Config, ReplayPool, build_costs
from amortized_opt.warmstart import arm_slice

from conftest import pick_cuda_device

DEV = pick_cuda_device() or "cpu"
HORIZON = 25
_ARM = {0: arm_slice("left"), 1: arm_slice("right")}


# ============================================ 1. tensorized strike table =====
def test_strike_geometry_table_matches_numpy():
    """32 randomized kits x every drum, every field, 1e-6.

    Frames are the trap: ``_strike_row`` works in the PELVIS frame while the
    context tokens (``drum_pos`` / ``goal_pos`` / ``future_pos``) are ENV.  The
    table is handed pelvis-frame centres, so the caller must subtract
    ``PELVIS_IN_ENV`` -- if it does not, every re-priced strike row is 80 cm off
    in z and the cost silently mis-prices.
    """
    kit0 = DrumKit.from_layout()
    rng = np.random.default_rng(7)
    kits = [kit0.randomized(rng, KitRandomization()) for _ in range(32)]
    D = kit0.num_drums

    pos = torch.as_tensor(np.stack([k.positions_in("pelvis") for k in kits]), dtype=torch.float64)
    axis = torch.as_tensor(np.stack([k.axes for k in kits]), dtype=torch.float64)
    rad = torch.as_tensor(np.stack([k.radii for k in kits]), dtype=torch.float64)
    tab = strike_geometry_table(pos, axis, rad, kit0.names)

    err: dict[str, float] = {}

    def bump(name: str, a, b) -> None:
        e = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
        err[name] = max(err.get(name, 0.0), e)

    for b, kit in enumerate(kits):
        for d in range(D):
            aim, center, disk_r, half = strike_region(kit, d, frame="pelvis")
            bump("aim", aim, tab["aim"][b, d].numpy())
            bump("disk_center", center, tab["disk_center"][b, d].numpy())
            bump("hit_radius", disk_r, tab["hit_radius"][b, d].item())
            bump("hp_dir", np.zeros(3) if half is None else half, tab["hp_dir"][b, d].numpy())
            bump("has_hp", 0.0 if half is None else 1.0, tab["has_hp"][b, d].item())

            # a goal ON that drum: jittered off the aim so disk_off is non-zero
            p_env = kit.positions[d] + np.array([0.017, -0.023, 0.004])
            v = np.array([0.4, -0.3, -2.2])
            row = _strike_row(kit, d, p_env, v)
            bump("sigma", row["sigma"], tab["sigma"][b, d].item())
            bump("shaft", row["shaft"], tab["shaft"][b, d].item())
            bump("elev", row["elev"], tab["elev"][b, d].item())

            p_star = p_env - PELVIS_IN_ENV
            off = tab["disk_center"][b, d].numpy() - p_star
            a = tab["axis"][b, d].numpy()
            off = off - float(off @ a) * a
            bump("disk_off", row["disk_off"], off)
            bump("hp_dir_row", row["hp_dir"], tab["hp_dir"][b, d].numpy())
            bump("has_hp_row", row["has_hp"], tab["has_hp"][b, d].item())
            bump("hit_radius_row", row["hit_radius"], tab["hit_radius"][b, d].item())

    worst = max(err.values())
    assert worst < 1e-6, f"tensorized strike table disagrees with numpy: {err}"


# ==================================================== roller fixtures ========
@pytest.fixture(scope="module")
def bits():
    if torch.device(DEV).type == "cuda":
        torch.cuda.set_device(torch.device(DEV).index or 0)
    kit = DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = Phase2ScenarioConfig(horizon=HORIZON, buckets=(HORIZON,))
    raw = generate_cost_scenarios(300, clips, seed=3, cfg=cfg)
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device=DEV)
    torch.manual_seed(0)
    model = A.build_model(A.ModelConfig(max_horizon=HORIZON), DEV)
    p2 = Phase2Config(compile_mode=None, device=DEV)
    costs = build_costs(pool.buckets, p2.cost, device=DEV)
    sim = BatchedPDSim(PDSimConfig(dt=cfg.dt), device=DEV, dtype=torch.float32,
                       rnea=costs[HORIZON].rnea)
    return pool, model, sim, costs[HORIZON]


def make_roller(bits, *, slots=24, seed=0, regoal=True, chain_frac=0.0):
    pool, model, sim, cost = bits
    return MPCRoller(
        pool, model=model, sim=sim, rnea=cost.rnea, fk=cost.fk,
        horizon=HORIZON, slots=slots, chain_frac=chain_frac, device=DEV,
        dtype=torch.float32, seed=seed, regoal=regoal,
    )


class _HoldModel:
    """Replace the decoder with "command the measured pose for every waypoint".

    A randomly-initialized decoder drives ~8 % of slots out of the joint box per
    tick, so any assertion about a *particular* slot surviving N ticks would be
    flaky.  Holding makes the plant deterministic and quiet; the goal-switch
    bookkeeping under test does not depend on what the decoder emits.
    """

    def __init__(self, roller):
        self.core = getattr(roller.model, "module", roller.model)
        self._fwd = self.core.forward

        def fwd(init, state, enc, masks):
            n = state.shape[-1] // 2
            q0 = state[:, :n]
            return SimpleNamespace(
                traj=q0.unsqueeze(1).expand(-1, init.shape[1], -1).contiguous()
            )

        self.core.forward = fwd

    def restore(self):
        self.core.forward = self._fwd


class _Spy:
    """Records the plan, the PD output and every respawn of one tick."""

    def __init__(self, roller):
        self.roller = roller
        self.spawned: list[torch.Tensor] = []
        self.traj = None
        self.pd = None
        core = getattr(roller.model, "module", roller.model)
        self._core = core
        self._fwd, self._spawn, self._step = core.forward, roller._spawn, roller.sim.step

        def fwd(*a, **k):
            out = self._fwd(*a, **k)
            self.traj = out.traj.detach().clone()
            return out

        def spawn(slots, carry):
            self.spawned.append(slots.detach().clone())
            return self._spawn(slots, carry)

        def step(*a, **k):
            # ``BatchedPDSim.step`` returns (q, v, integral) since the PID
            # change -- the spy must pass the third value through or the roller
            # unpacks two from a three-tuple.
            q, v, integral = self._step(*a, **k)
            self.pd = (q.detach().clone(), v.detach().clone())
            return q, v, integral

        core.forward, roller._spawn, roller.sim.step = fwd, spawn, step

    def restore(self):
        self.roller.model.forward = self._fwd
        self.roller._spawn = self._spawn
        self.roller.sim.step = self._step

    def respawned(self, slot: int) -> bool:
        return any(int(slot) in s.tolist() for s in self.spawned)

    @property
    def shifted(self) -> torch.Tensor:
        return torch.cat([self.traj[:, 1:], self.traj[:, -1:]], dim=1)


def _quiet(roller, slot: int) -> None:
    """Park a slot: both hands active, far from their hits, no futures."""
    st = roller.state
    st["active"][slot] = True
    st["goal_valid"][slot] = True
    st["hit"][slot] = HORIZON - 2
    st["goal_tth"][slot] = (HORIZON - 2) * roller.dt
    st["future_valid"][slot] = False
    st["age"][slot] = 0


def _armed(roller, slot: int, hand: int, *, tth: float = 0.4) -> int:
    """Arm ``hand`` of ``slot`` to fire on the next tick with a valid future.

    Returns the drum index the promoted future lands on.
    """
    st = roller.state
    other = 1 - hand
    _quiet(roller, slot)
    st["hit"][slot, hand] = 1
    st["goal_tth"][slot, hand] = roller.dt
    # a future the library can actually seed for this hand
    drum = next(
        d for d in range(roller.D)
        if bool(roller.lib_ok[hand, d]) and st["drum_valid"][slot, d]
    )
    st["future_valid"][slot, hand, 0] = True
    st["future_drum"][slot, hand, 0] = drum
    st["future_pos"][slot, hand, 0] = st["drum_pos"][slot, drum]
    st["future_vel"][slot, hand, 0] = -2.0 * st["drum_axis"][slot, drum]
    st["future_tth"][slot, hand, 0] = tth + roller.dt   # one tick of decay
    st["drum"][slot, other] = st["drum"][slot, other].clamp_min(0)
    return int(drum)


# ============================== 2. mid-flight switch, no state reset =========
def test_midflight_regoal_keeps_the_episode(bits):
    roller = make_roller(bits, slots=16, seed=11)
    slot, hand = 3, 1
    other = 1 - hand
    for s in range(roller.R):
        _quiet(roller, s)
    drum = _armed(roller, slot, hand, tth=0.40)

    st = roller.state
    old_drum = int(st["drum"][slot, hand])
    before = {k: v[slot].clone() for k, v in st.items()}

    spy = _Spy(roller)
    try:
        stats = roller.tick()
    finally:
        spy.restore()

    # --- the slot was NOT respawned and the plant carried through ------------
    assert not spy.respawned(slot), "a mid-flight re-goal must not respawn the slot"
    assert stats["regoal"] >= 1.0
    assert torch.equal(st["q0"][slot], spy.pd[0][slot])
    assert torch.equal(st["qd0"][slot], spy.pd[1][slot])
    assert torch.equal(st["kp"][slot], before["kp"])
    assert torch.equal(st["kd"][slot], before["kd"])
    assert int(st["age"][slot]) == int(before["age"]) + 1
    # same slot, same (randomized) kit
    assert torch.equal(st["drum_pos"][slot], before["drum_pos"])

    # --- the switched hand took the promoted goal ----------------------------
    assert int(st["hit"][slot, hand]) > 1
    assert int(st["hit"][slot, hand]) == min(
        max(int(round(0.40 / roller.dt)), 2), HORIZON
    )
    assert int(st["drum"][slot, hand]) == drum
    assert int(st["goal_drum"][slot, hand]) == drum
    assert bool(st["goal_valid"][slot, hand])
    assert int(st["start_drum"][slot, hand]) == old_drum
    assert float(st["start_shaft"][slot, hand]) == float(roller.shaft_tab[old_drum])
    # the promoted token is consumed
    assert not bool(st["future_valid"][slot, hand, 0])

    # --- and it is priced against THIS slot's own kit ------------------------
    tab = strike_geometry_table(
        st["drum_pos"][slot: slot + 1] - roller.pool._pelvis,
        st["drum_axis"][slot: slot + 1],
        st["drum_radius"][slot: slot + 1],
        roller.kit_names,
    )
    p_star = before["future_pos"][hand, 0] - roller.pool._pelvis
    assert torch.allclose(st["p_star"][slot, hand], p_star, atol=1e-6)
    assert abs(float(st["hit_radius"][slot, hand]) - float(tab["hit_radius"][0, drum])) < 1e-6
    off = tab["disk_center"][0, drum] - p_star
    ax = tab["axis"][0, drum]
    off = off - (off * ax).sum() * ax
    assert torch.allclose(st["disk_off"][slot, hand], off, atol=1e-6)
    assert not torch.equal(st["p_star"][slot, hand], before["p_star"][hand])

    # --- the OTHER hand is untouched -----------------------------------------
    ocols = _ARM[other]
    assert torch.equal(st["init"][slot][:, ocols], spy.shifted[slot][:, ocols])
    assert int(st["hit"][slot, other]) == int(before["hit"][other]) - 1
    for f in ("p_star", "v_star", "n_out", "hit_radius", "disk_off", "hp_dir",
              "has_hp", "sigma", "shaft", "start_shaft", "elev", "drum", "start_drum"):
        assert torch.equal(st[f][slot, other], before[f][other]), f"{f} moved on the idle hand"

    # --- the switched hand's own columns DID move ----------------------------
    hcols = _ARM[hand]
    assert not torch.equal(st["init"][slot][:, hcols], spy.shifted[slot][:, hcols])
    # ...and the seed starts at the measured pose and ends at the library pose
    ramp = st["init"][slot][:, hcols]
    yaw_bin = int(round((float(st["q0"][slot, roller.yaw_i]) - roller.yaw_lo)
                        / roller.yaw_step))
    yaw_bin = min(max(yaw_bin, 0), roller.n_bins - 1)
    term = roller.lib_q[hand, drum, yaw_bin]
    k = int(st["hit"][slot, hand])
    assert torch.allclose(ramp[k - 1], term, atol=1e-5)
    assert torch.allclose(ramp[0], st["q0"][slot][hcols], atol=5e-2)


# ================================================ 3. tth bookkeeping =========
def test_goal_tth_ticks_with_the_mask(bits):
    roller = make_roller(bits, slots=12, seed=5)
    slot = 0
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    tth0 = st["goal_tth"][slot].clone()
    hit0 = st["hit"][slot].clone()
    n = 5
    hold = _HoldModel(roller)
    try:
        for _ in range(n):
            stats = roller.tick()
            assert stats["regoal"] == 0.0 and stats["reset"] == 0.0, stats
    finally:
        hold.restore()

    got = st["goal_tth"][slot]
    assert torch.allclose(got, tth0 - n * roller.dt, atol=1e-5), (got, tth0)
    assert torch.equal(st["hit"][slot], hit0 - n)
    # the context and the mask's Fourier tau encodings tell the same story
    hit_s = st["hit"][slot].to(got.dtype) * roller.dt
    assert torch.all((got - hit_s).abs() <= roller.dt + 1e-6)


def test_future_tth_ticks_too(bits):
    roller = make_roller(bits, slots=8, seed=6)
    slot = 1
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    st["future_valid"][slot, 0, 0] = True
    st["future_tth"][slot, 0, 0] = 5.0          # far outside the horizon
    st["future_drum"][slot, 0, 0] = 0
    before = float(st["future_tth"][slot, 0, 0])
    invalid_before = st["future_tth"][slot, 1].clone()
    hold = _HoldModel(roller)
    try:
        for _ in range(4):
            roller.tick()
    finally:
        hold.restore()
    assert abs(float(st["future_tth"][slot, 0, 0]) - (before - 4 * roller.dt)) < 1e-5
    # invalid entries do not drift
    assert torch.equal(st["future_tth"][slot, 1], invalid_before)


def test_out_of_range_future_waits_instead_of_becoming_the_goal(bits):
    """``goal_snapshot``'s horizon cap, mirrored: a far hit stays in ``future``.

    Promoting it anyway would put a ``goal_tth`` of (say) 1.4 s next to a mask
    built from a horizon clamped at ``H`` -- the exact context/mask disagreement
    this fix exists to remove.  The hand idles and picks the hit up later.
    """
    roller = make_roller(bits, slots=8, seed=17, chain_frac=0.0)
    slot, hand = 3, 0
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    far = roller.H * roller.dt + 6 * roller.dt      # 6 ticks out of range
    drum = _armed(roller, slot, hand, tth=far - roller.dt)
    st["hit"][slot, 1 - hand] = HORIZON - 2         # keep the slot alive

    hold = _HoldModel(roller)
    spy = _Spy(roller)
    try:
        roller.tick()
        assert not spy.respawned(slot)
        assert int(st["hit"][slot, hand]) == -1, "far hit must not become the goal"
        assert not bool(st["active"][slot, hand])
        # the token is NOT consumed -- it is still waiting in future[0]
        assert bool(st["future_valid"][slot, hand, 0])
        assert int(st["future_drum"][slot, hand, 0]) == drum
        # ...and it is taken as soon as it ticks into range
        for _ in range(7):
            roller.tick()
    finally:
        spy.restore()
        hold.restore()
    assert bool(st["active"][slot, hand])
    assert int(st["hit"][slot, hand]) > 1
    assert int(st["drum"][slot, hand]) == drum
    assert float(st["goal_tth"][slot, hand]) <= roller.H * roller.dt + 1e-6


# ============================ 4. exhausted futures -> idle -> respawn ========
def test_exhausted_futures_idle_then_respawn(bits):
    roller = make_roller(bits, slots=8, seed=9, chain_frac=0.0)
    slot = 2
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    st["hit"][slot] = 1
    st["future_valid"][slot] = False
    st["age"][slot] = 17
    old_kit = st["drum_pos"][slot].clone()

    hold = _HoldModel(roller)
    spy = _Spy(roller)
    try:
        stats = roller.tick()
    finally:
        spy.restore()
        hold.restore()

    assert stats["idle"] >= 2.0, stats
    assert spy.respawned(slot)
    assert int(st["age"][slot]) == 0
    # a respawn draws a NEW scenario, hence (with probability ~1) a new kit
    assert not torch.equal(st["drum_pos"][slot], old_kit)


def test_one_hand_idle_keeps_the_slot_alive(bits):
    """Only the fired hand goes idle; the slot lives on for the other hand."""
    roller = make_roller(bits, slots=8, seed=13, chain_frac=0.0)
    slot = 4
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    st["hit"][slot, 0] = 1
    st["future_valid"][slot] = False
    before_other = st["hit"][slot, 1].clone()

    hold = _HoldModel(roller)
    spy = _Spy(roller)
    try:
        stats = roller.tick()
    finally:
        spy.restore()
        hold.restore()

    assert not spy.respawned(slot)
    assert stats["idle"] >= 1.0
    assert int(st["hit"][slot, 0]) == -1
    assert not bool(st["active"][slot, 0])
    assert not bool(st["goal_valid"][slot, 0])
    assert int(st["drum"][slot, 0]) == -1
    assert float(st["has_hp"][slot, 0]) == 0.0
    assert int(st["hit"][slot, 1]) == int(before_other) - 1
    assert bool(st["active"][slot, 1])


# =========================================== 5. divergence hard-respawns =====
def test_divergence_still_hard_respawns(bits):
    roller = make_roller(bits, slots=8, seed=21, chain_frac=1.0)
    slot = 5
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    st["q0"][slot] = roller.q_hi + 5.0
    st["prev_cmd"][slot] = st["q0"][slot].clone()

    hold = _HoldModel(roller)
    spy = _Spy(roller)
    try:
        stats = roller.tick()
    finally:
        spy.restore()
        hold.restore()

    assert stats["diverged"] >= 1.0
    assert spy.respawned(slot)
    assert int(st["age"][slot]) == 0
    assert torch.isfinite(st["q0"][slot]).all()
    # no carry: the fresh scenario's own start, inside the box
    assert bool((st["q0"][slot] <= roller.q_hi + 1e-4).all())
    assert bool((st["q0"][slot] >= roller.q_lo - 1e-4).all())


# ============================================ 6. harvest still collates ======
def test_harvest_collates_into_the_replay_pool(bits):
    pool = bits[0]
    roller = make_roller(bits, slots=32, seed=2)
    replay = ReplayPool(pool, 512, buckets=list(pool.buckets))
    for _ in range(10):
        for b, cols in roller.harvest().items():
            replay.add(b, cols)
        roller.tick()
    assert len(replay) > 0
    assert torch.isfinite(roller.state["q0"]).all()
    assert torch.isfinite(roller.state["init"]).all()

    b = pool.buckets[0]
    g = torch.Generator(device=pool.device)
    g.manual_seed(0)
    idx = replay.sample_idx(b, 8, generator=g)
    got = replay.batch(b, idx, generator=g, noise=False)
    ref = pool.batch(b, torch.arange(8, device=pool.device), generator=g, noise=False)
    for k in ("q0", "v0", "state", "init", "hit", "active"):
        assert got[k].shape == ref[k].shape, k
        assert got[k].dtype == ref[k].dtype, k
    for f in ("p_star", "hit_radius", "disk_offset", "shaft_allowed", "target_drum"):
        a, c = getattr(got["strike"], f), getattr(ref["strike"], f)
        assert a.shape == c.shape and a.dtype == c.dtype, f
    assert torch.isfinite(got["init"]).all()


# ======================================== 7. the A/B switch is a real A/B ====
def test_regoal_false_reproduces_respawn_on_hit(bits):
    roller = make_roller(bits, slots=8, seed=31, regoal=False, chain_frac=0.0)
    slot = 6
    for s in range(roller.R):
        _quiet(roller, s)
    st = roller.state
    st["hit"][slot] = 2
    # a perfectly promotable future -- the old path must ignore it
    _ = _armed(roller, slot, 1, tth=0.4)
    st["hit"][slot] = 2
    tth_before = st["goal_tth"][slot].clone()

    hold = _HoldModel(roller)
    spy = _Spy(roller)
    try:
        stats = roller.tick()
    finally:
        spy.restore()
        hold.restore()

    assert stats["regoal"] == 0.0 and stats["idle"] == 0.0
    assert spy.respawned(slot), "respawn-on-hit must still fire when hit reaches 1"
    # ...and the old path leaves the tth tokens alone (the bug this fix removes)
    assert not torch.allclose(st["goal_tth"][slot], tth_before - roller.dt)


def test_regoal_true_does_not_respawn_where_false_does(bits):
    """Same armed slot, the two branches disagree -- so the A/B is real."""
    roller = make_roller(bits, slots=8, seed=31, regoal=True, chain_frac=0.0)
    slot = 6
    for s in range(roller.R):
        _quiet(roller, s)
    _armed(roller, slot, 1, tth=0.4)
    roller.state["hit"][slot, 0] = 2

    hold = _HoldModel(roller)
    spy = _Spy(roller)
    try:
        roller.tick()
    finally:
        spy.restore()
        hold.restore()
    assert not spy.respawned(slot)
    assert int(roller.state["hit"][slot, 1]) > 1
