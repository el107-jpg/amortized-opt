"""Frontier collocation clouds: 64-chart seed, along-traj slices, TTH jitter."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.frontier import (
    N_CHART,
    FrontierCloud,
    FrontierConfig,
    apply_q_ball,
    ball_q_scale,
    chart_id,
    ingest_pool,
    jitter_hit,
    patch_spawn_q0,
    slice_at_knot,
    steal_q0,
)
from amortized_opt.warmstart_data import bucket_for

DT = CONTROL_DT
DEV = torch.device("cpu")
BUCKETS = (25, 50, 100)
IDLE_KIND, FAR_KIND = 4, 6
F = 2


def _cfg(**kw) -> FrontierConfig:
    return FrontierConfig(dt=DT, buckets=BUCKETS, **kw)


def _cloud(**kw) -> FrontierCloud:
    return FrontierCloud(_cfg(**kw), device=DEV)


def _cols(
    n: int,
    H: int,
    *,
    active,
    drum,
    hit=None,
    q0=None,
    future_empty: bool = False,
    future_gap=None,
    future_drum=None,
):
    n = int(n)
    H = int(H)
    act = torch.as_tensor(active, dtype=torch.bool)
    if act.dim() == 1:
        act = act.unsqueeze(0)
    if act.shape[0] == 1 and n > 1:
        act = act.expand(n, 2).clone()
    drm = torch.as_tensor(drum, dtype=torch.long)
    if drm.dim() == 1:
        drm = drm.unsqueeze(0)
    if drm.shape[0] == 1 and n > 1:
        drm = drm.expand(n, 2).clone()
    if hit is None:
        hit = torch.full((n, 2), H, dtype=torch.long)
        hit = torch.where(act, hit, torch.full_like(hit, -1))
    else:
        hit = torch.as_tensor(hit, dtype=torch.long)
        if hit.dim() == 1:
            hit = hit.unsqueeze(0)
        if hit.shape[0] == 1 and n > 1:
            hit = hit.expand(n, 2).clone()
    if q0 is None:
        q0 = torch.zeros(n, N_DOF)
    init = torch.zeros(n, H, N_DOF)
    # Distinctive path: knot t (1-indexed) has joint-0 = t, so q_full[t, 0] = t.
    init[:, :, 0] = torch.arange(1, H + 1, dtype=torch.float32).view(1, H)
    qd0 = torch.zeros(n, N_DOF)
    if future_empty:
        fg = torch.zeros(n, 2, F)
        fd = torch.zeros(n, 2, F, dtype=torch.long)
        fv = torch.zeros(n, 2, F, dtype=torch.bool)
    else:
        fg = torch.as_tensor(future_gap, dtype=torch.float32) if future_gap is not None else (
            torch.tensor([0.4, 0.8]).view(1, 1, F).expand(n, 2, F).clone()
        )
        fd = torch.as_tensor(future_drum, dtype=torch.long) if future_drum is not None else (
            torch.tensor([2, 5]).view(1, 1, F).expand(n, 2, F).clone()
        )
        fv = torch.ones(n, 2, F, dtype=torch.bool)
    gt = hit.clamp_min(0).to(torch.float32) * DT
    gt = torch.where(act, gt, torch.zeros_like(gt))
    return {
        "q0": q0,
        "qd0": qd0,
        "init": init,
        "hit": hit,
        "active": act,
        "drum": drm,
        "goal_tth": gt,
        "yaw_now": q0[:, 0].clone(),
        "future_gap": fg,
        "future_drum": fd,
        "future_valid": fv,
        "kind": torch.zeros(n, dtype=torch.long),
    }


def _pool(data: dict[int, dict], q_lo=None, q_hi=None):
    return SimpleNamespace(
        data=data,
        buckets=tuple(sorted(data)),
        q_lo=q_lo if q_lo is not None else torch.full((N_DOF,), -2.0),
        q_hi=q_hi if q_hi is not None else torch.full((N_DOF,), 2.0),
    )


def _live(cloud: FrontierCloud, bucket: int) -> dict[str, torch.Tensor]:
    m = cloud.size(bucket)
    if m == 0:
        return {}
    return cloud.gather(bucket, torch.arange(m))


# -------------------------------------------------------------- chart id --
def test_chart_occupancy_idle_one_hand_dual():
    idle = chart_id(
        torch.tensor([[False, False]]), torch.tensor([[3, 4]]),
    )
    assert int(idle[0]) == 0

    left = chart_id(
        torch.tensor([[True, False]]), torch.tensor([[3, 0]]),
    )
    assert int(left[0]) == (3 + 1) * 8 + 0

    right = chart_id(
        torch.tensor([[False, True]]), torch.tensor([[0, 2]]),
    )
    assert int(right[0]) == 0 * 8 + (2 + 1)

    dual = chart_id(
        torch.tensor([[True, True]]), torch.tensor([[0, 6]]),
    )
    assert int(dual[0]) == (0 + 1) * 8 + (6 + 1)

    cloud = _cloud(chart_cap=8, traj_stride=10_000)
    H = 25
    ingest_pool(cloud, _pool({
        H: _cols(1, H, active=[False, False], drum=[0, 0], hit=[-1, -1],
                 q0=torch.zeros(1, N_DOF)),
    }))
    ingest_pool(cloud, _pool({
        H: _cols(1, H, active=[True, False], drum=[3, 0], hit=[20, -1],
                 q0=torch.ones(1, N_DOF)),
    }))
    ingest_pool(cloud, _pool({
        H: _cols(1, H, active=[False, True], drum=[0, 2], hit=[-1, 20],
                 q0=torch.full((1, N_DOF), 2.0)),
    }))
    ingest_pool(cloud, _pool({
        H: _cols(1, H, active=[True, True], drum=[0, 6], hit=[20, 20],
                 q0=torch.full((1, N_DOF), 3.0)),
    }))
    occ = cloud.occupancy_of(H)
    assert occ.shape == (N_CHART,)
    assert int(occ[0]) == 1
    assert int(occ[int(left[0])]) == 1
    assert int(occ[int(right[0])]) == 1
    assert int(occ[int(dual[0])]) == 1
    assert int(occ.sum()) == 4


def test_frontier_disabled_by_default():
    assert FrontierConfig().enabled is False
    assert FrontierConfig().traj_stride == 4
    assert FrontierConfig().chart_cap == 2048
    assert FrontierConfig().ema_alpha == 0.25
    assert FrontierConfig().ema_nmin == 2


# ---------------------------------------------------------- seed ingest --
def test_ingest_pool_keeps_all_k0_rows_beyond_chart_cap():
    """Seed grows; chart_cap * 64 must not wrap away pool starts."""
    chart_cap = 2
    child_cap = chart_cap * N_CHART  # 128
    n = child_cap + 72  # would wrap under the old global cap
    H = 25
    q0 = torch.zeros(n, N_DOF)
    q0[:, 0] = torch.arange(n, dtype=torch.float32)
    cols = _cols(
        n, H, active=[True, False], drum=[1, 0], hit=[8, -1], q0=q0,
    )
    cloud = _cloud(chart_cap=chart_cap, traj_stride=10_000)
    added = ingest_pool(cloud, _pool({H: cols}))
    assert added[H] == n
    assert cloud.size(H) == n
    live = _live(cloud, H)
    assert bool(live["from_pool"].all())
    got = live["q0"][:, 0].sort().values
    assert torch.equal(got, q0[:, 0])
    # children wrap in a later region; seed q0 stays
    child = _cols(
        child_cap + 40, H, active=[True, False], drum=[1, 0], hit=[8, -1],
        q0=torch.full((child_cap + 40, N_DOF), 99.0),
    )
    cloud.ingest(H, child, from_pool=False)
    live = _live(cloud, H)
    seed = live["from_pool"]
    assert int(seed.sum()) == n
    seed_q = live["q0"][seed, 0].sort().values
    assert torch.equal(seed_q, q0[:, 0])
    assert not bool((live["q0"][seed] == 99.0).all())


# ---------------------------------------------------------- along-traj --
def test_ingest_pool_stride_slices_land_in_shorter_bucket():
    cloud = _cloud(chart_cap=8, traj_stride=4)
    cols = _cols(1, 50, active=[True, True], drum=[0, 1], hit=[50, 50])
    ingest_pool(cloud, _pool({50: cols}))
    live50 = _live(cloud, 50)
    assert bool((live50["traj_k"] == 0).any())
    assert cloud.size(25) > 0
    live25 = _live(cloud, 25)
    assert bool((live25["traj_k"] > 0).all())
    assert torch.equal(
        live25["future_gap"], cols["future_gap"].expand_as(live25["future_gap"]),
    )


def test_slice_dual_k31_goes_to_bucket_25_and_keeps_future():
    H = 50
    fg = torch.tensor([[[0.55, 1.10], [0.55, 1.10]]])
    fd = torch.tensor([[[3, 1], [3, 1]]], dtype=torch.long)
    cols = _cols(
        1, H, active=[True, True], drum=[0, 1], hit=[50, 50],
        future_gap=fg, future_drum=fd,
    )
    split = slice_at_knot(cols, 31, dt=DT, buckets=BUCKETS)
    assert set(split) == {25}
    sub = split[25]
    assert torch.equal(sub["hit"], torch.tensor([[19, 19]]))
    assert bucket_for(19, BUCKETS) == 25
    assert torch.allclose(sub["goal_tth"], torch.full((1, 2), 19 * DT))
    assert torch.equal(sub["future_gap"], fg)
    assert torch.equal(sub["future_drum"], fd)
    assert torch.equal(sub["drum"], cols["drum"])
    assert torch.equal(sub["active"], cols["active"])
    # suffix of q_full[32:] = init[31:] = 32..50 on joint 0, then hold to 25
    path = sub["init"][0, :, 0]
    assert path.shape == (25,)
    assert torch.allclose(path[:19], torch.arange(32, 51, dtype=torch.float32))
    assert torch.allclose(path[19:], torch.full((6,), 50.0))
    assert torch.allclose(sub["q0"][0, 0], torch.tensor(31.0))
    # qd0 from q_full[32] - q_full[31]
    assert torch.allclose(sub["qd0"][0, 0], torch.tensor(1.0 / DT))


def test_slice_skips_k_that_would_make_hit_below_one():
    H = 50
    cols = _cols(1, H, active=[True, True], drum=[0, 1], hit=[50, 50])
    assert slice_at_knot(cols, 50, dt=DT, buckets=BUCKETS) == {}
    assert slice_at_knot(cols, 51, dt=DT, buckets=BUCKETS) == {}
    kept = slice_at_knot(cols, 49, dt=DT, buckets=BUCKETS)
    assert 25 in kept
    assert torch.equal(kept[25]["hit"], torch.tensor([[1, 1]]))

    one = _cols(1, H, active=[True, False], drum=[2, 0], hit=[10, -1])
    assert slice_at_knot(one, 10, dt=DT, buckets=BUCKETS) == {}
    # dual with mismatched TTH: right would go to 0
    mismatch = _cols(1, H, active=[True, True], drum=[0, 1], hit=[50, 40])
    assert slice_at_knot(mismatch, 40, dt=DT, buckets=BUCKETS) == {}


def test_jitter_rejects_bucket_change_slice_does_not():
    # Parent lives in bucket 50 (hit=30). Jitter down into 1..25 must drop.
    hit = torch.tensor([[30, -1]])
    act = torch.tensor([[True, False]])
    g = torch.Generator().manual_seed(0)
    hit2, keep = jitter_hit(
        hit, act, dt=DT, bucket=50, sigma_t=1.0, generator=g, buckets=BUCKETS,
    )
    k_new = torch.where(act, hit2.clamp_min(0), torch.zeros_like(hit2)).max(1).values
    for i in range(hit.shape[0]):
        if bool(keep[i]):
            assert bucket_for(int(k_new[i]), BUCKETS) == 50
        else:
            assert int(k_new[i]) < 1 or bucket_for(int(k_new[i]), BUCKETS) != 50
    # Large noise on a batch almost surely produces a reject.
    hit_b = torch.full((64, 2), 30, dtype=torch.long)
    hit_b[:, 1] = -1
    act_b = torch.zeros(64, 2, dtype=torch.bool)
    act_b[:, 0] = True
    _, keep_b = jitter_hit(
        hit_b, act_b, dt=DT, bucket=50, sigma_t=1.0,
        generator=torch.Generator().manual_seed(1), buckets=BUCKETS,
    )
    assert not bool(keep_b.all()), "jitter must reject some bucket-crossing TTH"

    cols = _cols(1, 50, active=[True, False], drum=[1, 0], hit=[30, -1])
    split = slice_at_knot(cols, 10, dt=DT, buckets=BUCKETS)
    # hit' = 20 → bucket 25, and that is kept
    assert 25 in split
    assert torch.equal(split[25]["hit"][:, 0], torch.tensor([20]))


# -------------------------------------------------------------- jitter --
def test_dual_jitter_only_left_tth():
    hit = torch.tensor([[20, 40], [15, 35]])
    act = torch.ones(2, 2, dtype=torch.bool)
    hit2, _keep = jitter_hit(
        hit, act, dt=DT, bucket=50, sigma_t=0.2,
        generator=torch.Generator().manual_seed(2), buckets=BUCKETS,
    )
    assert torch.equal(hit2[:, 1], hit[:, 1])
    # right-only still jitters the right clock
    right = torch.tensor([[-1, 30]])
    ract = torch.tensor([[False, True]])
    r2, _ = jitter_hit(
        right, ract, dt=DT, bucket=50, sigma_t=0.3,
        generator=torch.Generator().manual_seed(3), buckets=BUCKETS,
    )
    # not asserting inequality (noise can be tiny); just that left stayed
    assert torch.equal(r2[:, 0], right[:, 0])

    cloud = _cloud(chart_cap=8, sigma_tth=0.15, sigma_q=0.0, traj_stride=10_000)
    parent = _cols(
        4, 50, active=[True, True], drum=[0, 1], hit=[40, 40],
        q0=torch.zeros(4, N_DOF),
        future_gap=torch.tensor([0.4, 0.8]).view(1, 1, F).expand(4, 2, F).clone(),
        future_drum=torch.tensor([2, 5]).view(1, 1, F).expand(4, 2, F).clone(),
    )
    cloud.ingest(50, parent, from_pool=True)
    child = cloud.propose(50, 4, expand=True, generator=torch.Generator().manual_seed(4))
    assert child is not None
    n_c = child["q0"].shape[0]
    assert torch.equal(child["future_gap"], parent["future_gap"][:n_c])
    assert torch.equal(child["future_drum"], parent["future_drum"][:n_c])
    assert torch.equal(child["hit"][:, 1], torch.full((n_c,), 40))


def test_child_future_copied_goal_tth_is_hit_dt():
    cloud = _cloud(chart_cap=8, sigma_tth=0.0, sigma_q=0.0, traj_stride=10_000)
    parent = _cols(
        3, 25, active=[True, False], drum=[2, 0], hit=[20, -1],
        q0=torch.zeros(3, N_DOF),
    )
    cloud.ingest(25, parent, from_pool=True)
    child = cloud.propose(25, 3, expand=True)
    assert child is not None
    assert torch.equal(child["future_gap"], parent["future_gap"][: child["q0"].shape[0]])
    assert torch.equal(child["future_drum"], parent["future_drum"][: child["q0"].shape[0]])
    act = child["active"]
    want = child["hit"].clamp_min(0).to(child["goal_tth"].dtype) * DT
    assert torch.allclose(child["goal_tth"][act], want[act])


def test_far_fetched_empty_future_stays_empty():
    cloud = _cloud(chart_cap=8, sigma_tth=0.0, sigma_q=0.0, traj_stride=10_000)
    parent = _cols(
        2, 25, active=[True, False], drum=[4, 0], hit=[18, -1],
        future_empty=True, q0=torch.zeros(2, N_DOF),
    )
    cloud.ingest(25, parent, from_pool=True)
    child = cloud.propose(25, 2, expand=True)
    assert child is not None
    assert torch.equal(child["future_gap"], torch.zeros_like(child["future_gap"]))
    assert torch.equal(child["future_drum"], torch.zeros_like(child["future_drum"]))
    assert not bool(child["future_valid"].any())


# -------------------------------------------------------- steal / spawn --
def test_patch_spawn_replaces_q0_keeps_drawn_clocks():
    cloud = _cloud(chart_cap=8, traj_stride=10_000)
    parent = _cols(
        2, 25, active=[True, False], drum=[1, 0], hit=[12, -1],
        q0=torch.full((2, N_DOF), 0.7),
    )
    cloud.ingest(25, parent, from_pool=True)
    drawn = _cols(
        3, 25, active=[True, False], drum=[1, 0], hit=[9, -1],
        q0=torch.full((3, N_DOF), -0.3),
        future_gap=torch.full((3, 2, F), 0.9),
    )
    hit0 = drawn["hit"].clone()
    gt0 = drawn["goal_tth"].clone()
    fg0 = drawn["future_gap"].clone()
    mask = patch_spawn_q0(drawn, cloud, frac=1.0)
    assert bool(mask.all())
    assert torch.allclose(drawn["q0"], torch.full((3, N_DOF), 0.7))
    assert torch.equal(drawn["hit"], hit0)
    assert torch.equal(drawn["goal_tth"], gt0)
    assert torch.equal(drawn["future_gap"], fg0)


def test_steal_empty_chart_leaves_q0():
    cloud = _cloud(chart_cap=8, traj_stride=10_000)
    idle = _cols(
        2, 25, active=[False, False], drum=[0, 0], hit=[-1, -1],
        q0=torch.zeros(2, N_DOF),
    )
    cloud.ingest(25, idle, from_pool=True)
    cols = _cols(
        3, 25, active=[True, True], drum=[0, 1], hit=[20, 20],
        q0=torch.full((3, N_DOF), 1.25),
    )
    q_before = cols["q0"].clone()
    cid = steal_q0(
        cols, cloud, 25, frac=1.0,
        kind=torch.full((3,), IDLE_KIND, dtype=torch.long),
        idle_kind=IDLE_KIND, far_kind=FAR_KIND,
    )
    assert torch.equal(cid, torch.zeros(3, dtype=torch.long))
    assert torch.equal(cols["q0"], q_before)
    assert cloud.n_in_chart(25, int(chart_id(cols["active"], cols["drum"])[0])) == 0


def test_sample_idx_without_replacement():
    cloud = _cloud(chart_cap=8, traj_stride=10_000)
    cols = _cols(
        10, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(10, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    g = torch.Generator().manual_seed(0)
    loc = cloud.sample_idx(25, 10, generator=g)
    assert loc.numel() == 10
    assert torch.equal(loc.sort().values, torch.arange(10))
    loc = cloud.sample_idx(25, 4, generator=g)
    assert loc.numel() == 4
    assert int(loc.unique().numel()) == 4
    assert cloud.sample_idx(25, 0).numel() == 0
    among = torch.tensor([1, 3, 5, 7, 9])
    loc = cloud.sample_idx(25, 5, among=among)
    assert torch.equal(loc.sort().values, among)
    loc = cloud.sample_idx(25, 2, among=among, generator=g)
    assert loc.numel() == 2
    assert set(loc.tolist()) <= set(among.tolist())
    assert cloud.scored_loc(25).numel() == 0


def test_ema_update_by_cid():
    cloud = _cloud(chart_cap=8, ema_alpha=0.5, traj_stride=10_000)
    cols = _cols(
        2, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(2, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    live = _live(cloud, 25)
    cid = live["cid"][:1]
    cloud.ema_update(25, cid, torch.tensor([1.0]))
    cloud.ema_update(25, cid, torch.tensor([0.0]))
    live = _live(cloud, 25)
    # m0=0.5 → 0.5*0.5+0.5*1=0.75 → 0.5*0.75+0.5*0=0.375
    assert abs(float(live["ema_m"][0]) - 0.375) < 1e-5
    assert int(live["ema_n"][0]) == 2
    assert int(live["ema_n"][1]) == 0


def test_parent_weights_prefer_ema_hits():
    cloud = _cloud(chart_cap=8, ema_alpha=1.0, ema_nmin=1, traj_stride=10_000)
    cols = _cols(
        2, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(2, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    live = _live(cloud, 25)
    cloud.ema_update(25, live["cid"][:1], torch.tensor([1.0]))
    cloud.ema_update(25, live["cid"][1:], torch.tensor([0.0]))
    w = cloud._parent_weights(25, torch.arange(2))
    assert float(w[0]) > float(w[1])


def test_parent_weights_stay_positive_when_seed_is_old():
    cloud = _cloud(chart_cap=8, traj_stride=10_000)
    cols = _cols(
        2, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(2, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    cloud._pos[25] = 10_000_000
    w = cloud._parent_weights(25, torch.arange(2))
    assert torch.isfinite(w).all()
    assert float(w.sum()) > 0.0


def test_two_hits_clear_ema_top():
    """alpha=0.25: two hits from 0.5 prior exceed 0.70; three reach ~0.79."""
    cloud = _cloud(chart_cap=8, ema_alpha=0.25, ema_nmin=2, traj_stride=10_000)
    cols = _cols(
        1, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(1, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    cid = _live(cloud, 25)["cid"][:1]
    cloud.ema_update(25, cid, torch.tensor([1.0]))
    m1 = float(_live(cloud, 25)["ema_m"][0])
    assert abs(m1 - 0.625) < 1e-5
    cloud.ema_update(25, cid, torch.tensor([1.0]))
    m2 = float(_live(cloud, 25)["ema_m"][0])
    assert m2 > 0.70
    assert abs(m2 - 0.71875) < 1e-5
    cloud.ema_update(25, cid, torch.tensor([1.0]))
    m3 = float(_live(cloud, 25)["ema_m"][0])
    assert abs(m3 - 0.7890625) < 1e-5


def test_ema_nmin_holds_prior_until_trusted():
    cloud = _cloud(chart_cap=8, ema_alpha=0.25, ema_nmin=2, traj_stride=10_000)
    cols = _cols(
        2, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(2, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    d = cloud._buf[25]
    d["recency"][:2] = 0
    d["n_children"][:2] = 0
    live = _live(cloud, 25)
    cloud.ema_update(25, live["cid"][:1], torch.tensor([1.0]))
    w = cloud._parent_weights(25, torch.arange(2))
    assert abs(float(w[0]) - float(w[1])) < 1e-5
    cloud.ema_update(25, live["cid"][:1], torch.tensor([1.0]))
    w2 = cloud._parent_weights(25, torch.arange(2))
    assert float(w2[0]) > float(w2[1])


def test_child_ring_grows_when_chart_cap_raised():
    """Resume with a bigger chart_cap must append, not wrap into holes."""
    H = 25
    cloud = _cloud(chart_cap=2, traj_stride=10_000)
    seed = _cols(
        4, H, active=[True, False], drum=[1, 0], hit=[8, -1],
        q0=torch.zeros(4, N_DOF),
    )
    cloud.ingest(H, seed, from_pool=True)
    old_cap = 2 * N_CHART
    child = _cols(
        old_cap, H, active=[True, False], drum=[1, 0], hit=[8, -1],
        q0=torch.full((old_cap, N_DOF), 3.0),
    )
    cloud.ingest(H, child, from_pool=False)
    assert int(cloud._child_alloc[H]) == old_cap
    assert cloud.size(H) == 4 + old_cap
    cloud.cfg.chart_cap = 8
    extra = _cols(
        10, H, active=[True, False], drum=[1, 0], hit=[8, -1],
        q0=torch.full((10, N_DOF), 7.0),
    )
    cloud.ingest(H, extra, from_pool=False)
    assert int(cloud._child_alloc[H]) == 8 * N_CHART
    live = _live(cloud, H)
    seed_ok = live["from_pool"]
    assert int(seed_ok.sum()) == 4
    kids = live["q0"][~seed_ok, 0]
    assert int((kids == 3.0).sum()) == old_cap
    assert int((kids == 7.0).sum()) == 10


def test_landmarks_skip_near_duplicates():
    wr = G1_UPPER_BODY_JOINTS.index("left_wrist_pitch_joint")
    cloud = _cloud(
        chart_cap=8, ema_alpha=1.0, ema_nmin=1, landmark_k=8,
        landmark_tau=2.0, traj_stride=10_000,
    )
    q0 = torch.zeros(3, N_DOF)
    q0[1, wr] = 1.0
    q0[2, wr] = 0.02
    cols = _cols(3, 25, active=[True, False], drum=[0, 0], hit=[10, -1], q0=q0)
    cloud.ingest(25, cols, from_pool=True)
    cid = _live(cloud, 25)["cid"]
    cloud.ema_update(25, cid, torch.ones(3))
    assert cloud.n_landmarks() == 2
    key = (25, int(_live(cloud, 25)["chart"][0]))
    assert cloud._landmarks[key].shape[0] == 2


def test_propose_rejects_children_inside_tau():
    cloud = _cloud(
        chart_cap=8, ema_alpha=1.0, ema_nmin=1, landmark_k=4,
        landmark_tau=8.0, sigma_tth=0.0, traj_stride=10_000,
    )
    cols = _cols(
        1, 25, active=[True, False], drum=[0, 0], hit=[10, -1],
        q0=torch.zeros(1, N_DOF),
    )
    cloud.ingest(25, cols, from_pool=True)
    cloud.ema_update(25, _live(cloud, 25)["cid"], torch.ones(1))
    assert cloud.n_landmarks() == 1
    g = torch.Generator().manual_seed(0)
    child = cloud.propose(25, 8, expand=True, generator=g)
    if child is not None:
        dmin = cloud._min_dist_to_landmarks(25, child["chart"], child["q0"])
        keep = ~torch.isfinite(dmin) | (dmin >= 8.0 - 1e-4)
        assert bool(keep.all())


def test_parent_pool_prefers_far_from_landmarks():
    wr = G1_UPPER_BODY_JOINTS.index("left_wrist_pitch_joint")
    cloud = _cloud(
        chart_cap=8, ema_alpha=1.0, ema_nmin=2, hull_ema=0.55,
        landmark_k=4, landmark_tau=2.0, propose_candidates=64,
        traj_stride=10_000,
    )
    q0 = torch.zeros(2, N_DOF)
    q0[1, wr] = 1.0
    cols = _cols(2, 25, active=[True, False], drum=[0, 0], hit=[10, -1], q0=q0)
    cloud.ingest(25, cols, from_pool=True)
    d = cloud._buf[25]
    d["recency"][:2] = 0
    d["n_children"][:2] = 0
    cloud.ema_update(25, _live(cloud, 25)["cid"][:1], torch.ones(1))
    cloud.ema_update(25, _live(cloud, 25)["cid"][:1], torch.ones(1))
    cloud.ema_update(25, _live(cloud, 25)["cid"][1:], torch.ones(1))
    hits, w = cloud._chart_parent_pool(25, int(d["chart"][0]), None)
    # index 1 is far from the first landmark
    wmap = {int(hits[i]): float(w[i]) for i in range(hits.numel())}
    assert wmap[1] > wmap[0]


def test_sample_idx_inv_frac_oversamples_far():
    wr = G1_UPPER_BODY_JOINTS.index("left_wrist_pitch_joint")
    cloud = _cloud(
        chart_cap=8, ema_alpha=1.0, ema_nmin=1, landmark_k=4,
        landmark_tau=2.0, train_inv_pool=32, traj_stride=10_000,
    )
    n = 24
    q0 = torch.zeros(n, N_DOF)
    q0[20:, wr] = 1.0
    cols = _cols(n, 25, active=[True, False], drum=[0, 0], hit=[10, -1], q0=q0)
    cloud.ingest(25, cols, from_pool=True)
    cloud.ema_update(25, _live(cloud, 25)["cid"], torch.ones(n))
    g = torch.Generator().manual_seed(1)
    far = 0
    for _ in range(40):
        loc = cloud.sample_idx(25, 6, generator=g, inv_frac=1.0)
        far += int((loc >= 20).sum())
    # uniform would be 6*(4/24)*40 = 40; inverse-density should beat that
    assert far > 40


def test_ball_waist_and_shoulder_scaled_down():
    yaw = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
    sh = G1_UPPER_BODY_JOINTS.index("left_shoulder_pitch_joint")
    wr = G1_UPPER_BODY_JOINTS.index("left_wrist_pitch_joint")
    w = ball_q_scale()
    assert float(w[yaw]) < float(w[sh]) < float(w[wr])
    g = torch.Generator().manual_seed(0)
    q0 = torch.zeros(1024, N_DOF)
    qd0 = torch.zeros(1024, N_DOF)
    lo = torch.full((N_DOF,), -4.0)
    hi = torch.full((N_DOF,), 4.0)
    q2, qd2 = apply_q_ball(
        q0, qd0, sigma_q=0.05, sigma_qd=0.10, q_lo=lo, q_hi=hi, generator=g,
    )
    sy, ss, sw = float(q2[:, yaw].std()), float(q2[:, sh].std()), float(q2[:, wr].std())
    assert sy < ss < sw
    assert abs(sy / sw - 0.25) < 0.08
    assert float(qd2[:, yaw].std()) < float(qd2[:, wr].std())


def test_tth_jitter_is_fraction_of_remaining():
    """Same σ: 1 s remaining moves more in seconds than 0.2 s remaining."""
    act = torch.ones(256, 2, dtype=torch.bool)
    act[:, 1] = False
    long_h = torch.full((256, 2), 50, dtype=torch.long)
    long_h[:, 1] = -1
    short_h = torch.full((256, 2), 10, dtype=torch.long)
    short_h[:, 1] = -1
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    h1, _ = jitter_hit(long_h, act, dt=DT, bucket=50, sigma_t=0.2, generator=g1, buckets=BUCKETS)
    h2, _ = jitter_hit(short_h, act, dt=DT, bucket=25, sigma_t=0.2, generator=g2, buckets=BUCKETS)
    d1 = ((h1[:, 0] - 50).to(torch.float32) * DT).abs().mean()
    d2 = ((h2[:, 0] - 10).to(torch.float32) * DT).abs().mean()
    assert float(d1) > float(d2) * 1.5

