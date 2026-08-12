"""Sanity tests for the phase-1 APEX warm-start variant (``warmstart_apex``).

These are the pre-training gates asked for by the experiment brief:

1. the terminal waypoint equals the chosen terminal pose EXACTLY, in both the
   apex target and the straight-line init that accompanies it;
2. the emitted path visits the apex cylinder (fraction + apex-height stats);
3. the path spline is C1/C2-continuous across the apex knot;
4. masks / boundary conditions are format-identical to the straight-line pool,
   and one batch feeds through the model forward;
5. the input init really is the straight-line quintic to the same terminal
   (Eric's 2026-08-10 clarification), not a noised apex;
6. the apex target is not *worse* than the straight line on path clearance --
   the whole point of the variant.

Run with ``python -m pytest tests/test_warmstart_apex.py -q -s`` (``-s`` shows
the measured distributions, which are the report the brief asks for).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.spline import retime_joint_path
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice
from amortized_opt.warmstart_apex import (
    ApexConfig,
    ApexSolver,
    ApexWarmstartPool,
    generate_apex_scenarios,
    path_derivative_jump,
    select_terminal,
)
from amortized_opt.warmstart_data import ScenarioConfig, WarmstartPool, generate_scenarios

HANDS = ("left", "right")
N_SCEN = 250


@pytest.fixture(scope="module")
def bundle():
    torch.set_num_threads(8)
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)
    cfg = ScenarioConfig()
    acfg = ApexConfig()
    dbg: list = []
    out = generate_apex_scenarios(
        N_SCEN, clips, seed=3, cfg=cfg, apex_cfg=acfg, kit=kit, debug_out=dbg
    )
    return {"kit": kit, "clips": clips, "cfg": cfg, "acfg": acfg,
            "out": out, "dbg": dbg, "library": GoalPoseLibrary()}


# ------------------------------------------------------------------ 1. terminal
def test_terminal_waypoint_is_exact(bundle):
    """Row ``k_hit`` of both the target and the init IS the terminal pose."""
    worst_t = worst_i = 0.0
    n = 0
    for d in bundle["dbg"]:
        ws = d["ws"]
        for hand in HANDS:
            k = ws.hit_index[hand]
            if k < 0:
                continue
            idx = arm_slice(hand)
            term = ws.terminal_arm[hand]
            worst_t = max(worst_t, float(np.abs(ws.target[k - 1, idx] - term).max()))
            worst_i = max(worst_i, float(np.abs(ws.init[k - 1, idx] - term).max()))
            n += 1
    print(f"\n[terminal] {n} hands: max|target-term|={worst_t:.2e} "
          f"max|init-term|={worst_i:.2e} rad")
    assert n > 100
    assert worst_t < 1e-12
    assert worst_i < 1e-12


# --------------------------------------------------------------- 2. cylinder
def test_path_visits_the_apex_cylinder(bundle):
    acfg = bundle["acfg"]
    fk = A.ChainFK(dtype=torch.float64)
    qs, hands, centers, dz_apex, outs = [], [], [], [], []
    for d in bundle["dbg"]:
        ws = d["ws"]
        for hand in HANDS:
            k = ws.hit_index[hand]
            if k < 0 or hand not in ws.apex_arm:
                continue
            qs.append(ws.q[: k + 1])
            hands.append(hand)
            centers.append(np.asarray(d["aim"][hand], float)
                           + np.array([0.0, 0.0, acfg.height]))
            a = d["apex"][hand]
            dz_apex.append(a["dz"])
            outs.append(a["outside"])
    assert len(qs) > 100
    K = max(x.shape[0] for x in qs)
    Q = np.stack([np.concatenate([x, np.repeat(x[-1:], K - x.shape[0], 0)]) for x in qs])
    rep = A_cyl(Q, hands, np.stack(centers), fk=fk, cfg=acfg)

    outs = np.asarray(outs)
    dz = np.asarray(dz_apex)
    print(f"\n[cylinder] apex POSE: outside p50={np.percentile(outs, 50)*1000:.2f} mm "
          f"p90={np.percentile(outs, 90)*1000:.1f} mm; "
          f"<=1 mm {float((outs <= 1e-3).mean()):.3f}, <=1 cm {float((outs <= 1e-2).mean()):.3f}")
    print(f"[cylinder] apex tip height above the strike: p10={np.percentile(dz, 10):.3f} "
          f"p50={np.percentile(dz, 50):.3f} p90={np.percentile(dz, 90):.3f} m "
          f"(target {acfg.height:.2f} +/- {acfg.half_height:.2f})")
    print(f"[cylinder] emitted PATH passes inside: {rep['inside_frac']:.3f}; "
          f"closest-approach miss p50={np.percentile(rep['min_outside_m'], 50)*1000:.2f} mm, "
          f"p90={np.percentile(rep['min_outside_m'], 90)*1000:.1f} mm; "
          f"height at closest approach p50={np.percentile(rep['apex_height_m'], 50):.3f} m")
    assert float((outs <= 1e-2).mean()) > 0.80
    assert rep["inside_frac"] > 0.50
    assert 0.15 < float(np.median(rep["apex_height_m"])) < 0.45


def A_cyl(Q, hands, centers, *, fk, cfg):
    from amortized_opt.warmstart_apex import cylinder_report

    return cylinder_report(Q, hands, centers, fk=fk, cfg=cfg)


# ------------------------------------------------------------- 3. continuity
def test_apex_knot_is_smooth(bundle):
    """No corner at the apex: the path spline is one polynomial piece."""
    big = small = 0.0
    j2 = 0.0
    n = 0
    for d in bundle["dbg"][:120]:
        ws = d["ws"]
        for hand in HANDS:
            if hand not in ws.apex_arm:
                continue
            idx = arm_slice(hand)
            Q = np.stack([ws.q0[idx], ws.apex_arm[hand], ws.terminal_arm[hand]])
            big = max(big, path_derivative_jump(Q, 1, eps=1e-5))
            small = max(small, path_derivative_jump(Q, 1, eps=1e-8))
            j2 = max(j2, path_derivative_jump(Q, 2, eps=1e-8))
            n += 1
    # A genuine corner gives an eps-INDEPENDENT jump; a smooth path gives one
    # proportional to eps.  Both numbers together are the discrimination.
    print(f"\n[smooth] {n} apex knots: max |dq/du| jump  eps=1e-5 -> {big:.2e}, "
          f"eps=1e-8 -> {small:.2e} (ratio {big / max(small, 1e-30):.0f}, "
          f"1000 == perfectly smooth); max |d2q/du2| jump = {j2:.2e}")
    assert n > 50
    assert small < 1e-6
    assert j2 < 1e-6


def test_target_acceleration_has_no_spike_at_the_apex(bundle):
    """The retimed target's 2nd difference at the apex index is unexceptional."""
    ratios = []
    for d in bundle["dbg"]:
        ws = d["ws"]
        for hand in HANDS:
            k = ws.hit_index[hand]
            if k < 4 or hand not in ws.apex_wp:
                continue
            idx = arm_slice(hand)
            X = ws.q[: k + 1, idx]
            acc = np.abs(np.diff(X, n=2, axis=0)).max(axis=1) / (ws.dt ** 2)
            ka = int(np.clip(ws.apex_wp[hand] - 1, 0, acc.size - 1))
            med = float(np.median(acc)) + 1e-9
            ratios.append(float(acc[ka]) / med)
    r = np.asarray(ratios)
    print(f"\n[smooth] |q''| at the apex index / median over the path: "
          f"p50={np.percentile(r, 50):.2f} p90={np.percentile(r, 90):.2f} max={r.max():.2f}")
    assert np.percentile(r, 90) < 6.0


# ------------------------------------------------------------ 4. pool format
def test_pool_format_matches_the_straight_line_pool(bundle):
    cfg = bundle["cfg"]
    kit, clips = bundle["kit"], bundle["clips"]
    dev = "cpu"
    ref_raw = generate_scenarios(160, clips, seed=11, cfg=cfg, kit=kit)
    ref = WarmstartPool(ref_raw["buckets"], cfg=cfg, device=dev)
    ap = ApexWarmstartPool(bundle["out"]["buckets"], cfg=cfg, device=dev)

    b = min(set(ref.buckets) & set(ap.buckets))
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    nb = min(8, ref.size(b), ap.size(b))
    ix = torch.arange(nb)
    rb = ref.batch(b, ix, generator=g)
    ab = ap.batch(b, ix, generator=g)

    assert set(rb) - {"init_base"} == set(ab) - {"init_base"}
    for k in ("state", "q0", "v0", "init", "target", "hit", "active", "drum"):
        assert rb[k].shape[1:] == ab[k].shape[1:], k
        assert rb[k].dtype == ab[k].dtype, k
    for k in ("elem_mask", "point_mask"):
        assert getattr(rb["masks"], k).shape[1:] == getattr(ab["masks"], k).shape[1:]
        assert getattr(rb["masks"], k).dtype == getattr(ab["masks"], k).dtype
    for k in rb["aux"]:
        assert rb["aux"][k].shape[1:] == ab["aux"][k].shape[1:], k

    # masked elements are exactly zero in the init, as the trainer assumes
    em = ab["masks"].elem_mask
    assert float((ab["init"] * (1 - em)).abs().max()) == 0.0

    model = A.build_model(A.ModelConfig(max_horizon=cfg.horizon), dev)
    enc = model.encode_context(ab["context"])
    out = model(ab["init"], ab["state"], enc, ab["masks"])
    assert out.traj.shape == ab["target"].shape
    for k in ("vel", "acc", "torque"):
        assert getattr(out, k).shape == ab["target"].shape
    assert out.fk_pos.shape == ab["aux"]["fk_pos"].shape
    print(f"\n[pool] bucket {b}: model forward OK, traj {tuple(out.traj.shape)}, "
          f"aux fk_pos {tuple(out.fk_pos.shape)}")


# ------------------------------------------------- 5. the init IS the straight line
def test_input_init_is_the_straight_quintic(bundle):
    """``init_base`` reproduces a 2-knot retime to the SAME terminal pose."""
    from amortized_opt.limits import isaac_velocity_limits, urdf_position_limits

    names = A.G1_UPPER_BODY_JOINTS
    lo, hi = urdf_position_limits(names)
    vlim = isaac_velocity_limits(names)
    worst = 0.0
    n = 0
    for d in bundle["dbg"][:120]:
        ws = d["ws"]
        for hand in HANDS:
            k = ws.hit_index[hand]
            if k < 0:
                continue
            idx = arm_slice(hand)
            Y, _ = retime_joint_path(
                np.stack([ws.q0[idx], ws.terminal_arm[hand]]), k + 1, ws.dt,
                v_start=d["qd0"][idx],
                v_limit=vlim[idx], q_lower=lo[idx], q_upper=hi[idx],
            )
            worst = max(worst, float(np.abs(ws.init[:k, idx] - Y[1:]).max()))
            n += 1
    print(f"\n[init] {n} hands: max|init - straight quintic| = {worst:.2e} rad")
    assert worst < 1e-9

    # ...and it is genuinely a different path from the apex target.
    gaps = []
    for d in bundle["dbg"]:
        ws = d["ws"]
        for hand in HANDS:
            k = ws.hit_index[hand]
            if k < 2 or hand not in ws.apex_arm:
                continue
            idx = arm_slice(hand)
            gaps.append(float(np.abs(ws.target[:k, idx] - ws.init[:k, idx]).max()))
    gp = np.asarray(gaps)
    print(f"[init] max|apex target - straight init| over the arm: "
          f"p50={np.percentile(gp, 50):.3f} p90={np.percentile(gp, 90):.3f} rad")
    assert float(np.median(gp)) > 0.05


def test_pool_init_channel_is_noised_init_base(bundle):
    cfg = bundle["cfg"]
    ap = ApexWarmstartPool(bundle["out"]["buckets"], cfg=cfg, device="cpu")
    b = ap.buckets[0]
    ix = torch.arange(min(16, ap.size(b)))
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    clean = ap.batch(b, ix, init_mode="straight_clean")
    noisy = ap.batch(b, ix, init_mode="straight", generator=g)
    em = clean["masks"].elem_mask
    d_to_base = float(((noisy["init"] - clean["init_base"] * em) * em).abs().mean())
    d_to_tgt = float(((noisy["init"] - clean["target"] * em) * em).abs().mean())
    print(f"\n[pool] mean|noised init - init_base| = {d_to_base:.4f} rad, "
          f"mean|noised init - target| = {d_to_tgt:.4f} rad")
    assert d_to_base < d_to_tgt


def test_mix_init_channel(bundle):
    """``init_mode="mix"`` really is a per-sample coin flip between the two."""
    cfg = bundle["cfg"]
    ap = ApexWarmstartPool(bundle["out"]["buckets"], cfg=cfg, device="cpu")
    ap.init_mix = 0.5
    b = ap.buckets[0]
    n = min(256, ap.size(b))
    ix = torch.arange(n)
    g = torch.Generator(device="cpu")
    g.manual_seed(3)
    clean = ap.batch(b, ix, init_mode="straight_clean")
    mixed = ap.batch(b, ix, init_mode="mix", generator=g)
    em = clean["masks"].elem_mask
    per = lambda x: ((x * em) ** 2).sum(dim=(1, 2))  # noqa: E731
    d_base = per(mixed["init"] - clean["init_base"])
    d_tgt = per(mixed["init"] - clean["target"])
    frac_apex = float((d_tgt < d_base).float().mean())
    print(f"\n[mix] {n} samples: {frac_apex:.3f} nearer the apex target "
          f"(expected ~{ap.init_mix:.2f})")
    assert 0.30 < frac_apex < 0.70
    # the default channel the trainer uses
    assert ap.default_init_mode == "mix"


def test_white_noise_matches_variance_and_kills_correlation(bundle):
    """The (implemented but untrained) ablation hook behaves as advertised."""
    cfg = bundle["cfg"]
    b_ou = ApexWarmstartPool(bundle["out"]["buckets"], cfg=cfg, device="cpu")
    b_wn = ApexWarmstartPool(bundle["out"]["buckets"], cfg=cfg, device="cpu")
    b_wn.white_noise = True
    bk = b_ou.buckets[0]
    n = min(512, b_ou.size(bk))
    ix = torch.arange(n)
    sig = torch.full((n, 1, 1), 0.10)
    stats = {}
    for tag, pool in (("ou", b_ou), ("white", b_wn)):
        g = torch.Generator(device="cpu")
        g.manual_seed(5)
        base = pool.data[bk]["init_base"].index_select(0, ix)
        d = pool.noised_init(base, bk, generator=g, sigma_corr=sig) - base
        x = d - d.mean()
        var = float((x ** 2).mean())
        lag1 = float((x[:, :-1] * x[:, 1:]).mean() / max(var, 1e-12))
        stats[tag] = (var ** 0.5, lag1)
    print(f"\n[noise] sd/lag-1 autocorr: OU {stats['ou'][0]:.4f}/{stats['ou'][1]:.3f}  "
          f"white {stats['white'][0]:.4f}/{stats['white'][1]:.3f}")
    assert abs(stats["ou"][0] - stats["white"][0]) < 0.02 * stats["ou"][0] + 0.005
    assert stats["ou"][1] > 0.6
    assert abs(stats["white"][1]) < 0.15


# -------------------------------------------------------- 6. does it help? --
def test_apex_target_clears_obstacles_at_least_as_well(bundle):
    """min SDF over the target path: apex vs the straight line, paired."""
    fk = A.ChainFK(dtype=torch.float64)
    sdf = A.SdfPenalty(A.DrumSceneSpec.from_layout(), fk=fk, dtype=torch.float64)
    rec: dict[str, list] = {f"{s}_{m}": [] for s in ("apex", "straight")
                            for m in ("min", "drum", "under", "self")}
    for d in bundle["dbg"][:150]:
        ws = d["ws"]
        k = max(ws.hit_index[h] for h in HANDS)
        if k < 3:
            continue
        heads = torch.as_tensor(d["kit"].positions_in("env"), dtype=torch.float64)
        tgt_d = {}
        for hi, hand in enumerate(HANDS):
            g = d["goals"][hand]
            tgt_d[hand] = -1 if g is None else int(d["kit"].index(g[0]))
        for src, tag in (("target", "apex"), ("init", "straight")):
            X = np.concatenate([ws.q0[None, :], getattr(ws, src)[:k]], axis=0)
            q = torch.as_tensor(X, dtype=torch.float64)
            n = q.shape[0]
            tl = torch.full((n,), tgt_d["left"], dtype=torch.long)
            tr = torch.full((n,), tgt_d["right"], dtype=torch.long)
            with torch.no_grad():
                _p, det = sdf.forward(
                    q, drum_head_centers=heads[None].expand(n, -1, -1),
                    target_drum_left=tl, target_drum_right=tr, return_details=True,
                )
            # exclude the last 2 waypoints (the strike itself) as the scorer does
            rec[f"{tag}_min"].append(float(det.min_sdf[:-2].min()))
            rec[f"{tag}_drum"].append(float(det.pen_drum[:-2].sum()))
            rec[f"{tag}_under"].append(float(det.pen_under[:-2].sum()))
            rec[f"{tag}_self"].append(float(det.pen_self[:-2].sum()))
    R = {k: np.asarray(v) for k, v in rec.items()}
    a, s = R["apex_min"] * 1000.0, R["straight_min"] * 1000.0
    print(f"\n[clearance] over the path, target-drum exemption ON, n={a.size}")
    print(f"  min SDF (mm, higher better)  apex p10={np.percentile(a, 10):7.1f} "
          f"p50={np.percentile(a, 50):7.1f} p90={np.percentile(a, 90):7.1f} | "
          f"straight p10={np.percentile(s, 10):7.1f} p50={np.percentile(s, 50):7.1f} "
          f"p90={np.percentile(s, 90):7.1f}")
    print(f"  min SDF >= 0 over the whole path: apex {float((a >= 0).mean()):.3f} "
          f"straight {float((s >= 0).mean()):.3f}; paired win rate "
          f"{float((a > s).mean()):.3f}")
    for term in ("drum", "under", "self"):
        pa, ps = R[f"apex_{term}"], R[f"straight_{term}"]
        print(f"  penalty[{term:5s}] (lower better) apex p50={np.percentile(pa, 50):.3e} "
              f"mean={pa.mean():.3e} | straight p50={np.percentile(ps, 50):.3e} "
              f"mean={ps.mean():.3e} | apex better on {float((pa < ps).mean()):.3f}")
    assert np.percentile(a, 50) >= np.percentile(s, 50) - 10.0


def test_terminal_choice_does_not_move_the_tip(bundle):
    """``nearest_apex`` picks a branch; it must not move the contact point."""
    from amortized_opt.ik import tip_position

    fk = A.ChainFK(dtype=torch.float64)
    lib = bundle["library"]
    acfg = bundle["acfg"]
    yaw_cfg = ApexConfig(terminal_mode="nearest_yaw")
    err_a, err_y, moved = [], [], 0
    for d in bundle["dbg"]:
        ws = d["ws"]
        for hand in HANDS:
            if ws.hit_index[hand] < 0:
                continue
            drum = d["goals"][hand][0]
            qa = ws.apex_arm.get(hand)
            e_a = select_terminal(lib, hand, drum, yaw_target=ws.yaw_target,
                                  apex_arm=qa, cfg=acfg)
            e_y = select_terminal(lib, hand, drum, yaw_target=ws.yaw_target,
                                  apex_arm=qa, cfg=yaw_cfg)
            moved += int(float(np.linalg.norm(e_a.arm_q - e_y.arm_q)) > 1e-9)
            aim = np.asarray(d["aim"][hand], float)
            for e, acc in ((e_a, err_a), (e_y, err_y)):
                q = ws.q[ws.hit_index[hand]].copy()
                q[arm_slice(hand)] = e.arm_q
                p = tip_position(fk, torch.as_tensor(q[None, :], dtype=torch.float64),
                                 hand, frame="env")[0].numpy()
                acc.append(float(np.linalg.norm(p - aim)))
    a, y = np.asarray(err_a) * 1000, np.asarray(err_y) * 1000
    print(f"\n[terminal] nearest-apex moved the pick on {moved}/{a.size} hands "
          f"({100 * moved / max(a.size, 1):.1f} %)")
    print(f"  tip error to the aim point (mm): nearest_apex p50={np.percentile(a, 50):.1f} "
          f"p90={np.percentile(a, 90):.1f} | nearest_yaw p50={np.percentile(y, 50):.1f} "
          f"p90={np.percentile(y, 90):.1f}")
    assert np.percentile(a, 50) < np.percentile(y, 50) + 25.0


# ------------------------------------------------------------------ 7. stats
def test_report_construction_stats(bundle):
    s = bundle["out"]["stats"]
    acc, req = s["accepted"], s["requested"]
    print(f"\n[stats] accepted {acc}/{req} = {100 * acc / req:.1f} % "
          f"(drop {100 * (1 - acc / req):.1f} %: |qd| {s['reject_velocity']}, "
          f"idle {s['reject_idle']})")
    print(f"[stats] apex solved {s['apex_solved']}, no room (k_hit<2) {s['apex_no_room']}, "
          f"rejected for missing the cylinder {s['apex_rejected_outside']}, "
          f"collision-free {s['apex_collision_free']}/{s['apex_solved']}")
    pr, pi = s["peak_rate_ratio"], s["peak_rate_ratio_init"]
    print(f"[stats] peak |qd| / rating -- apex target p50={np.percentile(pr, 50):.2f} "
          f"p99={np.percentile(pr, 99):.2f} | straight init p50={np.percentile(pi, 50):.2f} "
          f"p99={np.percentile(pi, 99):.2f}")
    assert acc > 0
