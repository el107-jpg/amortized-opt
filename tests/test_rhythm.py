"""RCC loading, hit quantization, kit randomization, tokenizer compatibility."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt import constants as C
from amortized_opt.model import AmortizedOptimizer, ModelConfig
from amortized_opt.rhythm import (
    DrumKit,
    KitRandomization,
    build_context_inputs,
    default_rcc_dir,
    goal_snapshot,
    load_rcc,
    load_rcc_pool,
    quantize_time,
    sample_hit_point,
    strike_region,
    strike_velocity,
)
from amortized_opt.tokenizer import build_waypoint_masks, pack_state


@pytest.fixture(scope="module")
def kit():
    return DrumKit.from_layout()


@pytest.fixture(scope="module")
def clips(kit):
    return load_rcc_pool(default_rcc_dir("train"), kit)


def test_vendored_clip_corpus_is_present(clips, capsys):
    ev = load_rcc_pool(default_rcc_dir("eval"), DrumKit.from_layout())
    n_hits = sum(c.num_hits for c in clips)
    with capsys.disabled():
        print(f"\n[rcc] {len(clips)} train clips ({n_hits} hits), {len(ev)} eval clips "
              f"({sum(c.num_hits for c in ev)} hits)")
    assert len(clips) >= 10 and len(ev) >= 4
    assert all(c.num_hits > 0 for c in clips + ev)


def test_quantization_error_is_at_most_half_a_tick(clips, capsys):
    worst = 0.0
    off_grid = 0
    for c in clips:
        for h in c.hits:
            worst = max(worst, abs(h.time - h.raw_time))
            if abs(h.time / C.CONTROL_DT - round(h.time / C.CONTROL_DT)) > 1e-9:
                off_grid += 1
    with capsys.disabled():
        print(f"\n[rcc quantization] max |t_q - t_raw| = {worst * 1e3:.3f} ms "
              f"(grid {C.HIT_QUANTIZATION_S * 1e3:.0f} ms); off-grid hits={off_grid}")
    assert worst <= C.HIT_QUANTIZATION_S / 2 + 1e-9
    assert worst <= 0.010 + 1e-9
    assert off_grid == 0


def test_quantize_time_rounds_to_nearest():
    assert quantize_time(0.0) == 0.0
    assert quantize_time(0.009) == 0.0
    assert abs(quantize_time(0.011) - 0.02) < 1e-12
    assert abs(quantize_time(-0.011) + 0.02) < 1e-12


def test_hand_and_drum_vocabulary_matches_the_repo_loader(kit, capsys):
    """The vendored parser must map limbs / aliases exactly like the env's.

    ``conftest.import_repo`` puts ``control/`` on the path; the rhythm loader
    lives under ``tasks/``, whose package ``__init__`` drags in Isaac Lab, so
    the module file is loaded directly instead (it only needs json/yaml).
    """
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    mod = (
        root / "source" / "g1_drummer" / "g1_drummer" / "tasks" / "direct" / "g1_drum" / "rhythm.py"
    )
    if not mod.is_file():
        pytest.skip(f"repo rhythm loader not present at {mod}")
    try:
        spec = importlib.util.spec_from_file_location("_repo_rhythm", mod)
        repo = importlib.util.module_from_spec(spec)
        # @dataclass resolves annotations through sys.modules, so register it.
        sys.modules[spec.name] = repo
        spec.loader.exec_module(repo)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import repo rhythm loader: {exc}")
    layout = repo.DrumLayout.from_yaml(C.default_layout_path())
    path = sorted(default_rcc_dir("train").glob("*.json"))[0]
    theirs = repo.RhythmSequence.from_rcc(path, layout)
    mine = load_rcc(path, kit, quantize=False, on_ineligible="keep")
    with capsys.disabled():
        print(f"\n[rcc vs repo loader] {path.name}: mine={mine.num_hits} theirs={theirs.num_beats}")
    assert mine.num_hits == theirs.num_beats
    ids = {"left": 0, "right": 1}
    theirs_set = sorted(zip(theirs.times, theirs.hands, theirs.drum_ids, theirs.velocities))
    mine_set = sorted((h.raw_time, ids[h.hand], kit.index(h.drum_name), h.velocity) for h in mine.hits)
    for a, b in zip(mine_set, theirs_set):
        assert abs(a[0] - b[0]) < 1e-9 and a[1] == b[1] and a[2] == b[2]
        assert abs(a[3] - b[3]) < 1e-9


def test_ineligible_hits_are_handled_as_requested(kit):
    path = sorted(default_rcc_dir("train").glob("*.json"))[0]
    keep = load_rcc(path, kit, on_ineligible="keep")
    drop = load_rcc(path, kit, on_ineligible="drop")
    swap = load_rcc(path, kit, on_ineligible="reassign")
    assert drop.num_hits + keep.ineligible == keep.num_hits
    assert swap.num_hits >= drop.num_hits
    elig = {h: set(kit.eligible(h)) for h in ("left", "right")}
    assert all(h.drum in elig[h.hand] for h in swap.hits)
    assert all(h.drum in elig[h.hand] for h in drop.hits)


def test_randomization_bounds_over_1000_draws(kit, capsys):
    """Each component is bounded exactly; isolate them so the bound is tight.

    Measuring both at once only bounds their sum, which would let a 10 cm
    per-drum bug hide behind the global allowance. So the per-drum draw is
    measured with the global term switched off and vice versa, then a combined
    draw checks that both are actually applied.
    """
    cfg = KitRandomization()
    rng = np.random.default_rng(0)
    per_drum, gxy, gz = [], [], []
    only_drum = KitRandomization(cfg.per_drum_xy, 0.0, 0.0)
    only_glob = KitRandomization(0.0, cfg.global_xy, cfg.global_z)
    for _ in range(1000):
        d = kit.randomized(rng, only_drum).positions - kit.positions
        assert np.allclose(d[:, 2], 0.0)  # a single drum NEVER moves vertically
        per_drum.append(float(np.linalg.norm(d[:, :2], axis=1).max()))

        g = kit.randomized(rng, only_glob).positions - kit.positions
        assert np.allclose(g, g[0])  # the global offset is common to every drum
        gxy.append(float(np.linalg.norm(g[0, :2])))
        gz.append(abs(float(g[0, 2])))

        k = kit.randomized(rng, cfg)
        assert np.array_equal(k.radii, kit.radii) and np.array_equal(k.axes, kit.axes)
        assert np.array_equal(k.heights, kit.heights) and k.names == kit.names
        dd = k.positions - kit.positions
        assert np.allclose(dd[:, 2], dd[0, 2])  # only the global term moves Z
        assert not np.allclose(dd[:, :2], dd[0, :2])  # ...and the drums differ in XY
    with capsys.disabled():
        print(f"\n[kit randomization, 1000 draws each] per-drum |dXY| max="
              f"{max(per_drum) * 100:.2f} cm (bound {cfg.per_drum_xy * 100:.1f}); "
              f"global |dXY| max={max(gxy) * 100:.2f} cm (bound {cfg.global_xy * 100:.1f}); "
              f"global |dZ| max={max(gz) * 100:.2f} cm (bound {cfg.global_z * 100:.1f})")
    assert max(per_drum) <= cfg.per_drum_xy + 1e-12
    assert max(gxy) <= cfg.global_xy + 1e-12
    assert max(gz) <= cfg.global_z + 1e-12
    # ...and the bounds are approached, so a silently-zeroed term would fail.
    assert max(per_drum) > 0.95 * cfg.per_drum_xy
    assert max(gxy) > 0.95 * cfg.global_xy
    assert max(gz) > 0.95 * cfg.global_z


def test_randomization_never_moves_a_single_drum_vertically(kit):
    rng = np.random.default_rng(1)
    k = kit.randomized(rng, KitRandomization(per_drum_xy=0.05, global_xy=0.0, global_z=0.0))
    assert np.allclose(k.positions[:, 2], kit.positions[:, 2])


def test_randomization_is_seeded(kit):
    a = kit.randomized(np.random.default_rng(4))
    b = kit.randomized(np.random.default_rng(4))
    c = kit.randomized(np.random.default_rng(5))
    assert np.array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions, c.positions)


def test_hit_points_stay_inside_the_legal_strike_region(kit):
    """Goal points must satisfy the SAME geometry the terminal cost prices."""
    rng = np.random.default_rng(0)
    for d in range(kit.num_drums):
        aim, center, disk_r, half = strike_region(kit, d)
        for _ in range(200):
            p = sample_hit_point(kit, d, rng, hit_radius_frac=0.6)
            off = p - center
            assert abs(float(off @ kit.axes[d])) < 1e-12  # in the head plane
            assert float(np.linalg.norm(off)) <= disk_r + 1e-9  # in the hit disk
            if half is not None:
                assert float(off @ half) >= -1e-9  # on the playable semicircle
    assert np.array_equal(sample_hit_point(kit, 0, None), strike_region(kit, 0)[0])


def test_hit_points_track_the_strike_aim(kit):
    """A cymbal goal sits out toward the near edge, not on the bell."""
    rng = np.random.default_rng(1)
    for name in ("cymbal_right", "snare"):
        if name not in kit.names:
            continue
        d = kit.index(name)
        aim, center, disk_r, _half = strike_region(kit, d)
        pts = np.stack([sample_hit_point(kit, d, rng) for _ in range(400)])
        mean_off = float(np.linalg.norm(pts.mean(axis=0) - center))
        aim_off = float(np.linalg.norm(aim - center))
        # the sample cloud is centred on the aim (up to the disk clamp), not on
        # the head centre
        assert abs(mean_off - aim_off) < 0.5 * disk_r
        if aim_off > 1e-6:
            assert mean_off > 0.2 * aim_off


def test_strike_velocity_is_into_the_head_and_scales_with_loudness(kit):
    for d in range(kit.num_drums):
        soft = strike_velocity(kit, d, 0.0)
        loud = strike_velocity(kit, d, 1.0)
        assert float(soft @ kit.axes[d]) < 0.0
        assert abs(float(np.linalg.norm(soft)) - 1.0) < 1e-12
        assert abs(float(np.linalg.norm(loud)) - 3.0) < 1e-12


def test_goal_snapshot_shape_and_ordering(kit, clips):
    clip = clips[0]
    snap = goal_snapshot(clip, kit, 0.0, num_future=3, rng=np.random.default_rng(0))
    for hand in ("left", "right"):
        g = snap.goal[hand]
        if g.active:
            assert g.time_to_hit > 0.0
            assert kit.is_eligible(hand, g.drum)
        times = [f.time_to_hit for f in snap.future[hand]]
        assert times == sorted(times)
        assert all(t >= (g.time_to_hit if g.active else 0.0) for t in times)


def test_context_feeds_the_model_end_to_end(kit, clips, capsys):
    """Goal sequences must be exactly what encode_context / the decoder want."""
    torch.manual_seed(0)
    B, H, F, D = 4, 40, 3, 8
    rng = np.random.default_rng(0)
    snaps = [
        goal_snapshot(clips[i % len(clips)], kit, 0.3 * i + 0.1, num_future=F, rng=rng)
        for i in range(B)
    ]
    kits = [kit.randomized(rng) for _ in range(B)]
    ctx = build_context_inputs(kits, snaps, num_drums=D, num_future=F)

    model = AmortizedOptimizer(ModelConfig(max_horizon=H, num_drums=D, num_future=F))
    enc = model.encode_context(ctx)
    tl = torch.as_tensor([s.time_to_hit("left", inactive=0.0) for s in snaps], dtype=torch.float32)
    tr = torch.as_tensor([s.time_to_hit("right", inactive=0.0) for s in snaps], dtype=torch.float32)
    al = torch.as_tensor([s.active("left") for s in snaps])
    ar = torch.as_tensor([s.active("right") for s in snaps])
    masks = build_waypoint_masks(tl, tr, max_horizon=H, active_left=al, active_right=ar)
    init = torch.zeros(B, H, C.N_DOF)
    state = pack_state(torch.zeros(B, C.N_DOF), torch.zeros(B, C.N_DOF))
    out = model(init, state, enc, masks)
    with capsys.disabled():
        print(f"\n[rhythm -> tokenizer -> model] memory={tuple(enc.memory.shape)} "
              f"traj={tuple(out.traj.shape)} active tokens per row="
              f"{masks.token_mask.sum(1).tolist()}")
    assert out.traj.shape == (B, H, C.N_DOF)
    assert torch.isfinite(out.traj).all() and torch.isfinite(out.torque).all()
    # Masked elements must come back exactly zero.
    assert float((out.traj * (1.0 - masks.elem_mask)).abs().max()) == 0.0


def test_context_indices_line_up_with_the_sdf_drum_order(kit):
    from amortized_opt.sdf import DrumSceneSpec

    scene = DrumSceneSpec.from_layout()
    assert scene.names == kit.names
    assert np.allclose(scene.head_centers, kit.positions)
    assert np.allclose(scene.axes, kit.axes)
