"""SDF conformance vs the repo original, including target-drum exemptions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import DRUM_LAYOUT_PATH
from amortized_opt.fk import ChainFK
from amortized_opt.sdf import DrumSceneSpec, SdfPenalty
from conftest import import_repo


def _q(n: int = 96, dtype=torch.float64, seed: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 17, generator=g, dtype=torch.float64) * 2.0 - 1.0).to(dtype)


def _scene_pair(frame: str):
    repo = import_repo("safety.sdf_torch")
    ref_scene = repo.DrumSceneSpec.from_layout(str(DRUM_LAYOUT_PATH), frame=frame)
    our_scene = DrumSceneSpec.from_layout(DRUM_LAYOUT_PATH, frame=frame)
    assert ref_scene.names == our_scene.names
    return repo, ref_scene, our_scene


@pytest.mark.parametrize("frame", ["env", "pelvis"])
def test_sdf_matches_repo_no_targets(frame, capsys):
    repo, ref_scene, our_scene = _scene_pair(frame)
    ref = repo.SdfPenalty(ref_scene, dtype=torch.float64)
    ours = SdfPenalty(our_scene, dtype=torch.float64)
    q = _q()
    a = ref.forward(q)
    b = ours.forward(q)
    err = float((a - b).abs().max())
    with capsys.disabled():
        print(f"\n[sdf conformance frame={frame}] max abs err = {err:.3e}")
    assert err <= 1e-6


def test_sdf_matches_repo_with_targets(capsys):
    repo, ref_scene, our_scene = _scene_pair("env")
    ref = repo.SdfPenalty(ref_scene, dtype=torch.float64)
    ours = SdfPenalty(our_scene, dtype=torch.float64)
    q = _q(128, seed=17)
    B = q.shape[0]
    D = our_scene.num_drums
    g = torch.Generator().manual_seed(3)
    # mix of -1 (no target) and every valid drum id, on both hands
    tl = torch.randint(-1, D, (B,), generator=g)
    tr = torch.randint(-1, D, (B,), generator=g)

    a, da = ref.forward(q, target_drum_left=tl, target_drum_right=tr, return_details=True)
    b, db = ours.forward(q, target_drum_left=tl, target_drum_right=tr, return_details=True)
    err = float((a - b).abs().max())
    finite = torch.isfinite(da.min_sdf) & torch.isfinite(db.min_sdf)
    err_min = float((da.min_sdf[finite] - db.min_sdf[finite]).abs().max())
    with capsys.disabled():
        print(
            f"\n[sdf conformance with targets] penalty {err:.3e}, min_sdf {err_min:.3e}, "
            f"n_exempt_left={int((tl >= 0).sum())} n_exempt_right={int((tr >= 0).sum())}"
        )
    assert err <= 1e-6
    assert err_min <= 1e-6


def test_sdf_all_targets_set(capsys):
    """The branch-free exemption must also match when every row is exempt."""
    repo, ref_scene, our_scene = _scene_pair("env")
    ref = repo.SdfPenalty(ref_scene, dtype=torch.float64)
    ours = SdfPenalty(our_scene, dtype=torch.float64)
    q = _q(64, seed=23)
    B, D = q.shape[0], our_scene.num_drums
    tl = torch.arange(B) % D
    tr = (torch.arange(B) + 1) % D
    err = float(
        (
            ref.forward(q, target_drum_left=tl, target_drum_right=tr)
            - ours.forward(q, target_drum_left=tl, target_drum_right=tr)
        )
        .abs()
        .max()
    )
    with capsys.disabled():
        print(f"\n[sdf conformance all-exempt] max abs err = {err:.3e}")
    assert err <= 1e-6


def test_sdf_matches_repo_with_jittered_heads(capsys):
    """Domain-randomized drum positions path."""
    repo, ref_scene, our_scene = _scene_pair("env")
    ref = repo.SdfPenalty(ref_scene, dtype=torch.float64)
    ours = SdfPenalty(our_scene, dtype=torch.float64)
    q = _q(48, seed=29)
    B, D = q.shape[0], our_scene.num_drums
    g = torch.Generator().manual_seed(5)
    heads = torch.tensor(our_scene.head_centers.tolist(), dtype=torch.float64).expand(B, D, 3)
    heads = heads + torch.randn(B, D, 3, generator=g, dtype=torch.float64) * 0.03
    err = float(
        (ref.forward(q, drum_head_centers=heads) - ours.forward(q, drum_head_centers=heads))
        .abs()
        .max()
    )
    with capsys.disabled():
        print(f"\n[sdf conformance jittered heads] max abs err = {err:.3e}")
    assert err <= 1e-6


def test_shared_fk_path_matches_internal_fk():
    """forward(frames=...) must equal forward(q)."""
    fk = ChainFK(dtype=torch.float64)
    pen = SdfPenalty(fk=fk, dtype=torch.float64)
    q = _q(32, seed=31)
    a = pen.forward(q)
    b = pen.forward(frames=fk.forward(q))
    assert float((a - b).abs().max()) == 0.0


def test_exemption_lowers_penalty():
    """Exempting the target drum can only reduce the penalty."""
    pen = SdfPenalty(dtype=torch.float64)
    q = _q(64, seed=37)
    D = pen.num_drums
    base = pen.forward(q)
    for d in range(D):
        tgt = torch.full((q.shape[0],), d, dtype=torch.long)
        assert bool((pen.forward(q, target_drum_left=tgt, target_drum_right=tgt) <= base + 1e-12).all())
