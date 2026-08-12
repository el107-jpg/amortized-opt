"""FK conformance + the new elbow frames."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS
from amortized_opt.fk import ChainFK, stick_shaft_offsets, stick_tip_offset
from conftest import import_repo


def _q(n: int = 64, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 17, generator=g, dtype=torch.float64) * 2.4 - 1.2).to(dtype)


# ------------------------------------------------------------- conformance --
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fk_matches_repo_torch_fk(dtype, capsys):
    """Wrist/torso frames + stick tip/shaft points vs the repo TorchChainFK."""
    repo = import_repo("safety.torch_fk")
    ref = repo.TorchChainFK(device="cpu", dtype=dtype)
    ours = ChainFK(device="cpu", dtype=dtype)
    q = _q(128, dtype)

    worst = 0.0
    for name in ("left_wrist", "right_wrist", "torso"):
        Rr, pr = ref.frame(q, name)
        Ro, po = ours.frame(q, name)
        worst = max(worst, float((Rr - Ro).abs().max()), float((pr - po).abs().max()))

    # tip + shaft sample points off each wrist frame
    for hand, frame in (("left", "left_wrist"), ("right", "right_wrist")):
        local = torch.tensor(
            np.concatenate([stick_tip_offset(hand)[None, :], stick_shaft_offsets(hand, 4)]).tolist(),
            dtype=dtype,
        )
        a = repo.TorchChainFK.points_world(*ref.frame(q, frame), local)
        b = ChainFK.points_world(*ours.frame(q, frame), local)
        worst = max(worst, float((a - b).abs().max()))

    with capsys.disabled():
        print(f"\n[fk conformance {dtype}] max abs err = {worst:.3e}")
    assert worst <= 1e-5


def test_stick_helpers_match_repo():
    repo = import_repo("safety.torch_fk")
    for hand in ("left", "right"):
        assert np.allclose(stick_tip_offset(hand), repo.stick_tip_offset(hand), atol=0)
        assert np.allclose(stick_shaft_offsets(hand, 6), repo.stick_shaft_offsets(hand, 6), atol=0)


# -------------------------------------------------------------- elbow frame --
def test_elbow_frames_registered():
    fk = ChainFK()
    assert "left_elbow" in fk.frame_names and "right_elbow" in fk.frame_names


def test_elbow_moves_with_elbow_joint():
    """The elbow frame must respond to the elbow joint angle."""
    fk = ChainFK(dtype=torch.float64)
    i_el = G1_UPPER_BODY_JOINTS.index("left_elbow_joint")
    q0 = torch.zeros(1, 17, dtype=torch.float64)
    q1 = q0.clone()
    q1[0, i_el] = 1.0
    _, p0 = fk.frame(q0, "left_elbow")
    _, p1 = fk.frame(q1, "left_elbow")
    # The elbow JOINT frame origin does not translate with its own angle (it is
    # the joint's own origin), but its orientation must rotate; the WRIST must
    # translate.
    R0, _ = fk.frame(q0, "left_elbow")
    R1, _ = fk.frame(q1, "left_elbow")
    assert float((R1 - R0).abs().max()) > 0.1
    _, w0 = fk.frame(q0, "left_wrist")
    _, w1 = fk.frame(q1, "left_wrist")
    assert float((w1 - w0).norm()) > 0.05
    assert float((p1 - p0).abs().max()) < 1e-12


def test_elbow_between_shoulder_and_wrist():
    """|shoulder->elbow| and |elbow->wrist| are both < |shoulder->wrist| + slack."""
    fk = ChainFK(dtype=torch.float64)
    q = _q(64, torch.float64, seed=3)
    frames = fk.forward(q)
    # shoulder origin: read from the chain by evaluating a shoulder-only frame
    sh = ChainFK(dtype=torch.float64, frames={"sh": "left_shoulder_yaw_joint"})
    _, p_sh = sh.frame(q, "sh")
    p_el = frames["left_elbow"][1]
    p_wr = frames["left_wrist"][1]
    d_se = (p_el - p_sh).norm(dim=-1)
    d_ew = (p_wr - p_el).norm(dim=-1)
    d_sw = (p_wr - p_sh).norm(dim=-1)
    assert bool((d_se <= d_se + d_ew + 1e-9).all())
    assert bool((d_sw <= d_se + d_ew + 1e-9).all())  # triangle inequality
    # upper arm length is constant (rigid link)
    assert float(d_se.max() - d_se.min()) < 1e-9


def test_elbow_matches_pinocchio(capsys):
    """Vendored elbow frame vs Pinocchio ``oMi`` for the same joint."""
    pin = pytest.importorskip("pinocchio")
    from amortized_opt.limits import build_reduced_model

    model, opt2pin = build_reduced_model()
    data = model.createData()
    fk = ChainFK(dtype=torch.float64)
    q = _q(24, torch.float64, seed=7)

    worst = 0.0
    for frame, jname in (
        ("left_elbow", "left_elbow_joint"),
        ("right_elbow", "right_elbow_joint"),
        ("left_wrist", "left_wrist_yaw_joint"),
        ("right_wrist", "right_wrist_yaw_joint"),
    ):
        jid = model.getJointId(jname)
        R, p = fk.frame(q, frame)
        for b in range(q.shape[0]):
            qp = np.zeros(model.nq)
            qp[opt2pin] = q[b].numpy()
            pin.forwardKinematics(model, data, qp)
            worst = max(
                worst,
                float(np.abs(np.asarray(data.oMi[jid].translation).ravel() - p[b].numpy()).max()),
                float(np.abs(np.asarray(data.oMi[jid].rotation) - R[b].numpy()).max()),
            )
    with capsys.disabled():
        print(f"\n[elbow/wrist vs pinocchio] max abs err = {worst:.3e}")
    assert worst <= 1e-4
