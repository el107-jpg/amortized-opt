"""The vendored Isaac scene assets (``g1_scene``) agree with the analytic model.

``g1_scene`` replaces what ``isaac_rl``/``isaac_rl_ppo`` used to import out of
the main repo. Two things have to hold for that port to be safe:

1. **Geometry parity.** The stick colliders PhysX simulates and the stick tip
   :mod:`amortized_opt.fk` scores must be the same object. They are now built
   from one set of constants, but the quaternion still has to be normalized on
   the way in -- see ``g1_scene.g1._unit_quat`` -- or the raw Hamilton rotation
   scales by |q|^2 and the two drift ~2e-7 m apart.
2. **Importable headless.** The geometry half must not need a launched Kit app,
   so a plain ``pytest`` run (no Isaac) still covers it.

The articulation cfg itself (``get_g1_cfg``) needs Isaac Lab and the upstream
Unitree USD, so it is exercised by ``isaac_rl_ppo/tests`` under a live app, not
here.
"""

from __future__ import annotations

import numpy as np
import pytest

from amortized_opt.constants import SHAFT_HIT_SPAN, STICK_LENGTH
from amortized_opt.fk import stick_contact_offset, stick_tip_offset
from g1_scene import DRUM_KIT_USD_PATH, DRUM_LAYOUT_PATH, ordered_drum_names
from g1_scene.g1 import (
    LEFT_WRIST_LINK,
    RIGHT_WRIST_LINK,
    SHAFT_HIT_SPAN_M,
    STICK_MOUNT_ROT_LEFT,
    STICK_MOUNT_ROT_RIGHT,
    TIP_OFFSET_IN_WRIST_LEFT,
    TIP_OFFSET_IN_WRIST_RIGHT,
    stick_shaft_offsets_in_wrist,
)

#: Machine-epsilon agreement, not a tolerance to relax. The two paths compute
#: the same quantity from the same numbers; anything above ~1e-15 means someone
#: reintroduced a second copy of the mount transform.
TOL = 1e-12


def test_mount_quaternions_are_unit():
    for q in (STICK_MOUNT_ROT_LEFT, STICK_MOUNT_ROT_RIGHT):
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-15


@pytest.mark.parametrize(
    "hand,tip",
    [("left", TIP_OFFSET_IN_WRIST_LEFT), ("right", TIP_OFFSET_IN_WRIST_RIGHT)],
)
def test_collider_tip_matches_analytic_fk(hand, tip):
    """Where PhysX puts the tip sphere == where the cost function thinks it is."""
    assert np.abs(np.asarray(tip) - stick_tip_offset(hand)).max() < TOL


@pytest.mark.parametrize("hand", ["left", "right"])
def test_shaft_probes_match_analytic_contact_offsets(hand):
    """Same for every shaft probe point (the cymbal/hi-hat contact samples)."""
    n = 4
    probes = stick_shaft_offsets_in_wrist(hand, n_samples=n)
    assert len(probes) == n
    for k, probe in enumerate(probes, start=1):
        sigma = STICK_LENGTH / 2.0 - k * (SHAFT_HIT_SPAN_M / n)
        assert np.abs(np.asarray(probe) - stick_contact_offset(hand, sigma)).max() < TOL


def test_shaft_span_tracks_the_shared_constant():
    assert SHAFT_HIT_SPAN_M == SHAFT_HIT_SPAN


def test_sticks_are_sagittal_mirrors():
    """Left and right differ only in the sign of y -- a mirrored mount, not a copy."""
    left = np.asarray(TIP_OFFSET_IN_WRIST_LEFT)
    right = np.asarray(TIP_OFFSET_IN_WRIST_RIGHT)
    assert np.abs(left * np.array([1.0, -1.0, 1.0]) - right).max() < TOL
    assert left[1] < 0.0 < right[1]


def test_wrist_links_are_the_yaw_links():
    """The stick hangs off wrist_yaw, so all three wrist DoF move the tip."""
    assert LEFT_WRIST_LINK == "left_wrist_yaw_link"
    assert RIGHT_WRIST_LINK == "right_wrist_yaw_link"


# ------------------------------------------------------------------ assets --
def test_scene_assets_are_vendored():
    assert DRUM_KIT_USD_PATH.is_file(), DRUM_KIT_USD_PATH
    assert DRUM_LAYOUT_PATH.is_file(), DRUM_LAYOUT_PATH


def test_drum_ids_are_contiguous_and_match_the_kit():
    """Contact-sensor filters are emitted in this order and indexed by drum id."""
    from amortized_opt.rhythm import DrumKit

    names = ordered_drum_names()
    assert len(names) == len(set(names)) and names
    # DrumKit sorts the same YAML by id; the two orderings must not diverge.
    assert list(names) == list(DrumKit.from_layout().names)


def test_bad_layout_ids_are_rejected(tmp_path):
    """A gap in the ids would silently misattribute every hit -- so it raises."""
    import yaml

    data = yaml.safe_load(DRUM_LAYOUT_PATH.read_text())
    first = next(iter(data["drums"]))
    data["drums"][first]["id"] = 99
    bad = tmp_path / "bad_layout.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(RuntimeError, match="0..D-1"):
        ordered_drum_names(bad)


def test_geometry_half_imports_without_isaac():
    """``g1_scene.g1`` must be importable with no Kit app running.

    The module-level import of ``isaaclab.sim`` was the reason the repo original
    could not be; it is deferred into ``get_g1_cfg`` here.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "import g1_scene.g1  # noqa: F401\n"
        "heavy = ('isaaclab', 'omni', 'unitree_rl_lab', 'pxr')\n"
        "leaked = [m for m in sys.modules if m.startswith(heavy)]\n"
        "assert not leaked, leaked\n"
        "print('HEADLESS_OK')\n"
    )
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "HEADLESS_OK" in out.stdout
