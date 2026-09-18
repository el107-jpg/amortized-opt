"""Layout-graph stick assignment: closer stick, parked on last drum."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PATH = _ROOT / "amortized_opt" / "hand_assign.py"
_SPEC = importlib.util.spec_from_file_location("hand_assign_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HA = importlib.util.module_from_spec(_SPEC)
sys.modules["hand_assign_under_test"] = _HA
_SPEC.loader.exec_module(_HA)

LIMB_FOOT = _HA.LIMB_FOOT
LIMB_LEFT = _HA.LIMB_LEFT
LIMB_RIGHT = _HA.LIMB_RIGHT
assign_rcc_groups = _HA.assign_rcc_groups
assign_simultaneous = _HA.assign_simultaneous
kit_graph_from_layout = _HA.kit_graph_from_layout


def _group(t: float, *drums: str) -> dict:
    return {
        "time": t,
        "targets": [{"drum": d, "drum_id": 0, "limb": LIMB_FOOT, "velocity": 1.0} for d in drums],
    }


def _limbs(groups) -> list[tuple[float, str, str]]:
    return [(g["time"], t["drum"], t["limb"]) for g in groups for t in g["targets"]]


def test_idle_side_and_no_crossover_bans():
    g = kit_graph_from_layout()
    assert g.idle[LIMB_LEFT][1] > g.idle[LIMB_RIGHT][1]
    names = list(g.names)
    assert names[g.banned[LIMB_LEFT]] == "floor_tom"
    assert names[g.banned[LIMB_RIGHT]] == "hihat"
    assert not g.eligible(LIMB_LEFT, "floor_tom")
    assert not g.eligible(LIMB_RIGHT, "hihat")
    assert g.eligible(LIMB_LEFT, "snare") and g.eligible(LIMB_RIGHT, "snare")


def test_first_snare_goes_to_left_from_idle():
    graph = kit_graph_from_layout()
    parked = {LIMB_LEFT: None, LIMB_RIGHT: None}
    got = assign_simultaneous(["snare"], parked, graph)
    assert got == [("snare", LIMB_LEFT)]


def test_ride_from_idle_goes_to_right():
    graph = kit_graph_from_layout()
    parked = {LIMB_LEFT: None, LIMB_RIGHT: None}
    got = assign_simultaneous(["cymbal_right"], parked, graph)
    assert got == [("cymbal_right", LIMB_RIGHT)]


def test_snare_plus_ride_is_left_snare_right_ride():
    graph = kit_graph_from_layout()
    parked = {LIMB_LEFT: None, LIMB_RIGHT: None}
    got = dict(assign_simultaneous(["snare", "cymbal_right"], parked, graph))
    assert got == {"snare": LIMB_LEFT, "cymbal_right": LIMB_RIGHT}


def test_crash_and_snare_do_not_cross_after_a_ride():
    """Left parked on snare, right on ride: snare+crash must not give the
    crash to the right stick just because left-on-snare has travel 0."""
    graph = kit_graph_from_layout()
    parked = {LIMB_LEFT: "snare", LIMB_RIGHT: "cymbal_right"}
    got = dict(assign_simultaneous(["snare", "cymbal_left"], parked, graph))
    assert got["cymbal_left"] == LIMB_LEFT
    assert got["snare"] == LIMB_RIGHT


def test_sequence_crash_stays_on_the_left_stick():
    groups = assign_rcc_groups(
        [
            _group(0.0, "snare", "cymbal_right"),
            _group(0.4, "snare", "cymbal_left"),
            _group(0.8, "cymbal_right"),
        ]
    )
    by = {(t, d): limb for t, d, limb in _limbs(groups)}
    assert by[(0.4, "cymbal_left")] == LIMB_LEFT
    assert by[(0.4, "snare")] == LIMB_RIGHT
    assert by[(0.8, "cymbal_right")] == LIMB_RIGHT


def test_later_snare_stays_on_the_hand_that_already_owns_it():
    """The old pref-list gave solo snares to the right after a snare+ride."""
    groups = assign_rcc_groups(
        [
            _group(0.0, "snare", "cymbal_right"),
            _group(0.4, "cymbal_right"),
            _group(0.8, "snare"),
            _group(1.2, "snare", "cymbal_right"),
        ]
    )
    by = {(t, d): limb for t, d, limb in _limbs(groups)}
    assert by[(0.0, "snare")] == LIMB_LEFT
    assert by[(0.0, "cymbal_right")] == LIMB_RIGHT
    assert by[(0.4, "cymbal_right")] == LIMB_RIGHT
    assert by[(0.8, "snare")] == LIMB_LEFT
    assert by[(1.2, "snare")] == LIMB_LEFT
    assert by[(1.2, "cymbal_right")] == LIMB_RIGHT


def test_hihat_and_floor_tom_cannot_cross():
    groups = assign_rcc_groups([_group(0.0, "hihat", "floor_tom")])
    by = {d: limb for _t, d, limb in _limbs(groups)}
    assert by["hihat"] == LIMB_LEFT
    assert by["floor_tom"] == LIMB_RIGHT


def test_kick_does_not_take_a_stick():
    groups = assign_rcc_groups([_group(0.0, "kick", "snare", "cymbal_right")])
    by = {d: limb for _t, d, limb in _limbs(groups)}
    assert by["kick"] == LIMB_FOOT
    assert by["snare"] == LIMB_LEFT
    assert by["cymbal_right"] == LIMB_RIGHT
    sticks = [limb for _t, _d, limb in _limbs(groups) if limb != LIMB_FOOT]
    assert len(sticks) == 2 and len(set(sticks)) == 2


def test_third_hand_drum_is_dropped():
    groups = assign_rcc_groups([_group(0.0, "hihat", "snare", "cymbal_right")])
    drums = {d for _t, d, _l in _limbs(groups)}
    assert len(drums) == 2
    assert "hihat" in drums  # closer to idle left than snare is
    assert "cymbal_right" in drums
