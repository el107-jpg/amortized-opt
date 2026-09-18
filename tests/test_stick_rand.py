"""Stick randomization plumbing: cost fallback, bounds, package exports.

Adversarial-review regressions:

* ``cost.stick_cost_from_context`` promised a nominal-geometry fallback for
  rows without a stick but only delivered it for the axis and tip sigma -- the
  POINTS of an invalid row collapsed to a zero-length stub at the wrist, so a
  single valid row poisoned every invalid row in the batch.
* ``StickRandomization`` rejected ``length >= STICK_LENGTH`` but the real
  invariant is that the shortest drawable stick keeps
  ``tip_sigma - SHAFT_TIP_CLEAR > butt_sigma`` (total length
  ``> SHAFT_TIP_CLEAR``), or ``stick_shaft_offsets``'s linspace inverts.
* ``amortized_opt`` exported ``StickSpec`` but not ``StickRandomization``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import SHAFT_TIP_CLEAR, STICK_LENGTH
from amortized_opt.cost import stick_cost_from_context
from amortized_opt.fk import (
    StickRandomization,
    StickSpec,
    stick_axis_local,
    stick_shaft_offsets,
    stick_tip_offset,
)

DT = torch.float64
HANDS = ("left", "right")
N_SHAFT = 6


def _context_columns(specs_per_row):
    """The pool's four stick columns for ``[(left_spec|None, right_spec|None)]``."""
    B = len(specs_per_row)
    ax = np.zeros((B, 2, 3))
    tip = np.zeros((B, 2, 3))
    ln = np.zeros((B, 2))
    valid = np.zeros((B, 2), bool)
    for b, pair in enumerate(specs_per_row):
        for hi, (hand, sp) in enumerate(zip(HANDS, pair)):
            if sp is None:
                continue
            ax[b, hi] = stick_axis_local(hand, sp)
            tip[b, hi] = stick_tip_offset(hand, sp)
            ln[b, hi] = sp.length
            valid[b, hi] = True
    t = lambda a: torch.as_tensor(a, dtype=DT)  # noqa: E731
    return t(ax), t(tip), t(ln), torch.as_tensor(valid)


def _nominal_points(hand):
    return np.concatenate(
        [stick_tip_offset(hand)[None, :], stick_shaft_offsets(hand, N_SHAFT),
         np.zeros((1, 3))], axis=0
    )


def _spec_points(hand, sp):
    return np.concatenate(
        [stick_tip_offset(hand, sp)[None, :], stick_shaft_offsets(hand, N_SHAFT, sp),
         np.zeros((1, 3))], axis=0
    )


# ---------------------------------------------- 3. invalid-row point fallback
def test_invalid_rows_get_the_nominal_points():
    """A mixed batch: the invalid row's POINTS equal the nominal stick's."""
    sp_l = StickSpec("left", tilt=0.05, roll=-0.03, length=STICK_LENGTH + 0.04)
    sp_r = StickSpec("right", tilt=-0.04, length=STICK_LENGTH - 0.02)
    rows = [(sp_l, sp_r), (None, None), (sp_l, None)]
    pts, axis, sig = stick_cost_from_context(*_context_columns(rows), N_SHAFT)

    for b, pair in enumerate(rows):
        for hi, (hand, sp) in enumerate(zip(HANDS, pair)):
            want = _nominal_points(hand) if sp is None else _spec_points(hand, sp)
            got = pts[b, hi].numpy()
            assert np.abs(got - want).max() < 1e-9, (b, hand)
            want_ax = stick_axis_local(hand, sp)
            assert np.abs(axis[b, hi].numpy() - want_ax).max() < 1e-12
            want_sig = (StickSpec(hand) if sp is None else sp).tip_sigma
            assert abs(float(sig[b, hi]) - want_sig) < 1e-12


def test_no_stick_anywhere_returns_none_triple():
    _ax, _tip, _ln, _valid = _context_columns([(None, None)])
    assert stick_cost_from_context(_ax, _tip, _ln, _valid, N_SHAFT) == (None, None, None)
    assert stick_cost_from_context(_ax, _tip, _ln, None, N_SHAFT) == (None, None, None)


# ------------------------------------------------------ 7. length invariant
def test_length_bound_is_the_shaft_invariant():
    limit = STICK_LENGTH - SHAFT_TIP_CLEAR
    with pytest.raises(ValueError, match="SHAFT_TIP_CLEAR"):
        StickRandomization(length=limit)
    with pytest.raises(ValueError, match="SHAFT_TIP_CLEAR"):
        StickRandomization(length=limit + 0.01)
    ok = StickRandomization(length=limit - 1e-6)
    # the shortest drawable stick must keep the shaft linspace monotone
    sp = StickSpec("left", length=STICK_LENGTH - ok.length)
    z = stick_shaft_offsets("left", N_SHAFT, sp)
    d = np.diff(z @ stick_axis_local("left", sp))
    assert (d > 0).all(), "shaft samples must walk butt -> tip"


def test_negative_bounds_still_rejected():
    for k in ("tilt", "roll", "length"):
        with pytest.raises(ValueError, match=">= 0"):
            StickRandomization(**{k: -0.1})


# ------------------------------------------------------------- 6. exports
def test_package_exports():
    assert A.StickRandomization is StickRandomization
    assert A.StickSpec is StickSpec
    assert "StickRandomization" in A.__all__
    from amortized_opt.tokenizer import STICK_RAW_DIM, stick_features
    assert A.STICK_RAW_DIM == STICK_RAW_DIM
    assert A.stick_features is stick_features
