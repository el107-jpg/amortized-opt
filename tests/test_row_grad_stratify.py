"""Unit tests for per-row torque grad stratification helpers."""

import numpy as np
import pytest

from amortized_opt.train_cost import (
    _stratified_row_indices,
    summarize_torque_grad_stratify,
)


def test_summarize_torque_grad_top_tail_dominates():
    viol = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], float)
    norms = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0], float)
    rep = summarize_torque_grad_stratify(
        norms, {"viol": viol}, n_bins=4, top_fracs=(0.25,),
    )
    top = rep["strata"]["viol"]["tails"]["top_25pct"]
    assert top["k"] == 2
    assert top["energy_frac"] > 0.4


def test_stratified_row_indices_covers_quartiles():
    viol = np.arange(100, dtype=float)
    idx = _stratified_row_indices(viol, 40, n_bins=4, seed=0)
    assert idx.shape[0] == 40
    assert idx.min() >= 0
    assert idx.max() < 100
