"""Train minibatch size invariants (``batch_for`` vs drawn tensors)."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.train_cost import Phase2Config, _require_train_batch_size


def test_batch_for_rhythm_buckets_default():
    cfg = Phase2Config(batch_size=512)
    assert cfg.batch_for(25) == 512
    assert cfg.batch_for(50) == 512


def test_batch_for_token_budget_caps_long_horizon():
    cfg = Phase2Config(batch_size=512, token_budget=512 * 50)
    assert cfg.batch_for(100) == 256
    assert cfg.batch_for(50) == 512


def test_batch_for_disabled_token_budget():
    cfg = Phase2Config(batch_size=512, token_budget=0)
    assert cfg.batch_for(100) == 512


def test_require_train_batch_size_ok():
    b = 8
    batch = {"q0": torch.zeros(b, 17)}
    assert _require_train_batch_size(batch, b, 50) == b


def test_require_train_batch_size_raises():
    batch = {"q0": torch.zeros(7, 17)}
    with pytest.raises(RuntimeError, match="batch size mismatch"):
        _require_train_batch_size(batch, 8, 25)
