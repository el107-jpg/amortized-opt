"""CLI-level tests for the replay vector-gradient diagnostic."""

from __future__ import annotations

import pytest

from scripts.probe_row_grad_stratify import (
    _resolve_batch_sizes,
    parse_args,
)


class _Cfg:
    def batch_for(self, bucket: int) -> int:
        assert bucket == 25
        return 512


def test_batch_size_defaults_to_configured_training_size():
    assert _resolve_batch_sizes(_Cfg(), 25, None) == (512, 512)


def test_batch_size_override_is_diagnostic_only():
    assert _resolve_batch_sizes(_Cfg(), 25, 64) == (512, 64)
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_batch_sizes(_Cfg(), 25, 0)


def test_parse_batch_size_and_seed():
    args = parse_args([
        "--ckpt", "last.pt",
        "--pool-cache", "pool.npz",
        "--batch-size", "64",
        "--seed", "3",
    ])
    assert args.batch_size == 64
    assert args.seed == 3
