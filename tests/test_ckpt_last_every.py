"""Checkpoint cadence helpers: atomic last.pt and shutdown flag."""

from __future__ import annotations

import signal
from pathlib import Path

import torch

from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.model import AmortizedOptimizer, ModelConfig
from amortized_opt.train_cost import (
    Phase2Config,
    _ShutdownFlag,
    _save_step_pt,
    _step_ckpt_path,
    save_phase2_checkpoint,
)


def test_save_phase2_checkpoint_atomic(tmp_path: Path):
    cfg = Phase2Config(device="cpu", max_steps=10)
    model_cfg = ModelConfig(
        d_model=32, dec_layers=1, ctx_layers=1, max_horizon=25, num_drums=7,
    )
    model = AmortizedOptimizer(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    al = AugmentedLagrangian()

    path = tmp_path / "last.pt"
    save_phase2_checkpoint(
        path,
        model=model,
        optimizer=opt,
        al=al,
        step=501,
        cfg=cfg,
        model_cfg=model_cfg,
        atomic=True,
    )
    assert path.is_file()
    assert not path.with_name("last.pt.tmp").exists()
    ck = torch.load(path, weights_only=False)
    assert ck["step"] == 501


def test_step_ckpt_path():
    assert _step_ckpt_path(Path("/runs/foo"), 4000) == Path("/runs/foo/step_004000.pt")


def test_save_step_pt_respects_keep_every(tmp_path: Path):
    cfg = Phase2Config(device="cpu", max_steps=10, ckpt_keep_every=500)
    model_cfg = ModelConfig(
        d_model=32, dec_layers=1, ctx_layers=1, max_horizon=25, num_drums=7,
    )
    model = AmortizedOptimizer(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    al = AugmentedLagrangian()

    assert _save_step_pt(
        tmp_path, model=model, optimizer=opt, al=al, step=500,
        cfg=cfg, model_cfg=model_cfg, scenario_cfg=None, ls_rt=None,
    ) == tmp_path / "step_000500.pt"
    assert (tmp_path / "step_000500.pt").is_file()

    cfg_off = Phase2Config(device="cpu", max_steps=10, ckpt_keep_every=0)
    assert _save_step_pt(
        tmp_path, model=model, optimizer=opt, al=al, step=1000,
        cfg=cfg_off, model_cfg=model_cfg, scenario_cfg=None, ls_rt=None,
    ) is None
    assert not (tmp_path / "step_001000.pt").exists()


def test_shutdown_flag_install():
    _ShutdownFlag.requested = False
    _ShutdownFlag.install()
    signal.raise_signal(signal.SIGTERM)
    assert _ShutdownFlag.requested
