"""Replay-freezing contract for critic-selected timing batches."""

from types import SimpleNamespace

import torch

from amortized_opt.objective_replay import FrozenControls, ObjectiveReplayContext
from amortized_opt.train_cost import _refresh_timing_line_search_replay


def test_refresh_timing_replay_uses_retimed_seed_only_for_fresh_rows():
    gen = torch.Generator(device="cpu").manual_seed(7)
    mgr = ObjectiveReplayContext({"gen": gen}, capture_global=False)
    original_init = torch.tensor([
        [[10.0], [11.0]],
        [[20.0], [21.0]],
        [[30.0], [31.0]],
    ])
    batch = {
        "init": original_init.clone(),
        "seed": torch.zeros_like(original_init),
        "state": torch.zeros(3, 2, 1),
        "mpc_init": torch.tensor([False, True, False]),
        "masks": SimpleNamespace(elem_mask=torch.ones_like(original_init)),
    }
    rec = mgr.begin_record(
        primary_batch=batch,
        controls=FrozenControls(step=0),
        snapshot_mutable=True,
    )

    # Mimic timing assist committing new raw seeds after the baseline's first
    # initializer rewrite. MPC row 1 was not retimed and must keep its shifted
    # initializer from the original replay snapshot.
    batch["seed"] = torch.tensor([
        [[1.0], [2.0]],
        [[3.0], [4.0]],
        [[5.0], [6.0]],
    ])
    batch["init"] = batch["seed"] + 100.0
    _refresh_timing_line_search_replay(rec, batch)

    assert torch.equal(rec.mutable_pre["seed"], batch["seed"])
    assert torch.equal(rec.mutable_pre["init"][0], batch["seed"][0])
    assert torch.equal(rec.mutable_pre["init"][1], original_init[1])
    assert torch.equal(rec.mutable_pre["init"][2], batch["seed"][2])


def test_refresh_timing_replay_respects_final_decoder_mask():
    gen = torch.Generator(device="cpu").manual_seed(8)
    mgr = ObjectiveReplayContext({"gen": gen}, capture_global=False)
    mask = torch.tensor([[[1.0], [0.0]]])
    batch = {
        "init": torch.full((1, 2, 1), 9.0),
        "seed": torch.zeros(1, 2, 1),
        "state": torch.zeros(1, 2, 1),
        "mpc_init": torch.tensor([False]),
        "masks": SimpleNamespace(elem_mask=mask),
    }
    rec = mgr.begin_record(primary_batch=batch, snapshot_mutable=True)
    batch["seed"] = torch.tensor([[[2.0], [3.0]]])
    _refresh_timing_line_search_replay(rec, batch)
    assert torch.equal(rec.mutable_pre["init"], torch.tensor([[[2.0], [0.0]]]))
