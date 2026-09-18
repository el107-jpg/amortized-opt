from __future__ import annotations

import math

import numpy as np
import torch

from amortized_opt.checkpoint_diagnostics import (
    adjacent_adam_report,
    analyze_adam_state,
    direction_alignment_report,
    parameter_module,
    tensor_mapping_delta_report,
)
from amortized_opt.constants import G1_UPPER_BODY_JOINTS
from amortized_opt.fk import StickRandomization
from amortized_opt.rhythm import KitRandomization
from amortized_opt.waist_aim import WaistAimConfig
from scripts.diagnose_loss_spike import _scenario_from_checkpoint


def test_parameter_delta_reports_overall_modules_and_shape_mismatches():
    before = {
        "encoder.layer.weight": torch.tensor([1.0, 2.0]),
        "decoder.head.weight": torch.tensor([3.0]),
        "decoder.old.weight": torch.ones(2),
        "shape.moves": torch.ones(2),
    }
    after = {
        "encoder.layer.weight": torch.tensor([2.0, 4.0]),
        "decoder.head.weight": torch.tensor([2.0]),
        "decoder.new.weight": torch.ones(2),
        "shape.moves": torch.ones(3),
    }

    report = tensor_mapping_delta_report(before, after, module_depth=2)

    assert parameter_module("decoder.blocks.0.weight", 2) == "decoder.blocks"
    assert report["overall"]["numel"] == 3
    assert math.isclose(report["overall"]["delta_l2"], math.sqrt(6.0))
    assert math.isclose(
        report["by_module"]["encoder.layer"]["delta_l2"], math.sqrt(5.0)
    )
    assert report["by_module"]["decoder.head"]["delta_l2"] == 1.0
    assert report["only_before"] == ["decoder.old.weight"]
    assert report["only_after"] == ["decoder.new.weight"]
    assert report["shape_mismatch"] == [
        {"name": "shape.moves", "before": [2], "after": [3]}
    ]


def _adam_fixture():
    parameters = {
        "weight": torch.tensor([1.0, -2.0]),
        "bias": torch.tensor([0.5]),
        "token_emb": torch.tensor([0.25]),
    }
    order = ["weight", "bias", "token_emb"]
    optimizer = {
        "param_groups": [
            {
                "params": [0],
                "lr": 0.1,
                "betas": (0.5, 0.75),
                "eps": 0.0,
                "weight_decay": 0.2,
            },
            {
                "params": [1, 2],
                "lr": 0.1,
                "betas": (0.5, 0.75),
                "eps": 0.0,
                "weight_decay": 0.0,
            },
        ],
        "state": {
            0: {
                "step": torch.tensor(2.0),
                "exp_avg": torch.tensor([0.75, -1.5]),
                "exp_avg_sq": torch.tensor([1.75, 7.0]),
            },
            1: {
                "step": torch.tensor(2.0),
                "exp_avg": torch.tensor([0.75]),
                "exp_avg_sq": torch.tensor([1.75]),
            },
            2: {
                "step": torch.tensor(2.0),
                "exp_avg": torch.tensor([-0.75]),
                "exp_avg_sq": torch.tensor([1.75]),
            },
        },
    }
    return parameters, order, optimizer


def test_adam_analysis_bias_correction_decay_and_coverage():
    parameters, order, optimizer = _adam_fixture()
    analysis = analyze_adam_state(optimizer, order, parameters, module_depth=1)

    # At step 2: m_hat=[1,-2], v_hat=[4,16], so direction=[0.5,-0.5].
    assert torch.allclose(
        analysis.direction["weight"], torch.tensor([0.5, -0.5])
    )
    # AdamW proposal is -lr * (direction + weight_decay * parameter).
    assert torch.allclose(
        analysis.proposed_update["weight"], torch.tensor([-0.07, 0.09])
    )
    assert torch.allclose(
        analysis.proposed_update["bias"], torch.tensor([-0.05])
    )
    assert analysis.report["available"]
    assert analysis.report["coverage"] == 1.0
    assert analysis.report["step"] == {"min": 2.0, "mean": 2.0, "max": 2.0}
    assert analysis.report["second_moment"]["negative_count"] == 0


def test_effective_update_alignment_and_adjacent_moment_report():
    parameters, order, optimizer = _adam_fixture()
    analysis = analyze_adam_state(optimizer, order, parameters)
    after = {
        name: value + analysis.proposed_update[name]
        for name, value in parameters.items()
    }

    alignment = direction_alignment_report(
        parameters, after, analysis.proposed_update
    )
    assert math.isclose(alignment["overall"]["cosine"], 1.0, abs_tol=1e-7)
    assert math.isclose(
        alignment["overall"]["best_fit_scale"], 1.0, abs_tol=5e-7
    )

    pair = adjacent_adam_report(parameters, after, analysis, analysis)
    assert pair["exp_avg_delta"]["overall"]["delta_l2"] == 0.0
    assert math.isclose(
        pair["before_effective_update_alignment"]["overall"]["cosine"],
        1.0,
        abs_tol=1e-7,
    )


def test_absent_optimizer_state_is_reported_not_raised():
    parameters = {"weight": torch.ones(2)}
    analysis = analyze_adam_state(None, ["weight"], parameters)
    assert not analysis.report["available"]
    assert analysis.report["mapped_state_tensors"] == 0
    assert "absent" in analysis.report["warnings"][0]


def test_scenario_loader_rebuilds_nested_config_and_uses_pool_shapes():
    ck = {
        "scenario_cfg": {
            "horizon": 100,
            "buckets": [25, 50, 100],
            "joint_names": list(G1_UPPER_BODY_JOINTS),
            "kit_randomization": {"per_drum_tilt": 0.1},
            "waist_cfg": {"future_tau": 0.75, "future_max": 3},
            "stick_rand": {"tilt": 0.02, "roll": 0.03, "length": 0.01},
        }
    }
    arrays = {
        "drum_pos": np.zeros((4, 7, 3), np.float32),
        "future_pos": np.zeros((4, 2, 1, 3), np.float32),
    }

    cfg = _scenario_from_checkpoint(ck, (25, 50), arrays)

    assert cfg.horizon == 50
    assert cfg.buckets == (25, 50)
    assert cfg.num_drums == 7
    assert cfg.num_future == 1
    assert isinstance(cfg.kit_randomization, KitRandomization)
    assert isinstance(cfg.waist_cfg, WaistAimConfig)
    assert isinstance(cfg.stick_rand, StickRandomization)
    assert cfg.waist_cfg.future_tau == 0.75
