"""TensorBoard record flattening."""

from amortized_opt.tensorboard_log import MetricsTensorboard, scalars_from_record


def test_scalars_from_train_record():
    rec = {
        "event": "train",
        "step": 50,
        "cost": 100.0,
        "c_torque": 2.5,
        "v_torque_waist": 0.1,
        "d_pos_err": 0.02,
        "lam_i": {"torque_waist": 1.2, "torque_shoulder": 0.5},
        "lr": 1e-4,
        "timing_critic": {
            "loss": 1.7,
            "entropy_ood": 0.91,
            "calibration_precision": 0.8,
            "curriculum_open": 1.0,
        },
        "roller": {
            "timing_boundary_accepted": 3.0,
            "timing_mpc_refresh": 7.0,
            "unrelated": 99.0,
        },
        "cfg": {"max_steps": 20000},
    }
    pairs = scalars_from_record(rec)
    tags = {p[0] for p in pairs}
    assert "loss/cost" in tags
    assert "terms/torque" in tags
    assert "violations/torque_waist" in tags
    assert "al/lambda/torque_waist" in tags
    assert "diagnostics/hit/d_pos_err" in tags
    assert "optimizer/lr" in tags
    assert "timing_critic/loss" in tags
    assert "timing_critic/entropy_ood" in tags
    assert "timing_critic/calibration_precision" in tags
    assert "timing_critic/curriculum_open" in tags
    assert "timing_curriculum/boundary_accepted" in tags
    assert "timing_curriculum/mpc_refresh" in tags
    assert "timing_curriculum/unrelated" not in tags
    assert "train/cost" not in tags
    assert all(p[2] == 50 for p in pairs)


def test_scalars_from_eval_record():
    rec = {
        "event": "eval",
        "step": 2000,
        "accepted": 40,
        "n": 81,
        "ok_hit": 0.85,
        "acceptance": 0.49,
        "contact_pos_err_mm_p50": 12.3,
    }
    pairs = scalars_from_record(rec)
    tags = {p[0] for p in pairs}
    assert "eval/rates/ok_hit" in tags
    assert "eval/counts/accepted" in tags
    assert "eval/errors/contact_pos_err_mm_p50" in tags


def test_scalars_from_probe_record():
    pairs = scalars_from_record({
        "event": "probe", "step": 1000, "score": 0.42, "hits": 3,
    })
    assert ("probe/score", 0.42, 1000) in pairs


def test_metrics_tensorboard_noop_when_disabled(tmp_path):
    tb = MetricsTensorboard(tmp_path / "tb", enabled=False)
    tb.log_record({"event": "train", "step": 1, "cost": 1.0})
    tb.close()
    assert not (tmp_path / "tb").exists()
