"""Guard the perturbation controls used to attribute decoder plan failures."""
import numpy as np
import pytest
import torch

from scripts.probe_mpc_plan_drift import masked_rms, phase_lag, summarize, unit_drift


@pytest.mark.parametrize("mode", ["ou", "anchored_ou", "smooth_bias"])
def test_drift_has_matched_rms_and_respects_per_joint_horizons(mode):
    init = torch.zeros(2, 7, 3)
    mask = torch.ones_like(init)
    mask[0, 1:, 0] = 0
    mask[1, 3:, 1:] = 0
    drift = unit_drift(init, mask, mode, torch.Generator().manual_seed(9))
    assert torch.equal(drift[mask == 0], torch.zeros_like(drift[mask == 0]))
    assert torch.allclose(masked_rms(drift, mask), torch.ones(2))
    assert torch.equal(init, torch.zeros_like(init))


def test_phase_lag_is_identity_at_zero_and_uses_measured_boundary():
    init = torch.tensor([[[2., 3.], [4., 5.], [6., 7.]]])
    q0 = torch.tensor([[0., 1.]])
    mask = torch.ones_like(init)
    mask[:, -1, 1] = 0
    init = init * mask
    assert torch.equal(phase_lag(init, q0, mask, 0), init)
    lag = phase_lag(init, q0, mask, .5)
    assert torch.equal(lag[0, 0], torch.tensor([1., 2.]))
    assert lag[0, -1, 1] == 0
    assert torch.equal(phase_lag(init, q0, mask, 5)[0, 0], q0[0])


def test_recovery_denominator_excludes_already_bad_clean_inputs():
    from scripts.probe_mpc_plan_drift import DIAGNOSTICS

    def values(flags):
        data = {k: np.zeros(3) for k in DIAGNOSTICS}
        data.update(cost=np.zeros(3))
        data.update({k: np.array(flags) for k in
                     ("on_head", "within_5mm", "geometric_ok", "hard_ok")})
        return data

    result = summarize(values([False, False, True]), values([True, False, True]),
                       values([True, False, True]), values([True, False, True]),
                       np.ones(3, dtype=bool), np.ones(3), np.ones(3))
    assert result["damaged_on_head_n"] == 1
    assert result["repair_damaged_on_head_rate"] == 1
    assert result["preserve_clean_input_on_head_rate"] == 1
    assert result["preserve_clean_output_on_head_rate"] == 1
    empty = summarize(values([True]*3), values([True]*3), values([True]*3),
                      values([True]*3), np.zeros(3, dtype=bool),
                      np.ones(3), np.ones(3))
    assert empty["output_on_head_rate"] is None
    assert empty["repair_damaged_on_head_rate"] is None
