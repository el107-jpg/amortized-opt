import json
import numpy as np
import torch

from scripts.probe_coarse_transition_qp import load_method_result, save_method_result


def test_verified_probe_results_save_a_real_witness_and_can_be_resumed(tmp_path):
    q = torch.zeros(1, 17)
    trajectory = torch.ones(1, 9, 17)
    result = dict(method='fine20', minimum_s=.18, final_metrics={'success': True})
    save_method_result(tmp_path, 2, 'fine20', result, q, q, trajectory)
    with np.load(tmp_path/'case_2_fine20.npz') as saved:
        np.testing.assert_array_equal(saved['trajectory'], trajectory.numpy())
        assert float(saved['duration_s']) == .18
    restored = load_method_result(tmp_path, 2, 'fine20')
    assert restored['witness_saved'] and restored['minimum_s'] == .18
    assert load_method_result(tmp_path, 2, 'coarse60_refine20') is None
    # Measurements from the original failed witness-save remain valid, but
    # resuming must explicitly distinguish them from saved path artifacts.
    (tmp_path/'case_2_fine20.npz').unlink()
    assert not load_method_result(tmp_path, 2, 'fine20')['witness_saved']
    assert json.loads((tmp_path/'case_2_fine20.json').read_text())['final_metrics']['success']
