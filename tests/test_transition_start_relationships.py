import numpy as np

from scripts.plot_transition_start_relationships import linear_fit


def test_exact_linear_relation_predicts_held_out_starts_and_ignores_unknowns():
    x = np.arange(8.)
    fit = linear_fit(np.append(x, 99), np.append(120+5*x, np.inf))
    assert fit['n'] == 8
    np.testing.assert_allclose([fit['slope'], fit['intercept']], [5, 120])
    assert fit['r2'] == 1
    assert fit['linear_loo_rmse_ms'] < 1e-9
    assert fit['constant_loo_rmse_ms'] > 0


def test_curvature_can_be_distinguished_using_held_out_prediction():
    x = np.arange(8.)
    fit = linear_fit(x, 100+x*x)
    assert fit['linear_loo_rmse_ms'] > 1
    assert fit['quadratic_loo_rmse_ms'] < 1e-9


def test_tiny_samples_and_constant_predictors_do_not_claim_linearity():
    assert linear_fit([1, 2], [100, 200])['r2'] is None
    fit = linear_fit([1]*5, [100, 200, 300, 400, 500])
    assert fit['slope'] is None and fit['linear_loo_rmse_ms'] is None
    assert linear_fit(np.arange(8.), np.ones(8))['r2'] is None


def test_high_training_r2_can_still_fail_to_beat_a_constant_predictor():
    # An isolated extreme start dominates the fit but cannot be predicted
    # from the remaining starts when it is held out.
    fit = linear_fit([0, 1, 2, 3, 20], [0, 10, -10, 0, 100])
    assert fit['r2'] > .9
    assert fit['linear_loo_rmse_ms'] > fit['constant_loo_rmse_ms']
