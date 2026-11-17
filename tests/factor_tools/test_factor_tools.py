"""Tests for factor_tools test_factor_tools."""

import numpy as np

import pytest

from fmbqml.factor_tools.var_selection import fit_var


def test_pca_two_paths_same_projection_space():
    from fmbqml.factor_tools.panel_factor import estimate_panel_factors
    rng = np.random.default_rng(82)
    for shape in ((75, 20), (20, 75)):
        x = rng.standard_normal(shape)
        f1, _, _ = estimate_panel_factors(x, 3)
        u, _, _ = np.linalg.svd(x, full_matrices=False)
        f2 = u[:, :3]
        q1, _ = np.linalg.qr(f1)
        q2, _ = np.linalg.qr(f2)
        np.testing.assert_allclose(q1 @ q1.T, q2 @ q2.T, atol=2e-12)

def test_fit_var_recovers_stable_ar1_coefficient():
    rng = np.random.default_rng(14)
    n = 600
    series = np.zeros((n, 1))
    for t in range(1, n):
        series[t, 0] = 0.45 * series[t - 1, 0] + rng.normal(scale=0.5)
    coef, resid = fit_var(series, p_min=1, p_max=1, cons=0)
    assert coef.shape == (1, 1)
    assert abs(coef[0, 0] - 0.45) < 0.13
    assert resid.shape == (n - 1, 1)


def test_fit_var_rejects_rank_deficient_design():
    x = np.ones((50, 2))
    with pytest.raises(np.linalg.LinAlgError):
        fit_var(x, 1, 1, 0)


def test_var_ar1_bic_selects_order_one_when_signal_is_strong():
    rng = np.random.default_rng(67)
    x = np.zeros((700, 1))
    x[0, 0] = 1
    for idx in range(1, len(x)):
        x[idx, 0] = .87 * x[idx-1, 0] + rng.normal(scale=.3)
    coefficients, residuals = fit_var(x, p_min=0, p_max=1, cons=0)
    assert coefficients.shape == (1, 1)
    assert residuals.shape[0] == len(x) - 1


def test_var_zero_lag_branch_with_intercept():
    rng = np.random.default_rng(81)
    x = 2.3 + rng.normal(size=(120, 1))
    coefficients, residuals = fit_var(x, p_min=0, p_max=0, cons=1)
    assert coefficients.shape == (1, 2)
    assert coefficients[0, 0] == pytest.approx(float(x.mean()))
    assert np.mean(residuals) == pytest.approx(0, abs=1e-14)
