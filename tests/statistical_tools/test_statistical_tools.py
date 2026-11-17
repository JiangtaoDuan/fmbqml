"""Tests for statistical_tools test_statistical_tools."""

import numpy as np

import pytest

from fmbqml.statistical_tools.brownian import simulate_brownian_motion

import fmbqml.statistical_tools.hac as hac_module

from fmbqml.statistical_tools.hac import hac_newey_west_94

from fmbqml.statistical_tools import select_critical_value

from fmbqml.statistical_tools.critical_values import (
    bridge_score_matrix,
    joint_sup_lr_cv,
    simulate_single_sup_lr_critical_values,
)


def test_brownian_reproducibility_and_original_rng_sequence():
    seed = 121
    n, d = 8, 3
    got = simulate_brownian_motion(1.0, n, d, np.random.default_rng(seed))
    ref_rng = np.random.default_rng(seed)
    increments = np.column_stack([ref_rng.standard_normal(d) for _ in range(n)])
    expected = np.cumsum(increments / np.sqrt(n), axis=1)
    np.testing.assert_array_equal(got, expected)


def test_brownian_rejects_a_grid_with_no_points():
    with pytest.raises(ValueError, match="no simulation points"):
        simulate_brownian_motion(0.01, 1, 1)

def test_hac_prewhitening_preserves_original_inverse_path(monkeypatch):
    X = np.arange(24, dtype=float).reshape(8, 3)
    e = np.ones((8, 1))
    betahat = 0.25 * np.eye(3)
    residuals = X.copy()
    monkeypatch.setattr(
        hac_module,
        "fit_var",
        lambda data, p_min, p_max, cons: (betahat, residuals),
    )

    result = hac_module.hac_newey_west_94(
        X, e, cons=0, kernel=0, a=4, pw=1, p_min=1, p_max=1, m=1.5
    )
    V = residuals.T @ residuals / residuals.shape[0]
    D = np.eye(3) - betahat
    expected = np.linalg.inv(D) @ V @ np.linalg.inv(D.T)
    np.testing.assert_array_equal(result, expected)


def test_hac_prewhitening_rejects_singular_matrix(monkeypatch):
    X = np.arange(16, dtype=float).reshape(8, 2)
    e = np.ones((8, 1))
    betahat = np.eye(2)
    residuals = X.copy()
    monkeypatch.setattr(
        hac_module,
        "fit_var",
        lambda data, p_min, p_max, cons: (betahat, residuals),
    )

    with pytest.raises(np.linalg.LinAlgError, match="[Ss]ingular"):
        hac_module.hac_newey_west_94(
            X, e, cons=0, kernel=0, a=4, pw=1, p_min=1, p_max=1, m=1.5
        )


def test_hac_four_cons_pw_combinations_real_var_path():
    rng = np.random.default_rng(20261110)
    T = 160
    z = rng.standard_normal((T, 2))
    x = np.empty_like(z)
    x[0] = z[0]
    for t in range(1, T):
        x[t] = np.array([[0.25, 0.03], [-0.02, -0.13]]) @ x[t - 1] + z[t]
    e = np.ones(T)
    outputs = []
    for cons in (0, 1):
        for pw in (0, 1):
            hac = hac_newey_west_94(x, e, cons, 1, 4, pw, 1, 1, 1.5)
            assert hac.shape == (2, 2)
            np.testing.assert_allclose(hac, hac.T, atol=1e-12)
            assert np.all(np.isfinite(hac))
            outputs.append(hac)
    assert len(outputs) == 4

def test_select_critical_value_function():
    cv = np.array([1.0, 2.0, 3.0])

    assert select_critical_value(cv, 0.10) == 1.0
    assert select_critical_value(cv, 0.05) == 2.0
    assert select_critical_value(cv, 0.01) == 3.0

    with pytest.raises(ValueError, match="alpha"):
        select_critical_value(cv, 0.2)


def test_bridge_scores_accept_singular_psd_covariance():
    grid = np.array([0., 0.1, -0.3, 0.2, 0.])
    bridge = np.vstack((grid, grid))
    scores = bridge_score_matrix(bridge, np.array([[1., 1.], [1., 1.]]), 1)
    assert np.isfinite(scores[0, 1])
    assert scores[0, 1] >= 0
    assert np.isneginf(scores[0, 0])


def test_critical_value_simulators_reproducible():
    omega = np.eye(1)
    first, draws1 = simulate_single_sup_lr_critical_values(
        omega, 0.2, n_sim=6, grid_size=20, random_state=91)
    second, draws2 = simulate_single_sup_lr_critical_values(
        omega, 0.2, n_sim=6, grid_size=20, random_state=91)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(draws1, draws2)
    j1, d1 = joint_sup_lr_cv(omega, 1, .2, 4, 12, 5)
    j2, d2 = joint_sup_lr_cv(omega, 1, .2, 4, 12, 5)
    np.testing.assert_array_equal(j1, j2)
    np.testing.assert_array_equal(d1, d2)
