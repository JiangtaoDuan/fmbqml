"""Deterministic numerical checks against independent references."""

import numpy as np
import pytest

from fmbqml.multiple_break.joint_testing import _normalize_factors, _run_joint_sup_lr
from fmbqml.statistical_tools.brownian import simulate_brownian_motion
from fmbqml.statistical_tools.critical_values import (
    simulate_single_sup_lr_critical_values,
)
from fmbqml.statistical_tools.hac import hac_newey_west_94
from tests.audit_numerical_verification import independent_hac_reference


def _reference_factors():
    rng = np.random.default_rng(541)
    innovations = rng.standard_normal((100, 2))
    x = np.empty_like(innovations)
    x[0] = innovations[0]
    for t in range(1, len(x)):
        x[t] = np.array([[0.35, 0.02], [0.01, -0.12]]) @ x[t - 1] + innovations[t]
    return x, rng.standard_normal((90, 2))


@pytest.mark.parametrize("cons", [0, 1])
@pytest.mark.parametrize("pw", [0, 1])
def test_hac_matches_independent_numpy_reference(cons, pw):
    x, _ = _reference_factors()
    actual = hac_newey_west_94(x, np.ones(len(x)), cons, 1, 4, pw, 1, 1, 1.5)
    expected = independent_hac_reference(x, cons, pw)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_brownian_motion_increment_scaling():
    seed, grid_size, dimension = 181, 20, 2
    motion = simulate_brownian_motion(1, grid_size, dimension, np.random.default_rng(seed))
    increments = np.diff(np.column_stack((np.zeros(dimension), motion)), axis=1)
    rng = np.random.default_rng(seed)
    expected = np.array([rng.standard_normal(dimension) for _ in range(grid_size)]).T
    np.testing.assert_allclose(increments * np.sqrt(grid_size), expected, rtol=1e-13, atol=1e-13)


def test_joint_lr_and_simulated_critical_values_against_reference():
    _, factors = _reference_factors()
    result = _run_joint_sup_lr(
        _normalize_factors(factors), n_breaks=1, trim_ratio=0.2,
        alpha=0.05, n_sim=6, grid_size=20, random_state=73,
    )
    assert result["breakpoints"] == [40]
    assert result["test_statistic"] == pytest.approx(9.917383631547388, abs=1e-8)
    assert result["critical_value"] == pytest.approx(13.66295571090166, abs=1e-8)
    assert not result["reject_null"]
    cv, simulated = simulate_single_sup_lr_critical_values(
        np.eye(1), trim_ratio=0.3, n_sim=5, grid_size=60, random_state=19,
    )
    np.testing.assert_allclose(cv, [1.5717881900938602] * 3, rtol=1e-9, atol=1e-10)
    assert simulated[0] == pytest.approx(0.4157258398890229, abs=1e-10)
