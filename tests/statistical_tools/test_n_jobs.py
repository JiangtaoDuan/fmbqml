"""Regression tests for Monte Carlo worker-count control."""

import numpy as np
import pytest

from fmbqml.statistical_tools.critical_values import (
    _validate_n_jobs,
    joint_sup_lr_cv,
    simulate_single_sup_lr_critical_values,
)


@pytest.mark.parametrize("workers", [-1, 1, 2, 4, np.int64(2)])
def test_n_jobs_accepts_supported_values(workers):
    assert _validate_n_jobs(workers) == int(workers)


@pytest.mark.parametrize("workers", [0, -2])
def test_n_jobs_rejects_invalid_values(workers):
    with pytest.raises(ValueError, match="n_jobs"):
        _validate_n_jobs(workers)


@pytest.mark.parametrize("workers", [None, 1.5, "2", True, np.bool_(True)])
def test_n_jobs_rejects_invalid_types(workers):
    with pytest.raises(TypeError, match="n_jobs"):
        _validate_n_jobs(workers)


def test_single_break_simulation_is_worker_invariant():
    kwargs = {
        "omega": np.eye(1),
        "trim_ratio": 0.2,
        "n_sim": 8,
        "grid_size": 30,
        "random_state": 42,
    }
    cv1, sim1 = simulate_single_sup_lr_critical_values(**kwargs, n_jobs=1)
    cv2, sim2 = simulate_single_sup_lr_critical_values(**kwargs, n_jobs=2)
    np.testing.assert_array_equal(sim1, sim2)
    np.testing.assert_array_equal(cv1, cv2)


def test_joint_simulation_is_worker_invariant():
    kwargs = {
        "omega": np.eye(1),
        "n_breaks": 2,
        "trim_ratio": 0.15,
        "n_sim": 8,
        "grid_size": 30,
        "random_state": 42,
    }
    cv1, sim1 = joint_sup_lr_cv(**kwargs, n_jobs=1)
    cv2, sim2 = joint_sup_lr_cv(**kwargs, n_jobs=2)
    np.testing.assert_array_equal(sim1, sim2)
    np.testing.assert_array_equal(cv1, cv2)
