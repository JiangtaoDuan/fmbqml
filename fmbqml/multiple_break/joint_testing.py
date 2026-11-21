"""Joint sup-LR test for a prespecified number of factor-model breaks.

This module implements the statistics and limiting distributions described by
Theorem 5 of the multiple-change extension of the factor-model LR test.
The supremum over admissible partitions is solved by dynamic programming.
"""


import numpy as np

import fmbqml.utils as U
from fmbqml.multiple_break.qml_optimization import find_min_qml
from fmbqml.statistical_tools import (
    factor_long_run_covariance,
    joint_sup_lr_cv,
    select_critical_value,
)


def _normalize_factors(F_hat: np.ndarray) -> np.ndarray:
    """Impose the full-sample PC normalization ``F'F/T = I``.

    Raises
    ------
    ValueError
        If the full-sample factor second-moment matrix is rank deficient.
    """
    gram = F_hat.T @ F_hat / len(F_hat)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    # A relative numerical-rank threshold is essential: rescaling all
    # factors must not turn a full-rank Gram matrix into a rank-deficient one.
    # This leaves the original threshold unchanged when the largest
    # eigenvalue is >= 1, and removes the artificial unit-scale floor.
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    if not np.isfinite(spectral_scale) or spectral_scale <= 0.0:
        raise ValueError("Estimated factors have no positive finite Gram spectrum.")
    tolerance = np.finfo(float).eps * max(gram.shape) * spectral_scale
    if np.min(eigenvalues) <= tolerance:
        raise ValueError("Estimated factors are rank deficient under full-sample normalization.")
    inverse_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    return F_hat @ inverse_sqrt


def _run_joint_sup_lr(
    F_hat: np.ndarray,
    n_breaks: int,
    trim_ratio: float = 0.15,
    alpha: float = 0.01,
    n_sim: int = 2000,
    grid_size: int = 500,
    random_state: int | None = None,
    n_jobs: int = -1,
) -> dict[str, object]:
    """Run the joint sup-LR test for a prespecified number of breaks.

    ``grid_size`` controls only the numerical approximation of the continuous
    Brownian-bridge supremum. The observed statistic is optimized on the full
    sample grid.

    Parameters
    ----------
    F_hat : ndarray of shape (T, r)
        Full-sample estimated factors.
    n_breaks : int
        Exact number of breaks under the alternative hypothesis.
    trim_ratio : float, default=0.15
        Minimum regime length as a fraction of the sample.
    alpha : {0.10, 0.05, 0.01}, default=0.01
        Significance level.
    n_sim : int, default=2000
        Number of Monte Carlo replications used for critical values.
    grid_size : int, default=500
        Brownian-bridge grid size used only in critical-value simulation.
    random_state : int or None, default=None
        Seed for reproducible critical-value simulation.
    n_jobs : int, default=-1
        Number of parallel Monte Carlo workers.

    Returns
    -------
    dict
        Test statistic, estimated breakpoints, critical values, rejection
        decision, objective values, and simulated limiting statistics.

    Notes
    -----
    This is an internal algorithm routine. The public model method validates
    user-facing options and the source panel before calling it. Direct callers
    are responsible for supplying finite real-valued factor estimates and
    valid scalar options; this routine retains structural and feasibility
    checks required by the statistic.

    The full-sample Newey--West HAC covariance is the same as in the
    single-break test (Theorem 1).  Critical values are generated from the
    multiple-break Brownian-bridge limit in Theorem 5: the simulator optimizes
    the sum of all ``n_breaks + 1`` segment-increment quadratic forms.
    """
    # The model API validates user-facing arguments and the original panel.
    # This internal routine receives estimated factors, so it only enforces
    # the structural and feasibility conditions required by the statistic.
    F_hat = np.asarray(F_hat, dtype=float)
    if F_hat.ndim != 2 or F_hat.shape[0] < 2 or F_hat.shape[1] < 1:
        raise ValueError("F_hat must be a non-empty two-dimensional factor matrix.")
    T = F_hat.shape[0]
    min_length = max(1, int(np.ceil(T * trim_ratio)))
    if (n_breaks + 1) * min_length > T:
        raise ValueError(
            "The requested number of breaks is infeasible under trim_ratio: "
            f"({n_breaks + 1}) * {min_length} > {T}."
        )
    grid_min_length = max(1, int(np.ceil(grid_size * trim_ratio)))
    if grid_size < 10 or (n_breaks + 1) * grid_min_length > grid_size:
        raise ValueError("grid_size is too small for n_breaks and trim_ratio.")

    F_hat = _normalize_factors(F_hat)

    alternative, breakpoint_array = find_min_qml(
        F_hat, T, min_length, n_breaks
    )
    breakpoints = [int(point) for point in breakpoint_array]
    null_value = T * U.safe_logdet(F_hat.T @ F_hat / T)
    statistic = max(0.0, null_value - alternative)

    omega = factor_long_run_covariance(F_hat)
    critical_values, simulated = joint_sup_lr_cv(
        omega,
        n_breaks,
        trim_ratio,
        n_sim,
        grid_size,
        random_state,
        n_jobs=n_jobs,
    )
    critical_value = select_critical_value(critical_values, alpha)

    return {
        "test_type": "joint_sup_lr",
        "n_breaks": n_breaks,
        "breakpoints": breakpoints,
        "test_statistic": statistic,
        "critical_value": critical_value,
        "critical_values": {
            0.10: float(critical_values[0]),
            0.05: float(critical_values[1]),
            0.01: float(critical_values[2]),
        },
        "reject_null": bool(statistic > critical_value),
        "significance_level": float(alpha),
        "trim_ratio": float(trim_ratio),
        "min_segment_length": min_length,
        "grid_size": grid_size,
        "monte_carlo_replications": n_sim,
        "random_state": random_state,
        "null_objective": null_value,
        "alternative_objective": alternative,
        "simulated_statistics": simulated,
    }


__all__ = []
