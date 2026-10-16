"""Critical-value simulation and HAC helpers for factor-model LR tests.

The single-break and joint multiple-break tests share the same HAC estimator
for the long-run covariance of ``vec(F_t F_t' - I)`` but have different
limiting sup-LR statistics.  Consequently, this module provides separate
simulation routines:

``simulate_single_sup_lr_critical_values``
    Maximizes over one candidate break fraction.
``joint_sup_lr_cv``
    Jointly optimizes a partition with a prespecified number of breaks.

``critical_value_lr_hac_m`` is the high-level single-break convenience
function used by :class:`~fmbqml.single_break.SingleBreakQML`.  The
multiple-break test estimates the HAC matrix itself and passes it to
``joint_sup_lr_cv`` so that the positive-semidefinite projection required by
that implementation remains explicit.
"""

from numbers import Real
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from fmbqml.statistical_tools.brownian import simulate_brownian_motion
from fmbqml.statistical_tools.hac import hac_newey_west_94
from fmbqml.utils.partition import optimize_additive_partition


DEFAULT_CRITICAL_LEVELS = (0.90, 0.95, 0.99)


def factor_product_deviations(F_hat: np.ndarray) -> np.ndarray:
    """Construct centered factor-product vectors for HAC estimation.

    Parameters
    ----------
    F_hat : ndarray of shape (T, r)
        Estimated factors, with observations in rows.

    Returns
    -------
    ndarray of shape (T, r**2)
        Row ``t`` contains ``vec(F_t F_t' - I_r)`` under the package's
        column-wise vectorization convention.

    Raises
    ------
    ValueError
        If ``F_hat`` is not two-dimensional.
    """
    F_hat = np.asarray(F_hat, dtype=float)
    if F_hat.ndim != 2:
        raise ValueError("F_hat must be a 2D array.")

    T, r = F_hat.shape
    products = F_hat[:, :, None] * F_hat[:, None, :] - np.eye(r)
    return products.reshape(T, r * r, order="F")


def factor_long_run_covariance(F_hat: np.ndarray, project_psd: bool = False) -> np.ndarray:
    """Estimate the HAC long-run covariance of factor-product deviations.

    Parameters
    ----------
    F_hat : ndarray of shape (T, r)
        Estimated factors.
    project_psd : bool, default=False
        If True, replace negative eigenvalues of the symmetrized HAC estimate
        by zero.  The joint multiple-break simulator uses this option to avoid
        negative quadratic-form scores caused by finite-sample estimation.

    Returns
    -------
    ndarray of shape (r**2, r**2)
        Symmetric Newey-West HAC estimate for
        ``vec(F_t F_t' - I_r)``.
    """
    vectorized = factor_product_deviations(F_hat)
    T = vectorized.shape[0]
    omega = hac_newey_west_94(
        vectorized,
        np.ones((T, 1)),
        0,
        1,
        4,
        0,
        1,
        1,
        1.5,
    )
    omega = 0.5 * (omega + omega.T)

    if project_psd:
        eigenvalues, eigenvectors = np.linalg.eigh(omega)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return omega


def _validate_n_sim(n_sim: int) -> int:
    if isinstance(n_sim, bool) or not isinstance(n_sim, int) or n_sim <= 0:
        raise ValueError(f"n_sim must be a positive integer. Got {n_sim}.")
    return n_sim


def _critical_values_from_simulated(
    simulated: np.ndarray,
    levels=DEFAULT_CRITICAL_LEVELS,
    use_ceiling: bool = True,
) -> np.ndarray:
    ordered = np.sort(np.asarray(simulated, dtype=float))
    n_sim = len(ordered)
    indices = []
    for level in levels:
        raw_index = np.ceil(level * n_sim) if use_ceiling else int(level * n_sim)
        indices.append(max(0, int(raw_index) - 1))
    return ordered[indices]


def simulate_single_sup_lr_critical_values(
    omega: np.ndarray,
    trim_ratio: float,
    n_sim: int = 5000,
    grid_size: int = 2000,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the limiting single-break sup-LR distribution.

    For every Monte Carlo draw, the routine constructs a Brownian bridge and
    maximizes its LR quadratic form over break fractions in
    ``[trim_ratio, 1 - trim_ratio]``.  This is the one-break distribution; it
    does not optimize over multiple-segment partitions.

    Parameters
    ----------
    omega : ndarray of shape (d, d)
        HAC long-run covariance matrix of the factor-product deviations.
    trim_ratio : float
        Fraction trimmed from each end of the unit interval.
    n_sim : int, default=5000
        Number of Monte Carlo replications.
    grid_size : int, default=2000
        Number of intervals used to approximate the unit interval.
    random_state : int or None, default=None
        Seed for reproducible simulation.

    Returns
    -------
    critical_values : ndarray of shape (3,)
        Simulated 10%, 5%, and 1% significance-level critical values, in that
        order.
    simulated_statistics : ndarray of shape (n_sim,)
        Sup-LR statistic from every Monte Carlo replication.
    """
    n_sim = _validate_n_sim(n_sim)
    omega = np.asarray(omega, dtype=float)
    dimension = omega.shape[0]
    base_rng = np.random.default_rng(random_state)
    seeds = base_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_sim,
        dtype=np.uint32,
    ).tolist()

    h_grid = int(np.ceil(grid_size * trim_ratio))
    j_min = h_grid
    j_max = grid_size - h_grid

    def monte_carlo_iteration(seed):
        rng = np.random.default_rng(int(seed))
        motion = simulate_brownian_motion(1, grid_size, dimension, rng=rng)
        terminal = motion[:, grid_size - 1].reshape(-1, 1)
        values = []
        for j in range(j_min, j_max + 1):
            fraction = j / grid_size
            bridge = motion[:, j - 1].reshape(-1, 1) - fraction * terminal
            statistic = (
                bridge.T
                @ omega
                @ bridge
                / (2 * fraction * (1 - fraction))
            )
            values.append(float(statistic[0, 0]))
        return max(values)

    simulated = np.asarray(
        Parallel(n_jobs=-1)(
            delayed(monte_carlo_iteration)(seed) for seed in seeds
        ),
        dtype=float,
    )
    critical_values = _critical_values_from_simulated(
        simulated,
        use_ceiling=False,
    )
    return critical_values, simulated


def critical_value_lr_hac_m(
    F_hat,
    trim_ratio,
    n_sim=5000,
    random_state=None,
    debug=False,
):
    """Compute HAC-based critical values for the single-break LR test.

    This is the public high-level helper used by
    :meth:`~fmbqml.single_break.SingleBreakQML.lr_test`.  It estimates the HAC
    matrix from ``F_hat`` and then calls
    :func:`simulate_single_sup_lr_critical_values`.  It is intentionally
    single-break-specific; joint tests with ``n_breaks > 1`` must use
    :func:`joint_sup_lr_cv`.

    Parameters
    ----------
    F_hat : array_like of shape (T, r)
        Estimated factors.
    trim_ratio : float
        Fraction excluded from each end of the candidate-break range.
    n_sim : int, default=5000
        Number of Monte Carlo replications.
    random_state : int or None, default=None
        Seed for reproducible simulation.
    debug : bool, default=False
        If True, print HAC eigenvalue diagnostics and the simulated critical
        values.

    Returns
    -------
    list of float
        Critical values for significance levels 10%, 5%, and 1%, in that
        order.

    Raises
    ------
    ValueError
        If ``F_hat`` is not two-dimensional or ``n_sim`` is not a positive
        integer.
    """
    F_hat = np.asarray(F_hat, dtype=float)
    if F_hat.ndim != 2:
        raise ValueError("F_hat must be a 2D array.")

    omega_hat = factor_long_run_covariance(F_hat)

    if debug:
        eig_omega = np.linalg.eigvalsh(omega_hat)
        print("Omega_hat eigenvalues:")
        print(eig_omega)
        print("Omega_hat min eigenvalue:", np.min(eig_omega))
        print("Omega_hat max eigenvalue:", np.max(eig_omega))
        print("Omega_hat trace:", np.trace(omega_hat))

    critical_values, _ = simulate_single_sup_lr_critical_values(
        omega_hat,
        trim_ratio,
        n_sim=n_sim,
        grid_size=2000,
        random_state=random_state,
    )

    if debug:
        print("Critical values [10%, 5%, 1%]:")
        print(critical_values)

    return critical_values.tolist()


def bridge_score_matrix(
    bridge: np.ndarray,
    omega: np.ndarray,
    min_length: int,
    denominator_scale: float = 2.0,
) -> np.ndarray:
    """Score all admissible Brownian-bridge segment increments.

    The returned upper-triangular matrix is consumed by the dynamic-programming
    partition optimizer in :func:`joint_sup_lr_cv`.  Inadmissible intervals
    retain a score of negative infinity.

    Parameters
    ----------
    bridge : ndarray of shape (d, grid_size + 1)
        Brownian bridge evaluated on a grid including both endpoints.
    omega : ndarray of shape (d, d)
        HAC long-run covariance matrix.
    min_length : int
        Minimum admissible segment length measured in grid intervals.
    denominator_scale : float, default=2.0
        Scale in the denominator of each segment score.

    Returns
    -------
    ndarray of shape (grid_size + 1, grid_size + 1)
        Entry ``[start, end]`` is the LR contribution of that segment.
    """
    grid_size = bridge.shape[1] - 1
    scores = np.full((grid_size + 1, grid_size + 1), -np.inf)
    for start in range(grid_size):
        ends = np.arange(start + min_length, grid_size + 1)
        increments = (bridge[:, ends] - bridge[:, start, None]).T
        quadratic = np.einsum(
            "ij,jk,ik->i", increments, omega, increments, optimize=True
        )
        fractions = (ends - start) / grid_size
        scores[start, ends] = quadratic / (denominator_scale * fractions)
    return scores


def joint_sup_lr_cv(
    omega: np.ndarray,
    n_breaks: int,
    trim_ratio: float,
    n_sim: int,
    grid_size: int,
    random_state: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the limiting joint multiple-break sup-LR distribution.

    Each replication constructs a Brownian bridge, scores every admissible
    segment, and uses dynamic programming to find the best partition into
    ``n_breaks + 1`` regimes.  Thus the critical values depend on the
    prespecified number of breaks.  For ``n_breaks=1`` the statistic reduces
    to the one-break form up to numerical implementation details, but this
    routine preserves one consistent partition-based path for all joint tests.

    Parameters
    ----------
    omega : ndarray of shape (d, d)
        HAC long-run covariance matrix of the factor-product deviations.
    n_breaks : int
        Prespecified number of breaks under the alternative.
    trim_ratio : float
        Minimum regime length as a fraction of the unit interval.
    n_sim : int
        Number of Monte Carlo replications.
    grid_size : int
        Number of intervals used to approximate the unit interval.
    random_state : int or None
        Seed for reproducible simulation.

    Returns
    -------
    critical_values : ndarray of shape (3,)
        Simulated 10%, 5%, and 1% significance-level critical values, in that
        order.
    simulated_statistics : ndarray of shape (n_sim,)
        Optimized joint sup-LR statistic from every replication.

    Notes
    -----
    This is a lower-level simulator: unlike
    :func:`critical_value_lr_hac_m`, it receives ``omega`` rather than
    estimating it from factors.  The caller is responsible for validating all
    arguments other than ``n_sim``.
    """
    n_sim = _validate_n_sim(n_sim)
    omega = np.asarray(omega, dtype=float)
    dimension = omega.shape[0]
    min_length = int(np.ceil(grid_size * trim_ratio))
    n_segments = n_breaks + 1
    fractions = np.arange(grid_size + 1) / grid_size
    base_rng = np.random.default_rng(random_state)
    seeds = base_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_sim,
        dtype=np.uint32,
    ).tolist()

    def monte_carlo_iteration(seed):
        rng = np.random.default_rng(int(seed))
        motion = simulate_brownian_motion(
            t=1.0,
            n=grid_size,
            dim=dimension,
            rng=rng,
        )
        motion = np.column_stack([np.zeros(dimension), motion])
        bridge = motion - motion[:, -1, None] * fractions
        scores = bridge_score_matrix(bridge, omega, min_length)
        statistic, _ = optimize_additive_partition(
            scores,
            n_segments,
            min_length,
            maximize=True,
        )
        return statistic

    simulated = np.asarray(
        Parallel(n_jobs=-1)(
            delayed(monte_carlo_iteration)(seed) for seed in seeds
        ),
        dtype=float,
    )

    critical_values = _critical_values_from_simulated(simulated, use_ceiling=True)
    return critical_values, simulated


def select_critical_value(cv, alpha: float) -> float:
    """Select the critical value corresponding to a significance level.

    Parameters
    ----------
    cv : array_like
        At least three critical values ordered for significance levels 10%,
        5%, and 1%.
    alpha : {0.10, 0.05, 0.01}
        Requested significance level.

    Returns
    -------
    float
        Selected finite critical value.

    Raises
    ------
    ValueError
        If ``alpha`` is unsupported, fewer than three critical values are
        supplied, or the selected value is not finite.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError(
            f"alpha must be one of 0.10, 0.05, or 0.01. Got alpha={alpha}."
        )

    alpha = float(alpha)
    alpha_map = {0.10: 0, 0.05: 1, 0.01: 2}
    if alpha not in alpha_map:
        raise ValueError(
            f"alpha must be one of 0.10, 0.05, or 0.01. Got alpha={alpha}."
        )

    values = np.asarray(cv, dtype=float).reshape(-1)
    if values.size < 3:
        raise ValueError(
            "cv must contain at least three critical values ordered as "
            "alpha = (0.10, 0.05, 0.01)."
        )

    value = values[alpha_map[alpha]]
    if not np.isfinite(value):
        raise ValueError(f"The selected critical value is not finite. Got {value}.")
    return float(value)


__all__ = [
    "bridge_score_matrix",
    "critical_value_lr_hac_m",
    "factor_long_run_covariance",
    "factor_product_deviations",
    "select_critical_value",
    "joint_sup_lr_cv",
    "simulate_single_sup_lr_critical_values",
]
