"""Dynamic-programming tools for exact multiple-break QML partitions."""

from numbers import Integral

import numpy as np

from fmbqml.utils.partition import optimize_additive_partition
from fmbqml.utils.safe_logdet import safe_logdet


def build_qml_cost_matrix(
        F_hat: np.ndarray,
        min_length: int,
) -> np.ndarray:
    """Build QML segment costs for zero-based half-open intervals.

    The cost for segment ``[start, end)`` is

        length * logdet(F_segment.T @ F_segment / length).

    Only segments satisfying ``length >= min_length`` are evaluated.
    """
    F_hat = np.asarray(F_hat, dtype=float)
    T, r = F_hat.shape

    outer_products = np.einsum("ti,tj->tij", F_hat, F_hat)
    cumulative = np.concatenate(
        (
            np.zeros((1, r, r)),
            np.cumsum(outer_products, axis=0),
        ),
        axis=0,
    )

    costs = np.full((T + 1, T + 1), np.inf)

    for start in range(T):
        # A segment with fewer observations than factors has a
        # necessarily rank-deficient second-moment matrix, so its logdet
        # objective is not defined. Keep it inadmissible even when the
        # caller's trimming length is smaller than r.
        for end in range(start + max(min_length, r), T + 1):
            length = end - start
            second_moment = (cumulative[end] - cumulative[start]) / length
            costs[start, end] = length * safe_logdet(second_moment)

    return costs


def find_min_qml(F_hat, T, h, m, *, cost_matrix=None):
    """Minimize the QML criterion over exactly ``m`` breakpoints.

    Returned breakpoint locations are zero-based segment boundaries. Thus,
    a returned location ``k`` separates observations ``[:k]`` and ``[k:]``;
    it corresponds to MATLAB's one-based breakpoint location ``k``.

    Parameters
    ----------
    cost_matrix : ndarray, optional
        Precomputed segment-cost matrix from :func:`build_qml_cost_matrix`.
        It must have been built from the same ``F_hat`` and ``h``. Supplying
        it allows callers evaluating several values of ``m`` to reuse the
        expensive segment costs. If omitted, the matrix is built internally.
    """
    F_hat = np.asarray(F_hat, dtype=float)

    if isinstance(T, (bool, np.bool_)) or not isinstance(T, Integral) or T <= 0:
        raise ValueError("T must be a positive integer.")
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Integral) or h <= 0:
        raise ValueError("h must be a positive integer.")
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or m < 0:
        raise ValueError("m must be a non-negative integer.")
    if F_hat.ndim != 2 or F_hat.shape[0] != T:
        raise ValueError("F_hat must be a two-dimensional array with T rows.")
    if (m + 1) * h > T:
        raise ValueError("The requested number of breaks is infeasible for T and h.")

    # In joint estimation, the segment-cost matrix is invariant across the
    # candidate number of breaks. Allow reuse of a precomputed matrix; direct
    # callers retain the original self-contained behavior.
    costs = (
        build_qml_cost_matrix(F_hat, h)
        if cost_matrix is None
        else np.asarray(cost_matrix, dtype=float)
    )
    if costs.shape != (T + 1, T + 1):
        raise ValueError("cost_matrix must have shape (T + 1, T + 1).")

    minimum, breakpoints = optimize_additive_partition(
        costs,
        n_segments=m + 1,
        min_length=h,
        maximize=False,
    )

    return minimum, np.asarray(breakpoints, dtype=int)


__all__ = ["build_qml_cost_matrix", "find_min_qml"]
