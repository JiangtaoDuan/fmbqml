"""Dynamic-programming tools for exact multiple-break partitions."""

import numpy as np

from fmbqml.utils.safe_logdet import safe_logdet
from fmbqml.utils.partition import optimize_additive_partition


def build_qml_cost_matrix(
        F_hat: np.ndarray,
        min_length: int,
) -> np.ndarray:
    """Build the segment QML cost matrix on zero-based half-open intervals."""
    F_hat = np.asarray(F_hat, dtype=float)
    T, r = F_hat.shape
    outer_products = np.einsum("ti,tj->tij", F_hat, F_hat)
    cumulative = np.concatenate([
        np.zeros((1, r, r)),
        np.cumsum(outer_products, axis=0),
    ])
    costs = np.full((T + 1, T + 1), np.inf)

    for start in range(T):
        for end in range(start + min_length, T + 1):
            length = end - start
            second_moment = (cumulative[end] - cumulative[start]) / length
            costs[start, end] = length * safe_logdet(second_moment)
    return costs


def find_min_qml(F_hat, T, h, m):
    """Minimize the QML criterion over exactly ``m`` breakpoints."""
    F_hat = np.asarray(F_hat, dtype=float)
    if F_hat.ndim != 2 or F_hat.shape[0] != T:
        raise ValueError("F_hat must be a two-dimensional array with T rows.")

    costs = build_qml_cost_matrix(F_hat, h)
    minimum, breakpoints = optimize_additive_partition(
        costs,
        n_segments=m + 1,
        min_length=h,
        maximize=False,
    )
    return minimum, np.asarray(breakpoints, dtype=int)


__all__ = [
    "build_qml_cost_matrix",
    "find_min_qml",
]
