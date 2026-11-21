
"""Joint QML estimation for multiple factor-model breakpoints."""

import numpy as np

from fmbqml.factor_tools.nb_factors import bai_ng_factor_count
from fmbqml.multiple_break.eig_sort import eig_sort
from fmbqml.multiple_break.qml_optimization import (
    build_qml_cost_matrix,
    find_min_qml,
)
from fmbqml.utils.criteria import normalize_factor_criterion
from fmbqml.utils.safe_logdet import safe_logdet


def _standardize_columns(X):
    """Standardize each variable using its full-sample mean and sample SD."""
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    return (X - mu) / sd


def _select_factor_number(X, max_factors, factor_criterion):
    """Select the number of pseudo-factors using a Bai--Ng criterion."""
    T, N = X.shape
    criterion_index = normalize_factor_criterion(factor_criterion)
    kmax_eff = min(int(max_factors), T - 1, N - 1)

    if kmax_eff < 1:
        return 0

    result = bai_ng_factor_count(X, kmax=kmax_eff)
    return int(result["khat"][criterion_index])


def _estimate_factor_var_spectral_radius(F_hat):
    """Fit a no-intercept factor VAR(1) and return its largest eigenvalue modulus.

    Least squares accommodates singular lagged-factor matrices; the radius
    enters the information-criterion penalty for selecting break counts.
    """
    Y = F_hat[1:, :]
    X_lag = F_hat[:-1, :]
    phi = np.linalg.lstsq(X_lag, Y, rcond=None)[0].T

    _, D_rho = eig_sort(phi, F_hat.shape[1])
    return float(np.abs(D_rho[0, 0]))


def joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
        *,
        standardize=False,
):
    """
    Jointly estimate multiple structural breakpoints using the QML criterion.

    The number of pseudo-factors is selected first, followed by full-sample
    principal-component estimation. For each candidate number of breaks,
    dynamic programming minimizes the original segment-wise QML objective.

    The IC uses the same segment-wise objective as the joint sup-LR test.

    The number of breaks is selected by

        IC(m) = Loss(m) + m * log(min(N, T)) * r**2 * (1 + abs(rho)),

    where ``rho`` is the spectral radius of the no-intercept factor VAR(1).

    Parameters
    ----------
    X : array-like
        Input data matrix with shape (T, N).
    min_segment_length : int
        Minimum segment length used in breakpoint search.
    max_factors : int
        Maximum number of pseudo-factors considered.
    max_break : int
        Maximum number of breaks considered.
    min_break : int
        Minimum number of breaks considered.
    factor_criterion : int or str
        Bai--Ng factor-number selection criterion.
    standardize : bool, default=False
        Whether to standardize the input data before estimation.
    Returns
    -------
    breakpoints_qml : list or str
        Estimated breakpoints. Returns "no breakpoint" if no break is selected.
    m_hat : int
        Selected number of breaks.
    ic_path : list of dict
        Information-criterion path over candidate numbers of breaks.
    """
    X = np.asarray(X, dtype=float)

    if min_break > max_break:
        raise ValueError("min_break cannot be greater than max_break.")

    X_work = _standardize_columns(X) if standardize else X.copy()
    T, N = X_work.shape

    r_full = _select_factor_number(
        X_work,
        max_factors=max_factors,
        factor_criterion=factor_criterion,
    )

    if r_full <= 0:
        raise ValueError(
            "The estimated number of factors is zero. "
            "Multiple-break QML requires at least one estimated factor."
        )

    U_full, D_full = eig_sort(X_work @ X_work.T, r_full)
    order = np.argsort(np.diag(D_full))[::-1]
    F_hat = np.sqrt(T) * U_full[:, order[:r_full]]

    rho_hat = _estimate_factor_var_spectral_radius(F_hat)

    penalty_per_break = (
        np.log(min(N, T)) * (r_full ** 2) * (1.0 + abs(rho_hat))
    )

    first_m = 0 if min_break == 0 else min_break

    loss_values = {}
    ic_values = {}
    breakpoint_store = {}
    raw_loss_store = {}

    # Segment QML costs depend on F_hat and min_segment_length, not on the
    # candidate number of breaks. Build once and reuse for every m, avoiding
    # repeated Cholesky/log-determinant work while preserving the same cost
    # entries and dynamic-programming routine.
    qml_cost_matrix = (
        build_qml_cost_matrix(F_hat, min_segment_length)
        if max_break >= max(first_m, 1)
        else None
    )

    for m in range(first_m, max_break + 1):

        if m == 0:
            breakpoints = []
            raw_min_value = np.nan
            loss_m = T * safe_logdet(F_hat.T @ F_hat / T)

        else:
            raw_min_value, raw_breakpoints = find_min_qml(
                F_hat,
                T,
                min_segment_length,
                m,
                cost_matrix=qml_cost_matrix,
            )
            breakpoints = raw_breakpoints.tolist()
            loss_m = raw_min_value

        breakpoint_store[int(m)] = breakpoints
        raw_loss_store[int(m)] = raw_min_value

        loss_values[int(m)] = float(loss_m)
        ic_values[int(m)] = float(
            loss_m + m * penalty_per_break
        )

    valid_ms = range(first_m, max_break + 1)
    m_hat = min(valid_ms, key=lambda m: ic_values[m])

    breakpoints_qml = (
        "no breakpoint" if m_hat == 0 else breakpoint_store[m_hat]
    )

    ic_path = []

    for m in range(first_m, max_break + 1):
        raw_value = raw_loss_store[m]

        ic_path.append({
            "m": int(m),
            "IC": float(ic_values[m]),
            "loss": float(loss_values[m]),
            "estimated_breakpoints": breakpoint_store[m],
            "n_factors": int(r_full),
            "rho": float(rho_hat),
            "penalty_per_break": float(penalty_per_break),
            "raw_find_min_qml_value": (
                None
                if m == 0 or not np.isfinite(raw_value)
                else float(raw_value)
            ),
            "standardize": bool(standardize),
        })

    return breakpoints_qml, int(m_hat), ic_path


__all__ = ["joint_estimation"]
