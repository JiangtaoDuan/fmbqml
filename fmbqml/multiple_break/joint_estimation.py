"""Joint QML estimation for multiple factor-model breakpoints."""

import warnings

import numpy as np

from fmbqml.factor_tools.nb_factors import bai_ng_factor_count
from fmbqml.multiple_break.eig_sort import eig_sort
from fmbqml.multiple_break.find_min_qml import find_min_qml
from fmbqml.utils.criteria import normalize_factor_criterion
from fmbqml.utils.safe_logdet import safe_logdet


def _clean_breakpoint_list(breakpoints):
    """
    Convert breakpoint candidates to a clean list of integer locations.

    Finite numerical values are rounded to the nearest integer, while non-finite
    or invalid entries are ignored.
    """
    if breakpoints is None:
        return []

    if isinstance(breakpoints, str):
        return breakpoints

    arr = np.asarray(breakpoints).reshape(-1)
    out = []

    for v in arr:
        try:
            if np.isfinite(v):
                out.append(int(round(float(v))))
        except (TypeError, ValueError, OverflowError):
            pass

    return out


def _standardize_columns(X):
    """
    Standardize each variable using its full-sample mean and sample standard
    deviation.
    """
    X = np.asarray(X, dtype=float)

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0

    return (X - mu) / sd


def _select_factor_number(X, max_factors, factor_criterion):
    """
    Select the number of pseudo-factors using Bai--Ng information criteria.
    """
    X = np.asarray(X, dtype=float)

    T, N = X.shape
    criterion_index = normalize_factor_criterion(factor_criterion)

    kmax_eff = min(int(max_factors), T - 1, N - 1)

    if kmax_eff < 1:
        return 0

    result = bai_ng_factor_count(X, kmax=kmax_eff)

    return int(result["khat"][criterion_index])

def _compute_segment_qml_loss(
        F_hat,
        breakpoints,
        T,
        r_full,
        jitter0=1e-10
):
    """
    Compute the total QML loss over all regimes implied by a set of breakpoints.
    """
    if breakpoints is None or len(breakpoints) == 0:
        bk = []
    else:
        bk = list(breakpoints)

    edges = [0] + bk + [T]
    loss_value = 0.0

    for j in range(len(edges) - 1):
        a = int(edges[j])
        b = int(edges[j + 1])
        length = b - a

        if length <= 0:
            continue

        Fseg = F_hat[a:b, :]

        Sig = (Fseg.T @ Fseg) / length
        Sig = 0.5 * (Sig + Sig.T)

        logdet_sig = safe_logdet(Sig)

        loss_value += length * logdet_sig

    return float(loss_value)


def _estimate_factor_var_spectral_radius(F_hat):
    """
    Estimate the spectral radius of the VAR(1) coefficient matrix fitted to
    the estimated factor series.
    """
    T, r_full = F_hat.shape

    if T <= 2 or r_full == 0:
        return 0.0

    Y = F_hat[1:T, :]
    Xlag = F_hat[0:T - 1, :]

    Phi = np.linalg.lstsq(Xlag, Y, rcond=None)[0].T

    _, D_rho = eig_sort(Phi, r_full)

    rho_hat = np.abs(D_rho[0, 0])

    if np.isnan(rho_hat) or np.isinf(rho_hat):
        rho_hat = 0.0

    return float(rho_hat)


def joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
        *,
        standardize=False,
        jitter0=1e-10,
):
    """
    Jointly estimate multiple structural breakpoints using the QML criterion.

    The procedure first selects the number of pseudo-factors, estimates the
    full-sample factor matrix, and then evaluates candidate numbers of breaks.
    For each candidate number of breaks, the optimal breakpoint locations are
    obtained by dynamic programming. The final number of breaks is selected
    using an information criterion.

    The information criterion is defined as

        IC(m) = Loss(m) + m * log(min(N, T)) * r^2 * (1 + abs(rho)),

    where r is the selected number of pseudo-factors and rho is the spectral
    radius of the factor VAR(1) coefficient matrix.

    Parameters
    ----------
    X : array-like
        Input data matrix with shape (T, N).
    min_segment_length : int
        Minimum segment length used in breakpoint search.
    max_factors : int
        Maximum number of pseudo-factors considered in factor selection.
    max_break : int
        Maximum number of breaks allowed.
    min_break : int
        Minimum number of breaks allowed.
    factor_criterion : int or str
        Bai--Ng factor-number selection criterion.
    standardize : bool, default=False
        Whether to standardize the input data before estimation.
    jitter0 : float, default=1e-10
        Initial diagonal regularization used in log-determinant computation.

    Returns
    -------
    breakpoints_qml : list or str
        Estimated breakpoints. If no break is selected, returns
        "no breakpoint".
    m_hat : int
        Selected number of breaks.
    ic_path : list of dict
        Information-criterion path over candidate numbers of breaks.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf.")

    if min_break > max_break:
        raise ValueError("min_break cannot be greater than max_break.")

    if standardize:
        X_work = _standardize_columns(X)
    else:
        X_work = X.copy()

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

    F_hat = np.sqrt(T) * U_full[:, 0:r_full]

    rho_hat = _estimate_factor_var_spectral_radius(F_hat)

    penalty_per_break = (
        np.log(min(N, T))
        * (r_full ** 2)
        * (1.0 + abs(rho_hat))
    )

    loss_values = np.full(max_break + 1, np.nan)
    ic_values = np.full(max_break + 1, np.nan)

    breakpoint_store = {}
    raw_loss_store = {}

    first_m = 0 if min_break == 0 else min_break

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for m in range(first_m, max_break + 1):
            if m == 0:
                breakpoints = []
                raw_min_value = np.nan
            else:
                raw_min_value, raw_breakpoints = find_min_qml(
                    F_hat,
                    T,
                    min_segment_length,
                    m
                )

                breakpoints = _clean_breakpoint_list(raw_breakpoints)

                if isinstance(breakpoints, str):
                    breakpoints = []

            breakpoint_store[int(m)] = breakpoints
            raw_loss_store[int(m)] = raw_min_value

            loss_m = _compute_segment_qml_loss(
                F_hat=F_hat,
                breakpoints=breakpoints,
                T=T,
                r_full=r_full,
                jitter0=jitter0,
            )

            loss_values[m] = loss_m
            ic_values[m] = loss_m + m * penalty_per_break

    valid_ms = list(range(first_m, max_break + 1))

    m_hat = int(valid_ms[int(np.nanargmin(ic_values[valid_ms]))])

    if m_hat == 0:
        breakpoints_qml = "no breakpoint"
    else:
        breakpoints_qml = breakpoint_store[m_hat]

    ic_path = []

    for m in valid_ms:
        raw_value = raw_loss_store.get(int(m), np.nan)

        ic_path.append({
            "m": int(m),
            "IC": float(ic_values[m]),
            "loss": float(loss_values[m]),
            "estimated_breakpoints": breakpoint_store.get(int(m), []),
            "n_factors": int(r_full),
            "rho": float(rho_hat),
            "penalty_per_break": float(penalty_per_break),
            "raw_find_min_qml_value": (
                None
                if (m == 0 or not np.isfinite(raw_value))
                else float(raw_value)
            ),
            "standardize": bool(standardize),
            "x_checksum": float(np.sum(X_work)),
            "x_fro_norm": float(np.linalg.norm(X_work)),
        })

    return breakpoints_qml, int(m_hat), ic_path
