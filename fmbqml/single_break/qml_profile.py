
"""QML objective-profile computation for a single breakpoint."""

import numpy as np

from fmbqml.utils.safe_logdet import safe_logdet


def _segment_logdet(second_moment: np.ndarray) -> float:
    """Return a segment's log-determinant, or infinity if it is unusable.

    Nonfinite inputs and linear-algebra failures disqualify the candidate;
    nonsquare matrices indicate a programming error and raise ``ValueError``.
    """
    matrix = np.asarray(second_moment, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Segment second-moment matrix must be square.")

    if not np.isfinite(matrix).all():
        return np.inf

    try:
        return safe_logdet(matrix)
    except np.linalg.LinAlgError:
        return np.inf


def compute_single_break_qml_profile(
        F_hat: np.ndarray,
        T: int,
        t_min: int,
        t_max: int,
        n_factors: int
) -> np.ndarray:
    """Compute the single-break QML objective profile.

    Candidate locations are evaluated over the supplied trimming range.
    A candidate is assigned an infinite objective value if either regime
    contains fewer observations than the number of estimated factors,
    or if its segment second-moment matrix cannot be evaluated.

    Parameters
    ----------
    F_hat : ndarray of shape (T, n_factors)
        Estimated factor matrix.
    T : int
        Sample size.
    t_min : int
        Lower bound of the breakpoint search range.
    t_max : int
        Upper bound of the breakpoint search range.
    n_factors : int
        Estimated number of factors.

    Returns
    -------
    qml_profile : ndarray
        QML objective values for candidate breakpoints from t_min
        through t_max. Numerically unusable candidates are assigned
        infinite objective values.

    Raises
    ------
    ValueError
        If the factor matrix dimensions or candidate range are invalid,
        or if no candidate has numerically usable segment matrices.
    """
    F_hat = np.asarray(F_hat, dtype=float)

    if F_hat.ndim != 2:
        raise ValueError("F_hat must be a two-dimensional array.")

    if F_hat.shape[0] != T:
        raise ValueError("F_hat must have T rows.")

    if F_hat.shape[1] != n_factors:
        raise ValueError(
            "n_factors must match the number of columns in F_hat."
        )

    if (
        isinstance(t_min, (bool, np.bool_))
        or not isinstance(t_min, (int, np.integer))
        or isinstance(t_max, (bool, np.bool_))
        or not isinstance(t_max, (int, np.integer))
    ):
        raise TypeError("t_min and t_max must be integers.")

    if t_min < 1 or t_max >= T or t_min > t_max:
        raise ValueError(
            f"Invalid candidate range [{t_min}, {t_max}] for T={T}."
        )

    qml_profile = np.full(
        t_max - t_min + 1,
        np.inf,
        dtype=float
    )

    for index, k in enumerate(range(t_min, t_max + 1)):

        # A segment shorter than the factor dimension
        # cannot produce a full-rank second-moment matrix.
        if k < n_factors or (T - k) < n_factors:
            continue

        sigma_left = (
            F_hat[:k, :].T @ F_hat[:k, :]
        ) / k

        sigma_right = (
            F_hat[k:, :].T @ F_hat[k:, :]
        ) / (T - k)

        logdet_left = _segment_logdet(sigma_left)
        logdet_right = _segment_logdet(sigma_right)

        if np.isfinite(logdet_left) and np.isfinite(logdet_right):
            qml_profile[index] = (
                k * logdet_left
                + (T - k) * logdet_right
            )

    if not np.isfinite(qml_profile).any():
        raise ValueError(
            "No candidate breakpoint has numerically usable segment "
            "second-moment matrices. The trimming-based candidate range "
            "was retained, but all candidate costs were non-finite."
        )

    return qml_profile
