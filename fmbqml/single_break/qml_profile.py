"""QML objective-profile computation for a single breakpoint."""

import numpy as np

from fmbqml.utils.safe_logdet import safe_logdet


def compute_single_break_qml_profile(
        F_hat: np.ndarray,
        T: int,
        t_min: int,
        t_max: int,
        n_factors: int
) -> np.ndarray:
    """
    Compute the single-break QML objective profile over candidate breakpoints.

    Parameters
    ----------
    F_hat : np.ndarray
        Estimated factor matrix with shape (T, n_factors).
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
    np.ndarray
        Array of QML objective values for candidate breakpoints.

    Raises
    ------
    ValueError
        If no valid candidate breakpoints are available.
    """
    qml_profile = []

    for k in range(t_min, t_max + 1):
        if k <= n_factors + 1 or (T - k) <= n_factors + 1:
            qml_profile.append(np.inf)
            continue

        sigma_left = F_hat[:k, :].T @ F_hat[:k, :] / k
        sigma_right = F_hat[k:, :].T @ F_hat[k:, :] / (T - k)

        logdet_left = safe_logdet(sigma_left)
        logdet_right = safe_logdet(sigma_right)

        qml_profile.append(k * logdet_left + (T - k) * logdet_right)

    qml_profile = np.asarray(qml_profile, dtype=float)

    if np.all(np.isinf(qml_profile)):
        raise ValueError(
            "No valid candidate breakpoints found. "
            "Check sample size T, trim_ratio, or the estimated number of factors."
        )

    return qml_profile
