"""Numerically stable log-determinant evaluation."""

import numpy as np


def safe_logdet(A, eps=1e-8, max_tries=8):
    """
    Compute a numerically stable log-determinant.

    The matrix is first symmetrized to remove small floating-point
    asymmetries. The function first tries to compute the log-determinant of
    the original matrix using numpy.linalg.slogdet. A diagonal jitter is added
    only if the unregularized calculation is not numerically reliable. As a
    final fallback, the log-determinant is computed from lower-bounded
    eigenvalues.

    Parameters
    ----------
    A : np.ndarray
        Input square matrix.
    eps : float, default=1e-8
        Initial positive regularization constant added to the diagonal.
    max_tries : int, default=8
        Maximum number of jitter-increasing attempts.

    Returns
    -------
    float
        Numerically stable log-determinant value.
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    # Remove small numerical asymmetry.
    A = 0.5 * (A + A.T)

    sign, logdet = np.linalg.slogdet(A)

    if sign > 0 and np.isfinite(logdet):
        return float(logdet)

    jitter = eps

    for _ in range(max_tries):
        A_reg = A + jitter * np.eye(n)

        sign, logdet = np.linalg.slogdet(A_reg)

        if sign > 0 and np.isfinite(logdet):
            return float(logdet)

        jitter *= 10.0

    # Final fallback: eigenvalue truncation.
    eigvals = np.linalg.eigvalsh(A)
    eigvals = np.clip(eigvals, eps, None)

    return float(np.sum(np.log(eigvals)))
