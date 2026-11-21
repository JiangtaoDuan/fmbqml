"""Principal-component estimation for static factor models."""

import numpy as np


def estimate_panel_factors(X, r):
    """Estimate common factors using principal components.

    Parameters
    ----------
    X : ndarray of shape (T, N)
        Panel data matrix, with observations in rows and series in columns.
    r : int
        Number of factors to estimate.

    Returns
    -------
    factor : ndarray of shape (T, r)
        Estimated common factors.
    lambda_ : ndarray of shape (N, r)
        Estimated factor loadings.
    VNT : ndarray of shape (r, r)
        Diagonal matrix of eigenvalues.
    """

    T, N = X.shape

    if T < N:
        XX = (X @ X.T) / (N * T)
        Fhat0, S, _ = np.linalg.svd(XX, full_matrices=True)

        factor = Fhat0[:, :r] * np.sqrt(T)
        lambda_ = (X.T @ factor) / T

        VNT = np.diag(S[:r])

    else:
        XX = (X.T @ X) / (N * T)
        Fhat0, S, _ = np.linalg.svd(XX, full_matrices=True)

        lambda_ = Fhat0[:, :r] * np.sqrt(N)
        factor = (X @ lambda_) / N
        VNT = np.diag(S[:r])

    return factor, lambda_, VNT
