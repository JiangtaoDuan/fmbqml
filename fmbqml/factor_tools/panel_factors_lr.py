"""Factor-estimation helpers used by likelihood-ratio procedures."""

import numpy as np


def estimate_panel_factors_lr(X, r):
    """
    MATLAB-style factor-normalized panel factor estimation.

    This function corresponds to the MATLAB code:

        XX = X * X' / (N * T)
        [Fhat0, eigval, Fhat1] = svd(XX)
        factor = Fhat0(:, 1:r) * sqrt(T)
        lambda = X' * factor / T
        VNT = eigval(1:r, 1:r)

    Normalization
    -------------
    The estimated factors satisfy:

        factor.T @ factor / T = I

    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape (T, N).
    r : int
        Number of factors.

    Returns
    -------
    factor : np.ndarray
        Estimated factor matrix with shape (T, r).
    lambda_ : np.ndarray
        Estimated loading matrix with shape (N, r).
    VNT : np.ndarray
        Diagonal matrix of eigenvalues with shape (r, r).
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (T, N).")

    T, N = X.shape

    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise ValueError(f"r must be a positive integer. Got {r}.")

    if r > min(T, N):
        raise ValueError(
            f"r is too large. Require r <= min(T, N) = {min(T, N)}, got {r}."
        )

    XX = (X @ X.T) / (N * T)

    Fhat0, S, Fhat1 = np.linalg.svd(XX, full_matrices=True)

    factor = Fhat0[:, :r] * np.sqrt(T)

    lambda_ = (X.T @ factor) / T

    VNT = np.diag(S[:r])

    return factor, lambda_, VNT
