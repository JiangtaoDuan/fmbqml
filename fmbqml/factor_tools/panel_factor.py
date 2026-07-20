"""Principal-component estimation for static factor models."""

import numpy as np

def estimate_panel_factors(X, r):
    """
    Input:
        X : (T x N) data matrix
        r : number of factors
    Output:
        factor : (T x r) estimated common factors
        lambda_ : (N x r) estimated factor loadings
        VNT : (r x r) diagonal matrix of eigenvalues
    """

    T, N = X.shape

    if T < N:
        XX = (X @ X.T) / (N * T)
        Fhat0, S, Fhat1 = np.linalg.svd(XX, full_matrices=True)

        factor = Fhat0[:, :r] * np.sqrt(T)
        lambda_ = (X.T @ factor) / T

        VNT = np.diag(S[:r])

    else:
        XX = (X.T @ X) / (N * T)
        Fhat0, S, Fhat1 = np.linalg.svd(XX, full_matrices=True)

        lambda_ = Fhat0[:, :r] * np.sqrt(N)
        factor = (X @ lambda_) / N
        VNT = np.diag(S[:r])

    return factor, lambda_, VNT
