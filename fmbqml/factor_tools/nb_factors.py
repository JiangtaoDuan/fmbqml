"""Bai--Ng information criteria for selecting the number of factors."""

import numpy as np
from numpy.linalg import eigh
from scipy.linalg import cholesky, inv


def _safe_cholesky(matrix, *, context, jitter=1e-10):
    """
    Compute a Cholesky factor with a small numerical safeguard.

    The first attempt uses the original matrix. A tiny diagonal jitter is used
    only when the matrix is numerically non-positive-definite. This keeps the
    standard Bai--Ng calculation unchanged for regular data while producing a
    clearer failure mode for nearly rank-deficient panels.
    """
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)

    try:
        return cholesky(matrix)
    except np.linalg.LinAlgError:
        scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
        adjusted = matrix + jitter * scale * np.eye(matrix.shape[0])
        try:
            return cholesky(adjusted)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Bai-Ng factor-number selection failed because a loading "
                "covariance matrix is not positive definite. "
                "Check whether the data are nearly rank deficient, contain "
                "constant or duplicate columns, or use a smaller max_factors. "
                f"Failure occurred while computing {context}."
            ) from exc


def bai_ng_factor_count(X, kmax=None):
    """Select factor counts using Bai and Ng information criteria.

    Implements criteria from Bai and Ng (2002), "Determining the Number of
    Factors in Approximate Factor Models," Econometrica, 70(1), 191-221.

    Parameters
    ----------
    X : ndarray of shape (T, N)
        Panel of observations, with time periods in rows and series in columns.
    kmax : int, optional
        Maximum number of factors considered. If omitted or NaN, uses
        ``min(N, T)``.

    Returns
    -------
    results : dict
        Dictionary with the following entries:

        * ``khat``: selected counts for IC1, IC2, IC3, PC1, PC2, PC3,
          AIC3 and BIC3, in that order.
        * ``khat_IC`` and ``khat_PC``: selected counts for the three IC and
          PC criteria, respectively.
        * ``khat_AIC3`` and ``khat_BIC3``: selected counts for AIC3 and BIC3.
        * ``kmax``: maximum number of factors considered.
        * ``IC`` and ``PC``: arrays containing the factor count followed by
          the three criterion values for each count from zero to ``kmax``.
        * ``AIC3`` and ``BIC3``: arrays containing the factor count and the
          corresponding criterion value for each count from zero to ``kmax``.
        * ``Vkmax``: estimated residual variance with ``kmax`` factors.
    """

    T, N = X.shape

    if (kmax is None) or (isinstance(kmax, float) and np.isnan(kmax)):
        kmax = min(N, T)

    # ----------------------------------------
    # --- Computation of the V(kmax,Fkmax) ---
    # ----------------------------------------

    if T < N:
        eigenvalues, eigenvectors = eigh(X @ X.T)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        factors = np.sqrt(T) * eigenvectors[:, :kmax]
        loadings = X.T @ factors / T
        betahat = loadings @ _safe_cholesky(
            loadings.T @ loadings / N,
            context="the kmax-factor residual variance",
        )
    else:
        eigenvalues, eigenvectors = eigh(X.T @ X)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        loadings = np.sqrt(N) * eigenvectors[:, :kmax]
        factors = X @ loadings / N
        betahat = (X.T @ X) @ loadings / (N * T)

    G = betahat.T @ betahat
    G = 0.5 * (G + G.T)

    try:
        G_inv = inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G, rcond=1e-10)

    Z = X - X @ betahat @ G_inv @ betahat.T
    var_Z_kmax = np.sum(Z ** 2) / (N * T)
    # ----------------------------------
    # --- Computation of the V(k,Fk) ---
    # ----------------------------------
    V = np.zeros(kmax)  # Vector of V(k,Fk) for k=1,...,kmax

    for k in range(1, kmax + 1):
        if T < N:
            factors_k = np.sqrt(T) * eigenvectors[:, :k]
            loadings_k = X.T @ factors_k / T
            betahat_k = loadings_k @ _safe_cholesky(
                loadings_k.T @ loadings_k / N,
                context=f"the {k}-factor residual variance",
            )
        else:
            loadings_k = np.sqrt(N) * eigenvectors[:, :k]
            factors_k = X @ loadings_k / N
            betahat_k = (X.T @ X) @ loadings_k / (N * T)

        G = betahat_k.T @ betahat_k
        G = 0.5 * (G + G.T)

        try:
            G_inv = inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G, rcond=1e-10)

        Z_k = X - X @ betahat_k @ G_inv @ betahat_k.T

        V[k - 1] = np.sum(Z_k ** 2) / (N * T)

    # -----------------------------------
    # --- Panel Information Criteria ----
    # -----------------------------------

    IC = np.zeros((kmax + 1, 4))
    IC[:, 0] = np.arange(kmax + 1)

    PC = np.zeros((kmax + 1, 4))
    PC[:, 0] = np.arange(kmax + 1)

    AIC3 = np.zeros((kmax + 1, 2))
    AIC3[:, 0] = np.arange(kmax + 1)

    BIC3 = np.zeros((kmax + 1, 2))
    BIC3[:, 0] = np.arange(kmax + 1)

    mean_X_squared = np.mean(np.sum(X * X / T, axis=0))
    IC[0, 1:4] = np.log(mean_X_squared)

    PC[0, 1:4] = mean_X_squared

    AIC3[0, 1] = mean_X_squared
    BIC3[0, 1] = mean_X_squared

    CNT = min(N, T)
    penalty = np.array([
        (N + T) / (N * T) * np.log((N * T) / (N + T)),
        (N + T) / (N * T) * np.log(CNT),
        np.log(CNT) / CNT
    ])

    kk = np.arange(1, kmax + 1).reshape(-1, 1)

    # PC information Criteria (PC1, PC2 and PC3)
    PC[1:, 1:4] = V.reshape(-1, 1) + var_Z_kmax * kk * penalty

    # IC information Criteria (IC1, IC2 and IC3)
    with np.errstate(divide="ignore", invalid="ignore"):
        IC[1:, 1:4] = np.log(V.reshape(-1, 1)) + kk * penalty

    # BIC3 criterion
    BIC3[1:, 1] = V + kk.flatten() * var_Z_kmax * (N + T - kk.flatten()) * np.log(N * T) / (N * T)

    # AIC3 criterion
    AIC3[1:, 1] = V + kk.flatten() * var_Z_kmax * (N + T - kk.flatten()) * 2 / (N * T)

    # ------------------------------------------
    # --- Estimated Number of Common Factors ---
    # ------------------------------------------

    # Criterion rows start at zero factors, so the minimizing index is the count.
    PCs = PC[:, 1:4]
    khat_PC = np.argmin(PCs, axis=0)

    # Estimated Numbers of Factor with IC
    ICs = IC[:, 1:4]
    khat_IC = np.argmin(ICs, axis=0)

    # Estimated Numbers of Factor with BIC3
    BIC3s = BIC3[:, 1]
    khat_BIC3 = np.argmin(BIC3s)

    # Estimated Numbers of Factor with AIC3
    AIC3s = AIC3[:, 1]
    khat_AIC3 = np.argmin(AIC3s)

    khat = np.concatenate([khat_IC, khat_PC, [khat_AIC3], [khat_BIC3]])

    # ================
    # === RESULTS ====
    # ================

    results = {
        'khat': khat,
        'khat_IC': khat_IC,
        'khat_PC': khat_PC,
        'khat_BIC3': khat_BIC3,
        'khat_AIC3': khat_AIC3,
        'kmax': kmax,
        'IC': IC,
        'PC': PC,
        'AIC3': AIC3,
        'BIC3': BIC3,
        'Vkmax': var_Z_kmax
    }

    return results
