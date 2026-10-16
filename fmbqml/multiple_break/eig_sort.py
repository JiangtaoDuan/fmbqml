"""Eigenvalue sorting helpers for multiple-break estimation."""

import numpy as np
from scipy.sparse.linalg import eigs


def eig_sort(X, r):

    """
    Perform eigenvalue decomposition of a square matrix X, sort the eigenvalues
    in descending order, and return the leading r eigenvectors together with the
    corresponding diagonal eigenvalue matrix.

    Parameters
    ----------
    X : array-like, shape (n, n)
        Input square matrix.
    r : int
        Number of leading eigenvalues/eigenvectors to retain.

    Returns
    -------
    Us : ndarray, shape (n, r)
        The leading r eigenvectors after sorting.
    Ds : ndarray, shape (r, r)
        Diagonal matrix containing the corresponding leading r eigenvalues.
    """

    X = np.asarray(X)

    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix.")

    n = X.shape[0]
    r = int(r)

    if r <= 0:
        raise ValueError("r must be positive.")

    r = min(r, n)

    if np.allclose(X, X.T.conj(), rtol=1e-10, atol=1e-12):
        eigvals, eigvecs = np.linalg.eigh(X)

    else:
        if r >= n - 1:
            eigvals, eigvecs = np.linalg.eig(X)
        else:
            eigvals, eigvecs = eigs(X, k=r)

    if np.max(np.abs(np.imag(eigvals))) < 1e-10:
        eigvals_real = np.real(eigvals)
        eigvecs_real = np.real(eigvecs)

        idx = np.argsort(eigvals_real)[::-1][:r]

        Ds = np.diag(eigvals_real[idx])
        Us = eigvecs_real[:, idx]

    else:
        idx = np.argsort(np.abs(eigvals))[::-1][:r]

        Ds = np.diag(eigvals[idx])
        Us = eigvecs[:, idx]

    return Us, Ds
