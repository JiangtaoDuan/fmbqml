"""Variance-based utilities for factor and rank selection."""

import numpy as np
from typing import Union

def rth(array: Union[list, np.ndarray], r: int) -> np.ndarray:
    """
    Extract the r largest elements from an array.

    Parameters
    ----------
    array : list or np.ndarray
        Input array of numeric values.
    r : int
        Number of largest elements to extract.

    Returns
    -------
    np.ndarray
        Array of shape (r, 2), where the first column contains
        0-based indices of the selected elements and the second
        column contains their values.
    """
    arr = np.asarray(array, dtype=float).ravel()
    k = arr.size
    m = arr.min() - 1.0

    indices = np.zeros(r, dtype=int)
    values = np.full(r, m, dtype=float)

    working = arr.copy()
    for a in range(r):
        for b in range(k):
            if values[a] < working[b]:
                values[a] = working[b]
                indices[a] = b
        working[indices[a]] = m - 2.0

    return np.column_stack([indices, values])


def fit_var(data: np.ndarray, p_min: int, p_max: int, cons: int):
    """
    Fit a VAR model over a range of lag orders and select the
    optimal specification using the BIC criterion.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix with shape (T, k), where T is the
        sample size and k is the number of variables.
    p_min : int
        Minimum lag order considered.
    p_max : int
        Maximum lag order considered.
    cons : int
        Indicator for including an intercept term.
        Use 0 for no intercept and 1 for an intercept.

    Returns
    -------
    tuple
        (phi_opt, e_opt) where

        phi_opt : np.ndarray
            Estimated coefficient matrix for the selected VAR model.

        e_opt : np.ndarray
            Residual matrix from the selected VAR model.
    """
    data = np.asarray(data, dtype=float)
    T, k = data.shape

    V = [None] * (p_max + 1)
    ehat = [None] * (p_max + 1)
    Phi = [None] * (p_max + 1)
    BIC = np.zeros(p_max + 1)

    for p in range(p_min, p_max + 1):
        Y = data[p:, :]
        X = np.ones((T - p, 1))

        if p == 0:
            if cons == 0:
                Phizero = np.zeros((k, k))
                ehatzero = data.copy()
                num_para = 0
            elif cons == 1:
                mean_vec = np.mean(data, axis=0)
                Phizero = np.column_stack([mean_vec, np.zeros((k, k))])
                ehatzero = data - mean_vec
                num_para = k
            else:
                raise ValueError("cons must be 0 or 1")

            Vzero = ehatzero.T @ ehatzero / (T - num_para)
            BICzero = np.log(np.linalg.det(Vzero)) + num_para * np.log(T) / T

        else:
            for r in range(1, p + 1):
                X = np.column_stack([X, data[p - r:T - r, :]])

            if cons == 0:
                X = X[:, 1:]
                num_para = (k ** 2) * p
            elif cons == 1:
                num_para = (k ** 2) * p + k
            else:
                raise ValueError("cons must be 0 or 1")

            XtX = X.T @ X
            XtY = X.T @ Y
            Phi_p = np.linalg.solve(XtX, XtY)
            ehat_p = Y - X @ Phi_p
            V_p = ehat_p.T @ ehat_p / T
            BIC_p = np.log(np.linalg.det(V_p)) + num_para * np.log(T) / T
            Phi[p] = Phi_p
            ehat[p] = ehat_p
            V[p] = V_p
            BIC[p] = BIC_p

    start = max(p_min, 1)
    segment = -BIC[start:p_max + 1].reshape(-1, 1)
    selected_idx = rth(segment, 1)

    phat = int(selected_idx[0, 0] + start)

    if phat == 0:
        phi_opt = Phizero
        e_opt = ehatzero
    else:
        phi_opt = Phi[phat].T
        e_opt = ehat[phat]

    if p_min == 0:
        if BICzero < BIC[phat]:
            phi_opt = Phizero
            e_opt = ehatzero

    return phi_opt, e_opt
