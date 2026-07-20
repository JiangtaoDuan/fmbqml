"""HAC covariance estimation shared by break tests."""

import numpy as np

from fmbqml.factor_tools.var_selection import fit_var


def hac_newey_west_94(X, e, cons, kernel, a, pw, p_min, p_max, m):
    """Estimate a HAC covariance matrix using Newey and West (1994)."""
    _, k = X.shape

    if cons == 0:
        w = np.ones((k, 1))
    elif cons == 1:
        w = np.zeros((k, 1))
        w[1:] = 1
    else:
        raise ValueError("cons must be 0 or 1.")

    Y = X * np.tile(e, (1, k))
    if pw == 0:
        Z = Y
    elif pw == 1:
        Betahat, ehat = fit_var(Y, p_min, p_max, 0)
        Z = ehat
    else:
        raise ValueError("pw must be 0 or 1.")

    T = Z.shape[0]
    if kernel == 0:
        V = Z.T @ Z / T
    elif kernel == 1:
        q = 1
        cr = 1.1447
        n = int(np.floor(a * (T / 100) ** (2 / 9)))
        s = np.zeros(n + 1)
        s0 = 0.0
        sq = 0.0

        for lag in range(n + 1):
            scalar = w.T @ (Z[:T - lag, :].T @ Z[lag:, :] / T) @ w
            s[lag] = float(scalar.item())
            s0 += 2 * s[lag]
            sq += 2 * (lag ** q) * s[lag]

        s0 -= s[0]
        rhat = m * cr * ((sq / s0) ** 2) ** (1 / (2 * q + 1))
        ST = int(np.floor(rhat * T ** (1 / (2 * q + 1))))
        gamma = [None] * (ST + 1)
        phi_hat = np.zeros((k, k))

        for lag in range(ST + 1):
            if T - lag <= 0:
                gamma[lag] = np.zeros((k, k))
            else:
                z1 = Z[:T - lag, :]
                z2 = Z[lag:, :]
                gamma[lag] = z1.T @ z2 / T
            phi_hat += (1 - lag / (ST + 1)) * (
                gamma[lag] + gamma[lag].T
            )
        V = phi_hat - gamma[0]
    else:
        raise ValueError("kernel must be 0 or 1.")

    if pw == 0:
        return V

    D = np.eye(k)
    for lag in range(1, Betahat.shape[1] // k + 1):
        D -= Betahat[:, (lag - 1) * k: lag * k]
    rcond = 1e-10
    try:
        if np.linalg.cond(D) < 1.0 / rcond:
            return np.linalg.inv(D) @ V @ np.linalg.inv(D.T)
    except np.linalg.LinAlgError:
        pass

    # Preserve the original inverse-based calculation for well-conditioned D;
    # use a Moore-Penrose inverse only when D is singular or nearly singular.
    D_pinv = np.linalg.pinv(D, rcond=rcond)
    return D_pinv @ V @ D_pinv.T


__all__ = ["hac_newey_west_94"]
