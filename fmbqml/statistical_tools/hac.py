
"""HAC covariance estimation shared by break tests."""

from numbers import Real

import numpy as np

from fmbqml.factor_tools.var_selection import fit_var


def hac_newey_west_94(X, e, cons, kernel, a, pw, p_min, p_max, m):
    """Estimate a HAC covariance matrix using Newey and West (1994).

    The ``cons`` option controls the bandwidth-selector weight vector.
    VAR prewhitening uses ``fit_var(..., cons=0)`` without an intercept,
    irrespective of the HAC-level ``cons`` value.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    T_x, k = X.shape

    e = np.asarray(e, dtype=float)
    if e.ndim == 1:
        e = e.reshape(-1, 1)
    elif e.ndim == 2 and e.shape[1] == 1:
        pass
    else:
        raise ValueError("e must be a one-dimensional vector or a (T, 1) column.")
    if e.shape[0] != T_x:
        raise ValueError("e must have the same number of rows as X.")

    if cons not in (0, 1):
        raise ValueError("cons must be 0 or 1.")

    if kernel not in (0, 1):
        raise ValueError("kernel must be 0 or 1.")

    if pw not in (0, 1):
        raise ValueError("pw must be 0 or 1.")

    if cons == 0:
        w = np.ones((k, 1))
    else:
        w = np.zeros((k, 1))
        w[1:] = 1.0

    Y = X * e

    if pw == 0:
        Z = Y
    else:
        Betahat, Z = fit_var(Y, p_min, p_max, 0)

    T = Z.shape[0]

    if kernel == 0:
        V = Z.T @ Z / T

    else:
        if (
            isinstance(a, bool)
            or not isinstance(a, Real)
            or not np.isfinite(a)
            or a <= 0
        ):
            raise ValueError("a must be a positive finite real number.")

        if (
            isinstance(m, bool)
            or not isinstance(m, Real)
            or not np.isfinite(m)
            or m <= 0
        ):
            raise ValueError("m must be a positive finite real number.")

        q = 1
        cr = 1.1447

        # Newey--West (1994) preliminary bandwidth.
        n = min(
            T - 1,
            int(np.floor(a * (T / 100.0) ** (2.0 / 9.0))),
        )

        s = np.zeros(n + 1)
        s0 = 0.0
        sq = 0.0

        for lag in range(n + 1):
            scalar = (
                w.T
                @ (Z[:T - lag, :].T @ Z[lag:, :] / T)
                @ w
            )

            s[lag] = float(scalar.item())
            s0 += 2.0 * s[lag]
            sq += 2.0 * (lag ** q) * s[lag]

        s0 -= s[0]

        rhat = m * cr * ((sq / s0) ** 2) ** (
            1.0 / (2 * q + 1)
        )

        ST = int(
            np.floor(rhat * T ** (1.0 / (2 * q + 1)))
        )

        ST = min(max(ST, 0), T - 1)

        gamma0 = Z.T @ Z / T
        phi_hat = np.zeros((k, k))

        for lag in range(ST + 1):
            z1 = Z[:T - lag, :]
            z2 = Z[lag:, :]

            gamma = z1.T @ z2 / T

            weight = 1.0 - lag / (ST + 1.0)

            phi_hat += weight * (gamma + gamma.T)

        V = phi_hat - gamma0

    V = 0.5 * (V + V.T)

    if pw == 0:
        return V

    # Recolor using the inverse VAR transformation.
    D = np.eye(k)

    for lag in range(Betahat.shape[1] // k):
        start = lag * k
        D -= Betahat[:, start:start + k]

    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "The VAR recoloring matrix is singular; HAC prewhitening cannot be applied."
        ) from exc

    output = D_inv @ V @ D_inv.T

    return 0.5 * (output + output.T)


__all__ = ["hac_newey_west_94"]
