
"""Brownian-motion simulation shared by break tests."""

import numpy as np


def simulate_brownian_motion(t, n, dim, rng=None):
    """Simulate a discrete-time approximation to Brownian motion.

    Parameters
    ----------
    t : float
        Time horizon used to determine the number of simulated grid points.
    n : int
        Number of time steps per unit of time; increments have variance
        ``1 / n`` in each dimension.
    dim : int
        Number of independent Brownian-motion dimensions.
    rng : numpy.random.Generator or None, default=None
        Random number generator. A new default generator is used when None.

    Returns
    -------
    path : ndarray of shape (dim, round(t * n))
        Simulated paths at steps ``1 / n, 2 / n, ...``. The initial zero at
        time zero is not included.

    Raises
    ------
    ValueError
        If the requested time grid has no points.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_points = round(t * n)

    if grid_points < 1:
        raise ValueError(
            "The requested time grid has no simulation points."
        )

    path = np.zeros((dim, grid_points))

    path[:, 0] = rng.standard_normal(dim) / np.sqrt(n)

    for step in range(1, grid_points):
        path[:, step] = (
            path[:, step - 1]
            + rng.standard_normal(dim) / np.sqrt(n)
        )

    return path


__all__ = ["simulate_brownian_motion"]
