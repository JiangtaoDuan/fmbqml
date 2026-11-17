
"""Brownian-motion simulation shared by break tests."""

import numpy as np


def simulate_brownian_motion(t, n, dim, rng=None):
    """Simulate a discrete-time approximation to Brownian motion."""
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
