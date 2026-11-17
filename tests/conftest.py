"""Shared deterministic data for user-facing workflow tests."""

import numpy as np
import pytest


@pytest.fixture
def loading_break_panel():
    """A low-noise two-factor panel with one loading break at t=50."""
    rng = np.random.default_rng(42)
    factors = rng.normal(size=(100, 2))
    before = rng.normal(size=(16, 2))
    after = rng.normal(size=(16, 2))
    signal = np.vstack((factors[:50] @ before.T, factors[50:] @ after.T))
    return signal + 0.15 * rng.normal(size=(100, 16))
