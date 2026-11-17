"""Deterministic helpers for single-break tests."""

import numpy as np

from fmbqml import SingleBreakQML


def generate_panel_data(T: int = 300, N: int = 100, seed: int = 123) -> np.ndarray:
    """Generate a synthetic panel dataset for testing."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N))


def make_fake_factor_estimator(model: SingleBreakQML, r: int = 2, seed: int = 1234):
    """Create a deterministic fake factor-estimation method for fast tests."""
    rng = np.random.default_rng(seed)
    fake_F_hat = rng.normal(size=(model.T, r))
    fake_L_hat = rng.normal(size=(model.N, r))
    fake_VNT = np.array([[1.0]])

    def fake_calculate_factors():
        model._nbhat_r = r
        model._F_hat = fake_F_hat
        model._L_hat = fake_L_hat
        model._VNT = fake_VNT
        return r, fake_F_hat

    return fake_calculate_factors


def make_panel_data():
    rng = np.random.default_rng(42)
    factors = rng.normal(size=(80, 2))
    return factors @ rng.normal(size=(2, 12)) + rng.normal(scale=0.1, size=(80, 12))
