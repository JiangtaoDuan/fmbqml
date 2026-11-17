"""Tests for structural-break classification."""

import numpy as np

from fmbqml.break_tools.classify_break_types import classify_break_types


def test_classification_minimum_length_uses_ceiling():
    X = np.zeros((101, 10))
    result = classify_break_types(
        X,
        breakpoints=[],
        max_factors=3,
        trim_ratio=0.1,
        full_sample_factor_number=2,
    )
    assert result["min_regime_length"] == 11


def test_classification_real_factor_selection_runs_without_mocks():
    rng = np.random.default_rng(81)
    T, N = 90, 16
    f = rng.normal(size=(T, 2))
    loadings = rng.normal(size=(2, N))
    x = f @ loadings + 0.1 * rng.normal(size=(T, N))
    result = classify_break_types(
        x, [45], max_factors=4, min_regime_length=10,
        factor_criterion="IC2", full_sample_factor_number=3,
    )
    assert len(result["break_types"]) == 1
    assert result["regime_boundaries"] == [0, 45, 90]
    assert len(result["combined_factors"]) == 1
    assert result["break_types"][0] in {
        "Type 1 (Singular)", "Type 2 (Rotational)",
        "Unknown (invalid rank relation)",
    }
