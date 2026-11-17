"""Deterministic helpers and stubs for multiple-break tests."""

import numpy as np


def generate_panel_data(T: int = 300, N: int = 20, seed: int = 123) -> np.ndarray:
    """Generate a deterministic synthetic panel dataset for tests."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N))


def normalized_factors(seed: int = 0, T: int = 36, r: int = 2) -> np.ndarray:
    """Generate factors normalized so that ``F'F/T`` is the identity."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(T, r))
    gram = factors.T @ factors / T
    values, vectors = np.linalg.eigh(gram)
    return factors @ vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def make_fake_ic_path():
    """Create a minimal information-criterion path."""
    return [
        {"m": 0, "IC": 10.0, "loss": 10.0, "estimated_breakpoints": []},
        {"m": 1, "IC": 8.0, "loss": 7.0, "estimated_breakpoints": [20]},
        {"m": 2, "IC": 6.0, "loss": 5.0, "estimated_breakpoints": [20, 50]},
    ]


def fake_estimate_panel_factors(X, n_factors):
    """Fast deterministic stand-in for factor estimation."""
    T, N = X.shape
    F_hat = np.zeros((T, n_factors))
    L_hat = np.zeros((N, n_factors))
    VNT = np.eye(n_factors)
    return F_hat, L_hat, VNT


def fake_classification_fields(T: int = 80):
    """Classification fields with one Type 1 and one Type 2 breakpoint."""
    break_types = ["Type 1 (Singular)", "Type 2 (Rotational)"]
    return {
        "break_types": break_types,
        "break_type": break_types[0],
        "regime_factors": [2, 2, 2],
        "combined_factors": [3, 2],
        "classification_criteria": [
            {
                "break_index": 20,
                "r_j": 2,
                "r_j1": 2,
                "r_j_j1": 3,
                "min_r": 2,
                "condition": "r_j,j+1 (3) > min(r_j,r_j+1) (2)",
                "type": "Type 1 (Singular)",
            },
            {
                "break_index": 50,
                "r_j": 2,
                "r_j1": 2,
                "r_j_j1": 2,
                "min_r": 2,
                "condition": "r_j = r_j+1 = r_j,j+1 = 2",
                "type": "Type 2 (Rotational)",
            },
        ],
        "regime_boundaries": [0, 20, 50, T],
        "n_regimes": 3,
        "classification_r_full": 3,
        "classification_factor_criterion": 2,
        "min_regime_length": None,
        "break_type_summary": {
            "type1_count": 1,
            "type2_count": 1,
            "other_count": 0,
        },
        "classification_full_sample_standardize": False,
        "classification_segment_standardize": True,
    }


def make_fake_classify_breakpoint_result(
    expected_breakpoints=None, expected_trim_ratio=None,
    expected_min_regime_length=None,
    expected_factor_criterion=None,
    T: int = 80,
):
    """
    Return a fake classify_breakpoint_result() function matching the current
    MultiBreakQML implementation.
    """
    def fake_classify_breakpoint_result(
        X,
        breakpoints,
        result_dict,
        max_factors,
        factor_criterion,
        full_sample_factor_number,
        T,
        trim_ratio=None,
        min_regime_length=None,
        full_sample_standardize=False,
        segment_standardize=True,
        verbose=False,
    ):
        if expected_breakpoints is not None:
            assert breakpoints == expected_breakpoints
        if expected_trim_ratio is not None:
            assert trim_ratio == expected_trim_ratio
        if expected_min_regime_length is not None:
            assert min_regime_length == expected_min_regime_length
        if expected_factor_criterion is not None:
            assert factor_criterion == expected_factor_criterion

        assert full_sample_factor_number == result_dict.get("estimated_factors")
        assert full_sample_standardize is False
        assert segment_standardize is True

        classification = fake_classification_fields(T=T)
        classification["min_regime_length"] = min_regime_length

        updated_result = result_dict.copy()
        updated_result.update(classification)

        return updated_result, classification

    return fake_classify_breakpoint_result
