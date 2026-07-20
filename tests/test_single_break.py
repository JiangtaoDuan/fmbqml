import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from fmbqml import SingleBreakQML
import fmbqml.single_break.single_break_qml as qml_module
import fmbqml.statistical_tools.hac as hac_module

from fmbqml.statistical_tools import select_critical_value
from fmbqml.single_break.qml_profile import compute_single_break_qml_profile
from fmbqml.break_tools.classify_break_types import classify_break_types
from fmbqml.utils.breakpoint_utils import calculate_break_range, map_breakpoint_to_label
from fmbqml.utils.validation import validate_alpha


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


def test_single_break_classification_passes_ceiling_length(monkeypatch):
    X = generate_panel_data(T=101, N=20)
    model = SingleBreakQML(X, trim_ratio=0.1, max_factors=3)
    model.is_tested = True
    model.test_results = {"break_point": 50, "estimated_factors": 2}
    captured = {}

    def fake_classify_breakpoint_result(**kwargs):
        captured.update(kwargs)
        return kwargs["result_dict"].copy(), {}

    monkeypatch.setattr(
        qml_module,
        "classify_breakpoint_result",
        fake_classify_breakpoint_result,
    )
    model.classify_break()
    assert captured["min_regime_length"] == 11


def test_hac_prewhitening_preserves_original_inverse_path(monkeypatch):
    X = np.arange(24, dtype=float).reshape(8, 3)
    e = np.ones((8, 1))
    betahat = 0.25 * np.eye(3)
    residuals = X.copy()
    monkeypatch.setattr(
        hac_module,
        "fit_var",
        lambda data, p_min, p_max, cons: (betahat, residuals),
    )

    result = hac_module.hac_newey_west_94(
        X, e, cons=0, kernel=0, a=4, pw=1, p_min=1, p_max=1, m=1.5
    )
    V = residuals.T @ residuals / residuals.shape[0]
    D = np.eye(3) - betahat
    expected = np.linalg.inv(D) @ V @ np.linalg.inv(D.T)
    np.testing.assert_array_equal(result, expected)


def test_hac_prewhitening_uses_pseudoinverse_for_singular_matrix(monkeypatch):
    X = np.arange(16, dtype=float).reshape(8, 2)
    e = np.ones((8, 1))
    betahat = np.eye(2)
    residuals = X.copy()
    monkeypatch.setattr(
        hac_module,
        "fit_var",
        lambda data, p_min, p_max, cons: (betahat, residuals),
    )

    result = hac_module.hac_newey_west_94(
        X, e, cons=0, kernel=0, a=4, pw=1, p_min=1, p_max=1, m=1.5
    )
    np.testing.assert_array_equal(result, np.zeros((2, 2)))
    assert np.isfinite(result).all()


def test_hac_prewhitening_uses_pseudoinverse_for_nearly_singular_matrix(
    monkeypatch,
):
    X = np.arange(16, dtype=float).reshape(8, 2)
    e = np.ones((8, 1))
    D = np.diag([1.0, 1e-12])
    betahat = np.eye(2) - D
    residuals = X.copy()
    monkeypatch.setattr(
        hac_module,
        "fit_var",
        lambda data, p_min, p_max, cons: (betahat, residuals),
    )

    result = hac_module.hac_newey_west_94(
        X, e, cons=0, kernel=0, a=4, pw=1, p_min=1, p_max=1, m=1.5
    )
    V = residuals.T @ residuals / residuals.shape[0]
    D_pinv = np.linalg.pinv(D, rcond=1e-10)
    expected = D_pinv @ V @ D_pinv.T
    np.testing.assert_allclose(result, expected)
    assert np.isfinite(result).all()


def test_single_break_result_getter_returns_nested_copies():
    model = SingleBreakQML(
        np.arange(80, dtype=float).reshape(20, 4),
        max_factors=2,
    )
    model.is_tested = True
    model.test_results = {
        "candidate_breaks": np.array([5, 6]),
        "classification_criteria": [{"condition": "original"}],
    }

    returned = model.get_test_results()
    returned["candidate_breaks"][0] = 999
    returned["classification_criteria"][0]["condition"] = "changed"

    assert model.test_results["candidate_breaks"][0] == 5
    assert model.test_results["classification_criteria"][0]["condition"] == "original"


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


# ----------------------------
# Initialization and input checks
# ----------------------------

def test_init_with_valid_numpy_array():
    X = generate_panel_data()
    model = SingleBreakQML(
        X,
        trim_ratio=0.3,
        max_factors=5,
        factor_criterion=2,
    )

    assert model.T == 300
    assert model.N == 100
    assert model.trim_ratio == 0.3
    assert model.max_factors == 5
    assert model.factor_criterion == 2
    assert model.is_tested is False
    assert model.test_results == {}


def test_init_with_valid_dataframe():
    X = pd.DataFrame(generate_panel_data())
    model = SingleBreakQML(X)

    assert model.T == X.shape[0]
    assert model.N == X.shape[1]

    returned = model.get_data()
    assert isinstance(returned, pd.DataFrame)
    assert returned.shape == X.shape


def test_init_with_dataframe_index_labels():
    X = pd.DataFrame(
        generate_panel_data(T=20, N=5),
        index=pd.date_range("2000-01-01", periods=20, freq="MS"),
    )
    model = SingleBreakQML(X, trim_ratio=0.2, max_factors=2)

    assert map_breakpoint_to_label(1, model._row_index) == X.index[0]
    assert map_breakpoint_to_label(10, model._row_index) == X.index[9]
    assert map_breakpoint_to_label(None, model._row_index) is None


def test_init_with_invalid_dimension():
    X = np.random.randn(100)

    with pytest.raises(ValueError, match="2D array"):
        SingleBreakQML(X)


def test_init_with_nan_input():
    X = generate_panel_data()
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN|Inf|infinite"):
        SingleBreakQML(X)


def test_init_with_inf_input():
    X = generate_panel_data()
    X[0, 0] = np.inf

    with pytest.raises(ValueError, match="NaN|Inf|infinite"):
        SingleBreakQML(X)


# ----------------------------
# Parameter validation and utility functions
# ----------------------------

def test_invalid_trim_ratio():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="trim_ratio"):
        SingleBreakQML(X, trim_ratio=0.0)

    with pytest.raises(ValueError, match="trim_ratio"):
        SingleBreakQML(X, trim_ratio=0.6)

    with pytest.raises(ValueError, match="trim_ratio"):
        SingleBreakQML(X, trim_ratio=True)


def test_validate_alpha_function():
    with pytest.raises(ValueError, match="Significance level"):
        validate_alpha(0.2)

    with pytest.raises(ValueError, match="Significance level"):
        validate_alpha(True)

    assert validate_alpha(0.01) == 0.01
    assert validate_alpha(0.05) == 0.05
    assert validate_alpha(0.10) == 0.10


def test_invalid_max_factors_bool():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="max_factors"):
        SingleBreakQML(X, max_factors=True)


def test_invalid_max_factors_too_large():
    X = generate_panel_data(T=20, N=5)

    with pytest.raises(ValueError, match="max_factors"):
        SingleBreakQML(X, max_factors=6)


def test_invalid_factor_criterion():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="factor_criterion"):
        SingleBreakQML(X, factor_criterion=8)

    with pytest.raises(ValueError, match="factor_criterion"):
        SingleBreakQML(X, factor_criterion=True)

    with pytest.raises(ValueError, match="factor_criterion"):
        SingleBreakQML(X, factor_criterion="BAD")


def test_valid_factor_criterion_string_is_normalized():
    X = generate_panel_data()
    model = SingleBreakQML(X, factor_criterion="IC2")

    assert model.factor_criterion == 1


def test_invalid_break_search_range():
    X = generate_panel_data(T=2, N=3)

    with pytest.raises(ValueError, match="Invalid break search range"):
        SingleBreakQML(X, trim_ratio=0.49, max_factors=1)


def test_calculate_break_range_function():
    t_min, t_max = calculate_break_range(T=200, trim_ratio=0.1)

    assert t_min == 20
    assert t_max == 180


def test_select_critical_value_function():
    cv = np.array([1.0, 2.0, 3.0])

    assert select_critical_value(cv, 0.10) == 1.0
    assert select_critical_value(cv, 0.05) == 2.0
    assert select_critical_value(cv, 0.01) == 3.0

    with pytest.raises(ValueError, match="alpha"):
        select_critical_value(cv, 0.2)


# ----------------------------
# Accessors before running tests
# ----------------------------

def test_get_test_results_before_run():
    X = generate_panel_data()
    model = SingleBreakQML(X)

    with pytest.raises(ValueError, match=r"Run lr_test\(\) or estimate_breakpoint\(\)"):
        model.get_test_results()


def test_summary_before_run():
    X = generate_panel_data()
    model = SingleBreakQML(X)

    with pytest.raises(ValueError, match="No test has been executed yet"):
        model.summary()


def test_get_profiles_before_run():
    X = generate_panel_data()
    model = SingleBreakQML(X)

    with pytest.raises(ValueError, match="Run lr_test"):
        model.get_lr_profile()

    with pytest.raises(ValueError, match=r"Run lr_test\(\) or estimate_breakpoint\(\)"):
        model.get_qml_profile()


# ----------------------------
# get_data
# ----------------------------

def test_get_data_returns_copy_for_numpy():
    X = generate_panel_data()
    model = SingleBreakQML(X)

    returned = model.get_data()
    assert isinstance(returned, np.ndarray)
    assert returned.shape == X.shape

    returned[0, 0] = 999.0
    assert model.X[0, 0] != 999.0


def test_get_data_returns_copy_for_dataframe():
    X = pd.DataFrame(generate_panel_data())
    model = SingleBreakQML(X)

    returned = model.get_data()
    assert isinstance(returned, pd.DataFrame)

    returned.iloc[0, 0] = 999.0
    assert model.get_data().iloc[0, 0] != 999.0


# ----------------------------
# set_data and set_parameters
# ----------------------------

def test_set_data_updates_shape_and_resets_state():
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X)

    new_X = generate_panel_data(T=100, N=100, seed=456)
    model.set_data(new_X)

    assert model.T == 100
    assert model.N == 100
    assert model.is_tested is False
    assert model.test_results == {}
    assert model._F_hat is None
    assert model._L_hat is None
    assert model._VNT is None
    assert model._nbhat_r is None


def test_set_data_invalid_reverts_state():
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, max_factors=5)

    old_T, old_N = model.T, model.N
    old_X = model.X.copy()
    bad_X = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        model.set_data(bad_X)

    assert model.T == old_T
    assert model.N == old_N
    np.testing.assert_allclose(model.X, old_X)


def test_set_data_revalidates_parameters():
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, max_factors=50)

    new_X = generate_panel_data(T=40, N=20)

    with pytest.raises(ValueError, match="max_factors"):
        model.set_data(new_X)

    assert model.T == 200
    assert model.N == 100


def test_set_parameters_updates_values():
    X = generate_panel_data()
    model = SingleBreakQML(X)

    model.set_parameters(
        trim_ratio=0.2,
        max_factors=4,
        factor_criterion="PC1",
    )

    assert model.trim_ratio == 0.2
    assert model.max_factors == 4
    assert model.factor_criterion == 3
    assert model.is_tested is False
    assert model.test_results == {}


def test_set_parameters_invalid_factor_criterion_reverts_state():
    X = generate_panel_data()
    model = SingleBreakQML(
        X,
        trim_ratio=0.3,
        max_factors=5,
        factor_criterion=0,
    )

    old_trim_ratio = model.trim_ratio
    old_max_factors = model.max_factors
    old_factor_criterion = model.factor_criterion

    with pytest.raises(ValueError, match="factor_criterion"):
        model.set_parameters(factor_criterion=8)

    assert model.trim_ratio == old_trim_ratio
    assert model.max_factors == old_max_factors
    assert model.factor_criterion == old_factor_criterion


def test_set_parameters_invalid_trim_ratio_reverts_state():
    X = generate_panel_data()
    model = SingleBreakQML(
        X,
        trim_ratio=0.3,
        max_factors=5,
        factor_criterion=0,
    )

    old_trim_ratio = model.trim_ratio
    old_max_factors = model.max_factors
    old_factor_criterion = model.factor_criterion

    with pytest.raises(ValueError):
        model.set_parameters(trim_ratio=0.6)

    assert model.trim_ratio == old_trim_ratio
    assert model.max_factors == old_max_factors
    assert model.factor_criterion == old_factor_criterion


# ----------------------------
# Factor estimation accessor
# ----------------------------

def test_calculate_factors_zero_factor_raises(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, max_factors=5, factor_criterion=0)

    def fake_bai_ng_factor_count(X, max_factors):
        return {"khat": np.zeros(8, dtype=int)}

    monkeypatch.setattr(qml_module, "bai_ng_factor_count", fake_bai_ng_factor_count)

    with pytest.raises(ValueError, match="at least one factor"):
        model._calculate_factors()

    assert model._nbhat_r == 0
    assert model._F_hat is None
    assert model._L_hat is None
    assert model._VNT is None


def test_calculate_factors_positive_factor_runs(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, max_factors=5, factor_criterion=0)

    def fake_bai_ng_factor_count(X, max_factors):
        return {"khat": np.array([2, 2, 2, 2, 2, 2, 2, 2])}

    def fake_estimate_panel_factors_lr(X, r):
        F_hat = np.ones((X.shape[0], r))
        L_hat = np.ones((X.shape[1], r))
        VNT = np.array([[1.0]])
        return F_hat, L_hat, VNT

    monkeypatch.setattr(qml_module, "bai_ng_factor_count", fake_bai_ng_factor_count)
    monkeypatch.setattr(qml_module, "estimate_panel_factors_lr", fake_estimate_panel_factors_lr)

    r_hat, F_hat = model._calculate_factors()

    assert r_hat == 2
    assert F_hat.shape == (model.T, 2)
    assert model._L_hat.shape == (model.N, 2)
    assert model._VNT.shape == (1, 1)


def test_get_factor_estimates_returns_dict(monkeypatch):
    X = generate_panel_data()
    model = SingleBreakQML(X, factor_criterion=2)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    result = model.get_factor_estimates()

    assert isinstance(result, dict)
    assert result["estimated_factors"] == 2
    assert result["F_hat"].shape == (model.T, 2)
    assert result["L_hat"].shape == (model.N, 2)
    assert result["factor_criterion"] == 2


def test_get_factor_estimates_recalculate_false_uses_cache():
    X = generate_panel_data()
    model = SingleBreakQML(X, factor_criterion=1)
    model._nbhat_r = 3
    model._F_hat = np.ones((model.T, 3))
    model._L_hat = np.ones((model.N, 3))
    model._VNT = np.array([[1.0]])

    result = model.get_factor_estimates(recalculate=False)

    assert result["estimated_factors"] == 3
    assert result["F_hat"].shape == (model.T, 3)
    assert result["L_hat"].shape == (model.N, 3)


def test_get_factor_estimates_recalculate_true_recomputes(monkeypatch):
    X = generate_panel_data()
    model = SingleBreakQML(X, factor_criterion=1)
    model._nbhat_r = 3
    model._F_hat = np.ones((model.T, 3))
    model._L_hat = np.ones((model.N, 3))
    model._VNT = np.array([[1.0]])

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    result = model.get_factor_estimates(recalculate=True)

    assert result["estimated_factors"] == 2
    assert result["F_hat"].shape == (model.T, 2)
    assert result["L_hat"].shape == (model.N, 2)


# ----------------------------
# QML profile function
# ----------------------------

def test_compute_single_break_qml_profile_returns_array():
    T = 200
    rng = np.random.default_rng(321)
    F_hat = rng.normal(size=(T, 2))
    t_min, t_max = calculate_break_range(T, trim_ratio=0.1)

    result = compute_single_break_qml_profile(
        F_hat=F_hat,
        T=T,
        t_min=t_min,
        t_max=t_max,
        n_factors=2,
    )

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == (t_max - t_min + 1)


# ----------------------------
# classify_break
# ----------------------------

def test_classify_break_without_break_point():
    X = generate_panel_data()
    model = SingleBreakQML(X)
    model.is_tested = True
    model.test_results = {
        "test_type": "lr_test",
        "break_point": None,
        "estimated_factors": 2,
        "factor_criterion": model.factor_criterion,
        "T": model.T,
        "N": model.N,
    }

    result = model.classify_break(break_point=None)

    assert result["break_types"] == []
    assert result["regime_factors"] == []
    assert result["combined_factors"] == []
    assert result["classification_criteria"] == []
    assert result["regime_boundaries"] == [0, model.T]
    assert result["n_regimes"] == 1
    assert "min_regime_length" in result


def test_classify_break_updates_test_results(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, max_factors=5, factor_criterion=2)
    model.is_tested = True
    model.test_results = {
        "test_type": "estimate_breakpoint",
        "break_point": 80,
        "estimated_factors": 2,
        "factor_criterion": model.factor_criterion,
        "T": model.T,
        "N": model.N,
    }

    def fake_classify_breakpoint_result(**kwargs):
        updated = kwargs["result_dict"].copy()
        updated.update({
            "break_types": ["Type 1: singular"],
            "break_type": "Type 1: singular",
            "regime_factors": [1, 2],
            "combined_factors": [2],
            "classification_criteria": [
                {
                    "break_index": 80,
                    "type": "Type 1",
                    "r_j": 1,
                    "r_j1": 2,
                    "r_j_j1": 3,
                    "condition": "test condition",
                }
            ],
            "regime_boundaries": [0, 80, 200],
            "n_regimes": 2,
            "classification_r_full": 2,
            "classification_full_sample_standardize": False,
            "classification_segment_standardize": True,
            "break_type_summary": {
                "type1_count": 1,
                "type2_count": 0,
                "other_count": 0,
            },
        })
        return updated, updated.copy()

    monkeypatch.setattr(qml_module, "classify_breakpoint_result", fake_classify_breakpoint_result)

    result = model.classify_break(break_point=80)

    assert result["break_types"] == ["Type 1: singular"]
    assert result["break_type"] == "Type 1: singular"
    assert result["break_type_summary"]["type1_count"] == 1
    assert result["break_type_summary"]["type2_count"] == 0
    assert result["break_type_summary"]["other_count"] == 0
    assert result["classification_r_full"] == 2
    assert result["classification_full_sample_standardize"] is False
    assert result["classification_segment_standardize"] is True

    assert model.test_results["break_type"] == "Type 1: singular"
    assert model.test_results["break_type_summary"]["type1_count"] == 1


def test_classify_break_passes_expected_options(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, max_factors=5, factor_criterion="IC1")
    model.is_tested = True
    model.test_results = {
        "test_type": "estimate_breakpoint",
        "break_point": 100,
        "estimated_factors": 2,
        "factor_criterion": model.factor_criterion,
        "T": model.T,
        "N": model.N,
    }

    captured_kwargs = {}

    def fake_classify_breakpoint_result(**kwargs):
        captured_kwargs.update(kwargs)
        updated = kwargs["result_dict"].copy()
        updated.update({
            "break_types": ["Type 2: rotational"],
            "break_type": "Type 2: rotational",
            "regime_factors": [2, 2],
            "combined_factors": [2],
            "classification_criteria": [],
            "regime_boundaries": [0, 100, 200],
            "n_regimes": 2,
            "classification_r_full": 2,
            "classification_full_sample_standardize": False,
            "classification_segment_standardize": True,
            "break_type_summary": {
                "type1_count": 0,
                "type2_count": 1,
                "other_count": 0,
            },
        })
        return updated, updated.copy()

    monkeypatch.setattr(qml_module, "classify_breakpoint_result", fake_classify_breakpoint_result)

    result = model.classify_break(break_point=100)

    assert captured_kwargs["breakpoints"] == [100]
    assert captured_kwargs["max_factors"] == 5
    assert captured_kwargs["min_regime_length"] == 20
    assert captured_kwargs["factor_criterion"] == 0
    assert captured_kwargs["full_sample_factor_number"] == 2
    assert captured_kwargs["full_sample_standardize"] is False
    assert captured_kwargs["segment_standardize"] is True

    assert result["break_type"] == "Type 2: rotational"
    assert result["break_type_summary"]["type1_count"] == 0
    assert result["break_type_summary"]["type2_count"] == 1
    assert result["break_type_summary"]["other_count"] == 0


# ----------------------------
# estimate_breakpoint and QML profile
# ----------------------------

def test_estimate_breakpoint_runs_and_returns_dict(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    result = model.estimate_breakpoint(classify=False)

    assert isinstance(result, dict)
    assert result["test_type"] == "estimate_breakpoint"
    assert "break_point" in result
    assert "estimated_break_candidate" in result
    assert "estimated_break_candidate_label" in result
    assert "estimated_factors" in result
    assert "factor_criterion" in result
    assert "search_range" in result
    assert "candidate_breaks" in result
    assert "qml_profile" in result
    assert result["factor_criterion"] == 1
    assert isinstance(result["break_point"], (int, np.integer))
    assert model.is_tested is True


def test_estimate_breakpoint_default_does_not_print(monkeypatch, capsys):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    model.estimate_breakpoint(classify=False)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_estimate_breakpoint_verbose_prints_summary(monkeypatch, capsys):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    model.estimate_breakpoint(classify=False, verbose=True)

    captured = capsys.readouterr()
    assert "SingleBreakQML BREAKPOINT ESTIMATION SUMMARY" in captured.out
    assert "Accepted breakpoint:" in captured.out


def test_get_qml_profile_after_estimate_breakpoint(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    model.estimate_breakpoint(classify=False)
    qml_profile = model.get_qml_profile()

    assert isinstance(qml_profile, pd.DataFrame)
    assert list(qml_profile.columns) == ["candidate_break", "qml_objective"]
    assert len(qml_profile) == len(model.test_results["candidate_breaks"])


def test_plot_qml_profile_after_estimate_breakpoint(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")

    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    model.estimate_breakpoint(classify=False)
    fig, ax = model.plot_qml_profile(true_break=80, show=False)

    assert fig is not None
    assert ax is not None
    plt.close(fig)


# ----------------------------
# lr_test and LR profile
# ----------------------------

def test_lr_test_runs_and_returns_dict(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    result = model.lr_test(alpha=0.05, classify=False)

    assert isinstance(result, dict)
    assert result["test_type"] == "lr_test"
    assert "break_point" in result
    assert "estimated_break_candidate" in result
    assert "estimated_break_candidate_label" in result
    assert "reject_null" in result
    assert "test_statistic" in result
    assert "critical_value" in result
    assert "significance_level" in result
    assert "factor_criterion" in result
    assert "candidate_breaks" in result
    assert "lr_profile" in result
    assert "qml_profile" in result
    assert result["factor_criterion"] == 4
    assert result["significance_level"] == 0.05
    assert result["critical_value"] == 2.0
    assert result["monte_carlo_replications"] == 5000
    assert result["random_state"] is None
    assert isinstance(result["reject_null"], (bool, np.bool_))
    assert model.is_tested is True


def test_lr_test_passes_simulation_options(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    captured = {}

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        captured["trim_ratio"] = trim_ratio
        captured["n_sim"] = n_sim
        captured["random_state"] = random_state
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    result = model.lr_test(
        alpha=0.05,
        classify=False,
        n_sim=123,
        random_state=456,
    )

    assert captured["trim_ratio"] == 0.1
    assert captured["n_sim"] == 123
    assert captured["random_state"] == 456
    assert result["monte_carlo_replications"] == 123
    assert result["random_state"] == 456


def test_lr_test_default_does_not_print(monkeypatch, capsys):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_lr_test_verbose_prints_summary(monkeypatch, capsys):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(
        alpha=0.05,
        classify=False,
        verbose=True,
        n_sim=123,
        random_state=456,
    )

    captured = capsys.readouterr()
    assert "SingleBreakQML LR TEST SUMMARY" in captured.out
    assert "Reject null hypothesis of no break:" in captured.out
    assert "Break type:" in captured.out
    assert "Monte Carlo replications:" not in captured.out
    assert "Random state:" not in captured.out


def test_lr_test_invalid_alpha_raises_after_factor_step(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    with pytest.raises(ValueError, match="Significance level"):
        model.lr_test(alpha=0.2, classify=False)


def test_lr_test_rejects_and_classifies(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=0)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([-1e9, -1e9, -1e9])

    def fake_classify_break(break_point=None, verbose=False):
        model.test_results["break_types"] = ["Type 2: rotational"]
        model.test_results["break_type"] = "Type 2: rotational"
        model.test_results["break_type_summary"] = {
            "type1_count": 0,
            "type2_count": 1,
            "other_count": 0,
        }
        return model.test_results.copy()

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)
    monkeypatch.setattr(model, "classify_break", fake_classify_break)

    result = model.lr_test(alpha=0.05, classify=True)

    assert bool(result["reject_null"]) is True
    assert result["break_point"] is not None
    assert result["estimated_break_candidate"] == result["break_point"]
    assert model.test_results["break_type"] == "Type 2: rotational"


def test_lr_test_non_rejection_keeps_candidate(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=0)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1e9, 1e9, 1e9])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    result = model.lr_test(alpha=0.05, classify=True)

    assert bool(result["reject_null"]) is False
    assert result["break_point"] is None
    assert result["estimated_break_candidate"] is not None


def test_get_lr_profile_after_lr_test(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)
    lr_profile = model.get_lr_profile()

    assert isinstance(lr_profile, pd.DataFrame)
    assert list(lr_profile.columns) == ["candidate_break", "lr_statistic"]
    assert len(lr_profile) == len(model.test_results["candidate_breaks"])


def test_get_qml_profile_after_lr_test(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)
    qml_profile = model.get_qml_profile()

    assert isinstance(qml_profile, pd.DataFrame)
    assert list(qml_profile.columns) == ["candidate_break", "qml_objective"]
    assert len(qml_profile) == len(model.test_results["candidate_breaks"])


def test_plot_lr_profile_after_lr_test(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")

    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)
    fig, ax = model.plot_lr_profile(true_break=80, show=False)

    assert fig is not None
    assert ax is not None
    plt.close(fig)


# ----------------------------
# summary
# ----------------------------

def test_summary_after_estimate_breakpoint(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=6)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    model.estimate_breakpoint(classify=False)
    text = model.summary()

    assert isinstance(text, str)
    assert "SingleBreakQML BREAKPOINT ESTIMATION SUMMARY" in text
    assert "Accepted breakpoint:" in text
    assert "Estimated break candidate:" in text
    assert "Factor criterion:" in text


def test_summary_after_lr_test(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=0)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)
    text = model.summary()

    assert isinstance(text, str)
    assert "SingleBreakQML LR TEST SUMMARY" in text
    assert "Estimated break candidate:" in text
    assert "Reject null hypothesis of no break:" in text
    assert "Critical value" in text
    assert "alpha = 0.05" in text


def test_summary_does_not_duplicate_break_label(monkeypatch):
    X = pd.DataFrame(
        generate_panel_data(T=200, N=100),
        index=pd.date_range("2000-01-01", periods=200, freq="MS")
    )

    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model, r=2))

    model.estimate_breakpoint(classify=False)
    text = model.summary()

    label_mentions = text.count("Accepted breakpoint label:") + text.count("Estimated break label:")
    assert label_mentions <= 1
