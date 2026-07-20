import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

import fmbqml.multiple_break.multiple_break_qml as mqml_module
from fmbqml import MultiBreakQML
from fmbqml.utils.breakpoint_utils import (
    calculate_break_range,
    compute_min_segment_length,
    map_breakpoints_to_labels,
)


def test_multiple_break_result_getters_return_nested_copies():
    model = MultiBreakQML(
        np.arange(80, dtype=float).reshape(20, 4),
        max_break=2,
        max_factors=2,
    )
    model.joint_estimated = True
    model.joint_tested = True
    model.joint_results = {
        "breakpoints": [5, 10],
        "ic_path": [{"estimated_breakpoints": [5]}],
    }
    model.joint_test_results = {"joint_lr_profile": np.array([1.0, 2.0])}

    estimated = model.get_joint_results()
    tested = model.get_joint_test_results()
    estimated["breakpoints"][0] = 999
    estimated["ic_path"][0]["estimated_breakpoints"][0] = 999
    tested["joint_lr_profile"][0] = 999.0

    assert model.joint_results["breakpoints"] == [5, 10]
    assert model.joint_results["ic_path"][0]["estimated_breakpoints"] == [5]
    np.testing.assert_array_equal(
        model.joint_test_results["joint_lr_profile"],
        np.array([1.0, 2.0]),
    )

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
        "classification_full_sample_standardize": True,
        "classification_segment_standardize": True,
    }


def make_fake_classify_breakpoint_result(expected_breakpoints=None, expected_trim_ratio=None,
                                         expected_min_regime_length=None,
                                         expected_factor_criterion=None,
                                         T: int = 80):
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

        assert full_sample_factor_number is None
        assert full_sample_standardize is True
        assert segment_standardize is True

        classification = fake_classification_fields(T=T)
        classification["min_regime_length"] = min_regime_length

        updated_result = result_dict.copy()
        updated_result.update(classification)

        return updated_result, classification

    return fake_classify_breakpoint_result


# ----------------------------
# Initialization and input checks
# ----------------------------

def test_init_with_valid_numpy_array():
    X = generate_panel_data()
    model = MultiBreakQML(X, max_break=3, max_factors=10, factor_criterion=2)

    assert model.T == 300
    assert model.N == 20
    assert model.max_break == 3
    assert model.max_factors == 10
    assert model.factor_criterion == 2
    assert model.trim_ratio == 0.1
    assert model.joint_estimated is False
    assert model.joint_results == {}
    assert model._index is None
    assert model._is_dataframe is False
    assert map_breakpoints_to_labels([1, 10], model._index) == []


def test_init_with_valid_dataframe():
    index = pd.date_range("2000-01-31", periods=50, freq="MS")
    columns = [f"x{i}" for i in range(8)]
    X = pd.DataFrame(
        generate_panel_data(T=50, N=8),
        index=index,
        columns=columns,
    )

    model = MultiBreakQML(X, max_factors=5)

    assert model.T == X.shape[0]
    assert model.N == X.shape[1]
    assert model._is_dataframe is True

    returned = model.get_data()
    assert isinstance(returned, pd.DataFrame)
    assert returned.shape == X.shape

    assert map_breakpoints_to_labels([1, 10], model._index) == [
        index[0],
        index[9],
    ]


def test_init_with_invalid_max_break():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="max_break"):
        MultiBreakQML(X, max_break=-1)

    with pytest.raises(ValueError, match="max_break"):
        MultiBreakQML(X, max_break=True)


def test_init_with_invalid_max_factors():
    X = generate_panel_data(T=20, N=12)

    with pytest.raises(ValueError, match="max_factors"):
        MultiBreakQML(X, max_break=2, max_factors=0)

    with pytest.raises(ValueError, match="max_factors"):
        MultiBreakQML(X, max_break=2, max_factors=True)

    with pytest.raises(ValueError, match="max_factors"):
        MultiBreakQML(X, max_break=2, max_factors=21)


def test_init_with_invalid_factor_criterion():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="factor_criterion"):
        MultiBreakQML(X, factor_criterion=8)

    with pytest.raises(ValueError, match="factor_criterion"):
        MultiBreakQML(X, factor_criterion=True)

    with pytest.raises(ValueError, match="factor_criterion"):
        MultiBreakQML(X, factor_criterion="BAD")


def test_init_with_invalid_trim_ratio():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="trim_ratio"):
        MultiBreakQML(X, trim_ratio=0.8)

    with pytest.raises(ValueError, match="trim_ratio"):
        MultiBreakQML(X, trim_ratio=True)


def test_valid_factor_criterion_string_is_normalized():
    X = generate_panel_data()
    model = MultiBreakQML(X, factor_criterion="IC2")

    assert model.factor_criterion == 1

    model.set_parameters(factor_criterion="PC1")
    assert model.factor_criterion == 3


def test_init_with_non_2d_input():
    X = np.random.randn(100)

    with pytest.raises(ValueError, match="2D array"):
        MultiBreakQML(X)


def test_init_with_nan_input():
    X = generate_panel_data()
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite"):
        MultiBreakQML(X)


def test_init_with_inf_input():
    X = generate_panel_data()
    X[0, 0] = np.inf

    with pytest.raises(ValueError, match="NaN or infinite"):
        MultiBreakQML(X)


# ----------------------------
# Utility function checks
# ----------------------------

def test_compute_min_segment_length_function():
    assert compute_min_segment_length(100, 0.1) == 10
    assert compute_min_segment_length(200, 0.05) == 10
    assert compute_min_segment_length(101, 0.1) == 11


def test_compute_min_segment_length_invalid():
    with pytest.raises(ValueError, match="trim_ratio"):
        compute_min_segment_length(100, 0.0)

    with pytest.raises(ValueError, match="trim_ratio"):
        compute_min_segment_length(100, 0.6)

    with pytest.raises(ValueError, match="trim_ratio"):
        compute_min_segment_length(100, True)

    with pytest.raises(ValueError, match="trim_ratio"):
        compute_min_segment_length(100, "0.1")


def test_calculate_break_range_function():
    t_min, t_max = calculate_break_range(100, 0.1)

    assert t_min == 10
    assert t_max == 90


def test_validate_joint_parameters():
    X = generate_panel_data()
    model = MultiBreakQML(X, max_break=2, max_factors=3)

    with pytest.raises(ValueError, match="min_break"):
        model.estimate_breaks_jointly(min_break=-1)

    with pytest.raises(ValueError, match="min_break"):
        model.estimate_breaks_jointly(min_break=3)


def test_joint_methods_use_object_trim_ratio_by_default(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, trim_ratio=0.2)

    captured = {}

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        captured["joint_min_segment_length"] = min_segment_length
        return [20], 1, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    result = model.estimate_breaks_jointly(min_break=0, classify=False)

    assert result["trim_ratio"] == 0.2
    assert result["min_segment_length"] == 16
    assert captured["joint_min_segment_length"] == 16


def test_multibreak_public_joint_test_api(monkeypatch):
    X = np.random.default_rng(9).normal(size=(40, 8))
    model = MultiBreakQML(X, max_break=2, max_factors=3, trim_ratio=0.15)
    factors = normalized_factors(T=40, r=2)

    monkeypatch.setattr(
        model,
        "get_factor_estimates",
        lambda recalculate=False: {
            "r_hat": 2,
            "F_hat": factors,
            "L_hat": np.zeros((8, 2)),
            "VNT": np.eye(2),
            "factor_criterion": 0,
            "criterion_name": "IC1",
        },
    )
    fake_result = {
        "test_type": "joint_sup_lr",
        "n_breaks": 2,
        "breakpoints": [13, 27],
        "test_statistic": 20.0,
        "critical_value": 10.0,
        "critical_values": {0.10: 8.0, 0.05: 10.0, 0.01: 14.0},
        "reject_null": True,
        "significance_level": 0.05,
        "trim_ratio": 0.15,
        "min_segment_length": 6,
        "grid_size": 20,
        "monte_carlo_replications": 4,
        "random_state": 1,
        "null_objective": 0.0,
        "alternative_objective": -20.0,
        "simulated_statistics": np.arange(4.0),
    }
    monkeypatch.setattr(
        mqml_module,
        "_run_joint_sup_lr",
        lambda **kwargs: fake_result.copy(),
    )

    result = model.joint_sup_lr_test(
        n_breaks=2,
        n_sim=4,
        random_state=1,
        classify=False,
    )

    assert result["reject_null"] is True
    assert result["breakpoints"] == [13, 27]
    assert result["estimated_factors"] == 2
    assert model.get_joint_test_results()["test_statistic"] == 20.0


def test_prespecified_lr_uses_object_trim_ratio_by_default(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, trim_ratio=0.2)

    factors = {
        "r_hat": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(model, "get_factor_estimates", lambda recalculate=False: factors)

    def fake_run_joint_sup_lr(**kwargs):
        assert kwargs["trim_ratio"] == 0.2
        assert kwargs["alpha"] == 0.01
        assert kwargs["n_sim"] == 5000
        return {
            "test_type": "joint_sup_lr",
            "n_breaks": kwargs["n_breaks"],
            "breakpoints": [20],
            "test_statistic": 2.0,
            "critical_value": 1.0,
            "critical_values": {0.10: 0.8, 0.05: 1.0, 0.01: 1.2},
            "reject_null": True,
            "significance_level": kwargs["alpha"],
            "trim_ratio": kwargs["trim_ratio"],
            "min_segment_length": 16,
            "grid_size": 200,
            "monte_carlo_replications": kwargs["n_sim"],
            "random_state": kwargs["random_state"],
            "null_objective": 0.0,
            "alternative_objective": -2.0,
            "simulated_statistics": np.arange(3.0),
        }

    monkeypatch.setattr(mqml_module, "_run_joint_sup_lr", fake_run_joint_sup_lr)

    result = model.joint_sup_lr_test(n_breaks=1, classify=False)

    assert result["trim_ratio"] == 0.2


def test_prespecified_lr_classifies_breakpoints_by_default(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4, factor_criterion=2)

    factors = {
        "r_hat": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(model, "get_factor_estimates", lambda recalculate=False: factors)

    def fake_run_joint_sup_lr(**kwargs):
        return {
            "test_type": "joint_sup_lr",
            "n_breaks": kwargs["n_breaks"],
            "breakpoints": [20, 50],
            "test_statistic": 2.0,
            "critical_value": 1.0,
            "critical_values": {0.10: 0.8, 0.05: 1.0, 0.01: 1.2},
            "reject_null": True,
            "significance_level": kwargs["alpha"],
            "trim_ratio": kwargs["trim_ratio"],
            "min_segment_length": 8,
            "grid_size": 200,
            "monte_carlo_replications": kwargs["n_sim"],
            "random_state": kwargs["random_state"],
            "null_objective": 0.0,
            "alternative_objective": -2.0,
            "simulated_statistics": np.arange(3.0),
        }

    monkeypatch.setattr(mqml_module, "_run_joint_sup_lr", fake_run_joint_sup_lr)
    monkeypatch.setattr(
        mqml_module,
        "classify_breakpoint_result",
        make_fake_classify_breakpoint_result(
            expected_breakpoints=[20, 50],
            expected_trim_ratio=0.1,
            expected_min_regime_length=None,
            expected_factor_criterion=2,
            T=80,
        ),
    )

    result = model.joint_sup_lr_test(n_breaks=2)

    assert result["break_types"] == ["Type 1 (Singular)", "Type 2 (Rotational)"]
    assert result["break_type_summary"]["type1_count"] == 1
    assert result["break_type_summary"]["type2_count"] == 1


def test_prespecified_lr_can_skip_classification(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4)

    factors = {
        "r_hat": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(model, "get_factor_estimates", lambda recalculate=False: factors)

    def fake_run_joint_sup_lr(**kwargs):
        return {
            "test_type": "joint_sup_lr",
            "n_breaks": kwargs["n_breaks"],
            "breakpoints": [20],
            "test_statistic": 2.0,
            "critical_value": 1.0,
            "critical_values": {0.10: 0.8, 0.05: 1.0, 0.01: 1.2},
            "reject_null": True,
            "significance_level": kwargs["alpha"],
            "trim_ratio": kwargs["trim_ratio"],
            "min_segment_length": 8,
            "grid_size": 200,
            "monte_carlo_replications": kwargs["n_sim"],
            "random_state": kwargs["random_state"],
            "null_objective": 0.0,
            "alternative_objective": -2.0,
            "simulated_statistics": np.arange(3.0),
        }

    def fail_classification(*args, **kwargs):
        raise AssertionError("classification should not run")

    monkeypatch.setattr(mqml_module, "_run_joint_sup_lr", fake_run_joint_sup_lr)
    monkeypatch.setattr(mqml_module, "classify_breakpoint_result", fail_classification)

    result = model.joint_sup_lr_test(n_breaks=1, classify=False)

    assert "break_types" not in result


def test_prespecified_lr_verbose_prints_summary(monkeypatch, capsys):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4, factor_criterion=1)

    factors = {
        "r_hat": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(model, "get_factor_estimates", lambda recalculate=False: factors)

    def fake_run_joint_sup_lr(**kwargs):
        return {
            "test_type": "joint_sup_lr",
            "n_breaks": kwargs["n_breaks"],
            "breakpoints": [20],
            "test_statistic": 2.0,
            "critical_value": 1.0,
            "critical_values": {0.10: 0.8, 0.05: 1.0, 0.01: 1.2},
            "reject_null": True,
            "significance_level": kwargs["alpha"],
            "trim_ratio": kwargs["trim_ratio"],
            "min_segment_length": 8,
            "grid_size": 200,
            "monte_carlo_replications": kwargs["n_sim"],
            "random_state": kwargs["random_state"],
            "null_objective": 0.0,
            "alternative_objective": -2.0,
            "simulated_statistics": np.arange(3.0),
        }

    monkeypatch.setattr(mqml_module, "_run_joint_sup_lr", fake_run_joint_sup_lr)

    model.joint_sup_lr_test(
        n_breaks=1,
        classify=False,
        verbose=True,
        random_state=123,
    )

    captured = capsys.readouterr()
    assert "MultiBreakQML JOINT SUP-LR TEST SUMMARY" in captured.out
    assert "Reject null hypothesis of no break:" in captured.out
    assert "Detected breakpoints:" in captured.out


def test_joint_estimate_min_break_infeasible_raises():
    X = generate_panel_data(T=20, N=12)
    model = MultiBreakQML(X, max_break=10, max_factors=3, trim_ratio=0.2)

    with pytest.raises(ValueError, match="infeasible"):
        model.estimate_breaks_jointly(
            min_break=5,
            classify=False,
        )


def test_joint_estimate_warns_when_max_break_too_large(monkeypatch):
    X = generate_panel_data(T=20, N=12)
    model = MultiBreakQML(X, max_break=15, max_factors=3)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [], 0, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    with pytest.warns(RuntimeWarning, match="max_break"):
        result = model.estimate_breaks_jointly(
            min_break=0,
            classify=False,
        )

    assert result["breakpoints"] == []
    assert result["n_breaks"] == 0


# ----------------------------
# Accessors before running procedures
# ----------------------------

def test_get_joint_results_before_run():
    X = generate_panel_data()
    model = MultiBreakQML(X)

    with pytest.raises(ValueError, match="estimate_breaks_jointly"):
        model.get_joint_results()


def test_summary_before_run():
    X = generate_panel_data()
    model = MultiBreakQML(X)

    with pytest.raises(ValueError, match="No joint estimation"):
        model.summary(mode="joint")

    with pytest.raises(ValueError, match="mode"):
        model.summary(mode="bad")


def test_classification_before_estimation_raises():
    X = generate_panel_data()
    model = MultiBreakQML(X)

    with pytest.raises(ValueError, match="estimate_breaks_jointly"):
        model.classify_joint_breaks()

# ----------------------------
# get_data, set_data, and set_parameters
# ----------------------------

def test_get_data_returns_copy_for_numpy():
    X = generate_panel_data()
    model = MultiBreakQML(X)

    returned = model.get_data()
    assert isinstance(returned, np.ndarray)

    returned[0, 0] = 999.0
    assert model.X[0, 0] != 999.0


def test_get_data_returns_copy_for_dataframe():
    X = pd.DataFrame(generate_panel_data())
    model = MultiBreakQML(X)

    returned = model.get_data()
    assert isinstance(returned, pd.DataFrame)

    returned.iloc[0, 0] = 999.0
    assert model.get_data().iloc[0, 0] != 999.0


def test_set_data_updates_shape_and_resets_results():
    X = generate_panel_data(T=100, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4)

    model.joint_estimated = True
    model.joint_results = {"breakpoints": [20]}
    model._r_hat = 2
    model._F_hat = np.ones((100, 2))
    model._L_hat = np.ones((12, 2))
    model._VNT = np.eye(2)

    new_X = generate_panel_data(T=80, N=12, seed=456)
    model.set_data(new_X)

    assert model.T == 80
    assert model.N == 12
    assert model.joint_estimated is False
    assert model.joint_results == {}
    assert model._r_hat is None
    assert model._F_hat is None
    assert model._L_hat is None
    assert model._VNT is None


def test_set_data_dataframe_updates_index():
    X = generate_panel_data(T=100, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4)

    index = pd.date_range("2010-01-31", periods=80, freq="MS")
    new_X = pd.DataFrame(
        generate_panel_data(T=80, N=12, seed=456),
        index=index,
    )

    model.set_data(new_X)

    assert model._is_dataframe is True
    assert map_breakpoints_to_labels([1, 10], model._index) == [
        index[0],
        index[9],
    ]


def test_set_data_invalid_reverts_state():
    X = generate_panel_data(T=100, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4)

    old_T, old_N = model.T, model.N
    old_X = model.X.copy()
    bad_X = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        model.set_data(bad_X)

    assert model.T == old_T
    assert model.N == old_N
    np.testing.assert_allclose(model.X, old_X)


def test_set_data_revalidates_parameters():
    X = generate_panel_data(T=100, N=50)
    model = MultiBreakQML(X, max_break=2, max_factors=40)
    new_X = generate_panel_data(T=20, N=10, seed=456)

    with pytest.raises(ValueError, match="max_factors"):
        model.set_data(new_X)

    assert model.T == 100
    assert model.N == 50


def test_set_parameters_updates_values_and_resets_results():
    X = generate_panel_data()
    model = MultiBreakQML(X, max_break=2, max_factors=5, factor_criterion=0)

    model.joint_estimated = True
    model.joint_results = {"breakpoints": [20]}

    model.set_parameters(
        max_break=4,
        max_factors=6,
        factor_criterion="PC1",
    )

    assert model.max_break == 4
    assert model.max_factors == 6
    assert model.factor_criterion == 3
    assert model.trim_ratio == 0.1
    assert model.joint_estimated is False
    assert model.joint_results == {}


def test_set_parameters_invalid_reverts_state():
    X = generate_panel_data()
    model = MultiBreakQML(X, max_break=2, max_factors=5, factor_criterion=0)

    old_max_break = model.max_break
    old_max_factors = model.max_factors
    old_factor_criterion = model.factor_criterion
    with pytest.raises(ValueError):
        model.set_parameters(max_break=-1)

    assert model.max_break == old_max_break
    assert model.max_factors == old_max_factors
    assert model.factor_criterion == old_factor_criterion


# ----------------------------
# Factor estimates
# ----------------------------

def test_get_factor_estimates_computes_and_returns_dict(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=6, factor_criterion=1)

    def fake_bai_ng_factor_count(X, max_factors):
        return {"khat": [2, 3, 2, 2, 2, 2, 2, 2]}

    monkeypatch.setattr(mqml_module, "bai_ng_factor_count", fake_bai_ng_factor_count)
    monkeypatch.setattr(mqml_module, "estimate_panel_factors", fake_estimate_panel_factors)

    result = model.get_factor_estimates()

    assert isinstance(result, dict)
    assert result["r_hat"] == 3
    assert result["F_hat"].shape == (80, 3)
    assert result["L_hat"].shape == (12, 3)
    assert result["VNT"].shape == (3, 3)
    assert result["factor_criterion"] == 1
    assert result["criterion_name"] == "IC2"


def test_get_factor_estimates_zero_factor_raises(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=6, factor_criterion=1)

    def fake_bai_ng_factor_count(X, max_factors):
        return {"khat": [1, 0, 1, 1, 1, 1, 1, 1]}

    monkeypatch.setattr(mqml_module, "bai_ng_factor_count", fake_bai_ng_factor_count)

    with pytest.raises(ValueError, match="at least one factor"):
        model.get_factor_estimates()


def test_get_factor_estimates_uses_cache_when_recalculate_false(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=6, factor_criterion=1)

    model._r_hat = 2
    model._F_hat = np.ones((80, 2))
    model._L_hat = np.ones((12, 2))
    model._VNT = np.eye(2)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("factor estimation should not be called")

    monkeypatch.setattr(mqml_module, "estimate_panel_factors", fail_if_called)

    result = model.get_factor_estimates(recalculate=False)

    assert result["r_hat"] == 2
    assert result["F_hat"].shape == (80, 2)


def test_get_factor_estimates_recalculate_true(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=6, factor_criterion=1)

    model._r_hat = 2
    model._F_hat = np.ones((80, 2))
    model._L_hat = np.ones((12, 2))
    model._VNT = np.eye(2)

    def fake_bai_ng_factor_count(X, max_factors):
        return {"khat": [1, 3, 1, 1, 1, 1, 1, 1]}

    monkeypatch.setattr(mqml_module, "bai_ng_factor_count", fake_bai_ng_factor_count)
    monkeypatch.setattr(mqml_module, "estimate_panel_factors", fake_estimate_panel_factors)

    result = model.get_factor_estimates(recalculate=True)

    assert result["r_hat"] == 3
    assert result["F_hat"].shape == (80, 3)


# ----------------------------
# Joint estimation
# ----------------------------

def test_joint_estimate_runs_and_returns_dict(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [15, 40], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    result = model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
    )

    assert isinstance(result, dict)
    assert result["test_type"] == "mqml_joint"
    assert result["breakpoints"] == [15, 40]
    assert "break_labels" not in result
    assert result["n_breaks"] == 2
    assert result["factor_criterion"] == 1
    assert result["ic_path"] == make_fake_ic_path()
    assert model.joint_estimated is True


def test_joint_estimate_default_does_not_print(monkeypatch, capsys):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20], 1, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)
    model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_joint_estimate_verbose_prints_summary(monkeypatch, capsys):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20], 1, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "MultiBreakQML JOINT ESTIMATION SUMMARY" in captured.out
    assert "Estimated number of factors:" in captured.out
    assert "Break types:" in captured.out
    assert "trim_ratio=" in captured.out
    assert "epsilon=" not in captured.out


def test_joint_estimate_dataframe_break_labels(monkeypatch):
    index = pd.date_range("2000-01-31", periods=80, freq="MS")
    X = pd.DataFrame(generate_panel_data(T=80, N=12), index=index)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20, 50], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    result = model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
    )

    assert result["break_labels"] == [index[19], index[49]]


def test_joint_estimate_no_breakpoint_string(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return np.array(["no breakpoint"], dtype=object), 0, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    result = model.estimate_breaks_jointly(
        min_break=0,
        classify=True,
    )

    assert result["breakpoints"] == []
    assert "break_labels" not in result
    assert result["n_breaks"] == 0
    assert "break_types" not in result


def test_get_joint_results_and_ic_path_after_joint_estimation(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20, 50], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
    )

    result = model.get_joint_results()
    ic_table = model.format_ic_path_table()

    assert result["breakpoints"] == [20, 50]
    assert "IC" in ic_table
    assert "m=0:" in ic_table
    assert "U_NT=" in ic_table


def test_format_ic_path_table_before_joint_estimation():
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X)

    with pytest.raises(ValueError, match="estimate_breaks_jointly"):
        model.format_ic_path_table()


# ----------------------------
# MQML profile plotting
# ----------------------------

def test_plot_mqml_profile_before_joint_estimation():
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X)

    with pytest.raises(ValueError, match="estimate_breaks_jointly"):
        model.plot_mqml_profile(show=False)


def test_plot_mqml_profile_after_joint_estimation(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")

    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20, 50], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
    )

    fig, ax = model.plot_mqml_profile(show=False)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Number of breaks"
    assert "MQML" in ax.get_title()

    plt.close(fig)


def test_plot_mqml_profile_save_path(monkeypatch, tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")

    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3, factor_criterion=1)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20, 50], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)

    model.estimate_breaks_jointly(
        min_break=0,
        classify=False,
    )

    save_path = tmp_path / "mqml_profiles.png"
    fig, ax = model.plot_mqml_profile(save_path=str(save_path), show=False)

    assert save_path.exists()
    assert save_path.stat().st_size > 0

    plt.close(fig)


def test_plot_mqml_profile_empty_ic_path_raises():
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2)

    model.joint_estimated = True
    model.joint_results = {
        "test_type": "mqml_joint",
        "breakpoints": [],
        "n_breaks": 0,
        "max_break": 2,
        "min_break": 0,
        "factor_criterion": 0,
        "max_factors": 10,
        "trim_ratio": 0.1,
        "min_segment_length": 8,
        "T": 80,
        "N": 12,
        "ic_path": [],
    }

    with pytest.raises(ValueError, match="empty"):
        model.plot_mqml_profile(show=False)


def test_plot_joint_lr_profile_after_joint_lr_test(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")

    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3)

    factors = {
        "r_hat": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(model, "get_factor_estimates", lambda recalculate=False: factors)

    def fake_run_joint_sup_lr(**kwargs):
        n_breaks = kwargs["n_breaks"]
        return {
            "test_type": "joint_sup_lr",
            "n_breaks": n_breaks,
            "breakpoints": [20] * n_breaks,
            "test_statistic": float(n_breaks),
            "critical_value": float(n_breaks + 1),
            "critical_values": {0.10: 0.8, 0.05: 1.0, 0.01: 1.2},
            "reject_null": False,
            "significance_level": kwargs["alpha"],
            "trim_ratio": kwargs["trim_ratio"],
            "min_segment_length": 8,
            "grid_size": 200,
            "monte_carlo_replications": kwargs["n_sim"],
            "random_state": kwargs["random_state"],
            "null_objective": 0.0,
            "alternative_objective": -float(n_breaks),
            "simulated_statistics": np.arange(3.0),
        }

    monkeypatch.setattr(mqml_module, "_run_joint_sup_lr", fake_run_joint_sup_lr)

    model.joint_sup_lr_test(n_breaks=1, classify=False)
    fig, ax = model.plot_joint_lr_profile(n_sim=3, show=False)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Number of breaks"
    assert "joint_lr_profile" in model.get_joint_test_results()

    plt.close(fig)


def test_plot_joint_lr_profile_save_path(monkeypatch, tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")

    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3)

    factors = {
        "r_hat": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(model, "get_factor_estimates", lambda recalculate=False: factors)

    def fake_run_joint_sup_lr(**kwargs):
        n_breaks = kwargs["n_breaks"]
        return {
            "test_type": "joint_sup_lr",
            "n_breaks": n_breaks,
            "breakpoints": [20] * n_breaks,
            "test_statistic": float(n_breaks),
            "critical_value": float(n_breaks + 1),
            "critical_values": {0.10: 0.8, 0.05: 1.0, 0.01: 1.2},
            "reject_null": False,
            "significance_level": kwargs["alpha"],
            "trim_ratio": kwargs["trim_ratio"],
            "min_segment_length": 8,
            "grid_size": 200,
            "monte_carlo_replications": kwargs["n_sim"],
            "random_state": kwargs["random_state"],
            "null_objective": 0.0,
            "alternative_objective": -float(n_breaks),
            "simulated_statistics": np.arange(3.0),
        }

    monkeypatch.setattr(mqml_module, "_run_joint_sup_lr", fake_run_joint_sup_lr)

    save_path = tmp_path / "joint_lr_profile.png"
    fig, ax = model.plot_joint_lr_profile(
        n_sim=3,
        save_path=str(save_path),
        show=False,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0

    plt.close(fig)


# ----------------------------
# Joint classification
# ----------------------------

def test_joint_estimate_with_classification(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4, factor_criterion=2)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20, 50], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)
    monkeypatch.setattr(
        mqml_module,
        "classify_breakpoint_result",
        make_fake_classify_breakpoint_result(
            expected_breakpoints=[20, 50],
            expected_trim_ratio=0.1,
            expected_min_regime_length=None,
            expected_factor_criterion=2,
            T=80,
        ),
    )

    result = model.estimate_breaks_jointly(
        min_break=0,
        classify=True,
    )

    assert result["breakpoints"] == [20, 50]
    assert result["break_types"] == ["Type 1 (Singular)", "Type 2 (Rotational)"]
    assert result["break_type_summary"]["type1_count"] == 1
    assert result["break_type_summary"]["type2_count"] == 1
    assert result["break_type_summary"]["other_count"] == 0


def test_classify_joint_breaks_returns_dict_after_joint_estimation(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4, factor_criterion=1)

    model.joint_estimated = True
    model.joint_results = {
        "test_type": "mqml_joint",
        "breakpoints": [20, 50],
        "break_labels": [20, 50],
        "n_breaks": 2,
        "max_break": 2,
        "min_break": 0,
        "factor_criterion": 1,
        "max_factors": 4,
        "trim_ratio": 0.1,
        "min_segment_length": 8,
        "T": 80,
        "N": 12,
        "ic_path": make_fake_ic_path(),
    }

    monkeypatch.setattr(
        mqml_module,
        "classify_breakpoint_result",
        make_fake_classify_breakpoint_result(
            expected_breakpoints=[20, 50],
            expected_factor_criterion=1,
            T=80,
        ),
    )

    result = model.classify_joint_breaks(verbose=False)

    assert isinstance(result, dict)
    assert result["breakpoints"] == [20, 50]
    assert result["break_types"] == ["Type 1 (Singular)", "Type 2 (Rotational)"]
    assert "break_type_summary" in result
    assert model.joint_results["break_type_summary"]["type1_count"] == 1


def test_classify_joint_breaks_with_no_breakpoints_returns_existing_results():
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2)

    model.joint_estimated = True
    model.joint_results = {
        "test_type": "mqml_joint",
        "breakpoints": [],
        "n_breaks": 0,
        "max_break": 2,
        "min_break": 0,
        "factor_criterion": 0,
        "max_factors": 10,
        "trim_ratio": 0.1,
        "min_segment_length": 8,
        "T": 80,
        "N": 12,
        "ic_path": make_fake_ic_path(),
    }

    result = model.classify_joint_breaks(verbose=True)

    assert result["breakpoints"] == []
    assert result["break_types"] == []


# ----------------------------
# Summaries
# ----------------------------

def test_summary_after_joint_estimation(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4, factor_criterion=5)

    def fake_joint_estimation(
        X,
        min_segment_length,
        max_factors,
        max_break,
        min_break,
        factor_criterion,
    ):
        return [20, 50], 2, make_fake_ic_path()

    monkeypatch.setattr(mqml_module, "joint_estimation", fake_joint_estimation)
    monkeypatch.setattr(
        mqml_module,
        "classify_breakpoint_result",
        make_fake_classify_breakpoint_result(
            expected_breakpoints=[20, 50],
            expected_trim_ratio=0.1,
            expected_factor_criterion=5,
            T=80,
        ),
    )

    model.estimate_breaks_jointly(
        min_break=0,
        classify=True,
    )

    text = model.summary(mode="joint")

    assert isinstance(text, str)
    assert "MultiBreakQML JOINT ESTIMATION SUMMARY" in text
    assert "Detected breakpoints:" in text
    assert "Break types:" in text
    assert "epsilon=" not in text
    assert "Factor criterion:" in text
    assert "JOINT BREAKPOINT TYPE CLASSIFICATION" in text
    assert "Information criterion path:" in text
