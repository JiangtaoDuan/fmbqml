"""Tests for the MultiBreakQML model API."""

import numpy as np
import pandas as pd
import pytest

import fmbqml.multiple_break.model as mqml_module
from fmbqml import MultiBreakQML
from fmbqml.utils.breakpoint_utils import map_breakpoints_to_labels

from ._helpers import fake_estimate_panel_factors, generate_panel_data


def test_multiple_model_init_with_valid_numpy_array():
    X = generate_panel_data()
    model = MultiBreakQML(X, max_break=3, max_factors=10, factor_criterion=2)

    assert model.T == 300
    assert model.N == 20
    assert model.max_break == 3
    assert model.max_factors == 10
    assert model.factor_criterion == 2
    assert model.trim_ratio == 0.1
    assert model.joint_estimated is False
    assert model._joint_results == {}
    assert model._index is None
    assert model._is_dataframe is False
    assert map_breakpoints_to_labels([1, 10], model._index) == []


def test_multiple_model_init_with_valid_dataframe():
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


def test_init_with_non_2d_input():
    X = np.random.randn(100)

    with pytest.raises(ValueError, match="2D array"):
        MultiBreakQML(X)


def test_multiple_model_init_with_nan_input():
    X = generate_panel_data()
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        MultiBreakQML(X)


def test_multiple_model_init_with_inf_input():
    X = generate_panel_data()
    X[0, 0] = np.inf

    with pytest.raises(ValueError, match="NaN or Inf"):
        MultiBreakQML(X)


def test_multiple_model_get_data_returns_copy_for_numpy():
    X = generate_panel_data()
    model = MultiBreakQML(X)

    returned = model.get_data()
    assert isinstance(returned, np.ndarray)

    returned[0, 0] = 999.0
    assert model.X[0, 0] != 999.0


def test_set_data_updates_shape_and_resets_results():
    X = generate_panel_data(T=100, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4)

    model.joint_estimated = True
    model._joint_results = {"breakpoints": [20]}
    model._r_hat = 2
    model._F_hat = np.ones((100, 2))
    model._L_hat = np.ones((12, 2))
    model._VNT = np.eye(2)

    new_X = generate_panel_data(T=80, N=12, seed=456)
    model.set_data(new_X)

    assert model.T == 80
    assert model.N == 12
    assert model.joint_estimated is False
    assert model._joint_results == {}
    assert model._r_hat is None
    assert model._F_hat is None
    assert model._L_hat is None
    assert model._VNT is None


def test_multiple_model_set_data_invalid_reverts_state():
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


def test_set_parameters_updates_values_and_resets_results():
    X = generate_panel_data()
    model = MultiBreakQML(X, max_break=2, max_factors=5, factor_criterion=0)

    model.joint_estimated = True
    model._joint_results = {"breakpoints": [20]}

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
    assert model._joint_results == {}


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


def test_get_factor_estimates_computes_and_returns_dict(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=6, factor_criterion=1)

    def fake_bai_ng_factor_count(X, max_factors):
        return {"khat": [2, 3, 2, 2, 2, 2, 2, 2]}

    monkeypatch.setattr(mqml_module, "bai_ng_factor_count", fake_bai_ng_factor_count)
    monkeypatch.setattr(
        mqml_module, "estimate_panel_factors", fake_estimate_panel_factors
    )

    result = model.get_factor_estimates()

    assert isinstance(result, dict)
    assert result["estimated_factors"] == 3
    assert result["F_hat"].shape == (80, 3)
    assert result["L_hat"].shape == (12, 3)
    assert result["VNT"].shape == (3, 3)
    assert result["factor_criterion"] == 1


def test_multiple_break_public_boolean_options_are_validated():
    model = MultiBreakQML(
        generate_panel_data(T=80, N=12),
        max_break=2,
        max_factors=3,
    )

    with pytest.raises(TypeError, match="recalculate"):
        model.get_factor_estimates(recalculate="false")

    with pytest.raises(TypeError, match="verbose"):
        model.classify_joint_breaks(verbose="false")


def test_configuration_direct_assignment_and_get_parameters():
    model = MultiBreakQML(generate_panel_data(), max_break=2, max_factors=5)
    initial = model.get_parameters()
    assert set(initial) == {"max_break", "trim_ratio", "max_factors", "factor_criterion"}
    initial["max_break"] = 123
    assert model.max_break == 2
    model._F_hat = np.ones((model.T, 1))
    model._joint_results = {"breakpoints": [20]}
    model.joint_estimated = True
    model.max_break = 3
    assert model.max_break == 3
    assert model._F_hat is None and model._joint_results == {}
    assert not model.joint_estimated
    model.factor_criterion = "PC1"
    assert model.get_parameters()["factor_criterion"] == 3
    model._joint_results = {"breakpoints": [20]}
    with pytest.raises(ValueError):
        model.max_break = -1
    assert model.max_break == 3 and model._joint_results == {"breakpoints": [20]}
    with pytest.raises(AttributeError):
        model.N = 10
    with pytest.raises(AttributeError):
        model.X = np.zeros_like(model.X)


def test_factor_matrix_properties_return_independent_copies():
    model = MultiBreakQML(
        generate_panel_data(T=80, N=12), max_break=2, max_factors=5
    )
    model._r_hat = 2
    model._F_hat = np.arange(model.T * 2, dtype=float).reshape(model.T, 2)
    model._L_hat = np.arange(model.N * 2, dtype=float).reshape(model.N, 2)
    model._VNT = np.eye(2)

    factors = model.factors_
    loadings = model.factor_loadings_

    assert factors.shape == (model.T, 2)
    assert loadings.shape == (model.N, 2)
    factors[0, 0] = -999.0
    loadings[0, 0] = -999.0
    assert model._F_hat[0, 0] != -999.0
    assert model._L_hat[0, 0] != -999.0
