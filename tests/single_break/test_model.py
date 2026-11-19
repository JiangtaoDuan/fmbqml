"""Tests for single_break test_model."""

import numpy as np

import pandas as pd

import pytest

from ._helpers import generate_panel_data, make_panel_data

from fmbqml import SingleBreakQML

from fmbqml.utils.breakpoint_utils import map_breakpoint_to_label

def test_single_model_init_with_valid_numpy_array():
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
    assert model._test_results == {}


def test_single_model_init_with_valid_dataframe():
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


def test_single_model_init_with_nan_input():
    X = generate_panel_data()
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN|Inf|infinite"):
        SingleBreakQML(X)


def test_single_model_init_with_inf_input():
    X = generate_panel_data()
    X[0, 0] = np.inf

    with pytest.raises(ValueError, match="NaN|Inf|infinite"):
        SingleBreakQML(X)


def test_invalid_trim_ratio():
    X = generate_panel_data()

    with pytest.raises(ValueError, match="trim_ratio"):
        SingleBreakQML(X, trim_ratio=0.0)

    with pytest.raises(ValueError, match="trim_ratio"):
        SingleBreakQML(X, trim_ratio=0.6)

    with pytest.raises(ValueError, match="trim_ratio"):
        SingleBreakQML(X, trim_ratio=True)


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


def test_single_model_get_data_returns_copy_for_numpy():
    X = generate_panel_data()
    model = SingleBreakQML(X)

    returned = model.get_data()
    assert isinstance(returned, np.ndarray)
    assert returned.shape == X.shape

    returned[0, 0] = 999.0
    assert model.X[0, 0] != 999.0


def test_set_data_updates_shape_and_resets_state():
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X)

    new_X = generate_panel_data(T=100, N=100, seed=456)
    model.set_data(new_X)

    assert model.T == 100
    assert model.N == 100
    assert model.is_tested is False
    assert model._test_results == {}
    assert model._F_hat is None
    assert model._L_hat is None
    assert model._VNT is None
    assert model._nbhat_r is None


def test_single_model_set_data_invalid_reverts_state():
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
    assert model._test_results == {}


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


def test_configuration_requires_setter():
    model = SingleBreakQML(make_panel_data(), max_factors=3)
    with pytest.raises(AttributeError):
        model.trim_ratio = 0.2


def test_single_break_public_boolean_options_are_validated():
    model = SingleBreakQML(generate_panel_data(T=80, N=12), max_factors=3)

    with pytest.raises(TypeError, match="recalculate"):
        model.get_factor_estimates(recalculate="false")

    with pytest.raises(TypeError, match="verbose"):
        model.classify_break(verbose="false")

def test_classify_break_requires_stored_estimated_breakpoint():
    model = SingleBreakQML(
        np.arange(80, dtype=float).reshape(20, 4),
        max_factors=2,
    )
    model._last_result_type = "test"
    model._test_results = {"breakpoint": None, "candidate_breakpoint": 7}

    with pytest.raises(ValueError, match="No estimated breakpoint"):
        model.classify_break()


def test_classify_break_does_not_accept_user_supplied_location():
    import inspect

    parameters = inspect.signature(SingleBreakQML.classify_break).parameters
    assert "breakpoint" not in parameters
