"""Tests for single-break result objects and reporting."""

import numpy as np
import pytest

import fmbqml.single_break.model as qml_module
from fmbqml import SingleBreakQML, SingleBreakResults

from ._helpers import generate_panel_data, make_fake_factor_estimator, make_panel_data


def test_single_break_result_getter_returns_nested_copies():
    model = SingleBreakQML(
        np.arange(80, dtype=float).reshape(20, 4),
        max_factors=2,
    )
    model.is_tested = True
    model._last_result_type = "test"
    model._test_results = {
        "candidate_breaks": np.array([5, 6]),
        "classification_criteria": [{"condition": "original"}],
    }

    returned = SingleBreakResults(model._get_active_results()).to_dict()
    returned["candidate_breaks"][0] = 999
    returned["classification_criteria"][0]["condition"] = "changed"

    assert model._test_results["candidate_breaks"][0] == 5
    assert model._test_results["classification_criteria"][0]["condition"] == "original"


def test_single_break_result_print_is_compact_and_summary_is_detailed():
    result = SingleBreakResults({
        "test_type": "lr_test",
        "T": 200,
        "N": 100,
        "factor_criterion": 0,
        "estimated_factors": 4,
        "test_statistic": 120.1234,
        "critical_value": 10.5678,
        "significance_level": 0.05,
        "reject_null": True,
        "breakpoint": 100,
        "candidate_breakpoint": 100,
        "break_type": "Type 1 (Singular)",
    })

    printed = str(result)
    detailed = str(result.summary())

    assert "SingleBreakQML LR Test" in printed
    assert "LR statistic: 120.1234" in printed
    assert "Decision: Reject H0" in printed
    assert "Accepted breakpoint: 100" in printed
    assert "\n\nLR statistic:" in printed
    assert "\n\nAccepted breakpoint:" in printed
    assert "Breakpoint type: Type 1 (Singular)" in printed
    assert "Call:" not in printed
    assert "SingleBreakQML LR Test Results" in detailed
    assert "Candidate breakpoint under H1: 100" in detailed
    assert "Accepted breakpoint:" in detailed and "100" in detailed
    # The standard summary remains distinct from verbose diagnostics.
    assert "Call:" not in detailed
    assert str(result) == repr(result)


def test_single_break_estimation_print_uses_same_compact_layout():
    result = SingleBreakResults({
        "test_type": "estimate_breakpoint",
        "T": 200,
        "N": 100,
        "factor_criterion": 0,
        "estimated_factors": 4,
        "breakpoint": 100,
        "break_type": "Type 1 (Singular)",
    })

    printed = str(result)
    assert printed.startswith("SingleBreakQML Breakpoint Estimation\n")
    assert "Breakpoint type: Type 1 (Singular)" in printed
    assert str(result) == repr(result)


def test_single_break_result_print_distinguishes_unaccepted_candidate():
    result = SingleBreakResults({
        "test_type": "lr_test",
        "T": 100,
        "N": 20,
        "test_statistic": 5.0,
        "critical_value": 8.0,
        "significance_level": 0.05,
        "reject_null": False,
        "breakpoint": None,
        "candidate_breakpoint": 42,
    })

    printed = str(result)
    detailed = str(result.summary())
    assert "Accepted breakpoint: None" in printed
    assert "Breakpoint type:" not in printed
    assert "Candidate breakpoint under H1: 42" in detailed
    assert "Accepted breakpoint:" in detailed and "None" in detailed
    assert "Breakpoint type:" not in printed


def test_single_break_estimation_print_is_compact():
    result = SingleBreakResults({
        "test_type": "estimate_breakpoint",
        "T": 100,
        "N": 20,
        "factor_criterion": 1,
        "estimated_factors": 3,
        "breakpoint": 42,
        "break_type": "Type 2 (Rotational)",
    })

    printed = str(result)
    assert "SingleBreakQML Breakpoint Estimation" in printed
    assert "Breakpoint type: Type 2 (Rotational)" in printed
    assert "Model specification" not in printed


def test_summary_after_lr_test(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=0)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None, n_jobs=-1):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)
    text = str(SingleBreakResults(model._get_active_results()).summary())

    assert isinstance(text, str)
    assert "SingleBreakQML LR Test Results" in text
    assert "Breakpoint" in text
    assert "Decision:" in text
    assert "Critical value" in text
    assert "5%" in text


def test_break_type_property_is_none_without_classification():
    model = SingleBreakQML(generate_panel_data(T=40, N=8), max_factors=3)
    model.is_tested = True
    model._last_result_type = "test"
    model._test_results = {
        "test_type": "lr_test", "reject_null": True, "breakpoint": 15,
    }
    assert SingleBreakResults(model._get_active_results()).break_type is None


def test_single_result_snapshot_survives_model_update():
    model = SingleBreakQML(make_panel_data(), max_factors=3)
    result = model.estimate_breakpoint(classify=False)
    assert isinstance(result, SingleBreakResults)
    assert result.breakpoint is not None
    assert result.reject_null is None
    original = result.qml_values.copy()
    exported = result.to_dict()
    exported["qml_profile"][:] = 0
    np.testing.assert_array_equal(result.qml_values, original)
    model.set_parameters(trim_ratio=0.2)
    np.testing.assert_array_equal(result.qml_values, original)
    assert str(result.summary())


def test_single_break_to_dict_can_omit_diagnostic_arrays():
    result = SingleBreakResults({
        "test_type": "lr_test",
        "breakpoint": 10,
        "candidate_breakpoint": 10,
        "candidate_breaks": np.array([5, 6, 7, 8, 9, 10]),
        "lr_profile": np.arange(6, dtype=float),
        "qml_profile": np.arange(6, dtype=float) + 1,
        "test_statistic": 12.5,
    })

    full = result.to_dict()
    compact = result.to_dict(include_diagnostics=False)

    assert "candidate_breaks" in full
    assert "lr_profile" in full
    assert "qml_profile" in full
    assert "candidate_breaks" not in compact
    assert "lr_profile" not in compact
    assert "qml_profile" not in compact
    assert compact["breakpoint"] == 10
    assert compact["test_statistic"] == 12.5

    with pytest.raises(TypeError, match="include_diagnostics"):
        result.to_dict(include_diagnostics="false")
