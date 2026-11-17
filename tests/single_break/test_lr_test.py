"""Tests for single-break likelihood-ratio testing."""

import numpy as np
import pandas as pd

from ._helpers import generate_panel_data, make_fake_factor_estimator
from fmbqml import SingleBreakQML, SingleBreakResults
import fmbqml.single_break.model as qml_module


def test_lr_test_runs_and_returns_result(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    result = model.lr_test(alpha=0.05, classify=False)
    assert isinstance(result, SingleBreakResults)
    result = result.to_dict()

    assert isinstance(result, dict)
    assert result["test_type"] == "lr_test"
    assert "breakpoint" in result
    assert "candidate_breakpoint" in result
    assert "candidate_break_label" in result
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


def test_lr_test_rejects_and_classifies(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=0)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([-1e9, -1e9, -1e9])

    def fake_classify_breakpoint_result(**kwargs):
        updated = kwargs["result_dict"].copy()
        updated["break_type"] = "Type 2: rotational"
        updated["break_types"] = ["Type 2: rotational"]
        return updated, {}

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)
    monkeypatch.setattr(
        qml_module, "classify_breakpoint_result", fake_classify_breakpoint_result
    )

    result = model.lr_test(alpha=0.05, classify=True)
    assert isinstance(result, SingleBreakResults)
    result = result.to_dict()

    assert bool(result["reject_null"]) is True
    assert result["breakpoint"] is not None
    assert result["candidate_breakpoint"] == result["breakpoint"]
    assert model._test_results["break_type"] == "Type 2: rotational"


def test_lr_test_non_rejection_keeps_candidate(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=0)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1e9, 1e9, 1e9])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    result = model.lr_test(alpha=0.05, classify=True)
    assert isinstance(result, SingleBreakResults)
    result = result.to_dict()

    assert bool(result["reject_null"]) is False
    assert result["breakpoint"] is None
    assert result["candidate_breakpoint"] is not None


def test_get_lr_profile_after_lr_test(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=4)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    def fake_cv(F_hat, trim_ratio, n_sim=5000, random_state=None):
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(qml_module, "critical_value_lr_hac_m", fake_cv)

    model.lr_test(alpha=0.05, classify=False)
    lr_profile = SingleBreakResults(model._get_active_results()).lr_profile

    assert isinstance(lr_profile, pd.DataFrame)
    assert list(lr_profile.columns) == ["candidate_break", "lr_statistic"]
    assert len(lr_profile) == len(model._get_active_results()["candidate_breaks"])
