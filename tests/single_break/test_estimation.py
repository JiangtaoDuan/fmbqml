"""Tests for single-break QML estimation."""

import numpy as np
import pandas as pd

import fmbqml.single_break.model as qml_module
from fmbqml import SingleBreakQML, SingleBreakResults
from fmbqml.single_break.qml_profile import compute_single_break_qml_profile
from fmbqml.utils.breakpoint_utils import calculate_break_range

from ._helpers import generate_panel_data, make_fake_factor_estimator


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
    monkeypatch.setattr(
        qml_module, "estimate_panel_factors_lr", fake_estimate_panel_factors_lr
    )

    r_hat, F_hat = model._calculate_factors()

    assert r_hat == 2
    assert F_hat.shape == (model.T, 2)
    assert model._L_hat.shape == (model.N, 2)
    assert model._VNT.shape == (1, 1)


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


def test_estimate_breakpoint_runs_and_returns_result(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    result = model.estimate_breakpoint(classify=False)
    assert isinstance(result, SingleBreakResults)
    result = result.to_dict()

    assert isinstance(result, dict)
    assert result["test_type"] == "estimate_breakpoint"
    assert "breakpoint" in result
    assert "candidate_breakpoint" in result
    assert "candidate_break_label" in result
    assert "estimated_factors" in result
    assert "factor_criterion" in result
    assert "search_range" in result
    assert "candidate_breaks" in result
    assert "qml_profile" in result
    assert result["factor_criterion"] == 1
    assert isinstance(result["breakpoint"], (int, np.integer))
    assert model.is_tested is False
    assert model._estimation_results["breakpoint"] == result["breakpoint"]


def test_estimate_breakpoint_summary_is_explicit(monkeypatch, capsys):
    model = SingleBreakQML(generate_panel_data(T=80, N=12), max_factors=3)
    monkeypatch.setattr(model, "_calculate_factors", make_fake_factor_estimator(model))
    monkeypatch.setattr(
        qml_module,
        "critical_value_lr_hac_m",
        lambda *a, **k: np.array([1.0, 2.0, 3.0]),
    )
    result = model.estimate_breakpoint(classify=False)
    assert capsys.readouterr().out == ""
    print(result.summary())
    assert "SingleBreakQML" in capsys.readouterr().out


def test_qml_profile_property_after_estimate_breakpoint(monkeypatch):
    X = generate_panel_data(T=200, N=100)
    model = SingleBreakQML(X, trim_ratio=0.1, factor_criterion=1)

    monkeypatch.setattr(
        model, "_calculate_factors", make_fake_factor_estimator(model, r=2)
    )

    model.estimate_breakpoint(classify=False)
    qml_profile = SingleBreakResults(model._get_active_results()).qml_profile

    assert isinstance(qml_profile, pd.DataFrame)
    assert list(qml_profile.columns) == ["candidate_break", "qml_objective"]
    assert len(qml_profile) == len(model._get_active_results()["candidate_breaks"])


def test_factor_estimates_are_reused_across_single_break_workflows(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = SingleBreakQML(X, trim_ratio=0.15, max_factors=3, factor_criterion="IC1")
    calls = {"count": 0, "estimate": 0}
    rng = np.random.default_rng(2026)
    F_hat = rng.normal(size=(model.T, 2))
    L_hat = rng.normal(size=(model.N, 2))
    VNT = np.eye(2)

    def fake_factor_count(X, max_factors):
        calls["count"] += 1
        return {"khat": {0: 2}}

    def fake_factor_estimation(X, r):
        calls["estimate"] += 1
        return F_hat.copy(), L_hat.copy(), VNT.copy()

    monkeypatch.setattr(qml_module, "bai_ng_factor_count", fake_factor_count)
    monkeypatch.setattr(qml_module, "estimate_panel_factors_lr", fake_factor_estimation)
    monkeypatch.setattr(
        qml_module,
        "critical_value_lr_hac_m",
        lambda *args, **kwargs: np.array([1.0, 2.0, 3.0]),
    )

    model.get_factor_estimates()
    model.estimate_breakpoint(classify=False)
    model.lr_test(alpha=0.05, classify=False, n_sim=5, random_state=7)

    assert calls == {"count": 1, "estimate": 1}

    model.get_factor_estimates(recalculate=True)
    assert calls == {"count": 2, "estimate": 2}


def test_factor_matrix_properties_return_independent_copies():
    model = SingleBreakQML(generate_panel_data(), max_factors=5)
    model._nbhat_r = 2
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
