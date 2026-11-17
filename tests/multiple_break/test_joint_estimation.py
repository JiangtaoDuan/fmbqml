"""Tests for multiple_break test_joint_estimation."""

import numpy as np

import pandas as pd

import pytest

from ._helpers import (
    generate_panel_data,
    make_fake_classify_breakpoint_result,
    make_fake_ic_path,
)

from fmbqml import MultiBreakQML, MultiBreakResults

import fmbqml.multiple_break.model as mqml_module

from fmbqml.multiple_break.qml_optimization import build_qml_cost_matrix, find_min_qml


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
        assert isinstance(result, MultiBreakResults)
        result = result.to_dict()

    assert result["breakpoints"] == []
    assert result["n_breaks"] == 0


def test_joint_estimate_runs_and_returns_result(monkeypatch):
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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

    assert isinstance(result, dict)
    assert result["test_type"] == "mqml_joint"
    assert result["breakpoints"] == [15, 40]
    assert "break_labels" not in result
    assert result["n_breaks"] == 2
    assert result["factor_criterion"] == 1
    assert result["ic_path"] == make_fake_ic_path()
    assert model.joint_estimated is True


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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

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

    result = MultiBreakResults(model._joint_results).to_dict()
    ic_table = MultiBreakResults(model._joint_results).format_ic_path_table()

    assert result["breakpoints"] == [20, 50]
    assert "IC" in ic_table
    assert "10.0000" in ic_table
    assert "estimated_breakpoints" in ic_table
    assert "U_NT" not in ic_table


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

    fig, ax = MultiBreakResults(model._joint_results).plot_mqml_profile(show=False)

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Number of breaks"
    assert "MQML" in ax.get_title()

    plt.close(fig)


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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

    assert result["breakpoints"] == [20, 50]
    assert result["break_types"] == ["Type 1 (Singular)", "Type 2 (Rotational)"]
    assert result["break_type_summary"]["type1_count"] == 1
    assert result["break_type_summary"]["type2_count"] == 1
    assert result["break_type_summary"]["other_count"] == 0


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

    text = str(MultiBreakResults(model._joint_results).summary())

    assert isinstance(text, str)
    assert "MultiBreakQML Joint Estimation Results" in text
    assert "Estimated breakpoints:" in text
    assert "Type 1 (Singular)" in text
    assert "Type 2 (Rotational)" in text
    assert "epsilon=" not in text
    assert "Factor criterion:" in text
    assert "Break Classification" in text
    ic_table = MultiBreakResults(model._joint_results).format_ic_path_table()
    assert "estimated_breakpoints" in ic_table


def test_joint_estimation_failure_preserves_previous_results(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3)

    # Simulate results from a previous successful estimation.
    model.joint_estimated = True
    model._joint_results = {
        "breakpoints": [20, 50],
        "n_breaks": 2,
        "ic_path": make_fake_ic_path(),
    }

    # Simulate a failure during the new estimation.
    def fake_joint_estimation(*args, **kwargs):
        raise RuntimeError("Simulated estimation failure")

    monkeypatch.setattr(
        mqml_module,
        "joint_estimation",
        fake_joint_estimation,
    )

    with pytest.raises(RuntimeError, match="Simulated estimation failure"):
        model.estimate_breaks_jointly(
            min_break=0,
            classify=False,
        )

    # A failed computation leaves the previous successful state intact.
    assert model.joint_estimated is True
    assert model._joint_results["breakpoints"] == [20, 50]


def test_joint_estimation_builds_qml_cost_matrix_once(monkeypatch):
    """All candidate break counts reuse the same segment-cost matrix."""
    import fmbqml.multiple_break.joint_estimation as joint_module

    original_builder = joint_module.build_qml_cost_matrix
    calls = []

    def counted_builder(F_hat, min_length):
        calls.append((F_hat.shape, min_length))
        return original_builder(F_hat, min_length)

    monkeypatch.setattr(joint_module, "build_qml_cost_matrix", counted_builder)
    monkeypatch.setattr(
        joint_module, "_select_factor_number", lambda *args, **kwargs: 2
    )

    X = generate_panel_data(T=90, N=12, seed=918)
    model = MultiBreakQML(
        X,
        max_break=3,
        trim_ratio=0.15,
        max_factors=3,
        factor_criterion="IC2",
    )
    model.estimate_breaks_jointly(classify=False)

    assert len(calls) == 1
    assert calls[0][0][0] == X.shape[0]


def test_classify_joint_breaks_rejects_empty_estimated_breakpoint_set():
    model = MultiBreakQML(
        np.arange(80, dtype=float).reshape(20, 4),
        max_break=2,
        max_factors=2,
    )
    model.joint_estimated = True
    model._joint_results = {"breakpoints": []}

    with pytest.raises(ValueError, match="No estimated breakpoints"):
        model.classify_joint_breaks()

def test_qml_cost_matches_direct_segment_products():
    factors = np.random.default_rng(12).normal(size=(12, 3))
    costs = build_qml_cost_matrix(factors, 2)
    for start in range(12):
        for end in range(start + 1, 13):
            length = end - start
            if length < 3:
                assert np.isposinf(costs[start, end])
            else:
                segment = factors[start:end]
                expected = length * np.linalg.slogdet(segment.T @ segment / length)[1]
                assert costs[start, end] == pytest.approx(expected)


def test_find_min_qml_respects_minimum_regime_length():
    factors = np.random.default_rng(71).normal(size=(12, 2))
    value, breaks = find_min_qml(factors, 12, 3, 2)
    assert np.isfinite(value)
    assert len(breaks) == 2
    assert np.diff([0, *breaks, 12]).min() >= 3


@pytest.mark.parametrize("T", [True, 4.0, 0, -1])
def test_find_min_qml_rejects_invalid_sample_size(T):
    with pytest.raises(ValueError, match="T"):
        find_min_qml(np.ones((4, 1)), T, 1, 0)
