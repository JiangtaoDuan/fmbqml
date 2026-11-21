"""Tests for joint multiple-break likelihood-ratio testing."""

import numpy as np
import pytest

import fmbqml.multiple_break.model as mqml_module
from fmbqml import MultiBreakQML, MultiBreakResults
from fmbqml.multiple_break.joint_testing import _normalize_factors

from ._helpers import generate_panel_data, make_fake_ic_path, normalized_factors


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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

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
            "estimated_factors": 2,
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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

    assert result["reject_null"] is True
    assert result["breakpoints"] == [13, 27]
    assert result["estimated_factors"] == 2
    stored_result = MultiBreakResults(model._joint_test_results).to_dict()
    assert stored_result["test_statistic"] == 20.0


def test_prespecified_lr_can_skip_classification(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=4)

    factors = {
        "estimated_factors": 2,
        "F_hat": np.ones((80, 2)),
        "L_hat": np.ones((12, 2)),
        "VNT": np.eye(2),
        "factor_criterion": 0,
        "criterion_name": "IC1",
    }
    monkeypatch.setattr(
        model, "get_factor_estimates", lambda recalculate=False: factors
    )

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
    assert isinstance(result, MultiBreakResults)
    result = result.to_dict()

    assert "break_types" not in result


def test_joint_lr_result_nonrejection():
    result = MultiBreakResults({"test_type": "joint_sup_lr", "reject_null": False,
                               "breakpoints": [], "candidate_breakpoints": [20]})
    assert result.breakpoints == []
    candidates = result.candidate_breakpoints
    candidates.append(30)
    assert result.candidate_breakpoints == [20]


@pytest.mark.parametrize("reject", [True, False])
def test_public_joint_lr_accessors_preserve_candidate_breaks(reject):
    model = MultiBreakQML(generate_panel_data(T=80, N=8), max_break=2, max_factors=3)
    model.joint_tested = True
    model._joint_test_results = {
        "breakpoints": [20, 50], "reject_null": reject,
        "break_types": ["Type 1 (Singular)", "Type 2 (Rotational)"],
    }
    assert MultiBreakResults(model._joint_test_results).reject_null is reject
    assert MultiBreakResults(model._joint_test_results).breakpoints == [20, 50]
    assert MultiBreakResults(model._joint_test_results).break_types == [
        "Type 1 (Singular)", "Type 2 (Rotational)"
    ]


def test_joint_lr_failure_preserves_previous_results(monkeypatch):
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3)

    # Simulate results from a previous successful LR test.
    model.joint_tested = True
    model._joint_test_results = {
        "breakpoints": [20, 50],
        "reject_null": True,
    }

    # Avoid real factor estimation.
    monkeypatch.setattr(
        model,
        "get_factor_estimates",
        lambda recalculate=False: {
            "estimated_factors": 2,
            "F_hat": np.ones((80, 2)),
        },
    )

    # Simulate a failure during the new LR test.
    def fake_run_joint_sup_lr(**kwargs):
        raise RuntimeError("Simulated LR failure")

    monkeypatch.setattr(
        mqml_module,
        "_run_joint_sup_lr",
        fake_run_joint_sup_lr,
    )

    with pytest.raises(RuntimeError, match="Simulated LR failure"):
        model.joint_sup_lr_test(
            n_breaks=2,
            classify=False,
        )

    # A failed computation leaves the previous successful state intact.
    assert model.joint_tested is True
    assert model._joint_test_results["breakpoints"] == [20, 50]


def test_factor_normalization_is_scale_invariant():
    rng = np.random.default_rng(917)
    factors = rng.standard_normal((80, 3))
    baseline = _normalize_factors(factors)
    for scale in (1e-8, 1e-4, 1e4, 1e8):
        candidate = _normalize_factors(scale * factors)
        np.testing.assert_allclose(candidate, baseline, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(candidate.T @ candidate / len(candidate),
                                   np.eye(3), atol=2e-13)


def test_factor_normalization_rejects_real_rank_deficiency():
    factors = np.ones((60, 2))
    with pytest.raises(ValueError, match="rank deficient"):
        _normalize_factors(factors)
    with pytest.raises(ValueError, match="positive finite"):
        _normalize_factors(np.zeros((60, 2)))
