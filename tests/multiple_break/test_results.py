"""Tests for multiple-break result objects and reporting."""

import numpy as np
import pandas as pd
import pytest

from ._helpers import generate_panel_data, make_fake_ic_path
from fmbqml import MultiBreakQML, MultiBreakResults


def test_multiple_break_result_getters_return_nested_copies():
    model = MultiBreakQML(
        np.arange(80, dtype=float).reshape(20, 4),
        max_break=2,
        max_factors=2,
    )
    model.joint_estimated = True
    model.joint_tested = True
    model._joint_results = {
        "breakpoints": [5, 10],
        "ic_path": [{"estimated_breakpoints": [5]}],
    }
    model._joint_test_results = {"joint_lr_profile": np.array([1.0, 2.0])}

    estimated = MultiBreakResults(model._joint_results).to_dict()
    tested = MultiBreakResults(model._joint_test_results).to_dict()
    estimated["breakpoints"][0] = 999
    estimated["ic_path"][0]["estimated_breakpoints"][0] = 999
    tested["joint_lr_profile"][0] = 999.0

    assert model._joint_results["breakpoints"] == [5, 10]
    assert model._joint_results["ic_path"][0]["estimated_breakpoints"] == [5]
    np.testing.assert_array_equal(
        model._joint_test_results["joint_lr_profile"],
        np.array([1.0, 2.0]),
    )


def test_multiple_break_result_print_is_compact_and_summary_is_detailed():
    result = MultiBreakResults({
        "test_type": "joint_sup_lr",
        "T": 600,
        "N": 100,
        "factor_criterion": 0,
        "estimated_factors": 12,
        "n_breaks": 3,
        "test_statistic": 13228.7429,
        "critical_value": 3921.1606,
        "significance_level": 0.05,
        "reject_null": True,
        "candidate_breakpoints": [150, 300, 450],
        "breakpoints": [150, 300, 450],
        "break_types": ["Type 1 (Singular)"] * 3,
        "regime_factors": [3, 3, 3, 3],
        "combined_factors": [6, 6, 6],
        "monte_carlo_replications": 1000,
    })

    printed = str(result)
    detailed = str(result.summary())

    assert "MultiBreakQML Joint LR Test" in printed
    assert "LR statistic: 13228.7429" in printed
    assert "Accepted breakpoints: 150, 300, 450" in printed
    assert "Breakpoint types:" in printed
    assert printed.count("Type 1 (Singular)") == 3
    assert "\n\nLR statistic:" in printed
    assert "\n\nAccepted breakpoints:" in printed
    assert "Alternative:" not in printed
    assert "Accepted breakpoints: 150, 300, 450" in printed
    assert "Break Classification" not in printed
    assert "Model Information" in detailed
    assert "Hypothesis Test" in detailed
    assert "Break Classification" in detailed
    assert "Prespecified number of breaks" not in detailed
    assert "Candidate breakpoints under H1: [150, 300, 450]" in detailed
    assert "Accepted breakpoints:" in detailed and "[150, 300, 450]" in detailed
    assert "Estimated breakpoints:" not in detailed
    assert str(result) == repr(result)


def test_multiple_break_result_summary_distinguishes_unaccepted_candidates():
    result = MultiBreakResults({
        "test_type": "joint_sup_lr",
        "T": 100,
        "N": 20,
        "n_breaks": 2,
        "test_statistic": 10.0,
        "critical_value": 12.0,
        "significance_level": 0.05,
        "reject_null": False,
        "candidate_breakpoints": [30, 70],
        "breakpoints": [],
    })

    detailed = str(result.summary())
    printed = str(result)

    assert "Candidate breakpoints under H1: [30, 70]" in detailed
    assert "Accepted breakpoints:" in detailed and "None" in detailed
    assert "Accepted breakpoints: None" in printed
    assert "Breakpoint types:" not in printed
    assert result.candidate_breakpoints == [30, 70]


def test_multiple_break_estimation_print_is_compact():
    result = MultiBreakResults({
        "test_type": "mqml_joint",
        "T": 80,
        "N": 12,
        "factor_criterion": 5,
        "estimated_factors": 3,
        "n_breaks": 2,
        "breakpoints": [20, 50],
        "break_types": ["Type 1 (Singular)", "Type 2 (Rotational)"],
        "regime_factors": [2, 2, 2],
        "combined_factors": [3, 2],
    })

    printed = str(result)
    detailed = str(result.summary())

    assert "MultiBreakQML Joint Estimation" in printed
    assert "Selected breaks: 2" in printed
    assert "Estimated breakpoints: 20, 50" in printed
    assert "Breakpoint types: Type 1 (Singular), Type 2 (Rotational)" in printed
    assert "\n\nSelected breaks: 2" in printed
    assert "Break Classification" not in printed
    assert "Break Classification" in detailed


def test_result_constructor_requires_dictionary():
    with pytest.raises(TypeError, match="dictionary"):
        MultiBreakResults([])


def test_get_ic_path_returns_dataframe_and_independent_copy():
    X = generate_panel_data(T=80, N=12)
    model = MultiBreakQML(X, max_break=2, max_factors=3)

    model.joint_estimated = True
    model._joint_results = {
        "ic_path": make_fake_ic_path()
    }

    ic_path = MultiBreakResults(model._joint_results).ic_path

    assert isinstance(ic_path, pd.DataFrame)
    assert len(ic_path) == 3
    assert list(ic_path["m"]) == [0, 1, 2]
    assert list(ic_path["IC"]) == [10.0, 8.0, 6.0]

    assert ic_path.loc[2, "estimated_breakpoints"] == [20, 50]

    # Modify the returned DataFrame.
    ic_path.at[0, "IC"] = 999.0
    ic_path.at[2, "estimated_breakpoints"] = [999]

    # The original cached results must remain unchanged.
    assert model._joint_results["ic_path"][0]["IC"] == 10.0
    assert model._joint_results["ic_path"][2]["estimated_breakpoints"] == [20, 50]

    # A subsequent call should return the original values.
    new_ic_path = MultiBreakResults(model._joint_results).ic_path
    assert new_ic_path.loc[0, "IC"] == 10.0
    assert new_ic_path.loc[2, "estimated_breakpoints"] == [20, 50]
