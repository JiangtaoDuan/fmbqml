"""Tests for test workflows."""

import numpy as np
import pandas as pd
import pytest

from fmbqml import MultiBreakQML, SingleBreakQML


@pytest.mark.parametrize("model_class", [SingleBreakQML, MultiBreakQML])
def test_model_accepts_panel_and_preserves_input(model_class, loading_break_panel):
    original = loading_break_panel.copy()
    model = model_class(loading_break_panel, max_factors=3)
    np.testing.assert_array_equal(model.get_data(), original)
    assert not model.X.flags.writeable
    np.testing.assert_array_equal(loading_break_panel, original)


def test_single_break_recovers_known_loading_change(loading_break_panel):
    model = SingleBreakQML(
        loading_break_panel, trim_ratio=0.15,
        max_factors=3, factor_criterion="IC2",
    )
    result = model.estimate_breakpoint(classify=False)
    assert abs(result.breakpoint - 50) <= 2
    assert 1 <= result.estimated_factors <= 3
    assert "50" in str(result.summary())
    assert model.get_data().shape == loading_break_panel.shape


def test_multiple_break_recovers_known_loading_change(loading_break_panel):
    model = MultiBreakQML(
        loading_break_panel, max_break=2, trim_ratio=0.15,
        max_factors=3, factor_criterion="IC2",
    )
    result = model.estimate_breaks_jointly(classify=False)
    assert result.breakpoints == [50]
    assert result.n_breaks == 1
    assert result.ic_path is not None
    assert "50" in str(result.summary())


def test_dataframe_labels_are_preserved(loading_break_panel):
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    panel = pd.DataFrame(loading_break_panel, index=index)
    model = SingleBreakQML(panel, trim_ratio=0.15, max_factors=3,
                           factor_criterion="IC2")
    result = model.estimate_breakpoint(classify=False)
    assert result.breakpoint == 50
    assert result.break_label == index[49]
    pd.testing.assert_frame_equal(model.get_data(), panel)


@pytest.mark.parametrize("model_class", [SingleBreakQML, MultiBreakQML])
def test_model_rejects_invalid_panel_at_boundary(model_class, loading_break_panel):
    invalid = loading_break_panel.copy()
    invalid[10, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model_class(invalid, max_factors=3)


def test_invalid_set_data_does_not_change_existing_panel(loading_break_panel):
    model = MultiBreakQML(loading_break_panel, max_factors=3, max_break=2)
    before = model.get_data()
    with pytest.raises(ValueError, match="time variation"):
        model.set_data(np.ones_like(before))
    np.testing.assert_array_equal(model.get_data(), before)
