"""Tests for utils test_utils."""

import numpy as np
import pytest

from fmbqml.utils.breakpoint_utils import (
    calculate_break_range,
    compute_min_segment_length,
)
from fmbqml.utils.partition import optimize_additive_partition
from fmbqml.utils.safe_logdet import safe_logdet


@pytest.mark.parametrize("maximize,sentinel", [(False, np.inf), (True, -np.inf)])
def test_partition_preserves_directional_infinity_markers(maximize, sentinel):
    scores = np.full((7, 7), sentinel)
    for start in range(6):
        for end in range(start + 1, 7):
            scores[start, end] = float(end - start)
    optimum, points = optimize_additive_partition(scores, 2, 2, maximize=maximize)
    assert np.isfinite(optimum)
    assert len(points) == 1


def test_partition_nan_rejected_even_in_unused_cell():
    values = np.zeros((8, 8))
    values[0, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        optimize_additive_partition(values, 1, 1)


def test_partition_matches_brute_force_for_two_segments():
    rng = np.random.default_rng(501)
    score = rng.normal(size=(10, 10))
    for maximize in (False, True):
        optimum, breaks = optimize_additive_partition(score, 2, 2, maximize)
        alternatives = [(float(score[0, j] + score[j, 9]), j)
                        for j in range(2, 8)]
        brute = max(alternatives) if maximize else min(alternatives)
        assert optimum == pytest.approx(brute[0])
        assert breaks == [brute[1]]


@pytest.mark.parametrize("field,value", [
    ("n_segments", True), ("min_length", np.bool_(True)),
    ("n_segments", 1.5), ("min_length", 1.5),
    ("n_segments", "2"), ("min_length", None),
])
def test_partition_rejects_non_integral_segment_options(field, value):
    kwargs = {"n_segments": 2, "min_length": 1}
    kwargs[field] = value
    with pytest.raises(TypeError, match=field):
        optimize_additive_partition(np.zeros((7, 7)), **kwargs)


@pytest.mark.parametrize("field,value", [
    ("n_segments", 0), ("min_length", 0),
    ("n_segments", -2), ("min_length", -1),
])
def test_partition_rejects_nonpositive_segment_options(field, value):
    kwargs = {"n_segments": 2, "min_length": 1}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        optimize_additive_partition(np.zeros((7, 7)), **kwargs)


def test_partition_accepts_numpy_integer_parameters():
    optimum, breakpoints = optimize_additive_partition(
        np.zeros((7, 7)), np.int64(2), np.int64(2))
    assert optimum == 0
    assert len(breakpoints) == 1

def test_calculate_break_range_function():
    t_min, t_max = calculate_break_range(T=200, trim_ratio=0.1)

    assert t_min == 20
    assert t_max == 180


def test_compute_min_segment_length_function():
    assert compute_min_segment_length(100, 0.1) == 10
    assert compute_min_segment_length(200, 0.05) == 10
    assert compute_min_segment_length(101, 0.1) == 11

def test_safe_logdet_cholesky_matches_unregularized_spd():
    matrix = np.array([[2.0, 0.3], [0.3, 0.6]])
    expected = np.linalg.slogdet(matrix)[1]
    assert safe_logdet(matrix) == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_safe_logdet_bounded_regularization_for_singular_covariance():
    matrix = np.diag([1.0, 0.0])
    expected = np.log(1.0 + 1e-8) + np.log(1e-8)
    assert safe_logdet(matrix) == pytest.approx(expected, abs=1e-9)


def test_safe_logdet_rejects_strongly_indefinite_matrix():
    with pytest.raises(np.linalg.LinAlgError, match="Cholesky"):
        safe_logdet(np.diag([1.0, -0.5]))


def test_safe_logdet_rejects_material_asymmetry():
    with pytest.raises(ValueError, match="symmetric"):
        safe_logdet(np.array([[1.0, 0.9], [0.0, 1.0]]))


def test_safe_logdet_progresses_to_next_jitter_only_when_needed():
    # 1e-8 is insufficient; the next bounded step, 1e-7, succeeds.
    matrix = np.diag([1.0, -5e-8])
    expected = np.log(1.0 + 1e-7) + np.log(5e-8)
    assert safe_logdet(matrix) == pytest.approx(expected, abs=1e-8)
