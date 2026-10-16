
"""Utilities for validating and labeling breakpoint locations."""

from typing import Tuple
from numbers import Integral, Real
from typing import Any, Iterable, List, Optional
import numpy as np


def calculate_break_range(T: int, trim_ratio: float) -> Tuple[int, int]:
    """
    Compute and validate the admissible breakpoint search range.

    Parameters
    ----------
    T : int
        Sample size.
    trim_ratio : float
        Trimming ratio. Must satisfy 0 < trim_ratio < 0.5.

    Returns
    -------
    tuple
        (t_min, t_max), the lower and upper bounds of the breakpoint
        search interval.
    """
    if isinstance(T, bool) or not isinstance(T, Integral) or T <= 0:
        raise ValueError(f"T must be a positive integer. Got T={T}.")

    if (
        isinstance(trim_ratio, bool)
        or not isinstance(trim_ratio, Real)
        or not (0 < trim_ratio < 0.5)
    ):
        raise ValueError(
            f"trim_ratio must be a real number in (0, 0.5). "
            f"Got trim_ratio={trim_ratio}."
        )

    h = compute_min_segment_length(T, trim_ratio)

    t_min = h
    t_max = T - h

    if t_min >= t_max:
        raise ValueError(
            f"Invalid break search range: ({t_min}, {t_max}). "
            f"Check trim_ratio={trim_ratio} or sample size T={T}."
        )

    return t_min, t_max


def compute_min_segment_length(T: int, trim_ratio: float) -> int:
    """
    Compute the minimum segment length implied by trim_ratio.

    Parameters
    ----------
    T : int
        Sample size.
    trim_ratio : float
        Trimming ratio for breakpoint search.
        Must satisfy 0 < trim_ratio < 0.5.

    Returns
    -------
    int
        Minimum segment length, computed as ceil(T * trim_ratio).
    """
    if isinstance(T, bool) or not isinstance(T, Integral) or T <= 0:
        raise ValueError(f"T must be a positive integer. Got T={T}.")

    if (
        isinstance(trim_ratio, bool)
        or not isinstance(trim_ratio, Real)
        or not (0 < trim_ratio < 0.5)
    ):
        raise ValueError(
            f"trim_ratio must be a real number in (0, 0.5). "
            f"Got trim_ratio={trim_ratio}."
        )

    min_segment_length = int(np.ceil(T * trim_ratio - 1e-12))

    if min_segment_length < 1:
        raise ValueError(
            "The minimum segment length is less than 1. "
            f"Got T={T}, trim_ratio={trim_ratio}, "
            f"min_segment_length={min_segment_length}."
        )

    return min_segment_length

def map_breakpoint_to_label(
        breakpoint: Optional[int],
        index
) -> Optional[Any]:
    """
    Map a 1-indexed breakpoint position to the corresponding index label.
    """
    if breakpoint is None or index is None:
        return None

    idx = int(breakpoint) - 1

    if idx < 0 or idx >= len(index):
        return None

    return index[idx]


def map_breakpoints_to_labels(
        breakpoints: Iterable[int],
        index
) -> List[Any]:
    """
    Map 1-indexed breakpoint positions to corresponding index labels.
    """
    if breakpoints is None or index is None:
        return []

    labels = []

    for breakpoint in breakpoints:
        label = map_breakpoint_to_label(breakpoint, index)
        if label is not None:
            labels.append(label)

    return labels
