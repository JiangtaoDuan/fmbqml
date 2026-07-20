

"""Formatting helpers for breakpoint-classification results."""

from typing import Dict, Iterable, Optional


def summarize_break_types(
        break_types: Optional[Iterable],
        n_breakpoints: Optional[int] = None
) -> Dict[str, int]:
    """
    Summarize breakpoint types.

    Parameters
    ----------
    break_types : iterable or None
        Breakpoint type labels, such as "Type 1 (Singular)" or
        "Type 2 (Rotational)".
    n_breakpoints : int, optional
        Total number of breakpoints. If provided, it is used to compute
        the number of other / ambiguous / unknown breakpoints.

    Returns
    -------
    dict
        Dictionary containing counts of Type 1, Type 2, and other breakpoints.
    """
    if break_types is None:
        break_types = []

    break_types = list(break_types)

    type1_count = sum(
        1 for bt in break_types
        if isinstance(bt, str) and bt.startswith("Type 1")
    )

    type2_count = sum(
        1 for bt in break_types
        if isinstance(bt, str) and bt.startswith("Type 2")
    )

    if n_breakpoints is None:
        total = len(break_types)
    else:
        total = int(n_breakpoints)

    other_count = max(0, total - type1_count - type2_count)

    return {
        "type1_count": type1_count,
        "type2_count": type2_count,
        "other_count": other_count,
    }
