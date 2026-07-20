
"""High-level orchestration for breakpoint classification."""

from typing import Any, Dict, Optional, Sequence, Tuple
from fmbqml.break_tools.classification_summary import summarize_break_types
from fmbqml.break_tools.classify_break_types import classify_break_types


def classify_breakpoint_result(
        X,
        breakpoints: Optional[Sequence[int]],
        result_dict: Dict[str, Any],
        max_factors: int,
        factor_criterion: int,
        full_sample_factor_number: Optional[int],
        T: int,
        trim_ratio: Optional[float] = None,
        min_regime_length: Optional[int] = None,
        full_sample_standardize: bool = False,
        segment_standardize: bool = True,
        verbose: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Classify one or multiple breakpoints and update a result dictionary.

    This function provides a common wrapper for both SingleBreakQML and
    MultiBreakQML. It calls classify_break_types(), stores the classification
    fields in the result dictionary, and adds a summary of Type 1, Type 2,
    and other breakpoint types.

    Parameters
    ----------
    X : array-like
        Input data matrix.
    breakpoints : sequence of int or None
        Breakpoint locations. For a single-break result, pass [break_point].
    result_dict : dict
        Existing result dictionary to update.
    max_factors : int
        Maximum number of factors considered in classification.
    factor_criterion : int
        Factor-selection criterion index.
    full_sample_factor_number : int or None
        Full-sample factor number used as the upper bound in classification.
    T : int
        Sample size.
    trim_ratio : float, optional
        Minimum spacing ratio used by classify_break_types().
    min_regime_length : int, optional
        Minimum regime length used by classify_break_types().
    full_sample_standardize : bool, default=False
        Whether to standardize the full sample before classification.
    segment_standardize : bool, default=True
        Whether to standardize each segment before classification.
    verbose : bool, default=False
        Whether to print a short classification message.

    Returns
    -------
    updated_result : dict
        Updated result dictionary.
    classification : dict
        Classification output.
    """
    updated_result = result_dict.copy()

    if breakpoints is None:
        breakpoints = []
    else:
        breakpoints = [int(bp) for bp in breakpoints]

    if len(breakpoints) == 0:
        classification = {
            "break_types": [],
            "regime_factors": [],
            "combined_factors": [],
            "classification_criteria": [],
            "regime_boundaries": [0, T],
            "n_regimes": 1,
            "min_regime_length": min_regime_length,
            "classification_r_full": None,
            "break_type": None,
            "break_type_summary": summarize_break_types([], n_breakpoints=0),
            "classification_full_sample_standardize": full_sample_standardize,
            "classification_segment_standardize": segment_standardize,
        }

        updated_result.update(classification)

        if verbose:
            print("No breakpoints available for classification.")

        return updated_result, classification

    classification_r_full = None

    if full_sample_factor_number is not None:
        classification_r_full = int(full_sample_factor_number)

        if classification_r_full < 0:
            raise ValueError(
                "full_sample_factor_number must be a non-negative integer or None. "
                f"Got {full_sample_factor_number}."
            )

    kwargs = {
        "X": X,
        "breakpoints": breakpoints,
        "max_factors": max_factors,
        "factor_criterion": factor_criterion,
        "full_sample_factor_number": classification_r_full,
        "full_sample_standardize": full_sample_standardize,
        "segment_standardize": segment_standardize,
    }

    if min_regime_length is not None:
        kwargs["min_regime_length"] = min_regime_length
    elif trim_ratio is not None:
        kwargs["trim_ratio"] = trim_ratio
    else:
        kwargs["trim_ratio"] = 0.1

    try:
        classification = classify_break_types(**kwargs)
    except TypeError as exc:
        msg = str(exc)
        if (
                "full_sample_factor_number" in msg
                or "full_sample_standardize" in msg
                or "segment_standardize" in msg
        ):
            raise TypeError(
                "classify_break_types() must support the arguments "
                "full_sample_factor_number, full_sample_standardize, and "
                "segment_standardize."
            ) from exc
        raise

    break_types = classification.get("break_types", [])

    updated_result["break_types"] = break_types
    updated_result["regime_factors"] = classification.get("regime_factors", [])
    updated_result["combined_factors"] = classification.get("combined_factors", [])
    updated_result["classification_criteria"] = classification.get(
        "classification_criteria", []
    )
    updated_result["regime_boundaries"] = classification.get(
        "regime_boundaries", []
    )
    updated_result["n_regimes"] = classification.get(
        "n_regimes",
        len(breakpoints) + 1
    )

    updated_result["classification_r_full"] = classification.get(
        "r_full",
        classification_r_full
    )
    updated_result["classification_factor_criterion"] = classification.get(
        "factor_criterion",
        factor_criterion
    )

    if min_regime_length is not None:
        updated_result["min_regime_length"] = min_regime_length
    else:
        updated_result["min_regime_length"] = classification.get(
            "min_regime_length"
        )

    updated_result["break_type_summary"] = summarize_break_types(
        break_types,
        n_breakpoints=len(breakpoints)
    )

    if len(break_types) > 0:
        updated_result["break_type"] = break_types[0]
    else:
        updated_result["break_type"] = None

    updated_result["classification_full_sample_standardize"] = (
        full_sample_standardize
    )
    updated_result["classification_segment_standardize"] = segment_standardize

    classification["break_type"] = updated_result["break_type"]
    classification["break_type_summary"] = updated_result["break_type_summary"]
    classification["classification_r_full"] = updated_result[
        "classification_r_full"
    ]
    classification["classification_full_sample_standardize"] = (
        full_sample_standardize
    )
    classification["classification_segment_standardize"] = segment_standardize

    if verbose:
        summary = updated_result["break_type_summary"]
        print(
            "Break type summary: "
            f"Type 1 = {summary['type1_count']}, "
            f"Type 2 = {summary['type2_count']}, "
            f"Other = {summary['other_count']}"
        )

    return updated_result, classification
