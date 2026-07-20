"""Statistical procedures for classifying factor-model breaks."""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import fmbqml.utils as U
from fmbqml.factor_tools.nb_factors import bai_ng_factor_count

def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """
    Standardize each column using the sample standard deviation.
    """
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)

    sd[~np.isfinite(sd) | (sd == 0)] = 1.0

    return (X - mu) / sd


def _criterion_vector_from_bai_ng_result(
    result: Dict[str, Any],
    criterion: str
) -> np.ndarray:
    """
    Extract criterion values for k = 0, ..., kmax from bai_ng_factor_count output.

    Parameters
    ----------
    result : dict
        Output dictionary returned by bai_ng_factor_count().
    criterion : str
        Criterion name.

    Returns
    -------
    ndarray
        Criterion value vector over the candidate factor numbers.
    """
    if criterion == "IC1":
        return np.asarray(result["IC"][:, 1], dtype=float)
    if criterion == "IC2":
        return np.asarray(result["IC"][:, 2], dtype=float)
    if criterion == "IC3":
        return np.asarray(result["IC"][:, 3], dtype=float)
    if criterion == "PC1":
        return np.asarray(result["PC"][:, 1], dtype=float)
    if criterion == "PC2":
        return np.asarray(result["PC"][:, 2], dtype=float)
    if criterion == "PC3":
        return np.asarray(result["PC"][:, 3], dtype=float)
    if criterion == "AIC3":
        return np.asarray(result["AIC3"][:, 1], dtype=float)
    if criterion == "BIC3":
        return np.asarray(result["BIC3"][:, 1], dtype=float)

    raise ValueError(f"Unsupported criterion: {criterion}")


def _estimate_factor_number_by_nb(
    X_segment: np.ndarray,
    max_factors: int,
    factor_criterion: str,
    min_length: int,
    lower_bound: int = 0,
    upper_bound: Optional[int] = None,
    standardize_segment: bool = True,
) -> Optional[int]:
    """
    Estimate the number of factors in a segment using Bai--Ng criteria.

    The search can be restricted to a constrained grid
    [lower_bound, upper_bound]. This is useful when estimating the number
    of pseudo-factors in a merged adjacent regime.

    Parameters
    ----------
    X_segment : ndarray, shape (T_seg, N)
        Data matrix for the segment.
    max_factors : int
        Maximum number of factors considered.
    factor_criterion : str
        Factor-selection criterion.
    min_length : int
        Minimum segment length required for factor-number selection.
    lower_bound : int, default=0
        Lower bound of the candidate factor-number grid.
    upper_bound : int, optional
        Upper bound of the candidate factor-number grid.
    standardize_segment : bool, default=True
        Whether to standardize the segment before factor-number selection.

    Returns
    -------
    int or None
        Estimated number of factors. Returns None if the segment is too short
        or no feasible search grid is available.
    """
    T_seg, N_seg = X_segment.shape

    if T_seg < min_length:
        warnings.warn(
            f"Segment is too short for factor-number selection: "
            f"T_seg={T_seg}, required min_length={min_length}. "
            "This segment will be skipped and the corresponding break type "
            "may be reported as Unknown.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if T_seg < 2 or N_seg < 2:
        warnings.warn(
            f"Segment is too small for factor-number selection: "
            f"T_seg={T_seg}, N_seg={N_seg}. "
            "At least two observations and two variables are required.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    Z = _zscore_columns(X_segment) if standardize_segment else X_segment.copy()

    kmax_eff = min(int(max_factors), T_seg - 1, N_seg - 1)

    if upper_bound is not None:
        kmax_eff = min(kmax_eff, int(upper_bound))

    lower_eff = max(0, int(lower_bound))

    if kmax_eff < lower_eff:
        warnings.warn(
            f"No feasible factor-number search grid: "
            f"lower_bound={lower_eff}, upper_bound={kmax_eff}. "
            "This may occur when the segment is too short or when the "
            "merged-regime lower bound exceeds the feasible upper bound.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    result = bai_ng_factor_count(Z, kmax=kmax_eff)
    values = _criterion_vector_from_bai_ng_result(result, factor_criterion)

    sub_values = values[lower_eff:kmax_eff + 1]
    idx = int(np.argmin(sub_values))

    return int(lower_eff + idx)


def classify_break_types(
    X: Union[np.ndarray, pd.DataFrame],
    breakpoints: List[int],
    max_factors: int = 10,
    trim_ratio: Optional[float] = 0.1,
    min_regime_length: Optional[int] = None,
    factor_criterion: Union[int, str] = "IC2",
    full_sample_factor_number: Optional[int] = None,
    full_sample_standardize: bool = True,
    segment_standardize: bool = True,
) -> Dict[str, Any]:
    """
    Classify estimated breakpoints according to factor-number relationships.

    For each estimated breakpoint, the procedure estimates:
    1. the number of effective factors in the left regime, r_j;
    2. the number of effective factors in the right regime, r_{j+1};
    3. the number of pseudo-factors in the merged adjacent regime, r_{j,j+1}.

    The breakpoint type is then determined by comparing these quantities.

    Parameters
    ----------
    X : ndarray or pandas.DataFrame, shape (T, N)
        Input data matrix after missing-value processing or imputation.
    breakpoints : list of int
        Estimated break indices. The convention is
        edges = [0] + breakpoints + [T], so X[0:bp] is the left segment
        for a breakpoint bp.
    max_factors : int, default=10
        Maximum number of candidate factors.
    trim_ratio : float, optional, default=0.1
        Minimum regime-length fraction. Used only when min_regime_length
        is not provided.
    min_regime_length : int, optional
        Minimum number of observations in each regime. If provided, it
        overrides trim_ratio.
    factor_criterion : str or int, default="IC2"
        Factor-selection criterion. Supported names are
        {'IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3', 'AIC3', 'BIC3'}.
        Integer values 0--7 are also supported.
    full_sample_factor_number : int, optional
        Full-sample number of pseudo-factors. If not provided, it is estimated
        using the selected Bai--Ng criterion.
    full_sample_standardize : bool, default=True
        Whether to standardize the full input data before classification.
    segment_standardize : bool, default=True
        Whether to standardize each single or merged segment before
        factor-number selection.

    Classification rule
    -------------------
    Type 1 (Singular):
        r_{j,j+1} > min(r_j, r_{j+1})

    Type 2 (Rotational):
        r_{j,j+1} = min(r_j, r_{j+1})

    Returns
    -------
    dict
        Dictionary containing break types, regime-specific factor numbers,
        merged-regime factor numbers, classification criteria, regime
        boundaries, and related settings.
    """
    factor_criterion_index = U.normalize_factor_criterion(factor_criterion)

    factor_criterion_name = U.CRITERION_NAMES[factor_criterion_index]

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)

    T, N = U.validate_input_matrix(X, name="X")

    max_factors = U.validate_max_factors(max_factors, T, N)

    breakpoints = sorted(set(int(b) for b in breakpoints if 0 < int(b) < T))

    if min_regime_length is not None:
        min_length = U.validate_positive_integer(
            min_regime_length,
            "min_regime_length"
        )
    else:
        trim_ratio = U.validate_trim_ratio(trim_ratio)
        min_length = U.compute_min_segment_length(T, trim_ratio)

    X_work = _zscore_columns(X) if full_sample_standardize else X.copy()

    # Estimate or use the full-sample number of pseudo-factors.
    if full_sample_factor_number is None:
        r_full = _estimate_factor_number_by_nb(
            X_segment=X_work,
            max_factors=max_factors,
            factor_criterion=factor_criterion_name,
            min_length=1,
            lower_bound=0,
            upper_bound=max_factors,
            standardize_segment=False,
        )
        if r_full is None:
            r_full = 0
    else:
        r_full = U.validate_nonnegative_integer(
            full_sample_factor_number,
            "full_sample_factor_number"
        )
    if len(breakpoints) == 0:
        return {
            "break_types": [],
            "regime_factors": [],
            "combined_factors": [],
            "classification_criteria": [],
            "regime_boundaries": [0, T],
            "n_regimes": 1,
            "min_regime_length": min_length,
            "factor_criterion": factor_criterion,
            "r_full": r_full,
        }

    regime_boundaries = [0] + breakpoints + [T]
    n_regimes = len(regime_boundaries) - 1

    # Estimate r_j for each individual regime.
    regime_factors = []

    for i in range(n_regimes):
        start = regime_boundaries[i]
        end = regime_boundaries[i + 1]

        r_i = _estimate_factor_number_by_nb(
            X_segment=X_work[start:end, :],
            max_factors=max_factors,
            factor_criterion=factor_criterion_name,
            min_length=min_length,
            lower_bound=0,
            upper_bound=r_full,
            standardize_segment=segment_standardize,
        )
        regime_factors.append(r_i)

    # Estimate r_{j,j+1} for each merged adjacent regime and classify breaks.
    break_types = []
    combined_factors = []
    classification_criteria = []

    for j, bp in enumerate(breakpoints):
        left_regime = j
        right_regime = j + 1

        r_j = regime_factors[left_regime]
        r_j1 = regime_factors[right_regime]

        if r_j is None or r_j1 is None:
            break_type = "Unknown (insufficient data)"
            r_j_j1 = None
            min_r = None
            max_r = None
            r_lower = None
            r_upper = None
            condition = (
                "At least one adjacent regime is shorter than min_regime_length."
            )
        else:
            combined_start = regime_boundaries[left_regime]
            combined_end = regime_boundaries[right_regime + 1]

            min_r = min(r_j, r_j1)
            max_r = max(r_j, r_j1)

            # The merged-regime factor number should not be smaller than
            # the larger factor number of the two adjacent regimes.
            r_lower = max_r
            r_upper = r_full

            r_j_j1 = _estimate_factor_number_by_nb(
                X_segment=X_work[combined_start:combined_end, :],
                max_factors=max_factors,
                factor_criterion=factor_criterion_name,
                min_length=min_length,
                lower_bound=r_lower,
                upper_bound=r_upper,
                standardize_segment=segment_standardize,
            )

            if r_j_j1 is None:
                # Fallback when the constrained search grid is empty.
                q_merge = min(combined_end - combined_start, N)
                r_j_j1 = min(r_upper, q_merge - 1)
                condition_extra = (
                    " Constrained grid was empty; used feasible upper bound."
                )
            else:
                condition_extra = ""

            if r_j_j1 > min_r:
                break_type = "Type 1 (Singular)"
                condition = (
                    f"r_j,j+1 ({r_j_j1}) > min(r_j,r_j+1) ({min_r}); "
                    f"merged search grid=[{r_lower},{r_upper}]."
                    + condition_extra
                )
            elif r_j_j1 == min_r:
                break_type = "Type 2 (Rotational)"
                condition = (
                    f"r_j,j+1 ({r_j_j1}) = min(r_j,r_j+1) ({min_r}); "
                    f"merged search grid=[{r_lower},{r_upper}]."
                    + condition_extra
                )
            else:
                break_type = "Unknown (invalid rank relation)"
                condition = (
                    f"Estimated r_j,j+1 ({r_j_j1}) < "
                    f"min(r_j,r_j+1) ({min_r}); "
                    "this violates the expected rank relation."
                    + condition_extra
                )

        combined_factors.append(r_j_j1)

        classification_criteria.append({
            "break_index": bp,
            "r_j": r_j,
            "r_j1": r_j1,
            "r_j_j1": r_j_j1,
            "min_r": min_r,
            "max_r": max_r,
            "r_lower_for_merge": r_lower,
            "r_upper_for_merge": r_upper,
            "condition": condition,
            "type": break_type,
        })

        break_types.append(break_type)

    return {
        "break_types": break_types,
        "regime_factors": regime_factors,
        "combined_factors": combined_factors,
        "classification_criteria": classification_criteria,
        "regime_boundaries": regime_boundaries,
        "n_regimes": n_regimes,
        "min_regime_length": min_length,
        "factor_criterion": factor_criterion_name,
        "r_full": r_full,
    }
