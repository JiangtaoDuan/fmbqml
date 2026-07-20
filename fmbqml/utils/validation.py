
"""Argument validation shared by public estimators and tests."""

from numbers import Integral, Real
from typing import Tuple
import numpy as np



def validate_input_matrix(X, name: str = "X") -> Tuple[int, int]:
    """
    Validate a numerical input matrix and return its shape.
    """
    X = np.asarray(X)

    if X.size == 0:
        raise ValueError(f"Input {name} is empty.")

    if X.ndim != 2:
        raise ValueError(
            f"Input {name} must be a 2D array (T x N). Got {X.ndim}D array."
        )

    T, N = X.shape

    if T < 2:
        raise ValueError(f"Sample size T must be at least 2. Got T={T}.")

    if N < 1:
        raise ValueError(f"Number of variables N must be at least 1. Got N={N}.")

    try:
        is_finite = np.isfinite(X).all()
    except TypeError as exc:
        raise ValueError(f"Input {name} must contain numerical values only.") from exc

    if not is_finite:
        raise ValueError(f"Input {name} contains NaN or infinite values.")

    return T, N


def validate_trim_ratio(trim_ratio: float) -> float:
    """
    Validate trimming ratio.
    """
    if (
        isinstance(trim_ratio, bool)
        or not isinstance(trim_ratio, Real)
        or not (0 < trim_ratio < 0.5)
    ):
        raise ValueError(
            f"trim_ratio must be a real number in (0, 0.5). "
            f"Got trim_ratio={trim_ratio}."
        )

    return float(trim_ratio)


def validate_alpha(alpha: float) -> float:
    """
    Validate significance level for the LR test.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError(
            f"Significance level 'alpha' must be one of "
            f"(0.01, 0.05, 0.1). Got {alpha}."
        )

    alpha = float(alpha)

    if alpha not in (0.01, 0.05, 0.1):
        raise ValueError(
            f"Significance level 'alpha' must be one of "
            f"(0.01, 0.05, 0.1). Got {alpha}."
        )

    return alpha


def validate_positive_integer(value, name: str) -> int:
    """
    Validate a strictly positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {name}={value}.")

    return int(value)


def validate_nonnegative_integer(value, name: str) -> int:
    """
    Validate a non-negative integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer. Got {name}={value}."
        )

    return int(value)

def _get_max_factor_dim(T: int, N: int) -> int:
    """
    Return the maximum admissible number of factors.

    The number of factors must be smaller than min(T, N),
    so that at least one residual dimension is left.
    """
    max_factor_dim = min(T, N) - 1

    if max_factor_dim < 1:
        raise ValueError(
            "The panel dimensions are too small for factor estimation. "
            f"Require min(T, N) >= 2, got T={T}, N={N}."
        )

    return max_factor_dim


def validate_max_factors(max_factors: int, T: int, N: int) -> int:
    """
    Validate max_factors relative to panel dimensions.
    """
    max_factors = validate_positive_integer(max_factors, "max_factors")

    max_factor_dim = _get_max_factor_dim(T, N)

    if max_factors > max_factor_dim:
        raise ValueError(
            f"Parameter 'max_factors' is too large relative to sample size. "
            f"Require max_factors <= min(T, N) - 1 = {max_factor_dim}, "
            f"got max_factors={max_factors}."
        )

    return max_factors


def validate_joint_sup_lr_arguments(
        F_hat,
        n_breaks,
        trim_ratio,
        alpha,
        n_sim,
        grid_size,
):
    """Validate and normalize all arguments of the joint sup-LR test."""
    F_hat = np.asarray(F_hat, dtype=float)
    T, r = validate_input_matrix(F_hat, name="F_hat")
    n_breaks = validate_positive_integer(n_breaks, "n_breaks")
    trim_ratio = validate_trim_ratio(trim_ratio)
    alpha = validate_alpha(alpha)
    n_sim = validate_positive_integer(n_sim, "n_sim")
    grid_size = validate_positive_integer(grid_size, "grid_size")

    if grid_size < 10:
        raise ValueError("grid_size must be an integer of at least 10.")

    # Theorem 5 and the breakpoint-estimation API use ceil(T * trim_ratio).
    min_length = int(np.ceil(T * trim_ratio))
    if min_length <= r:
        raise ValueError(
            "The minimum segment length must exceed the factor dimension. "
            "Increase trim_ratio or reduce the number of factors."
        )
    if (n_breaks + 1) * min_length > T:
        raise ValueError(
            "The requested number of breaks is infeasible under trim_ratio: "
            f"({n_breaks + 1}) * {min_length} > {T}."
        )

    grid_min_length = int(np.ceil(grid_size * trim_ratio))
    if (n_breaks + 1) * grid_min_length > grid_size:
        raise ValueError("grid_size is too small for n_breaks and trim_ratio.")

    return (
        F_hat,
        min_length,
        n_breaks,
        trim_ratio,
        alpha,
        n_sim,
        grid_size,
    )
