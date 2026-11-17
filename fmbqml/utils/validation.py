
"""Argument validation shared by public estimators and tests."""

from numbers import Integral, Real
import numpy as np




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



def validate_boolean(value, name: str) -> bool:
    """Validate a public Boolean option and normalize NumPy Boolean scalars."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a Boolean (True or False). Got {value!r}.")
    return bool(value)


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


def validate_random_state(random_state):
    """Validate a public random seed and normalize NumPy integer scalars.

    The public API accepts a non-negative integer or ``None``. Boolean values
    are rejected explicitly because ``bool`` is a subclass of ``int``.
    """
    if random_state is None:
        return None

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError(
            "random_state must be a non-negative integer or None. "
            f"Got random_state={random_state}."
        )

    return int(random_state)

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
