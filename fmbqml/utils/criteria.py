"""Information-criterion names and normalization helpers."""

from numbers import Integral
from typing import Union


CRITERION_NAMES = {
    0: "IC1",
    1: "IC2",
    2: "IC3",
    3: "PC1",
    4: "PC2",
    5: "PC3",
    6: "AIC3",
    7: "BIC3",
}

CRITERION_INDEX = {v: k for k, v in CRITERION_NAMES.items()}

VALID_CRITERION_NAMES = set(CRITERION_INDEX.keys())


def normalize_factor_criterion(criterion: Union[int, str]) -> int:
    """
    Convert a factor-selection criterion name or index to an integer index.

    Parameters
    ----------
    criterion : int or str
        Factor-selection criterion. It can be specified as an integer in
        {0,1,2,3,4,5,6,7} or as one of
        {"IC1", "IC2", "IC3", "PC1", "PC2", "PC3", "AIC3", "BIC3"}.

    Returns
    -------
    int
        Integer index of the criterion.

    Raises
    ------
    ValueError
        If the criterion is invalid.
    """
    if isinstance(criterion, bool):
        raise ValueError(_criterion_error_message(criterion))

    if isinstance(criterion, Integral):
        criterion = int(criterion)
        if criterion in CRITERION_NAMES:
            return criterion
        raise ValueError(_criterion_error_message(criterion))

    if isinstance(criterion, str):
        criterion_upper = criterion.upper()
        if criterion_upper in CRITERION_INDEX:
            return CRITERION_INDEX[criterion_upper]
        raise ValueError(_criterion_error_message(criterion))

    raise ValueError(_criterion_error_message(criterion))


def get_criterion_name(criterion: Union[int, str]) -> str:
    """
    Return the string name of a factor-selection criterion.

    Examples
    --------
    >>> get_criterion_name(0)
    'IC1'
    >>> get_criterion_name("IC2")
    'IC2'
    """
    criterion_index = normalize_factor_criterion(criterion)
    return CRITERION_NAMES[criterion_index]


def _criterion_error_message(criterion) -> str:
    return (
        "Parameter 'factor_criterion' must be one of "
        "{'IC1','IC2','IC3','PC1','PC2','PC3','AIC3','BIC3'} "
        "or an integer in {0,1,2,3,4,5,6,7}. "
        f"Got {criterion}."
    )
