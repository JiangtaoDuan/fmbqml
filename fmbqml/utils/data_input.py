"""Input conversion helpers for panel-data interfaces."""

from typing import Any

import numpy as np
import pandas as pd


def validate_real_input(X, name: str = "X") -> np.ndarray:
    """Convert input to a finite real-valued float array without discarding data.

    Complex-valued inputs are rejected before conversion. This check also covers
    object arrays containing Python or NumPy complex scalars, for which
    ``np.iscomplexobj`` on the array itself is not sufficient.
    """
    try:
        raw = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be convertible to a real numeric array."
        ) from exc

    contains_complex = np.iscomplexobj(raw)
    if not contains_complex and raw.dtype.kind == "O":
        contains_complex = any(np.iscomplexobj(value) for value in raw.flat)

    if contains_complex:
        raise ValueError(
            f"{name} must be real-valued; complex input is not supported."
        )

    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be convertible to a real numeric array."
        ) from exc

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")

    return array


def _validate_panel_data(X, name: str = "X") -> np.ndarray:
    """Validate the panel once at the model/input-preparation boundary."""
    array = validate_real_input(X, name=name)
    if array.ndim != 2:
        raise ValueError(
            f"Input {name} must be a 2D array (T x N). Got {array.ndim}D array."
        )
    T, N = array.shape
    if T < 2:
        raise ValueError(f"Sample size T must be at least 2. Got T={T}.")
    if N < 1:
        raise ValueError(f"Number of variables N must be at least 1. Got N={N}.")
    return array



def _validate_break_panel(X) -> np.ndarray:
    """Validate panel shape and variation once when data enter a model."""
    array = _validate_panel_data(X)
    if np.all(array == array[0]):
        raise ValueError("Input X must have time variation for break estimation.")
    return array

def prepare_input_data(
        X: np.ndarray | pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | pd.DataFrame, Any | None]:
    """
    Convert input data to a numerical numpy array and preserve original metadata.

    Parameters
    ----------
    X : np.ndarray or pandas.DataFrame
        Input data matrix.

    Returns
    -------
    data_array : np.ndarray
        Numerical data matrix.
    original_data : np.ndarray or pandas.DataFrame
        Copy of the original input data.
    row_index : pandas.Index or None
        Original row index if X is a DataFrame; otherwise None.
    """
    data_array = _validate_break_panel(X)
    if isinstance(X, pd.DataFrame):
        original_data = X.copy()
        row_index = X.index.copy()
    else:
        original_data = data_array.copy()
        row_index = None

    return data_array, original_data, row_index

def prepare_multi_input_data(
        X: np.ndarray | pd.DataFrame
) -> tuple[
    np.ndarray,
    np.ndarray | pd.DataFrame,
    Any | None,
    Any | None,
    bool,
]:
    """Convert multiple-break input data and preserve tabular metadata.

    Parameters
    ----------
    X : ndarray or pandas.DataFrame
        Input panel with observations in rows and variables in columns.

    Returns
    -------
    data_array : ndarray
        Numerical representation of the panel.
    original_data : ndarray or pandas.DataFrame
        Copy of the input in its original container type.
    row_index : pandas.Index or None
        Original row labels when ``X`` is a DataFrame.
    columns : pandas.Index or None
        Original column labels when ``X`` is a DataFrame.
    is_dataframe : bool
        Whether the supplied input was a DataFrame.
    """
    data_array = _validate_break_panel(X)
    if isinstance(X, pd.DataFrame):
        original_data = X.copy()
        row_index = X.index.copy()
        columns = X.columns.copy()
        is_dataframe = True
    else:
        original_data = data_array.copy()
        row_index = None
        columns = None
        is_dataframe = False

    return data_array, original_data, row_index, columns, is_dataframe
