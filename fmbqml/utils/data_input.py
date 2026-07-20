"""Input conversion helpers for panel-data interfaces."""

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd


def prepare_input_data(
        X: Union[np.ndarray, pd.DataFrame]
) -> Tuple[np.ndarray, Union[np.ndarray, pd.DataFrame], Optional[Any]]:
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
    if isinstance(X, pd.DataFrame):
        data_array = X.to_numpy(dtype=float)
        original_data = X.copy()
        row_index = X.index.copy()
    else:
        data_array = np.asarray(X, dtype=float)
        original_data = data_array.copy()
        row_index = None

    return data_array, original_data, row_index

def prepare_multi_input_data(
        X: Union[np.ndarray, pd.DataFrame]
) -> Tuple[
    np.ndarray,
    Union[np.ndarray, pd.DataFrame],
    Optional[Any],
    Optional[Any],
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
    if isinstance(X, pd.DataFrame):
        data_array = X.to_numpy(dtype=float)
        original_data = X.copy()
        row_index = X.index.copy()
        columns = X.columns.copy()
        is_dataframe = True
    else:
        data_array = np.asarray(X, dtype=float)
        original_data = data_array.copy()
        row_index = None
        columns = None
        is_dataframe = False

    return data_array, original_data, row_index, columns, is_dataframe
