import numpy as np


def remove_outliers(X):
    """
    Python version of remove_outliers.m

    Parameters
    ----------
    X : array-like, shape (T, N)
        Dataset, one series per column.

    Returns
    -------
    Y : ndarray, shape (T, N)
        Dataset with outliers replaced by np.nan.

    n : ndarray, shape (N,)
        Number of outliers found in each series.
    """

    X = np.asarray(X, dtype=float)

    # Calculate median of each series, ignoring NaN
    median_X = np.nanmedian(X, axis=0)

    # Repeat median over all rows
    median_X_mat = np.tile(median_X, (X.shape[0], 1))

    # Calculate quartiles, ignoring NaN
    Q = np.nanpercentile(X, [25, 50, 75], axis=0)

    # Interquartile range
    IQR = Q[2, :] - Q[0, :]

    # Repeat IQR over all rows
    IQR_mat = np.tile(IQR, (X.shape[0], 1))

    # Determine outliers
    Z = np.abs(X - median_X_mat)
    outlier = Z > (10.0 * IQR_mat)

    # Replace outliers with NaN
    Y = X.copy()
    Y[outlier] = np.nan

    # Count number of outliers in each series
    n = np.sum(outlier, axis=0)

    return Y, n