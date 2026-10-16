import numpy as np


def prepare_missing(rawdata, tcode):
    """
    Python version of prepare_missing.m

    Parameters
    ----------
    rawdata : array-like, shape (T, N)
        Raw data matrix. Each column is one time series.

    tcode : array-like, shape (N,)
        Transformation code for each series.

    Returns
    -------
    yt : ndarray, shape (T, N)
        Transformed data matrix.
    """

    rawdata = np.asarray(rawdata, dtype=float)
    tcode = np.asarray(tcode).astype(int).ravel()

    T, N = rawdata.shape
    yt = np.full((T, N), np.nan, dtype=float)

    for i in range(N):
        yt[:, i] = transxf(rawdata[:, i], tcode[i])

    return yt


def transxf(x, tcode):
    """
    Python version of MATLAB subfunction transxf.
    """

    x = np.asarray(x, dtype=float).ravel()

    n = x.shape[0]
    small = 1e-6

    y = np.full(n, np.nan, dtype=float)

    if tcode == 1:
        # Level: x(t)
        y = x.copy()

    elif tcode == 2:
        # First difference: x(t)-x(t-1)
        y[1:n] = x[1:n] - x[0:n - 1]

    elif tcode == 3:
        # Second difference
        y[2:n] = x[2:n] - 2.0 * x[1:n - 1] + x[0:n - 2]

    elif tcode == 4:
        # Natural log: ln(x)
        if np.all(np.isnan(x)):
            y[:] = np.nan
        elif np.nanmin(x) < small:
            y[:] = np.nan
        else:
            y = np.log(x)

    elif tcode == 5:
        # First difference of natural log
        if np.all(np.isnan(x)):
            y[:] = np.nan
        elif np.nanmin(x) > small:
            lx = np.log(x)
            y[1:n] = lx[1:n] - lx[0:n - 1]

    elif tcode == 6:
        # Second difference of natural log
        if np.all(np.isnan(x)):
            y[:] = np.nan
        elif np.nanmin(x) > small:
            lx = np.log(x)
            y[2:n] = lx[2:n] - 2.0 * lx[1:n - 1] + lx[0:n - 2]

    elif tcode == 7:
        # First difference of percent change
        y1 = np.full(n, np.nan, dtype=float)
        y1[1:n] = (x[1:n] - x[0:n - 1]) / x[0:n - 1]
        y[2:n] = y1[2:n] - y1[1:n - 1]

    else:
        raise ValueError(f"Unknown transformation code: {tcode}")

    y[~np.isfinite(y)] = np.nan

    return y