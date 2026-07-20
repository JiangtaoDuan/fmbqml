import numpy as np


def factors_em(x, kmax, jj, DEMEAN):
    """
    Python version of factors_em.m

    Parameters
    ----------
    x : array-like, shape (T, N)
        Dataset, one series per column.

    kmax : int
        Maximum number of factors to be estimated.
        If kmax == 99, number of factors is forced to 8.

    jj : int
        Information criterion:
            1 -> PC_p1
            2 -> PC_p2
            3 -> PC_p3

    DEMEAN : int
        Transformation type:
            0 -> no transformation
            1 -> demean only
            2 -> demean and standardize
            3 -> recursively demean and then standardize

    Returns
    -------
    ehat : ndarray
        Difference between x and values predicted by factors.

    Fhat : ndarray
        Estimated factors.

    lamhat : ndarray
        Factor loadings.

    ve2 : ndarray
        Eigenvalues of x3' x3.

    x2 : ndarray
        Original x with missing values replaced by EM algorithm.
    """

    x = np.asarray(x, dtype=float)

    # =========================================================================
    # PART 1: CHECKS
    # =========================================================================

    if np.any(np.sum(np.isnan(x), axis=1) == x.shape[1]):
        raise ValueError("Input x contains entire row of missing values.")

    if np.any(np.sum(np.isnan(x), axis=0) == x.shape[0]):
        raise ValueError("Input x contains entire column of missing values.")

    if not ((kmax <= x.shape[1] and kmax >= 1 and int(kmax) == kmax) or kmax == 99):
        raise ValueError("Input kmax is specified incorrectly.")

    if jj not in (1, 2, 3):
        raise ValueError("Input jj is specified incorrectly.")

    if DEMEAN not in (0, 1, 2, 3):
        raise ValueError("Input DEMEAN is specified incorrectly.")

    # =========================================================================
    # PART 2: SETUP
    # =========================================================================

    maxit = 50

    T = x.shape[0]
    N = x.shape[1]

    err = 999.0
    it = 0

    x1 = np.isnan(x)

    # =========================================================================
    # PART 3: INITIALIZE EM ALGORITHM
    # =========================================================================

    mut0 = np.tile(np.nanmean(x, axis=0), (T, 1))

    x2 = x.copy()
    x2[np.isnan(x)] = mut0[np.isnan(x)]

    x3, mut, sdt = transform_data(x2, DEMEAN)

    if kmax != 99:
        icstar, _, _, _ = baing(x3, kmax, jj)
    else:
        icstar = 8

    chat, Fhat, lamhat, ve2 = pc2(x3, icstar)

    chat0 = chat.copy()

    # =========================================================================
    # PART 4: PERFORM EM ALGORITHM
    # =========================================================================

    while err > 0.000001 and it < maxit:

        it += 1

        print(f"Iteration {it}: obj {err:10f} IC {icstar}")

        # ---------------------------------------------------------------------
        # UPDATE MISSING VALUES
        # ---------------------------------------------------------------------

        for t in range(T):
            for j in range(N):
                if x1[t, j]:
                    x2[t, j] = chat[t, j] * sdt[t, j] + mut[t, j]
                else:
                    x2[t, j] = x[t, j]

        # ---------------------------------------------------------------------
        # ESTIMATE FACTORS
        # ---------------------------------------------------------------------

        x3, mut, sdt = transform_data(x2, DEMEAN)

        if kmax != 99:
            icstar, _, _, _ = baing(x3, kmax, jj)
        else:
            icstar = 8

        chat, Fhat, lamhat, ve2 = pc2(x3, icstar)

        # ---------------------------------------------------------------------
        # CALCULATE NEW ERROR VALUE
        # ---------------------------------------------------------------------

        diff = chat - chat0

        v1 = diff.reshape(-1, 1)
        v2 = chat0.reshape(-1, 1)

        denominator = float(v2.T @ v2)

        if denominator == 0:
            denominator = np.finfo(float).eps

        err = float((v1.T @ v1) / denominator)

        chat0 = chat.copy()

    if it == maxit:
        print("Warning: Maximum number of iterations reached in EM algorithm")

    # -------------------------------------------------------------------------
    # FINAL DIFFERENCE
    # -------------------------------------------------------------------------

    ehat = x - chat * sdt - mut

    return ehat, Fhat, lamhat, ve2, x2


def baing(X, kmax, jj):
    """
    Python version of baing() subfunction in factors_em.m.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Dataset, one series per column.

    kmax : int
        Maximum number of factors.

    jj : int
        Information criterion:
            1 -> PC_p1
            2 -> PC_p2
            3 -> PC_p3

    Returns
    -------
    ic1 : int
        Selected number of factors.

    chat : ndarray
        Values of X predicted by factors.

    Fhat : ndarray
        Factors.

    eigval : ndarray
        Eigenvalues of X'X or XX'.
    """

    X = np.asarray(X, dtype=float)

    T = X.shape[0]
    N = X.shape[1]

    NT = N * T
    NT1 = N + T

    CT = np.zeros(kmax)

    ii = np.arange(1, kmax + 1)

    GCT = min(N, T)

    if jj == 1:
        CT[:] = np.log(NT / NT1) * ii * NT1 / NT

    elif jj == 2:
        CT[:] = (NT1 / NT) * np.log(min(N, T)) * ii

    elif jj == 3:
        CT[:] = ii * np.log(GCT) / GCT

    else:
        raise ValueError("Input jj is specified incorrectly.")

    # -------------------------------------------------------------------------
    # RUN PRINCIPAL COMPONENT ANALYSIS
    # -------------------------------------------------------------------------

    if T < N:
        ev, eigval, _ = np.linalg.svd(X @ X.T, full_matrices=True)

        Fhat0 = np.sqrt(T) * ev

        Lambda0 = X.T @ Fhat0 / T

    else:
        ev, eigval, _ = np.linalg.svd(X.T @ X, full_matrices=True)

        Lambda0 = np.sqrt(N) * ev

        Fhat0 = X @ Lambda0 / N

    # -------------------------------------------------------------------------
    # SELECT NUMBER OF FACTORS
    # -------------------------------------------------------------------------

    Sigma = np.zeros(kmax + 1)
    IC1 = np.zeros(kmax + 1)

    for i in range(kmax, 0, -1):

        Fhat_i = Fhat0[:, :i]

        lambda_i = Lambda0[:, :i]

        chat_i = Fhat_i @ lambda_i.T

        ehat_i = X - chat_i

        Sigma[i - 1] = np.mean(np.sum(ehat_i * ehat_i / T, axis=0))

        IC1[i - 1] = np.log(Sigma[i - 1]) + CT[i - 1]

    Sigma[kmax] = np.mean(np.sum(X * X / T, axis=0))

    IC1[kmax] = np.log(Sigma[kmax])

    ic_pos = minindc(IC1.reshape(-1, 1))[0]

    ic1 = ic_pos

    if ic1 > kmax:
        ic1 = 0

    # -------------------------------------------------------------------------
    # SAVE OTHER OUTPUT
    # -------------------------------------------------------------------------

    Fhat = Fhat0[:, :kmax]

    Lambda = Lambda0[:, :kmax]

    chat = Fhat @ Lambda.T

    return int(ic1), chat, Fhat, eigval


def pc2(X, nfac):
    """
    Python version of pc2() subfunction in factors_em.m.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Dataset, one series per column.

    nfac : int
        Number of factors to be selected.

    Returns
    -------
    chat : ndarray
        Values of X predicted by factors.

    fhat : ndarray
        Factors scaled by 1 / sqrt(N).

    lambda_hat : ndarray
        Factor loadings scaled by sqrt(N).

    ss : ndarray
        Eigenvalues of X'X.
    """

    X = np.asarray(X, dtype=float)

    N = X.shape[1]

    U, S, _ = np.linalg.svd(X.T @ X, full_matrices=True)

    if nfac == 0:
        lambda_hat = np.empty((N, 0))
        fhat = np.empty((X.shape[0], 0))
        chat = np.zeros_like(X)
    else:
        lambda_hat = U[:, :nfac] * np.sqrt(N)

        fhat = X @ lambda_hat / N

        chat = fhat @ lambda_hat.T

    ss = S

    return chat, fhat, lambda_hat, ss


def minindc(x):
    """
    Python version of minindc() subfunction in factors_em.m.

    This returns MATLAB-style 1-based row indices.
    """

    x = np.asarray(x, dtype=float)

    nrows = x.shape[0]
    ncols = x.shape[1]

    pos = np.zeros(ncols, dtype=int)

    seq = np.arange(1, nrows + 1)

    for i in range(ncols):

        min_i = np.min(x[:, i])

        colmin_i = seq * ((x[:, i] - min_i) == 0)

        if np.sum(colmin_i > 0) > 1:
            raise ValueError("Minimum value occurs more than once.")

        pos[i] = int(np.sum(colmin_i))

    return pos


def transform_data(x2, DEMEAN):
    """
    Python version of transform_data() subfunction in factors_em.m.

    Parameters
    ----------
    x2 : ndarray, shape (T, N)
        Dataset without missing values.

    DEMEAN : int
        0 -> no transformation
        1 -> demean only
        2 -> demean and standardize
        3 -> recursively demean and then standardize

    Returns
    -------
    x22 : ndarray
        Transformed dataset.

    mut : ndarray
        Matrix containing values subtracted from x2.

    sdt : ndarray
        Matrix containing values by which x2 was divided.
    """

    x2 = np.asarray(x2, dtype=float)

    T = x2.shape[0]
    N = x2.shape[1]

    if DEMEAN == 0:

        mut = np.tile(np.zeros(N), (T, 1))

        sdt = np.tile(np.ones(N), (T, 1))

        x22 = x2.copy()

    elif DEMEAN == 1:

        mut = np.tile(np.mean(x2, axis=0), (T, 1))

        sdt = np.tile(np.ones(N), (T, 1))

        x22 = x2 - mut

    elif DEMEAN == 2:

        mut = np.tile(np.mean(x2, axis=0), (T, 1))

        sdt = np.tile(np.std(x2, axis=0, ddof=1), (T, 1))

        x22 = (x2 - mut) / sdt

    elif DEMEAN == 3:

        mut = np.full_like(x2, np.nan)

        for t in range(T):
            mut[t, :] = np.mean(x2[:t + 1, :], axis=0)

        sdt = np.tile(np.std(x2, axis=0, ddof=1), (T, 1))

        x22 = (x2 - mut) / sdt

    else:
        raise ValueError("Input DEMEAN is specified incorrectly.")

    return x22, mut, sdt