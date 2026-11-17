"""Cholesky log-determinants with bounded diagonal-jitter fallback."""

from numbers import Real

import numpy as np


def safe_logdet(
        A,
        *,
        symmetry_rtol: float = 1e-10,
        symmetry_atol: float = 0.0,
) -> float:
    """Compute a symmetric matrix log-determinant using Cholesky factorization.

    The unmodified matrix is attempted first. Only if that factorization fails,
    retry with diagonal jitter of 1e-8, 1e-7, and 1e-6, each added to the
    ORIGINAL matrix. A fallback result is the log determinant of A + jitter*I,
    not of A itself. The jitter bounds are fixed numerical implementation
    settings; they are not estimated statistical parameters.

    Parameters
    ----------
    A : array-like
        Non-empty square real-valued matrix.
    symmetry_rtol : float, default=1e-10
        Relative tolerance for floating-point asymmetry.
    symmetry_atol : float, default=0.0
        Absolute tolerance for floating-point asymmetry.

    Returns
    -------
    float
        Natural log determinant of A, or of the jittered matrix on fallback.

    Raises
    ------
    TypeError
        If the input cannot be converted to a real numeric array.
    ValueError
        If the input is complex, non-square, empty, non-finite, materially
        non-symmetric, or a symmetry tolerance is invalid.
    numpy.linalg.LinAlgError
        If Cholesky fails even after the bounded jitter attempts.
    """
    for tolerance in (symmetry_rtol, symmetry_atol):
        if (isinstance(tolerance, (bool, np.bool_))
                or not isinstance(tolerance, Real)
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Symmetry tolerances must be finite non-negative real scalars.")

    try:
        raw = np.asarray(A)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("A must be convertible to a real numeric array.") from exc

    contains_complex = np.iscomplexobj(raw)
    if not contains_complex and raw.dtype.kind == "O":
        contains_complex = any(np.iscomplexobj(value) for value in raw.flat)
    if contains_complex:
        raise ValueError("A must be real-valued; complex input is not supported.")

    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("A must be convertible to a real numeric array.") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("A must be a non-empty square matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("A contains NaN or Inf.")

    scale = float(np.max(np.abs(matrix)))
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    symmetry_limit = symmetry_atol + symmetry_rtol * scale
    if asymmetry > symmetry_limit:
        raise ValueError(
            "A must be symmetric within the specified numerical tolerance."
        )

    matrix = 0.5 * (matrix + matrix.T)

    identity = np.eye(matrix.shape[0], dtype=float)
    for jitter in (0.0, 1e-8, 1e-7, 1e-6):
        candidate = matrix if jitter == 0.0 else matrix + jitter * identity
        try:
            lower = np.linalg.cholesky(candidate)
        except np.linalg.LinAlgError:
            continue

        logdet = 2.0 * np.sum(np.log(np.diag(lower)))
        if np.isfinite(logdet):
            return float(logdet)

    raise np.linalg.LinAlgError(
        "Cholesky log determinant failed with or without bounded diagonal jitter."
    )


__all__ = ["safe_logdet"]
