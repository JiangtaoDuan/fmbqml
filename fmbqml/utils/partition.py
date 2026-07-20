"""Generic dynamic-programming tools for additive partitions."""

from typing import List, Tuple

import numpy as np


def optimize_additive_partition(
        values: np.ndarray,
        n_segments: int,
        min_length: int,
        maximize: bool = False,
) -> Tuple[float, List[int]]:
    """Optimize an additive criterion over exactly ``n_segments`` segments."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("values must be a square matrix.")
    if n_segments < 1 or min_length < 1:
        raise ValueError("n_segments and min_length must be positive integers.")

    final_point = values.shape[0] - 1
    if n_segments * min_length > final_point:
        raise ValueError("No feasible partition exists under min_length.")

    initial = -np.inf if maximize else np.inf
    dynamic = np.full((n_segments + 1, final_point + 1), initial)
    previous = np.full((n_segments + 1, final_point + 1), -1, dtype=int)
    dynamic[0, 0] = 0.0

    for segment in range(1, n_segments + 1):
        earliest_end = segment * min_length
        latest_end = final_point - (n_segments - segment) * min_length
        for end in range(earliest_end, latest_end + 1):
            starts = np.arange(
                (segment - 1) * min_length,
                end - min_length + 1,
            )
            candidates = dynamic[segment - 1, starts] + values[starts, end]
            index = int(
                np.argmax(candidates) if maximize else np.argmin(candidates)
            )
            dynamic[segment, end] = candidates[index]
            previous[segment, end] = starts[index]

    optimum = float(dynamic[n_segments, final_point])
    if not np.isfinite(optimum):
        raise RuntimeError("No feasible partition was found.")

    breakpoints = []
    end = final_point
    for segment in range(n_segments, 0, -1):
        start = int(previous[segment, end])
        if segment > 1:
            breakpoints.append(start)
        end = start
    breakpoints.reverse()
    return optimum, breakpoints


__all__ = ["optimize_additive_partition"]
