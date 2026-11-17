

"""Shared validation, formatting, plotting, and numerical utilities."""

from .criteria import (
    CRITERION_NAMES,
    normalize_factor_criterion,
)

from .breakpoint_utils import (
    calculate_break_range,
    compute_min_segment_length,
    map_breakpoint_to_label,
    map_breakpoints_to_labels,
)

from .formatting import (
    Summary,
    format_single_break_result_summary,
    format_multi_break_result_summary,
)

from .plotting import (
    plot_lr_profile,
    plot_qml_profile,
    plot_mqml_profile,
)

from .data_input import (
    validate_real_input,
    prepare_input_data,
    prepare_multi_input_data,
)

from .safe_logdet import safe_logdet
from .partition import optimize_additive_partition

from .validation import (
    validate_alpha,
    validate_boolean,
    validate_max_factors,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_random_state,
    validate_trim_ratio,
)

__all__ = [
    "CRITERION_NAMES",
    "normalize_factor_criterion",

    "calculate_break_range",
    "compute_min_segment_length",
    "map_breakpoint_to_label",
    "map_breakpoints_to_labels",

    "Summary",
    "format_single_break_result_summary",
    "format_multi_break_result_summary",

    "plot_lr_profile",
    "plot_qml_profile",
    "plot_mqml_profile",

    "validate_real_input",
    "prepare_input_data",
    "prepare_multi_input_data",

    "safe_logdet",
    "optimize_additive_partition",

    "validate_alpha",
    "validate_boolean",
    "validate_max_factors",
    "validate_nonnegative_integer",
    "validate_positive_integer",
    "validate_random_state",
    "validate_trim_ratio",
]
