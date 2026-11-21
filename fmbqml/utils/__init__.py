

"""Shared validation, formatting, plotting, and numerical utilities."""

from .breakpoint_utils import (
    calculate_break_range,
    compute_min_segment_length,
    map_breakpoint_to_label,
    map_breakpoints_to_labels,
)
from .criteria import (
    CRITERION_NAMES,
    normalize_factor_criterion,
)
from .data_input import (
    prepare_input_data,
    prepare_multi_input_data,
    validate_real_input,
)
from .formatting import (
    Summary,
    format_multi_break_result_summary,
    format_single_break_result_summary,
)
from .partition import optimize_additive_partition
from .plotting import (
    plot_lr_profile,
    plot_mqml_profile,
    plot_qml_profile,
)
from .safe_logdet import safe_logdet
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
    "Summary",
    "calculate_break_range",
    "compute_min_segment_length",
    "format_multi_break_result_summary",
    "format_single_break_result_summary",
    "map_breakpoint_to_label",
    "map_breakpoints_to_labels",
    "normalize_factor_criterion",
    "optimize_additive_partition",
    "plot_lr_profile",
    "plot_mqml_profile",
    "plot_qml_profile",
    "prepare_input_data",
    "prepare_multi_input_data",
    "safe_logdet",
    "validate_alpha",
    "validate_boolean",
    "validate_max_factors",
    "validate_nonnegative_integer",
    "validate_positive_integer",
    "validate_random_state",
    "validate_real_input",
    "validate_trim_ratio",
]
