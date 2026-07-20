

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
    format_single_break_summary,
    format_multi_break_summary,
    format_ic_path_table,
)

from .plotting import (
    plot_joint_lr_profile,
    plot_lr_profile,
    plot_qml_profile,
    plot_mqml_profile,
)

from .data_input import (
    prepare_input_data,
    prepare_multi_input_data,
)

from .safe_logdet import safe_logdet
from .partition import optimize_additive_partition

from .validation import (
    validate_alpha,
    validate_input_matrix,
    validate_joint_sup_lr_arguments,
    validate_max_factors,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_trim_ratio,
)

__all__ = [
    "CRITERION_NAMES",
    "normalize_factor_criterion",

    "calculate_break_range",
    "compute_min_segment_length",
    "map_breakpoint_to_label",
    "map_breakpoints_to_labels",

    "format_single_break_summary",
    "format_multi_break_summary",
    "format_ic_path_table",

    "plot_joint_lr_profile",
    "plot_lr_profile",
    "plot_qml_profile",
    "plot_mqml_profile",

    "prepare_input_data",
    "prepare_multi_input_data",

    "safe_logdet",
    "optimize_additive_partition",

    "validate_alpha",
    "validate_input_matrix",
    "validate_joint_sup_lr_arguments",
    "validate_max_factors",
    "validate_nonnegative_integer",
    "validate_positive_integer",
    "validate_trim_ratio",
]
