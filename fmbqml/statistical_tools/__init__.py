"""Statistical tools shared by single- and multiple-break procedures."""

from .brownian import simulate_brownian_motion
from .critical_values import (
    bridge_score_matrix,
    critical_value_lr_hac_m,
    factor_long_run_covariance,
    factor_product_deviations,
    joint_sup_lr_cv,
    select_critical_value,
    simulate_single_sup_lr_critical_values,
)
from .hac import hac_newey_west_94

__all__ = [
    "hac_newey_west_94",
    "bridge_score_matrix",
    "critical_value_lr_hac_m",
    "factor_long_run_covariance",
    "factor_product_deviations",
    "joint_sup_lr_cv",
    "select_critical_value",
    "simulate_single_sup_lr_critical_values",
    "simulate_brownian_motion",
]
