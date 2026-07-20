"""Formatting helpers for user-facing estimation and test summaries."""

from typing import Any, Dict, Optional

import numpy as np

from fmbqml.utils.criteria import CRITERION_NAMES


def _format_float(value: Any, digits: int = 4) -> str:
    """
    Safely format a numeric value.
    """
    if value is None:
        return "None"

    try:
        value = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(value):
        return "nan"

    return f"{value:.{digits}f}"


def _get_list_item(values: Any, index: int, default: Any = "") -> Any:
    """
    Safely get one item from a list-like object.
    """
    if values is None:
        return default

    try:
        if len(values) > index:
            value = values[index]
            return default if value is None else value
    except Exception:
        pass

    return default


def _get_dict_value(data: Any, keys, default: Any = "") -> Any:
    """
    Safely get a value from a dictionary using several possible keys.
    """
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def _safe_min_text(a: Any, b: Any) -> str:
    """
    Return min(a, b) as text when possible.
    """
    try:
        return str(min(int(a), int(b)))
    except Exception:
        return ""


def _format_breakpoints(x: Any) -> str:
    """
    Format breakpoint candidates in a compact readable form.
    """
    if x is None:
        return "[]"

    if isinstance(x, str):
        if x.lower() == "no breakpoint":
            return "[]"
        return x

    try:
        arr = np.asarray(x).reshape(-1)
    except Exception:
        return str(x)

    if arr.size == 0:
        return "[]"

    out = []

    for v in arr:
        try:
            if np.isfinite(v):
                out.append(str(int(round(float(v)))))
        except Exception:
            pass

    return "[" + ", ".join(out) + "]" if out else "[]"


def _format_break_types(result: Dict[str, Any]) -> Any:
    break_types = result.get("break_types")

    if break_types:
        return break_types

    break_type = result.get("break_type")
    if break_type:
        return break_type

    return None


def _format_grid_text(criteria: Any) -> str:
    """
    Format merged search grid information when available.
    """
    grid = _get_dict_value(
        criteria,
        keys=[
            "merged_search_grid",
            "search_grid",
            "grid",
            "candidate_grid",
        ],
        default=None
    )

    if grid is None:
        return ""

    return f"; merged search grid={grid}."


def _format_condition_text(
        break_type: Any,
        r_left: Any,
        r_right: Any,
        r_combined: Any,
        criteria: Any
) -> str:
    """
    Format breakpoint-type classification condition.
    """
    condition = _get_dict_value(
        criteria,
        keys=[
            "condition",
            "classification_condition",
            "decision_rule",
            "rule",
        ],
        default=None
    )

    if condition is not None:
        return str(condition) + _format_grid_text(criteria)

    min_text = _safe_min_text(r_left, r_right)

    if str(break_type).startswith("Type 1"):
        if min_text:
            return (
                f"r_j,j+1 ({r_combined}) > "
                f"min(r_j,r_j+1) ({min_text})"
                + _format_grid_text(criteria)
            )
        return "Type 1 condition satisfied" + _format_grid_text(criteria)

    if str(break_type).startswith("Type 2"):
        if min_text:
            return (
                f"r_j,j+1 ({r_combined}) <= "
                f"min(r_j,r_j+1) ({min_text})"
                + _format_grid_text(criteria)
            )
        return "Type 2 condition satisfied" + _format_grid_text(criteria)

    return "No classification condition available" + _format_grid_text(criteria)


def format_ic_path_table(ic_path: Any) -> str:
    """
    Format the information criterion path as readable text.
    """
    if ic_path is None or len(ic_path) == 0:
        return "No information criterion path available."

    lines = []

    for row in ic_path:
        m = int(row.get("m", 0))

        loss = row.get("loss", np.nan)
        ic = row.get("IC", np.nan)
        breaks = row.get("estimated_breakpoints", [])

        loss_str = _format_float(loss)
        ic_str = _format_float(ic)
        breaks_str = _format_breakpoints(breaks)

        lines.append(
            f"m={m}: U_NT={loss_str}, IC={ic_str}, breakpoints={breaks_str}"
        )

    return "\n".join(lines)


def format_break_classification_table(
        result: Dict[str, Any],
        mode: str = "joint"
) -> Optional[str]:
    """
    Format breakpoint classification results as a console-style table.
    """
    break_types = result.get("break_types", [])

    if not break_types:
        return None

    breakpoints = result.get("breakpoints", None)

    if breakpoints is None:
        break_point = result.get("break_point", None)
        breakpoints = [] if break_point is None else [break_point]

    labels = result.get("break_labels", None)

    if labels is None:
        break_label = result.get("break_label", None)
        labels = [] if break_label is None else [break_label]

    regime_factors = result.get("regime_factors", [])
    combined_factors = result.get("combined_factors", [])
    criteria_list = result.get("classification_criteria", [])

    if mode == "joint":
        title = "JOINT BREAKPOINT TYPE CLASSIFICATION"
    elif mode == "joint_lr":
        title = "JOINT LR BREAKPOINT TYPE CLASSIFICATION"
    elif mode == "single":
        title = "SINGLE BREAKPOINT TYPE CLASSIFICATION"
    else:
        title = "BREAKPOINT TYPE CLASSIFICATION"

    sep = "-" * 112

    break_w = 14
    loc_w = 10
    type_w = 23
    r_left_w = 7
    r_right_w = 8
    r_combined_w = 10
    condition_w = 55
    label_w = 14

    lines = [
        sep,
        title,
        sep,
        (
            f"{'Break':<{break_w}}"
            f"{'Location':<{loc_w}}"
            f"{'Type':<{type_w}}"
            f"{'r_j':<{r_left_w}}"
            f"{'r_j+1':<{r_right_w}}"
            f"{'r_j,j+1':<{r_combined_w}}"
            f"{'Condition':<{condition_w}}"
            f"{'Label':<{label_w}}"
        )
    ]

    n_breaks = len(break_types)

    for i in range(n_breaks):
        bp = _get_list_item(breakpoints, i, "")
        btype = _get_list_item(break_types, i, "")
        label = _get_list_item(labels, i, "")

        criteria = _get_list_item(criteria_list, i, {})

        r_left = _get_dict_value(
            criteria,
            keys=[
                "r_j",
                "r_left",
                "left_factor",
                "left_factors",
                "r_before",
            ],
            default=_get_list_item(regime_factors, i, "")
        )

        r_right = _get_dict_value(
            criteria,
            keys=[
                "r_j_plus_1",
                "r_right",
                "right_factor",
                "right_factors",
                "r_after",
            ],
            default=_get_list_item(regime_factors, i + 1, "")
        )

        r_combined = _get_dict_value(
            criteria,
            keys=[
                "r_j_j_plus_1",
                "r_combined",
                "combined_factor",
                "combined_factors",
                "r_merged",
            ],
            default=_get_list_item(combined_factors, i, "")
        )

        condition = _format_condition_text(
            break_type=btype,
            r_left=r_left,
            r_right=r_right,
            r_combined=r_combined,
            criteria=criteria
        )

        lines.append(
            f"{'Breakpoint ' + str(i + 1):<{break_w}}"
            f"{str(bp):<{loc_w}}"
            f"{str(btype):<{type_w}}"
            f"{str(r_left):<{r_left_w}}"
            f"{str(r_right):<{r_right_w}}"
            f"{str(r_combined):<{r_combined_w}}"
            f"{str(condition):<{condition_w}}"
            f"{str(label):<{label_w}}"
        )

    lines.append(sep)

    if result.get("break_type_summary") is not None:
        summary = result["break_type_summary"]
        lines.append(
            "Break type summary: "
            f"Type 1 = {summary.get('type1_count', 0)}, "
            f"Type 2 = {summary.get('type2_count', 0)}, "
            f"Other = {summary.get('other_count', 0)}"
        )
        lines.append(sep)

    return "\n".join(lines)


def format_single_break_summary(result: Dict[str, Any]) -> str:
    """
    Format summary text for SingleBreakQML results.
    """
    test_type = result.get("test_type", "")

    if test_type == "lr_test":
        title = "SingleBreakQML LR TEST SUMMARY"
    elif test_type == "estimate_breakpoint":
        title = "SingleBreakQML BREAKPOINT ESTIMATION SUMMARY"
    else:
        title = "SingleBreakQML SUMMARY"

    criterion_name = CRITERION_NAMES.get(
        result.get("factor_criterion", 0),
        "IC1"
    )

    lines = [
        "=" * 70,
        title,
        "=" * 70,
        f"Test type: {result.get('test_type')}",
        f"Sample size: T = {result.get('T')}, N = {result.get('N')}",
    ]

    if result.get("search_range") is not None:
        lines.append(
            f"Search range: {result.get('search_range')}"
        )

    if result.get("estimated_factors") is not None:
        lines.append(
            f"Estimated number of factors: {result.get('estimated_factors')}"
        )

    lines.append(
        f"Factor criterion: {criterion_name}"
    )

    if result.get("estimated_break_candidate") is not None:
        lines.append(
            f"Estimated break candidate: {result.get('estimated_break_candidate')}"
        )

    if result.get("estimated_break_candidate_label") is not None:
        lines.append(
            "Estimated break candidate label: "
            f"{result.get('estimated_break_candidate_label')}"
        )

    lines.append(
        f"Accepted breakpoint: {result.get('break_point')}"
    )

    if result.get("break_label") is not None:
        lines.append(
            f"Accepted breakpoint label: {result.get('break_label')}"
        )

    lines.append(
        f"Break type: {result.get('break_type')}"
    )

    if result.get("reject_null") is not None:
        lines.append(
            f"Reject null hypothesis of no break: {result.get('reject_null')}"
        )

    if result.get("test_statistic") is not None:
        lines.append(
            f"Test statistic: {float(result.get('test_statistic')):.4f}"
        )

    if result.get("critical_value") is not None:
        if result.get("significance_level") is not None:
            lines.append(
                "Critical value "
                f"(alpha = {result.get('significance_level')}): "
                f"{float(result.get('critical_value')):.4f}"
            )
        else:
            lines.append(
                f"Critical value: {float(result.get('critical_value')):.4f}"
            )

    if result.get("min_regime_length") is not None:
        lines.append(
            f"Classification min regime length: {result.get('min_regime_length')}"
        )

    lines.append("=" * 70)

    classification_table = format_break_classification_table(
        result=result,
        mode="single"
    )

    if classification_table is not None:
        lines.append(classification_table)
        lines.append("=" * 70)

    return "\n".join(lines)


def format_multi_break_summary(
        result: Dict[str, Any],
        mode: str = "joint"
) -> str:
    """
    Format summary text for MultiBreakQML results.
    """
    criterion_name = CRITERION_NAMES.get(
        result.get("factor_criterion", 0),
        "IC1"
    )

    T = result.get("T")
    N = result.get("N")

    if mode == "joint":
        lines = [
            "=" * 70,
            "MultiBreakQML JOINT ESTIMATION SUMMARY",
            "=" * 70,
            f"Test type: {result['test_type']}",
            f"Sample size: T = {T}, N = {N}",
            f"Allowed breaks: max={result['max_break']}, min={result['min_break']}",
            (
                f"Max factors: {result['max_factors']}, "
                f"trim_ratio={result['trim_ratio']} "
                f"(min_segment_length={result['min_segment_length']})"
            ),
            f"Estimated number of factors: {result.get('estimated_factors')}",
            f"Detected breakpoints: {result['breakpoints']}",
            f"Break types: {_format_break_types(result)}",
        ]

        if result.get("break_labels") is not None:
            lines.append(f"Breakpoint labels: {result['break_labels']}")

        lines.extend([
            f"Number of breaks: {result['n_breaks']}",
            f"Factor criterion: {criterion_name}",
        ])

        if "ic_path" in result:
            lines.extend([
                "",
                "Information criterion path:",
                format_ic_path_table(result.get("ic_path")),
                ""
            ])

    elif mode == "joint_lr":
        lines = [
            "=" * 70,
            "MultiBreakQML JOINT SUP-LR TEST SUMMARY",
            "=" * 70,
            f"Test type: {result['test_type']}",
            f"Sample size: T = {T}, N = {N}",
            f"Prespecified number of breaks: {result['n_breaks']}",
        ]

        if result.get("max_break") is not None:
            lines.append(f"Allowed breaks: max={result['max_break']}")

        lines.extend([
            (
                f"Max factors: {result.get('max_factors')}, "
                f"trim_ratio={result['trim_ratio']} "
                f"(min_segment_length={result['min_segment_length']})"
            ),
            f"Estimated number of factors: {result.get('estimated_factors')}",
            f"Factor criterion: {criterion_name}",
            f"Detected breakpoints: {result['breakpoints']}",
            f"Break types: {_format_break_types(result)}",
        ])

        if result.get("break_labels") is not None:
            lines.append(f"Breakpoint labels: {result['break_labels']}")

        lines.extend([
            f"Reject null hypothesis of no break: {result['reject_null']}",
            f"Test statistic: {float(result['test_statistic']):.4f}",
            (
                f"Critical value (alpha = {result['significance_level']}): "
                f"{float(result['critical_value']):.4f}"
            ),
            f"Monte Carlo replications: {result.get('monte_carlo_replications')}",
        ])

        if result.get("random_state") is not None:
            lines.append(f"Random state: {result['random_state']}")

    else:
        raise ValueError("mode must be 'joint' or 'joint_lr'.")

    lines.append("=" * 70)

    classification_table = format_break_classification_table(
        result=result,
        mode=mode
    )

    if classification_table is not None:
        lines.append(classification_table)
        lines.append("=" * 70)

    return "\n".join(lines)
