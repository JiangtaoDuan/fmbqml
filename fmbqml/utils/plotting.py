

"""Plotting helpers for LR and QML result profiles."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _break_label(result):
    """Describe a plotted breakpoint according to the analysis decision."""
    if result.get("test_type") == "lr_test":
        if result.get("reject_null") is not None and bool(result["reject_null"]):
            return "Accepted break"
        return "Candidate break (under H1)"
    return "Estimated break"


def plot_lr_profile(
        result: dict[str, Any],
        true_break: int | None = None,
        figsize: tuple[int, int] = (8, 4),
        save_path: str | None = None,
        show: bool = True
):
    """
    Plot the LR profile for single-break analysis.
    """
    if "candidate_breaks" not in result or "lr_profile" not in result:
        raise ValueError(
            "LR profile is not available. Run lr_test() first."
        )

    candidate_breaks = result["candidate_breaks"]
    lr_profile = result["lr_profile"]

    fig, ax = plt.subplots(figsize=figsize)

    base_color = "blue"

    ax.plot(
        candidate_breaks,
        lr_profile,
        color=base_color,
        label="LR statistic"
    )

    # Read the current result-object field names.
    estimated_break = result.get("candidate_breakpoint")
    if estimated_break is None:
        estimated_break = result.get("breakpoint")

    if estimated_break is not None:
        ax.axvline(
            estimated_break,
            color="red",
            linestyle="--",
            label=_break_label(result)
        )

    if true_break is not None:
        ax.axvline(
            true_break,
            color=base_color,
            linestyle=":",
            linewidth=2,
            label="True break"
        )

    critical_value = result.get("critical_value", None)

    if critical_value is not None:
        ax.axhline(
            critical_value,
            color=base_color,
            linestyle="-.",
            label="Critical value"
        )

    ax.set_xlabel("Candidate break location")
    ax.set_ylabel("LR statistic")
    ax.set_title("LR Profile")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_qml_profile(
        result: dict[str, Any],
        true_break: int | None = None,
        figsize: tuple[int, int] = (8, 4),
        save_path: str | None = None,
        show: bool = True
):
    """
    Plot the QML objective profile for single-break analysis.
    """
    if "candidate_breaks" not in result or "qml_profile" not in result:
        raise ValueError(
            "QML profile is not available. Run lr_test() or "
            "estimate_breakpoint() first."
        )

    candidate_breaks = result["candidate_breaks"]
    qml_profile = result["qml_profile"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        candidate_breaks,
        qml_profile,
        color="blue",
        label="QML objective"
    )

    # Read the current result-object field names.
    estimated_break = result.get("candidate_breakpoint")
    if estimated_break is None:
        estimated_break = result.get("breakpoint")

    if estimated_break is not None:
        ax.axvline(
            estimated_break,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=_break_label(result)
        )

    if true_break is not None:
        ax.axvline(
            true_break,
            color="blue",
            linestyle=":",
            linewidth=2.2,
            label="True break"
        )

    ax.set_xlabel("Candidate break location")
    ax.set_ylabel("QML objective")
    ax.set_title("QML Profile")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def _find_column(df: pd.DataFrame, candidates: list[str]):
    """
    Find a column using a list of candidate names.
    """
    column_map = {str(col).lower(): col for col in df.columns}

    for name in candidates:
        key = name.lower()
        if key in column_map:
            return column_map[key]

    return None


def plot_mqml_profile(
        result: dict[str, Any],
        figsize: tuple[int, int] = (8, 4),
        plot_loss: bool = True,
        save_path: str | None = None,
        show: bool = True
):
    """
    Plot the information criterion path for joint multiple-break estimation.

    If a QML loss column is available, it can also be plotted on a secondary
    vertical axis.
    """
    ic_path = result.get("ic_path", None)

    if ic_path is None:
        raise ValueError(
            "Information criterion path is not available. "
            "Run estimate_breaks_jointly() first."
        )

    ic_df = pd.DataFrame(ic_path)

    if ic_df.empty:
        raise ValueError("Information criterion path is empty.")

    m_col = _find_column(
        ic_df,
        ["m", "n_breaks", "num_breaks", "break_number", "number_of_breaks"]
    )

    ic_col = _find_column(
        ic_df,
        ["IC", "ic", "information_criterion", "criterion"]
    )

    loss_col = _find_column(
        ic_df,
        ["loss", "Loss", "qml_loss", "mqml_loss", "objective", "qml_objective"]
    )

    numeric_cols = [
        col for col in ic_df.columns
        if pd.api.types.is_numeric_dtype(ic_df[col])
    ]

    if m_col is None:
        if len(numeric_cols) >= 1:
            m_col = numeric_cols[0]
        else:
            ic_df["_m_index"] = np.arange(len(ic_df))
            m_col = "_m_index"

    if ic_col is None:
        candidates = [col for col in numeric_cols if col != m_col]
        if candidates:
            ic_col = candidates[0]

    if loss_col is None:
        candidates = [
            col for col in numeric_cols
            if col not in {m_col, ic_col}
        ]
        if candidates:
            loss_col = candidates[0]

    if ic_col is None and loss_col is None:
        raise ValueError(
            "Cannot identify information criterion or loss columns in ic_path."
        )

    fig, ax = plt.subplots(figsize=figsize)

    m_values = ic_df[m_col]

    handles = []
    labels = []

    if ic_col is not None:
        line_ic, = ax.plot(
            m_values,
            ic_df[ic_col],
            marker="o",
            color="blue",
            label="Information criterion"
        )
        handles.append(line_ic)
        labels.append("Information criterion")
        ax.set_ylabel("Information criterion")
    else:
        ax.set_ylabel("Value")

    selected_m = result.get("n_breaks", None)

    if selected_m is not None:
        line_selected = ax.axvline(
            selected_m,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label="Selected number of breaks"
        )
        handles.append(line_selected)
        labels.append("Selected number of breaks")

    if plot_loss and loss_col is not None:
        ax2 = ax.twinx()
        line_loss, = ax2.plot(
            m_values,
            ic_df[loss_col],
            marker="s",
            linestyle=":",
            color="gray",
            label="QML loss"
        )
        ax2.set_ylabel("QML loss")
        handles.append(line_loss)
        labels.append("QML loss")

    ax.set_xlabel("Number of breaks")
    ax.set_title("MQML Joint-Estimation Profiles")
    ax.legend(handles, labels)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
