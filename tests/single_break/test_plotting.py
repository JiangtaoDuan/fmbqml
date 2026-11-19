"""Check how profile plots describe estimated and accepted breaks."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmbqml.utils.plotting import plot_lr_profile, plot_qml_profile


@pytest.mark.parametrize("plot", [plot_lr_profile, plot_qml_profile])
@pytest.mark.parametrize(
    "test_type,reject_null,expected",
    [
        ("lr_test", np.bool_(False), "Candidate break (under H1)"),
        ("lr_test", np.bool_(True), "Accepted break"),
        ("estimate_breakpoint", None, "Estimated break"),
    ],
)
def test_profile_break_label(plot, test_type, reject_null, expected):
    result = {
        "test_type": test_type,
        "reject_null": reject_null,
        "candidate_breakpoint": 2,
        "candidate_breaks": [1, 2, 3],
        "lr_profile": [1, 2, 1],
        "qml_profile": [2, 1, 2],
    }
    fig, ax = plot(result, show=False)
    try:
        assert expected in ax.get_legend_handles_labels()[1]
    finally:
        plt.close(fig)
