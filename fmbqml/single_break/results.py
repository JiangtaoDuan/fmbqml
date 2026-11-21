"""Result object for single_break QML."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import fmbqml.utils as U
from fmbqml.utils.formatting import _DisplayDict

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class SingleBreakResults:
    """Statistical result object with attribute-based access."""

    def __init__(self, results: dict[str, Any]) -> None:
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary of computed statistics.")
        self._results: dict[str, Any] = deepcopy(results)

    def _get(self, key: str, default: Any = None) -> Any:
        """Internal lookup for display methods; not a public mapping API."""
        return self._results.get(key, default)

    def to_dict(self, include_diagnostics: bool = True) -> dict[str, Any]:
        """Return an independent dictionary containing the stored results.

        Parameters
        ----------
        include_diagnostics : bool, default=True
            If True, include diagnostic paths and arrays. If False, omit
            potentially large diagnostic data while retaining the principal
            estimation or test results.

        Returns
        -------
        dict
            A deep copy of the stored results. The returned dictionary is
            independent of this result object.

        Notes
        -----
        The returned dictionary displays one top-level field per line.
        """
        include_diagnostics = U.validate_boolean(
            include_diagnostics, "include_diagnostics"
        )
        if include_diagnostics:
            results = deepcopy(self._results)
        else:
            excluded = {'candidate_breaks', 'lr_profile', 'qml_profile'}
            results = deepcopy({
                key: value for key, value in self._results.items() if key not in excluded
            })
        return _DisplayDict(results)

    @property
    def test_type(self) -> str | None:
        """Analysis type that produced this result."""
        return self._get("test_type")

    @property
    def test_statistic(self) -> float | None:
        """Observed likelihood-ratio test statistic, when available."""
        return self._get("test_statistic")

    @property
    def critical_value(self) -> float | None:
        """Simulated critical value at the requested significance level."""
        return self._get("critical_value")

    @property
    def reject_null(self) -> bool | None:
        """Whether the null hypothesis is rejected; None when no test was performed."""
        return self._get("reject_null")

    @property
    def estimated_factors(self) -> int | None:
        """Number of pseudo-factors used in the analysis."""
        return self._get("estimated_factors")

    @property
    def factor_criterion(self) -> Any:
        """Criterion used to select the number of pseudo-factors."""
        return self._get("factor_criterion")

    @property
    def max_factors(self) -> int | None:
        """Maximum candidate number of pseudo-factors."""
        return self._get("max_factors")

    @property
    def trim_ratio(self) -> float | None:
        """Trimming proportion used to define the breakpoint search region."""
        return self._get("trim_ratio")

    @property
    def significance_level(self) -> float | None:
        """Significance level used for the hypothesis test, when available."""
        return self._get("significance_level", self._get("alpha"))

    @property
    def runtime_seconds(self) -> float | None:
        """Wall-clock runtime of the analysis, in seconds."""
        return self._get("runtime_seconds")

    @property
    def T(self) -> int | None:
        """Number of time observations in the input panel."""
        return self._get("T")

    @property
    def N(self) -> int | None:
        """Number of cross-sectional series in the input panel."""
        return self._get("N")

    @property
    def break_types(self) -> Any:
        """Classification labels for the reported breakpoints, when available."""
        return deepcopy(self._get("break_types"))

    @property
    def classification_criteria(self) -> Any:
        """Factor-count comparisons used to classify the breakpoints."""
        return deepcopy(self._get("classification_criteria"))

    @property
    def breakpoint(self) -> int | None:
        """Estimated breakpoint, or None when no break is accepted."""
        return self._get("breakpoint")

    @property
    def candidate_breakpoint(self) -> int | None:
        """LR-profile maximizing location, regardless of the test decision."""
        return self._get("candidate_breakpoint")

    @property
    def break_type(self) -> str | None:
        """Classification label for the reported single breakpoint, when available."""
        typ = self._get("break_type")
        if typ is None:
            types = self._get("break_types")
            if types is not None and len(types):
                typ = types[0]
        return typ

    @property
    def search_range(self) -> Any:
        """Range of candidate breakpoint locations evaluated by the estimator."""
        return deepcopy(self._get("search_range"))

    @property
    def lr_profile(self) -> pd.DataFrame | None:
        """LR statistics indexed by candidate breakpoint (independent copy)."""
        breaks, values = self._get("candidate_breaks"), self._get("lr_profile")
        if breaks is None or values is None:
            return None
        return pd.DataFrame({"candidate_break": deepcopy(breaks),
                             "lr_statistic": deepcopy(values)})

    @property
    def qml_profile(self) -> pd.DataFrame | None:
        """QML objective by candidate breakpoint (independent copy)."""
        breaks, values = self._get("candidate_breaks"), self._get("qml_profile")
        if breaks is None or values is None:
            return None
        return pd.DataFrame({"candidate_break": deepcopy(breaks),
                             "qml_objective": deepcopy(values)})

    @property
    def lr_values(self) -> np.ndarray | None:
        """Raw LR profile values, returned as an independent copy."""
        return deepcopy(self._get("lr_profile"))

    @property
    def qml_values(self) -> np.ndarray | None:
        """Raw QML objective values, returned as an independent copy."""
        return deepcopy(self._get("qml_profile"))

    @property
    def candidate_breaks(self) -> np.ndarray | None:
        """Candidate breakpoint locations evaluated in the profile search."""
        return deepcopy(self._get("candidate_breaks"))

    @property
    def break_label(self) -> Any:
        """Original observation label for the accepted/estimated breakpoint."""
        return deepcopy(self._get("break_label"))

    @property
    def candidate_break_label(self) -> Any:
        """Original observation label for the LR-profile candidate breakpoint."""
        return deepcopy(self._get("candidate_break_label"))

    @staticmethod
    def _format_compact_value(value: Any, digits: int = 4) -> str:
        """Format a scalar for the compact string representation."""
        if value is None:
            return "N/A"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if pd.isna(number):
            return "N/A"
        return f"{number:.{digits}f}"

    def _compact_text(self) -> str:
        """Return a concise overview; use ``summary()`` for the full report."""
        is_lr = self._get("test_type") == "lr_test"
        title = (
            "SingleBreakQML LR Test"
            if is_lr else "SingleBreakQML Breakpoint Estimation"
        )
        lines = [title, "-" * len(title)]

        T, N = self._get("T"), self._get("N")
        if T is not None or N is not None:
            lines.append(
                f"Sample size: T = {T if T is not None else 'N/A'}, "
                f"N = {N if N is not None else 'N/A'}"
            )

        n_factors = self._get("estimated_factors")
        criterion = self._get("factor_criterion")
        if n_factors is not None:
            factor_text = f"Estimated factors: {n_factors}"
            if criterion is not None:
                criterion = U.CRITERION_NAMES.get(criterion, criterion)
                factor_text += f" ({criterion})"
            lines.append(factor_text)

        if is_lr:
            # Match the multiple-break output: model details, test results,
            # and the accepted breakpoint are separated into clear blocks.
            if len(lines) > 2:
                lines.append("")

            lines.append(
                f"LR statistic: "
                f"{self._format_compact_value(self._get('test_statistic'))}"
            )

            alpha = self._get("significance_level", self._get("alpha"))
            critical = self._get("critical_value")
            if alpha is not None:
                try:
                    alpha_text = f"{float(alpha) * 100:g}%"
                except (TypeError, ValueError):
                    alpha_text = str(alpha)
                lines.append(
                    f"Critical value ({alpha_text}): "
                    f"{self._format_compact_value(critical)}"
                )
            elif critical is not None:
                lines.append(
                    f"Critical value: {self._format_compact_value(critical)}"
                )

            reject = self._get("reject_null")
            decision = (
                "Not available" if reject is None
                else "Reject H0" if bool(reject)
                else "Do not reject H0"
            )
            lines.append(f"Decision: {decision}")

            # Compact output reports accepted breakpoints only. Candidate
            # locations under H1 remain available in summary(); when H0 is
            # not rejected, no breakpoint is reported as accepted.
            if reject is not None and bool(reject):
                breakpoint = self._get("breakpoint")
                if breakpoint is None:
                    breakpoint = self._get("candidate_breakpoint")
                breakpoint_label = "Accepted breakpoint"
            else:
                breakpoint = None
                breakpoint_label = "Accepted breakpoint"

            lines.append("")
            lines.append(
                f"{breakpoint_label}: "
                f"{breakpoint if breakpoint is not None else 'None'}"
            )
        else:
            if len(lines) > 2:
                lines.append("")
            breakpoint = self._get("breakpoint")
            lines.append(
                f"Estimated breakpoint: "
                f"{breakpoint if breakpoint is not None else 'None'}"
            )

        # Classification is available only for an accepted LR-test break,
        # or for a breakpoint returned by the standalone QML estimator.
        show_break_type = (
            not is_lr or (self._get("reject_null") is not None
                          and bool(self._get("reject_null")))
        )
        if breakpoint is not None and show_break_type:
            break_type = self._get("break_type")
            if break_type is None:
                break_types = self._get("break_types")
                if isinstance(break_types, (list, tuple, np.ndarray)) and len(break_types):
                    break_type = break_types[0]
            if break_type:
                lines.append(f"Breakpoint type: {break_type}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Display a concise overview; call ``summary()`` for full details."""
        return self._compact_text()

    def __repr__(self) -> str:
        """Use the concise overview in interactive Python sessions as well."""
        return self._compact_text()

    @property
    def random_state(self) -> Any:
        """Stored simulation setting, if available."""
        return deepcopy(self._get("random_state"))

    @property
    def monte_carlo_replications(self) -> Any:
        """Stored simulation setting, if available."""
        return deepcopy(self._get("monte_carlo_replications"))

    @property
    def grid_size(self) -> Any:
        """Stored simulation setting, if available."""
        return deepcopy(self._get("grid_size"))

    def summary(self, include_labels: bool = True) -> U.Summary:
        """Return a structured summary of the stored QML estimate or LR test.

        Parameters
        ----------
        include_labels : bool, default=True
            Whether to include original observation labels for reported and
            candidate breakpoints when available.

        Returns
        -------
        Summary
            Structured statistical report.
        """
        result = self._results.copy()
        if not include_labels:
            for key in ("break_label", "candidate_break_label", "break_labels", "candidate_break_labels"):
                result.pop(key, None)
        return deepcopy(U.format_single_break_result_summary(result))

    def plot_lr_profile(
            self,
            true_break: int | None = None,
            figsize: tuple[int, int] = (8, 4),
            save_path: str | None = None,
            show: bool = True
    ) -> tuple["Figure", "Axes"]:
        """Plot the stored single-break LR profile without recomputation.

        Available for results returned by ``lr_test()``.

        Parameters
        ----------
        true_break : int or None, default=None
            Optional known breakpoint location to mark on the plot.
        figsize : tuple of float, default=(8, 4)
            Figure width and height in inches.
        save_path : str or path-like or None, default=None
            Destination for the figure; no file is written when None.
        show : bool, default=True
            Whether to call ``matplotlib.pyplot.show()``. When False, the
            figure can still be saved and is returned without being displayed.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        ax : matplotlib.axes.Axes
            Axes containing the LR profile.

        Raises
        ------
        ValueError
            If the result was not produced by ``lr_test()``.
        """
        if self._get("test_type") != "lr_test":
            raise ValueError("plot_lr_profile() requires a result from lr_test().")
        return U.plot_lr_profile(
            result=self._results, true_break=true_break, figsize=figsize,
            save_path=save_path, show=show,
        )

    def plot_qml_profile(
            self,
            true_break: int | None = None,
            figsize: tuple[int, int] = (8, 4),
            save_path: str | None = None,
            show: bool = True
    ) -> tuple["Figure", "Axes"]:
        """Plot the saved QML objective profile without re-estimation.

        Available after either ``estimate_breakpoint()`` or ``lr_test()``.

        Parameters
        ----------
        true_break : int or None, default=None
            Optional known breakpoint location to mark on the plot.
        figsize : tuple of float, default=(8, 4)
            Figure width and height in inches.
        save_path : str or path-like or None, default=None
            Destination for the figure; no file is written when None.
        show : bool, default=True
            Whether to call ``matplotlib.pyplot.show()``. When False, the
            figure can still be saved and is returned without being displayed.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        ax : matplotlib.axes.Axes
            Axes containing the QML objective profile.

        Raises
        ------
        ValueError
            If the result is not from a single-break estimation or LR test.
        """
        if self._get("test_type") not in {"lr_test", "estimate_breakpoint"}:
            raise ValueError("plot_qml_profile() requires a single-break result.")
        return U.plot_qml_profile(
            result=self._results, true_break=true_break, figsize=figsize,
            save_path=save_path, show=show,
        )
