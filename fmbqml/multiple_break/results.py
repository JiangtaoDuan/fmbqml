"""Result object for multiple_break QML."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, List

import pandas as pd
import fmbqml.utils as U

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class MultiBreakResults:
    """Statistical result object with attribute-based access."""

    def __init__(self, results: Dict[str, Any]) -> None:
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary of computed statistics.")
        self._results: Dict[str, Any] = deepcopy(results)

    def _get(self, key: str, default: Any = None) -> Any:
        """Internal lookup for display methods; not a public mapping API."""
        return self._results.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of all stored results.

        The returned dictionary is independent of this result object, so
        modifying it does not alter the stored results.
        """
        return deepcopy(self._results)

    @property
    def test_type(self) -> Optional[str]:
        """Analysis type that produced this result."""
        return self._get("test_type")

    @property
    def test_statistic(self) -> Optional[float]:
        """Observed likelihood-ratio test statistic, when available."""
        return self._get("test_statistic")

    @property
    def critical_value(self) -> Optional[float]:
        """Simulated critical value at the requested significance level."""
        return self._get("critical_value")

    @property
    def reject_null(self) -> Optional[bool]:
        """Whether the null hypothesis is rejected; None when no test was performed."""
        return self._get("reject_null")

    @property
    def estimated_factors(self) -> Optional[int]:
        """Number of pseudo-factors used in the analysis."""
        return self._get("estimated_factors")

    @property
    def factor_criterion(self) -> Any:
        """Criterion used to select the number of pseudo-factors."""
        return self._get("factor_criterion")

    @property
    def max_factors(self) -> Optional[int]:
        """Maximum candidate number of pseudo-factors."""
        return self._get("max_factors")

    @property
    def trim_ratio(self) -> Optional[float]:
        """Trimming proportion used to define the breakpoint search region."""
        return self._get("trim_ratio")

    @property
    def significance_level(self) -> Optional[float]:
        """Significance level used for the hypothesis test, when available."""
        return self._get("significance_level", self._get("alpha"))

    @property
    def runtime_seconds(self) -> Optional[float]:
        """Wall-clock runtime of the analysis, in seconds."""
        return self._get("runtime_seconds")

    @property
    def T(self) -> Optional[int]:
        """Number of time observations in the input panel."""
        return self._get("T")

    @property
    def N(self) -> Optional[int]:
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
    def candidate_breakpoints(self) -> Optional[List[int]]:
        """Breakpoints selected under the alternative in a joint sup-LR test."""
        values = self._get("candidate_breakpoints")
        return None if values is None else deepcopy(list(values))

    @property
    def candidate_break_labels(self) -> Any:
        """Original observation labels for candidate breakpoints under H1."""
        return deepcopy(self._get("candidate_break_labels"))

    @property
    def breakpoints(self) -> List[int]:
        """Accepted breakpoints, or QML-estimated breakpoints for joint estimation."""
        values = self._get("breakpoints")
        return [] if values is None else deepcopy(list(values))

    @property
    def n_breaks(self) -> Optional[int]:
        """Number of breaks selected by joint QML estimation or specified for testing."""
        return self._get("n_breaks")

    @property
    def max_break(self) -> Optional[int]:
        """Maximum number of breaks considered by the joint estimator."""
        return self._get("max_break")

    @property
    def min_break(self) -> Optional[int]:
        """Minimum number of breaks considered by the joint estimator."""
        return self._get("min_break")

    @property
    def ic_path(self) -> Optional[pd.DataFrame]:
        """Public IC path with the statistics relevant to model comparison.

        Diagnostic fields remain in the internal result dictionary, but are
        intentionally excluded from this user-facing DataFrame. All values
        are copied; this property never modifies the stored results.
        """
        value = self._get("ic_path")
        if value is None:
            return None

        table = pd.DataFrame(deepcopy(value))
        columns = ("m", "IC", "loss", "estimated_breakpoints")
        return table.loc[:, [key for key in columns if key in table.columns]].copy()

    def format_ic_path_table(self) -> str:
        """Format the public IC path with fixed-point IC and loss values."""
        if self._get("test_type") != "mqml_joint":
            raise ValueError(
                "An information-criterion path requires joint QML estimation."
            )

        table = self.ic_path
        if table is None or table.empty:
            return "No information-criterion path available."

        formatters = {
            key: (lambda value: f"{value:.4f}")
            for key in ("IC", "loss")
            if key in table.columns
        }
        return table.to_string(index=False, formatters=formatters)

    @property
    def break_labels(self) -> Any:
        """Original observation labels for accepted or estimated breakpoints."""
        return deepcopy(self._get("break_labels"))

    @property
    def min_segment_length(self) -> Optional[int]:
        """Minimum number of observations required in each regime."""
        return self._get("min_segment_length")



    @staticmethod
    def _format_compact_value(value: Any, digits: int = 4) -> str:
        """Format a scalar for the compact string representation."""
        if value is None:
            return "N/A"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not pd.notna(number):
            return "N/A"
        return f"{number:.{digits}f}"

    @staticmethod
    def _compact_breakpoint_values(values: Any) -> List[Any]:
        """Normalize breakpoint values without relying on array truthiness."""
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            return [values]
        try:
            return list(values)
        except TypeError:
            return [values]

    @classmethod
    def _format_compact_breakpoints(cls, values: Any) -> str:
        """Format breakpoint locations for the compact string representation."""
        values = cls._compact_breakpoint_values(values)
        if not values:
            return "None"
        return ", ".join(str(value) for value in values)

    def _compact_text(self) -> str:
        """Return a concise overview; use ``summary()`` for the full report."""
        test_type = str(self._get("test_type") or "").lower()
        is_lr = (
            "joint_sup_lr" in test_type
            or self._get("test_statistic") is not None
            or self._get("critical_value") is not None
        )
        title = (
            "MultiBreakQML Joint LR Test"
            if is_lr else "MultiBreakQML Joint Estimation"
        )
        lines = [title, "-" * len(title)]

        T, N = self._get("T"), self._get("N")
        if T is not None or N is not None:
            lines.append(f"Sample size: T = {T if T is not None else 'N/A'}, "
                         f"N = {N if N is not None else 'N/A'}")

        n_factors = self._get("estimated_factors")
        criterion = self._get("factor_criterion")
        if n_factors is not None:
            factor_text = f"Estimated factors: {n_factors}"
            if criterion is not None:
                criterion = U.CRITERION_NAMES.get(criterion, criterion)
                factor_text += f" ({criterion})"
            lines.append(factor_text)

        if is_lr:
            # Compact output reports accepted breakpoints only. Candidate
            # locations under H1 remain available in summary(); when H0 is
            # not rejected, the accepted-breakpoint list is empty.
            if len(lines) > 2:
                lines.append("")

            lines.append(
                f"LR statistic: {self._format_compact_value(self._get('test_statistic'))}"
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

            if reject is not None and bool(reject):
                breakpoints = self._compact_breakpoint_values(
                    self._get("breakpoints", [])
                )
                if not breakpoints:
                    breakpoints = self._compact_breakpoint_values(
                        self._get("candidate_breakpoints")
                    )
                breakpoint_label = "Accepted breakpoints"
            else:
                breakpoints = []
                breakpoint_label = "Accepted breakpoints"

            lines.append("")
            lines.append(
                f"{breakpoint_label}: "
                f"{self._format_compact_breakpoints(breakpoints)}"
            )
        else:
            if len(lines) > 2:
                lines.append("")
            breakpoints = self._compact_breakpoint_values(
                self._get("breakpoints", [])
            )
            n_breaks = self._get("n_breaks")
            if n_breaks is None:
                n_breaks = len(breakpoints)
            lines.append(f"Selected breaks: {n_breaks}")
            lines.append(
                f"Estimated breakpoints: "
                f"{self._format_compact_breakpoints(breakpoints)}"
            )

        # Only show types for accepted LR-test breaks, or QML estimates.
        reported_breakpoints = (
            breakpoints
            if (not is_lr or (self._get("reject_null") is not None
                              and bool(self._get("reject_null"))))
            else []
        )
        break_types = self._get("break_types")
        if break_types is None:
            break_type = self._get("break_type")
            break_types = [] if break_type is None else [break_type]
        elif isinstance(break_types, str):
            break_types = [break_types]
        else:
            try:
                break_types = list(break_types)
            except TypeError:
                break_types = [break_types]

        if reported_breakpoints and break_types:
            lines.append(
                "Breakpoint types: " + ", ".join(str(value) for value in break_types)
            )

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
        """Return a structured result summary without recomputing estimates."""
        result = self.to_dict()
        if not include_labels:
            for key in ("break_label", "candidate_break_label", "break_labels", "candidate_break_labels"):
                result.pop(key, None)
        return U.format_multi_break_result_summary(result)

    def plot_mqml_profile(
            self,
            figsize: Tuple[int, int] = (8, 4),
            plot_loss: bool = True,
            save_path: Optional[str] = None,
            show: bool = True
    ) -> Tuple["Figure", "Axes"]:
        """Plot the saved joint-QML information-criterion path; return (fig, ax)."""
        if self._get("test_type") != "mqml_joint":
            raise ValueError("plot_mqml_profile() requires a result from estimate_breaks_jointly().")
        return U.plot_mqml_profile(
            result=self._results, figsize=figsize, plot_loss=plot_loss,
            save_path=save_path, show=show,
        )
