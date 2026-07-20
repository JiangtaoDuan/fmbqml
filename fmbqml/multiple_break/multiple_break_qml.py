"""Object-oriented interface for multiple-break QML procedures."""

from copy import deepcopy

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fmbqml.break_tools import classify_breakpoint_result
from fmbqml.factor_tools import (
    bai_ng_factor_count,
    estimate_panel_factors,
)

from fmbqml.multiple_break.joint_estimation import joint_estimation
from fmbqml.multiple_break.joint_testing import _run_joint_sup_lr

import fmbqml.utils as U



class MultiBreakQML:
    """
    Multiple-break quasi-maximum likelihood (MultiBreakQML) procedure
    for detecting and estimating multiple structural breaks
    in high-dimensional factor models.

    This class jointly estimates the number and locations of structural breaks
    via global QML optimization and supports breakpoint-type classification
    through ``classify_joint_breaks()``.
    """
    CRITERION_NAMES = U.CRITERION_NAMES

    def __init__(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            max_break: int = 5,
            max_factors: int = 10,
            factor_criterion: Union[int, str] = "IC1",
            trim_ratio: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        X : np.ndarray or pandas.DataFrame
            Input data matrix with shape (T, N), where T is the sample size
            and N is the cross-sectional dimension.
        max_break : int, default=5
            Maximum number of structural breaks allowed.
            Must satisfy max_break >= 0.
        max_factors : int, default=10
            Maximum number of latent factors considered when estimating
            the factor dimension. Must satisfy max_factors <= min(T, N).
        factor_criterion : int or str, default="IC1"
            Bai--Ng factor-number selection criterion. It can be specified
            either as an integer in {0,1,2,3,4,5,6,7} or as one of
            {"IC1", "IC2", "IC3", "PC1", "PC2", "PC3", "AIC3", "BIC3"}.
        trim_ratio : float, default=0.1
            Default minimum regime-length ratio used by joint estimation,
            joint sup-LR testing, and joint breakpoint classification.
        """

        self.X, self._original_X, self._index, self._columns, self._is_dataframe = (
            U.prepare_multi_input_data(X)
        )

        self.T, self.N = U.validate_input_matrix(self.X, name="X")

        self.max_break = U.validate_nonnegative_integer(max_break, "max_break")
        self.max_factors = U.validate_max_factors(max_factors, self.T, self.N)
        self.factor_criterion = U.normalize_factor_criterion(factor_criterion)
        self.trim_ratio = U.validate_trim_ratio(trim_ratio)

        self._reset_state()

    def _reset_state(self) -> None:
        self.joint_estimated = False
        self.joint_tested = False

        self.joint_results: Dict[str, Any] = {}
        self.joint_test_results: Dict[str, Any] = {}

        self._r_hat: Optional[int] = None
        self._F_hat: Optional[np.ndarray] = None
        self._L_hat: Optional[np.ndarray] = None
        self._VNT: Optional[np.ndarray] = None

    def _classify_breakpoints(
            self,
            breakpoints: List[int],
            result_dict: Dict[str, Any],
            trim_ratio: Optional[float] = None,
            min_regime_length: Optional[int] = None,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Shared breakpoint classification logic for joint-estimation results.
        """
        updated_result, _ = classify_breakpoint_result(
            X=self.X,
            breakpoints=breakpoints,
            result_dict=result_dict,
            max_factors=self.max_factors,
            factor_criterion=self.factor_criterion,
            full_sample_factor_number=None,
            T=self.T,
            trim_ratio=trim_ratio,
            min_regime_length=min_regime_length,
            full_sample_standardize=True,
            segment_standardize=True,
            verbose=verbose,
        )

        return updated_result.copy()

    def classify_joint_breaks(
            self,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """Classify breakpoints from the latest joint estimation.

        Parameters
        ----------
        verbose : bool, default=False
            If True, print the classification summary.

        Returns
        -------
        dict
            Updated joint-estimation results containing breakpoint
            classifications.

        Raises
        ------
        ValueError
            If :meth:`estimate_breaks_jointly` has not been run.
        """
        if not self.joint_estimated:
            raise ValueError("Run estimate_breaks_jointly() first.")

        breakpoints = self.joint_results.get("breakpoints", [])

        updated_result = self._classify_breakpoints(
            breakpoints=breakpoints,
            result_dict=self.joint_results,
            trim_ratio=self.trim_ratio,
            min_regime_length=None,
            verbose=verbose
        )

        self.joint_results = updated_result
        return deepcopy(self.joint_results)

    def plot_mqml_profile(
            self,
            figsize: Tuple[int, int] = (8, 4),
            plot_loss: bool = True,
            save_path: Optional[str] = None,
            show: bool = True
    ):
        """
        Plot MQML joint-estimation profiles over the number of breaks.
        """
        if not self.joint_estimated:
            raise ValueError(
                "Run estimate_breaks_jointly() before plotting MQML profiles."
            )

        return U.plot_mqml_profile(
            result=self.joint_results,
            figsize=figsize,
            plot_loss=plot_loss,
            save_path=save_path,
            show=show,
        )

    def plot_joint_lr_profile(
            self,
            alpha: float = 0.01,
            n_sim: int = 5000,
            random_state: Optional[int] = None,
            figsize: Tuple[int, int] = (8, 4),
            save_path: Optional[str] = None,
            show: bool = True
    ):
        """
        Plot joint sup-LR statistics over candidate numbers of breaks.
        """
        alpha = U.validate_alpha(alpha)
        factors = self.get_factor_estimates(recalculate=False)

        profile = []
        for n_breaks in range(1, self.max_break + 1):
            try:
                result = _run_joint_sup_lr(
                    F_hat=factors["F_hat"],
                    n_breaks=n_breaks,
                    trim_ratio=self.trim_ratio,
                    alpha=alpha,
                    n_sim=n_sim,
                    random_state=random_state,
                )
            except ValueError:
                continue

            profile.append({
                "n_breaks": n_breaks,
                "test_statistic": result["test_statistic"],
                "critical_value": result["critical_value"],
                "reject_null": result["reject_null"],
                "breakpoints": result["breakpoints"],
            })

        if not profile:
            raise ValueError("No feasible joint LR profile points are available.")

        if self.joint_tested:
            plot_result = self.joint_test_results.copy()
            plot_result["joint_lr_profile"] = profile
            self.joint_test_results = plot_result
        else:
            plot_result = {
                "test_type": "joint_sup_lr_profile",
                "n_breaks": None,
                "joint_lr_profile": profile,
            }

        return U.plot_joint_lr_profile(
            result=plot_result,
            figsize=figsize,
            save_path=save_path,
            show=show,
        )

    def joint_sup_lr_test(
            self,
            n_breaks: int,
            alpha: float = 0.01,
            n_sim: int = 5000,
            random_state: Optional[int] = None,
            classify: bool = True,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """Test no structural change against exactly ``n_breaks`` changes.

        This implements the joint sup-LR statistic for covariance changes. The
        observed partition is optimized on the sample grid, while the limiting
        Brownian-bridge supremum is approximated internally.

        Parameters
        ----------
        n_breaks : int
            Prespecified number of changes under the alternative.
        alpha : {0.10, 0.05, 0.01}, default=0.01
            Significance level.
        n_sim : int, default=5000
            Number of Brownian-bridge Monte Carlo replications.
        random_state : int, optional
            Seed for critical-value simulation.
        classify : bool, default=True
            If True, classify the detected breakpoints after the LR test.
        verbose : bool, default=False
            Print a compact result summary.

        Returns
        -------
        dict
            Joint LR statistic, estimated breakpoints, simulated critical
            values, rejection decision, factor metadata, and optional
            breakpoint classifications.

        Notes
        -----
        The critical-value distribution is simulated for the specified
        ``n_breaks`` using the joint partition-based routine
        ``joint_sup_lr_cv``. It is not obtained from the single-break helper
        ``critical_value_lr_hac_m``. When ``n_breaks=1``, the limiting
        statistic reduces to the single-break form, subject to differences in
        simulation grids and other numerical conventions.
        """
        n_breaks = U.validate_positive_integer(n_breaks, "n_breaks")
        if n_breaks > self.max_break:
            raise ValueError(
                f"n_breaks must be <= max_break={self.max_break}. "
                f"Got n_breaks={n_breaks}."
            )
        alpha = U.validate_alpha(alpha)

        factors = self.get_factor_estimates(recalculate=False)
        result = _run_joint_sup_lr(
            F_hat=factors["F_hat"],
            n_breaks=n_breaks,
            trim_ratio=self.trim_ratio,
            alpha=alpha,
            n_sim=n_sim,
            random_state=random_state,
        )
        result.update({
            "estimated_factors": factors["r_hat"],
            "factor_criterion": self.factor_criterion,
            "max_break": self.max_break,
            "max_factors": self.max_factors,
            "T": self.T,
            "N": self.N,
        })
        labels = U.map_breakpoints_to_labels(result["breakpoints"], self._index)
        if labels:
            result["break_labels"] = labels

        if classify and len(result["breakpoints"]) > 0:
            result = self._classify_breakpoints(
                breakpoints=result["breakpoints"],
                result_dict=result,
                trim_ratio=self.trim_ratio,
                min_regime_length=None,
                verbose=False,
            )

        self.joint_tested = True
        self.joint_test_results = result

        if verbose:
            print(U.format_multi_break_summary(result, mode="joint_lr"))
        return deepcopy(self.joint_test_results)

    def estimate_breaks_jointly(
            self,
            min_break: int = 0,
            classify: bool = True,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Jointly estimate structural breakpoints via global QML optimization.

        Parameters
        ----------
        min_break : int, default=0
            Minimum number of breaks allowed.
        classify : bool, default=True
            If True, classify the detected breakpoints into structural
            breakpoint types after joint estimation.
        verbose : bool, default=False
            If True, print the joint-estimation summary.

        Returns
        -------
        dict
            Dictionary containing joint-estimation results, including
            detected breakpoints, the selected number of breaks, and the
            information criterion path.
        """

        min_break = U.validate_nonnegative_integer(min_break, "min_break")

        if min_break > self.max_break:
            raise ValueError(
                f"min_break must be <= max_break. Got min_break={min_break}, "
                f"max_break={self.max_break}."
            )

        min_segment_length = U.compute_min_segment_length(self.T, self.trim_ratio)

        # Theoretical upper bound under the minimum segment length constraint:
        # with m breaks, there are m + 1 regimes, so approximately (m + 1) * h <= T.
        max_feasible_breaks = max(0, self.T // min_segment_length - 1)

        if min_break > max_feasible_breaks:
            raise ValueError(
                f"min_break={min_break} is infeasible under the current "
                f"minimum segment length={min_segment_length}. "
                f"At most {max_feasible_breaks} breaks can be accommodated "
                f"when T={self.T} and trim_ratio={self.trim_ratio}."
            )

        if self.max_break > max_feasible_breaks:
            warnings.warn(
                f"max_break={self.max_break} may be too large for the current "
                f"sample size and spacing constraint. "
                f"With T={self.T} and min_segment_length={min_segment_length}, at most about "
                f"{max_feasible_breaks} breaks are feasible. "
                f"The joint estimation routine may ignore infeasible candidates "
                f"or return fewer breaks.",
                RuntimeWarning,
                stacklevel=2,
            )

        breakpoints, break_number, ic_path = joint_estimation(
            self.X,
            min_segment_length,
            self.max_factors,
            self.max_break,
            min_break,
            self.factor_criterion
        )

        if breakpoints is None:
            breakpoints = []
        else:
            flat_breaks = np.asarray(breakpoints).flatten()

            if len(flat_breaks) == 0:
                breakpoints = []
            elif len(flat_breaks) == 1 and str(flat_breaks[0]) == "no breakpoint":
                breakpoints = []
            else:
                breakpoints = sorted(set(int(b) for b in flat_breaks))

        break_number = len(breakpoints)

        self.joint_estimated = True

        break_labels = U.map_breakpoints_to_labels(breakpoints, self._index)
        estimated_factors = None
        if ic_path:
            for row in ic_path:
                if int(row.get("m", -1)) == break_number:
                    estimated_factors = row.get("n_factors")
                    break
            if estimated_factors is None:
                estimated_factors = ic_path[0].get("n_factors")

        self.joint_results = {
            'test_type': 'mqml_joint',
            'breakpoints': breakpoints,
            'n_breaks': break_number,
            'max_break': self.max_break,
            'min_break': min_break,
            'factor_criterion': self.factor_criterion,
            'max_factors': self.max_factors,
            'estimated_factors': estimated_factors,
            'trim_ratio': self.trim_ratio,
            'min_segment_length': min_segment_length,
            'T': self.T,
            'N': self.N,
            'ic_path': ic_path
        }

        if break_labels:
            self.joint_results['break_labels'] = break_labels

        if classify and len(breakpoints) > 0:
            self.classify_joint_breaks(
                verbose=False
            )

        if verbose:
            print(self.summary(mode="joint"))

        return deepcopy(self.joint_results)

    def format_ic_path_table(self) -> str:
        """Format the latest information-criterion path as a table.

        Returns
        -------
        str
            Text table containing the information criterion for each
            candidate number of breaks.

        Raises
        ------
        ValueError
            If :meth:`estimate_breaks_jointly` has not been run.
        """
        if not self.joint_estimated:
            raise ValueError("Run estimate_breaks_jointly() first.")

        return U.format_ic_path_table(
            self.joint_results.get("ic_path", None)
        )

    def summary(self, mode: str = "joint") -> str:
        """Return a formatted summary of the latest joint estimation.

        Parameters
        ----------
        mode : {"joint"}, default="joint"
            Result mode to summarize. Only joint estimation is currently
            supported.

        Returns
        -------
        str
            Formatted estimation summary.

        Raises
        ------
        ValueError
            If no joint estimation is available or ``mode`` is unsupported.
        """
        if mode == "joint":
            if not self.joint_estimated:
                raise ValueError("No joint estimation has been executed yet.")

            return U.format_multi_break_summary(
                self.joint_results,
                mode="joint"
            )

        raise ValueError("mode must be 'joint'.")

    def get_joint_results(self) -> Dict[str, Any]:
        """
        Return the joint-estimation results.

        Returns
        -------
        dict
            Dictionary containing the current joint-estimation results.

        Raises
        ------
        ValueError
            If estimate_breaks_jointly() has not been executed yet.
        """
        if not self.joint_estimated:
            raise ValueError("Run estimate_breaks_jointly() before retrieving joint results.")
        return deepcopy(self.joint_results)

    def get_joint_test_results(self) -> Dict[str, Any]:
        """Return results from :meth:`joint_sup_lr_test`."""
        if not self.joint_tested:
            raise ValueError(
                "Run joint_sup_lr_test() before retrieving joint test results."
            )
        return deepcopy(self.joint_test_results)

    def get_data(self) -> Union[np.ndarray, pd.DataFrame]:
        """
        Return the input X in its original format.

        Returns
        -------
        np.ndarray or pandas.DataFrame
            Copy of the original input X.
        """
        return self._original_X.copy()

    def get_factor_estimates(self, recalculate: bool = False) -> Dict[str, Any]:
        """
        Return estimated factors and loadings used in multi-break analysis.

        Parameters
        ----------
        recalculate : bool, default=False
            If True, re-estimate the factor structure even if cached values
            already exist. If False, reuse cached results when available.

        Returns
        -------
        dict
            Dictionary containing:
            - r_hat : estimated number of factors
            - F_hat : estimated factor matrix
            - L_hat : estimated loading matrix
            - VNT   : diagonal matrix of leading eigenvalues

        Notes
        -----
        The number of factors is selected using the Bai-Ng information
        criterion specified by `self.factor_criterion`. Factor estimation
        is then performed by `estimate_panel_factors()`.
        """
        if recalculate or self._F_hat is None or self._L_hat is None or self._VNT is None:
            nb_factors_result = bai_ng_factor_count(self.X, self.max_factors)
            self._r_hat = int(nb_factors_result["khat"][self.factor_criterion])

            if self._r_hat <= 0:
                raise ValueError(
                    "Estimated number of factors is zero. "
                    "MultiBreakQML requires at least one factor."
                )

            self._F_hat, self._L_hat, self._VNT = estimate_panel_factors(
                self.X,
                self._r_hat
            )

        return {
            "r_hat": self._r_hat,
            "F_hat": self._F_hat.copy() if self._F_hat is not None else None,
            "L_hat": self._L_hat.copy() if self._L_hat is not None else None,
            "VNT": self._VNT.copy() if self._VNT is not None else None,
            "factor_criterion": self.factor_criterion,
            "criterion_name": self.CRITERION_NAMES.get(self.factor_criterion, "IC1"),
        }

    def set_data(
            self,
            new_X: Union[np.ndarray, pd.DataFrame],
            verbose: bool = False
    ) -> "MultiBreakQML":
        """Replace the panel data and clear all cached results.

        Parameters
        ----------
        new_X : ndarray or pandas.DataFrame
            New panel with observations in rows and variables in columns.
        verbose : bool, default=False
            If True, print the dimensions of the new panel.

        Returns
        -------
        MultiBreakQML
            The updated instance, enabling method chaining.

        Raises
        ------
        ValueError
            If the new panel or current factor settings are invalid.
        """

        new_array, new_original_X, new_index, new_columns, new_is_dataframe = (
            U.prepare_multi_input_data(new_X)
        )

        new_T, new_N = U.validate_input_matrix(new_array, name="X")
        U.validate_max_factors(self.max_factors, new_T, new_N)

        self.X = new_array
        self._original_X = new_original_X
        self._index = new_index
        self._columns = new_columns
        self._is_dataframe = new_is_dataframe
        self.T, self.N = new_T, new_N

        self._reset_state()

        if verbose:
            print(f"X updated: T = {self.T}, N = {self.N}")

        return self

    def set_parameters(
            self,
            max_break: Optional[int] = None,
            max_factors: Optional[int] = None,
            factor_criterion: Optional[Union[int, str]] = None,
            verbose: bool = False
    ) -> "MultiBreakQML":
        """Update model settings and clear all cached results.

        Parameters
        ----------
        max_break : int or None, default=None
            New maximum number of breaks; ``None`` preserves the current
            value.
        max_factors : int or None, default=None
            New maximum number of factors; ``None`` preserves the current
            value.
        factor_criterion : int, str, or None, default=None
            New Bai--Ng criterion; ``None`` preserves the current value.
        verbose : bool, default=False
            If True, print the updated settings.

        Returns
        -------
        MultiBreakQML
            The updated instance, enabling method chaining.

        Raises
        ------
        ValueError
            If any supplied setting is invalid for the current panel.
        """

        new_max_break = (
            self.max_break
            if max_break is None
            else U.validate_nonnegative_integer(max_break, "max_break")
        )

        new_max_factors = (
            self.max_factors
            if max_factors is None
            else U.validate_max_factors(max_factors, self.T, self.N)
        )

        new_factor_criterion = (
            self.factor_criterion
            if factor_criterion is None
            else U.normalize_factor_criterion(factor_criterion)
        )

        self.max_break = new_max_break
        self.max_factors = new_max_factors
        self.factor_criterion = new_factor_criterion

        self._reset_state()

        if verbose:
            criterion_name = self.CRITERION_NAMES.get(
                self.factor_criterion,
                self.factor_criterion
            )
            print(
                "Parameters updated: "
                f"max_break = {self.max_break}, "
                f"max_factors = {self.max_factors}, "
                f"factor_criterion = {criterion_name}"
            )

        return self
