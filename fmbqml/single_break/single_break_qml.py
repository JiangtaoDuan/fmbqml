"""Object-oriented interface for single-break QML procedures."""

from copy import deepcopy

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fmbqml.break_tools import classify_breakpoint_result
from fmbqml.factor_tools import (
    bai_ng_factor_count,
    estimate_panel_factors_lr,
)

from fmbqml.single_break.qml_profile import compute_single_break_qml_profile
from fmbqml.statistical_tools import critical_value_lr_hac_m, select_critical_value
import fmbqml.utils as U

class SingleBreakQML:
    """
    SingleBreakQML is a Python class for detecting and estimating a single structural break in approximate factor models
    using quasi-maximum likelihood methods. The class provides functionality for validating panel data input, estimating
    the number of latent factors, computing factor estimates, performing likelihood ratio tests for structural breaks,
    estimating the most likely breakpoint location, and classifying the type of break. In addition to break detection,
    the class also includes utility methods for summarizing test results, retrieving estimated factors and loadings,
    and updating model data or parameters for repeated analysis.
    """

    CRITERION_NAMES = U.CRITERION_NAMES

    def __init__(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            trim_ratio: float = 0.30,
            max_factors: int = 10,
            factor_criterion: Union[int, str] = "IC1"
    ) -> None:
        """
        Initialize a SingleBreakQML object for structural break detection,
        breakpoint estimation, and break-type classification in factor models.

        Parameters
        ----------
        X : np.ndarray or pandas.DataFrame
            Input data matrix with shape (T, N).
        trim_ratio : float, default=0.30
            Truncation parameter defining the admissible break search range.
            Must satisfy 0 < trim_ratio < 0.5.
        max_factors : int, default=10
            Maximum number of factors considered when estimating
            the latent factor dimension.
        factor_criterion : int or str, default="IC1"
            Bai--Ng factor-number selection criterion. It can be specified
            either as an integer in {0,1,2,3,4,5,6,7} or as one of
            {"IC1", "IC2", "IC3", "PC1", "PC2", "PC3", "AIC3", "BIC3"}.
        """
        self.X, self._original_data, self._row_index = U.prepare_input_data(X)

        self.T, self.N = U.validate_input_matrix(self.X, name="X")

        self.trim_ratio = U.validate_trim_ratio(trim_ratio)

        self.max_factors = U.validate_max_factors(max_factors, self.T, self.N)

        self.factor_criterion = U.normalize_factor_criterion(factor_criterion)

        U.calculate_break_range(self.T, self.trim_ratio)

        self._reset_state()

    def _reset_state(self) -> None:
        self._F_hat = None
        self._L_hat = None
        self._VNT = None
        self._nbhat_r = None
        self.is_tested = False
        self.test_results = {}

    def _estimate_factor_count(self) -> int:
        """
        Estimate the number of latent factors.

        Returns
        -------
        int
            Estimated number of factors selected by the
            specified Bai-Ng information criterion.
        """
        nb_factors_result = bai_ng_factor_count(self.X, self.max_factors)
        return int(nb_factors_result['khat'][self.factor_criterion])

    def _calculate_factors(self) -> Tuple[int, np.ndarray]:
        """
        Estimate the number of factors and the factor matrix.

        Raises
        ------
        ValueError
            If the selected number of factors is zero.
        """
        self._nbhat_r = self._estimate_factor_count()

        if self._nbhat_r <= 0:
            raise ValueError(
                "Estimated number of factors is zero. "
                "SingleBreakQML requires at least one factor."
            )

        self._F_hat, self._L_hat, self._VNT = estimate_panel_factors_lr(
            self.X,
            self._nbhat_r
        )

        return self._nbhat_r, self._F_hat

    def classify_break(
            self,
            break_point: Optional[int] = None,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Classify a single estimated breakpoint.
        """
        if break_point is None:
            break_point = self.test_results.get("break_point", None)

        breakpoints = [] if break_point is None else [break_point]

        min_regime_length = U.compute_min_segment_length(
            self.T,
            self.trim_ratio
        )
        full_sample_factor_number = self.test_results.get(
            "estimated_factors",
            None
        )

        updated_result, classification = classify_breakpoint_result(
            X=self.X,
            breakpoints=breakpoints,
            result_dict=self.test_results,
            max_factors=self.max_factors,
            factor_criterion=self.factor_criterion,
            full_sample_factor_number=full_sample_factor_number,
            T=self.T,
            min_regime_length=min_regime_length,
            full_sample_standardize=False,
            segment_standardize=True,
            verbose=verbose,
        )

        self.test_results = updated_result
        return classification

    def plot_lr_profile(
            self,
            true_break: Optional[int] = None,
            figsize: Tuple[int, int] = (8, 4),
            save_path: Optional[str] = None,
            show: bool = True
    ):
        """
        Plot the LR profile over candidate breakpoint locations.
        """
        if not self.is_tested:
            raise ValueError("Run lr_test() before plotting the LR profile.")

        return U.plot_lr_profile(
            result=self.test_results,
            true_break=true_break,
            figsize=figsize,
            save_path=save_path,
            show=show
        )

    def plot_qml_profile(
            self,
            true_break: Optional[int] = None,
            figsize: Tuple[int, int] = (8, 4),
            save_path: Optional[str] = None,
            show: bool = True
    ):
        """
        Plot the QML objective profile over candidate breakpoint locations.
        """
        if not self.is_tested:
            raise ValueError(
                "Run lr_test() or estimate_breakpoint() before plotting the QML profile."
            )

        return U.plot_qml_profile(
            result=self.test_results,
            true_break=true_break,
            figsize=figsize,
            save_path=save_path,
            show=show
        )

    def lr_test(
            self,
            alpha: float = 0.01,
            classify: bool = True,
            verbose: bool = False,
            n_sim: int = 5000,
            random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform the likelihood ratio (LR) test for a structural break.

        Parameters
        ----------
        alpha : float, default=0.01
            Significance level. Must be one of 0.01, 0.05, or 0.10.
        classify : bool, default=True
            If True and the null of no break is rejected, classify the
            estimated breakpoint.
        verbose : bool, default=False
            Whether to print the test summary.
        n_sim : int, default=5000
            Number of Monte Carlo replications used to compute critical values.
        random_state : int or None, default=None
            Random seed used for reproducible Monte Carlo simulation.

        Returns
        -------
        dict
            Dictionary containing the LR test results, including the estimated
            break candidate, accepted breakpoint, rejection decision, LR statistic,
            critical value, factor number, search range, LR profile, QML profile,
            and optional breakpoint classification results.

        Notes
        -----
        ``break_point`` is reported only when the null hypothesis is rejected,
        whereas ``estimated_break_candidate`` is the maximizer of the LR profile
        regardless of the rejection decision.

        Critical values are computed by ``critical_value_lr_hac_m``, which
        estimates the HAC long-run covariance from the factors and simulates
        the single-break Brownian-bridge supremum. The joint multiple-break LR
        test uses a separate partition-based critical-value simulator.
        """
        T_min, T_max = U.calculate_break_range(self.T, self.trim_ratio)

        nbhat_r, F_hat = self._calculate_factors()
        alpha = U.validate_alpha(alpha)

        cv = critical_value_lr_hac_m(
            F_hat,
            self.trim_ratio,
            n_sim=n_sim,
            random_state=random_state
        )

        qml_profile = compute_single_break_qml_profile(
            F_hat=F_hat,
            T=self.T,
            t_min=T_min,
            t_max=T_max,
            n_factors=nbhat_r,
        )

        candidate_breaks = np.arange(T_min, T_max + 1)

        hatSigma0 = F_hat.T @ F_hat / self.T
        logdet0 = U.safe_logdet(hatSigma0)

        lr_profile = self.T * logdet0 - qml_profile
        lr_profile = np.asarray(lr_profile, dtype=float)
        lr_profile[~np.isfinite(lr_profile)] = np.nan

        hatk = int(candidate_breaks[np.nanargmax(lr_profile)])

        lr_stat = float(np.nanmax(lr_profile))

        level = select_critical_value(cv, alpha)

        result_rej = lr_stat > level

        # break_point means "accepted estimated break after LR rejection"
        break_point = hatk if result_rej else None

        # estimated_break_candidate means the maximizer of LR profile,
        # regardless of whether the LR test rejects the null.
        estimated_break_candidate = hatk

        self.is_tested = True

        break_label = U.map_breakpoint_to_label(break_point, self._row_index)
        estimated_break_candidate_label = U.map_breakpoint_to_label(
            estimated_break_candidate,
            self._row_index
        )

        self.test_results = {
            'test_type': 'lr_test',
            'break_point': break_point,
            'estimated_break_candidate': estimated_break_candidate,
            'break_label': break_label,
            'estimated_break_candidate_label': estimated_break_candidate_label,
            'reject_null': result_rej,
            'test_statistic': lr_stat,
            'critical_value': level,
            'significance_level': alpha,
            'estimated_factors': nbhat_r,
            'factor_criterion': self.factor_criterion,
            'search_range': (T_min, T_max),
            'candidate_breaks': candidate_breaks,
            'lr_profile': lr_profile,
            'qml_profile': qml_profile,
            'monte_carlo_replications': n_sim,
            'random_state': random_state,
            'T': self.T,
            'N': self.N,
        }

        if classify and break_point is not None:
            self.classify_break(
                break_point=break_point,
                verbose=False
            )

        if verbose:
            print(U.format_single_break_summary(self.test_results))

        return deepcopy(self.test_results)

    def estimate_breakpoint(
            self,
            classify: bool = True,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Estimate the structural breakpoint using the SingleBreakQML criterion.

        Parameters
        ----------
        classify : bool, default=True
            If True, classify the estimated breakpoint.
        verbose : bool, default=False
            Whether to print the estimation summary.

        Returns
        -------
        dict
            Dictionary containing the estimated breakpoint, its label when
            available, the estimated number of factors, factor criterion,
            search range, candidate break locations, QML objective profile,
            and optional breakpoint classification results.
        """

        T_min, T_max = U.calculate_break_range(self.T, self.trim_ratio)

        nbhat_r, F_hat = self._calculate_factors()

        qml_profile = compute_single_break_qml_profile(
            F_hat=F_hat,
            T=self.T,
            t_min=T_min,
            t_max=T_max,
            n_factors=nbhat_r,
        )

        candidate_breaks = np.arange(T_min, T_max + 1)

        break_point = int(candidate_breaks[np.argmin(qml_profile)])
        break_label = U.map_breakpoint_to_label(break_point, self._row_index)

        self.is_tested = True
        self.test_results = {
            'test_type': 'estimate_breakpoint',
            'break_point': break_point,
            'estimated_break_candidate': break_point,
            'break_label': break_label,
            'estimated_break_candidate_label': break_label,
            'reject_null': None,
            'test_statistic': None,
            'critical_value': None,
            'significance_level': None,
            'estimated_factors': nbhat_r,
            'factor_criterion': self.factor_criterion,
            'search_range': (T_min, T_max),
            'candidate_breaks': candidate_breaks,
            'qml_profile': qml_profile,
            'T': self.T,
            'N': self.N,
        }

        if classify and break_point is not None:
            self.classify_break(
                break_point=break_point,
                verbose=False
            )

        if verbose:
            print(U.format_single_break_summary(self.test_results))

        return deepcopy(self.test_results)

    def summary(self) -> str:
        """Return a formatted summary of the latest single-break procedure.

        Returns
        -------
        str
            Formatted test or estimation summary.

        Raises
        ------
        ValueError
            If neither :meth:`lr_test` nor :meth:`estimate_breakpoint` has
            been run.
        """
        if not self.is_tested:
            raise ValueError("No test has been executed yet.")
        return U.format_single_break_summary(self.test_results)

    def get_test_results(self) -> Dict[str, Any]:
        """
        Return the LR or SingleBreakQML test results.

        Returns
        -------
        dict
            Dictionary containing the test results.

        Raises
        ------
        ValueError
            If the test has not been executed yet.
        """
        if not self.is_tested:
            raise ValueError("Run lr_test() or estimate_breakpoint() before retrieving results.")
        return deepcopy(self.test_results)

    def get_data(self) -> Union[np.ndarray, pd.DataFrame]:
        """
        Return the input data in its original format.

        Returns
        -------
        np.ndarray or pandas.DataFrame
            Copy of the original input data.
        """
        return self._original_data.copy()

    def get_factor_estimates(self, recalculate: bool = False) -> Dict[str, Any]:
        """
        Return estimated factors and loadings.

        Parameters
        ----------
        recalculate : bool, default=False
            If True, recompute factor estimates.

        Returns
        -------
        dict
            Dictionary containing

            - F_hat : estimated factor matrix
            - L_hat : estimated loading matrix
            - VNT : normalization constant
            - estimated_factors : number of factors
            - factor_criterion : Bai-Ng criterion index used for factor selection
        """

        if recalculate or self._F_hat is None:
            self._calculate_factors()

        return {
            'F_hat': self._F_hat.copy() if self._F_hat is not None else None,
            'L_hat': self._L_hat.copy() if self._L_hat is not None else None,
            'VNT': self._VNT.copy() if hasattr(self._VNT, 'copy') else self._VNT,
            'estimated_factors': self._nbhat_r,
            'factor_criterion': self.factor_criterion
        }

    def get_lr_profile(self) -> pd.DataFrame:
        """
        Return the LR profile as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with candidate break locations and LR statistics.
        """
        if not self.is_tested:
            raise ValueError("Run lr_test() before retrieving the LR profile.")

        if 'candidate_breaks' not in self.test_results or 'lr_profile' not in self.test_results:
            raise ValueError(
                "LR profile is not available. Re-run lr_test()."
            )

        return pd.DataFrame({
            'candidate_break': self.test_results['candidate_breaks'],
            'lr_statistic': self.test_results['lr_profile']
        })

    def get_qml_profile(self) -> pd.DataFrame:
        """
        Return the QML objective profile as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with candidate break locations and QML objective values.
        """
        if not self.is_tested:
            raise ValueError(
                "Run lr_test() or estimate_breakpoint() before retrieving the QML profile."
            )

        if 'candidate_breaks' not in self.test_results or 'qml_profile' not in self.test_results:
            raise ValueError(
                "QML profile is not available. Re-run lr_test() or estimate_breakpoint()."
            )

        return pd.DataFrame({
            'candidate_break': self.test_results['candidate_breaks'],
            'qml_objective': self.test_results['qml_profile']
        })

    def set_data(
            self,
            new_data: Union[np.ndarray, pd.DataFrame],
            verbose: bool = False
    ) -> "SingleBreakQML":
        """Replace the panel data and clear all cached results.

        Parameters
        ----------
        new_data : ndarray or pandas.DataFrame
            New panel with observations in rows and variables in columns.
        verbose : bool, default=False
            If True, print the dimensions of the new panel.

        Returns
        -------
        SingleBreakQML
            The updated instance, enabling method chaining.

        Raises
        ------
        ValueError
            If the new panel is invalid for the current model settings.
        """

        new_X, new_original_data, new_row_index = U.prepare_input_data(new_data)

        new_T, new_N = U.validate_input_matrix(new_X, name="X")

        new_max_factors = U.validate_max_factors(
            self.max_factors,
            new_T,
            new_N
        )

        _ = U.calculate_break_range(new_T, self.trim_ratio)

        self.X = new_X
        self._original_data = new_original_data
        self._row_index = new_row_index
        self.T, self.N = new_T, new_N
        self.max_factors = new_max_factors

        self._reset_state()

        if verbose:
            print(f"Data updated: T = {self.T}, N = {self.N}")

        return self

    def set_parameters(
            self,
            trim_ratio: Optional[float] = None,
            max_factors: Optional[int] = None,
            factor_criterion: Optional[Union[int, str]] = None,
            verbose: bool = False
    ) -> "SingleBreakQML":
        """Update model settings and clear all cached results.

        Parameters
        ----------
        trim_ratio : float or None, default=None
            New trimming fraction; ``None`` preserves the current value.
        max_factors : int or None, default=None
            New maximum number of factors; ``None`` preserves the current
            value.
        factor_criterion : int, str, or None, default=None
            New Bai--Ng criterion; ``None`` preserves the current value.
        verbose : bool, default=False
            If True, print the updated settings.

        Returns
        -------
        SingleBreakQML
            The updated instance, enabling method chaining.

        Raises
        ------
        ValueError
            If any supplied setting is invalid for the current panel.
        """

        new_trim_ratio = (
            self.trim_ratio
            if trim_ratio is None
            else U.validate_trim_ratio(trim_ratio)
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

        _ = U.calculate_break_range(self.T, new_trim_ratio)

        self.trim_ratio = new_trim_ratio
        self.max_factors = new_max_factors
        self.factor_criterion = new_factor_criterion

        self._reset_state()

        if verbose:
            criterion_name = U.CRITERION_NAMES[self.factor_criterion]
            print(
                f"Parameters updated: trim_ratio = {self.trim_ratio}, "
                f"max_factors = {self.max_factors}, "
                f"factor_criterion = {criterion_name}"
            )

        return self
