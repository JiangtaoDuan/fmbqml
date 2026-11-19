"""Object-oriented interface for single-break QML procedures."""

from copy import deepcopy
from time import perf_counter
import logging

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

from .results import SingleBreakResults


_logger = logging.getLogger(__name__)


class SingleBreakQML:
    """
    SingleBreakQML is a Python class for detecting and estimating a single structural break in approximate factor models
    using quasi-maximum likelihood methods. The class provides functionality for validating panel data input, estimating
    the number of latent factors, computing factor estimates, performing likelihood ratio tests for structural breaks,
    estimating the most likely breakpoint location, and classifying the type of break. In addition to break detection,
    the class also includes utility methods for summarizing test results, retrieving estimated factors and loadings,
    and updating model data or parameters for repeated analysis. LR test results
    and direct breakpoint-estimation results are stored separately.
    """

    # Configuration and data are updated only through validated setters.
    _PROTECTED_SETTINGS = ('T', 'N', 'trim_ratio', 'max_factors', 'factor_criterion')

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_X" and "_X" in self.__dict__:
            raise AttributeError("Internal panel data cannot be replaced directly; use set_data().")
        if name in self._PROTECTED_SETTINGS and name in self.__dict__:
            raise AttributeError(
                f"{name} is read-only; use set_parameters() or set_data() instead."
            )
        super().__setattr__(name, value)

    @property
    def X(self) -> np.ndarray:
        """Read-only view of the panel used in estimation."""
        view = self._X.view()
        view.setflags(write=False)
        return view

    def _set_input_array(self, array: np.ndarray) -> None:
        """Store a private, independent, non-writeable panel array."""
        values = np.ascontiguousarray(array, dtype=float)
        # Immutable bytes backing prevents re-enabling WRITEABLE through _X.
        frozen = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(self, "_X", frozen)

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
        array, self._original_data, self._row_index = U.prepare_input_data(X)
        self._set_input_array(array)

        self.T, self.N = self.X.shape

        self.trim_ratio = U.validate_trim_ratio(trim_ratio)

        self.max_factors = U.validate_max_factors(max_factors, self.T, self.N)

        self.factor_criterion = U.normalize_factor_criterion(factor_criterion)

        U.calculate_break_range(self.T, self.trim_ratio)

        self._reset_state()
        _logger.debug(
            "Initialized SingleBreakQML (T=%d, N=%d, trim_ratio=%s, max_factors=%d).",
            self.T, self.N, self.trim_ratio, self.max_factors,
        )

    def __repr__(self) -> str:
        """Return a one-line representation of the model configuration."""
        criterion = (
            U.CRITERION_NAMES.get(self.factor_criterion, str(self.factor_criterion))
            if isinstance(self.factor_criterion, (int, np.integer))
            else str(self.factor_criterion)
        )
        return (
            f"SingleBreakQML(T={self.T}, N={self.N}, "
            f"max_factors={self.max_factors}, "
            f"factor_criterion={criterion!r}, "
            f"trim_ratio={self.trim_ratio:.4f})"
        )

    __str__ = __repr__

    def _reset_state(self) -> None:
        self._F_hat = None
        self._L_hat = None
        self._VNT = None
        self._nbhat_r = None
        self.is_tested = False
        self._test_results = {}
        self._estimation_results = {}
        self._last_result_type = None

    def _get_active_results(self) -> Dict[str, Any]:
        """Return the result dictionary associated with the latest procedure."""
        if self._last_result_type == "test" and self._test_results:
            return self._test_results
        if self._last_result_type == "estimation" and self._estimation_results:
            return self._estimation_results
        if self._test_results:
            return self._test_results
        if self._estimation_results:
            return self._estimation_results
        raise ValueError(
            "Run lr_test() or estimate_breakpoint() before accessing results."
        )

    def _set_active_results(self, result_type: str, results: Dict[str, Any]) -> None:
        """Store results separately and mark the procedure as the latest analysis."""
        if result_type == "test":
            self._test_results = results
        elif result_type == "estimation":
            self._estimation_results = results
        else:
            raise ValueError("result_type must be 'test' or 'estimation'.")
        self._last_result_type = result_type
        self.is_tested = bool(self._test_results)

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

    def _calculate_factors(self, *, recalculate: bool = False) -> Tuple[int, np.ndarray]:
        """
        Return cached factors or estimate them when needed.

        Raises
        ------
        ValueError
            If the selected number of factors is zero.
        """
        if (
            not recalculate
            and self._F_hat is not None
            and self._L_hat is not None
            and self._VNT is not None
            and self._nbhat_r is not None
        ):
            return self._nbhat_r, self._F_hat

        # Compute into local variables first. Update the factor cache only
        # after the complete factor-estimation step has succeeded.
        nbhat_r = self._estimate_factor_count()

        if nbhat_r <= 0:
            raise ValueError(
                "Estimated number of factors is zero. "
                "SingleBreakQML requires at least one factor."
            )

        F_hat, L_hat, VNT = estimate_panel_factors_lr(self.X, nbhat_r)

        self._nbhat_r = nbhat_r
        self._F_hat = F_hat
        self._L_hat = L_hat
        self._VNT = VNT

        return nbhat_r, F_hat

    def classify_break(
            self,
            verbose: bool = False
    ) -> SingleBreakResults:
        """Classify the latest estimated or accepted single breakpoint.

        The breakpoint is retrieved from the latest stored analysis result;
        users cannot supply an arbitrary breakpoint location. After an LR
        non-rejection, the candidate breakpoint is not treated as an
        accepted breakpoint for classification.

        Parameters
        ----------
        verbose : bool, default=False
            If True, print the classification summary.

        Returns
        -------
        SingleBreakResults
            Updated result object containing the breakpoint classification.

        Raises
        ------
        ValueError
            If no estimated or accepted breakpoint is available.
        """
        verbose = U.validate_boolean(verbose, "verbose")
        result = self._get_active_results()
        breakpoint = result.get("breakpoint")

        if breakpoint is None:
            raise ValueError(
                "No estimated breakpoint is available for classification. "
                "Run estimate_breakpoint() or obtain an accepted LR-test result first."
            )

        breakpoints = [breakpoint]

        min_regime_length = U.compute_min_segment_length(
            self.T,
            self.trim_ratio
        )
        full_sample_factor_number = result.get(
            "estimated_factors",
            None
        )

        updated_result, _ = classify_breakpoint_result(
            X=self.X,
            breakpoints=breakpoints,
            result_dict=result,
            max_factors=self.max_factors,
            factor_criterion=self.factor_criterion,
            full_sample_factor_number=full_sample_factor_number,
            T=self.T,
            min_regime_length=min_regime_length,
            full_sample_standardize=False,
            segment_standardize=True,
            verbose=verbose,
        )

        if self._last_result_type == "test":
            self._test_results = updated_result
        else:
            self._estimation_results = updated_result
        return SingleBreakResults(deepcopy(updated_result))

    def lr_test(
            self,
            alpha: float = 0.01,
            classify: bool = True,
            n_sim: int = 5000,
            random_state: Optional[int] = None
    ) -> SingleBreakResults:
        """
        Perform the likelihood ratio (LR) test for a structural break.

        Parameters
        ----------
        alpha : float, default=0.01
            Significance level. Must be one of 0.01, 0.05, or 0.10.
        classify : bool, default=True
            If True and the null of no break is rejected, classify the
            estimated breakpoint.
        n_sim : int, default=5000
            Number of Monte Carlo replications used to compute critical values.
        random_state : int or None, default=None
            Random seed used for reproducible Monte Carlo simulation.

        Returns
        -------
        SingleBreakResults
            Result object providing attribute-based access to the LR test statistics.
            Printing the result displays a concise overview. Call result.summary() for the complete statistical report.
            The LR and QML profiles remain available through result attributes.

        Notes
        -----
        ``breakpoint`` is reported only when the null hypothesis is rejected,
        whereas ``candidate_breakpoint`` is the maximizer of the LR profile
        regardless of the rejection decision.

        Critical values are computed by ``critical_value_lr_hac_m``, which
        estimates the HAC long-run covariance from the factors and simulates
        the single-break Brownian-bridge supremum. The joint multiple-break LR
        test uses a separate partition-based critical-value simulator.

        If validation or any stage of the computation fails, the most recent
        successfully completed result remains available on the model.
        """
        # Validate all user-controlled arguments before doing any work or
        # touching the previously committed result state.
        alpha = U.validate_alpha(alpha)
        classify = U.validate_boolean(classify, "classify")
        n_sim = U.validate_positive_integer(n_sim, "n_sim")
        random_state = U.validate_random_state(random_state)

        start_time = perf_counter()
        _logger.info(
            "Starting single-break LR test (T=%d, N=%d, alpha=%s).",
            self.T, self.N, alpha,
        )

        T_min, T_max = U.calculate_break_range(self.T, self.trim_ratio)

        nbhat_r, F_hat = self._calculate_factors()

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

        # breakpoint means "accepted estimated break after LR rejection"
        breakpoint = hatk if result_rej else None

        # candidate_breakpoint means the maximizer of LR profile,
        # regardless of whether the LR test rejects the null.
        candidate_breakpoint = hatk

        break_label = U.map_breakpoint_to_label(breakpoint, self._row_index)
        candidate_break_label = U.map_breakpoint_to_label(
            candidate_breakpoint,
            self._row_index
        )

        test_results = {
            'test_type': 'lr_test',
            'breakpoint': breakpoint,
            'candidate_breakpoint': candidate_breakpoint,
            'break_label': break_label,
            'candidate_break_label': candidate_break_label,
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
            'classify': bool(classify),
            'max_factors': self.max_factors,
            'trim_ratio': self.trim_ratio,
            'T': self.T,
            'N': self.N,
        }

        # Classification is part of this analysis transaction.  Work on the
        # local result so a classification failure cannot partially replace
        # the last successful model result.
        if classify and breakpoint is not None:
            min_regime_length = U.compute_min_segment_length(
                self.T, self.trim_ratio
            )
            test_results, _ = classify_breakpoint_result(
                X=self.X,
                breakpoints=[breakpoint],
                result_dict=test_results,
                max_factors=self.max_factors,
                factor_criterion=self.factor_criterion,
                full_sample_factor_number=test_results.get("estimated_factors"),
                T=self.T,
                min_regime_length=min_regime_length,
                full_sample_standardize=False,
                segment_standardize=True,
                verbose=False,
            )

        # Commit only after estimation, simulation, and optional classification
        # have all completed successfully.
        test_results["runtime_seconds"] = perf_counter() - start_time
        self._set_active_results("test", test_results)
        _logger.info(
            "Completed single-break LR test (LR=%.6g, reject_null=%s, runtime=%.3fs).",
            test_results["test_statistic"],
            test_results["reject_null"],
            test_results["runtime_seconds"],
        )

        return SingleBreakResults(deepcopy(test_results))

    def estimate_breakpoint(
            self,
            classify: bool = True
    ) -> SingleBreakResults:
        """
        Estimate the structural breakpoint using the SingleBreakQML criterion.

        Parameters
        ----------
        classify : bool, default=True
            If True, classify the estimated breakpoint.

        Returns
        -------
        SingleBreakResults
            Result object containing the estimated breakpoint, its label when
            available, the estimated number of factors, factor criterion,
            search range, candidate break locations, QML objective profile,
            and optional breakpoint classification results.

        Notes
        -----
        If estimation or optional classification fails, the most recent
        successfully completed result remains available on the model.
        """
        classify = U.validate_boolean(classify, "classify")
        start_time = perf_counter()
        _logger.info(
            "Starting single-break QML estimation (T=%d, N=%d).",
            self.T, self.N,
        )

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

        breakpoint = int(candidate_breaks[np.argmin(qml_profile)])
        break_label = U.map_breakpoint_to_label(breakpoint, self._row_index)

        estimation_results = {
            'test_type': 'estimate_breakpoint',
            'breakpoint': breakpoint,
            'candidate_breakpoint': breakpoint,
            'break_label': break_label,
            'candidate_break_label': break_label,
            'reject_null': None,
            'test_statistic': None,
            'critical_value': None,
            'significance_level': None,
            'estimated_factors': nbhat_r,
            'factor_criterion': self.factor_criterion,
            'search_range': (T_min, T_max),
            'candidate_breaks': candidate_breaks,
            'qml_profile': qml_profile,
            'max_factors': self.max_factors,
            'trim_ratio': self.trim_ratio,
            'T': self.T,
            'N': self.N,
            'classify': bool(classify),
        }

        if classify and breakpoint is not None:
            min_regime_length = U.compute_min_segment_length(
                self.T, self.trim_ratio
            )
            estimation_results, _ = classify_breakpoint_result(
                X=self.X,
                breakpoints=[breakpoint],
                result_dict=estimation_results,
                max_factors=self.max_factors,
                factor_criterion=self.factor_criterion,
                full_sample_factor_number=estimation_results.get("estimated_factors"),
                T=self.T,
                min_regime_length=min_regime_length,
                full_sample_standardize=False,
                segment_standardize=True,
                verbose=False,
            )

        estimation_results["runtime_seconds"] = perf_counter() - start_time
        self._set_active_results("estimation", estimation_results)
        _logger.info(
            "Completed single-break QML estimation (breakpoint=%s, runtime=%.3fs).",
            estimation_results.get("breakpoint"),
            estimation_results["runtime_seconds"],
        )

        return SingleBreakResults(deepcopy(estimation_results))



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

        recalculate = U.validate_boolean(recalculate, "recalculate")
        if recalculate:
            self._calculate_factors(recalculate=True)
        elif self._F_hat is None or self._L_hat is None or self._VNT is None:
            self._calculate_factors()

        return {
            'F_hat': self._F_hat.copy() if self._F_hat is not None else None,
            'L_hat': self._L_hat.copy() if self._L_hat is not None else None,
            'VNT': self._VNT.copy() if hasattr(self._VNT, 'copy') else self._VNT,
            'estimated_factors': self._nbhat_r,
            'factor_criterion': self.factor_criterion
        }

    def set_data(
            self,
            new_data: Union[np.ndarray, pd.DataFrame]
    ) -> "SingleBreakQML":
        """Replace the panel data and clear all cached results.

        Previously returned result objects remain independent snapshots.
        Direct assignment to configuration attributes and X is not supported.

        Parameters
        ----------
        new_data : ndarray or pandas.DataFrame
            New panel with observations in rows and variables in columns.
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

        new_T, new_N = new_X.shape

        new_max_factors = U.validate_max_factors(
            self.max_factors,
            new_T,
            new_N
        )

        _ = U.calculate_break_range(new_T, self.trim_ratio)

        self._set_input_array(new_X)
        self._original_data = new_original_data
        self._row_index = new_row_index
        object.__setattr__(self, "T", new_T)
        object.__setattr__(self, "N", new_N)
        object.__setattr__(self, "max_factors", new_max_factors)

        self._reset_state()
        _logger.info("Updated single-break input data (T=%d, N=%d); cleared cached state.", self.T, self.N)

        return self

    def set_parameters(
            self,
            trim_ratio: Optional[float] = None,
            max_factors: Optional[int] = None,
            factor_criterion: Optional[Union[int, str]] = None
    ) -> "SingleBreakQML":
        """Update model settings and clear all cached results.

        Results returned by earlier calls are independent snapshots and remain
        unchanged. Future computations use the new settings. Configuration
        attributes are read-only outside this method.

        Parameters
        ----------
        trim_ratio : float or None, default=None
            New trimming fraction; ``None`` preserves the current value.
        max_factors : int or None, default=None
            New maximum number of factors; ``None`` preserves the current
            value.
        factor_criterion : int, str, or None, default=None
            New Bai--Ng criterion; ``None`` preserves the current value.
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

        object.__setattr__(self, "trim_ratio", new_trim_ratio)
        object.__setattr__(self, "max_factors", new_max_factors)
        object.__setattr__(self, "factor_criterion", new_factor_criterion)

        self._reset_state()
        _logger.info(
            "Updated single-break model parameters (trim_ratio=%s, max_factors=%d, factor_criterion=%s); cleared cached state.",
            self.trim_ratio, self.max_factors, self.factor_criterion,
        )

        return self
