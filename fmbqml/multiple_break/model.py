"""Object-oriented interface for multiple-break QML procedures."""

from copy import deepcopy
from time import perf_counter
import logging

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

from .results import MultiBreakResults


_logger = logging.getLogger(__name__)


class MultiBreakQML:
    """
    Multiple-break quasi-maximum likelihood (MultiBreakQML) procedure
    for detecting and estimating multiple structural breaks
    in high-dimensional factor models.

    This class jointly estimates the number and locations of structural breaks
    via global QML optimization and supports breakpoint-type classification
    through ``classify_joint_breaks()``.
    """
    # Configuration and data are updated only through validated setters.
    _PROTECTED_SETTINGS = ('T', 'N', 'max_break', 'max_factors', 'factor_criterion', 'trim_ratio')

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
            the factor dimension. Must satisfy max_factors <= min(T, N) - 1.
        factor_criterion : int or str, default="IC1"
            Bai--Ng factor-number selection criterion. It can be specified
            either as an integer in {0,1,2,3,4,5,6,7} or as one of
            {"IC1", "IC2", "IC3", "PC1", "PC2", "PC3", "AIC3", "BIC3"}.
        trim_ratio : float, default=0.1
            Default minimum regime-length ratio used by joint estimation,
            joint sup-LR testing, and joint breakpoint classification.
        """

        array, self._original_X, self._index, self._columns, self._is_dataframe = (
            U.prepare_multi_input_data(X)
        )
        self._set_input_array(array)

        self.T, self.N = self.X.shape

        self.max_break = U.validate_nonnegative_integer(max_break, "max_break")
        self.max_factors = U.validate_max_factors(max_factors, self.T, self.N)
        self.factor_criterion = U.normalize_factor_criterion(factor_criterion)
        self.trim_ratio = U.validate_trim_ratio(trim_ratio)

        self._reset_state()
        _logger.debug(
            "Initialized MultiBreakQML (T=%d, N=%d, max_break=%d, trim_ratio=%s).",
            self.T, self.N, self.max_break, self.trim_ratio,
        )

    def __repr__(self) -> str:
        """Return a one-line representation of the model configuration."""
        criterion = (
            U.CRITERION_NAMES.get(self.factor_criterion, str(self.factor_criterion))
            if isinstance(self.factor_criterion, (int, np.integer))
            else str(self.factor_criterion)
        )
        return (
            f"MultiBreakQML(T={self.T}, N={self.N}, "
            f"max_break={self.max_break}, "
            f"max_factors={self.max_factors}, "
            f"factor_criterion={criterion!r}, "
            f"trim_ratio={self.trim_ratio:.2f})"
        )

    __str__ = __repr__

    def _reset_state(self) -> None:
        self.joint_estimated = False
        self.joint_tested = False

        self._joint_results: Dict[str, Any] = {}
        self._joint_test_results: Dict[str, Any] = {}

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
            full_sample_factor_number=result_dict.get("estimated_factors"),
            T=self.T,
            trim_ratio=trim_ratio,
            min_regime_length=min_regime_length,
            full_sample_standardize=False,
            segment_standardize=True,
            verbose=verbose,
        )

        return updated_result.copy()

    def classify_joint_breaks(
            self,
            verbose: bool = False
    ) -> MultiBreakResults:
        """Classify breakpoints from the latest joint estimation.

        Parameters
        ----------
        verbose : bool, default=False
            If True, print the classification summary.

        Returns
        -------
        MultiBreakResults
            Updated joint-estimation result with breakpoint classifications.

        Raises
        ------
        ValueError
            If :meth:`estimate_breaks_jointly` has not been run.
        """
        verbose = U.validate_boolean(verbose, "verbose")
        if not self.joint_estimated:
            raise ValueError("Run estimate_breaks_jointly() first.")

        breakpoints = self._joint_results.get("breakpoints", [])
        if not breakpoints:
            raise ValueError(
                "No estimated breakpoints are available for classification."
            )

        updated_result = self._classify_breakpoints(
            breakpoints=breakpoints,
            result_dict=self._joint_results,
            trim_ratio=self.trim_ratio,
            min_regime_length=None,
            verbose=verbose
        )

        self._joint_results = updated_result
        return MultiBreakResults(deepcopy(self._joint_results))


    def joint_sup_lr_test(
            self,
            n_breaks: int,
            alpha: float = 0.01,
            n_sim: int = 2000,
            random_state: Optional[int] = None,
            classify: bool = True
    ) -> MultiBreakResults:
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
        n_sim : int, default=2000
            Number of Brownian-bridge Monte Carlo replications.
        random_state : int, optional
            Seed for critical-value simulation.
        classify : bool, default=True
            If True, classify the estimated breakpoints only when the test
            rejects the null hypothesis of no structural break.

        Returns
        -------
        MultiBreakResults
            Result object containing the joint LR statistic, candidate
            breakpoints, simulated critical value, rejection decision,
            factor metadata, and optional breakpoint classifications.

        Notes
        -----
        The critical-value distribution is simulated for the specified
        ``n_breaks`` using the joint partition-based routine
        ``joint_sup_lr_cv``. It is not obtained from the single-break helper
        ``critical_value_lr_hac_m``. When ``n_breaks=1``, the limiting
        statistic reduces to the single-break form, subject to differences in
        simulation grids and other numerical conventions.

        If validation or any stage of the computation fails, the most recent
        successfully completed joint LR result remains available on the model.
        """
        start_time = perf_counter()
        _logger.info(
            "Starting joint sup-LR test (T=%d, N=%d, n_breaks=%s, alpha=%s).",
            self.T, self.N, n_breaks, alpha,
        )

        # Validate every public argument before factor estimation or any
        # change to the previously committed result state.
        n_breaks = U.validate_positive_integer(n_breaks, "n_breaks")
        classify = U.validate_boolean(classify, "classify")
        n_sim = U.validate_positive_integer(n_sim, "n_sim")
        random_state = U.validate_random_state(random_state)
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
            "estimated_factors": factors["estimated_factors"],
            "factor_criterion": self.factor_criterion,
            "max_break": self.max_break,
            "max_factors": self.max_factors,
            "T": self.T,
            "N": self.N,
            "classify": bool(classify),
            "n_sim": n_sim,
        })
        # Preserve the partition selected under H1 separately from the
        # breakpoints accepted after the joint sup-LR test.
        raw_candidate_breakpoints = result.get("breakpoints", [])
        if raw_candidate_breakpoints is None:
            raw_candidate_breakpoints = []
        else:
            raw_candidate_breakpoints = np.asarray(
                raw_candidate_breakpoints, dtype=object
            ).reshape(-1).tolist()
            if (len(raw_candidate_breakpoints) == 1
                    and str(raw_candidate_breakpoints[0]).lower() == "no breakpoint"):
                raw_candidate_breakpoints = []
        candidate_breakpoints = [int(value) for value in raw_candidate_breakpoints]
        candidate_break_labels = U.map_breakpoints_to_labels(
            candidate_breakpoints, self._index
        )
        reject_null = bool(result.get("reject_null", False))
        accepted_breakpoints = candidate_breakpoints if reject_null else []
        accepted_break_labels = candidate_break_labels if reject_null else []

        result["candidate_breakpoints"] = candidate_breakpoints
        result["candidate_break_labels"] = candidate_break_labels
        result["breakpoints"] = accepted_breakpoints
        result["break_labels"] = accepted_break_labels

        if classify and reject_null and accepted_breakpoints:
            result = self._classify_breakpoints(
                breakpoints=accepted_breakpoints,
                result_dict=result,
                trim_ratio=self.trim_ratio,
                min_regime_length=None,
                verbose=False,
            )

        result["runtime_seconds"] = perf_counter() - start_time

        self.joint_tested = True
        self._joint_test_results = result
        _logger.info(
            "Completed joint sup-LR test (LR=%.6g, reject_null=%s, runtime=%.3fs).",
            result.get("test_statistic", float("nan")),
            result.get("reject_null"),
            result["runtime_seconds"],
        )

        return MultiBreakResults(
            deepcopy(self._joint_test_results)
        )

    def estimate_breaks_jointly(
            self,
            min_break: int = 0,
            classify: bool = True
    ) -> MultiBreakResults:
        """
        Jointly estimate structural breakpoints via global QML optimization.

        Parameters
        ----------
        min_break : int, default=0
            Minimum number of breaks allowed.
        classify : bool, default=True
            If True, classify the detected breakpoints into structural
            breakpoint types after joint estimation.
        Returns
        -------
        MultiBreakResults
            Joint-estimation result, including detected breakpoints, the
            selected number of breaks, and the information criterion path.

        Notes
        -----
        If optimization or optional classification fails, the most recent
        successfully completed joint-estimation result remains available.
        """
        classify = U.validate_boolean(classify, "classify")
        start_time = perf_counter()
        _logger.info(
            "Starting joint multiple-break estimation (T=%d, N=%d, max_break=%d, min_break=%s).",
            self.T, self.N, self.max_break, min_break,
        )

        min_break = U.validate_nonnegative_integer(min_break, "min_break")

        if min_break > self.max_break:
            raise ValueError(
                f"min_break must be <= max_break. Got min_break={min_break}, "
                f"max_break={self.max_break}."
            )

        min_segment_length = U.compute_min_segment_length(self.T, self.trim_ratio)

        # With m breaks there are m + 1 regimes. Each regime must contain
        # at least h observations, so (m + 1) * h <= T.
        max_feasible_breaks = max(0, self.T // min_segment_length - 1)

        if min_break > max_feasible_breaks:
            raise ValueError(
                f"min_break={min_break} is infeasible under the current "
                f"minimum segment length={min_segment_length}. "
                f"At most {max_feasible_breaks} breaks can be accommodated "
                f"when T={self.T} and trim_ratio={self.trim_ratio}."
            )

        effective_max_break = min(self.max_break, max_feasible_breaks)

        if self.max_break > max_feasible_breaks:
            warnings.warn(
                f"max_break={self.max_break} exceeds the feasible upper "
                f"bound under the current minimum segment length "
                f"{min_segment_length}. "
                f"The search is restricted to at most "
                f"{effective_max_break} breaks.",
                RuntimeWarning,
                stacklevel=2,
            )

        breakpoints, break_number, ic_path = joint_estimation(
            self.X,
            min_segment_length,
            self.max_factors,
            effective_max_break,
            min_break,
            self.factor_criterion,
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

        break_labels = U.map_breakpoints_to_labels(breakpoints, self._index)
        estimated_factors = None
        if ic_path:
            for row in ic_path:
                if int(row.get("m", -1)) == break_number:
                    estimated_factors = row.get("n_factors")
                    break
            if estimated_factors is None:
                estimated_factors = ic_path[0].get("n_factors")

        joint_results = {
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
            'ic_path': ic_path,
            'classify': bool(classify),
        }

        if break_labels:
            joint_results['break_labels'] = break_labels

        if classify and len(breakpoints) > 0:
            # Classify the in-progress result without requiring joint_estimated=True.
            joint_results = self._classify_breakpoints(
                breakpoints=breakpoints,
                result_dict=joint_results,
                trim_ratio=self.trim_ratio,
                min_regime_length=None,
                verbose=False,
            )

        joint_results["runtime_seconds"] = perf_counter() - start_time

        # Commit the new estimation only after optimization and optional
        # classification have both completed successfully.
        self._joint_results = joint_results
        self.joint_estimated = True
        _logger.info(
            "Completed joint multiple-break estimation (n_breaks=%d, breakpoints=%s, runtime=%.3fs).",
            joint_results["n_breaks"],
            joint_results["breakpoints"],
            joint_results["runtime_seconds"],
        )

        return MultiBreakResults(deepcopy(joint_results))

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
            - estimated_factors : estimated number of factors
            - F_hat : estimated factor matrix
            - L_hat : estimated loading matrix
            - VNT   : diagonal matrix of leading eigenvalues
            - factor_criterion : Bai--Ng criterion used

        Notes
        -----
        The number of factors is selected using the Bai-Ng information
        criterion specified by `self.factor_criterion`. Factor estimation
        is then performed by `estimate_panel_factors()`.
        """
        recalculate = U.validate_boolean(recalculate, "recalculate")
        if recalculate or self._F_hat is None or self._L_hat is None or self._VNT is None:
            nb_factors_result = bai_ng_factor_count(self.X, self.max_factors)
            # Keep the existing factor cache intact unless the complete
            # factor-count and factor-estimation sequence succeeds.
            r_hat = int(nb_factors_result["khat"][self.factor_criterion])

            if r_hat <= 0:
                raise ValueError(
                    "Estimated number of factors is zero. "
                    "MultiBreakQML requires at least one factor."
                )

            F_hat, L_hat, VNT = estimate_panel_factors(self.X, r_hat)

            self._r_hat = r_hat
            self._F_hat = F_hat
            self._L_hat = L_hat
            self._VNT = VNT

        return {
            "estimated_factors": self._r_hat,
            "F_hat": self._F_hat.copy() if self._F_hat is not None else None,
            "L_hat": self._L_hat.copy() if self._L_hat is not None else None,
            "VNT": self._VNT.copy() if self._VNT is not None else None,
            "factor_criterion": self.factor_criterion,
        }

    def set_data(
            self,
            new_X: Union[np.ndarray, pd.DataFrame]
    ) -> "MultiBreakQML":
        """Replace the panel data and clear all cached results.

        Previously returned result objects remain independent snapshots.
        Direct assignment to configuration attributes and X is not supported.

        Parameters
        ----------
        new_X : ndarray or pandas.DataFrame
            New panel with observations in rows and variables in columns.
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

        new_T, new_N = new_array.shape
        U.validate_max_factors(self.max_factors, new_T, new_N)

        self._set_input_array(new_array)
        self._original_X = new_original_X
        self._index = new_index
        self._columns = new_columns
        self._is_dataframe = new_is_dataframe
        object.__setattr__(self, "T", new_T)
        object.__setattr__(self, "N", new_N)

        self._reset_state()
        _logger.info("Updated multiple-break input data (T=%d, N=%d); cleared cached state.", self.T, self.N)

        return self

    def set_parameters(
            self,
            max_break: Optional[int] = None,
            max_factors: Optional[int] = None,
            factor_criterion: Optional[Union[int, str]] = None,
            *,
            trim_ratio: Optional[float] = None
    ) -> "MultiBreakQML":
        """Update model settings and clear all cached results.

        Results returned by earlier calls are independent snapshots and remain
        unchanged. Future computations use the new settings. Configuration
        attributes are read-only outside this method.

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
        trim_ratio : float or None, keyword-only, default=None
            New minimum regime-length ratio; ``None`` preserves the current
            value.

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

        new_trim_ratio = (
            self.trim_ratio
            if trim_ratio is None
            else U.validate_trim_ratio(trim_ratio)
        )

        object.__setattr__(self, "max_break", new_max_break)
        object.__setattr__(self, "max_factors", new_max_factors)
        object.__setattr__(self, "factor_criterion", new_factor_criterion)
        object.__setattr__(self, "trim_ratio", new_trim_ratio)

        self._reset_state()
        _logger.info(
            "Updated multiple-break model parameters (max_break=%d, max_factors=%d, factor_criterion=%s, trim_ratio=%s); cleared cached state.",
            self.max_break, self.max_factors, self.factor_criterion, self.trim_ratio,
        )

        return self
