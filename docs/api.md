
# API guide

## Input conventions

Both estimators accept a NumPy array or pandas DataFrame of shape `(T, N)`, where rows are observations and columns are variables. Values must be numeric and finite. DataFrame row labels are preserved and used to report breakpoint labels. Breakpoint positions are integer split locations in the observation sequence.

A breakpoint at position `k` separates the first `k` observations from the remaining observations.

## `SingleBreakQML`

```python
SingleBreakQML(
    X,
    trim_ratio=0.30,
    max_factors=10,
    factor_criterion="IC1",
)
```

The trimming ratio defines the fraction excluded from each end of the candidate-break range. The argument `max_factors` specifies the upper bound for factor-number selection.

### Main methods

- `lr_test(alpha=0.01, classify=True, verbose=False, n_sim=5000, random_state=None)` tests the null hypothesis of no structural break. Valid significance levels are `0.01`, `0.05`, and `0.10`.
- `estimate_breakpoint(classify=True, verbose=False)` estimates one breakpoint under the assumption that a break exists, without performing a significance test.
- `classify_break(break_point=None, verbose=False)` classifies an estimated or supplied breakpoint.
- `plot_lr_profile(...)` and `plot_qml_profile(...)` return Matplotlib figure and axes objects. Set `show=False` for scripts and automated runs.
- `summary()` returns a formatted text summary of the latest analysis.

### Dedicated result access methods

The following methods provide direct access to commonly used results without requiring users to inspect internal result dictionaries.

- `is_rejected()` returns whether the latest LR test rejects the no-break null hypothesis. This method is available only after `lr_test()`.
- `get_breakpoint()` returns the breakpoint location from the latest single-break procedure. After `lr_test()`, it returns `None` if the null hypothesis is not rejected. After `estimate_breakpoint()`, it returns the estimated location under the assumption that one break exists.
- `get_break_type()` returns the classification of the estimated breakpoint. It returns `None` if no breakpoint is reported and raises `ValueError` if a breakpoint exists but classification results are unavailable.
- `get_lr_profile()` returns the LR profile as a pandas DataFrame containing candidate breakpoint locations and corresponding LR statistics.
- `get_qml_profile()` returns the QML objective profile as a pandas DataFrame containing candidate breakpoint locations and objective values.

The `is_rejected()` method should not be used after `estimate_breakpoint()`, because that procedure does not perform a significance test.

### Additional results and model updates

- `get_test_results()` returns an independent copy of the latest result dictionary.
- `get_factor_estimates(recalculate=False)` returns estimated factors, loadings, normalization values, and the selected factor count.
- `get_data()` returns a copy of the input panel in its original format.
- `set_data(new_data)` replaces the panel and clears previously computed results.
- `set_parameters(trim_ratio=None, max_factors=None, factor_criterion=None)` updates validated settings and clears cached results.

For advanced analysis, the result dictionary contains additional fields such as `break_point`, `estimated_break_candidate`, `reject_null`, `test_statistic`, `critical_value`, `estimated_factors`, `candidate_breaks`, `lr_profile`, and `qml_profile`.

When classification is performed, the results also contain `break_type`, `break_types`, `classification_criteria`, and `break_type_summary`.

## `MultiBreakQML`

```python
MultiBreakQML(
    X,
    max_break=5,
    max_factors=10,
    factor_criterion="IC1",
    trim_ratio=0.10,
)
```

The argument `max_break` specifies the maximum number of candidate breaks. The trimming ratio controls the minimum admissible regime length.

### Main methods

- `estimate_breaks_jointly(min_break=0, classify=True, verbose=False)` jointly selects the number and locations of structural breaks using the QML information criterion.
- `joint_sup_lr_test(n_breaks, alpha=0.01, n_sim=5000, random_state=None, classify=True, verbose=False)` tests the null hypothesis of no structural change against an alternative with exactly `n_breaks` changes.
- `classify_joint_breaks(verbose=False)` classifies the breakpoints obtained from the latest joint estimation.
- `plot_mqml_profile(...)` plots the information-criterion path over candidate numbers of breaks.
- `plot_joint_lr_profile(...)` plots joint sup-LR statistics over feasible candidate numbers of breaks.
- `format_ic_path_table()` returns a formatted text table of the information-criterion path.
- `summary(mode="joint")` returns a formatted summary of joint estimation or joint sup-LR testing. Use `mode="joint_lr"` for the joint sup-LR test summary.

### Dedicated result access methods

The following methods allow users to retrieve commonly used results without directly accessing internal result dictionaries.

- `get_breakpoints(mode="joint")` returns the estimated breakpoint locations as a list. The default mode retrieves locations from joint estimation. Use `mode="joint_lr"` to retrieve candidate locations fitted under the prespecified alternative in the joint sup-LR test.
- `get_n_breaks()` returns the number of breaks selected by `estimate_breaks_jointly()`. It returns `0` when the no-break model is selected.
- `is_rejected()` returns whether the latest `joint_sup_lr_test()` rejects the null hypothesis of no structural change.
- `get_break_types(mode="joint")` returns a list of breakpoint classifications corresponding to the estimated locations. Use `mode="joint_lr"` to retrieve classifications from joint sup-LR testing.
- `get_ic_path()` returns the information-criterion path as a pandas DataFrame after joint estimation.

When joint estimation selects no breakpoints, `get_breakpoints()` and `get_break_types()` return empty lists, and `get_n_breaks()` returns `0`.

For `mode="joint_lr"`, breakpoint locations are estimated under the prespecified alternative and may be returned even when the test does not reject the no-break null hypothesis. Users should check `is_rejected()` before interpreting these candidate locations as evidence of structural change.

Breakpoint classifications are available when the corresponding procedure is run with `classify=True`, or when joint-estimation classifications are subsequently computed using `classify_joint_breaks()`.

### Information-criterion path

The information-criterion path can be accessed in two forms.

- `get_ic_path()` returns a pandas DataFrame containing information-criterion results for the candidate numbers of breaks.
- `format_ic_path_table()` returns a formatted text representation of the same information.

For example:

```python
model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
    verbose=False,
)

ic_path = model.get_ic_path()
print(ic_path)

print(model.format_ic_path_table())
```

The DataFrame returned by `get_ic_path()` can be used for additional analysis, visualization, or export.

### Additional results and model updates

- `get_joint_results()` returns an independent copy of the latest joint-estimation results.
- `get_joint_test_results()` returns an independent copy of the latest joint sup-LR test results.
- `get_factor_estimates(recalculate=False)` returns estimated factors, loadings, normalization values, and the selected factor count.
- `get_data()` returns a copy of the input panel.
- `set_data(new_X)` replaces the input data and clears previously computed results.
- `set_parameters(max_break=None, max_factors=None, factor_criterion=None)` updates validated settings and clears cached results.

Joint-estimation results contain additional fields such as `breakpoints`, `n_breaks`, `ic_path`, `estimated_factors`, and classification results when requested.

Joint sup-LR test results include the rejection decision, test statistic, critical value, candidate breakpoint locations, and optional breakpoint classifications.

The complete result dictionaries remain available for advanced users who need detailed diagnostic information beyond the dedicated access methods.

## Errors and boundary conditions

The public classes raise `ValueError` for non-finite or incorrectly shaped input data, invalid significance levels, infeasible trimming constraints, invalid factor bounds, and attempts to access results before the corresponding analysis has been performed.

A selected factor count of zero is reported as an error because the QML procedures require at least one factor.

Result access methods also validate the availability of the requested information. For example, `is_rejected()` requires a completed LR test, while `get_ic_path()` requires completed joint estimation with an available information-criterion path.

When breakpoint classification has not been performed, the corresponding classification access methods raise `ValueError` if breakpoint locations exist.

Result access methods return independent copies of stored mutable results where applicable. Modifying a returned DataFrame, list, or result dictionary does not modify the cached model results.

Calling `set_data()` or `set_parameters()` clears previously computed results. The corresponding estimation or testing procedures must be run again before their results can be accessed.
