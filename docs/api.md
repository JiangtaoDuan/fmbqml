# API guide

## Input conventions

Both estimators accept a NumPy array or pandas DataFrame of shape `(T, N)`,
where rows are observations and columns are variables. Values must be numeric
and finite. DataFrame row labels are preserved and used to report breakpoint
labels. Breakpoint positions are integer split locations in the observation
sequence.

## `SingleBreakQML`

```python
SingleBreakQML(
    X,
    trim_ratio=0.30,
    max_factors=10,
    factor_criterion="IC1",
)
```

The trimming ratio defines the fraction excluded from each end of the
candidate-break range.

### Main methods

- `lr_test(alpha=0.01, classify=True, verbose=False, n_sim=5000,
  random_state=None)` tests the no-break null. Valid significance levels are
  `0.01`, `0.05`, and `0.10`.
- `estimate_breakpoint(classify=True, verbose=False)` estimates one breakpoint
  without deciding whether a break exists.
- `classify_break(break_point=None, verbose=False)` classifies an estimated or
  supplied breakpoint.
- `plot_lr_profile(...)` and `plot_qml_profile(...)` return Matplotlib figure
  and axes objects. Set `show=False` for scripts and automated runs.
- `summary()` returns a formatted text summary.

### Result access and updates

- `get_test_results()` returns the latest result dictionary.
- `get_factor_estimates(recalculate=False)` returns factors, loadings,
  normalization values, and the selected factor count.
- `get_lr_profile()` and `get_qml_profile()` return pandas DataFrames.
- `get_data()` returns a copy in the original container type.
- `set_data(new_data)` replaces the panel and clears cached results.
- `set_parameters(trim_ratio=None, max_factors=None,
  factor_criterion=None)` updates validated settings and clears cached results.

Important LR result fields include `break_point`,
`estimated_break_candidate`, `reject_null`, `test_statistic`,
`critical_value`, `estimated_factors`, `candidate_breaks`, `lr_profile`, and
`qml_profile`. Classification adds `break_type`, `break_types`,
`classification_criteria`, and `break_type_summary`.

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

`max_break` is the largest candidate number of breaks. The trimming ratio
controls the minimum admissible regime length.

### Main methods

- `estimate_breaks_jointly(min_break=0, classify=True, verbose=False)` jointly
  selects the number and locations of breaks by the QML information criterion.
- `joint_sup_lr_test(n_breaks, alpha=0.01, n_sim=5000,
  random_state=None, classify=True, verbose=False)` tests no change against
  exactly `n_breaks` changes.
- `classify_joint_breaks(verbose=False)` classifies the latest jointly
  estimated breakpoints.
- `plot_mqml_profile(...)` plots the information-criterion path.
- `plot_joint_lr_profile(...)` plots joint sup-LR results over feasible break
  counts.
- `format_ic_path_table()` and `summary()` return formatted text.

### Result access and updates

- `get_joint_results()` and `get_joint_test_results()` return independent
  copies of the latest results.
- `get_factor_estimates(recalculate=False)` returns factor estimates.
- `get_data()` returns a copy of the input panel.
- `set_data(new_X)` replaces data and clears all cached results.
- `set_parameters(max_break=None, max_factors=None,
  factor_criterion=None)` updates settings and clears cached results.

Joint-estimation results include `breakpoints`, `n_breaks`, `ic_path`,
`estimated_factors`, and classification fields when requested.

## Errors and boundary conditions

The public classes raise `ValueError` for non-finite or incorrectly shaped
data, invalid significance levels, infeasible trimming constraints, invalid
factor bounds, and result access before estimation. A selected factor count of
zero is reported as an error because the QML procedures require at least one
factor.
