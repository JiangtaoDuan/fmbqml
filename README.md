# fmbqml: Structural Break Analysis in High-Dimensional Factor Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/fmbqml.svg)](https://pypi.org/project/fmbqml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/JiangtaoDuan/fmbqml/blob/master/LICENSE.txt)

`fmbqml` is a Python package for quasi-maximum-likelihood analysis of
structural breaks in high-dimensional factor models. It provides tools for
single-break testing and estimation, joint multiple-break estimation, sup-LR
testing, factor-number selection, break-type classification, and diagnostic
plots.

## Links

- [Source code](https://github.com/JiangtaoDuan/fmbqml)
- [Issue tracker](https://github.com/JiangtaoDuan/fmbqml/issues)

## Installation

Install the package from PyPI with:

```bash
python -m pip install fmbqml
```

For development from a source checkout, use:

```bash
python -m pip install -e .
```

## Input data

The input data should be a finite two-dimensional NumPy array or pandas
DataFrame with observations in rows and variables in columns. Missing values
should be imputed before constructing a model.

## Single-break analysis

Assuming that `panel_data` contains the input panel dataset:

```python
from fmbqml import SingleBreakQML

model = SingleBreakQML(
    panel_data,
    trim_ratio=0.15,
    max_factors=10,
    factor_criterion="IC2",
)

result = model.lr_test(
    alpha=0.05,
    classify=True,
    n_sim=5000,
    random_state=12345,
)

print(result["reject_null"])
print(result["break_point"])
print(result.get("break_type"))
```
Use `estimate_breakpoint()` when a break is assumed to exist and significance
testing is not required. In `lr_test()`, `break_point` is returned when the
no-break null is rejected, while `estimated_break_candidate` records the
profile optimizer.

## Multiple-break analysis

```python
from fmbqml import MultiBreakQML

model = MultiBreakQML(
    panel_data,
    max_break=5,
    max_factors=8,
    factor_criterion="IC2",
    trim_ratio=0.10,
)

result = model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
)

print(result["n_breaks"])
print(result["breakpoints"])
print(result.get("break_types"))
```

The option `min_break=0` permits selection of a no-break model. Use
`joint_sup_lr_test(n_breaks=...)` to test no structural change against an exact
number of breaks.

## Factor-number selection

`factor_criterion` accepts `IC1`, `IC2`, `IC3`, `PC1`, `PC2`, `PC3`, `AIC3`,
or `BIC3`. Integer indices `0` through `7` are also accepted in that order.
The argument `max_factors` is a search bound and should be smaller than
`min(T, N)`.

## Results and model state

Result dictionaries contain NumPy arrays for available profiles and nested
structures for breakpoint classifications and information-criterion paths.
For example, single-break procedures store candidate break locations together
with LR or QML profile values, while multiple-break joint estimation stores
the selected breakpoints and the information-criterion path used to choose the
number of breaks.

Public result methods return independent copies of the cached results, so
changing a returned object does not modify the model's internal state. Calling
`set_data()` or `set_parameters()` validates the new values and clears previous
estimates. Retrieve results with `get_test_results()`, `get_joint_results()`,
or `get_joint_test_results()` after running the corresponding procedure.

## Reproducibility

All stochastic procedures accept an explicit random seed. For reproducible
critical values, pass `random_state` to `lr_test()`, `joint_sup_lr_test()`, and
profile plotting routines that simulate critical values.

Example scripts demonstrating the main functionality of `fmbqml` are available
in the project repository:

- [Examples](https://github.com/JiangtaoDuan/fmbqml/tree/master/examples)

## Dependencies

`fmbqml` depends on NumPy, SciPy, pandas, matplotlib, and joblib. Python 3.8 or
newer is required.


## License

`fmbqml` is distributed under the MIT License. See
[LICENSE.txt](https://github.com/JiangtaoDuan/fmbqml/blob/master/LICENSE.txt).
