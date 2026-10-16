
# fmbqml: Structural Break Analysis in High-Dimensional Factor Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/fmbqml.svg)](https://pypi.org/project/fmbqml/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

`fmbqml` is a Python package for quasi-maximum-likelihood analysis of structural breaks in high-dimensional factor models. It provides tools for single-break testing and estimation, joint multiple-break estimation, sup-LR testing, factor-number selection, break-type classification, and diagnostic plots.

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

The input data should be a finite two-dimensional NumPy array or pandas DataFrame with observations in rows and variables in columns. Missing values should be imputed before constructing a model.

## Single-break analysis

This example generates a panel with 300 observations and 100 variables. The factor loadings change after observation 100. Rows represent observations in time order, and columns represent variables.

```python
import numpy as np
from fmbqml import SingleBreakQML

# Generate a panel with one loading break.
rng = np.random.default_rng(2024)

T, N, r = 300, 100, 3
true_break = 100

factors = rng.normal(size=(T, r))
loadings_before = rng.normal(size=(N, r))
loadings_after = loadings_before + rng.normal(size=(N, r))
errors = rng.normal(scale=np.sqrt(r), size=(T, N))

panel_data = np.empty((T, N))
panel_data[:true_break] = (
    factors[:true_break] @ loadings_before.T
    + errors[:true_break]
)
panel_data[true_break:] = (
    factors[true_break:] @ loadings_after.T
    + errors[true_break:]
)

# Test the null hypothesis of no structural break.
model = SingleBreakQML(
    panel_data,
    trim_ratio=0.15,
    max_factors=10,
    factor_criterion="IC2",
)

model.lr_test(
    alpha=0.05,
    classify=True,
    verbose=False,
    n_sim=1000,
    random_state=12345,
)

# Display the summary and retrieve individual results.
print(model.summary())
print("Reject the no-break null:", model.is_rejected())
print("Estimated breakpoint:", model.get_breakpoint())
print("Break type:", model.get_break_type())
```

A breakpoint at `k` separates the first `k` observations from the remaining observations. If the LR test does not reject the no-break null, `get_breakpoint()` returns `None`.

Use `estimate_breakpoint()` when one break is assumed to exist and only its location needs to be estimated. This estimation procedure does not perform a significance test.

## Multiple-break analysis

This example generates a panel with loading changes after observations 100 and 200. The number and locations of the breaks are estimated from the panel data.

```python
import numpy as np
from fmbqml import MultiBreakQML

# Generate a panel with two loading breaks.
rng = np.random.default_rng(42)

T, N, r = 300, 100, 3
break_1, break_2 = 100, 200

factors = rng.normal(size=(T, r))
loadings_1 = rng.normal(size=(N, r))
loadings_2 = rng.normal(size=(N, r))
loadings_3 = rng.normal(size=(N, r))
errors = rng.normal(size=(T, N))

panel_data = np.empty((T, N))
panel_data[:break_1] = (
    factors[:break_1] @ loadings_1.T
    + errors[:break_1]
)
panel_data[break_1:break_2] = (
    factors[break_1:break_2] @ loadings_2.T
    + errors[break_1:break_2]
)
panel_data[break_2:] = (
    factors[break_2:] @ loadings_3.T
    + errors[break_2:]
)

# Select the number of breaks and estimate their locations.
model = MultiBreakQML(
    panel_data,
    max_break=3,
    max_factors=10,
    factor_criterion="IC2",
    trim_ratio=0.10,
)

model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
    verbose=False,
)

# Display the summary and retrieve individual results.
print(model.summary())
print("Selected number of breaks:", model.get_n_breaks())
print("Estimated breakpoints:", model.get_breakpoints())
print("Break types:", model.get_break_types())

# Retrieve and display the information-criterion path.
ic_path = model.get_ic_path()
print(ic_path)

# Continue with the model constructed above.
# Test no break against an alternative with two prespecified breaks.
model.joint_sup_lr_test(
    n_breaks=2,
    alpha=0.05,
    n_sim=1000,
    random_state=123,
    classify=True,
    verbose=False,
)

print(model.summary(mode="joint_lr"))
print("Reject the no-break null:", model.is_rejected())

if model.is_rejected():
    print(
        "Estimated breakpoints:",
        model.get_breakpoints(mode="joint_lr"),
    )
    print(
        "Break types:",
        model.get_break_types(mode="joint_lr"),
    )
```

`max_break` sets the maximum number of candidate breaks, while `min_break=0` allows selection of a no-break model. The true break locations used to generate the data are not supplied to the estimator.

## Factor-number selection

`factor_criterion` accepts `IC1`, `IC2`, `IC3`, `PC1`, `PC2`, `PC3`, `AIC3`, or `BIC3`. Integer indices `0` through `7` are also accepted in that order.

The argument `max_factors` must be a positive integer and specifies the upper bound for factor-number selection. It must not exceed `min(T, N) - 1`.

## Results and model state

The package provides dedicated methods for accessing commonly used results without directly inspecting internal result dictionaries.

For single-break analysis, `is_rejected()` returns the LR test decision, `get_breakpoint()` returns the estimated breakpoint, and `get_break_type()` returns its classification. If the LR test does not reject the no-break null, `get_breakpoint()` returns `None`.

For multiple-break analysis, `get_n_breaks()` returns the number of breaks selected by joint estimation, while `get_breakpoints()` and `get_break_types()` return the estimated break locations and their classifications. The `get_ic_path()` method returns the information-criterion path as a pandas DataFrame. For joint sup-LR testing, `is_rejected()` returns the test decision, and the breakpoint locations and types can be accessed by specifying `mode="joint_lr"`.

The `summary()` method provides a formatted overview of the results. Diagnostic profiles can be accessed through dedicated methods and plotting functions.

For advanced analysis, complete result dictionaries remain available through `get_test_results()`, `get_joint_results()`, and `get_joint_test_results()`. These dictionaries contain additional information, including LR and QML profiles, breakpoint classifications, and information-criterion paths. The result accessors return independent copies of the stored results, preventing modifications to the model's internal state.

Calling `set_data()` or `set_parameters()` validates the new values and clears previously computed results.

## Reproducibility

Monte Carlo procedures used to approximate LR critical values support reproducibility through the `random_state` argument. Users can specify `random_state` when calling `lr_test()`, `joint_sup_lr_test()`, or LR profile plotting methods that perform Monte Carlo simulations.

The examples use fixed random seeds for data generation and critical-value simulation to facilitate reproducibility.

Example scripts demonstrating the main functionality of `fmbqml` are available in the project repository:

- [Examples](https://github.com/JiangtaoDuan/fmbqml/tree/master/examples)

## Dependencies

`fmbqml` depends on NumPy, SciPy, pandas, matplotlib, and joblib. Python 3.8 or newer is required.

## License

`fmbqml` is distributed under the GNU General Public License version 3 (GPLv3).
See [LICENSE.txt](https://github.com/JiangtaoDuan/fmbqml/blob/master/LICENSE.txt).
