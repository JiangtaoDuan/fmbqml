# fmbqml

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/fmbqml)](https://pypi.org/project/fmbqml/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue?style=flat-square)](LICENSE.txt)
## Introduction

`fmbqml` is a Python package for detecting and estimating structural breaks in high-dimensional factor models using quasi-maximum likelihood (QML) methods. It provides procedures for single-break testing and estimation, joint multiple-break estimation, joint sup-LR testing, and break-type classification.

The package accepts two-dimensional NumPy arrays and pandas DataFrames, with time observations in rows and variables in columns. It is designed for researchers and practitioners studying structural changes in high-dimensional panel data.

## Features

| Feature | Description |
| --- | --- |
| Single-break LR test | Tests the null hypothesis of no structural break against a single-break alternative. Candidate breakpoints under the alternative are retained in the result summary; a breakpoint is reported as accepted only when the null is rejected. |
| Single-break QML estimation | Directly estimates a breakpoint under the assumption that one break is present, without performing an LR test. |
| Multiple-break QML estimation | Jointly selects the number and locations of breaks using a QML information criterion. The no-break specification can also be selected. |
| Joint sup-LR test | Tests the no-break null against an alternative with a prespecified number of breaks. |
| Break-type classification | Optionally classifies estimated or tested breakpoints. |
| Result reporting and plots | Provides concise printed results, detailed statistical summaries, dictionary representations, and diagnostic plots of estimation and test profiles. |

## Installation

Install the released package from PyPI:

```bash
python -m pip install fmbqml
```

The package requires Python 3.8 or later. Its runtime dependencies are NumPy, SciPy, pandas, Matplotlib, and joblib.

To run the test suite from a source checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Quickstart

The examples below use a finite panel-data matrix `X` with shape `(T, N)`, where rows are time observations and columns are variables. NumPy arrays and pandas DataFrames are supported. Handle missing and infinite values before fitting a model.

### Single-break analysis

The following example generates panel data with a change in factor loadings and performs a single-break LR test.

```python
import numpy as np
from fmbqml import SingleBreakQML

# Generate panel data with a change in factor loadings.
rng = np.random.default_rng(2024)
T, N, r = 120, 30, 2
true_break = 60

factors = rng.normal(size=(T, r))
loadings_before = rng.normal(size=(N, r))
loadings_after = loadings_before + 0.5 * rng.normal(size=(N, r))
errors = rng.normal(scale=0.5, size=(T, N))

X = np.empty((T, N))
X[:true_break] = (
    factors[:true_break] @ loadings_before.T
    + errors[:true_break]
)
X[true_break:] = (
    factors[true_break:] @ loadings_after.T
    + errors[true_break:]
)

# Test the null hypothesis of no structural break.
model = SingleBreakQML(
    X,
    trim_ratio=0.15,
    max_factors=5,
)

result = model.lr_test(
    alpha=0.05,
    classify=True,
    n_sim=1000,
    random_state=12345,
)

print(result)
print(result.summary())

if result.reject_null:
    print("Accepted breakpoint:", result.breakpoint)
    print("Breakpoint type:", result.break_type)
else:
    print("The no-break null was not rejected.")
    print("Candidate breakpoint under H1:", result.candidate_breakpoint)
```

The LR test evaluates the null hypothesis of no structural break against a single-break alternative. The candidate breakpoint under the alternative is retained in the detailed summary. A breakpoint is reported as accepted only when the null hypothesis is rejected.

If a single break is assumed to exist and only its location is required, use `model.estimate_breakpoint()`. This direct QML estimation procedure does not perform an LR test.

### Multiple-break analysis

The following example generates panel data with two changes in factor loadings. It first estimates the number and locations of breaks jointly and then performs a joint sup-LR test against an alternative with two breaks.

```python
import numpy as np
from fmbqml import MultiBreakQML

# Generate panel data with two changes in factor loadings.
rng = np.random.default_rng(2025)
T, N, r = 150, 40, 2
break_1, break_2 = 50, 100

factors = rng.normal(size=(T, r))
loadings_1 = rng.normal(size=(N, r))
loadings_2 = rng.normal(size=(N, r))
loadings_3 = rng.normal(size=(N, r))
errors = rng.normal(scale=0.5, size=(T, N))

X = np.empty((T, N))
X[:break_1] = (
    factors[:break_1] @ loadings_1.T
    + errors[:break_1]
)
X[break_1:break_2] = (
    factors[break_1:break_2] @ loadings_2.T
    + errors[break_1:break_2]
)
X[break_2:] = (
    factors[break_2:] @ loadings_3.T
    + errors[break_2:]
)

model = MultiBreakQML(
    X,
    max_break=3,
    max_factors=5,
    trim_ratio=0.10,
)

# Joint QML estimation with information-criterion-based break selection.
estimate = model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
)

print(estimate)
print(estimate.summary())
print("Selected number of breaks:", estimate.n_breaks)
print("Estimated breakpoints:", estimate.breakpoints)
print("Breakpoint types:", estimate.break_types)

# Test no structural break against an alternative with two breaks.
test = model.joint_sup_lr_test(
    n_breaks=2,
    alpha=0.05,
    n_sim=1000,
    random_state=12345,
    classify=True,
)

print(test)
print(test.summary())
print("Candidate breakpoints under H1:", test.candidate_breakpoints)
print("Accepted breakpoints:", test.breakpoints)
```

In the joint sup-LR test, candidate breakpoints selected under the alternative are distinct from accepted breakpoints. The candidate locations remain available in the result, while accepted breakpoints are reported only when the no-break null hypothesis is rejected.

## Examples

Runnable Python scripts and Jupyter notebooks are available in the GitHub [`examples/`](https://github.com/JiangtaoDuan/fmbqml/tree/master/examples) directory. They cover the package's single-break and multiple-break workflows, including the FRED-MD empirical application.

The Sphinx documentation source is available in the GitHub [`docs/`](https://github.com/JiangtaoDuan/fmbqml/tree/master/docs) directory. It includes the user guide, API reference, software-design description, and validation information.

## References

To cite the software, see [CITATION.cff](CITATION.cff). Please also cite
the methodology papers relevant to your analysis.

The following publications provide background on high-dimensional factor models and structural-break estimation:

- Duan, J., Bai, J., & Han, X. (2026). Taxonomy and Estimation of Multiple Breakpoints in High-Dimensional Factor Models. *Journal of the American Statistical Association*. https://doi.org/10.1080/01621459.2026.2719782
- Duan, J., Bai, J., & Han, X. (2023). Quasi-maximum likelihood estimation of break point in high-dimensional factor models. *Journal of Econometrics*, 233(1), 209–236. https://doi.org/10.1016/j.jeconom.2021.12.011
- Bai, J., Duan, J., & Han, X. (2024). The likelihood ratio test for structural changes in factor models. *Journal of Econometrics*, 238(2), 105631. https://doi.org/10.1016/j.jeconom.2023.105631
- Bai, J. (2003). Inferential Theory for Factor Models of Large Dimensions. *Econometrica*, 71(1), 135–171. https://doi.org/10.1111/1468-0262.00392

## License

`fmbqml` is distributed under the GNU General Public License, version 3 or (at your option) any later version (GPL-3.0-or-later). See [LICENSE.txt](LICENSE.txt) for the full license.
