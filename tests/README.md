# Tests

The test suite is organized by the package's main functional areas. Related
low-level checks are grouped together rather than split into very small files,
while the two main model families retain dedicated test areas.

```text
tests/
├── conftest.py
├── test_workflows.py
├── single_break/
│   ├── _helpers.py
│   ├── test_model.py
│   ├── test_estimation.py
│   ├── test_lr_test.py
│   └── test_results.py
├── multiple_break/
│   ├── _helpers.py
│   ├── test_model.py
│   ├── test_joint_estimation.py
│   ├── test_joint_testing.py
│   └── test_results.py
├── break_tools/
│   └── test_classification.py
├── factor_tools/
│   └── test_factor_tools.py
├── statistical_tools/
│   └── test_statistical_tools.py
├── utils/
│   └── test_utils.py
└── audit_numerical_verification.py
```

## Test organization

- `single_break/` and `multiple_break/` independently cover their model APIs,
  estimation, LR testing, and result reporting.
- `break_tools/` tests structural-break classification independently of the
  model-level classification interfaces.
- `factor_tools/`, `statistical_tools/`, and `utils/` group related lower-level
  routines into cohesive test modules.
- `test_workflows.py` contains seeded end-to-end examples with known
  structural changes.
- `_helpers.py` modules contain deterministic synthetic data and test doubles
  shared within their respective model areas.
- `audit_numerical_verification.py` is an independent diagnostic script and is
  intentionally not collected as a pytest test module.

The suite avoids splitting every small helper into a separate test file.
Tests use fixed random seeds and check observable behavior, mathematical
properties, or known results. This organization is intended to keep the
suite navigable without sacrificing coverage.

## Running tests

Run the complete suite:

```bash
python -m pytest -q
```

Run tests with warnings treated as errors:

```bash
python -m pytest -q -W error
```

Run a focused test module:

```bash
python -m pytest tests/single_break/test_lr_test.py -q
```

Run the independent numerical diagnostic:

```bash
python -m tests.audit_numerical_verification
```
