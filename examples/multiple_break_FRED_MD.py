"""FRED-MD empirical application with fmbqml.

Demonstrates:
- Empirical multiple-break analysis using the November 2025 FRED-MD vintage
- Baseline information-criterion path
- Sensitivity analysis for the maximum candidate number of breaks
- The extended information-criterion path in Appendix C

Related paper sections: Section 5.5 and Appendix C.

The source example includes the FRED-MD data and preprocessing routines in
examples/fred_md/. These example files are not part of the installed package.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow this source example to run directly from any working directory.
source_root = Path(__file__).resolve().parent.parent
if (source_root / "fmbqml" / "__init__.py").is_file():
    sys.path.insert(0, str(source_root))

from fmbqml import MultiBreakQML

print("\n" + "=" * 72)
print("FRED-MD empirical application with fmbqml")
print("=" * 72)

def find_replication_root():
    """Locate bundled data and preprocessing modules without changing the data."""
    candidates = []
    override = os.environ.get("FMBQML_REPLICATION_ROOT")
    if override:
        override_path = Path(override).expanduser()
        candidates.extend([override_path, override_path / "examples"])
    candidates.extend([Path(__file__).resolve().parent, *Path(__file__).resolve().parents])
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for root in candidates:
        fred_dir = root / "fred_md"
        required = [fred_dir / "data" / "FRED_MD_2025_11.csv"]
        required.extend(fred_dir / name for name in (
            "__init__.py", "factors_em.py", "remove_outliers.py", "prepare_missing.py"
        ))
        if all(path.is_file() for path in required):
            return root.resolve()
    raise FileNotFoundError(
        "Could not find the FRED-MD data and preprocessing modules. "
        "Ensure examples/fred_md/ is present, or set FMBQML_REPLICATION_ROOT "
        "to the directory containing fred_md/ or to the project root."
    )

PROJECT_ROOT = find_replication_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fred_md.factors_em import factors_em
from fred_md.prepare_missing import prepare_missing
from fred_md.remove_outliers import remove_outliers

# 5.5. Prepare the FRED-MD November 2025 vintage
print("\n[1/5] Loading and preprocessing the November 2025 FRED-MD vintage...")

# Load the November 2025 vintage of the FRED-MD dataset.
fred_file = PROJECT_ROOT / "fred_md" / "data" / "FRED_MD_2025_11.csv"

if not fred_file.exists():
    raise FileNotFoundError(
        f"FRED-MD vintage file not found: {fred_file}. "
        "Ensure the source example's fred_md/data directory is present."
    )

fred_df = pd.read_csv(fred_file)

# Extract the transformation codes and raw observations.
transform_codes = (
    fred_df.iloc[0, 1:].astype(float).astype(int).to_numpy()
)

raw_data = (
    fred_df.iloc[1:, 1:]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)

# Construct the monthly calendar range based on the final date in the dataset.
date_column = fred_df.columns[0]
date_strings = fred_df[date_column].astype(str)
final_date = pd.to_datetime(date_strings.iloc[-1], errors="coerce")

if pd.isna(final_date):
    raise ValueError("Could not parse the final date in the FRED-MD vintage.")

monthly_dates = pd.date_range(
    start="1959-01-01",
    end=f"{final_date.year:04d}-{final_date.month:02d}-01",
    freq="MS",
)

num_raw_dates = len(monthly_dates)
raw_data = raw_data[:num_raw_dates, :]

# Apply the FRED-MD transformation codes and remove the initial
# observations lost through differencing.
transformed_data = prepare_missing(raw_data, transform_codes)
transformed_data = transformed_data[2:num_raw_dates, :]

# Treat outliers in the transformed panel.
clean_data, outlier_counts = remove_outliers(transformed_data)

# Set the factor-selection and demeaning options for EM completion.
max_factors = 8
factor_ic_method = 2
demean_method = 2

# Estimate the factor structure and obtain the completed panel.
_, _, _, _, filled_data = factors_em(
    clean_data,
    max_factors,
    factor_ic_method,
    demean_method,
)

# Standardize each series using its sample mean and standard deviation.
sample_size, num_series = filled_data.shape

column_means = np.mean(filled_data, axis=0)
column_stds = np.std(filled_data, axis=0, ddof=1)
column_stds[column_stds == 0] = 1.0

standardized_data = (filled_data - column_means) / column_stds

# Construct monthly labels corresponding to the retained observations,
# starting in March 1959 after removing the first two observations.
calendar_labels = []

for time_index in range(1, sample_size + 1):
    month_offset = 2 + (time_index - 1)
    year = 1959 + month_offset // 12
    month = month_offset % 12 + 1
    calendar_labels.append(f"{year:04d}-{month:02d}")

# 5.5. Construct the indexed FRED-MD panel and estimate breakpoints
print("\n[2/5] Estimating structural breaks in the FRED-MD panel...")

qml_panel = pd.DataFrame(
    standardized_data,
    index=calendar_labels,
)

min_segment_length = 20
max_breaks = 5
max_factors = 8

model = MultiBreakQML(
    qml_panel,
    max_break=max_breaks,
    max_factors=max_factors,
    factor_criterion="IC2",
    trim_ratio=min_segment_length / len(qml_panel),
)

result = model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
)
print(result.summary())

# 5.5. Display the baseline information-criterion path
print("\n[3/5] Displaying the baseline information-criterion path...")

print(result.format_ic_path_table())

# 5.5. Sensitivity to the maximum candidate number of breaks
print("\n[4/5] Running sensitivity analysis for the maximum candidate number of breaks...")

sensitivity_grid = (4, 5, 6, 7)
sensitivity_rows = []
extended_fred_result = None

for candidate_max_break in sensitivity_grid:
    if candidate_max_break == max_breaks:
        # Reuse the baseline estimate rather than fitting max_break=5 twice.
        candidate_result = result
    else:
        sensitivity_model = MultiBreakQML(
            qml_panel,
            max_break=candidate_max_break,
            max_factors=max_factors,
            factor_criterion="IC2",
            trim_ratio=min_segment_length / len(qml_panel),
        )
        classify_for_appendix = candidate_max_break == 7
        candidate_result = sensitivity_model.estimate_breaks_jointly(
            min_break=0,
            classify=classify_for_appendix,
        )
        if classify_for_appendix:
            extended_fred_result = candidate_result

    sensitivity_rows.append({
        "max_break": candidate_max_break,
        "Selected number of breaks": candidate_result.n_breaks,
        "Estimated breakpoints": candidate_result.breakpoints,
        "Estimated break dates": candidate_result.break_labels,
    })

sensitivity_table = pd.DataFrame(sensitivity_rows)
print("\nSensitivity to the maximum candidate number of breaks:")
print(sensitivity_table.to_string(index=False))

# Appendix C. Extended FRED-MD information-criterion path
print("\n[5/5] Computing the extended information-criterion path for Appendix C...")

# Appendix C is executed in every complete tutorial run. The max_break=7
# fit from the Section 5.5 sensitivity analysis is reused when available.
if extended_fred_result is not None:
    fred_results = extended_fred_result
else:
    fred_model = MultiBreakQML(
        qml_panel,
        max_break=7,
        max_factors=8,
        factor_criterion="IC2",
        trim_ratio=min_segment_length / len(qml_panel),
    )
    fred_results = fred_model.estimate_breaks_jointly(
        min_break=0,
        classify=True,
    )
print(fred_results.summary())

print("\nExtended information-criterion path (m = 0,...,7):")
print(fred_results.format_ic_path_table())

appendix_breakpoints = fred_results.breakpoints or []
appendix_labels = fred_results.break_labels or []
appendix_types = fred_results.break_types or []

print("\nSelected model under the extended candidate range:")
print("  Number of breaks:", fred_results.n_breaks)
print("  Breakpoints:", appendix_breakpoints if appendix_breakpoints else "None")
print("  Break dates:", appendix_labels if appendix_labels else "None")

if appendix_breakpoints:
    classification_rows = []
    for index, breakpoint in enumerate(appendix_breakpoints):
        classification_rows.append({
            "Breakpoint": int(breakpoint),
            "Date": (
                appendix_labels[index]
                if index < len(appendix_labels) else ""
            ),
            "Type": (
                appendix_types[index]
                if index < len(appendix_types) else ""
            ),
        })
    print("\nBreak-type classification under the extended candidate range:")
    print(pd.DataFrame(classification_rows).to_string(index=False))
else:
    print("\nNo breakpoints were selected under the extended candidate range.")

