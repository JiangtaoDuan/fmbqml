"""Multiple-break detection in the FRED-MD dataset.

Converted from multiple_break_FRED_MD.ipynb. Run from this examples/ directory
(for example, ``python multiple_break_FRED_MD.py``) with fmbqml installed.
"""

# %% [markdown]
# # Multiple-Break Detection in the FRED-MD Dataset#
# We apply the proposed method to the FRED-MD (Federal Reserve Economic Data - Monthly Data) dataset to illustrate its empirical performance in detecting structural breakpoints in macroeconomic time series. The FRED database is maintained by the Research Division of the Federal Reserve Bank of St. Louis, and the data are publicly available and continuously updated. The raw sample contains 126 monthly macroeconomic series spanning from January 1959 to November 2025; the processed balanced panel is used for estimation. According to the standard FRED-MD classification, these series are grouped into eight economic categories: output and income, labor market, housing, consumption (including orders and inventories), money and credit, interest and exchange rates, prices, and the stock market.

# %% Code cell 1
import numpy as np
import pandas as pd

from fred.remove_outliers import remove_outliers
from fred.prepare_missing import prepare_missing
from fred.factors_em import factors_em
from fmbqml import MultiBreakQML

# %% [markdown]
# Using the FRED-MD macroeconomic database, this study first preprocesses the raw time-series data. Missing values in the original dataset are imputed using the expectation-maximization (**EM**) algorithm. Subsequently, variable-specific transformation rules, such as differencing or log-differencing, are applied to obtain approximately stationary time series, and outlying observations are adjusted accordingly. After these preprocessing steps, each macroeconomic variable contains a balanced monthly sample of length (T=801). Next, the information criteria proposed by Bai and Ng (2002) are employed to determine the optimal number of common factors within a prespecified maximum-factor range; in this study, the **IC2** criterion is used. Once the number of factors is selected, the latent common factors are estimated using a principal components approach combined with the **EM** algorithm, which yields stable factor series even in the presence of missing observations. The resulting factor matrix is then organized with time as the index and used for subsequent econometric analysis. For detailed descriptions of the construction of the FRED-MD dataset and the variable transformation procedures, see McCracken and Ng (2016). The first five observations of the processed data are also reported in the paper for illustration.

# %% Code cell 2
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
input_csv = EXAMPLES_DIR / "data" / "FRED_MD_2025_11.csv"

demean_method = 2
factor_ic_method = 2
max_factors = 8
min_segment_length = 20
max_breaks = 5

# ========================================================================
# PART 1: LOAD AND LABEL DATA
# ========================================================================

fred_df = pd.read_csv(input_csv)

transform_codes = fred_df.iloc[0, 1:].astype(float).astype(int).to_numpy()
raw_data = fred_df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

date_column = fred_df.columns[0]
date_strings = fred_df[date_column].astype(str)

final_date = pd.to_datetime(date_strings.iloc[-1], errors="coerce")
if pd.isna(final_date):
    raise ValueError("Cannot parse final date from CSV first column.")

final_month = final_date.month
final_year = final_date.year

monthly_dates = pd.date_range(
    start="1959-01-01",
    end=f"{final_year:04d}-{final_month:02d}-01",
    freq="MS"
)

num_raw_dates = len(monthly_dates)
raw_data = raw_data[:num_raw_dates, :]

# ========================================================================
# PART 2: PROCESS DATA
# ========================================================================

transformed_data = prepare_missing(raw_data, transform_codes)

transformed_data = transformed_data[2:num_raw_dates, :]

clean_data, outlier_counts = remove_outliers(transformed_data)

# ========================================================================
# PART 3: EM IMPUTATION
# ========================================================================

residuals, em_factors, loadings, eigenvalues, filled_data = factors_em(
    clean_data,
    max_factors,
    factor_ic_method,
    demean_method
)


# ========================================================================
# PART 4: FULL-SAMPLE STANDARDIZATION
# ========================================================================

em_filled_data = filled_data
sample_size, num_series = em_filled_data.shape

column_means = np.mean(em_filled_data, axis=0)
column_stds = np.std(em_filled_data, axis=0, ddof=1)
column_stds[column_stds == 0] = 1.0

standardized_data = (em_filled_data - column_means) / column_stds

print("\n============================================================")
print("FRED-MD empirical application with MultiBreakQML")
print(f"Vintage file: {input_csv}")
print(f"After EM + outlier replacement: T = {sample_size}, N = {num_series}")
print(
    f"Minimum segment length h = {min_segment_length} months, "
    f"trim_ratio = {min_segment_length / sample_size:.4f}"
)
print(f"Candidate #breaks: m = 0,1,...,{max_breaks}")
print("============================================================\n")


# ========================================================================
# PART 5: CREATE CALENDAR INDEX
# ========================================================================

start_year = 1959
start_month = 3

calendar_labels = []
for time_index in range(1, sample_size + 1):
    month_offset = (start_month - 1) + (time_index - 1)
    year = start_year + month_offset // 12
    month = month_offset % 12 + 1
    calendar_labels.append(f"{year:04d}-{month:02d}")

# %% [markdown]
# After obtaining the EM-imputed and standardized panel, the `MultiBreakQML` procedure is employed to detect structural changes in its latent factor structure. Specifically, the standardized panel is used to construct a `MultiBreakQML` object, with the maximum number of allowable breakpoints set to `max_break=5`, the upper bound on the number of candidate factors set to `max_factors=8`, `factor_criterion=1` (i.e., **IC2**) adopted for factor-dimension selection, and `trim_ratio=20/T` corresponding to a minimum segment length of approximately 20 observations. Subsequently, the method `estimate_breaks_jointly(min_break=0, classify=True)` is called to jointly estimate the breakpoint locations and model dimension. Here, setting `min_break=0` allows for the possibility of no structural break, so the procedure can simultaneously determine whether structural breaks are present in the data. The object-level `trim_ratio` parameter controls the minimum admissible segment length used in model selection and breakpoint determination. Since `classify=True`, the method also classifies the type of each estimated breakpoint. Ultimately, the procedure returns the estimated set of breakpoints, the corresponding number of breakpoints, and the type of each breakpoint, thereby providing a basis for subsequent segmented analysis and the study of structural changes.

# %% Code cell 3: Joint breakpoint estimation
qml_panel = pd.DataFrame(standardized_data, index=calendar_labels)


model = MultiBreakQML(
    qml_panel,
    max_break=max_breaks,
    max_factors=max_factors,
    factor_criterion="IC2",
    trim_ratio=min_segment_length / len(qml_panel)
)

result = model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
)
print(result.summary())

# %% Code cell 4: Breakpoint dates and classification details
# Access commonly used results through dedicated public attributes.
print("\nFinal result:")
print("Number of breaks:", result.n_breaks)
print("Breakpoints:", result.breakpoints)
print("Break dates:", result.break_labels or [])
print("Breakpoint types:", result.break_types)

# The detailed factor counts are advanced classification diagnostics.
# No dedicated getter is currently provided for these fields.
classification_details = result.to_dict()
print("Regime factors:", classification_details.get("regime_factors"))
print("Combined factors:", classification_details.get("combined_factors"))

# %% [markdown]
# ## Main text, Section 5.5: Sensitivity to the maximum number of breaks
# Following Table 10, the baseline specification is compared with max_break values
# of 4, 5, 6, and 7. The preprocessing, factor-selection settings, and minimum
# segment length are held fixed.

# %% Code cell 5: Main-text sensitivity analysis
from pathlib import Path

max_break_grid = (4, 5, 6, 7)
max_break_sensitivity_results = {}
max_break_sensitivity_rows = []

for candidate_max_break in max_break_grid:
    if candidate_max_break == max_breaks:
        sensitivity_model = model
        sensitivity_result = result
    else:
        sensitivity_model = MultiBreakQML(
            qml_panel,
            max_break=candidate_max_break,
            max_factors=max_factors,
            factor_criterion="IC2",
            trim_ratio=min_segment_length / len(qml_panel),
        )
        sensitivity_result = sensitivity_model.estimate_breaks_jointly(
            min_break=0,
            classify=False,
        )

    max_break_sensitivity_results[candidate_max_break] = sensitivity_result
    estimated_breakpoints = sensitivity_result.breakpoints
    estimated_dates = [qml_panel.index[k - 1] for k in estimated_breakpoints]

    max_break_sensitivity_rows.append({
        "max_break": candidate_max_break,
        "Selected number of breaks": sensitivity_result.n_breaks,
        "Estimated break dates": ", ".join(map(str, estimated_dates)),
    })

max_break_sensitivity_table = pd.DataFrame(max_break_sensitivity_rows)
print("\\nSensitivity analysis (main-text Table 10):")
print(max_break_sensitivity_table.to_string(index=False))

table_output_dir = EXAMPLES_DIR / "output" / "tables"
table_output_dir.mkdir(parents=True, exist_ok=True)
csv_path = table_output_dir / "fred_md_max_break_sensitivity.csv"
latex_path = table_output_dir / "fred_md_max_break_sensitivity.tex"

max_break_sensitivity_table.to_csv(csv_path, index=False)
max_break_sensitivity_table.to_latex(
    latex_path,
    index=False,
    caption="Sensitivity of FRED-MD break estimates to the maximum number of breaks.",
    label="tab:fred_md_max_break_sensitivity",
)

print(f"CSV table saved to: {csv_path}")
print(f"LaTeX table saved to: {latex_path}")


# %% [markdown]
# ## Appendix C: Additional results for the FRED-MD case
# Appendix C expands the candidate number of breaks to m = 0,...,7 while keeping
# T, N, max_factors, IC2, and the minimum segment length unchanged. It reports
# the complete information-criterion path and the classification of the selected
# five breakpoints. No figures are generated.

# %% Code cell 6: Appendix C - expanded IC path and classification
appendix_max_break = 7

appendix_model = MultiBreakQML(
    qml_panel,
    max_break=appendix_max_break,
    max_factors=max_factors,
    factor_criterion="IC2",
    trim_ratio=min_segment_length / len(qml_panel),
)

appendix_result = appendix_model.estimate_breaks_jointly(
    min_break=0,
    classify=True,
)

print("\\nAppendix C - joint-estimation summary:")
print(appendix_result.summary())

print("\\nAppendix C - complete information-criterion path (m = 0,...,7):")
print(appendix_result.format_ic_path_table())

appendix_breakpoints = appendix_result.breakpoints
appendix_break_dates = [qml_panel.index[k - 1] for k in appendix_breakpoints]

print("\\nAppendix C - selected breakpoints and dates:")
for break_number, (location, date_label) in enumerate(
    zip(appendix_breakpoints, appendix_break_dates), start=1
):
    print(f"Breakpoint {break_number}: location={location}, date={date_label}")

print("\\nAppendix C - break types:")
print(appendix_result.break_types)

# Save the Appendix C IC path for direct use in the paper's supplementary tables.
appendix_ic_path = appendix_result.ic_path.copy()
appendix_ic_csv = table_output_dir / "fred_md_appendix_C_ic_path.csv"
appendix_ic_latex = table_output_dir / "fred_md_appendix_C_ic_path.tex"

appendix_ic_path.to_csv(appendix_ic_csv, index=False)
appendix_ic_path.to_latex(
    appendix_ic_latex,
    index=False,
    caption="Information-criterion path for the FRED-MD case with max_break=7.",
    label="tab:fred_md_appendix_C_ic_path",
)

print(f"Appendix C IC path CSV saved to: {appendix_ic_csv}")
print(f"Appendix C IC path LaTeX saved to: {appendix_ic_latex}")
