# DataGuard AI

**Inspect CSV data quality and measure how specific imputation choices affect a regression model.**

DataGuard AI is a Python application with two workflows:

- **CSV Scanner:** identify missing values and duplicate rows without changing the dataset.
- **ML Impact Lab:** compare mean and median imputation using the same Ridge regression model and identical, seeded cross-validation splits.

Version 0.2 introduces a focused experiment workflow: it reports measured differences for a specific dataset and evaluation setup. It does not automatically rank all data problems or establish a universally best repair.

[View the repository on GitHub](https://github.com/JaberAhmad555/DataGuard-AI)

## Features

### CSV Scanner

- Row and column counts.
- Missing values per column and across the dataset.
- Exact duplicate-row counts after the first occurrence.
- Human-readable text and structured JSON reports.
- Reusable, typed Python functions.
- No runtime dependencies for the base scanner.

### ML Impact Lab

- Numeric features and a user-selected numeric regression target.
- Mean and median imputation through scikit-learn `SimpleImputer`.
- Training-fold-only imputation and standardization inside `Pipeline`.
- Ridge regression with identical settings for both methods.
- Seeded repeated K-fold cross-validation with paired splits.
- Per-fold MAE, mean MAE, sample standard deviation, and baseline differences.
- Dataset and split fingerprints, configuration, and environment versions.
- Explicit errors for unsupported inputs, unusable training folds, and numerical failures.

### Engineering

- Python 3.11 or newer.
- Installable package with a `src/` layout.
- Optional ML dependencies loaded only when needed.
- Automated unit and integration tests.
- Read-only access to source CSV files.

## Installation

The following commands are for Windows PowerShell. Run installation and usage commands from the project root: the folder containing `pyproject.toml`.

If you do not already have a checkout:

```powershell
git clone https://github.com/JaberAhmad555/DataGuard-AI.git
Set-Location .\DataGuard-AI
```

If you already have the project, open PowerShell in its existing folder instead.

### Create a virtual environment

Confirm that `python --version` reports Python 3.11 or newer.

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Choose an installation

For the CSV scanner:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e .
```

For the scanner and ML Impact Lab:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

For development and the complete test suite:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
```

The `ml` extra installs NumPy and scikit-learn, including their dependencies. The `dev` extra installs pytest. The `-e` option installs the local project in editable mode.

Installation may require internet access to obtain dependencies. Scanning, running the included experiments, and executing tests do not require network access after installation.

These commands use the virtual environment's executables directly, so activation and PowerShell execution-policy changes are unnecessary.

In VS Code, use **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`.

## CSV Scanner

Scan the existing employee example:

```powershell
& .\.venv\Scripts\dataguard.exe scan .\employees.csv
```

Generate JSON:

```powershell
& .\.venv\Scripts\dataguard.exe scan .\employees.csv --json
```

The module entry point is also supported:

```powershell
& .\.venv\Scripts\python.exe -m dataguard scan .\employees.csv
```

Inspect command help:

```powershell
& .\.venv\Scripts\dataguard.exe scan --help
```

Quote paths containing spaces. In an activated virtual environment, the equivalent command is `dataguard scan .\employees.csv`.

### Scanner report example

Run the small test fixture:

```powershell
& .\.venv\Scripts\dataguard.exe scan .\tests\fixtures\normal.csv --json
```

Its expected report is:

```json
{
  "row_count": 4,
  "column_count": 3,
  "missing_values_per_column": {
    "name": 1,
    "age": 1,
    "city": 1
  },
  "total_missing_values": 3,
  "duplicate_count": 1
}
```

These are descriptive dataset counts, not ML performance measurements.

## ML Impact Lab

Install the `ml` extra before running experiments.

Compare imputation methods on the included synthetic housing dataset:

```powershell
& .\.venv\Scripts\dataguard.exe impact .\data\examples\housing_regression.csv --target price
```

Set the evaluation configuration explicitly and generate JSON:

```powershell
& .\.venv\Scripts\dataguard.exe impact .\data\examples\housing_regression.csv --target price --folds 5 --repeats 3 --seed 42 --ridge-alpha 1.0 --json
```

The equivalent module command is:

```powershell
& .\.venv\Scripts\python.exe -m dataguard impact .\data\examples\housing_regression.csv --target price --folds 5 --repeats 3 --seed 42 --ridge-alpha 1.0 --json
```

View the available options:

```powershell
& .\.venv\Scripts\dataguard.exe impact --help
```

| Argument | Purpose | Default |
| --- | --- | --- |
| CSV path | Input UTF-8 CSV file | Required |
| `--target` | Exact name of the numeric target column | Required |
| `--folds` | Folds per repetition; at least 2 | `5` |
| `--repeats` | Number of repetitions; at least 1 | `3` |
| `--seed` | Integer from 0 through 4294967295 | `42` |
| `--ridge-alpha` | Positive, finite Ridge regularization strength | `1.0` |
| `--json` | Emit JSON instead of readable text | Text output |

Using `--repeats 1` performs a single K-fold evaluation.

All columns except the selected target become features. There is currently no CLI option for selecting or excluding individual feature columns.

### Synthetic housing example

[data/examples/housing_regression.csv](data/examples/housing_regression.csv) contains **60 fictional housing records**.

| Column | Meaning | Missing values |
| --- | --- | ---: |
| `size_sqft` | Property size in square feet | 4 |
| `bedrooms` | Number of bedrooms | 3 |
| `building_age_years` | Building age in years | 5 |
| `distance_to_center_km` | Distance to a fictional city center in kilometres | 4 |
| `price` | Synthetic price expressed in USD | 0 |

The data includes variation in property sizes, ages, distances, and prices, with larger properties and longer distances introducing skew. Missing feature values are represented by empty CSV cells.

Each feature has at most five missing values. Consequently, every 48-record training fold in a standard five-fold split contains at least 43 observed values for each feature, regardless of seed.

**This synthetic demonstration is not evidence of universal ML performance.** It is intended to demonstrate the workflow and support software testing. Run the experiment to obtain its actual results; the project does not promise that median imputation will outperform mean imputation.

## How the experiment works

### Mean and median imputation

Mean imputation fills a missing feature value with that feature's training-fold average.

Median imputation uses the training-fold median. The median is less sensitive to extreme observed values, but this does not guarantee lower prediction error.

Neither strategy fills missing target values. Missing targets cause an error.

### Ridge regression and preprocessing

Each method uses a fresh pipeline for every fold:

```text
SimpleImputer → StandardScaler → Ridge
```

`StandardScaler` centers and scales the imputed numeric features. Ridge regression fits a linear model with regularization controlled by `--ridge-alpha`.

Both methods use the same Ridge configuration and the deterministic SVD solver. Only the imputation strategy differs. No hyperparameter search is performed.

### Repeated cross-validation

The experiment constructs one sequence of `RepeatedKFold` splits using the selected seed. Both methods reuse those exact training and validation indices.

For example, five folds and three repetitions produce 15 evaluations per method and 30 pipeline fits in total.

Every training fold is validated before fitting begins. If a feature has no observed values in any training fold, the experiment fails explicitly. It does not silently drop the feature, fill it with zero, or skip the fold.

### Leakage prevention

Imputation and scaling are fitted only on the training records within each fold. Held-out validation records are transformed with those fitted parameters.

No feature means, medians, or scaling parameters are learned from the full dataset before cross-validation. The target column is excluded from the feature matrix.

This prevents preprocessing leakage within the implemented protocol. It does not automatically detect target-derived features, repeated entities, temporal relationships, or other forms of leakage in the supplied data.

## Interpreting the results

**Mean absolute error (MAE)** is the average absolute difference between predicted and observed target values. Lower MAE is better, and its units match the target: USD for the synthetic housing example.

Each method reports:

- Its MAE for every validation fold.
- Mean MAE across folds.
- Sample standard deviation across fold MAEs, using `ddof=1`.
- Difference from the mean-imputation baseline.

The aggregate difference is:

```text
method mean MAE − mean-imputation mean MAE
```

A negative difference for median imputation favors median under this experiment. A positive difference favors mean. The baseline's difference from itself is zero.

The report also includes paired differences for individual folds:

```text
median fold MAE − mean-imputation fold MAE
```

Fold MAEs receive equal weight. When validation folds have different sizes, this differs from pooling all individual prediction errors before averaging.

Repeated cross-validation folds are dependent. Their standard deviation is descriptive variability, **not a confidence interval or a statistical significance test**.

The results are a comparison within this evaluation setup. Choosing a method based on these scores does not provide an independent test-set estimate of the selected method's performance.

Text reports display eight significant digits. JSON retains the report's numeric values without additional rounding and rejects non-finite JSON numbers.

## Supported inputs

### Shared CSV rules

- Files must be UTF-8, optionally with a UTF-8 BOM, and comma-separated.
- The first nonempty record is the header.
- Column names must be unique and must not be empty or whitespace-only.
- Header text is otherwise preserved exactly.
- Every data record must contain the same number of fields as the header.
- Standard double-quoted fields, escaped double quotes, embedded commas, and quoted multiline fields are supported through Python's strict CSV reader.
- Empty physical lines outside quoted fields are skipped.
- A record such as `,` in a two-column CSV is a data row containing two empty fields.
- Zero-byte and blank-line-only files are rejected because no header is available.

Delimiter, encoding, and header detection are not implemented. A semicolon-separated file may appear to be a single-column CSV; the application cannot infer the intended delimiter.

### Scanner counting rules

- Empty and whitespace-only fields count as missing.
- Literal `NA`, `N/A`, `null`, `None`, and `NaN` are ordinary strings.
- Duplicates are exact matches of all parsed field strings, counted after the first occurrence. Three identical rows produce a duplicate count of two.
- Quoting differences that decode to identical strings do not distinguish rows.
- Case, whitespace, and numeric spelling remain significant for duplicate detection: `Ada` differs from `Ada `, and `1` differs from `01`.
- Scanner checks inspect strings without normalizing or converting the stored values.
- Row counts exclude the header and skipped blank lines.
- Missing-value counts include duplicate records.
- Header-only CSV files are valid for scanning and report zero rows.

### Additional ML requirements

- The target name must exactly match a header, including case and any whitespace.
- There must be at least one feature column.
- Observed feature values and all target values must convert to finite `float64` numbers.
- Empty or whitespace-only feature fields become `NaN` in the in-memory feature matrix.
- Literal missing-value tokens such as `NA`, `null`, and `NaN` are rejected rather than converted into missing values.
- Missing or constant targets are rejected.
- Entirely missing feature columns are rejected.
- The loader requires at least two records. The selected cross-validation configuration must also be feasible: the fold count cannot exceed the record count, and every training fold must contain at least two records.
- Every feature must have an observed value in every training fold.
- No columns or records are silently dropped.

Numeric target values are treated as regression responses, including integer-valued targets. Numeric category codes cannot reliably be distinguished from genuine regression targets automatically.

The CSV scanner can inspect categorical strings; the ML experiment cannot use those strings as features.

## Reproducibility

Experiment reports include:

- Fold count, repetition count, seed, and Ridge regularization strength.
- Target name, feature order, dataset dimensions, and missing-value counts.
- Per-fold scores and training/validation sizes.
- A SHA-256 fingerprint of the parsed CSV contents and selected target.
- A SHA-256 fingerprint of the ordered cross-validation split indices.
- Python, platform, DataGuard, NumPy, SciPy, and scikit-learn versions.

The data fingerprint represents parsed content, not raw file bytes. It includes column order, row order, original parsed field strings, and target selection. It excludes the file path and serialization differences such as equivalent quoting, a UTF-8 BOM, and skipped blank lines.

The split fingerprint identifies the generated splits; it does not contain the split indices themselves.

To reproduce an experiment, preserve the dataset and row order, use the same settings, and recreate the software environment. Fixed seeds do not guarantee bit-for-bit equality across operating systems, numerical libraries, hardware, or dependency versions.

Dependency ranges in `pyproject.toml` are compatibility constraints, not an environment lock. You can record the installed packages:

```powershell
New-Item -ItemType Directory -Force -Path .\artifacts | Out-Null
& .\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 .\artifacts\environment.txt
```

## Python API

### Scan a CSV

```python
from dataguard import CSVInputError, scan_csv

try:
    report = scan_csv("employees.csv")
except CSVInputError as exc:
    print(f"Cannot scan dataset: {exc}")
else:
    print(report.total_missing_values)
    print(report.to_dict())
```

The individual checks can also be used independently:

```python
from dataguard.checks import count_duplicate_rows, count_missing_values

columns = ["name", "age"]
rows = [("Ada", ""), ("Ada", ""), ("Ben", "30")]

print(count_missing_values(columns, rows))
print(count_duplicate_rows(rows))
```

These functions expect strings. They consume row iterables, so use a reusable collection or a new iterator when running multiple checks.

`count_missing_values` raises `ValueError` for duplicate column names or rows whose widths do not match the supplied columns.

### Run an imputation experiment

```python
from dataguard.experiments import (
    ExperimentConfig,
    ImputationExperimentError,
    RegressionDataError,
    run_imputation_experiment,
)
from dataguard.reporting.impact import (
    format_impact_json,
    format_impact_report,
)

config = ExperimentConfig(
    n_splits=5,
    n_repeats=3,
    random_state=42,
    ridge_alpha=1.0,
)

try:
    report = run_imputation_experiment(
        "data/examples/housing_regression.csv",
        target_column="price",
        config=config,
    )
except (RegressionDataError, ImputationExperimentError) as exc:
    print(f"Cannot run experiment: {exc}")
else:
    print(format_impact_report(report))
    json_report = format_impact_json(report)
```

For an already loaded dataset, use `load_regression_csv()` followed by `compare_imputation()` from `dataguard.experiments`.

Importing the experiments package requires the ML dependencies. The existing scanner and CLI help do not require them.

## Errors and exit codes

| Code | Meaning |
| --- | --- |
| `0` | Successful scan or experiment, or help displayed |
| `1` | CSV input error, experiment failure, or unavailable/incompatible ML imports |
| `2` | Invalid or missing command-line arguments |
| `130` | ML experiment interrupted during execution |

Finding missing values or duplicates does not itself cause a scan to exit unsuccessfully.

Successful reports go to standard output. Errors go to standard error. Input and experiment failures produce an error instead of a partial result report.

The scanner and experiment read source CSVs without writing repairs back to them. The experiment does not save fitted models or corrected datasets.

## Testing

Install the complete development environment:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
```

On Windows, create the artifacts directory and use an explicit temporary-test directory:

```powershell
New-Item -ItemType Directory -Force -Path .\artifacts | Out-Null
& .\.venv\Scripts\python.exe -m pytest --basetemp=./artifacts/pytest_tmp
```

The `pytest_tmp` directory is reserved for disposable test files. Pytest clears its configured base temporary directory on subsequent runs, so do not store reports or other files you want to keep inside it.

Run just the ML integration tests:

```powershell
New-Item -ItemType Directory -Force -Path .\artifacts | Out-Null
& .\.venv\Scripts\python.exe -m pytest .\tests\integration\test_impact.py --basetemp=./artifacts/pytest_tmp
```

For scanner-only development, install the `dev` extra and select the original test files:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
New-Item -ItemType Directory -Force -Path .\artifacts | Out-Null
& .\.venv\Scripts\python.exe -m pytest .\tests\unit\test_checks.py .\tests\unit\test_csv_io.py .\tests\unit\test_cli.py .\tests\integration\test_scan.py --basetemp=./artifacts/pytest_tmp
```

The complete suite requires both `dev` and `ml`; the ML test modules import the optional dependencies.

Tests cover:

- Scanner counts, CSV parsing, malformed input, encoding, and filesystem errors.
- Numeric regression validation, missing targets, non-finite values, and fingerprints.
- Identical seeded splits and reproducible results.
- Training-only imputation statistics and scaling.
- Independent reference calculations for MAE and summary statistics.
- Unusable folds, numerical failures, and source-data preservation.
- CLI text/JSON output, arguments, errors, and simulated missing ML dependencies.
- End-to-end experiments using the synthetic housing example.
- The installed module entry point running outside the repository.

Tests use synthetic fixtures and temporary files. They do not require `employees.csv` or network access. Run the suite in your environment to obtain current results; no benchmark or coverage percentage is implied by the presence of these tests.

## Project structure

```text
DataGuard-AI/
├── README.md
├── pyproject.toml
├── .gitignore
├── main.py
├── employees.csv
├── data/
│   └── examples/
│       └── housing_regression.csv
├── src/
│   └── dataguard/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── csv_io.py
│       ├── scan.py
│       ├── checks/
│       │   ├── __init__.py
│       │   └── quality.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── data.py
│       │   └── imputation.py
│       └── reporting/
│           ├── __init__.py
│           ├── report.py
│           └── impact.py
├── tests/
│   ├── fixtures/
│   │   ├── normal.csv
│   │   ├── header_only.csv
│   │   ├── empty.csv
│   │   └── malformed.csv
│   ├── unit/
│   │   ├── test_checks.py
│   │   ├── test_csv_io.py
│   │   ├── test_cli.py
│   │   ├── test_regression_data.py
│   │   ├── test_imputation.py
│   │   └── test_impact_cli.py
│   └── integration/
│       ├── test_scan.py
│       └── test_impact.py
└── artifacts/                  Local generated files; ignored by Git
```

`csv_io.py` validates CSV structure. `checks/` contains reusable scanner checks, and `scan.py` assembles their results. `experiments/` validates numeric regression data and runs paired evaluations. `reporting/` formats results, while `cli.py` handles arguments and command dispatch.

The existing `main.py` remains a separate prototype and may require its original dependencies, including pandas. The installed CLI does not import or execute it. Pandas is not required by the scanner or ML Impact Lab.

## Current scope and limitations

Version 0.2 implements CSV scanning and one controlled comparison of numeric-feature imputation methods for regression.

- Only UTF-8, comma-separated CSV files with headers are supported. There is no Excel, TSV, or configurable-dialect support.
- Validation uses Python's CSV parser plus the documented structural rules; it is not a business-schema validator.
- Missing-value tokens are not configurable.
- Duplicate detection uses complete parsed rows, without fuzzy matching or selected-key matching.
- Data, feature arrays, and cross-validation indices are held in memory. The scanner also keeps a set of distinct rows. There are no streaming or large-dataset performance claims.
- The standard library's default CSV field-size limit applies.
- ML experiments use all non-target columns as numeric features. Identifiers, encoded categories, and target-derived features need user review.
- Shuffled K-fold evaluation is unsuitable for many time-series, grouped, or repeated-entity datasets. These structures are not automatically detected.
- There is no classification, categorical encoding, model search, or hyperparameter tuning.
- Mean imputation is the comparison baseline. There is no separate untreated-data or dummy-regressor benchmark in this workflow.
- There is no independent final holdout evaluation, confidence interval, significance test, or automatic strategy selection.
- The scanner does not detect suspicious ages, outliers, inconsistent types, target leakage, or dataset drift.
- The application does not overwrite source data, export repaired datasets, or persist trained models.

## Future direction

Potential next steps include:

- Additional quality checks and explicitly configured repair strategies.
- Time-aware and group-aware evaluation protocols.
- Broader comparisons across datasets and models, with documented methodology.
- Evaluation approaches that address uncertainty and selection bias.
- Explainable reports connecting detected issues with measured repair effects.
- Evidence-based ranking of data problems under clearly defined experiments.
- FastAPI, a web interface, Docker packaging, and CI/CD.

These are roadmap items, not implemented capabilities or delivery commitments. DataGuard v0.2 has no AI API integration, database, hosted service, or deployment requirement.