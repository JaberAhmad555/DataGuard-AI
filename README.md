# DataGuard AI

DataGuard AI v0.1 is a small Python application for inspecting CSV data quality.
It reports missing values and duplicate rows without changing the source file.
The package has no runtime dependencies and requires Python 3.11 or newer.

The project's longer-term goal is evidence-based measurement of how data repairs
affect machine-learning performance. That experiment engine is not part of v0.1.

## Current features

- Reusable, typed Python functions for missing values and duplicate rows.
- A structured report with row count, column count, missing values per column,
  total missing values, and duplicate count.
- A `dataguard scan` command with readable text or JSON output.
- Input validation and actionable errors for unreadable or invalid CSV files.
- Deterministic results for the same input bytes and supported Python environment.
- pytest tests with small fixtures and temporary files.

## Installation on Windows PowerShell

Open PowerShell in the DataGuard project folder. Confirm that `python` selects
Python 3.11 or newer, then create a virtual environment and install the package:

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Using the executable inside `.venv` works without activating it or changing
PowerShell's execution policy. `-e` installs the local source in editable mode,
and `[dev]` also installs pytest. Installation may need internet access to obtain
build and development dependencies. For installation without pytest, use
`& .\.venv\Scripts\python.exe -m pip install -e .` instead.

In VS Code, use **Python: Select Interpreter** and choose
`.venv\Scripts\python.exe`.

## Command-line usage

From the project folder:

```powershell
& .\.venv\Scripts\dataguard.exe scan .\employees.csv
& .\.venv\Scripts\dataguard.exe scan .\employees.csv --json
& .\.venv\Scripts\python.exe -m dataguard scan .\employees.csv
& .\.venv\Scripts\dataguard.exe scan --help
```

Quote paths that contain spaces. If the virtual environment is already activated,
the equivalent command is `dataguard scan <csv-path>`; replace `<csv-path>` with
the actual file path.

For a reproducible example using the bundled test fixture:

```powershell
& .\.venv\Scripts\dataguard.exe scan .\tests\fixtures\normal.csv --json
```

Expected report for that fixture:

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

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Scan completed, even if quality issues were found; or help was displayed. |
| `1` | The CSV could not be read or failed input validation. |
| `2` | Invalid or missing command-line arguments. |

Errors go to standard error. Successful reports go to standard output. An invalid
file produces an error rather than a partial report.

## Python usage

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

The checks are also reusable independently of CSV loading:

```python
from dataguard.checks import count_duplicate_rows, count_missing_values

columns = ["name", "age"]
rows = [("Ada", ""), ("Ada", ""), ("Ben", "30")]

print(count_missing_values(columns, rows))  # {"name": 0, "age": 2}
print(count_duplicate_rows(rows))           # 1
```

These functions expect strings. They accept row iterables and consume them;
use a reusable collection, or a new iterator, when running multiple checks.
`count_missing_values` raises `ValueError` for duplicate column names or rows
whose widths do not match the supplied columns.

## CSV rules and counting definitions

- Input must be UTF-8, optionally with a UTF-8 BOM, and comma-separated.
- The first nonempty record is the header. Column names must be unique and must
  not be empty or whitespace-only. Header text is otherwise preserved exactly.
- Standard double-quoted fields, escaped double quotes, commas inside quoted
  fields, and line breaks inside quoted fields are supported through Python's
  `csv.reader` with strict parsing enabled.
- Each data record must have exactly the same number of fields as the header.
- Empty physical lines outside quoted fields are skipped. A record such as `,`
  in a two-column CSV is a data row containing two missing values.
- Missing values are empty or whitespace-only fields, including quoted empty
  fields. Literal `NA`, `N/A`, `null`, `None`, and `NaN` are ordinary strings.
- Duplicate rows are exact matches of all parsed field strings, counted after
  their first occurrence. Three identical rows produce a duplicate count of two.
  CSV quoting differences that decode to identical strings do not distinguish rows.
- Whitespace, case, and numeric spelling matter for duplicates: `1` and `01`
  differ, as do `Ada` and `Ada `. No values are trimmed or converted in memory.
- Row counts exclude the header and skipped blank lines. Missing-value counts
  include duplicate records; no rows are removed before counting.
- A header-only CSV is valid and reports zero rows. A zero-byte or blank-line-only
  file is rejected because its columns cannot be determined.

## Tests

After installing the development extras, run from the project folder:

```powershell
& .\.venv\Scripts\python.exe -m pytest
```

Tests cover normal and header-only inputs, empty files, malformed records, invalid
headers, encoding errors, quoted fields, duplicate semantics, filesystem errors,
CLI output and exit codes, repeatability, and preservation of source-file bytes.
They use synthetic data and do not require `employees.csv` or network access.

## Project layout

```text
DataGuard/
|-- main.py                    Existing prototype, preserved
|-- employees.csv              Existing example dataset, preserved
|-- .gitignore
|-- pyproject.toml
|-- README.md
|-- src/dataguard/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   |-- csv_io.py
|   |-- scan.py
|   |-- checks/
|   |   |-- __init__.py
|   |   `-- quality.py
|   `-- reporting/
|       |-- __init__.py
|       `-- report.py
`-- tests/
    |-- fixtures/
    |   |-- normal.csv
    |   |-- header_only.csv
    |   |-- empty.csv
    |   `-- malformed.csv
    |-- unit/
    |   |-- test_checks.py
    |   |-- test_csv_io.py
    |   `-- test_cli.py
    `-- integration/
        `-- test_scan.py
```

`csv_io.py` validates input, `checks/` contains the independent checks,
`scan.py` combines them, and `reporting/` defines their output. `cli.py` handles
arguments and presentation. The package is installed from `src/`.

The existing `main.py` remains a separate prototype and may require its original
dependencies, including pandas. The new CLI does not execute or import it; pandas
is not needed by the new package. `employees.csv` remains at its original path.

## Limitations

- Only UTF-8 CSV with a header and a comma delimiter is supported. No delimiter,
  encoding, or header detection is attempted. TSV, Excel, and other formats are
  not supported. A semicolon-separated file can appear to be a single-column CSV;
  the scanner cannot infer the author's intended delimiter.
- All parsed rows and a set of distinct rows are held in memory. Memory use grows
  with dataset size. This release has no streaming or large-dataset performance
  claims. The standard library's default CSV field-size limit also applies.
- Validation follows Python's CSV parser plus the documented header and row-width
  rules; it does not impose additional CSV dialect or business-schema rules.
- Missing-value tokens cannot yet be configured. Duplicate detection does not
  perform fuzzy matching, normalization, or matching on selected columns.
- This release does not detect outliers, suspicious ages, type inconsistencies,
  target leakage, or dataset drift, and it never repairs input data.
- There are no ML experiments, impact rankings, benchmarks, AI APIs, database,
  web interface, Docker configuration, or automated deployment in v0.1.
