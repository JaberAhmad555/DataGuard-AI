"""Coordinate CSV loading, quality checks, and report construction."""

from pathlib import Path

from dataguard.checks import count_duplicate_rows, count_missing_values
from dataguard.csv_io import read_csv
from dataguard.reporting import ScanReport


def scan_csv(path: str | Path) -> ScanReport:
    """Scan a CSV without modifying it; raise CSVInputError for invalid input."""
    columns, rows = read_csv(path)
    missing = count_missing_values(columns, rows)
    return ScanReport(
        row_count=len(rows),
        column_count=len(columns),
        missing_values_per_column=missing,
        total_missing_values=sum(missing.values()),
        duplicate_count=count_duplicate_rows(rows),
    )
