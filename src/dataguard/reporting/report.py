"""Represent scan results independently of the command-line interface."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScanReport:
    """Dataset dimensions and quality counts, excluding the header row."""

    row_count: int
    column_count: int
    missing_values_per_column: dict[str, int]
    total_missing_values: int
    duplicate_count: int

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        """Return a detached, JSON-serializable copy of the report."""
        return asdict(self)


def format_report(report: ScanReport) -> str:
    """Render a deterministic, human-readable report in CSV column order."""
    lines = [
        "DataGuard AI scan report",
        f"Rows: {report.row_count}",
        f"Columns: {report.column_count}",
        "Missing values per column:",
    ]
    lines.extend(
        f"  {column!r}: {count}"
        for column, count in report.missing_values_per_column.items()
    )
    lines.extend([
        f"Total missing values: {report.total_missing_values}",
        f"Duplicate rows (after first occurrence): {report.duplicate_count}",
    ])
    return "\n".join(lines)
