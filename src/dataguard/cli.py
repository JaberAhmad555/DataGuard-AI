"""Command-line interface for DataGuard AI."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dataguard.csv_io import CSVInputError
from dataguard.reporting import format_report
from dataguard.scan import scan_csv


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return 0 on success or 1 for CSV input errors.

    argparse handles help and invalid arguments with exit codes 0 and 2.
    Reported data quality issues do not themselves cause a nonzero exit.
    """
    parser = argparse.ArgumentParser(
        prog="dataguard", description="Read-only CSV data quality checks."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan", help="Scan a UTF-8 CSV with a header.")
    scan_parser.add_argument("csv_path", type=Path, help="Path to a comma-separated CSV.")
    scan_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Print a structured JSON report."
    )
    args = parser.parse_args(argv)

    try:
        report = scan_csv(args.csv_path)
    except CSVInputError as exc:
        print(f"dataguard: error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0
