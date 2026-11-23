"""Command-line interface for DataGuard AI.

The CSV scanner requires no optional ML dependencies. Experiment modules are
imported only when an ML Impact Lab command is executed.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from dataguard.csv_io import CSVInputError
from dataguard.reporting import format_report
from dataguard.scan import scan_csv


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Exit behavior:
        0: Successful scan or experiment, including scans finding issues.
        1: Input, experiment, or optional ML dependency error.
        2: Invalid arguments, handled by argparse through SystemExit.
        130: An ML experiment was interrupted by the user.

    argparse also displays help and exits with status 0. Help commands do not
    import the optional ML dependencies.
    """
    parser = argparse.ArgumentParser(
        prog="dataguard", description="Read-only CSV data quality checks."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Preserve the existing v0.1 scanner arguments and defaults.
    scan_parser = commands.add_parser(
        "scan", help="Scan a UTF-8 CSV with a header."
    )
    scan_parser.add_argument(
        "csv_path", type=Path, help="Path to a comma-separated CSV."
    )
    scan_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print a structured JSON report.",
    )

    impact_parser = commands.add_parser(
        "impact",
        help="Compare mean and median imputation for numeric regression.",
        description=(
            "Compare mean and median imputation using identical seeded "
            "cross-validation splits and Ridge regression. Imputation and "
            "scaling are fitted only on training folds."
        ),
        epilog=(
            "All columns except the target must be numeric features. "
            "Use independent records suitable for shuffled cross-validation. "
            "Requires the optional ML dependencies: install with .[ml] "
            "from the project root."
        ),
    )
    impact_parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to a UTF-8, comma-separated CSV with a header.",
    )
    impact_parser.add_argument(
        "--target",
        required=True,
        type=_target_name,
        metavar="COLUMN",
        help="Exact name of the numeric regression target column.",
    )
    impact_parser.add_argument(
        "--folds",
        type=_fold_count,
        default=5,
        metavar="N",
        help="Cross-validation folds per repetition; at least 2 (default: 5).",
    )
    impact_parser.add_argument(
        "--repeats",
        type=_positive_integer,
        default=3,
        metavar="N",
        help="Cross-validation repetitions; at least 1 (default: 3).",
    )
    impact_parser.add_argument(
        "--seed",
        type=_random_seed,
        default=42,
        metavar="N",
        help="Integer split seed from 0 to 4294967295 (default: 42).",
    )
    impact_parser.add_argument(
        "--ridge-alpha",
        type=_positive_finite_float,
        default=1.0,
        metavar="VALUE",
        help="Positive, finite Ridge regularization strength (default: 1.0).",
    )
    impact_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the complete structured report as JSON instead of text.",
    )

    args = parser.parse_args(argv)

    if args.command == "impact":
        return _run_impact(args)

    # Keep the v0.1 scan execution, output, and error behavior unchanged.
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


def _run_impact(args: argparse.Namespace) -> int:
    """Load optional ML components and execute one regression experiment."""
    install_command = (
        f'& "{sys.executable}" -m pip install -e ".[ml]"'
    )

    try:
        # Keep these imports inside the command handler. Importing the CLI,
        # displaying help, and running scans must not require NumPy or sklearn.
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
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown"

        if (
            missing_module == "dataguard"
            or missing_module.startswith("dataguard.")
        ):
            message = (
                f"ML Impact Lab module {missing_module!r} is unavailable. "
                "Ensure all DataGuard v0.2 source files have been saved."
            )
        else:
            message = (
                f"ML Impact Lab requires an unavailable dependency: "
                f"{missing_module!r}."
            )

        print(
            f"dataguard: error: {message}\n"
            "From the DataGuard project root, install the ML extra using "
            "the interpreter running this command:\n"
            f"  {install_command}",
            file=sys.stderr,
        )
        return 1
    except ImportError as exc:
        print(
            "dataguard: error: ML Impact Lab could not be imported. "
            "Check that the v0.2 files are complete and the ML dependencies "
            f"are compatible with this Python environment. Details: {exc}\n"
            "From the DataGuard project root, install the ML extra with:\n"
            f"  {install_command}",
            file=sys.stderr,
        )
        return 1

    try:
        config = ExperimentConfig(
            n_splits=args.folds,
            n_repeats=args.repeats,
            random_state=args.seed,
            ridge_alpha=args.ridge_alpha,
        )
        report = run_imputation_experiment(
            args.csv_path,
            target_column=args.target,
            config=config,
        )

        # Prepare the complete output before printing. Input or experiment
        # failures therefore do not produce a partial report on stdout.
        if args.as_json:
            output = format_impact_json(report)
        else:
            output = format_impact_report(report)

    except (RegressionDataError, ImputationExperimentError) as exc:
        print(f"dataguard: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "dataguard: error: ML Impact Lab experiment interrupted.",
            file=sys.stderr,
        )
        return 130

    print(output)
    return 0


def _target_name(value: str) -> str:
    """Reject blank target names while preserving exact header matching."""
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "target column must not be empty or whitespace-only"
        )
    return value


def _positive_integer(value: str) -> int:
    """Parse an integer greater than or equal to one."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer greater than or equal to 1"
        ) from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "must be an integer greater than or equal to 1"
        )

    return parsed


def _fold_count(value: str) -> int:
    """Parse a cross-validation fold count of at least two."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "folds must be an integer greater than or equal to 2"
        ) from exc

    if parsed < 2:
        raise argparse.ArgumentTypeError(
            "folds must be an integer greater than or equal to 2"
        )

    return parsed


def _random_seed(value: str) -> int:
    """Parse a seed supported by the experiment's random-state settings."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seed must be an integer from 0 to 4294967295"
        ) from exc

    if not 0 <= parsed <= 2**32 - 1:
        raise argparse.ArgumentTypeError(
            "seed must be an integer from 0 to 4294967295"
        )

    return parsed


def _positive_finite_float(value: str) -> float:
    """Parse a positive float, rejecting NaN, infinity, and overflow."""
    try:
        parsed = float(value)
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive, finite number"
        ) from exc

    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive, finite number"
        )

    return parsed