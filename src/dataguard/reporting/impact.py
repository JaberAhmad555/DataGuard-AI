"""Format ML Impact Lab results without running experiments or writing files.

Text output displays eight significant digits for readability. JSON preserves
the report's numeric values without additional rounding and rejects non-finite
numbers rather than emitting invalid JSON.

ML types are imported only for static type checking, so importing this module
does not introduce ML dependencies into the existing CSV scanner.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataguard.experiments.imputation import (
        ImputationMethodResult,
        ImputationReport,
    )


def format_impact_report(report: ImputationReport) -> str:
    """Return a readable experiment report without a trailing newline.

    Includes both methods' aggregate MAE statistics, paired fold scores,
    baseline differences, experiment settings, input and split fingerprints,
    dependency versions, and interpretation notes.

    Differences are method MAE minus mean-imputation baseline MAE. Negative
    differences favor the compared method. No statistical significance,
    confidence interval, or universal winner is inferred.

    Raises:
        ValueError: If methods or fold counts are inconsistent, or a displayed
            metric is non-finite.
    """
    mean_result, median_result = _validated_methods(report)
    config = report.config
    total_folds = config.n_splits * config.n_repeats

    lines = [
        "DataGuard AI - ML Impact Lab",
        "Mean versus median imputation for numeric regression",
        "",
        "Dataset",
        f"  Target: {report.target_name!r}",
        f"  Records: {report.n_samples}",
        f"  Numeric features: {report.n_features}",
        "  Feature order: "
        + ", ".join(repr(name) for name in report.feature_names),
        "  Missing feature values: "
        + str(sum(report.missing_values_per_feature.values())),
        "  Missing values per feature:",
    ]

    lines.extend(
        f"    {name!r}: {report.missing_values_per_feature[name]}"
        for name in report.feature_names
    )

    lines.extend([
        "",
        "Experiment settings",
        f"  Model: {report.model}",
        f"  Ridge alpha: {config.ridge_alpha!r}",
        f"  Ridge solver: {report.ridge_solver}",
        "  Pipeline: "
        + " -> ".join((*report.preprocessing, report.model)),
        "  Imputation and scaling: fitted only on each training fold",
        "  Cross-validation: RepeatedKFold",
        f"  Folds per repetition: {config.n_splits}",
        f"  Repetitions: {config.n_repeats}",
        f"  Seed: {config.random_state}",
        f"  Evaluations per method: {total_folds}",
        "  Split pairing: identical training and validation indices",
        f"  Metric: {report.metric} (lower is better)",
        f"  Standard deviation: sample SD with ddof={report.std_ddof}",
        "  Aggregation: equal weight for each fold MAE",
        f"  Baseline strategy: {report.baseline_strategy}",
        "",
        "MAE summary",
        f"{'Method':<10} {'Mean MAE':>16} {'Std MAE':>16} "
        f"{'Delta vs mean':>16}",
        "-" * 61,
    ])

    for result in (mean_result, median_result):
        lines.append(
            f"{result.strategy:<10} "
            f"{_format_number(result.mean_mae):>16} "
            f"{_format_number(result.std_mae):>16} "
            f"{_format_number(result.mae_difference_from_baseline, signed=True):>16}"
        )

    lines.extend([
        "",
        "Delta = method mean MAE - mean-imputation baseline mean MAE.",
        "Negative delta favors the compared method; positive delta favors "
        "mean imputation.",
        "MAE and its differences are expressed in the target's units.",
        "",
        "Paired fold scores",
        f"{'Repeat':>6} {'Fold':>5} {'Train':>7} {'Valid':>7} "
        f"{'Mean MAE':>16} {'Median MAE':>16} {'Median - mean':>16}",
        "-" * 79,
    ])

    paired_folds = zip(
        report.train_sizes,
        report.validation_sizes,
        mean_result.fold_mae,
        median_result.fold_mae,
        report.paired_mae_differences,
        strict=True,
    )

    for split_index, (
        train_size,
        validation_size,
        mean_mae,
        median_mae,
        difference,
    ) in enumerate(paired_folds):
        repetition, fold = divmod(split_index, config.n_splits)
        lines.append(
            f"{repetition + 1:>6} {fold + 1:>5} "
            f"{train_size:>7} {validation_size:>7} "
            f"{_format_number(mean_mae):>16} "
            f"{_format_number(median_mae):>16} "
            f"{_format_number(difference, signed=True):>16}"
        )

    lines.extend([
        "",
        "Reproducibility metadata",
        "  Parsed-data and target fingerprint (SHA-256):",
        f"    {report.data_fingerprint}",
        "  Ordered cross-validation splits fingerprint (SHA-256):",
        f"    {report.split_fingerprint}",
        "  Environment:",
    ])

    lines.extend(
        f"    {name}: {value}"
        for name, value in sorted(report.versions.items())
    )

    lines.extend([
        "",
        "Interpretation and limitations",
        "  Fold scores are dependent. Their standard deviation is descriptive "
        "variability, not a confidence interval or significance test.",
        "  Text metrics use eight significant digits; JSON retains unrounded "
        "report values.",
    ])

    lines.extend(f"  - {note}" for note in report.notes)

    return "\n".join(lines)


def format_impact_json(report: ImputationReport) -> str:
    """Return the complete report as valid, indented JSON.

    All fields from ImputationReport.to_dict() are retained, including settings,
    fold scores, baseline differences, paired differences, fingerprints,
    environment versions, and notes.

    ASCII escaping makes the returned text safe to write through terminals
    with limited character encodings. A JSON parser restores the original
    Unicode strings. Tuple fields become JSON arrays.

    The function does not modify the report and returns no trailing newline.

    Raises:
        ValueError: If methods or fold counts are inconsistent, or the report
            contains NaN or infinity.
        TypeError: If the report contains a value that is not JSON serializable.
    """
    _validated_methods(report)

    return json.dumps(
        report.to_dict(),
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
    )


def _validated_methods(
    report: ImputationReport,
) -> tuple[ImputationMethodResult, ImputationMethodResult]:
    """Check paired-report structure and return baseline followed by median."""
    if (
        len(report.methods) != 2
        or {method.strategy for method in report.methods} != {"mean", "median"}
    ):
        raise ValueError(
            "An imputation report must contain exactly one mean result "
            "and one median result."
        )

    if report.baseline_strategy != "mean":
        raise ValueError(
            "This formatter requires mean imputation as the baseline."
        )

    if report.config.n_splits < 2 or report.config.n_repeats < 1:
        raise ValueError("The report contains invalid cross-validation settings.")

    methods = {method.strategy: method for method in report.methods}
    mean_result = methods["mean"]
    median_result = methods["median"]
    expected_folds = report.config.n_splits * report.config.n_repeats

    fold_counts = (
        len(report.train_sizes),
        len(report.validation_sizes),
        len(mean_result.fold_mae),
        len(median_result.fold_mae),
        len(report.paired_mae_differences),
    )

    if any(count != expected_folds for count in fold_counts):
        raise ValueError(
            "Training sizes, validation sizes, method scores, and paired "
            f"differences must each contain {expected_folds} entries."
        )

    if set(report.missing_values_per_feature) != set(report.feature_names):
        raise ValueError(
            "Missing-value counts must correspond to the reported feature names."
        )

    return mean_result, median_result


def _format_number(value: float, *, signed: bool = False) -> str:
    """Format a finite metric without suggesting more display precision."""
    if not math.isfinite(value):
        raise ValueError(
            "Cannot format a non-finite experiment metric."
        )

    # Avoid displaying negative zero while leaving the report untouched.
    if value == 0:
        value = 0.0

    return format(value, "+.8g" if signed else ".8g")