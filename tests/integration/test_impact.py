"""Integration tests for the complete ML Impact Lab workflow.

The housing example contains fictional, synthetic records. These tests verify
software behavior; they do not establish real-world model performance or claim
that one imputation strategy must outperform another.

Requires an installed package with development and ML dependencies:
    python -m pip install -e ".[dev,ml]"

Successful experiments use real CSV loading, preprocessing, Ridge regression,
and cross-validation. No experiment components are mocked.
"""

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dataguard import scan_csv
from dataguard.cli import main
from dataguard.experiments import (
    ExperimentConfig,
    ImputationReport,
    load_regression_csv,
    run_imputation_experiment,
)
from dataguard.reporting.impact import (
    format_impact_json,
    format_impact_report,
)


@pytest.fixture(scope="module")
def housing_csv() -> Path:
    """Locate the repository's synthetic housing example."""
    project_root = Path(__file__).resolve().parents[2]
    source = project_root / "data" / "examples" / "housing_regression.csv"

    assert source.is_file(), (
        "The synthetic housing example is required for integration tests: "
        f"{source}"
    )
    return source


@pytest.fixture(scope="module")
def original_housing_bytes(housing_csv: Path) -> bytes:
    """Capture the source bytes before this module runs any experiments."""
    return housing_csv.read_bytes()


@pytest.fixture(scope="module")
def reference_config() -> ExperimentConfig:
    """Use repeated five-fold evaluation with a fixed seed."""
    return ExperimentConfig(
        n_splits=5,
        n_repeats=2,
        random_state=42,
        ridge_alpha=1.0,
    )


@pytest.fixture(scope="module")
def reference_report(
    housing_csv: Path,
    original_housing_bytes: bytes,
    reference_config: ExperimentConfig,
) -> ImputationReport:
    """Run one real experiment shared by report-validation tests."""
    report = run_imputation_experiment(
        housing_csv,
        target_column="price",
        config=reference_config,
    )

    assert housing_csv.read_bytes() == original_housing_bytes
    return report


def _reject_nonstandard_json_constant(value: str) -> Any:
    """Reject NaN and Infinity rather than accepting nonstandard JSON."""
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _parse_json_report(text: str) -> dict[str, Any]:
    """Parse exactly one JSON report and require an object at its root."""
    payload = json.loads(
        text,
        parse_constant=_reject_nonstandard_json_constant,
    )
    assert isinstance(payload, dict)
    return payload


def test_synthetic_example_loads_and_remains_scannable(
    housing_csv: Path,
    original_housing_bytes: bytes,
) -> None:
    """Check the sample through both the v0.1 scanner and v0.2 loader."""
    scan = scan_csv(housing_csv)
    dataset = load_regression_csv(housing_csv, target_column="price")

    expected_missing = {
        "size_sqft": 4,
        "bedrooms": 3,
        "building_age_years": 5,
        "distance_to_center_km": 4,
    }

    assert scan.row_count == 60
    assert scan.column_count == 5
    assert scan.duplicate_count == 0
    assert scan.total_missing_values == 16
    assert scan.missing_values_per_column == {
        **expected_missing,
        "price": 0,
    }

    assert dataset.n_samples == 60
    assert dataset.n_features == 4
    assert dataset.target_name == "price"
    assert dataset.feature_names == (
        "size_sqft",
        "bedrooms",
        "building_age_years",
        "distance_to_center_km",
    )
    assert dataset.missing_values_per_feature == expected_missing
    assert dataset.total_missing_values == 16
    assert dataset.X.dtype.name == "float64"
    assert dataset.y.dtype.name == "float64"
    assert all(math.isfinite(value) for value in dataset.y)
    assert not dataset.X.flags.writeable
    assert not dataset.y.flags.writeable

    assert housing_csv.read_bytes() == original_housing_bytes


def test_complete_experiment_has_consistent_results_and_fold_sizes(
    housing_csv: Path,
    original_housing_bytes: bytes,
    reference_config: ExperimentConfig,
    reference_report: ImputationReport,
) -> None:
    """Validate real training results without hardcoded performance values."""
    report = reference_report
    expected_evaluations = (
        reference_config.n_splits * reference_config.n_repeats
    )

    assert report.config == reference_config
    assert report.n_samples == 60
    assert report.n_features == 4
    assert report.target_name == "price"
    assert report.missing_values_per_feature == {
        "size_sqft": 4,
        "bedrooms": 3,
        "building_age_years": 5,
        "distance_to_center_km": 4,
    }
    assert report.model == "Ridge"
    assert report.ridge_solver == "svd"
    assert report.preprocessing == ("SimpleImputer", "StandardScaler")
    assert report.metric == "MAE"
    assert report.std_ddof == 1
    assert report.baseline_strategy == "mean"

    assert report.train_sizes == (48,) * expected_evaluations
    assert report.validation_sizes == (12,) * expected_evaluations

    for train_size, validation_size in zip(
        report.train_sizes,
        report.validation_sizes,
        strict=True,
    ):
        assert train_size + validation_size == report.n_samples

    methods = {result.strategy: result for result in report.methods}
    assert len(report.methods) == 2
    assert set(methods) == {"mean", "median"}

    for result in methods.values():
        assert len(result.fold_mae) == expected_evaluations
        assert all(
            math.isfinite(score) and score >= 0
            for score in result.fold_mae
        )
        assert math.isfinite(result.mean_mae)
        assert math.isfinite(result.std_mae)
        assert result.std_mae >= 0

        assert result.mean_mae == pytest.approx(
            statistics.mean(result.fold_mae)
        )
        assert result.std_mae == pytest.approx(
            statistics.stdev(result.fold_mae)
        )
        assert result.mae_difference_from_baseline == pytest.approx(
            result.mean_mae - methods["mean"].mean_mae
        )

    assert methods["mean"].mae_difference_from_baseline == 0.0
    assert report.paired_mae_differences == pytest.approx(tuple(
        median_score - mean_score
        for median_score, mean_score in zip(
            methods["median"].fold_mae,
            methods["mean"].fold_mae,
            strict=True,
        )
    ))

    dataset = load_regression_csv(housing_csv, target_column="price")
    assert report.data_fingerprint == dataset.data_fingerprint

    for fingerprint in (
        report.data_fingerprint,
        report.split_fingerprint,
    ):
        assert len(fingerprint) == 64
        assert all(
            character in "0123456789abcdef"
            for character in fingerprint
        )

    assert {
        "python",
        "platform",
        "dataguard-ai",
        "numpy",
        "scipy",
        "scikit-learn",
    } <= report.versions.keys()
    assert all(
        isinstance(value, str) and value
        for value in report.versions.values()
    )
    assert housing_csv.read_bytes() == original_housing_bytes


def test_real_results_render_as_text_and_strict_json(
    reference_report: ImputationReport,
) -> None:
    """Exercise both formatters using results from the actual experiment."""
    text = format_impact_report(reference_report)
    payload = _parse_json_report(format_impact_json(reference_report))

    assert text.startswith("DataGuard AI - ML Impact Lab\n")

    for section in (
        "Dataset",
        "Experiment settings",
        "MAE summary",
        "Paired fold scores",
        "Reproducibility metadata",
        "Interpretation and limitations",
    ):
        assert section in text

    assert "Target: 'price'" in text
    assert "Records: 60" in text
    assert "Numeric features: 4" in text
    assert "Missing feature values: 16" in text
    assert "not a confidence interval or significance test" in text
    assert reference_report.data_fingerprint in text
    assert reference_report.split_fingerprint in text

    for result in reference_report.methods:
        assert format(result.mean_mae, ".8g") in text
        assert format(result.std_mae, ".8g") in text

    assert payload["config"] == {
        "n_splits": 5,
        "n_repeats": 2,
        "random_state": 42,
        "ridge_alpha": 1.0,
    }
    assert payload["target_name"] == reference_report.target_name
    assert payload["feature_names"] == list(reference_report.feature_names)
    assert payload["n_samples"] == reference_report.n_samples
    assert payload["n_features"] == reference_report.n_features
    assert (
        payload["missing_values_per_feature"]
        == reference_report.missing_values_per_feature
    )
    assert payload["train_sizes"] == list(reference_report.train_sizes)
    assert payload["validation_sizes"] == list(
        reference_report.validation_sizes
    )
    assert payload["data_fingerprint"] == reference_report.data_fingerprint
    assert payload["split_fingerprint"] == reference_report.split_fingerprint
    assert payload["versions"] == reference_report.versions
    assert payload["notes"] == list(reference_report.notes)

    serialized_methods = {
        result["strategy"]: result
        for result in payload["methods"]
    }
    assert set(serialized_methods) == {"mean", "median"}

    for result in reference_report.methods:
        serialized = serialized_methods[result.strategy]
        assert serialized["fold_mae"] == list(result.fold_mae)
        assert serialized["mean_mae"] == result.mean_mae
        assert serialized["std_mae"] == result.std_mae
        assert (
            serialized["mae_difference_from_baseline"]
            == result.mae_difference_from_baseline
        )

    assert payload["paired_mae_differences"] == list(
        reference_report.paired_mae_differences
    )


def test_installed_cli_reproduces_the_experiment_in_a_separate_process(
    housing_csv: Path,
    original_housing_bytes: bytes,
    reference_config: ExperimentConfig,
    reference_report: ImputationReport,
    tmp_path: Path,
) -> None:
    """Verify the installed module entry point and same-setting reproducibility.

    The subprocess runs outside the repository. Its results are compared with
    the separately executed Python API experiment, rather than fixed scores.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "dataguard",
            "impact",
            str(housing_csv),
            "--target",
            "price",
            "--folds",
            str(reference_config.n_splits),
            "--repeats",
            str(reference_config.n_repeats),
            "--seed",
            str(reference_config.random_state),
            "--ridge-alpha",
            str(reference_config.ridge_alpha),
            "--json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""

    payload = _parse_json_report(result.stdout)

    assert payload["config"] == {
        "n_splits": reference_config.n_splits,
        "n_repeats": reference_config.n_repeats,
        "random_state": reference_config.random_state,
        "ridge_alpha": reference_config.ridge_alpha,
    }
    assert payload["data_fingerprint"] == reference_report.data_fingerprint
    assert payload["split_fingerprint"] == reference_report.split_fingerprint
    assert payload["train_sizes"] == list(reference_report.train_sizes)
    assert payload["validation_sizes"] == list(
        reference_report.validation_sizes
    )
    assert payload["versions"] == reference_report.versions

    subprocess_methods = {
        method["strategy"]: method
        for method in payload["methods"]
    }
    assert set(subprocess_methods) == {"mean", "median"}

    for expected in reference_report.methods:
        actual = subprocess_methods[expected.strategy]

        assert actual["fold_mae"] == pytest.approx(
            expected.fold_mae,
            rel=1e-10,
            abs=1e-8,
        )
        assert actual["mean_mae"] == pytest.approx(
            expected.mean_mae,
            rel=1e-10,
            abs=1e-8,
        )
        assert actual["std_mae"] == pytest.approx(
            expected.std_mae,
            rel=1e-10,
            abs=1e-8,
        )
        assert actual["mae_difference_from_baseline"] == pytest.approx(
            expected.mae_difference_from_baseline,
            rel=1e-10,
            abs=1e-8,
        )

    assert payload["paired_mae_differences"] == pytest.approx(
        reference_report.paired_mae_differences,
        rel=1e-10,
        abs=1e-8,
    )
    assert housing_csv.read_bytes() == original_housing_bytes


def test_alternative_settings_change_the_evaluation_configuration(
    housing_csv: Path,
    original_housing_bytes: bytes,
    reference_report: ImputationReport,
) -> None:
    """Exercise a different fold count, repetition count, seed, and alpha."""
    alternative_config = ExperimentConfig(
        n_splits=4,
        n_repeats=1,
        random_state=7,
        ridge_alpha=2.5,
    )

    alternative = run_imputation_experiment(
        housing_csv,
        target_column="price",
        config=alternative_config,
    )

    assert alternative.config == alternative_config
    assert alternative.config != reference_report.config
    assert alternative.data_fingerprint == reference_report.data_fingerprint
    assert alternative.split_fingerprint != reference_report.split_fingerprint
    assert alternative.train_sizes == (45,) * 4
    assert alternative.validation_sizes == (15,) * 4

    for method in alternative.methods:
        assert len(method.fold_mae) == 4
        assert all(
            math.isfinite(score) and score >= 0
            for score in method.fold_mae
        )

    # Different settings need not produce a particular ordering or improvement.
    assert housing_csv.read_bytes() == original_housing_bytes


@pytest.mark.parametrize(
    ("content", "expected_error"),
    [
        (
            "feature,price\n1,100\n2,\n3,300\n",
            "Target column 'price' is missing",
        ),
        (
            "feature,price\n1,100\nurban,200\n3,300\n",
            "nonnumeric or unrepresentable",
        ),
        (
            "feature,price\n1,100\n2,200,extra\n3,300\n",
            "expected 2",
        ),
        (
            "feature,price\n,100\n,200\n,300\n",
            "no observed numeric values",
        ),
    ],
    ids=[
        "missing-target",
        "categorical-feature",
        "malformed-record",
        "entirely-missing-feature",
    ],
)
def test_invalid_csv_workflow_returns_an_error_without_a_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    expected_error: str,
) -> None:
    """Exercise real CLI error propagation without mocking the loader."""
    source = tmp_path / "invalid.csv"
    source.write_text(content, encoding="utf-8", newline="")
    original_bytes = source.read_bytes()

    status = main([
        "impact",
        str(source),
        "--target",
        "price",
        "--folds",
        "3",
        "--repeats",
        "2",
        "--seed",
        "42",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err.startswith("dataguard: error: ")
    assert expected_error in captured.err
    assert "Traceback" not in captured.err
    assert source.read_bytes() == original_bytes


def test_invalid_cv_for_valid_housing_data_produces_no_partial_report(
    housing_csv: Path,
    original_housing_bytes: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate the full loading-to-experiment error path on the real sample."""
    status = main([
        "impact",
        str(housing_csv),
        "--target",
        "price",
        "--folds",
        "61",
        "--repeats",
        "2",
        "--seed",
        "42",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err.startswith("dataguard: error: ")
    assert "exceeds the number of data records" in captured.err
    assert "Traceback" not in captured.err
    assert housing_csv.read_bytes() == original_housing_bytes