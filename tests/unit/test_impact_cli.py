"""Tests for the ML Impact Lab CLI and scanner compatibility.

Requires the development and ML extras: pip install -e ".[dev,ml]".

Only the end-to-end JSON test and the shared reference-report fixture run
actual experiments. Argument-forwarding and failure tests mock execution.

Missing-dependency tests intercept imports explicitly, so previously cached
ML modules do not invalidate the simulation.
"""

import builtins
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

import dataguard.experiments as experiments
from dataguard.cli import main
from dataguard.experiments import (
    ExperimentConfig,
    ImputationExperimentError,
    ImputationReport,
    load_regression_csv,
    run_imputation_experiment,
)


def _write_regression_csv(directory: Path) -> Path:
    """Create a small numeric dataset with sparse missing feature values."""
    source = directory / "housing sample.csv"
    source.write_text(
        "size,age,price\n"
        "720,38,185000\n"
        "850,26,212000\n"
        ",18,238000\n"
        "1100,12,285000\n"
        "1250,,302000\n"
        "1380,8,348000\n"
        "1500,22,315000\n"
        ",15,276000\n"
        "1680,6,395000\n"
        "1820,11,418000\n"
        "2050,5,489000\n"
        "3100,4,815000\n",
        encoding="utf-8",
        newline="",
    )
    return source


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    """Provide an independent input file for each CLI test."""
    return _write_regression_csv(tmp_path)


@pytest.fixture(scope="module")
def reference_report(
    tmp_path_factory: pytest.TempPathFactory,
) -> ImputationReport:
    """Run one small experiment for tests that only exercise CLI behavior."""
    directory = tmp_path_factory.mktemp("impact-cli-reference")
    source = _write_regression_csv(directory)
    return run_imputation_experiment(
        source,
        target_column="price",
        config=ExperimentConfig(
            n_splits=3,
            n_repeats=1,
            random_state=17,
            ridge_alpha=0.75,
        ),
    )


def _reject_nonstandard_json_constant(value: str) -> Any:
    """Reject NaN and Infinity, which json.loads otherwise accepts."""
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def _block_ml_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Block ML imports even if their modules already exist in sys.modules."""
    original_import = builtins.__import__
    attempts: list[str] = []

    blocked_prefixes = (
        "dataguard.experiments",
        "dataguard.reporting.impact",
        "numpy",
        "scipy",
        "sklearn",
        "joblib",
        "threadpoolctl",
    )

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in blocked_prefixes
        ):
            attempts.append(name)
            raise ModuleNotFoundError(
                f"No module named {name!r}",
                name=name,
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return attempts


def test_impact_json_end_to_end(
    regression_csv: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the actual CLI, CSV loader, experiment, and JSON formatter."""
    original_bytes = regression_csv.read_bytes()

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
        "--folds",
        "3",
        "--repeats",
        "1",
        "--seed",
        "17",
        "--ridge-alpha",
        "0.75",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.err == ""

    payload = json.loads(
        captured.out,
        parse_constant=_reject_nonstandard_json_constant,
    )

    assert payload["config"] == {
        "n_splits": 3,
        "n_repeats": 1,
        "random_state": 17,
        "ridge_alpha": 0.75,
    }
    assert payload["target_name"] == "price"
    assert payload["feature_names"] == ["size", "age"]
    assert payload["n_samples"] == 12
    assert payload["n_features"] == 2
    assert payload["missing_values_per_feature"] == {
        "size": 2,
        "age": 1,
    }
    assert payload["metric"] == "MAE"
    assert payload["baseline_strategy"] == "mean"
    assert payload["model"] == "Ridge"
    assert payload["ridge_solver"] == "svd"
    assert payload["preprocessing"] == ["SimpleImputer", "StandardScaler"]
    assert payload["std_ddof"] == 1
    assert payload["train_sizes"] == [8, 8, 8]
    assert payload["validation_sizes"] == [4, 4, 4]

    methods = {
        result["strategy"]: result
        for result in payload["methods"]
    }
    assert len(payload["methods"]) == 2
    assert set(methods) == {"mean", "median"}

    for result in methods.values():
        assert len(result["fold_mae"]) == 3
        assert all(
            math.isfinite(score) and score >= 0
            for score in result["fold_mae"]
        )
        assert math.isfinite(result["mean_mae"])
        assert result["mean_mae"] >= 0
        assert math.isfinite(result["std_mae"])
        assert result["std_mae"] >= 0
        assert math.isfinite(result["mae_difference_from_baseline"])

    assert methods["mean"]["mae_difference_from_baseline"] == 0.0
    assert methods["median"]["mae_difference_from_baseline"] == pytest.approx(
        methods["median"]["mean_mae"] - methods["mean"]["mean_mae"]
    )
    assert payload["paired_mae_differences"] == pytest.approx([
        median - mean
        for median, mean in zip(
            methods["median"]["fold_mae"],
            methods["mean"]["fold_mae"],
            strict=True,
        )
    ])

    dataset = load_regression_csv(regression_csv, target_column="price")
    assert payload["data_fingerprint"] == dataset.data_fingerprint

    split_fingerprint = payload["split_fingerprint"]
    assert len(split_fingerprint) == 64
    assert all(
        character in "0123456789abcdef"
        for character in split_fingerprint
    )
    assert {
        "python",
        "platform",
        "dataguard-ai",
        "numpy",
        "scipy",
        "scikit-learn",
    } <= payload["versions"].keys()
    assert all(
        isinstance(value, str) and value
        for value in payload["versions"].values()
    )
    assert payload["notes"]
    assert regression_csv.read_bytes() == original_bytes


def test_impact_defaults_to_readable_text(
    regression_csv: Path,
    reference_report: ImputationReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def return_report(*args: Any, **kwargs: Any) -> ImputationReport:
        return reference_report

    monkeypatch.setattr(
        experiments,
        "run_imputation_experiment",
        return_report,
    )

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
        "--folds",
        "3",
        "--repeats",
        "1",
        "--seed",
        "17",
        "--ridge-alpha",
        "0.75",
    ])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.err == ""
    assert captured.out.startswith("DataGuard AI - ML Impact Lab\n")

    for expected_text in (
        "Target: 'price'",
        "Records: 12",
        "Numeric features: 2",
        "Model: Ridge",
        "Ridge alpha: 0.75",
        "Folds per repetition: 3",
        "Repetitions: 1",
        "Seed: 17",
        "MAE summary",
        "Mean MAE",
        "Std MAE",
        "Delta vs mean",
        "Paired fold scores",
        "Median - mean",
        "Reproducibility metadata",
        "not a confidence interval or significance test",
    ):
        assert expected_text in captured.out

    assert reference_report.data_fingerprint in captured.out
    assert reference_report.split_fingerprint in captured.out
    assert any(
        line.startswith("mean ")
        for line in captured.out.splitlines()
    )
    assert any(
        line.startswith("median ")
        for line in captured.out.splitlines()
    )


@pytest.mark.parametrize(
    ("options", "overrides"),
    [
        ([], {}),
        (["--folds", "2"], {"n_splits": 2}),
        (["--repeats", "1"], {"n_repeats": 1}),
        (["--seed", "0"], {"random_state": 0}),
        (
            ["--seed", "4294967295"],
            {"random_state": 4294967295},
        ),
        (["--ridge-alpha", "2.5"], {"ridge_alpha": 2.5}),
        (
            [
                "--folds", "3",
                "--repeats", "2",
                "--seed", "123",
                "--ridge-alpha", "0.25",
            ],
            {
                "n_splits": 3,
                "n_repeats": 2,
                "random_state": 123,
                "ridge_alpha": 0.25,
            },
        ),
    ],
    ids=[
        "defaults",
        "folds",
        "repeats",
        "zero-seed",
        "maximum-seed",
        "ridge-alpha",
        "combined-settings",
    ],
)
def test_experiment_settings_are_forwarded(
    regression_csv: Path,
    reference_report: ImputationReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    overrides: dict[str, Any],
) -> None:
    calls: list[tuple[Path, str, ExperimentConfig]] = []

    def capture_experiment(
        path: Path,
        *,
        target_column: str,
        config: ExperimentConfig,
    ) -> ImputationReport:
        calls.append((path, target_column, config))
        return reference_report

    monkeypatch.setattr(
        experiments,
        "run_imputation_experiment",
        capture_experiment,
    )

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
        *options,
    ])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.err == ""
    assert len(calls) == 1

    path, target, config = calls[0]
    assert path == regression_csv
    assert target == "price"

    expected_settings: dict[str, Any] = {
        "n_splits": 5,
        "n_repeats": 3,
        "random_state": 42,
        "ridge_alpha": 1.0,
    }
    expected_settings.update(overrides)

    assert config == ExperimentConfig(**expected_settings)


def test_target_name_is_forwarded_without_trimming(
    tmp_path: Path,
    reference_report: ImputationReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "spaced-header.csv"
    source.write_text(
        "feature, sale price \n1,100\n2,200\n3,300\n",
        encoding="utf-8",
    )
    targets: list[str] = []

    def capture_target(
        path: Path,
        *,
        target_column: str,
        config: ExperimentConfig,
    ) -> ImputationReport:
        targets.append(target_column)
        return reference_report

    monkeypatch.setattr(
        experiments,
        "run_imputation_experiment",
        capture_target,
    )

    assert main([
        "impact",
        str(source),
        "--target",
        " sale price ",
    ]) == 0

    assert targets == [" sale price "]
    assert capsys.readouterr().err == ""


def test_required_target_argument_is_enforced_without_ml_imports(
    regression_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import_attempts = _block_ml_imports(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main(["impact", str(regression_csv)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "--target" in captured.err
    assert "required" in captured.err
    assert "Traceback" not in captured.err
    assert import_attempts == []


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--target", ""),
        ("--target", "   "),
        ("--folds", "0"),
        ("--folds", "1"),
        ("--folds", "-1"),
        ("--folds", "2.5"),
        ("--folds", "many"),
        ("--repeats", "0"),
        ("--repeats", "-1"),
        ("--repeats", "1.5"),
        ("--repeats", "many"),
        ("--seed", "-1"),
        ("--seed", "4294967296"),
        ("--seed", "1.5"),
        ("--seed", "random"),
        ("--ridge-alpha", "0"),
        ("--ridge-alpha", "-1"),
        ("--ridge-alpha", "nan"),
        ("--ridge-alpha", "inf"),
        ("--ridge-alpha", "-inf"),
        ("--ridge-alpha", "1e309"),
        ("--ridge-alpha", "invalid"),
    ],
)
def test_invalid_arguments_exit_before_loading_ml_dependencies(
    regression_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    import_attempts = _block_ml_imports(monkeypatch)
    arguments = ["impact", str(regression_csv)]

    if option != "--target":
        arguments.extend(["--target", "price"])

    # The equals form also handles negative infinity as an option value.
    arguments.append(f"{option}={value}")

    with pytest.raises(SystemExit) as error:
        main(arguments)

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert option in captured.err
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert import_attempts == []


def test_missing_csv_returns_one_without_partial_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "absent.csv"

    status = main([
        "impact",
        str(missing),
        "--target",
        "price",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err.startswith("dataguard: error: ")
    assert "not found" in captured.err
    assert "absent.csv" in captured.err
    assert "Traceback" not in captured.err


def test_unknown_target_returns_one(
    regression_csv: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "unknown_target",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert "unknown_target" in captured.err
    assert "was not found" in captured.err
    assert "Available columns:" in captured.err
    assert "Traceback" not in captured.err


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
            "feature,price\n1,low\n2,medium\n3,high\n",
            "nonnumeric or unrepresentable",
        ),
        (
            "feature,price\n1,100\ninf,200\n3,300\n",
            "non-finite numeric value",
        ),
        (
            "feature,price\n1,100\n2,100\n3,100\n",
            "is constant",
        ),
        (
            "feature,price\n,100\n,200\n,300\n",
            "no observed numeric values",
        ),
        (
            "feature,price\n1,100\n2,200,extra\n",
            "expected 2",
        ),
        (
            "feature,price\n",
            "at least two data records",
        ),
        (
            "price\n100\n200\n300\n",
            "at least one feature column",
        ),
        (
            "",
            "header is required",
        ),
    ],
    ids=[
        "missing-target-value",
        "categorical-feature",
        "categorical-target",
        "infinite-feature",
        "constant-target",
        "entirely-missing-feature",
        "malformed-record",
        "header-only",
        "no-features",
        "empty-file",
    ],
)
def test_unsupported_or_invalid_datasets_return_clear_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    expected_error: str,
) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text(content, encoding="utf-8", newline="")
    original_bytes = source.read_bytes()

    status = main([
        "impact",
        str(source),
        "--target",
        "price",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err.startswith("dataguard: error: ")
    assert expected_error in captured.err
    assert "Traceback" not in captured.err
    assert source.read_bytes() == original_bytes


def test_fold_count_exceeding_record_count_returns_one(
    regression_csv: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
        "--folds",
        "13",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert "exceeds the number of data records" in captured.err
    assert "Traceback" not in captured.err


def test_experiment_failure_has_no_partial_report(
    regression_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_experiment(*args: Any, **kwargs: Any) -> ImputationReport:
        raise ImputationExperimentError(
            "Numerical failure in repetition 1, fold 1."
        )

    monkeypatch.setattr(
        experiments,
        "run_imputation_experiment",
        fail_experiment,
    )

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err == (
        "dataguard: error: Numerical failure in repetition 1, fold 1.\n"
    )


def test_interrupted_experiment_returns_130(
    regression_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> ImputationReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        experiments,
        "run_imputation_experiment",
        interrupt,
    )

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
    ])
    captured = capsys.readouterr()

    assert status == 130
    assert captured.out == ""
    assert "experiment interrupted" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "missing_module",
    ["numpy", "sklearn", "scipy", "joblib", "threadpoolctl"],
)
def test_missing_optional_dependencies_produce_installation_guidance(
    regression_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing_module: str,
) -> None:
    original_import = builtins.__import__
    intercepted: list[str] = []

    def simulate_missing_dependency(
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if name == "dataguard.experiments":
            intercepted.append(name)
            raise ModuleNotFoundError(
                f"No module named {missing_module!r}",
                name=missing_module,
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        builtins,
        "__import__",
        simulate_missing_dependency,
    )

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
        "--json",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert intercepted == ["dataguard.experiments"]
    assert captured.out == ""
    assert "unavailable dependency" in captured.err
    assert repr(missing_module) in captured.err
    assert 'pip install -e ".[ml]"' in captured.err
    assert sys.executable in captured.err
    assert "Traceback" not in captured.err


def test_incompatible_ml_import_produces_clear_error(
    regression_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def simulate_import_failure(
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if name == "dataguard.experiments":
            raise ImportError("deliberate dependency import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        builtins,
        "__import__",
        simulate_import_failure,
    )

    status = main([
        "impact",
        str(regression_csv),
        "--target",
        "price",
    ])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert "could not be imported" in captured.err
    assert "deliberate dependency import failure" in captured.err
    assert 'pip install -e ".[ml]"' in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("arguments", "expected_text"),
    [
        (["--help"], "impact"),
        (["impact", "--help"], "--target"),
        (["scan", "--help"], "--json"),
    ],
    ids=["main-help", "impact-help", "scan-help"],
)
def test_help_works_without_ml_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected_text: str,
) -> None:
    import_attempts = _block_ml_imports(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main(arguments)

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert captured.err == ""
    assert "usage:" in captured.out
    assert expected_text in captured.out
    assert import_attempts == []

    if arguments == ["impact", "--help"]:
        for option in (
            "--target",
            "--folds",
            "--repeats",
            "--seed",
            "--ridge-alpha",
            "--json",
        ):
            assert option in captured.out


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_original_scanner_output_is_preserved_without_ml_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    source = tmp_path / "scanner.csv"
    source.write_text(
        "name,age\n"
        "Ada,36\n"
        "Ben,\n"
        "Ada,36\n"
        ",29\n",
        encoding="utf-8",
    )
    original_bytes = source.read_bytes()
    import_attempts = _block_ml_imports(monkeypatch)

    arguments = ["scan", str(source)]
    if as_json:
        arguments.append("--json")

    status = main(arguments)
    captured = capsys.readouterr()

    assert status == 0
    assert captured.err == ""
    assert import_attempts == []
    assert source.read_bytes() == original_bytes

    if as_json:
        expected = {
            "row_count": 4,
            "column_count": 2,
            "missing_values_per_column": {"name": 1, "age": 1},
            "total_missing_values": 2,
            "duplicate_count": 1,
        }
        assert json.loads(captured.out) == expected

        # Preserve the original indentation, key order, and final newline.
        assert captured.out == json.dumps(expected, indent=2) + "\n"
    else:
        assert captured.out == (
            "DataGuard AI scan report\n"
            "Rows: 4\n"
            "Columns: 2\n"
            "Missing values per column:\n"
            "  'name': 1\n"
            "  'age': 1\n"
            "Total missing values: 2\n"
            "Duplicate rows (after first occurrence): 1\n"
        )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("missing", "CSV file not found"),
        ("malformed", "expected 2"),
    ],
)
def test_original_scanner_errors_are_preserved_without_ml_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected_error: str,
) -> None:
    source = tmp_path / "scanner-error.csv"
    if case == "malformed":
        source.write_text("name,age\nAda\n", encoding="utf-8")

    import_attempts = _block_ml_imports(monkeypatch)

    status = main(["scan", str(source), "--json"])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err.startswith("dataguard: error: ")
    assert expected_error in captured.err
    assert "Traceback" not in captured.err
    assert import_attempts == []


@pytest.mark.parametrize(
    "arguments",
    [[], ["scan"], ["unknown"]],
)
def test_original_argument_errors_remain_exit_code_two_without_ml(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    import_attempts = _block_ml_imports(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main(arguments)

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert import_attempts == []