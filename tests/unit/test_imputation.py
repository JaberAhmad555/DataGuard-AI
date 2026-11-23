"""Tests for paired, leakage-safe imputation experiments.

Requires the development and ML extras: pip install -e ".[dev,ml]".

Tests use temporary synthetic datasets and small deterministic experiments.
Reference scores are calculated independently with scikit-learn pipelines.
No test assumes that median imputation must outperform mean imputation.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dataguard.experiments.imputation as imputation
from dataguard.experiments.data import (
    RegressionDataset,
    load_regression_csv,
)
from dataguard.experiments.imputation import (
    ExperimentConfig,
    ImputationExperimentError,
    compare_imputation,
    run_imputation_experiment,
)


@pytest.fixture
def regression_dataset(tmp_path: Path) -> RegressionDataset:
    """Provide skewed numeric features with sparse, genuine missing cells."""
    source = tmp_path / "regression.csv"
    source.write_text(
        "signal,context,price\n"
        "1,11,12\n"
        "2,19,18\n"
        ",17,25\n"
        "4,31,32\n"
        "8,,49\n"
        "16,47,60\n"
        "32,59,82\n"
        ",73,105\n"
        "128,83,140\n"
        "256,97,200\n"
        "512,109,310\n"
        "1024,127,490\n",
        encoding="utf-8",
        newline="",
    )
    return load_regression_csv(source, target_column="price")


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    """Keep tests small while exercising repeated cross-validation."""
    return ExperimentConfig(
        n_splits=3,
        n_repeats=2,
        random_state=17,
        ridge_alpha=0.75,
    )


def test_identical_seeded_splits_and_training_only_preprocessing(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe actual pipeline inputs and fitted preprocessing statistics."""
    fit_calls: list[
        tuple[Pipeline, NDArray[np.float64], NDArray[np.float64]]
    ] = []
    prediction_calls: list[
        tuple[Pipeline, NDArray[np.float64]]
    ] = []

    original_fit = Pipeline.fit
    original_predict = Pipeline.predict

    def record_fit(
        self: Pipeline,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        **kwargs: Any,
    ) -> Pipeline:
        fit_calls.append((self, X.copy(), y.copy()))
        return original_fit(self, X, y, **kwargs)

    def record_predict(
        self: Pipeline,
        X: NDArray[np.float64],
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        prediction_calls.append((self, X.copy()))
        return original_predict(self, X, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    monkeypatch.setattr(Pipeline, "predict", record_predict)

    report = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )

    expected_splits = tuple(
        RepeatedKFold(
            n_splits=experiment_config.n_splits,
            n_repeats=experiment_config.n_repeats,
            random_state=experiment_config.random_state,
        ).split(regression_dataset.X)
    )

    assert len(fit_calls) == 2 * len(expected_splits)
    assert len(prediction_calls) == len(fit_calls)

    # Each strategy in each fold must receive a fresh pipeline instance.
    assert len({id(pipeline) for pipeline, _, _ in fit_calls}) == len(fit_calls)

    differs_from_full_dataset = {"mean": False, "median": False}

    for split_index, (train_indices, validation_indices) in enumerate(
        expected_splits
    ):
        assert not np.intersect1d(train_indices, validation_indices).size

        expected_X_train = regression_dataset.X[train_indices]
        expected_y_train = regression_dataset.y[train_indices]
        expected_X_validation = regression_dataset.X[validation_indices]

        assert report.train_sizes[split_index] == len(train_indices)
        assert report.validation_sizes[split_index] == len(validation_indices)

        for strategy_index, strategy in enumerate(("mean", "median")):
            call_index = 2 * split_index + strategy_index
            pipeline, actual_X_train, actual_y_train = fit_calls[call_index]
            prediction_pipeline, actual_X_validation = prediction_calls[
                call_index
            ]

            assert prediction_pipeline is pipeline

            # Raw missing values must reach the pipeline. These checks detect
            # preprocessing applied to the full dataset before splitting.
            np.testing.assert_array_equal(
                actual_X_train, expected_X_train
            )
            np.testing.assert_array_equal(
                actual_y_train, expected_y_train
            )
            np.testing.assert_array_equal(
                actual_X_validation, expected_X_validation
            )

            fitted_imputer = pipeline.named_steps["imputer"]
            fitted_scaler = pipeline.named_steps["scaler"]
            fitted_model = pipeline.named_steps["regressor"]

            assert isinstance(fitted_imputer, SimpleImputer)
            assert isinstance(fitted_scaler, StandardScaler)
            assert isinstance(fitted_model, Ridge)
            assert fitted_imputer.strategy == strategy
            assert fitted_imputer.add_indicator is False
            assert fitted_model.alpha == experiment_config.ridge_alpha
            assert fitted_model.solver == "svd"

            statistic = np.nanmean if strategy == "mean" else np.nanmedian
            expected_statistics = statistic(expected_X_train, axis=0)
            full_dataset_statistics = statistic(
                regression_dataset.X, axis=0
            )

            np.testing.assert_allclose(
                fitted_imputer.statistics_,
                expected_statistics,
                rtol=1e-12,
                atol=1e-12,
            )

            # Establish that this fixture can distinguish training-only
            # statistics from statistics learned using held-out records.
            if not np.allclose(
                expected_statistics,
                full_dataset_statistics,
                rtol=1e-12,
                atol=1e-12,
            ):
                differs_from_full_dataset[strategy] = True

            expected_imputed_training = np.where(
                np.isnan(expected_X_train),
                expected_statistics,
                expected_X_train,
            )

            # Scaling must also learn exclusively from imputed training data.
            np.testing.assert_allclose(
                fitted_scaler.mean_,
                expected_imputed_training.mean(axis=0),
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                fitted_scaler.var_,
                expected_imputed_training.var(axis=0, ddof=0),
                rtol=1e-12,
                atol=1e-12,
            )

    assert all(differs_from_full_dataset.values())


def test_scores_and_summaries_match_independent_reference_evaluation(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    report = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )
    splits = tuple(
        RepeatedKFold(
            n_splits=experiment_config.n_splits,
            n_repeats=experiment_config.n_repeats,
            random_state=experiment_config.random_state,
        ).split(regression_dataset.X)
    )

    expected_scores: dict[str, list[float]] = {
        "mean": [],
        "median": [],
    }

    # Do not use the production pipeline builder or its MAE helper.
    for strategy in ("mean", "median"):
        for train_indices, validation_indices in splits:
            reference = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=strategy,
                            copy=True,
                            add_indicator=False,
                            keep_empty_features=False,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        Ridge(
                            alpha=experiment_config.ridge_alpha,
                            fit_intercept=True,
                            copy_X=True,
                            solver="svd",
                        ),
                    ),
                ]
            )
            reference.fit(
                regression_dataset.X[train_indices],
                regression_dataset.y[train_indices],
            )
            predictions = reference.predict(
                regression_dataset.X[validation_indices]
            )
            absolute_errors = np.abs(
                regression_dataset.y[validation_indices] - predictions
            )
            expected_scores[strategy].append(
                float(np.mean(absolute_errors))
            )

    methods = {result.strategy: result for result in report.methods}
    assert set(methods) == {"mean", "median"}

    baseline_mean = float(np.mean(expected_scores["mean"]))

    for strategy, result in methods.items():
        expected = np.asarray(expected_scores[strategy])

        assert np.isfinite(result.fold_mae).all()
        assert all(score >= 0 for score in result.fold_mae)

        np.testing.assert_allclose(
            result.fold_mae,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )
        assert result.mean_mae == pytest.approx(
            float(np.mean(expected))
        )
        assert result.std_mae == pytest.approx(
            float(np.std(expected, ddof=1))
        )
        assert result.mae_difference_from_baseline == pytest.approx(
            float(np.mean(expected)) - baseline_mean
        )

    assert methods["mean"].mae_difference_from_baseline == 0.0

    np.testing.assert_allclose(
        report.paired_mae_differences,
        np.asarray(expected_scores["median"])
        - np.asarray(expected_scores["mean"]),
        rtol=1e-10,
        atol=1e-10,
    )

    assert report.metric == "MAE"
    assert report.std_ddof == 1
    assert report.baseline_strategy == "mean"
    assert report.config == experiment_config


def test_repeated_runs_reproduce_scores_and_fingerprints(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    first = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )
    second = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )

    assert first.data_fingerprint == regression_dataset.data_fingerprint
    assert first.data_fingerprint == second.data_fingerprint
    assert first.split_fingerprint == second.split_fingerprint
    assert first.train_sizes == second.train_sizes
    assert first.validation_sizes == second.validation_sizes
    assert first.versions == second.versions

    assert len(first.split_fingerprint) == 64
    assert all(
        character in "0123456789abcdef"
        for character in first.split_fingerprint
    )

    for left, right in zip(first.methods, second.methods, strict=True):
        assert left.strategy == right.strategy
        np.testing.assert_allclose(
            left.fold_mae,
            right.fold_mae,
            rtol=1e-12,
            atol=1e-12,
        )
        assert left.mean_mae == pytest.approx(right.mean_mae)
        assert left.std_mae == pytest.approx(right.std_mae)


def test_changing_seed_changes_splits_but_not_data_fingerprint(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    first = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )
    second = compare_imputation(
        regression_dataset,
        config=replace(experiment_config, random_state=29),
    )

    assert first.data_fingerprint == second.data_fingerprint
    assert first.split_fingerprint != second.split_fingerprint


def test_changing_model_alpha_preserves_split_fingerprint(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    first = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )
    second = compare_imputation(
        regression_dataset,
        config=replace(experiment_config, ridge_alpha=2.0),
    )

    assert first.data_fingerprint == second.data_fingerprint
    assert first.split_fingerprint == second.split_fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_splits", -1),
        ("n_splits", 0),
        ("n_splits", 1),
        ("n_splits", 2.5),
        ("n_splits", "3"),
        ("n_splits", True),
        ("n_repeats", -1),
        ("n_repeats", 0),
        ("n_repeats", 1.5),
        ("n_repeats", "2"),
        ("n_repeats", False),
        ("random_state", -1),
        ("random_state", 2**32),
        ("random_state", 1.5),
        ("random_state", "42"),
        ("random_state", True),
        ("ridge_alpha", 0),
        ("ridge_alpha", -1),
        ("ridge_alpha", float("nan")),
        ("ridge_alpha", float("inf")),
        ("ridge_alpha", float("-inf")),
        ("ridge_alpha", True),
        ("ridge_alpha", "1.0"),
        ("ridge_alpha", None),
        ("ridge_alpha", 10**400),
    ],
)
def test_invalid_configuration_is_rejected(
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ImputationExperimentError, match=field):
        ExperimentConfig(**{field: value})


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_boundaries_are_accepted(seed: int) -> None:
    config = ExperimentConfig(random_state=seed)

    assert config.random_state == seed


@pytest.mark.parametrize("invalid_config", [0, "", {}])
def test_invalid_configuration_object_is_rejected(
    regression_dataset: RegressionDataset,
    invalid_config: Any,
) -> None:
    with pytest.raises(ImputationExperimentError, match="ExperimentConfig"):
        compare_imputation(
            regression_dataset,
            config=invalid_config,
        )


def test_single_repetition_is_supported(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    config = replace(experiment_config, n_repeats=1)

    report = compare_imputation(regression_dataset, config=config)

    for result in report.methods:
        assert len(result.fold_mae) == config.n_splits
        assert math_is_finite_nonnegative(result.std_mae)


def math_is_finite_nonnegative(value: float) -> bool:
    """Check metric validity without imposing an expected performance level."""
    return bool(np.isfinite(value) and value >= 0)


@pytest.mark.parametrize(
    ("n_records", "n_splits", "message"),
    [
        (2, 3, "exceeds the number"),
        (3, 4, "exceeds the number"),
        (2, 2, "fewer than two training records"),
        (3, 2, "fewer than two training records"),
    ],
)
def test_insufficient_records_or_training_fold_size(
    tmp_path: Path,
    n_records: int,
    n_splits: int,
    message: str,
) -> None:
    source = tmp_path / "small.csv"
    source.write_text(
        "feature,price\n"
        + "".join(
            f"{index + 1},{100 + index}\n"
            for index in range(n_records)
        ),
        encoding="utf-8",
    )
    dataset = load_regression_csv(source, target_column="price")

    with pytest.raises(ImputationExperimentError, match=message):
        compare_imputation(
            dataset,
            config=ExperimentConfig(
                n_splits=n_splits,
                n_repeats=1,
                random_state=17,
            ),
        )


def test_training_fold_with_entirely_missing_feature_fails_before_fitting(
    tmp_path: Path,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_records = 12
    splitter = RepeatedKFold(
        n_splits=experiment_config.n_splits,
        n_repeats=experiment_config.n_repeats,
        random_state=experiment_config.random_state,
    )
    _, first_validation_indices = next(
        splitter.split(np.zeros((n_records, 2)))
    )
    observed_record = int(first_validation_indices[0])

    source = tmp_path / "sparse-feature.csv"
    records = ["sparse,observed,price"]
    for index in range(n_records):
        sparse = "10" if index == observed_record else ""
        records.append(f"{sparse},{index + 1},{100 + index}")

    source.write_text(
        "\n".join(records) + "\n",
        encoding="utf-8",
    )
    dataset = load_regression_csv(source, target_column="price")

    # The whole dataset has an observation, so loading correctly succeeds.
    assert np.count_nonzero(~np.isnan(dataset.X[:, 0])) == 1

    def unexpected_pipeline_creation(*args: Any, **kwargs: Any) -> Pipeline:
        raise AssertionError(
            "No pipeline should be constructed before fold validation passes."
        )

    monkeypatch.setattr(
        imputation,
        "_build_pipeline",
        unexpected_pipeline_creation,
    )

    with pytest.raises(
        ImputationExperimentError,
        match="no observed training values",
    ) as error:
        compare_imputation(dataset, config=experiment_config)

    assert "'sparse'" in str(error.value)
    assert "repetition 1, fold 1" in str(error.value)


@pytest.mark.parametrize("stage", ["fit", "predict"])
@pytest.mark.parametrize(
    "exception_type",
    [
        ValueError,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
    ],
)
def test_numerical_failures_include_strategy_and_fold_context(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    exception_type: type[Exception],
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise exception_type("deliberate numerical failure")

    monkeypatch.setattr(Pipeline, stage, fail)

    with pytest.raises(
        ImputationExperimentError,
        match="deliberate numerical failure",
    ) as error:
        compare_imputation(
            regression_dataset,
            config=experiment_config,
        )

    assert "'mean'" in str(error.value)
    assert "repetition 1, fold 1" in str(error.value)
    assert "No partial experiment report" in str(error.value)
    assert isinstance(error.value.__cause__, exception_type)


def test_runtime_warning_is_converted_to_experiment_error(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warn_during_fit(
        self: Pipeline,
        *args: Any,
        **kwargs: Any,
    ) -> Pipeline:
        warnings.warn(
            "deliberate numerical warning",
            RuntimeWarning,
            stacklevel=2,
        )
        return self

    monkeypatch.setattr(Pipeline, "fit", warn_during_fit)

    with pytest.raises(
        ImputationExperimentError,
        match="deliberate numerical warning",
    ) as error:
        compare_imputation(
            regression_dataset,
            config=experiment_config,
        )

    assert isinstance(error.value.__cause__, RuntimeWarning)


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_rejected(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    prediction: float,
) -> None:
    def invalid_predict(
        self: Pipeline,
        X: NDArray[np.float64],
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        return np.full(X.shape[0], prediction, dtype=np.float64)

    monkeypatch.setattr(Pipeline, "predict", invalid_predict)

    with pytest.raises(
        ImputationExperimentError,
        match="non-finite predictions",
    ):
        compare_imputation(
            regression_dataset,
            config=experiment_config,
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.0])
def test_invalid_mae_is_rejected(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
) -> None:
    def invalid_mae(*args: Any, **kwargs: Any) -> float:
        return score

    monkeypatch.setattr(imputation, "mean_absolute_error", invalid_mae)

    with pytest.raises(
        ImputationExperimentError,
        match="MAE must be a finite, nonnegative number",
    ):
        compare_imputation(
            regression_dataset,
            config=experiment_config,
        )


def test_read_only_arrays_and_original_file_are_preserved(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    original_X = regression_dataset.X.copy()
    original_y = regression_dataset.y.copy()
    original_bytes = regression_dataset.source_path.read_bytes()
    original_fingerprint = regression_dataset.data_fingerprint

    assert not regression_dataset.X.flags.writeable
    assert not regression_dataset.y.flags.writeable

    report = compare_imputation(
        regression_dataset,
        config=experiment_config,
    )

    np.testing.assert_array_equal(regression_dataset.X, original_X)
    np.testing.assert_array_equal(regression_dataset.y, original_y)
    assert not regression_dataset.X.flags.writeable
    assert not regression_dataset.y.flags.writeable
    assert regression_dataset.source_path.read_bytes() == original_bytes
    assert regression_dataset.data_fingerprint == original_fingerprint
    assert report.data_fingerprint == original_fingerprint


def test_failed_experiment_preserves_arrays_and_source_file(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_X = regression_dataset.X.copy()
    original_y = regression_dataset.y.copy()
    original_bytes = regression_dataset.source_path.read_bytes()

    def fail_fit(*args: Any, **kwargs: Any) -> Any:
        raise FloatingPointError("deliberate failure")

    monkeypatch.setattr(Pipeline, "fit", fail_fit)

    with pytest.raises(ImputationExperimentError, match="deliberate failure"):
        compare_imputation(
            regression_dataset,
            config=experiment_config,
        )

    np.testing.assert_array_equal(regression_dataset.X, original_X)
    np.testing.assert_array_equal(regression_dataset.y, original_y)
    assert not regression_dataset.X.flags.writeable
    assert not regression_dataset.y.flags.writeable
    assert regression_dataset.source_path.read_bytes() == original_bytes


def test_csv_entry_point_uses_requested_settings(
    regression_dataset: RegressionDataset,
    experiment_config: ExperimentConfig,
) -> None:
    report = run_imputation_experiment(
        regression_dataset.source_path,
        target_column="price",
        config=experiment_config,
    )

    assert report.config == experiment_config
    assert report.target_name == "price"
    assert report.feature_names == regression_dataset.feature_names
    assert report.data_fingerprint == regression_dataset.data_fingerprint
    assert report.n_samples == regression_dataset.n_samples
    assert report.n_features == regression_dataset.n_features
    assert report.missing_values_per_feature == {
        "signal": 2,
        "context": 1,
    }


def test_complete_data_produces_equivalent_strategy_scores_and_note(
    tmp_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    source = tmp_path / "complete.csv"
    source.write_text(
        "feature,context,price\n"
        + "".join(
            f"{index},{index * index},{3 * index + index % 3}\n"
            for index in range(1, 13)
        ),
        encoding="utf-8",
    )
    dataset = load_regression_csv(source, target_column="price")

    report = compare_imputation(dataset, config=experiment_config)
    methods = {result.strategy: result for result in report.methods}

    assert report.missing_values_per_feature == {
        "feature": 0,
        "context": 0,
    }
    np.testing.assert_allclose(
        methods["mean"].fold_mae,
        methods["median"].fold_mae,
        rtol=1e-12,
        atol=1e-12,
    )
    assert methods["median"].mae_difference_from_baseline == pytest.approx(
        0.0, abs=1e-12
    )
    assert any(
        "No missing feature values were present" in note
        for note in report.notes
    )