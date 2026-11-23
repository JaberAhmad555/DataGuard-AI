"""Compare mean and median imputation using paired regression experiments.

Both strategies use identical repeated cross-validation splits and identical
Ridge regression settings. Every fold receives a newly constructed pipeline;
imputation and scaling are fitted exclusively on that fold's training records.

Results describe this dataset, model, and evaluation protocol. Cross-validation
folds are dependent, so their standard deviation is descriptive variability,
not a confidence interval or evidence of statistical significance.
"""

from __future__ import annotations

import hashlib
import math
import platform
import statistics
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataguard.experiments.data import (
    RegressionDataset,
    load_regression_csv,
)

ImputationStrategy = Literal["mean", "median"]

STRATEGIES: tuple[ImputationStrategy, ...] = ("mean", "median")


class ImputationExperimentError(ValueError):
    """Raised when an experiment cannot be configured or evaluated safely."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Settings shared by both imputation strategies.

    n_splits:
        Number of folds per repetition; must be at least two.
    n_repeats:
        Number of independently shuffled repetitions. One is allowed for a
        single K-fold evaluation; the default performs three repetitions.
    random_state:
        Integer seed used to generate one shared sequence of splits.
    ridge_alpha:
        Positive, finite regularization strength for Ridge regression.

    Ridge uses the deterministic SVD solver. The random seed controls splits;
    the selected Ridge solver does not use a random number generator.
    """

    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 42
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid settings before any data is fitted."""
        for name in ("n_splits", "n_repeats", "random_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ImputationExperimentError(
                    f"{name} must be a Python integer."
                )

        if self.n_splits < 2:
            raise ImputationExperimentError("n_splits must be at least 2.")

        if self.n_repeats < 1:
            raise ImputationExperimentError("n_repeats must be at least 1.")

        if not 0 <= self.random_state <= 2**32 - 1:
            raise ImputationExperimentError(
                "random_state must be between 0 and 2**32 - 1."
            )

        if isinstance(self.ridge_alpha, bool) or not isinstance(
            self.ridge_alpha, (int, float)
        ):
            raise ImputationExperimentError(
                "ridge_alpha must be a positive, finite number."
            )

        try:
            alpha = float(self.ridge_alpha)
        except (ValueError, OverflowError) as exc:
            raise ImputationExperimentError(
                "ridge_alpha cannot be represented as a finite float."
            ) from exc

        if not math.isfinite(alpha) or alpha <= 0:
            raise ImputationExperimentError(
                "ridge_alpha must be a positive, finite number."
            )

        object.__setattr__(self, "ridge_alpha", alpha)


@dataclass(frozen=True, slots=True)
class ImputationMethodResult:
    """MAE measurements for one strategy.

    fold_mae follows repetition order, then fold order within each repetition.
    Corresponding positions across strategies refer to identical held-out rows.

    std_mae is the sample standard deviation across fold MAEs, using ddof=1.
    mae_difference_from_baseline equals this method's mean MAE minus the
    mean-imputation method's mean MAE. Negative differences favor this method.
    """

    strategy: ImputationStrategy
    fold_mae: tuple[float, ...]
    mean_mae: float
    std_mae: float
    mae_difference_from_baseline: float


@dataclass(frozen=True, slots=True)
class ImputationReport:
    """Structured results and settings needed to interpret an experiment.

    Split indices are zero-based positions in the loaded dataset. Their hash,
    together with the data fingerprint, seed, settings, and dependency
    versions, helps identify the exact evaluation setup.

    Fold MAEs receive equal weight in the summary. When fold sizes differ,
    this is not exactly the same as pooling every held-out absolute error.
    """

    config: ExperimentConfig
    target_name: str
    feature_names: tuple[str, ...]
    n_samples: int
    n_features: int
    missing_values_per_feature: dict[str, int]
    data_fingerprint: str
    split_fingerprint: str
    train_sizes: tuple[int, ...]
    validation_sizes: tuple[int, ...]
    methods: tuple[ImputationMethodResult, ...]
    paired_mae_differences: tuple[float, ...]
    versions: dict[str, str]
    notes: tuple[str, ...]
    metric: str = "MAE"
    baseline_strategy: str = "mean"
    model: str = "Ridge"
    ridge_solver: str = "svd"
    preprocessing: tuple[str, ...] = ("SimpleImputer", "StandardScaler")
    std_ddof: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return a detached representation suitable for JSON serialization."""
        return asdict(self)


def run_imputation_experiment(
    path: str | Path,
    *,
    target_column: str,
    config: ExperimentConfig | None = None,
) -> ImputationReport:
    """Load a CSV and compare numeric-feature imputation strategies.

    Raises:
        RegressionDataError: Propagated from the dataset loader for invalid
            CSV input or unsupported regression data.
        ImputationExperimentError: Invalid settings, unusable training folds,
            or numerical failures during evaluation.

    The original CSV is only read. No repaired dataset or fitted model is
    written to disk, and no final model is fitted on the complete dataset.
    """
    settings = _resolve_config(config)
    dataset = load_regression_csv(path, target_column=target_column)
    return compare_imputation(dataset, config=settings)


def compare_imputation(
    dataset: RegressionDataset,
    *,
    config: ExperimentConfig | None = None,
) -> ImputationReport:
    """Evaluate a loaded dataset using identical splits for both strategies.

    Independent copies of the supplied arrays are used. No caller-owned array
    is modified, and no learned preprocessing occurs before splitting.

    Every training fold is checked before any model is fitted. If a feature
    has no observed values in a training fold, the entire experiment fails
    explicitly: no zero fallback, column dropping, or fold skipping is used.

    Observations must be suitable for shuffled K-fold evaluation. This function
    cannot infer temporal order, repeated entities, numeric category codes, or
    target-derived features from numeric arrays.
    """
    settings = _resolve_config(config)
    X, y = _validated_array_copies(dataset)

    if settings.n_splits > len(y):
        raise ImputationExperimentError(
            f"n_splits={settings.n_splits} exceeds the number of "
            f"data records ({len(y)})."
        )

    splitter = RepeatedKFold(
        n_splits=settings.n_splits,
        n_repeats=settings.n_repeats,
        random_state=settings.random_state,
    )

    # Materialize once: both strategies reuse these exact index arrays.
    splits = tuple(splitter.split(X))

    for split_index, (train_indices, _) in enumerate(splits):
        label = _split_label(split_index, settings.n_splits)

        if len(train_indices) < 2:
            raise ImputationExperimentError(
                f"{label} has fewer than two training records. "
                "Provide more data or choose suitable cross-validation settings."
            )

        empty_features = np.isnan(X[train_indices]).all(axis=0)
        if np.any(empty_features):
            names = ", ".join(
                repr(name)
                for name, is_empty in zip(
                    dataset.feature_names, empty_features, strict=True
                )
                if is_empty
            )
            raise ImputationExperimentError(
                f"{label} has no observed training values for: {names}. "
                "Mean and median cannot be estimated for these features. "
                "The experiment was not run."
            )

    scores: dict[ImputationStrategy, list[float]] = {
        strategy: [] for strategy in STRATEGIES
    }

    for split_index, (train_indices, validation_indices) in enumerate(splits):
        label = _split_label(split_index, settings.n_splits)

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        for strategy in STRATEGIES:
            pipeline = _build_pipeline(strategy, settings.ridge_alpha)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    with np.errstate(
                        over="raise", invalid="raise", divide="raise"
                    ):
                        pipeline.fit(X_train, y_train)
                        predictions = pipeline.predict(X_validation)

                        if not np.isfinite(predictions).all():
                            raise ValueError(
                                "The model produced non-finite predictions."
                            )

                        mae = float(
                            mean_absolute_error(y_validation, predictions)
                        )

                        if not math.isfinite(mae) or mae < 0:
                            raise ValueError(
                                "MAE must be a finite, nonnegative number."
                            )

            except (
                ValueError,
                FloatingPointError,
                OverflowError,
                np.linalg.LinAlgError,
                RuntimeWarning,
            ) as exc:
                raise ImputationExperimentError(
                    f"{strategy!r} imputation failed in {label}: {exc}. "
                    "No partial experiment report is returned."
                ) from exc

            scores[strategy].append(mae)

    summaries = {
        strategy: _summarize_scores(scores[strategy])
        for strategy in STRATEGIES
    }
    baseline_mean = summaries["mean"][0]

    results = tuple(
        ImputationMethodResult(
            strategy=strategy,
            fold_mae=tuple(scores[strategy]),
            mean_mae=summaries[strategy][0],
            std_mae=summaries[strategy][1],
            mae_difference_from_baseline=(
                summaries[strategy][0] - baseline_mean
            ),
        )
        for strategy in STRATEGIES
    )

    missing_counts = np.isnan(X).sum(axis=0)
    missing_by_feature = {
        name: int(count)
        for name, count in zip(
            dataset.feature_names, missing_counts, strict=True
        )
    }

    notes = [
        "Lower MAE is better; MAE is expressed in the target's units.",
        "Differences are median MAE minus mean-imputation baseline MAE; "
        "negative differences favor median imputation.",
        "Summary statistics weight each fold equally. Standard deviation "
        "uses ddof=1 across fold MAEs.",
        "Repeated cross-validation folds are dependent. Their standard "
        "deviation is not a confidence interval or a significance test.",
        "These are comparative cross-validation results, not an independent "
        "test-set evaluation after choosing an imputation strategy.",
        "Shuffled K-fold evaluation assumes independent, exchangeable records. "
        "Time series and grouped or repeated-entity data require other splits.",
        "Numeric category codes and target-derived features are not detected "
        "automatically. Duplicate observations can also bias evaluation.",
        "Reproduction requires the same data, row order, settings, and software "
        "environment. Floating-point results may vary across platforms.",
    ]

    if not any(missing_by_feature.values()):
        notes.append(
            "No missing feature values were present. This comparison does "
            "not measure the effect of repairing missing values."
        )

    return ImputationReport(
        config=settings,
        target_name=dataset.target_name,
        feature_names=dataset.feature_names,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        missing_values_per_feature=missing_by_feature,
        data_fingerprint=dataset.data_fingerprint,
        split_fingerprint=_fingerprint_splits(splits),
        train_sizes=tuple(len(train) for train, _ in splits),
        validation_sizes=tuple(len(valid) for _, valid in splits),
        methods=results,
        paired_mae_differences=tuple(
            median_mae - mean_mae
            for median_mae, mean_mae in zip(
                scores["median"], scores["mean"], strict=True
            )
        ),
        versions={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dataguard-ai": _package_version("dataguard-ai"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
        },
        notes=tuple(notes),
    )


def _resolve_config(
    config: ExperimentConfig | None,
) -> ExperimentConfig:
    """Supply defaults or reject an incompatible configuration object."""
    if config is None:
        return ExperimentConfig()

    if not isinstance(config, ExperimentConfig):
        raise ImputationExperimentError(
            "config must be an ExperimentConfig instance or None."
        )

    return config


def _validated_array_copies(
    dataset: RegressionDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Defensively check the dataset contract and copy its numeric arrays."""
    if not isinstance(dataset, RegressionDataset):
        raise ImputationExperimentError(
            "dataset must be a RegressionDataset from load_regression_csv."
        )

    if not isinstance(dataset.X, np.ndarray) or not isinstance(
        dataset.y, np.ndarray
    ):
        raise ImputationExperimentError("Dataset X and y must be NumPy arrays.")

    if dataset.X.ndim != 2 or dataset.y.ndim != 1:
        raise ImputationExperimentError(
            "X must be a two-dimensional matrix and y a one-dimensional array."
        )

    if dataset.X.dtype != np.float64 or dataset.y.dtype != np.float64:
        raise ImputationExperimentError(
            "X and y must have float64 dtype. Use load_regression_csv "
            "to validate and convert CSV input."
        )

    if dataset.X.shape[0] != dataset.y.shape[0]:
        raise ImputationExperimentError(
            "X and y must contain the same number of records."
        )

    if dataset.X.shape[0] < 2 or dataset.X.shape[1] < 1:
        raise ImputationExperimentError(
            "At least two records and one feature are required."
        )

    names = dataset.feature_names
    if (
        not isinstance(names, tuple)
        or len(names) != dataset.X.shape[1]
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ImputationExperimentError(
            "feature_names must contain one unique, nonblank name per feature."
        )

    if (
        not isinstance(dataset.target_name, str)
        or not dataset.target_name.strip()
        or dataset.target_name in names
    ):
        raise ImputationExperimentError(
            "The target name must be nonblank and excluded from feature_names."
        )

    X = dataset.X.copy()
    y = dataset.y.copy()

    if np.isinf(X).any():
        raise ImputationExperimentError(
            "Feature values must be finite numbers or NaN for missing values."
        )

    if not np.isfinite(y).all():
        raise ImputationExperimentError(
            "Target values must be finite and cannot be missing."
        )

    if np.all(y == y[0]):
        raise ImputationExperimentError(
            "A constant target is unsupported for this experiment."
        )

    return X, y


def _build_pipeline(
    strategy: ImputationStrategy,
    ridge_alpha: float,
) -> Pipeline:
    """Construct a fresh pipeline with explicitly shared model settings."""
    return Pipeline(
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
            (
                "scaler",
                StandardScaler(
                    copy=True,
                    with_mean=True,
                    with_std=True,
                ),
            ),
            (
                "regressor",
                Ridge(
                    alpha=ridge_alpha,
                    fit_intercept=True,
                    copy_X=True,
                    solver="svd",
                ),
            ),
        ]
    )


def _summarize_scores(scores: list[float]) -> tuple[float, float]:
    """Calculate mean MAE and sample standard deviation without rounding."""
    if len(scores) < 2:
        raise ImputationExperimentError(
            "At least two fold scores are required to calculate standard deviation."
        )

    if any(not math.isfinite(score) or score < 0 for score in scores):
        raise ImputationExperimentError(
            "Cannot summarize non-finite or negative MAE scores."
        )

    try:
        mean_mae = float(statistics.mean(scores))
        std_mae = float(statistics.stdev(scores))
    except (OverflowError, ValueError) as exc:
        raise ImputationExperimentError(
            "MAE summary statistics could not be represented safely."
        ) from exc

    if not math.isfinite(mean_mae) or not math.isfinite(std_mae):
        raise ImputationExperimentError(
            "MAE summary statistics are non-finite."
        )

    return mean_mae, std_mae


def _split_label(split_index: int, n_splits: int) -> str:
    """Return a one-based repetition and fold label for error messages."""
    repetition, fold = divmod(split_index, n_splits)
    return f"repetition {repetition + 1}, fold {fold + 1}"


def _fingerprint_splits(
    splits: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...],
) -> str:
    """Hash ordered split indices using a platform-independent byte format."""
    digest = hashlib.sha256()
    digest.update(b"dataguard-repeated-kfold-splits-v1\0")
    digest.update(len(splits).to_bytes(8, byteorder="little", signed=False))

    for train_indices, validation_indices in splits:
        for indices in (train_indices, validation_indices):
            digest.update(
                len(indices).to_bytes(8, byteorder="little", signed=False)
            )
            digest.update(
                np.asarray(indices, dtype="<i8").tobytes(order="C")
            )

    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    """Read installed package metadata without inventing a version."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"