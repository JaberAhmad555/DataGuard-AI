"""Public API for DataGuard's numeric regression imputation experiments.

Requires the optional ML dependencies installed with ``dataguard-ai[ml]``.
Importing this subpackage does not load datasets or run experiments.
"""

from dataguard.experiments.data import (
    RegressionDataError,
    RegressionDataset,
    load_regression_csv,
)
from dataguard.experiments.imputation import (
    ExperimentConfig,
    ImputationExperimentError,
    ImputationMethodResult,
    ImputationReport,
    ImputationStrategy,
    compare_imputation,
    run_imputation_experiment,
)

__all__ = [
    "ExperimentConfig",
    "ImputationExperimentError",
    "ImputationMethodResult",
    "ImputationReport",
    "ImputationStrategy",
    "RegressionDataError",
    "RegressionDataset",
    "compare_imputation",
    "load_regression_csv",
    "run_imputation_experiment",
]