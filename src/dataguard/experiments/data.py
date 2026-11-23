"""Load and validate numeric regression datasets for the ML Impact Lab.

This module performs structural validation and numeric conversion only.
It never imputes, scales, selects features, fits a model, or splits data.

CSV parsing uses DataGuard's existing reader, preserving v0.1 parsing rules.
Missing feature values remain NaN until a training-fold pipeline handles them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dataguard.csv_io import CSVInputError, read_csv


class RegressionDataError(ValueError):
    """Raised when input cannot be used for a numeric regression experiment."""


@dataclass(frozen=True, slots=True, eq=False)
class RegressionDataset:
    """Validated regression data returned by ``load_regression_csv``.

    Attributes:
        X: Feature matrix with shape (n_samples, n_features). Missing feature
            values are represented by NaN; observed values are finite floats.
        y: Finite target values with shape (n_samples,).
        feature_names: Feature names in their original CSV column order.
        target_name: Exact name of the selected target column.
        source_path: Input path, retained for identification.
        data_fingerprint: SHA-256 fingerprint of the parsed CSV contents and
            selected target. This is not a hash of the original file bytes.

    Attribute reassignment is disabled. The loader also marks X and y as
    read-only to guard against accidental in-place changes. This is a
    programming safeguard, not a guarantee against deliberate mutation.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    feature_names: tuple[str, ...]
    target_name: str
    source_path: Path
    data_fingerprint: str

    @property
    def n_samples(self) -> int:
        """Return the number of data records, excluding the CSV header."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Return the number of feature columns, excluding the target."""
        return int(self.X.shape[1])

    @property
    def missing_values_per_feature(self) -> dict[str, int]:
        """Return descriptive missing counts in original feature order.

        These counts are for reporting only. They must not be used to derive
        imputation values or other fitted preprocessing parameters.
        """
        counts = np.isnan(self.X).sum(axis=0)
        return {
            name: int(count)
            for name, count in zip(self.feature_names, counts, strict=True)
        }

    @property
    def total_missing_values(self) -> int:
        """Return the number of missing feature cells."""
        return int(np.count_nonzero(np.isnan(self.X)))


def load_regression_csv(
    path: str | Path,
    *,
    target_column: str,
) -> RegressionDataset:
    """Read a CSV and validate it for numeric, single-target regression.

    Every column other than target_column becomes a feature. Columns are
    never silently dropped, and the target is never included in X.

    Input rules:
        * The existing CSV reader's encoding, header, and row-width rules apply.
        * The target name must exactly match a CSV column name.
        * At least two data records and one feature column are required.
        * Empty or whitespace-only feature fields become NaN.
        * All observed feature values and all target values must convert to
          finite float64 values.
        * Missing targets, constant targets, and entirely missing feature
          columns are rejected.
        * Literal missing-value tokens such as "NA", "null", and "NaN" are
          rejected, rather than silently interpreted as missing data.

    Numeric targets, including integer-valued targets, are treated as
    regression responses. Numeric category codes cannot reliably be
    distinguished from regression responses automatically; callers must
    supply a genuine regression target.

    Missing values are not repaired here. A feature can still be entirely
    missing within an individual training fold even if it passes this loader.
    The experiment runner must validate that condition before fitting.

    Record numbers in errors are one-based data-record positions, not physical
    file line numbers. Blank lines and multiline quoted fields follow the
    existing CSV reader's behavior.

    Args:
        path: Path to the input CSV.
        target_column: Exact name of the numeric regression target.

    Returns:
        A dataset with read-only numeric arrays and reproducibility metadata.

    Raises:
        RegressionDataError: If the file cannot be read, the CSV is invalid,
            or the data violates the numeric regression requirements.

    The entire dataset is loaded into memory. The source file is never written.
    """
    if not isinstance(target_column, str) or not target_column.strip():
        raise RegressionDataError(
            "target_column must be a nonempty column name."
        )

    if not isinstance(path, (str, Path)):
        raise RegressionDataError("path must be a string or pathlib.Path.")

    if isinstance(path, str) and not path.strip():
        raise RegressionDataError("The CSV path must not be empty.")

    source = Path(path)

    try:
        columns, rows = read_csv(source)
    except CSVInputError as exc:
        raise RegressionDataError(
            f"Cannot load regression dataset: {exc}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise RegressionDataError(
            f"Cannot read CSV path {str(source)!r}: {exc}"
        ) from exc

    if target_column not in columns:
        available = ", ".join(repr(name) for name in columns)
        raise RegressionDataError(
            f"Target column {target_column!r} was not found. "
            f"Available columns: {available}."
        )

    target_index = columns.index(target_column)
    feature_indices = tuple(
        index for index in range(len(columns)) if index != target_index
    )
    feature_names = tuple(columns[index] for index in feature_indices)

    if not feature_names:
        raise RegressionDataError(
            "The dataset must contain at least one feature column "
            "in addition to the target."
        )

    if len(rows) < 2:
        raise RegressionDataError(
            "Regression experiments require at least two data records. "
            "Cross-validation settings may require additional records."
        )

    X = np.empty((len(rows), len(feature_names)), dtype=np.float64)
    y = np.empty(len(rows), dtype=np.float64)

    for row_index, row in enumerate(rows):
        record_number = row_index + 1

        y[row_index] = _parse_numeric_field(
            row[target_index],
            column=target_column,
            record_number=record_number,
            allow_missing=False,
        )

        for feature_index, csv_index in enumerate(feature_indices):
            X[row_index, feature_index] = _parse_numeric_field(
                row[csv_index],
                column=columns[csv_index],
                record_number=record_number,
                allow_missing=True,
            )

    entirely_missing = np.isnan(X).all(axis=0)
    missing_columns = [
        name
        for name, is_missing in zip(
            feature_names, entirely_missing, strict=True
        )
        if is_missing
    ]

    if missing_columns:
        names = ", ".join(repr(name) for name in missing_columns)
        raise RegressionDataError(
            f"Feature columns contain no observed numeric values: {names}. "
            "Mean and median imputation require observed training values."
        )

    if np.all(y == y[0]):
        raise RegressionDataError(
            f"Target column {target_column!r} is constant. "
            "This experiment requires a varying regression target."
        )

    fingerprint = _fingerprint_dataset(columns, rows, target_column)

    X.setflags(write=False)
    y.setflags(write=False)

    return RegressionDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_column,
        source_path=source,
        data_fingerprint=fingerprint,
    )


def _parse_numeric_field(
    value: str,
    *,
    column: str,
    record_number: int,
    allow_missing: bool,
) -> float:
    """Convert one field without learning anything from other records."""
    stripped = value.strip()

    if not stripped:
        if allow_missing:
            return float("nan")
        raise RegressionDataError(
            f"Target column {column!r} is missing at data record "
            f"{record_number}. Targets must be present; no records are dropped."
        )

    try:
        number = float(stripped)
    except (ValueError, OverflowError) as exc:
        raise RegressionDataError(
            f"Column {column!r} contains a nonnumeric or unrepresentable "
            f"value at data record {record_number}. "
            "Only numeric features and a numeric regression target are supported."
        ) from exc

    if not math.isfinite(number):
        raise RegressionDataError(
            f"Column {column!r} contains a non-finite numeric value at "
            f"data record {record_number}. Infinity, NaN tokens, and values "
            "outside the finite float64 range are unsupported. "
            "Use an empty field for a missing feature value."
        )

    return number


def _fingerprint_dataset(
    columns: tuple[str, ...],
    rows: list[tuple[str, ...]],
    target_column: str,
) -> str:
    """Hash the parsed table and target selection using a versioned format.

    Column order, row order, and original parsed field strings are significant.
    Paths, CSV quoting syntax, skipped blank lines, and a UTF-8 BOM are not.

    Hashing the already-loaded records avoids rereading a file that could have
    changed after parsing. Incremental JSON encoding avoids constructing an
    additional serialized copy of the entire dataset.
    """
    payload = {
        "columns": columns,
        "rows": rows,
        "target_column": target_column,
    }
    encoder = json.JSONEncoder(
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256()
    digest.update(b"dataguard-regression-input-v1\0")

    for chunk in encoder.iterencode(payload):
        digest.update(chunk.encode("utf-8"))

    return digest.hexdigest()