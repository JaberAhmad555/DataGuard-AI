"""Tests for numeric regression CSV loading and validation.

Requires the development and ML extras: pip install -e ".[dev,ml]".

All input files are created under pytest's tmp_path. These tests do not
depend on the housing example, modify project datasets, or train models.
"""

from pathlib import Path

import numpy as np
import pytest

from dataguard.csv_io import CSVInputError
from dataguard.experiments.data import (
    RegressionDataError,
    RegressionDataset,
    load_regression_csv,
)


def _write_csv(
    tmp_path: Path,
    content: str,
    *,
    filename: str = "regression.csv",
    encoding: str = "utf-8",
) -> Path:
    """Write a temporary CSV without newline translation."""
    source = tmp_path / filename
    source.write_text(content, encoding=encoding, newline="")
    return source


def test_valid_dataset_preserves_feature_order_and_excludes_target(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "size,price,bedrooms,age\n"
        "1000,200000,2,10\n"
        "1500,320000,3,5\n"
        "2000,450000,4,2\n",
    )

    dataset = load_regression_csv(source, target_column="price")

    assert isinstance(dataset, RegressionDataset)
    assert dataset.source_path == source
    assert dataset.target_name == "price"
    assert dataset.feature_names == ("size", "bedrooms", "age")
    assert dataset.target_name not in dataset.feature_names
    assert dataset.n_samples == 3
    assert dataset.n_features == 3
    assert dataset.X.shape == (3, 3)
    assert dataset.y.shape == (3,)
    assert dataset.X.dtype == np.dtype("float64")
    assert dataset.y.dtype == np.dtype("float64")

    np.testing.assert_array_equal(
        dataset.X,
        np.array(
            [[1000, 2, 10], [1500, 3, 5], [2000, 4, 2]],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        dataset.y,
        np.array([200000, 320000, 450000], dtype=np.float64),
    )

    assert dataset.missing_values_per_feature == {
        "size": 0,
        "bedrooms": 0,
        "age": 0,
    }
    assert dataset.total_missing_values == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0.0),
        ("0042", 42.0),
        (" 1.25 ", 1.25),
        ("+2.5", 2.5),
        ("-3.75", -3.75),
        ("1e3", 1000.0),
        ("-2.5E-2", -0.025),
    ],
)
def test_finite_numeric_strings_convert_in_features_and_target(
    tmp_path: Path,
    text: str,
    expected: float,
) -> None:
    source = _write_csv(
        tmp_path,
        f"feature,price\n{text},{text}\n99.5,123.5\n",
    )

    dataset = load_regression_csv(source, target_column="price")

    assert dataset.X[0, 0] == pytest.approx(expected)
    assert dataset.y[0] == pytest.approx(expected)
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.y).all()


@pytest.mark.parametrize("missing", ["", " ", "\t", " \t "])
def test_blank_feature_fields_become_nan(
    tmp_path: Path,
    missing: str,
) -> None:
    source = _write_csv(
        tmp_path,
        f"feature,price\n{missing},100\n2,200\n3,300\n",
    )

    dataset = load_regression_csv(source, target_column="price")

    assert np.isnan(dataset.X[0, 0])
    np.testing.assert_array_equal(dataset.X[1:, 0], [2.0, 3.0])
    assert dataset.missing_values_per_feature == {"feature": 1}
    assert dataset.total_missing_values == 1
    assert np.isfinite(dataset.y).all()


def test_missing_counts_cover_multiple_features_without_imputation(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "size,bedrooms,age,price\n"
        "1000,2,,200000\n"
        ",3,5,320000\n"
        "2000,,2,450000\n",
    )

    dataset = load_regression_csv(source, target_column="price")

    np.testing.assert_allclose(
        dataset.X,
        [
            [1000.0, 2.0, np.nan],
            [np.nan, 3.0, 5.0],
            [2000.0, np.nan, 2.0],
        ],
        equal_nan=True,
    )
    assert dataset.missing_values_per_feature == {
        "size": 1,
        "bedrooms": 1,
        "age": 1,
    }
    assert list(dataset.missing_values_per_feature) == [
        "size",
        "bedrooms",
        "age",
    ]
    assert dataset.total_missing_values == 3

    # Mutating an exported count dictionary does not alter dataset state.
    counts = dataset.missing_values_per_feature
    counts["size"] = 999
    assert dataset.missing_values_per_feature["size"] == 1


@pytest.mark.parametrize("missing", ["", " ", "\t", " \t "])
def test_missing_target_is_rejected_without_dropping_records(
    tmp_path: Path,
    missing: str,
) -> None:
    source = _write_csv(
        tmp_path,
        f"feature,price\n1,{missing}\n2,200\n3,300\n",
    )

    with pytest.raises(
        RegressionDataError,
        match=r"Target column 'price' is missing at data record 1",
    ):
        load_regression_csv(source, target_column="price")


@pytest.mark.parametrize("target", ["unknown", "Price", " price"])
def test_target_name_must_match_header_exactly(
    tmp_path: Path,
    target: str,
) -> None:
    source = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,200\n",
    )

    with pytest.raises(RegressionDataError, match="was not found") as error:
        load_regression_csv(source, target_column=target)

    assert "Available columns:" in str(error.value)
    assert "'feature'" in str(error.value)
    assert "'price'" in str(error.value)


def test_target_header_whitespace_is_preserved_when_matched_exactly(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "feature, price \n1,100\n2,200\n",
    )

    dataset = load_regression_csv(source, target_column=" price ")

    assert dataset.target_name == " price "
    assert dataset.feature_names == ("feature",)
    np.testing.assert_array_equal(dataset.y, [100.0, 200.0])


@pytest.mark.parametrize("target", ["", " ", "\t"])
def test_blank_target_argument_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    source = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,200\n",
    )

    with pytest.raises(RegressionDataError, match="nonempty column name"):
        load_regression_csv(source, target_column=target)


@pytest.mark.parametrize("column", ["feature", "price"])
@pytest.mark.parametrize(
    "value",
    ["Alice", "NA", "N/A", "null", "None", "2026-01-01", "1,200"],
)
def test_nonnumeric_values_are_rejected(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    # Quoting keeps values such as "1,200" in one syntactically valid field.
    feature = f'"{value}"' if column == "feature" else "1"
    target = f'"{value}"' if column == "price" else "100"
    source = _write_csv(
        tmp_path,
        f"feature,price\n{feature},{target}\n2,200\n3,300\n",
    )

    with pytest.raises(
        RegressionDataError,
        match="nonnumeric or unrepresentable",
    ) as error:
        load_regression_csv(source, target_column="price")

    assert repr(column) in str(error.value)
    assert "data record 1" in str(error.value)


@pytest.mark.parametrize("column", ["feature", "price"])
@pytest.mark.parametrize(
    "value",
    ["NaN", "nan", "inf", "+inf", "-Infinity", "1e309", "-1e309"],
)
def test_nonfinite_values_are_rejected(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    feature = value if column == "feature" else "1"
    target = value if column == "price" else "100"
    source = _write_csv(
        tmp_path,
        f"feature,price\n{feature},{target}\n2,200\n3,300\n",
    )

    with pytest.raises(
        RegressionDataError,
        match="non-finite numeric value",
    ) as error:
        load_regression_csv(source, target_column="price")

    assert repr(column) in str(error.value)
    assert "data record 1" in str(error.value)


def test_constant_target_is_rejected_after_numeric_conversion(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,100.0\n3,1e2\n",
    )

    with pytest.raises(RegressionDataError, match="is constant"):
        load_regression_csv(source, target_column="price")


def test_entirely_missing_feature_is_rejected(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "size,empty_feature,price\n"
        "1000,,100\n"
        "1500, ,200\n"
        "2000,\t,300\n",
    )

    with pytest.raises(
        RegressionDataError,
        match="no observed numeric values",
    ) as error:
        load_regression_csv(source, target_column="price")

    assert "'empty_feature'" in str(error.value)


def test_multiple_entirely_missing_features_are_identified(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "empty_a,observed,empty_b,price\n"
        ",1,,100\n"
        ",2,,200\n",
    )

    with pytest.raises(RegressionDataError) as error:
        load_regression_csv(source, target_column="price")

    message = str(error.value)
    assert "'empty_a'" in message
    assert "'empty_b'" in message
    assert "'observed'" not in message


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        ("", "header is required"),
        ("\n\r\n", "header is required"),
        ("feature,feature,price\n1,2,100\n", "must be unique"),
        ("feature,,price\n1,2,100\n", "must not be empty"),
        ("feature,   ,price\n1,2,100\n", "must not be empty"),
        ("feature,price\n1\n", "expected 2"),
        ("feature,price\n1,100,extra\n", "expected 2"),
        ('feature,price\n"unfinished,100\n', "Malformed CSV"),
        ('feature,price\n"1"oops,100\n', "Malformed CSV"),
    ],
)
def test_invalid_csv_errors_are_wrapped_with_the_original_cause(
    tmp_path: Path,
    content: str,
    detail: str,
) -> None:
    source = _write_csv(tmp_path, content)

    with pytest.raises(
        RegressionDataError,
        match="Cannot load regression dataset",
    ) as error:
        load_regression_csv(source, target_column="price")

    assert detail in str(error.value)
    assert isinstance(error.value.__cause__, CSVInputError)


@pytest.mark.parametrize(
    "content",
    [
        "feature,price\n",
        "feature,price\n1,100\n",
    ],
    ids=["header-only", "one-record"],
)
def test_insufficient_data_records_are_rejected(
    tmp_path: Path,
    content: str,
) -> None:
    source = _write_csv(tmp_path, content)

    with pytest.raises(RegressionDataError, match="at least two data records"):
        load_regression_csv(source, target_column="price")


def test_two_records_are_valid_at_the_loading_stage(
    tmp_path: Path,
) -> None:
    # Fold-count and training-fold-size requirements belong to the runner.
    source = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,200\n",
    )

    dataset = load_regression_csv(source, target_column="price")

    assert dataset.n_samples == 2
    assert dataset.n_features == 1


def test_target_only_dataset_is_rejected(
    tmp_path: Path,
) -> None:
    source = _write_csv(tmp_path, "price\n100\n200\n")

    with pytest.raises(
        RegressionDataError,
        match="at least one feature column",
    ):
        load_regression_csv(source, target_column="price")


def test_missing_file_raises_a_regression_data_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(RegressionDataError, match="not found"):
        load_regression_csv(
            tmp_path / "does-not-exist.csv",
            target_column="price",
        )


def test_directory_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RegressionDataError, match="directory"):
        load_regression_csv(tmp_path, target_column="price")


def test_non_utf8_input_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "invalid-encoding.csv"
    source.write_bytes(b"feature,price\n\xff,100\n2,200\n")

    with pytest.raises(RegressionDataError, match="UTF-8"):
        load_regression_csv(source, target_column="price")


@pytest.mark.parametrize("path", ["", " ", "\t"])
def test_blank_path_is_rejected(path: str) -> None:
    with pytest.raises(RegressionDataError, match="path must not be empty"):
        load_regression_csv(path, target_column="price")


def test_fingerprint_is_repeatable_for_path_and_string_inputs(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,200\n3,300\n",
    )

    first = load_regression_csv(source, target_column="price")
    second = load_regression_csv(str(source), target_column="price")

    assert first.data_fingerprint == second.data_fingerprint
    assert len(first.data_fingerprint) == 64
    assert all(
        character in "0123456789abcdef"
        for character in first.data_fingerprint
    )


def test_fingerprint_ignores_path_and_equivalent_csv_serialization(
    tmp_path: Path,
) -> None:
    plain = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,200\n3,300\n",
        filename="plain.csv",
    )
    formatted = _write_csv(
        tmp_path,
        '\r\n"feature","price"\r\n'
        '"1","100"\r\n\r\n'
        '"2","200"\r\n'
        '"3","300"\r\n',
        filename="quoted-with-bom.csv",
        encoding="utf-8-sig",
    )

    assert plain.read_bytes() != formatted.read_bytes()

    first = load_regression_csv(plain, target_column="price")
    second = load_regression_csv(formatted, target_column="price")

    assert first.source_path != second.source_path
    assert first.data_fingerprint == second.data_fingerprint
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)


@pytest.mark.parametrize(
    "changed_content",
    [
        "feature,other,price\n3,30,300\n2,20,200\n1,10,100\n",
        "other,feature,price\n10,1,100\n20,2,200\n30,3,300\n",
        "feature,other,price\n9,10,100\n2,20,200\n3,30,300\n",
        "feature,other,price\n1,10,101\n2,20,200\n3,30,300\n",
        "feature,other,price\n1.0,10,100\n2,20,200\n3,30,300\n",
        "renamed,other,price\n1,10,100\n2,20,200\n3,30,300\n",
    ],
    ids=[
        "row-order",
        "column-order",
        "feature-value",
        "target-value",
        "numeric-spelling",
        "header-name",
    ],
)
def test_fingerprint_changes_when_parsed_content_changes(
    tmp_path: Path,
    changed_content: str,
) -> None:
    original = _write_csv(
        tmp_path,
        "feature,other,price\n1,10,100\n2,20,200\n3,30,300\n",
        filename="original.csv",
    )
    changed = _write_csv(
        tmp_path,
        changed_content,
        filename="changed.csv",
    )

    first = load_regression_csv(original, target_column="price")
    second = load_regression_csv(changed, target_column="price")

    assert first.data_fingerprint != second.data_fingerprint


def test_fingerprint_includes_target_selection(tmp_path: Path) -> None:
    source = _write_csv(
        tmp_path,
        "feature,other,price\n1,10,100\n2,20,200\n3,30,300\n",
    )

    price_target = load_regression_csv(source, target_column="price")
    other_target = load_regression_csv(source, target_column="other")

    assert price_target.data_fingerprint != other_target.data_fingerprint
    assert price_target.feature_names == ("feature", "other")
    assert other_target.feature_names == ("feature", "price")


def test_feature_and_target_arrays_are_read_only(tmp_path: Path) -> None:
    source = _write_csv(
        tmp_path,
        "feature,price\n1,100\n2,200\n3,300\n",
    )
    dataset = load_regression_csv(source, target_column="price")

    assert not dataset.X.flags.writeable
    assert not dataset.y.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        dataset.X[0, 0] = 999.0

    with pytest.raises(ValueError, match="read-only"):
        dataset.y[0] = 999.0

    np.testing.assert_array_equal(dataset.X[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dataset.y, [100.0, 200.0, 300.0])


def test_successful_loading_preserves_original_file_bytes(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path,
        '\r\n"feature","price"\r\n'
        '" 1 ","100"\r\n'
        ',"200"\r\n'
        '"3","300"\r\n',
        encoding="utf-8-sig",
    )
    before = source.read_bytes()

    load_regression_csv(source, target_column="price")

    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        "feature,price\ninvalid,100\n2,200\n",
        "feature,price\n1,\n2,200\n",
        "feature,price\n1,100,extra\n2,200\n",
    ],
    ids=["nonnumeric-feature", "missing-target", "malformed-record"],
)
def test_failed_loading_preserves_original_file_bytes(
    tmp_path: Path,
    content: str,
) -> None:
    source = _write_csv(tmp_path, content)
    before = source.read_bytes()

    with pytest.raises(RegressionDataError):
        load_regression_csv(source, target_column="price")

    assert source.read_bytes() == before