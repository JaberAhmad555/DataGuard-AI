"""Unit tests for reusable quality checks."""

import pytest

from dataguard.checks import count_duplicate_rows, count_missing_values


def test_missing_values_preserve_input_and_column_order() -> None:
    rows = [["", " "], ["NA", "\t"], ["null", "NaN"], ["0", "value"]]
    original = [row.copy() for row in rows]
    result = count_missing_values(["b", "a"], rows)
    assert result == {"b": 1, "a": 2}
    assert list(result) == ["b", "a"]
    assert rows == original


def test_missing_values_with_no_rows() -> None:
    assert count_missing_values(["name", "age"], []) == {"name": 0, "age": 0}


def test_missing_values_accept_a_generator() -> None:
    rows = (row for row in [("",), ("Ada",)])
    assert count_missing_values(["name"], rows) == {"name": 1}


@pytest.mark.parametrize("row", [["Ada"], ["Ada", "36", "extra"]])
def test_missing_values_reject_wrong_row_width(row: list[str]) -> None:
    with pytest.raises(ValueError, match="same number"):
        count_missing_values(["name", "age"], [row])


def test_missing_values_reject_duplicate_columns() -> None:
    with pytest.raises(ValueError, match="unique"):
        count_missing_values(["name", "name"], [])


def test_duplicate_count_excludes_first_occurrence() -> None:
    rows = [["Ada", "36"], ["Ben", "30"], ["Ada", "36"], ["Ada", "36"]]
    original = [row.copy() for row in rows]
    assert count_duplicate_rows(rows) == 2
    assert rows == original


def test_duplicate_check_preserves_case_whitespace_and_numeric_spelling() -> None:
    rows = [("Ada", "1"), ("ada", "1"), ("Ada ", "1"), ("Ada", "01")]
    assert count_duplicate_rows(rows) == 0


def test_duplicate_check_accepts_generator() -> None:
    assert count_duplicate_rows(row for row in [("",), ("",), (" ",)]) == 1


def test_duplicate_count_with_no_rows() -> None:
    assert count_duplicate_rows([]) == 0
