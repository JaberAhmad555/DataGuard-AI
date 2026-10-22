"""CLI output and exit-status tests."""

import json
from pathlib import Path

import pytest

from dataguard.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_text_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", str(FIXTURES / "normal.csv")]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "DataGuard AI scan report\n"
        "Rows: 4\n"
        "Columns: 3\n"
        "Missing values per column:\n"
        "  'name': 1\n"
        "  'age': 1\n"
        "  'city': 1\n"
        "Total missing values: 3\n"
        "Duplicate rows (after first occurrence): 1\n"
    )


def test_json_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", str(FIXTURES / "normal.csv"), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "row_count": 4,
        "column_count": 3,
        "missing_values_per_column": {"name": 1, "age": 1, "city": 1},
        "total_missing_values": 3,
        "duplicate_count": 1,
    }


@pytest.mark.parametrize("filename", ["empty.csv", "malformed.csv", "absent.csv"])
def test_input_error_is_printed_to_stderr(
    filename: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["scan", str(FIXTURES / filename)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("dataguard: error: ")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("arguments", [[], ["scan"], ["unknown"]])
def test_invalid_arguments_exit_with_two(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "scan" in capsys.readouterr().out
