"""End-to-end tests of installed package behavior."""

import json
import subprocess
import sys
from pathlib import Path

from dataguard import scan_csv

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_scan_is_repeatable_and_does_not_modify_file(tmp_path: Path) -> None:
    source = tmp_path / "dataset with spaces.csv"
    original = (FIXTURES / "normal.csv").read_bytes()
    source.write_bytes(original)
    report = scan_csv(source)
    assert report == scan_csv(str(source))
    assert report.to_dict() == {
        "row_count": 4,
        "column_count": 3,
        "missing_values_per_column": {"name": 1, "age": 1, "city": 1},
        "total_missing_values": 3,
        "duplicate_count": 1,
    }
    assert source.read_bytes() == original


def test_header_only_report() -> None:
    assert scan_csv(FIXTURES / "header_only.csv").to_dict() == {
        "row_count": 0,
        "column_count": 2,
        "missing_values_per_column": {"name": 0, "age": 0},
        "total_missing_values": 0,
        "duplicate_count": 0,
    }


def test_dictionary_export_is_detached() -> None:
    report = scan_csv(FIXTURES / "normal.csv")
    exported = report.to_dict()
    counts = exported["missing_values_per_column"]
    assert isinstance(counts, dict)
    counts["name"] = 999
    assert report.missing_values_per_column["name"] == 1


def test_module_entry_point_outside_project(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dataguard", "scan", str(FIXTURES / "normal.csv"), "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["duplicate_count"] == 1
