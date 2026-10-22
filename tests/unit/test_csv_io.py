"""CSV boundary tests, including malformed and unreadable inputs."""

from pathlib import Path

import pytest

from dataguard.csv_io import CSVInputError, read_csv

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_read_normal_fixture() -> None:
    columns, rows = read_csv(FIXTURES / "normal.csv")
    assert columns == ("name", "age", "city")
    assert len(rows) == 4
    assert rows[-1] == ("", "29", "   ")


def test_header_only_is_valid() -> None:
    assert read_csv(FIXTURES / "header_only.csv") == (("name", "age"), [])


def test_empty_fixture_is_rejected() -> None:
    with pytest.raises(CSVInputError, match="header is required"):
        read_csv(FIXTURES / "empty.csv")


def test_malformed_fixture_is_rejected() -> None:
    with pytest.raises(CSVInputError, match="line 2 has 1 fields; expected 2"):
        read_csv(FIXTURES / "malformed.csv")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("\n\r\n", "header is required"),
        ("name,name\nAda,Ben\n", "unique"),
        ("name,\nAda,36\n", "column names must not be empty"),
        ("name,   \nAda,36\n", "column names must not be empty"),
        ("name,age\nAda,36,extra\n", "expected 2"),
        ('name,age\n"Ada,36\n', "Malformed CSV"),
        ('name,age\n"Ada"oops,36\n', "Malformed CSV"),
    ],
)
def test_invalid_csv(tmp_path: Path, content: str, message: str) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text(content, encoding="utf-8", newline="")
    with pytest.raises(CSVInputError, match=message):
        read_csv(source)


def test_utf8_bom_quoting_and_blank_lines(tmp_path: Path) -> None:
    source = tmp_path / "quoted.csv"
    source.write_text(
        '\nname,note\r\n"Zoë","comma, and ""quote"""\r\n\r\n"Ada","two\nlines"\r\n',
        encoding="utf-8-sig",
        newline="",
    )
    assert read_csv(source) == (
        ("name", "note"),
        [("Zoë", 'comma, and "quote"'), ("Ada", "two\nlines")],
    )


def test_blank_lines_are_skipped_but_empty_fields_are_records(tmp_path: Path) -> None:
    source = tmp_path / "blank.csv"
    source.write_text('name,age\n\n,\n"",""\n', encoding="utf-8")
    assert read_csv(source) == (("name", "age"), [("", ""), ("", "")])


def test_invalid_encoding(tmp_path: Path) -> None:
    source = tmp_path / "encoding.csv"
    source.write_bytes(b"name\n\xff\n")
    with pytest.raises(CSVInputError, match="UTF-8"):
        read_csv(source)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CSVInputError, match="not found"):
        read_csv(tmp_path / "absent.csv")


def test_directory_input(tmp_path: Path) -> None:
    with pytest.raises(CSVInputError, match="directory"):
        read_csv(tmp_path)


@pytest.mark.parametrize(
    ("error", "message"),
    [(PermissionError("denied"), "Permission denied"), (OSError(5, "read failure"), "Could not read")],
)
def test_filesystem_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError, message: str
) -> None:
    def fail_open(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(CSVInputError, match=message):
        read_csv(tmp_path / "unreadable.csv")
