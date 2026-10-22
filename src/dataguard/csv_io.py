"""Read and validate CSV input without changing its contents."""

import csv
from pathlib import Path


class CSVInputError(ValueError):
    """Raised when a CSV cannot be read or has an invalid table structure."""


def read_csv(path: str | Path) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Read a UTF-8, comma-separated CSV and return its header and data rows.

    A UTF-8 BOM is accepted. Empty physical lines are skipped. Headers must
    be nonblank and unique, and every data row must match the header width.
    Field strings are preserved, including their whitespace. A header-only
    file is valid; a file without a header raises CSVInputError.
    """
    source = Path(path)
    try:
        if source.is_dir():
            raise CSVInputError(f"Expected a CSV file, but received a directory: {source}")

        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next((row for row in reader if row), None)
            if header is None:
                raise CSVInputError("CSV is empty or contains only blank lines; a header is required.")
            if any(not name.strip() for name in header):
                raise CSVInputError("CSV column names must not be empty or whitespace-only.")
            if len(set(header)) != len(header):
                raise CSVInputError("CSV column names must be unique.")

            rows: list[tuple[str, ...]] = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise CSVInputError(
                        f"CSV record ending at line {reader.line_num} has {len(row)} fields; "
                        f"expected {len(header)}."
                    )
                rows.append(tuple(row))
            return tuple(header), rows
    except FileNotFoundError as exc:
        raise CSVInputError(f"CSV file not found: {source}") from exc
    except PermissionError as exc:
        raise CSVInputError(f"Permission denied when reading CSV: {source}") from exc
    except UnicodeDecodeError as exc:
        raise CSVInputError(f"CSV must use UTF-8 encoding: {source}") from exc
    except csv.Error as exc:
        raise CSVInputError(
            f"Malformed CSV near line {reader.line_num}: {exc}"
        ) from exc
    except OSError as exc:
        raise CSVInputError(f"Could not read CSV {source}: {exc.strerror}") from exc
