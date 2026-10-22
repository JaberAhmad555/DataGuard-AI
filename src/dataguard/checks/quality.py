"""Deterministic missing-value and duplicate-row checks."""

from collections.abc import Iterable, Sequence


def count_missing_values(
    columns: Sequence[str], rows: Iterable[Sequence[str]]
) -> dict[str, int]:
    """Count empty or whitespace-only fields per column, in column order.

    Names must be unique and each row must match the number of columns;
    otherwise ValueError is raised. Literal NA, null, and NaN are ordinary
    strings. Values are inspected without modifying the supplied rows.
    """
    if len(set(columns)) != len(columns):
        raise ValueError("Column names must be unique.")
    counts = dict.fromkeys(columns, 0)
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("Every row must have the same number of fields as columns.")
        for column, value in zip(columns, row, strict=True):
            if not value.strip():
                counts[column] += 1
    return counts


def count_duplicate_rows(rows: Iterable[Sequence[str]]) -> int:
    """Count repeated rows after their first occurrence using exact strings.

    For three identical rows, return two. Whitespace, case, and numeric
    spelling remain significant; no normalization or type conversion occurs.
    """
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = tuple(row)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates
