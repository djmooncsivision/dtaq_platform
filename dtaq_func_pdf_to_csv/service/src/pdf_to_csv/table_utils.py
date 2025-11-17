"""
Utility helpers for cleaning table cells and parsing OCR output.
"""

from __future__ import annotations

import re
from typing import List, Sequence

MULTISPACE_PATTERN = re.compile(r"\s{2,}")
NUMERIC_PATTERN = re.compile(r"[+-]?((\d+(\.\d*)?)|(\.\d+))(E[+-]?\d+)?", re.IGNORECASE)


def clean_cell(value: object) -> str:
    """
    Normalize cell values by replacing newlines/tabs and trimming whitespace.
    """

    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    text = text.replace("\n", " ").replace("\t", " ")
    return MULTISPACE_PATTERN.sub(" ", text).strip()


def is_empty_row(row: Sequence[str]) -> bool:
    """Return True when every cell is empty."""

    return all(cell == "" for cell in row)


def normalize_row_length(row: Sequence[str], target: int) -> List[str]:
    """Pad or trim rows to keep column counts consistent."""

    normalized = list(row[:target])
    if len(normalized) < target:
        normalized.extend([""] * (target - len(normalized)))
    return normalized


def merge_rows(upper: Sequence[str], lower: Sequence[str]) -> List[str]:
    """
    Merge two header rows, preferring non-empty cells from the lower row.
    """

    width = max(len(upper), len(lower))
    merged: List[str] = []
    for idx in range(width):
        upper_cell = upper[idx] if idx < len(upper) else ""
        lower_cell = lower[idx] if idx < len(lower) else ""
        merged_cell = lower_cell or upper_cell
        merged.append(merged_cell.strip())
    return merged


def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def count_numeric_cells(rows: Sequence[Sequence[str]]) -> int:
    """Count numeric-looking cells to infer measurement columns."""

    return sum(1 for row in rows for cell in row if NUMERIC_PATTERN.fullmatch(cell))


def parse_ocr_text_block(text: str) -> List[List[str]]:
    """
    Convert OCR text (where columns are separated by multiple spaces) to rows.
    """

    rows: List[List[str]] = []
    for raw_line in text.splitlines():
        cleaned_line = raw_line.strip()
        if not cleaned_line:
            continue
        cells = [col.strip() for col in MULTISPACE_PATTERN.split(cleaned_line) if col.strip()]
        if cells:
            rows.append(cells)
    return rows
