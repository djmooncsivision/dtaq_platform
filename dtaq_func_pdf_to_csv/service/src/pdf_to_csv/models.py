"""
Lightweight data models used across the PDF-to-CSV pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(slots=True)
class ExtractedTable:
    """
    Represents a single table captured from a PDF page (or OCR fallback).
    """

    pdf_path: Path
    page_number: int
    table_index: int
    header: Sequence[str]
    rows: Sequence[Sequence[str]]
    source: str = "pdf"
    confidence: float = 0.7
    context: Dict[str, Any] = field(default_factory=dict)

    def to_csv(self, output_path: Path, encoding: str = "utf-8-sig") -> None:
        """
        Persist the table to disk as UTF-8 CSV with BOM (Excel-friendly).
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding=encoding, newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(list(self.header))
            for row in self.rows:
                writer.writerow(list(row))

    def as_dict(self) -> Dict[str, Any]:
        """
        A serializable representation that callers can feed into JSON or APIs.
        """

        return {
            "pdf_path": str(self.pdf_path),
            "page_number": self.page_number,
            "table_index": self.table_index,
            "header": list(self.header),
            "rows": [list(row) for row in self.rows],
            "source": self.source,
            "confidence": self.confidence,
            "context": self.context,
        }

