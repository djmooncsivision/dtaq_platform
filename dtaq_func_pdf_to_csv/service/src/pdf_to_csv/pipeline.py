"""
High-level helpers to batch convert PDFs to CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .config import PDFExtractionConfig
from .extractor import PDFTableExtractor

LOGGER = logging.getLogger(__name__)


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def convert_pdf(
    pdf_path: Path | str,
    output_dir: Path | str,
    config: PDFExtractionConfig | None = None,
) -> List[Path]:
    """
    Convert a single PDF file to one or more CSV outputs (one per detected table).
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    _ensure_output_dir(output_dir)

    effective_config = config or PDFExtractionConfig()
    extractor = PDFTableExtractor(config=effective_config)
    tables = extractor.extract(pdf_path)
    written_files: List[Path] = []

    for table in tables:
        csv_name = (
            f"{pdf_path.stem}_p{table.page_number:02d}_"
            f"t{table.table_index:02d}_{table.source}.csv"
        )
        destination = output_dir / csv_name
        table.to_csv(destination, encoding=effective_config.output_encoding)
        written_files.append(destination)
        LOGGER.info("Wrote %s", destination)

    if not tables:
        LOGGER.warning("No exam tables detected in %s", pdf_path.name)

    return written_files


def _iter_pdf_files(
    root: Path, recursive: bool, glob_pattern: str = "*.pdf"
) -> Sequence[Path]:
    if root.is_file():
        return [root]
    if recursive:
        return sorted(root.rglob(glob_pattern))
    return sorted(root.glob(glob_pattern))


def convert_directory(
    input_path: Path | str,
    output_dir: Path | str,
    config: PDFExtractionConfig | None = None,
    recursive: bool = False,
    glob_pattern: str = "*.pdf",
) -> Dict[Path, List[Path]]:
    """
    Convert every PDF found under ``input_path`` and return a mapping of
    PDF -> list of generated CSV files.
    """

    input_path = Path(input_path)
    pdf_files = _iter_pdf_files(input_path, recursive, glob_pattern)
    results: Dict[Path, List[Path]] = {}

    for pdf_file in pdf_files:
        results[pdf_file] = convert_pdf(pdf_file, output_dir, config=config)

    return results
