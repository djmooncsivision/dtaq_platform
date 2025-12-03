"""
Configuration dataclasses that control how PDF tables are detected and exported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

DEFAULT_TABLE_KEYWORDS: tuple[str, ...] = (
    "시험결과",
    "시험 결과",
    "시험",
    "결과",
    "기준",
    "단위",
    "시험항목",
    "점검항목",
    "세부항목",
    "기준값",
    "허용범위",
    "측정값",
    "판정",
    "아날로그 신호 점검",
    "점검",
    "측정",
    "스퀴브",
    "Test Result",
    "Test Item",
    "Result",
    "Specification",
)

DEFAULT_HEADER_KEYWORDS: tuple[str, ...] = (
    "시험항목",
    "시험항 목",
    "품명",
    "항목",
    "결과",
    "판정",
    "점검항목",
    "세부항목",
    "기준값",
    "허용범위",
    "측정값",
    "Result",
    "Item",
    "Spec",
    "Criteria",
)

DEFAULT_TABLE_SETTINGS: Dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 3,
}


@dataclass(slots=True)
class OCRConfig:
    """Optional OCR fall-back configuration for scanned PDFs."""

    enabled: bool = False
    lang: str = "kor+eng"
    dpi: int = 150
    poppler_path: str | None = None
    tesseract_cmd: str | None = None
    max_pages: int | None = None
    max_width: int | None = 2200


@dataclass(slots=True)
class PDFExtractionConfig:
    """
    Tunable parameters for :class:`PDFTableExtractor`.
    """

    min_columns: int = 3
    max_columns: int = 12
    min_rows: int = 2
    page_keyword_threshold: int = 1
    keywords: Sequence[str] = field(
        default_factory=lambda: tuple(DEFAULT_TABLE_KEYWORDS)
    )
    header_keywords: Sequence[str] = field(
        default_factory=lambda: tuple(DEFAULT_HEADER_KEYWORDS)
    )
    table_settings: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_TABLE_SETTINGS)
    )
    rotations: Sequence[int] = field(default_factory=lambda: (0, 90, 180, 270))
    output_encoding: str = "utf-8-sig"
    cleaned_column_name_fallback: str = "column_{index}"
    stopwords: Sequence[str] = field(default_factory=lambda: ("요약", "비고", "총계"))
    retain_intermediate_debug_csv: bool = False
    debug_dir: Path | None = None
    ocr: OCRConfig = field(default_factory=OCRConfig)
