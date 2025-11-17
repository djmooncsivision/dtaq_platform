"""
Core PDF table extraction logic.
"""

from __future__ import annotations

import logging
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

from .config import PDFExtractionConfig
from .models import ExtractedTable
from .table_utils import (
    clean_cell,
    contains_keyword,
    count_numeric_cells,
    is_empty_row,
    merge_rows,
    normalize_row_length,
    parse_ocr_text_block,
)

try:  # pragma: no cover - import check only
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover - handled dynamically at runtime
    pdfplumber = None


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency (pdfplumber or OCR libs) is absent."""


class PDFTableExtractor:
    """
    Extract exam-result tables from PDF files and emit normalized CSV-friendly data.
    """

    def __init__(
        self,
        config: PDFExtractionConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or PDFExtractionConfig()
        self.logger = logger or logging.getLogger(__name__)

    def extract(self, pdf_path: Path | str) -> List[ExtractedTable]:
        """
        Extract tables from the provided PDF path.
        """

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        tables = self._extract_with_pdfplumber(path)
        if tables:
            return tables

        if self.config.ocr.enabled:
            self.logger.info(
                "No vector tables detected in %s; falling back to OCR mode", path.name
            )
            return self._extract_with_ocr(path)

        return []

    # ------------------------------------------------------------------ #
    # pdfplumber-powered extraction
    # ------------------------------------------------------------------ #
    def _ensure_pdfplumber(self) -> None:
        if pdfplumber is None:
            raise MissingDependencyError(
                "pdfplumber is required for vector PDF extraction. "
                "Install it via `pip install pdfplumber`."
            )

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[ExtractedTable]:
        self._ensure_pdfplumber()

        extracted: List[ExtractedTable] = []
        rotations = self._normalized_rotations()
        pdf_handles, temp_streams = self._open_rotation_pdfs(pdf_path, rotations)
        try:
            page_count = len(pdf_handles[rotations[0]].pages)
            for page_idx in range(1, page_count + 1):
                table_counter = 0
                seen_signatures: Set[
                    Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]
                ] = set()

                for rotation in rotations:
                    page = pdf_handles[rotation].pages[page_idx - 1]
                    tables = self._extract_tables_from_page(
                        page,
                        pdf_path,
                        page_idx,
                        rotation,
                        seen_signatures,
                    )
                    for table in tables:
                        table_counter += 1
                        table.table_index = table_counter
                        extracted.append(table)
        finally:
            for pdf in pdf_handles.values():
                pdf.close()
            for stream in temp_streams:
                stream.close()

        return extracted

    def _normalized_rotations(self) -> Tuple[int, ...]:
        rotations = tuple(dict.fromkeys(angle % 360 for angle in self.config.rotations))
        if not rotations:
            return (0,)
        if 0 not in rotations:
            rotations = (0,) + rotations
        return rotations

    def _open_rotation_pdfs(
        self, pdf_path: Path, rotations: Sequence[int]
    ) -> Tuple[Dict[int, "pdfplumber.pdf.PDF"], List[io.BytesIO]]:
        pdf_handles: Dict[int, "pdfplumber.pdf.PDF"] = {}
        temp_streams: List[io.BytesIO] = []

        pdf_handles[0] = pdfplumber.open(str(pdf_path))
        for rotation in rotations:
            if rotation == 0:
                continue
            buffer = self._create_rotated_pdf_stream(pdf_path, rotation)
            temp_streams.append(buffer)
            pdf_handles[rotation] = pdfplumber.open(buffer)

        return pdf_handles, temp_streams

    def _create_rotated_pdf_stream(self, pdf_path: Path, rotation: int) -> io.BytesIO:
        try:  # pragma: no cover - optional dependency
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError(
                "Scanning additional rotations requires PyPDF2. "
                "Install it via `pip install PyPDF2` or set `rotations=(0,)`."
            ) from exc

        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        normalized_rotation = rotation % 360
        for page in reader.pages:
            rotated_page = page
            if normalized_rotation:
                rotate_fn = getattr(page, "rotate", None)
                if callable(rotate_fn):
                    rotated_page = rotate_fn(normalized_rotation)
                else:
                    rotate_clockwise = getattr(page, "rotate_clockwise", None)
                    if not callable(rotate_clockwise):
                        raise MissingDependencyError(
                            "Installed PyPDF2 version does not expose rotation helpers."
                        )
                    rotated_page = rotate_clockwise(normalized_rotation)
            writer.add_page(rotated_page)

        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)
        return buffer

    def _extract_tables_from_page(
        self,
        page: "pdfplumber.page.Page",
        pdf_path: Path,
        page_idx: int,
        rotation: int,
        seen_signatures: Set[Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]],
    ) -> List[ExtractedTable]:
        results: List[ExtractedTable] = []

        if not self._page_contains_target_keywords(page):
            return results

        tables = page.extract_tables(table_settings=dict(self.config.table_settings))
        if not tables:
            return results

        for raw_table in tables:
            normalized = self._normalize_table(raw_table)
            if not normalized:
                continue
            header, rows = normalized
            if not self._is_exam_table(header, rows):
                continue

            signature = self._table_signature(header, rows)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            confidence = self._score_table(header, rows)
            results.append(
                ExtractedTable(
                    pdf_path=pdf_path,
                    page_number=page_idx,
                    table_index=-1,  # overwritten by caller
                    header=header,
                    rows=rows,
                    confidence=confidence,
                    source="pdf",
                    context={
                        "page_text_preview": self._page_preview(page),
                        "columns": len(header),
                        "rows": len(rows),
                        "rotation": rotation % 360,
                    },
                )
            )

        return results

    def _page_contains_target_keywords(self, page: "pdfplumber.page.Page") -> bool:
        text = page.extract_text() or ""
        lowered = text.lower()
        hits = sum(1 for kw in self.config.keywords if kw.lower() in lowered)
        return hits >= self.config.page_keyword_threshold or not text

    def _normalize_table(
        self, table: Sequence[Sequence[object] | None]
    ) -> Optional[Tuple[List[str], List[List[str]]]]:
        clean_rows: List[List[str]] = []
        max_cols = 0

        for raw_row in table:
            if not raw_row:
                continue
            cleaned = [clean_cell(cell) for cell in raw_row]
            if is_empty_row(cleaned):
                continue
            max_cols = max(max_cols, len(cleaned))
            clean_rows.append(cleaned)

        if (
            max_cols < self.config.min_columns
            or max_cols > self.config.max_columns
            or len(clean_rows) < self.config.min_rows
        ):
            return None

        normalized_rows = [normalize_row_length(row, max_cols) for row in clean_rows]
        header, data_rows = self._split_header_and_body(normalized_rows)

        if not header or not data_rows:
            return None

        return header, data_rows

    def _split_header_and_body(
        self, rows: Sequence[Sequence[str]]
    ) -> Tuple[List[str], List[List[str]]]:
        header = list(rows[0])
        data_start = 1

        header_text = " ".join(header)
        if len(rows) > 1 and not contains_keyword(header_text, self.config.header_keywords):
            merged = merge_rows(header, rows[1])
            if contains_keyword(" ".join(merged), self.config.header_keywords):
                header = merged
                data_start = 2

        body = [list(row) for row in rows[data_start:]]
        return header, body

    def _is_exam_table(
        self, header: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> bool:
        header_text = " ".join(header)
        if not contains_keyword(header_text, self.config.header_keywords):
            return False

        if count_numeric_cells(rows) == 0:
            return False

        return True

    def _score_table(
        self, header: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> float:
        header_text = " ".join(header).lower()
        keyword_hits = sum(
            1 for kw in self.config.keywords if kw.lower() in header_text
        )
        numeric_cells = count_numeric_cells(rows)
        total_cells = max(1, len(rows) * len(header))
        numeric_ratio = numeric_cells / total_cells
        score = 0.4 + 0.1 * keyword_hits + 0.5 * numeric_ratio
        return min(1.0, round(score, 2))

    def _page_preview(self, page: "pdfplumber.page.Page") -> str:
        text = (page.extract_text() or "").strip()
        return text[:160]

    def _table_signature(
        self, header: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
        return tuple(header), tuple(tuple(row) for row in rows)

    # ------------------------------------------------------------------ #
    # OCR fallback (scanned PDFs)
    # ------------------------------------------------------------------ #
    def _extract_with_ocr(self, pdf_path: Path) -> List[ExtractedTable]:
        cfg = self.config.ocr
        try:  # pragma: no cover - heavy optional deps
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError(
                "OCR mode requires `pdf2image` and `pytesseract`. "
                "Install them via `pip install pdf2image pytesseract` "
                "and ensure Poppler/Tesseract binaries are available."
            ) from exc

        if cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

        images = convert_from_path(
            str(pdf_path),
            dpi=cfg.dpi,
            poppler_path=cfg.poppler_path,
        )

        if cfg.max_pages:
            images = images[: cfg.max_pages]

        extracted: List[ExtractedTable] = []
        for page_idx, image in enumerate(images, start=1):
            base_image = image
            if cfg.max_width and base_image.width > cfg.max_width:
                ratio = cfg.max_width / float(base_image.width)
                resized_height = int(base_image.height * ratio)
                base_image = base_image.resize((cfg.max_width, resized_height))

            table_counter = 0
            seen_signatures: Set[
                Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]
            ] = set()

            for rotation in self.config.rotations:
                normalized_rotation = rotation % 360
                rotated_image = (
                    base_image
                    if normalized_rotation == 0
                    else base_image.rotate(normalized_rotation, expand=True)
                )

                text = pytesseract.image_to_string(
                    rotated_image, lang=cfg.lang, config="--psm 6"
                )
                rows = parse_ocr_text_block(text)
                if not rows:
                    continue

                normalized = self._normalize_table(rows)
                if not normalized:
                    continue

                header, data_rows = normalized
                if not self._is_exam_table(header, data_rows):
                    continue

                signature = self._table_signature(header, data_rows)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                table_counter += 1

                extracted.append(
                    ExtractedTable(
                        pdf_path=pdf_path,
                        page_number=page_idx,
                        table_index=table_counter,
                        header=header,
                        rows=data_rows,
                        source="ocr",
                        confidence=0.45,
                        context={"ocr_lang": cfg.lang, "rotation": normalized_rotation},
                    )
                )

        return extracted
