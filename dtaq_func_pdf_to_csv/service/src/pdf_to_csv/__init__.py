"""
High-level helpers for converting structured exam-result tables inside PDFs to CSV.

The module centers around :class:`PDFTableExtractor`, which uses ``pdfplumber`` for
vector PDFs and can optionally fall back to a lightweight OCR pipeline for scanned
documents. Convenience functions are re-exported here so downstream callers can
either integrate with the extractor directly or run the batch-oriented helpers.
"""

from .config import OCRConfig, PDFExtractionConfig
from .pipeline import convert_directory, convert_pdf
from .extractor import PDFTableExtractor

__all__ = [
    "OCRConfig",
    "PDFExtractionConfig",
    "PDFTableExtractor",
    "convert_directory",
    "convert_pdf",
]

