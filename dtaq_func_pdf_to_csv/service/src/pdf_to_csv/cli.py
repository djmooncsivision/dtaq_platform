"""
Command-line interface for the PDF-to-CSV converter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PDFExtractionConfig
from .pipeline import convert_directory, convert_pdf

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("pdf_to_csv.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract exam-result tables from PDFs and export them as CSV files.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Single PDF file or directory containing PDFs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="converted_csv",
        help="Directory that will store the generated CSV files.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan directories for PDF files.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR fallback (requires pdf2image + pytesseract).",
    )
    parser.add_argument(
        "--rotations",
        default="0,90,180,270",
        help=(
            "Comma-separated angles (degrees) for sideways tables. "
            "Non-zero values require PyPDF2. Default: 0,90,180,270"
        ),
    )
    parser.add_argument(
        "--tesseract-path",
        help="Custom path to the tesseract executable.",
    )
    parser.add_argument(
        "--poppler-path",
        help="Path to Poppler binaries (used by pdf2image).",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Custom glob used when crawling directories (default: *.pdf).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PDFExtractionConfig()
    if args.rotations:
        config.rotations = tuple(
            int(angle.strip()) for angle in args.rotations.split(",") if angle.strip()
        ) or (0,)
    if args.ocr:
        config.ocr.enabled = True
        config.ocr.poppler_path = args.poppler_path
        config.ocr.tesseract_cmd = args.tesseract_path

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_dir():
        result = convert_directory(
            input_path,
            output_dir,
            config=config,
            recursive=args.recursive,
            glob_pattern=args.glob,
        )
        total_tables = sum(len(csv_paths) for csv_paths in result.values())
        LOGGER.info(
            "Converted %d PDFs and produced %d CSV files under %s",
            len(result),
            total_tables,
            output_dir,
        )
    else:
        generated = convert_pdf(input_path, output_dir, config=config)
        LOGGER.info(
            "Processed %s and generated %d CSV file(s) in %s",
            input_path,
            len(generated),
            output_dir,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
