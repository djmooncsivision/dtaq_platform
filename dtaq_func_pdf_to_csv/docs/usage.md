<!--
Usage guide for the PDF-to-CSV microservice.
-->

# PDF → CSV Conversion Guide

The `dtaq_func_pdf_to_csv` service focuses on extracting 시험 결과 (test result)
tables from PDF attachments and emitting normalized CSV files that downstream data
pipelines can ingest.

## Quick Start

```bash
cd dtaq_func_pdf_to_csv/service
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install pdfplumber pdf2image pytesseract pillow
pip install -r ../../requirements.txt  # optional shared deps
python -m pdf_to_csv.cli -i ../sample_data -o ../sample_data/converted
```

Each detected table becomes a CSV named `TalkFile_.._p01_t01_pdf.csv` (page/table
indices plus extraction source). Files are encoded as `UTF-8-SIG` so Excel opens
them cleanly.

## Dependencies

- **Vector PDF parsing**: [`pdfplumber`](https://github.com/jsvine/pdfplumber)
- **OCR fallback (optional)**: `pdf2image`, `pytesseract`, `Pillow`
  - Install the [Tesseract OCR binary](https://github.com/tesseract-ocr/tesseract)
    and ensure it is discoverable (`tesseract` on PATH or pass `--tesseract-path`).
    - Install Poppler utilities and pass `--poppler-path` on Windows.
- **Page rotation search (optional)**: [`PyPDF2`](https://pypi.org/project/PyPDF2/)
  - Required only if you keep rotation angles other than `0`. Set `--rotations 0`
    to skip this dependency.

## CLI Flags

`python -m pdf_to_csv.cli --help`

- `--input/-i`: Single PDF or a directory of PDFs.
- `--output/-o`: Destination directory for CSV files (created automatically).
- `--recursive`: Recurse into sub-directories when the input is a folder.
- `--rotations`: Comma-separated angles (defaults to `0,90,180,270`). The extractor
  retries each page at those rotations so sideways/landscape tables are detected.
- `--ocr`: Enable OCR backup mode (requires the extra deps above).
- `--tesseract-path` / `--poppler-path`: Explicit binary locations when the tools
  are not on PATH (common on Windows).

## Extraction Notes

1. The extractor focuses on tables whose headers contain Korean or English
   keywords such as `시험항목`, `결과`, `Spec`, or `Result`.
2. Only the table sections are exported; other narrative text is ignored.
3. Each page is re-processed at the rotations listed in the config (default:
   `0, 90, 180, 270`) so vertically oriented 시험 결과 표도 수집됩니다.
   - If you cannot install `PyPDF2`, run with `--rotations 0` to skip this step.
4. When no vector table is detected, enabling `--ocr` rasterizes each page and
   rebuilds tables using whitespace-separated columns (best-effort).
5. If your PDFs use bespoke layouts, adjust the heuristics through
   `PDFExtractionConfig` (e.g., column bounds, keyword lists, OCR DPI).

## Integrating as a Module

```python
from pathlib import Path
from pdf_to_csv import PDFTableExtractor, PDFExtractionConfig

config = PDFExtractionConfig()
extractor = PDFTableExtractor(config)
tables = extractor.extract(Path("sample_data/TalkFile.pdf"))

for table in tables:
    table.to_csv(Path("out") / "table.csv")
```

The `.as_dict()` helper makes it easy to push data into APIs or message queues
if you prefer streaming the results downstream instead of writing CSVs locally.
