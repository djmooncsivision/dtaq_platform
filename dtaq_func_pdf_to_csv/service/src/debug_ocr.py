# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from pdf_to_csv.table_utils import parse_ocr_text_block
from pdf_to_csv.extractor import PDFTableExtractor
from pdf_to_csv.config import PDFExtractionConfig

def debug_ocr(pdf_path):
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"Debugging {pdf_path}")
    images = convert_from_path(str(pdf_path), dpi=150)
    if not images:
        print("No images converted")
        return

    config = PDFExtractionConfig()
    extractor = PDFTableExtractor(config=config)

    for i, image in enumerate(images):
        print(f"--- Page {i+1} ---")
        # text = pytesseract.image_to_string(image, lang="kor+eng", config="--psm 6")
        text = extractor._run_tesseract(image, lang="kor+eng", config="--psm 6")
        print("Text repr:", repr(text))
        
        if "점검항목" in text:
            print("FOUND KEYWORD: 점검항목")
        else:
            print("Keyword '점검항목' NOT found")

        rows = parse_ocr_text_block(text)
        normalized = extractor._normalize_table(rows)
        
        if normalized:
            header, data_rows = normalized
            print("Normalized Header:", header)
            print("Is Exam Table:", extractor._is_exam_table(header, data_rows))
        else:
            print("Normalization failed")

if __name__ == "__main__":
    pdf_path = r"c:\Users\msi\Desktop\workspace\055_기품원_consulting\dtaq_platform\dtaq_func_pdf_to_csv\sample_data\신궁_test_1-min.pdf"
    debug_ocr(pdf_path)
