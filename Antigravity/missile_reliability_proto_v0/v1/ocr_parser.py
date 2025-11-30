import pdfplumber
import easyocr
import io
import numpy as np
from typing import List, Tuple, Any

class OCRParser:
    def __init__(self, languages: List[str] = ['ko', 'en'], gpu: bool = False):
        """
        Initialize EasyOCR reader.
        Args:
            languages: List of languages to support (default: Korean and English).
            gpu: Whether to use GPU for OCR (default: False).
        """
        print("Initializing EasyOCR... (This may take a moment)")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def parse_pdf(self, pdf_path: str) -> List[List[Tuple[Any, str, float]]]:
        """
        Parse a PDF file and extract text using OCR.
        Args:
            pdf_path: Path to the PDF file.
        Returns:
            A list of pages, where each page is a list of OCR results.
            Each result is a tuple of (bbox, text, confidence).
        """
        results = []
        
        try:
            # Open PDF from bytes to avoid file locking issues
            with open(pdf_path, 'rb') as f:
                pdf_bytes = io.BytesIO(f.read())

            with pdfplumber.open(pdf_bytes) as pdf:
                print(f"Processing PDF: {pdf_path} ({len(pdf.pages)} pages)")
                
                for i, page in enumerate(pdf.pages):
                    print(f"  - Processing page {i+1}...")
                    
                    # Render page to image (300 DPI for better accuracy)
                    im = page.to_image(resolution=300).original
                    im_np = np.array(im)
                    
                    # Run OCR
                    page_result = self.reader.readtext(im_np)
                    
                    # Sanitize result to ensure no numpy types
                    sanitized_result = []
                    for item in page_result:
                        bbox, text, prob = item
                        # bbox is list of lists, make sure elements are floats/ints
                        bbox = [[float(c) for c in p] for p in bbox]
                        sanitized_result.append((bbox, str(text), float(prob)))
                        
                    results.append(sanitized_result)
                    
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise e

        return results

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        parser = OCRParser()
        results = parser.parse_pdf(sys.argv[1])
        for i, page_data in enumerate(results):
            print(f"--- Page {i+1} ---")
            for item in page_data[:5]: # Print first 5 items
                print(item)
