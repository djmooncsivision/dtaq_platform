import pdfplumber
import easyocr
import io
import sys
import numpy as np

def analyze_ocr(pdf_path):
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['ko', 'en']) # Assuming Korean and English
    
    # Open PDF from bytes to avoid file locking issues (just in case)
    with open(pdf_path, 'rb') as f:
        pdf_bytes = io.BytesIO(f.read())

    with pdfplumber.open(pdf_bytes) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        
        for i, page in enumerate(pdf.pages):
            print(f"\n--- Page {i+1} Processing ---")
            
            # Render page to image (resolution 300 DPI for better OCR)
            im = page.to_image(resolution=300).original
            
            # Convert PIL image to numpy array (EasyOCR expects this or file path)
            im_np = np.array(im)
            
            print("Running OCR...")
            result = reader.readtext(im_np)
            
            print(f"--- Page {i+1} OCR Results ---")
            for (bbox, text, prob) in result:
                print(f"{text} (Prob: {prob:.2f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_ocr(sys.argv[1])
    else:
        print("Usage: python analyze_ocr.py <pdf_path>")
