from pypdf import PdfReader
import sys

def analyze_pypdf(pdf_path):
    reader = PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    for i, page in enumerate(reader.pages):
        print(f"\n--- Page {i+1} Text ---")
        print(page.extract_text())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_pypdf(sys.argv[1])
    else:
        print("Usage: python analyze_pypdf.py <pdf_path>")
