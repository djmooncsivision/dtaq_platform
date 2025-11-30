import pdfplumber
import sys

def analyze_pdf(pdf_path):
    import io
    with open(pdf_path, 'rb') as f:
        pdf_bytes = io.BytesIO(f.read())
    
    with pdfplumber.open(pdf_bytes) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        
    for i, page in enumerate(pdf.pages):
        print(f"\n--- Page {i+1} Text ---")
        print(page.extract_text())
        
        print(f"\n--- Page {i+1} Tables ---")
        tables = page.extract_tables()
        for j, table in enumerate(tables):
            print(f"Table {j+1}:")
            for row in table:
                print(row)
        
        print(f"\n--- Page {i+1} Images ---")
        print(f"Count: {len(page.images)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_pdf(sys.argv[1])
    else:
        print("Usage: python analyze_pdf.py <pdf_path>")
