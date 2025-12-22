import ocr_parser
import sys

def debug_page1(pdf_path):
    print(f"Debugging Page 1 of: {pdf_path}")
    parser = ocr_parser.OCRParser()
    
    # Extract only page 1
    # We can use the existing parse_pdf but it processes all pages.
    # Let's just use it and break after page 1 or inspect the result for page 1.
    
    ocr_results = parser.parse_pdf(pdf_path)
    
    if not ocr_results:
        print("No results found!")
        return

    page1_data = ocr_results[0] # Page 1 is index 0
    print(f"--- Page 1 Raw Items ({len(page1_data)} items) ---")
    
    # Sort by Y then X for readability
    sorted_items = sorted(page1_data, key=lambda x: (x[0][0][1], x[0][0][0]))
    
    for item in sorted_items:
        bbox, text, prob = item
        # Print simplified bbox (TL x, y) and text
        print(f"Y={bbox[0][1]:.1f} | X={bbox[0][0]:.1f} | Text: {text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_page1.py <pdf_path>")
        sys.exit(1)
    
    debug_page1(sys.argv[1])
