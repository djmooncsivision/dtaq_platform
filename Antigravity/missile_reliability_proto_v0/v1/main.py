import sys
import os
from ocr_parser import OCRParser
from data_extractor import DataExtractor
from csv_writer import CSVWriter

def main(pdf_path, output_csv_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found - {pdf_path}")
        return

    print(f"Starting conversion for: {pdf_path}")
    
    # 1. Parse PDF with OCR
    ocr_parser = OCRParser(gpu=False) # Set gpu=True if CUDA is available
    ocr_results = ocr_parser.parse_pdf(pdf_path)
    
    # 2. Extract Data
    extractor = DataExtractor()
    extracted_data = extractor.extract_data(ocr_results)
    
    print(f"Extracted {len(extracted_data)} records.")
    
    # 3. Save to CSV
    writer = CSVWriter(output_csv_path)
    writer.save_data(extracted_data)
    
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path> [output_csv_path]")
    else:
        input_pdf = sys.argv[1]
        if len(sys.argv) > 2:
            output_csv = sys.argv[2]
        else:
            # Default output name
            base_name = os.path.splitext(input_pdf)[0]
            output_csv = f"{base_name}.csv"
            
        main(input_pdf, output_csv)
