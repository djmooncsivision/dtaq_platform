import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "vgui3.0"))
from modules.pdf_converter import PDFConverter

def debug_run():
    # Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "pdf2csv", "test_sample", "PGM_ELEC_test_5-min (1).pdf")
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found at {pdf_path}")
        return

    # Mock UploadedFile
    class MockFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self.path = path
        def getbuffer(self):
            with open(self.path, "rb") as f:
                return f.read()

    uploaded_file = MockFile(pdf_path)
    
    # Init Converter
    # Point to reference limit file if needed, or None
    limits_path = os.path.join(base_dir, "ref_data", "upper_lower_limit.csv")
    converter = PDFConverter(limits_path)
    
    # Run Extract Data Step-by-Step
    print("--- Parsing PDF ---")
    temp_path = "temp_debug.pdf"
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    ocr_results = converter.ocr_parser.parse_pdf(temp_path)
    
    print("\n--- Extracting Data ---")
    extracted_data, sn, date = converter.extractor.extract_data(ocr_results)
    
    print(f"SN: {sn}, Date: {date}")
    
    print("\n--- Extracted Entries with Mapping ---")
    
    # We need to access the logic inside convert() effectively, but validly we can just instantiate PDFConverter and run a customized flow or copy the logic.
    # Actually, let's just use the converter.convert() on the mock file and inspect the raw_df if possible, 
    # OR replicate the loop here.
    # Replicating the loop from convert() for debug visibility:
    
    extracted_data, extracted_sn, extracted_date = converter.extractor.extract_data(ocr_results)
    print(f"SN: {extracted_sn}, Date: {extracted_date}")
    
    mapping = converter.item_map
    name_counts = {}
    
    print("\n[Step 1: Initial Mapping]")
    for i, entry in enumerate(extracted_data):
        name = entry['Item_Name'].replace(" ", "").upper().replace("·", "").replace(".", "")
        candidate_ids = mapping.get(name)
        if not candidate_ids:
             for k, v in mapping.items():
                if k in name or name in k: candidate_ids = v; break
        
        item_id = None
        if candidate_ids:
            if len(candidate_ids) == 1:
                item_id = candidate_ids[0]
            else:
                count_key = name
                idx = name_counts.get(count_key, 0)
                if idx < len(candidate_ids):
                    item_id = candidate_ids[idx]
                    name_counts[count_key] = idx + 1
                else:
                    item_id = candidate_ids[-1]
        
        entry['_mapped_id'] = int(item_id) if item_id else None
        print(f"Row {i}: Name='{entry['Item_Name']}' (Clean='{name}') -> ID: {entry['_mapped_id']} | Val: {entry['Measured_Value']}")

    print("\n[Step 2: Sequential Gap Filling]")
    last_id = 0
    for i, entry in enumerate(extracted_data):
        current_id = entry.get('_mapped_id')
        if current_id:
            last_id = current_id
        else:
            has_val_or_res = entry.get('Result') or entry.get('Measured_Value')
            if last_id > 0 and last_id < 27 and has_val_or_res:
                 # Check Filter
                 if entry.get('Result'):
                    guessed_id = last_id + 1
                    print(f"Row {i}: UNMAPPED. Guessing ID {guessed_id} (Prev {last_id})")
                    current_id = guessed_id
                    entry['_mapped_id'] = current_id
                    last_id = current_id
            else:
                 print(f"Row {i}: UNMAPPED. SKIPPED (Prev {last_id})")


    # clean
    if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    debug_run()
