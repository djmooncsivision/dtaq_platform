import sys
import os
import pandas as pd
import numpy as np

# Add pdf2csv to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pdf2csv')))

from ocr_parser import OCRParser
from data_extractor import DataExtractor

class PDFProcessor:
    def __init__(self, limits_path=None):
        self.ocr_parser = OCRParser(gpu=False) # GPU can be toggled
        self.extractor = DataExtractor()
        self.item_map = self._build_item_mapping(limits_path)
        
    def _build_item_mapping(self, limits_path):
        """Builds a mapping from Korean Item Name to Item ID (1~27) from limits CSV."""
        mapping = {}
        if not limits_path or not os.path.exists(limits_path):
            # Fallback / Hardcoded common names if file missing
            return mapping

        try:
            df = pd.read_csv(limits_path, header=None, encoding='utf-8')
            # Columns 5 to 31 correspond to Item 1 to 27 (0-indexed)
            # Rows 1 to 4 (1-indexed in file, so indices 1-4 in 0-indexed DF) contain names
            
            for col_idx in range(5, 32):
                item_id = str(col_idx - 4) # 5 -> 1, 31 -> 27
                
                # Get all unique strings in this column for rows 1-4 (Header rows)
                names = df.iloc[1:5, col_idx].dropna().unique()
                
                for name in names:
                    clean_name = str(name).strip().replace(" ", "")
                    if clean_name:
                        mapping[clean_name] = item_id
                        
            # Add manual manual mapping for known discrepancies from data_extractor
            manual_map = {
                "자전안정화주파수(f0)": "13", # Inspecting limit csv col 17 (13+4) -> 자전안정화주파수
                "보조채널조종(ACPA1)": "15", # ACPA1
                "보조채널조종(ACPA2)": "16",
                "보조채널조종(ACPA3)": "17",
                "시선각(SLAY)": "27", # Last item? Need to verify
                "5'측정(ACPA)": "15", # Guessing
            }
            mapping.update(manual_map)
            
        except Exception as e:
            print(f"Error building mapping: {e}")
            
        return mapping

    def process_pdfs(self, uploaded_files):
        """
        Process list of uploaded PDF files.
        Returns:
            df (pd.DataFrame): Transformed Wide format data
            raw_df (pd.DataFrame): Raw Long format data
        """
        all_rows = []
        all_raw_data = [] # To store raw extractions
        
        for uploaded_file in uploaded_files:
            # Save temporary file for OCR (OCRParser expects path)
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                # 1. OCR & Extract
                print(f"Processing {uploaded_file.name}...")
                ocr_results = self.ocr_parser.parse_pdf(temp_path)
                extracted_data = self.extractor.extract_data(ocr_results)
                
                # 2. Transform to Single Row
                row_data = self._transform_to_row(extracted_data, uploaded_file.name)
                all_rows.append(row_data)
                
                # Collect Raw Data for debug/view
                for item in extracted_data:
                    item['Source_File'] = uploaded_file.name
                    all_raw_data.append(item)
                
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Create DataFrame
        if not all_rows:
            return pd.DataFrame(), pd.DataFrame()
            
        df = pd.DataFrame(all_rows)
        raw_df = pd.DataFrame(all_raw_data)
        
        # Sort columns to match Item 1, 2... 27
        # Ensure metadata columns come first
        meta_cols = ['일련번호', '품번', '시험일자', '운용월', '합격여부']
        item_cols = [str(i) for i in range(1, 28)]
        
        # Ensure all columns exist
        for col in meta_cols + item_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        final_cols = meta_cols + item_cols
        return df[final_cols], raw_df

    def _transform_to_row(self, extracted_data, filename):
        """
        Transforms a list of extracted records (Long format) into a single dictionary (Wide format).
        """
        row = {}
        
        # Placeholder Metadata (Try to extract from filename or content if possible in future)
        # e.g. "SG 16 D312 00.pdf" -> Serial
        row['일련번호'] = os.path.splitext(filename)[0]
        row['품번'] = '81040050' # Default/Placeholder
        row['시험일자'] = '2024.01.01' # Placeholder
        row['운용월'] = 0 # Placeholder
        row['합격여부'] = 'Unknown'

        # Map Items
        for entry in extracted_data:
            name = entry['Item_Name'].replace(" ", "")
            value = entry['Measured_Value']
            result = entry['Result']
            
            # Map Name to ID
            # 1. Try exact match
            item_id = self.item_map.get(name)
            
            # 2. Fuzzy/Partial match if needed
            if not item_id:
                for map_name, map_id in self.item_map.items():
                    if map_name in name or name in map_name:
                        item_id = map_id
                        break
            
            if item_id:
                # Clean value (handle numbers)
                try:
                    clean_val = float(str(value).replace(',', ''))
                    row[item_id] = clean_val
                except:
                    row[item_id] = value # Keep specific string if not float
            
            # Update Pass/Fail logic (Simple heuristic: if any Fail, then Fail)
            if result.upper() in ['FAIL', 'F', '불량', 'NG']:
                row['합격여부'] = '불량'
            elif row.get('합격여부') != '불량' and result.upper() in ['PASS', 'P', '정상', 'OK']:
                row['합격여부'] = '합격'
                
        return row
