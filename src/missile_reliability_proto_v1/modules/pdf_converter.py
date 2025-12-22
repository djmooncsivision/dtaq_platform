import sys
import os
import io
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict
import re

# Try imports with debug
pdfplumber = None
easyocr = None
import_error_msg = ""
import sys

debug_info = f"Python Executable: {sys.executable}\nPython Version: {sys.version}\nSys Path: {sys.path}\n"

try:
    import pdfplumber
except ImportError as e:
    import_error_msg += f"pdfplumber import failed: {e}\n"
except Exception as e:
    import_error_msg += f"pdfplumber error: {e}\n"

try:
    import easyocr
except ImportError as e:
    import_error_msg += f"easyocr import failed: {e}\n"
except Exception as e:
    import_error_msg += f"easyocr error: {e}\n"

class OCRParser:
    def __init__(self, languages: List[str] = ['ko', 'en'], gpu: bool = False):
        if easyocr is None:
            # Raise the captured error to show the user with ENV INFO
            raise ImportError(f"easyocr library is not installed.\n\n[Debug Info]\n{debug_info}\n\n[Import Errors]\n{import_error_msg}")
        print("Initializing EasyOCR... (This may take a moment)")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def parse_pdf(self, pdf_path: str) -> List[List[Tuple[Any, str, float]]]:
        results = []
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = io.BytesIO(f.read())

            with pdfplumber.open(pdf_bytes) as pdf:
                print(f"Processing PDF: {pdf_path} ({len(pdf.pages)} pages)")
                for i, page in enumerate(pdf.pages):
                    im = page.to_image(resolution=300).original
                    im_np = np.array(im)
                    page_result = self.reader.readtext(im_np)
                    
                    sanitized_result = []
                    for item in page_result:
                        bbox, text, prob = item
                        bbox = [[float(c) for c in p] for p in bbox]
                        sanitized_result.append((bbox, str(text), float(prob)))
                    results.append(sanitized_result)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise e
        return results

class DataExtractor:
    def __init__(self, y_tolerance: int = 80):
        self.y_tolerance = y_tolerance

    def extract_serial_number(self, ocr_results: List[List[Tuple[Any, str, float]]]) -> str:
        # Try to find "S/N" or similar pattern
        import re
        for page_items in ocr_results:
            for item in page_items:
                text = item[1]
                # Regex for patterns like "S/N : SG 11 B440 00" or similar
                match = re.search(r'(S/N|Serial|일련번호)\s*[:.]?\s*([A-Za-z0-9\s-]+)', text, re.IGNORECASE)
                if match:
                    candidate = match.group(2).strip()
                    if len(candidate) > 5: # Basic filter
                        return candidate
                
                # Direct pattern search without prefix "SG ..."
                match_sg = re.search(r'(SG\s*\d+\s*[A-Z]\d+\s*\d+)', text)
                if match_sg:
                    return match_sg.group(1).strip()
        return ""

    def extract_test_date(self, ocr_results: List[List[Tuple[Any, str, float]]]) -> str:
        # Look for date pattern 20XX.MM.DD, 20XX-MM-DD, or 20XX년 XX월 XX일
        import re
        
        # Pattern 1: Korean Format (Prioritized)
        korean_pattern = re.compile(r'(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일')
        
        # Pattern 2: Dot/Dash Format
        standard_pattern = re.compile(r'(20\d{2})[.-](\d{1,2})[.-](\d{1,2})')
        
        for page_items in ocr_results:
            for item in page_items:
                text = item[1]
                
                # Check Korean pattern first
                match_ko = korean_pattern.search(text)
                if match_ko:
                    y, m, d = match_ko.groups()
                    return f"{y}.{m.zfill(2)}.{d.zfill(2)}"
                
                # Check Standard pattern
                match_std = standard_pattern.search(text)
                if match_std:
                    y, m, d = match_std.groups()
                    return f"{y}.{m.zfill(2)}.{d.zfill(2)}"
                    
        return "2024.01.01" # Default fallback if not found

    def extract_data(self, ocr_results: List[List[Tuple[Any, str, float]]]) -> Tuple[List[Dict[str, Any]], str, str]:
        all_data = []
        serial_number = self.extract_serial_number(ocr_results) # Extract SN
        test_date = self.extract_test_date(ocr_results) # Extract Date
        
        for page_idx, page_items in enumerate(ocr_results):
            rows = self._cluster_into_rows(page_items)
            page_data = self._parse_page_rows(rows, page_idx + 1)
            all_data.extend(page_data)
        return all_data, serial_number, test_date

    def _cluster_into_rows(self, items):
        # Sort by Y-coordinate
        sorted_items = sorted(items, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
        rows = []
        if not sorted_items:
            return rows

        first_item = sorted_items[0]
        current_row = [first_item]
        current_y = (first_item[0][0][1] + first_item[0][2][1]) / 2
        
        for item in sorted_items[1:]:
            item_y = (item[0][0][1] + item[0][2][1]) / 2
            if abs(item_y - current_y) <= self.y_tolerance:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x[0][0][0])
                rows.append(current_row)
                current_row = [item]
                current_y = item_y
                
        if current_row:
            current_row.sort(key=lambda x: x[0][0][0])
            rows.append(current_row)
        return rows

    def _correct_item_name(self, name: str, page_num: int) -> str:
        # if page_num != 1: return name
        name_clean = name.replace(" ", "").replace("(", "").replace(")", "")
        
        # Aggressive correction for common OCR errors
        if "스귀스" in name_clean or "스키브" in name_clean:
            name_clean = name_clean.replace("스귀스", "스퀴브").replace("스키브", "스퀴브")
        if "저향" in name_clean:
            name_clean = name_clean.replace("저향", "저항")
            
        corrections = {
            "자전안정화": "자전 안정화 주파수(f0)",
            "보조채널조증ACPA7": "보조채널조종(ACPA1)", "보조채널조A0PA2": "보조채널조종(ACPA2)",
            "보조채널조증AOPA": "보조채널조종(ACPA2)", "보조채널조증A0PA3": "보조채널조종(ACPA3)",
            "보조채닐조증": "보조채널조종(ACPA3)", "보조채널조증": "보조채널조종(ACPA1)",
            "시선PSAY": "시선각(SLAY)", "5혹정스PA": "5' 측정(ACPA)",
            "보조지널획의스PA": "보조채널 확인(ACPA)", "보조지널": "보조채널 확인(ACPA)",
            # "스퀴브저항": "열전지스퀴브저항", # Too broad, causes conflict with 사출모터
            "열전지스퀴브": "열전지스퀴브저항"
        }
        
        # Specific fix for Squib ambiguity
        if "스퀴브저항" in name_clean and "열전지" not in name_clean and "사출" not in name_clean:
             # If just "Squib Res", assume Thermal Battery if undefined? Or leave it?
             # Based on data, Item 1 is Thermal, Item 2 is Injection.
             pass 

        for key, val in corrections.items():
            if key in name_clean: return val
        return name_clean 

    def _clean_number_string(self, value: str) -> str:
        if not value: return value
        cleaned = value.replace(" ", "").replace(",.", ".").replace(".,", ".")
        import re
        if re.search(r',\d{2}$', cleaned):
            cleaned = cleaned[::-1].replace(',', '.', 1)[::-1]
        return cleaned

    def _parse_page_rows(self, rows, page_num):
        extracted_rows = []
        header_keywords = ["점검", "항목", "기준", "측정", "판정", "Item", "Value", "Result"]
        result_keywords = ["정상", "불량", "Pass", "Fail", "OK", "NG", "F", "P"]
        
        measured_val_xs = [] # Track X coordinates of valid measures
        
        for row in rows:
            text_row = [item[1] for item in row]
            row_text_joined = " ".join(text_row)
            
            # Skip Header / Noise
            if sum(1 for k in header_keywords if k in row_text_joined) >= 2 or "0x" in row_text_joined:
                continue
            
            # Identify Result
            result_val, result_idx = "", -1
            for i in range(len(text_row) - 1, -1, -1):
                item = text_row[i]
                found = False
                for rk in result_keywords:
                    if (rk in ["F", "P"] and item == rk) or (rk not in ["F", "P"] and rk in item):
                        found = True; break
                if found:
                    result_val, result_idx = item, i
                    break
            
            entry = {
                "Page": page_num, "Item_Name": "", "Measured_Value": "", "Result": "", 
                "_row_items": row # Store for Pass 2
            }

            if result_idx != -1:
                entry["Result"] = result_val
                
                # Extract Item Name
                item_name = ""
                start_idx = 0
                if result_idx > 0:
                    first_item = text_row[0]
                    is_number = False
                    try:
                        float(first_item.replace(',', ''))
                        is_number = True
                    except: pass
                    
                    if is_number:
                        item_name = ""; start_idx = 0
                    else:
                        item_name = text_row[0]; start_idx = 1
                
                entry["Item_Name"] = self._correct_item_name(item_name, page_num)
                
                # Extract Measured Value
                if result_idx > start_idx:
                    candidate = text_row[result_idx - 1]
                    clean_cand = self._clean_number_string(candidate)
                    try:
                        float(clean_cand.replace(',', ''))
                        entry["Measured_Value"] = clean_cand
                        # Track X coordinate
                        bbox = row[result_idx - 1][0]
                        center_x = (bbox[0][0] + bbox[1][0]) / 2
                        measured_val_xs.append(center_x)
                    except: pass
                
                extracted_rows.append(entry)
            else:
                # Potential wrapped name line
                has_numbers = any(any(c.isdigit() for c in t) for t in text_row)
                if not has_numbers and len(text_row) > 0:
                    name = self._correct_item_name(" ".join(text_row), page_num)
                    extracted_rows.append({
                        "Page": page_num, "Item_Name": name, "Measured_Value": "", "Result": "", "_row_items": row
                    })

        # Pass 2: Recover missing values using Column Alignment
        if measured_val_xs:
            min_x, max_x = min(measured_val_xs) - 20, max(measured_val_xs) + 20
            for entry in extracted_rows:
                if not entry["Measured_Value"] and "_row_items" in entry:
                    for item in entry["_row_items"]:
                        text = item[1]; bbox = item[0]
                        center_x = (bbox[0][0] + bbox[1][0]) / 2
                        if text == entry["Result"] or text == entry["Item_Name"]: continue
                        
                        if min_x <= center_x <= max_x:
                            clean_val = self._clean_number_string(text)
                            try:
                                float(clean_val.replace(',', ''))
                                entry["Measured_Value"] = clean_val
                                break
                            except: pass
                            
        # Cleanup
        for entry in extracted_rows:
            entry.pop("_row_items", None)
            
        return extracted_rows

class PDFConverter:
    def __init__(self, limits_path=None):
        self.ocr_parser = OCRParser(gpu=False)
        self.extractor = DataExtractor()
        self.item_map = self._build_item_mapping(limits_path)
        
    def _build_item_mapping(self, limits_path):
        # Hardcoded reference from csv_sample.csv headers
        ref_items = {
            "1": "열전지스퀴브저항", "2": "사출모터스퀴브저항",
            "3": "22P - PGND", "4": "PGND - 22P", "5": "22N - PGND", "6": "PGND - 22N", "7": "22P - 22N", "8": "22N - 22P",
            "9": "자전기동시간", "10": "자전안정화주파수",
            "11": "ASLAP", "12": "ASLAY", "13": "ALOSRP", "14": "ALOSRY", "15": "AVIFSC", "16": "AFROLL", "17": "AMRSIN",
            "18": "+5V", "19": "+15V", "20": "-15V",
            "21": "SLAP", "22": "SLAY", "23": "SLAP", "24": "SLAY", "25": "SLAP", "26": "SLAY",
            "27": "냉각유지시간"
        }
        
        mapping = {}
        for id, name in ref_items.items():
            clean_name = name.replace(" ", "").upper()
            if clean_name not in mapping:
                mapping[clean_name] = []
            mapping[clean_name].append(id)
            
        # Manual variations for fuzzy matching
        variations = {
            "자전안정화": ["10"], "자전안정화주파수(f0)": ["10"],
            "보조채널조종(ACPA1)": ["15"], "5'측정(ACPA)": ["15"], 
            "시선각(SLAY)": ["22", "24", "26"],
            "22PPGND": ["3"], "PGND22P": ["4"], "22NPGND": ["5"], "PGND22N": ["6"],
            "22P22N": ["7"], "22N22P": ["8"],
            "냉각유지": ["27"], "냉각유지시간": ["27"]
        }
        for k, v in variations.items():
            clean_k = k.replace(" ", "").upper()
            if clean_k not in mapping: mapping[clean_k] = v
            else: mapping[clean_k].extend(v) 
        
        return mapping

    def convert(self, uploaded_files) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_rows = []
        all_raw_data = []
        
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            try:
                print(f"Converting {uploaded_file.name}...")
                ocr_results = self.ocr_parser.parse_pdf(temp_path)
                extracted_data, extracted_sn, extracted_date = self.extractor.extract_data(ocr_results)
                
                final_sn = extracted_sn if extracted_sn else os.path.splitext(uploaded_file.name)[0]
                final_date = extracted_date if extracted_date else '2024.01.01'
                
                row = {
                    '일련번호': final_sn,
                    '품번': '81040050', 
                    '시험일자': final_date, 
                    '운용월': 0,
                    'Code': '',
                    'LOT 번호': '',
                    '합격여부': 'Unknown',
                    '불합격 항목': ''
                }
                for i in range(1, 28): row[str(i)] = np.nan
                
                # --- Step 1: Identify Known Items and Store in List ---
                mapped_entries = [] # List of (index, item_id, entry)
                name_counts = {} 
                failed_items_list = []
                
                for i, entry in enumerate(extracted_data):
                    entry['Source_File'] = uploaded_file.name
                    entry['_original_index'] = i
                    all_raw_data.append(entry)
                    
                    name = entry['Item_Name'].replace(" ", "").upper()
                    name = name.replace("·", "").replace(".", "")
                    
                    # Try Mapping
                    candidate_ids = self.item_map.get(name)
                    if not candidate_ids:
                        # Improved Fuzzy Search: Collect ALL candidates
                        fuzzy_candidates = []
                        for k, v in self.item_map.items():
                            if k in name or name in k: 
                                fuzzy_candidates.extend(v)
                        # Remove duplicates and sort numerically
                        if fuzzy_candidates:
                            candidate_ids = sorted(list(set(fuzzy_candidates)), key=lambda x: int(x))
                    
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
                    
                    # Store mapping result directly in entry for Step 2
                    entry['_mapped_id'] = int(item_id) if item_id else None
                
                # --- Step 2: Sequential Gap Filling ---
                # Iterate through extracted data and fill gaps for entries with Value/Result but no ID
                last_id = 0
                for entry in extracted_data:
                    current_id = entry.get('_mapped_id')
                    
                    if current_id:
                        last_id = current_id
                    else:
                        # If unmapped, but has Result/Value, try to guess
                        # Only guess if we are in the flow (last_id was set)
                        has_val_or_res = entry.get('Result') or entry.get('Measured_Value')
                        if last_id > 0 and last_id < 27 and has_val_or_res:
                            guessed_id = last_id + 1
                            # Check if guessed_id is already taken by the NEXT mapped item?
                            # For simple sequential filling, we assume strict order.
                            # But we should be careful about "garbage" rows.
                            
                            # Filter: Only fill if it looks like a measurement row (has Result)
                            if entry.get('Result'):
                                current_id = guessed_id
                                entry['_mapped_id'] = current_id
                                last_id = current_id
                                # Debug print
                                print(f"Gap Filled: ID {current_id} for entry {entry['Item_Name']}")

                    # --- Step 3: Populate Row ---
                    if current_id:
                        item_id_str = str(current_id)
                        value = entry['Measured_Value']
                        result = entry['Result']
                        
                        try: row[item_id_str] = float(str(value).replace(',', ''))
                        except: row[item_id_str] = value
                        
                        if result and result.upper() in ['FAIL', 'F', '불량', 'NG']:
                            row['합격여부'] = '불합격'
                            target_name = entry['Item_Name'] if entry['Item_Name'] else f"Item {item_id_str}"
                            failed_items_list.append(target_name)
                        elif row.get('합격여부') != '불합격' and result and result.upper() in ['PASS', 'P', '정상', 'OK']:
                            row['합격여부'] = '합격'

                row['불합격 항목'] = ", ".join(failed_items_list)
                all_rows.append(row)

            except Exception as e:
                print(f"Error converting {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
                
        if not all_rows: return pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(all_rows)
        raw_df = pd.DataFrame(all_raw_data)
        
        base_cols = ['일련번호', '품번', '시험일자']
        item_cols = [str(i) for i in range(1, 28)]
        extra_cols = ['합격여부', '불합격 항목', '운용월', 'Code', 'LOT 번호']
        all_cols = base_cols + item_cols + extra_cols
        
        for col in all_cols:
            if col not in df.columns: df[col] = np.nan
        
        return df[all_cols], raw_df
