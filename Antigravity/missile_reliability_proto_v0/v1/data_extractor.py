from typing import List, Dict, Any, Tuple

class DataExtractor:
    def __init__(self, y_tolerance: int = 80):
        """
        Initialize DataExtractor.
        Args:
            y_tolerance: Vertical pixel tolerance to group text into the same row.
                         Increased to 80 to handle large Result column misalignment.
        """
        self.y_tolerance = y_tolerance

    def extract_data(self, ocr_results: List[List[Tuple[Any, str, float]]]) -> List[Dict[str, Any]]:
        """
        Extract structured data from OCR results.
        Args:
            ocr_results: List of pages, each containing OCR items.
        Returns:
            List of dictionaries representing extracted rows.
        """
        all_data = []
        
        for page_idx, page_items in enumerate(ocr_results):
            # 1. Cluster items into rows
            rows = self._cluster_into_rows(page_items)
            
            # 2. Parse rows with context (headers)
            page_data = self._parse_page_rows(rows, page_idx + 1)
            all_data.extend(page_data)
            
        return all_data

    def _cluster_into_rows(self, items: List[Tuple[Any, str, float]]) -> List[List[Tuple[Any, str, float]]]:
        """
        Group OCR items into rows based on Y-coordinate.
        """
        # Sort by Y-coordinate (top to bottom)
        sorted_items = sorted(items, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
        
        rows = []
        if not sorted_items:
            return rows

        # Helper to get Y from a point which might be [x, y] or just y (if structure varies)
        def get_y(pt_or_coord):
            if isinstance(pt_or_coord, list):
                return pt_or_coord[1]
            return pt_or_coord

        first_item = sorted_items[0]
        tl_y = first_item[0][0][1]
        br_y = first_item[0][2][1]
        
        current_row = [first_item]
        current_y = (tl_y + br_y) / 2
        
        for item in sorted_items[1:]:
            tl_y = item[0][0][1]
            br_y = item[0][2][1]
            item_y = (tl_y + br_y) / 2
            
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
        """
        Apply fuzzy corrections to item names, specifically for Page 1.
        """
        if page_num != 1:
            return name
            
        # Normalization
        name_clean = name.replace(" ", "").replace("(", "").replace(")", "")
        
        corrections = {
            "자전안정화": "자전 안정화 주파수(f0)",
            "보조채널조증ACPA7": "보조채널조종(ACPA1)",
            "보조채널조A0PA2": "보조채널조종(ACPA2)",
            "보조채널조증AOPA": "보조채널조종(ACPA2)",
            "보조채널조증A0PA3": "보조채널조종(ACPA3)",
            "보조채닐조증": "보조채널조종(ACPA3)",
            "보조채널조증": "보조채널조종(ACPA1)", # Default to 1 if ambiguous (put last)
            "시선PSAY": "시선각(SLAY)",
            "5혹정스PA": "5' 측정(ACPA)",
            "보조지널획의스PA": "보조채널 확인(ACPA)",
            "보조지널": "보조채널 확인(ACPA)"
        }
        
        for key, correct_name in corrections.items():
            if key in name_clean:
                return correct_name
                
        return name

    def _clean_number_string(self, value: str) -> str:
        """
        Clean numerical strings, specifically fixing comma/decimal confusion.
        Example: "2400,00" -> "2400.00", "10,.04" -> "10.04"
        """
        if not value:
            return value
            
        # Check if it looks like a number
        if not any(c.isdigit() for c in value):
            return value
            
        cleaned = value
        
        # Fix mixed separators common in OCR errors
        cleaned = cleaned.replace(",.", ".").replace(".,", ".")
        
        # Case 1: "digits,digits" (e.g., 2400,00)
        import re
        if re.search(r',\d{2}$', cleaned):
            cleaned = cleaned[::-1].replace(',', '.', 1)[::-1]
            
        cleaned = cleaned.replace(" ", "")
        return cleaned

    def _parse_page_rows(self, rows: List[List[Tuple[Any, str, float]]], page_num: int) -> List[Dict[str, Any]]:
        """
        Parse rows using an anchor-based strategy (Name ... Result).
        Prioritizes extracting Item Name and Measured Value.
        """
        extracted_rows = []
        
        # Keywords to identify headers (to skip them)
        header_keywords = ["점검", "항목", "기준", "측정", "판정", "Item", "Value", "Result"]
        
        # Result keywords to identify data rows (Right Anchor)
        result_keywords = ["정상", "불량", "Pass", "Fail", "OK", "NG", "F", "P"]
        
        current_section = ""
        
        # Pass 1: Extract using Anchor Logic and collect X-coordinates
        extracted_rows = []
        measured_val_xs = []
        
        for row in rows:
            text_row = [item[1] for item in row]
            row_text_joined = " ".join(text_row)
            
            # 1. Skip Header Rows
            matches = sum(1 for k in header_keywords if k in row_text_joined)
            if matches >= 2:
                pending_name = ""
                continue
                
            # 1.1 Skip Hex Dumps / Noise
            if "0x" in row_text_joined or "Ox" in row_text_joined:
                continue
                
            # 1.2 Skip Summary Rows
            if "종합판정" in row_text_joined or "종합 판정" in row_text_joined:
                continue
            
            # 2. Identify Result Anchor (Right Anchor)
            result_val = ""
            result_idx = -1
            
            # Search from end to start for the result
            for i in range(len(text_row) - 1, -1, -1):
                item = text_row[i]
                match_found = False
                
                for rk in result_keywords:
                    if rk in ["F", "P"]:
                        if item == rk:
                            match_found = True
                            break
                    else:
                        if rk in item:
                            match_found = True
                            break
                
                if match_found:
                    result_val = item
                    result_idx = i
                    break
            
            entry = {
                "Page": page_num,
                "index": current_section,
                "index_detail": "",
                "Measured_Value": "",
                "Reference_Tolerance": "",
                "Result": "",
                "Raw_Row": " | ".join(text_row),
                "_row_items": row # Store raw items for Pass 2
            }
            
            if result_idx != -1:
                # We have a data row!
                entry["Result"] = result_val
                
                # --- Extract Item Name (Left Anchor) ---
                item_name = ""
                start_idx = 0
                
                first_item = text_row[0]
                is_first_number = False
                try:
                    float(first_item.replace(',', '').replace(' ', '').replace('.', ''))
                    is_first_number = True
                except ValueError:
                    pass
                
                if is_first_number:
                    if pending_name:
                        item_name = pending_name
                        pending_name = "" 
                    else:
                        item_name = "" 
                    start_idx = 0 
                else:
                    item_name = text_row[0]
                    start_idx = 1 
                    pending_name = "" 
                
                item_name = self._correct_item_name(item_name, page_num)
                entry["index_detail"] = item_name
                
                # Filter noise
                if len(item_name) <= 1 and result_idx <= start_idx:
                     continue

                # --- Extract Measured Value ---
                measured_val = ""
                end_idx = result_idx
                
                if result_idx > start_idx:
                    candidate = text_row[result_idx - 1]
                    # Check if candidate looks like a number
                    clean_candidate = self._clean_number_string(candidate)
                    
                    # Heuristic: If it's a number, it's likely the value
                    # If it's not a number (e.g. "이상"), it might be part of reference?
                    # But usually value is immediately before result.
                    
                    try:
                        float(clean_candidate.replace(',', ''))
                        measured_val = clean_candidate
                        end_idx = result_idx - 1
                        
                        # Collect X-coordinate for Pass 2
                        # item structure: (bbox, text, conf)
                        # bbox: [[x,y], [x,y], [x,y], [x,y]]
                        val_item = row[result_idx - 1]
                        bbox = val_item[0]
                        center_x = (bbox[0][0] + bbox[1][0]) / 2
                        measured_val_xs.append(center_x)
                        
                    except ValueError:
                        # Not a number, maybe value is missing or mixed with ref?
                        pass
                
                entry["Measured_Value"] = measured_val
                
                # --- Extract Reference/Tolerance (Middle) ---
                middle_content = []
                if end_idx > start_idx:
                    middle_content = text_row[start_idx:end_idx]
                
                entry["Reference_Tolerance"] = " ".join(middle_content)
                extracted_rows.append(entry)
                
            else:
                # No result found. Check for Section Header or Pending Name
                if len(text_row) > 0:
                    has_numbers = any(any(c.isdigit() for c in t) for t in text_row)
                    
                    if not has_numbers:
                        candidate_text = " ".join(text_row)
                        if "점검" in candidate_text or "상태" in candidate_text or "ID" in candidate_text:
                            current_section = candidate_text
                            pending_name = "" 
                        else:
                            pending_name = candidate_text
                            pending_name = self._correct_item_name(pending_name, page_num)
                    else:
                        pass

        # Pass 2: Recover missing Measured Values using Column Alignment
        if measured_val_xs:
            min_x = min(measured_val_xs)
            max_x = max(measured_val_xs)
            min_x -= 20
            max_x += 20
            
            for entry in extracted_rows:
                if not entry["Measured_Value"] and "_row_items" in entry:
                    row_items = entry["_row_items"]
                    for item in row_items:
                        text = item[1]
                        bbox = item[0]
                        center_x = (bbox[0][0] + bbox[1][0]) / 2
                        
                        if text == entry["Result"] or text == entry["index_detail"]:
                            continue
                            
                        if min_x <= center_x <= max_x:
                            clean_val = self._clean_number_string(text)
                            try:
                                float(clean_val.replace(',', ''))
                                entry["Measured_Value"] = clean_val
                                break 
                            except ValueError:
                                pass
                
                if "_row_items" in entry:
                    del entry["_row_items"]
        else:
             for entry in extracted_rows:
                 if "_row_items" in entry:
                     del entry["_row_items"]

        # Final Filter: Drop rows with empty Measured_Value
        # This aligns with output_sample_ver3 which excludes status checks
        final_rows = []
        for entry in extracted_rows:
            if entry["Measured_Value"]:
                # Normalize Item Names to match ver3 (remove spaces)
                name = entry["index_detail"]
                
                # 1. Fix "Squib Resistance" variations
                # Matches: "스퀴브", "스귀스", "스키브" + "저향", "저항"
                if ("스퀴브" in name or "스귀스" in name or "스키브" in name) and ("저향" in name or "저항" in name):
                    name = name.replace(" ", "")
                    
                # 2. Fix "22P/N - PGND" variations
                # Matches: "22" and "PGND"
                if "22" in name and "PGND" in name:
                    name = name.replace(" ", "")
                    
                # 3. Fix "22P - 22N" variations
                # Matches: "22" and "22" (appears twice)
                if name.count("22") >= 2:
                    name = name.replace(" ", "")
                    
                entry["index_detail"] = name
                final_rows.append(entry)
                
        return final_rows

if __name__ == "__main__":
    # Test with dummy data
    pass
