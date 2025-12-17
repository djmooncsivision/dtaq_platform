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

    def _parse_page_rows(self, rows: List[List[Tuple[Any, str, float]]], page_num: int) -> List[Dict[str, Any]]:
        """
        Parse rows using an anchor-based strategy (Name ... Result).
        Prioritizes extracting Item Name and Measured Value.
        """
        extracted_rows = []
        
        # Keywords to identify headers (to skip them)
        header_keywords = ["점검", "항목", "기준", "측정", "판정", "Item", "Value", "Result"]
        
        # Result keywords to identify data rows (Right Anchor)
        # Added "F", "P" for single-letter results
        result_keywords = ["정상", "불량", "Pass", "Fail", "OK", "NG", "F", "P"]
        
        pending_name = ""
        
        for row in rows:
            text_row = [item[1] for item in row]
            row_text_joined = " ".join(text_row)
            
            # 1. Skip Header Rows
            matches = sum(1 for k in header_keywords if k in row_text_joined)
            if matches >= 2:
                pending_name = ""
                continue
            
            # 2. Identify Result Anchor (Right Anchor)
            result_val = ""
            result_idx = -1
            
            # Search from end to start for the result
            for i in range(len(text_row) - 1, -1, -1):
                item = text_row[i]
                if any(rk in item for rk in result_keywords):
                    # Strict check for single letters to avoid false positives in normal text
                    if item in ["F", "P"]:
                        result_val = item
                        result_idx = i
                        break
                    else:
                        result_val = item
                        result_idx = i
                        break
            
            if result_idx != -1:
                # We have a data row!
                
                # --- Extract Item Name (Left Anchor) ---
                item_name = ""
                start_idx = 0
                
                # Heuristic: If the first item looks like a number, it's probably NOT the name
                first_item = text_row[0]
                is_first_number = False
                try:
                    float(first_item.replace(',', '').replace(' ', ''))
                    is_first_number = True
                except ValueError:
                    pass
                
                if is_first_number:
                    # Name is missing in this row
                    if pending_name:
                        item_name = pending_name
                        pending_name = "" # Consume it
                    else:
                        item_name = "" # Truly missing
                    start_idx = 0 # Data starts at 0
                else:
                    # First item is the Name
                    item_name = text_row[0]
                    start_idx = 1 # Data starts after name
                    pending_name = "" # Clear pending
                
                # Apply correction to item name
                item_name = self._correct_item_name(item_name, page_num)
                
                # --- Extract Measured Value ---
                # Look for the value immediately before the Result
                measured_val = ""
                end_idx = result_idx
                
                if result_idx > start_idx:
                    # The item before result is likely the measured value
                    candidate = text_row[result_idx - 1]
                    measured_val = candidate
                    end_idx = result_idx - 1
                
                # --- Extract Reference/Tolerance (Middle) ---
                # Everything between Name and Measured Value
                middle_content = []
                if end_idx > start_idx:
                    middle_content = text_row[start_idx:end_idx]
                
                ref_tol_str = " ".join(middle_content)
                
                entry = {
                    "Page": page_num,
                    "Item_Name": item_name,
                    "Measured_Value": measured_val,
                    "Reference_Tolerance": ref_tol_str,
                    "Result": result_val,
                    "Raw_Row": " | ".join(text_row)
                }
                extracted_rows.append(entry)
                
            else:
                # No result found.
                # Check if it's a "Pending Name" row (text only, no numbers)
                if len(text_row) > 0:
                    # If it has no numbers, assume it's a name for the next row
                    has_numbers = any(any(c.isdigit() for c in t) for t in text_row)
                    if not has_numbers:
                        pending_name = " ".join(text_row)
                        # Apply correction immediately to pending name too
                        pending_name = self._correct_item_name(pending_name, page_num)
                    else:
                        pass

        return extracted_rows

if __name__ == "__main__":
    # Test with dummy data
    pass
