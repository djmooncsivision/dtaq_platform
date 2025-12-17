import pandas as pd
import os
import sys
from typing import List

def normalize_item_name(name: str) -> str:
    """
    Normalizes item names to handle OCR variations and merge columns.
    """
    name = str(name).strip()
    
    # Mapping Dictionary (Partial matches or exact matches)
    # Key: Canonical Name, Value: List of keywords/substrings to match
    mappings = {
        "자전안정화주파수(f0)": ["자전안정화주파수", "지전안정화주파수", "자전 안정화 주파수"],
        "자전기동시간(초)": ["자전기동시간"],
        "보조채널조종(ACPA1)": ["ACPA1", "ACPA 1", "ACPA7"], # OCR often reads 1 as 7 or l
        "보조채널조종(ACPA2)": ["ACPA2", "ACPA 2", "A0PA2"],
        "보조채널조종(ACPA3)": ["ACPA3", "ACPA 3", "A0PA3"],
        "보조채널조종(ACPA4)": ["ACPA4", "ACPA 4", "AOPA"], # Sometimes just AOPA if 4 is missed?
        "시선각(SLAY)": ["시선각", "시선리", "시선P"],
        "5' 측정(ACPA)": ["5' 측정", "5혹정"],
        "보조채널 확인(ACPA)": ["보조채널 확인", "보조지널", "보조채널되애"],
        "열전지스퀴브저항": ["열전지스퀴브", "열전지스귀스", "열전지스키브"],
        "사출모터스퀴브저항": ["사출모터스퀴브", "사출모터스키브", "사출모터스귀스"],
        "그 5V": ["그 5V", "그5V"],
        "+5V": ["+5V", "5V"], # Be careful not to match +15V
        "냉각유지시간": ["냉각유지"],
        "ASLAP": ["ASLAP"],
        "ASLAY": ["ASLAY"],
        "ALOSRP": ["ALOSRP"],
        "ALOSRY": ["ALOSRY", "ALSORY"],
        "AVIFSC": ["AVIFSC"],
        "AFROLL": ["AFROLL"],
        "AMRSIN": ["AMRSIN"],
        "SLAP": ["SLAP"],
        "SLAY": ["SLAY"],
        "22P": ["22P"],
        "22N": ["22N"],
        "PGND": ["PGND"],
        "+15V": ["+15V"],
        "-15V": ["-15V"]
    }
    
    # Specific Order: Check longer/more specific matches first
    # For example, check "ACPA1" before "ACPA"
    
    # 1. Exact/Close matches for complex names
    if "ACPA1" in name or "ACPA 1" in name or "ACPA7" in name: return "보조채널조종(ACPA1)"
    if "ACPA2" in name or "ACPA 2" in name or "A0PA2" in name: return "보조채널조종(ACPA2)"
    if "ACPA3" in name or "ACPA 3" in name or "A0PA3" in name: return "보조채널조종(ACPA3)"
    # ACPA4 is tricky if it reads as AOPA without number. 
    # Let's assume if it contains "AOPA" or "ACPA" and NOT 1,2,3, it might be 4? 
    # But wait, "5' 측정" also has ACPA.
    
    if "5'" in name or "5혹정" in name: return "5' 측정(ACPA)"
    if "확인" in name or "획의" in name or "되애" in name: return "보조채널 확인(ACPA)"
    
    # Fallback for ACPA4 if needed, or just rely on specific unique strings
    
    if "시선" in name: return "시선각(SLAY)"
    
    if "자전" in name and "주파수" in name: return "자전안정화주파수(f0)"
    if "지전" in name and "주파수" in name: return "자전안정화주파수(f0)"
    
    if "자전" in name and "기동" in name: return "자전기동시간(초)"
    
    if "열전지" in name: return "열전지스퀴브저항"
    if "사출모터" in name: return "사출모터스퀴브저항"
    
    if "냉각" in name: return "냉각유지시간"
    
    if "그" in name and "5V" in name: return "그 5V"
    
    # Exact matches for short codes
    if name in ["+5V", "5V"]: return "+5V"
    if name in ["+15V", "15V"]: return "+15V"
    if name in ["-15V"]: return "-15V"
    
    # General substring matching for others
    for canonical, keywords in mappings.items():
        for kw in keywords:
            if kw in name:
                return canonical
                
    return name

def aggregate_csvs(file_paths: List[str], output_path: str):
    """
    Aggregates multiple CSV files into a single summary CSV.
    Rows: Files
    Columns: Item Names (index_detail)
    Values: Measured_Value
    """
    aggregated_data = []
    
    # Collect all unique item names to ensure consistent columns
    all_items = set()
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            if 'index_detail' not in df.columns or 'Measured_Value' not in df.columns:
                print(f"Warning: Missing required columns in {file_path}")
                continue
            
            # Create a dictionary for this file
            file_data = {'Filename': os.path.basename(file_path)}
            
            for _, row in df.iterrows():
                raw_name = row['index_detail']
                item_name = normalize_item_name(raw_name) # Normalize here
                measured_val = row['Measured_Value']
                
                file_data[item_name] = measured_val
                all_items.add(item_name)
            
            aggregated_data.append(file_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not aggregated_data:
        print("No data aggregated.")
        return

    # Create DataFrame
    # Ensure 'Filename' is the first column
    # Convert all items to string to avoid comparison errors
    sorted_items = sorted([str(i) for i in all_items])
    cols = ['Filename'] + sorted_items
    result_df = pd.DataFrame(aggregated_data)
    
    # Reorder columns, filling missing with NaN
    result_df = result_df.reindex(columns=cols)
    
    # Save
    try:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully aggregated {len(result_df)} files to {output_path}")
    except Exception as e:
        print(f"Error saving aggregated CSV: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python aggregate_csv.py <output_file> <input_csv1> <input_csv2> ...")
        sys.exit(1)
        
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    
    aggregate_csvs(input_files, output_file)
