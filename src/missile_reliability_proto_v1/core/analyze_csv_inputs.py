import sys
import os
import pandas as pd
import glob

# Add logic path
sys.path.append(os.path.abspath("c:/Antigravity/missile_reliability_proto_v1/vgui3.0"))
from modules.csv_integrator import CSVIntegrator

def analyze_inputs():
    folder = "c:/Antigravity/missile_reliability_proto_v1/csv2csv"
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    integrator = CSVIntegrator()
    
    assigned_ids = {} # ID -> List of (File, RowVal, RawName)
    
    print(f"--- Analyzing {len(files)} CSV files in {folder} ---")
    
    for file_path in files:
        fname = os.path.basename(file_path)
        if "sample" in fname: continue
        
        print(f"\n>> Analyzing {fname}")
        try:
            df = pd.read_csv(file_path, header=None, names=range(30), encoding='utf-8', engine='python')
        except:
            df = pd.read_csv(file_path, header=None, names=range(30), encoding='cp949', engine='python')
            
        for idx, row in df.iterrows():
            # Skip likely header rows (checking if col 10 is numeric)
            val_raw = row[10]
            try:
                # Basic check if it's a value row
                float(str(val_raw).replace(',', ''))
            except:
                continue # Skip non-data rows
            
            # Extract potential names
            name_candidates = [str(row[2]), str(row[3])]
            
            # Try to map using integrator logic (copy-paste-ish logic)
            matched_id = None
            matched_reason = ""
            
            for name_raw in name_candidates:
                if pd.isna(name_raw) or name_raw == 'nan' or not name_raw.strip(): continue
                
                clean_name = integrator._correct_item_name(name_raw).replace(" ", "").upper()
                clean_name = clean_name.replace("·", "").replace(".", "")
                
                # Direct Map
                candidate_ids = integrator.item_map.get(clean_name)
                
                # Check Fuzzy
                if not candidate_ids:
                     fuzzy_candidates = []
                     for k, v in integrator.item_map.items():
                        if k in clean_name or clean_name in k: 
                            fuzzy_candidates.extend(v)
                     if fuzzy_candidates:
                        candidate_ids = sorted(list(set(fuzzy_candidates)), key=lambda x: int(x))

                if candidate_ids:
                    # In this analysis, we just take the first candidate to see what matches
                    # The integrator has sequential logic, but here we just want to see IF it matches anything.
                    matched_id = candidate_ids
                    matched_reason = f"Matched '{clean_name}'"
                    break
            
            val = row[10]
            print(f"  Row {idx}: Col2='{row[2]}', Col3='{row[3]}' -> ID: {matched_id} | Val: {val}")
            
            if matched_id:
                for mid in matched_id:
                    if mid not in assigned_ids: assigned_ids[mid] = []
                    assigned_ids[mid].append((fname, val, row[2]))

    print("\n\n--- Summary of Assignments ---")
    all_ids = [str(i) for i in range(1, 28)]
    for i in all_ids:
        if i in assigned_ids:
            entries = assigned_ids[i]
            count = len(entries)
            status = "OK" if count == 1 else "DUPLICATE/AMBIGUOUS"
            print(f"Item {i}: {status} ({count} found)")
            for e in entries:
                print(f"    - {e[0]}: {e[1]} (Name: {e[2]})")
        else:
            print(f"Item {i}: MISSING")

if __name__ == "__main__":
    analyze_inputs()
