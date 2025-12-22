import sys
import os
import pandas as pd

# Add vgui3.0 paths
sys.path.append(os.path.abspath("c:/Antigravity/missile_reliability_proto_v1/vgui3.0"))

from modules.csv_integrator import CSVIntegrator

def test_integration():
    integrator = CSVIntegrator()
    target_folder = "c:/Antigravity/missile_reliability_proto_v1/csv2csv"
    
    print(f"Testing integration on: {target_folder}")
    df = integrator.integrate_folder(target_folder)
    
    if df.empty:
        print("Result is Empty!")
    else:
        print("Integration Successful!")
        print("Columns:", df.columns.tolist())
        print("Row Content:")
        print(df.iloc[0])
        
    print("\n--- Diagnostic for Item 2 (Injection Squib) ---")
    keys_for_2 = [k for k,v in integrator.item_map.items() if '2' in v]
    print(f"Keys for ID 2 in Map: {keys_for_2}")
    
    # Simulate extraction for 1200.csv
    f1200 = os.path.join(target_folder, "1200.csv")
    if os.path.exists(f1200):
        try:
             df = pd.read_csv(f1200, header=None, names=range(30), encoding='utf-8', engine='python')
             # Row 9 (Index 8) is Item 2
             r = df.iloc[8]
             raw_name = str(r[2]) # "사출모터스퀴브 저항"
             # Apply cleaning exactly as inside integrator
             clean_name = integrator._correct_item_name(raw_name).replace(" ", "").upper()
             clean_name = clean_name.replace("·", "").replace(".", "").replace("-", "")
             
             print(f"Raw Input: '{raw_name}'")
             print(f"Cleaned Input: '{clean_name}'")
             print(f"Expected Key matches: {[k for k in keys_for_2 if k == clean_name]}")
             print(f"Is Clean Name in Keys? {clean_name in keys_for_2}")
             
        except Exception as e:
            print(f"Error checking 1200: {e}")

    print("\n--- Diagnostic for Item 4 (PGND-22P) ---")
    keys_for_4 = [k for k,v in integrator.item_map.items() if '4' in v]
    print(f"Keys for ID 4 in Map: {keys_for_4}")

if __name__ == "__main__":
    test_integration()
