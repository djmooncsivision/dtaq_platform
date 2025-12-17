import pandas as pd
import os

class MockIntegrator:
    def __init__(self):
        self.item_map = {'열전지스퀴브저항': ['1'], '사출모터스퀴브저항': ['2']}
    
    def _correct_item_name(self, name):
        return name

    def test(self, file_path):
        print(f"Reading {file_path}")
        df_raw = pd.read_csv(file_path, header=None, names=range(30), encoding='utf-8', engine='python')
        print(f"Rows: {len(df_raw)}")
        
        row = {}
        
        for idx, r in df_raw.iterrows():
            print(f"--- Row {idx} ---")
            if len(r) < 11: 
                print("Skipped len")
                continue
            
            name_candidates = [str(r[2]), str(r[3]) if len(r) > 3 else ""]
            print(f"Candidates: {name_candidates}")
            
            for name_raw in name_candidates:
                if pd.isna(name_raw) or name_raw == 'nan': continue
                
                clean_name = str(name_raw).replace(" ", "").upper().replace("-", "")
                print(f"  Checking '{clean_name}'")
                
                item_id = None
                
                # logic copy
                candidate_ids = self.item_map.get(clean_name)
                if candidate_ids:
                    item_id = candidate_ids[0]
                
                if item_id:
                    print(f"  MATCH: {item_id}")
                    row[str(item_id)] = r[10]
                    break
        
        print("Final Row Keys:", row.keys())

if __name__ == "__main__":
    t = MockIntegrator()
    t.test("c:/Antigravity/missile_reliability_proto_v1/csv2csv/1200.csv")
