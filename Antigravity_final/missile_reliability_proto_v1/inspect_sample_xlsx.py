import pandas as pd

def inspect(path):
    print(f"\n--- Inspecting {path} ---")
    try:
        df = pd.read_csv(path, encoding='cp949') # Try Korean encoding first
        print("Columns:", df.columns.tolist())
        print("Row 0:", df.iloc[0].to_dict() if not df.empty else "Empty")
    except Exception as e:
        try:
            df = pd.read_csv(path, encoding='utf-8')
            print("Columns (utf-8):", df.columns.tolist())
            print("Row 0:", df.iloc[0].to_dict() if not df.empty else "Empty")
        except Exception as e2:
            print(f"Error: {e2}")

inspect('c:/Antigravity/missile_reliability_proto_v1/csv_sample.csv')
inspect('c:/Antigravity/missile_reliability_proto_v1/PGM_ELEC_test_5-min (1)_sample_csv.csv')
