import pandas as pd
try:
    df = pd.read_csv('c:/Antigravity/missile_reliability_proto_v1/ref_data/upper_lower_limit.csv', encoding='utf-8', header=None)
    print(df.iloc[4:8, 31]) # Check column 31 (Item 27?)
except Exception as e:
    print(e)
