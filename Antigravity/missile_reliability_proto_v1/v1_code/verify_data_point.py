import pandas as pd

# Load QIM data without header to access by absolute index
df = pd.read_csv('c:/Antigravity/missile_reliability_proto_v1/ref_data/QIM_data.csv', encoding='utf-8', header=None)

# Item 5 Column Index
# Index 0: Code, 1: Unnamed, 2: Serial, 3: Part, 4: Date
# Index 5: Item 1
# Index 9: Item 5
item_5_col_idx = 9

# 1. Get Limits
# LCL is at Row 5 (Index 5)
lcl = df.iloc[5, item_5_col_idx]
# UCL is at Row 6 (Index 6)
ucl = df.iloc[6, item_5_col_idx]

print(f"Item 5 LCL: {lcl}")
print(f"Item 5 UCL: {ucl}")

# 2. Get Data for Serial 'SG 16 D074K 00'
# Serial column is Index 2
target_serial = 'SG 16 D074K 00'
# Find row index
row_idx = df.index[df[2] == target_serial].tolist()

if row_idx:
    idx = row_idx[0]
    value = df.iloc[idx, item_5_col_idx]
    print(f"Serial {target_serial} Item 5 Value: {value}")
else:
    print(f"Serial {target_serial} not found.")
