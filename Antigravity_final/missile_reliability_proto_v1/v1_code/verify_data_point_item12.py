import pandas as pd

# Load QIM data without header
df = pd.read_csv('c:/Antigravity/missile_reliability_proto_v1/ref_data/QIM_data.csv', encoding='utf-8', header=None)

# Item 12 Column Index
# Item 1 is at Index 5
# Item 12 is at Index 5 + 11 = 16
item_12_col_idx = 16

# 1. Get Limits
lcl = df.iloc[5, item_12_col_idx]
ucl = df.iloc[6, item_12_col_idx]

print(f"Item 12 LCL: {lcl}")
print(f"Item 12 UCL: {ucl}")

# 2. Get Data for Serial 'SG 16 D074K 00'
target_serial = 'SG 16 D074K 00'
row_idx = df.index[df[2] == target_serial].tolist()

if row_idx:
    idx = row_idx[0]
    value = df.iloc[idx, item_12_col_idx]
    print(f"Serial {target_serial} Item 12 Value: {value}")
else:
    print(f"Serial {target_serial} not found.")
