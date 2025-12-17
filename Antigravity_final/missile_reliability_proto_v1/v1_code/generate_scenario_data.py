import pandas as pd
import numpy as np
import os
import random

def generate_scenario_data(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameters
    # Ratio: QIM:Overhaul:ASRP = 8:8:2
    # Let's set a base unit of 25.
    # QIM = 8 * 25 = 200
    # Overhaul = 8 * 25 = 200
    # ASRP = 2 * 25 = 50
    n_qim = 200
    n_overhaul = 200
    n_asrp = 50
    n_items = 27
    
    # Base parameters for items (Mean, Std)
    # Most items are stable around Mean=100, Std=2
    item_params = {i: {'mean': 100, 'std': 2} for i in range(1, n_items + 1)}
    
    # Define Degrading Items (Items 23, 24, 25, 26, 27)
    # Trend: Mean decreases by 'drift' per month, Std increases by 'noise_growth' per month
    degrading_items = [23, 24, 25, 26, 27]
    drift_rates = {
        23: 0.05,  # Fast degradation
        24: 0.04,
        25: 0.03,
        26: 0.02,
        27: 0.01   # Slow degradation
    }
    
    # --- 1. Generate QIM Data (Time = 0) ---
    qim_data = []
    for _ in range(n_qim):
        row = {
            'Code': 'A',
            '일련번호': f'SN_Q_{random.randint(1000, 9999)}',
            '품번': 'P12345',
            '시험일자': '2023-01-01',
            '운용월': 0,
            'Dataset': 'QIM'
        }
        for i in range(1, n_items + 1):
            # QIM is baseline
            val = np.random.normal(item_params[i]['mean'], item_params[i]['std'])
            row[str(i)] = round(val, 4)
        qim_data.append(row)
    
    # Add Limits (embedded in QIM file as per v1 logic: Rows 0-5 metadata, Row 5=LCL, Row 6=UCL)
    # We need to structure the CSV exactly as v1 loader expects.
    # v1 loader expects:
    # Header: Row 0
    # Metadata: Rows 1-4
    # LCL: Row 5 (Index 5)
    # UCL: Row 6 (Index 6)
    # Data: Row 7+ (Index 7+)
    
    # Create DataFrame
    qim_df = pd.DataFrame(qim_data)
    
    # Create Limits Rows
    lcl_row = {col: '' for col in qim_df.columns}
    ucl_row = {col: '' for col in qim_df.columns}
    
    for i in range(1, n_items + 1):
        # Limits: Mean +/- 4*Std (wide enough for QIM)
        mean = item_params[i]['mean']
        std = item_params[i]['std']
        lcl_row[str(i)] = mean - 4 * std
        ucl_row[str(i)] = mean + 4 * std
        
    # Create Metadata Rows (Empty placeholders)
    meta_rows = [{col: '' for col in qim_df.columns} for _ in range(4)]
    
    # Combine for QIM CSV
    # Order: Header(already in df columns), Meta(4), LCL(1), UCL(1), Data
    # We'll write this manually to CSV to ensure structure
    
    qim_final_rows = meta_rows + [lcl_row, ucl_row] + qim_data
    qim_final_df = pd.DataFrame(qim_final_rows)
    # Reorder columns to ensure 1..27 are present
    cols = ['Code', '일련번호', '품번', '시험일자', '운용월', 'Dataset'] + [str(i) for i in range(1, n_items + 1)]
    qim_final_df = qim_final_df[cols]
    
    qim_path = os.path.join(output_dir, 'QIM_data.csv')
    qim_final_df.to_csv(qim_path, index=False)
    
    # --- 2. Generate ASRP Data (Time: 102 months) ---
    asrp_data = []
    for _ in range(n_asrp):
        month = 102 # Fixed at 102 months as requested
            
        row = {
            'Code': 'B',
            '일련번호': f'SN_A_{random.randint(1000, 9999)}',
            '품번': 'P12345',
            '시험일자': '2023-06-01',
            '운용월': month,
            'Dataset': 'ASRP'
        }
        for i in range(1, n_items + 1):
            base_mean = item_params[i]['mean']
            base_std = item_params[i]['std']
            
            if i in degrading_items:
                # Apply degradation
                # Mean decreases
                current_mean = base_mean - (drift_rates[i] * month)
                # Variance increases (Std increases)
                std_growth = 1 + (month / 240.0)
                current_std = base_std * std_growth
            else:
                # Stable
                current_mean = base_mean
                current_std = base_std
                
            val = np.random.normal(current_mean, current_std)
            row[str(i)] = round(val, 4)
        asrp_data.append(row)
        
    asrp_final_rows = meta_rows + [lcl_row, ucl_row] + asrp_data
    asrp_final_df = pd.DataFrame(asrp_final_rows)
    asrp_final_df = asrp_final_df[cols]
    asrp_path = os.path.join(output_dir, 'ASRP_data.csv')
    asrp_final_df.to_csv(asrp_path, index=False)
    
    # --- 3. Generate Overhaul Data (Time: 108~132) ---
    overhaul_data = []
    for _ in range(n_overhaul):
        month = random.randint(108, 132)
        row = {
            'Code': 'C',
            '일련번호': f'SN_O_{random.randint(1000, 9999)}',
            '품번': 'P12345',
            '시험일자': '2023-09-01',
            '운용월': month,
            'Dataset': 'Overhaul'
        }
        for i in range(1, n_items + 1):
            base_mean = item_params[i]['mean']
            base_std = item_params[i]['std']
            
            if i in degrading_items:
                # Apply degradation similar to ASRP
                current_mean = base_mean - (drift_rates[i] * month)
                std_growth = 1 + (month / 240.0)
                current_std = base_std * std_growth
            else:
                # Stable
                current_mean = base_mean
                current_std = base_std
                
            val = np.random.normal(current_mean, current_std)
            row[str(i)] = round(val, 4)
        overhaul_data.append(row)
        
    overhaul_final_rows = meta_rows + [lcl_row, ucl_row] + overhaul_data
    overhaul_final_df = pd.DataFrame(overhaul_final_rows)
    overhaul_final_df = overhaul_final_df[cols]
    overhaul_path = os.path.join(output_dir, 'Overhaul_data.csv')
    overhaul_final_df.to_csv(overhaul_path, index=False)
    
    print(f"Scenario Data Generated in {output_dir}")
    print(f"Degrading Items: {degrading_items}")

if __name__ == "__main__":
    generate_scenario_data("c:/Antigravity/missile_reliability_proto_v1/v1_code/scenario_data")
