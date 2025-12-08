import pandas as pd
import numpy as np
import random
import os
import sys

# Add v1_code to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v1_code')))
from data_loader import DataLoader

def generate_synthetic_data(
    n_qim=200, n_asrp=50, n_overhaul=200,
    asrp_time_range=(96, 144), overhaul_time_range=(108, 132),
    degrading_items=None, drift_rate=0.05, noise_growth=1.0,
    base_mean=100, base_std=2
):
    """
    Generates a synthetic DataFrame with QIM, ASRP, and Overhaul data.
    Returns a pandas DataFrame ready for analysis.
    """
    if degrading_items is None:
        degrading_items = []

    n_items = 27
    item_params = {i: {'mean': base_mean, 'std': base_std} for i in range(1, n_items + 1)}
    
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
            val = np.random.normal(item_params[i]['mean'], item_params[i]['std'])
            row[str(i)] = round(val, 4)
        qim_data.append(row)
        
    # --- 2. Generate ASRP Data ---
    asrp_data = []
    for _ in range(n_asrp):
        # Random month within range
        month = random.randint(asrp_time_range[0], asrp_time_range[1])
        
        row = {
            'Code': 'B',
            '일련번호': f'SN_A_{random.randint(1000, 9999)}',
            '품번': 'P12345',
            '시험일자': '2023-06-01',
            '운용월': month,
            'Dataset': 'ASRP'
        }
        for i in range(1, n_items + 1):
            base_m = item_params[i]['mean']
            base_s = item_params[i]['std']
            
            if i in degrading_items:
                # Apply degradation
                current_mean = base_m - (drift_rate * month)
                std_growth_factor = 1 + (month / 240.0) * noise_growth
                current_std = base_s * std_growth_factor
            else:
                current_mean = base_m
                current_std = base_s
                
            val = np.random.normal(current_mean, current_std)
            row[str(i)] = round(val, 4)
        asrp_data.append(row)
        
    # --- 3. Generate Overhaul Data ---
    overhaul_data = []
    for _ in range(n_overhaul):
        month = random.randint(overhaul_time_range[0], overhaul_time_range[1])
        
        row = {
            'Code': 'C',
            '일련번호': f'SN_O_{random.randint(1000, 9999)}',
            '품번': 'P12345',
            '시험일자': '2023-09-01',
            '운용월': month,
            'Dataset': 'Overhaul'
        }
        for i in range(1, n_items + 1):
            base_m = item_params[i]['mean']
            base_s = item_params[i]['std']
            
            if i in degrading_items:
                # Apply degradation (same logic as ASRP for now)
                current_mean = base_m - (drift_rate * month)
                std_growth_factor = 1 + (month / 240.0) * noise_growth
                current_std = base_s * std_growth_factor
            else:
                current_mean = base_m
                current_std = base_s
                
            val = np.random.normal(current_mean, current_std)
            row[str(i)] = round(val, 4)
        overhaul_data.append(row)
        
    # Combine all data
    all_data = qim_data + asrp_data + overhaul_data
    df = pd.DataFrame(all_data)
    
    # Reorder columns
    cols = ['Code', '일련번호', '품번', '시험일자', '운용월', 'Dataset'] + [str(i) for i in range(1, n_items + 1)]
    df = df[cols]
    
    return df

def create_limits_df(n_items=27, base_mean=100, base_std=2):
    """Creates a DataFrame for Limits (LSL, USL)."""
    limits_data = []
    for i in range(1, n_items + 1):
        limits_data.append({
            'Item': str(i),
            'LSL': base_mean - 4 * base_std,
            'USL': base_mean + 4 * base_std
        })
    return pd.DataFrame(limits_data)

class InMemoryLoader:
    """Mock DataLoader that holds a DataFrame directly."""
    def __init__(self, df, limits_df=None):
        self.df = df
        self.limits_df = limits_df
        
    def split_data(self):
        qim = self.df[self.df['Dataset'] == 'QIM']
        asrp = self.df[self.df['Dataset'] == 'ASRP']
        return qim, asrp
