import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.qim_df = None
        self.asrp_df = None

    def load_data(self):
        """Loads data from CSV and handles encoding."""
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.file_path, encoding='cp949')
        
        # Basic cleaning if necessary (e.g., dropping empty rows)
        self.df.dropna(how='all', inplace=True)
        
        # Ensure '운용월' is numeric
        self.df['운용월'] = pd.to_numeric(self.df['운용월'], errors='coerce').fillna(0)
        
        return self.df

    def split_data(self):
        """Splits data into QIM (Month 0) and ASRP (Month > 0)."""
        if self.df is None:
            self.load_data()
            
        self.qim_df = self.df[self.df['운용월'] == 0].copy()
        self.asrp_df = self.df[self.df['운용월'] > 0].copy()
        
        return self.qim_df, self.asrp_df

    def get_measurement_columns(self):
        """Identifies measurement columns (1 to 27)."""
        # Assuming columns '1' through '27' exist
        cols = [str(i) for i in range(1, 28)]
        existing_cols = [c for c in cols if c in self.df.columns]
        return existing_cols
