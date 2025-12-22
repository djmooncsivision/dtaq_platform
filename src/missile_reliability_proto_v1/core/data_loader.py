import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, file_path_or_dir):
        self.file_path_or_dir = file_path_or_dir
        self.df = None
        self.limits_df = None
        self.qim_df = None
        self.asrp_df = None

    def load_data(self):
        """Loads data from CSV file or directory of CSV files."""
        if os.path.isdir(self.file_path_or_dir):
            # If directory, load and merge all CSVs (Legacy logic from scenario_data)
            all_files = [f for f in os.listdir(self.file_path_or_dir) if f.endswith('.csv')]
            dfs = []
            for f in all_files:
                path = os.path.join(self.file_path_or_dir, f)
                try:
                    df_temp = pd.read_csv(path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df_temp = pd.read_csv(path, encoding='cp949')
                dfs.append(df_temp)
            if dfs:
                self.df = pd.concat(dfs, ignore_index=True)
        else:
            # Single file
            try:
                self.df = pd.read_csv(self.file_path_or_dir, encoding='utf-8-sig')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.file_path_or_dir, encoding='cp949')
        
        if self.df is not None:
            self.df.dropna(how='all', inplace=True)
            if '운용월' in self.df.columns:
                self.df['운용월'] = pd.to_numeric(self.df['운용월'], errors='coerce').fillna(0)
            if 'Dataset' not in self.df.columns:
                # Infer Dataset from Month
                self.df['Dataset'] = self.df['운용월'].apply(lambda x: 'QIM' if x == 0 else 'ASRP')
        
        return self.df

    def split_data(self):
        """Splits data into QIM (Month 0) and ASRP (Month > 0)."""
        if self.df is None:
            self.load_data()
            
        self.qim_df = self.df[self.df['Dataset'] == 'QIM'].copy()
        self.asrp_df = self.df[self.df['Dataset'] == 'ASRP'].copy()
        
        return self.qim_df, self.asrp_df

    def get_measurement_columns(self):
        """Identifies measurement columns (1 to 27)."""
        cols = [str(i) for i in range(1, 28)]
        existing_cols = [c for c in cols if c in self.df.columns]
        return existing_cols
