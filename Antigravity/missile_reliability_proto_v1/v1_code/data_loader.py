import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = None
        self.qim_df = None
        self.asrp_df = None
        self.overhaul_df = None
        self.limits_df = None

    def load_data(self):
        """Loads QIM, ASRP, Overhaul, and Limits data."""
        # 1. Load QIM Data
        qim_path = os.path.join(self.data_dir, 'QIM_data.csv')
        self.qim_df = self._read_csv_with_skip(qim_path, encoding='utf-8')
        self.qim_df['Dataset'] = 'QIM'
        self.qim_df['운용월'] = 0  # QIM is always 0 months

        # 2. Load ASRP Data
        asrp_path = os.path.join(self.data_dir, 'ASRP_data.csv')
        self.asrp_df = self._read_csv_with_skip(asrp_path, encoding='utf-8')
        self.asrp_df['Dataset'] = 'ASRP'
        
        # 3. Load Overhaul Data
        overhaul_path = os.path.join(self.data_dir, 'overhaul_data.csv')
        try:
            self.overhaul_df = self._read_csv_with_skip(overhaul_path, encoding='cp949')
        except:
            self.overhaul_df = self._read_csv_with_skip(overhaul_path, encoding='utf-8')
        self.overhaul_df['Dataset'] = 'Overhaul'

        # 4. Load Limits (From QIM Data Header)
        # User confirmed: Row 6 (Index 5) is LCL, Row 7 (Index 6) is UCL
        # Data starts at Row 8 (Index 7)
        try:
            # Read QIM raw for limits (header=None to get absolute indices)
            qim_raw = pd.read_csv(qim_path, encoding='utf-8', header=None)
            
            # Extract LCL and UCL
            # Columns 5 to 31 (Item 1 to 27) -> Index 5 is '1', Index 31 is '27'
            lcl_row = qim_raw.iloc[5, 5:32]
            ucl_row = qim_raw.iloc[6, 5:32]
            
            # Convert to numeric
            lcl = pd.to_numeric(lcl_row, errors='coerce').values
            ucl = pd.to_numeric(ucl_row, errors='coerce').values
            
            self.limits_df = pd.DataFrame({
                'Item': [str(i) for i in range(1, 28)],
                'USL': ucl,
                'LSL': lcl
            })
            
        except Exception as e:
            print(f"Error loading limits from QIM: {e}")
            self.limits_df = None

        # 5. Merge Main DataFrames
        # Ensure common columns exist. We need '1'...'27' and '운용월'.
        
        # Standardize column names if they are like '1_측정치'
        self._standardize_columns(self.qim_df)
        self._standardize_columns(self.asrp_df)
        self._standardize_columns(self.overhaul_df)

        self.df = pd.concat([self.qim_df, self.asrp_df, self.overhaul_df], ignore_index=True)
        print(f"DEBUG: Combined DF Columns: {self.df.columns.tolist()}")
        
        # Ensure numeric types
        measurement_cols = [str(i) for i in range(1, 28)]
        for col in measurement_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df['운용월'] = pd.to_numeric(self.df['운용월'], errors='coerce').fillna(0)
        
        # Refresh individual DFs to ensure they have the numeric data and standardized columns
        self.qim_df = self.df[self.df['Dataset'] == 'QIM'].copy()
        self.asrp_df = self.df[self.df['Dataset'] == 'ASRP'].copy()
        self.overhaul_df = self.df[self.df['Dataset'] == 'Overhaul'].copy()
        
        return self.df

    def _read_csv_with_skip(self, path, encoding):
        """Reads CSV with header at row 0, then drops first 6 rows (metadata + limits)."""
        # Read with default header=0
        df = pd.read_csv(path, encoding=encoding)
        # Drop first 6 rows (Indices 0-5 are metadata/limits, Data starts at Index 6 -> Row 7 in raw)
        # Wait, Raw Index 7 is Data.
        # Header is Raw Index 0.
        # DF Index 0 is Raw Index 1.
        # DF Index 6 is Raw Index 7.
        # So df.iloc[6:] is correct.
        df = df.iloc[6:].reset_index(drop=True)
        
        # Drop 'Unnamed' columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df

    def _standardize_columns(self, df):
        """Renames columns to '1', '2', ... '27' if needed."""
        # Convert all columns to string first
        df.columns = df.columns.astype(str)
        
        # Handle cases where columns might be '1.0', '2.0' etc.
        new_columns = {}
        for col in df.columns:
            if col.endswith('.0'):
                new_col = col.replace('.0', '')
                if new_col.isdigit() and 1 <= int(new_col) <= 27:
                    new_columns[col] = new_col
        
        if new_columns:
            df.rename(columns=new_columns, inplace=True)
            
        # Also handle '1_측정치' style if present (though ref code created them, input might have them)
        # But ref code input had '1', '2'...
        # Just ensure '1'...'27' exist.
        pass

    def split_data(self):
        """Splits data into QIM and Aged (ASRP + Overhaul)."""
        if self.df is None:
            self.load_data()
            
        qim = self.df[self.df['Dataset'] == 'QIM'].copy()
        aged = self.df[self.df['Dataset'].isin(['ASRP', 'Overhaul'])].copy()
        
        return qim, aged

    def get_measurement_columns(self):
        cols = [str(i) for i in range(1, 28)]
        existing_cols = [c for c in cols if c in self.df.columns]
        return existing_cols
