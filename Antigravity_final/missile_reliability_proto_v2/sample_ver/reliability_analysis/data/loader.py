import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from ..config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()

    def load_stockpile_data(self) -> pd.DataFrame:
        """Loads the stockpile composition data."""
        path = os.path.join(self.config.DATA_DIR, self.config.STOCKPILE_DATA_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stockpile data not found at {path}")
        return pd.read_csv(path)

    def load_observed_data(self, filename: str) -> pd.DataFrame:
        """Loads the observed reliability test data."""
        path = os.path.join(self.config.DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Observed data not found at {path}")
        return pd.read_csv(path)

    def prepare_data_for_analysis(self, observed_filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads raw data, aggregates it by LOT, and prepares indices for hierarchical models.
        Returns:
            agg_data (pd.DataFrame): Data aggregated by production_lot.
            indices (Dict): Indices and mappings for the model.
        """
        # 1. Load Data
        raw_observed = self.load_observed_data(observed_filename)
        stockpile = self.load_stockpile_data()

        # 2. Aggregate Data by LOT
        # Calculate num_tested and num_failures for each lot in observed data
        agg_data = raw_observed.groupby('lot_id').apply(lambda x: pd.Series({
            'num_tested': len(x),
            'num_failures': (x['result'] == 'Failure').sum()
        })).reset_index()
        
        # Rename 'lot_id' to 'production_lot' to match stockpile data
        agg_data = agg_data.rename(columns={'lot_id': 'production_lot'})
        agg_data['num_success'] = agg_data['num_tested'] - agg_data['num_failures']

        # 3. Create Indices and Mappings
        all_lots = stockpile['production_lot'].tolist()
        all_years = sorted(stockpile['production_year'].unique())
        
        lot_map = {lot: i for i, lot in enumerate(all_lots)}
        year_map = {year: i for i, year in enumerate(all_years)}
        
        # Get indices for observed lots only
        # Filter out lots that might be in observed data but not in stockpile (if any, though unlikely in this context)
        valid_observed_lots = [lot for lot in agg_data['production_lot'] if lot in lot_map]
        observed_lot_idx = [lot_map[lot] for lot in valid_observed_lots]

        # Ensure agg_data order matches observed_lot_idx order if needed, 
        # but here we just need the indices to point to the correct lot in all_lots.
        
        indices = {
            "all_lots": all_lots,
            "all_years": np.array(all_years),
            "year_idx_of_lot": stockpile['production_year'].map(year_map).values,
            "observed_lot_idx": observed_lot_idx,
            "year_of_lot": stockpile['production_year'].values,
            "lot_quantities": stockpile['quantity'].values
        }
        
        return agg_data, indices
