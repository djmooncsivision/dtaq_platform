import pandas as pd
from typing import List, Dict, Any

class CSVWriter:
    def __init__(self, output_path: str):
        """
        Initialize CSVWriter.
        Args:
            output_path: Path to save the CSV file.
        """
        self.output_path = output_path

    def save_data(self, data: List[Dict[str, Any]]):
        """
        Save list of dictionaries to CSV.
        Args:
            data: List of data entries.
        """
        if not data:
            print("No data to save.")
            return

        df = pd.DataFrame(data)
        
        # Ensure columns are in a good order
        # Updated columns based on reference output
        cols = [
            "Page", 
            "index",            # Section Header (e.g., 정렬 점검)
            "index_detail",     # Item Name (e.g., 보조채널조종)
            "Reference_Tolerance", 
            "Measured_Value", 
            "Result", 
            "Raw_Row"
        ]
        # Filter cols that exist in df
        cols = [c for c in cols if c in df.columns]
        # Add any other columns
        remaining_cols = [c for c in df.columns if c not in cols]
        cols.extend(remaining_cols)
        
        df = df[cols]
        
        try:
            df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            print(f"Successfully saved {len(df)} rows to {self.output_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    # Test
    pass
