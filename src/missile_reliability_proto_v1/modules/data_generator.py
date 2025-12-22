import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self):
        pass

    def generate_date(self, start_year=2015, end_year=2017):
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        delta = end - start
        random_days = random.randrange(delta.days)
        return (start + timedelta(days=random_days)).strftime("%Y.%m.%d")

    def generate(self, num_samples=100, qim_ratio=0.7):
        """
        Generates synthetic reliability data.
        Args:
            num_samples: Number of rows to generate.
            qim_ratio: Ratio of QIM (Initial) data vs ASRP (Storage) data.
        Returns:
            pd.DataFrame: Synthetic data in standard Wide format.
        """
        data = []
        
        for i in range(num_samples):
            # Basic Info
            serial_no = f"SG {random.randint(15, 17)} D{random.randint(10, 999):03d} 00"
            part_no = "81040050"
            test_date = self.generate_date()
            
            # Operation Month
            # QIM (0 months) vs ASRP (110-130 months)
            if random.random() < qim_ratio:
                op_month = 0
            else:
                op_month = random.randint(110, 130)
                
            # Measurements (Simulating some items)
            # Ranges based on Sample Data
            row_dict = {
                '일련번호': serial_no,
                '품번': part_no,
                '시험일자': test_date,
                '운용월': op_month
            }
            
            # Item 1 (Voltage?) ~ 2.2
            row_dict['1'] = round(random.uniform(2.1, 2.3), 2)
            # Item 2 ~ 2.05
            row_dict['2'] = round(random.uniform(2.0, 2.1), 2)
            # Item 6 ~ 2700
            row_dict['6'] = random.randint(2400, 3000)
            
            # Fill others with random or fixed values if not specified
            for j in range(3, 28):
                if str(j) not in row_dict:
                    if j in [3, 6, 7]: # Large integers
                         row_dict[str(j)] = random.randint(1000, 6000)
                    else:
                         row_dict[str(j)] = round(random.uniform(0, 50), 2)
            
            # Specific logic for trend simulation (decay over time)
            # If op_month is high, degrade Item 1 slightly
            if op_month > 0:
                row_dict['1'] -= (op_month * 0.001) # Decay
            
            # Result
            row_dict['합격여부'] = "합격" if random.random() > 0.05 else "불하격"
            
            data.append(row_dict)
            
        columns = ["일련번호", "품번", "시험일자", "운용월"] + [str(i) for i in range(1, 28)] + ["합격여부"]
        df = pd.DataFrame(data)
        
        # Ensure column order
        return df[columns]
