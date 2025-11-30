import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_date(start_year=2015, end_year=2017):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randrange(delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y.%m.%d")

def generate_data(num_rows=100):
    data = []
    
    for i in range(num_rows):
        # Basic Info
        serial_no = f"SG {random.randint(15, 17)} D{random.randint(10, 999):03d} 00"
        part_no = "81040050"
        test_date = generate_date()
        
        # Operation Month (Mix of QIM=70% and ASRP=30%)
        if random.random() < 0.7: # 70% QIM
            op_month = 0
        else:
            op_month = random.randint(110, 130)
            
        # Measurements (Approximate ranges based on sample)
        col1 = round(random.uniform(2.1, 2.3), 2)
        col2 = round(random.uniform(2.0, 2.1), 2)
        col3 = random.randint(1700, 2400)
        col4 = random.randint(580, 600)
        col5 = random.randint(550, 590)
        col6 = random.randint(2400, 3000)
        col7 = random.randint(5400, 6000)
        
        # Cols 8-18 seem to be 0 in sample, keeping them 0 for now
        cols_8_18 = [0] * 11
        
        col19 = round(random.uniform(14.8, 15.2), 2)
        col20 = round(random.uniform(-15.5, -14.5), 2)
        col21 = round(random.uniform(9.4, 10.2), 2)
        col22 = round(random.uniform(0.1, 0.3), 2)
        col23 = round(random.uniform(20.0, 22.5), 2)
        col24 = round(random.uniform(0.6, 1.1), 2)
        col25 = round(random.uniform(9.3, 10.3), 2)
        col26 = round(random.uniform(0.15, 0.35), 2)
        col27 = round(random.uniform(21.0, 24.5), 2)
        
        # Result
        result = "합격" if random.random() > 0.05 else "불합격"
        
        row = [serial_no, part_no, test_date, op_month, col1, col2, col3, col4, col5, col6, col7] + \
              cols_8_18 + \
              [col19, col20, col21, col22, col23, col24, col25, col26, col27, result]
        
        data.append(row)
        
    columns = ["일련번호", "품번", "시험일자", "운용월"] + [str(i) for i in range(1, 28)] + ["합격여부"]
    df = pd.DataFrame(data, columns=columns)
    
    return df

if __name__ == "__main__":
    df = generate_data(150) # Generate 150 rows
    # Save with BOM for Excel compatibility
    df.to_csv("c:/Antigravity/missile_reliability_proto_v1/sample_reliability_data.csv", index=False, encoding="utf-8-sig")
    print("Generated 150 rows of synthetic data.")
