import pandas as pd
import numpy as np
import os

def create_stockpile_data_csv(missiles_per_lot: int = 400):
    """
    전체 비축 물량(모집단)의 구성 정보를 생성하고 CSV 파일로 저장합니다.
    이 데이터는 분석의 전체 대상을 정의합니다.
    """
    # --- 데이터 설명 ---
    # - production_lot: 전체 비축 물량을 구성하는 각 생산 로트의 식별자.
    # - production_year: 해당 로트가 생산된 연도.
    # - quantity: 해당 로트에 포함된 유도탄의 수량.

    # 1. 생산년도 정의 (2015년부터 2019년까지)
    all_years = np.arange(2015, 2020)

    # 2. 연간 로트 수 정의 (매년 6개 로트)
    lots_per_year = 6

    # 3. 전체 로트 리스트 생성
    all_lots = [f"{year}-LOT-{i:02d}" for year in all_years for i in range(1, lots_per_year + 1)]
    
    # 4. DataFrame 생성
    stockpile_data = {
        'production_lot': all_lots,
        'production_year': [int(lot.split('-')[0]) for lot in all_lots],
        'quantity': [missiles_per_lot] * len(all_lots)
    }
    df = pd.DataFrame(stockpile_data)

    # 5. CSV 파일로 저장
    output_folder = 'data'
    os.makedirs(output_folder, exist_ok=True)
    
    file_path = os.path.join(output_folder, 'stockpile_composition_data.csv')
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"'{output_folder}' 폴더에 전체 비축 물량 구성 파일을 생성했습니다: {file_path}")
    print(f"  - 총 로트 수: {len(df)}")
    print(f"  - 로트당 유도탄 수: {missiles_per_lot}")
    print(f"  - 총 유도탄 수: {len(df) * missiles_per_lot}")

if __name__ == '__main__':
    # 로트당 유도탄 수를 400개로 설정하여 데이터 생성
    create_stockpile_data_csv(missiles_per_lot=400)