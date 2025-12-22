import pandas as pd
import numpy as np
import os
import random

def generate_individual_test_data(production_year: int, test_year: int, n_samples: int):
    """
    'stockpile_composition_data.csv'를 참조하여 특정 연도에 생산된 로트에서
    추출한 n개의 샘플에 대한 시험 데이터를 생성합니다.
    '0개 실패'와 '1개 실패' 두 가지 시나리오의 CSV 파일을 개별 샘플 단위로 생성합니다.

    Args:
        production_year (int): 시험 대상 유도탄의 생산 연도.
        test_year (int): 시험이 수행된 연도.
        n_samples (int): 시험할 샘플의 수.
    """
    # --- 1. 설정 및 폴더 생성 ---
    output_folder = 'data'
    stockpile_data_path = os.path.join(output_folder, 'stockpile_composition_data.csv')
    os.makedirs(output_folder, exist_ok=True)

    # --- 2. 비축 재고 데이터 로드 및 필터링 ---
    try:
        stockpile_df = pd.read_csv(stockpile_data_path)
    except FileNotFoundError:
        print(f"오류: '{stockpile_data_path}' 파일을 찾을 수 없습니다.")
        print("먼저 storage_data_generator.py를 실행하여 비축 재고 구성 파일을 생성해야 합니다.")
        return

    # 지정된 생산 연도의 로트 정보 필터링
    year_lots = stockpile_df[stockpile_df['production_year'] == production_year]
    
    if year_lots.empty:
        print(f"오류: {production_year}년도에 해당하는 로트 정보가 '{stockpile_data_path}'에 없습니다.")
        return
        
    available_lots = year_lots['production_lot'].tolist()

    # --- 3. 시험 샘플 데이터 생성 ---
    # n개의 샘플에 대해 사용 가능한 로트를 무작위로 할당
    assigned_lots = random.choices(available_lots, k=n_samples)
    
    # 기본 데이터프레임 생성
    test_data = {
        'test_id': [f"TEST-{i:03d}" for i in range(1, n_samples + 1)],
        'test_date': f"{test_year}-01-15", # 시험일자는 임의로 지정
        'production_year': production_year,
        'lot_id': assigned_lots
    }
    base_df = pd.DataFrame(test_data)

    # --- 4. 시나리오별 데이터 생성 및 저장 ---

    # 시나리오 1: 0개 실패 (전부 성공)
    df_scenario_1 = base_df.copy()
    df_scenario_1['result'] = 'Success'
    
    output_path_1 = os.path.join(output_folder, 'observed_reliability_data_0_failures.csv')
    df_scenario_1.to_csv(output_path_1, index=False, encoding='utf-8-sig')
    print(f"'{output_folder}' 폴더에 0개 실패 시나리오 파일을 생성했습니다: {os.path.basename(output_path_1)}")
    print(f"  - 총 샘플 수: {n_samples}, 실패: 0")

    # 시나리오 2: 1개 실패
    df_scenario_2 = base_df.copy()
    results = ['Success'] * (n_samples - 1) + ['Failure']
    random.shuffle(results) # 실패 위치를 무작위로 섞음
    df_scenario_2['result'] = results

    output_path_2 = os.path.join(output_folder, 'observed_reliability_data_1_failure.csv')
    df_scenario_2.to_csv(output_path_2, index=False, encoding='utf-8-sig')
    print(f"'{output_folder}' 폴더에 1개 실패 시나리오 파일을 생성했습니다: {os.path.basename(output_path_2)}")
    print(f"  - 총 샘플 수: {n_samples}, 실패: 1")


if __name__ == '__main__':
    # 2015년 생산된 유도탄 중 11개를 2025년에 시험하는 시나리오
    generate_individual_test_data(
        production_year=2015, 
        test_year=2025, 
        n_samples=11
    )
