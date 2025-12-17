import pandas as pd
import numpy as np
import os
from typing import Dict, List

# --- 상수 정의 ---
DATA_DIR = 'data'
SPEC_FILE = os.path.join(DATA_DIR, 'test_specifications.csv')
OBSERVED_SUMMARY_FILE = os.path.join(DATA_DIR, 'observed_reliability_data.csv')
STOCKPILE_FILE = os.path.join(DATA_DIR, 'stockpile_composition_data.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'observed_detailed_reliability_data.csv')

def generate_measured_data(
    spec_df: pd.DataFrame,
    production_lot: str,
    num_samples: int,
    num_failures: int,
    random_seed: int = None
) -> pd.DataFrame:
    """
    주어진 시험 기준과 조건에 따라 가상의 계측 데이터를 생성합니다.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    df = pd.DataFrame({
        'sample_id': [f"{production_lot}_S{i+1:02d}" for i in range(num_samples)],
        'production_lot': production_lot
    })

    for _, spec_row in spec_df.iterrows():
        param, mean, tol = spec_row['parameter'], spec_row['mean'], spec_row['tolerance']
        lower, upper = spec_row['lower_bound'], spec_row['upper_bound']
        std_dev = tol / 4
        df[param] = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
        df[param] = np.clip(df[param], lower, upper)

    if num_failures > 0 and num_failures <= num_samples:
        failure_indices = np.random.choice(df.index, num_failures, replace=False)
        for idx in failure_indices:
            fail_param = np.random.choice(spec_df['parameter'])
            spec = spec_df[spec_df['parameter'] == fail_param].iloc[0]
            lower, upper, tol = spec['lower_bound'], spec['upper_bound'], spec['tolerance']
            
            if np.random.rand() > 0.5:
                df.loc[idx, fail_param] = upper + np.random.uniform(low=tol*0.1, high=tol*0.3)
            else:
                df.loc[idx, fail_param] = lower - np.random.uniform(low=tol*0.1, high=tol*0.3)

    # 최종 판정 및 데이터 정리
    for _, spec_row in spec_df.iterrows():
        param, lower, upper = spec_row['parameter'], spec_row['lower_bound'], spec_row['upper_bound']
        df[f'{param}_pass'] = df[param].between(lower, upper)
    
    pass_cols = [col for col in df.columns if col.endswith('_pass')]
    df['overall_pass'] = df[pass_cols].all(axis=1)

    df['Pmax'] = df['Pmax'].round(2)
    df['Td'] = df['Td'].round(4)

    return df

def load_and_prepare_input_data() -> pd.DataFrame | None:
    """
    분석에 필요한 모든 입력 CSV 파일을 로드하고 병합하여 반환합니다.
    """
    required_files = {
        "시험 기준": SPEC_FILE,
        "관측 시험 요약": OBSERVED_SUMMARY_FILE,
        "전체 재고 구성": STOCKPILE_FILE
    }
    for name, f_path in required_files.items():
        if not os.path.exists(f_path):
            print(f"에러: '{name}' 파일('{f_path}')을 찾을 수 없습니다.")
            print("데이터 생성기 스크립트(test_reference_generator.py, test_data_generator.py, storage_data_generator.py)를 모두 실행했는지 확인해주세요.")
            return None

    observed_df = pd.read_csv(OBSERVED_SUMMARY_FILE)
    stockpile_df = pd.read_csv(STOCKPILE_FILE)
    
    # 관측 데이터에 생산년도 정보 추가
    observed_with_year_df = pd.merge(observed_df, stockpile_df[['production_lot', 'production_year']], on='production_lot', how='left')
    return observed_with_year_df

def main():
    """메인 실행 함수"""
    # --- 1. 데이터 로드 및 전처리 ---
    observed_scenarios = load_and_prepare_input_data()
    if observed_scenarios is None:
        return

    spec_df = pd.read_csv(SPEC_FILE)

    # --- 2. 상세 계측 데이터 생성 ---
    all_data_frames: List[pd.DataFrame] = []
    print(f"가상 계측 데이터 생성을 시작합니다... (저장 위치: {DATA_DIR})")
    for i, scenario in observed_scenarios.iterrows():
        df = generate_measured_data(
            spec_df=spec_df, 
            production_lot=scenario['production_lot'], 
            num_samples=scenario['num_tested'], 
            num_failures=scenario['num_failures'],
            random_seed=101 + i
        )
        all_data_frames.append(df)

    # --- 3. 데이터 통합 및 저장 ---
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    combined_df['test_year'] = 2025
    
    os.makedirs(DATA_DIR, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    total_tests = len(combined_df)
    total_failures = len(combined_df[combined_df['overall_pass'] == False])
    print(f"-> 통합 데이터 파일 '{OUTPUT_FILE}'이 생성되었습니다.")
    print(f"   (총 시험 수: {total_tests}, 총 실패 수: {total_failures})")

if __name__ == '__main__':
    main()
