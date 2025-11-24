
import pandas as pd
import numpy as np
import os

def generate_detailed_results(reliability_data_path, specs_path, output_path):
    """
    Generates detailed test results by adding simulated measurement values to reliability data.

    Args:
        reliability_data_path (str): Path to the observed reliability data CSV file.
        specs_path (str): Path to the test specifications CSV file.
        output_path (str): Path to save the output detailed results CSV file.
    """
    # 데이터 폴더 확인 및 생성
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 데이터 읽기
    try:
        reliability_df = pd.read_csv(reliability_data_path)
        specs_df = pd.read_csv(specs_path)
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
        return

    # 상세 결과 데이터프레임 초기화
    detailed_results_df = reliability_df.copy()

    # 각 계측 항목에 대해 가상 데이터 생성
    for _, spec_row in specs_df.iterrows():
        param = spec_row['parameter']
        lower_bound = spec_row['lower_bound']
        upper_bound = spec_row['upper_bound']
        
        # 성공 케이스에 대한 랜덤 데이터 생성 (규격 내)
        detailed_results_df[param] = np.random.uniform(
            low=lower_bound,
            high=upper_bound,
            size=len(detailed_results_df)
        ).round(4)

    # 실패 케이스 처리
    failure_indices = detailed_results_df[detailed_results_df['result'] == 'Failure'].index
    if not failure_indices.empty:
        # 실패한 샘플에 대해
        for idx in failure_indices:
            # 실패를 유발할 계측 항목을 무작위로 선택
            failing_param_row = specs_df.sample(n=1).iloc[0]
            param_to_fail = failing_param_row['parameter']
            lower_bound = failing_param_row['lower_bound']
            upper_bound = failing_param_row['upper_bound']
            
            # 규격을 벗어나는 값 생성 (50% 확률로 상한 또는 하한 초과)
            if np.random.rand() > 0.5:
                # 상한 초과
                failure_value = upper_bound + np.random.uniform(0.1, 0.5) * (upper_bound - lower_bound)
            else:
                # 하한 미달
                failure_value = lower_bound - np.random.uniform(0.1, 0.5) * (upper_bound - lower_bound)
            
            detailed_results_df.loc[idx, param_to_fail] = round(failure_value, 4)

    # CSV 파일로 저장
    detailed_results_df.to_csv(output_path, index=False)
    print(f"Successfully generated detailed test results: {output_path}")


if __name__ == "__main__":
    # 기본 경로 설정
    DATA_DIR = "data"
    SPECS_FILE = os.path.join(DATA_DIR, "test_specifications.csv")

    # 0개 실패 시나리오
    generate_detailed_results(
        reliability_data_path=os.path.join(DATA_DIR, "observed_reliability_data_0_failures.csv"),
        specs_path=SPECS_FILE,
        output_path=os.path.join(DATA_DIR, "detailed_test_results_0_failure.csv")
    )

    # 1개 실패 시나리오
    generate_detailed_results(
        reliability_data_path=os.path.join(DATA_DIR, "observed_reliability_data_1_failure.csv"),
        specs_path=SPECS_FILE,
        output_path=os.path.join(DATA_DIR, "detailed_test_results_1_failure.csv")
    )
