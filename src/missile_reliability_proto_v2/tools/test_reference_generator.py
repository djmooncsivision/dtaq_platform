import pandas as pd
import os

def create_test_spec_csv():
    """
    시험 항목별 기준(Specification)을 정의하고 CSV 파일로 저장합니다.
    """
    # --- 기준값 설명 ---
    # - parameter: 시험 계측 항목 (Pmax, Td)
    # - mean: 규격의 중심값
    # - tolerance: 중심값 기준 허용 오차 (±)
    # - lower_bound: 합격 기준 하한값
    # - upper_bound: 합격 기준 상한값

    spec_data = {
        'parameter': ['Pmax', 'Td'],
        'mean': [800.0, 1.2],
        'tolerance': [200.0, 0.2],
    }
    
    df = pd.DataFrame(spec_data)
    df['lower_bound'] = df['mean'] - df['tolerance']
    df['upper_bound'] = df['mean'] + df['tolerance']

    output_folder = 'data'
    os.makedirs(output_folder, exist_ok=True)
    
    file_path = os.path.join(output_folder, 'test_specifications.csv')
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"'{output_folder}' 폴더에 시험 기준값 파일을 생성했습니다: {file_path}")

if __name__ == '__main__':
    create_test_spec_csv()