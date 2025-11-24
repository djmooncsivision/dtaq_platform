import pandas as pd
import numpy as np
import os
import argparse # 1. argparse 라이브러리 추가

def generate_controlled_failure_data(
    # ... 함수 내용은 이전과 동일 ...
    num_samples: int,
    num_failures: int,
    production_year: int = 2015,
    test_year: int = 2025,
    pressure_spec: tuple = (800, 200),
    delay_spec: tuple = (1.2, 0.2),
    random_seed: int = None
) -> pd.DataFrame:
    if num_failures > num_samples:
        raise ValueError("실패 개수는 총 샘플 개수보다 많을 수 없습니다.")
    if random_seed is not None:
        np.random.seed(random_seed)
    available_lots = [f"{production_year}-LOT-{i:02d}" for i in range(1, 7)]
    sampled_lots = np.random.choice(available_lots, num_samples, replace=True)
    data = {'sample_id': [f"SAMPLE_{i:03d}" for i in range(1, num_samples + 1)],'production_lot': sampled_lots,'production_year': production_year,'test_year': test_year,'age_at_test': test_year - production_year,}
    df = pd.DataFrame(data)
    pressure_mean, pressure_tol = pressure_spec
    pressure_std = pressure_tol / 4
    df['max_pressure'] = np.random.normal(loc=pressure_mean, scale=pressure_std, size=num_samples)
    delay_mean, delay_tol = delay_spec
    delay_std = delay_tol / 4
    df['delay_time'] = np.random.normal(loc=delay_mean, scale=delay_std, size=num_samples)
    if num_failures > 0:
        failure_indices = np.random.choice(df.index, num_failures, replace=False)
        for idx in failure_indices:
            fail_pressure = np.random.rand() > 0.5
            if fail_pressure:
                exceeds_upper = np.random.rand() > 0.5
                df.loc[idx, 'max_pressure'] = (pressure_mean + pressure_tol + np.random.uniform(5, 20)) if exceeds_upper else (pressure_mean - pressure_tol - np.random.uniform(5, 20))
            else:
                exceeds_upper = np.random.rand() > 0.5
                df.loc[idx, 'delay_time'] = (delay_mean + delay_tol + np.random.uniform(0.01, 0.05)) if exceeds_upper else (delay_mean - delay_tol - np.random.uniform(0.01, 0.05))
    df['max_pressure'] = df['max_pressure'].round(2)
    df['delay_time'] = df['delay_time'].round(4)
    df['pressure_pass'] = df['max_pressure'].between(pressure_mean - pressure_tol, pressure_mean + pressure_tol)
    df['delay_pass'] = df['delay_time'].between(delay_mean - delay_tol, delay_mean + delay_tol)
    df['overall_pass'] = df['pressure_pass'] & df['delay_pass']
    return df

def main():
    # 2. 실행 인자(argument)를 처리하는 부분 추가
    parser = argparse.ArgumentParser(description="가상 신뢰성 테스트 데이터셋을 생성합니다.")
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=os.path.join('data', 'raw'), 
        help="CSV 파일을 저장할 디렉토리 경로 (기본값: data/raw)"
    )
    args = parser.parse_args()

    print(f"가상 데이터셋 생성을 시작합니다... (저장 위치: {args.output_dir})")
    
    # 3. 하드코딩된 경로 대신, 입력받은 인자(args.output_dir)를 사용
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    scenarios_to_generate = {
        "0_failures": {"num_failures": 0, "seed": 101},
        "1_failure": {"num_failures": 1, "seed": 102},
        "2_failures": {"num_failures": 2, "seed": 103},
    }

    for name, params in scenarios_to_generate.items():
        df = generate_controlled_failure_data(
            num_samples=11,
            num_failures=params["num_failures"],
            random_seed=params["seed"]
        )
        file_name = f'sample_data_11_samples_{name}.csv'
        file_path = os.path.join(output_dir, file_name)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"-> '{file_path}' 경로에 데이터가 성공적으로 저장되었습니다.")
    
    print("모든 데이터셋 생성이 완료되었습니다.")

if __name__ == "__main__":
    main()
