# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_one_sided_lower_bound(
    num_samples: int, 
    num_failures: int, 
    confidence_level: float = 0.90
) -> dict:
    """
    Clopper-Pearson 방법을 사용하여 신뢰도의 단측 하한(Lower Confidence Bound)을 계산합니다.

    Args:
        num_samples (int): 총 샘플 수 (n).
        num_failures (int): 실패 수 (f).
        confidence_level (float): 계산하고자 하는 신뢰수준 (기본값: 0.90).

    Returns:
        dict: 분석 결과를 담은 딕셔너리.
    """
    if num_samples == 0:
        return {
            "technique": "Clopper-Pearson (One-sided)",
            "num_samples": 0,
            "num_failures": 0,
            "confidence_level": confidence_level,
            "reliability_point_estimate": 0,
            "lower_confidence_bound": 0,
        }

    num_successes = num_samples - num_failures
    alpha = 1 - confidence_level

    # 신뢰도 점 추정치 (Point Estimate of Reliability)
    reliability_point_estimate = num_successes / num_samples

    # 신뢰도 하한 (Lower Confidence Bound)
    # 실패가 0개일 때와 아닐 때를 구분하여 계산
    if num_successes == num_samples:
        lower_bound = alpha**(1 / num_samples)
    else:
        lower_bound = beta.ppf(alpha, num_successes, num_failures + 1)

    return {
        "technique": "Clopper-Pearson (One-sided)",
        "num_samples": num_samples,
        "num_failures": num_failures,
        "confidence_level": confidence_level,
        "reliability_point_estimate": reliability_point_estimate,
        "lower_confidence_bound": lower_bound,
    }

def analyze_and_present_results(results_df: pd.DataFrame):
    """
    분석 결과를 출력하고 시각화합니다.
    """
    print("--- 신뢰도 단측 하한 분석 결과 (90% 신뢰수준) ---")
    
    # 분석 결과 텍스트로 출력
    for index, row in results_df.iterrows():
        print(f"\n[시나리오: {row['scenario']}]")
        print(f"  - 총 샘플 수: {int(row['num_samples'])}, 총 실패 수: {int(row['num_failures'])}")
        print(f"  - 신뢰도 점 추정치: {row['reliability_point_estimate']:.3f} ({(row['reliability_point_estimate'])*100:.1f}%)")
        print(f"  - 90% 신뢰도 하한: {row['lower_confidence_bound']:.3f} ({(row['lower_confidence_bound'])*100:.1f}%)")
        print(f"    -> 해석: '전체 비축물량의 신뢰도는 90%의 신뢰수준으로 최소 {row['lower_confidence_bound']*100:.1f}% 이상이다.'")

    # 결과 시각화
    # 시각화 스타일 설정
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = results_df.index

    bars1 = ax.bar(index - bar_width/2, results_df['reliability_point_estimate'], bar_width, 
                   label='신뢰도 점 추정치 (R)', color='skyblue')

    bars2 = ax.bar(index + bar_width/2, results_df['lower_confidence_bound'], bar_width, 
                   label='90% 신뢰도 하한 (LCB)', color='salmon')

    ax.set_xlabel('분석 시나리오', fontsize=12)
    ax.set_ylabel('신뢰도', fontsize=12)
    ax.set_title('시나리오별 신뢰도 추정치 비교 (Clopper-Pearson)', fontsize=15, pad=20)
    ax.set_xticks(index)
    ax.set_xticklabels(results_df['scenario'], rotation=0)
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', 
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def main():
    """
    데이터 파일을 읽어 Clopper-Pearson 분석을 수행하고 결과를 출력/시각화합니다.
    """
    # 분석할 데이터 파일 목록
    data_files = [
        'data/observed_reliability_data_0_failures.csv',
        'data/observed_reliability_data_1_failure.csv'
    ]
    
    results = []
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"경고: 파일이 존재하지 않습니다 - {file_path}")
            continue
            
        # 데이터 로드
        observed_data = pd.read_csv(file_path)
        
        # 전체 샘플 수와 실패 수 집계
        total_samples = len(observed_data)
        total_failures = (observed_data['result'] == 'Failure').sum()
        
        # 시나리오 이름 설정
        scenario_name = f"시험 데이터 ({total_failures}개 실패)"
        
        # Clopper-Pearson 분석 실행
        result = calculate_one_sided_lower_bound(
            num_samples=total_samples,
            num_failures=total_failures
        )
        result['scenario'] = scenario_name
        results.append(result)

    if not results:
        print("분석할 데이터가 없습니다.")
        return

    # 결과를 Pandas DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 결과 출력 및 시각화
    analyze_and_present_results(results_df)

if __name__ == "__main__":
    main()