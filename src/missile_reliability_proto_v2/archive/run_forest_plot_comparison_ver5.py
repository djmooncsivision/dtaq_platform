import pytensor
pytensor.config.cxx = ""

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import os

# Constants
CURRENT_YEAR = 2025
# 시나리오별 색상 지정
SCENARIO_COLORS = {
    "낙관적 (Optimistic)": "cornflowerblue",
    "보수적 (Pessimistic)": "salmon",
    "이항분포 모델": "darkorange", # 비교용 색상 추가
    "초기하분포 모델": "darkcyan" # 비교용 색상 추가
}

# 데이터 파일 경로 상수
DATA_DIR = 'data'
OBSERVED_DATA_FILE = os.path.join(DATA_DIR, 'observed_reliability_data.csv')
STOCKPILE_DATA_FILE = os.path.join(DATA_DIR, 'stockpile_composition_data.csv')

def prepare_data_and_indices(observed_data_file: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """CSV 파일에서 데이터를 로드하고 분석에 필요한 인덱스를 준비합니다."""
    # 1. 데이터 파일 로드
    if not os.path.exists(observed_data_file) or not os.path.exists(STOCKPILE_DATA_FILE):
        raise FileNotFoundError(
            f"데이터 파일({observed_data_file} 또는 {STOCKPILE_DATA_FILE})을 찾을 수 없습니다. "
            "test_data_generator_ver2.py와 storage_data_generator.py를 먼저 실행해주세요."
        )
    observed_data = pd.read_csv(observed_data_file)
    stockpile_data = pd.read_csv(STOCKPILE_DATA_FILE)

    observed_data['num_success'] = observed_data['num_tested'] - observed_data['num_failures']

    # 2. 모집단 및 인덱스 정보 생성
    all_lots = stockpile_data['production_lot'].tolist()
    all_years = sorted(stockpile_data['production_year'].unique())
    
    lot_map = {lot: i for i, lot in enumerate(all_lots)}
    year_map = {year: i for i, year in enumerate(all_years)}
    
    year_of_lot = stockpile_data['production_year'].values
    year_idx_of_lot = stockpile_data['production_year'].map(year_map).values
    observed_lot_idx = [lot_map[lot] for lot in observed_data['production_lot']]
    
    # LOT별 수량 정보 추가
    lot_quantities = stockpile_data['quantity'].values

    indices = {
        "all_lots": all_lots,
        "all_years": np.array(all_years),
        "year_idx_of_lot": year_idx_of_lot,
        "observed_lot_idx": observed_lot_idx,
        "year_of_lot": year_of_lot,
        "lot_quantities": lot_quantities # 수량 정보 추가
    }
    return observed_data, indices

def run_binomial_model(data: pd.DataFrame, indices: Dict[str, Any], model_params: Dict[str, Any]) -> az.InferenceData:
    """[비교용] 이항분포를 사용하여 베이지안 모델을 실행합니다."""
    with pm.Model() as model:
        mu_global_logit = pm.Normal('mu_global_logit', mu=3.89, sigma=0.5)
        sigma_year = pm.HalfNormal('sigma_year', sigma=model_params["inter_year_sigma"])
        sigma_lot_base = pm.HalfNormal('sigma_lot_base', sigma=model_params["intra_lot_sigma"])

        if model_params["degradation_effect_on_variance"]:
            variance_degradation_rate = pm.HalfNormal('variance_degradation_rate', sigma=0.05)
            age_of_lot = CURRENT_YEAR - indices["year_of_lot"]
            sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base + age_of_lot * variance_degradation_rate)
        else:
            sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base)

        theta_year = pm.Normal('theta_year', mu=mu_global_logit, sigma=sigma_year, shape=len(indices["all_years"]))
        theta_lot = pm.Normal('theta_lot', mu=theta_year[indices["year_idx_of_lot"]], sigma=sigma_lot_effective, shape=len(indices["all_lots"]))
        reliability_lot = pm.Deterministic('reliability_lot', pm.invlogit(theta_lot))

        y_obs = pm.Binomial('y_obs', n=data['num_tested'].values, p=reliability_lot[indices["observed_lot_idx"]], observed=data['num_success'].values)
        
        return pm.sample(2000, tune=1500, cores=1, return_inferencedata=True, random_seed=2024, progressbar=True)

def run_hypergeometric_model(data: pd.DataFrame, indices: Dict[str, Any], model_params: Dict[str, Any]) -> az.InferenceData:
    """초기하분포를 사용하여 베이지안 모델을 실행하고 추론 결과를 반환합니다."""
    with pm.Model() as model:
        mu_global_logit = pm.Normal('mu_global_logit', mu=3.89, sigma=0.5)
        sigma_year = pm.HalfNormal('sigma_year', sigma=model_params["inter_year_sigma"])
        sigma_lot_base = pm.HalfNormal('sigma_lot_base', sigma=model_params["intra_lot_sigma"])

        if model_params["degradation_effect_on_variance"]:
            variance_degradation_rate = pm.HalfNormal('variance_degradation_rate', sigma=0.05)
            age_of_lot = CURRENT_YEAR - indices["year_of_lot"]
            sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base + age_of_lot * variance_degradation_rate)
        else:
            sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base)

        theta_year = pm.Normal('theta_year', mu=mu_global_logit, sigma=sigma_year, shape=len(indices["all_years"]))
        theta_lot = pm.Normal('theta_lot', mu=theta_year[indices["year_idx_of_lot"]], sigma=sigma_lot_effective, shape=len(indices["all_lots"]))
        
        # --- 모델 수정: Binomial -> HyperGeometric ---
        # 1. 각 LOT의 잠재적 신뢰도(p)를 추정 (기존과 동일)
        p_lot = pm.invlogit(theta_lot)
        
        # 2. 각 LOT의 전체 수량(N)에서 예상되는 성공 개수(k)를 추정
        # k는 정수여야 하므로, p_lot * N을 중심으로 하는 이항분포로 k를 모델링
        lot_quantities = indices["lot_quantities"]
        k_lot = pm.Binomial('k_lot', n=lot_quantities, p=p_lot, shape=len(indices["all_lots"]))

        # 3. 관측 데이터(y_obs)를 초기하분포와 연결
        # N: 시험한 LOT의 전체 수량, k: 해당 LOT의 추정된 성공 개수, n: 시험한 샘플 수
        y_obs = pm.HyperGeometric('y_obs', N=lot_quantities[indices["observed_lot_idx"]], 
                                  k=k_lot[indices["observed_lot_idx"]], 
                                  n=data['num_tested'].values, 
                                  observed=data['num_success'].values)

        # 4. 최종적으로 LOT별 신뢰도를 Deterministic으로 계산하여 저장
        reliability_lot = pm.Deterministic('reliability_lot', k_lot / lot_quantities)
        
        trace = pm.sample(2000, tune=1500, cores=1, return_inferencedata=True, random_seed=2024, progressbar=True)
    return trace

def plot_stockpile_reliability_comparison(traces: Dict[str, az.InferenceData], indices: Dict[str, Any], case_name: str):
    """
    여러 시나리오의 전체 비축물량 평균 신뢰도 분포를 비교하는 밀도 그림을 생성합니다.
    (LOT별 수량을 반영한 가중 평균 사용)
    """
    plt.figure(figsize=(12, 8))
    
    print("\n--- Overall Stockpile Reliability Analysis (Weighted by Quantity) ---")
    # 가중치로 사용할 LOT별 수량
    weights = indices["lot_quantities"]

    for name, trace in traces.items():
        posterior_samples = trace.posterior['reliability_lot'].values
        # 각 MCMC 샘플(draw)에 대해 LOT별 수량으로 가중 평균 계산
        weighted_mean_per_draw = np.average(posterior_samples, axis=-1, weights=weights).flatten()
        
        sns.kdeplot(
            weighted_mean_per_draw, 
            label=f'시나리오: {name}', 
            fill=True, 
            alpha=0.6, 
            color=SCENARIO_COLORS.get(name)
        )
        
        hdi_lower_bound = az.hdi(weighted_mean_per_draw, hdi_prob=0.9)[0]
        print(f"[{name} 시나리오] 90% 신뢰수준에서 전체 평균 신뢰도는 최소 {hdi_lower_bound:.4f} 이상일 것으로 추정됩니다.")
        hdi_bounds = az.hdi(weighted_mean_per_draw, hdi_prob=0.9)
        plt.axvspan(hdi_bounds[0], hdi_bounds[1], color=SCENARIO_COLORS.get(name), alpha=0.1)
        print(f"[{name} 시나리오] 90% 신뢰구간(HDI): [{hdi_bounds[0]:.4f}, {hdi_bounds[1]:.4f}]")
    
    plt.title(f'[{case_name}] 시나리오별 전체 비축물량 가중 평균 신뢰도 분포 비교', fontsize=20, pad=20)
    plt.axvline(x=0.98, color='grey', linestyle='--', linewidth=1.5, label='목표 신뢰도 (98%)')
    plt.text(0.98, plt.ylim()[1]*0.9, ' 목표 신뢰도', color='dimgray', fontsize=12, ha='left')
    plt.xlabel('전체 가중 평균 신뢰도', fontsize=15)
    plt.ylabel('확률 밀도', fontsize=15)
    plt.legend(title='시나리오', fontsize=13, title_fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

def plot_posterior_distributions(traces: Dict[str, az.InferenceData], case_name: str):
    """모델의 주요 파라미터에 대한 사후 분포를 시각화합니다."""
    # 시각화할 파라미터와 그림 제목을 딕셔너리로 정의
    params_to_plot = {
        'mu_global_logit': '전체 평균 신뢰도 (로짓 스케일)',
        'sigma_year': '연도 간 편차',
        'sigma_lot_base': 'LOT 간 기본 편차',
        'variance_degradation_rate': '성능 저하율 (편차 증가 속도)'
    }
    
    for param, title in params_to_plot.items():
        # 각 파라미터에 대해 새로운 Figure를 생성
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        has_plot = False
        for name, trace in traces.items():
            if param in trace.posterior:
                has_plot = True
                samples = trace.posterior[param].values.flatten()
                sns.kdeplot(
                    samples,
                    label=name,
                    color=SCENARIO_COLORS.get(name),
                    fill=True,
                    alpha=0.6,
                    ax=ax
                )
        
        if not has_plot:
            plt.close() # 데이터가 없는 파라미터(예: 낙관적 시나리오의 variance_degradation_rate)는 플롯을 닫음
            continue

        ax.set_title(f'[{case_name}] 사후 분포 비교: {title}', fontsize=20, pad=20)
        ax.set_xlabel('파라미터 값', fontsize=15)
        ax.set_ylabel('확률 밀도', fontsize=15)
        ax.legend(title='시나리오', fontsize=13, title_fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

def plot_lot_reliability_forest(traces: Dict[str, az.InferenceData], indices: Dict[str, Any], case_name: str):
    """시나리오별로 각 LOT의 신뢰도를 하나의 Forest Plot으로 비교하여 시각화합니다."""
    plt.figure(figsize=(12, 16))
    
    y_positions = np.arange(len(indices["all_lots"]))
    
    # 각 시나리오의 결과를 순회하며 플롯에 추가
    for i, (name, trace) in enumerate(traces.items()):
        # 데이터 추출
        posterior_data = trace.posterior['reliability_lot']
        hdi_data = az.hdi(posterior_data, hdi_prob=0.9)['reliability_lot'].values
        mean_data = posterior_data.mean(dim=['chain', 'draw']).values
        
        # y축 위치 조정을 위한 오프셋
        y_offset = -0.15 + i * 0.3
        
        # 90% HDI 에러바 (가로 방향)
        plt.errorbar(x=mean_data, y=y_positions + y_offset, 
                     xerr=[mean_data - hdi_data[:, 0], hdi_data[:, 1] - mean_data],
                     fmt='none', elinewidth=1.5, capsize=4, 
                     color=SCENARIO_COLORS.get(name), alpha=0.7)
        
        # 평균 추정치 점
        plt.plot(mean_data, y_positions + y_offset, 'o', markersize=6, color=SCENARIO_COLORS.get(name), label=name)

    plt.yticks(ticks=y_positions, labels=indices["all_lots"], fontsize=10)
    plt.title(f'[{case_name}] 시나리오별 LOT 신뢰도 추정 비교 (90% HDI)', fontsize=20, pad=20)
    plt.xlabel('추정 신뢰도', fontsize=15)
    plt.ylabel('생산 LOT', fontsize=15)
    plt.xlim(0.9, 1.0) # x축 범위를 조정하여 가독성 향상
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(title='시나리오', fontsize=13, title_fontsize=14)
    plt.tight_layout()

def plot_model_comparison_forest(traces_to_compare: Dict[str, az.InferenceData], indices: Dict[str, Any], case_name: str):
    """
    이항분포 모델과 초기하분포 모델의 LOT별 신뢰도 추정 결과를 비교하는 Forest Plot을 생성합니다.
    """
    plt.figure(figsize=(12, 16))
    
    y_positions = np.arange(len(indices["all_lots"]))
    
    # 각 모델의 결과를 순회하며 플롯에 추가
    for i, (name, trace) in enumerate(traces_to_compare.items()):
        # 데이터 추출
        posterior_data = trace.posterior['reliability_lot']
        hdi_data = az.hdi(posterior_data, hdi_prob=0.9)['reliability_lot'].values
        mean_data = posterior_data.mean(dim=['chain', 'draw']).values
        
        # y축 위치 조정을 위한 오프셋
        y_offset = -0.15 + i * 0.3
        
        # 90% HDI 에러바 (가로 방향)
        plt.errorbar(x=mean_data, y=y_positions + y_offset, 
                     xerr=[mean_data - hdi_data[:, 0], hdi_data[:, 1] - mean_data],
                     fmt='none', elinewidth=1.5, capsize=4, 
                     color=SCENARIO_COLORS.get(name), alpha=0.7)
        
        # 평균 추정치 점
        plt.plot(mean_data, y_positions + y_offset, 'o', markersize=6, color=SCENARIO_COLORS.get(name), label=name)

    plt.yticks(ticks=y_positions, labels=indices["all_lots"], fontsize=10)
    plt.title(f'[{case_name}] 모델별 LOT 신뢰도 추정 비교 (보수적 시나리오, 90% HDI)', fontsize=20, pad=20)
    plt.xlabel('추정 신뢰도', fontsize=15)
    plt.ylabel('생산 LOT', fontsize=15)
    plt.xlim(0.9, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(title='모델', fontsize=13, title_fontsize=14)
    plt.tight_layout()

def main():
    """메인 실행 함수"""
    # 시각화를 위한 전역 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
    except:
        plt.rcParams['font.family'] = 'AppleGothic' # Mac
    plt.rcParams['axes.unicode_minus'] = False

    # --- 분석할 시험 데이터 케이스 정의 ---
    test_cases = {
        "시험 1 (0개 실패)": os.path.join(DATA_DIR, 'observed_reliability_data_0_failures.csv'),
        "시험 2 (1개 실패)": os.path.join(DATA_DIR, 'observed_reliability_data_1_failure.csv')
    }

    for case_name, data_file in test_cases.items():
        print(f"\n{'='*20} 분석 시작: {case_name} {'='*20}")
        
        data, indices = prepare_data_and_indices(data_file)

        scenarios = {
            "낙관적 (Optimistic)": {"inter_year_sigma": 0.01, "intra_lot_sigma": 0.02, "degradation_effect_on_variance": False},
            "보수적 (Pessimistic)": {"inter_year_sigma": 0.2, "intra_lot_sigma": 0.1, "degradation_effect_on_variance": True}
        }
        traces = {}

        for name, params in scenarios.items():
            print(f"\n--- Running analysis for scenario: [{name}] ---")
            traces[name] = run_hypergeometric_model(data, indices, params)
            print(f"--- Scenario analysis complete: [{name}] ---")
        
        # --- 분석 결과 시각화 ---
        print(f"\n--- '{case_name}'에 대한 플롯 생성 중 ---")
        # 1. 시나리오별 전체 평균 신뢰도 비교 플롯 (가중 평균 적용)
        plot_stockpile_reliability_comparison(traces, indices, case_name)

        # 2. 시나리오별 주요 파라미터 사후 분포 플롯
        plot_posterior_distributions(traces, case_name)

        # 3. 시나리오별 LOT 신뢰도 Forest 플롯
        plot_lot_reliability_forest(traces, indices, case_name)

        # 4. [추가] 이항분포 vs 초기하분포 모델 비교 (보수적 시나리오 기준)
        print("\n--- Running model comparison (Binomial vs. Hypergeometric) ---")
        pessimistic_params = scenarios["보수적 (Pessimistic)"]
        traces_to_compare = {
            "이항분포 모델": run_binomial_model(data, indices, pessimistic_params),
            "초기하분포 모델": traces["보수적 (Pessimistic)"] # 이미 실행했으므로 재사용
        }
        plot_model_comparison_forest(traces_to_compare, indices, case_name)

    plt.show()

if __name__ == '__main__':
    main()