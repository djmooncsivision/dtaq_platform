import pytensor
pytensor.config.cxx = ""

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List

# Constants
CURRENT_YEAR = 2025
# 시나리오별 색상 지정
SCENARIO_COLORS = {
    "낙관적 (Optimistic)": "cornflowerblue",
    "보수적 (Pessimistic)": "salmon"
}



def prepare_data_and_indices() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """분석에 필요한 데이터와 인덱스를 준비합니다."""
    data = pd.DataFrame({
        'production_lot': ['2015-LOT-01', '2015-LOT-02', '2015-LOT-04'],
        'num_tested': [11, 10, 12],
        'num_failures': [0, 1, 1]
    })
    data['num_success'] = data['num_tested'] - data['num_failures']

    all_years = np.arange(2015, 2020)
    all_lots = [f"{year}-LOT-{i:02d}" for year in all_years for i in range(1, 7)]
    lot_map = {lot: i for i, lot in enumerate(all_lots)}
    year_map = {year: i for i, year in enumerate(all_years)}
    year_of_lot = np.array([int(lot.split('-')[0]) for lot in all_lots])
    year_idx_of_lot = np.array([year_map[y] for y in year_of_lot])
    observed_lot_idx = [lot_map[lot] for lot in data['production_lot']]
    
    indices = {
        "all_lots": all_lots,
        "all_years": all_years,
        "year_idx_of_lot": year_idx_of_lot,
        "observed_lot_idx": observed_lot_idx,
        "year_of_lot": year_of_lot
    }
    return data, indices

def run_bayesian_model(data: pd.DataFrame, indices: Dict[str, Any], model_params: Dict[str, Any]) -> az.InferenceData:
    """주어진 파라미터로 베이지안 모델을 실행하고 추론 결과를 반환합니다."""
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
        
        trace = pm.sample(2000, tune=1500, cores=1, return_inferencedata=True, random_seed=2024, progressbar=True)
    return trace

def plot_stockpile_reliability_comparison(traces: Dict[str, az.InferenceData]):
    """
    여러 시나리오의 전체 비축물량 평균 신뢰도 분포를 비교하는 밀도 그림을 생성합니다.
    """
    plt.figure(figsize=(12, 7))
    
    print("\n--- Overall Stockpile Reliability Analysis ---")
    for name, trace in traces.items():
        # 모든 LOT의 신뢰도에 대한 사후 샘플을 가져와서, 각 샘플마다 평균을 계산
        posterior_samples = trace.posterior['reliability_lot'].values
        # (chains, draws, lots) -> (chains * draws)
        # 각 MCMC 샘플(draw)에 대한 전체 LOT의 평균 신뢰도를 계산
        mean_per_draw = posterior_samples.mean(axis=-1).flatten()
        
        sns.kdeplot(
            mean_per_draw, 
            label=f'시나리오: {name}', 
            fill=True, 
            alpha=0.6, 
            color=SCENARIO_COLORS.get(name) # 지정된 색상 사용
        )
        
        # hdi 함수는 [lower, upper] 형태의 numpy 배열을 반환
        hdi_lower_bound = az.hdi(mean_per_draw, hdi_prob=0.9)[0]
        print(f"[{name} 시나리오] 90% 신뢰수준에서 전체 평균 신뢰도는 최소 {hdi_lower_bound:.4f} 이상일 것으로 추정됩니다.")
    
    plt.title('시나리오별 전체 비축물량 평균 신뢰도 분포 비교', fontsize=18)
    plt.xlabel('전체 평균 신뢰도', fontsize=12)
    plt.ylabel('확률 밀도', fontsize=12)
    plt.legend(title='시나리오', fontsize=12)
    plt.show()

def main():
    """메인 실행 함수"""
    # 시각화를 위한 전역 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
    except:
        plt.rcParams['font.family'] = 'AppleGothic' # Mac
    plt.rcParams['axes.unicode_minus'] = False

    data, indices = prepare_data_and_indices()

    scenarios = {
        "낙관적 (Optimistic)": {"inter_year_sigma": 0.01, "intra_lot_sigma": 0.02, "degradation_effect_on_variance": False},
        "보수적 (Pessimistic)": {"inter_year_sigma": 0.2, "intra_lot_sigma": 0.1, "degradation_effect_on_variance": True}
    }
    traces = {}

    for name, params in scenarios.items():
        print(f"\n--- Running analysis for scenario: [{name}] ---")
        traces[name] = run_bayesian_model(data, indices, params)
        print(f"--- Scenario analysis complete: [{name}] ---")
    
    plot_stockpile_reliability_comparison(traces)

if __name__ == '__main__':
    main()