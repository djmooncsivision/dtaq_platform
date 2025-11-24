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
    plt.figure(figsize=(12, 8))
    
    print("\n--- Overall Stockpile Reliability Analysis ---")
    for name, trace in traces.items():
        posterior_samples = trace.posterior['reliability_lot'].values
        mean_per_draw = posterior_samples.mean(axis=-1).flatten()
        
        sns.kdeplot(
            mean_per_draw, 
            label=f'시나리오: {name}', 
            fill=True, 
            alpha=0.6, 
            color=SCENARIO_COLORS.get(name)
        )
        
        hdi_lower_bound = az.hdi(mean_per_draw, hdi_prob=0.9)[0]
        print(f"[{name} 시나리오] 90% 신뢰수준에서 전체 평균 신뢰도는 최소 {hdi_lower_bound:.4f} 이상일 것으로 추정됩니다.")
        hdi_bounds = az.hdi(mean_per_draw, hdi_prob=0.9)
        plt.axvspan(hdi_bounds[0], hdi_bounds[1], color=SCENARIO_COLORS.get(name), alpha=0.1)
        print(f"[{name} 시나리오] 90% 신뢰구간(HDI): [{hdi_bounds[0]:.4f}, {hdi_bounds[1]:.4f}]")
    
    plt.title('시나리오별 전체 비축물량 평균 신뢰도 분포 비교', fontsize=20, pad=20)
    plt.axvline(x=0.98, color='grey', linestyle='--', linewidth=1.5, label='목표 신뢰도 (98%)')
    plt.text(0.98, plt.ylim()[1]*0.9, ' 목표 신뢰도', color='dimgray', fontsize=12, ha='left')
    plt.xlabel('전체 평균 신뢰도', fontsize=15)
    plt.ylabel('확률 밀도', fontsize=15)
    plt.legend(title='시나리오', fontsize=13, title_fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

def plot_posterior_distributions(traces: Dict[str, az.InferenceData]):
    """모델의 주요 파라미터에 대한 사후 분포를 시각화합니다."""
    for name, trace in traces.items():
        vars_to_plot = ['mu_global_logit', 'sigma_year', 'sigma_lot_base']
        if 'variance_degradation_rate' in trace.posterior:
            vars_to_plot.append('variance_degradation_rate')
    # 보수적 시나리오에만 있는 파라미터까지 포함하여 전체 파라미터 목록 생성
    all_params = ['mu_global_logit', 'sigma_year', 'sigma_lot_base', 'variance_degradation_rate']
    
    for param in all_params:
        plt.figure(figsize=(10, 6))
        
        az.plot_posterior(
            trace,
            var_names=vars_to_plot,
            hdi_prob=0.9,
        )
        plt.suptitle(f'[{name} 시나리오] 주요 파라미터 사후 분포', fontsize=20, y=1.02)
        has_data = False
        for name, trace in traces.items():
            if param in trace.posterior:
                has_data = True
                samples = trace.posterior[param].values.flatten()
                sns.kdeplot(
                    samples,
                    label=name,
                    color=SCENARIO_COLORS.get(name),
                    fill=True,
                    alpha=0.6
                )
        
        if not has_data:
            plt.close() # 데이터가 없는 파라미터(예: 낙관적 시나리오의 variance_degradation_rate)는 플롯을 닫음
            continue

        plt.title(f'주요 파라미터 사후 분포 비교: {param}', fontsize=20, pad=20)
        plt.xlabel('파라미터 값', fontsize=15)
        plt.ylabel('확률 밀도', fontsize=15)
        plt.legend(title='시나리오', fontsize=13, title_fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

def plot_lot_reliability_forest(traces: Dict[str, az.InferenceData], indices: Dict[str, Any]):
    """시나리오별로 각 LOT의 신뢰도를 Forest Plot으로 시각화합니다."""
    for name, trace in traces.items():
        plt.figure(figsize=(16, 8))
        
        # 데이터 추출
        posterior_data = trace.posterior['reliability_lot']
        hdi_data = az.hdi(posterior_data, hdi_prob=0.9)['reliability_lot'].values
        mean_data = posterior_data.mean(dim=['chain', 'draw']).values
        
        lot_indices = np.arange(len(indices["all_lots"]))
        
        # 90% HDI 에러바
        plt.errorbar(x=lot_indices, y=mean_data, yerr=[mean_data - hdi_data[:, 0], hdi_data[:, 1] - mean_data],
                     fmt='none', # 점은 따로 그림
                     color='lightgray', elinewidth=2, capsize=5, label='90% HDI (신뢰구간)')
        
        # 평균 추정치 점
        plt.plot(lot_indices, mean_data, 'o', color=SCENARIO_COLORS.get(name), markersize=8, label='평균 추정치')
        
        plt.xticks(ticks=lot_indices, labels=indices["all_lots"], rotation=90, fontsize=10)
        plt.title(f'[{name} 시나리오] LOT별 신뢰도 추정', fontsize=20, pad=20)
        plt.ylabel('추정 신뢰도', fontsize=15)
        plt.xlabel('생산 LOT', fontsize=15)
        plt.legend(fontsize=12)
        plt.ylim(0.9, 1.0) # y축 범위를 조정하여 가독성 향상
        plt.tight_layout()

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
    
    # 1. 전체 평균 신뢰도 비교 플롯
    plot_stockpile_reliability_comparison(traces)

    # 2. 주요 파라미터 사후 분포 플롯
    plot_posterior_distributions(traces)

    # 3. LOT별 신뢰도 Forest 플롯
    plot_lot_reliability_forest(traces, indices)

    plt.show()

if __name__ == '__main__':
    main()