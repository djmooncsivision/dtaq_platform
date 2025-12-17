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

def plot_comparison_results(traces: Dict[str, az.InferenceData], lot_labels: List[str]):
    """여러 시나리오의 추론 결과를 비교하는 Forest Plot을 생성합니다."""
    # squeeze=False ensures that `axes` is always a 2D array, even with 1 plot
    fig, axes = plt.subplots(1, len(traces), figsize=(8 * len(traces), 12), sharey=True, squeeze=False)
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    for i, (name, trace) in enumerate(traces.items()):
        ax = axes[i]
        az.plot_forest(trace, var_names=['reliability_lot'], combined=True, hdi_prob=0.9, ax=ax)
        ax.set_title(f'Scenario: {name}', fontsize=16)
        ax.set_yticklabels(lot_labels[::-1] if i == 0 else [])

    fig.suptitle('Lot Reliability Estimation Comparison by Scenario (90% HDI)', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    """메인 실행 함수"""
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
    
    plot_comparison_results(traces, indices["all_lots"])

if __name__ == '__main__':
    main()