import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def run_density_plot_comparison():
    """
    두 가지 가정 시나리오(낙관적/보수적)를 실행하고,
    전체 비축물량의 평균 신뢰도 분포를 비교하여 시각화합니다.
    """
    # --- 1. 환경 설정 및 데이터 준비 ---
    sns.set_theme(style="whitegrid")
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
    except:
        plt.rcParams['font.family'] = 'AppleGothic' # Mac
    plt.rcParams['axes.unicode_minus'] = False

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

    # --- 2. 시나리오별 모델 실행 ---
    scenarios = {
        "낙관적 (Optimistic)": {"inter_year_sigma": 0.01, "intra_lot_sigma": 0.02, "degradation_effect_on_variance": False},
        "보수적 (Pessimistic)": {"inter_year_sigma": 0.2, "intra_lot_sigma": 0.1, "degradation_effect_on_variance": True}
    }
    traces = {}

    for name, params in scenarios.items():
        print(f"\n--- [{name}] 시나리오 분석 실행... ---")
        with pm.Model() as model:
            mu_global_logit = pm.Normal('mu_global_logit', mu=3.89, sigma=0.5)
            sigma_year = pm.HalfNormal('sigma_year', sigma=params["inter_year_sigma"])
            sigma_lot_base = pm.HalfNormal('sigma_lot_base', sigma=params["intra_lot_sigma"])

            if params["degradation_effect_on_variance"]:
                variance_degradation_rate = pm.HalfNormal('variance_degradation_rate', sigma=0.05)
                age_of_lot = 2025 - year_of_lot
                sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base + age_of_lot * variance_degradation_rate)
            else:
                sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base)

            theta_year = pm.Normal('theta_year', mu=mu_global_logit, sigma=sigma_year, shape=len(all_years))
            theta_lot = pm.Normal('theta_lot', mu=theta_year[year_idx_of_lot], sigma=sigma_lot_effective, shape=len(all_lots))
            reliability_lot = pm.Deterministic('reliability_lot', pm.invlogit(theta_lot))

            y_obs = pm.Binomial('y_obs', n=data['num_tested'].values, p=reliability_lot[observed_lot_idx], observed=data['num_success'].values)
            traces[name] = pm.sample(2000, tune=1500, cores=1, return_inferencedata=True, random_seed=2024, progressbar=True)
        print(f"--- [{name}] 시나리오 분석 완료. ---")

    # --- 3. 결과 비교 시각화 ---
    plt.figure(figsize=(12, 7))

    for name, trace in traces.items():
        posterior_samples = trace.posterior['reliability_lot'].values
        mean_stockpile_reliability = posterior_samples.mean(axis=2)
        
        sns.kdeplot(mean_stockpile_reliability.flatten(), label=f'시나리오: {name}', fill=True, alpha=0.5)
        
        hdi_lower_bound = az.hdi(mean_stockpile_reliability, hdi_prob=0.9)[0]
        print(f"[{name} 시나리오] 90% 신뢰수준에서 전체 비축물량의 평균 신뢰도는 최소 {hdi_lower_bound:.3f} 이상일 것으로 추정됩니다.")

    plt.title('가정 시나리오별 전체 비축물량 평균 신뢰도 분포 비교', fontsize=18)
    plt.xlabel('전체 평균 신뢰도', fontsize=12)
    plt.ylabel('확률 밀도', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == '__main__':
    run_density_plot_comparison()