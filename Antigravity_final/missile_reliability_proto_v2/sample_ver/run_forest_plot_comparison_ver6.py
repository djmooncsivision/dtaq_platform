# run_forest_plot_comparison_ver6.py

import pytensor
pytensor.config.cxx = ""

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Callable
import os
import config  # 설정 파일 임포트

# --- 1. 데이터 준비 ---

def prepare_data_and_indices(observed_data_filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    CSV 파일에서 원본 시험 데이터를 로드하고, LOT별로 집계한 후,
    분석에 필요한 데이터프레임과 인덱스를 준비합니다.
    """
    observed_data_path = os.path.join(config.DATA_DIR, observed_data_filename)
    stockpile_data_path = os.path.join(config.DATA_DIR, config.STOCKPILE_DATA_FILE)

    if not os.path.exists(observed_data_path) or not os.path.exists(stockpile_data_path):
        raise FileNotFoundError(
            f"데이터 파일({observed_data_path} 또는 {stockpile_data_path})을 찾을 수 없습니다. "
            "데이터 생성 스크립트를 먼저 실행해주세요."
        )
        
    # 1. 원본 데이터 로드
    raw_observed_data = pd.read_csv(observed_data_path)
    stockpile_data = pd.read_csv(stockpile_data_path)

    # 2. LOT ID를 기준으로 데이터 집계
    # num_tested: 각 lot_id의 총 등장 횟수
    # num_failures: 각 lot_id에 대해 result가 'Failure'인 경우의 수
    agg_data = raw_observed_data.groupby('lot_id').apply(lambda x: pd.Series({
        'num_tested': len(x),
        'num_failures': (x['result'] == 'Failure').sum()
    })).reset_index()
    
    # 'lot_id' 컬럼명을 모델의 다른 부분에서 사용하는 'production_lot'으로 변경
    agg_data = agg_data.rename(columns={'lot_id': 'production_lot'})

    # 3. 성공 횟수 계산
    agg_data['num_success'] = agg_data['num_tested'] - agg_data['num_failures']
    
    # 4. 인덱스 정보 생성
    all_lots = stockpile_data['production_lot'].tolist()
    all_years = sorted(stockpile_data['production_year'].unique())
    
    lot_map = {lot: i for i, lot in enumerate(all_lots)}
    year_map = {year: i for i, year in enumerate(all_years)}
    
    # 집계된 데이터에 있는 lot만 사용
    observed_lot_idx = [lot_map[lot] for lot in agg_data['production_lot']]

    indices = {
        "all_lots": all_lots,
        "all_years": np.array(all_years),
        "year_idx_of_lot": stockpile_data['production_year'].map(year_map).values,
        "observed_lot_idx": observed_lot_idx,
        "year_of_lot": stockpile_data['production_year'].values,
        "lot_quantities": stockpile_data['quantity'].values
    }
    
    # 집계된 데이터를 반환
    return agg_data, indices

# --- 2. 모델 정의 ---

def _build_hierarchical_structure(model_params: Dict[str, Any], indices: Dict[str, Any]) -> Dict[str, Any]:
    """계층 모델의 공통 구조를 정의합니다."""
    mu_global_logit = pm.Normal('mu_global_logit', 
                                mu=config.MODEL_PRIORS['MU_GLOBAL_LOGIT_MU'], 
                                sigma=config.MODEL_PRIORS['MU_GLOBAL_LOGIT_SIGMA'])
    
    sigma_year = pm.HalfNormal('sigma_year', sigma=model_params["inter_year_sigma"])
    sigma_lot_base = pm.HalfNormal('sigma_lot_base', sigma=model_params["intra_lot_sigma"])

    if model_params["degradation_effect_on_variance"]:
        variance_degradation_rate = pm.HalfNormal('variance_degradation_rate', 
                                                  sigma=config.MODEL_PRIORS['VARIANCE_DEGRADATION_RATE_SIGMA'])
        age_of_lot = 2025 - indices["year_of_lot"]
        sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base + age_of_lot * variance_degradation_rate)
    else:
        sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base)

    theta_year = pm.Normal('theta_year', mu=mu_global_logit, sigma=sigma_year, shape=len(indices["all_years"]))
    theta_lot = pm.Normal('theta_lot', mu=theta_year[indices["year_idx_of_lot"]], sigma=sigma_lot_effective, shape=len(indices["all_lots"]))
    
    return {"theta_lot": theta_lot}

def run_bayesian_model(
    model_builder: Callable, 
    model_name: str,
    case_name: str, 
    scenario_name: str,
    data: pd.DataFrame, 
    indices: Dict[str, Any], 
    model_params: Dict[str, Any]
) -> az.InferenceData:
    """
    지정된 모델을 실행하고 결과를 캐싱합니다.
    파일이 존재하면 캐시된 결과를 로드하고, 그렇지 않으면 모델을 실행하고 결과를 저장합니다.
    """
    cache_filename = f"{model_name}_{case_name.replace(' ', '_')}_{scenario_name.replace(' ', '_')}.nc"
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    if os.path.exists(cache_path):
        print(f"--- 캐시된 결과 로드: {cache_filename} ---")
        return az.from_netcdf(cache_path)
    
    print(f"--- 모델 실행 시작: {model_name} ({scenario_name}) ---")
    trace = model_builder(data, indices, model_params)
    
    print(f"--- 결과 캐싱: {cache_filename} ---")
    trace.to_netcdf(cache_path)
    
    return trace

def build_binomial_model(data: pd.DataFrame, indices: Dict[str, Any], model_params: Dict[str, Any]) -> az.InferenceData:
    """이항분포를 사용한 베이지안 모델을 구성하고 샘플링을 실행합니다."""
    with pm.Model() as model:
        structure = _build_hierarchical_structure(model_params, indices)
        theta_lot = structure["theta_lot"]
        
        reliability_lot = pm.Deterministic('reliability_lot', pm.invlogit(theta_lot))
        
        pm.Binomial('y_obs', 
                    n=data['num_tested'].values, 
                    p=reliability_lot[indices["observed_lot_idx"]], 
                    observed=data['num_success'].values)
        
        mcmc_config = config.MCMC_CONFIG
        # PyMC v4/v5 호환성을 위해 'cores'를 'chains'로 처리
        chains = mcmc_config.get('chains', mcmc_config.get('cores', 1))
        
        return pm.sample(
            draws=mcmc_config['draws'],
            tune=mcmc_config['tune'],
            chains=chains,
            random_seed=mcmc_config['random_seed'],
            progressbar=mcmc_config['progressbar'],
            return_inferencedata=True
        )

def build_hypergeometric_model(data: pd.DataFrame, indices: Dict[str, Any], model_params: Dict[str, Any]) -> az.InferenceData:
    """초기하분포를 사용한 베이지안 모델을 구성하고 샘플링을 실행합니다."""
    with pm.Model() as model:
        structure = _build_hierarchical_structure(model_params, indices)
        theta_lot = structure["theta_lot"]
        
        p_lot = pm.invlogit(theta_lot)
        lot_quantities = indices["lot_quantities"]
        k_lot = pm.Binomial('k_lot', n=lot_quantities, p=p_lot, shape=len(indices["all_lots"]))

        pm.HyperGeometric('y_obs', 
                          N=lot_quantities[indices["observed_lot_idx"]], 
                          k=k_lot[indices["observed_lot_idx"]], 
                          n=data['num_tested'].values, 
                          observed=data['num_success'].values)

        reliability_lot = pm.Deterministic('reliability_lot', k_lot / lot_quantities)
        
        mcmc_config = config.MCMC_CONFIG
        # PyMC v4/v5 호환성을 위해 'cores'를 'chains'로 처리
        chains = mcmc_config.get('chains', mcmc_config.get('cores', 1))
        
        return pm.sample(
            draws=mcmc_config['draws'],
            tune=mcmc_config['tune'],
            chains=chains,
            random_seed=mcmc_config['random_seed'],
            progressbar=mcmc_config['progressbar'],
            return_inferencedata=True
        )

# --- 3. 시각화 및 결과 저장 ---

def save_plot_and_csv(
    fig: plt.Figure, 
    df: pd.DataFrame, 
    base_filename: str, 
    case_name: str
):
    """플롯과 데이터프레임을 지정된 경로에 저장합니다."""
    clean_case_name = case_name.replace(' ', '_')
    
    # 플롯 저장
    plot_path = os.path.join(config.PLOT_DIR, f"{base_filename}_{clean_case_name}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"플롯 저장 완료: {plot_path}")

    # CSV 저장
    csv_path = os.path.join(config.CSV_DIR, f"{base_filename}_{clean_case_name}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 저장 완료: {csv_path}")
    
    plt.close(fig)

def plot_stockpile_reliability_comparison(traces: Dict[str, az.InferenceData], indices: Dict[str, Any], case_name: str):
    """전체 비축물량 가중 평균 신뢰도 분포를 비교하고 저장합니다."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    weights = indices["lot_quantities"]
    all_draws_df = []

    for name, trace in traces.items():
        posterior_samples = trace.posterior['reliability_lot'].values
        weighted_mean_per_draw = np.average(posterior_samples, axis=-1, weights=weights).flatten()
        
        sns.kdeplot(weighted_mean_per_draw, label=f'시나리오: {name}', fill=True, 
                    alpha=0.6, color=config.SCENARIO_COLORS.get(name), ax=ax)
        
        hdi_bounds = az.hdi(weighted_mean_per_draw, hdi_prob=0.9)
        ax.axvspan(hdi_bounds[0], hdi_bounds[1], color=config.SCENARIO_COLORS.get(name), alpha=0.1)
        
        print(f"[{name} 시나리오] 90% 신뢰수준에서 전체 평균 신뢰도는 최소 {hdi_bounds[0]:.4f} 이상입니다.")
        print(f"[{name} 시나리오] 90% 신뢰구간(HDI): [{hdi_bounds[0]:.4f}, {hdi_bounds[1]:.4f}]")

        all_draws_df.append(pd.DataFrame({
            'scenario': name,
            'weighted_mean_reliability': weighted_mean_per_draw
        }))

    ax.set_title(f'[{case_name}] 시나리오별 전체 비축물량 가중 평균 신뢰도 분포 비교', fontsize=20, pad=20)
    ax.axvline(x=config.TARGET_RELIABILITY, color='grey', linestyle='--', linewidth=1.5, label=f'목표 신뢰도 ({config.TARGET_RELIABILITY})')
    ax.text(config.TARGET_RELIABILITY, ax.get_ylim()[1]*0.9, ' 목표 신뢰도', color='dimgray', fontsize=12, ha='left')
    ax.set_xlabel('전체 가중 평균 신뢰도', fontsize=15)
    ax.set_ylabel('확률 밀도', fontsize=15)
    ax.legend(title='시나리오', fontsize=13, title_fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()

    save_plot_and_csv(fig, pd.concat(all_draws_df), "stockpile_reliability_comparison", case_name)

def plot_posterior_distributions(traces: Dict[str, az.InferenceData], case_name: str):
    """주요 파라미터의 사후 분포를 비교하고 저장합니다."""
    params_to_plot = {
        'mu_global_logit': '전체 평균 신뢰도 (로짓 스케일)',
        'sigma_year': '연도 간 편차',
        'sigma_lot_base': 'LOT 간 기본 편차',
        'variance_degradation_rate': '성능 저하율 (편차 증가 속도)'
    }
    
    for param, title in params_to_plot.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        all_samples_df = []
        has_plot = False

        for name, trace in traces.items():
            if param in trace.posterior:
                has_plot = True
                samples = trace.posterior[param].values.flatten()
                sns.kdeplot(samples, label=name, color=config.SCENARIO_COLORS.get(name), 
                            fill=True, alpha=0.6, ax=ax)
                all_samples_df.append(pd.DataFrame({'scenario': name, 'parameter': param, 'value': samples}))

        if not has_plot:
            plt.close(fig)
            continue

        ax.set_title(f'[{case_name}] 사후 분포 비교: {title}', fontsize=20, pad=20)
        ax.set_xlabel('파라미터 값', fontsize=15)
        ax.set_ylabel('확률 밀도', fontsize=15)
        ax.legend(title='시나리오', fontsize=13, title_fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()

        save_plot_and_csv(fig, pd.concat(all_samples_df), f"posterior_{param}", case_name)

def plot_forest(
    traces: Dict[str, az.InferenceData], 
    indices: Dict[str, Any], 
    case_name: str,
    title: str,
    legend_title: str,
    base_filename: str
):
    """LOT 신뢰도 추정 결과를 Forest Plot으로 시각화하고 저장합니다."""
    fig, ax = plt.subplots(figsize=(12, 16))
    
    y_positions = np.arange(len(indices["all_lots"]))
    all_hdi_df = []

    for i, (name, trace) in enumerate(traces.items()):
        posterior_data = trace.posterior['reliability_lot']
        hdi_data = az.hdi(posterior_data, hdi_prob=0.9)['reliability_lot'].values
        mean_data = posterior_data.mean(dim=['chain', 'draw']).values
        
        y_offset = -0.15 + i * 0.3
        
        ax.errorbar(x=mean_data, y=y_positions + y_offset, 
                    xerr=[mean_data - hdi_data[:, 0], hdi_data[:, 1] - mean_data],
                    fmt='none', elinewidth=1.5, capsize=4, 
                    color=config.SCENARIO_COLORS.get(name), alpha=0.7)
        
        ax.plot(mean_data, y_positions + y_offset, 'o', markersize=6, 
                color=config.SCENARIO_COLORS.get(name), label=name)

        hdi_df = pd.DataFrame({
            'lot': indices["all_lots"],
            'mean': mean_data,
            'hdi_5%': hdi_data[:, 0],
            'hdi_95%': hdi_data[:, 1]
        })
        hdi_df['source'] = name
        all_hdi_df.append(hdi_df)

    ax.set_yticks(ticks=y_positions, labels=indices["all_lots"], fontsize=10)
    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel('추정 신뢰도', fontsize=15)
    ax.set_ylabel('생산 LOT', fontsize=15)
    ax.set_xlim(0.9, 1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend(title=legend_title, fontsize=13, title_fontsize=14)
    fig.tight_layout()

    save_plot_and_csv(fig, pd.concat(all_hdi_df), base_filename, case_name)

# --- 4. 메인 실행 로직 ---

def main():
    """메인 실행 함수"""
    plt.rcParams['font.family'] = config.DEFAULT_FONT
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(config.CSV_DIR, exist_ok=True)

    for case_name, data_filename in config.TEST_CASES.items():
        print(f"\n{'='*25} 분석 시작: {case_name} {'='*25}")
        
        data, indices = prepare_data_and_indices(data_filename)
        
        # --- 시나리오별 초기하분포 모델 분석 ---
        scenario_traces = {}
        for name, params in config.SCENARIOS.items():
            scenario_traces[name] = run_bayesian_model(
                model_builder=build_hypergeometric_model,
                model_name="Hypergeometric",
                case_name=case_name,
                scenario_name=name,
                data=data,
                indices=indices,
                model_params=params
            )
        
        # --- 분석 결과 시각화 및 저장 ---
        print(f"\n--- '{case_name}'에 대한 결과물 생성 중 ---")
        
        plot_stockpile_reliability_comparison(scenario_traces, indices, case_name)
        plot_posterior_distributions(scenario_traces, case_name)
        plot_forest(
            traces=scenario_traces,
            indices=indices,
            case_name=case_name,
            title=f'[{case_name}] 시나리오별 LOT 신뢰도 추정 비교 (90% HDI)',
            legend_title='시나리오',
            base_filename="lot_reliability_by_scenario"
        )

        # --- 모델 비교 분석 (이항 vs 초기하) ---
        print("\n--- 모델 비교 분석 실행 (Binomial vs. Hypergeometric) ---")
        pessimistic_params = config.SCENARIOS["보수적 (Pessimistic)"]
        
        traces_to_compare = {
            "이항분포 모델": run_bayesian_model(
                model_builder=build_binomial_model,
                model_name="Binomial",
                case_name=case_name,
                scenario_name="보수적 (Pessimistic)",
                data=data,
                indices=indices,
                model_params=pessimistic_params
            ),
            "초기하분포 모델": scenario_traces["보수적 (Pessimistic)"]
        }
        
        plot_forest(
            traces=traces_to_compare,
            indices=indices,
            case_name=case_name,
            title=f'[{case_name}] 모델별 LOT 신뢰도 추정 비교 (보수적 시나리오, 90% HDI)',
            legend_title='모델',
            base_filename="lot_reliability_by_model"
        )

    print(f"\n{'='*25} 모든 분석 완료 {'='*25}")
    # plt.show() # 모든 플롯이 파일로 저장되므로, 화면 표시는 선택사항

if __name__ == '__main__':
    main()
