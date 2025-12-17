import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 가상 데이터 생성 (2015년 생산된 3개 롯트만 시험) ---
# 실제로는 이 부분에 실제 시험 데이터를 불러옵니다.
data = pd.DataFrame({
    'production_lot': ['2015-LOT-01', '2015-LOT-02', '2015-LOT-04'],
    'num_tested': [11, 10, 12],
    'num_failures': [0, 1, 1]
})
data['num_success'] = data['num_tested'] - data['num_failures']

# --- 2. 데이터 전처리 및 모델을 위한 인덱스 생성 ---
all_years = np.arange(2015, 2020)
all_lots = [f"{year}-LOT-{i:02d}" for year in all_years for i in range(1, 7)]

# 각 데이터가 어떤 연도/롯트에 해당하는지 인덱스로 변환
lot_map = {lot: i for i, lot in enumerate(all_lots)}
year_of_lot = np.array([int(lot.split('-')[0]) for lot in all_lots])
year_map = {year: i for i, year in enumerate(all_years)}
year_idx_of_lot = np.array([year_map[y] for y in year_of_lot])

# 관측된 데이터의 롯트 인덱스
observed_lot_idx = [lot_map[lot] for lot in data['production_lot']]

# --- 3. 시나리오 설정 (분석가가 제어하는 부분) ---
# 'optimistic': 연도별 차이가 작을 것이다.
# 'pessimistic': 연도별 차이가 클 수도 있다.
scenario_assumption = 'pessimistic' 

print(f"--- 분석 시나리오: {scenario_assumption.upper()} ---")

# --- 4. PyMC를 이용한 계층적 베이즈 모델 정의 ---
with pm.Model() as hierarchical_model:
    # === 사전 확률 (Priors) 설정 ===
    # 1. 전체 평균 신뢰도 (상위 레벨)
    # Beta 분포는 0과 1 사이 값을 가지므로 신뢰도 모델링에 적합
    mu_global = pm.Beta('mu_global', alpha=98.0, beta=2.0) # 평균 98% 신뢰도 가정

    # 2. 연도별 변동성 (분석가의 가정이 들어가는 핵심 부분)
    if scenario_assumption == 'optimistic':
        # 연도별 차이가 작을 것이라는 가정 -> 표준편차를 작게 제한
        sigma_year = pm.HalfNormal('sigma_year', sigma=0.01)
    else: # pessimistic
        # 연도별 차이가 클 수 있다는 가정 -> 표준편차를 더 크게 허용
        sigma_year = pm.HalfNormal('sigma_year', sigma=0.1)

    # 3. 롯트별 변동성 (연내 변동성)
    sigma_lot = pm.HalfNormal('sigma_lot', sigma=0.05)

    # === 계층 구조 정의 ===
    # 4. 각 연도의 신뢰도 (관측되지 않음)
    # mu_global을 중심으로 sigma_year만큼 흩어짐
    theta_year = pm.Normal('theta_year', mu=mu_global, sigma=sigma_year, shape=len(all_years))

    # 5. 각 롯트의 신뢰도 (관측되지 않음)
    # 각 롯트는 자신이 속한 연도의 신뢰도(theta_year)를 중심으로 sigma_lot만큼 흩어짐
    theta_lot = pm.Normal('theta_lot', mu=theta_year[year_idx_of_lot], sigma=sigma_lot, shape=len(all_lots))
    
    # 6. 신뢰도 값은 0과 1 사이여야 하므로 로짓 변환의 역함수(pm.invlogit) 사용
    reliability_lot = pm.Deterministic('reliability_lot', pm.invlogit(theta_lot))

    # === 가능도 (Likelihood) ===
    # 7. 관측된 데이터(성공/실패)가 모델과 어떻게 연결되는지 정의
    # Binomial 분포: n번 시험해서 p의 확률로 성공할 때의 성공 횟수
    y_obs = pm.Binomial(
        'y_obs',
        n=data['num_tested'].values,
        p=reliability_lot[observed_lot_idx],
        observed=data['num_success'].values
    )

    # --- 5. MCMC 샘플링 실행 ---
    # 모델의 사후 확률 분포를 추정
    trace = pm.sample(2000, tune=1000, cores=1, return_inferencedata=True)

# --- 6. 결과 분석 ---
# 각 롯트별 신뢰도의 사후 분포 요약
summary = az.summary(trace, var_names=['reliability_lot'])
print(summary)
