# config.py

"""
분석 스크립트에 사용될 모든 설정을 관리합니다.
"""

import os
import platform
from typing import Dict, Any, List

# 1. 기본 경로 설정
# --------------------------------
DATA_DIR = 'data'
CACHE_DIR = 'cache'
OUTPUT_DIR = 'output'
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')


# 2. MCMC 샘플링 설정
# --------------------------------
MCMC_CONFIG = {
    'draws': 2000,
    'tune': 1000,
    'cores': 1,
    'random_seed': 42,
    'progressbar': True
}


# 3. 모델 공통 파라미터 (사전 확률)
# --------------------------------
MODEL_PRIORS = {
    "MU_GLOBAL_LOGIT_MU": 1.386,   # 80% 신뢰도에 해당하는 로짓 값
    "MU_GLOBAL_LOGIT_SIGMA": 0.5,
    "VARIANCE_DEGRADATION_RATE_SIGMA": 0.05
}


# 4. 분석 시나리오 정의
# --------------------------------
# 각 시나리오는 사전 확률의 표준편차 값을 조정하여 정의됩니다.
# inter_year_sigma: 연도 간 편차의 사전 분포 스케일
# intra_lot_sigma: LOT 간 편차의 사전 분포 스케일
# degradation_effect_on_variance: 성능 저하 효과 반영 여부
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "낙관적 (Optimistic)": {
        "inter_year_sigma": 0.01, 
        "intra_lot_sigma": 0.02, 
        "degradation_effect_on_variance": False
    },
    "보수적 (Pessimistic)": {
        "inter_year_sigma": 0.2, 
        "intra_lot_sigma": 0.1, 
        "degradation_effect_on_variance": True
    }
}


# 5. 시각화 설정
# --------------------------------
# 시나리오 및 모델별 색상 지정
SCENARIO_COLORS: Dict[str, str] = {
    "낙관적 (Optimistic)": "cornflowerblue",
    "보수적 (Pessimistic)": "salmon",
    "이항분포 모델": "darkorange",
    "초기하분포 모델": "darkcyan"
}

# 목표 신뢰도 설정
TARGET_RELIABILITY = 0.98

# 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
DEFAULT_FONT = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'


# 6. 분석 대상 데이터 파일
# --------------------------------
TEST_CASES: Dict[str, str] = {
    "시험 1 (0개 실패)": "observed_reliability_data_0_failures.csv",
    "시험 2 (1개 실패)": "observed_reliability_data_1_failure.csv"
}

STOCKPILE_DATA_FILE = "stockpile_composition_data.csv"
