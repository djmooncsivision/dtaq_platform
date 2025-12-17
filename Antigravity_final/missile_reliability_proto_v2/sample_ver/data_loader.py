import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, List
import config

def validate_data(df: pd.DataFrame, required_columns: List[str], name: str):
    """
    데이터프레임의 필수 컬럼 존재 여부와 기본 유효성을 검사합니다.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"'{name}' 데이터에 필수 컬럼이 누락되었습니다: {missing_cols}")
    
    if df.empty:
        raise ValueError(f"'{name}' 데이터가 비어 있습니다.")

def prepare_data_and_indices(observed_data_filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    CSV 파일에서 원본 시험 데이터를 로드하고, LOT별로 집계한 후,
    분석에 필요한 데이터프레임과 인덱스를 준비합니다.
    """
    observed_data_path = os.path.join(config.DATA_DIR, observed_data_filename)
    stockpile_data_path = os.path.join(config.DATA_DIR, config.STOCKPILE_DATA_FILE)

    if not os.path.exists(observed_data_path):
        raise FileNotFoundError(f"관측 데이터 파일을 찾을 수 없습니다: {observed_data_path}")
    if not os.path.exists(stockpile_data_path):
        raise FileNotFoundError(f"비축 현황 데이터 파일을 찾을 수 없습니다: {stockpile_data_path}")
        
    # 1. 원본 데이터 로드
    raw_observed_data = pd.read_csv(observed_data_path)
    stockpile_data = pd.read_csv(stockpile_data_path)

    # 데이터 검증
    validate_data(raw_observed_data, [config.COL_LOT_ID, config.COL_RESULT], "관측 데이터")
    validate_data(stockpile_data, [config.COL_PRODUCTION_LOT, config.COL_PRODUCTION_YEAR, config.COL_QUANTITY], "비축 데이터")

    # 2. LOT ID를 기준으로 데이터 집계
    # num_tested: 각 lot_id의 총 등장 횟수
    # num_failures: 각 lot_id에 대해 result가 'Failure'인 경우의 수
    agg_data = raw_observed_data.groupby(config.COL_LOT_ID).apply(lambda x: pd.Series({
        config.COL_NUM_TESTED: len(x),
        config.COL_NUM_FAILURES: (x[config.COL_RESULT] == config.RESULT_FAILURE).sum()
    })).reset_index()
    
    # 'lot_id' 컬럼명을 모델의 다른 부분에서 사용하는 'production_lot'으로 변경
    agg_data = agg_data.rename(columns={config.COL_LOT_ID: config.COL_PRODUCTION_LOT})

    # 3. 성공 횟수 계산
    agg_data[config.COL_NUM_SUCCESS] = agg_data[config.COL_NUM_TESTED] - agg_data[config.COL_NUM_FAILURES]
    
    # 4. 인덱스 정보 생성
    all_lots = stockpile_data[config.COL_PRODUCTION_LOT].tolist()
    all_years = sorted(stockpile_data[config.COL_PRODUCTION_YEAR].unique())
    
    lot_map = {lot: i for i, lot in enumerate(all_lots)}
    year_map = {year: i for i, year in enumerate(all_years)}
    
    # 집계된 데이터에 있는 lot만 사용
    # 검증: 관측된 LOT가 비축 데이터에 존재하는지 확인
    unknown_lots = set(agg_data[config.COL_PRODUCTION_LOT]) - set(all_lots)
    if unknown_lots:
        raise ValueError(f"비축 데이터에 존재하지 않는 LOT가 관측 데이터에 있습니다: {unknown_lots}")

    observed_lot_idx = [lot_map[lot] for lot in agg_data[config.COL_PRODUCTION_LOT]]

    indices = {
        "all_lots": all_lots,
        "all_years": np.array(all_years),
        "year_idx_of_lot": stockpile_data[config.COL_PRODUCTION_YEAR].map(year_map).values,
        "observed_lot_idx": observed_lot_idx,
        "year_of_lot": stockpile_data[config.COL_PRODUCTION_YEAR].values,
        "lot_quantities": stockpile_data[config.COL_QUANTITY].values
    }
    
    # 집계된 데이터를 반환
    return agg_data, indices
