import pandas as pd
import os

def create_observed_data_csv():
    """
    베이지안 모델 분석에 사용될 관측된 시험 데이터를 생성하고 CSV 파일로 저장합니다.
    """
    # --- 데이터 설명 ---
    # 이 데이터는 실제 신뢰성 시험을 수행한 일부 LOT의 결과를 나타냅니다.
    # 이 관측 데이터를 "증거"로 사용하여, 시험하지 않은 전체 LOT의 신뢰도를 추론합니다.
    #
    # - production_lot: 시험 샘플이 채취된 생산 LOT의 식별자.
    # - num_tested: 해당 LOT에서 시험한 샘플의 총 개수.
    # - num_failures: 시험 결과 실패(불량)로 판정된 샘플의 개수.
    
    observed_data = {
        'production_lot': ['2015-LOT-01', '2015-LOT-02', '2015-LOT-04'],
        'num_tested': [11, 10, 12],
        'num_failures': [0, 1, 1]
    }
    
    df = pd.DataFrame(observed_data)

    # CSV 파일로 저장
    output_folder = 'data'
    os.makedirs(output_folder, exist_ok=True)
    
    file_path = os.path.join(output_folder, 'observed_reliability_data.csv')
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"'{output_folder}' 폴더에 관측 데이터 파일을 생성했습니다: {file_path}")

if __name__ == '__main__':
    create_observed_data_csv()