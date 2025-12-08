import streamlit as st
import pandas as pd
import os
import sys

# Add v1_code and vgui2.0 to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'v1_code')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from utils import generate_synthetic_data, create_limits_df, InMemoryLoader

st.set_page_config(page_title="데이터 설정", page_icon="⚙️", layout="wide")

st.title("⚙️ 데이터 설정 (Data Configuration)")

tab1, tab2 = st.tabs(["📂 실제 데이터 로드 (Real Data)", "🧪 가상 데이터 생성 (Synthetic Data)"])

# --- Tab 1: Load Real Data ---
with tab1:
    st.header("기존 데이터 로드")
    
    data_source = st.radio("데이터 소스 선택", ["디렉토리 (CSV 파일들)", "단일 CSV 파일 업로드"])
    
    if data_source == "디렉토리 (CSV 파일들)":
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'v1_code', 'scenario_data'))
        input_dir = st.text_input("데이터 디렉토리 경로", value=default_dir)
        
        if st.button("디렉토리에서 로드"):
            if os.path.exists(input_dir):
                try:
                    loader = DataLoader(input_dir)
                    loader.load_data()
                    st.session_state['data_loader'] = loader
                    st.session_state['data_source_type'] = 'Real'
                    st.success(f"성공적으로 로드됨: {len(loader.df)} 행 (디렉토리)")
                except Exception as e:
                    st.error(f"데이터 로드 중 오류 발생: {e}")
            else:
                st.error("디렉토리가 존재하지 않습니다.")
                
    elif data_source == "단일 CSV 파일 업로드":
        uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")
        if uploaded_file is not None:
            if st.button("CSV 로드"):
                try:
                    df = pd.read_csv(uploaded_file)
                    # Basic validation
                    if 'Dataset' not in df.columns:
                        st.warning("CSV에 'Dataset' 컬럼이 없습니다. 모두 'ASRP'로 가정합니다.")
                        df['Dataset'] = 'ASRP'
                    
                    # Create a mock loader
                    loader = InMemoryLoader(df)
                    st.session_state['data_loader'] = loader
                    st.session_state['data_source_type'] = 'Real'
                    st.success(f"성공적으로 로드됨: {len(df)} 행")
                except Exception as e:
                    st.error(f"CSV 읽기 오류: {e}")

# --- Tab 2: Generate Synthetic Data ---
with tab2:
    st.header("가상 데이터 생성기")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. 수량 설정 (Quantity)")
        n_qim = st.number_input("QIM (초기) 수량", value=200, step=10)
        n_asrp = st.number_input("ASRP (저장) 수량", value=50, step=10)
        n_overhaul = st.number_input("Overhaul (창정비) 수량", value=200, step=10)
        
    with col2:
        st.subheader("2. 시점 설정 (Timing)")
        asrp_range = st.slider("ASRP 운용월 범위", 0, 240, (96, 144))
        overhaul_range = st.slider("Overhaul 운용월 범위", 0, 240, (108, 132))
        
    with col3:
        st.subheader("3. 노후화 설정 (Degradation)")
        degrading_items_str = st.text_input("노후화 적용 항목 (쉼표로 구분)", value="23, 24, 25, 26, 27")
        drift_rate = st.slider("변화율 (월별 평균 감소량)", 0.0, 0.1, 0.05, step=0.01)
        noise_growth = st.slider("분산 증가율 (Noise Growth)", 1.0, 5.0, 1.0, step=0.1)
        
    if st.button("가상 데이터 생성"):
        try:
            # Parse items
            degrading_items = [int(x.strip()) for x in degrading_items_str.split(',') if x.strip().isdigit()]
            
            with st.spinner("데이터 생성 중..."):
                df = generate_synthetic_data(
                    n_qim=n_qim, n_asrp=n_asrp, n_overhaul=n_overhaul,
                    asrp_time_range=asrp_range, overhaul_time_range=overhaul_range,
                    degrading_items=degrading_items, drift_rate=drift_rate, noise_growth=noise_growth
                )
                
                limits_df = create_limits_df()
                loader = InMemoryLoader(df, limits_df)
                
                st.session_state['data_loader'] = loader
                st.session_state['data_source_type'] = 'Synthetic'
                
                st.success(f"생성 완료: 총 {len(df)} 행 (QIM:{n_qim}, ASRP:{n_asrp}, Overhaul:{n_overhaul})")
                st.dataframe(df.head())
                
        except Exception as e:
            st.error(f"데이터 생성 오류: {e}")

# Check status
if st.session_state.get('data_loader'):
    st.info(f"현재 로드된 데이터: {st.session_state['data_source_type']} 데이터 ({len(st.session_state['data_loader'].df)} 행)")
    st.markdown("👉 **'분석 대시보드 (Analysis Dashboard)' 페이지로 이동하여 결과를 확인하세요.**")
