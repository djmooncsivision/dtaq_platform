import streamlit as st

st.set_page_config(
    page_title="미사일 신뢰도 분석 도구 v2.0",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 미사일 신뢰도 분석 도구 v2.0")

st.markdown("""
### 고급 신뢰도 분석 도구에 오신 것을 환영합니다

이 도구는 고급 통계 모델과 머신러닝 기법을 사용하여 미사일 신뢰도 데이터를 분석합니다.

#### 🌟 주요 기능
1.  **유연한 데이터 소스 (Flexible Data Sourcing)**:
    *   **실제 데이터 (Real Data)**: 기존 CSV 파일을 로드하여 분석합니다.
    *   **가상 데이터 (Synthetic Data)**: 사용자 정의 파라미터(QIM, ASRP, 창정비)로 테스트 데이터셋을 생성합니다.
2.  **고급 분석 (Advanced Analysis)**:
    *   **추세 예측**: 6가지 회귀 모델(선형, GPR, 신경망 등)을 비교 분석합니다.
    *   **위험 스크리닝**: 노후화 경향을 기반으로 고위험 항목을 자동으로 식별합니다.
3.  **인터랙티브 시각화**:
    *   데이터 분포 및 추세 분석을 위한 동적 그래프를 제공합니다.
    *   RMSE 비교를 통해 모델 성능을 검증합니다.

#### 👈 사용 방법
왼쪽 **사이드바**를 사용하여 이동하세요:
1.  **데이터 설정 (Data Configuration)** 메뉴로 이동하여 데이터를 선택하거나 생성합니다.
2.  데이터 준비가 완료되면 **분석 대시보드 (Analysis Dashboard)** 메뉴로 이동하여 결과를 확인합니다.

---
*Developed by Antigravity*
""")

# Initialize Session State for Data if not exists
if 'data_loader' not in st.session_state:
    st.session_state['data_loader'] = None
if 'data_source_type' not in st.session_state:
    st.session_state['data_source_type'] = None
