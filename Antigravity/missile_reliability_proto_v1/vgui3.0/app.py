import streamlit as st

st.set_page_config(
    page_title="미사일 신뢰도 분석 도구 v3.0",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 미사일 신뢰도 분석 도구 v3.0")

st.markdown("""
### 환영합니다!

이 도구는 미사일의 저장 신뢰도를 분석하고, 미래의 성능 저하를 예측하기 위해 개발되었습니다.
**v3.0**에서는 성적서 PDF 파일로부터 데이터를 직접 획득하는 기능이 추가되었습니다.

#### 📋 주요 기능
1.  **데이터 획득 (Data Acquisition)**: PDF 성적서 파일을 분석 가능한 CSV 데이터로 변환합니다.
2.  **데이터 설정 (Data Configuration)**: 실제 데이터를 로드하거나, 시뮬레이션을 위한 가상 데이터를 생성합니다.
3.  **분석 대시보드 (Analysis Dashboard)**:
    *   **데이터 개요**: 데이터의 분포와 통계적 특성을 확인합니다.
    *   **추세 예측**: 다양한 회귀 모델을 통해 향후 성능 변화를 예측합니다.
    *   **스크리닝**: 전체 항목의 위험도를 분석하여 우선순위를 도출합니다.
    *   **보고서 생성**: 분석 결과를 Word 보고서로 출력합니다.

#### 👈 사용 방법
왼쪽 **사이드바**를 사용하여 이동하세요:
1.  **데이터 획득** 메뉴에서 PDF 파일을 업로드하여 데이터를 준비합니다.
2.  **데이터 설정** 메뉴에서 사용할 데이터를 선택합니다.
3.  **분석 대시보드** 메뉴에서 상세 분석을 수행하고 보고서를 생성합니다.

---
*Developed by Antigravity*
""")

# Initialize Session State for Data if not exists
if 'data_loader' not in st.session_state:
    st.session_state['data_loader'] = None
if 'data_source_type' not in st.session_state:
    st.session_state['data_source_type'] = None
