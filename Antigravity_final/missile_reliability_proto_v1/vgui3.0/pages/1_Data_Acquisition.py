import streamlit as st
import pandas as pd
import os
import sys

# Add current directory and parent to path for modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from modules.pdf_converter import PDFConverter
    from modules.csv_integrator import CSVIntegrator
    from utils import InMemoryLoader # Import for compatibility
except ImportError as e:
    st.error(f"모듈 로드 중 오류 발생: {e}")
    # Fallback to local import if run from pages dir
    from modules.pdf_converter import PDFConverter
    from modules.csv_integrator import CSVIntegrator
    from utils import InMemoryLoader

st.set_page_config(page_title="데이터 획득 (Data Acquisition)", page_icon="📂", layout="wide")

st.title("📂 데이터 획득 (Data Acquisition)")
st.markdown("Raw 데이터(PDF 성적서 또는 분할된 CSV 파일들)를 통합하여 분석 가능한 포맷으로 변환합니다.")

tab1, tab2 = st.tabs(["📄 PDF 성적서 변환", "📊 CSV 폴더 통합"])

# --- Tab 1: PDF Conversion ---
with tab1:
    st.header("1. PDF 성적서 업로드 및 변환")
    st.caption("스캔된 PDF 이미지 성적서를 OCR로 분석하여 데이터를 추출합니다.")

    uploaded_pdfs = st.file_uploader("PDF 파일 업로드 (복수 선택 가능)", type="pdf", accept_multiple_files=True, key="pdf_uploader")

    if uploaded_pdfs:
        if st.button("PDF 변환 실행 (Convert)", key="btn_pdf"):
            with st.spinner("PDF 분석 및 변환 중... (EasyOCR 엔진 사용)"):
                try:
                    # Limits path for mapping (Legacy arg in PDFConverter, unused but required)
                    limits_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ref_data', 'upper_lower_limit.csv'))
                    converter = PDFConverter(limits_path)
                    
                    df, raw_df = converter.convert(uploaded_pdfs)
                    
                    if not df.empty:
                        st.success(f"변환 성공! 총 {len(df)}개의 데이터를 확보했습니다.")
                        
                        st.session_state['uploaded_data'] = df
                        st.session_state['data_source'] = "PDF"
                        
                        if 'Dataset' not in df.columns:
                            df['Dataset'] = 'Real'
                            
                        loader = InMemoryLoader(df)
                        st.session_state['data_loader'] = loader
                        st.session_state['data_source_type'] = 'PDF'
                        
                        st.subheader("변환된 데이터 미리보기")
                        st.dataframe(df.head())
                        
                        if len(uploaded_pdfs) == 1:
                            base_name = os.path.splitext(uploaded_pdfs[0].name)[0]
                            csv_fname = f"{base_name}_sample_csv.csv"
                        else:
                            csv_fname = "converted_data_sample_csv.csv"
                        
                        csv = df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(label=f"💾 CSV 다운로드 ({csv_fname})", data=csv, file_name=csv_fname, mime='text/csv')
                        
                        with st.expander("원본 추출 데이터 (Raw Data) 보기"):
                            st.dataframe(raw_df)
                            
                        st.info("데이터가 로드되었습니다. '분석 대시보드' 메뉴로 이동하세요.")
                    else:
                        st.error("데이터 추출에 실패했습니다.")
                        
                except Exception as e:
                    st.error(f"오류 발생: {e}")

# --- Tab 2: CSV Integration ---
with tab2:
    st.header("2. CSV 폴더 통합 (CSV to CSV)")
    st.caption("여러 단계별 성적서 CSV 파일들이 있는 폴더를 선택하여 하나의 통합된 성적서(Row)로 합칩니다.")
    
    # Default path setup
    default_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'csv2csv'))
    input_dir = st.text_input("CSV 폴더 경로 입력 (Input Folder Path)", value=default_csv_path)
    
    if st.button("폴더 통합 및 변환 (Integrate)", key="btn_csv"):
        if os.path.exists(input_dir) and os.path.isdir(input_dir):
            with st.spinner("CSV 파일 통합 분석 중..."):
                try:
                    integrator = CSVIntegrator()
                    df_integrated = integrator.integrate_folder(input_dir)
                    
                    if not df_integrated.empty:
                        st.success("통합 성공! 1개의 통합된 레코드를 생성했습니다.")
                        
                        st.dataframe(df_integrated)
                        
                        # Set to Session State
                        df_integrated['Dataset'] = 'Real'
                        st.session_state['uploaded_data'] = df_integrated
                        st.session_state['data_source'] = "CSV_Integrated"
                        
                        loader = InMemoryLoader(df_integrated)
                        st.session_state['data_loader'] = loader
                        st.session_state['data_source_type'] = 'CSV_Integrated'
                        
                        csv_data = df_integrated.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="💾 통합된 CSV 다운로드 (converted_from_csv.csv)",
                            data=csv_data,
                            file_name="converted_from_csv.csv",
                            mime="text/csv"
                        )
                        st.info("데이터가 로드되었습니다. '분석 대시보드' 메뉴로 이동하세요.")
                    else:
                        st.warning("통합할 유효한 데이터가 폴더에 없습니다.")
                except Exception as e:
                    st.error(f"통합 중 오류 발생: {e}")
        else:
            st.error("올바른 디렉토리 경로가 아닙니다.")

st.divider()
st.markdown("""
### 💡 참고 사항
*   **PDF 변환**: 이미지 기반의 PDF도 OCR을 통해 텍스트를 인식합니다.
*   **CSV 통합**: `1200.csv`, `1300.csv` 등 분리된 시험 단계별 결과를 하나의 행(Row)으로 합칩니다.
""")

