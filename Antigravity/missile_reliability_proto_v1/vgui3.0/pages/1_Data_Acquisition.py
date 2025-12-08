import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="데이터 획득 (Data Acquisition)", page_icon="📂", layout="wide")

st.title("📂 데이터 획득 (Data Acquisition)")
st.markdown("""
이 페이지에서는 **성적서 PDF 파일**을 업로드하여 분석 가능한 **CSV 형식**으로 변환합니다.
""")

# --- 1. PDF Upload Section ---
st.header("1. 성적서 PDF 파일 업로드")
uploaded_files = st.file_uploader("PDF 파일을 선택하세요 (여러 개 가능)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
    
    # Display file names
    file_names = [f.name for f in uploaded_files]
    st.write("선택된 파일 목록:", file_names)

    # --- 2. Conversion Settings (Placeholder) ---
    st.header("2. 변환 설정")
    st.checkbox("표(Table) 데이터만 추출", value=True)
    st.checkbox("이미지 내 텍스트 추출 (OCR 필요)", value=False, disabled=True, help="추후 지원 예정")

    # --- 3. Convert Button ---
    if st.button("CSV로 변환 실행 (Convert to CSV)"):
        with st.spinner("PDF 파일을 분석하여 데이터를 추출하는 중입니다... (현재는 데모 기능입니다)"):
            # Placeholder for conversion logic
            import time
            time.sleep(2) 
            
            st.success("변환이 완료되었습니다!")
            
            # Create a dummy CSV for demonstration
            dummy_data = {
                'Item': ['Item 1', 'Item 2', 'Item 3'],
                'Value': [10.5, 11.2, 9.8],
                'Status': ['Pass', 'Pass', 'Pass']
            }
            df = pd.DataFrame(dummy_data)
            
            st.subheader("변환 결과 미리보기")
            st.dataframe(df)
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 변환된 CSV 다운로드",
                data=csv,
                file_name="converted_data.csv",
                mime="text/csv"
            )

else:
    st.info("좌측의 'Browse files' 버튼을 눌러 PDF 파일을 업로드해주세요.")
